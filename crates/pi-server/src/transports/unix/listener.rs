//! Port of `packages/server/src/transports/unix/listener.ts`.


use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use pi_protocol::DEFAULT_MAX_FRAME_LENGTH;
use tokio::io::AsyncReadExt;
use tokio::net::UnixListener as TokioUnixListener;

use crate::connection::{ByteConnection, ByteConnectionAcceptor};
use crate::listener::ServerListener;
use crate::session_router::uuid_v4;
use crate::server::MAX_TIMER_DELAY_MS;
use crate::transports::unix::socket::{TokioSocket, UnixByteConnection};
use crate::types::{AnyError, ErrorCallback, io_error};

const DEFAULT_SOCKET_MODE: u32 = 0o600;
const DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS: u64 = 5_000;
const SOCKET_PROBE_TIMEOUT_MS: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileIdentity {
	dev: u64,
	ino: u64,
}

struct ResolvedUnixListenerOptions {
	path: PathBuf,
	mode: u32,
	graceful_close_timeout_ms: u64,
	max_pending_bytes: usize,
	on_error: Option<ErrorCallback>,
}

/// State of a started listener (upstream `UnixListener` instance fields).
struct LiveListener {
	accept_task: tokio::task::JoinHandle<()>,
	shared: Arc<ListenerShared>,
	socket_identity: FileIdentity,
}

/// Upstream `UnixListener` (returned by [`create_unix_listener`]).
pub struct UnixListener {
	options: ResolvedUnixListenerOptions,
	closing: AtomicBool,
	live: parking_lot::Mutex<Option<LiveListener>>,
	close_lock: tokio::sync::Mutex<()>,
}

/// Upstream `createUnixListener`.
pub fn create_unix_listener(options: crate::transports::unix::types::UnixListenerOptions) -> Arc<dyn ServerListener> {
	let resolved = resolve_unix_listener_options(options);
	Arc::new(UnixListener {
		options: resolved,
		closing: AtomicBool::new(false),
		live: parking_lot::Mutex::new(None),
		close_lock: tokio::sync::Mutex::new(()),
	})
}

fn resolve_unix_listener_options(
	options: crate::transports::unix::types::UnixListenerOptions,
) -> ResolvedUnixListenerOptions {
	if options.path.is_empty() {
		panic!("Server Unix socket path must not be empty");
	}
	let mode = options.mode.unwrap_or(DEFAULT_SOCKET_MODE);
	if mode > 0o777 {
		panic!("Server Unix socket mode must be an integer between 0 and 0o777");
	}
	let max_frame_length = options.max_frame_length.unwrap_or(DEFAULT_MAX_FRAME_LENGTH);
	if max_frame_length == 0 || max_frame_length > crate::server::MAX_UINT32 {
		panic!(
			"Server maxFrameLength must be an integer between 1 and {}",
			crate::server::MAX_UINT32
		);
	}
	let max_pending_bytes = options.max_pending_bytes.unwrap_or(max_frame_length * 4);
	if max_pending_bytes < max_frame_length + 4 {
		panic!("Server maxPendingBytes must be a safe integer at least maxFrameLength + 4");
	}
	let graceful_close_timeout_ms = options.graceful_close_timeout_ms.unwrap_or(DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS);
	if graceful_close_timeout_ms == 0 || graceful_close_timeout_ms > MAX_TIMER_DELAY_MS {
		panic!(
			"Server gracefulCloseTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}"
		);
	}
	ResolvedUnixListenerOptions {
		path: PathBuf::from(&options.path),
		mode,
		graceful_close_timeout_ms,
		max_pending_bytes,
		on_error: options.on_error,
	}
}

impl UnixListener {
	async fn start(&self, accept: ByteConnectionAcceptor) -> Result<(), AnyError> {
		if self.live.lock().is_some() {
			return Err(listener_error("Unix listener is already started"));
		}
		if self.closing.load(Ordering::SeqCst) {
			return Err(listener_error("Unix listener is closing or closed"));
		}
		let path = self.options.path.clone();
		let owned_bind_path = get_owned_bind_path(&path);
		if let Some(parent) = path.parent() {
			std::os::unix::fs::DirBuilderExt::mode(
				std::fs::DirBuilder::new().recursive(true),
				0o700,
			)
			.create(parent)
				.map_err(io_error)?;
		}
		remove_stale_socket(&path).await?;
		remove_stale_socket(&owned_bind_path).await?;

		let server = match TokioUnixListener::bind(&owned_bind_path) {
			Ok(server) => Arc::new(server),
			Err(error) => return Err(io_error(error)),
		};
		let stats = match std::fs::symlink_metadata(&owned_bind_path) {
			Ok(stats) => stats,
			Err(error) => return Err(io_error(error)),
		};
		if !is_socket_file(&stats) {
			return Err(listener_error(format!(
				"Unix listener path is not a socket after binding: {}",
				owned_bind_path.display()
			)));
		}
		let identity = FileIdentity { dev: stats.dev(), ino: stats.ino() };
		let bind_result: Result<(), AnyError> = async {
			std::fs::hard_link(&owned_bind_path, &path).map_err(io_error)?;
			set_socket_mode(&path, self.options.mode)?;
			remove_path(&owned_bind_path)?;
			Ok(())
		}
		.await;
		if let Err(error) = bind_result {
			// Startup failed after binding: tear down what we created (no
			// accept loop exists yet).
			drop(server);
			let _ = remove_path(&owned_bind_path);
			let _ = cleanup_owned_socket(&path, identity).await;
			return Err(error);
		}
		let shared = Arc::new(ListenerShared {
			connections: parking_lot::Mutex::new(Vec::new()),
			options: ResolvedUnixListenerOptions {
				path: self.options.path.clone(),
				mode: self.options.mode,
				graceful_close_timeout_ms: self.options.graceful_close_timeout_ms,
				max_pending_bytes: self.options.max_pending_bytes,
				on_error: self.options.on_error.clone(),
			},
			closing: AtomicBool::new(false),
			accept,
		});
		let accept_task = {
			let accept_server = Arc::clone(&server);
			let shared = Arc::clone(&shared);
			tokio::spawn(async move {
				loop {
					match accept_server.accept().await {
						Ok((stream, _address)) => accept_socket(&shared, stream),
						Err(error) => {
							if let Some(on_error) = &shared.options.on_error {
								on_error(io_error(error));
							}
						}
					}
				}
			})
		};
		drop(server);
		*self.live.lock() = Some(LiveListener { accept_task, shared, socket_identity: identity });
		Ok(())
	}

	async fn close(&self) -> Result<(), AnyError> {
		let _guard = self.close_lock.lock().await;
		self.closing.store(true, Ordering::SeqCst);
		let live = self.live.lock().take();
		if let Some(live) = live {
			live.shared.closing.store(true, Ordering::SeqCst);
			// Stop accepting (upstream closeNetServer), removing the owned bind
			// path first.
			live.accept_task.abort();
			let _ = live.accept_task.await;
			let cleanup = cleanup_owned_socket(&self.options.path, live.socket_identity).await;
			// Close connections concurrently (upstream Promise.all).
			let connections: Vec<Arc<UnixByteConnection>> = live.shared.connections.lock().iter().cloned().collect();
			let joins: Vec<_> = connections
				.iter()
				.map(|connection| {
					let connection = Arc::clone(connection);
					tokio::spawn(async move { connection.close(None).await })
				})
				.collect();
			for join in joins {
				let _ = join.await;
			}
			live.shared.connections.lock().clear();
			cleanup
		} else {
			Ok(())
		}
	}
}

struct ListenerShared {
	connections: parking_lot::Mutex<Vec<Arc<UnixByteConnection>>>,
	options: ResolvedUnixListenerOptions,
	closing: AtomicBool,
	accept: ByteConnectionAcceptor,
}

fn accept_socket(shared: &Arc<ListenerShared>, stream: tokio::net::UnixStream) {
	if shared.closing.load(Ordering::SeqCst) {
		return;
	}
	let (read_half, write_half) = stream.into_split();
	let connection = Arc::new(UnixByteConnection::new(
		TokioSocket::new(write_half),
		shared.options.graceful_close_timeout_ms,
		shared.options.max_pending_bytes,
	));
	shared.connections.lock().push(Arc::clone(&connection));
	let handler = (shared.accept)(Arc::clone(&connection) as Arc<dyn ByteConnection>);
	let state = Arc::clone(shared);
	tokio::spawn(async move {
		let mut read_half = read_half;
		let mut buffer = vec![0u8; 64 * 1024];
		loop {
			match read_half.read(&mut buffer).await {
				Ok(0) => break,
				Ok(read) => handler.on_data(&buffer[..read]),
				Err(error) => {
					handler.on_error(io_error(error));
					break;
				}
			}
		}
		connection.mark_closed();
		state.connections.lock().retain(|existing| !Arc::ptr_eq(existing, &connection));
		handler.on_close();
	});
}

fn get_owned_bind_path(path: &Path) -> PathBuf {
	// Upstream: sha256(path) hex, first 8 chars. Rust-side digest via a small
	// std-only FNV-1a keeps the shape (unique suffix per path) without a hash
	// crate; the suffix never crosses the wire.
	let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
	for byte in path.to_string_lossy().as_bytes() {
		hash ^= u64::from(*byte);
		hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
	}
	path
		.parent()
		.unwrap_or_else(|| Path::new("."))
		.join(format!("bind-{hash:016x}"))
}

async fn remove_stale_socket(path: &Path) -> Result<(), AnyError> {
	let original = match std::fs::symlink_metadata(path) {
		Ok(original) => original,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
		Err(error) => return Err(io_error(error)),
	};
	if !is_socket_file(&original) {
		return Err(listener_error(format!(
			"Refusing to remove non-socket Unix listener path: {}",
			path.display()
		)));
	}
	if is_socket_live(path).await? {
		return Err(listener_error(format!("Unix listener is already running: {}", path.display())));
	}
	let original_identity = FileIdentity { dev: original.dev(), ino: original.ino() };
	let preserved = path
		.parent()
		.unwrap_or_else(|| Path::new("."))
		.join(format!("stale-{}", &uuid_v4()[..6]));
	if let Err(error) = std::fs::rename(path, &preserved) {
		if error.kind() == std::io::ErrorKind::NotFound {
			return Ok(());
		}
		return Err(io_error(error));
	}
	let current = std::fs::symlink_metadata(&preserved).map_err(io_error)?;
	if !is_socket_file(&current)
		|| current.dev() != original_identity.dev
		|| current.ino() != original_identity.ino
	{
		match std::fs::symlink_metadata(path) {
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
				std::fs::rename(&preserved, path).map_err(io_error)?;
			}
			Err(error) => return Err(io_error(error)),
			Ok(_) => {}
		}
		return Err(listener_error(format!(
			"Unix listener path changed while checking for a stale socket: {}",
			path.display()
		)));
	}
	remove_path(&preserved)?;
	Ok(())
}

fn remove_path(path: &Path) -> Result<(), AnyError> {
	match std::fs::remove_file(path) {
		Ok(()) => Ok(()),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
		Err(error) => Err(io_error(error)),
	}
}

async fn is_socket_live(path: &Path) -> Result<bool, AnyError> {
	match tokio::time::timeout(
		Duration::from_millis(SOCKET_PROBE_TIMEOUT_MS),
		tokio::net::UnixStream::connect(path),
	)
	.await
	{
		// Probe timeout: assume live (upstream resolves true).
		Err(_elapsed) => Ok(true),
		Ok(Ok(_stream)) => Ok(true),
		Ok(Err(error)) => match error.kind() {
			std::io::ErrorKind::ConnectionRefused
			| std::io::ErrorKind::NotFound
			| std::io::ErrorKind::BrokenPipe
			| std::io::ErrorKind::ConnectionReset => Ok(false),
			_ => Err(io_error(error)),
		},
	}
}

fn set_socket_mode(path: &Path, mode: u32) -> Result<(), AnyError> {
	match std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)) {
		Ok(()) => Ok(()),
		// ENOSYS (78) / ENOTSUP (45): filesystems without chmod support.
		Err(error) if matches!(error.raw_os_error(), Some(45) | Some(78)) => Ok(()),
		Err(error) => Err(io_error(error)),
	}
}

async fn cleanup_owned_socket(path: &Path, identity: FileIdentity) -> Result<(), AnyError> {
	let current = match std::fs::symlink_metadata(path) {
		Ok(current) => current,
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
		Err(error) => return Err(io_error(error)),
	};
	if !is_socket_file(&current) || current.dev() != identity.dev || current.ino() != identity.ino {
		return Ok(());
	}
	let preserved = path
		.parent()
		.unwrap_or_else(|| Path::new("."))
		.join(format!("cleanup-{}", &uuid_v4()[..6]));
	if let Err(error) = std::fs::rename(path, &preserved) {
		if error.kind() == std::io::ErrorKind::NotFound {
			return Ok(());
		}
		return Err(io_error(error));
	}
	let moved = std::fs::symlink_metadata(&preserved).map_err(io_error)?;
	if is_socket_file(&moved) && moved.dev() == identity.dev && moved.ino() == identity.ino {
		return remove_path(&preserved);
	}
	match std::fs::symlink_metadata(path) {
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
			std::fs::rename(&preserved, path).map_err(io_error)?;
		}
		Err(error) => return Err(io_error(error)),
		Ok(_) => {}
	}
	Err(listener_error(format!(
		"Unix listener path changed during cleanup; preserved replacement at {}",
		preserved.display()
	)))
}

fn is_socket_file(metadata: &std::fs::Metadata) -> bool {
	use std::os::unix::fs::FileTypeExt;
	metadata.file_type().is_socket()
}

fn listener_error(message: impl Into<String>) -> AnyError {
	Arc::new(pi_chord::ChordError { message: message.into() })
}

#[async_trait::async_trait]
impl ServerListener for UnixListener {
	async fn start(&self, accept: ByteConnectionAcceptor) -> Result<(), AnyError> {
		UnixListener::start(self, accept).await
	}

	async fn close(&self) -> Result<(), AnyError> {
		UnixListener::close(self).await
	}
}
