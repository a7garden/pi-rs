//! Port of `packages/client/src/unix.ts`.
//!
//! Node `net` sockets become `tokio::net::UnixSocket` halves: one reader task
//! feeds [`ByteTransportHandlers`], writes are serialized through a mutex
//! (upstream's `writeTail` promise chain) and bounded by the same pending-byte
//! limit. Discovery keeps the 16-probe concurrency cap and the upstream
//! probe-omission error classes.
//!
//! Deviations:
//! - `readdir`/`lstat` become `read_dir`/`symlink_metadata`; `localeCompare`
//!   ordering becomes byte-wise string ordering.
//! - Upstream's `isErrorCode` cause-chain walk maps to `io::Error`
//!   `raw_os_error()` checks on the boxed error.
//! - Upstream's `setTimeout(...).unref()` race becomes `tokio::time::timeout`
//!   around connect + handshake, with the probe client always disposed.

use std::io;
use std::os::unix::fs::FileTypeExt;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use pi_chord::services::errors::{error, type_error, AnyError};
use pi_protocol::{is_server_id, ProtocolValidationError, DEFAULT_MAX_FRAME_LENGTH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::unix::OwnedWriteHalf;
use tokio::sync::Mutex as AsyncMutex;

use crate::client::connect_client;
use crate::errors::{DisconnectedError, ServerError};
use crate::transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};
use crate::types::ClientOptions;

const DEFAULT_DISCOVERY_TIMEOUT_MS: u64 = 1_000;
const MAX_TIMER_DELAY_MS: u64 = 2_147_483_647;
const UNIX_SOCKET_SUFFIX: &str = ".sock";
const MAX_CONCURRENT_DISCOVERY_PROBES: usize = 16;

// Probe-omission error classes (upstream checks `code` strings on the error
// cause chain); mapped to `io::ErrorKind` for darwin/linux portability.
use io::ErrorKind as IoKind;
const ENOENT: IoKind = IoKind::NotFound;
const EPIPE: IoKind = IoKind::BrokenPipe;
const ECONNRESET: IoKind = IoKind::ConnectionReset;
const ECONNREFUSED: IoKind = IoKind::ConnectionRefused;
const ETIMEDOUT: IoKind = IoKind::TimedOut;

/// Upstream `UnixTransportOptions`.
#[derive(Debug, Clone)]
pub struct UnixTransportOptions {
	pub path: String,
	pub max_pending_bytes: Option<usize>,
}

/// Upstream `UnixServerRoute`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnixServerRoute {
	pub server_id: String,
	pub path: String,
}

/// Upstream `DiscoverUnixServersOptions`.
#[derive(Debug, Clone)]
pub struct DiscoverUnixServersOptions {
	/// Directory containing server-addressed Unix sockets.
	pub directory: String,
	/// Maximum time for each connection and handshake. Defaults to 1,000 ms.
	pub timeout_ms: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
#[error("Unix discovery timed out")]
struct DiscoveryTimeout;

fn is_errno(error: &AnyError, kind: io::ErrorKind) -> bool {
	// Upstream `isErrorCode` walks the `cause` chain; Rust mirrors that with
	// `Error::source`. Kinds map the errnos portably (mac/linux differ).
	let mut current: &(dyn std::error::Error + 'static) = &**error;
	loop {
		if let Some(io_error) = current.downcast_ref::<io::Error>() {
			if io_error.kind() == kind {
				return true;
			}
		}
		current = match current.source() {
			Some(source) => source,
			None => return false,
		};
	}
}

/// Discover reachable local servers by probing server-addressed Unix sockets
/// (upstream `discoverUnixServers`).
pub async fn discover_unix_servers(options: DiscoverUnixServersOptions) -> Result<Vec<UnixServerRoute>, AnyError> {
	if cfg!(windows) {
		return Err(error("Unix transport is not supported on Windows"));
	}
	let timeout_ms = options.timeout_ms.unwrap_or(DEFAULT_DISCOVERY_TIMEOUT_MS);
	if timeout_ms == 0 || timeout_ms > MAX_TIMER_DELAY_MS {
		return Err(type_error(format!(
			"Unix discovery timeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}"
		)));
	}
	let directory = std::path::PathBuf::from(&options.directory);
	let mut entries = match tokio::fs::read_dir(&directory).await {
		Ok(entries) => entries,
		Err(read_error) => {
			if read_error.kind() == io::ErrorKind::NotFound {
				return Ok(Vec::new());
			}
			return Err(Box::new(read_error));
		}
	};
	let mut candidates: Vec<UnixServerRoute> = Vec::new();
	while let Some(entry) = entries.next_entry().await.map_err(|error| Box::new(error) as AnyError)? {
		let Some(name) = entry.file_name().to_str().map(str::to_string) else { continue };
		let Some(server_id) = name.strip_suffix(UNIX_SOCKET_SUFFIX) else { continue };
		if is_server_id(server_id) {
			candidates.push(UnixServerRoute {
				server_id: server_id.to_string(),
				path: directory.join(&name).to_string_lossy().into_owned(),
			});
		}
	}

	let routes = Arc::new(parking_lot::Mutex::new(Vec::new()));
	let failure: Arc<parking_lot::Mutex<Option<AnyError>>> = Arc::new(parking_lot::Mutex::new(None));
	let next_index = Arc::new(parking_lot::Mutex::new(0usize));
	let candidates = Arc::new(candidates);
	let worker_count = MAX_CONCURRENT_DISCOVERY_PROBES.min(candidates.len());
	let mut workers = Vec::with_capacity(worker_count);
	for _ in 0..worker_count {
		let routes = Arc::clone(&routes);
		let failure = Arc::clone(&failure);
		let next_index = Arc::clone(&next_index);
		let candidates = Arc::clone(&candidates);
		workers.push(tokio::spawn(async move {
			loop {
				if failure.lock().is_some() {
					return;
				}
				let candidate = {
					let mut index = next_index.lock();
					let candidate = candidates.get(*index).cloned();
					if candidate.is_some() {
						*index += 1;
					}
					candidate
				};
				let Some(candidate) = candidate else { return };
				match probe_route(&candidate, timeout_ms).await {
					Ok(Some(route)) => routes.lock().push(route),
					Ok(None) => {}
					Err(probe_error) => {
						let mut failure = failure.lock();
						if failure.is_none() {
							*failure = Some(probe_error);
						}
						return;
					}
				}
			}
		}));
	}
	for worker in workers {
		let _ = worker.await;
	}
	if let Some(probe_error) = failure.lock().take() {
		return Err(probe_error);
	}
	let mut routes = std::mem::take(&mut *routes.lock());
	routes.sort_by(|left, right| left.server_id.cmp(&right.server_id));
	Ok(routes)
}

/// One candidate probe: lstat gate (a socket can disappear between readdir and
/// lstat during normal shutdown — ENOENT is skipped, other errors propagate)
/// followed by the handshake probe.
async fn probe_route(candidate: &UnixServerRoute, timeout_ms: u64) -> Result<Option<UnixServerRoute>, AnyError> {
	let metadata = match tokio::fs::symlink_metadata(&candidate.path).await {
		Ok(metadata) => metadata,
		Err(stat_error) => {
			if stat_error.kind() == io::ErrorKind::NotFound {
				return Ok(None);
			}
			return Err(Box::new(stat_error));
		}
	};
	if !metadata.file_type().is_socket() {
		return Ok(None);
	}
	probe_unix_server(candidate, timeout_ms).await
}

/// Upstream `probeUnixServer`: connect + handshake within `timeout_ms`, treating
/// stale/refused/mismatched endpoints as omissions.
pub async fn probe_unix_server(route: &UnixServerRoute, timeout_ms: u64) -> Result<Option<UnixServerRoute>, AnyError> {
	validate_unix_transport_options(&UnixTransportOptions { path: route.path.clone(), max_pending_bytes: None })?;
	let probe = async {
		let factory = create_unix_transport_factory(UnixTransportOptions {
			path: route.path.clone(),
			max_pending_bytes: None,
		})?;
		let client = connect_client(ClientOptions {
			server_id: route.server_id.clone(),
			transport_factory: factory,
			max_frame_length: None,
			on_listener_error: None,
		})
		.await?;
		client.dispose().await;
		Ok::<UnixServerRoute, AnyError>(route.clone())
	};
	match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), probe).await {
		Ok(Ok(route)) => Ok(Some(route)),
		Ok(Err(probe_error)) => {
			// Missing/refused sockets are stale or shutting down. Protocol
			// failures mean the endpoint is not the advertised server. Both
			// are safe to omit.
			let omit = probe_error.is::<DiscoveryTimeout>()
				|| probe_error.downcast_ref::<ProtocolValidationError>().is_some()
				|| probe_error
					.downcast_ref::<DisconnectedError>()
					.map(|disconnected| disconnected.cause.is_none())
					.unwrap_or(false)
				|| probe_error
					.downcast_ref::<ServerError>()
					.map(|server_error| server_error.code == "version")
					.unwrap_or(false)
				|| is_errno(&probe_error, ENOENT)
				|| is_errno(&probe_error, ECONNREFUSED)
				|| is_errno(&probe_error, ECONNRESET)
				|| is_errno(&probe_error, EPIPE)
				|| is_errno(&probe_error, ETIMEDOUT);
			if omit {
				Ok(None)
			} else {
				Err(probe_error)
			}
		}
		Err(_elapsed) => Ok(None),
	}
}

/// Creates fresh Unix-domain socket transports for client connection attempts
/// (upstream `createUnixTransportFactory`).
pub fn create_unix_transport_factory(options: UnixTransportOptions) -> Result<ByteTransportFactory, AnyError> {
	let max_pending_bytes = validate_unix_transport_options(&options)?;
	let path = options.path;
	Ok(Arc::new(move |handlers| {
		let path = path.clone();
		Box::pin(async move { connect_unix_socket(&path, max_pending_bytes as i64, handlers).await })
	}))
}

fn validate_unix_transport_options(options: &UnixTransportOptions) -> Result<usize, AnyError> {
	if options.path.is_empty() {
		return Err(type_error("Unix transport path must not be empty"));
	}
	let max_pending_bytes = options.max_pending_bytes.unwrap_or(DEFAULT_MAX_FRAME_LENGTH * 4);
	if max_pending_bytes == 0 {
		return Err(type_error("Unix transport maxPendingBytes must be a positive safe integer"));
	}
	if cfg!(windows) {
		return Err(error("Unix transport is not supported on Windows"));
	}
	Ok(max_pending_bytes)
}

struct UnixByteTransport {
	writer: Arc<AsyncMutex<OwnedWriteHalf>>,
	pending_bytes: Arc<AtomicI64>,
	max_pending_bytes: i64,
	closed: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl ByteTransport for UnixByteTransport {
	async fn send(&self, chunk: Vec<u8>) -> Result<(), AnyError> {
		if self.closed.load(Ordering::SeqCst) {
			return Err(error("Unix transport is closed"));
		}
		let byte_length = chunk.len() as i64;
		if self.pending_bytes.load(Ordering::SeqCst) + byte_length > self.max_pending_bytes {
			return Err(error("Unix transport exceeded its pending byte limit"));
		}
		self.pending_bytes.fetch_add(byte_length, Ordering::SeqCst);
		// Serialized writes (upstream `writeTail`); the pending count drops
		// once this write leaves the queue.
		let result = async {
			if self.closed.load(Ordering::SeqCst) {
				return Err(error("Unix transport is closed"));
			}
			let mut writer = self.writer.lock().await;
			if self.closed.load(Ordering::SeqCst) {
				return Err(error("Unix transport is closed"));
			}
			writer.write_all(&chunk).await.map_err(|write_error| Box::new(write_error) as AnyError)?;
			Ok(())
		}
		.await;
		self.pending_bytes.fetch_sub(byte_length, Ordering::SeqCst);
		result
	}

	fn close(&self) {
		if self.closed.swap(true, Ordering::SeqCst) {
			return;
		}
		// Shut down the write half; the reader observes EOF. Local closes do
		// not report `onClose` (upstream marks the close terminal first).
		let writer = Arc::clone(&self.writer);
		tokio::spawn(async move {
			writer.lock().await.shutdown().await.ok();
		});
	}
}

async fn connect_unix_socket(
	path: &str,
	max_pending_bytes: i64,
	handlers: ByteTransportHandlers,
) -> Result<Box<dyn ByteTransport>, AnyError> {
	let socket = unix_socket_connect(path).await?;
	let (mut read_half, write_half) = socket.into_split();

	let closed = Arc::new(AtomicBool::new(false));
	let pending_bytes = Arc::new(AtomicI64::new(0));
	let transport = UnixByteTransport {
		writer: Arc::new(AsyncMutex::new(write_half)),
		pending_bytes: Arc::clone(&pending_bytes),
		max_pending_bytes,
		closed: Arc::clone(&closed),
	};

	// Reader task: feeds inbound chunks and reports the terminal event.
	let reader_handlers = handlers;
	let reader_closed = Arc::clone(&closed);
	tokio::spawn(async move {
		let mut buffer = vec![0u8; 64 * 1024];
		loop {
			match read_half.read(&mut buffer).await {
				Ok(0) | Err(_) => break,
				Ok(read) => {
					(reader_handlers.on_data)(buffer[..read].to_vec());
					if reader_closed.load(Ordering::SeqCst) {
						return;
					}
				}
			}
		}
		if !reader_closed.swap(true, Ordering::SeqCst) {
			(reader_handlers.on_close)();
		}
	});

	Ok(Box::new(transport) as Box<dyn ByteTransport>)
}

async fn unix_socket_connect(path: &str) -> Result<tokio::net::UnixStream, AnyError> {
	let socket = tokio::net::UnixSocket::new_stream().map_err(|error| Box::new(error) as AnyError)?;
	let stream = socket.connect(path).await.map_err(|connect_error| Box::new(connect_error) as AnyError)?;
	Ok(stream)
}

#[cfg(test)]
mod tests {
	//! Ports of `packages/client/test/unix-transport.test.ts` and
	//! `packages/client/test/unix.test.ts`.
	//!
	//! The upstream discovery tests drive a real `pi-server` runtime; this
	//! crate cannot depend on pi-server (PORT_MAP crate graph), so the tests
	//! use minimal probe-answering socket servers that exercise the same
	//! observable client/discovery behavior.

	use std::path::{Path, PathBuf};
	use std::sync::atomic::{AtomicUsize, Ordering};

	use pi_protocol::{encode_server_message, ClientMessageDecoder, PROTOCOL_VERSION};
	use serde_json::json;
	use tempfile::TempDir;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	use tokio::net::{UnixListener, UnixStream};

	use super::*;
	use crate::client::Client;
	use crate::test_support::DEFAULT_SERVER_ID;
	use crate::types::{ConnectionState, RpcTarget};

	const SERVER_ID: &str = DEFAULT_SERVER_ID;

	fn socket_id(value: usize) -> String {
		format!("00000000-0000-4000-8000-{value:012x}")
	}

	fn make_socket_path() -> (TempDir, PathBuf) {
		let directory = tempfile::Builder::new().prefix("pi-client-transport-").tempdir().unwrap();
		let path = directory.path().join("pi.sock");
		(directory, path)
	}

	fn make_directory() -> TempDir {
		tempfile::Builder::new().prefix("pc-").tempdir().unwrap()
	}

	/// Minimal server that answers the client hello like `RuntimeServer` does
	/// for probe connections (upstream's test used the real server).
	async fn start_probe_server(path: &Path, reported_server_id: &str) -> tokio::task::JoinHandle<()> {
		let listener = UnixListener::bind(path).expect("bind probe server");
		let reported_server_id = reported_server_id.to_string();
		tokio::spawn(async move {
			loop {
				let Ok((mut stream, _)) = listener.accept().await else { return };
				let reported = reported_server_id.clone();
				tokio::spawn(async move {
					let mut decoder = ClientMessageDecoder::new(None).unwrap();
					let mut buffer = [0u8; 4096];
					loop {
						match stream.read(&mut buffer).await {
							Ok(0) | Err(_) => return,
							Ok(read) => {
								let Ok(messages) = decoder.push(&buffer[..read]) else { return };
								for message in messages {
									if message.get("type").and_then(|value| value.as_str()) == Some("hello") {
										let frame = encode_server_message(
											&json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": reported }),
											None,
										)
										.unwrap();
										if stream.write_all(&frame).await.is_err() {
											return;
										}
									}
								}
							}
						}
					}
				});
			}
		})
	}

	struct ConnectionCounters {
		active: AtomicUsize,
		maximum: AtomicUsize,
		total: AtomicUsize,
	}

	/// Silent socket: accepts connections but never speaks (upstream
	/// `startSilentSocket`).
	async fn start_silent_socket(path: &Path, counters: Option<Arc<ConnectionCounters>>) -> tokio::task::JoinHandle<()> {
		let listener = UnixListener::bind(path).expect("bind silent socket");
		tokio::spawn(async move {
			loop {
				let Ok((stream, _)) = listener.accept().await else { return };
				if let Some(counters) = &counters {
					let active = counters.active.fetch_add(1, Ordering::SeqCst) + 1;
					counters.maximum.fetch_max(active, Ordering::SeqCst);
					counters.total.fetch_add(1, Ordering::SeqCst);
					let counters = Arc::clone(counters);
					tokio::spawn(async move {
						drop_silently(stream).await;
						counters.active.fetch_sub(1, Ordering::SeqCst);
					});
				} else {
					tokio::spawn(drop_silently(stream));
				}
			}
		})
	}

	async fn drop_silently(mut stream: UnixStream) {
		let mut buffer = [0u8; 1024];
		while let Ok(read) = stream.read(&mut buffer).await {
			if read == 0 {
				return;
			}
		}
	}

	// ── upstream unix-transport.test.ts ─────────────────────────────────────

	#[test]
	fn rejects_invalid_unix_transport_options() {
		let empty = create_unix_transport_factory(UnixTransportOptions { path: String::new(), max_pending_bytes: None });
		let Err(empty_error) = empty else { panic!("empty path rejected") };
		assert!(empty_error.to_string().contains("must not be empty"));

		let zero = create_unix_transport_factory(UnixTransportOptions {
			path: "/tmp/pi.sock".into(),
			max_pending_bytes: Some(0),
		});
		let Err(zero_error) = zero else { panic!("zero maxPendingBytes rejected") };
		assert!(zero_error.to_string().contains("positive"));
	}

	#[tokio::test]
	async fn carries_a_complete_client_handshake_and_request_over_a_real_unix_socket() {
		let (_dir, path) = make_socket_path();
		let received_members: Arc<parking_lot::Mutex<Vec<String>>> = Arc::default();
		let listener = UnixListener::bind(&path).unwrap();
		let members = Arc::clone(&received_members);
		tokio::spawn(async move {
			loop {
				let Ok((mut stream, _)) = listener.accept().await else { return };
				let members = Arc::clone(&members);
				tokio::spawn(async move {
					let mut decoder = ClientMessageDecoder::new(None).unwrap();
					let mut buffer = [0u8; 4096];
					loop {
						match stream.read(&mut buffer).await {
							Ok(0) | Err(_) => return,
							Ok(read) => {
								let Ok(messages) = decoder.push(&buffer[..read]) else { return };
								for message in messages {
									match message.get("type").and_then(|value| value.as_str()) {
										Some("hello") => {
											let frame = encode_server_message(
												&json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": SERVER_ID }),
												None,
											)
											.unwrap();
											// Byte-by-byte write exercises reassembly.
											for byte in &frame {
												if stream.write_all(std::slice::from_ref(byte)).await.is_err() {
													return;
												}
											}
										}
										Some("cancel") => {}
										_ => {
											let Ok(call) = pi_chord::parse_service_call(&message["call"]) else { return };
											members.lock().push(format!("{}.{}", call.service_id, call.member));
											let frame = encode_server_message(
												&json!({ "type": "response", "id": message["id"], "ok": true, "result": [] }),
												None,
											)
											.unwrap();
											let split = frame.len() / 2;
											if stream.write_all(&frame[..split]).await.is_err() {
												return;
											}
											if stream.write_all(&frame[split..]).await.is_err() {
												return;
											}
										}
									}
								}
							}
						}
					}
				});
			}
		});

		let client = Client::new(ClientOptions {
			server_id: SERVER_ID.into(),
			transport_factory: create_unix_transport_factory(UnixTransportOptions {
				path: path.to_string_lossy().into_owned(),
				max_pending_bytes: None,
			})
			.unwrap(),
			max_frame_length: None,
			on_listener_error: None,
		})
		.unwrap();

		let hello = client.connect().await.expect("handshake over unix socket");
		assert_eq!(hello.server_id, SERVER_ID);
		let result = client
			.request(
				&RpcTarget::Server { server_id: SERVER_ID.into() },
				&pi_chord::ServiceCall { service_id: "test.server".into(), instance: None, member: "list".into(), args: vec![] },
				None,
			)
			.await
			.expect("request over unix socket");
		assert_eq!(result, Some(json!([])));
		assert_eq!(*received_members.lock(), vec!["test.server.list".to_string()]);
		client.dispose().await;
	}

	#[tokio::test]
	async fn reports_truncated_final_frames_through_client() {
		let (_dir, path) = make_socket_path();
		let listener = UnixListener::bind(&path).unwrap();
		tokio::spawn(async move {
			loop {
				let Ok((mut stream, _)) = listener.accept().await else { return };
				tokio::spawn(async move {
					let mut decoder = ClientMessageDecoder::new(None).unwrap();
					let mut buffer = [0u8; 4096];
					loop {
						match stream.read(&mut buffer).await {
							Ok(0) | Err(_) => return,
							Ok(read) => {
								let Ok(messages) = decoder.push(&buffer[..read]) else { return };
								for message in messages {
									if message.get("type").and_then(|value| value.as_str()) == Some("hello") {
										let frame = encode_server_message(
											&json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": SERVER_ID }),
											None,
										)
										.unwrap();
										if stream.write_all(&frame).await.is_err() {
											return;
										}
									} else {
										let _ = stream.write_all(&[0, 0, 0, 2, 1]).await;
										let _ = stream.shutdown().await;
										return;
									}
								}
							}
						}
					}
				});
			}
		});
		let client = Client::new(ClientOptions {
			server_id: SERVER_ID.into(),
			transport_factory: create_unix_transport_factory(UnixTransportOptions {
				path: path.to_string_lossy().into_owned(),
				max_pending_bytes: None,
			})
			.unwrap(),
			max_frame_length: None,
			on_listener_error: None,
		})
		.unwrap();

		client.connect().await.expect("handshake");
		let failure = client
			.request(
				&RpcTarget::Server { server_id: SERVER_ID.into() },
				&pi_chord::ServiceCall { service_id: "test.server".into(), instance: None, member: "list".into(), args: vec![] },
				None,
			)
			.await
			.err()
			.expect("truncated frame rejects");
		let Some(validation_error) = failure.downcast_ref::<ProtocolValidationError>() else { panic!("ProtocolValidationError expected") };
		assert!(validation_error.to_string().to_lowercase().contains("truncated"));
		assert_eq!(client.connection_state(), ConnectionState::Disconnected);
		client.dispose().await;
	}

	#[tokio::test]
	async fn rejects_connection_attempts_to_missing_sockets() {
		let (_dir, path) = make_socket_path();
		let factory = create_unix_transport_factory(UnixTransportOptions {
			path: path.to_string_lossy().into_owned(),
			max_pending_bytes: None,
		})
		.unwrap();
		let handlers = ByteTransportHandlers {
			on_data: Arc::new(|_| {}),
			on_close: Arc::new(|| {}),
			on_error: Arc::new(|_| {}),
		};
		let failure = (factory)(handlers).await.err().expect("missing socket rejected");
		assert!(is_errno(&failure, ENOENT));
	}

	// ── upstream unix.test.ts (discoverUnixServers) ─────────────────────────

	#[tokio::test]
	async fn returns_no_routes_when_the_server_directory_is_missing() {
		let directory = make_directory();
		let missing = directory.path().join("missing");
		let routes = discover_unix_servers(DiscoverUnixServersOptions {
			directory: missing.to_string_lossy().into_owned(),
			timeout_ms: None,
		})
		.await
		.unwrap();
		assert!(routes.is_empty());
	}

	#[tokio::test]
	async fn discovers_reachable_servers_in_server_id_order() {
		let directory = make_directory();
		let first = socket_id(1);
		let second = socket_id(2);
		let _second = start_probe_server(&directory.path().join(format!("{second}.sock")), &second).await;
		let _first = start_probe_server(&directory.path().join(format!("{first}.sock")), &first).await;

		let routes = discover_unix_servers(DiscoverUnixServersOptions {
			directory: directory.path().to_string_lossy().into_owned(),
			timeout_ms: None,
		})
		.await
		.unwrap();
		assert_eq!(
			routes,
			vec![
				UnixServerRoute { server_id: first.clone(), path: directory.path().join(format!("{first}.sock")).to_string_lossy().into_owned() },
				UnixServerRoute { server_id: second.clone(), path: directory.path().join(format!("{second}.sock")).to_string_lossy().into_owned() },
			]
		);
	}

	#[tokio::test]
	async fn ignores_malformed_entries_non_sockets_and_mismatched_servers() {
		let directory = make_directory();
		std::fs::write(directory.path().join(format!("{}.sock", socket_id(1))), "not a socket").unwrap();
		std::fs::write(directory.path().join("not-a-server.sock"), "ignored").unwrap();
		std::fs::create_dir(directory.path().join(format!("{}.sock", socket_id(2)))).unwrap();
		let _mismatched =
			start_probe_server(&directory.path().join(format!("{}.sock", socket_id(3))), &socket_id(4)).await;

		let routes = discover_unix_servers(DiscoverUnixServersOptions {
			directory: directory.path().to_string_lossy().into_owned(),
			timeout_ms: None,
		})
		.await
		.unwrap();
		assert!(routes.is_empty());
	}

	#[tokio::test]
	async fn ignores_stale_sockets_without_deleting_them() {
		let directory = make_directory();
		let id = socket_id(1);
		let path = directory.path().join(format!("{id}.sock"));
		// Bind and immediately drop the listener: the socket file survives but
		// connections are refused (upstream kills a forked stale-socket child).
		let stale = std::os::unix::net::UnixListener::bind(&path).unwrap();
		drop(stale);

		let routes = discover_unix_servers(DiscoverUnixServersOptions {
			directory: directory.path().to_string_lossy().into_owned(),
			timeout_ms: None,
		})
		.await
		.unwrap();
		assert!(routes.is_empty());
		assert!(path.symlink_metadata().unwrap().file_type().is_socket());
	}

	#[tokio::test]
	async fn times_out_an_unresponsive_socket_without_deleting_it() {
		let directory = make_directory();
		let id = socket_id(1);
		let path = directory.path().join(format!("{id}.sock"));
		let _silent = start_silent_socket(&path, None).await;

		let routes = discover_unix_servers(DiscoverUnixServersOptions {
			directory: directory.path().to_string_lossy().into_owned(),
			timeout_ms: Some(20),
		})
		.await
		.unwrap();
		assert!(routes.is_empty());
		assert!(path.symlink_metadata().unwrap().file_type().is_socket());
	}

	#[tokio::test]
	async fn limits_concurrent_probes_to_16() {
		let directory = make_directory();
		let counters = Arc::new(ConnectionCounters {
			active: AtomicUsize::new(0),
			maximum: AtomicUsize::new(0),
			total: AtomicUsize::new(0),
		});
		let mut silent = Vec::new();
		for index in 1..=20usize {
			silent.push(
				start_silent_socket(&directory.path().join(format!("{}.sock", socket_id(index))), Some(Arc::clone(&counters)))
					.await,
			);
		}

		// Rust futures are lazy; drive discovery concurrently while polling the
		// connection counters (upstream starts the promise eagerly).
		let discovery = tokio::spawn(discover_unix_servers(DiscoverUnixServersOptions {
			directory: directory.path().to_string_lossy().into_owned(),
			timeout_ms: Some(100),
		}));
		for _ in 0..400 {
			if counters.active.load(Ordering::SeqCst) == 16 {
				break;
			}
			tokio::time::sleep(std::time::Duration::from_millis(5)).await;
		}
		assert_eq!(counters.active.load(Ordering::SeqCst), 16);
		assert_eq!(counters.maximum.load(Ordering::SeqCst), 16);
		let routes = discovery.await.unwrap().unwrap();
		assert!(routes.is_empty());
		assert_eq!(counters.total.load(Ordering::SeqCst), 20);
	}

	#[tokio::test]
	async fn ignores_an_endpoint_that_closes_before_its_handshake() {
		let directory = make_directory();
		let id = socket_id(1);
		let path = directory.path().join(format!("{id}.sock"));
		let listener = UnixListener::bind(&path).unwrap();
		tokio::spawn(async move {
			loop {
				let Ok((stream, _)) = listener.accept().await else { return };
				drop(stream);
			}
		});

		let routes = discover_unix_servers(DiscoverUnixServersOptions {
			directory: directory.path().to_string_lossy().into_owned(),
			timeout_ms: None,
		})
		.await
		.unwrap();
		assert!(routes.is_empty());
	}

	#[tokio::test]
	async fn propagates_unexpected_filesystem_errors() {
		let directory = make_directory();
		let file = directory.path().join("not-a-directory");
		std::fs::write(&file, "content").unwrap();

		let failure = discover_unix_servers(DiscoverUnixServersOptions {
			directory: file.to_string_lossy().into_owned(),
			timeout_ms: None,
		})
		.await
		.err()
		.expect("filesystem error propagates");
		assert_eq!(failure.downcast_ref::<std::io::Error>().unwrap().raw_os_error(), Some(20)); // ENOTDIR
	}
}
