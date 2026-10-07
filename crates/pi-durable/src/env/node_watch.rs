//! Port of `packages/durable/src/env/node-watch.ts` — the local `FileWatcher`.
//!
//! Watches by snapshots: native events only trigger a debounced rescan, and changes are the
//! difference between snapshots plus the event paths. A replaced file, a renamed or recreated
//! ancestor, or a directory created with its contents therefore never depends on which events
//! an operating system sends. New directories get their watchers before they are scanned
//! again, so nothing written into them before the watcher existed is missed.
//!
//! The worker runs on a dedicated OS thread (upstream relies on libuv's thread pool and
//! timers); `close` signals it and joins the thread, so no callback starts after `close`
//! resolves. Snapshot scans use `std::fs` directly: the worker thread has no async runtime.
//!
//! Deviations:
//! - D1: Node's `fs.watch` maps to the `notify` crate (v7). `notify::RecommendedWatcher`
//!   already provides inotify/kqueue/FSEvents/ReadDirectoryChangesW, including recursive
//!   watches everywhere, so the upstream Linux-only "one watcher per directory" fallback is
//!   unnecessary: a recursive root watcher per target is installed on every platform.
//! - D2: `statfs`-based detection of file systems that do not report remote changes (NFS,
//!   SMB, FUSE, …) needs `libc` on Linux, which is not a dependency of this crate; the check
//!   is deferred, so auto mode polls only on Windows. Callers on unreliable file systems
//!   should pass `mode: Some(WatchMode::Polling)` explicitly.
//! - D3: the polling-mode content hash of recently modified small files needs a SHA-256
//!   dependency upstream (`node:crypto`) that this crate does not have; hash comparison is
//!   omitted, so a same-size same-timestamp rewrite within the file system's timestamp
//!   granularity can be missed in polling mode. All other snapshot comparisons are kept.
//! - D4: upstream's `#scheduleSettle` macOS rescan (FSEvents stream liveness) is kept as a
//!   one-shot delayed flush 500 ms after watchers are (re)installed.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use async_trait::async_trait;
use pi_chord::context::Context;

use crate::env::{
	err, FileError, FileErrorCode, FileKind, FileWatcher, WatchCallback, WatchChange, WatchMode, WatchTarget,
};

const DEBOUNCE_MS: u64 = 50;
/// On macOS, a rescan this long after installing watchers catches changes made before the
/// FSEvents stream went live (see D4).
const FSEVENTS_SETTLE_MS: u64 = 500;
const DEFAULT_POLL_MS: u64 = 2000;
const DEFAULT_MAX_DIRECTORIES: usize = 10_000;
/// Upper bound of rescan rounds triggered by newly installed watchers (upstream: 10 rounds).
const MAX_SYNC_ROUNDS: u32 = 10;

/// How `NodeExecutionEnv` watches.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeWatchOptions {
	/// Force a mode. By default `polling` is chosen on Windows, where native watchers keep
	/// directories open and so block renaming their parents (see also D2).
	pub mode: Option<WatchMode>,
	/// Interval between snapshots in `polling` mode; default 2000 ms.
	pub poll_interval_ms: Option<u64>,
	/// Most directories one watcher covers; default 10,000.
	pub max_directories: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
	File,
	Directory,
	Symlink,
	Other,
}

impl EntryKind {
	fn from_metadata(metadata: &std::fs::Metadata) -> Self {
		if metadata.is_file() {
			Self::File
		} else if metadata.is_dir() {
			Self::Directory
		} else if metadata.file_type().is_symlink() {
			Self::Symlink
		} else {
			Self::Other
		}
	}

	fn from_kind(kind: FileKind) -> Self {
		match kind {
			FileKind::File => Self::File,
			FileKind::Directory => Self::Directory,
			FileKind::Symlink => Self::Symlink,
		}
	}
}

/// What a snapshot remembers of one path. Directories and ancestors are compared by identity only.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
	kind: EntryKind,
	dev: u64,
	ino: u64,
	size: u64,
	mtime_ms: u64,
}

#[derive(Debug, Clone)]
struct ResolvedTarget {
	path: String,
	recursive: bool,
	hidden: bool,
	names: HashSet<String>,
}

fn same_entry(a: &Entry, b: &Entry) -> bool {
	a.kind == b.kind && a.dev == b.dev && a.ino == b.ino && a.size == b.size && a.mtime_ms == b.mtime_ms
}

/// Ancestors of `path`, nearest first, up to the root.
fn ancestors_of(path: &str) -> Vec<String> {
	let mut result = Vec::new();
	let mut current = Path::new(path);
	while let Some(parent) = current.parent() {
		result.push(parent.to_string_lossy().into_owned());
		if parent == Path::new("/") || parent.as_os_str().is_empty() {
			break;
		}
		current = parent;
	}
	result
}

/// Whether `path` is `ancestor` or below it.
fn is_within(path: &str, ancestor: &str) -> bool {
	path == ancestor || path.starts_with(if ancestor.ends_with('/') { ancestor.to_string() } else { format!("{ancestor}/") }.as_str())
}

fn excluded(target: &ResolvedTarget, name: &str) -> bool {
	(target.hidden && name.starts_with('.')) || target.names.contains(name)
}

fn join_dir(dir: &str, name: &str) -> String {
	if dir.ends_with('/') {
		format!("{dir}{name}")
	} else {
		format!("{dir}/{name}")
	}
}

fn dev_ino(metadata: &std::fs::Metadata) -> (u64, u64) {
	#[cfg(unix)]
	{
		use std::os::unix::fs::MetadataExt as _;
		(metadata.dev(), metadata.ino())
	}
	#[cfg(windows)]
	{
		// Windows has no stable file identity through std; volume+index needs winapi bindings.
		// Identity falls back to zero, making directory identity comparison exact for
		// kind/size/mtime only. See the D-ledger in `node.rs` (D2) for platform notes.
		let _ = metadata;
		(0, 0)
	}
}

fn mtime_ms(metadata: &std::fs::Metadata) -> u64 {
	metadata
		.modified()
		.ok()
		.and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
		.map(|since| since.as_millis() as u64)
		.unwrap_or(0)
}

fn entry_of(metadata: &std::fs::Metadata) -> Entry {
	let kind = EntryKind::from_metadata(metadata);
	let (dev, ino) = dev_ino(metadata);
	Entry {
		kind,
		dev,
		ino,
		size: if kind == EntryKind::Directory { 0 } else { metadata.len() },
		mtime_ms: if kind == EntryKind::Directory { 0 } else { mtime_ms(metadata) },
	}
}

enum ScanError {
	/// The watched tree grew past the environment's directory budget.
	Budget(String),
	Io(std::io::Error),
}

impl From<std::io::Error> for ScanError {
	fn from(error: std::io::Error) -> Self {
		Self::Io(error)
	}
}

enum Message {
	Notify(notify::Event),
	Stop,
}

struct Worker {
	targets: Vec<ResolvedTarget>,
	established: Arc<AtomicBool>,
	on_change: WatchCallback,
	poll_interval: Duration,
	max_directories: usize,
	closed: Arc<AtomicBool>,
	mode: WatchMode,
	snapshot: HashMap<String, Entry>,
	watchers: HashMap<String, notify::RecommendedWatcher>,
	/// Channel shared with the notify callbacks, so newly installed watchers feed the loop.
	events_tx: Sender<Message>,
	events: BTreeSet<String>,
	/// Deadline of the pending debounced flush, if one is scheduled.
	debounce: Option<Instant>,
	/// One-shot delayed flush after watchers were installed (macOS FSEvents settle, D4).
	settle: Option<Instant>,
}

impl Worker {
	fn deliver(&self, change: WatchChange) {
		if self.closed.load(Ordering::SeqCst) {
			return;
		}
		// A panicking callback must not stop watching.
		let callback = self.on_change.clone();
		let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(change)));
	}

	fn schedule_debounce(&mut self) {
		if self.debounce.is_none() {
			self.debounce = Some(Instant::now() + Duration::from_millis(DEBOUNCE_MS));
		}
	}

	/// Rescan, report differences, and install watchers for new directories, rescanning until
	/// none are new. Returns the set of changed paths to report.
	fn sync(&mut self, mut report: bool) -> Result<BTreeSet<String>, ScanError> {
		let mut changed = BTreeSet::new();
		for _round in 0..MAX_SYNC_ROUNDS {
			if self.closed.load(Ordering::SeqCst) {
				break;
			}
			let next = self.scan()?;
			if report {
				for path in self.diff(&self.snapshot, &next) {
					changed.insert(path);
				}
			}
			self.snapshot = next;
			if self.mode == WatchMode::Polling || !self.reconcile_watchers()? {
				break;
			}
			if cfg!(target_os = "macos") {
				self.settle = Some(Instant::now() + Duration::from_millis(FSEVENTS_SETTLE_MS));
			}
			// Something written into a new directory before its watcher existed shows up in the next round.
			report = true;
		}
		Ok(changed)
	}

	fn diff(&self, previous: &HashMap<String, Entry>, next: &HashMap<String, Entry>) -> Vec<String> {
		let mut changed = Vec::new();
		for (path, entry) in next {
			match previous.get(path) {
				Some(before) if same_entry(before, entry) => {}
				_ => changed.push(self.reported(path)),
			}
		}
		for path in previous.keys() {
			if !next.contains_key(path) {
				changed.push(self.reported(path));
			}
		}
		changed
	}

	/// An ancestor that changed identity moved every target below it; report those targets.
	fn reported(&self, path: &str) -> String {
		if self.targets.iter().any(|target| is_within(path, &target.path)) {
			return path.to_string();
		}
		self.targets
			.iter()
			.find(|target| is_within(&target.path, path))
			.map(|target| target.path.clone())
			.unwrap_or_else(|| path.to_string())
	}

	fn scan(&mut self) -> Result<HashMap<String, Entry>, ScanError> {
		let mut snapshot = HashMap::new();
		let mut directories = 0usize;
		for target in &self.targets {
			for ancestor in ancestors_of(&target.path) {
				if snapshot.contains_key(&ancestor) {
					continue;
				}
				// Identity only: an ancestor's own timestamps change with every unrelated sibling.
				if let Ok(metadata) = std::fs::symlink_metadata(&ancestor) {
					let mut entry = entry_of(&metadata);
					entry.size = 0;
					entry.mtime_ms = 0;
					snapshot.insert(ancestor, entry);
				}
			}
			// The target itself may be a symbolic link to what is watched; follow it.
			let Ok(metadata) = std::fs::metadata(&target.path) else {
				continue;
			};
			snapshot.insert(target.path.clone(), entry_of(&metadata));
			if metadata.is_dir() {
				directories += 1;
				if directories > self.max_directories {
					return Err(ScanError::Budget(format!(
						"Watched paths exceed {} directories",
						self.max_directories
					)));
				}
				self.scan_directory(target, &target.path, &mut snapshot, &mut directories)?;
			}
		}
		Ok(snapshot)
	}

	fn scan_directory(
		&self,
		target: &ResolvedTarget,
		directory: &str,
		snapshot: &mut HashMap<String, Entry>,
		directories: &mut usize,
	) -> Result<(), ScanError> {
		let entries = match std::fs::read_dir(directory) {
			Ok(entries) => entries,
			// A directory that disappeared or is unreadable is simply absent from the snapshot.
			Err(error)
				if matches!(
					error.kind(),
					std::io::ErrorKind::NotFound
						| std::io::ErrorKind::PermissionDenied
						| std::io::ErrorKind::NotADirectory
				) =>
			{
				return Ok(())
			}
			Err(error) => return Err(error.into()),
		};
		for entry in entries {
			let Ok(entry) = entry else { continue };
			let name = entry.file_name().to_string_lossy().into_owned();
			if excluded(target, &name) {
				continue;
			}
			let path = join_dir(directory, &name);
			if snapshot.contains_key(&path) {
				continue;
			}
			// Do not follow symlinks below a recursive target (upstream `lstat`).
			let Ok(metadata) = std::fs::symlink_metadata(&path) else { continue };
			snapshot.insert(path.clone(), entry_of(&metadata));
			if target.recursive && metadata.is_dir() {
				*directories += 1;
				if *directories > self.max_directories {
					return Err(ScanError::Budget(format!(
						"Watched paths exceed {} directories",
						self.max_directories
					)));
				}
				self.scan_directory(target, &path, snapshot, directories)?;
			}
		}
		Ok(())
	}

	/// Watch every existing ancestor of each target and each target directory (a recursive
	/// root watcher per target; see D1). Returns whether a watcher was added.
	fn reconcile_watchers(&mut self) -> Result<bool, ScanError> {
		let mut wanted: Vec<(String, bool)> = Vec::new();
		for target in &self.targets {
			for ancestor in ancestors_of(&target.path) {
				if self.snapshot.get(&ancestor).map(|entry| entry.kind) == Some(EntryKind::Directory)
					&& !wanted.iter().any(|(path, _)| *path == ancestor)
				{
					wanted.push((ancestor, false));
				}
			}
			if self.snapshot.get(&target.path).map(|entry| entry.kind) != Some(EntryKind::Directory) {
				continue;
			}
			if !wanted.iter().any(|(path, _)| *path == target.path) {
				wanted.push((target.path.clone(), target.recursive));
			}
		}
		let wanted_set: HashSet<&str> = wanted.iter().map(|(path, _)| path.as_str()).collect();
		self.watchers.retain(|path, _| wanted_set.contains(path.as_str()));
		let mut added = false;
		for (path, recursive) in wanted {
			if self.watchers.contains_key(&path) {
				continue;
			}
			match install_watcher(&path, recursive, &self.events_tx) {
				Ok(watcher) => {
					self.watchers.insert(path, watcher);
					added = true;
				}
				// A directory that disappeared: rescan, which also installs the watcher.
				Err(notify::Error { kind: notify::ErrorKind::PathNotFound, .. }) => {}
				Err(notify::Error { kind: notify::ErrorKind::Io(error), .. })
					if matches!(error.kind(), std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied) => {}
				// Out of watches or unsupported: compare snapshots from now on, and say
				// coverage was uncertain.
				Err(_) => {
					self.switch_to_polling();
					return Ok(false);
				}
			}
		}
		Ok(added)
	}

	fn switch_to_polling(&mut self) {
		if self.mode == WatchMode::Polling {
			return;
		}
		self.mode = WatchMode::Polling;
		self.watchers.clear();
		self.deliver(WatchChange::Overflow);
		self.debounce = None;
	}

	/// One flush cycle: rescan, union the diff with collected event paths, deliver.
	fn flush(&mut self) {
		loop {
			if self.closed.load(Ordering::SeqCst) {
				return;
			}
			let result = self.sync(true);
			let events: Vec<String> = std::mem::take(&mut self.events).into_iter().collect();
			match result {
				Ok(changed) => {
					let mut paths: Vec<String> = changed.into_iter().collect();
					for path in events {
						if !paths.contains(&path) {
							paths.push(path);
						}
					}
					if !paths.is_empty() {
						paths.sort();
						self.deliver(WatchChange::Paths(paths));
					}
				}
				Err(ScanError::Budget(message)) => {
					self.deliver(WatchChange::Error(FileError::new(FileErrorCode::Invalid, message, None)));
					self.stop();
					return;
				}
				Err(ScanError::Io(error)) => {
					self.deliver(WatchChange::Error(FileError::new(FileErrorCode::Invalid, error.to_string(), None)));
					self.stop();
					return;
				}
			}
			// notify events that arrived while scanning are already in `self.events`; the
			// worker loop reschedules a flush for them, so one pass here is enough.
			return;
		}
	}

	fn stop(&mut self) {
		self.closed.store(true, Ordering::SeqCst);
		self.watchers.clear();
	}

	fn on_event(&mut self, directory: &str, filename: Option<&str>) {
		if self.closed.load(Ordering::SeqCst) {
			return;
		}
		let path = match filename {
			Some(name) => join_dir(directory, name),
			None => directory.to_string(),
		};
		let relevant = self.in_scope(&path);
		if relevant {
			self.events.insert(self.reported(&path));
		}
		if relevant || filename.is_none() {
			self.schedule_debounce();
		}
	}

	fn in_scope(&self, path: &str) -> bool {
		for target in &self.targets {
			if is_within(&target.path, path) {
				return true;
			}
			if !is_within(path, &target.path) || path == target.path {
				continue;
			}
			let components: Vec<&str> = path[target.path.len() + 1..].split('/').collect();
			if !target.recursive && components.len() > 1 {
				continue;
			}
			if components.iter().any(|name| excluded(target, name)) {
				continue;
			}
			return true;
		}
		false
	}

	/// The worker loop: establish coverage with the initial scan, then debounced flushes on
	/// native events, periodic scans while polling, and the one-shot macOS settle flush (D4).
	fn run(mut self, control: &Receiver<Message>) {
		match self.sync(false) {
			Ok(_) => self.established.store(true, Ordering::SeqCst),
			Err(ScanError::Budget(message)) => {
				self.deliver(WatchChange::Error(FileError::new(FileErrorCode::Invalid, message, None)));
				self.stop();
				return;
			}
			Err(ScanError::Io(error)) => {
				self.deliver(WatchChange::Error(FileError::new(FileErrorCode::Invalid, error.to_string(), None)));
				self.stop();
				return;
			}
		}
		loop {
			if self.closed.load(Ordering::SeqCst) {
				return;
			}
			let now = Instant::now();
			match self.next_deadline(now) {
				Some(at) if at > now => match control.recv_timeout(at - now) {
					Ok(Message::Stop) | Err(RecvTimeoutError::Disconnected) => {
						self.stop();
						return;
					}
					Ok(Message::Notify(event)) => self.absorb(event),
					Err(RecvTimeoutError::Timeout) => {}
				},
				Some(_) => {}
				None => match control.recv() {
					Ok(Message::Notify(event)) => self.absorb(event),
					Ok(Message::Stop) | Err(_) => {
						self.stop();
						return;
					}
				},
			}
			let now = Instant::now();
			if self.debounce.is_some_and(|at| now >= at) {
				self.debounce = None;
				self.flush();
			} else if self.settle.is_some_and(|at| now >= at) {
				self.settle = None;
				self.flush();
			} else if self.mode == WatchMode::Polling {
				// Poll tick: the deadline was the next poll time.
				self.flush();
			}
		}
	}

	fn next_deadline(&self, now: Instant) -> Option<Instant> {
		let mut deadline: Option<Instant> = None;
		if self.mode == WatchMode::Polling {
			deadline = Some(now + self.poll_interval);
		}
		if let Some(at) = self.debounce {
			deadline = Some(deadline.map_or(at, |current| current.min(at)));
		}
		if let Some(at) = self.settle {
			deadline = Some(deadline.map_or(at, |current| current.min(at)));
		}
		deadline
	}

	fn absorb(&mut self, event: notify::Event) {
		for path in event.paths {
			let path = path.to_string_lossy().into_owned();
			self.on_event(&path, None);
		}
	}
}

fn install_watcher(
	path: &str,
	recursive: bool,
	tx: &Sender<Message>,
) -> Result<notify::RecommendedWatcher, notify::Error> {
	let tx = tx.clone();
	let mut watcher = notify::recommended_watcher(move |result: Result<notify::Event, notify::Error>| {
		if let Ok(event) = result {
			let _ = tx.send(Message::Notify(event));
		}
	})?;
	let mode = if recursive { notify::RecursiveMode::Recursive } else { notify::RecursiveMode::NonRecursive };
	use notify::Watcher as _;
	watcher.watch(Path::new(path), mode)?;
	Ok(watcher)
}

/// Upstream `NodeFileWatcher`.
pub struct NodeFileWatcher {
	mode: WatchMode,
	closed: Arc<AtomicBool>,
	established: Arc<AtomicBool>,
	handle: StdMutex<Option<std::thread::JoinHandle<()>>>,
	/// Dropping this signals the worker (a disconnected channel also stops it).
	control: Sender<Message>,
}

impl NodeFileWatcher {
	/// Establish coverage: watchers first, then the snapshot later changes are compared with.
	pub async fn open(
		targets: &[WatchTarget],
		resolve_path: impl Fn(&str) -> String,
		on_change: WatchCallback,
		options: NodeWatchOptions,
	) -> Result<NodeFileWatcher, FileError> {
		let resolved: Vec<ResolvedTarget> = targets
			.iter()
			.map(|target| ResolvedTarget {
				path: resolve_path(&target.path),
				recursive: target.recursive,
				hidden: target.exclude.as_ref().is_some_and(|exclude| exclude.hidden),
				names: target.exclude.as_ref().map(|exclude| exclude.names.iter().cloned().collect()).unwrap_or_default(),
			})
			.collect();
		// Windows refuses to rename a directory while another directory below it is open, and a
		// native watcher keeps every watched directory open: watching would break renames of
		// their parents, so Windows polls. See D2 for unreliable-file-system detection.
		let mode = options.mode.unwrap_or(if cfg!(windows) { WatchMode::Polling } else { WatchMode::Native });
		let poll_interval =
			Duration::from_millis(options.poll_interval_ms.unwrap_or(DEFAULT_POLL_MS).max(1));
		let max_directories = options.max_directories.unwrap_or(DEFAULT_MAX_DIRECTORIES);
		let closed = Arc::new(AtomicBool::new(false));
		let established = Arc::new(AtomicBool::new(false));
		let (tx, rx) = std::sync::mpsc::channel::<Message>();
		let worker = Worker {
			targets: resolved,
			established: Arc::clone(&established),
			on_change,
			poll_interval,
			max_directories,
			closed: Arc::clone(&closed),
			mode,
			snapshot: HashMap::new(),
			watchers: HashMap::new(),
			events: BTreeSet::new(),
			debounce: None,
			settle: None,
			events_tx: tx.clone(),
		};
		let handle = std::thread::Builder::new()
			.name("pi-durable-env-watch".to_string())
			.spawn(move || worker.run(&rx))
			.map_err(|error| FileError::new(FileErrorCode::Unknown, error.to_string(), None))?;
		let watcher = NodeFileWatcher {
			mode,
			closed,
			established,
			handle: StdMutex::new(Some(handle)),
			control: tx,
		};
		// Wait until the initial scan inside `run` finished, so that watching before loading
		// cannot miss a change made during the load (upstream `open` awaits the first `#sync`).
		watcher.wait_established().await?;
		Ok(watcher)
	}

	/// Waits until the first snapshot exists (coverage established). A scan error is surfaced
	/// as the watcher's terminal error change, mirroring upstream's throwing `open`.
	async fn wait_established(&self) -> Result<(), FileError> {
		let deadline = Instant::now() + Duration::from_secs(60);
		while !self.established.load(Ordering::SeqCst) {
			if self.closed.load(Ordering::SeqCst) {
				return err(FileError::new(FileErrorCode::Invalid, "watcher stopped during open", None));
			}
			if Instant::now() > deadline {
				return err(FileError::new(FileErrorCode::Unknown, "watcher did not establish coverage", None));
			}
			tokio::time::sleep(Duration::from_millis(5)).await;
		}
		Ok(())
	}
}

#[async_trait]
impl FileWatcher for NodeFileWatcher {
	fn mode(&self) -> WatchMode {
		self.mode
	}

	async fn close(&self, _context: &Context) {
		if self.closed.swap(true, Ordering::SeqCst) {
			return;
		}
		let _ = self.control.send(Message::Stop);
		let handle = self.handle.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
		if let Some(handle) = handle {
			let _ = tokio::task::spawn_blocking(move || handle.join()).await;
		}
	}
}
