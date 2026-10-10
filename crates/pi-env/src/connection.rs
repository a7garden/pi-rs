//! Client half of the daemon wire protocol (upstream `packages/env/src/connection.ts`): frames of
//! `u32 length | u8 type | u32 id | u32 jsonLength | json | payload` over the daemon's stdin/stdout.
//!
//! One connection starts its daemon lazily on the first request and starts a new daemon after the
//! old one is lost (crash, kill, transport failure). Each daemon run is a *session* with its own id:
//! handles opened in it are only valid there, and session-scoped requests fail instead of reaching a
//! daemon that never opened them. Requests in flight when a session is lost fail with a `lost` error
//! (`is_connection_lost`), which the next request may recover from by starting the daemon again.

use std::{collections::HashMap, process::Stdio, sync::Arc, time::Duration};

use parking_lot::Mutex;
use pi_chord::context::AbortSignal;
use serde_json::{Map, Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{ChildStdin, Command},
    sync::{Mutex as AsyncMutex, oneshot},
};

/// Frame types (docs/protocol.md).
const REQUEST: u8 = 1;
const RESULT: u8 = 2;
const ERROR: u8 = 3;
const EVENT: u8 = 4;
const CANCEL: u8 = 5;
const PING: u8 = 6;

/// Largest frame either side accepts.
pub const MAX_FRAME: usize = 16 * 1024 * 1024;
const PING_INTERVAL: Duration = Duration::from_secs(5);
/// The daemon pings every five seconds; this much silence means the connection is gone.
const SILENCE_LIMIT: Duration = Duration::from_secs(30);
/// How long starting the daemon and its `hello` may take.
const START_TIMEOUT: Duration = Duration::from_secs(60);

pub type Json = Map<String, Value>;

/// A failure reported by the daemon, with its Node-style code; `fields` carries the whole record
/// (including the client's own `lost` marker for transport failures).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct RemoteError {
    pub code: String,
    pub message: String,
    pub path: Option<String>,
    pub fields: Json,
}

impl RemoteError {
    pub fn from_fields(fields: Json) -> RemoteError {
        let code = fields
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let message = fields
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Remote operation failed")
            .to_string();
        let path = fields
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_owned);
        RemoteError {
            code,
            message,
            path,
            fields,
        }
    }

    pub fn new(code: &str, message: impl Into<String>) -> RemoteError {
        let mut fields = Json::new();
        fields.insert("code".into(), json!(code));
        let message = message.into();
        fields.insert("message".into(), Value::String(message.clone()));
        RemoteError {
            code: code.to_string(),
            message,
            path: None,
            fields,
        }
    }

    /// A connection failure rather than a daemon refusal; a request may succeed once the daemon is
    /// started again.
    pub fn lost(&self) -> bool {
        self.fields
            .get("lost")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

/// Whether an error means the connection was lost, not that the daemon refused the request.
pub fn is_connection_lost(error: &RemoteError) -> bool {
    error.lost()
}

fn lost(message: impl Into<String>) -> RemoteError {
    let mut error = RemoteError::new("unknown", message);
    error.fields.insert("lost".into(), json!(true));
    error
}

fn closed_error() -> RemoteError {
    RemoteError::new("unknown", "Connection closed")
}

/// What `hello` reports about the remote machine.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RemoteInfo {
    pub protocol: u32,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub home: String,
    pub tmpdir: String,
    /// Path separator: `/`, or `\` on Windows.
    pub separator: String,
    /// The daemon's working directory, where Node's `path.resolve` falls back to `process.cwd()`.
    pub cwd: String,
    /// Windows: per-drive working directories (`C:` → `C:\work`) of the `=C:` variables.
    pub drive_cwds: HashMap<String, String>,
    pub pid: u32,
}

impl Default for RemoteInfo {
    fn default() -> Self {
        RemoteInfo {
            protocol: 0,
            version: String::new(),
            os: String::new(),
            arch: String::new(),
            home: String::new(),
            tmpdir: String::new(),
            separator: "/".to_string(),
            cwd: String::new(),
            drive_cwds: HashMap::new(),
            pid: 0,
        }
    }
}

/// Where the daemon command comes from. A computed command runs at each start — detecting the remote
/// system and deploying the daemon first, for example; its failure fails that start with its message,
/// and the next request tries again.
#[derive(Clone)]
pub enum CommandSource {
    Fixed(Vec<String>),
    Computed(Arc<dyn Fn() -> CommandFuture + Send + Sync>),
}

pub type CommandFuture =
    std::pin::Pin<Box<dyn Future<Output = Result<Vec<String>, String>> + Send>>;

#[derive(Clone)]
pub struct ConnectionOptions {
    pub command: CommandSource,
    /// Receives the daemon's and the transport's diagnostic output.
    pub on_log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

impl ConnectionOptions {
    pub fn new(command: impl IntoIterator<Item = String>) -> ConnectionOptions {
        ConnectionOptions {
            command: CommandSource::Fixed(command.into_iter().collect()),
            on_log: None,
        }
    }

    /// A command computed at each start (upstream's `command` function).
    pub fn computed<F>(compute: F) -> ConnectionOptions
    where
        F: Fn() -> CommandFuture + Send + Sync + 'static,
    {
        ConnectionOptions {
            command: CommandSource::Computed(Arc::new(compute)),
            on_log: None,
        }
    }
}

/// What one request answered: the daemon's JSON, its payload, and the session that answered.
#[derive(Debug, Clone)]
pub struct Reply {
    pub json: Json,
    pub payload: Vec<u8>,
    pub session: u64,
}

pub struct PendingEntry {
    sender: oneshot::Sender<Result<Reply, RemoteError>>,
    on_event: Option<Arc<dyn Fn(Json, Vec<u8>) + Send + Sync>>,
    disposer: Option<Box<dyn FnOnce() + Send>>,
}

#[derive(Default)]
pub struct RequestOptions {
    pub payload: Vec<u8>,
    /// Aborting sends `cancel`; the request still settles with the daemon's result or error.
    pub signal: Option<AbortSignal>,
    /// Progress events of the request, such as `exec` output.
    pub on_event: Option<Arc<dyn Fn(Json, Vec<u8>) + Send + Sync>>,
    /// The request's id and session, for `kill`.
    pub on_start: Option<Arc<dyn Fn(u32, u64) + Send + Sync>>,
    /// Only run in this daemon session, for requests on handles it opened.
    pub session: Option<u64>,
}

/// One running daemon.
struct Session {
    id: u64,
    token: String,
    writer: Arc<AsyncMutex<ChildStdin>>,
    pending: Mutex<HashMap<u32, PendingEntry>>,
    /// Milliseconds since the epoch of the last received byte.
    last_seen: Arc<atomic::AtomicU64>,
    live: atomic::AtomicBool,
    /// Wakes the reader, which owns the child process, so a teardown kills and reaps it.
    stop: Arc<tokio::sync::Notify>,
    /// The connection this session belongs to, so any teardown clears it as the current session.
    owner: Mutex<std::sync::Weak<Inner>>,
}

mod atomic {
    pub use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
}

impl Session {
    async fn write(&self, frame: Vec<u8>) -> Result<(), RemoteError> {
        let mut writer = self.writer.lock().await;
        writer
            .write_all(&frame)
            .await
            .map_err(|error| lost(format!("pi-env write failed: {error}")))
    }

    fn touch(&self) {
        self.last_seen.store(now_ms(), atomic::Ordering::Relaxed);
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

struct State {
    session: Option<(Arc<Session>, RemoteInfo)>,
    sessions: u64,
    next_id: u32,
    closed: bool,
}

struct Inner {
    options: ConnectionOptions,
    state: Mutex<State>,
    /// Serializes starts: waiters either reuse the session a running start produced or start anew.
    starting: AsyncMutex<()>,
}

/// One connection to a pi-env daemon: started lazily on the first request and started again after it
/// is lost. Requests in flight when it is lost fail with a `lost` error; for mutations their outcome
/// is then unknown.
#[derive(Clone)]
pub struct Connection {
    inner: Arc<Inner>,
}

impl Connection {
    pub fn new(options: ConnectionOptions) -> Connection {
        Connection {
            inner: Arc::new(Inner {
                options,
                state: Mutex::new(State {
                    session: None,
                    sessions: 0,
                    next_id: 1,
                    closed: false,
                }),
                starting: AsyncMutex::new(()),
            }),
        }
    }

    /// Connect if needed and return what the daemon reported about its machine.
    pub async fn info(&self) -> Result<RemoteInfo, RemoteError> {
        Ok(self.connect().await?.1)
    }

    /// Connect if needed and return the live session's id.
    pub async fn session_id(&self) -> Result<u64, RemoteError> {
        Ok(self.connect().await?.0.id)
    }

    pub async fn request(
        &self,
        op: &str,
        fields: Json,
        options: RequestOptions,
    ) -> Result<Reply, RemoteError> {
        if let Some(wanted) = options.session {
            let current = self.inner.state.lock().session.clone();
            let Some((session, _)) = current.filter(|(session, _)| session.id == wanted) else {
                return Err(lost("pi-env connection lost"));
            };
            return self.send(&session, op, fields, options).await;
        }
        let (session, _) = self.connect().await?;
        self.send(&session, op, fields, options).await
    }

    /// Like [`Connection::request`], but aborting passes the caller's signal.
    pub async fn request_with_signal(
        &self,
        op: &str,
        fields: Json,
        mut options: RequestOptions,
        signal: Option<AbortSignal>,
    ) -> Result<Reply, RemoteError> {
        if options.signal.is_none() {
            options.signal = signal;
        }
        self.request(op, fields, options).await
    }

    /// Stop the daemon; it kills everything it started.
    pub async fn close(&self) {
        self.inner.state.lock().closed = true;
        let current = self.inner.state.lock().session.take();
        if let Some((session, _)) = current {
            Connection::teardown_owned(&session, closed_error());
        }
    }

    /// Kill one `exec` without aborting it: the command settles with the killed process's status.
    pub fn kill(&self, id: u32, session: u64) {
        let current = self.inner.state.lock().session.clone();
        let Some((current, _)) = current.filter(|(current, _)| {
            current.id == session && current.live.load(atomic::Ordering::SeqCst)
        }) else {
            return;
        };
        if let Ok(frame) = encode_frame(CANCEL, id, &json_map([("mode", json!("kill"))]), &[]) {
            tokio::spawn(async move {
                let _ = current.write(frame).await;
            });
        }
    }

    async fn connect(&self) -> Result<(Arc<Session>, RemoteInfo), RemoteError> {
        if self.inner.state.lock().closed {
            return Err(closed_error());
        }
        let _starting = self.inner.starting.lock().await;
        if let Some(current) = self.inner.state.lock().session.clone() {
            return Ok(current);
        }
        self.start().await
    }

    /// Spawn the daemon, wait for its sync line and `hello`; a failure is not remembered.
    async fn start(&self) -> Result<(Arc<Session>, RemoteInfo), RemoteError> {
        let command = match &self.inner.options.command {
            CommandSource::Fixed(command) => Ok(command.clone()),
            CommandSource::Computed(compute) => compute().await,
        }
        .map_err(|message| {
            let mut fields = Json::new();
            fields.insert("code".into(), json!("spawn_error"));
            fields.insert("message".into(), json!(message));
            fields.insert("lost".into(), json!(true));
            RemoteError::from_fields(fields)
        })?;
        if self.inner.state.lock().closed {
            return Err(closed_error());
        }
        let (program, arguments) = command
            .split_first()
            .ok_or_else(|| RemoteError::new("spawn_error", "No daemon command"))?;
        let program = program.clone();
        let arguments = arguments.to_vec();
        let token = random_token();
        let mut child = Command::new(&program)
            .args(&arguments)
            .args(["serve", "--token", &token])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| RemoteError::new("spawn_error", error.to_string()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| RemoteError::new("spawn_error", "Missing daemon stdin"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| RemoteError::new("spawn_error", "Missing daemon stdout"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| RemoteError::new("spawn_error", "Missing daemon stderr"))?;
        // The handshake races `hello` against the start deadline and an exit before the sync line.
        let (failed_sender, mut failed_receiver) = oneshot::channel::<RemoteError>();
        let failed_slot = Arc::new(Mutex::new(Some(failed_sender)));
        let session = {
            let mut state = self.inner.state.lock();
            state.sessions += 1;
            let session = Arc::new(Session {
                id: state.sessions,
                writer: Arc::new(AsyncMutex::new(stdin)),
                token,
                pending: Mutex::new(HashMap::new()),
                last_seen: Arc::new(atomic::AtomicU64::new(now_ms())),
                live: atomic::AtomicBool::new(true),
                stop: Arc::new(tokio::sync::Notify::new()),
                owner: Mutex::new(Arc::downgrade(&self.inner)),
            });
            state.session = Some((session.clone(), RemoteInfo::default()));
            session
        };
        // Shell startup files may print before the daemon runs; the reader skips everything before its
        // sync line and then parses frames. It owns the child, so it kills it on teardown and reports
        // how an exit before the sync line ended; corruption and end of output tear the session down.
        let reader_session = session.clone();
        let start_failure = failed_slot.clone();
        let log = self.inner.options.on_log.clone();
        let reader = tokio::spawn(async move {
            let mut data = Vec::new();
            let mut synced = false;
            let mut corrupt = false;
            let marker = format!("PI-ENV {}\n", reader_session.token).into_bytes();
            let mut buffer = [0u8; 16 * 1024];
            'outer: loop {
                let read = tokio::select! {
                    read = stdout.read(&mut buffer) => read,
                    _ = reader_session.stop.notified() => {
                        // The session is gone: the daemon must not outlive it.
                        let _ = child.start_kill();
                        break;
                    }
                };
                match read {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        reader_session.touch();
                        data.extend_from_slice(&buffer[..read]);
                    }
                }
                if !synced {
                    let Some(position) = find_bytes(&data, &marker) else {
                        if data.len() > 1024 * 1024 {
                            let keep = marker.len().saturating_sub(1);
                            data.drain(..data.len() - keep);
                        }
                        continue;
                    };
                    let noise = String::from_utf8_lossy(&data[..position]).trim().to_owned();
                    if !noise.is_empty()
                        && let Some(log) = &log
                    {
                        log(&noise);
                    }
                    data.drain(..position + marker.len());
                    synced = true;
                }
                while data.len() >= 4 {
                    let length = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
                    if !(9..=MAX_FRAME).contains(&length) {
                        corrupt = true;
                        break 'outer;
                    }
                    if data.len() < 4 + length {
                        break;
                    }
                    let body = &data[4..4 + length];
                    let kind = body[0];
                    let id = u32::from_be_bytes(body[1..5].try_into().unwrap());
                    let json_length = u32::from_be_bytes(body[5..9].try_into().unwrap()) as usize;
                    let frame_json = if 9 + json_length > length {
                        corrupt = true;
                        break 'outer;
                    } else {
                        match serde_json::from_slice::<Json>(&body[9..9 + json_length]) {
                            Ok(fields) => fields,
                            Err(_) => {
                                corrupt = true;
                                break 'outer;
                            }
                        }
                    };
                    let payload = body[9 + json_length..].to_vec();
                    data.drain(..4 + length);
                    reader_session.dispatch(kind, id, frame_json, payload);
                }
            }
            // Reap the child; a sync line it never printed is the start failure the handshake reports.
            let exit_code = child.wait().await.ok().and_then(|status| status.code());
            if !synced && let Some(sender) = start_failure.lock().take() {
                let _ = sender.send(lost(match exit_code {
                    Some(code) => format!("pi-env exited with code {code} before it was ready"),
                    None => "pi-env exited before it was ready".to_string(),
                }));
            }
            let error = if corrupt {
                lost("Corrupt frame from pi-env")
            } else {
                lost("pi-env connection lost")
            };
            Connection::teardown_owned(&reader_session, error);
        });
        // The daemon's and the transport's diagnostics.
        let log = self.inner.options.on_log.clone();
        tokio::spawn(async move {
            let mut buffer = [0u8; 16 * 1024];
            loop {
                match stderr.read(&mut buffer).await {
                    Ok(0) | Err(_) => return,
                    Ok(read) => {
                        if let Some(log) = &log {
                            log(&String::from_utf8_lossy(&buffer[..read]));
                        }
                    }
                }
            }
        });
        // Pings keep both ends watching the link; silence tears the session down.
        let ping_session = session.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(PING_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                if !ping_session.live.load(atomic::Ordering::SeqCst) {
                    return;
                }
                if now_ms().saturating_sub(ping_session.last_seen.load(atomic::Ordering::Relaxed))
                    > SILENCE_LIMIT.as_millis() as u64
                {
                    Connection::teardown_owned(&ping_session, lost("pi-env connection timed out"));
                    return;
                }
                let Ok(frame) = encode_frame(PING, 0, &Json::new(), &[]) else {
                    return;
                };
                if ping_session.write(frame).await.is_err() {
                    return;
                }
            }
        });
        // Handshake: `hello` raced against the start deadline (and the reader's exit-before-ready failure).
        let failed_sender_deadline = failed_slot.clone();
        let timeout = tokio::spawn(async move {
            tokio::time::sleep(START_TIMEOUT).await;
            if let Some(sender) = failed_sender_deadline.lock().take() {
                let _ = sender.send(lost(format!(
                    "pi-env did not answer within {} s",
                    START_TIMEOUT.as_secs()
                )));
            }
        });
        let mut fields = Json::new();
        fields.insert("protocol".into(), json!(1));
        let hello = tokio::select! {
            reply = self.send(&session, "hello", fields, RequestOptions::default()) => Some(reply.map(|reply| (reply, ()))),
            // The channel carries the reader's start failure; without one (it dropped the sender),
            // the reader owns the teardown, so the handshake gives up.
            failure = &mut failed_receiver => match failure {
                Ok(failure) => Some(Err(failure)),
                Err(_) => None,
            },
        };
        // The deadline only guards the handshake.
        timeout.abort();
        let (reply, ()) = match hello {
            Some(Ok(reply)) => reply,
            Some(Err(error)) => {
                self.teardown(&session, error.clone());
                return Err(error);
            }
            None => {
                let error = lost("pi-env connection lost");
                self.teardown(&session, error.clone());
                return Err(error);
            }
        };
        if reply.json.get("protocol").and_then(Value::as_u64) != Some(1) {
            let error = lost(format!(
                "Unsupported protocol {}",
                reply
                    .json
                    .get("protocol")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            ));
            self.teardown(&session, error.clone());
            return Err(error);
        }
        let info: RemoteInfo =
            serde_json::from_value(Value::Object(reply.json)).map_err(|error| {
                let mapped = RemoteError::new("unknown", error.to_string());
                self.teardown(&session, mapped.clone());
                mapped
            })?;
        {
            let mut state = self.inner.state.lock();
            if state
                .session
                .as_ref()
                .is_some_and(|(current, _)| current.id == session.id)
            {
                state.session = Some((session.clone(), info.clone()));
            }
        }
        Ok((session, info))
    }

    async fn send(
        &self,
        session: &Arc<Session>,
        op: &str,
        mut fields: Json,
        options: RequestOptions,
    ) -> Result<Reply, RemoteError> {
        if !session.live.load(atomic::Ordering::SeqCst) {
            return Err(lost("pi-env connection lost"));
        }
        let id = {
            let mut state = self.inner.state.lock();
            let id = state.next_id;
            state.next_id = state.next_id.wrapping_add(1).max(1);
            id
        };
        fields.insert("op".into(), json!(op));
        let (sender, receiver) = oneshot::channel();
        // Aborting writes one `cancel`; the request still settles with the daemon's answer.
        let disposer = options.signal.as_ref().map(|signal| {
            let cancel_session = session.clone();
            signal.add_listener(move || {
                let cancel_session = cancel_session.clone();
                if let Ok(frame) = encode_frame(CANCEL, id, &Json::new(), &[]) {
                    tokio::spawn(async move {
                        let _ = cancel_session.write(frame).await;
                    });
                }
            })
        });
        session.pending.lock().insert(
            id,
            PendingEntry {
                sender,
                on_event: options.on_event.clone(),
                disposer,
            },
        );
        if let Some(on_start) = &options.on_start {
            on_start(id, session.id);
        }
        let frame = match encode_frame(REQUEST, id, &fields, &options.payload) {
            Ok(frame) => frame,
            Err(error) => {
                session.pending.lock().remove(&id);
                return Err(error);
            }
        };
        if let Err(error) = session.write(frame).await {
            session.pending.lock().remove(&id);
            return Err(error);
        }
        // The abort listener above fires for a signal that was already aborted before the request was
        // written, which the daemon cannot match yet; send the cancel again, as upstream does.
        if let Some(signal) = &options.signal
            && signal.aborted()
        {
            let Ok(frame) = encode_frame(CANCEL, id, &Json::new(), &[]) else {
                return Err(RemoteError::new(
                    "unknown",
                    "unreachable: empty json serializes",
                ));
            };
            let _ = session.write(frame).await;
        }
        // The cancel makes the daemon settle the request; the answer (or the session teardown) resolves it.
        match receiver.await {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(lost("pi-env connection lost")),
        }
    }

    /// Fail every pending request and stop the session's tasks and child.
    fn teardown(&self, session: &Arc<Session>, error: RemoteError) {
        session.live.store(false, atomic::Ordering::SeqCst);
        Connection::teardown_owned(session, error);
    }

    fn teardown_owned(session: &Arc<Session>, error: RemoteError) {
        if !session.live.swap(false, atomic::Ordering::SeqCst) {
            return;
        }
        // The session is no longer current: the next request starts the daemon again.
        if let Some(owner) = session.owner.lock().upgrade() {
            let mut state = owner.state.lock();
            if state
                .session
                .as_ref()
                .is_some_and(|(current, _)| current.id == session.id)
            {
                state.session = None;
            }
        }
        // The reader owns the child: waking it kills and reaps the daemon.
        session.stop.notify_one();
        let pending: Vec<PendingEntry> = session
            .pending
            .lock()
            .drain()
            .map(|(_, entry)| entry)
            .collect();
        for entry in pending {
            let PendingEntry {
                sender, disposer, ..
            } = entry;
            if let Some(disposer) = disposer {
                disposer();
            }
            let _ = sender.send(Err(error.clone()));
        }
    }
}

impl Session {
    fn dispatch(&self, kind: u8, id: u32, fields: Json, payload: Vec<u8>) {
        if kind == EVENT {
            let on_event = self
                .pending
                .lock()
                .get(&id)
                .and_then(|entry| entry.on_event.clone());
            if let Some(on_event) = on_event {
                on_event(fields, payload);
            }
            return;
        }
        if kind != RESULT && kind != ERROR {
            return;
        }
        let Some(entry) = self.pending.lock().remove(&id) else {
            return;
        };
        let PendingEntry {
            sender, disposer, ..
        } = entry;
        if let Some(disposer) = disposer {
            disposer();
        }
        let _ = sender.send(match kind {
            RESULT => Ok(Reply {
                json: fields,
                payload,
                session: self.id,
            }),
            _ => Err(RemoteError::from_fields(fields)),
        });
    }
}

/// Lone surrogates become U+FFFD as Node's `JSON.stringify` does; Rust strings cannot hold them.
pub fn encode_frame(
    kind: u8,
    id: u32,
    fields: &Json,
    payload: &[u8],
) -> Result<Vec<u8>, RemoteError> {
    let body = serde_json::to_vec(fields)
        .map_err(|error| RemoteError::new("unknown", error.to_string()))?;
    let length = 9 + body.len() + payload.len();
    if length > MAX_FRAME {
        return Err(RemoteError::new("unknown", "Frame exceeds 16 MiB"));
    }
    let mut frame = Vec::with_capacity(4 + length);
    frame.extend_from_slice(&(length as u32).to_be_bytes());
    frame.push(kind);
    frame.extend_from_slice(&id.to_be_bytes());
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    frame.extend_from_slice(payload);
    Ok(frame)
}

fn json_map(pairs: impl IntoIterator<Item = (&'static str, Value)>) -> Json {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

fn random_token() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("system randomness");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn find_bytes(data: &[u8], needle: &[u8]) -> Option<usize> {
    data.windows(needle.len())
        .position(|window| window == needle)
}
