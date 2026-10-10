//! Port of `packages/mcp/src/transports/transport.ts`.

use parking_lot::Mutex;
use std::sync::Arc;

use async_trait::async_trait;

use crate::protocol::jsonrpc::{JsonRpcMessage, McpError};

pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

pub type MessageListener = Arc<dyn Fn(JsonRpcMessage) + Send + Sync>;
pub type ErrorListener = Arc<dyn Fn(&McpError) + Send + Sync>;
pub type CloseListener = Arc<dyn Fn() + Send + Sync>;
/// Unsubscribe handle returned by the `on*` registration methods.
pub type Unsubscribe = Box<dyn FnOnce() + Send>;

/// Upstream `McpTransport`: the client-side transport interface.
#[async_trait]
pub trait McpTransport: Send + Sync {
    async fn start(&self) -> Result<(), McpError>;
    async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError>;
    async fn close(&self) -> Result<(), McpError>;
    fn on_message(&self, listener: MessageListener) -> Unsubscribe;
    fn on_error(&self, listener: ErrorListener) -> Unsubscribe;
    fn on_close(&self, listener: CloseListener) -> Unsubscribe;
    /// Optional upstream `setProtocolVersion?`; the default is a no-op.
    fn set_protocol_version(&self, _version: &str) {}
}

struct ListenerList<L> {
    next_id: u64,
    listeners: Vec<(u64, L)>,
}

impl<L: Clone> ListenerList<L> {
    fn new() -> Self {
        ListenerList {
            next_id: 1,
            listeners: Vec::new(),
        }
    }

    fn add(&mut self, listener: L) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.listeners.push((id, listener));
        id
    }

    fn remove(&mut self, id: u64) {
        self.listeners.retain(|(existing, _)| *existing != id);
    }

    fn snapshot(&self) -> Vec<L> {
        self.listeners
            .iter()
            .map(|(_, listener)| listener.clone())
            .collect()
    }
}

type ListenerCell<L> = Arc<Mutex<ListenerList<L>>>;

/// Listener bookkeeping shared by transports. `emit_close` fires at most once per transport.
/// (Port of upstream's `TransportEvents` base class; transports compose it.)
pub struct TransportEvents {
    messages: ListenerCell<MessageListener>,
    errors: ListenerCell<ErrorListener>,
    closes: ListenerCell<CloseListener>,
    close_emitted: Arc<Mutex<bool>>,
}

fn lock_list<L>(cell: &ListenerCell<L>) -> parking_lot::MutexGuard<'_, ListenerList<L>> {
    cell.lock()
}

impl TransportEvents {
    pub fn new() -> Self {
        TransportEvents {
            messages: Arc::new(Mutex::new(ListenerList::new())),
            errors: Arc::new(Mutex::new(ListenerList::new())),
            closes: Arc::new(Mutex::new(ListenerList::new())),
            close_emitted: Arc::new(Mutex::new(false)),
        }
    }

    pub fn on_message(&self, listener: MessageListener) -> Unsubscribe {
        let id = lock_list(&self.messages).add(listener);
        let messages = Arc::clone(&self.messages);
        Box::new(move || lock_list(&messages).remove(id))
    }

    pub fn on_error(&self, listener: ErrorListener) -> Unsubscribe {
        let id = lock_list(&self.errors).add(listener);
        let errors = Arc::clone(&self.errors);
        Box::new(move || lock_list(&errors).remove(id))
    }

    pub fn on_close(&self, listener: CloseListener) -> Unsubscribe {
        let id = lock_list(&self.closes).add(listener);
        let closes = Arc::clone(&self.closes);
        Box::new(move || lock_list(&closes).remove(id))
    }

    pub fn emit_message(&self, message: JsonRpcMessage) {
        for listener in lock_list(&self.messages).snapshot() {
            listener(message.clone());
        }
    }

    pub fn emit_error(&self, error: McpError) {
        for listener in lock_list(&self.errors).snapshot() {
            listener(&error);
        }
    }

    pub fn emit_close(&self) {
        {
            let mut emitted = self.close_emitted.lock();
            if *emitted {
                return;
            }
            *emitted = true;
        }
        for listener in lock_list(&self.closes).snapshot() {
            listener();
        }
    }
}
