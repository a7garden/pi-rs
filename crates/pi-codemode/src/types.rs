//! Wire and host-facing types, mirroring upstream `src/types.ts` 1:1.
//!
//! JSON shapes are byte-compatible with the upstream protocol: `CodemodeOutputItem`,
//! `CodemodeCall`, `CodemodeError`, `CodemodeResult`, and `CodemodeStoreWrites` all
//! serialize with the same field names and tags as the TypeScript definitions, so
//! messages that cross a process boundary keep their JSON representation.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Notify;

/// A JSON Schema document. Only used to render declarations; values are not validated against it.
///
/// Upstream this is `{ [key: string]: unknown } | boolean`; here it is the raw JSON value
/// (objects and booleans are meaningful; any other value renders as `unknown`, matching the
/// upstream `isObject` guards).
pub type CodemodeJsonSchema = Value;

/// Aborted when the script finishes (including unawaited calls), the execution times out,
/// the caller aborts, or the sandbox is closed.
///
/// Rust counterpart of the upstream `AbortSignal` handed to tool `execute`. Subscribe with
/// [`AbortSignal::wait_aborted`] or poll [`AbortSignal::is_aborted`].
#[derive(Clone, Default)]
pub struct AbortSignal {
    inner: Arc<SignalInner>,
}

struct SignalInner {
    reason: Mutex<Option<String>>,
    notify: Notify,
}

impl Default for SignalInner {
    fn default() -> Self {
        Self {
            reason: Mutex::new(None),
            notify: Notify::new(),
        }
    }
}

impl std::fmt::Debug for AbortSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbortSignal")
            .field("aborted", &self.is_aborted())
            .finish()
    }
}

impl AbortSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Abort with a reason. Later calls are no-ops (first reason wins, like upstream).
    pub fn abort(&self, reason: impl Into<String>) {
        {
            let mut guard = self.inner.reason.lock();
            if guard.is_some() {
                return;
            }
            *guard = Some(reason.into());
        }
        // Wake everyone; waiters re-check `reason`, so a notify racing a subscribe is safe.
        self.inner.notify.notify_waiters();
    }

    pub fn is_aborted(&self) -> bool {
        self.inner.reason.lock().is_some()
    }

    /// The abort reason, if aborted.
    pub fn reason(&self) -> Option<String> {
        self.inner.reason.lock().clone()
    }

    /// Resolves once the signal aborts (returns immediately if already aborted).
    pub async fn wait_aborted(&self) {
        loop {
            // `notified()` registers interest before the check below, so an abort racing
            // this wait is always observed.
            let notified = self.inner.notify.notified();
            if self.is_aborted() {
                return;
            }
            notified.await;
        }
    }
}

/// What a tool implementation receives and returns. `Ok(None)` is the upstream
/// `undefined` result; `Ok(Some(Value))` must be JSON-serializable (a thrown error
/// surfaces in the script as an `Error` with the same message, so failures are
/// `Err(message)`).
pub type ToolExecuteFn = Arc<
    dyn Fn(
            Option<Value>,
            CodemodeToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, String>> + Send>>
        + Send
        + Sync,
>;

/// Context passed to tool `execute`: carries the per-call [`AbortSignal`].
#[derive(Clone, Default)]
pub struct CodemodeToolContext {
    pub signal: AbortSignal,
}

/// A tool callable from the script as `tools.<id>(args)`.
#[derive(Clone)]
pub struct CodemodeTool {
    /// The script calls tools as `tools.<id>(args)`, where `<id>` is the name with characters
    /// that are not valid in identifiers replaced by `_` (see
    /// [`to_codemode_identifier`](crate::identifier::to_codemode_identifier)), and also as
    /// `tools["<name>"](args)`. Globals are called as `<name>(args)` and must be identifiers,
    /// or `<namespace>.<member>`, which groups them into a frozen namespace object.
    pub name: String,
    /// Shown as a doc comment in [`render_declarations`](crate::declarations::render_declarations),
    /// and listed in `ALL_TOOLS` for tools.
    pub description: Option<String>,
    /// Schema of the single argument. Rendered as the parameter type; `unknown` when omitted.
    pub input_schema: Option<CodemodeJsonSchema>,
    /// Schema of the resolved value. Rendered as the promise type; `unknown` when omitted.
    pub output_schema: Option<CodemodeJsonSchema>,
    /// Globals only: `execute` receives all call arguments as an array instead of the first one.
    pub spread: bool,
    /// Globals only: TypeScript parameter list and return type for
    /// [`render_declarations`](crate::declarations::render_declarations), for example
    /// `(type: string, id?: string): Promise<Model[]>`. Replaces the rendering from the schemas.
    pub signature: Option<String>,
    /// `args` is whatever the script passed, after a JSON round trip. The return value must be
    /// JSON-serializable; a thrown error surfaces in the script as an `Error` with the same message.
    pub execute: ToolExecuteFn,
}

impl std::fmt::Debug for CodemodeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodemodeTool")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CodemodeTool {
    /// Convenience constructor with a synchronous-ish async closure.
    pub fn new<F, Fut>(name: impl Into<String>, execute: F) -> Self
    where
        F: Fn(Option<Value>, CodemodeToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<Value>, String>> + Send + 'static,
    {
        let execute = Arc::new(execute);
        Self {
            name: name.into(),
            description: None,
            input_schema: None,
            output_schema: None,
            spread: false,
            signature: None,
            execute: Arc::new(move |args, ctx| {
                let fut = execute(args, ctx);
                Box::pin(fut)
            }),
        }
    }

    /// Builder-style description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Builder-style input schema.
    pub fn with_input_schema(mut self, schema: CodemodeJsonSchema) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Builder-style output schema.
    pub fn with_output_schema(mut self, schema: CodemodeJsonSchema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Builder-style explicit signature (globals only).
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }

    /// Builder-style spread flag (globals only).
    pub fn with_spread(mut self, spread: bool) -> Self {
        self.spread = spread;
        self
    }
}

/// One item of the script's output, in the order the script produced it: `text()` and
/// `console.*` produce text items, with `console: true` for `console.*`, and `image()`
/// image items. `data` is base64.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodemodeOutputItem {
    #[serde(rename = "text")]
    Text {
        text: String,
        /// `true` when the item came from `console.*` rather than `text()` (upstream
        /// `console?: true`; never serialized as any other value).
        #[serde(skip_serializing_if = "Option::is_none", default)]
        console: Option<True>,
    },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

impl CodemodeOutputItem {
    /// Characters this item contributes to the output budget (text or base64 data).
    pub fn char_len(&self) -> usize {
        match self {
            CodemodeOutputItem::Text { text, .. } => text.chars().count(),
            CodemodeOutputItem::Image { data, .. } => data.chars().count(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodemodeCallStatus {
    #[serde(rename = "ok")]
    Ok,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeCall {
    pub name: String,
    pub status: CodemodeCallStatus,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodemodeErrorKind {
    /// The script threw or failed to parse. `name` and `stack` come from the script's error.
    Script,
    /// The overall deadline expired. The worker was terminated.
    Timeout,
    /// The caller's signal fired or the sandbox was closed. The worker was terminated.
    Aborted,
    /// The worker or VM failed outside the script's control (for example a wasm trap or a
    /// missing worker file).
    Sandbox,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeError {
    pub kind: CodemodeErrorKind,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stack: Option<String>,
}

impl CodemodeError {
    pub fn new(kind: CodemodeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            name: None,
            message: message.into(),
            stack: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_stack(mut self, stack: impl Into<String>) -> Self {
        self.stack = Some(stack.into());
        self
    }
}

/// Keys the script changed with `store()`. Only successful executions report writes.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeStoreWrites {
    pub set: serde_json::Map<String, Value>,
    /// Keys stored as `undefined`.
    pub delete: Vec<String>,
}

/// `output` is kept for failed executions too, up to the failure. `exit()` completes with
/// `value: undefined` (encoded as `None`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CodemodeResult {
    Ok {
        ok: True,
        /// The script's return value after its JSON round trip; `None` is the JS `undefined`.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        value: Option<Value>,
        output: Vec<CodemodeOutputItem>,
        calls: Vec<CodemodeCall>,
        store_writes: CodemodeStoreWrites,
    },
    Err {
        ok: False,
        error: CodemodeError,
        output: Vec<CodemodeOutputItem>,
        calls: Vec<CodemodeCall>,
    },
}

/// Serde helper: serializes as JSON `true` (a bool-typed tag field on the result).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct True;

impl Serialize for True {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bool(true)
    }
}

impl<'de> Deserialize<'de> for True {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(if <bool as Deserialize>::deserialize(deserializer)? {
            True
        } else {
            return Err(serde::de::Error::custom("expected true"));
        })
    }
}

/// Serde helper: serializes as JSON `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct False;

impl Serialize for False {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bool(false)
    }
}

impl<'de> Deserialize<'de> for False {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(if <bool as Deserialize>::deserialize(deserializer)? {
            return Err(serde::de::Error::custom("expected false"));
        } else {
            False
        })
    }
}

impl CodemodeResult {
    /// `true` for the `ok` variant.
    pub fn is_ok(&self) -> bool {
        matches!(self, CodemodeResult::Ok { .. })
    }

    /// The script's return value (`None` for a JS `undefined` result).
    pub fn value(&self) -> Option<&Value> {
        match self {
            CodemodeResult::Ok { value, .. } => value.as_ref(),
            CodemodeResult::Err { .. } => None,
        }
    }

    pub fn error(&self) -> Option<&CodemodeError> {
        match self {
            CodemodeResult::Ok { .. } => None,
            CodemodeResult::Err { error, .. } => Some(error),
        }
    }

    pub fn output(&self) -> &[CodemodeOutputItem] {
        match self {
            CodemodeResult::Ok { output, .. } | CodemodeResult::Err { output, .. } => output,
        }
    }

    pub fn calls(&self) -> &[CodemodeCall] {
        match self {
            CodemodeResult::Ok { calls, .. } | CodemodeResult::Err { calls, .. } => calls,
        }
    }

    pub fn store_writes(&self) -> Option<&CodemodeStoreWrites> {
        match self {
            CodemodeResult::Ok { store_writes, .. } => Some(store_writes),
            CodemodeResult::Err { .. } => None,
        }
    }
}

/// Per-execution options for [`crate::runtime::host::CodemodeSandbox::execute`], mirroring
/// upstream `CodemodeExecuteOptions`.
#[derive(Debug, Clone, Default)]
pub struct CodemodeExecuteOptions {
    /// Aborts the execution (the worker is terminated) when fired.
    pub signal: Option<AbortSignal>,
    /// Overrides the sandbox default for this execution. `f64::INFINITY` disables the deadline.
    pub timeout_ms: Option<f64>,
    /// Values the script reads with `load(key)`. Must be JSON-serializable. The script's own
    /// `store()` calls come back as `result.store_writes`; persisting them is up to the caller.
    pub store: Option<serde_json::Map<String, Value>>,
}
