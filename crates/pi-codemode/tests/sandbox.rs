//! Port of upstream `test/sandbox.test.ts`, scoped to host-observable behavior.
//!
//! Upstream runs real JavaScript in a QuickJS VM inside a Node worker. Here the
//! [`CodeModeWorkerFactory`] is replaced by an in-process fake that reproduces the upstream
//! host/worker protocol semantics (`runtime/protocol.ts`): the fake emits `call`/`output`/
//! `done`/`crash` messages, consumes `result` messages, and honors the interrupt flag and
//! termination. Cases whose assertions live inside the VM's JavaScript semantics are deferred
//! to pi-runtime's QuickJS integration (see the crate-level deviation notes in `lib.rs`).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc};
use {parking_lot::Mutex};

use parking_lot::Mutex as ParkingMutex;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use pi_codemode::runtime::protocol::{
    CallTarget, DoneOutcome, HostToWorkerMessage, WorkerData, WorkerToHostMessage,
};
use pi_codemode::runtime::host::{
    CodemodeSandbox, CodemodeSandboxOptions, CodeModeWorkerFactory, RunningWorker, WorkerHandle,
};
use pi_codemode::types::{
    AbortSignal, CodemodeErrorKind, CodemodeExecuteOptions, CodemodeResult, CodemodeStoreWrites,
    CodemodeTool, CodemodeToolContext,
};

/// A loader that always succeeds: the fake worker ignores the wasm module.
fn dummy_loader() -> Arc<dyn pi_codemode::wasm::WasmLoader> {
    struct DummyLoader;
    impl pi_codemode::wasm::WasmLoader for DummyLoader {
        fn load(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<pi_codemode::CodemodeWasmModule, String>> + Send + '_>> {
            let module: pi_codemode::CodemodeWasmModule = Arc::new(());
            Box::pin(async move { Ok(module) })
        }
    }
    Arc::new(DummyLoader)
}

type Behavior = Arc<
    dyn Fn(FakeWorkerApi) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync,
>;

/// The API a fake worker behavior uses to talk to the host, mirroring the real worker's
/// `post`/message surface.
struct FakeWorkerApi {
    data: WorkerData,
    to_host: mpsc::Sender<WorkerToHostMessage>,
    /// Completed replies by call id; the prelude's `pending` map routes by id, so replies
    /// arriving out of order must not be dropped.
    replies: Arc<ParkingMutex<std::collections::HashMap<u64, (bool, Option<String>)>>>,
    reply_notify: Arc<tokio::sync::Notify>,
}

impl FakeWorkerApi {
    async fn send(&self, message: WorkerToHostMessage) {
        let _ = self.to_host.send(message).await;
    }

    /// Emulates the prelude's tool/global caller: posts a `call` and awaits the host's reply.
    async fn call(&self, target: CallTarget, name: &str, args: Option<Value>) -> Result<Option<Value>, String> {
        // Call ids count up like the prelude's `nextId`.
        static NEXT_ID: AtomicI32 = AtomicI32::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst) as u64;
        self.send(WorkerToHostMessage::Call {
            id,
            target,
            name: name.to_string(),
            args: args.as_ref().map(|value| value.to_string()),
        })
        .await;
        loop {
            {
                let mut replies = self.replies.lock();
                if let Some((ok, payload)) = replies.remove(&id) {
                    return match (ok, payload) {
                        (true, None) => Ok(None),
                        (true, Some(json)) => Ok(Some(serde_json::from_str(&json).unwrap())),
                        (false, Some(message)) => Err(message),
                        (false, None) => Err(String::new()),
                    };
                }
            }
            let notified = self.reply_notify.notified();
            if let Some((ok, payload)) = self.replies.lock().remove(&id) {
                return match (ok, payload) {
                    (true, None) => Ok(None),
                    (true, Some(json)) => Ok(Some(serde_json::from_str(&json).unwrap())),
                    (false, Some(message)) => Err(message),
                    (false, None) => Err(String::new()),
                };
            }
            notified.await;
        }
    }

    async fn output_text(&self, text: &str) {
        self.send(WorkerToHostMessage::Output {
            item: pi_codemode::types::CodemodeOutputItem::Text { text: text.to_string() },
        })
        .await;
    }

    async fn done_ok(&self, value: Option<Value>, writes: CodemodeStoreWrites) {
        let entries: Vec<Value> = writes
            .set
            .into_iter()
            .map(|(key, value)| json!([key, value.to_string()]))
            .chain(writes.delete.into_iter().map(|key| json!([key])))
            .collect();
        self.send(WorkerToHostMessage::Done {
            ok: true,
            outcome: DoneOutcome::Ok {
                value: value.map(|value| value.to_string()),
                writes: serde_json::to_string(&entries).unwrap(),
            },
        })
        .await;
    }

    async fn done_err(&self, error_json: &str) {
        self.send(WorkerToHostMessage::Done {
            ok: false,
            outcome: DoneOutcome::Err { error: error_json.to_string() },
        })
        .await;
    }
}

struct FakeHandle {
    join: ParkingMutex<Option<tokio::task::JoinHandle<()>>>,
    exit_code: i32,
}

impl WorkerHandle for FakeHandle {
    fn terminate(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let join = self.join.lock().take();
        Box::pin(async move {
            if let Some(join) = join {
                // A well-behaved worker exits on the interrupt flag; allow that before
                // hard-killing, mirroring the upstream terminate ordering.
                for _ in 0..50 {
                    if join.is_finished() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                join.abort();
                let _ = join.await;
            }
        })
    }

    fn exit_code(&self) -> i32 {
        self.exit_code
    }
}

#[derive(Default)]
struct FakeWorkerFactoryInner {
    behavior: Option<Behavior>,
    spawn_error: Option<String>,
    exit_code: i32,
    spawned: usize,
}

/// In-process stand-in for the upstream Node worker.
struct FakeWorkerFactory {
    inner: Arc<ParkingMutex<FakeWorkerFactoryInner>>,
}

impl FakeWorkerFactory {
    fn new(behavior: impl Fn(FakeWorkerApi) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync + 'static) -> Self {
        Self {
            inner: Arc::new(ParkingMutex::new(FakeWorkerFactoryInner {
                behavior: Some(Arc::new(behavior)),
                ..Default::default()
            })),
        }
    }

    fn with_exit_code(behavior: impl Fn(FakeWorkerApi) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync + 'static, exit_code: i32) -> Self {
        Self {
            inner: Arc::new(ParkingMutex::new(FakeWorkerFactoryInner {
                behavior: Some(Arc::new(behavior)),
                exit_code,
                ..Default::default()
            })),
        }
    }

}

impl CodeModeWorkerFactory for FakeWorkerFactory {
    fn spawn(&self, data: WorkerData) -> Result<RunningWorker, String> {
        let mut inner = self.inner.lock();
        inner.spawned += 1;
        if let Some(error) = inner.spawn_error.clone() {
            return Err(error);
        }
        let behavior = inner.behavior.clone().expect("fake behavior");
        let exit_code = inner.exit_code;
        drop(inner);
        let worker_interrupt = data.interrupt.clone();

        let (to_host, from_host_rx) = mpsc::channel(64);
        let (to_worker_tx, from_host_rx_2) = mpsc::channel(64);
        let replies: Arc<ParkingMutex<std::collections::HashMap<u64, (bool, Option<String>)>>> =
            Arc::new(ParkingMutex::new(std::collections::HashMap::new()));
        let reply_notify = Arc::new(tokio::sync::Notify::new());
        // Relay task: routes host replies by id, like the prelude's `settle`.
        {
            let replies = replies.clone();
            let reply_notify = reply_notify.clone();
            let mut from_host_rx_2 = from_host_rx_2;
            tokio::spawn(async move {
                while let Some(HostToWorkerMessage::Result { id, ok, payload }) = from_host_rx_2.recv().await {
                    replies.lock().insert(id, (ok, payload));
                    reply_notify.notify_waiters();
                }
            });
        }
        let api = FakeWorkerApi {
            data,
            to_host,
            replies,
            reply_notify,
        };
        // Keep the worker->host sender alive even if the behavior drops the api: a real
        // worker stays alive until terminated.
        let keep_alive = api.to_host.clone();
        let join = tokio::spawn(async move {
            behavior(api).await;
            drop(keep_alive); // closes the worker->host channel like a worker exit
        });
        Ok(RunningWorker {
            to_worker: to_worker_tx,
            from_worker: from_host_rx,
            interrupt: worker_interrupt,
            handle: Box::new(FakeHandle { join: ParkingMutex::new(Some(join)), exit_code }),
        })
    }
}

fn sandbox_with(behavior: impl Fn(FakeWorkerApi) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync + 'static) -> CodemodeSandbox {
    CodemodeSandbox::new(CodemodeSandboxOptions {
        worker_factory: Some(Arc::new(FakeWorkerFactory::new(behavior))),
        wasm_loader: Some(dummy_loader()),
        ..Default::default()
    })
    .unwrap()
}

fn sync_tool(name: &str, result: Value) -> CodemodeTool {
    CodemodeTool::new(name.to_string(), move |_args, _ctx| {
        let result = result.clone();
        async move { Ok(Some(result)) }
    })
}

fn script_error_json(name: &str, message: &str) -> String {
    json!({ "name": name, "message": message }).to_string()
}

async fn result_of(future: impl Future<Output = Result<CodemodeResult, String>>) -> CodemodeResult {
    future.await.expect("execute should not reject")
}

#[tokio::test]
async fn returns_the_scripts_value_after_a_json_round_trip() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            let _ = api.data.code.as_str();
            // The value crossed as JSON text; the host parses it back.
            api.done_ok(Some(json!({ "a": 1, "b": [true, "x"] })), CodemodeStoreWrites::default()).await;
        })
    });
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!({ "a": 1, "b": [true, "x"] })));
    assert!(result.is_ok());
    assert!(result.output().is_empty());
    assert!(result.calls().is_empty());

    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    let _ = result;
    let sandbox2 = sandbox_with(|api| {
        Box::pin(async move {
            api.done_ok(Some(json!("plain")), CodemodeStoreWrites::default()).await;
        })
    });
    let result = result_of(sandbox2.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!("plain")));

    let sandbox3 = sandbox_with(|api| {
        Box::pin(async move {
            api.done_ok(None, CodemodeStoreWrites::default()).await;
        })
    });
    let result = result_of(sandbox3.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), None, "empty script completes with value undefined");
}

#[tokio::test]
async fn terminates_a_script_that_never_settles_on_timeout() {
    let sandbox = sandbox_with(|_api| {
        Box::pin(async move {
            // A script spinning in the VM never posts anything.
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        })
    });
    let started = std::time::Instant::now();
    let result = result_of(
        sandbox.execute(String::new(), CodemodeExecuteOptions { timeout_ms: Some(200.0), ..Default::default() }),
    )
    .await;
    assert_eq!(result.error().map(|error| error.kind), Some(CodemodeErrorKind::Timeout), "msg={:?}", result.error().map(|e| e.message.clone()));
    assert_eq!(
        result.error().map(|error| error.message.clone()),
        Some("Execution timed out after 200 ms".to_string())
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

#[tokio::test]
async fn runs_without_a_deadline_when_timeout_is_infinity() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            let value = api.call(CallTarget::Tool, "wait", Some(json!(1))).await.unwrap();
            api.done_ok(value, CodemodeStoreWrites::default()).await;
        })
    });
    sandbox
        .register_tool(CodemodeTool::new("wait", |args, _ctx| async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Ok(args)
        }))
        .unwrap();
    let result = result_of(
        sandbox.execute(
            String::new(),
            CodemodeExecuteOptions { timeout_ms: Some(f64::INFINITY), ..Default::default() },
        ),
    )
    .await;
    assert_eq!(result.value(), Some(&json!(1)));
}

#[tokio::test]
async fn fails_a_script_that_waits_on_a_promise_nothing_can_settle() {
    // The prelude's `stalled()` reports this; the host surfaces it as a script error.
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            api.done_err(
                &json!({
                    "name": "Error",
                    "message": "The script is waiting on a promise that can never settle: no tool call is pending, and timers do not exist here.",
                })
                .to_string(),
            )
            .await;
        })
    });
    let result = result_of(
        sandbox.execute(
            String::new(),
            CodemodeExecuteOptions { timeout_ms: Some(f64::INFINITY), ..Default::default() },
        ),
    )
    .await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Script);
    assert!(error.message.contains("can never settle"));
}

#[tokio::test]
async fn collects_output_in_order() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            api.output_text("hello 1 {\"a\":1}").await;
            api.output_text("second").await;
            api.send(WorkerToHostMessage::Output {
                item: pi_codemode::types::CodemodeOutputItem::Image {
                    data: "iVBORw0KGgo=".to_string(),
                    mime_type: "image/png".to_string(),
                },
            })
            .await;
            api.done_ok(None, CodemodeStoreWrites::default()).await;
        })
    });
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert!(result.is_ok());
    assert_eq!(result.output().len(), 3);
    assert_eq!(
        result.output()[0],
        pi_codemode::types::CodemodeOutputItem::Text { text: "hello 1 {\"a\":1}".to_string() }
    );
    assert_eq!(
        result.output()[2],
        pi_codemode::types::CodemodeOutputItem::Image {
            data: "iVBORw0KGgo=".to_string(),
            mime_type: "image/png".to_string()
        }
    );
}

#[tokio::test]
async fn reports_a_crash_as_a_sandbox_error() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            api.send(WorkerToHostMessage::Crash { message: "RuntimeError: unreachable".to_string() }).await;
        })
    });
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Sandbox);
    assert_eq!(error.message, "RuntimeError: unreachable");
}

#[tokio::test]
async fn reports_a_worker_exit_before_settle_as_a_sandbox_error() {
    let sandbox = {
        let factory = FakeWorkerFactory::with_exit_code(
            |_api| Box::pin(async move { /* exits without reporting anything */ }),
            3,
        );
        CodemodeSandbox::new(CodemodeSandboxOptions {
            worker_factory: Some(Arc::new(factory)),
            wasm_loader: Some(dummy_loader()),
            ..Default::default()
        })
        .unwrap()
    };
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Sandbox);
    assert_eq!(error.message, "Worker exited with code 3 before the script settled");
}

#[tokio::test]
async fn reports_a_missing_worker_file_as_a_sandbox_error() {
    // The default factory has no worker implementation, mirroring a missing worker file.
    let sandbox = CodemodeSandbox::new(CodemodeSandboxOptions {
        wasm_loader: Some(dummy_loader()),
        ..Default::default()
    })
    .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Sandbox);
    assert!(error.message.starts_with("Failed to start worker: "));
}

#[tokio::test]
async fn reports_a_failing_wasm_module_as_a_sandbox_error() {
    struct FailingLoader;
    impl pi_codemode::wasm::WasmLoader for FailingLoader {
        fn load(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<pi_codemode::CodemodeWasmModule, String>> + Send + '_>> {
            Box::pin(async { Err("no wasm".to_string()) })
        }
    }
    let sandbox = CodemodeSandbox::new(CodemodeSandboxOptions {
        wasm_loader: Some(Arc::new(FailingLoader)),
        ..Default::default()
    })
    .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Sandbox);
    assert_eq!(error.message, "Failed to load QuickJS: no wasm");
}

#[tokio::test]
async fn exposes_tools_and_records_calls() {
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_tool = seen.clone();
    let sandbox = sandbox_with(move |api| {
        let seen = seen_tool.clone();
        Box::pin(async move {
            let first = api.call(CallTarget::Tool, "add", Some(json!({ "a": 1, "b": 2 }))).await.unwrap();
            let second = api.call(CallTarget::Tool, "add", Some(json!({ "a": 3, "b": 10 }))).await.unwrap();
            let sum = second.as_ref().and_then(|v| v.get("sum")).cloned().unwrap();
            drop(first);
            api.done_ok(Some(json!({ "sum": sum })), CodemodeStoreWrites::default()).await;
            seen.lock().push(json!({ "a": 1, "b": 2 }));
            seen.lock().push(json!({ "a": 3, "b": 10 }));
        })
    });
    sandbox
        .register_tool(CodemodeTool::new("add", |args, _ctx| async move {
            let a = args.as_ref().and_then(|args| args.get("a")).and_then(Value::as_i64).unwrap();
            let b = args.as_ref().and_then(|args| args.get("b")).and_then(Value::as_i64).unwrap();
            Ok(Some(json!({ "sum": a + b })))
        }))
        .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!({ "sum": 13 })));
    assert_eq!(
        seen.lock().as_slice(),
        &[json!({ "a": 1, "b": 2 }), json!({ "a": 3, "b": 10 })]
    );
    let calls = result.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.name == "add" && call.status == pi_codemode::types::CodemodeCallStatus::Ok));
    assert!(calls.iter().all(|call| call.duration_ms >= 0.0));
}

#[tokio::test]
async fn turns_tool_errors_into_catchable_errors() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            let outcome = api.call(CallTarget::Tool, "fail", None).await;
            match outcome {
                Err(message) => {
                    // The script catches it as an Error with the same message.
                    api.done_ok(Some(json!({ "isError": true, "message": message })), CodemodeStoreWrites::default()).await;
                }
                Ok(_) => {
                    api.done_ok(Some(json!("no error")), CodemodeStoreWrites::default()).await;
                }
            }
        })
    });
    sandbox
        .register_tool(CodemodeTool::new("fail", |_args, _ctx| async move {
            Err("tool exploded".to_string())
        }))
        .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!({ "isError": true, "message": "tool exploded" })));
    let calls = result.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "fail");
    assert_eq!(calls[0].status, pi_codemode::types::CodemodeCallStatus::Error);
}

#[tokio::test]
async fn rejects_calls_to_unknown_tools() {
    let reply: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let reply_clone = reply.clone();
    let sandbox = sandbox_with(move |api| {
        let reply = reply_clone.clone();
        Box::pin(async move {
            let outcome = api.call(CallTarget::Tool, "missing", None).await;
            match outcome {
                Err(message) => {
                    reply.lock().push(message);
                    api.done_err(&script_error_json("TypeError", "tools.missing is not a function")).await;
                }
                Ok(_) => api.done_ok(Some(json!("imported")), CodemodeStoreWrites::default()).await,
            }
        })
    });
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(*reply.lock(), vec!["Unknown tool \"missing\"".to_string()]);
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Script);
    assert_eq!(error.name.as_deref(), Some("TypeError"));
}

#[tokio::test]
async fn aborts_unawaited_calls_when_the_script_returns() {
    let aborted = Arc::new(AtomicI32::new(0));
    let aborted_tool = aborted.clone();
    let sandbox = sandbox_with(move |api| {
        Box::pin(async move {
            // The script fires the call without awaiting and returns.
            let _ = api.to_host.send(WorkerToHostMessage::Call {
                id: 1,
                target: CallTarget::Tool,
                name: "slow".to_string(),
                args: None,
            }).await;
            api.done_ok(Some(json!("early")), CodemodeStoreWrites::default()).await;
        })
    });
    sandbox
        .register_tool(CodemodeTool::new("slow", move |_args, ctx: CodemodeToolContext| {
            let aborted = aborted_tool.clone();
            async move {
                ctx.signal.wait_aborted().await;
                aborted.store(1, Ordering::SeqCst);
                Err("aborted".to_string())
            }
        }))
        .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!("early")));
    let calls = result.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "slow");
    assert_eq!(calls[0].status, pi_codemode::types::CodemodeCallStatus::Cancelled);
    for _ in 0..100 {
        if aborted.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(aborted.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn runs_concurrent_calls() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            let (a, b, c) = tokio::join!(
                api.call(CallTarget::Tool, "delay", Some(json!(1))),
                api.call(CallTarget::Tool, "delay", Some(json!(2))),
                api.call(CallTarget::Tool, "echo", Some(json!(3))),
            );
            api.done_ok(
                Some(json!({ "values": [a.unwrap(), b.unwrap(), c.unwrap()] })),
                CodemodeStoreWrites::default(),
            )
            .await;
        })
    });
    sandbox
        .register_tool(CodemodeTool::new("echo", |args, _ctx| async move { Ok(args) }))
        .unwrap();
    sandbox
        .register_tool(CodemodeTool::new("delay", |args, _ctx| async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Ok(args)
        }))
        .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!({ "values": [1, 2, 3] })));
    assert_eq!(result.calls().len(), 3);
}

#[tokio::test]
async fn passes_undefined_arguments_and_results_through() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            let first = api.call(CallTarget::Tool, "noop", None).await.unwrap();
            let second = api.call(CallTarget::Tool, "noop", Some(json!(null))).await.unwrap();
            api.done_ok(Some(json!([first, second])), CodemodeStoreWrites::default()).await;
        })
    });
    sandbox
        .register_tool(CodemodeTool::new("noop", |args, _ctx| async move { Ok(args) }))
        .unwrap();
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!([null, null])));
}

#[tokio::test]
async fn exposes_globals_without_recording_calls() {
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_attach = seen.clone();
    let sandbox = {
        let factory = FakeWorkerFactory::new(move |api| {
            Box::pin(async move {
                let _ = api.call(CallTarget::Global, "attach", Some(json!({ "ref": 1 }))).await;
                let _ = api.call(CallTarget::Global, "attach", Some(json!("not awaited"))).await;
                let echo = api.call(CallTarget::Tool, "echo", Some(json!(2))).await.unwrap();
                api.done_ok(Some(json!(["function", "function", echo])), CodemodeStoreWrites::default()).await;
            })
        });
        CodemodeSandbox::new(CodemodeSandboxOptions {
            wasm_loader: Some(dummy_loader()),
            tools: vec![CodemodeTool::new("echo", |args, _ctx| async move { Ok(args) })],
            globals: vec![CodemodeTool::new("attach", move |args, _ctx| {
                let seen = seen_attach.clone();
                async move {
                    seen.lock().push(args.unwrap_or(Value::Null));
                    Ok(None)
                }
            })],
            worker_factory: Some(Arc::new(factory)),
            ..Default::default()
        })
        .unwrap()
    };
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!(["function", "function", 2])));
    // Globals are not recorded as calls.
    let calls = result.calls();
    assert_eq!(calls.iter().map(|call| call.name.as_str()).collect::<Vec<_>>(), vec!["echo"]);
    assert_eq!(
        seen.lock().as_slice(),
        &[json!({ "ref": 1 }), json!("not awaited")]
    );
}

#[tokio::test]
async fn reads_the_store_snapshot_and_reports_writes() {
    let snapshot: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let snapshot_clone = snapshot.clone();
    let sandbox = sandbox_with(move |api| {
        let snapshot = snapshot_clone.clone();
        Box::pin(async move {
            snapshot.lock().push(json!(api.data.store));
            let writes = CodemodeStoreWrites {
                set: {
                    let mut set = serde_json::Map::new();
                    set.insert("counter".to_string(), json!(42));
                    set.insert("list".to_string(), json!([1, { "a": null }]));
                    set
                },
                delete: vec!["old".to_string()],
            };
            api.done_ok(Some(json!([41, 42, null, null])), writes).await;
        })
    });
    let mut store = serde_json::Map::new();
    store.insert("counter".to_string(), json!(41));
    store.insert("old".to_string(), json!("x"));
    let result = result_of(
        sandbox.execute(
            String::new(),
            CodemodeExecuteOptions { store: Some(store), ..Default::default() },
        ),
    )
    .await;
    // The worker receives key -> JSON text.
    assert_eq!(snapshot.lock()[0], json!({ "counter": "41", "old": "\"x\"" }));
    assert_eq!(result.value(), Some(&json!([41, 42, null, null])), "result: {:?}", result);
    let writes = result.store_writes().unwrap();
    assert_eq!(writes.set.get("counter"), Some(&json!(42)));
    assert_eq!(writes.set.get("list"), Some(&json!([1, { "a": null }])));
    assert_eq!(writes.delete, vec!["old".to_string()]);
}

#[tokio::test]
async fn supports_register_and_unregister_between_executions() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            let _ = api.call(CallTarget::Tool, "echo", Some(json!("a"))).await.unwrap();
            api.done_ok(Some(json!("a")), CodemodeStoreWrites::default()).await;
        })
    });
    sandbox.register_tool(sync_tool("echo", json!(0))).unwrap();
    assert!(sandbox.register_tool(sync_tool("echo", json!(0))).is_err());
    assert!(!sandbox.register_tool(sync_tool("echo", json!(0))).unwrap_err().is_empty());
    assert!(sandbox
        .register_tool(sync_tool("echo", json!(0)))
        .unwrap_err()
        .contains("already registered"));
    assert_eq!(sandbox.tools().iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(), vec!["echo"]);
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert_eq!(result.value(), Some(&json!("a")));
    assert!(sandbox.unregister_tool("echo"));
    assert!(!sandbox.unregister_tool("echo"));
}

#[tokio::test]
async fn rejects_invalid_and_reserved_global_names() {
    let execute_tool = || CodemodeTool::new("x", |_args, _ctx| async { Ok(None) });
    for name in ["a.b.c", "a.", ".a", "tools.x", "store.x", "a.not-valid"] {
        let error = CodemodeSandbox::new(CodemodeSandboxOptions {
            globals: vec![{
                let mut tool = execute_tool();
                tool.name = name.to_string();
                tool
            }],
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.contains("Invalid global"), "name: {name}");
    }
    let namespace_conflict = CodemodeSandbox::new(CodemodeSandboxOptions {
        globals: vec![
            {
                let mut tool = execute_tool();
                tool.name = "models".to_string();
                tool
            },
            {
                let mut tool = execute_tool();
                tool.name = "models.list".to_string();
                tool
            },
        ],
        ..Default::default()
    })
    .unwrap_err();
    assert!(namespace_conflict.contains("conflicts with the namespace"));
    for name in ["not-valid", "tools", "console", "store", "load"] {
        let error = CodemodeSandbox::new(CodemodeSandboxOptions {
            globals: vec![{
                let mut tool = execute_tool();
                tool.name = name.to_string();
                tool
            }],
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.contains("Invalid global"), "name: {name}");
    }
}

#[tokio::test]
async fn rejects_duplicate_tool_registration() {
    let sandbox = CodemodeSandbox::new(CodemodeSandboxOptions {
        tools: vec![sync_tool("echo", json!(1))],
        ..Default::default()
    })
    .unwrap();
    assert!(sandbox.register_tool(sync_tool("echo", json!(2))).unwrap_err().contains("already registered"));
}

#[tokio::test]
async fn aborts_via_signal_and_cancels_in_flight_calls() {
    let tool_signal: Arc<ParkingMutex<Option<AbortSignal>>> = Arc::new(ParkingMutex::new(None));
    let (called_tx, mut called_rx) = mpsc::channel::<()>(8);

    let factory_signal = tool_signal.clone();
    let factory = Arc::new(FakeWorkerFactory::new(move |api| {
        let tool_signal = factory_signal.clone();
        Box::pin(async move {
            // Mirrors the prelude awaiting `tools.hang()`: the reply only ever arrives via
            // cancellation.
            let outcome = api.call(CallTarget::Tool, "hang", None).await;
            assert!(outcome.is_err(), "hang resolves only through cancellation");
            let _ = tool_signal;
        })
    }));

    let sandbox = {
        let tool_signal = tool_signal.clone();
        let called_tx = called_tx.clone();
        CodemodeSandbox::new(CodemodeSandboxOptions {
            tools: vec![CodemodeTool::new("hang", move |_args, ctx: CodemodeToolContext| {
                let tool_signal = tool_signal.clone();
                let called_tx = called_tx.clone();
                async move {
                    *tool_signal.lock() = Some(ctx.signal.clone());
                    let _ = called_tx.send(()).await;
                    ctx.signal.wait_aborted().await;
                    Err("cancelled".to_string())
                }
            })],
            worker_factory: Some(factory),
            wasm_loader: Some(dummy_loader()),
            ..Default::default()
        })
        .unwrap()
    };

    // First execution hangs on the tool; it is only resolved by close() below. Spawned so it
    // starts eagerly, like an upstream promise.
    let promise = {
        let sandbox = sandbox.clone();
        tokio::spawn(async move { sandbox.execute(String::new(), CodemodeExecuteOptions::default()).await })
    };
    let _ = called_rx.recv().await;

    let controller = AbortSignal::new();
    controller.abort("user cancelled");
    // An already-aborted signal aborts the execution immediately.
    let result = result_of(
        sandbox.execute(
            String::new(),
            CodemodeExecuteOptions { signal: Some(controller.clone()), ..Default::default() },
        ),
    )
    .await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Aborted);
    assert_eq!(error.message, "user cancelled");

    // A pending execution aborted mid-call records the cancelled call.
    let second = AbortSignal::new();
    let pending = {
        let sandbox = sandbox.clone();
        let second = second.clone();
        tokio::spawn(async move {
            sandbox
                .execute(
                    String::new(),
                    CodemodeExecuteOptions { signal: Some(second), ..Default::default() },
                )
                .await
        })
    };
    let _ = called_rx.recv().await;
    second.abort("stop");
    let aborted = pending.await.unwrap().unwrap();
    assert_eq!(aborted.error().unwrap().kind, CodemodeErrorKind::Aborted);
    assert_eq!(aborted.calls().len(), 1);
    assert_eq!(aborted.calls()[0].status, pi_codemode::types::CodemodeCallStatus::Cancelled);
    assert!(tool_signal.lock().as_ref().map(|signal| signal.is_aborted()).unwrap_or(false));

    while called_rx.try_recv().is_ok() {}
    // close() aborts in-flight executions.
    sandbox.close().await;
    let closed = promise.await.unwrap().unwrap();
    let error = closed.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Aborted);
    assert_eq!(error.message, "Sandbox closed");
    assert_eq!(closed.error().unwrap().kind, CodemodeErrorKind::Aborted);
    assert_eq!(closed.error().unwrap().message, "Sandbox closed");
}

#[tokio::test]
async fn rejects_execute_after_close() {
    let sandbox = sandbox_with(|_api| Box::pin(async {}));
    sandbox.close().await;
    let error = sandbox.execute(String::new(), CodemodeExecuteOptions::default()).await.unwrap_err();
    assert!(error.contains("closed"));
}

#[tokio::test]
async fn runs_executions_in_parallel_without_sharing_state() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            // Each worker sees only its own data; the fake echoes the code it was handed,
            // which is the per-execution state the real VM would isolate.
            let shared = if api.data.code == "a" { "a" } else if api.data.code == "b" { "b" } else { "undefined" };
            api.done_ok(Some(json!(shared)), CodemodeStoreWrites::default()).await;
        })
    });
    let results = tokio::join!(
        sandbox.execute("a".to_string(), CodemodeExecuteOptions::default()),
        sandbox.execute("b".to_string(), CodemodeExecuteOptions::default()),
        sandbox.execute(String::new(), CodemodeExecuteOptions::default()),
    );
    let (r1, r2, r3) = results;
    let values: Vec<Value> = [r1, r2, r3]
        .into_iter()
        .map(|result| result.unwrap().value().cloned().unwrap_or(Value::Null))
        .collect();
    assert_eq!(values, vec![json!("a"), json!("b"), json!("undefined")]);
}

#[tokio::test]
async fn keeps_output_before_a_failure() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            api.output_text("partial").await;
            api.done_err(&script_error_json("Error", "boom")).await;
        })
    });
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    assert!(!result.is_ok());
    assert_eq!(
        result.output(),
        &[pi_codemode::types::CodemodeOutputItem::Text { text: "partial".to_string() }]
    );
    assert_eq!(result.error().unwrap().message, "boom");
}

#[tokio::test]
async fn reports_script_errors_with_parsed_name_and_stack() {
    let sandbox = sandbox_with(|api| {
        Box::pin(async move {
            api.done_err(
                &json!({
                    "name": "TypeError",
                    "message": "boom 1",
                    "stack": "TypeError: boom 1\n  at codemode.js:2",
                })
                .to_string(),
            )
            .await;
        })
    });
    let result = result_of(sandbox.execute(String::new(), CodemodeExecuteOptions::default())).await;
    let error = result.error().unwrap();
    assert_eq!(error.kind, CodemodeErrorKind::Script);
    assert_eq!(error.name.as_deref(), Some("TypeError"));
    assert_eq!(error.message, "boom 1");
    assert_eq!(error.stack.as_deref(), Some("TypeError: boom 1\n  at codemode.js:2"));
}

#[tokio::test]
async fn sets_the_interrupt_flag_before_terminating() {
    let interrupt_seen = Arc::new(AtomicI32::new(0));
    let interrupt_for_behavior = interrupt_seen.clone();
    let sandbox = {
        let factory = FakeWorkerFactory::new(move |api| {
            let interrupt_seen = interrupt_for_behavior.clone();
            Box::pin(async move {
                // Mirror the real worker: watch the interrupt flag.
                let interrupt = api.data.interrupt.clone();
                loop {
                    if interrupt.load(Ordering::SeqCst) != 0 {
                        interrupt_seen.store(1, Ordering::SeqCst);
                        return;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            })
        });
        CodemodeSandbox::new(CodemodeSandboxOptions {
            worker_factory: Some(Arc::new(factory)),
            wasm_loader: Some(dummy_loader()),
            ..Default::default()
        })
        .unwrap()
    };
    let result = result_of(
        sandbox.execute(String::new(), CodemodeExecuteOptions { timeout_ms: Some(150.0), ..Default::default() }),
    )
    .await;
    assert_eq!(result.error().unwrap().kind, CodemodeErrorKind::Timeout);
    assert_eq!(interrupt_seen.load(Ordering::SeqCst), 1);
}
