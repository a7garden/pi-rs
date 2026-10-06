//! pi-codemode trait implementations over an embedded deno_core `JsRuntime`.
//!
//! Upstream spawns a Node `worker_thread` owning a QuickJS-wasi VM per
//! execution. Here, [`DenoCodeModeWorkerFactory::spawn`] starts one OS thread
//! owning a fresh `JsRuntime`; the thread evaluates the verbatim upstream
//! prelude ([`pi_codemode::runtime::prelude_source::PRELUDE_SOURCE`]) and drives
//! the host/worker message protocol through a single sync bridge op.
//!
//! Termination: the host sets the shared [`WorkerData::interrupt`] flag and
//! calls [`WorkerHandle::terminate`]. V8's `IsolateHandle::terminate_execution`
//! is explicitly thread-safe ("can be used by any thread"), so no dedicated
//! watchdog thread is needed: terminate requests an execution-terminating
//! exception from any thread, and the worker loop — which polls the interrupt
//! flag between event-loop ticks — exits promptly. `memory_limit_bytes` maps to
//! a V8 near-heap-limit callback that requests termination (upstream: a wasm
//! memory cap).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;

use deno_core::v8::IsolateHandle;
use deno_core::{OpState, PollEventLoopOptions};
use pi_codemode::runtime::host::{CodeModeWorkerFactory, RunningWorker, WorkerHandle};
use pi_codemode::runtime::prelude_source::PRELUDE_SOURCE;
use pi_codemode::runtime::protocol::{
    CallTarget, DoneOutcome, HostToWorkerMessage, WorkerData, WorkerToHostMessage,
};
use pi_codemode::types::CodemodeOutputItem;
use pi_codemode::wasm::{CodemodeWasmModule, WasmLoader};
use tokio::sync::{mpsc, oneshot};

use crate::host::{make_runtime, spawn_js_thread};

/// The per-execution state shared between the bridge op and the worker loop.
struct WorkerShared {
    /// The worker→host channel; ops push protocol messages onto it.
    to_host: mpsc::Sender<WorkerToHostMessage>,
    /// `bridge("call"|"global", …)` calls awaiting a `result` reply.
    pending: AtomicU64,
    /// Set once `bridge("done", …)` fired; the worker exits soon after.
    done: AtomicBool,
}

/// pi-codemode worker factory backed by an embedded `JsRuntime`.
pub struct DenoCodeModeWorkerFactory;

impl CodeModeWorkerFactory for DenoCodeModeWorkerFactory {
    fn spawn(&self, data: WorkerData) -> Result<RunningWorker, String> {
        spawn_worker(data)
    }
}

/// [`WasmLoader`] for this runtime. Upstream loads the compiled QuickJS-wasi
/// binary; the embedded runtime replaces QuickJS with V8, so no wasm is
/// required. The loader yields an opaque no-op token to satisfy the sandbox's
/// loading contract (failures would surface as sandbox errors instead).
pub struct DenoWasmLoader;

#[derive(Default)]
struct WasmToken;

impl WasmLoader for DenoWasmLoader {
    fn load(&self) -> Pin<Box<dyn Future<Output = Result<CodemodeWasmModule, String>> + Send + '_>> {
        Box::pin(async { Ok(Arc::new(WasmToken) as CodemodeWasmModule) })
    }
}

/// Termination surface of a `DenoCodeModeWorkerFactory` worker.
struct DenoWorkerHandle {
    /// Published by the worker thread right after runtime creation.
    isolate: tokio::sync::watch::Receiver<Option<IsolateHandle>>,
    /// Exit notification (oneshot consumed by the first terminate).
    exit: Mutex<Option<oneshot::Receiver<()>>>,
    /// Host-requested termination (set before isolating).
    terminate_requested: Arc<AtomicI32>,
    exit_code: Arc<AtomicI32>,
}

impl WorkerHandle for DenoWorkerHandle {
    fn terminate(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        self.terminate_requested.store(1, Ordering::SeqCst);
        let mut isolate = self.isolate.clone();
        let exit = self.exit.lock().take();
        Box::pin(async move {
            // The handle is published before any script runs, so this resolves
            // immediately in practice.
            if isolate.changed().await.is_ok() {
                if let Some(isolate) = isolate.borrow().as_ref() {
                    isolate.terminate_execution();
                }
            }
            if let Some(exit) = exit {
                let _ = exit.await;
            }
        })
    }

    fn exit_code(&self) -> i32 {
        self.exit_code.load(Ordering::SeqCst)
    }
}

fn spawn_worker(data: WorkerData) -> Result<RunningWorker, String> {
    let (to_worker, mut to_worker_rx) = mpsc::channel::<HostToWorkerMessage>(64);
    let (to_host, from_worker) = mpsc::channel::<WorkerToHostMessage>(256);
    let (exit_tx, exit_rx) = oneshot::channel::<()>();
    let (isolate_tx, isolate_rx) = tokio::sync::watch::channel(None::<IsolateHandle>);
    let terminate_requested = Arc::new(AtomicI32::new(0));
    let exit_code = Arc::new(AtomicI32::new(1));

    let thread_data = data;
    let thread_terminate = Arc::clone(&terminate_requested);
    let thread_exit_code = Arc::clone(&exit_code);
    let code = thread_data.code.clone();
    // The prelude reads the upstream workerData JSON shape (`jsName` is
    // camelCase there; pi-codemode's Rust structs use `js_name`).
    let tools_json = serde_json::to_string(
        &thread_data
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "jsName": tool.js_name,
                    "description": tool.description,
                })
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    let globals_json = serde_json::to_string(
        &thread_data
            .globals
            .iter()
            .map(|global| serde_json::json!({ "name": global.name, "spread": global.spread }))
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    let store_json = serde_json::to_string(&thread_data.store).map_err(|e| e.to_string())?;
    let memory_limit = thread_data.memory_limit_bytes;
    let interrupt = Arc::clone(&thread_data.interrupt);
    // `data.wasm` (QuickJS binary) is intentionally ignored: V8 replaces QuickJS.

    let shared = Arc::new(WorkerShared {
        to_host: to_host.clone(),
        pending: AtomicU64::new(0),
        done: AtomicBool::new(false),
    });
    let loop_shared = Arc::clone(&shared);

    let worker_interrupt = Arc::clone(&interrupt);
    spawn_js_thread(move || async move {
        let mut runtime = make_runtime(vec![deno_codemode_ops::init()], None);
        runtime.op_state().borrow_mut().put(shared.clone());
        let _ = isolate_tx.send(Some(runtime.v8_isolate().thread_safe_handle()));

        if memory_limit.is_some() {
            let isolate = runtime.v8_isolate().thread_safe_handle();
            // The callback fires near V8's internal limit scaled from the
            // requested cap; request V8 termination instead of a fatal OOM
            // crash, and give the isolate headroom to unwind the terminating
            // exception.
            runtime.add_near_heap_limit_callback(move |current, _heap| {
                isolate.terminate_execution();
                current.saturating_add(64 * 1024 * 1024)
            });
        }

        run_worker_loop(
            &mut runtime,
            loop_shared,
            &mut to_worker_rx,
            interrupt,
            thread_terminate,
            thread_exit_code,
            code,
            tools_json,
            globals_json,
            store_json,
        )
        .await;

        let _ = exit_tx.send(());
    });

    Ok(RunningWorker {
        to_worker,
        from_worker,
        interrupt: worker_interrupt,
        handle: Box::new(DenoWorkerHandle {
            isolate: isolate_rx,
            exit: Mutex::new(Some(exit_rx)),
            terminate_requested,
            exit_code,
        }),
    })
}

/// The single synchronous bridge the upstream prelude calls with primitives
/// only: `bridge(kind, a, b, c)`. Empty strings encode `undefined`.
#[deno_core::op2(fast)]
fn op_codemode_bridge(
    state: &mut OpState,
    #[string] kind: &str,
    #[string] a: &str,
    #[string] b: &str,
    #[string] c: &str,
) -> Result<(), deno_error::JsErrorBox> {
    let shared = state.borrow::<Arc<WorkerShared>>().clone();
    let message = match kind {
        "call" | "global" => {
            let id = a.parse::<u64>().unwrap_or(0);
            shared.pending.fetch_add(1, Ordering::SeqCst);
            WorkerToHostMessage::Call {
                id,
                target: if kind == "call" { CallTarget::Tool } else { CallTarget::Global },
                name: b.to_string(),
                args: (!c.is_empty()).then(|| c.to_string()),
            }
        }
        "output" => {
            let item = if a == "image" {
                CodemodeOutputItem::Image { data: b.to_string(), mime_type: c.to_string() }
            } else {
                CodemodeOutputItem::Text { text: b.to_string() }
            };
            WorkerToHostMessage::Output { item }
        }
        "done" => {
            shared.done.store(true, Ordering::SeqCst);
            let ok = a == "true" || a == "1";
            if ok {
                WorkerToHostMessage::Done {
                    ok: true,
                    outcome: DoneOutcome::Ok {
                        value: (!b.is_empty()).then(|| b.to_string()),
                        writes: c.to_string(),
                    },
                }
            } else {
                WorkerToHostMessage::Done {
                    ok: false,
                    outcome: DoneOutcome::Err { error: b.to_string() },
                }
            }
        }
        other => {
            return Err(deno_error::JsErrorBox::generic(format!(
                "unknown bridge kind {other}"
            )));
        }
    };
    // The host drains this channel continuously while an execution runs.
    if shared.to_host.try_send(message).is_err() {
        return Err(deno_error::JsErrorBox::generic(
            "codemode worker: host message channel is full or closed",
        ));
    }
    Ok(())
}

deno_core::extension!(
    deno_codemode_ops,
    ops = [op_codemode_bridge],
);

#[allow(clippy::too_many_arguments)]
async fn run_worker_loop(
    runtime: &mut deno_core::JsRuntime,
    shared: Arc<WorkerShared>,
    to_worker_rx: &mut mpsc::Receiver<HostToWorkerMessage>,
    interrupt: Arc<AtomicI32>,
    terminate_requested: Arc<AtomicI32>,
    exit_code: Arc<AtomicI32>,
    code: String,
    tools_json: String,
    globals_json: String,
    store_json: String,
) {
    // Evaluate the upstream prelude verbatim and bind its API plus a bridge
    // wrapper normalizing the prelude's primitive arguments to strings.
    let bootstrap = format!(
        "globalThis.__piBridge = (...args) => Deno.core.ops.op_codemode_bridge(\n\
         \x20   String(args[0]),\n\
         \x20   String(args[1] === undefined ? '' : args[1]),\n\
         \x20   String(args[2] === undefined ? '' : args[2]),\n\
         \x20   String(args[3] === undefined ? '' : args[3]),\n\
         );\n\
         const __piApi = ({prelude})(globalThis.__piBridge, {tools}, {globals}, {store});\n\
         globalThis.__piSettle = (id, ok, payload) => __piApi.settle(id, ok, payload);\n\
         globalThis.__piStalled = () => __piApi.stalled();\n\
         globalThis.__piRun = (fn) => __piApi.run(fn);",
        prelude = PRELUDE_SOURCE,
        tools = crate::host::js_string_literal(&tools_json),
        globals = crate::host::js_string_literal(&globals_json),
        store = crate::host::js_string_literal(&store_json),
    );
    if let Err(error) = runtime.execute_script("codemode-prelude.js", bootstrap) {
        send_crash(&shared, &error.to_string());
        return;
    }

    // Compile the user script. The wrapper shares the first line with the
    // script so reported line numbers match it as written (upstream parity).
    let script = format!("(globalThis.__piUserFn = (async (tools, console) => {{{code}\n}}))");
    if let Err(error) = runtime.execute_script("codemode.js", script) {
        if stopping(&interrupt, &terminate_requested) {
            return;
        }
        let error_json = crate::host::script_error_json_string(&error);
        let message = WorkerToHostMessage::Done {
            ok: false,
            outcome: DoneOutcome::Err { error: error_json },
        };
        if shared.to_host.try_send(message).is_ok() {
            shared.done.store(true, Ordering::SeqCst);
        }
        return;
    }
    if let Err(error) = runtime.execute_script("codemode-run.js", "__piRun(globalThis.__piUserFn)") {
        if stopping(&interrupt, &terminate_requested) {
            return;
        }
        send_crash(&shared, &error.to_string());
        return;
    }

    loop {
        if shared.done.load(Ordering::SeqCst) || stopping(&interrupt, &terminate_requested) {
            break;
        }
        tokio::select! {
            biased;
            message = to_worker_rx.recv() => {
                match message {
                    Some(HostToWorkerMessage::Result { id, ok, payload }) => {
                        shared.pending.fetch_sub(1, Ordering::SeqCst);
                        let payload_js = match &payload {
                            Some(payload) => serde_json::to_string(payload)
                                .unwrap_or_else(|_| "undefined".to_string()),
                            None => "undefined".to_string(),
                        };
                        let settle = format!("__piSettle({id}, {ok}, {payload_js})");
                        if let Err(error) = runtime.execute_script("codemode-settle.js", settle) {
                            if stopping(&interrupt, &terminate_requested) {
                                break;
                            }
                            send_crash(&shared, &error.to_string());
                            break;
                        }
                    }
                    None => break, // host dropped the worker
                }
            }
            polled = runtime.run_event_loop(PollEventLoopOptions::default()) => {
                if let Err(error) = polled {
                    if stopping(&interrupt, &terminate_requested) {
                        break;
                    }
                    send_crash(&shared, &error.to_string());
                    break;
                }
                if shared.done.load(Ordering::SeqCst) {
                    break;
                }
                if shared.pending.load(Ordering::SeqCst) == 0 {
                    // Idle with nothing that can ever resume the script
                    // (upstream drains pending jobs, then reports `stalled`).
                    if let Err(error) =
                        runtime.execute_script("codemode-stalled.js", "__piStalled()")
                    {
                        if stopping(&interrupt, &terminate_requested) {
                            break;
                        }
                        send_crash(&shared, &error.to_string());
                        break;
                    }
                    if shared.done.load(Ordering::SeqCst) {
                        break;
                    }
                }
            }
        }
    }
    exit_code.store(if shared.done.load(Ordering::SeqCst) { 0 } else { 1 }, Ordering::SeqCst);
}

fn stopping(interrupt: &AtomicI32, terminate_requested: &AtomicI32) -> bool {
    interrupt.load(Ordering::SeqCst) != 0 || terminate_requested.load(Ordering::SeqCst) != 0
}

fn send_crash(shared: &WorkerShared, message: &str) {
    let _ = shared.to_host.try_send(WorkerToHostMessage::Crash {
        message: message.to_string(),
    });
}
