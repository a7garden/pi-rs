//! Integration tests: running codemode scripts through the worker traits
//! (`CodeModeWorkerFactory` + `WasmLoader`) over an embedded `JsRuntime`,
//! including timeout/abort termination of runaway scripts.

use std::time::Duration;

use pi_codemode::runtime::host::{CodemodeSandbox, CodemodeSandboxOptions};
use pi_codemode::types::{CodemodeErrorKind, CodemodeOutputItem, CodemodeTool};
use pi_runtime::{DenoCodeModeWorkerFactory, DenoWasmLoader};
use serde_json::json;

fn sandbox(tools: Vec<CodemodeTool>) -> CodemodeSandbox {
    CodemodeSandbox::new(CodemodeSandboxOptions {
        tools,
        wasm_loader: Some(Arc::new(DenoWasmLoader)),
        worker_factory: Some(Arc::new(DenoCodeModeWorkerFactory)),
        ..Default::default()
    })
    .unwrap()
}

use std::sync::Arc;

async fn ok_result(result: Result<pi_codemode::types::CodemodeResult, String>) -> pi_codemode::types::CodemodeResult {
    result.unwrap_or_else(|error| panic!("sandbox error: {error}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_a_script_that_calls_tools_and_produces_output() {
    let sandbox = sandbox(vec![CodemodeTool::new("echo", |args: Option<serde_json::Value>, _ctx| async move {
        Ok(args)
    })]);

    let result = ok_result(
        sandbox
            .execute(
                r#"
const echoed = await tools.echo({ hello: "world" });
text("echoed " + JSON.stringify(echoed));
return { echoed, all: ALL_TOOLS.map((tool) => tool.name) };
"#
                .to_string(),
                Default::default(),
            )
            .await,
    )
    .await;

    let pi_codemode::types::CodemodeResult::Ok { value, output, calls, .. } = result else {
        panic!("expected ok result, got {result:?}");
    };
    assert_eq!(value, Some(json!({ "echoed": { "hello": "world" }, "all": ["echo"] })));
    assert_eq!(
        output,
        vec![CodemodeOutputItem::Text { text: "echoed {\"hello\":\"world\"}".to_string() }]
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].status, pi_codemode::types::CodemodeCallStatus::Ok);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_and_load_round_trip_through_the_snapshot() {
    let sandbox = sandbox(Vec::new());
    let mut store = serde_json::Map::new();
    store.insert("count".to_string(), json!(2));
    let result = ok_result(
        sandbox
            .execute(
                r#"
const previous = load("count");
store("count", previous + 1);
return load("count");
"#
                .to_string(),
                pi_codemode::types::CodemodeExecuteOptions {
                    store: Some(store),
                    ..Default::default()
                },
            )
            .await,
    )
    .await;

    let pi_codemode::types::CodemodeResult::Ok { value, store_writes, .. } = result else {
        panic!("expected ok result, got {result:?}");
    };
    assert_eq!(value, Some(json!(3)));
    assert_eq!(store_writes.set.get("count"), Some(&json!(3)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn script_errors_report_the_script_error_shape() {
    let sandbox = sandbox(Vec::new());
    let result = ok_result(
        sandbox
            .execute("throw new TypeError(\"boom\");".to_string(), Default::default())
            .await,
    )
    .await;

    let pi_codemode::types::CodemodeResult::Err { error, .. } = result else {
        panic!("expected error result, got {result:?}");
    };
    assert_eq!(error.kind, CodemodeErrorKind::Script);
    assert_eq!(error.name.as_deref(), Some("TypeError"));
    assert_eq!(error.message, "boom");
    assert!(error.stack.as_deref().unwrap_or_default().contains("codemode.js"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeouts_terminate_a_spinning_script() {
    let sandbox = sandbox(Vec::new());
    let started = std::time::Instant::now();
    let result = ok_result(
        sandbox
            .execute(
                "while (true) { /* spin */ }".to_string(),
                pi_codemode::types::CodemodeExecuteOptions {
                    timeout_ms: Some(250.0),
                    ..Default::default()
                },
            )
            .await,
    )
    .await;

    let pi_codemode::types::CodemodeResult::Err { error, .. } = result else {
        panic!("expected error result, got {result:?}");
    };
    assert_eq!(error.kind, CodemodeErrorKind::Timeout);
    assert!(started.elapsed() < Duration::from_secs(10), "termination must not hang");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_signals_terminate_an_execution() {
    let sandbox = sandbox(Vec::new());
    let signal = pi_codemode::types::AbortSignal::new();
    let aborted_signal = signal.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        aborted_signal.abort("caller went away");
    });

    let result = ok_result(
        sandbox
            .execute(
                "while (true) { /* spin */ }".to_string(),
                pi_codemode::types::CodemodeExecuteOptions {
                    signal: Some(signal),
                    ..Default::default()
                },
            )
            .await,
    )
    .await;

    let pi_codemode::types::CodemodeResult::Err { error, .. } = result else {
        panic!("expected error result, got {result:?}");
    };
    assert_eq!(error.kind, CodemodeErrorKind::Aborted);
    assert_eq!(error.message, "caller went away");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stalled_script_reports_a_sandbox_script_error() {
    let sandbox = sandbox(Vec::new());
    let result = ok_result(
        sandbox
            .execute("await new Promise(() => {});".to_string(), Default::default())
            .await,
    )
    .await;

    let pi_codemode::types::CodemodeResult::Err { error, .. } = result else {
        panic!("expected error result, got {result:?}");
    };
    // The prelude reports scripts waiting on a promise nothing can settle.
    assert!(
        error.message.contains("can never settle"),
        "unexpected error: {error:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_errors_surface_in_the_script() {
    let sandbox = sandbox(vec![CodemodeTool::new("fail", |_args: Option<serde_json::Value>, _ctx| async move {
        Err("tool exploded".to_string())
    })]);
    let result = ok_result(
        sandbox
            .execute("try { await tools.fail(); } catch (error) { return String(error.message); }".to_string(), Default::default())
            .await,
    )
    .await;
    let pi_codemode::types::CodemodeResult::Ok { value, calls, .. } = result else {
        panic!("expected ok result, got {result:?}");
    };
    assert_eq!(value, Some(json!("tool exploded")));
    assert_eq!(calls[0].status, pi_codemode::types::CodemodeCallStatus::Error);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sandbox_capabilities_match_the_upstream_surface() {
    let sandbox = sandbox(Vec::new());
    let result = ok_result(
        sandbox
            .execute(
                r#"
const results = [];
results.push(typeof fetch);
results.push(typeof setTimeout);
results.push(typeof process);
results.push(typeof require);
return results;
"#
                .to_string(),
                Default::default(),
            )
            .await,
    )
    .await;
    let pi_codemode::types::CodemodeResult::Ok { value, .. } = result else {
        panic!("expected ok result, got {result:?}");
    };
    // Upstream: no timers, no fetch, no process, no require — only the prelude surface.
    assert_eq!(value, Some(json!(["undefined", "undefined", "undefined", "undefined"])));
}
