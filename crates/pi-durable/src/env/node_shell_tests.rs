//! Shell tests of `test/env-node.test.ts` (`describe("NodeExecutionEnv shell")`).
//!
//! Deviations:
//! - D1: the detached-descendant tests rely on Windows-only `taskkill` behavior or Node
//!   child-process internals; they are Windows/Node-specific upstream (`skipIf` guards) and
//!   are not ported.
//! - D2: the legacy-WSL stdin-transport test mutates `process.platform` at runtime, which is
//!   a compile-time `cfg` here (node.rs D2); transport selection is unit-tested instead.
//! - D3: the taskkill-spawn-error abort test is Windows-only upstream and not ported.
//! - D4: "decodes UTF-8 split across raw process chunks" uses two shell writes with a delay
//!   instead of Node `process.stdout` timing.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pi_chord::context::{with_abort_signal, AbortController, Context, BACKGROUND_CONTEXT};

use crate::env::{
    get_or_throw, ExecCommand, ExecutionErrorCode, FileSystem, NodeEnvOptions, NodeExecutionEnv,
    Shell, ShellExecOptions, ShellSpillOptions,
};

use super::node_tests::{aborted_context, env_at, remove_dir, unique_dir};

type OutputCollector = Arc<Mutex<String>>;

fn collector() -> OutputCollector {
    Arc::new(Mutex::new(String::new()))
}

fn with_output(collector: &OutputCollector) -> ShellExecOptions {
    let collector = Arc::clone(collector);
    ShellExecOptions {
        on_output: Some(Arc::new(
            move |text: &str, _info: &crate::env::ShellOutputInfo| {
                collector.lock().expect("output lock").push_str(text);
            },
        )),
        ..Default::default()
    }
}

async fn collect_shell_output(
    env: &NodeExecutionEnv,
    command: &str,
    options: ShellExecOptions,
) -> (
    std::result::Result<crate::env::ShellExecResult, crate::env::ExecutionError>,
    String,
) {
    let output = collector();
    let mut options = options;
    options.on_output = with_output(&output).on_output;
    let result = env
        .exec(
            ExecCommand::from(command),
            Some(options),
            &BACKGROUND_CONTEXT,
        )
        .await;
    let text = output.lock().expect("output lock").clone();
    (result, text)
}

#[tokio::test]
async fn executes_commands_in_cwd_with_env_overrides() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let mut options = ShellExecOptions::default();
    let mut env_overrides = std::collections::HashMap::new();
    env_overrides.insert("NODE_ENV_TEST".to_string(), "ok".to_string());
    options.env = Some(env_overrides);
    let (result, output) = collect_shell_output(
		env.as_ref(),
		"printf '%s' \"$NODE_ENV_TEST\" > cwd-marker.txt; printf '%s:%s' \"$PWD\" \"$NODE_ENV_TEST\"",
		options,
	)
	.await;
    let result = get_or_throw(result);
    assert_eq!(result.exit_code, 0);
    assert_eq!(
        std::fs::read_to_string(root.join("cwd-marker.txt")).expect("marker"),
        "ok"
    );
    let real_root = std::fs::canonicalize(&root).unwrap();
    assert_eq!(output, format!("{}:ok", real_root.to_string_lossy()));
    remove_dir(&root);
}

#[tokio::test]
async fn applies_string_shell_environment_overrides() {
    // Upstream's it.each table: a missing override preserves the base value, an empty one
    // shadows it, and a string replaces it.
    for (overrides, expected_session_file) in [
        (None, "x:/stale/parent.jsonl"),
        (Some(""), "x:"),
        (Some("/sessions/current.jsonl"), "x:/sessions/current.jsonl"),
    ] {
        let root = unique_dir("pi-durable-env-node");
        let mut shell_env = std::collections::HashMap::new();
        shell_env.insert(
            "PI_SESSION_FILE".to_string(),
            "/stale/parent.jsonl".to_string(),
        );
        shell_env.insert("PI_CODING_AGENT".to_string(), "true".to_string());
        shell_env.insert(
            "PI_NODE_ENV_PRESERVED_TEST".to_string(),
            "preserved".to_string(),
        );
        let env = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
            cwd: root.to_string_lossy().into_owned(),
            shell_env: Some(shell_env),
            ..Default::default()
        }));
        let mut options = ShellExecOptions::default();
        let mut env_overrides = std::collections::HashMap::new();
        if let Some(session_file) = overrides {
            env_overrides.insert("PI_SESSION_FILE".to_string(), session_file.to_string());
            options.env = Some(env_overrides);
        }
        let (result, output) = collect_shell_output(
			env.as_ref(),
			"printf '%s:%s|%s|%s' \"${PI_SESSION_FILE+x}\" \"${PI_SESSION_FILE-}\" \"$PI_CODING_AGENT\" \"$PI_NODE_ENV_PRESERVED_TEST\"",
			options,
		)
		.await;
        get_or_throw(result);
        assert_eq!(
            output,
            format!("{expected_session_file}|true|preserved"),
            "overrides {overrides:?}"
        );
        remove_dir(&root);
    }
}

#[tokio::test]
async fn can_replace_rather_than_inherit_the_default_shell_environment() {
    let root = unique_dir("pi-durable-env-node");
    let inherited_key = "PI_NODE_ENV_INHERITED_TEST";
    let configured_key = "PI_NODE_ENV_CONFIGURED_TEST";
    let explicit_key = "PI_NODE_ENV_EXPLICIT_TEST";
    std::env::set_var(inherited_key, "host");
    let mut shell_env = std::collections::HashMap::new();
    shell_env.insert(configured_key.to_string(), "configured".to_string());
    let env = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
        cwd: root.to_string_lossy().into_owned(),
        shell_env: Some(shell_env),
        ..Default::default()
    }));
    let mut options = ShellExecOptions::default();
    options.inherit_env = Some(false);
    let mut env_overrides = std::collections::HashMap::new();
    env_overrides.insert(explicit_key.to_string(), "explicit".to_string());
    options.env = Some(env_overrides);
    let (result, output) = collect_shell_output(
		env.as_ref(),
		&format!("printf '%s:%s:%s' \"${{{inherited_key}-}}\" \"${{{configured_key}-}}\" \"${{{explicit_key}-}}\""),
		options,
	)
	.await;
    get_or_throw(result);
    assert_eq!(output, "::explicit");
    std::env::remove_var(inherited_key);
    remove_dir(&root);
}

#[tokio::test]
async fn cleanup_terminates_active_shell_processes() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let execution = {
        let env = Arc::clone(&env);
        tokio::spawn(async move {
            env.exec(
                ExecCommand::from("touch started; sleep 60"),
                None,
                &BACKGROUND_CONTEXT,
            )
            .await
        })
    };
    // Wait for the command to start, then tear everything down.
    for _ in 0..200 {
        if get_or_throw(env.exists("started", &BACKGROUND_CONTEXT).await) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    Shell::cleanup(env.as_ref(), &BACKGROUND_CONTEXT).await;
    // Upstream awaits the same in-flight promise; the spawned task mirrors that.
    let result = tokio::time::timeout(std::time::Duration::from_secs(3), execution)
        .await
        .expect("settles")
        .expect("join");
    assert!(
        get_or_throw(result).exit_code != 0,
        "killed command does not exit cleanly"
    );
    remove_dir(&root);
}

#[tokio::test]
async fn streams_combined_stdout_and_stderr() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let (result, output) = collect_shell_output(
        env.as_ref(),
        "printf out; printf err >&2",
        ShellExecOptions::default(),
    )
    .await;
    assert_eq!(get_or_throw(result).exit_code, 0);
    assert!(output.contains("out"));
    assert!(output.contains("err"));
    remove_dir(&root);
}

#[tokio::test]
async fn decodes_utf8_split_across_raw_process_chunks() {
    // D4: two delayed shell writes instead of Node `process.stdout` timing.
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let (result, output) = collect_shell_output(
        env.as_ref(),
        "printf '\\xf0\\x9f'; sleep 0.05; printf '\\x98\\x80'",
        ShellExecOptions::default(),
    )
    .await;
    get_or_throw(result);
    assert_eq!(output, "😀");
    remove_dir(&root);
}

#[tokio::test]
async fn reports_a_missing_working_directory_before_spawning() {
    let root = unique_dir("pi-durable-env-node");
    let env = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
        cwd: root.join("missing").to_string_lossy().into_owned(),
        ..Default::default()
    }));
    let error = env
        .exec(ExecCommand::from("printf ok"), None, &BACKGROUND_CONTEXT)
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::SpawnError);
    assert!(
        error.message.contains("Working directory does not exist"),
        "{}",
        error.message
    );
    remove_dir(&root);
}

#[tokio::test]
async fn returns_non_zero_command_exit_codes_as_successful_execution_results() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let result = get_or_throw(
        env.exec(ExecCommand::from("exit 7"), None, &BACKGROUND_CONTEXT)
            .await,
    );
    assert_eq!(result.exit_code, 7);
    remove_dir(&root);
}

#[tokio::test]
async fn maps_signal_killed_processes_to_a_non_zero_exit_code() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let result = get_or_throw(
        env.exec(ExecCommand::from("kill -9 $$"), None, &BACKGROUND_CONTEXT)
            .await,
    );
    assert_eq!(result.exit_code, 128 + 9);
    remove_dir(&root);
}

#[tokio::test]
async fn returns_timeout_errors_for_commands_exceeding_the_timeout() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let mut options = ShellExecOptions::default();
    options.timeout = Some(0.01);
    let error = env
        .exec(
            ExecCommand::from("sleep 5"),
            Some(options),
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Timeout);
    remove_dir(&root);
}

#[tokio::test]
async fn rejects_invalid_timeouts_before_spawning() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY, 2_147_484.0] {
        let mut options = ShellExecOptions::default();
        options.timeout = Some(timeout);
        let error = env
            .exec(
                ExecCommand::from("touch spawned"),
                Some(options),
                &BACKGROUND_CONTEXT,
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ExecutionErrorCode::Timeout, "timeout {timeout}");
        assert!(
            error.message.contains("Invalid timeout"),
            "{}",
            error.message
        );
    }
    assert!(!get_or_throw(
        env.exists("spawned", &BACKGROUND_CONTEXT).await
    ));
    remove_dir(&root);
}

#[tokio::test]
async fn returns_callback_errors_from_exec_stream_handlers() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let options = ShellExecOptions {
        on_output: Some(Arc::new(
            |_text: &str, _info: &crate::env::ShellOutputInfo| {
                panic!("callback failed");
            },
        )),
        ..Default::default()
    };
    let error = env
        .exec(
            ExecCommand::from("printf out"),
            Some(options),
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::CallbackError);
    assert!(error.message.contains("callback"), "{}", error.message);
    remove_dir(&root);
}

#[tokio::test]
async fn returns_shell_unavailable_and_spawn_errors() {
    let root = unique_dir("pi-durable-env-node");
    let missing_shell_env = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
        cwd: root.to_string_lossy().into_owned(),
        shell_path: Some(root.join("missing-shell").to_string_lossy().into_owned()),
        ..Default::default()
    }));
    let error = missing_shell_env
        .exec(ExecCommand::from("printf ok"), None, &BACKGROUND_CONTEXT)
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::ShellUnavailable);

    let env = env_at(&root);
    let shell_path = root.join("not-executable-shell");
    get_or_throw(
        env.write_file(
            &shell_path.to_string_lossy(),
            crate::env::FileContent::from("not executable"),
            &BACKGROUND_CONTEXT,
        )
        .await,
    );
    let spawn_error_env = Arc::new(NodeExecutionEnv::new(NodeEnvOptions {
        cwd: root.to_string_lossy().into_owned(),
        shell_path: Some(shell_path.to_string_lossy().into_owned()),
        ..Default::default()
    }));
    let error = spawn_error_env
        .exec(ExecCommand::from("printf ok"), None, &BACKGROUND_CONTEXT)
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::SpawnError);
    remove_dir(&root);
}

#[tokio::test]
async fn returns_an_aborted_result_for_pre_aborted_and_aborted_commands() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let error = env
        .exec(ExecCommand::from("touch spawned"), None, &aborted_context())
        .await
        .unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Aborted);
    assert!(!get_or_throw(
        env.exists("spawned", &BACKGROUND_CONTEXT).await
    ));

    let controller = AbortController::new();
    let context = with_abort_signal(controller.signal(), BACKGROUND_CONTEXT.clone());
    let promise = env.exec(ExecCommand::from("sleep 5"), None, &context);
    controller.abort_default();
    let error = promise.await.unwrap_err();
    assert_eq!(error.code, ExecutionErrorCode::Aborted);
    remove_dir(&root);
}

#[tokio::test]
async fn preserves_complete_large_output_in_the_spill() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let size = 500_000;
    let result = get_or_throw(
        env.exec(
            ExecCommand::from("head -c 500000 </dev/zero | tr '\\0' x"),
            Some(ShellExecOptions {
                spill: Some(ShellSpillOptions {
                    after_bytes: 10,
                    after_lines: 10,
                }),
                ..Default::default()
            }),
            &BACKGROUND_CONTEXT,
        )
        .await,
    );
    let spill_path = result.spill_path.expect("spill path");
    assert_eq!(
        get_or_throw(env.read_text_file(&spill_path, &BACKGROUND_CONTEXT).await).len(),
        size
    );
    let _ = std::fs::remove_file(&spill_path);
    remove_dir(&root);
}

#[tokio::test]
async fn streams_every_line_and_spills_them_all_once_output_crosses_its_line_threshold() {
    let root = unique_dir("pi-durable-env-node");
    let env = env_at(&root);
    let (result, output) = collect_shell_output(
        env.as_ref(),
        "i=1; while [ $i -le 15000 ]; do echo line-$i; i=$((i+1)); done",
        ShellExecOptions {
            spill: Some(ShellSpillOptions {
                after_bytes: 1024 * 1024,
                after_lines: 100,
            }),
            ..Default::default()
        },
    )
    .await;
    let result = get_or_throw(result);
    let expected: String = (1..=15000).map(|index| format!("line-{index}\n")).collect();
    assert_eq!(output, expected);
    let spill_path = result.spill_path.expect("spill path");
    let spilled = get_or_throw(
        env.read_text_lines(&spill_path, None, &BACKGROUND_CONTEXT)
            .await,
    );
    assert_eq!(spilled.len(), 15000);
    assert_eq!(spilled.last().map(String::as_str), Some("line-15000"));
    let _ = std::fs::remove_file(&spill_path);
    remove_dir(&root);
}

// Keep imports used on non-test builds of the helper module.
#[allow(unused)]
fn _paths(_: PathBuf, _: &Path, _: Context) {}
