//! Port of upstream `test/rpc.test.ts` — the live-CLI RPC mode suite,
//! previously deferred per crate-docs D94 (it needs the built CLI process).
//! Each case spawns the real `pi` binary (`CARGO_BIN_EXE_pi`, D109) in RPC
//! mode through the ported [`RpcClient`] and drives the agent over the wire,
//! asserting the same shapes as upstream.
//!
//! Port adaptations (**D110**):
//! - The upstream `describe.skipIf(!ANTHROPIC_API_KEY && !ANTHROPIC_OAUTH_TOKEN)`
//!   credential gate becomes a runtime early-return, so the suite stays
//!   green without credentials (mirroring a vitest skip).
//! - Vitest per-test timeouts (`30_000`–`120_000`ms) become
//!   [`with_vitest_timeout`] wrappers around each body.
//! - Upstream runs the CLI with `cwd` = the coding-agent package dir; the
//!   port uses a fresh empty temp dir per test (hermetic; sessions still
//!   live under `PI_CODING_AGENT_DIR`, so the observable surface is the
//!   same). The client spawns the binary directly (custom `program`, no
//!   script argument — the D109 `RpcClient` spawn rule).
//! - `readdirSync(sessionsPath)[0]` becomes the sorted first entry; with a
//!   single cwd-keyed directory the observable result is identical.
//!
//! Tests within the file run sequentially (a shared permit), matching the
//! upstream single-file sequential vitest run.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_agent::types::AgentThinkingLevel;
use pi_coding_agent::core::session_manager::SessionEntry;
use pi_coding_agent::modes::rpc::rpc_client::parse_thinking_level;
use pi_coding_agent::modes::rpc::{RpcClient, RpcClientOptions};
use serde_json::Value as JsonValue;

/// Shared permit keeping the live cases sequential (upstream: one vitest
/// file runs its tests one at a time). Poison is ignored so one failed test
/// does not fail the rest of the suite.
static SEQUENTIAL_PERMIT: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn test_seq() -> u32 {
    static SEQ: AtomicU32 = AtomicU32::new(0);
    SEQ.fetch_add(1, Ordering::SeqCst)
}

fn epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

/// `skipIf(!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_OAUTH_TOKEN)`.
fn live_credentials_available() -> bool {
    ["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"]
        .iter()
        .any(|key| {
            std::env::var(key)
                .ok()
                .is_some_and(|value| !value.is_empty())
        })
}

fn report_skip() {
    eprintln!(
		"skipping: live-CLI RPC suite requires ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN (upstream describe.skipIf)"
	);
}

/// Vitest per-test timeout (`test("...", fn, 90_000)`).
async fn with_vitest_timeout(timeout: Duration, body: impl Future<Output = ()>) {
    if tokio::time::timeout(timeout, body).await.is_err() {
        panic!("test exceeded the upstream vitest timeout of {timeout:?}");
    }
}

/// `beforeEach`/`afterEach` state: client plus the temp dirs, removed on drop.
struct RpcTestSession {
    client: RpcClient,
    session_dir: PathBuf,
    cwd: PathBuf,
}

impl RpcTestSession {
    fn new() -> Self {
        let session_dir =
            std::env::temp_dir().join(format!("pi-rpc-test-{}-{}", epoch_millis(), test_seq()));
        let cwd =
            std::env::temp_dir().join(format!("pi-rpc-test-cwd-{}-{}", epoch_millis(), test_seq()));
        std::fs::create_dir_all(&cwd).unwrap();
        let mut env = HashMap::new();
        env.insert(
            "PI_CODING_AGENT_DIR".to_string(),
            session_dir.to_string_lossy().into_owned(),
        );
        env.insert("PI_SKIP_VERSION_CHECK".to_string(), "1".to_string());
        let client = RpcClient::new(RpcClientOptions {
            // Spawn the built `pi` binary directly (D109 spawn rule).
            program: Some(env!("CARGO_BIN_EXE_pi").to_string()),
            cwd: Some(cwd.to_string_lossy().into_owned()),
            env: Some(env),
            provider: Some("anthropic".to_string()),
            model: Some("claude-sonnet-4-5".to_string()),
            ..Default::default()
        });
        RpcTestSession {
            client,
            session_dir,
            cwd,
        }
    }
}

impl Drop for RpcTestSession {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.session_dir);
        let _ = std::fs::remove_dir_all(&self.cwd);
    }
}

/// Wire projection of a session entry (same serialization as the session
/// file, giving `entry.type`/`entry.id` access).
fn entry_wire(entry: &SessionEntry) -> JsonValue {
    serde_json::to_value(entry).unwrap()
}

/// Entry ids of a typed entry list, via the wire projection.
fn entry_ids(entries: &[SessionEntry]) -> Vec<String> {
    entries
        .iter()
        .map(|entry| entry_wire(entry)["id"].as_str().unwrap().to_string())
        .collect()
}

/// Read the session file written under `PI_CODING_AGENT_DIR`: parse each
/// JSONL line (upstream `entries` array of parsed objects).
fn read_session_entries(session_dir: &Path) -> Vec<JsonValue> {
    let sessions_path = session_dir.join("sessions");
    assert!(sessions_path.exists(), "sessions directory should exist");
    let mut session_dirs: Vec<PathBuf> = std::fs::read_dir(&sessions_path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert!(
        !session_dirs.is_empty(),
        "expected at least one cwd session directory"
    );
    session_dirs.sort();
    let cwd_session_dir = &session_dirs[0];
    let mut session_files: Vec<PathBuf> = std::fs::read_dir(cwd_session_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect();
    assert_eq!(session_files.len(), 1, "expected exactly one session file");
    session_files.sort();
    let content = std::fs::read_to_string(&session_files[0]).unwrap();
    content
        .trim()
        .split('\n')
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn millis_tag() -> String {
    format!("{}-{}", epoch_millis(), test_seq())
}

// ============================================================================
// should get state
// ============================================================================

#[tokio::test]
async fn should_get_state() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(30), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        let state = session.client.get_state().await.unwrap();
        let model = state.get("model").expect("state.model to be defined");
        assert_eq!(model["provider"], "anthropic");
        assert_eq!(model["id"], "claude-sonnet-4-5");
        assert_eq!(state["isStreaming"], false);
        assert_eq!(state["messageCount"], 0);

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should save messages to session file
// ============================================================================

#[tokio::test]
async fn should_save_messages_to_session_file() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Send prompt and wait for completion
        let events = session
            .client
            .prompt_and_wait(
                "Reply with just the word 'hello'",
                None,
                Duration::from_secs(90),
            )
            .await
            .unwrap();

        // Should have message events
        let message_end_events = events
            .iter()
            .filter(|event| event["type"] == "message_end")
            .count();
        assert!(message_end_events >= 2); // user + assistant

        // Wait for file writes
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify session file
        let entries = read_session_entries(&session.session_dir);

        // First entry should be session header
        assert_eq!(entries[0]["type"], "session");

        // Should have user and assistant messages
        let messages: Vec<&JsonValue> = entries
            .iter()
            .filter(|entry| entry["type"] == "message")
            .collect();
        assert!(messages.len() >= 2);

        let roles: Vec<&str> = messages
            .iter()
            .map(|message| message["message"]["role"].as_str().unwrap())
            .collect();
        assert!(roles.contains(&"user"));
        assert!(roles.contains(&"assistant"));

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should handle manual compaction
// ============================================================================

#[tokio::test]
async fn should_handle_manual_compaction() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(120), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // First send a prompt to have messages to compact
        session
            .client
            .prompt_and_wait("Say hello", None, Duration::from_secs(90))
            .await
            .unwrap();

        // Compact
        let result = session.client.compact(None).await.unwrap();
        assert!(result
            .get("summary")
            .is_some_and(|summary| !summary.is_null()));
        assert!(result["tokensBefore"].as_i64().unwrap() > 0);

        // Wait for file writes
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify compaction in session file
        let entries = read_session_entries(&session.session_dir);

        let compaction_entries: Vec<&JsonValue> = entries
            .iter()
            .filter(|entry| entry["type"] == "compaction")
            .collect();
        assert_eq!(compaction_entries.len(), 1);
        assert!(compaction_entries[0]
            .get("summary")
            .is_some_and(|summary| !summary.is_null()));

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should execute bash command
// ============================================================================

#[tokio::test]
async fn should_execute_bash_command() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(30), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        let result = session.client.bash("echo hello").await.unwrap();
        assert_eq!(result.output.trim(), "hello");
        assert_eq!(result.exit_code, Some(0));
        assert!(!result.cancelled);

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should add bash output to context
// ============================================================================

#[tokio::test]
async fn should_add_bash_output_to_context() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // First send a prompt to initialize session
        session
            .client
            .prompt_and_wait("Say hi", None, Duration::from_secs(90))
            .await
            .unwrap();

        // Run bash command
        let unique_value = format!("test-{}", millis_tag());
        session
            .client
            .bash(format!("echo {unique_value}"))
            .await
            .unwrap();

        // Wait for file writes
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify bash message in session
        let entries = read_session_entries(&session.session_dir);

        let bash_messages: Vec<&JsonValue> = entries
            .iter()
            .filter(|entry| {
                entry["type"] == "message" && entry["message"]["role"] == "bashExecution"
            })
            .collect();
        assert_eq!(bash_messages.len(), 1);
        assert!(bash_messages[0]["message"]["output"]
            .as_str()
            .unwrap()
            .contains(&unique_value));

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should include bash output in LLM context
// ============================================================================

#[tokio::test]
async fn should_include_bash_output_in_llm_context() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
		let session = RpcTestSession::new();
		session.client.start().await.unwrap();

		// Run a bash command with a unique value
		let unique_value = format!("unique-{}", millis_tag());
		session.client.bash(format!("echo {unique_value}")).await.unwrap();

		// Ask the LLM what the output was
		let events = session
			.client
			.prompt_and_wait(
				"What was the exact output of the echo command I just ran? Reply with just the value, nothing else.",
				None,
				Duration::from_secs(90),
			)
			.await
			.unwrap();

		// Find assistant's response
		let assistant_message = events
			.iter()
			.find(|event| event["type"] == "message_end" && event["message"]["role"] == "assistant")
			.expect("assistant message_end event");

		let text_content = assistant_message["message"]["content"]
			.as_array()
			.unwrap()
			.iter()
			.find(|content| content["type"] == "text")
			.expect("text content block");
		assert!(text_content["text"].as_str().unwrap().contains(&unique_value));

		session.client.stop().await;
	})
	.await;
}

// ============================================================================
// should set and get thinking level
// ============================================================================

#[tokio::test]
async fn should_set_and_get_thinking_level() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(30), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Set thinking level
        session
            .client
            .set_thinking_level(AgentThinkingLevel::High)
            .await
            .unwrap();

        // Verify via state
        let state = session.client.get_state().await.unwrap();
        assert_eq!(state["thinkingLevel"], "high");

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should cycle thinking level
// ============================================================================

#[tokio::test]
async fn should_cycle_thinking_level() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(30), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Get initial level
        let initial_state = session.client.get_state().await.unwrap();
        let initial_level =
            parse_thinking_level(initial_state["thinkingLevel"].as_str().unwrap()).unwrap();

        // Cycle
        let result = session.client.cycle_thinking_level().await.unwrap();
        let result_level = result.expect("cycle result to be defined");
        assert_ne!(result_level, initial_level);

        // Verify via state
        let new_state = session.client.get_state().await.unwrap();
        assert_eq!(
            parse_thinking_level(new_state["thinkingLevel"].as_str().unwrap()).unwrap(),
            result_level
        );

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should get available thinking levels
// ============================================================================

#[tokio::test]
async fn should_get_available_thinking_levels() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(30), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        let levels = session
            .client
            .get_available_thinking_levels()
            .await
            .unwrap();
        assert!(!levels.is_empty());

        // The current level reported by get_state must be in the available list
        let state = session.client.get_state().await.unwrap();
        let initial_level = parse_thinking_level(state["thinkingLevel"].as_str().unwrap()).unwrap();
        assert!(levels.contains(&initial_level));

        // cycle_thinking_level must only ever land on levels from get_available_thinking_levels
        let cycled = session.client.cycle_thinking_level().await.unwrap();
        if let Some(level) = cycled {
            assert!(levels.contains(&level));
            // distinct cycle step (unless only one level)
            if levels.len() > 1 {
                assert_ne!(level, initial_level);
            }
        }

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should get available models
// ============================================================================

#[tokio::test]
async fn should_get_available_models() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(30), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        let models = session.client.get_available_models().await.unwrap();
        assert!(!models.is_empty());

        // All models should have required fields
        for model in &models {
            assert!(!model.provider.is_empty());
            assert!(!model.id.is_empty());
            assert!(model.context_window > 0);
        }

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should get session stats
// ============================================================================

#[tokio::test]
async fn should_get_session_stats() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Send a prompt first
        session
            .client
            .prompt_and_wait("Hello", None, Duration::from_secs(90))
            .await
            .unwrap();

        let stats = session.client.get_session_stats().await.unwrap();
        assert!(stats
            .get("sessionFile")
            .is_some_and(|value| !value.is_null()));
        assert!(stats.get("sessionId").is_some_and(|value| !value.is_null()));
        assert!(stats["userMessages"].as_i64().unwrap() >= 1);
        assert!(stats["assistantMessages"].as_i64().unwrap() >= 1);

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should create new session
// ============================================================================

#[tokio::test]
async fn should_create_new_session() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Send a prompt
        session
            .client
            .prompt_and_wait("Hello", None, Duration::from_secs(90))
            .await
            .unwrap();

        // Verify messages exist
        let state = session.client.get_state().await.unwrap();
        assert!(state["messageCount"].as_u64().unwrap() > 0);

        // New session
        session.client.new_session(None).await.unwrap();

        // Verify messages cleared
        let state = session.client.get_state().await.unwrap();
        assert_eq!(state["messageCount"], 0);

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should export to HTML
// ============================================================================

#[tokio::test]
async fn should_export_to_html() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Send a prompt first
        session
            .client
            .prompt_and_wait("Hello", None, Duration::from_secs(90))
            .await
            .unwrap();

        // Export
        let result = session.client.export_html(None).await.unwrap();
        assert!(result.path.ends_with(".html"));
        assert!(Path::new(&result.path).exists());

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should get last assistant text
// ============================================================================

#[tokio::test]
async fn should_get_last_assistant_text() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Initially null
        let text = session.client.get_last_assistant_text().await.unwrap();
        assert_eq!(text, None);

        // Send prompt
        session
            .client
            .prompt_and_wait("Reply with just: test123", None, Duration::from_secs(90))
            .await
            .unwrap();

        // Should have text now
        let text = session.client.get_last_assistant_text().await.unwrap();
        assert!(text.unwrap().contains("test123"));

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should get session entries with since cursor
// ============================================================================

#[tokio::test]
async fn should_get_session_entries_with_since_cursor() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        session
            .client
            .prompt_and_wait("Reply with just 'ok'", None, Duration::from_secs(90))
            .await
            .unwrap();

        let result = session.client.get_entries(None).await.unwrap();
        let entries: Vec<JsonValue> = result.entries.iter().map(entry_wire).collect();
        assert!(entries.len() >= 2); // user + assistant
        for entry in &entries {
            assert!(entry.get("id").is_some());
        }
        assert_eq!(
            result.leaf_id.as_deref(),
            entries.last().unwrap()["id"].as_str(),
            "leafId should be the last entry id"
        );

        // since cursor returns only entries strictly after the given id
        let since = session
            .client
            .get_entries(Some(entries[0]["id"].as_str().unwrap()))
            .await
            .unwrap();
        let since_ids: Vec<String> = entry_ids(&since.entries);
        let expected_ids: Vec<String> = entries[1..]
            .iter()
            .map(|entry| entry["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(since_ids, expected_ids);
        assert_eq!(since.leaf_id, result.leaf_id);

        // unknown since id is an error response
        let error = session
            .client
            .get_entries(Some("nonexistent-id"))
            .await
            .unwrap_err();
        assert!(error.contains("Entry not found"), "got: {error}");

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should get session tree
// ============================================================================

#[tokio::test]
async fn should_get_session_tree() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(90), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        session
            .client
            .prompt_and_wait("Reply with just 'ok'", None, Duration::from_secs(90))
            .await
            .unwrap();

        let result = session.client.get_entries(None).await.unwrap();
        let entry_ids: Vec<String> = entry_ids(&result.entries);
        let tree_result = session.client.get_tree().await.unwrap();
        assert_eq!(
            tree_result["leafId"].as_str(),
            result.leaf_id.as_deref(),
            "tree leafId should match entries leafId"
        );

        // Single root whose chain matches the entries
        let mut nodes: Vec<JsonValue> = tree_result["tree"].as_array().expect("tree array").clone();
        assert_eq!(nodes.len(), 1);
        let mut chain_ids: Vec<String> = Vec::new();
        while nodes.len() == 1 {
            chain_ids.push(nodes[0]["entry"]["id"].as_str().unwrap().to_string());
            nodes = nodes[0]["children"].as_array().cloned().unwrap_or_default();
        }
        assert!(nodes.is_empty());
        assert_eq!(chain_ids, entry_ids);

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should retain pre-compaction entries in get_entries
// ============================================================================

#[tokio::test]
async fn should_retain_pre_compaction_entries_in_get_entries() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(120), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        session
            .client
            .prompt_and_wait("Reply with just 'ok'", None, Duration::from_secs(90))
            .await
            .unwrap();
        let before = session.client.get_entries(None).await.unwrap();

        session.client.compact(None).await.unwrap();

        let after = session.client.get_entries(None).await.unwrap();
        // Append-only: pre-compaction entries are still there, in the same order
        assert_eq!(
            entry_ids(&after.entries[..before.entries.len()]),
            entry_ids(&before.entries)
        );
        let has_compaction = after
            .entries
            .iter()
            .any(|entry| entry_wire(entry)["type"] == "compaction");
        assert!(has_compaction);

        session.client.stop().await;
    })
    .await;
}

// ============================================================================
// should set and get session name
// ============================================================================

#[tokio::test]
async fn should_set_and_get_session_name() {
    if !live_credentials_available() {
        report_skip();
        return;
    }
    let _permit = SEQUENTIAL_PERMIT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    with_vitest_timeout(Duration::from_secs(60), async {
        let session = RpcTestSession::new();
        session.client.start().await.unwrap();

        // Initially undefined
        let state = session.client.get_state().await.unwrap();
        assert!(
            state.get("sessionName").is_none(),
            "sessionName should be undefined"
        );

        // Send a prompt first - session files are only written after first assistant message
        session
            .client
            .prompt_and_wait("Reply with just 'ok'", None, Duration::from_secs(60))
            .await
            .unwrap();

        // Set name
        session
            .client
            .set_session_name("my-test-session")
            .await
            .unwrap();

        // Verify via state
        let state = session.client.get_state().await.unwrap();
        assert_eq!(state["sessionName"], "my-test-session");

        // Wait for file writes
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify session_info entry in session file
        let entries = read_session_entries(&session.session_dir);

        let session_info_entries: Vec<&JsonValue> = entries
            .iter()
            .filter(|entry| entry["type"] == "session_info")
            .collect();
        assert_eq!(session_info_entries.len(), 1);
        assert_eq!(session_info_entries[0]["name"], "my-test-session");

        session.client.stop().await;
    })
    .await;
}
