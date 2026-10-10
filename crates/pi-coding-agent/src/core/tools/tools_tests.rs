//! Port of `upstream/pi/packages/coding-agent/test/tools.test.ts` (1:1) —
//! tests for the built-in coding tools. The three read-tool image cases run
//! in `tests/tools_image_tests.rs` instead: the photon engine registry is
//! process-global, and the in-module suite must not register a fixture that
//! would break sibling tests asserting no-engine behavior (D36).
//!
//! Fixture substitutions (crate docs D35): Node `process.execPath` commands
//! become `sh` equivalents; the module-function `getShellConfig` spies become
//! injected shell-config resolvers; the `applyPatch` npm helper becomes the
//! local [`apply_unified_patch`].

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::LazyLock;

use pi_agent::{AgentTool, AgentToolResult};
use serde_json::{json, Value};

use super::bash::{
    create_bash_tool, create_bash_tool_definition, create_local_bash_operations,
    create_local_shell_operations, BashExecOptions, BashExecOutcome, BashOperations,
    BashToolOptions,
};
use super::edit::{create_edit_tool, create_edit_tool_definition, EditOperations, EditToolOptions};
use super::edit_diff::{compute_edits_diff, Edit, EditDiffOutcome};
use super::find::{create_find_tool, create_find_tool_definition};
use super::grep::{create_grep_tool, create_grep_tool_definition};
use super::ls::{create_ls_tool, create_ls_tool_definition};
use super::read::{create_read_tool, create_read_tool_definition};
use super::write::{create_write_tool, create_write_tool_definition};
use crate::core::bash_executor::execute_bash_with_operations;
use crate::core::extensions::types::{
    AutocompleteProviderFactory, CustomComponentFactory, CustomComponentOptions, EditorFactory,
    ExtensionContext, ExtensionMode, ExtensionToolContext, ExtensionUi, ExtensionUiDialogOptions,
    FooterFactory, HeaderFactory, ModelRegistry, NotifyKind, SetThemeArg, TerminalInputHandler,
    Theme, ThemeInfo, ThemeSwitchResult, WidgetContent, WorkingIndicatorOptions,
};
use crate::core::session_manager::SessionManager;

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

static COUNTER: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(0));

/// Unique temp directory per test (upstream `join(tmpdir(), ...Date.now())`).
fn test_dir(label: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("pi-rs-{label}-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dir");
    }
    std::fs::write(path, content).expect("write test file");
}

fn read_file(path: &Path) -> String {
    std::fs::read_to_string(path).expect("read test file")
}

/// The test file's local `getTextOutput` helper: join text blocks.
fn get_text_output(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            pi_ai::types::UserContent::Text(text) => Some(text.text.clone()),
            pi_ai::types::UserContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Execute a definition the way the tests do (upstream
/// `tool.execute(id, args, undefined, undefined, ctx?)`).
async fn execute_definition(
    tool: &crate::core::extensions::types::ToolDefinition,
    call_id: &str,
    args: Value,
    ctx: Option<ExtensionToolContext>,
) -> Result<AgentToolResult, String> {
    let execute = crate::core::extensions::types::ToolExecute::clone(&tool.execute);
    execute(call_id.to_string(), args, None, None, ctx).await
}

/// Execute a wrapped [`AgentTool`] (upstream `tool.execute(id, args)`).
async fn execute_tool(
    tool: &AgentTool,
    call_id: &str,
    args: Value,
) -> Result<AgentToolResult, String> {
    let map = args.as_object().cloned().expect("args must be an object");
    let execute = std::sync::Arc::clone(&tool.execute);
    execute(call_id.to_string(), map, None, None).await
}

// ---------------------------------------------------------------------------
// fake_ctx — `{ cwd } as ExtensionToolContext` upstream
// ---------------------------------------------------------------------------

/// No-op UI stand-in for the fake context.
struct TestUi;

impl ExtensionUi for TestUi {
    fn select(
        &self,
        _title: String,
        _options: Vec<String>,
        _opts: Option<ExtensionUiDialogOptions>,
    ) -> pi_agent::BoxFuture<'static, Option<String>> {
        Box::pin(async { None })
    }

    fn confirm(
        &self,
        _title: String,
        _message: String,
        _opts: Option<ExtensionUiDialogOptions>,
    ) -> pi_agent::BoxFuture<'static, bool> {
        Box::pin(async { false })
    }

    fn input(
        &self,
        _title: String,
        _placeholder: Option<String>,
        _opts: Option<ExtensionUiDialogOptions>,
    ) -> pi_agent::BoxFuture<'static, Option<String>> {
        Box::pin(async { None })
    }

    fn notify(&self, _message: String, _kind: Option<NotifyKind>) {}

    fn on_terminal_input(
        &self,
        _handler: TerminalInputHandler,
    ) -> crate::core::extensions::types::Unsubscribe {
        Box::new(|| {})
    }

    fn set_status(&self, _key: String, _text: Option<String>) {}
    fn set_working_message(&self, _message: Option<String>) {}
    fn set_working_visible(&self, _visible: bool) {}
    fn set_working_indicator(&self, _options: Option<WorkingIndicatorOptions>) {}
    fn set_hidden_thinking_label(&self, _label: Option<String>) {}
    fn set_widget(
        &self,
        _key: String,
        _content: Option<WidgetContent>,
        _options: Option<crate::core::extensions::types::ExtensionWidgetOptions>,
    ) {
    }
    fn set_footer(&self, _factory: Option<FooterFactory>) {}
    fn set_header(&self, _factory: Option<HeaderFactory>) {}
    fn set_title(&self, _title: String) {}

    fn custom(
        &self,
        _factory: CustomComponentFactory,
        _options: Option<CustomComponentOptions>,
    ) -> pi_agent::BoxFuture<'static, Value> {
        Box::pin(async { Value::Null })
    }

    fn paste_to_editor(&self, _text: String) {}
    fn set_editor_text(&self, _text: String) {}
    fn get_editor_text(&self) -> String {
        String::new()
    }

    fn editor(
        &self,
        _title: String,
        _prefill: Option<String>,
    ) -> pi_agent::BoxFuture<'static, Option<String>> {
        Box::pin(async { None })
    }

    fn add_autocomplete_provider(&self, _factory: AutocompleteProviderFactory) {}
    fn set_editor_component(&self, _factory: Option<EditorFactory>) {}
    fn get_editor_component(&self) -> Option<EditorFactory> {
        None
    }
    fn theme(&self) -> Theme {
        Theme
    }
    fn get_all_themes(&self) -> Vec<ThemeInfo> {
        Vec::new()
    }
    fn get_theme(&self, _name: &str) -> Option<Theme> {
        None
    }
    fn set_theme(&self, _theme: SetThemeArg) -> ThemeSwitchResult {
        ThemeSwitchResult {
            success: false,
            error: None,
        }
    }
    fn get_tools_expanded(&self) -> bool {
        false
    }
    fn set_tools_expanded(&self, _expanded: bool) {}
}

fn noop_context(cwd: &str) -> ExtensionContext {
    ExtensionContext {
        ui: std::sync::Arc::new(TestUi),
        mode: ExtensionMode::Print,
        has_ui: false,
        cwd: cwd.to_string(),
        session_manager: std::sync::Arc::new(parking_lot::Mutex::new(test_session_manager(cwd))),
        model_registry: crate::core::extensions::types::offline_registry_for_tests(),
        model: None,
        scoped_models: Vec::new(),
        thinking_level: None,
        signal: None,
        is_idle: std::sync::Arc::new(|| true),
        is_project_trusted: std::sync::Arc::new(|| false),
        abort: std::sync::Arc::new(|| ()),
        has_pending_messages: std::sync::Arc::new(|| false),
        shutdown: std::sync::Arc::new(|| ()),
        get_context_usage: std::sync::Arc::new(|| None),
        compact: std::sync::Arc::new(|_| ()),
        get_system_prompt: std::sync::Arc::new(String::new),
    }
}

fn test_session_manager(cwd: &str) -> SessionManager {
    static SESSION_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir =
            std::env::temp_dir().join(format!("pi-rs-tools-tests-sessions-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create session dir");
        dir
    });
    SessionManager::create(cwd, Some(SESSION_DIR.to_string_lossy().as_ref()), None)
        .expect("create session manager")
}

/// `fakeCtx(cwd)` — minimal context whose only meaningful member is `cwd`.
fn fake_ctx(cwd: &str) -> ExtensionToolContext {
    ExtensionToolContext {
        ctx: noop_context(cwd),
        tools: Vec::new(),
        execute_tool: std::sync::Arc::new(|_, _, _| {
            Box::pin(async { panic!("executeTool is not available in the fake context") })
        }),
    }
}

// ---------------------------------------------------------------------------
// applyPatch — minimal unified-patch applier standing in for npm `diff`
// ---------------------------------------------------------------------------

fn apply_unified_patch(original: &str, patch: &str) -> Option<String> {
    let mut hunks: Vec<(usize, Vec<&str>)> = Vec::new();
    let mut current: Option<(usize, Vec<&str>)> = None;
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("@@ -") {
            if let Some(inserted) = current.take() {
                hunks.push(inserted);
            }
            let old_range = rest.split(" +").next()?.trim_end_matches(' ');
            let start: usize = old_range
                .split(',')
                .next()?
                .parse::<usize>()
                .ok()
                .map(|zero_based| zero_based.saturating_sub(1))?;
            current = Some((start, Vec::new()));
        } else if current.is_some() {
            let header = line.starts_with("--- ") || line.starts_with("+++ ");
            if header {
                continue;
            }
            if let Some((_, body)) = current.as_mut() {
                body.push(line);
            }
        }
    }
    if let Some(inserted) = current.take() {
        hunks.push(inserted);
    }

    let mut lines: Vec<String> = original.lines().map(str::to_string).collect();
    let mut offset: isize = 0;
    for (start, body) in hunks {
        let at = (start as isize + offset).max(0) as usize;
        let mut replacement: Vec<String> = Vec::new();
        let mut cursor = at;
        for line in body {
            let (tag, text) = line.split_at(1);
            match tag {
                "-" | " " => {
                    if lines.get(cursor)?.as_str() != text {
                        return None;
                    }
                    if tag == "-" {
                        cursor += 1;
                    } else {
                        replacement.push(text.to_string());
                        cursor += 1;
                    }
                }
                "+" => replacement.push(text.to_string()),
                _ => {}
            }
        }
        let removed = cursor - at;
        lines.splice(at..at + removed, replacement);
        offset += cursor as isize - at as isize - removed as isize;
        offset += 0;
    }

    let mut result = lines.join("\n");
    if original.ends_with('\n') {
        result.push('\n');
    }
    Some(result)
}

// ---------------------------------------------------------------------------
// TestPhotonEngine — minimal BMP decode / PNG encode engine (D36)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// read tool
// ---------------------------------------------------------------------------

fn read_tool() -> AgentTool {
    create_read_tool(".", None)
}

#[tokio::test]
async fn read_reads_file_contents_that_fit_within_limits() {
    let dir = test_dir("read-fits");
    let test_file = dir.join("test.txt");
    let content = "Hello, world!\nLine 2\nLine 3";
    write_file(&test_file, &content);

    let result = execute_tool(
        &read_tool(),
        "test-call-1",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();
    assert_eq!(get_text_output(&result), content);
    // No truncation message since the file fits within limits.
    assert!(!get_text_output(&result).contains("Use offset="));
    assert_eq!(result.details, Value::Null);
    // Programmatic callers get the text as structured content (#10251).
    assert_eq!(result.structured_content.as_ref(), Some(&json!(content)));
}

#[tokio::test]
async fn read_handles_non_existent_files() {
    let dir = test_dir("read-missing");
    let test_file = dir.join("nonexistent.txt");

    let error = execute_tool(
        &read_tool(),
        "test-call-2",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap_err();
    assert!(error.to_lowercase().contains("enoent") || error.to_lowercase().contains("not found"));
}

#[tokio::test]
async fn read_truncates_files_exceeding_line_limit() {
    let dir = test_dir("read-line-trunc");
    let test_file = dir.join("large.txt");
    let lines: Vec<String> = (0..2500).map(|i| format!("Line {}", i + 1)).collect();
    write_file(&test_file, &lines.join("\n"));

    let result = execute_tool(
        &read_tool(),
        "test-call-3",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();
    let output = get_text_output(&result);

    assert!(output.contains("Line 1\n"));
    assert!(output.contains("\nLine 2000"));
    assert!(!output.contains("Line 2001"));
    assert!(output.contains("[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]"));
}

#[tokio::test]
async fn read_truncates_when_byte_limit_exceeded() {
    let dir = test_dir("read-byte-trunc");
    let test_file = dir.join("large-bytes.txt");
    let lines: Vec<String> = (0..500)
        .map(|i| format!("Line {}: {}", i + 1, "x".repeat(200)))
        .collect();
    write_file(&test_file, &lines.join("\n"));

    let result = execute_tool(
        &read_tool(),
        "test-call-4",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();
    let output = get_text_output(&result);

    assert!(output.contains("Line 1:"));
    // Should show the byte limit message (upstream regex:
    // /\[Showing lines 1-\d+ of 500 \(.* limit\)\. Use offset=\d+ to
    // continue\.\]/).
    assert!(output.contains("[Showing lines 1-"));
    assert!(output.contains(" of 500 ("));
    assert!(output.contains(" limit). Use offset="));
}

#[tokio::test]
async fn read_handles_offset_parameter() {
    let dir = test_dir("read-offset");
    let test_file = dir.join("offset-test.txt");
    let lines: Vec<String> = (0..100).map(|i| format!("Line {}", i + 1)).collect();
    write_file(&test_file, &lines.join("\n"));

    let result = execute_tool(
        &read_tool(),
        "test-call-5",
        json!({"path": test_file.to_string_lossy(), "offset": 51}),
    )
    .await
    .unwrap();
    let output = get_text_output(&result);

    assert!(!output.contains("Line 50\n") && !output.contains("\nLine 50"));
    assert!(output.contains("Line 51"));
    assert!(output.contains("Line 100"));
    assert!(!output.contains("Use offset="));
}

#[tokio::test]
async fn read_handles_limit_parameter() {
    let dir = test_dir("read-limit");
    let test_file = dir.join("limit-test.txt");
    let lines: Vec<String> = (0..100).map(|i| format!("Line {}", i + 1)).collect();
    write_file(&test_file, &lines.join("\n"));

    let result = execute_tool(
        &read_tool(),
        "test-call-6",
        json!({"path": test_file.to_string_lossy(), "limit": 10}),
    )
    .await
    .unwrap();
    let output = get_text_output(&result);

    assert!(output.contains("Line 1\n"));
    assert!(output.contains("\nLine 10"));
    assert!(!output.contains("Line 11"));
    assert!(output.contains("[90 more lines in file. Use offset=11 to continue.]"));
}

#[tokio::test]
async fn read_handles_offset_plus_limit_together() {
    let dir = test_dir("read-offset-limit");
    let test_file = dir.join("offset-limit-test.txt");
    let lines: Vec<String> = (0..100).map(|i| format!("Line {}", i + 1)).collect();
    write_file(&test_file, &lines.join("\n"));

    let result = execute_tool(
        &read_tool(),
        "test-call-7",
        json!({"path": test_file.to_string_lossy(), "offset": 41, "limit": 20}),
    )
    .await
    .unwrap();
    let output = get_text_output(&result);

    assert!(!output.contains("\nLine 40") && !output.starts_with("Line 40"));
    assert!(output.contains("Line 41"));
    assert!(output.contains("Line 60"));
    assert!(!output.contains("Line 61"));
    assert!(output.contains("[40 more lines in file. Use offset=61 to continue.]"));
}

#[tokio::test]
async fn read_shows_error_when_offset_is_beyond_file_length() {
    let dir = test_dir("read-offset-oob");
    let test_file = dir.join("short.txt");
    write_file(&test_file, "Line 1\nLine 2\nLine 3");

    let error = execute_tool(
        &read_tool(),
        "test-call-8",
        json!({"path": test_file.to_string_lossy(), "offset": 100}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("Offset 100 is beyond end of file (3 lines total)"));
}

#[tokio::test]
async fn read_includes_truncation_details_when_truncated() {
    let dir = test_dir("read-details");
    let test_file = dir.join("large-file.txt");
    let lines: Vec<String> = (0..2500).map(|i| format!("Line {}", i + 1)).collect();
    write_file(&test_file, &lines.join("\n"));

    let result = execute_tool(
        &read_tool(),
        "test-call-9",
        json!({"path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();

    assert_ne!(result.details, Value::Null);
    let truncation = &result.details["truncation"];
    assert_eq!(truncation["truncated"], json!(true));
    assert_eq!(truncation["truncatedBy"], json!("lines"));
    assert_eq!(truncation["totalLines"], json!(2500));
    assert_eq!(truncation["outputLines"], json!(2000));
}

#[tokio::test]
async fn write_writes_file_contents() {
    let dir = test_dir("write-content");
    let test_file = dir.join("write-test.txt");
    let content = "Test content";

    let result = execute_tool(
        &create_write_tool(".", None),
        "test-call-3",
        json!({"path": test_file.to_string_lossy(), "content": content}),
    )
    .await
    .unwrap();

    assert_eq!(
        get_text_output(&result),
        format!("Successfully wrote to {}", test_file.to_string_lossy())
    );
    assert_eq!(result.details, Value::Null);
}

#[tokio::test]
async fn write_creates_parent_directories() {
    let dir = test_dir("write-nested");
    let test_file = dir.join("nested/dir/test.txt");

    let result = execute_tool(
        &create_write_tool(".", None),
        "test-call-4",
        json!({"path": test_file.to_string_lossy(), "content": "Nested content"}),
    )
    .await
    .unwrap();

    assert!(get_text_output(&result).contains("Successfully wrote"));
}

// ---------------------------------------------------------------------------
// edit tool
// ---------------------------------------------------------------------------

fn edit_tool() -> AgentTool {
    create_edit_tool(".", None)
}

#[tokio::test]
async fn edit_replaces_text_in_file() {
    let dir = test_dir("edit-replace");
    let test_file = dir.join("edit-test.txt");
    let original_content = "Hello, world!";
    write_file(&test_file, original_content);

    let result = execute_tool(
		&edit_tool(),
		"test-call-5",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "world", "newText": "testing"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert_ne!(result.details, Value::Null);
    assert!(result.details["diff"].is_string());
    assert!(result.details["diff"].as_str().unwrap().contains("testing"));
    assert!(result.details["patch"].as_str().unwrap().contains("--- "));
    assert!(result.details["patch"].as_str().unwrap().contains("+++ "));
    assert!(result.details["patch"].as_str().unwrap().contains("@@"));
    assert!(result.details["patch"]
        .as_str()
        .unwrap()
        .contains("-Hello, world!"));
    assert!(result.details["patch"]
        .as_str()
        .unwrap()
        .contains("+Hello, testing!"));
    let patched =
        apply_unified_patch(original_content, result.details["patch"].as_str().unwrap()).unwrap();
    assert_eq!(patched, "Hello, testing!");
}

#[tokio::test]
async fn edit_fails_if_text_not_found() {
    let dir = test_dir("edit-not-found");
    let test_file = dir.join("edit-test.txt");
    write_file(&test_file, "Hello, world!");

    let error = execute_tool(
		&edit_tool(),
		"test-call-6",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "nonexistent", "newText": "testing"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains("Could not find the exact text"));
}

#[tokio::test]
async fn edit_includes_enoent_when_the_edit_target_does_not_exist() {
    let dir = test_dir("edit-enoent");
    let missing_file = dir.join("missing.txt");

    let error = execute_tool(
		&edit_tool(),
		"test-call-6b",
		json!({"path": missing_file.to_string_lossy(), "edits": [{"oldText": "hello", "newText": "world"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains(&format!(
        "Could not edit file: {}. Error code: ENOENT.",
        missing_file.to_string_lossy()
    )));
}

#[tokio::test]
async fn edit_fails_if_text_appears_multiple_times() {
    let dir = test_dir("edit-dup");
    let test_file = dir.join("edit-test.txt");
    write_file(&test_file, "foo foo foo");

    let error = execute_tool(
		&edit_tool(),
		"test-call-7",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "foo", "newText": "bar"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains("Found 3 occurrences"));
}

#[tokio::test]
async fn edit_replaces_multiple_disjoint_regions_in_one_call() {
    let dir = test_dir("edit-multi");
    let test_file = dir.join("edit-multi.txt");
    write_file(&test_file, "alpha\nbeta\ngamma\ndelta\n");

    let result = execute_tool(
        &edit_tool(),
        "test-call-8",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "alpha\n", "newText": "ALPHA\n"},
            {"oldText": "gamma\n", "newText": "GAMMA\n"}
        ]}),
    )
    .await
    .unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced 2 block(s)"));
    assert_eq!(read_file(&test_file), "ALPHA\nbeta\nGAMMA\ndelta\n");
    assert!(result.details["diff"].as_str().unwrap().contains("ALPHA"));
    assert!(result.details["diff"].as_str().unwrap().contains("GAMMA"));
}

#[tokio::test]
async fn edit_collapses_large_unchanged_gaps_in_multi_edit_diffs() {
    let dir = test_dir("edit-gap");
    let test_file = dir.join("edit-multi-large-gap.txt");
    let lines: Vec<String> = (0..600).map(|i| format!("line {:03}", i + 1)).collect();
    write_file(&test_file, &format!("{}\n", lines.join("\n")));

    let result = execute_tool(
        &edit_tool(),
        "test-call-8b",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "line 100\n", "newText": "LINE 100\n"},
            {"oldText": "line 300\n", "newText": "LINE 300\n"},
            {"oldText": "line 500\n", "newText": "LINE 500\n"}
        ]}),
    )
    .await
    .unwrap();

    let diff = result.details["diff"].as_str().unwrap_or("");
    assert!(diff.contains("LINE 100"));
    assert!(diff.contains("LINE 300"));
    assert!(diff.contains("LINE 500"));
    assert!(diff.contains("..."));
    assert!(!diff.contains("line 250"));
    assert!(diff.split('\n').count() < 50);
}

#[tokio::test]
async fn edit_matches_edits_against_the_original_file_not_incrementally() {
    let dir = test_dir("edit-original");
    let test_file = dir.join("edit-multi-original.txt");
    write_file(&test_file, "foo\nbar\nbaz\n");

    execute_tool(
        &edit_tool(),
        "test-call-9",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "foo\n", "newText": "foo bar\n"},
            {"oldText": "bar\n", "newText": "BAR\n"}
        ]}),
    )
    .await
    .unwrap();

    assert_eq!(read_file(&test_file), "foo bar\nBAR\nbaz\n");
}

#[tokio::test]
async fn edit_fails_when_edits_is_empty() {
    let dir = test_dir("edit-empty");
    let test_file = dir.join("edit-empty-edits.txt");
    write_file(&test_file, "hello\nworld\n");

    let error = execute_tool(
        &edit_tool(),
        "test-call-11",
        json!({"path": test_file.to_string_lossy(), "edits": []}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("edits must contain at least one replacement"));
}

#[tokio::test]
async fn edit_fails_when_multi_edit_regions_overlap() {
    let dir = test_dir("edit-overlap");
    let test_file = dir.join("edit-overlap.txt");
    write_file(&test_file, "one\ntwo\nthree\n");

    let error = execute_tool(
        &edit_tool(),
        "test-call-12",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
            {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"}
        ]}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("overlap"));
}

#[tokio::test]
async fn edit_does_not_partially_apply_edits_when_one_edit_fails() {
    let dir = test_dir("edit-no-partial");
    let test_file = dir.join("edit-no-partial.txt");
    let original_content = "alpha\nbeta\ngamma\n";
    write_file(&test_file, original_content);

    let error = execute_tool(
        &edit_tool(),
        "test-call-13",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "alpha\n", "newText": "ALPHA\n"},
            {"oldText": "missing\n", "newText": "MISSING\n"}
        ]}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("Could not find"));

    assert_eq!(read_file(&test_file), original_content);
}

#[cfg(unix)]
#[tokio::test]
async fn edit_includes_eacces_for_read_only_files() {
    let dir = test_dir("edit-readonly");
    let test_file = dir.join("edit-readonly.txt");
    write_file(&test_file, "hello\n");

    #[allow(clippy::permissions_readonly_false)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&test_file, std::fs::Permissions::from_mode(0o444)).unwrap();
    }

    let error = execute_tool(
		&edit_tool(),
		"test-call-14",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "hello", "newText": "world"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains(&format!(
        "Could not edit file: {}. Error code: EACCES.",
        test_file.to_string_lossy()
    )));
}

#[cfg(unix)]
#[tokio::test]
async fn edit_includes_the_original_error_message_for_unknown_edit_access_errors() {
    let dir = test_dir("edit-generic-failure");
    let generic_failure_tool = create_edit_tool(
        dir.to_string_lossy().as_ref(),
        Some(EditToolOptions {
            operations: Some(std::sync::Arc::new(GenericFailureOperations)),
        }),
    );

    let error = execute_tool(
        &generic_failure_tool,
        "test-call-16",
        json!({"path": "broken.txt", "edits": [{"oldText": "hello", "newText": "world"}]}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("Could not edit file: broken.txt. Error: disk offline."));
}

struct GenericFailureOperations;

impl EditOperations for GenericFailureOperations {
    fn read_file<'a>(
        &'a self,
        _absolute_path: String,
    ) -> pi_agent::BoxFuture<'a, Result<Vec<u8>, String>> {
        Box::pin(async { Ok(b"hello\n".to_vec()) })
    }

    fn write_file<'a>(
        &'a self,
        _absolute_path: String,
        _content: String,
    ) -> pi_agent::BoxFuture<'a, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }

    fn access<'a>(&'a self, _absolute_path: String) -> pi_agent::BoxFuture<'a, Result<(), String>> {
        Box::pin(async { Err("disk offline".to_string()) })
    }
}

#[tokio::test]
async fn edit_includes_enoent_in_diff_preview_for_missing_files() {
    let dir = test_dir("edit-preview-enoent");
    let missing_file = dir.join("missing-preview.txt");
    let result = compute_edits_diff(
        missing_file.to_string_lossy().as_ref(),
        &[Edit {
            old_text: "hello".to_string(),
            new_text: "world".to_string(),
        }],
        dir.to_string_lossy().as_ref(),
    )
    .await;

    assert!(matches!(&result, EditDiffOutcome::Error(error)
		if error.error == format!("Could not edit file: {}. Error code: ENOENT.", missing_file.to_string_lossy())));
}

#[cfg(unix)]
#[tokio::test]
async fn edit_includes_eaccels_in_diff_preview_for_unreadable_files() {
    let dir = test_dir("edit-preview-eacces");
    let unreadable_file = dir.join("unreadable-preview.txt");
    write_file(&unreadable_file, "hello\n");

    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&unreadable_file, std::fs::Permissions::from_mode(0o222)).unwrap();
    }

    let result = compute_edits_diff(
        unreadable_file.to_string_lossy().as_ref(),
        &[Edit {
            old_text: "hello".to_string(),
            new_text: "world".to_string(),
        }],
        dir.to_string_lossy().as_ref(),
    )
    .await;

    assert!(matches!(&result, EditDiffOutcome::Error(error)
		if error.error == format!("Could not edit file: {}. Error code: EACCES.", unreadable_file.to_string_lossy())));
}

// ---------------------------------------------------------------------------
// bash tool
// ---------------------------------------------------------------------------

fn bash_tool() -> AgentTool {
    create_bash_tool(".", None)
}

fn text_value(result: &AgentToolResult) -> String {
    get_text_output(result)
}

#[tokio::test]
async fn bash_executes_simple_commands() {
    let result = execute_tool(
        &bash_tool(),
        "test-call-8",
        json!({"command": "echo 'test output'"}),
    )
    .await
    .unwrap();

    assert!(text_value(&result).contains("test output"));
    assert_eq!(result.details, Value::Null);
}

#[tokio::test]
async fn bash_reports_non_zero_exit_codes_as_error_results_with_structured_content() {
    let result = execute_tool(
        &bash_tool(),
        "test-call-9",
        json!({"command": "echo out; exit 3"}),
    )
    .await
    .unwrap();
    assert_eq!(result.is_error, Some(true));
    assert_eq!(text_value(&result), "out\n\n\nCommand exited with code 3");
    let structured = &result.structured_content.as_ref().unwrap();
    assert_eq!(structured["output"], json!("out\n"));
    assert_eq!(structured["truncated"], json!(false));
    assert_eq!(structured["exit_code"], json!(3));
    assert!(structured["wall_time_seconds"].is_number());

    let ok = execute_tool(
        &bash_tool(),
        "test-call-9b",
        json!({"command": "echo fine"}),
    )
    .await
    .unwrap();
    assert_eq!(ok.is_error, None);
    let ok_structured = ok.structured_content.as_ref().unwrap();
    assert_eq!(ok_structured["output"], json!("fine\n"));
    assert_eq!(ok_structured["exit_code"], json!(0));

    let empty = execute_tool(&bash_tool(), "test-call-9c", json!({"command": "true"}))
        .await
        .unwrap();
    assert_eq!(text_value(&empty), "(no output)");
    let empty_structured = empty.structured_content.as_ref().unwrap();
    assert_eq!(empty_structured["output"], json!(""));
    assert_eq!(empty_structured["truncated"], json!(false));
}

#[tokio::test]
async fn bash_returns_up_to_1_mib_of_output_in_structured_content() {
    // 3000 lines exceed the model-facing 2000 line limit but not 1 MiB.
    let medium = execute_tool(
        &bash_tool(),
        "test-call-9d",
        json!({"command": "seq 1 3000"}),
    )
    .await
    .unwrap();
    assert!(!text_value(&medium).contains("\n1\n2\n"));
    assert_eq!(medium.details["truncation"]["truncated"], json!(true));
    let medium_output = &medium.structured_content.as_ref().unwrap();
    assert_eq!(medium_output["truncated"], json!(false));
    let expected: String = {
        let lines: Vec<String> = (1..=3000).map(|i| i.to_string()).collect();
        format!("{}\n", lines.join("\n"))
    };
    assert_eq!(medium_output["output"], json!(expected));

    // About 2 MB: keeps the first and last 512 KiB around an omission marker.
    let large = execute_tool(
        &bash_tool(),
        "test-call-9e",
        json!({"command": "seq 1 300000"}),
    )
    .await
    .unwrap();
    let large_output = large.structured_content.as_ref().unwrap();
    assert_eq!(large_output["truncated"], json!(true));
    assert!(large_output["output"]
        .as_str()
        .unwrap()
        .starts_with("1\n2\n3\n"));
    assert!(large_output["output"]
        .as_str()
        .unwrap()
        .ends_with("299999\n300000\n"));
    assert!(
        large_output["output"]
            .as_str()
            .unwrap()
            .contains("\n\n[... ")
            && large_output["output"]
                .as_str()
                .unwrap()
                .contains(" bytes omitted ...]\n\n")
    );
    assert!(large_output["output"].as_str().unwrap().len() < 1024 * 1024 + 100);
    assert_eq!(
        large_output["full_output_path"],
        large.details["fullOutputPath"]
    );
    assert!(read_file(Path::new(
        large_output["full_output_path"].as_str().unwrap()
    ))
    .ends_with("300000\n"));
}

#[tokio::test]
async fn bash_maps_signal_killed_commands_to_128_plus_the_signal_number() {
    let operations = create_local_bash_operations(None);
    for (signal, exit_code) in [("KILL", 137), ("TERM", 143)] {
        let result = operations
            .exec(
                &format!("kill -{signal} $$"),
                &test_dir("bash-signals").to_string_lossy(),
                BashExecOptions {
                    on_data: std::sync::Arc::new(|_| {}),
                    signal: None,
                    timeout: None,
                    env: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, Some(exit_code));
    }
}

#[tokio::test]
async fn bash_reports_signal_killed_commands_as_errors_while_preserving_partial_output() {
    for (signal, exit_code) in [("KILL", 137), ("TERM", 143)] {
        let result = execute_tool(
            &bash_tool(),
            &format!("test-call-signal-{signal}"),
            json!({"command": format!("printf 'before-kill\\n'; kill -{signal} $$")}),
        )
        .await
        .unwrap();
        assert_eq!(result.is_error, Some(true));
        let output = text_value(&result);
        assert!(
            output.contains("before-kill"),
            "partial output missing in {output:?}"
        );
        assert!(output
            .trim_end()
            .ends_with(&format!("Command exited with code {exit_code}")));
    }
}

#[tokio::test]
async fn bash_rejects_a_null_exit_code_from_custom_operations() {
    let operations: std::sync::Arc<dyn BashOperations> = std::sync::Arc::new(NullExitOperations);
    let bash = create_bash_tool(
        &test_dir("bash-null-exit").to_string_lossy(),
        Some(BashToolOptions {
            operations: Some(operations),
            ..Default::default()
        }),
    );

    let error = execute_tool(&bash, "test-call-null-exit", json!({"command": "remote"}))
        .await
        .unwrap_err();
    assert!(error.contains("partial"));
    assert!(error
        .trim_end()
        .ends_with("Command terminated without an exit code"));
}

struct NullExitOperations;

impl BashOperations for NullExitOperations {
    fn exec(
        &self,
        _command: &str,
        _cwd: &str,
        options: BashExecOptions,
    ) -> pi_agent::BoxFuture<'static, Result<BashExecOutcome, String>> {
        options.on_data.do_callback(b"partial\n");
        Box::pin(async { Ok(BashExecOutcome { exit_code: None }) })
    }
}

trait OnDataExt {
    fn do_callback(&self, data: &[u8]);
}

impl OnDataExt for std::sync::Arc<dyn Fn(Vec<u8>) + Send + Sync> {
    fn do_callback(&self, data: &[u8]) {
        self(data.to_vec())
    }
}

#[tokio::test]
async fn bash_respects_timeout() {
    // Upstream spawns `process.execPath -e "setInterval(...)"`; the sh
    // equivalent is a long sleep (D35 test substitution).
    let error = execute_tool(
        &bash_tool(),
        "test-call-10",
        json!({"command": "sleep 5", "timeout": 0.05}),
    )
    .await
    .unwrap_err();
    assert!(error.to_lowercase().contains("timed out"));
}

#[tokio::test]
async fn bash_includes_full_output_path_for_truncated_timeout_and_abort_errors() {
    for (error, expected) in [
        ("timeout:5", "Command timed out after 5 seconds"),
        ("aborted", "Command aborted"),
    ] {
        let operations: std::sync::Arc<dyn BashOperations> =
            std::sync::Arc::new(ChattyFailureOperations {
                error: error.to_string(),
            });
        let bash = create_bash_tool(
            &test_dir("bash-truncated-errors").to_string_lossy(),
            Some(BashToolOptions {
                operations: Some(operations),
                ..Default::default()
            }),
        );

        let error_message = execute_tool(
            &bash,
            &format!("test-call-{error}"),
            json!({"command": "chatty-fail"}),
        )
        .await
        .unwrap_err();

        assert!(error_message.contains(expected));
        assert!(error_message.contains("[Showing lines "));
        assert!(error_message.contains(" of "));
        assert!(error_message.contains(". Full output: "));
        assert!(!error_message.contains("Full output: undefined"));
        let full_output_path = error_message
            .split("Full output: ")
            .nth(1)
            .and_then(|rest| rest.split(']').next())
            .and_then(|rest| rest.split('\n').next())
            .unwrap()
            .to_string();
        assert!(Path::new(&full_output_path).exists());
        let full_output = read_file(Path::new(&full_output_path));
        assert!(full_output.contains("1\n2\n3"));
        assert!(full_output.contains("2998\n2999\n3000"));
    }
}

struct ChattyFailureOperations {
    error: String,
}

impl BashOperations for ChattyFailureOperations {
    fn exec(
        &self,
        _command: &str,
        _cwd: &str,
        options: BashExecOptions,
    ) -> pi_agent::BoxFuture<'static, Result<BashExecOutcome, String>> {
        let error = self.error.clone();
        for i in 1..=3000 {
            options.on_data.do_callback(format!("{i}\n").as_bytes());
        }
        Box::pin(async move { Err(error) })
    }
}

#[tokio::test]
async fn bash_throws_error_when_cwd_does_not_exist() {
    let nonexistent_cwd = "/this/directory/definitely/does/not/exist/12345";

    let bash_tool_with_bad_cwd = create_bash_tool(nonexistent_cwd, None);

    let error = execute_tool(
        &bash_tool_with_bad_cwd,
        "test-call-11",
        json!({"command": "echo test"}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("Working directory does not exist"));
}

#[tokio::test]
async fn bash_handles_process_spawn_errors() {
    // Upstream spies getShellConfig to return a nonexistent shell; the port
    // injects the same config through the resolver (D35).
    let operations = create_local_shell_operations(
        "bash",
        std::sync::Arc::new(|| {
            Ok(crate::utils::shell::ShellConfig {
                shell: "/nonexistent-shell-path-xyz123".to_string(),
                args: vec!["-c".to_string()],
                command_transport: None,
            })
        }),
    );
    let bash_with_bad_shell = create_bash_tool(
        &test_dir("bash-bad-shell").to_string_lossy(),
        Some(BashToolOptions {
            operations: Some(operations),
            ..Default::default()
        }),
    );

    let error = execute_tool(
        &bash_with_bad_shell,
        "test-call-12",
        json!({"command": "echo test"}),
    )
    .await
    .unwrap_err();
    assert!(error.contains("ENOENT") || error.to_lowercase().contains("no such file"));
}

#[tokio::test]
async fn bash_passes_shell_path_through_to_shell_resolution() {
    // Custom operations bypass shell resolution entirely (upstream asserts
    // the getShellConfig spy was not called).
    let bash_with_custom_shell = create_bash_tool(
        &test_dir("bash-shell-path").to_string_lossy(),
        Some(BashToolOptions {
            shell_path: Some("/custom/bash".to_string()),
            operations: Some(std::sync::Arc::new(NoopExitOperations)),
            ..Default::default()
        }),
    );

    let result = execute_tool(
        &bash_with_custom_shell,
        "test-call-12b",
        json!({"command": "echo test"}),
    )
    .await
    .unwrap();
    assert_eq!(result.is_error, None);

    let ops = create_local_bash_operations(Some("/custom/bash".to_string()));
    let error = ops
        .exec(
            "echo test",
            &test_dir("bash-shell-path-2").to_string_lossy(),
            BashExecOptions {
                on_data: std::sync::Arc::new(|_| {}),
                signal: None,
                timeout: None,
                env: None,
            },
        )
        .await
        .err()
        .expect("expected the custom shell path to be rejected");
    assert_eq!(error, "Custom shell path not found: /custom/bash");
}

struct NoopExitOperations;

impl BashOperations for NoopExitOperations {
    fn exec(
        &self,
        _command: &str,
        _cwd: &str,
        _options: BashExecOptions,
    ) -> pi_agent::BoxFuture<'static, Result<BashExecOutcome, String>> {
        Box::pin(async { Ok(BashExecOutcome { exit_code: Some(0) }) })
    }
}

#[tokio::test]
async fn bash_sends_commands_over_stdin_when_shell_resolution_requires_it() {
    // Upstream mocks getShellConfig to a node script echoing stdin; the
    // port injects the equivalent config (D35): `cat` echoes the command.
    let operations = create_local_shell_operations(
        "bash",
        std::sync::Arc::new(|| {
            Ok(crate::utils::shell::ShellConfig {
                shell: "cat".to_string(),
                args: Vec::new(),
                command_transport: Some(crate::utils::shell::CommandTransport::Stdin),
            })
        }),
    );
    let chunks: std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>> = Default::default();
    let chunks_for_callback = std::sync::Arc::clone(&chunks);
    let name_expansion = "${name}";
    let count_expansion = "${count}";
    let i_expansion = "${i}";
    let command = format!(
		"name='World'; echo \"Hello, {name_expansion}!\"; count=3; for i in $(seq 1 {count_expansion}); do echo \"Iteration {i_expansion} of {count_expansion}\"; done"
	);

    let result = operations
        .exec(
            &command,
            &test_dir("bash-stdin").to_string_lossy(),
            BashExecOptions {
                on_data: std::sync::Arc::new(move |data| {
                    chunks_for_callback.lock().unwrap().push(data);
                }),
                signal: None,
                timeout: None,
                env: None,
            },
        )
        .await
        .unwrap();

    assert_eq!(result.exit_code, Some(0));
    let collected: Vec<u8> = chunks.lock().unwrap().concat();
    assert_eq!(String::from_utf8(collected).unwrap(), command);
}

#[tokio::test]
async fn bash_resolves_legacy_wsl_bash_exe_to_stdin_command_transport() {
    // Upstream returns early on non-Windows (the test simulates win32
    // behavior, which is not representable without a real Windows process).
    if !cfg!(windows) {
        return;
    }
    let shell_path = "C:\\Windows\\System32\\bash.exe";
    let config = crate::utils::shell::get_shell_config(Some(shell_path)).unwrap();
    assert_eq!(config.shell, shell_path);
    assert_eq!(config.args, vec!["-s".to_string()]);
    assert_eq!(
        config.command_transport,
        Some(crate::utils::shell::CommandTransport::Stdin)
    );
}

#[tokio::test]
async fn bash_prepends_command_prefix_when_configured() {
    let bash_with_prefix = create_bash_tool(
        &test_dir("bash-prefix").to_string_lossy(),
        Some(BashToolOptions {
            command_prefix: Some("export TEST_VAR=hello".to_string()),
            ..Default::default()
        }),
    );

    let result = execute_tool(
        &bash_with_prefix,
        "test-prefix-1",
        json!({"command": "echo $TEST_VAR"}),
    )
    .await
    .unwrap();
    assert_eq!(text_value(&result).trim(), "hello");
}

#[tokio::test]
async fn bash_includes_output_from_both_prefix_and_command() {
    let bash_with_prefix = create_bash_tool(
        &test_dir("bash-prefix-2").to_string_lossy(),
        Some(BashToolOptions {
            command_prefix: Some("echo prefix-output".to_string()),
            ..Default::default()
        }),
    );

    let result = execute_tool(
        &bash_with_prefix,
        "test-prefix-2",
        json!({"command": "echo command-output"}),
    )
    .await
    .unwrap();
    assert_eq!(text_value(&result).trim(), "prefix-output\ncommand-output");
}

#[tokio::test]
async fn bash_works_without_command_prefix() {
    let bash_without_prefix = create_bash_tool(
        &test_dir("bash-prefix-3").to_string_lossy(),
        Some(BashToolOptions::default()),
    );

    let result = execute_tool(
        &bash_without_prefix,
        "test-prefix-3",
        json!({"command": "echo no-prefix"}),
    )
    .await
    .unwrap();
    assert_eq!(text_value(&result).trim(), "no-prefix");
}

#[tokio::test]
async fn bash_coalesces_streaming_updates_for_chatty_output() {
    let operations: std::sync::Arc<dyn BashOperations> = std::sync::Arc::new(ChattyOperations);
    let updates: std::sync::Arc<std::sync::Mutex<Vec<AgentToolResult>>> = Default::default();
    let bash = create_bash_tool(
        &test_dir("bash-chatty").to_string_lossy(),
        Some(BashToolOptions {
            operations: Some(operations),
            ..Default::default()
        }),
    );

    let updates_for_callback = std::sync::Arc::clone(&updates);
    let execute = std::sync::Arc::clone(&bash.execute);
    let on_update: pi_agent::AgentToolUpdateCallback =
        std::sync::Arc::new(move |update: AgentToolResult| {
            updates_for_callback.lock().unwrap().push(update);
        });
    let result = execute(
        "test-call-chatty-updates".to_string(),
        json!({"command": "chatty"}).as_object().unwrap().clone(),
        None,
        Some(on_update),
    )
    .await
    .unwrap();

    let updates = updates.lock().unwrap();
    assert!(updates.len() < 25);
    assert!(text_value(&result).contains("line 4999"));
}

struct ChattyOperations;

impl BashOperations for ChattyOperations {
    fn exec(
        &self,
        _command: &str,
        _cwd: &str,
        options: BashExecOptions,
    ) -> pi_agent::BoxFuture<'static, Result<BashExecOutcome, String>> {
        for i in 0..5000 {
            options
                .on_data
                .do_callback(format!("line {i}\n").as_bytes());
        }
        Box::pin(async { Ok(BashExecOutcome { exit_code: Some(0) }) })
    }
}

#[tokio::test]
async fn bash_does_not_count_a_trailing_newline_as_an_extra_truncated_bash_output_line() {
    let operations: std::sync::Arc<dyn BashOperations> = std::sync::Arc::new(PaddedLinesOperations);
    let bash = create_bash_tool(
        &test_dir("bash-trailing-newline").to_string_lossy(),
        Some(BashToolOptions {
            operations: Some(operations),
            ..Default::default()
        }),
    );

    let result = execute_tool(
        &bash,
        "test-call-trailing-newline-line-count",
        json!({"command": "many-lines"}),
    )
    .await
    .unwrap();
    let output = text_value(&result);

    assert_eq!(result.details["truncation"]["totalLines"], json!(4000));
    assert_eq!(result.details["truncation"]["outputLines"], json!(2000));
    assert!(output.contains("line-2001"));
    assert!(output.contains("line-4000"));
    assert!(output.contains("[Showing lines 2001-4000 of 4000. Full output: "));
    assert!(!output.contains("4001"));
}

struct PaddedLinesOperations;

impl BashOperations for PaddedLinesOperations {
    fn exec(
        &self,
        _command: &str,
        _cwd: &str,
        options: BashExecOptions,
    ) -> pi_agent::BoxFuture<'static, Result<BashExecOutcome, String>> {
        for i in 1..=4000 {
            options
                .on_data
                .do_callback(format!("line-{i:04}\n").as_bytes());
        }
        Box::pin(async { Ok(BashExecOutcome { exit_code: Some(0) }) })
    }
}

#[tokio::test]
async fn bash_decodes_utf8_characters_split_across_output_chunks() {
    let euro = "€\n".as_bytes().to_vec();
    let operations: std::sync::Arc<dyn BashOperations> =
        std::sync::Arc::new(SplitUtf8Operations { first: euro[0] });
    let bash = create_bash_tool(
        &test_dir("bash-split-utf8").to_string_lossy(),
        Some(BashToolOptions {
            operations: Some(operations),
            ..Default::default()
        }),
    );

    let result = execute_tool(
        &bash,
        "test-call-split-utf8",
        json!({"command": "split-utf8"}),
    )
    .await
    .unwrap();

    assert_eq!(text_value(&result).trim(), "€");
}

struct SplitUtf8Operations {
    first: u8,
}

impl BashOperations for SplitUtf8Operations {
    fn exec(
        &self,
        _command: &str,
        _cwd: &str,
        options: BashExecOptions,
    ) -> pi_agent::BoxFuture<'static, Result<BashExecOutcome, String>> {
        let euro = "€\n".as_bytes().to_vec();
        options.on_data.do_callback(&euro[0..1]);
        options.on_data.do_callback(&euro[1..]);
        let _ = self.first;
        Box::pin(async { Ok(BashExecOutcome { exit_code: Some(0) }) })
    }
}

#[tokio::test]
async fn bash_exposes_local_bash_operations_for_extension_reuse() {
    let ops = create_local_bash_operations(None);
    let chunks: std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>> = Default::default();
    let chunks_for_callback = std::sync::Arc::clone(&chunks);

    let result = ops
        .exec(
            "echo $TEST_LOCAL_BASH_OPS",
            &test_dir("bash-local-ops").to_string_lossy(),
            BashExecOptions {
                on_data: std::sync::Arc::new(move |data| {
                    chunks_for_callback.lock().unwrap().push(data);
                }),
                signal: None,
                timeout: None,
                env: Some(
                    std::env::vars()
                        .chain([(
                            "TEST_LOCAL_BASH_OPS".to_string(),
                            "from-local-ops".to_string(),
                        )])
                        .collect(),
                ),
            },
        )
        .await
        .unwrap();

    assert_eq!(result.exit_code, Some(0));
    let collected: Vec<u8> = chunks.lock().unwrap().concat();
    assert_eq!(
        String::from_utf8(collected).unwrap().trim(),
        "from-local-ops"
    );
}

#[tokio::test]
async fn bash_preserves_execute_bash_sanitization_when_using_local_bash_operations() {
    let result = execute_bash_with_operations(
        "printf '\\033[31mred\\033[0m\\r\\n'",
        ".",
        create_local_bash_operations(None),
        None,
    )
    .await
    .unwrap();

    assert_eq!(result.exit_code, Some(0));
    assert_eq!(result.output, "red\n");
}

#[tokio::test]
async fn bash_persists_full_output_when_truncation_happens_by_line_count_only() {
    let bash = create_bash_tool(&test_dir("bash-line-truncation").to_string_lossy(), None);
    let result = execute_tool(
        &bash,
        "test-call-line-truncation",
        json!({"command": "seq 3000"}),
    )
    .await
    .unwrap();
    let output = text_value(&result);
    let full_output_path = result.details["fullOutputPath"]
        .as_str()
        .map(str::to_string);

    assert_eq!(result.details["truncation"]["truncated"], json!(true));
    assert_eq!(result.details["truncation"]["truncatedBy"], json!("lines"));
    assert!(full_output_path.is_some());
    assert!(output.contains("[Showing lines "));
    assert!(output.contains(" of 3000. Full output: "));
    assert!(!output.contains("Full output: undefined"));

    let full_output_path = full_output_path.unwrap();
    for _ in 0..20 {
        if Path::new(&full_output_path).exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    assert!(Path::new(&full_output_path).exists());
    let full_output = read_file(Path::new(&full_output_path));
    assert!(full_output.contains("1\n2\n3"));
    assert!(full_output.contains("2998\n2999\n3000"));
}

#[tokio::test]
async fn execute_bash_persists_full_output_when_truncation_happens_by_line_count_only() {
    let result =
        execute_bash_with_operations("seq 3000", ".", create_local_bash_operations(None), None)
            .await
            .unwrap();
    let full_output_path = result.full_output_path.clone();

    assert!(result.truncated);
    assert!(full_output_path.is_some());

    let full_output_path = full_output_path.unwrap();
    for _ in 0..20 {
        if Path::new(&full_output_path).exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    assert!(Path::new(&full_output_path).exists());
    let full_output = read_file(Path::new(&full_output_path));
    assert!(full_output.contains("1\n2\n3"));
    assert!(full_output.contains("2998\n2999\n3000"));
}

// ---------------------------------------------------------------------------
// grep tool
// ---------------------------------------------------------------------------

fn grep_tool() -> AgentTool {
    create_grep_tool(".", None)
}

#[tokio::test]
async fn grep_includes_filename_when_searching_a_single_file() {
    let dir = test_dir("grep-single");
    let test_file = dir.join("example.txt");
    write_file(&test_file, "first line\nmatch line\nlast line");

    let result = execute_tool(
        &grep_tool(),
        "test-call-11",
        json!({"pattern": "match", "path": test_file.to_string_lossy()}),
    )
    .await
    .unwrap();

    let output = text_value(&result);
    assert!(
        output.contains("example.txt:2: match line"),
        "output: {output}"
    );
}

#[tokio::test]
async fn grep_respects_global_limit_and_includes_context_lines() {
    let dir = test_dir("grep-context");
    let test_file = dir.join("context.txt");
    let content = [
        "before",
        "match one",
        "after",
        "middle",
        "match two",
        "after two",
    ]
    .join("\n");
    write_file(&test_file, &content);

    let result = execute_tool(
        &grep_tool(),
        "test-call-12",
        json!({"pattern": "match", "path": test_file.to_string_lossy(), "limit": 1, "context": 1}),
    )
    .await
    .unwrap();

    let output = text_value(&result);
    assert!(output.contains("context.txt-1- before"), "output: {output}");
    assert!(output.contains("context.txt:2: match one"));
    assert!(output.contains("context.txt-3- after"));
    assert!(output.contains("[1 matches limit reached. Use limit=2 for more, or refine pattern]"));
    // Ensure the second match is not present.
    assert!(!output.contains("match two"));
}

#[tokio::test]
async fn grep_treats_flag_like_patterns_as_search_text() {
    let dir = test_dir("grep-injection");
    let marker = dir.join("grep-injection-marker");
    let payload = dir.join("payload.sh");
    let test_file = dir.join("target.txt");
    write_file(
        &payload,
        &format!(
            "#!/bin/sh\necho executed > {}\ncat \"$1\"\n",
            marker.to_string_lossy()
        ),
    );
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&payload, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    write_file(&test_file, "target\n");

    let result = execute_tool(
		&grep_tool(),
		"test-call-grep-injection",
		json!({"pattern": format!("--pre={}", payload.to_string_lossy()), "path": dir.to_string_lossy()}),
	)
	.await
	.unwrap();

    assert!(text_value(&result).contains("No matches found"));
    assert!(!marker.exists());
}

// ---------------------------------------------------------------------------
// find tool
// ---------------------------------------------------------------------------

fn find_tool() -> AgentTool {
    create_find_tool(".", None)
}

#[tokio::test]
async fn find_includes_hidden_files_that_are_not_gitignored() {
    let dir = test_dir("find-hidden");
    let hidden_dir = dir.join(".secret");
    std::fs::create_dir_all(&hidden_dir).unwrap();
    write_file(&hidden_dir.join("hidden.txt"), "hidden");
    write_file(&dir.join("visible.txt"), "visible");

    let result = execute_tool(
        &find_tool(),
        "test-call-13",
        json!({"pattern": "**/*.txt", "path": dir.to_string_lossy()}),
    )
    .await
    .unwrap();

    let output_lines: Vec<String> = text_value(&result)
        .split('\n')
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    assert!(
        output_lines.contains(&"visible.txt".to_string()),
        "lines: {output_lines:?}"
    );
    assert!(output_lines.contains(&".secret/hidden.txt".to_string()));
}

#[tokio::test]
async fn find_respects_gitignore() {
    let dir = test_dir("find-gitignore");
    write_file(&dir.join(".gitignore"), "ignored.txt\n");
    write_file(&dir.join("ignored.txt"), "ignored");
    write_file(&dir.join("kept.txt"), "kept");

    let result = execute_tool(
        &find_tool(),
        "test-call-14",
        json!({"pattern": "**/*.txt", "path": dir.to_string_lossy()}),
    )
    .await
    .unwrap();

    let output = text_value(&result);
    assert!(output.contains("kept.txt"));
    assert!(!output.contains("ignored.txt"));
}

#[tokio::test]
async fn find_surfaces_fd_glob_parse_errors() {
    let dir = test_dir("find-bad-glob");
    let error = execute_tool(
        &find_tool(),
        "test-call-15",
        json!({"pattern": "[", "path": dir.to_string_lossy()}),
    )
    .await
    .unwrap_err();
    let lower = error.to_lowercase();
    assert!(
        lower.contains("error parsing glob")
            || lower.contains("fd exited with code 1")
            || lower.contains("fd error"),
        "error: {error}"
    );
}

#[tokio::test]
async fn find_treats_flag_like_patterns_as_search_text() {
    let dir = test_dir("find-flag-pattern");
    let result = execute_tool(
        &find_tool(),
        "test-call-find-flag-pattern",
        json!({"pattern": "--help", "path": dir.to_string_lossy()}),
    )
    .await
    .unwrap();

    assert!(text_value(&result).contains("No files found matching pattern"));
}

// ---------------------------------------------------------------------------
// ls tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ls_lists_dotfiles_and_directories() {
    let dir = test_dir("ls-dotfiles");
    write_file(&dir.join(".hidden-file"), "secret");
    std::fs::create_dir_all(dir.join(".hidden-dir")).unwrap();

    let result = execute_tool(
        &create_ls_tool(".", None),
        "test-call-15",
        json!({"path": dir.to_string_lossy()}),
    )
    .await
    .unwrap();
    let output = text_value(&result);

    assert!(output.contains(".hidden-file"));
    assert!(output.contains(".hidden-dir/"));
}

// ---------------------------------------------------------------------------
// tool cwd resolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cwd_read_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-read");
    let test_file = dir.join("ctx-cwd-read.txt");
    write_file(&test_file, "hello from ctx.cwd");
    let tool = create_read_tool_definition("/", None);
    let result = execute_definition(
        &tool,
        "test-read-ctx-cwd",
        json!({"path": "ctx-cwd-read.txt"}),
        Some(fake_ctx(&dir.to_string_lossy())),
    )
    .await
    .unwrap();
    assert!(get_text_output(&result).contains("hello from ctx.cwd"));
}

#[tokio::test]
async fn cwd_write_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-write");
    let tool = create_write_tool_definition("/", None);
    execute_definition(
        &tool,
        "test-write-ctx-cwd",
        json!({"path": "ctx-cwd-write.txt", "content": "written via ctx.cwd"}),
        Some(fake_ctx(&dir.to_string_lossy())),
    )
    .await
    .unwrap();
    let content = read_file(&dir.join("ctx-cwd-write.txt"));
    assert_eq!(content, "written via ctx.cwd");
}

#[tokio::test]
async fn cwd_edit_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-edit");
    let test_file = dir.join("ctx-cwd-edit.txt");
    write_file(&test_file, "old text");
    let tool = create_edit_tool_definition("/", None);
    execute_definition(
		&tool,
		"test-edit-ctx-cwd",
		json!({"path": "ctx-cwd-edit.txt", "edits": [{"oldText": "old text", "newText": "new text"}]}),
		Some(fake_ctx(&dir.to_string_lossy())),
	)
	.await
	.unwrap();
    assert_eq!(read_file(&test_file), "new text");
}

#[tokio::test]
async fn cwd_grep_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-grep");
    let test_file = dir.join("ctx-cwd-grep.txt");
    write_file(&test_file, "match in ctx.cwd");
    let tool = create_grep_tool_definition("/", None);
    let result = execute_definition(
        &tool,
        "test-grep-ctx-cwd",
        json!({"pattern": "match"}),
        Some(fake_ctx(&dir.to_string_lossy())),
    )
    .await
    .unwrap();
    assert!(text_value(&result).contains("ctx-cwd-grep.txt"));
}

#[tokio::test]
async fn cwd_find_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-find");
    write_file(&dir.join("ctx-cwd-find.txt"), "find me");
    let tool = create_find_tool_definition("/", None);
    let result = execute_definition(
        &tool,
        "test-find-ctx-cwd",
        json!({"pattern": "ctx-cwd-find.txt"}),
        Some(fake_ctx(&dir.to_string_lossy())),
    )
    .await
    .unwrap();
    assert!(text_value(&result).contains("ctx-cwd-find.txt"));
}

#[tokio::test]
async fn cwd_ls_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-ls");
    write_file(&dir.join("ctx-cwd-ls.txt"), "list me");
    let tool = create_ls_tool_definition("/", None);
    let result = execute_definition(
        &tool,
        "test-ls-ctx-cwd",
        json!({}),
        Some(fake_ctx(&dir.to_string_lossy())),
    )
    .await
    .unwrap();
    assert!(text_value(&result).contains("ctx-cwd-ls.txt"));
}

#[tokio::test]
async fn cwd_bash_uses_ctx_cwd_when_provided() {
    let dir = test_dir("ctx-bash");
    let tool = create_bash_tool_definition(
        "/",
        Some(BashToolOptions {
            expose_session_environment: Some(false),
            ..Default::default()
        }),
    );
    let result = execute_definition(
        &tool,
        "test-bash-ctx-cwd",
        json!({"command": "pwd"}),
        Some(fake_ctx(&dir.to_string_lossy())),
    )
    .await
    .unwrap();
    assert!(text_value(&result).contains(dir.to_string_lossy().as_ref()));
}

// ---------------------------------------------------------------------------
// edit tool fuzzy matching
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fuzzy_matches_text_with_trailing_whitespace_stripped() {
    let dir = test_dir("fuzzy-trailing-ws");
    let test_file = dir.join("trailing-ws.txt");
    write_file(&test_file, "line one   \nline two  \nline three\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-1",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "line one\nline two\n", "newText": "replaced\n"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert_eq!(read_file(&test_file), "replaced\nline three\n");
}

#[tokio::test]
async fn fuzzy_matches_fullwidth_punctuation_in_chinese_text() {
    let dir = test_dir("fuzzy-chinese");
    let test_file = dir.join("chinese-punctuation.txt");
    write_file(&test_file, "你好，世界\n你好（世界）\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-chinese",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "你好,世界\n你好(世界)\n", "newText": "你好，pi\n你好(pi)\n"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert_eq!(read_file(&test_file), "你好，pi\n你好(pi)\n");
}

#[tokio::test]
async fn fuzzy_matches_compatibility_equivalent_unicode_forms() {
    let dir = test_dir("fuzzy-unicode");
    let test_file = dir.join("unicode-compatibility.txt");
    write_file(&test_file, "ＡＢＣ１２３\ncafe\u{301}\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-unicode",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "ABC123\ncafé\n", "newText": "XYZ789\ncoffee\n"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert_eq!(read_file(&test_file), "XYZ789\ncoffee\n");
}

#[tokio::test]
async fn fuzzy_matches_smart_single_quotes_to_ascii_quotes() {
    let dir = test_dir("fuzzy-smart-single");
    let test_file = dir.join("smart-quotes.txt");
    write_file(&test_file, "console.log(\u{2018}hello\u{2019});\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-2",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "console.log('hello');", "newText": "console.log('world');"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert!(read_file(&test_file).contains("world"));
}

#[tokio::test]
async fn fuzzy_matches_smart_double_quotes_to_ascii_quotes() {
    let dir = test_dir("fuzzy-smart-double");
    let test_file = dir.join("smart-double-quotes.txt");
    write_file(&test_file, "const msg = \u{201C}Hello World\u{201D};\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-3",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "const msg = \"Hello World\";", "newText": "const msg = \"Goodbye\";"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert!(read_file(&test_file).contains("Goodbye"));
}

#[tokio::test]
async fn fuzzy_matches_unicode_dashes_to_ascii_hyphen() {
    let dir = test_dir("fuzzy-dashes");
    let test_file = dir.join("unicode-dashes.txt");
    write_file(&test_file, "range: 1\u{2013}5\nbreak\u{2014}here\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-4",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "range: 1-5\nbreak-here", "newText": "range: 10-50\nbreak--here"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert!(read_file(&test_file).contains("10-50"));
}

#[tokio::test]
async fn fuzzy_matches_non_breaking_space_to_regular_space() {
    let dir = test_dir("fuzzy-nbsp");
    let test_file = dir.join("nbsp.txt");
    write_file(&test_file, "hello\u{A0}world\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-5",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "hello world", "newText": "hello universe"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert!(read_file(&test_file).contains("universe"));
}

#[tokio::test]
async fn fuzzy_prefers_exact_match_over_fuzzy_match() {
    let dir = test_dir("fuzzy-exact-preferred");
    let test_file = dir.join("exact-preferred.txt");
    write_file(&test_file, "const x = 'exact';\nconst y = 'other';\n");

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-6",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "const x = 'exact';", "newText": "const x = 'changed';"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
    assert_eq!(
        read_file(&test_file),
        "const x = 'changed';\nconst y = 'other';\n"
    );
}

#[tokio::test]
async fn fuzzy_still_fails_when_text_is_not_found() {
    let dir = test_dir("fuzzy-no-match");
    let test_file = dir.join("no-match.txt");
    write_file(&test_file, "completely different content\n");

    let error = execute_tool(
		&edit_tool(),
		"test-fuzzy-7",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "this does not exist", "newText": "replacement"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains("Could not find the exact text"));
}

#[tokio::test]
async fn fuzzy_detects_duplicates_after_fuzzy_normalization() {
    let dir = test_dir("fuzzy-dups");
    let test_file = dir.join("fuzzy-dups.txt");
    write_file(&test_file, "hello world   \nhello world\n");

    let error = execute_tool(
		&edit_tool(),
		"test-fuzzy-8",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "hello world", "newText": "replaced"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains("Found 2 occurrences"));
}

#[tokio::test]
async fn fuzzy_supports_fuzzy_matching_in_multi_edit_mode() {
    let dir = test_dir("fuzzy-multi");
    let test_file = dir.join("fuzzy-multi.txt");
    write_file(
        &test_file,
        "console.log(\u{2018}hello\u{2019});\nhello\u{A0}world\n",
    );

    execute_tool(
        &edit_tool(),
        "test-fuzzy-9",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "console.log('hello');\n", "newText": "console.log('world');\n"},
            {"oldText": "hello world\n", "newText": "hello universe\n"}
        ]}),
    )
    .await
    .unwrap();

    assert_eq!(
        read_file(&test_file),
        "console.log('world');\nhello universe\n"
    );
}

#[tokio::test]
async fn fuzzy_preserves_the_correct_occurrence_when_fuzzy_replacement_equals_a_nearby_line() {
    let dir = test_dir("fuzzy-preserve-dup-line");
    let test_file = dir.join("fuzzy-preserve-duplicate-line.txt");
    let original_content = ["replace me   ", "after   ", ""].join("\n");
    write_file(&test_file, &original_content);

    let result = execute_tool(
		&edit_tool(),
		"test-fuzzy-preserve-duplicate-line",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "replace me\n", "newText": "after\n"}]}),
	)
	.await
	.unwrap();

    let expected_content = ["after", "after   ", ""].join("\n");
    assert_eq!(read_file(&test_file), expected_content);
    let patched = apply_unified_patch(
        &original_content,
        result.details["patch"].as_str().unwrap_or(""),
    )
    .unwrap();
    assert_eq!(patched, expected_content);
}

#[tokio::test]
async fn fuzzy_preserves_untouched_lines_and_produces_an_applicable_patch_for_fuzzy_multi_edits() {
    let dir = test_dir("fuzzy-preserve-multi");
    let test_file = dir.join("fuzzy-preserve-multi.txt");
    let original_content = [
        "keep before  ",
        "first target  ",
        "first after",
        "keep middle   ",
        "second target  ",
        "second after",
        "keep after  ",
        "",
    ]
    .join("\n");
    write_file(&test_file, &original_content);

    let result = execute_tool(
        &edit_tool(),
        "test-fuzzy-preserve-multi",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "first target\nfirst after", "newText": "FIRST\nFIRST2"},
            {"oldText": "second target\nsecond after", "newText": "SECOND\nSECOND2"}
        ]}),
    )
    .await
    .unwrap();

    let expected_content = [
        "keep before  ",
        "FIRST",
        "FIRST2",
        "keep middle   ",
        "SECOND",
        "SECOND2",
        "keep after  ",
        "",
    ]
    .join("\n");
    assert_eq!(read_file(&test_file), expected_content);
    let patched = apply_unified_patch(
        &original_content,
        result.details["patch"].as_str().unwrap_or(""),
    )
    .unwrap();
    assert_eq!(patched, expected_content);
}

// ---------------------------------------------------------------------------
// edit tool CRLF handling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn crlf_matches_lf_old_text_against_crlf_file_content() {
    let dir = test_dir("crlf-match");
    let test_file = dir.join("crlf-test.txt");
    write_file(&test_file, "line one\r\nline two\r\nline three\r\n");

    let result = execute_tool(
		&edit_tool(),
		"test-crlf-1",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "line two\n", "newText": "replaced line\n"}]}),
	)
	.await
	.unwrap();

    assert!(get_text_output(&result).contains("Successfully replaced"));
}

#[tokio::test]
async fn crlf_preserves_crlf_line_endings_after_edit() {
    let dir = test_dir("crlf-preserve");
    let test_file = dir.join("crlf-preserve.txt");
    write_file(&test_file, "first\r\nsecond\r\nthird\r\n");

    execute_tool(
		&edit_tool(),
		"test-crlf-2",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}),
	)
	.await
	.unwrap();

    assert_eq!(read_file(&test_file), "first\r\nREPLACED\r\nthird\r\n");
}

#[tokio::test]
async fn crlf_preserves_lf_line_endings_for_lf_files() {
    let dir = test_dir("lf-preserve");
    let test_file = dir.join("lf-preserve.txt");
    write_file(&test_file, "first\nsecond\nthird\n");

    execute_tool(
		&edit_tool(),
		"test-lf-1",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}),
	)
	.await
	.unwrap();

    assert_eq!(read_file(&test_file), "first\nREPLACED\nthird\n");
}

#[tokio::test]
async fn crlf_detects_duplicates_across_crlf_lf_variants() {
    let dir = test_dir("crlf-dup");
    let test_file = dir.join("mixed-endings.txt");
    write_file(&test_file, "hello\r\nworld\r\n---\r\nhello\nworld\n");

    let error = execute_tool(
		&edit_tool(),
		"test-crlf-dup",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "hello\nworld\n", "newText": "replaced\n"}]}),
	)
	.await
	.unwrap_err();
    assert!(error.contains("Found 2 occurrences"));
}

#[tokio::test]
async fn crlf_preserves_utf8_bom_after_edit() {
    let dir = test_dir("crlf-bom");
    let test_file = dir.join("bom-test.txt");
    write_file(&test_file, "\u{FEFF}first\r\nsecond\r\nthird\r\n");

    execute_tool(
		&edit_tool(),
		"test-bom",
		json!({"path": test_file.to_string_lossy(), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}),
	)
	.await
	.unwrap();

    assert_eq!(
        read_file(&test_file),
        "\u{FEFF}first\r\nREPLACED\r\nthird\r\n"
    );
}

#[tokio::test]
async fn crlf_preserves_crlf_line_endings_and_bom_in_multi_edit_mode() {
    let dir = test_dir("crlf-multi");
    let test_file = dir.join("bom-crlf-multi.txt");
    write_file(&test_file, "\u{FEFF}first\r\nsecond\r\nthird\r\nfourth\r\n");

    execute_tool(
        &edit_tool(),
        "test-crlf-multi",
        json!({"path": test_file.to_string_lossy(), "edits": [
            {"oldText": "second\n", "newText": "SECOND\n"},
            {"oldText": "fourth\n", "newText": "FOURTH\n"}
        ]}),
    )
    .await
    .unwrap();

    assert_eq!(
        read_file(&test_file),
        "\u{FEFF}first\r\nSECOND\r\nthird\r\nFOURTH\r\n"
    );
}
