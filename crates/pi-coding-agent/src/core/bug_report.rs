//! Port of `packages/coding-agent/src/core/bug-report.ts` — sensitive-data
//! redaction, environment/metadata collection, and the report bundle shared
//! by upload and zip export.
//!
//! Deviations (crate docs):
//! - **D125 — runtime identifier.** `collectEnvironment`'s `runtime` field
//!   reports `bun/x.y` or `node/vX` upstream; the port reports `rust` (no
//!   JS-runtime version exists). `platform`/`arch` reuse the Node naming
//!   from [`crate::utils::pi_user_agent`].
//! - **D126 — OS release/version.** Upstream `os.release()`/`os.version()`
//!   map to `uname(2)` via `libc` on unix (`release`/`version` fields);
//!   both are empty strings where `uname` is unavailable (non-unix hosts).
//! - **D127 — URL redaction engine.** `redactUrl` parses with the `url`
//!   crate instead of the WHATWG `URL` class: unchanged URLs are returned
//!   byte-for-byte (matching the upstream `changed ? url.toString() :
//!   value` passthrough), and re-serialization of *changed* URLs can differ
//!   from JS in percent-encoding corners.
//! - **D128 — Radius gateway resolution** lives in
//!   [`crate::core::bug_report_upload`] (upstream `core/radius.ts` is not
//!   ported yet).
//! - **D129 — crash record capture.** Upstream strips `notified` via object
//!   rest spread; the port serializes each [`CrashRecord`] to JSON and
//!   removes the `notified` key, so `crash_log::CrashRecord` must implement
//!   `Serialize` with upstream camelCase field names.

use std::collections::BTreeSet;
use std::path::Path;

use pi_agent::{AgentMessage, AgentThinkingLevel, StreamFn};
use pi_ai::models::{Models, Provider};
use pi_ai::types::{
	AssistantContent, Context, Message, Model, ProviderEnv, ProviderHeaders, SimpleStreamOptions, StopReason,
	UserContent, UserContentField, UserMessage,
};
use pi_ai::utils::abort::AbortSignal;
use pi_ai::utils::retry::RetryPolicy;
use pi_ai::utils::text::content_blocks_text;
use pi_ai::utils::transcript::normalize_context;
use pi_ai::utils::uuid::uuidv7_now;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::VERSION;
use crate::core::compaction::compaction::{complete_summarization, estimate_tokens, get_summarization_failure};
use crate::core::compaction::utils::serialize_conversation;
use crate::core::crash_log::CrashRecord;
use crate::core::extensions::types::{Extension, LoadRecord};
use crate::core::messages::convert_to_llm;
use crate::core::model_runtime::ModelRuntime;
use crate::core::provider_composer::AuthStatus;
use crate::core::session_manager::{SessionEntry, SessionManager};
use crate::core::settings_manager::Settings;
use crate::utils::pi_user_agent::{get_pi_user_agent, node_arch, node_platform};
use crate::utils::time::{now_ms, to_iso_timestamp};
use crate::utils::zip::{write_zip_archive, ZipEntry};

/// Custom session-entry type for persisted bug-report records
/// (`BUG_REPORT_CUSTOM_ENTRY_TYPE`).
pub const BUG_REPORT_CUSTOM_ENTRY_TYPE: &str = "pi.bug-report";

const BUG_REPORT_SCHEMA_VERSION: u32 = 1;
const REDACTED: &str = "<redacted>";

// ============================================================================
// Redaction
// ============================================================================

/// camelCase → snake_case insertion (upstream
/// `key.replace(/([a-z0-9])([A-Z])/g, "$1_$2")`): an `_` is inserted between
/// a lowercase letter or digit and a following uppercase letter.
fn split_camel_case(key: &str) -> String {
	let chars: Vec<char> = key.chars().collect();
	let mut out = String::with_capacity(key.len() + 4);
	for (index, &current) in chars.iter().enumerate() {
		if index > 0 {
			let previous = chars[index - 1];
			if current.is_ascii_uppercase() && (previous.is_ascii_lowercase() || previous.is_ascii_digit()) {
				out.push('_');
			}
		}
		out.push(current);
	}
	out
}

/// Upstream `SENSITIVE_KEY` word list minus the `api[-_]?key` alternative,
/// which is handled as the compound `api` + optional separator + `key`
/// below. All matches require a boundary (`^`/`-`/`_`) on both sides.
const SENSITIVE_WORDS: &[&str] =
	&["secret", "token", "password", "passwd", "credential", "authorization", "cookie"];

/// `(?:$|[-_])` — a word must end at a boundary.
fn ends_at_boundary(rest: &str) -> bool {
	rest.is_empty() || rest.starts_with('-') || rest.starts_with('_')
}

fn is_sensitive_key(key: &str) -> bool {
	let normalized = split_camel_case(key);
	let normalized = normalized.to_ascii_lowercase();
	for start in 0..normalized.len() {
		// `(?:^|[-_])` boundary before the word.
		if start > 0 && !matches!(normalized.as_bytes()[start - 1], b'-' | b'_') {
			continue;
		}
		let tail = &normalized[start..];
		if SENSITIVE_WORDS.iter().any(|word| tail.strip_prefix(word).is_some_and(ends_at_boundary)) {
			return true;
		}
		// `api[-_]?key`: optional separator between `api` and `key`.
		if let Some(rest) = tail.strip_prefix("api") {
			let rest = rest.strip_prefix(['-', '_']).unwrap_or(rest);
			if let Some(rest) = rest.strip_prefix("key") {
				if ends_at_boundary(rest) {
					return true;
				}
			}
		}
	}
	false
}

/// Split `scheme:` off the front of `value` (upstream nested-scheme regex
/// group 1, case-insensitive `[a-z][a-z0-9+.-]*:`).
fn split_scheme(value: &str) -> Option<(&str, &str)> {
	let bytes = value.as_bytes();
	let mut index = 0;
	if index >= bytes.len() || !bytes[index].is_ascii_alphabetic() {
		return None;
	}
	index += 1;
	while index < bytes.len()
		&& (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'+' | b'.' | b'-'))
	{
		index += 1;
	}
	if index < bytes.len() && bytes[index] == b':' {
		Some((&value[..index], &value[index + 1..]))
	} else {
		None
	}
}

/// Strip credentials and secret-looking query parameters from a URL
/// (`redactUrl`). Unparseable values and URLs without userinfo or sensitive
/// parameters are returned unchanged (see D127).
pub fn redact_url(value: &str) -> String {
	// Nested scheme (`socket:http://…`): redact the inner URL only.
	if let Some((scheme, rest)) = split_scheme(value) {
		if let Some((_, after)) = split_scheme(rest) {
			if after.starts_with("//") {
				return format!("{scheme}:{}", redact_url(rest));
			}
		}
	}
	let Ok(mut url) = url::Url::parse(value) else {
		return value.to_string();
	};
	let mut changed = false;
	if !url.username().is_empty() || url.password().is_some() {
		let _ = url.set_username("");
		let _ = url.set_password(None);
		changed = true;
	}
	// Sensitive parameters are redacted in place; duplicate sensitive keys
	// collapse to their first occurrence (upstream `searchParams.set`).
	let mut seen_sensitive: BTreeSet<String> = BTreeSet::new();
	let mut replacement: Vec<(String, String)> = Vec::new();
	for (key, value) in url.query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())) {
		if is_sensitive_key(&key) {
			if seen_sensitive.insert(key.clone()) {
				replacement.push((key, REDACTED.to_string()));
			}
			changed = true;
		} else {
			replacement.push((key, value));
		}
	}
	if changed && url.query().is_some() {
		let mut pairs = url.query_pairs_mut();
		pairs.clear();
		pairs.extend_pairs(replacement.iter().map(|(key, value)| (key.as_str(), value.as_str())));
	}
	if changed {
		url.to_string()
	} else {
		value.to_string()
	}
}

/// Copy a JSON value while removing values that may contain credentials
/// (`redactJsonValue`): sensitive keys replace their entire subtree, string
/// leaves are URL-redacted, `null` passes through.
pub fn redact_json_value(value: &Value) -> Value {
	match value {
		Value::Object(map) => Value::Object(
			map.iter()
				.map(|(key, child)| {
					let redacted = if !child.is_null() && is_sensitive_key(key) {
						Value::String(REDACTED.to_string())
					} else {
						redact_json_value(child)
					};
					(key.clone(), redacted)
				})
				.collect(),
		),
		Value::Array(items) => Value::Array(items.iter().map(redact_json_value).collect()),
		Value::String(text) => Value::String(redact_url(text)),
		other => other.clone(),
	}
}

/// Redact an already-typed value (`redactJsonValue(x)` on a serializable).
fn redact_serialized<T: Serialize>(value: &T) -> Value {
	redact_json_value(&serde_json::to_value(value).unwrap_or(Value::Null))
}

/// Upstream `redactSettings`: drop the machine identifiers, then apply the
/// generic sensitive-key walk.
fn redact_settings(settings: &Settings) -> Settings {
	let mut rest = settings.0.clone();
	rest.remove("trackingId");
	rest.remove("deviceId");
	match redact_json_value(&Value::Object(rest)) {
		Value::Object(map) => Settings(map),
		_ => Settings::new(),
	}
}

// ============================================================================
// Environment
// ============================================================================

/// `process.env[name] || null`.
fn env_value(name: &str) -> Option<String> {
	std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// `Boolean(process.env[name])`.
fn env_flag(name: &str) -> bool {
	std::env::var_os(name).is_some_and(|value| !value.is_empty())
}

/// `uname(2)` field (upstream `os.release()`/`os.version()`; D126).
#[cfg(unix)]
fn uname_value(release: bool) -> Option<String> {
	let mut uts: libc::utsname = unsafe { std::mem::zeroed() };
	if unsafe { libc::uname(&mut uts) } != 0 {
		return None;
	}
	let field: &[libc::c_char] = if release { &uts.release } else { &uts.version };
	let bytes: &[u8] = unsafe { std::slice::from_raw_parts(field.as_ptr().cast::<u8>(), field.len()) };
	let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
	Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

/// [`uname_value`] counterpart on hosts without `uname` (D126).
#[cfg(not(unix))]
fn uname_value(_release: bool) -> Option<String> {
	None
}

fn os_release() -> String {
	uname_value(true).unwrap_or_default()
}

fn os_version() -> String {
	uname_value(false).unwrap_or_default()
}

/// Terminal shape of the environment (`collectEnvironment.terminal`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalEnvironment {
	pub term: Option<String>,
	pub program: Option<String>,
	pub program_version: Option<String>,
	pub colorterm: Option<String>,
	pub tmux: bool,
	pub ssh: bool,
	pub ci: bool,
}

/// Process/host environment summary (`collectEnvironment`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentReport {
	pub version: String,
	pub user_agent: String,
	/// `rust` where upstream reports `bun/x.y`/`node/vX` (D125).
	pub runtime: String,
	pub platform: String,
	pub arch: String,
	pub os_release: String,
	pub os_version: String,
	pub shell: Option<String>,
	pub terminal: TerminalEnvironment,
	/// Names help diagnose configuration; values never leave the machine.
	pub pi_environment_variables: Vec<String>,
}

fn collect_environment() -> EnvironmentReport {
	let pi_environment_variables: BTreeSet<String> = std::env::vars_os()
		.map(|(name, _)| name.to_string_lossy().into_owned())
		.filter(|name| name.starts_with("PI_"))
		.collect();
	EnvironmentReport {
		version: VERSION.to_string(),
		user_agent: get_pi_user_agent(VERSION),
		runtime: "rust".to_string(),
		platform: node_platform(),
		arch: node_arch(),
		os_release: os_release(),
		os_version: os_version(),
		shell: std::env::var("SHELL")
			.ok()
			.and_then(|shell| shell.rsplit(['/', '\\']).next().map(str::to_string))
			.filter(|shell| !shell.is_empty()),
		terminal: TerminalEnvironment {
			term: env_value("TERM"),
			program: env_value("TERM_PROGRAM"),
			program_version: env_value("TERM_PROGRAM_VERSION"),
			colorterm: env_value("COLORTERM"),
			tmux: env_flag("TMUX"),
			ssh: env_flag("SSH_CONNECTION") || env_flag("SSH_CLIENT") || env_flag("SSH_TTY"),
			ci: env_flag("CI"),
		},
		pi_environment_variables: pi_environment_variables.into_iter().collect(),
	}
}

// ============================================================================
// Describers
// ============================================================================

/// `describeModel(model)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DescribeModel {
	pub provider: String,
	pub id: String,
	pub name: String,
	pub api: String,
	pub base_url: String,
	pub reasoning: bool,
	pub input: Vec<pi_ai::types::ModelInput>,
	pub context_window: u64,
	pub max_tokens: u64,
	pub sampling_params: Option<Value>,
	pub compat: Option<Value>,
	pub thinking_level_map: Option<Value>,
	pub header_names: Vec<String>,
}

fn describe_model(model: &Model) -> DescribeModel {
	DescribeModel {
		provider: model.base.provider.clone(),
		id: model.base.id.clone(),
		name: model.base.name.clone(),
		api: model.base.api.clone(),
		base_url: redact_url(&model.base.base_url),
		reasoning: model.reasoning,
		input: model.base.input.clone(),
		context_window: model.context_window,
		max_tokens: model.max_tokens,
		sampling_params: model.sampling_params.as_ref().map(redact_serialized),
		compat: model.compat.as_ref().map(redact_serialized),
		thinking_level_map: model
			.thinking_level_map
			.as_ref()
			.map(|map| serde_json::to_value(map).unwrap_or(Value::Null)),
		header_names: model.base.headers.as_ref().map(|headers| headers.keys().cloned().collect()).unwrap_or_default(),
	}
}

/// Serialized `getProviderAuthStatus` result: `source`/`label` are omitted
/// when absent, like `JSON.stringify` drops `undefined` fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DescribeAuthStatus {
	pub configured: bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub source: Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub label: Option<String>,
}

fn describe_auth_status(status: AuthStatus) -> DescribeAuthStatus {
	DescribeAuthStatus {
		configured: status.configured,
		source: status.source.map(|source| source.as_str()),
		label: status.label,
	}
}

/// `describeProvider(modelRuntime, provider)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DescribeProvider {
	pub id: String,
	pub name: String,
	pub base_url: Option<String>,
	pub header_names: Vec<String>,
	pub auth_types: Vec<&'static str>,
	pub auth_status: DescribeAuthStatus,
	pub using_oauth: bool,
	pub registered_by_extension: bool,
}

fn describe_provider(model_runtime: &ModelRuntime, provider: &dyn Provider) -> DescribeProvider {
	let mut auth_types: Vec<&'static str> = Vec::new();
	let auth = provider.auth();
	if auth.api_key.is_some() {
		auth_types.push("api_key");
	}
	if auth.oauth.is_some() {
		auth_types.push("oauth");
	}
	let provider_id = provider.id();
	DescribeProvider {
		id: provider_id.to_string(),
		name: provider.name().to_string(),
		base_url: provider.base_url().map(redact_url),
		header_names: provider.headers().map(|headers| headers.keys().cloned().collect()).unwrap_or_default(),
		auth_types,
		auth_status: describe_auth_status(model_runtime.get_provider_auth_status(provider_id)),
		using_oauth: model_runtime.is_using_oauth(provider_id),
		registered_by_extension: model_runtime.get_registered_provider_ids().iter().any(|id| id == provider_id),
	}
}

/// `describeExtension(extension)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DescribeExtension {
	pub path: String,
	pub source: String,
	pub scope: crate::core::source_info::SourceScope,
	pub origin: crate::core::source_info::SourceOrigin,
	pub hidden: bool,
}

fn describe_extension(extension: &Extension) -> DescribeExtension {
	DescribeExtension {
		path: extension.path.clone(),
		source: redact_url(&extension.source_info.source),
		scope: extension.source_info.scope,
		origin: extension.source_info.origin,
		hidden: extension.hidden == Some(true),
	}
}

// ============================================================================
// Metadata collection
// ============================================================================

/// Upstream `CollectBugReportMetadataOptions`.
pub struct CollectBugReportMetadataOptions<'a> {
	pub id: Option<&'a str>,
	pub hint: Option<&'a str>,
	pub session_id: &'a str,
	pub cwd: &'a str,
	pub include_session: bool,
	pub include_summary: bool,
	pub message_count: usize,
	pub model: Option<&'a Model>,
	pub model_runtime: &'a ModelRuntime,
	pub thinking_level: AgentThinkingLevel,
	pub extensions: &'a [Extension],
	pub extension_errors: &'a [LoadRecord],
	pub global_settings: &'a Settings,
	pub project_settings: &'a Settings,
}

/// Session coordinates (`collectBugReportMetadata().session`). `cwd` is only
/// present when the session transcript is included.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BugReportSessionInfo {
	pub id: String,
	pub included: bool,
	pub summary_included: bool,
	pub message_count: usize,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cwd: Option<String>,
}

/// Redacted global/project settings pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BugReportSettingsPair {
	pub global: Settings,
	pub project: Settings,
}

/// Collected report metadata (`BugReportMetadata`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BugReportMetadata {
	pub schema_version: u32,
	pub id: String,
	pub created_at: String,
	pub hint: Option<String>,
	pub environment: EnvironmentReport,
	pub session: BugReportSessionInfo,
	pub model: Option<DescribeModel>,
	pub provider: Option<DescribeProvider>,
	pub thinking_level: AgentThinkingLevel,
	pub extensions: Vec<DescribeExtension>,
	pub extension_errors: Vec<LoadRecord>,
	pub settings: BugReportSettingsPair,
}

/// Assemble the redacted report metadata (`collectBugReportMetadata`).
pub fn collect_bug_report_metadata(options: CollectBugReportMetadataOptions<'_>) -> BugReportMetadata {
	let hint = options.hint.map(str::trim).filter(|hint| !hint.is_empty());
	let provider = options.model.and_then(|model| options.model_runtime.get_provider(model.provider()));
	BugReportMetadata {
		schema_version: BUG_REPORT_SCHEMA_VERSION,
		id: options.id.map(str::to_string).unwrap_or_else(|| uuidv7_now().expect("uuidv7 clock")),
		created_at: to_iso_timestamp(now_ms()),
		hint: hint.map(str::to_string),
		environment: collect_environment(),
		session: BugReportSessionInfo {
			id: options.session_id.to_string(),
			included: options.include_session,
			summary_included: options.include_summary,
			message_count: options.message_count,
			cwd: options.include_session.then(|| options.cwd.to_string()),
		},
		model: options.model.map(describe_model),
		provider: provider.as_deref().map(|provider| describe_provider(options.model_runtime, provider)),
		thinking_level: options.thinking_level,
		extensions: options.extensions.iter().map(describe_extension).collect(),
		extension_errors: options.extension_errors.to_vec(),
		settings: BugReportSettingsPair {
			global: redact_settings(options.global_settings),
			project: redact_settings(options.project_settings),
		},
	}
}

// ============================================================================
// Diagnostics collection
// ============================================================================

/// One failed assistant turn (`collectBugReportDiagnostics` entries).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantDiagnostic {
	pub entry_id: String,
	pub timestamp: String,
	pub provider: String,
	pub model: String,
	pub api: String,
	pub stop_reason: StopReason,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub raw_stop_reason: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub error_message: Option<String>,
	pub diagnostics: Vec<pi_ai::types::AssistantMessageDiagnostic>,
}

/// Failed-turn diagnostics without conversation content
/// (`BugReportDiagnostics`). Crash records carry every upstream field except
/// `notified` (D129).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BugReportDiagnostics {
	pub schema_version: u32,
	pub session_id: String,
	pub entry_count: usize,
	pub assistant_message_count: usize,
	pub assistant: Vec<AssistantDiagnostic>,
	pub crashes: Vec<Value>,
}

/// Serialize a crash record without the `notified` flag (D129).
fn describe_crash(record: &CrashRecord) -> Value {
	let mut value = serde_json::to_value(record).unwrap_or(Value::Null);
	if let Some(map) = value.as_object_mut() {
		map.remove("notified");
	}
	value
}

/// Collect failed assistant turns without collecting conversation content
/// (`collectBugReportDiagnostics`).
pub fn collect_bug_report_diagnostics(
	session_manager: &SessionManager,
	crashes: &[CrashRecord],
) -> BugReportDiagnostics {
	let entries = session_manager.get_entries();
	let mut assistant: Vec<AssistantDiagnostic> = Vec::new();
	let mut assistant_message_count = 0usize;
	for entry in &entries {
		let SessionEntry::Message(entry) = entry else { continue };
		let AgentMessage::Llm(Message::Assistant(message)) = &entry.message else { continue };
		assistant_message_count += 1;
		let diagnostics = message.diagnostics.clone().unwrap_or_default();
		if diagnostics.is_empty()
			&& message.stop_reason != StopReason::Error
			&& message.stop_reason != StopReason::Aborted
			&& message.error_message.is_none()
		{
			continue;
		}
		assistant.push(AssistantDiagnostic {
			entry_id: entry.base.id.clone(),
			timestamp: entry.base.timestamp.clone(),
			provider: message.provider.clone(),
			model: message.model.clone(),
			api: message.api.clone(),
			stop_reason: message.stop_reason,
			raw_stop_reason: message.raw_stop_reason.clone(),
			error_message: message.error_message.clone(),
			diagnostics,
		});
	}
	BugReportDiagnostics {
		schema_version: BUG_REPORT_SCHEMA_VERSION,
		session_id: session_manager.get_session_id().to_string(),
		entry_count: entries.len(),
		assistant_message_count,
		assistant,
		crashes: crashes.iter().map(describe_crash).collect(),
	}
}

// ============================================================================
// Bundle / files
// ============================================================================

/// Everything a delivery needs (`BugReportBundle`).
#[derive(Debug, Clone, PartialEq)]
pub struct BugReportBundle {
	pub metadata: BugReportMetadata,
	pub diagnostics: BugReportDiagnostics,
	pub session_jsonl: Option<String>,
	pub summary: Option<String>,
}

/// Persisted bug-report session-entry payload (`BugReportSessionEntryData`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BugReportSessionEntryData {
	pub id: String,
	pub created_at: String,
	pub hint: Option<String>,
	pub session_included: bool,
	pub summary_included: bool,
	pub delivery: BugReportDelivery,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub path: Option<String>,
}

/// How the report left the machine (`BugReportSessionEntryData.delivery`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BugReportDelivery {
	Zip,
	Upload,
}

/// One file shared by upload and zip export (`BugReportFile`).
#[derive(Debug, Clone, PartialEq)]
pub struct BugReportFile {
	pub name: String,
	pub content_type: String,
	pub data: String,
}

/// Files shared by upload and zip export (`bugReportFiles`), in upstream
/// order: report.json, diagnostics.json, session.jsonl?, summary.md?.
pub fn bug_report_files(bundle: &BugReportBundle) -> Vec<BugReportFile> {
	let mut files: Vec<BugReportFile> = vec![
		BugReportFile {
			name: "report.json".to_string(),
			content_type: "application/json".to_string(),
			data: format!("{}\n", serde_json::to_string_pretty(&bundle.metadata).expect("report.json serializes")),
		},
		BugReportFile {
			name: "diagnostics.json".to_string(),
			content_type: "application/json".to_string(),
			data: format!(
				"{}\n",
				serde_json::to_string_pretty(&bundle.diagnostics).expect("diagnostics.json serializes")
			),
		},
	];
	if let Some(session_jsonl) = &bundle.session_jsonl {
		files.push(BugReportFile {
			name: "session.jsonl".to_string(),
			content_type: "application/x-ndjson".to_string(),
			data: session_jsonl.clone(),
		});
	}
	if let Some(summary) = &bundle.summary {
		files.push(BugReportFile {
			name: "summary.md".to_string(),
			content_type: "text/markdown".to_string(),
			data: if summary.ends_with('\n') { summary.clone() } else { format!("{summary}\n") },
		});
	}
	files
}

/// Write the report bundle as a ZIP archive (`writeBugReportArchive`).
pub fn write_bug_report_archive(bundle: &BugReportBundle, file_path: impl AsRef<Path>) -> std::io::Result<()> {
	let entries: Vec<ZipEntry> = bug_report_files(bundle)
		.into_iter()
		.map(|file| ZipEntry::new(file.name, file.data.into_bytes()))
		.collect();
	write_zip_archive(file_path, &entries)
}

/// `bugReportArchiveFileName`.
pub fn bug_report_archive_file_name(id: &str) -> String {
	format!("pi-bug-report-{id}.zip")
}

// ============================================================================
// Summary generation
// ============================================================================

const BUG_SUMMARY_SYSTEM_PROMPT: &str = "You are helping a user file a bug report about pi, the coding agent they are talking to. You will be shown the conversation transcript. Write a report for the pi developers describing what the user was doing and what went wrong.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the report.";

const BUG_SUMMARY_INSTRUCTIONS: &str = "Write the bug report in Markdown with these sections:\n\n## What the user was doing\nOne short paragraph.\n\n## What went wrong\nConcrete description of the failure: wrong output, errors, hangs, tool failures, unexpected behavior. Quote error messages and tool output verbatim where they exist.\n\n## Steps to reproduce\nNumbered list, as specific as the transcript allows.\n\n## Relevant details\nTool calls involved, files touched, model behavior, anything else that helps a developer reproduce or locate the problem.\n\nDo not include file contents, secrets, or credentials from the transcript; refer to files by path only. Keep the report factual and concise.";

/// Newest-first walk under `tokenBudget`; the newest message is always kept
/// (`selectMessages`).
fn select_messages(messages: &[AgentMessage], token_budget: usize) -> Vec<AgentMessage> {
	let mut selected: Vec<AgentMessage> = Vec::new();
	let mut tokens = 0usize;
	for message in messages.iter().rev() {
		let next = estimate_tokens(message);
		if !selected.is_empty() && tokens + next > token_budget {
			break;
		}
		selected.push(message.clone());
		tokens += next;
	}
	selected.reverse();
	selected
}

/// Upstream `GenerateBugReportSummaryOptions`.
pub struct GenerateBugReportSummaryOptions<'a> {
	pub messages: &'a [AgentMessage],
	pub hint: Option<&'a str>,
	pub model: &'a Model,
	pub api_key: Option<&'a str>,
	pub headers: Option<&'a ProviderHeaders>,
	pub env: Option<&'a ProviderEnv>,
	pub signal: AbortSignal,
	pub thinking_level: Option<AgentThinkingLevel>,
	pub stream_fn: Option<StreamFn>,
	pub retry: Option<&'a RetryPolicy>,
	pub session_id: Option<&'a str>,
}

/// Ask the session model for a report when the user does not share the
/// transcript (`generateBugReportSummary`).
pub async fn generate_bug_report_summary(options: GenerateBugReportSummaryOptions<'_>) -> Result<String, String> {
	let model = options.model;
	let context_window = if model.context_window > 0 { model.context_window } else { 128_000 };
	let messages = select_messages(options.messages, (context_window as f64 * 0.6).floor() as usize);
	let hint = options.hint.map(str::trim).filter(|hint| !hint.is_empty());
	let conversation = serialize_conversation(&convert_to_llm(&messages));
	let mut parts: Vec<String> = Vec::new();
	if messages.len() < options.messages.len() {
		parts.push(format!(
			"Note: only the last {} of {} messages are shown.",
			messages.len(),
			options.messages.len()
		));
	}
	parts.push(format!("<conversation>\n{conversation}\n</conversation>"));
	if let Some(hint) = hint {
		parts.push(format!("<user-report>\n{hint}\n</user-report>"));
	}
	parts.push(BUG_SUMMARY_INSTRUCTIONS.to_string());
	let prompt = parts.join("\n\n");

	let mut request_options = SimpleStreamOptions::default();
	request_options.base.max_tokens = Some(4096.min(if model.max_tokens > 0 { model.max_tokens } else { u64::MAX }));
	request_options.base.base.signal = Some(options.signal.clone());
	request_options.base.base.api_key = options.api_key.map(str::to_string);
	request_options.base.base.headers = options.headers.cloned();
	request_options.base.base.env = options.env.cloned();
	request_options.base.session_id = options.session_id.map(str::to_string);
	if model.reasoning && matches!(options.thinking_level, Some(level) if level != AgentThinkingLevel::Off) {
		request_options.reasoning = options.thinking_level.and_then(AgentThinkingLevel::to_reasoning);
	}

	let context = normalize_context(&Context {
		system_prompt: Some(BUG_SUMMARY_SYSTEM_PROMPT.to_string()),
		messages: vec![Message::User(UserMessage {
			content: UserContentField::Blocks(vec![UserContent::Text(pi_ai::types::TextContent::new(prompt))]),
			timestamp: now_ms(),
		})],
		tools: None,
	});
	let response = complete_summarization(model, context, request_options, options.stream_fn, options.retry, None).await?;
	if response.stop_reason == StopReason::Aborted {
		return Err("Bug report summary was cancelled".to_string());
	}
	if let Some(failure) = get_summarization_failure(&response, "Bug report summary") {
		return Err(failure);
	}
	if response.content.iter().any(|block| matches!(block, AssistantContent::ToolCall(_))) {
		return Err("Bug report summary attempted to call a tool".to_string());
	}
	let text = content_blocks_text(&response.content, "\n").trim().to_string();
	if text.is_empty() {
		return Err("Bug report summary was empty".to_string());
	}
	Ok(text)
}

// ============================================================================
// Test fixtures
// ============================================================================

/// Minimal valid metadata shared by [`bug_report_files`] tests and the
/// upload-module tests.
#[cfg(test)]
pub(crate) fn sample_bug_report_metadata() -> BugReportMetadata {
	BugReportMetadata {
		schema_version: BUG_REPORT_SCHEMA_VERSION,
		id: "01990000-0000-7000-8000-000000000000".to_string(),
		created_at: "2026-01-01T00:00:00.000Z".to_string(),
		hint: None,
		environment: EnvironmentReport {
			version: VERSION.to_string(),
			user_agent: get_pi_user_agent(VERSION),
			runtime: "rust".to_string(),
			platform: "darwin".to_string(),
			arch: "arm64".to_string(),
			os_release: "25.0.0".to_string(),
			os_version: "Darwin Kernel Version 25.0.0".to_string(),
			shell: None,
			terminal: TerminalEnvironment {
				term: None,
				program: None,
				program_version: None,
				colorterm: None,
				tmux: false,
				ssh: false,
				ci: false,
			},
			pi_environment_variables: Vec::new(),
		},
		session: BugReportSessionInfo {
			id: "ses".to_string(),
			included: true,
			summary_included: true,
			message_count: 3,
			cwd: Some("/tmp/project".to_string()),
		},
		model: None,
		provider: None,
		thinking_level: AgentThinkingLevel::Off,
		extensions: Vec::new(),
		extension_errors: Vec::new(),
		settings: BugReportSettingsPair { global: Settings::new(), project: Settings::new() },
	}
}

/// Minimal diagnostics shared by the bundle tests.
#[cfg(test)]
pub(crate) fn sample_bug_report_diagnostics() -> BugReportDiagnostics {
	BugReportDiagnostics {
		schema_version: BUG_REPORT_SCHEMA_VERSION,
		session_id: "ses".to_string(),
		entry_count: 0,
		assistant_message_count: 0,
		assistant: Vec::new(),
		crashes: Vec::new(),
	}
}

#[cfg(test)]
pub(crate) fn sample_bug_report_bundle() -> BugReportBundle {
	BugReportBundle {
		metadata: sample_bug_report_metadata(),
		diagnostics: sample_bug_report_diagnostics(),
		session_jsonl: Some("{\"type\":\"message\"}\n".to_string()),
		summary: Some("# Report\n".to_string()),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	fn user_message(text: &str) -> AgentMessage {
		AgentMessage::Llm(Message::User(UserMessage {
			content: UserContentField::Text(text.to_string()),
			timestamp: 0,
		}))
	}

	#[test]
	fn detects_sensitive_keys_across_casings() {
		for key in [
			"apiKey", "api_key", "API_KEY", "x-api-key", "my_apikey", "authToken", "SECRET", "db_password",
			"Authorization", "session-cookie", "provider_api_key",
		] {
			assert!(is_sensitive_key(key), "expected sensitive: {key}");
		}
		for key in [
			"token2", "tokens", "csrftoken", "authenticator", "name", "passwords_field", "cooking", "credentials",
		] {
			assert!(!is_sensitive_key(key), "expected plain: {key}");
		}
	}

	#[test]
	fn redact_url_leaves_plain_urls_untouched() {
		let value = "https://example.com/path?a=b%20c&x=1#frag";
		assert_eq!(redact_url(value), value);
		let value = "not a url";
		assert_eq!(redact_url(value), value);
	}

	#[test]
	fn redact_url_strips_userinfo() {
		assert_eq!(redact_url("https://user:pass@example.com/path"), "https://example.com/path");
		assert_eq!(redact_url("https://:pass@example.com/"), "https://example.com/");
	}

	#[test]
	fn redact_url_redacts_sensitive_query_parameters() {
		assert_eq!(redact_url("https://example.com/?api_key=abc&x=1"), "https://example.com/?api_key=%3Credacted%3E&x=1");
		assert_eq!(redact_url("https://example.com/?apiKey=abc"), "https://example.com/?apiKey=%3Credacted%3E");
		assert_eq!(redact_url("https://example.com/?auth-token=abc"), "https://example.com/?auth-token=%3Credacted%3E");
	}

	#[test]
	fn redact_url_handles_nested_schemes() {
		assert_eq!(
			redact_url("socket:http://user:pass@example.com/?token=x"),
			"socket:http://example.com/?token=%3Credacted%3E"
		);
	}

	#[test]
	fn redact_url_collapses_duplicate_sensitive_keys() {
		// Upstream `searchParams.set` keeps the first occurrence only.
		assert_eq!(
			redact_url("https://example.com/?token=a&x=1&token=b"),
			"https://example.com/?token=%3Credacted%3E&x=1"
		);
	}

	#[test]
	fn redact_json_value_walks_objects_arrays_and_strings() {
		let value = json!({
			"apiKey": { "nested": 1 },
			"list": ["http://user:pass@host/path"],
			"note": "http://user:pass@host/other",
			"token": null,
			"plain": 42
		});
		assert_eq!(
			redact_json_value(&value),
			json!({
				"apiKey": "<redacted>",
				"list": ["http://host/path"],
				"note": "http://host/other",
				"token": null,
				"plain": 42
			})
		);
	}

	#[test]
	fn redact_settings_drops_machine_identifiers() {
		let settings = Settings(
			json!({ "theme": "dark", "trackingId": "t", "deviceId": "d", "apiKey": "x" })
				.as_object()
				.expect("object")
				.clone(),
		);
		assert_eq!(
			redact_settings(&settings).0,
			json!({ "theme": "dark", "apiKey": "<redacted>" }).as_object().expect("object").clone()
		);
	}

	#[test]
	fn collect_environment_reports_crate_identity() {
		let environment = collect_environment();
		assert_eq!(environment.version, VERSION);
		assert_eq!(environment.user_agent, get_pi_user_agent(VERSION));
		assert_eq!(environment.runtime, "rust");
		assert!(!environment.platform.is_empty());
		assert!(!environment.arch.is_empty());
		let sorted = environment.pi_environment_variables.clone();
		let mut expected = sorted.clone();
		expected.sort();
		assert_eq!(sorted, expected);
		assert!(sorted.iter().all(|name| name.starts_with("PI_")));
	}

	#[test]
	fn bug_report_files_are_ordered_and_json_is_pretty() {
		let bundle = sample_bug_report_bundle();
		let files = bug_report_files(&bundle);
		assert_eq!(
			files.iter().map(|file| file.name.as_str()).collect::<Vec<_>>(),
			["report.json", "diagnostics.json", "session.jsonl", "summary.md"]
		);
		assert_eq!(
			files[0],
			BugReportFile {
				name: "report.json".to_string(),
				content_type: "application/json".to_string(),
				data: format!("{}\n", serde_json::to_string_pretty(&bundle.metadata).unwrap()),
			}
		);
		assert_eq!(
			files[1],
			BugReportFile {
				name: "diagnostics.json".to_string(),
				content_type: "application/json".to_string(),
				data: format!("{}\n", serde_json::to_string_pretty(&bundle.diagnostics).unwrap()),
			}
		);
		assert_eq!(files[2].content_type, "application/x-ndjson");
		assert_eq!(files[2].data, "{\"type\":\"message\"}\n");
		assert_eq!(files[3].content_type, "text/markdown");
		assert_eq!(files[3].data, "# Report\n");
	}

	#[test]
	fn bug_report_files_omit_absent_session_and_summary() {
		let bundle = BugReportBundle {
			metadata: sample_bug_report_metadata(),
			diagnostics: sample_bug_report_diagnostics(),
			session_jsonl: None,
			summary: None,
		};
		assert_eq!(bug_report_files(&bundle).len(), 2);
	}

	#[test]
	fn summary_files_end_with_a_newline() {
		let mut bundle = sample_bug_report_bundle();
		bundle.summary = Some("# Report".to_string());
		let files = bug_report_files(&bundle);
		assert_eq!(files[3].data, "# Report\n");
	}

	#[test]
	fn archive_file_name_follows_upstream_shape() {
		assert_eq!(bug_report_archive_file_name("br_1"), "pi-bug-report-br_1.zip");
	}

	#[test]
	fn write_bug_report_archive_produces_a_zip() {
		let path = std::env::temp_dir().join(format!("pi-bug-report-test-{}-{}.zip", std::process::id(), now_ms()));
		write_bug_report_archive(&sample_bug_report_bundle(), &path).expect("archive writes");
		let bytes = std::fs::read(&path).expect("archive readable");
		assert!(bytes.starts_with(b"PK\x03\x04"), "missing local file header");
		assert!(bytes.windows(4).any(|window| window == b"PK\x05\x06"), "missing EOCD");
		assert!(bytes.windows(11).any(|window| window == b"report.json"), "missing report entry");
		std::fs::remove_file(&path).expect("cleanup");
	}

	#[test]
	fn summary_prompts_are_verbatim() {
		assert_eq!(
			BUG_SUMMARY_SYSTEM_PROMPT,
			"You are helping a user file a bug report about pi, the coding agent they are talking to. You will be shown the conversation transcript. Write a report for the pi developers describing what the user was doing and what went wrong.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the report."
		);
		assert_eq!(
			BUG_SUMMARY_INSTRUCTIONS,
			"Write the bug report in Markdown with these sections:\n\n## What the user was doing\nOne short paragraph.\n\n## What went wrong\nConcrete description of the failure: wrong output, errors, hangs, tool failures, unexpected behavior. Quote error messages and tool output verbatim where they exist.\n\n## Steps to reproduce\nNumbered list, as specific as the transcript allows.\n\n## Relevant details\nTool calls involved, files touched, model behavior, anything else that helps a developer reproduce or locate the problem.\n\nDo not include file contents, secrets, or credentials from the transcript; refer to files by path only. Keep the report factual and concise."
		);
	}

	#[test]
	fn select_messages_keeps_newest_under_budget() {
		let messages: Vec<AgentMessage> =
			["one", "two two", "three three three", "four four four four"].iter().map(|text| user_message(text)).collect();
		let estimates: Vec<usize> = messages.iter().map(estimate_tokens).collect();
		assert!(estimates[0] < estimates[3], "fixture estimates must differ: {estimates:?}");

		// Budget fits exactly the last two.
		let budget = estimates[2] + estimates[3];
		let selected = select_messages(&messages, budget);
		assert_eq!(selected, vec![messages[2].clone(), messages[3].clone()]);

		// The newest message is always included, even over budget.
		assert_eq!(select_messages(&messages, 0), vec![messages[3].clone()]);

		// A generous budget keeps everything in original order.
		assert_eq!(select_messages(&messages, usize::MAX), messages);
	}
}
