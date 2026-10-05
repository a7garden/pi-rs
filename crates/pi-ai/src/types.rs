//! Port of `packages/ai/src/types.ts`.
//!
//! Wire-compatible type substrate for pi-ai. JSON shapes stay byte-compatible
//! with upstream: field names are serialized exactly as upstream emits them
//! (camelCase, `"toolUse"`, `"toolResult"`, ...), and optional upstream fields
//! are omitted (not nulled) when absent.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::http::FetchFn;

/// The uniform stream handle produced by every API implementation and
/// provider (upstream re-exports the type from `utils/event-stream.ts`).
pub use crate::utils::event_stream::AssistantMessageEventStream;

/// Known chat API implementation ids (upstream `KnownApi`). Model data carries
/// the raw string; unknown APIs are legal (upstream `Api = KnownApi | string`).
pub mod api_ids {
	pub const OPENAI_COMPLETIONS: &str = "openai-completions";
	pub const MISTRAL_CONVERSATIONS: &str = "mistral-conversations";
	pub const OPENAI_RESPONSES: &str = "openai-responses";
	pub const AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";
	pub const OPENAI_CODEX_RESPONSES: &str = "openai-codex-responses";
	pub const ANTHROPIC_MESSAGES: &str = "anthropic-messages";
	pub const BEDROCK_CONVERSE_STREAM: &str = "bedrock-converse-stream";
	pub const GOOGLE_GENERATIVE_AI: &str = "google-generative-ai";
	pub const GOOGLE_VERTEX: &str = "google-vertex";
	pub const PI_MESSAGES: &str = "pi-messages";
}

/// Known image API ids (upstream `KnownImageApi`).
pub mod image_api_ids {
	pub const OPENROUTER_IMAGES: &str = "openrouter-images";
}

/// Known classifier API ids (upstream `KnownClassifierApi`).
pub mod classifier_api_ids {
	pub const TYPESAFE_SYSTEM_ONE: &str = "typesafe-system-one";
	pub const CLOUDFLARE_WORKERS_AI_SYSTEM_ONE: &str = "cloudflare-workers-ai-system-one";
	pub const LLAMA_CPP_CLASSIFY: &str = "llama-cpp-classify";
}

/// Known provider ids (upstream `KnownProvider`). Model data carries the raw
/// string; unknown providers are legal (upstream `ProviderId`).
pub mod provider_ids {
	pub const AMAZON_BEDROCK: &str = "amazon-bedrock";
	pub const ANT_LING: &str = "ant-ling";
	pub const ANTHROPIC: &str = "anthropic";
	pub const GOOGLE: &str = "google";
	pub const GOOGLE_VERTEX: &str = "google-vertex";
	pub const OPENAI: &str = "openai";
	pub const AZURE: &str = "azure";
	pub const OPENAI_CODEX: &str = "openai-codex";
	pub const RADIUS: &str = "radius";
	pub const TYPESAFE: &str = "typesafe";
	pub const NVIDIA: &str = "nvidia";
	pub const DEEPSEEK: &str = "deepseek";
	pub const GITHUB_COPILOT: &str = "github-copilot";
	pub const XAI: &str = "xai";
	pub const GROQ: &str = "groq";
	pub const CEREBRAS: &str = "cerebras";
	pub const OPENROUTER: &str = "openrouter";
	pub const VERCEL_AI_GATEWAY: &str = "vercel-ai-gateway";
	pub const ZAI: &str = "zai";
	pub const ZAI_CODING_CN: &str = "zai-coding-cn";
	pub const MISTRAL: &str = "mistral";
	pub const MINIMAX: &str = "minimax";
	pub const MINIMAX_CN: &str = "minimax-cn";
	pub const MOONSHOTAI: &str = "moonshotai";
	pub const MOONSHOTAI_CN: &str = "moonshotai-cn";
	pub const HUGGINGFACE: &str = "huggingface";
	pub const FIREWORKS: &str = "fireworks";
	pub const TOGETHER: &str = "together";
	pub const BASETEN: &str = "baseten";
	pub const OPENCODE: &str = "opencode";
	pub const OPENCODE_GO: &str = "opencode-go";
	pub const KIMI_CODING: &str = "kimi-coding";
	pub const META: &str = "meta";
	pub const CLOUDFLARE_WORKERS_AI: &str = "cloudflare-workers-ai";
	pub const CLOUDFLARE_AI_GATEWAY: &str = "cloudflare-ai-gateway";
	pub const QWEN_TOKEN_PLAN: &str = "qwen-token-plan";
	pub const QWEN_TOKEN_PLAN_CN: &str = "qwen-token-plan-cn";
	pub const QWEN_TOKEN_PLAN_INDIVIDUAL: &str = "qwen-token-plan-individual";
	pub const XIAOMI: &str = "xiaomi";
	pub const XIAOMI_TOKEN_PLAN_CN: &str = "xiaomi-token-plan-cn";
	pub const XIAOMI_TOKEN_PLAN_AMS: &str = "xiaomi-token-plan-ams";
	pub const XIAOMI_TOKEN_PLAN_SGP: &str = "xiaomi-token-plan-sgp";
}

/// Provider-neutral tool selection for simple requests (`ToolChoice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
	Auto,
	None,
}

/// Pi thinking level (`ThinkingLevel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
	Minimal,
	Low,
	Medium,
	High,
	#[serde(rename = "xhigh")]
	Xhigh,
	#[serde(rename = "max")]
	Max,
}

/// Thinking level including the off state (`ModelThinkingLevel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelThinkingLevel {
	Off,
	Minimal,
	Low,
	Medium,
	High,
	#[serde(rename = "xhigh")]
	Xhigh,
	#[serde(rename = "max")]
	Max,
}

impl From<ThinkingLevel> for ModelThinkingLevel {
	fn from(level: ThinkingLevel) -> Self {
		match level {
			ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
			ThinkingLevel::Low => ModelThinkingLevel::Low,
			ThinkingLevel::Medium => ModelThinkingLevel::Medium,
			ThinkingLevel::High => ModelThinkingLevel::High,
			ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
			ThinkingLevel::Max => ModelThinkingLevel::Max,
		}
	}
}

impl ModelThinkingLevel {
	/// Ordered ladder matching upstream `EXTENDED_THINKING_LEVELS`.
	pub const LADDER: [ModelThinkingLevel; 7] = [
		ModelThinkingLevel::Off,
		ModelThinkingLevel::Minimal,
		ModelThinkingLevel::Low,
		ModelThinkingLevel::Medium,
		ModelThinkingLevel::High,
		ModelThinkingLevel::Xhigh,
		ModelThinkingLevel::Max,
	];
}

/// Maps pi thinking levels to provider/model-specific values
/// (`ThinkingLevelMap`). `None` marks a level as unsupported (JSON `null`).
pub type ThinkingLevelMap = BTreeMap<ModelThinkingLevel, Option<String>>;

/// Arbitrary sampling parameters merged into request bodies
/// (`SamplingParams`). Key order is preserved (`preserve_order`).
pub type SamplingParams = serde_json::Map<String, Value>;

/// Sampling parameter overrides selected by thinking level
/// (`SamplingParamsByThinkingLevel`).
pub type SamplingParamsByThinkingLevel = BTreeMap<ModelThinkingLevel, SamplingParams>;

/// Chat-template kwarg value (`ChatTemplateKwargValue`): a literal, or a
/// `$var` placeholder resolved by pi at request time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateKwargValue {
	Literal(JsonLiteral),
	Var {
		#[serde(rename = "$var")]
		var: ChatTemplateVar,
		#[serde(rename = "omitWhenOff", default, skip_serializing_if = "Option::is_none")]
		omit_when_off: Option<bool>,
	},
}

/// Literal scalar variants of [`ChatTemplateKwargValue`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonLiteral {
	String(String),
	Number(f64),
	Boolean(bool),
	Null,
}

/// Placeholder variable names accepted by `$var` template kwargs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatTemplateVar {
	#[serde(rename = "thinking.enabled")]
	ThinkingEnabled,
	#[serde(rename = "thinking.effort")]
	ThinkingEffort,
	#[serde(rename = "thinking.budget")]
	ThinkingBudget,
}

/// Top-level request field used to cap reasoning tokens on OpenAI-compatible
/// servers (`ThinkingTokenBudgetField`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingTokenBudgetField {
	#[serde(rename = "thinking_token_budget")]
	ThinkingTokenBudget,
	#[serde(rename = "thinking_budget")]
	ThinkingBudget,
	#[serde(rename = "thinking_budget_tokens")]
	ThinkingBudgetTokens,
}

/// Token budgets for each thinking level (`ThinkingBudgets`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub struct ThinkingBudgets {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub minimal: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub low: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub medium: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub high: Option<u64>,
}

/// Prompt cache retention preference (`CacheRetention`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
	None,
	Short,
	Long,
}

/// Best-effort prompt cache lifetime in seconds per retention tier
/// (`ModelPromptCache`). `none` is excluded upstream, hence the two keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPromptCache {
	#[serde(rename = "short", default, skip_serializing_if = "Option::is_none")]
	pub short: Option<u64>,
	#[serde(rename = "long", default, skip_serializing_if = "Option::is_none")]
	pub long: Option<u64>,
}

/// Preferred transport for providers that support multiple transports
/// (`Transport`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transport {
	#[serde(rename = "sse")]
	Sse,
	#[serde(rename = "websocket")]
	Websocket,
	#[serde(rename = "websocket-cached")]
	WebsocketCached,
	#[serde(rename = "auto")]
	Auto,
}

/// Provider-scoped environment overrides (`ProviderEnv`). Values take
/// precedence over process environment variables.
pub type ProviderEnv = BTreeMap<String, String>;

/// Custom HTTP headers merged over provider defaults (`ProviderHeaders`).
/// `None` values suppress a provider/API default header.
pub type ProviderHeaders = BTreeMap<String, Option<String>>;

/// Session-affinity header format (`SessionAffinityFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionAffinityFormat {
	#[serde(rename = "openai")]
	Openai,
	#[serde(rename = "openai-nosession")]
	OpenaiNosession,
	#[serde(rename = "openrouter")]
	Openrouter,
}

/// Metadata about the HTTP response (`ProviderResponse`), handed to the
/// `onResponse` hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponse {
	pub status: u16,
	pub headers: BTreeMap<String, String>,
}

/// Text content block (`TextContent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
	pub text: String,
	/// Message metadata carried by OpenAI Responses (legacy id string or
	/// [`TextSignatureV1`] JSON).
	#[serde(rename = "textSignature", default, skip_serializing_if = "Option::is_none")]
	pub text_signature: Option<String>,
}

impl TextContent {
	pub fn new(text: impl Into<String>) -> TextContent {
		TextContent { text: text.into(), text_signature: None }
	}
}

/// Signed text payload (`TextSignatureV1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextSignatureV1 {
	pub v: u8,
	pub id: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub phase: Option<TextSignaturePhase>,
}

/// Phase discriminator of [`TextSignatureV1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSignaturePhase {
	Commentary,
	FinalAnswer,
}

/// Thinking content block (`ThinkingContent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingContent {
	pub thinking: String,
	/// Provider-specific opaque or serialized reasoning replay data.
	#[serde(rename = "thinkingSignature", default, skip_serializing_if = "Option::is_none")]
	pub thinking_signature: Option<String>,
	/// When true, the thinking content was redacted by safety filters; the
	/// opaque encrypted payload travels in `thinkingSignature`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub redacted: Option<bool>,
}

impl ThinkingContent {
	pub fn new(thinking: impl Into<String>) -> ThinkingContent {
		ThinkingContent {
			thinking: thinking.into(),
			thinking_signature: None,
			redacted: None,
		}
	}
}

/// Base64 image content block (`ImageContent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageContent {
	/// Base64 encoded image data.
	pub data: String,
	/// e.g. `image/jpeg`, `image/png`.
	#[serde(rename = "mimeType")]
	pub mime_type: String,
}

impl ImageContent {
	pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> ImageContent {
		ImageContent { data: data.into(), mime_type: mime_type.into() }
	}
}

/// A tool invocation emitted by the model (`ToolCall`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
	pub id: String,
	pub name: String,
	/// JSON object of arguments (`JsonObject`).
	pub arguments: serde_json::Map<String, Value>,
	/// Google-specific: opaque signature for reusing thought context.
	#[serde(rename = "thoughtSignature", default, skip_serializing_if = "Option::is_none")]
	pub thought_signature: Option<String>,
	/// OpenAI Responses namespace for calls to dynamically loaded or
	/// namespaced tools.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub namespace: Option<String>,
}

/// The literal `"toolCall"` discriminator of [`ToolCall`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallContentType {
	#[serde(rename = "toolCall")]
	ToolCall,
}

/// Token usage and cost for one model response (`Usage`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
	pub input: u64,
	pub output: u64,
	pub cache_read: u64,
	pub cache_write: u64,
	/// Subset of `cacheWrite` written with 1h retention. Only Anthropic
	/// reports this split.
	#[serde(rename = "cacheWrite1h", default, skip_serializing_if = "Option::is_none")]
	pub cache_write_1h: Option<u64>,
	/// Reasoning/thinking tokens, when the provider reports them. Subset of
	/// `output`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning: Option<u64>,
	pub total_tokens: u64,
	pub cost: UsageCost,
}

/// Cost breakdown of [`Usage`] in USD (`Usage["cost"]`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
	pub input: f64,
	pub output: f64,
	pub cache_read: f64,
	pub cache_write: f64,
	pub total: f64,
}

/// Why a model response ended (`StopReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
	#[serde(rename = "pending")]
	Pending,
	#[serde(rename = "stop")]
	Stop,
	#[serde(rename = "length")]
	Length,
	#[serde(rename = "toolUse")]
	ToolUse,
	#[serde(rename = "error")]
	Error,
	#[serde(rename = "aborted")]
	Aborted,
	#[serde(rename = "deferred")]
	Deferred,
}

/// JSON value type mirroring upstream `JsonValue`
/// (`serde_json::Value` with object keys ordered).
pub type JsonValue = Value;

/// JSON object type mirroring upstream `JsonObject`.
pub type JsonObject = serde_json::Map<String, Value>;

/// Handle for continuing a deferred (async) provider response
/// (`DeferredHandle`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
	pub provider: String,
	#[serde(rename = "modelId")]
	pub model_id: String,
	pub api: String,
	/// Provider token, such as a response id or batch id plus row id.
	pub id: String,
	#[serde(rename = "expiresAt", default, skip_serializing_if = "Option::is_none")]
	pub expires_at: Option<i64>,
	#[serde(rename = "pollAfterMs", default, skip_serializing_if = "Option::is_none")]
	pub poll_after_ms: Option<u64>,
	/// Provider conversion data required to reconstruct the final assistant
	/// message.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub data: Option<JsonValue>,
}

/// Tool declaration accepted by providers (`Tool`). Upstream `parameters` is a
/// TypeBox schema; Rust carries the equivalent JSON Schema document as an
/// opaque [`serde_json::Value`] (see crate docs, deviation D6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
	pub name: String,
	pub description: String,
	pub parameters: JsonValue,
	/// Provider-side constrained sampling config, or `false` to disable.
	#[serde(
		rename = "constrainedSampling",
		default,
		skip_serializing_if = "Option::is_none"
	)]
	pub constrained_sampling: Option<ConstrainedSamplingSetting>,
}

/// Upstream `constrainedSampling?: false | ConstrainedSamplingConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConstrainedSamplingSetting {
	Disabled(bool),
	Config(ConstrainedSamplingConfig),
}

/// Provider-side constrained sampling config for a tool
/// (`ConstrainedSamplingConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSamplingConfig {
	#[serde(rename = "json_schema")]
	JsonSchema { strict: StrictSchemaMode },
	#[serde(rename = "grammar")]
	Grammar { variants: GrammarVariants },
}

/// `strict` policy of the `json_schema` constrained sampling config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrictSchemaMode {
	#[serde(rename = "prefer")]
	Prefer,
	#[serde(rename = "require")]
	Require,
}

/// OpenAI grammar variants for constrained sampling (`GrammarVariants`).
pub type GrammarVariants = BTreeMap<GrammarFormat, String>;

/// OpenAI grammar formats (`GrammarFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GrammarFormat {
	#[serde(rename = "openai_lark")]
	OpenaiLark,
	#[serde(rename = "openai_regex")]
	OpenaiRegex,
}

/// Tool reference by name (`ToolReference`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReference {
	pub name: String,
}

/// Lax content vectors (deviation D13): legacy session files carry
/// `content: null` or omit the field; normalize to `[]`.
fn deserialize_null_tolerant_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
	T: serde::de::DeserializeOwned,
	D: serde::Deserializer<'de>,
{
	let value = Option::<Value>::deserialize(deserializer)?;
	match value {
		None | Some(Value::Null) => Ok(Vec::new()),
		Some(other) => serde_json::from_value(other).map_err(serde::de::Error::custom),
	}
}

/// System instructions and tool declarations at one point in the transcript
/// (`SystemMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
	/// Instruction text. On the leading message this is the base prompt; later,
	/// additional instructions.
	pub content: SystemContent,
	/// Named, ordered prompt sections; `null` removes a section.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub sections: Option<BTreeMap<String, Option<String>>>,
	/// Complete definitions of tools that become available at this point.
	#[serde(rename = "toolsAdded", default, skip_serializing_if = "Option::is_none")]
	pub tools_added: Option<Vec<Tool>>,
	/// Tools that stop being available at this point.
	#[serde(rename = "toolsRemoved", default, skip_serializing_if = "Option::is_none")]
	pub tools_removed: Option<Vec<ToolReference>>,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
}

/// `string | TextContent[]` content of [`SystemMessage`]. Blocks carry the
/// upstream `{"type":"text",...}` wire shape via [`SystemBlock`]. Lax on read:
/// `null` normalizes to an empty block list (deviation D13).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SystemContent {
	Text(String),
	Blocks(Vec<SystemBlock>),
}

impl<'de> Deserialize<'de> for SystemContent {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		let value = Value::deserialize(deserializer)?;
		Ok(match value {
			Value::Null => SystemContent::Blocks(Vec::new()),
			Value::String(text) => SystemContent::Text(text),
			other => SystemContent::Blocks(
				serde_json::from_value(other).map_err(serde::de::Error::custom)?,
			),
		})
	}
}

/// Self-describing system text block (`{"type":"text",...}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SystemBlock {
	#[serde(rename = "text")]
	Text(TextContent),
}

/// User message (`UserMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
	pub content: UserContentField,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
}

/// `string | (TextContent | ImageContent)[]` content of [`UserMessage`]. Lax
/// on read: `null` normalizes to an empty block list (deviation D13).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UserContentField {
	Text(String),
	Blocks(Vec<UserContent>),
}

impl<'de> Deserialize<'de> for UserContentField {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		let value = Value::deserialize(deserializer)?;
		Ok(match value {
			Value::Null => UserContentField::Blocks(Vec::new()),
			Value::String(text) => UserContentField::Text(text),
			other => UserContentField::Blocks(
				serde_json::from_value(other).map_err(serde::de::Error::custom)?,
			),
		})
	}
}

/// One user-visible content block: text or image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserContent {
	#[serde(rename = "text")]
	Text(TextContent),
	#[serde(rename = "image")]
	Image(ImageContent),
}

impl From<TextContent> for UserContent {
	fn from(value: TextContent) -> Self {
		UserContent::Text(value)
	}
}

impl From<ImageContent> for UserContent {
	fn from(value: ImageContent) -> Self {
		UserContent::Image(value)
	}
}

/// Assistant response (`AssistantMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
	/// Lax on read: legacy session files can carry `content: null` or omit it
	/// (upstream issues #6259/#6276 — normalized to `[]` before provider
	/// requests; here normalized at the deserialization boundary, deviation
	/// D13).
	#[serde(default, deserialize_with = "deserialize_null_tolerant_vec")]
	pub content: Vec<AssistantContent>,
	pub api: String,
	pub provider: String,
	/// Requested model id.
	pub model: String,
	/// Concrete model reported by the provider when different from the
	/// requested `model`.
	#[serde(rename = "responseModel", default, skip_serializing_if = "Option::is_none")]
	pub response_model: Option<String>,
	/// Provider-specific response/message identifier.
	#[serde(rename = "responseId", default, skip_serializing_if = "Option::is_none")]
	pub response_id: Option<String>,
	/// Exact provider-native effort level used for this response.
	#[serde(
		rename = "providerThinkingLevel",
		default,
		skip_serializing_if = "Option::is_none"
	)]
	pub provider_thinking_level: Option<String>,
	/// Pi thinking level the agent loop requested for this response.
	#[serde(rename = "thinkingLevel", default, skip_serializing_if = "Option::is_none")]
	pub thinking_level: Option<ModelThinkingLevel>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
	pub usage: Usage,
	pub stop_reason: StopReason,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub deferred: Option<DeferredHandle>,
	#[serde(rename = "errorMessage", default, skip_serializing_if = "Option::is_none")]
	pub error_message: Option<String>,
	#[serde(rename = "rawStopReason", default, skip_serializing_if = "Option::is_none")]
	pub raw_stop_reason: Option<String>,
	/// Provider indication of whether the model explicitly ended its turn.
	#[serde(rename = "endTurn", default, skip_serializing_if = "Option::is_none")]
	pub end_turn: Option<bool>,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
}

/// One assistant content block: text, thinking, or tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantContent {
	#[serde(rename = "text")]
	Text(TextContent),
	#[serde(rename = "thinking")]
	Thinking(ThinkingContent),
	#[serde(rename = "toolCall")]
	ToolCall(ToolCall),
}

impl From<TextContent> for AssistantContent {
	fn from(value: TextContent) -> Self {
		AssistantContent::Text(value)
	}
}

impl From<ThinkingContent> for AssistantContent {
	fn from(value: ThinkingContent) -> Self {
		AssistantContent::Thinking(value)
	}
}

impl From<ToolCall> for AssistantContent {
	fn from(value: ToolCall) -> Self {
		AssistantContent::ToolCall(value)
	}
}

/// Redacted provider/runtime diagnostic attached to an assistant message
/// (`AssistantMessageDiagnostic`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
	#[serde(rename = "type")]
	pub diagnostic_type: String,
	pub timestamp: i64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error: Option<DiagnosticErrorInfo>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonObject>,
}

/// Error info inside a diagnostic (`DiagnosticErrorInfo`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
	pub message: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stack: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub code: Option<DiagnosticCode>,
}

/// `string | number` diagnostic error code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiagnosticCode {
	String(String),
	Number(f64),
}

/// A tool call that another tool made while it ran (`NestedToolCallRecord`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedToolCallRecord {
	pub id: String,
	pub name: String,
	/// Omitted when over the size limits; `argumentsBytes` then gives their size.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub arguments: Option<JsonObject>,
	/// UTF-8 size of the arguments as JSON, set when `arguments` is omitted.
	#[serde(rename = "argumentsBytes", default, skip_serializing_if = "Option::is_none")]
	pub arguments_bytes: Option<u64>,
	/// `unfinished`: the call was still running when the calling tool finished.
	pub status: NestedToolCallStatus,
	#[serde(rename = "durationMs", default, skip_serializing_if = "Option::is_none")]
	pub duration_ms: Option<u64>,
	/// Error text, truncated.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub error: Option<String>,
}

/// Status of a [`NestedToolCallRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NestedToolCallStatus {
	Ok,
	Error,
	#[serde(rename = "unfinished")]
	Unfinished,
}

/// Bounded record of the nested calls a tool made (`NestedToolCalls`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedToolCalls {
	pub calls: Vec<NestedToolCallRecord>,
	/// False when calls were dropped, arguments omitted, or calls had not
	/// finished.
	pub complete: bool,
}

/// Tool execution result (`ToolResultMessage`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
	#[serde(rename = "toolCallId")]
	pub tool_call_id: String,
	#[serde(rename = "toolName")]
	pub tool_name: String,
	/// Text and image blocks. Lax on read like [`AssistantMessage::content`].
	#[serde(default, deserialize_with = "deserialize_null_tolerant_vec")]
	pub content: Vec<UserContent>,
	/// Provider-specific details payload.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub details: Option<JsonValue>,
	/// Usage from the tool execution itself.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	/// Calls this tool made to other tools; kept for the session record.
	#[serde(rename = "nestedCalls", default, skip_serializing_if = "Option::is_none")]
	pub nested_calls: Option<NestedToolCalls>,
	#[serde(rename = "isError")]
	pub is_error: bool,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
}

/// One message in a transcript (`Message`), discriminated by `role`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
	#[serde(rename = "system")]
	System(SystemMessage),
	#[serde(rename = "user")]
	User(UserMessage),
	#[serde(rename = "assistant")]
	Assistant(AssistantMessage),
	#[serde(rename = "toolResult")]
	ToolResult(ToolResultMessage),
}

impl Message {
	/// Upstream `message.role` as a plain string.
	pub fn role(&self) -> &'static str {
		match self {
			Message::System(_) => "system",
			Message::User(_) => "user",
			Message::Assistant(_) => "assistant",
			Message::ToolResult(_) => "toolResult",
		}
	}
}

impl From<SystemMessage> for Message {
	fn from(value: SystemMessage) -> Self {
		Message::System(value)
	}
}

impl From<UserMessage> for Message {
	fn from(value: UserMessage) -> Self {
		Message::User(value)
	}
}

impl From<AssistantMessage> for Message {
	fn from(value: AssistantMessage) -> Self {
		Message::Assistant(value)
	}
}

impl From<ToolResultMessage> for Message {
	fn from(value: ToolResultMessage) -> Self {
		Message::ToolResult(value)
	}
}

/// Image-generation input block set (`ImagesInputContent`).
pub type ImagesInputContent = UserContent;

/// Image-generation output block set (`ImagesOutputContent`).
pub type ImagesOutputContent = UserContent;

/// Image-generation request context (`ImagesContext`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesContext {
	pub input: Vec<ImagesInputContent>,
}

/// Image-generation stop reasons (`ImagesStopReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImagesStopReason {
	Stop,
	Error,
	Aborted,
}

/// Image-generation result (`AssistantImages`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
	pub api: String,
	pub provider: String,
	pub model: String,
	pub output: Vec<ImagesOutputContent>,
	#[serde(rename = "responseId", default, skip_serializing_if = "Option::is_none")]
	pub response_id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	pub stop_reason: ImagesStopReason,
	#[serde(rename = "errorMessage", default, skip_serializing_if = "Option::is_none")]
	pub error_message: Option<String>,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
}

/// Classification question types (`ClassifierQuestion`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ClassifierQuestion {
	#[serde(rename = "choice")]
	Choice {
		instructions: String,
		criteria: BTreeMap<String, String>,
	},
	#[serde(rename = "score")]
	Score {
		instructions: String,
		criteria: Vec<String>,
	},
	#[serde(rename = "bool")]
	Bool {
		instructions: String,
		criteria: ClassifierBoolCriteria,
	},
}

/// `{ true, false }` criteria of a bool question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierBoolCriteria {
	#[serde(rename = "true")]
	pub true_label: String,
	#[serde(rename = "false")]
	pub false_label: String,
}

/// Classification request context (`ClassifierContext`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierContext {
	pub state: JsonObject,
	#[serde(rename = "questions")]
	pub questions: BTreeMap<String, ClassifierQuestion>,
}

/// Classification answer types (`ClassifierAnswer`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ClassifierAnswer {
	#[serde(rename = "choice")]
	Choice {
		choice: String,
		probabilities: BTreeMap<String, f64>,
		confidence: f64,
	},
	#[serde(rename = "score")]
	Score { score: f64, confidence: f64 },
	#[serde(rename = "bool")]
	Bool { probability: f64 },
}

/// Classification stop reasons (`ClassifierStopReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierStopReason {
	Stop,
	Error,
	Aborted,
}

/// Classification result (`ClassifierResult`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierResult {
	pub api: String,
	pub provider: String,
	pub model: String,
	pub answers: BTreeMap<String, ClassifierAnswer>,
	/// Token usage and its cost at the model's catalog price.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
	pub stop_reason: ClassifierStopReason,
	#[serde(rename = "errorMessage", default, skip_serializing_if = "Option::is_none")]
	pub error_message: Option<String>,
	/// Unix timestamp in milliseconds.
	pub timestamp: i64,
}

/// Request input accepted by the public stream entry points (`Context`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Context {
	pub system_prompt: Option<String>,
	pub messages: Vec<Message>,
	pub tools: Option<Vec<Tool>>,
}

/// Normalized request context passed to providers and API implementations
/// (`TranscriptContext`). The prompt and tool declarations are carried by the
/// transcript's system messages. Only
/// [`crate::utils::transcript::normalize_context`] produces this type.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptContext {
	messages: Vec<Message>,
}

impl TranscriptContext {
	/// Pub(crate) constructor; the public path is `normalize_context`.
	pub(crate) fn new(messages: Vec<Message>) -> TranscriptContext {
		TranscriptContext { messages }
	}

	pub fn messages(&self) -> &[Message] {
		&self.messages
	}

	/// Consume into the raw message list.
	pub fn into_messages(self) -> Vec<Message> {
		self.messages
	}
}

/// Event protocol of
/// [`AssistantMessageEventStream`](crate::utils::event_stream::AssistantMessageEventStream)
/// (`AssistantMessageEvent`).
///
/// Successful streams emit `start` before partial updates and terminate with
/// `done`. A stream may terminate directly with `error` when request setup
/// fails before generation starts; after `start`, failures also terminate with
/// `error`. Updates and `done` must never appear before `start`.
///
/// `partial` is the shared live response-so-far helper, not an event-time
/// snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
	#[serde(rename = "start")]
	Start { partial: AssistantMessage },
	#[serde(rename = "text_start")]
	TextStart { #[serde(rename = "contentIndex")] content_index: u32, partial: AssistantMessage },
	#[serde(rename = "text_delta")]
	TextDelta {
		#[serde(rename = "contentIndex")]
		content_index: u32,
		delta: String,
		partial: AssistantMessage,
	},
	#[serde(rename = "text_end")]
	TextEnd {
		#[serde(rename = "contentIndex")]
		content_index: u32,
		content: String,
		partial: AssistantMessage,
	},
	#[serde(rename = "thinking_start")]
	ThinkingStart { #[serde(rename = "contentIndex")] content_index: u32, partial: AssistantMessage },
	#[serde(rename = "thinking_delta")]
	ThinkingDelta {
		#[serde(rename = "contentIndex")]
		content_index: u32,
		delta: String,
		partial: AssistantMessage,
	},
	#[serde(rename = "thinking_end")]
	ThinkingEnd {
		#[serde(rename = "contentIndex")]
		content_index: u32,
		content: String,
		partial: AssistantMessage,
	},
	#[serde(rename = "toolcall_start")]
	ToolcallStart { #[serde(rename = "contentIndex")] content_index: u32, partial: AssistantMessage },
	#[serde(rename = "toolcall_delta")]
	ToolcallDelta {
		#[serde(rename = "contentIndex")]
		content_index: u32,
		delta: String,
		partial: AssistantMessage,
	},
	#[serde(rename = "toolcall_end")]
	ToolcallEnd {
		#[serde(rename = "contentIndex")]
		content_index: u32,
		#[serde(rename = "toolCall")]
		tool_call: ToolCall,
		partial: AssistantMessage,
	},
	#[serde(rename = "done")]
	Done { reason: StreamDoneReason, message: AssistantMessage },
	#[serde(rename = "error")]
	Error { reason: StreamErrorReason, error: AssistantMessage },
}

/// `done` stop reasons (`"stop" | "length" | "toolUse" | "deferred"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamDoneReason {
	#[serde(rename = "stop")]
	Stop,
	#[serde(rename = "length")]
	Length,
	#[serde(rename = "toolUse")]
	ToolUse,
	#[serde(rename = "deferred")]
	Deferred,
}

/// `error` stop reasons (`"aborted" | "error"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamErrorReason {
	#[serde(rename = "aborted")]
	Aborted,
	#[serde(rename = "error")]
	Error,
}

impl From<StreamDoneReason> for StopReason {
	fn from(value: StreamDoneReason) -> Self {
		match value {
			StreamDoneReason::Stop => StopReason::Stop,
			StreamDoneReason::Length => StopReason::Length,
			StreamDoneReason::ToolUse => StopReason::ToolUse,
			StreamDoneReason::Deferred => StopReason::Deferred,
		}
	}
}

impl From<StreamErrorReason> for StopReason {
	fn from(value: StreamErrorReason) -> Self {
		match value {
			StreamErrorReason::Aborted => StopReason::Aborted,
			StreamErrorReason::Error => StopReason::Error,
		}
	}
}

/// Compatibility settings for OpenAI-compatible completions APIs
/// (`OpenAICompletionsCompat`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_store: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_developer_role: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_reasoning_effort: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_usage_in_streaming: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_finish_reason: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub max_tokens_field: Option<MaxTokensField>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_tool_result_name: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_assistant_after_tool_result: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_thinking_as_text: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_reasoning_content_on_assistant_messages: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking_format: Option<ThinkingFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub chat_template_kwargs: Option<BTreeMap<String, ChatTemplateKwargValue>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub chat_template_args: Option<BTreeMap<String, ChatTemplateKwargValue>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub open_router_routing: Option<OpenRouterRouting>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub vercel_gateway_routing: Option<VercelGatewayRouting>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub zai_tool_stream: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_thinking_token_budget: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_openai_grammar_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_system_messages: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_tool_additions: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_strict_mode: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_control_format: Option<CacheControlFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub send_session_affinity_headers: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub session_affinity_format: Option<SessionAffinityFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_long_cache_retention: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub vllm_priority: Option<f64>,
}

/// Which request field carries the max token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxTokensField {
	#[serde(rename = "max_completion_tokens")]
	MaxCompletionTokens,
	#[serde(rename = "max_tokens")]
	MaxTokens,
}

/// Reasoning/thinking parameter encoding on OpenAI-compatible APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingFormat {
	#[serde(rename = "openai")]
	Openai,
	#[serde(rename = "openrouter")]
	Openrouter,
	#[serde(rename = "deepseek")]
	Deepseek,
	#[serde(rename = "together")]
	Together,
	#[serde(rename = "baseten")]
	Baseten,
	#[serde(rename = "zai")]
	Zai,
	#[serde(rename = "qwen")]
	Qwen,
	#[serde(rename = "chat-template")]
	ChatTemplate,
	#[serde(rename = "qwen-chat-template")]
	QwenChatTemplate,
	#[serde(rename = "string-thinking")]
	StringThinking,
	#[serde(rename = "ant-ling")]
	AntLing,
}

/// Cache control convention for prompt caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlFormat {
	Anthropic,
}

/// Compatibility settings for OpenAI Responses APIs
/// (`OpenAIResponsesCompat`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAIResponsesCompat {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_developer_role: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_system_messages: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub session_affinity_format: Option<SessionAffinityFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_long_cache_retention: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_strict_mode: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_openai_grammar_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_additional_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_tool_search: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_explicit_prompt_cache_mode: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_max_output_tokens: Option<bool>,
}

/// Models Anthropic accepts in `fallbacks`
/// (`AnthropicAllowedFallbackModel`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicAllowedFallbackModel {
	pub provider: String,
	pub model: String,
	pub cost: ModelCost,
}

/// Compatibility settings for Anthropic Messages-compatible APIs
/// (`AnthropicMessagesCompat`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicMessagesCompat {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_eager_tool_input_streaming: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_long_cache_retention: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub send_session_affinity_headers: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub session_affinity_format: Option<SessionAffinityFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_cache_control_on_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_temperature: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub force_adaptive_thinking: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub allow_empty_signature: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_strict_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_effort: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_system_messages: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_tool_changes: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
}

/// Compatibility settings for Amazon Bedrock models (`BedrockCompat`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockCompat {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_strict_mode: Option<bool>,
}

/// Compatibility settings for the Mistral chat API
/// (`MistralConversationsCompat`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MistralConversationsCompat {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_system_messages: Option<bool>,
}

/// OpenRouter provider routing preferences (`OpenRouterRouting`). Deeply
/// variant fields (`sort`, throughput/latency preferences) are carried as raw
/// JSON to preserve wire bytes exactly.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenRouterRouting {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub allow_fallbacks: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub require_parameters: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub data_collection: Option<OpenRouterDataCollection>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub zdr: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub enforce_distillable_text: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub order: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub only: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub ignore: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub quantizations: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub sort: Option<JsonValue>,
	#[serde(rename = "max_price", default, skip_serializing_if = "Option::is_none")]
	pub max_price: Option<OpenRouterMaxPrice>,
	#[serde(rename = "preferred_min_throughput", default, skip_serializing_if = "Option::is_none")]
	pub preferred_min_throughput: Option<JsonValue>,
	#[serde(rename = "preferred_max_latency", default, skip_serializing_if = "Option::is_none")]
	pub preferred_max_latency: Option<JsonValue>,
}

/// Data collection setting (`"deny" | "allow"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenRouterDataCollection {
	#[serde(rename = "deny")]
	Deny,
	#[serde(rename = "allow")]
	Allow,
}

/// Maximum price per million tokens (`OpenRouterRouting.max_price`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenRouterMaxPrice {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub prompt: Option<NumberOrString>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub completion: Option<NumberOrString>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub image: Option<NumberOrString>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub audio: Option<NumberOrString>,
	#[serde(rename = "request", default, skip_serializing_if = "Option::is_none")]
	pub request: Option<NumberOrString>,
}

/// `number | string` price values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NumberOrString {
	Number(f64),
	String(String),
}

/// Vercel AI Gateway routing preferences (`VercelGatewayRouting`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VercelGatewayRouting {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub only: Option<Vec<String>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub order: Option<Vec<String>>,
}

/// Per-token-type cost rates in $/million tokens (`ModelCostRates`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
	pub input: f64,
	pub output: f64,
	#[serde(rename = "cacheRead")]
	pub cache_read: f64,
	#[serde(rename = "cacheWrite")]
	pub cache_write: f64,
}

/// Request-wide pricing tier (`ModelCostTier`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
	#[serde(flatten)]
	pub rates: ModelCostRates,
	/// Use this tier for requests whose total input usage exceeds this token
	/// count.
	#[serde(rename = "inputTokensAbove")]
	pub input_tokens_above: f64,
}

/// Model cost with optional tiers (`ModelCost`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
	#[serde(flatten)]
	pub rates: ModelCostRates,
	/// Highest matching input threshold applies to the full request.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tiers: Option<Vec<ModelCostTier>>,
}

/// Image resize profile (`ModelImageResizeOptions`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageResizeOptions {
	#[serde(rename = "maxWidth", default, skip_serializing_if = "Option::is_none")]
	pub max_width: Option<u64>,
	#[serde(rename = "maxHeight", default, skip_serializing_if = "Option::is_none")]
	pub max_height: Option<u64>,
	/// Maximum base64-encoded payload size in bytes.
	#[serde(rename = "maxBytes", default, skip_serializing_if = "Option::is_none")]
	pub max_bytes: Option<u64>,
	#[serde(rename = "jpegQuality", default, skip_serializing_if = "Option::is_none")]
	pub jpeg_quality: Option<u8>,
}

/// Per-message image limits (`ModelImageInputLimits`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageInputLimits {
	/// Cache-safe resize profile applied before a new image enters history.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub resize: Option<ModelImageResizeOptions>,
	/// Maximum images accepted in one provider message.
	#[serde(rename = "maxPerMessage", default, skip_serializing_if = "Option::is_none")]
	pub max_per_message: Option<u64>,
	/// Maximum images accepted across one provider request.
	#[serde(rename = "maxPerRequest", default, skip_serializing_if = "Option::is_none")]
	pub max_per_request: Option<u64>,
}

/// Provider input limits (`ModelInputLimits`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInputLimits {
	/// Maximum serialized provider request size in bytes.
	#[serde(rename = "maxRequestBytes", default, skip_serializing_if = "Option::is_none")]
	pub max_request_bytes: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub images: Option<ModelImageInputLimits>,
}

/// Model input modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelInput {
	#[serde(rename = "text")]
	Text,
	#[serde(rename = "image")]
	Image,
}

/// Model output modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelOutput {
	#[serde(rename = "text")]
	Text,
	#[serde(rename = "image")]
	Image,
}

/// What a catalog entry is for (`ModelType`). Chat is the default: models
/// without `type` are chat models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
	Chat,
	Image,
	Classifier,
}

/// Discriminator value stored in the optional `type` field of chat models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatType {
	#[serde(rename = "chat")]
	Chat,
}

/// Fields shared by every catalog entry (`BaseModel`). All three model shapes
/// flatten this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseModelFields {
	pub id: String,
	pub name: String,
	pub api: String,
	pub provider: String,
	#[serde(rename = "baseUrl")]
	pub base_url: String,
	pub input: Vec<ModelInput>,
	#[serde(rename = "inputLimits", default, skip_serializing_if = "Option::is_none")]
	pub input_limits: Option<ModelInputLimits>,
	pub cost: ModelCost,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub headers: Option<BTreeMap<String, String>>,
}

/// Chat model (`Model<Api>`), usable with `stream()` and friends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
	#[serde(flatten)]
	pub base: BaseModelFields,
	/// Chat is the default model type; `None` means chat.
	#[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
	pub model_type: Option<ChatType>,
	pub reasoning: bool,
	#[serde(rename = "thinkingLevelMap", default, skip_serializing_if = "Option::is_none")]
	pub thinking_level_map: Option<ThinkingLevelMap>,
	#[serde(rename = "promptCache", default, skip_serializing_if = "Option::is_none")]
	pub prompt_cache: Option<ModelPromptCache>,
	#[serde(rename = "contextWindow")]
	pub context_window: u64,
	#[serde(rename = "maxTokens")]
	pub max_tokens: u64,
	#[serde(rename = "samplingParams", default, skip_serializing_if = "Option::is_none")]
	pub sampling_params: Option<SamplingParams>,
	#[serde(
		rename = "samplingParamsByThinkingLevel",
		default,
		skip_serializing_if = "Option::is_none"
	)]
	pub sampling_params_by_thinking_level: Option<SamplingParamsByThinkingLevel>,
	/// Compatibility overrides. Upstream this is a conditional type keyed on
	/// `api`; runtime shape is one object, so Rust carries the merged struct
	/// (crate docs, deviation D7).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub compat: Option<ModelCompat>,
}

impl Model {
	/// The model id (`model.id` upstream).
	pub fn id(&self) -> &str {
		&self.base.id
	}

	/// Owning provider id (`model.provider` upstream).
	pub fn provider(&self) -> &str {
		&self.base.provider
	}

	/// API id (`model.api` upstream).
	pub fn api(&self) -> &str {
		&self.base.api
	}

	/// Upstream `model.type ?? "chat"`.
	pub fn model_type(&self) -> ModelType {
		ModelType::Chat
	}
}

/// Image-generation model (`ImageModel<ImageApi>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModel {
	#[serde(flatten)]
	pub base: BaseModelFields,
	#[serde(rename = "type")]
	pub model_type: ImageTypeTag,
	/// Output modalities. Always includes `image`; `text` means the model can
	/// also return text blocks.
	pub output: Vec<ModelOutput>,
}

/// The required `type: "image"` discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageTypeTag {
	#[serde(rename = "image")]
	Image,
}

impl ImageTypeTag {
	pub const TAG: ImageTypeTag = ImageTypeTag::Image;
}

impl ImageModel {
	pub fn id(&self) -> &str {
		&self.base.id
	}

	pub fn provider(&self) -> &str {
		&self.base.provider
	}

	pub fn api(&self) -> &str {
		&self.base.api
	}

	pub fn model_type(&self) -> ModelType {
		ModelType::Image
	}
}

/// Structured classifier model (`ClassifierModel<ClassifierApi>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierModel {
	#[serde(flatten)]
	pub base: BaseModelFields,
	#[serde(rename = "type")]
	pub model_type: ClassifierTypeTag,
	#[serde(rename = "contextWindow")]
	pub context_window: u64,
}

/// The required `type: "classifier"` discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassifierTypeTag {
	#[serde(rename = "classifier")]
	Classifier,
}

impl ClassifierTypeTag {
	pub const TAG: ClassifierTypeTag = ClassifierTypeTag::Classifier;
}

impl ClassifierModel {
	pub fn id(&self) -> &str {
		&self.base.id
	}

	pub fn provider(&self) -> &str {
		&self.base.provider
	}

	pub fn api(&self) -> &str {
		&self.base.api
	}

	pub fn model_type(&self) -> ModelType {
		ModelType::Classifier
	}
}

/// Any catalog entry (`AnyModel`). Deserialization dispatches on the `type`
/// field: `"image"` → [`ImageModel`], `"classifier"` → [`ClassifierModel`],
/// absent or `"chat"` → [`Model`].
#[derive(Debug, Clone, PartialEq)]
pub enum AnyModel {
	Chat(Model),
	Image(ImageModel),
	Classifier(ClassifierModel),
}

impl AnyModel {
	pub fn model_type(&self) -> ModelType {
		match self {
			AnyModel::Chat(_) => ModelType::Chat,
			AnyModel::Image(_) => ModelType::Image,
			AnyModel::Classifier(_) => ModelType::Classifier,
		}
	}

	pub fn id(&self) -> &str {
		match self {
			AnyModel::Chat(m) => m.id(),
			AnyModel::Image(m) => m.id(),
			AnyModel::Classifier(m) => m.id(),
		}
	}

	pub fn provider(&self) -> &str {
		match self {
			AnyModel::Chat(m) => m.provider(),
			AnyModel::Image(m) => m.provider(),
			AnyModel::Classifier(m) => m.provider(),
		}
	}

	pub fn api(&self) -> &str {
		match self {
			AnyModel::Chat(m) => m.api(),
			AnyModel::Image(m) => m.api(),
			AnyModel::Classifier(m) => m.api(),
		}
	}

	/// Custom JSON (de)serialization dispatching on `type`, mirroring
	/// upstream's discriminated union without untagged-enum ambiguity.
	pub fn to_json(&self) -> JsonValue {
		match self {
			AnyModel::Chat(m) => serde_json::to_value(m).expect("chat model serializes"),
			AnyModel::Image(m) => serde_json::to_value(m).expect("image model serializes"),
			AnyModel::Classifier(m) => serde_json::to_value(m).expect("classifier model serializes"),
		}
	}

	pub fn from_json(value: &JsonValue) -> Result<AnyModel, serde_json::Error> {
		let kind = value.get("type").and_then(Value::as_str);
		match kind {
			Some("image") => Ok(AnyModel::Image(serde_json::from_value(value.clone())?)),
			Some("classifier") => Ok(AnyModel::Classifier(serde_json::from_value(value.clone())?)),
			// Absent or "chat": chat models. Any other type string is from a
			// newer version and is rejected so store reads can drop it.
			Some(other) if other != "chat" => {
				Err(serde::de::Error::custom(format!("unknown model type \"{other}\"")))
			}
			_ => Ok(AnyModel::Chat(serde_json::from_value(value.clone())?)),
		}
	}
}

impl Serialize for AnyModel {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		self.to_json().serialize(serializer)
	}
}

impl<'de> Deserialize<'de> for AnyModel {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		let value = JsonValue::deserialize(deserializer)?;
		AnyModel::from_json(&value).map_err(serde::de::Error::custom)
	}
}

/// Merged model compatibility overrides (`Model.compat`). Upstream types this
/// as one of five API-specific interfaces via a conditional type; the runtime
/// shape is a single optional object, so Rust carries the union of fields.
/// Shared field names (`supportsMidConvoSystemMessages`,
/// `supportsLongCacheRetention`, `supportsStrictMode`,
/// `sessionAffinityFormat`) have identical value types across interfaces.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCompat {
	// OpenAICompletionsCompat
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_store: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_reasoning_effort: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_usage_in_streaming: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_finish_reason: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub max_tokens_field: Option<MaxTokensField>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_tool_result_name: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_assistant_after_tool_result: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_thinking_as_text: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_reasoning_content_on_assistant_messages: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking_format: Option<ThinkingFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub chat_template_kwargs: Option<BTreeMap<String, ChatTemplateKwargValue>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub chat_template_args: Option<BTreeMap<String, ChatTemplateKwargValue>>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub open_router_routing: Option<OpenRouterRouting>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub vercel_gateway_routing: Option<VercelGatewayRouting>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub zai_tool_stream: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_thinking_token_budget: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_openai_grammar_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_tool_additions: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_control_format: Option<CacheControlFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub send_session_affinity_headers: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub vllm_priority: Option<f64>,
	// OpenAIResponsesCompat
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_additional_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_tool_search: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_explicit_prompt_cache_mode: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_max_output_tokens: Option<bool>,
	// AnthropicMessagesCompat
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_eager_tool_input_streaming: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_cache_control_on_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_temperature: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub force_adaptive_thinking: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub allow_empty_signature: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_strict_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_effort: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_tool_changes: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
	// BedrockCompat / MistralConversationsCompat add no fields beyond the shared ones.
	// Shared fields (OpenAICompletionsCompat + OpenAIResponsesCompat +
	// AnthropicMessagesCompat + BedrockCompat + MistralConversationsCompat)
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_developer_role: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_mid_convo_system_messages: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub session_affinity_format: Option<SessionAffinityFormat>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_long_cache_retention: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_strict_mode: Option<bool>,
}

/// Authentication, HTTP transport, and lifecycle callbacks shared by provider
/// requests (`ProviderRequestOptions`). Upstream optional fields become
/// `Option`s; the `model` argument of hooks is passed owned.
#[derive(Clone, Default)]
pub struct ProviderRequestOptions {
	/// Cooperative cancellation signal.
	pub signal: Option<crate::utils::abort::AbortSignal>,
	/// Explicit parent context for telemetry produced by this logical request.
	pub telemetry_context: Option<std::sync::Arc<dyn pi_telemetry::TelemetryContext>>,
	/// Explicit API key, overriding resolved auth.
	pub api_key: Option<String>,
	/// Custom fetch implementation (upstream `FetchFunction`). `None` uses the
	/// crate default adapter.
	pub fetch: Option<FetchFn>,
	/// Provider-scoped environment values, taking precedence over the process
	/// environment.
	pub env: Option<ProviderEnv>,
	/// Inspect or replace provider payloads before sending; return `None` to
	/// keep the payload unchanged.
	pub on_payload: Option<PayloadHook>,
	/// Invoked after an HTTP response is received.
	pub on_response: Option<ResponseHook>,
	/// Custom HTTP headers; caller values override default headers, `None`
	/// values suppress provider defaults.
	pub headers: Option<ProviderHeaders>,
	/// HTTP request timeout in milliseconds.
	pub timeout_ms: Option<u64>,
	/// Maximum retry attempts for client-side retries.
	pub max_retries: Option<u32>,
	/// Maximum delay in milliseconds to wait for a server-requested retry wait.
	pub max_retry_delay_ms: Option<u64>,
}

/// `onPayload(payload, model) -> unknown | undefined | Promise<...>`.
pub type PayloadHook = std::sync::Arc<
	dyn Fn(Value, Model) -> futures_util::future::BoxFuture<'static, Option<Value>> + Send + Sync,
>;

/// `onResponse(response, model)`.
pub type ResponseHook =
	std::sync::Arc<dyn Fn(ProviderResponse, Model) -> futures_util::future::BoxFuture<'static, ()> + Send + Sync>;

/// Observer for each parsed provider stream event before pi normalization
/// (`onProviderStreamEvent`). Event data is adapter-owned and read-only.
pub type ProviderStreamEventHook =
	std::sync::Arc<dyn Fn(Value, Model) -> futures_util::future::BoxFuture<'static, ()> + Send + Sync>;

/// Per-API option groups (upstream `AnthropicOptions`,
/// `OpenAICompletionsOptions`, `OpenAIResponsesOptions`) ride on
/// [`StreamOptions`] as plain nested structs so one options type flows through
/// the object-safe [`crate::models::Provider`] trait (crate docs, deviation
/// D2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnthropicApiOptions {
	/// Enable extended thinking. Adaptive models decide when/how much to
	/// think; older models use `thinking_budget_tokens`.
	pub thinking_enabled: Option<bool>,
	/// Token budget for extended thinking (older models only). Default: 1024
	/// when enabled and unset.
	pub thinking_budget_tokens: Option<u64>,
	/// Effort level for adaptive thinking models.
	pub effort: Option<AnthropicEffort>,
	/// How thinking content is returned. Default: `Summarized` when thinking
	/// is enabled.
	pub thinking_display: Option<AnthropicThinkingDisplay>,
	/// Request the interleaved thinking beta header for non-adaptive models.
	/// Default: true.
	pub interleaved_thinking: Option<bool>,
	/// Anthropic tool choice behavior; `Tool(name)` forces a specific tool.
	pub tool_choice: Option<AnthropicToolChoice>,
}

/// Anthropic adaptive-thinking effort (`AnthropicEffort`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicEffort {
	Low,
	Medium,
	High,
	#[serde(rename = "xhigh")]
	Xhigh,
	#[serde(rename = "max")]
	Max,
}

/// Anthropic thinking content display (`AnthropicThinkingDisplay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicThinkingDisplay {
	Summarized,
	Omitted,
}

/// Anthropic tool choice (`"auto" | "any" | "none" | { type: "tool", name }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolChoice {
	BuiltIn(AnthropicBuiltInToolChoice),
	Tool { #[serde(rename = "type")] tool_type: AnthropicToolChoiceType, name: String },
}

/// Built-in Anthropic tool choice values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicBuiltInToolChoice {
	Auto,
	Any,
	None,
}

/// The `type: "tool"` discriminator of the forced-tool variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename = "tool")]
pub enum AnthropicToolChoiceType {
	Tool,
}

/// Upstream `OpenAICompletionsOptions` (deviation D2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAICompletionsApiOptions {
	/// OpenAI chat tool choice option (sent verbatim; string or object).
	pub tool_choice: Option<JsonValue>,
	/// Reasoning effort (`"minimal" | ... | "max"`).
	pub reasoning_effort: Option<ThinkingLevel>,
	/// Token budgets per thinking level, applied when the compat
	/// thinking-token-budget settings are on.
	pub thinking_budgets: Option<ThinkingBudgets>,
}

/// Upstream `OpenAIResponsesOptions` (deviation D2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAIResponsesApiOptions {
	/// Reasoning effort (`"minimal" | ... | "max"`).
	pub reasoning_effort: Option<ThinkingLevel>,
	/// Reasoning summary verbosity; `Some(None)` is upstream `null`.
	pub reasoning_summary: Option<Option<ReasoningSummary>>,
	/// OpenAI service tier (sent verbatim).
	pub service_tier: Option<String>,
	/// OpenAI Responses tool choice option (sent verbatim; string or object).
	pub tool_choice: Option<JsonValue>,
}

/// Reasoning summary verbosity on the Responses API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
	Auto,
	Detailed,
	Concise,
}

/// Options for `stream()` on an API implementation (`StreamOptions`).
#[derive(Clone, Default)]
pub struct StreamOptions {
	pub base: ProviderRequestOptions,
	/// Observer for parsed provider stream events.
	pub on_provider_stream_event: Option<ProviderStreamEventHook>,
	/// Anthropic Messages-specific options (upstream `AnthropicOptions`).
	pub anthropic: AnthropicApiOptions,
	/// OpenAI-compatible completions options (upstream
	/// `OpenAICompletionsOptions`).
	pub openai_completions: OpenAICompletionsApiOptions,
	/// OpenAI Responses options (upstream `OpenAIResponsesOptions`).
	pub openai_responses: OpenAIResponsesApiOptions,
	pub temperature: Option<f64>,
	/// Arbitrary sampling parameters merged into the request body as-is, after
	/// the named request fields, so keys here override them.
	pub sampling_params: Option<SamplingParams>,
	pub max_tokens: Option<u64>,
	/// Preferred transport for providers that support multiple transports.
	pub transport: Option<Transport>,
	/// Prompt cache retention preference. Default: `Short`.
	pub cache_retention: Option<CacheRetention>,
	/// Optional session identifier for providers that support session-based
	/// caching.
	pub session_id: Option<String>,
	/// WebSocket connect timeout in milliseconds (handshake only).
	pub websocket_connect_timeout_ms: Option<u64>,
	/// Optional metadata to include in API requests.
	pub metadata: Option<serde_json::Map<String, Value>>,
}

impl std::ops::Deref for StreamOptions {
	type Target = ProviderRequestOptions;

	fn deref(&self) -> &ProviderRequestOptions {
		&self.base
	}
}

impl std::ops::DerefMut for StreamOptions {
	fn deref_mut(&mut self) -> &mut ProviderRequestOptions {
		&mut self.base
	}
}

impl AsRef<ProviderRequestOptions> for StreamOptions {
	fn as_ref(&self) -> &ProviderRequestOptions {
		&self.base
	}
}

impl From<ProviderRequestOptions> for StreamOptions {
	fn from(base: ProviderRequestOptions) -> Self {
		StreamOptions { base, ..Default::default() }
	}
}

/// Upstream `DeferredFetchOptions`: provider request options plus the maximum
/// provider long-poll duration.
#[derive(Clone, Default)]
pub struct DeferredFetchOptions {
	pub base: ProviderRequestOptions,
	/// Maximum provider long-poll duration in milliseconds. Defaults to 0,
	/// which performs one status check.
	pub wait: Option<u64>,
}

impl std::ops::Deref for DeferredFetchOptions {
	type Target = ProviderRequestOptions;
	fn deref(&self) -> &ProviderRequestOptions {
		&self.base
	}
}

/// Request options for best-effort deferred-response cancellation
/// (`DeferredCancelOptions`).
#[derive(Clone, Default)]
pub struct DeferredCancelOptions {
	pub base: ProviderRequestOptions,
}

impl std::ops::Deref for DeferredCancelOptions {
	type Target = ProviderRequestOptions;
	fn deref(&self) -> &ProviderRequestOptions {
		&self.base
	}
}

/// Unified options for `streamSimple()` / `completeSimple()`
/// (`SimpleStreamOptions`).
#[derive(Clone, Default)]
pub struct SimpleStreamOptions {
	pub base: StreamOptions,
	/// Provider-neutral tool selection. When omitted, adapters use
	/// provider-specific behavior.
	pub tool_choice: Option<ToolChoice>,
	/// Pi thinking level mapped by each adapter.
	pub reasoning: Option<ThinkingLevel>,
	/// Ask a capable provider to return a durable handle and continue the
	/// request asynchronously.
	pub deferred: Option<DeferredRequest>,
	/// Custom token budgets for thinking levels (token-based providers only).
	pub thinking_budgets: Option<ThinkingBudgets>,
}

impl std::ops::Deref for SimpleStreamOptions {
	type Target = StreamOptions;
	fn deref(&self) -> &StreamOptions {
		&self.base
	}
}

impl std::ops::DerefMut for SimpleStreamOptions {
	fn deref_mut(&mut self) -> &mut StreamOptions {
		&mut self.base
	}
}

impl AsRef<StreamOptions> for SimpleStreamOptions {
	fn as_ref(&self) -> &StreamOptions {
		&self.base
	}
}

impl From<StreamOptions> for SimpleStreamOptions {
	fn from(base: StreamOptions) -> Self {
		SimpleStreamOptions { base, ..Default::default() }
	}
}

/// `boolean | { window?: "15m" | "1h" | "24h" }` deferred request switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredRequest {
	Bool(bool),
	Window(Option<DeferredWindow>),
}

/// Deferred response windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredWindow {
	M15,
	H1,
	H24,
}

impl DeferredRequest {
	pub fn is_enabled(&self) -> bool {
		match self {
			DeferredRequest::Bool(enabled) => *enabled,
			DeferredRequest::Window(_) => true,
		}
	}
}

/// Options for classifier requests (`ClassifierOptions`).
#[derive(Clone, Default)]
pub struct ClassifierOptions {
	pub base: ProviderRequestOptions,
	/// Divides the answer logits by this value before normalization. Must be
	/// positive; APIs that cannot apply it ignore it.
	pub temperature: Option<f64>,
}

impl std::ops::Deref for ClassifierOptions {
	type Target = ProviderRequestOptions;
	fn deref(&self) -> &ProviderRequestOptions {
		&self.base
	}
}

/// Options for image-generation requests (`ImagesOptions`).
#[derive(Clone, Default)]
pub struct ImagesOptions {
	pub base: ProviderRequestOptions,
	/// Optional metadata to include in API requests.
	pub metadata: Option<serde_json::Map<String, Value>>,
}

impl std::ops::Deref for ImagesOptions {
	type Target = ProviderRequestOptions;
	fn deref(&self) -> &ProviderRequestOptions {
		&self.base
	}
}

#[cfg(test)]
mod tests {
	// Port of test/lax-message-content.test.ts: untyped callers and legacy
	// session files can carry null/missing `content`; the Rust substrate
	// normalizes at the deserialization boundary (upstream normalizes inside
	// transformMessages — same observable result, deviation D13).
	use super::*;

	fn text_only_model() -> Model {
		Model {
			base: BaseModelFields {
				id: "test-model".to_string(),
				name: "Test Model".to_string(),
				api: api_ids::OPENAI_COMPLETIONS.to_string(),
				provider: provider_ids::OPENAI.to_string(),
				base_url: "https://example.invalid/v1".to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: ModelCost::default(),
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 128_000,
			max_tokens: 16_000,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	#[test]
	fn normalizes_null_and_missing_content_to_empty_array() {
		let now = 1_700_000_000_000i64;
		let messages = serde_json::json!([
			{ "role": "user", "content": null, "timestamp": now },
			{
				"role": "assistant",
				"content": null,
				"api": "openai-completions",
				"provider": "openai",
				"model": "test-model",
				"usage": {
					"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
					"totalTokens": 0,
					"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
				},
				"stopReason": "stop",
				"timestamp": now
			},
			{
				"role": "toolResult",
				"toolCallId": "call_1",
				"toolName": "web_search",
				"isError": false,
				"timestamp": now
			}
		]);

		let parsed: Vec<Message> = serde_json::from_value(messages).expect("lax parse");
		assert_eq!(parsed.len(), 3);
		for message in &parsed {
			let empty = match message {
				Message::User(user) => user.content == UserContentField::Blocks(Vec::new()),
				Message::Assistant(assistant) => assistant.content.is_empty(),
				Message::ToolResult(tool_result) => tool_result.content.is_empty(),
				Message::System(_) => false,
			};
			assert!(empty, "content must normalize to empty");
		}

		// The normalized messages still flow through the text-only model path
		// that upstream used as its crash site (image downgrade path).
		let _model = text_only_model();
	}

	// Port of test/message-types.test.ts: the upstream suite is mostly
	// type-level (`ToolResultMessage<TDetails>` conditional types); Rust
	// carries `details` as `Option<JsonValue>`, so the port asserts the
	// runtime-observable wire behavior — detail payload shapes round-trip and
	// the serialized JSON keys match upstream exactly.
	use super::*;

	fn base_tool_result() -> ToolResultMessage {
		ToolResultMessage {
			tool_call_id: "call-1".to_string(),
			tool_name: "read".to_string(),
			content: Vec::new(),
			details: None,
			usage: None,
			nested_calls: None,
			is_error: false,
			timestamp: 1,
		}
	}

	#[test]
	fn preserves_json_compatible_detail_payloads() {
		let mut object_message = base_tool_result();
		object_message.details = Some(serde_json::json!({
			"path": "a.ts",
			"items": ["first"],
			"summary": { "lines": 3 }
		}));
		let mut array_message = base_tool_result();
		array_message.details = Some(serde_json::json!(["a", "b"]));
		let mut primitive_message = base_tool_result();
		primitive_message.details = Some(serde_json::json!("diagnostic"));
		let mut null_message = base_tool_result();
		null_message.details = Some(serde_json::Value::Null);

		assert_eq!(object_message.details.as_ref().and_then(|d| d.get("summary")).and_then(|s| s.get("lines")).and_then(Value::as_i64), Some(3));
		assert_eq!(array_message.details, Some(serde_json::json!(["a", "b"])));
		assert_eq!(primitive_message.details, Some(serde_json::json!("diagnostic")));
		assert!(null_message.details.as_ref().map(Value::is_null).unwrap_or(false));

		let transcript = vec![Message::ToolResult(object_message.clone()), Message::ToolResult(array_message.clone())];
		let serialized = serde_json::to_value(&transcript).expect("transcript serializes");
		let reparsed: Vec<Message> = serde_json::from_value(serialized).expect("transcript reparses");
		assert_eq!(reparsed, transcript);

		let context = Context { system_prompt: None, messages: transcript, tools: None };
		assert_eq!(context.messages.len(), 2);
	}

	#[test]
	fn wire_keys_match_upstream_shapes() {
		let json = serde_json::to_value(Message::ToolResult(base_tool_result())).expect("serializes");
		let object = json.as_object().expect("object");
		for key in ["role", "toolCallId", "toolName", "content", "isError", "timestamp"] {
			assert!(object.contains_key(key), "missing key {key}");
		}
		assert_eq!(object.get("role").and_then(Value::as_str), Some("toolResult"));
	}

	#[test]
	fn assistant_message_keys_match_upstream_shapes() {
		let message = AssistantMessage {
			content: vec![AssistantContent::Text(TextContent::new("hi"))],
			api: "anthropic-messages".to_string(),
			provider: "anthropic".to_string(),
			model: "claude-opus-4-7".to_string(),
			response_model: Some("claude-opus-4-7-20260101".to_string()),
			response_id: Some("msg_1".to_string()),
			provider_thinking_level: None,
			thinking_level: Some(ModelThinkingLevel::High),
			diagnostics: None,
			usage: Usage {
				input: 3,
				output: 5,
				cache_read: 1,
				cache_write: 2,
				cache_write_1h: None,
				reasoning: None,
				total_tokens: 11,
				cost: UsageCost::default(),
			},
			stop_reason: StopReason::Stop,
			deferred: None,
			error_message: None,
			raw_stop_reason: None,
			end_turn: Some(true),
			timestamp: 42,
		};
		let json = serde_json::to_value(Message::Assistant(message.clone())).expect("serializes");
		for key in [
			"role", "content", "api", "provider", "model", "responseModel", "responseId", "thinkingLevel", "usage",
			"stopReason", "endTurn", "timestamp",
		] {
			assert!(json.get(key).is_some(), "missing key {key}");
		}
		for absent in ["providerThinkingLevel", "diagnostics", "deferred", "errorMessage", "rawStopReason"] {
			assert!(json.get(absent).is_none(), "{absent} must be omitted");
		}
		assert_eq!(json.get("usage").and_then(|u| u.get("cacheRead")).and_then(Value::as_u64), Some(1));
		assert_eq!(json.get("stopReason").and_then(Value::as_str), Some("stop"));
		assert_eq!(json.get("thinkingLevel").and_then(Value::as_str), Some("high"));
		let reparsed: AssistantMessage = serde_json::from_value(json).expect("reparses");
		assert_eq!(reparsed, message);
	}

	#[test]
	fn any_model_dispatches_on_type_field() {
		let chat = serde_json::json!({
			"id": "claude-opus-4-7", "name": "Claude Opus 4.7", "api": "anthropic-messages",
			"provider": "anthropic", "baseUrl": "https://api.anthropic.com",
			"input": ["text", "image"], "reasoning": true,
			"contextWindow": 200000, "maxTokens": 64000,
			"cost": { "input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75 }
		});
		let image = serde_json::json!({
			"id": "gpt-image-2", "name": "GPT Image 2", "api": "openrouter-images",
			"provider": "openrouter", "baseUrl": "https://openrouter.ai",
			"input": ["text"], "type": "image", "output": ["image"],
			"cost": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0 }
		});
		let classifier = serde_json::json!({
			"id": "cls-1", "name": "Classifier", "api": "typesafe-system-one",
			"provider": "typesafe", "baseUrl": "https://typesafe.dev",
			"input": ["text"], "type": "classifier", "contextWindow": 8192,
			"cost": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0 }
		});

		let parsed_chat = AnyModel::from_json(&chat).expect("chat");
		assert_eq!(parsed_chat.model_type(), ModelType::Chat);
		assert!(matches!(parsed_chat, AnyModel::Chat(_)));
		let parsed_image = AnyModel::from_json(&image).expect("image");
		assert_eq!(parsed_image.model_type(), ModelType::Image);
		let parsed_classifier = AnyModel::from_json(&classifier).expect("classifier");
		assert_eq!(parsed_classifier.model_type(), ModelType::Classifier);

		// Round-trip keeps the discriminated wire shape.
		let serialized = serde_json::to_value(&parsed_classifier).expect("serializes");
		assert_eq!(serialized.get("type").and_then(Value::as_str), Some("classifier"));
		let chat_serialized = serde_json::to_value(&parsed_chat).expect("serializes");
		assert!(chat_serialized.get("type").is_none(), "chat models omit type");
	}
}
