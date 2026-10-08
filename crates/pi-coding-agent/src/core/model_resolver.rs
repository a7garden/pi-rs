//! Port of `packages/coding-agent/src/core/model-resolver.ts` — model
//! resolution, scoping, and initial selection (`parseModelPattern`,
//! `resolveModelScope*`, `resolveCliModel`, `findInitialModel`,
//! `restoreModelFromSession`, and `defaultModelPerProvider`).
//!
//! `ScopedModel` lives here (upstream home) and is re-exported by
//! [`crate::core::extensions::types`] (D29 lifecycle).
//!
//! # Deviations
//!
//! - **D47 — `ModelRuntime` is an injected seam trait.** Upstream resolver
//!   functions take a `ModelRuntime` instance; the runtime module is not
//!   ported yet, so the functions take [`ModelRuntimeSource`], an
//!   object-safe trait carrying exactly the five read methods they use
//!   (`getModels`, `getAvailableSnapshot`, `getAvailable`, `getModel`,
//!   `hasConfiguredAuth`). The future `core::model_runtime` port implements
//!   the trait; tests inject fakes mirroring upstream's duck-typed registries.
//! - **D48 — async signatures without awaits collapse.** Upstream
//!   `findInitialModel` and `restoreModelFromSession` are `async` but never
//!   await; the ported functions are synchronous.
//! - **Console output is plain and colorless.** `chalk.yellow/red/dim` and
//!   `console.*` become plain [`eprintln!`]/[`println!`] lines (the colored
//!   TUI layer is excluded by port decision).
//! - **`minimatch` becomes the `glob` crate.** Model-scope globs use
//!   [`glob::Pattern`] with `case_insensitive` and `require_literal_separator`
//!   (the package-manager D40a convention), matching minimatch's default
//!   `*`/`?`/`[...]` semantics (single `*` does not cross `/`); patterns that
//!   fail to compile match nothing where minimatch would throw.
//! - **`localeCompare` sorts become case-insensitive byte compares** (crate
//!   docs D39 precedent).

use glob::MatchOptions;
use pi_agent::AgentThinkingLevel;
use pi_ai::auth::types::AuthOperationOptions;
use pi_ai::types::Model;
use pi_ai::utils::thrown::ThrownValue;

use crate::core::defaults::DEFAULT_THINKING_LEVEL;

/// Boxed future returned by the async member of [`ModelRuntimeSource`].
pub type BoxFuture<'a, T> = futures_util::future::BoxFuture<'a, T>;


/// Default chat model IDs for providers with built-in chat models
/// (`defaultModelPerProvider`), in upstream declaration order.
pub const DEFAULT_MODEL_PER_PROVIDER: &[(&str, &str)] = &[
	("amazon-bedrock", "us.anthropic.claude-opus-4-6-v1"),
	("ant-ling", "Ring-2.6-1T"),
	("anthropic", "claude-opus-4-8"),
	("openai", "gpt-5.5"),
	("azure", "gpt-5.4"),
	("openai-codex", "gpt-6.1-sol"),
	("radius", "balanced"),
	("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
	("deepseek", "deepseek-v4-pro"),
	("google", "gemini-3.1-pro-preview"),
	("google-vertex", "gemini-3.1-pro-preview"),
	("github-copilot", "gpt-5.4"),
	("openrouter", "moonshotai/kimi-k2.6"),
	("vercel-ai-gateway", "zai/glm-5.1"),
	("xai", "grok-4.7"),
	("groq", "openai/gpt-oss-120b"),
	("cerebras", "gpt-oss-120b"),
	("zai", "glm-5.3"),
	("zai-coding-cn", "glm-5.3"),
	("mistral", "devstral-medium-latest"),
	("minimax", "MiniMax-M2.7"),
	("minimax-cn", "MiniMax-M2.7"),
	("moonshotai", "kimi-k2.6"),
	("moonshotai-cn", "kimi-k2.6"),
	("huggingface", "moonshotai/Kimi-K2.6"),
	("fireworks", "accounts/fireworks/models/kimi-k3"),
	("together", "moonshotai/Kimi-K3"),
	("baseten", "zai-org/GLM-5.2"),
	("opencode", "kimi-k2.6"),
	("opencode-go", "kimi-k3"),
	("kimi-coding", "kimi-for-coding"),
	("meta", "muse-spark-1.3"),
	("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6"),
	("cloudflare-ai-gateway", "workers-ai/@cf/moonshotai/kimi-k2.6"),
	("qwen-token-plan", "qwen3.7-max"),
	("qwen-token-plan-cn", "qwen3.7-max"),
	("qwen-token-plan-individual", "qwen3.8-max"),
	("xiaomi", "mimo-v2.5-pro"),
	("xiaomi-token-plan-cn", "mimo-v2.5-pro"),
	("xiaomi-token-plan-ams", "mimo-v2.5-pro"),
	("xiaomi-token-plan-sgp", "mimo-v2.5-pro"),
];

/// `defaultModelPerProvider[provider]`.
pub fn default_model_for_provider(provider: &str) -> Option<&'static str> {
	let mut found = None;
	for (id, model) in DEFAULT_MODEL_PER_PROVIDER {
		if *id == provider {
			found = Some(*model);
		}
	}
	found
}

/// Providers with a built-in default, in declaration order
/// (`Object.keys(defaultModelPerProvider)`).
pub fn default_model_providers() -> impl Iterator<Item = &'static str> {
	DEFAULT_MODEL_PER_PROVIDER.iter().map(|(provider, _)| *provider)
}

/// A model scoped to the session with an optional explicitly requested
/// thinking level (`ScopedModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScopedModel {
	pub model: Model,
	/// Thinking level if explicitly specified in the pattern (`"model:high"`).
	pub thinking_level: Option<AgentThinkingLevel>,
}

/// Upstream `cli/args.ts` `isValidThinkingLevel` (module not ported; the
/// resolver is its only core-side consumer).
pub fn is_valid_thinking_level(level: &str) -> bool {
	matches!(level, "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max")
}

/// `VALID_THINKING_LEVELS` from upstream `cli/args.ts`, in ladder order.
pub const VALID_THINKING_LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Helper to check if a model ID looks like an alias (no date suffix)
/// (`isAlias`). Dates are typically in format: -20241022 or -20250929.
fn is_alias(id: &str) -> bool {
	// Check if ID ends with -latest
	if id.ends_with("-latest") {
		return true;
	}

	// Check if ID ends with a date pattern (-YYYYMMDD)
	let bytes = id.as_bytes();
	if bytes.len() >= 9 {
		let suffix = &bytes[bytes.len() - 9..];
		if suffix[0] == b'-' && suffix[1..].iter().all(u8::is_ascii_digit) {
			return false;
		}
	}
	true
}

/// Find an exact model reference match (`findExactModelReferenceMatch`).
/// Supports either a bare model id or a canonical `provider/modelId`
/// reference. When matching by bare id, ambiguous matches across providers
/// are rejected.
pub fn find_exact_model_reference_match(model_reference: &str, available_models: &[Model]) -> Option<Model> {
	let trimmed_reference = model_reference.trim();
	if trimmed_reference.is_empty() {
		return None;
	}

	let normalized_reference = trimmed_reference.to_lowercase();

	let canonical_matches: Vec<&Model> = available_models
		.iter()
		.filter(|model| format!("{}/{}", model.provider(), model.id()).to_lowercase() == normalized_reference)
		.collect();
	if canonical_matches.len() == 1 {
		return Some(canonical_matches[0].clone());
	}
	if canonical_matches.len() > 1 {
		return None;
	}

	if let Some(slash_index) = trimmed_reference.find('/') {
		let provider = trimmed_reference[..slash_index].trim();
		let model_id = trimmed_reference[slash_index + 1..].trim();
		if !provider.is_empty() && !model_id.is_empty() {
			let provider_matches: Vec<&Model> = available_models
				.iter()
				.filter(|model| {
					model.provider().to_lowercase() == provider.to_lowercase()
						&& model.id().to_lowercase() == model_id.to_lowercase()
				})
				.collect();
			if provider_matches.len() == 1 {
				return Some(provider_matches[0].clone());
			}
			if provider_matches.len() > 1 {
				return None;
			}
		}
	}

	let id_matches: Vec<&Model> = available_models
		.iter()
		.filter(|model| model.id().to_lowercase() == normalized_reference)
		.collect();
	if id_matches.len() == 1 {
		Some(id_matches[0].clone())
	} else {
		None
	}
}

/// Try to match a pattern to a model from the available models list
/// (`tryMatchModel`). Returns the matched model or `None` if no match found.
fn try_match_model(model_pattern: &str, available_models: &[Model]) -> Option<Model> {
	if let Some(exact_match) = find_exact_model_reference_match(model_pattern, available_models) {
		return Some(exact_match);
	}

	// No exact match - fall back to partial matching
	let pattern_lower = model_pattern.to_lowercase();
	let matches: Vec<&Model> = available_models
		.iter()
		.filter(|m| m.id().to_lowercase().contains(&pattern_lower) || m.base.name.to_lowercase().contains(&pattern_lower))
		.collect();

	if matches.is_empty() {
		return None;
	}

	// Separate into aliases and dated versions
	let mut aliases: Vec<&Model> = matches.iter().filter(|m| is_alias(m.id())).map(|m| *m).collect();
	let mut dated_versions: Vec<&Model> = matches.iter().filter(|m| !is_alias(m.id())).map(|m| *m).collect();

	if !aliases.is_empty() {
		// Prefer alias - if multiple aliases, pick the one that sorts highest
		aliases.sort_by(|a, b| b.id().to_lowercase().cmp(&a.id().to_lowercase()));
		Some(aliases[0].clone())
	} else {
		// No alias found, pick latest dated version
		dated_versions.sort_by(|a, b| b.id().to_lowercase().cmp(&a.id().to_lowercase()));
		Some(dated_versions[0].clone())
	}
}

/// Result of [`parse_model_pattern`] (`ParsedModelResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedModelResult {
	pub model: Option<Model>,
	/// Thinking level if explicitly specified in the pattern.
	pub thinking_level: Option<AgentThinkingLevel>,
	pub warning: Option<String>,
}

fn build_fallback_model(provider: &str, model_id: &str, available_models: &[Model]) -> Option<Model> {
	let provider_models: Vec<&Model> = available_models.iter().filter(|m| m.provider() == provider).collect();
	if provider_models.is_empty() {
		return None;
	}

	let default_id = default_model_for_provider(provider);
	let base_model: &Model = match default_id {
		Some(default_id) => {
			provider_models.iter().copied().find(|m| m.id() == default_id).unwrap_or(provider_models[0])
		}
		None => provider_models[0],
	};

	let mut model = base_model.clone();
	model.base.id = model_id.to_string();
	model.base.name = model_id.to_string();
	Some(model)
}

/// Parse a pattern to extract model and thinking level (`parseModelPattern`).
/// Handles models with colons in their IDs (e.g., OpenRouter's `:exacto`
/// suffix).
///
/// Algorithm:
/// 1. Try to match full pattern as a model
/// 2. If found, return it with `None` thinking level
/// 3. If not found and has colons, split on last colon:
///    - If suffix is a valid thinking level, use it and recurse on prefix
///    - If suffix is invalid, warn and recurse on prefix with `None`
pub fn parse_model_pattern(
	pattern: &str,
	available_models: &[Model],
	options: Option<ParseModelPatternOptions>,
) -> ParsedModelResult {
	// Try exact match first
	if let Some(exact_match) = try_match_model(pattern, available_models) {
		return ParsedModelResult { model: Some(exact_match), thinking_level: None, warning: None };
	}

	// No match - try splitting on last colon if present
	let Some(last_colon_index) = pattern.rfind(':') else {
		// No colons, pattern simply doesn't match any model
		return ParsedModelResult::default();
	};

	let prefix = &pattern[..last_colon_index];
	let suffix = &pattern[last_colon_index + 1..];

	if is_valid_thinking_level(suffix) {
		// Valid thinking level - recurse on prefix and use this level
		let result = parse_model_pattern(prefix, available_models, options);
		if result.model.is_some() {
			// Only use this thinking level if no warning from inner recursion
			return ParsedModelResult {
				thinking_level: if result.warning.is_some() { None } else { Some(parse_level(suffix)) },
				warning: result.warning,
				model: result.model,
			};
		}
		result
	} else {
		// Invalid suffix
		let allow_fallback = options.and_then(|options| options.allow_invalid_thinking_level_fallback).unwrap_or(true);
		if !allow_fallback {
			// In strict mode (CLI --model parsing), treat it as part of the model id and fail.
			// This avoids accidentally resolving to a different model.
			return ParsedModelResult::default();
		}

		// Scope mode: recurse on prefix and warn
		let result = parse_model_pattern(prefix, available_models, options);
		if result.model.is_some() {
			return ParsedModelResult {
				thinking_level: None,
				warning: Some(format!("Invalid thinking level \"{suffix}\" in pattern \"{pattern}\". Using default instead.")),
				model: result.model,
			};
		}
		result
	}
}

/// Options bag of [`parse_model_pattern`] (`{ allowInvalidThinkingLevelFallback?: boolean }`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParseModelPatternOptions {
	pub allow_invalid_thinking_level_fallback: Option<bool>,
}

fn parse_level(suffix: &str) -> AgentThinkingLevel {
	match suffix {
		"off" => AgentThinkingLevel::Off,
		"minimal" => AgentThinkingLevel::Minimal,
		"low" => AgentThinkingLevel::Low,
		"medium" => AgentThinkingLevel::Medium,
		"high" => AgentThinkingLevel::High,
		"xhigh" => AgentThinkingLevel::Xhigh,
		_ => AgentThinkingLevel::Max,
	}
}

/// A structured warning from model-scope resolution (`ModelScopeDiagnostic`).
/// The upstream `type` is always `"warning"` and is carried by the code enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelScopeDiagnostic {
	pub code: ModelScopeDiagnosticCode,
	pub message: String,
	pub pattern: String,
}

/// `ModelScopeDiagnostic.code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelScopeDiagnosticCode {
	/// `"no-match"`
	NoMatch,
	/// `"invalid-thinking-level"`
	InvalidThinkingLevel,
}

/// Result of [`resolve_model_scope_from_models`] (`ResolveModelScopeResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolveModelScopeResult {
	pub scoped_models: Vec<ScopedModel>,
	pub diagnostics: Vec<ModelScopeDiagnostic>,
}

/// The read surface of `ModelRuntime` used by the resolver (crate docs D47).
/// The future `core::model_runtime` port implements this; tests inject fakes.
pub trait ModelRuntimeSource: Send + Sync {
	/// `getModels()` — every catalog chat model, synchronous.
	fn get_models(&self) -> Vec<Model>;
	/// `getAvailableSnapshot()` — last-known available chat models, sync.
	fn get_available_snapshot(&self) -> Vec<Model>;
	/// `getAvailable(provider, options)` — refresh-aware availability.
	fn get_available<'a>(
		&'a self,
		provider: Option<&'a str>,
		options: Option<AuthOperationOptions>,
	) -> BoxFuture<'a, Result<Vec<Model>, ThrownValue>>;
	/// `getModel(provider, modelId)`.
	fn get_model(&self, provider: &str, model_id: &str) -> Option<Model>;
	/// `hasConfiguredAuth(provider)`.
	fn has_configured_auth(&self, provider: &str) -> bool;
}

/// Resolve model patterns to actual models with optional thinking levels
/// (`resolveModelScopeFromModels`). Format: `"pattern:level"` where `:level`
/// is optional. For each pattern, finds all matching models and picks the
/// best version:
/// 1. Prefer alias (e.g., `claude-sonnet-4-5`) over dated versions
///    (`claude-sonnet-4-5-20250929`)
/// 2. If no alias, pick the latest dated version
///
/// Supports models with colons in their IDs (e.g., OpenRouter's
/// `model:exacto`): the algorithm tries to match the full pattern first,
/// then progressively strips colon-suffixes to find a match.
pub fn resolve_model_scope_from_models(patterns: &[String], models: &[Model]) -> ResolveModelScopeResult {
	let available_models: Vec<Model> = models.to_vec();
	let mut scoped_models: Vec<ScopedModel> = Vec::new();
	let mut diagnostics: Vec<ModelScopeDiagnostic> = Vec::new();

	for pattern in patterns {
		// Check if pattern contains glob characters
		if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
			// Extract optional thinking level suffix (e.g., "provider/*:high")
			let colon_idx = pattern.rfind(':');
			let mut glob_pattern = pattern.as_str();
			let mut thinking_level: Option<AgentThinkingLevel> = None;

			if let Some(colon_idx) = colon_idx {
				let suffix = &pattern[colon_idx + 1..];
				if is_valid_thinking_level(suffix) {
					thinking_level = Some(parse_level(suffix));
					glob_pattern = &pattern[..colon_idx];
				}
			}

			if let Some(exact_match) = find_exact_model_reference_match(glob_pattern, &available_models) {
				if !scoped_models
					.iter()
					.any(|sm| models_equal(&sm.model, &exact_match))
				{
					scoped_models.push(ScopedModel { model: exact_match, thinking_level });
				}
				continue;
			}

			// Match against "provider/modelId" format OR just model ID
			// This allows "*sonnet*" to match without requiring "anthropic/*sonnet*"
			// minimatch nocase semantics via the crate's `glob` convention
			// (package-manager D40a precedent): `*`/`?` do not cross `/`.
			let matcher = |value: &str| -> bool {
				match glob::Pattern::new(glob_pattern) {
					Ok(compiled) => compiled.matches_with(
						value,
						MatchOptions {
							case_sensitive: false,
							require_literal_separator: true,
							..MatchOptions::new()
						},
					),
					Err(_) => false,
				}
			};
			let matching_models: Vec<Model> = available_models
				.iter()
				.filter(|m| matcher(&format!("{}/{}", m.provider(), m.id())) || matcher(m.id()))
				.cloned()
				.collect();

			if matching_models.is_empty() {
				diagnostics.push(ModelScopeDiagnostic {
					code: ModelScopeDiagnosticCode::NoMatch,
					message: format!("No models match pattern \"{pattern}\""),
					pattern: pattern.clone(),
				});
				continue;
			}

			for model in matching_models {
				if !scoped_models
					.iter()
					.any(|sm| models_equal(&sm.model, &model))
				{
					scoped_models.push(ScopedModel { model, thinking_level });
				}
			}
			continue;
		}

		let result = parse_model_pattern(pattern, &available_models, None);

		if let Some(warning) = &result.warning {
			diagnostics.push(ModelScopeDiagnostic {
				code: ModelScopeDiagnosticCode::InvalidThinkingLevel,
				message: warning.clone(),
				pattern: pattern.clone(),
			});
		}

		let Some(model) = result.model else {
			diagnostics.push(ModelScopeDiagnostic {
				code: ModelScopeDiagnosticCode::NoMatch,
				message: format!("No models match pattern \"{pattern}\""),
				pattern: pattern.clone(),
			});
			continue;
		};

		// Avoid duplicates
		if !scoped_models
			.iter()
			.any(|sm| models_equal(&sm.model, &model))
		{
			scoped_models.push(ScopedModel { model, thinking_level: result.thinking_level });
		}
	}

	ResolveModelScopeResult { scoped_models, diagnostics }
}

/// `modelsAreEqual(a, b)` for two chat models (type/id/provider).
fn models_equal(a: &Model, b: &Model) -> bool {
	a.model_type() == b.model_type() && a.id() == b.id() && a.provider() == b.provider()
}

/// Async variant fetching the available list through the runtime
/// (`resolveModelScopeWithDiagnostics`).
pub async fn resolve_model_scope_with_diagnostics(
	patterns: &[String],
	model_runtime: &dyn ModelRuntimeSource,
	options: Option<AuthOperationOptions>,
) -> Result<ResolveModelScopeResult, ThrownValue> {
	let models = model_runtime.get_available(None, options).await?;
	Ok(resolve_model_scope_from_models(patterns, &models))
}

/// [`resolve_model_scope_with_diagnostics`] with warnings printed to stderr
/// (`resolveModelScope`).
pub async fn resolve_model_scope(
	patterns: &[String],
	model_runtime: &dyn ModelRuntimeSource,
	options: Option<AuthOperationOptions>,
) -> Result<Vec<ScopedModel>, ThrownValue> {
	let ResolveModelScopeResult { scoped_models, diagnostics } =
		resolve_model_scope_with_diagnostics(patterns, model_runtime, options).await?;
	for diagnostic in &diagnostics {
		eprintln!("Warning: {}", diagnostic.message);
	}
	Ok(scoped_models)
}

/// Result of [`resolve_cli_model`] (`ResolveCliModelResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolveCliModelResult {
	pub model: Option<Model>,
	pub thinking_level: Option<AgentThinkingLevel>,
	pub warning: Option<String>,
	/// Error message suitable for CLI display. When set, `model` is `None`.
	pub error: Option<String>,
}

/// Options bag of [`resolve_cli_model`].
pub struct ResolveCliModelOptions<'a> {
	pub cli_provider: Option<&'a str>,
	pub cli_model: Option<&'a str>,
	pub cli_thinking: Option<AgentThinkingLevel>,
	pub model_runtime: &'a dyn ModelRuntimeSource,
}

/// Resolve a single model from CLI flags (`resolveCliModel`).
///
/// Supports:
/// - `--provider <provider> --model <pattern>`
/// - `--model <provider>/<pattern>`
/// - Fuzzy matching (same rules as model scoping: exact id, then partial id/name)
///
/// Note: This does not apply the thinking level by itself, but it may *parse*
/// and return a thinking level from `"<pattern>:<thinking>"` so the caller
/// can apply it.
pub fn resolve_cli_model(options: ResolveCliModelOptions<'_>) -> ResolveCliModelResult {
	let ResolveCliModelOptions { cli_provider, cli_model, cli_thinking, model_runtime } = options;

	let Some(cli_model) = cli_model else {
		return ResolveCliModelResult::default();
	};

	// Important: use *all* models here, not just models with pre-configured auth.
	// This allows "--api-key" to be used for first-time setup.
	let available_models = model_runtime.get_models();
	if available_models.is_empty() {
		return ResolveCliModelResult {
			model: None,
			warning: None,
			error: Some("No models available. Check your installation or add models to models.json.".to_string()),
			thinking_level: None,
		};
	}

	// Build canonical provider lookup (case-insensitive)
	let mut provider_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
	for m in &available_models {
		provider_map.insert(m.provider().to_lowercase(), m.provider().to_string());
	}

	let mut provider = cli_provider.and_then(|cli_provider| provider_map.get(&cli_provider.to_lowercase()).cloned());
	if let Some(cli_provider) = cli_provider {
		if provider.is_none() {
			return ResolveCliModelResult {
				model: None,
				warning: None,
				error: Some(format!(
					"Unknown provider \"{cli_provider}\". Use --list-models to see available providers/models."
				)),
				thinking_level: None,
			};
		}
	}

	// If no explicit --provider, try to interpret "provider/model" format first.
	// When the prefix before the first slash matches a known provider, prefer that
	// interpretation over matching models whose IDs literally contain slashes
	// (e.g. "zai/glm-5" should resolve to provider=zai, model=glm-5, not to a
	// vercel-ai-gateway model with id "zai/glm-5").
	let mut pattern = cli_model.to_string();
	let mut inferred_provider = false;

	if provider.is_none() {
		if let Some(slash_index) = cli_model.find('/') {
			let maybe_provider = &cli_model[..slash_index];
			if let Some(canonical) = provider_map.get(&maybe_provider.to_lowercase()) {
				provider = Some(canonical.clone());
				pattern = cli_model[slash_index + 1..].to_string();
				inferred_provider = true;
			}
		}
	}

	// If no provider was inferred from the slash, try exact matches without provider inference.
	// This handles models whose IDs naturally contain slashes (e.g. OpenRouter-style IDs).
	// Bare exact IDs can exist in multiple providers, so do not choose by catalog order.
	// Prefer the sole authenticated provider when there is one; otherwise require an
	// explicit provider to avoid silently selecting an unusable provider.
	if provider.is_none() {
		let lower = cli_model.to_lowercase();
		let exact_matches: Vec<&Model> = available_models
			.iter()
			.filter(|m| m.id().to_lowercase() == lower || format!("{}/{}", m.provider(), m.id()).to_lowercase() == lower)
			.collect();
		if exact_matches.len() == 1 {
			return ResolveCliModelResult {
				model: Some(exact_matches[0].clone()),
				warning: None,
				thinking_level: None,
				error: None,
			};
		}
		if exact_matches.len() > 1 {
			let authenticated_exact_matches: Vec<&Model> = exact_matches
				.iter()
				.copied()
				.filter(|m| model_runtime.has_configured_auth(m.provider()))
				.collect();
			if authenticated_exact_matches.len() == 1 {
				return ResolveCliModelResult {
					model: Some(authenticated_exact_matches[0].clone()),
					warning: None,
					thinking_level: None,
					error: None,
				};
			}

			let mut references: Vec<String> =
				exact_matches.iter().map(|m| format!("{}/{}", m.provider(), m.id())).collect();
			references.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
			let matches = references.join(", ");
			let auth_hint = if authenticated_exact_matches.is_empty() {
				"No matching provider is authenticated."
			} else {
				"More than one matching provider is authenticated."
			};
			return ResolveCliModelResult {
				model: None,
				warning: None,
				thinking_level: None,
				error: Some(format!(
					"Model \"{cli_model}\" is ambiguous across providers: {matches}. {auth_hint} Use --provider or provider/model."
				)),
			};
		}
	}

	if let Some(provider) = &provider {
		if cli_provider.is_some() {
			// If both were provided, tolerate --model <provider>/<pattern> by stripping the provider prefix
			let prefix = format!("{provider}/");
			if cli_model.to_lowercase().starts_with(&prefix.to_lowercase()) {
				pattern = cli_model[prefix.len()..].to_string();
			}
		}
	}

	let candidates: Vec<Model> = match &provider {
		Some(provider) => available_models.iter().filter(|m| m.provider() == provider).cloned().collect(),
		None => available_models.clone(),
	};
	let result = parse_model_pattern(
		&pattern,
		&candidates,
		Some(ParseModelPatternOptions { allow_invalid_thinking_level_fallback: Some(false) }),
	);

	if let Some(model) = result.model {
		// If provider inference matched an unauthenticated provider/model pair, prefer
		// one exact raw model-id match that is authenticated. This keeps
		// "provider/model" syntax preferred when usable, but handles models whose
		// literal id starts with a known provider name (for example
		// commandcode model id "xiaomi/mimo-v2.5-pro").
		if inferred_provider {
			let raw_exact_matches: Vec<&Model> = available_models
				.iter()
				.filter(|m| m.id().to_lowercase() == cli_model.to_lowercase() && !models_equal(m, &model))
				.collect();
			if !raw_exact_matches.is_empty() && !model_runtime.has_configured_auth(model.provider()) {
				let authenticated_raw_matches: Vec<&Model> = raw_exact_matches
					.iter()
					.copied()
					.filter(|m| model_runtime.has_configured_auth(m.provider()))
					.collect();
				if authenticated_raw_matches.len() == 1 {
					return ResolveCliModelResult {
						model: Some(authenticated_raw_matches[0].clone()),
						thinking_level: None,
						warning: None,
						error: None,
					};
				}
			}
		}
		return ResolveCliModelResult {
			model: Some(model),
			thinking_level: result.thinking_level,
			warning: result.warning,
			error: None,
		};
	}

	// If we inferred a provider from the slash but found no match within that provider,
	// fall back to matching the full input as a raw model id across all models.
	// This handles OpenRouter-style IDs like "openai/gpt-4o:extended" where "openai"
	// looks like a provider but the full string is actually a model id on openrouter.
	if inferred_provider {
		let lower = cli_model.to_lowercase();
		let exact = available_models
			.iter()
			.find(|m| m.id().to_lowercase() == lower || format!("{}/{}", m.provider(), m.id()).to_lowercase() == lower);
		if let Some(exact) = exact {
			return ResolveCliModelResult {
				model: Some(exact.clone()),
				warning: None,
				thinking_level: None,
				error: None,
			};
		}
		// Also try parseModelPattern on the full input against all models
		let fallback = parse_model_pattern(
			cli_model,
			&available_models,
			Some(ParseModelPatternOptions { allow_invalid_thinking_level_fallback: Some(false) }),
		);
		if let Some(model) = fallback.model {
			return ResolveCliModelResult {
				model: Some(model),
				thinking_level: fallback.thinking_level,
				warning: fallback.warning,
				error: None,
			};
		}
	}

	if let Some(provider) = &provider {
		// Parse thinking level suffix from the pattern before building the fallback model,
		// but only when --thinking is not explicitly provided.
		// e.g. "zai-org/GLM-5.1-FP8:high" → modelId="zai-org/GLM-5.1-FP8", fallbackThinking="high"
		let mut fallback_pattern = pattern.clone();
		let mut fallback_thinking: Option<AgentThinkingLevel> = None;
		if cli_thinking.is_none() {
			if let Some(last_colon) = pattern.rfind(':') {
				let suffix = &pattern[last_colon + 1..];
				if is_valid_thinking_level(suffix) {
					fallback_pattern = pattern[..last_colon].to_string();
					fallback_thinking = Some(parse_level(suffix));
				}
			}
		}

		if let Some(mut fallback_model) = build_fallback_model(provider, &fallback_pattern, &available_models) {
			let requested_thinking = cli_thinking.or(fallback_thinking);
			if requested_thinking.is_some_and(|level| level != AgentThinkingLevel::Off) {
				fallback_model.reasoning = true;
			}
			let fallback_warning = match result.warning {
				Some(warning) => format!(
					"{warning} Model \"{fallback_pattern}\" not found for provider \"{provider}\". Using custom model id."
				),
				None => format!(
					"Model \"{fallback_pattern}\" not found for provider \"{provider}\". Using custom model id."
				),
			};
			return ResolveCliModelResult {
				model: Some(fallback_model),
				thinking_level: fallback_thinking,
				warning: Some(fallback_warning),
				error: None,
			};
		}
	}

	let display = match &provider {
		Some(provider) => format!("{provider}/{pattern}"),
		None => cli_model.to_string(),
	};
	ResolveCliModelResult {
		model: None,
		thinking_level: None,
		warning: result.warning,
		error: Some(format!("Model \"{display}\" not found. Use --list-models to see available models.")),
	}
}

/// Result of [`find_initial_model`] (`InitialModelResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct InitialModelResult {
	pub model: Option<Model>,
	pub thinking_level: AgentThinkingLevel,
	pub fallback_message: Option<String>,
}

/// Options bag of [`find_initial_model`].
pub struct FindInitialModelOptions<'a> {
	pub cli_provider: Option<&'a str>,
	pub cli_model: Option<&'a str>,
	pub scoped_models: &'a [ScopedModel],
	pub is_continuing: bool,
	pub default_provider: Option<&'a str>,
	pub default_model_id: Option<&'a str>,
	pub default_thinking_level: Option<AgentThinkingLevel>,
	pub model_thinking_levels: Option<&'a std::collections::HashMap<String, AgentThinkingLevel>>,
	pub model_runtime: &'a dyn ModelRuntimeSource,
}

/// Find the initial model to use based on priority (`findInitialModel`):
/// 1. CLI args (provider + model)
/// 2. First model from scoped models (if not continuing/resuming)
/// 3. Restored from session (if continuing/resuming)
/// 4. Saved default from settings
/// 5. First available model with valid API key
pub fn find_initial_model(options: FindInitialModelOptions<'_>) -> InitialModelResult {
	let FindInitialModelOptions {
		cli_provider,
		cli_model,
		scoped_models,
		is_continuing,
		default_provider,
		default_model_id,
		default_thinking_level,
		model_thinking_levels,
		model_runtime,
	} = options;

	// 1. CLI args take priority
	if let (Some(cli_provider), Some(cli_model)) = (cli_provider, cli_model) {
		let resolved = resolve_cli_model(ResolveCliModelOptions {
			cli_provider: Some(cli_provider),
			cli_model: Some(cli_model),
			cli_thinking: None,
			model_runtime,
		});
		if let Some(error) = resolved.error {
			eprintln!("{error}");
			std::process::exit(1);
		}
		if let Some(model) = resolved.model {
			return InitialModelResult { model: Some(model), thinking_level: DEFAULT_THINKING_LEVEL, fallback_message: None };
		}
	}

	// 2. Use first model from scoped models (skip if continuing/resuming)
	if !scoped_models.is_empty() && !is_continuing {
		let scoped_model = &scoped_models[0];
		let per_model = model_thinking_levels
			.and_then(|levels| levels.get(&format!("{}/{}", scoped_model.model.provider(), scoped_model.model.id())))
			.copied();
		return InitialModelResult {
			model: Some(scoped_model.model.clone()),
			thinking_level: scoped_model
				.thinking_level
				.or(per_model)
				.or(default_thinking_level)
				.unwrap_or(DEFAULT_THINKING_LEVEL),
			fallback_message: None,
		};
	}

	// 3. Try saved default from settings if auth is configured.
	if let (Some(default_provider), Some(default_model_id)) = (default_provider, default_model_id) {
		if let Some(found) = model_runtime.get_model(default_provider, default_model_id) {
			if model_runtime.has_configured_auth(found.provider()) {
				let per_model = model_thinking_levels
					.and_then(|levels| levels.get(&format!("{default_provider}/{default_model_id}")))
					.copied();
				let thinking_level = if let Some(per_model) = per_model {
					per_model
				} else if let Some(default_thinking_level) = default_thinking_level {
					default_thinking_level
				} else {
					DEFAULT_THINKING_LEVEL
				};
				return InitialModelResult { model: Some(found), thinking_level, fallback_message: None };
			}
		}
	}

	// 4. Try first available model with valid API key
	let available_models = model_runtime.get_available_snapshot();

	if !available_models.is_empty() {
		// Try to find a default model from known providers
		for provider in default_model_providers() {
			let Some(default_id) = default_model_for_provider(provider) else { continue };
			if let Some(matched) = available_models.iter().find(|m| m.provider() == provider && m.id() == default_id) {
				return InitialModelResult {
					model: Some(matched.clone()),
					thinking_level: DEFAULT_THINKING_LEVEL,
					fallback_message: None,
				};
			}
		}

		// If no default found, use first available
		return InitialModelResult {
			model: Some(available_models[0].clone()),
			thinking_level: DEFAULT_THINKING_LEVEL,
			fallback_message: None,
		};
	}

	// 5. No model found
	InitialModelResult { model: None, thinking_level: DEFAULT_THINKING_LEVEL, fallback_message: None }
}

/// Result of [`restore_model_from_session`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RestoreModelResult {
	pub model: Option<Model>,
	pub fallback_message: Option<String>,
}

/// Restore model from session, with fallback to available models
/// (`restoreModelFromSession`).
pub fn restore_model_from_session(
	saved_provider: &str,
	saved_model_id: &str,
	current_model: Option<&Model>,
	should_print_messages: bool,
	model_runtime: &dyn ModelRuntimeSource,
) -> RestoreModelResult {
	let restored_model = model_runtime.get_model(saved_provider, saved_model_id);

	// Check if restored model exists and still has auth configured
	let has_configured_auth = restored_model
		.as_ref()
		.map(|model| model_runtime.has_configured_auth(model.provider()))
		.unwrap_or(false);

	if let Some(restored) = &restored_model {
		if has_configured_auth {
			if should_print_messages {
				println!("Restored model: {saved_provider}/{saved_model_id}");
			}
			return RestoreModelResult { model: Some(restored.clone()), fallback_message: None };
		}
	}

	// Model not found or no API key - fall back
	let reason = if restored_model.is_none() { "model no longer exists" } else { "no auth configured" };

	if should_print_messages {
		eprintln!("Warning: Could not restore model {saved_provider}/{saved_model_id} ({reason}).");
	}

	// If we already have a model, use it as fallback
	if let Some(current_model) = current_model {
		if should_print_messages {
			println!("Falling back to: {}/{}", current_model.provider(), current_model.id());
		}
		return RestoreModelResult {
			fallback_message: Some(format!(
				"Could not restore model {saved_provider}/{saved_model_id} ({reason}). Using {}/{}.",
				current_model.provider(),
				current_model.id()
			)),
			model: Some(current_model.clone()),
		};
	}

	// Try to find any available model
	let available_models = model_runtime.get_available_snapshot();

	if !available_models.is_empty() {
		// Try to find a default model from known providers
		let mut fallback_model: Option<Model> = None;
		for provider in default_model_providers() {
			let Some(default_id) = default_model_for_provider(provider) else { continue };
			if let Some(matched) = available_models.iter().find(|m| m.provider() == provider && m.id() == default_id) {
				fallback_model = Some(matched.clone());
				break;
			}
		}

		// If no default found, use first available
		let fallback_model = fallback_model.unwrap_or_else(|| available_models[0].clone());

		if should_print_messages {
			println!("Falling back to: {}/{}", fallback_model.provider(), fallback_model.id());
		}

		return RestoreModelResult {
			model: Some(fallback_model.clone()),
			fallback_message: Some(format!(
				"Could not restore model {saved_provider}/{saved_model_id} ({reason}). Using {}/{}.",
				fallback_model.provider(),
				fallback_model.id()
			)),
		};
	}

	// No models available
	RestoreModelResult { model: None, fallback_message: None }
}

#[cfg(test)]
#[path = "model_resolver_tests.rs"]
mod model_resolver_tests;
