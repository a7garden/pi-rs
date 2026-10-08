//! Port of `upstream/pi/packages/coding-agent/test/model-registry.test.ts` —
//! every suite that does not construct a full `AgentSession` or require live
//! network (none do; the one unported upstream case is noted below).
//!
//! Skipped upstream suites:
//! - "streamSimple overlays do not mutate the global compat API registry" —
//!   the port has no global API registry; `streamSimple` overrides live on
//!   the composed provider only, so there is nothing to corrupt.
//!
//! Fixtures mirror `test/model-runtime-test-utils.ts::createModelRegistry`:
//! `ModelRuntime::create` with an [`InMemoryCredentialStore`], a temp
//! models.json path, an in-memory models store, and `allow_model_network:
//! false`; `refresh_on_create` stays enabled like the upstream helper (the
//! create-time refresh runs with network disabled). Direct `refresh` calls
//! pass `allow_network: Some(false)` to stay offline.

use std::collections::BTreeMap;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use pi_ai::auth::types::{Credential, CredentialStore as _, OAuthCredentials};
use pi_ai::models::{get_supported_thinking_levels, ModelsRefreshOptions, ModelsRefreshResult};
use pi_ai::compat::get_builtin_models;
use pi_ai::types::{
	AnyModel, MaxTokensField, Model, ModelCost, ModelCostRates, ModelInput, ModelThinkingLevel, ProviderEnv,
	ProviderHeaders, SimpleStreamOptions, ThinkingLevelMap, TranscriptContext,
};
use serde_json::{json, Value as JsonValue};

use crate::core::extensions::types::{
	ProviderChatModelConfig, ProviderConfig, ProviderModelConfig, ProviderModelConfigBase, ProviderOAuthConfig,
	StreamSimpleFn,
};
use crate::core::model_registry::{ModelRegistry, ResolvedRequestAuth};
use crate::core::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::core::models_store_wrapper::InMemoryCodingAgentModelsStore;
use crate::core::provider_composer::{AuthStatus, AuthStatusSource};
use crate::core::resolve_config_value::clear_config_value_cache;
use crate::core::virtual_models::{ModelRoute, RouteRequest, VirtualModelDefinition};

/// Serializes process-env mutation across the parallel test binary (upstream
/// tests mutate `process.env` inline; the port follows the
/// `resolve_config_value_tests` house pattern).
static ENV_LOCK: LazyLock<parking_lot::Mutex<()>> = LazyLock::new(|| parking_lot::Mutex::new(()));

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Temp directory removed on drop (upstream `afterEach` `rmSync`).
struct TempDir(PathBuf);

impl Drop for TempDir {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}

fn temp_dir(label: &str) -> TempDir {
	let dir = std::env::temp_dir().join(format!(
		"pi-model-registry-{label}-{}-{}",
		std::process::id(),
		DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
	));
	std::fs::create_dir_all(&dir).unwrap();
	TempDir(dir)
}

/// Upstream `beforeEach` state: temp dir, models.json path, in-memory auth
/// storage (`AuthStorage.inMemory()` — its reads resolve `$VAR` templates
/// from the credential env, which the plain pi-ai store does not).
struct Fixture {
	models_json_path: PathBuf,
	credentials: Arc<crate::core::auth_storage::AuthStorage>,
	_dir: TempDir,
}

impl Fixture {
	/// Upstream `writeModelsJson`/`writeRawModelsJson` — the providers record
	/// is stored under the top-level `providers` key.
	fn write_models_json(&self, providers: JsonValue) {
		std::fs::write(&self.models_json_path, serde_json::to_vec(&json!({ "providers": providers })).unwrap())
			.unwrap();
	}

	/// `authStorage.modify` — store one credential for a provider.
	async fn store_credential(&self, provider: &str, credential: Credential) {
		let provider = provider.to_string();
		self.credentials
			.modify(
				&provider,
				Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
				None,
			)
			.await
			.unwrap();
	}

	/// `createModelRegistry` — refresh with network disabled.
	async fn registry(&self) -> ModelRegistry {
		let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
			credentials: Some(self.credentials.clone()),
			models_path: Some(Some(self.models_json_path.to_string_lossy().into_owned())),
			models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::new())),
			allow_model_network: false,
			refresh_on_create: true,
			..Default::default()
		})
		.await
		.expect("ModelRuntime::create succeeds");
		ModelRegistry::new(runtime)
	}
}

fn fixture() -> Fixture {
	let dir = temp_dir("fixture");
	let models_json_path = dir.0.join("models.json");
	Fixture {
		models_json_path,
		credentials: Arc::new(crate::core::auth_storage::AuthStorage::in_memory(Default::default())),
		_dir: dir,
	}
}

fn fixture_with_models_json(providers: JsonValue) -> Fixture {
	let fixture = fixture();
	fixture.write_models_json(providers);
	fixture
}

/// Offline `registry.refresh()` (upstream bare `await registry.refresh()`).
async fn refresh_offline(registry: &ModelRegistry) -> ModelsRefreshResult {
	registry
		.refresh(Some(ModelsRefreshOptions { allow_network: Some(false), ..Default::default() }))
		.await
}

/// The spawned post-registration refresh rebuilds the models map non-atomically
/// (clear, then re-add), so an immediate synchronous read can race it — an
/// interleaving upstream JS cannot produce (single-threaded, sync asserts).
/// Bounded retry until the composition settles.
async fn eventually<T>(mut read: impl FnMut() -> Option<T>) -> T {
	for _ in 0..500 {
		if let Some(value) = read() {
			return value;
		}
		tokio::time::sleep(std::time::Duration::from_millis(10)).await;
	}
	panic!("registry read did not settle within 5s");
}

fn get_models_for_provider(registry: &ModelRegistry, provider: &str) -> Vec<Model> {
	registry.get_all().into_iter().filter(|model| model.provider() == provider).collect()
}

// ---------------------------------------------------------------------------
// Config helpers (upstream `providerConfig`/`overrideConfig`/`writeRawModelsJson`)
// ---------------------------------------------------------------------------

fn model_cost_free() -> ModelCost {
	ModelCost { rates: ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 }, tiers: None }
}

fn chat_model_config(id: &str) -> ProviderChatModelConfig {
	ProviderChatModelConfig {
		base: ProviderModelConfigBase {
			id: id.to_string(),
			name: id.to_string(),
			api: None,
			base_url: None,
			input: vec![ModelInput::Text],
			input_limits: None,
			cost: Some(model_cost_free()),
			headers: None,
		},
		model_type: None,
		api: None,
		reasoning: false,
		thinking_level_map: None,
		prompt_cache: None,
		context_window: 100_000,
		max_tokens: 8000,
		sampling_params: None,
		compat: None,
	}
}

/// Upstream `providerConfig` (struct form, for `registerProvider` calls).
fn provider_config(base_url: &str, model_ids: &[&str], api: Option<&str>) -> ProviderConfig {
	ProviderConfig {
		base_url: Some(base_url.to_string()),
		api_key: Some("test-key".to_string()),
		api: api.map(str::to_string),
		models: Some(model_ids.iter().map(|id| ProviderModelConfig::Chat(chat_model_config(id))).collect()),
		..Default::default()
	}
}

fn chat_model_json(id: &str) -> JsonValue {
	json!({
		"id": id, "name": id, "reasoning": false, "input": ["text"],
		"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
		"contextWindow": 100000, "maxTokens": 8000
	})
}

/// Upstream `providerConfig` (models.json form).
fn provider_config_json(base_url: &str, api: &str, model_ids: &[&str]) -> JsonValue {
	json!({
		"baseUrl": base_url,
		"apiKey": "test-key",
		"api": api,
		"models": model_ids.iter().map(|id| chat_model_json(id)).collect::<Vec<_>>(),
	})
}

/// Upstream `overrideConfig` — baseUrl-only (no custom models).
fn override_config(base_url: &str) -> JsonValue {
	json!({"baseUrl": base_url})
}

/// Upstream `providerWithApiKey`.
fn provider_with_api_key(api_key: &str) -> JsonValue {
	let mut provider = provider_config_json("https://example.com/v1", "anthropic-messages", &["test-model"]);
	provider["apiKey"] = json!(api_key);
	provider["models"][0]["name"] = json!("Test Model");
	provider
}

fn fixture_with_api_key(api_key: &str) -> Fixture {
	fixture_with_models_json(json!({ "custom-provider": provider_with_api_key(api_key) }))
}

fn status(configured: bool, source: Option<AuthStatusSource>, label: Option<&str>) -> AuthStatus {
	AuthStatus { configured, source, label: label.map(str::to_string) }
}

fn unwrap_ok(auth: ResolvedRequestAuth) -> (Option<String>, Option<ProviderHeaders>, Option<String>, Option<ProviderEnv>) {
	match auth {
		ResolvedRequestAuth::Ok { api_key, headers, base_url, env } => (api_key, headers, base_url, env),
		ResolvedRequestAuth::Error { error } => panic!("expected ok auth, got error: {error}"),
	}
}

fn header_value<'a>(headers: &'a Option<ProviderHeaders>, name: &str) -> Option<&'a str> {
	headers.as_ref().and_then(|headers| headers.get(name)).and_then(|value| value.as_deref())
}

fn now_ms() -> i64 {
	std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

/// Upstream `toShPath` — shell-safe path literal.
fn to_sh_path(value: &Path) -> String {
	value.to_string_lossy().replace('\\', "/").replace('"', "\\\"")
}

fn write_file(path: &Path, content: &str) {
	std::fs::write(path, content).unwrap();
}

fn read_file(path: &Path) -> String {
	std::fs::read_to_string(path).unwrap()
}

/// The text behind a thrown registration error (`ThrownValue` message).
fn thrown_text(value: &pi_ai::utils::thrown::ThrownValue) -> String {
	match value {
		pi_ai::utils::thrown::ThrownValue::Text(text) => text.clone(),
		pi_ai::utils::thrown::ThrownValue::Error(error) => error.to_string(),
	}
}

/// One env var set/restored around `test`, panic-safe (upstream try/finally).
fn run_with_env(name: &str, value: &str, test: impl FnOnce()) {
	run_with_envs(&[(name, value)], test);
}

/// Several env vars set/restored around `test`, panic-safe.
fn run_with_envs(env: &[(&str, &str)], test: impl FnOnce()) {
	let _guard = ENV_LOCK.lock();
	let previous: Vec<(&str, Option<String>)> =
		env.iter().map(|(name, _)| (*name, std::env::var(name).ok())).collect();
	for (name, value) in env {
		std::env::set_var(name, value);
	}
	let result = catch_unwind(AssertUnwindSafe(test));
	for ((name, _), (_, previous)) in env.iter().zip(previous) {
		match previous {
			Some(previous) => std::env::set_var(name, previous),
			None => std::env::remove_var(name),
		}
	}
	result.unwrap();
}

/// Every test body runs on its own multi-thread tokio runtime (the facade
/// spawns background refreshes through `tokio::spawn`).
fn block<F: Future>(future: F) -> F::Output {
	tokio::runtime::Runtime::new().unwrap().block_on(future)
}

fn oauth_config(name: &str) -> ProviderOAuthConfig {
	ProviderOAuthConfig {
		name: name.to_string(),
		is_subscription: None,
		uses_callback_server: None,
		login: Arc::new(|_callbacks| {
			Box::pin(async {
				Ok(OAuthCredentials {
					refresh: "refresh-token".to_string(),
					access: "access-token".to_string(),
					expires: now_ms() + 60_000,
					extra: BTreeMap::new(),
				})
			})
		}),
		refresh_token: Arc::new(|credentials, _signal| Box::pin(async move { Ok(credentials) })),
		get_api_key: Arc::new(|credentials: &OAuthCredentials| credentials.access.clone()),
		modify_models: None,
	}
}

// ---------------------------------------------------------------------------
// "baseUrl override (no custom models)"
// ---------------------------------------------------------------------------

#[test]
fn overriding_base_url_keeps_all_builtin_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": override_config("https://my-proxy.example.com/v1") }));
		let registry = fixture.registry().await;
		let anthropic_models = get_models_for_provider(&registry, "anthropic");
		assert!(anthropic_models.len() > 1);
		assert!(anthropic_models.iter().any(|model| model.id().contains("claude")));
	});
}

#[test]
fn overriding_base_url_changes_url_on_all_builtin_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": override_config("https://my-proxy.example.com/v1") }));
		let registry = fixture.registry().await;
		for model in get_models_for_provider(&registry, "anthropic") {
			assert_eq!(model.base.base_url, "https://my-proxy.example.com/v1");
		}
	});
}

#[test]
fn overriding_headers_resolves_at_request_time() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": {
			"baseUrl": "https://my-proxy.example.com/v1",
			"headers": {"X-Custom-Header": "custom-value"}
		} }));
		let registry = fixture.registry().await;
		for model in get_models_for_provider(&registry, "anthropic") {
			let (_, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
			assert_eq!(header_value(&headers, "X-Custom-Header"), Some("custom-value"));
		}
	});
}

#[test]
fn headers_only_override_resolves_at_request_time() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": {
			"headers": {"X-Custom-Header": "custom-value"}
		} }));
		let registry = fixture.registry().await;
		assert_eq!(registry.get_error(), None);
		for model in get_models_for_provider(&registry, "anthropic") {
			let (_, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
			assert_eq!(header_value(&headers, "X-Custom-Header"), Some("custom-value"));
		}
	});
}

#[test]
fn unconfigured_compatibility_auth_includes_static_model_headers() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;
		let mut model = registry.get_all()[0].clone();
		model.base.provider = "missing-provider".to_string();
		model.base.headers = Some(BTreeMap::from([("X-Static-Model".to_string(), "static-value".to_string())]));

		let auth = registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await;
		let (api_key, headers, base_url, env) = unwrap_ok(auth);
		assert_eq!(api_key, None);
		assert_eq!(header_value(&headers, "X-Static-Model"), Some("static-value"));
		assert_eq!(headers.as_ref().map(|headers| headers.len()), Some(1));
		assert_eq!(base_url, None);
		assert_eq!(env, None);
	});
}

#[test]
fn base_url_only_override_does_not_affect_other_providers() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": override_config("https://my-proxy.example.com/v1") }));
		let registry = fixture.registry().await;
		let google_models = get_models_for_provider(&registry, "google");
		assert!(google_models.len() > 0);
		assert_ne!(google_models[0].base.base_url, "https://my-proxy.example.com/v1");
	});
}

#[test]
fn can_mix_base_url_override_and_models_merge() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"anthropic": override_config("https://anthropic-proxy.example.com/v1"),
			"google": provider_config_json("https://google-proxy.example.com/v1", "google-generative-ai", &["gemini-custom"]),
		}));
		let registry = fixture.registry().await;

		let anthropic_models = get_models_for_provider(&registry, "anthropic");
		assert!(anthropic_models.len() > 1);
		assert_eq!(anthropic_models[0].base.base_url, "https://anthropic-proxy.example.com/v1");

		let google_models = get_models_for_provider(&registry, "google");
		assert!(google_models.len() > 1);
		assert!(google_models.iter().any(|model| model.id() == "gemini-custom"));
	});
}

#[test]
fn refresh_picks_up_base_url_override_changes() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": override_config("https://first-proxy.example.com/v1") }));
		let registry = fixture.registry().await;
		assert_eq!(
			get_models_for_provider(&registry, "anthropic")[0].base.base_url,
			"https://first-proxy.example.com/v1"
		);

		fixture.write_models_json(json!({ "anthropic": override_config("https://second-proxy.example.com/v1") }));
		refresh_offline(&registry).await;

		assert_eq!(
			get_models_for_provider(&registry, "anthropic")[0].base.base_url,
			"https://second-proxy.example.com/v1"
		);
	});
}

// ---------------------------------------------------------------------------
// "custom models merge behavior"
// ---------------------------------------------------------------------------

#[test]
fn builtin_provider_custom_models_inherit_api_and_base_url() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"models": [{
				"id": "fake-provider/fake-model",
				"name": "Fake model",
				"reasoning": true,
				"input": ["text"]
			}]
		} }));
		let registry = fixture.registry().await;
		assert_eq!(registry.get_error(), None);

		let model = registry.find("openrouter", "fake-provider/fake-model").expect("custom model registered");
		assert_eq!(model.api(), "openai-completions");
		assert_eq!(model.base.base_url, "https://openrouter.ai/api/v1");
	});
}

#[test]
fn non_builtin_provider_custom_models_still_require_base_url() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "my-custom-provider": {
			"apiKey": "test-key",
			"models": [{
				"id": "my-model",
				"api": "openai-completions",
				"reasoning": false,
				"input": ["text"]
			}]
		} }));
		let registry = fixture.registry().await;
		assert!(registry.get_error().unwrap().contains("baseUrl"));
	});
}

#[test]
fn reports_every_provider_composition_error() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"broken-one": {"api": "openai-completions", "models": [{"id": "one"}]},
			"broken-two": {"api": "openai-completions", "models": [{"id": "two"}]}
		}));
		let registry = fixture.registry().await;
		let error = registry.get_error().unwrap();
		assert!(error.contains("Provider \"broken-one\""), "error: {error}");
		assert!(error.contains("Provider \"broken-two\""), "error: {error}");
	});
}

#[test]
fn custom_provider_with_same_name_as_builtin_merges_with_builtin_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"anthropic": provider_config_json("https://my-proxy.example.com/v1", "anthropic-messages", &["claude-custom"])
		}));
		let registry = fixture.registry().await;
		let anthropic_models = get_models_for_provider(&registry, "anthropic");
		assert!(anthropic_models.len() > 1);
		assert!(anthropic_models.iter().any(|model| model.id() == "claude-custom"));
		assert!(anthropic_models.iter().any(|model| model.id().contains("claude")));
	});
}

#[test]
fn custom_model_with_same_id_replaces_builtin_model_by_id() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"openrouter": provider_config_json("https://my-proxy.example.com/v1", "openai-completions", &["anthropic/claude-sonnet-4"])
		}));
		let registry = fixture.registry().await;
		let sonnet_models: Vec<Model> = get_models_for_provider(&registry, "openrouter")
			.into_iter()
			.filter(|model| model.id() == "anthropic/claude-sonnet-4")
			.collect();
		assert_eq!(sonnet_models.len(), 1);
		assert_eq!(sonnet_models[0].base.base_url, "https://my-proxy.example.com/v1");
	});
}

#[test]
fn custom_provider_with_same_name_as_builtin_does_not_affect_other_builtins() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"anthropic": provider_config_json("https://my-proxy.example.com/v1", "anthropic-messages", &["claude-custom"])
		}));
		let registry = fixture.registry().await;
		assert!(get_models_for_provider(&registry, "google").len() > 0);
		assert!(get_models_for_provider(&registry, "openai").len() > 0);
	});
}

#[test]
fn provider_level_base_url_applies_to_both_builtin_and_custom_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"anthropic": provider_config_json("https://merged-proxy.example.com/v1", "anthropic-messages", &["claude-custom"])
		}));
		let registry = fixture.registry().await;
		for model in get_models_for_provider(&registry, "anthropic") {
			assert_eq!(model.base.base_url, "https://merged-proxy.example.com/v1");
		}
	});
}

#[test]
fn provider_level_compat_applies_to_custom_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "demo": {
			"baseUrl": "https://example.com/v1",
			"apiKey": "DEMO_KEY",
			"api": "openai-completions",
			"compat": {"supportsUsageInStreaming": false, "maxTokensField": "max_tokens"},
			"models": [{
				"id": "demo-model", "reasoning": false, "input": ["text"],
				"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": 1000, "maxTokens": 100
			}]
		} }));
		let registry = fixture.registry().await;
		let compat = registry.find("demo", "demo-model").expect("demo model").compat.expect("compat");
		assert_eq!(compat.supports_usage_in_streaming, Some(false));
		assert_eq!(compat.max_tokens_field, Some(MaxTokensField::MaxTokens));
	});
}

#[test]
fn model_level_compat_overrides_provider_level_compat_for_custom_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "demo": {
			"baseUrl": "https://example.com/v1",
			"apiKey": "DEMO_KEY",
			"api": "openai-completions",
			"compat": {"supportsUsageInStreaming": false, "maxTokensField": "max_tokens"},
			"models": [{
				"id": "demo-model", "reasoning": false, "input": ["text"],
				"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": 1000, "maxTokens": 100,
				"compat": {"supportsUsageInStreaming": true, "maxTokensField": "max_completion_tokens"}
			}]
		} }));
		let registry = fixture.registry().await;
		let compat = registry.find("demo", "demo-model").expect("demo model").compat.expect("compat");
		assert_eq!(compat.supports_usage_in_streaming, Some(true));
		assert_eq!(compat.max_tokens_field, Some(MaxTokensField::MaxCompletionTokens));
	});
}

#[test]
fn provider_level_compat_applies_to_builtin_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"compat": {"supportsUsageInStreaming": false, "supportsStrictMode": false}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		assert!(models.len() > 0);
		for model in models {
			let compat = model.compat.as_ref().expect("compat");
			assert_eq!(compat.supports_usage_in_streaming, Some(false));
			assert_eq!(compat.supports_strict_mode, Some(false));
		}
	});
}

#[test]
fn model_schema_accepts_thinking_level_map_and_compat_schema_accepts_strict_mode_and_cache_control() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "demo": {
			"baseUrl": "https://example.com/v1",
			"apiKey": "DEMO_KEY",
			"api": "openai-completions",
			"models": [{
				"id": "demo-model", "reasoning": true, "input": ["text"],
				"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": 1000, "maxTokens": 100,
				"thinkingLevelMap": {"minimal": null, "high": "max"},
				"compat": {"supportsStrictMode": false, "cacheControlFormat": "anthropic"}
			}]
		} }));
		let registry = fixture.registry().await;
		let model = registry.find("demo", "demo-model").expect("demo model");
		assert_eq!(registry.get_error(), None);
		let expected_map: ThinkingLevelMap = [
			(ModelThinkingLevel::Minimal, None),
			(ModelThinkingLevel::High, Some("max".to_string())),
		]
		.into_iter()
		.collect();
		assert_eq!(model.thinking_level_map.as_ref(), Some(&expected_map));
		let compat = model.compat.expect("compat");
		assert_eq!(compat.supports_strict_mode, Some(false));
		assert_eq!(serde_json::to_value(compat.cache_control_format).unwrap(), json!("anthropic"));
	});
}

#[test]
fn compat_schema_accepts_chat_template_thinking_configuration() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "demo": {
			"baseUrl": "https://example.com/v1",
			"apiKey": "DEMO_KEY",
			"api": "openai-completions",
			"models": [
				{
					"id": "kwargs-model", "reasoning": true, "input": ["text"],
					"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
					"contextWindow": 1000, "maxTokens": 100,
					"compat": {
						"thinkingFormat": "chat-template",
						"chatTemplateKwargs": {
							"preserve_thinking": true,
							"thinking": {"$var": "thinking.enabled"}
						}
					}
				},
				{
					"id": "args-model", "reasoning": true, "input": ["text"],
					"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
					"contextWindow": 1000, "maxTokens": 100,
					"compat": {
						"thinkingFormat": "baseten",
						"chatTemplateArgs": {
							"enable_thinking": {"$var": "thinking.enabled"}
						}
					}
				}
			]
		} }));
		let registry = fixture.registry().await;
		let kwargs_compat = registry.find("demo", "kwargs-model").unwrap().compat.unwrap();
		let args_compat = registry.find("demo", "args-model").unwrap().compat.unwrap();

		assert_eq!(registry.get_error(), None);
		assert_eq!(
			serde_json::to_value(kwargs_compat.thinking_format).unwrap(),
			json!("chat-template")
		);
		assert_eq!(
			serde_json::to_value(kwargs_compat.chat_template_kwargs).unwrap(),
			json!({"preserve_thinking": true, "thinking": {"$var": "thinking.enabled"}})
		);
		assert_eq!(serde_json::to_value(args_compat.thinking_format).unwrap(), json!("baseten"));
		assert_eq!(
			serde_json::to_value(args_compat.chat_template_args).unwrap(),
			json!({"enable_thinking": {"$var": "thinking.enabled"}})
		);
	});
}

#[test]
fn compat_schema_accepts_anthropic_eager_tool_input_streaming_flag() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "demo": {
			"baseUrl": "https://example.com",
			"apiKey": "DEMO_KEY",
			"api": "anthropic-messages",
			"compat": {"supportsEagerToolInputStreaming": false},
			"models": [{
				"id": "demo-model", "reasoning": true, "input": ["text"],
				"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": 1000, "maxTokens": 100
			}]
		} }));
		let registry = fixture.registry().await;
		let compat = registry.find("demo", "demo-model").unwrap().compat.unwrap();
		assert_eq!(registry.get_error(), None);
		assert_eq!(compat.supports_eager_tool_input_streaming, Some(false));
	});
}

#[test]
fn compat_schema_accepts_long_cache_retention_flag() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "demo": {
			"baseUrl": "https://example.com",
			"apiKey": "DEMO_KEY",
			"api": "anthropic-messages",
			"compat": {"supportsLongCacheRetention": false},
			"models": [{
				"id": "demo-model", "reasoning": true, "input": ["text"],
				"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": 1000, "maxTokens": 100
			}]
		} }));
		let registry = fixture.registry().await;
		let compat = registry.find("demo", "demo-model").unwrap().compat.unwrap();
		assert_eq!(registry.get_error(), None);
		assert_eq!(compat.supports_long_cache_retention, Some(false));
	});
}

#[test]
fn model_level_base_url_overrides_provider_level_base_url_for_custom_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "opencode-go": {
			"baseUrl": "https://opencode.ai/zen/go/v1",
			"apiKey": "TEST_KEY",
			"models": [
				{
					"id": "minimax-m2.5",
					"api": "anthropic-messages",
					"baseUrl": "https://opencode.ai/zen/go",
					"reasoning": true,
					"input": ["text"],
					"cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.03, "cacheWrite": 0},
					"contextWindow": 204800,
					"maxTokens": 131072
				},
				{
					"id": "glm-5",
					"api": "openai-completions",
					"reasoning": true,
					"input": ["text"],
					"cost": {"input": 1, "output": 3.2, "cacheRead": 0.2, "cacheWrite": 0},
					"contextWindow": 204800,
					"maxTokens": 131072
				}
			]
		} }));
		let registry = fixture.registry().await;
		let minimax = registry.find("opencode-go", "minimax-m2.5").unwrap();
		let glm5 = registry.find("opencode-go", "glm-5").unwrap();
		assert_eq!(minimax.base.base_url, "https://opencode.ai/zen/go");
		assert_eq!(glm5.base.base_url, "https://opencode.ai/zen/go/v1");
	});
}

#[test]
fn model_overrides_still_apply_when_provider_also_defines_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"baseUrl": "https://my-proxy.example.com/v1",
			"apiKey": "OPENROUTER_API_KEY",
			"api": "openai-completions",
			"models": [{
				"id": "custom/openrouter-model",
				"name": "Custom OpenRouter Model",
				"reasoning": false,
				"input": ["text"],
				"cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": 128000,
				"maxTokens": 16384
			}],
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {"name": "Overridden Built-in Sonnet"}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		assert!(models.iter().any(|model| model.id() == "custom/openrouter-model"));
		assert!(models
			.iter()
			.any(|model| model.id() == "anthropic/claude-sonnet-4" && model.base.name == "Overridden Built-in Sonnet"));
	});
}

#[test]
fn refresh_reloads_merged_custom_models_from_disk() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"anthropic": provider_config_json("https://first-proxy.example.com/v1", "anthropic-messages", &["claude-custom"])
		}));
		let registry = fixture.registry().await;
		assert!(get_models_for_provider(&registry, "anthropic").iter().any(|model| model.id() == "claude-custom"));

		fixture.write_models_json(json!({
			"anthropic": provider_config_json("https://second-proxy.example.com/v1", "anthropic-messages", &["claude-custom-2"])
		}));
		refresh_offline(&registry).await;

		let anthropic_models = get_models_for_provider(&registry, "anthropic");
		assert!(anthropic_models.iter().all(|model| model.id() != "claude-custom"));
		assert!(anthropic_models.iter().any(|model| model.id() == "claude-custom-2"));
		assert!(anthropic_models.iter().any(|model| model.id().contains("claude")));
	});
}

#[test]
fn removing_custom_models_from_models_json_keeps_builtin_provider_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"anthropic": provider_config_json("https://proxy.example.com/v1", "anthropic-messages", &["claude-custom"])
		}));
		let registry = fixture.registry().await;
		assert!(get_models_for_provider(&registry, "anthropic").iter().any(|model| model.id() == "claude-custom"));

		fixture.write_models_json(json!({}));
		refresh_offline(&registry).await;

		let anthropic_models = get_models_for_provider(&registry, "anthropic");
		assert!(anthropic_models.iter().all(|model| model.id() != "claude-custom"));
		assert!(anthropic_models.iter().any(|model| model.id().contains("claude")));
	});
}

// ---------------------------------------------------------------------------
// "modelOverrides (per-model customization)"
// ---------------------------------------------------------------------------

#[test]
fn model_override_applies_to_a_single_builtin_model() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Custom Sonnet Name"}}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");

		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		assert_eq!(sonnet.base.name, "Custom Sonnet Name");

		let opus = models.iter().find(|model| model.id() == "anthropic/claude-opus-4.1").unwrap();
		assert_ne!(opus.base.name, "Custom Sonnet Name");
	});
}

#[test]
fn anthropic_model_override_replaces_allowed_fallback_metadata() {
	block(async {
		let allowed_fallback_models = json!([
			{"provider": "anthropic", "model": "claude-opus-5", "cost": {"input": 5.0, "output": 25.0, "cacheRead": 0.5, "cacheWrite": 6.25}},
			{"provider": "anthropic", "model": "claude-opus-4-8", "cost": {"input": 4.0, "output": 20.0, "cacheRead": 0.4, "cacheWrite": 5.0}}
		]);
		let fixture = fixture_with_models_json(json!({ "anthropic": {
			"modelOverrides": {
				"claude-fable-5": {"compat": {"allowedFallbackModels": allowed_fallback_models}}
			}
		} }));
		let registry = fixture.registry().await;
		let compat = registry.find("anthropic", "claude-fable-5").unwrap().compat.unwrap();

		assert_eq!(registry.get_error(), None);
		assert_eq!(
			serde_json::to_value(compat.allowed_fallback_models).unwrap(),
			allowed_fallback_models
		);
	});
}

#[test]
fn empty_allowed_fallback_model_override_disables_server_side_fallback() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "anthropic": {
			"modelOverrides": {
				"claude-fable-5": {"compat": {"allowedFallbackModels": []}}
			}
		} }));
		let registry = fixture.registry().await;
		let compat = registry.find("anthropic", "claude-fable-5").unwrap().compat.unwrap();

		assert_eq!(registry.get_error(), None);
		assert_eq!(serde_json::to_value(compat.allowed_fallback_models).unwrap(), json!([]));
	});
}

#[test]
fn custom_models_and_model_overrides_carry_sampling_params() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"baseUrl": "https://my-proxy.example.com/v1",
			"api": "openai-completions",
			"models": [{
				"id": "custom/sampling-model",
				"samplingParams": {"temperature": 1, "top_p": 0.95, "top_k": 0},
				"samplingParamsByThinkingLevel": {
					"low": {"temperature": 0.6, "top_p": 0.95},
					"high": {"temperature": 0.8}
				}
			}],
			"modelOverrides": {
				"custom/sampling-model": {
					"samplingParamsByThinkingLevel": {
						"low": {"temperature": 0.5, "top_k": 20},
						"max": {"temperature": 1}
					}
				},
				"anthropic/claude-sonnet-4": {
					"samplingParams": {"top_p": 0.9},
					"samplingParamsByThinkingLevel": {"high": {"temperature": 0.8}}
				}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");

		let custom = models.iter().find(|model| model.id() == "custom/sampling-model").unwrap();
		assert_eq!(custom.sampling_params.as_ref().unwrap(), json!({"temperature": 1, "top_p": 0.95, "top_k": 0}).as_object().unwrap());
		let expected_custom_levels: BTreeMap<ModelThinkingLevel, serde_json::Map<String, JsonValue>> = [
			(ModelThinkingLevel::Low, json!({"temperature": 0.5, "top_p": 0.95, "top_k": 20})),
			(ModelThinkingLevel::High, json!({"temperature": 0.8})),
			(ModelThinkingLevel::Max, json!({"temperature": 1})),
		]
		.into_iter()
		.map(|(level, value)| (level, value.as_object().unwrap().clone()))
		.collect();
		assert_eq!(custom.sampling_params_by_thinking_level.as_ref(), Some(&expected_custom_levels));

		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		assert_eq!(sonnet.sampling_params.as_ref().unwrap(), json!({"top_p": 0.9}).as_object().unwrap());
		let expected_sonnet_levels: BTreeMap<ModelThinkingLevel, serde_json::Map<String, JsonValue>> = [
			(ModelThinkingLevel::High, json!({"temperature": 0.8})),
		]
		.into_iter()
		.map(|(level, value)| (level, value.as_object().unwrap().clone()))
		.collect();
		assert_eq!(sonnet.sampling_params_by_thinking_level.as_ref(), Some(&expected_sonnet_levels));

		let opus = models.iter().find(|model| model.id() == "anthropic/claude-opus-4.1").unwrap();
		assert_eq!(opus.sampling_params, None);
		assert_eq!(opus.sampling_params_by_thinking_level, None);
	});
}

#[test]
fn custom_model_and_model_override_carry_prompt_cache_lifetimes() {
	block(async {
		let fixture = fixture_with_models_json(json!({
			"openrouter": {
				"baseUrl": "https://my-proxy.example.com/v1",
				"api": "openai-completions",
				"models": [{"id": "custom/cached-model", "promptCache": {"short": 120}}],
				"modelOverrides": {
					"anthropic/claude-sonnet-4": {"promptCache": {"short": 300}}
				}
			},
			"anthropic": {
				"modelOverrides": {
					"claude-sonnet-4-6": {"promptCache": {"long": 1800}}
				}
			}
		}));
		let registry = fixture.registry().await;
		let openrouter = get_models_for_provider(&registry, "openrouter");

		assert_eq!(registry.get_error(), None);
		let cached = openrouter.iter().find(|model| model.id() == "custom/cached-model").unwrap();
		assert_eq!(serde_json::to_value(cached.prompt_cache.as_ref().unwrap()).unwrap(), json!({"short": 120}));
		let overridden = openrouter
			.iter()
			.find(|model| model.id() == "anthropic/claude-sonnet-4")
			.unwrap();
		assert_eq!(serde_json::to_value(overridden.prompt_cache.as_ref().unwrap()).unwrap(), json!({"short": 300}));
		let opus = openrouter.iter().find(|model| model.id() == "anthropic/claude-opus-4.1").unwrap();
		assert_eq!(opus.prompt_cache, None);
		// Overrides merge per tier with the built-in catalog.
		let sonnet_4_6 = registry.find("anthropic", "claude-sonnet-4-6").unwrap();
		assert_eq!(
			serde_json::to_value(sonnet_4_6.prompt_cache.as_ref().unwrap()).unwrap(),
			json!({"short": 300, "long": 1800})
		);
	});
}

#[test]
fn model_override_deep_merges_image_resize_limits() {
	block(async {
		// Regression test for https://github.com/earendil-works/pi/issues/9631
		let fixture = fixture_with_models_json(json!({ "test": {
			"baseUrl": "https://example.com",
			"apiKey": "test-key",
			"api": "openai-completions",
			"models": [{
				"id": "vision-model",
				"input": ["text", "image"],
				"inputLimits": {
					"maxRequestBytes": 33554432,
					"images": {
						"maxPerRequest": 100,
						"resize": {"maxWidth": 2000, "maxHeight": 2000, "maxBytes": 4718592, "jpegQuality": 80}
					}
				}
			}],
			"modelOverrides": {
				"vision-model": {
					"inputLimits": {
						"images": {"resize": {"maxWidth": 1568, "maxBytes": 524288, "jpegQuality": 75}}
					}
				}
			}
		} }));
		let registry = fixture.registry().await;
		let model = registry.find("test", "vision-model").unwrap();

		assert_eq!(registry.get_error(), None);
		assert_eq!(
			serde_json::to_value(model.base.input_limits).unwrap(),
			json!({
				"maxRequestBytes": 33554432,
				"images": {
					"maxPerRequest": 100,
					"resize": {"maxWidth": 1568, "maxHeight": 2000, "maxBytes": 524288, "jpegQuality": 75}
				}
			})
		);
	});
}

#[test]
fn model_override_with_compat_open_router_routing() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {
					"compat": {"openRouterRouting": {"only": ["amazon-bedrock"]}}
				}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		assert_eq!(
			serde_json::to_value(sonnet.compat.as_ref().unwrap().open_router_routing.clone()).unwrap(),
			json!({"only": ["amazon-bedrock"]})
		);
	});
}

#[test]
fn supports_finish_reason_can_be_configured_at_provider_and_model_levels() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"compat": {"supportsFinishReason": true},
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {"compat": {"supportsFinishReason": false}}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		let opus = models.iter().find(|model| model.id() == "anthropic/claude-opus-4.1").unwrap();
		assert_eq!(sonnet.compat.as_ref().unwrap().supports_finish_reason, Some(false));
		assert_eq!(opus.compat.as_ref().unwrap().supports_finish_reason, Some(true));
	});
}

#[test]
fn model_override_deep_merges_compat_settings() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {
					"compat": {"openRouterRouting": {"order": ["anthropic", "together"]}}
				}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		assert_eq!(
			serde_json::to_value(sonnet.compat.as_ref().unwrap().open_router_routing.clone()).unwrap(),
			json!({"order": ["anthropic", "together"]})
		);
	});
}

#[test]
fn multiple_model_overrides_on_same_provider() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {"compat": {"openRouterRouting": {"only": ["amazon-bedrock"]}}},
				"anthropic/claude-opus-4.1": {"compat": {"openRouterRouting": {"only": ["anthropic"]}}}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		let opus = models.iter().find(|model| model.id() == "anthropic/claude-opus-4.1").unwrap();
		assert_eq!(
			serde_json::to_value(sonnet.compat.as_ref().unwrap().open_router_routing.clone()).unwrap(),
			json!({"only": ["amazon-bedrock"]})
		);
		assert_eq!(
			serde_json::to_value(opus.compat.as_ref().unwrap().open_router_routing.clone()).unwrap(),
			json!({"only": ["anthropic"]})
		);
	});
}

#[test]
fn model_override_combined_with_base_url_override() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"baseUrl": "https://my-proxy.example.com/v1",
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {"name": "Proxied Sonnet"}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		assert_eq!(sonnet.base.base_url, "https://my-proxy.example.com/v1");
		assert_eq!(sonnet.base.name, "Proxied Sonnet");

		let opus = models.iter().find(|model| model.id() == "anthropic/claude-opus-4.1").unwrap();
		assert_eq!(opus.base.base_url, "https://my-proxy.example.com/v1");
		assert_ne!(opus.base.name, "Proxied Sonnet");
	});
}

#[test]
fn model_override_for_non_existent_model_id_is_ignored() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {
				"nonexistent/model-id": {"name": "This should not appear"}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		assert!(models.iter().all(|model| model.id() != "nonexistent/model-id"));
		assert_eq!(registry.get_error(), None);
	});
}

#[test]
fn model_override_can_change_cost_fields_partially() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {"cost": {"input": 99}}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models.iter().find(|model| model.id() == "anthropic/claude-sonnet-4").unwrap();
		assert_eq!(sonnet.base.cost.rates.input, 99.0);
		assert!(sonnet.base.cost.rates.output > 0.0);
	});
}

#[test]
fn model_override_can_add_headers_at_request_time() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {
				"anthropic/claude-sonnet-4": {"headers": {"X-Custom-Model-Header": "value"}}
			}
		} }));
		let registry = fixture.registry().await;
		let models = get_models_for_provider(&registry, "openrouter");
		let sonnet = models
			.iter()
			.find(|model| model.id() == "anthropic/claude-sonnet-4")
			.expect("sonnet model");

		let (_, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(sonnet.clone())).await);
		assert_eq!(header_value(&headers, "X-Custom-Model-Header"), Some("value"));
	});
}

#[test]
fn refresh_picks_up_model_override_changes() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "First Name"}}
		} }));
		let registry = fixture.registry().await;
		let sonnet_name = |registry: &ModelRegistry| {
			get_models_for_provider(registry, "openrouter")
				.into_iter()
				.find(|model| model.id() == "anthropic/claude-sonnet-4")
				.unwrap()
				.base
				.name
		};
		assert_eq!(sonnet_name(&registry), "First Name");

		fixture.write_models_json(json!({ "openrouter": {
			"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Second Name"}}
		} }));
		refresh_offline(&registry).await;
		assert_eq!(sonnet_name(&registry), "Second Name");
	});
}

#[test]
fn removing_model_override_restores_builtin_values() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "openrouter": {
			"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Custom Name"}}
		} }));
		let registry = fixture.registry().await;
		let sonnet_name = |registry: &ModelRegistry| {
			get_models_for_provider(registry, "openrouter")
				.into_iter()
				.find(|model| model.id() == "anthropic/claude-sonnet-4")
				.unwrap()
				.base
				.name
		};
		assert_eq!(sonnet_name(&registry), "Custom Name");

		fixture.write_models_json(json!({}));
		refresh_offline(&registry).await;
		assert_ne!(sonnet_name(&registry), "Custom Name");
	});
}

// ---------------------------------------------------------------------------
// "dynamic provider lifecycle"
// ---------------------------------------------------------------------------

#[test]
fn get_provider_display_name_resolves_registered_oauth_builtin_and_fallback_names() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		assert_eq!(registry.get_provider_display_name("openai"), "OpenAI");
		assert_eq!(registry.get_provider_display_name("github-copilot"), "GitHub Copilot");
		assert_eq!(registry.get_provider_display_name("zai"), "Z.AI");
		assert_eq!(registry.get_provider_display_name("unknown-provider"), "unknown-provider");

		let mut named = provider_config("https://provider.test/v1", &["demo-model"], Some("openai-completions"));
		named.name = Some("Named Provider".to_string());
		registry.register_provider("named-provider", named).unwrap();
		assert_eq!(
			eventually(|| {
				let name = registry.get_provider_display_name("named-provider");
				(name == "Named Provider").then_some(name)
			})
			.await,
			"Named Provider"
		);

		let mut oauth = provider_config("https://provider.test/v1", &["demo-model"], Some("openai-completions"));
		oauth.oauth = Some(oauth_config("OAuth Provider"));
		registry.register_provider("oauth-provider", oauth).unwrap();
		assert_eq!(
			eventually(|| {
				let name = registry.get_provider_display_name("oauth-provider");
				(name == "OAuth Provider").then_some(name)
			})
			.await,
			"OAuth Provider"
		);
	});
}

#[test]
fn model_overrides_apply_to_dynamically_registered_provider_models() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "extension-provider": {
			"modelOverrides": {
				"extension-model": {
					"name": "Overridden Extension Model",
					"thinkingLevelMap": {
						"off": null, "minimal": null, "low": null, "medium": null, "xhigh": "max"
					},
					"headers": {"x-model-override": "enabled"}
				}
			}
		} }));
		let registry = fixture.registry().await;

		let mut config = provider_config("https://provider.test/v1", &["extension-model"], Some("openai-completions"));
		if let Some(ProviderModelConfig::Chat(chat)) = config.models.as_mut().unwrap().first_mut() {
			chat.reasoning = true;
		}
		registry.register_provider("extension-provider", config).unwrap();

		let model = eventually(|| registry.find("extension-provider", "extension-model")).await;
		assert_eq!(model.base.name, "Overridden Extension Model");
		let expected_map: ThinkingLevelMap = [
			(ModelThinkingLevel::Off, None),
			(ModelThinkingLevel::Minimal, None),
			(ModelThinkingLevel::Low, None),
			(ModelThinkingLevel::Medium, None),
			(ModelThinkingLevel::Xhigh, Some("max".to_string())),
		]
		.into_iter()
		.collect();
		assert_eq!(model.thinking_level_map.as_ref(), Some(&expected_map));
		assert_eq!(get_supported_thinking_levels(&model), vec![ModelThinkingLevel::High, ModelThinkingLevel::Xhigh]);

		let (_, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
		assert_eq!(header_value(&headers, "x-model-override"), Some("enabled"));
	});
}

#[test]
fn stored_api_key_env_propagates_to_request_auth_and_resolves_headers() {
	block(async {
		let fixture = fixture();
		fixture
			.store_credential(
				"cloudflare-ai-gateway",
				Credential::ApiKey(pi_ai::auth::types::ApiKeyCredential {
					key: Some("$CLOUDFLARE_API_KEY".to_string()),
					env: Some(BTreeMap::from([
						("CLOUDFLARE_API_KEY".to_string(), "stored-cf-token".to_string()),
						("CLOUDFLARE_ACCOUNT_ID".to_string(), "stored-account".to_string()),
						("CLOUDFLARE_GATEWAY_ID".to_string(), "stored-gateway".to_string()),
					])),
				}),
			)
			.await;
		fixture.write_models_json(json!({ "cloudflare-ai-gateway": {
			"headers": {"x-account": "$CLOUDFLARE_ACCOUNT_ID"}
		} }));

		let registry = fixture.registry().await;
		let model = registry
			.get_all()
			.into_iter()
			.find(|model| model.provider() == "cloudflare-ai-gateway")
			.expect("cloudflare gateway model");

		let (api_key, headers, base_url, env) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
		assert_eq!(api_key, None);
		let headers = headers.expect("headers");
		assert_eq!(headers.get("cf-aig-authorization").and_then(|value| value.as_deref()), Some("Bearer stored-cf-token"));
		assert_eq!(headers.get("Authorization"), Some(&None));
		assert_eq!(headers.get("x-api-key"), Some(&None));
		assert_eq!(headers.get("x-account").and_then(|value| value.as_deref()), Some("stored-account"));
		let expected_env: ProviderEnv = BTreeMap::from([
			("CLOUDFLARE_ACCOUNT_ID".to_string(), "stored-account".to_string()),
			("CLOUDFLARE_GATEWAY_ID".to_string(), "stored-gateway".to_string()),
		]);
		assert_eq!(env, Some(expected_env));
		assert_eq!(base_url, None);
	});
}

#[test]
fn register_provider_treats_uppercase_api_key_and_headers_as_literals() {
	run_with_envs(&[("CUSTOM_NAME", "env-CUSTOM_NAME"), ("BEARER", "env-BEARER"), ("MODEL_TOKEN", "env-MODEL_TOKEN")], || {
		block(async {
			clear_config_value_cache();
			let fixture = fixture();
			let registry = fixture.registry().await;

			let mut config = ProviderConfig {
				api_key: Some("CUSTOM_NAME".to_string()),
				headers: Some(BTreeMap::from([("Authorization".to_string(), "BEARER".to_string())])),
				..provider_config("https://provider.test/v1", &[], Some("openai-completions"))
			};
			let mut model = chat_model_config("demo-model");
			model.base.headers =
				Some(BTreeMap::from([("x-model-token".to_string(), "MODEL_TOKEN".to_string())]));
			config.models = Some(vec![ProviderModelConfig::Chat(model)]);
			registry.register_provider("literal-provider", config).unwrap();

			// The background rebuild can transiently clear the provider; retry.
			let mut api_key = None;
			for _ in 0..500 {
				if let Some(key) = registry.get_api_key_for_provider("literal-provider").await {
					api_key = Some(key);
					break;
				}
				tokio::time::sleep(std::time::Duration::from_millis(10)).await;
			}
			assert_eq!(api_key.as_deref(), Some("CUSTOM_NAME"));
			let model = eventually(|| registry.find("literal-provider", "demo-model")).await;
			let (api_key, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
			assert_eq!(api_key.as_deref(), Some("CUSTOM_NAME"));
			assert_eq!(header_value(&headers, "Authorization"), Some("BEARER"));
			assert_eq!(header_value(&headers, "x-model-token"), Some("MODEL_TOKEN"));
		});
	});
}

#[test]
fn failed_register_provider_does_not_persist_invalid_stream_simple_config() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		let stream_simple: StreamSimpleFn = Arc::new(
			|_model: Model, _transcript: TranscriptContext, _options: Option<SimpleStreamOptions>| {
				unreachable!("streamSimple must not run during registration")
			},
		);
		let error = registry
			.register_provider(
				"broken-provider",
				ProviderConfig { stream_simple: Some(stream_simple), ..Default::default() },
			)
			.unwrap_err();
		assert_eq!(
			thrown_text(&error),
			"Provider broken-provider: \"api\" is required when registering streamSimple."
		);

		let result = refresh_offline(&registry).await;
		assert!(!result.aborted);
	});
}

#[test]
fn failed_register_provider_does_not_remove_existing_provider_models() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		registry
			.register_provider("demo-provider", provider_config("https://provider.test/v1", &["demo-model"], Some("openai-completions")))
			.unwrap();
		eventually(|| registry.find("demo-provider", "demo-model")).await;

		// Re-registration without any api at provider or model level fails
		// (upstream `toThrow` matches the prefix).
		let error = registry
			.register_provider("demo-provider", provider_config("https://provider.test/v2", &["broken-model"], None))
			.unwrap_err();
		assert!(thrown_text(&error)
			.starts_with("Provider demo-provider, model broken-model: no \"api\" specified."));

		eventually(|| registry.find("demo-provider", "demo-model")).await;
		let result = refresh_offline(&registry).await;
		assert!(!result.aborted);
		eventually(|| registry.find("demo-provider", "demo-model")).await;
	});
}

#[test]
fn unregister_provider_removes_the_runtime_oauth_overlay_without_mutating_global_state() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		let mut config = ProviderConfig::default();
		config.oauth = Some(oauth_config("Custom Anthropic OAuth"));
		registry.register_provider("anthropic", config).unwrap();

		assert_eq!(
			registry
				.get_registered_provider_config("anthropic")
				.expect("registered anthropic config")
				.oauth
				.map(|oauth| oauth.name),
			Some("Custom Anthropic OAuth".to_string())
		);

		registry.unregister_provider("anthropic");
		assert!(registry.get_registered_provider_config("anthropic").is_none());
	});
}

// -- "dynamic provider override persistence" --------------------------------

#[test]
fn base_url_only_override_keeps_builtin_provider_models_after_refresh() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		registry
			.register_provider(
				"anthropic",
				ProviderConfig { base_url: Some("https://proxy.test/anthropic".to_string()), ..Default::default() },
			)
			.unwrap();
		refresh_offline(&registry).await;

		let anthropic_models = eventually(|| {
			let models = get_models_for_provider(&registry, "anthropic");
			(models.len() > 1).then_some(models)
		})
		.await;
		assert!(anthropic_models.iter().all(|model| model.base.base_url == "https://proxy.test/anthropic"));
	});
}

#[test]
fn models_only_override_replaces_builtin_provider_models_after_refresh() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		let mut config = provider_config("https://custom.test/anthropic", &["custom-claude"], Some("anthropic-messages"));
		config.base_url = Some("https://custom.test/anthropic".to_string());
		registry.register_provider("anthropic", config).unwrap();
		refresh_offline(&registry).await;

		let ids = eventually(|| {
			let ids: Vec<String> =
				get_models_for_provider(&registry, "anthropic").into_iter().map(|model| model.id().to_string()).collect();
			(ids == vec!["custom-claude".to_string()]).then_some(ids)
		})
		.await;
		assert_eq!(ids, vec!["custom-claude".to_string()]);
		assert_eq!(
			registry.find("anthropic", "custom-claude").unwrap().base.base_url,
			"https://custom.test/anthropic"
		);
	});
}

#[test]
fn models_plus_base_url_override_replaces_builtin_provider_models_after_refresh() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		let mut config = provider_config("https://custom.test/anthropic", &["custom-claude"], Some("anthropic-messages"));
		config.base_url = Some("https://custom.test/anthropic".to_string());
		registry.register_provider("anthropic", config).unwrap();
		registry
			.register_provider(
				"anthropic",
				ProviderConfig { base_url: Some("https://proxy.test/anthropic".to_string()), ..Default::default() },
			)
			.unwrap();
		refresh_offline(&registry).await;

		let ids = eventually(|| {
			let ids: Vec<String> =
				get_models_for_provider(&registry, "anthropic").into_iter().map(|model| model.id().to_string()).collect();
			(ids == vec!["custom-claude".to_string()]).then_some(ids)
		})
		.await;
		assert_eq!(ids, vec!["custom-claude".to_string()]);
		assert_eq!(
			registry.find("anthropic", "custom-claude").unwrap().base.base_url,
			"https://proxy.test/anthropic"
		);
	});
}

#[test]
fn models_only_custom_provider_registration_survives_refresh() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		registry
			.register_provider(
				"custom-provider",
				provider_config("https://custom.test/v1", &["custom-a", "custom-b"], Some("openai-completions")),
			)
			.unwrap();
		refresh_offline(&registry).await;

		let ids = eventually(|| {
			let ids: Vec<String> = get_models_for_provider(&registry, "custom-provider")
				.into_iter()
				.map(|model| model.id().to_string())
				.collect();
			(ids == vec!["custom-a".to_string(), "custom-b".to_string()]).then_some(ids)
		})
		.await;
		assert_eq!(ids, vec!["custom-a".to_string(), "custom-b".to_string()]);
	});
}

#[test]
fn base_url_only_override_keeps_custom_provider_models_after_refresh() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		registry
			.register_provider(
				"custom-provider",
				provider_config("https://custom.test/v1", &["custom-a", "custom-b"], Some("openai-completions")),
			)
			.unwrap();
		registry
			.register_provider(
				"custom-provider",
				ProviderConfig { base_url: Some("https://proxy.test/custom".to_string()), ..Default::default() },
			)
			.unwrap();
		refresh_offline(&registry).await;

		let models = eventually(|| {
			let models = get_models_for_provider(&registry, "custom-provider");
			let ids: Vec<String> = models.iter().map(|model| model.id().to_string()).collect();
			(ids == vec!["custom-a".to_string(), "custom-b".to_string()]).then_some(models)
		})
		.await;
		assert!(models.iter().all(|model| model.base.base_url == "https://proxy.test/custom"));
	});
}

#[test]
fn headers_only_override_keeps_custom_provider_models_after_refresh() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		registry
			.register_provider(
				"custom-provider",
				provider_config("https://custom.test/v1", &["custom-a", "custom-b"], Some("openai-completions")),
			)
			.unwrap();
		registry
			.register_provider(
				"custom-provider",
				ProviderConfig {
					headers: Some(BTreeMap::from([("x-proxy".to_string(), "enabled".to_string())])),
					..Default::default()
				},
			)
			.unwrap();
		refresh_offline(&registry).await;

		let models = eventually(|| {
			let models = get_models_for_provider(&registry, "custom-provider");
			let ids: Vec<String> = models.iter().map(|model| model.id().to_string()).collect();
			(ids == vec!["custom-a".to_string(), "custom-b".to_string()]).then_some(models)
		})
		.await;
		assert!(models.iter().all(|model| model.base.base_url == "https://custom.test/v1"));
		let (_, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(models[0].clone())).await);
		assert_eq!(header_value(&headers, "x-proxy"), Some("enabled"));
	});
}

// ---------------------------------------------------------------------------
// Facade virtual-model registration (upstream covers `registerVirtualModel`
// deeply in virtual-models.test.ts; the facade delegation is exercised here).
// ---------------------------------------------------------------------------

#[test]
fn facade_registers_and_unregisters_virtual_models() {
	block(async {
		let fixture = fixture();
		let registry = fixture.registry().await;

		let definition = VirtualModelDefinition {
			provider: "virtual-provider".to_string(),
			id: "routed-model".to_string(),
			name: "Routed Model".to_string(),
			thinking_levels: None,
			context_window: Some(100_000),
			max_tokens: Some(8000),
			input: None,
			route: Arc::new(|request: RouteRequest| {
				Box::pin(async move {
					Ok(ModelRoute {
						model: request.model.clone(),
						thinking_level: request.thinking_level,
						state: None,
					})
				})
			}),
		};
		registry.register_virtual_model(definition).unwrap();

		let model = eventually(|| registry.find("virtual-provider", "routed-model")).await;
		assert!(crate::core::virtual_models::is_virtual_model(&model));

		registry.unregister_virtual_model("virtual-provider", "routed-model");
		assert!(registry.find("virtual-provider", "routed-model").is_none());
	});
}

// ---------------------------------------------------------------------------
// "API key resolution"
// ---------------------------------------------------------------------------

#[test]
fn api_key_with_bang_prefix_executes_command_and_uses_stdout() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!echo test-api-key-from-command");
		let registry = fixture.registry().await;
		assert_eq!(
			registry.get_api_key_for_provider("custom-provider").await.as_deref(),
			Some("test-api-key-from-command")
		);
	});
}

#[test]
fn api_key_with_bang_prefix_trims_whitespace_from_command_output() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!echo '  spaced-key  '");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("spaced-key"));
	});
}

#[test]
fn api_key_with_bang_prefix_handles_multiline_output() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!printf 'line1\\nline2'");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("line1\nline2"));
	});
}

#[test]
fn api_key_with_bang_prefix_returns_none_on_command_failure() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!exit 1");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await, None);
	});
}

#[test]
fn api_key_with_bang_prefix_returns_none_on_nonexistent_command() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!nonexistent-command-12345");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await, None);
	});
}

#[test]
fn api_key_with_bang_prefix_returns_none_on_empty_output() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!printf ''");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await, None);
	});
}

#[test]
fn api_key_with_dollar_prefix_resolves_to_env_value() {
	run_with_env("TEST_API_KEY_12345", "env-api-key-value", || {
		block(async {
			let fixture = fixture_with_api_key("$TEST_API_KEY_12345");
			let registry = fixture.registry().await;
			assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("env-api-key-value"));
		});
	});
}

#[test]
fn api_key_with_braced_env_syntax_resolves_to_env_value() {
	run_with_env("TEST_BRACED_API_KEY_12345", "braced-env-api-key-value", || {
		block(async {
			let fixture = fixture_with_api_key("${TEST_BRACED_API_KEY_12345}");
			let registry = fixture.registry().await;
			assert_eq!(
				registry.get_api_key_for_provider("custom-provider").await.as_deref(),
				Some("braced-env-api-key-value")
			);
		});
	});
}

#[test]
fn api_key_interpolates_braced_env_references_inside_literals() {
	run_with_envs(
		&[("TEST_INTERPOLATED_PART_A_12345", "left"), ("TEST_INTERPOLATED_PART_B_12345", "right")],
		|| {
			block(async {
				let fixture = fixture_with_api_key("${TEST_INTERPOLATED_PART_A_12345}_${TEST_INTERPOLATED_PART_B_12345}");
				let registry = fixture.registry().await;
				assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("left_right"));
			});
		},
	);
}

#[test]
fn api_key_with_double_dollar_prefix_escapes_a_leading_dollar() {
	block(async {
		let fixture = fixture_with_api_key("$$TEST_API_KEY_12345");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("$TEST_API_KEY_12345"));
	});
}

#[test]
fn api_key_with_dollar_bang_escapes_a_literal_bang_and_still_interpolates() {
	run_with_env("TEST_API_KEY_12345", "env-api-key-value", || {
		block(async {
			let fixture = fixture_with_api_key("$!literal-$TEST_API_KEY_12345");
			let registry = fixture.registry().await;
			assert_eq!(
				registry.get_api_key_for_provider("custom-provider").await.as_deref(),
				Some("!literal-env-api-key-value")
			);
		});
	});
}

#[test]
fn plain_api_key_is_used_directly_even_when_it_matches_an_env_var() {
	run_with_env("TEST_API_KEY_12345", "env-api-key-value", || {
		block(async {
			let fixture = fixture_with_api_key("TEST_API_KEY_12345");
			let registry = fixture.registry().await;
			assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("TEST_API_KEY_12345"));
		});
	});
}

#[test]
fn api_key_as_literal_value_is_used_directly_when_not_an_env_var() {
	block(async {
		let fixture = fixture_with_api_key("literal_api_key_value");
		let registry = fixture.registry().await;
		assert_eq!(
			registry.get_api_key_for_provider("custom-provider").await.as_deref(),
			Some("literal_api_key_value")
		);
	});
}

#[test]
fn api_key_command_can_use_shell_features_like_pipes() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_api_key("!echo 'hello world' | tr ' ' '-'");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_api_key_for_provider("custom-provider").await.as_deref(), Some("hello-world"));
	});
}

// -- "request-time resolution" ----------------------------------------------

#[test]
fn command_is_executed_on_every_provider_lookup() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("counter");
		write_file(&counter_file, "0");
		let command = format!(
			"!sh -c 'count=$(cat \"{}\"); echo $((count + 1)) > \"{}\"; echo \"key-value\"'",
			to_sh_path(&counter_file),
			to_sh_path(&counter_file)
		);
		fixture.write_models_json(json!({ "custom-provider": provider_with_api_key(&command) }));
		let registry = fixture.registry().await;

		registry.get_api_key_for_provider("custom-provider").await;
		registry.get_api_key_for_provider("custom-provider").await;
		registry.get_api_key_for_provider("custom-provider").await;

		assert_eq!(read_file(&counter_file).trim(), "3");
	});
}

#[test]
fn commands_are_re_executed_across_registry_instances() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("counter");
		write_file(&counter_file, "0");
		let command = format!(
			"!sh -c 'count=$(cat \"{}\"); echo $((count + 1)) > \"{}\"; echo \"key-value\"'",
			to_sh_path(&counter_file),
			to_sh_path(&counter_file)
		);
		fixture.write_models_json(json!({ "custom-provider": provider_with_api_key(&command) }));

		let registry1 = fixture.registry().await;
		registry1.get_api_key_for_provider("custom-provider").await;

		let registry2 = fixture.registry().await;
		registry2.get_api_key_for_provider("custom-provider").await;

		assert_eq!(read_file(&counter_file).trim(), "2");
	});
}

#[test]
fn different_commands_resolve_independently() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture_with_models_json(json!({
			"provider-a": provider_with_api_key("!echo key-a"),
			"provider-b": provider_with_api_key("!echo key-b")
		}));
		let registry = fixture.registry().await;

		let key_a = registry.get_api_key_for_provider("provider-a").await;
		let key_b = registry.get_api_key_for_provider("provider-b").await;

		assert_eq!(key_a.as_deref(), Some("key-a"));
		assert_eq!(key_b.as_deref(), Some("key-b"));
	});
}

#[test]
fn failed_commands_are_retried() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("counter");
		write_file(&counter_file, "0");
		let command = format!(
			"!sh -c 'count=$(cat \"{}\"); echo $((count + 1)) > \"{}\"; exit 1'",
			to_sh_path(&counter_file),
			to_sh_path(&counter_file)
		);
		fixture.write_models_json(json!({ "custom-provider": provider_with_api_key(&command) }));
		let registry = fixture.registry().await;

		let key1 = registry.get_api_key_for_provider("custom-provider").await;
		let key2 = registry.get_api_key_for_provider("custom-provider").await;
		assert_eq!(key1, None);
		assert_eq!(key2, None);
		assert_eq!(read_file(&counter_file).trim(), "2");
	});
}

#[test]
fn provider_auth_status_reports_api_key_environment_variables_from_models_json() {
	run_with_env("TEST_API_KEY_STATUS_TEST_98765", "status-test-key", || {
		block(async {
			let fixture = fixture_with_api_key("$TEST_API_KEY_STATUS_TEST_98765");
			let registry = fixture.registry().await;
			assert_eq!(
				registry.get_provider_auth_status("custom-provider"),
				status(true, Some(AuthStatusSource::Environment), Some("TEST_API_KEY_STATUS_TEST_98765"))
			);
		});
	});
}

#[test]
fn provider_auth_status_reports_interpolated_api_key_environment_variables() {
	run_with_envs(
		&[("TEST_API_KEY_STATUS_PART_A_98765", "left"), ("TEST_API_KEY_STATUS_PART_B_98765", "right")],
		|| {
			block(async {
				let fixture = fixture_with_api_key("${TEST_API_KEY_STATUS_PART_A_98765}_${TEST_API_KEY_STATUS_PART_B_98765}");
				let registry = fixture.registry().await;
				assert_eq!(
					registry.get_provider_auth_status("custom-provider"),
					status(
						true,
						Some(AuthStatusSource::Environment),
						Some("TEST_API_KEY_STATUS_PART_A_98765, TEST_API_KEY_STATUS_PART_B_98765")
					)
				);
			});
		},
	);
}

#[test]
fn provider_auth_status_reports_non_env_api_key_values_as_a_config_key() {
	block(async {
		let fixture = fixture_with_api_key("literal_api_key_value");
		let registry = fixture.registry().await;
		assert_eq!(
			registry.get_provider_auth_status("custom-provider"),
			status(true, Some(AuthStatusSource::ModelsJsonKey), None)
		);
	});
}

#[test]
fn missing_explicit_env_api_key_keeps_provider_unavailable() {
	block(async {
		std::env::remove_var("TEST_API_KEY_MISSING_TEST_98765");
		let fixture = fixture_with_api_key("$TEST_API_KEY_MISSING_TEST_98765");
		let registry = fixture.registry().await;
		assert_eq!(registry.get_provider_auth_status("custom-provider"), status(false, None, None));
		assert!(registry.get_available().iter().all(|model| model.provider() != "custom-provider"));
	});
}

#[test]
fn provider_auth_status_reports_command_api_key_values_without_executing_them() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("status-counter");
		write_file(&counter_file, "0");
		let command = format!("!sh -c 'echo 1 > \"{}\"; echo key-value'", to_sh_path(&counter_file));
		fixture.write_models_json(json!({ "custom-provider": provider_with_api_key(&command) }));
		let registry = fixture.registry().await;

		assert_eq!(
			registry.get_provider_auth_status("custom-provider"),
			status(true, Some(AuthStatusSource::ModelsJsonCommand), None)
		);
		assert_eq!(read_file(&counter_file), "0");
	});
}

#[test]
fn environment_variables_are_not_cached_changes_are_picked_up() {
	run_with_env("TEST_API_KEY_CACHE_TEST_98765", "first-value", || {
		block(async {
			let fixture = fixture_with_api_key("$TEST_API_KEY_CACHE_TEST_98765");
			let registry = fixture.registry().await;
			let key1 = registry.get_api_key_for_provider("custom-provider").await;
			assert_eq!(key1.as_deref(), Some("first-value"));

			std::env::set_var("TEST_API_KEY_CACHE_TEST_98765", "second-value");
			let key2 = registry.get_api_key_for_provider("custom-provider").await;
			assert_eq!(key2.as_deref(), Some("second-value"));
		});
	});
}

#[test]
fn get_available_does_not_execute_command_backed_api_key_resolution() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("counter");
		write_file(&counter_file, "0");
		let command = format!(
			"!sh -c 'count=$(cat \"{}\"); echo $((count + 1)) > \"{}\"; echo \"key-value\"'",
			to_sh_path(&counter_file),
			to_sh_path(&counter_file)
		);
		fixture.write_models_json(json!({ "custom-provider": provider_with_api_key(&command) }));
		let registry = fixture.registry().await;

		let available = registry.get_available();
		assert!(available.iter().any(|model| model.provider() == "custom-provider"));
		assert_eq!(read_file(&counter_file).trim(), "0");
	});
}

#[test]
fn get_available_filters_github_copilot_oauth_models_to_account_picker_availability() {
	block(async {
		let copilot_model = get_builtin_models(Some("github-copilot"))
			.into_iter()
			.next()
			.expect("at least one GitHub Copilot model");

		let fixture = fixture();
		fixture
			.store_credential(
				"github-copilot",
				Credential::OAuth(pi_ai::auth::types::OAuthCredential {
					credentials: OAuthCredentials {
						refresh: "github-access-token".to_string(),
						access: "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;".to_string(),
						expires: now_ms() + 60_000,
						extra: BTreeMap::from([(
							"availableModelIds".to_string(),
							json!([copilot_model.id()]),
						)]),
					},
				}),
			)
			.await;
		let registry = fixture.registry().await;

		let copilot_ids: Vec<String> = registry
			.get_available()
			.into_iter()
			.filter(|model| model.provider() == "github-copilot")
			.map(|model| model.id().to_string())
			.collect();
		assert_eq!(copilot_ids, vec![copilot_model.id().to_string()]);
	});
}

#[test]
fn get_api_key_and_headers_resolves_auth_header_on_every_request() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let token_file = fixture.models_json_path.parent().unwrap().join("token");
		write_file(&token_file, "token-1");

		let mut provider = provider_with_api_key(&format!("!sh -c 'cat \"{}\"'", to_sh_path(&token_file)));
		provider["authHeader"] = json!(true);
		fixture.write_models_json(json!({ "custom-provider": provider }));
		let registry = fixture.registry().await;
		let model = registry.find("custom-provider", "test-model").expect("test model");

		let (api_key, headers, base_url, env) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
		assert_eq!(api_key.as_deref(), Some("token-1"));
		assert_eq!(header_value(&headers, "Authorization"), Some("Bearer token-1"));
		assert_eq!(base_url, None);
		assert_eq!(env, None);

		write_file(&token_file, "token-2");

		let (api_key, headers, _, _) = unwrap_ok(registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await);
		assert_eq!(api_key.as_deref(), Some("token-2"));
		assert_eq!(header_value(&headers, "Authorization"), Some("Bearer token-2"));
	});
}

#[test]
fn get_api_key_and_headers_resolves_configured_auth_exactly_once() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("auth-counter");
		write_file(&counter_file, "0");
		let command = format!(
			"!sh -c 'count=$(cat \"{}\"); count=$((count + 1)); echo \"$count\" > \"{}\"; echo \"token-$count\"'",
			to_sh_path(&counter_file),
			to_sh_path(&counter_file)
		);
		let mut provider = provider_with_api_key(&command);
		provider["authHeader"] = json!(true);
		fixture.write_models_json(json!({ "custom-provider": provider }));

		let registry = fixture.registry().await;
		let (api_key, headers, _, _) = unwrap_ok(
			registry
				.get_api_key_and_headers(&AnyModel::Chat(registry.find("custom-provider", "test-model").unwrap().clone()))
				.await,
		);
		assert_eq!(api_key.as_deref(), Some("token-1"));
		assert_eq!(header_value(&headers, "Authorization"), Some("Bearer token-1"));
		assert_eq!(read_file(&counter_file).trim(), "1");
	});
}

#[test]
fn stored_credentials_bypass_lower_priority_configured_auth_commands() {
	block(async {
		clear_config_value_cache();
		let fixture = fixture();
		let counter_file = fixture.models_json_path.parent().unwrap().join("fallback-counter");
		write_file(&counter_file, "0");
		let command = format!("!sh -c 'echo 1 > \"{}\"; echo fallback-key'", to_sh_path(&counter_file));
		fixture.write_models_json(json!({ "custom-provider": provider_with_api_key(&command) }));
		fixture
			.store_credential(
				"custom-provider",
				Credential::ApiKey(pi_ai::auth::types::ApiKeyCredential {
					key: Some("stored-key".to_string()),
					env: None,
				}),
			)
			.await;

		let registry = fixture.registry().await;
		let (api_key, _, _, _) = unwrap_ok(
			registry
				.get_api_key_and_headers(&AnyModel::Chat(registry.find("custom-provider", "test-model").unwrap().clone()))
				.await,
		);
		assert_eq!(api_key.as_deref(), Some("stored-key"));
		assert_eq!(read_file(&counter_file).trim(), "0");
	});
}

#[test]
fn get_api_key_and_headers_preserves_the_legacy_missing_key_auth_header_error() {
	block(async {
		let fixture = fixture_with_models_json(json!({ "custom-provider": {
			"baseUrl": "https://example.test/v1",
			"api": "openai-completions",
			"authHeader": true,
			"models": [{"id": "test-model"}]
		} }));
		let registry = fixture.registry().await;
		let auth = registry
			.get_api_key_and_headers(&AnyModel::Chat(registry.find("custom-provider", "test-model").unwrap().clone()))
			.await;
		match auth {
			ResolvedRequestAuth::Error { error } => {
				assert_eq!(error, "No API key found for \"custom-provider\"");
			}
			other => panic!("expected error auth, got {other:?}"),
		}
	});
}

#[test]
fn get_api_key_and_headers_returns_an_error_for_failed_auth_header_resolution() {
	block(async {
		clear_config_value_cache();
		let mut provider = provider_with_api_key("!exit 1");
		provider["authHeader"] = json!(true);
		let fixture = fixture_with_models_json(json!({ "custom-provider": provider }));
		let registry = fixture.registry().await;
		let model = registry.find("custom-provider", "test-model").expect("test model");

		let auth = registry.get_api_key_and_headers(&AnyModel::Chat(model.clone())).await;
		match auth {
			ResolvedRequestAuth::Error { error } => {
				assert!(
					error.contains("Failed to resolve API key for provider \"custom-provider\""),
					"error: {error}"
				);
			}
			other => panic!("expected error auth, got {other:?}"),
		}
	});
}

