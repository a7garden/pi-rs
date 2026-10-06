//! Port of `packages/ai/src/api/azure-openai-config.ts` — Azure OpenAI
//! endpoint resolution (base URL normalization, deployment-name mapping, API
//! version).
//!
//! Deviations:
//! - **D15**: upstream `AzureEndpointOptions extends StreamOptions`; per-API
//!   options cannot ride `StreamOptions` (types.rs frozen; deviation follows
//!   the crate D2 pattern). The struct carries only the upstream extra fields;
//!   the scoped `env` map is a separate argument on the resolve functions
//!   (upstream reads it off the options object). Final integration should add
//!   the nested `StreamOptions` group (reported in the wave reexports).
//! - **D16**: URL normalization uses the `url` crate (WHATWG URL, the same
//!   grammar as the upstream `new URL`), with the trailing-slash and query
//!   handling reproduced exactly.

use std::collections::BTreeMap;

use crate::types::{Model, ProviderEnv};
use crate::utils::provider_env::get_provider_env_value;

/// Upstream `DEFAULT_AZURE_API_VERSION`.
pub const DEFAULT_AZURE_API_VERSION: &str = "v1";

/// Upstream `AzureEndpointOptions` (deviation D15: base `StreamOptions`
/// fields stay on `StreamOptions`).
#[derive(Debug, Clone, Default)]
pub struct AzureEndpointOptions {
	/// Overrides the `AZURE_OPENAI_API_VERSION` environment value.
	pub azure_api_version: Option<String>,
	/// Builds `https://<resource>.openai.azure.com/openai/v1` when no base
	/// URL is given.
	pub azure_resource_name: Option<String>,
	/// Explicit Azure OpenAI (or proxy) base URL.
	pub azure_base_url: Option<String>,
	/// Deployment name sent as the request model; defaults to `model.id`.
	pub azure_deployment_name: Option<String>,
}

/// Endpoint options view over the full responses options (upstream
/// `AzureOpenAIResponsesOptions extends AzureEndpointOptions`).
impl AzureEndpointOptions {
	pub fn from_parts(
		azure_api_version: Option<String>,
		azure_resource_name: Option<String>,
		azure_base_url: Option<String>,
		azure_deployment_name: Option<String>,
	) -> AzureEndpointOptions {
		AzureEndpointOptions { azure_api_version, azure_resource_name, azure_base_url, azure_deployment_name }
	}
}

/// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` parsing: `model-id=deployment-name`
/// pairs, comma-separated, each side trimmed; malformed entries are skipped.
fn parse_deployment_name_map(value: Option<&str>) -> BTreeMap<String, String> {
	let mut map = BTreeMap::new();
	let Some(value) = value else { return map };
	for entry in value.split(',') {
		let trimmed = entry.trim();
		if trimmed.is_empty() {
			continue;
		}
		let mut parts = trimmed.splitn(2, '=');
		let model_id = parts.next().unwrap_or_default().trim();
		let deployment_name = match parts.next() {
			Some(name) => name.trim(),
			None => continue,
		};
		if model_id.is_empty() || deployment_name.is_empty() {
			continue;
		}
		map.insert(model_id.to_string(), deployment_name.to_string());
	}
	map
}

/// Upstream `resolveDeploymentName`: the explicit option, then the
/// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` entry for the model id, then the model
/// id itself.
pub fn resolve_deployment_name(
	model: &Model,
	options: Option<&AzureEndpointOptions>,
	env: Option<&ProviderEnv>,
) -> String {
	if let Some(deployment_name) =
		options.and_then(|options| options.azure_deployment_name.as_deref()).filter(|name| !name.is_empty())
	{
		return deployment_name.to_string();
	}
	let mapped_deployment = parse_deployment_name_map(
		get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", env).as_deref(),
	)
	.get(model.id())
	.cloned();
	mapped_deployment.unwrap_or_else(|| model.id().to_string())
}

/// Upstream `normalizeAzureBaseUrl`: ensure Azure hosts carry `/openai/v1` as
/// the base path so `/responses` and `?api-version=v1` append correctly.
/// `Err` carries the upstream error message verbatim.
fn normalize_azure_base_url(base_url: &str) -> Result<String, String> {
	let trimmed = base_url.trim().trim_end_matches('/');
	let mut url =
		url::Url::parse(trimmed).map_err(|_| format!("Invalid Azure OpenAI base URL: {base_url}"))?;

	let hostname = url.host_str().unwrap_or_default();
	let is_azure_host = hostname.ends_with(".openai.azure.com")
		|| hostname.ends_with(".cognitiveservices.azure.com")
		|| hostname.ends_with(".ai.azure.com");
	let normalized_path = url.path().trim_end_matches('/').to_string();

	// Ensure Azure hosts have /openai/v1 as base path so the AzureOpenAI SDK
	// can append /deployments/<model>/... and ?api-version=v1 correctly.
	if is_azure_host
		&& (normalized_path.is_empty()
			|| normalized_path == "/openai"
			|| normalized_path == "/openai/v1/responses")
	{
		url.set_path("/openai/v1");
		url.set_query(None);
	}

	let mut result = url.to_string();
	// `url.toString().replace(/\/+$/, "")`
	while result.ends_with('/') {
		result.pop();
	}
	Ok(result)
}

/// Upstream `buildDefaultBaseUrl`.
fn build_default_base_url(resource_name: &str) -> String {
	format!("https://{resource_name}.openai.azure.com/openai/v1")
}

/// Upstream `resolveAzureBaseUrl` (option → env → model), normalized.
/// `Err` carries the upstream error message verbatim.
pub fn resolve_azure_base_url(
	model: &Model,
	options: Option<&AzureEndpointOptions>,
	env: Option<&ProviderEnv>,
) -> Result<String, String> {
	let base_url: Option<String> = options
		.and_then(|options| options.azure_base_url.as_deref())
		.map(|value| value.trim().to_string())
		.filter(|value| !value.is_empty())
		.or_else(|| {
			get_provider_env_value("AZURE_OPENAI_BASE_URL", env)
				.map(|value| value.trim().to_string())
				.filter(|value| !value.is_empty())
		});
	let resource_name: Option<String> = options
		.and_then(|options| options.azure_resource_name.as_deref())
		.filter(|value| !value.is_empty())
		.map(str::to_string)
		.or_else(|| get_provider_env_value("AZURE_OPENAI_RESOURCE_NAME", env));

	let mut resolved_base_url = base_url;

	if resolved_base_url.is_none() {
		if let Some(resource_name) = resource_name {
			resolved_base_url = Some(build_default_base_url(&resource_name));
		}
	}

	if resolved_base_url.is_none() && !model.base.base_url.is_empty() {
		resolved_base_url = Some(model.base.base_url.clone());
	}

	let resolved_base_url = resolved_base_url.ok_or_else(|| {
		"Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.".to_string()
	})?;

	normalize_azure_base_url(&resolved_base_url)
}

/// Upstream `resolveAzureConfig` return shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureResolvedConfig {
	pub base_url: String,
	pub api_version: String,
}

/// Upstream `resolveAzureConfig`.
pub fn resolve_azure_config(
	model: &Model,
	options: Option<&AzureEndpointOptions>,
	env: Option<&ProviderEnv>,
) -> Result<AzureResolvedConfig, String> {
	Ok(AzureResolvedConfig {
		base_url: resolve_azure_base_url(model, options, env)?,
		api_version: options
			.and_then(|options| options.azure_api_version.clone())
			.filter(|value| !value.is_empty())
			.or_else(|| get_provider_env_value("AZURE_OPENAI_API_VERSION", env))
			.unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string()),
	})
}

#[cfg(test)]
impl AzureEndpointOptions {
	/// Cross-module process-environment guard: the azure tests in the config,
	/// responses, and completions modules all mutate the process env exactly
	/// like the upstream suite; they serialize on this shared lock.
	pub(crate) fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
		static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
		LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// Direct unit coverage of the upstream test helpers (test/azure-utils.ts)
	// and the config resolver; the stream-level cases live in the responses /
	// completions module tests.

	fn model(id: &str, base_url: &str) -> Model {
		Model {
			base: crate::types::BaseModelFields {
				id: id.to_string(),
				name: id.to_string(),
				api: crate::types::api_ids::AZURE_OPENAI_RESPONSES.to_string(),
				provider: crate::types::provider_ids::AZURE.to_string(),
				base_url: base_url.to_string(),
				input: vec![crate::types::ModelInput::Text],
				input_limits: None,
				cost: Default::default(),
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 128_000,
			max_tokens: 16_384,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
	}

	#[test]
	fn deployment_name_map_parses_pairs_and_skips_malformed_entries() {
		let map = parse_deployment_name_map(Some("a=x, b = y , c=, =d, e,, f=g=h"));
		assert_eq!(map.get("a").map(String::as_str), Some("x"));
		assert_eq!(map.get("b").map(String::as_str), Some("y"));
		assert_eq!(map.len(), 3, "c=, =d, e are skipped; f=g=h keeps b=c");
		assert_eq!(map.get("f").map(String::as_str), Some("g=h"));
		assert!(parse_deployment_name_map(None).is_empty());
	}

	#[test]
	fn resolve_deployment_name_prefers_option_then_map_then_id() {
		let model = model("gpt-4o-mini", "");
		let mut options = AzureEndpointOptions::default();
		options.azure_deployment_name = Some("explicit-deployment".to_string());
		assert_eq!(resolve_deployment_name(&model, Some(&options), None), "explicit-deployment");

		let env = ProviderEnv::from([
			("AZURE_OPENAI_DEPLOYMENT_NAME_MAP".to_string(), "gpt-4o-mini=my-deployment".to_string()),
		]);
		assert_eq!(resolve_deployment_name(&model, None, Some(&env)), "my-deployment");
		assert_eq!(resolve_deployment_name(&model, None, None), "gpt-4o-mini");
	}

	#[test]
	fn resolve_azure_base_url_requires_a_base_url() {
		let _guard = AzureEndpointOptions::test_env_lock();
		std::env::remove_var("AZURE_OPENAI_BASE_URL");
		std::env::remove_var("AZURE_OPENAI_RESOURCE_NAME");
		let model = model("gpt-4o-mini", "");
		let error = resolve_azure_base_url(&model, None, None).unwrap_err();
		assert_eq!(
			error,
			"Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl."
		);
	}

	#[test]
	fn resolve_azure_base_url_uses_model_base_url() {
		let _guard = AzureEndpointOptions::test_env_lock();
		std::env::remove_var("AZURE_OPENAI_BASE_URL");
		std::env::remove_var("AZURE_OPENAI_RESOURCE_NAME");
		let model = model("gpt-4o-mini", "https://my-proxy.example.com/v1");
		assert_eq!(resolve_azure_base_url(&model, None, None).unwrap(), "https://my-proxy.example.com/v1");
	}

	#[test]
	fn invalid_base_url_carries_the_upstream_message() {
		let error = normalize_azure_base_url("not-a-url").unwrap_err();
		assert_eq!(error, "Invalid Azure OpenAI base URL: not-a-url");
	}

	#[test]
	fn resolve_azure_config_defaults_api_version() {
		let _guard = AzureEndpointOptions::test_env_lock();
		std::env::remove_var("AZURE_OPENAI_BASE_URL");
		std::env::remove_var("AZURE_OPENAI_RESOURCE_NAME");
		std::env::remove_var("AZURE_OPENAI_API_VERSION");
		let model = model("gpt-4o-mini", "https://my-resource.openai.azure.com");
		let config = resolve_azure_config(&model, None, None).unwrap();
		assert_eq!(config.base_url, "https://my-resource.openai.azure.com/openai/v1");
		assert_eq!(config.api_version, "v1");
	}
}
