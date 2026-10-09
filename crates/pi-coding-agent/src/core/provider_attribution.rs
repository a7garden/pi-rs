//! Port of `packages/coding-agent/src/core/provider-attribution.ts` —
//! provider attribution/session headers merged into provider requests.
//!
//! The telemetry opt-in comes from [`core::telemetry`] (upstream
//! `telemetry.ts`, folded into this crate as its own module).

use std::collections::BTreeMap;

use pi_ai::types::{Model, ProviderHeaders};

use crate::core::settings_manager::SettingsManager;
use crate::core::telemetry::is_install_telemetry_enabled;

const OPENROUTER_HOST: &str = "openrouter.ai";
const NVIDIA_NIM_HOST: &str = "integrate.api.nvidia.com";
const CLOUDFLARE_API_HOST: &str = "api.cloudflare.com";
const CLOUDFLARE_AI_GATEWAY_HOST: &str = "gateway.ai.cloudflare.com";
const OPENCODE_HOST: &str = "opencode.ai";

/// URL hostname comparison (`matchesHost`): `None` on parse failure
/// (upstream `catch { return false }`).
fn matches_host(base_url: &str, expected_host: &str) -> bool {
	url::Url::parse(base_url)
		.ok()
		.and_then(|url| url.host_str().map(|host| host == expected_host))
		.unwrap_or(false)
}

fn is_openrouter_model(model: &Model) -> bool {
	model.provider() == "openrouter" || model.base.base_url.contains(OPENROUTER_HOST)
}

fn is_nvidia_nim_model(model: &Model) -> bool {
	model.provider() == "nvidia" || matches_host(&model.base.base_url, NVIDIA_NIM_HOST)
}

fn is_cloudflare_model(model: &Model) -> bool {
	model.provider() == "cloudflare-workers-ai"
		|| model.provider() == "cloudflare-ai-gateway"
		|| matches_host(&model.base.base_url, CLOUDFLARE_API_HOST)
		|| matches_host(&model.base.base_url, CLOUDFLARE_AI_GATEWAY_HOST)
}

fn get_default_attribution_headers(model: &Model, settings_manager: &SettingsManager) -> ProviderHeaders {
	if !is_install_telemetry_enabled(settings_manager) {
		return BTreeMap::new();
	}

	if is_openrouter_model(model) {
		return BTreeMap::from([
			("HTTP-Referer".to_string(), Some("https://pi.dev".to_string())),
			("X-OpenRouter-Title".to_string(), Some("pi".to_string())),
			(
				"X-OpenRouter-Categories".to_string(),
				Some("cli-agent".to_string()),
			),
		]);
	}

	if is_nvidia_nim_model(model) {
		return BTreeMap::from([(
			"X-BILLING-INVOKE-ORIGIN".to_string(),
			Some("Pi".to_string()),
		)]);
	}

	if is_cloudflare_model(model) {
		return BTreeMap::from([("User-Agent".to_string(), Some("pi-coding-agent".to_string()))]);
	}

	BTreeMap::new()
}

fn get_session_headers(model: &Model, session_id: Option<&str>) -> ProviderHeaders {
	let Some(session_id) = session_id else {
		return BTreeMap::new();
	};
	if model.provider() != "opencode"
		&& model.provider() != "opencode-go"
		&& !matches_host(&model.base.base_url, OPENCODE_HOST)
	{
		return BTreeMap::new();
	}
	BTreeMap::from([
		("x-opencode-session".to_string(), Some(session_id.to_string())),
		("x-opencode-client".to_string(), Some("pi".to_string())),
	])
}

/// Merge session and attribution headers with caller-supplied header sources
/// (`mergeProviderAttributionHeaders`). Later sources win; `None` values are
/// preserved (upstream spreads `ProviderHeaders` records). Returns `None`
/// when nothing is set.
pub fn merge_provider_attribution_headers(
	model: &Model,
	settings_manager: &SettingsManager,
	session_id: Option<&str>,
	header_sources: &[&ProviderHeaders],
) -> Option<ProviderHeaders> {
	let mut merged: ProviderHeaders = get_session_headers(model, session_id);
	for (key, value) in get_default_attribution_headers(model, settings_manager) {
		merged.insert(key, value);
	}

	for headers in header_sources {
		for (key, value) in headers.iter() {
			merged.insert(key.clone(), value.clone());
		}
	}

	if merged.is_empty() {
		None
	} else {
		Some(merged)
	}
}

#[cfg(test)]
mod provider_attribution_tests {
	use super::*;
	use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};

	fn model(provider: &str, base_url: &str) -> Model {
		let handle = pi_ai::providers::faux::faux_provider(
			pi_ai::providers::faux::RegisterFauxProviderOptions {
				models: vec![pi_ai::providers::faux::FauxModelDefinition {
					id: "test-model".to_string(),
					..Default::default()
				}],
				..Default::default()
			},
		);
		let mut model = handle.get_model("test-model").expect("faux model");
		model.base.provider = provider.to_string();
		model.base.base_url = base_url.to_string();
		model
	}

	fn settings_with_telemetry(enabled: bool) -> SettingsManager {
		let mut settings = Settings::new();
		settings.0.insert("enableInstallTelemetry".to_string(), enabled.into());
		SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default())
	}

	#[test]
	fn no_headers_when_telemetry_disabled() {
		let model = model("openrouter", "https://openrouter.ai/api/v1");
		let settings = settings_with_telemetry(false);
		assert_eq!(
			merge_provider_attribution_headers(&model, &settings, None, &[]),
			None
		);
	}

	#[test]
	fn openrouter_attribution_headers() {
		let model = model("other-provider", "https://openrouter.ai/api/v1");
		let settings = settings_with_telemetry(true);
		let headers = merge_provider_attribution_headers(&model, &settings, None, &[]).unwrap();
		assert_eq!(headers.get("HTTP-Referer").unwrap().as_deref(), Some("https://pi.dev"));
		assert_eq!(headers.get("X-OpenRouter-Title").unwrap().as_deref(), Some("pi"));
		assert_eq!(
			headers.get("X-OpenRouter-Categories").unwrap().as_deref(),
			Some("cli-agent")
		);
	}

	#[test]
	fn nvidia_and_cloudflare_headers() {
		let settings = settings_with_telemetry(true);
		let nvidia = model("other", "https://integrate.api.nvidia.com/v1");
		let headers = merge_provider_attribution_headers(&nvidia, &settings, None, &[]).unwrap();
		assert_eq!(headers.get("X-BILLING-INVOKE-ORIGIN").unwrap().as_deref(), Some("Pi"));

		let cloudflare = model("other", "https://gateway.ai.cloudflare.com/v1/x");
		let headers = merge_provider_attribution_headers(&cloudflare, &settings, None, &[]).unwrap();
		assert_eq!(headers.get("User-Agent").unwrap().as_deref(), Some("pi-coding-agent"));
	}

	#[test]
	fn opencode_session_headers() {
		let settings = settings_with_telemetry(false);
		let opencode = model("opencode", "https://opencode.ai/zen/v1");
		let headers =
			merge_provider_attribution_headers(&opencode, &settings, Some("s-123"), &[]).unwrap();
		assert_eq!(headers.get("x-opencode-session").unwrap().as_deref(), Some("s-123"));
		assert_eq!(headers.get("x-opencode-client").unwrap().as_deref(), Some("pi"));

		// Session headers are opencode-only.
		let other = model("anthropic", "https://api.anthropic.com");
		assert_eq!(
			merge_provider_attribution_headers(&other, &settings, Some("s-123"), &[]),
			None
		);
	}

	#[test]
	fn caller_headers_win_over_defaults() {
		let model = model("openrouter", "https://openrouter.ai/api/v1");
		let settings = settings_with_telemetry(true);
		let overrides: ProviderHeaders = BTreeMap::from([(
			"HTTP-Referer".to_string(),
			Some("https://example.com".to_string()),
		)]);
		let headers =
			merge_provider_attribution_headers(&model, &settings, None, &[&overrides]).unwrap();
		assert_eq!(headers.get("HTTP-Referer").unwrap().as_deref(), Some("https://example.com"));
	}

	#[test]
	fn invalid_base_url_does_not_match_hosts() {
		let model = model("openrouter", "not-a-url");
		let settings = settings_with_telemetry(true);
		// Provider id still matches openrouter.
		assert!(merge_provider_attribution_headers(&model, &settings, None, &[]).is_some());
	}
}
