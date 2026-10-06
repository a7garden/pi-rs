//! Port of `packages/ai/src/env-api-keys.ts` — environment API-key discovery.
//!
//! The lazy `node:fs`/`node:os`/`node:path` imports do not apply to Rust
//! (deviation D38): filesystem checks run natively, and the
//! browser-cache-false nuance of `hasVertexAdcCredentials` collapses to a
//! plain memoized check (Rust always "has fs").

use std::collections::BTreeMap;
use parking_lot::Mutex;

use crate::types::ProviderEnv;
use crate::utils::provider_env::get_provider_env_value;

pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
pub const ANTHROPIC_OAUTH_TOKEN_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_FEDERATION_RULE_ID_ENV: &str = "ANTHROPIC_FEDERATION_RULE_ID";
pub const ANTHROPIC_ORGANIZATION_ID_ENV: &str = "ANTHROPIC_ORGANIZATION_ID";
pub const ANTHROPIC_SERVICE_ACCOUNT_ID_ENV: &str = "ANTHROPIC_SERVICE_ACCOUNT_ID";
pub const ANTHROPIC_IDENTITY_TOKEN_FILE_ENV: &str = "ANTHROPIC_IDENTITY_TOKEN_FILE";
pub const ANTHROPIC_WORKSPACE_ID_ENV: &str = "ANTHROPIC_WORKSPACE_ID";

fn cached_vertex_adc_credentials_exists() -> &'static Mutex<Option<bool>> {
	static CACHE: std::sync::LazyLock<Mutex<Option<bool>>> = std::sync::LazyLock::new(|| Mutex::new(None));
	&CACHE
}

fn file_exists(path: &str) -> bool {
	std::path::Path::new(expand_home(path).as_str()).exists()
}

/// Leading `~` expansion (upstream composes `join(homedir(), ...)`).
fn expand_home(path: &str) -> String {
	if let Some(rest) = path.strip_prefix("~/") {
		if let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) {
			return home.join(rest).to_string_lossy().into_owned();
		}
	}
	path.to_string()
}

/// Upstream `hasVertexAdcCredentials`.
fn has_vertex_adc_credentials(env: Option<&ProviderEnv>) -> bool {
	if let Some(explicit_credentials_path) = env.and_then(|env| env.get("GOOGLE_APPLICATION_CREDENTIALS")) {
		return file_exists(explicit_credentials_path);
	}

	let mut cache = cached_vertex_adc_credentials_exists().lock();
	if cache.is_none() {
		// Check GOOGLE_APPLICATION_CREDENTIALS env var first (standard way)
		if let Some(gac_path) = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env) {
			*cache = Some(file_exists(&gac_path));
		} else {
			// Fall back to default ADC path (lazy evaluation)
			*cache = Some(file_exists("~/.config/gcloud/application_default_credentials.json"));
		}
	}
	cache.unwrap_or(false)
}

fn get_api_key_env_vars(provider: &str) -> Option<Vec<&'static str>> {
	if provider == "github-copilot" {
		return Some(vec!["COPILOT_GITHUB_TOKEN"]);
	}

	// ANTHROPIC_AUTH_TOKEN participates in env discovery/status, but
	// getEnvApiKey() skips it because requests must pass it as
	// Authorization: Bearer.
	if provider == "anthropic" {
		return Some(vec![
			ANTHROPIC_AUTH_TOKEN_ENV,
			ANTHROPIC_OAUTH_TOKEN_ENV,
			ANTHROPIC_API_KEY_ENV,
		]);
	}

	let env_var = match provider {
		"ant-ling" => "ANT_LING_API_KEY",
		"qwen-token-plan" => "QWEN_TOKEN_PLAN_API_KEY",
		"qwen-token-plan-cn" => "QWEN_TOKEN_PLAN_CN_API_KEY",
		"qwen-token-plan-individual" => "QWEN_TOKEN_PLAN_API_KEY",
		"openai" => "OPENAI_API_KEY",
		"azure" => "AZURE_OPENAI_API_KEY",
		"nvidia" => "NVIDIA_API_KEY",
		"deepseek" => "DEEPSEEK_API_KEY",
		"google" => "GEMINI_API_KEY",
		"google-vertex" => "GOOGLE_CLOUD_API_KEY",
		"groq" => "GROQ_API_KEY",
		"cerebras" => "CEREBRAS_API_KEY",
		"xai" => "XAI_API_KEY",
		"typesafe" => "TYPESAFE_API_KEY",
		"radius" => "RADIUS_API_KEY",
		"openrouter" => "OPENROUTER_API_KEY",
		"vercel-ai-gateway" => "AI_GATEWAY_API_KEY",
		"zai" => "ZAI_API_KEY",
		"zai-coding-cn" => "ZAI_CODING_CN_API_KEY",
		"mistral" => "MISTRAL_API_KEY",
		"minimax" => "MINIMAX_API_KEY",
		"minimax-cn" => "MINIMAX_CN_API_KEY",
		"moonshotai" => "MOONSHOT_API_KEY",
		"moonshotai-cn" => "MOONSHOT_API_KEY",
		"huggingface" => "HF_TOKEN",
		"fireworks" => "FIREWORKS_API_KEY",
		"together" => "TOGETHER_API_KEY",
		"baseten" => "BASETEN_API_KEY",
		"opencode" => "OPENCODE_API_KEY",
		"opencode-go" => "OPENCODE_API_KEY",
		"kimi-coding" => "KIMI_API_KEY",
		"meta" => "META_API_KEY",
		"cloudflare-workers-ai" => "CLOUDFLARE_API_KEY",
		"cloudflare-ai-gateway" => "CLOUDFLARE_API_KEY",
		"xiaomi" => "XIAOMI_API_KEY",
		"xiaomi-token-plan-cn" => "XIAOMI_TOKEN_PLAN_CN_API_KEY",
		"xiaomi-token-plan-ams" => "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
		"xiaomi-token-plan-sgp" => "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
		_ => return None,
	};
	Some(vec![env_var])
}

/// Upstream `findEnvKeys` — configured env vars that can provide an API key.
/// Only reports actual API key variables; ambient credential sources (AWS
/// profiles, IAM, Google ADC) are intentionally excluded.
pub fn find_env_keys(provider: &str, env: Option<&ProviderEnv>) -> Option<Vec<String>> {
	let env_vars = get_api_key_env_vars(provider)?;
	let found: Vec<String> = env_vars
		.into_iter()
		.filter(|env_var| get_provider_env_value(env_var, env).is_some())
		.map(str::to_string)
		.collect();
	if found.is_empty() {
		None
	} else {
		Some(found)
	}
}

/// Upstream `getEnvApiKey` — will not return API keys for providers that
/// require OAuth tokens (ANTHROPIC_AUTH_TOKEN is skipped).
pub fn get_env_api_key(provider: &str, env: Option<&ProviderEnv>) -> Option<String> {
	let env_keys = find_env_keys(provider, env);
	if let Some(first) = env_keys.as_ref().and_then(|keys| keys.first()) {
		let api_key_env = if provider == "anthropic" {
			env_keys
				.as_ref()
				.unwrap()
				.iter()
				.find(|key| key.as_str() != ANTHROPIC_AUTH_TOKEN_ENV)
		} else {
			Some(first)
		};
		if let Some(api_key_env) = api_key_env {
			return get_provider_env_value(api_key_env, env);
		}
	}

	// Vertex AI supports either an explicit API key or Application Default
	// Credentials (`gcloud auth application-default login`).
	if provider == "google-vertex" {
		let has_credentials = has_vertex_adc_credentials(env);
		let has_project =
			get_provider_env_value("GOOGLE_CLOUD_PROJECT", env).is_some() || get_provider_env_value("GCLOUD_PROJECT", env).is_some();
		let has_location = get_provider_env_value("GOOGLE_CLOUD_LOCATION", env).is_some();

		if has_credentials && has_project && has_location {
			return Some("<authenticated>".to_string());
		}
	}

	if provider == "amazon-bedrock" {
		// Amazon Bedrock supports multiple credential sources:
		// 1. AWS_PROFILE - named profile from ~/.aws/credentials
		// 2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY - standard IAM keys
		// 3. AWS_BEARER_TOKEN_BEDROCK - Bedrock bearer token
		// 4. AWS_CONTAINER_CREDENTIALS_RELATIVE_URI - ECS task roles
		// 5. AWS_CONTAINER_CREDENTIALS_FULL_URI - ECS task roles (full URI)
		// 6. AWS_WEB_IDENTITY_TOKEN_FILE - IRSA (IAM Roles for Service Accounts)
		if get_provider_env_value("AWS_PROFILE", env).is_some()
			|| (get_provider_env_value("AWS_ACCESS_KEY_ID", env).is_some()
				&& get_provider_env_value("AWS_SECRET_ACCESS_KEY", env).is_some())
			|| get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env).is_some()
			|| get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env).is_some()
			|| get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env).is_some()
			|| get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env).is_some()
		{
			return Some("<authenticated>".to_string());
		}
	}

	None
}

// BTreeMap is part of the ProviderEnv shape referenced above.
#[allow(unused)]
fn _env_shape(_: BTreeMap<String, String>) {}

#[cfg(test)]
mod tests {
	use super::*;

	// Port of test/env-api-keys.test.ts. Env is process-global; the cases run
	// sequentially under one lock and restore variables afterwards.
	#[test]
	fn resolves_env_keys_like_upstream() {
		let mut saved: Vec<(String, Option<String>)> = Vec::new();
		for key in [
			"COPILOT_GITHUB_TOKEN",
			"GH_TOKEN",
			"GITHUB_TOKEN",
			"ZAI_CODING_CN_API_KEY",
			"ANTHROPIC_AUTH_TOKEN",
			"ANTHROPIC_OAUTH_TOKEN",
			"ANTHROPIC_API_KEY",
		] {
			saved.push((key.to_string(), std::env::var(key).ok()));
			std::env::remove_var(key);
		}

		// "does not treat generic GitHub tokens as GitHub Copilot credentials"
		std::env::set_var("GH_TOKEN", "gh-token");
		std::env::set_var("GITHUB_TOKEN", "github-token");
		assert_eq!(find_env_keys("github-copilot", None), None);
		assert_eq!(get_env_api_key("github-copilot", None), None);

		// "resolves GitHub Copilot credentials from COPILOT_GITHUB_TOKEN"
		std::env::set_var("COPILOT_GITHUB_TOKEN", "copilot-token");
		assert_eq!(
			find_env_keys("github-copilot", None),
			Some(vec!["COPILOT_GITHUB_TOKEN".to_string()])
		);
		assert_eq!(get_env_api_key("github-copilot", None).as_deref(), Some("copilot-token"));

		// "resolves ZAI China Coding Plan credentials"
		std::env::set_var("ZAI_CODING_CN_API_KEY", "zai-coding-cn-token");
		assert_eq!(
			find_env_keys("zai-coding-cn", None),
			Some(vec!["ZAI_CODING_CN_API_KEY".to_string()])
		);
		assert_eq!(get_env_api_key("zai-coding-cn", None).as_deref(), Some("zai-coding-cn-token"));

		// "reports ANTHROPIC_AUTH_TOKEN but preserves OAuth token API key lookup"
		std::env::set_var("ANTHROPIC_AUTH_TOKEN", "auth-token");
		std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "oauth-token");
		std::env::set_var("ANTHROPIC_API_KEY", "api-key");
		assert_eq!(
			find_env_keys("anthropic", None),
			Some(vec![
				"ANTHROPIC_AUTH_TOKEN".to_string(),
				"ANTHROPIC_OAUTH_TOKEN".to_string(),
				"ANTHROPIC_API_KEY".to_string()
			])
		);
		assert_eq!(get_env_api_key("anthropic", None).as_deref(), Some("oauth-token"));

		// "does not return ANTHROPIC_AUTH_TOKEN as an API key"
		std::env::remove_var("ANTHROPIC_OAUTH_TOKEN");
		std::env::remove_var("ANTHROPIC_API_KEY");
		assert_eq!(find_env_keys("anthropic", None), Some(vec!["ANTHROPIC_AUTH_TOKEN".to_string()]));
		assert_eq!(get_env_api_key("anthropic", None), None);

		// "preserves ANTHROPIC_OAUTH_TOKEN as an API key"
		std::env::remove_var("ANTHROPIC_AUTH_TOKEN");
		std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "oauth-token");
		assert_eq!(find_env_keys("anthropic", None), Some(vec!["ANTHROPIC_OAUTH_TOKEN".to_string()]));
		assert_eq!(get_env_api_key("anthropic", None).as_deref(), Some("oauth-token"));

		// "falls back to ANTHROPIC_API_KEY for API key lookup"
		std::env::remove_var("ANTHROPIC_OAUTH_TOKEN");
		std::env::set_var("ANTHROPIC_API_KEY", "api-key");
		assert_eq!(get_env_api_key("anthropic", None).as_deref(), Some("api-key"));

		// restore
		for (key, value) in saved {
			match value {
				Some(value) => std::env::set_var(key, value),
				None => std::env::remove_var(&key),
			}
		}
	}
}
