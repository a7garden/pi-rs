//! Port of `packages/coding-agent/src/core/radius.ts` — the built-in Radius
//! provider constants and gateway URL resolution. Normalization is delegated
//! to the shared `pi_ai::providers::radius_config` port (no duplication).
//!
//! No deviations. Upstream's module-load `RADIUS_MCP_URL` becomes a
//! `LazyLock` static (same value, computed on first use); `get_radius_gateway_url`
//! reads the env at call time like upstream (`process.env`).

use std::sync::LazyLock;

use pi_ai::providers::radius_config::{normalize_radius_gateway_url, DEFAULT_RADIUS_GATEWAY};

/// Upstream `RADIUS_PROVIDER_ID`.
pub const RADIUS_PROVIDER_ID: &str = "radius";

/// Upstream `ENV_RADIUS_GATEWAY`.
pub const ENV_RADIUS_GATEWAY: &str = "PI_RADIUS_GATEWAY";

/// MCP endpoint of the gateway the built-in Radius provider signs in to
/// (upstream `RADIUS_MCP_URL`).
pub static RADIUS_MCP_URL: LazyLock<String> =
	LazyLock::new(|| format!("{}/mcp", normalize_radius_gateway_url(DEFAULT_RADIUS_GATEWAY)));

/// Radius gateway origin, honoring the `PI_RADIUS_GATEWAY` override
/// (upstream `getRadiusGatewayUrl`). Like upstream's `??`, a set-but-empty
/// override is kept as-is (normalizing to `https://`).
pub fn get_radius_gateway_url() -> String {
	let gateway = std::env::var(ENV_RADIUS_GATEWAY).unwrap_or_else(|_| DEFAULT_RADIUS_GATEWAY.to_string());
	normalize_radius_gateway_url(&gateway)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Serialized env access: `PI_RADIUS_GATEWAY` may be set in the host
	/// environment, so save/restore around each case (same lock pattern as
	/// `resolve_config_value_tests`).
	static ENV_LOCK: LazyLock<std::sync::Mutex<()>> = LazyLock::new(std::sync::Mutex::default);

	#[test]
	fn radius_mcp_url_appends_mcp_to_normalized_default_gateway() {
		assert_eq!(&*RADIUS_MCP_URL, "https://radius.pi.dev/mcp");
	}

	#[test]
	fn get_radius_gateway_url_honors_env_override() {
		let _guard = ENV_LOCK.lock().expect("lock");
		let original = std::env::var(ENV_RADIUS_GATEWAY).ok();

		std::env::remove_var(ENV_RADIUS_GATEWAY);
		assert_eq!(get_radius_gateway_url(), "https://radius.pi.dev");

		std::env::set_var(ENV_RADIUS_GATEWAY, "gateway.example");
		assert_eq!(get_radius_gateway_url(), "https://gateway.example");

		std::env::set_var(ENV_RADIUS_GATEWAY, "https://gateway.example/");
		assert_eq!(get_radius_gateway_url(), "https://gateway.example");

		match original {
			Some(value) => std::env::set_var(ENV_RADIUS_GATEWAY, value),
			None => std::env::remove_var(ENV_RADIUS_GATEWAY),
		}
	}
}
