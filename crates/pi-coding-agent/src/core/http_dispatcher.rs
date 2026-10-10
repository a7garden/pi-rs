//! Port of `packages/coding-agent/src/core/http-dispatcher.ts` — shared HTTP
//! dispatcher/proxy configuration.
//!
//! All items here are the reusable seam for every entry point that mirrors an
//! upstream `configureHttpDispatcher` call site (`main`, `rpc-entry`,
//! `cli/setup`, `modes/interactive`, and the `experimental/{durable,vacation}`
//! `configureHarnessHttp` shims).
//!
//! Deviations:
//! - **D119 — the undici global-dispatcher replacement has no reqwest
//!   equivalent and is deferred.** Upstream swaps Node's global dispatcher
//!   for an `EnvHttpProxyAgent` (idle body/headers timeouts, `proxyTunnel`
//!   CONNECT semantics, 2s auto-select-family attempt timeout) and
//!   re-installs global fetch. Rust has no process-global dispatcher seam —
//!   pi-ai builds its clients via `pi_ai::http::new_reqwest_fetch` — so
//!   [`configure_http_dispatcher`] validates + normalizes the timeout with
//!   the upstream error string and returns it, instead of installing a
//!   dispatcher. Proxying keeps working through the `HTTP_PROXY`/`HTTPS_PROXY`
//!   env vars set by [`apply_http_proxy_settings`], which reqwest clients
//!   honor at build time. The undici-only
//!   `DEFAULT_AUTO_SELECT_FAMILY_ATTEMPT_TIMEOUT_MS` constant has no
//!   counterpart and is not carried over.
//! - [`parse_http_idle_timeout_ms`] and
//!   [`DEFAULT_HTTP_IDLE_TIMEOUT_MS`] previously lived as placeholders in
//!   `core::settings_manager` (upstream `settings-manager.ts` imports them
//!   from this module); flip that file to a re-export on integration.

use serde_json::Value;

/// Default idle timeout for HTTP headers/body chunks: five minutes
/// (`DEFAULT_HTTP_IDLE_TIMEOUT_MS`).
pub const DEFAULT_HTTP_IDLE_TIMEOUT_MS: u64 = 300_000;

/// One selectable idle-timeout choice (`HTTP_IDLE_TIMEOUT_CHOICES` item).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpIdleTimeoutChoice {
    pub label: &'static str,
    pub timeout_ms: u64,
}

/// Selectable idle-timeout choices (`HTTP_IDLE_TIMEOUT_CHOICES`).
pub const HTTP_IDLE_TIMEOUT_CHOICES: &[HttpIdleTimeoutChoice] = &[
    HttpIdleTimeoutChoice {
        label: "30 sec",
        timeout_ms: 30_000,
    },
    HttpIdleTimeoutChoice {
        label: "1 min",
        timeout_ms: 60_000,
    },
    HttpIdleTimeoutChoice {
        label: "2 min",
        timeout_ms: 120_000,
    },
    HttpIdleTimeoutChoice {
        label: "5 min",
        timeout_ms: 300_000,
    },
    HttpIdleTimeoutChoice {
        label: "disabled",
        timeout_ms: 0,
    },
];

fn value_as_finite_f64(value: &Value) -> Option<f64> {
    let number = value.as_f64()?;
    number.is_finite().then_some(number)
}

/// `parseHttpIdleTimeoutMs`: accepts a finite non-negative number (floored)
/// or a string (`"disabled"` → 0, empty → `None`, otherwise numeric).
pub fn parse_http_idle_timeout_ms(value: &Value) -> Option<u64> {
    if let Some(text) = value.as_str() {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("disabled") {
            return Some(0);
        }
        if trimmed.is_empty() {
            return None;
        }
        return parse_http_idle_timeout_ms(&Value::from(trimmed.parse::<f64>().ok()?));
    }
    let number = value_as_finite_f64(value)?;
    (number >= 0.0).then_some(number.floor() as u64)
}

/// `formatHttpIdleTimeoutMs` — the choice label, else `${timeoutMs / 1000}
/// sec`.
pub fn format_http_idle_timeout_ms(timeout_ms: u64) -> String {
    if let Some(choice) = HTTP_IDLE_TIMEOUT_CHOICES
        .iter()
        .find(|item| item.timeout_ms == timeout_ms)
    {
        return choice.label.to_string();
    }
    if timeout_ms % 1000 == 0 {
        format!("{} sec", timeout_ms / 1000)
    } else {
        format!("{} sec", timeout_ms as f64 / 1000.0)
    }
}

/// `applyHttpProxySettings` — seed `HTTP_PROXY`/`HTTPS_PROXY` from the
/// `httpProxy` setting without overriding existing values (upstream `??=`:
/// an already-present, empty value also wins).
pub fn apply_http_proxy_settings(http_proxy: Option<&str>) {
    let Some(proxy) = http_proxy.map(str::trim).filter(|proxy| !proxy.is_empty()) else {
        return;
    };
    if std::env::var_os("HTTP_PROXY").is_none() {
        std::env::set_var("HTTP_PROXY", proxy);
    }
    if std::env::var_os("HTTPS_PROXY").is_none() {
        std::env::set_var("HTTPS_PROXY", proxy);
    }
}

/// `configureHttpDispatcher(timeoutMs = DEFAULT_HTTP_IDLE_TIMEOUT_MS)` —
/// validates and normalizes the idle timeout with the upstream error string.
/// Returns the normalized value; the undici global-dispatcher installation is
/// deferred (D119). Callers pass the value they already own
/// (`getHttpIdleTimeoutMs`), or [`DEFAULT_HTTP_IDLE_TIMEOUT_MS`] for the
/// upstream no-argument form.
pub fn configure_http_dispatcher(timeout_ms: u64) -> Result<u64, String> {
    let normalized = parse_http_idle_timeout_ms(&Value::from(timeout_ms));
    match normalized {
        Some(normalized) => Ok(normalized),
        None => Err(format!("Invalid HTTP idle timeout: {timeout_ms}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_env_lock;

    fn parse(value: Value) -> Option<u64> {
        parse_http_idle_timeout_ms(&value)
    }

    #[test]
    fn parses_idle_timeout_values() {
        assert_eq!(parse(Value::Null), None);
        assert_eq!(parse(Value::Bool(true)), None);
        assert_eq!(parse(Value::from(0)), Some(0));
        assert_eq!(parse(Value::from(300_000)), Some(300_000));
        assert_eq!(parse(Value::from(1500.9)), Some(1500));
        assert_eq!(parse(Value::from(-1)), None);
        assert_eq!(parse(Value::from(f64::NAN)), None);
        assert_eq!(parse(Value::from(f64::INFINITY)), None);
    }

    #[test]
    fn parses_idle_timeout_strings() {
        assert_eq!(parse(Value::from("disabled")), Some(0));
        assert_eq!(parse(Value::from("DISABLED")), Some(0));
        assert_eq!(parse(Value::from("  Disabled  ")), Some(0));
        assert_eq!(parse(Value::from("")), None);
        assert_eq!(parse(Value::from("   ")), None);
        assert_eq!(parse(Value::from("300")), Some(300));
        assert_eq!(parse(Value::from(" 300.7 ")), Some(300));
        assert_eq!(parse(Value::from("junk")), None);
    }

    #[test]
    fn formats_idle_timeout_labels() {
        let labels = [
            (30_000, "30 sec"),
            (60_000, "1 min"),
            (120_000, "2 min"),
            (300_000, "5 min"),
            (0, "disabled"),
        ];
        for (timeout_ms, label) in labels {
            assert_eq!(format_http_idle_timeout_ms(timeout_ms), label);
        }
        assert_eq!(format_http_idle_timeout_ms(45_000), "45 sec");
        assert_eq!(format_http_idle_timeout_ms(45_500), "45.5 sec");
    }

    #[test]
    fn configures_the_dispatcher_with_the_default_timeout() {
        assert_eq!(
            configure_http_dispatcher(DEFAULT_HTTP_IDLE_TIMEOUT_MS),
            Ok(300_000)
        );
        assert_eq!(configure_http_dispatcher(60_000), Ok(60_000));
        assert_eq!(configure_http_dispatcher(0), Ok(0));
        assert_eq!(configure_http_dispatcher(u64::MAX), Ok(u64::MAX));
    }

    // Upstream "applies httpProxy to HTTP_PROXY and HTTPS_PROXY".
    #[test]
    fn applies_http_proxy_to_both_proxy_env_vars() {
        let _env = test_env_lock();
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");
        apply_http_proxy_settings(Some("http://127.0.0.1:7890"));
        assert_eq!(
            std::env::var("HTTP_PROXY").as_deref(),
            Ok("http://127.0.0.1:7890")
        );
        assert_eq!(
            std::env::var("HTTPS_PROXY").as_deref(),
            Ok("http://127.0.0.1:7890")
        );
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");
    }

    // Upstream "does not override existing proxy env vars".
    #[test]
    fn does_not_override_existing_proxy_env_vars() {
        let _env = test_env_lock();
        std::env::set_var("HTTP_PROXY", "http://env-http:8080");
        std::env::set_var("HTTPS_PROXY", "http://env-https:8080");
        apply_http_proxy_settings(Some("http://settings:7890"));
        assert_eq!(
            std::env::var("HTTP_PROXY").as_deref(),
            Ok("http://env-http:8080")
        );
        assert_eq!(
            std::env::var("HTTPS_PROXY").as_deref(),
            Ok("http://env-https:8080")
        );
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");
    }

    // Upstream "ignores empty values" — including a present-but-empty var
    // (JS `??=` only fills unset variables).
    #[test]
    fn ignores_empty_values_and_keeps_empty_env_vars() {
        let _env = test_env_lock();
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");
        apply_http_proxy_settings(Some("   "));
        assert!(std::env::var_os("HTTP_PROXY").is_none());
        assert!(std::env::var_os("HTTPS_PROXY").is_none());

        apply_http_proxy_settings(None);
        assert!(std::env::var_os("HTTP_PROXY").is_none());
        assert!(std::env::var_os("HTTPS_PROXY").is_none());

        std::env::set_var("HTTP_PROXY", "");
        apply_http_proxy_settings(Some("http://settings:7890"));
        assert_eq!(std::env::var("HTTP_PROXY").as_deref(), Ok(""));
        assert_eq!(
            std::env::var("HTTPS_PROXY").as_deref(),
            Ok("http://settings:7890")
        );
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");
    }
}
