//! Port of `packages/coding-agent/src/utils/version-check.ts` — pi.dev version
//! checks. Also carries the reduced `pi-user-agent.ts` port
//! (`get_pi_user_agent`).
//!
//! Deviations (crate docs, D12/D13): `semver` is a small in-module parser
//! (subset below); `formatVersionCheckError` collapses to the error's Display
//! text — Rust io errors carry no Node errno cause chain; the HTTP transport
//! goes through [`crate::utils::management_http::Fetcher`] (D12).

use serde_json::Value;

use crate::utils::management_http::{fetch_with_retry, FetchRetryOptions, Fetcher};
use crate::utils::pi_user_agent::get_pi_user_agent;
use std::sync::Arc;

const LATEST_VERSION_URL: &str = "https://pi.dev/api/latest-version";
const DEFAULT_VERSION_CHECK_TIMEOUT_MS: u64 = 10_000;

/// Latest release metadata (upstream `LatestPiRelease`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatestPiRelease {
	pub version: String,
	pub package_name: Option<String>,
	pub note: Option<String>,
}

/// Include useful errno details hidden behind Node's generic "fetch failed"
/// error (`formatVersionCheckError`). Rust errors have no errno cause chain,
/// so this renders the error's own message (D12).
pub fn format_version_check_error<E: std::fmt::Display>(error: &E) -> String {
	error.to_string()
}

/// A parsed semver subset: `major.minor.patch[-prerelease][+build]`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SemVer {
	major: u64,
	minor: u64,
	patch: u64,
	prerelease: Vec<PrereleaseId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PrereleaseId {
	Numeric(u64),
	Alpha(String),
}

impl SemVer {
	/// Upstream `valid()`: `None` unless the string is a full valid semver.
	fn parse(value: &str) -> Option<SemVer> {
		let (core, rest) = match value.find(['-', '+']) {
			Some(i) => (&value[..i], &value[i..]),
			None => (value, ""),
		};
		let mut parts = core.split('.');
		let major = parts.next()?.parse().ok()?;
		let minor = parts.next()?.parse().ok()?;
		let patch = parts.next()?.parse().ok()?;
		if parts.next().is_some() {
			return None;
		}
		let mut prerelease = Vec::new();
		if let Some(pre) = rest.strip_prefix('-') {
			let pre = pre.split('+').next().unwrap_or(pre);
			if pre.is_empty() {
				return None;
			}
			for id in pre.split('.') {
				if id.is_empty() {
					return None;
				}
				if !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()) {
					// Leading zeros are invalid numeric identifiers.
					if id.len() > 1 && id.starts_with('0') {
						return None;
					}
					prerelease.push(PrereleaseId::Numeric(id.parse().ok()?));
				} else {
					if !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
						return None;
					}
					prerelease.push(PrereleaseId::Alpha(id.to_string()));
				}
			}
		} else if rest.starts_with('+') {
			// Build metadata alone is fine; already handled by split below.
		} else if !rest.is_empty() {
			return None;
		}
		Some(SemVer { major, minor, patch, prerelease })
	}

	fn cmp(&self, other: &SemVer) -> std::cmp::Ordering {
		use std::cmp::Ordering;
		let ord = self
			.major
			.cmp(&other.major)
			.then(self.minor.cmp(&other.minor))
			.then(self.patch.cmp(&other.patch));
		if ord != Ordering::Equal {
			return ord;
		}
		match (self.prerelease.is_empty(), other.prerelease.is_empty()) {
			(true, true) => Ordering::Equal,
			// A release outranks any prerelease of the same version.
			(true, false) => Ordering::Greater,
			(false, true) => Ordering::Less,
			(false, false) => {
				for (a, b) in self.prerelease.iter().zip(other.prerelease.iter()) {
					let ord = match (a, b) {
						(PrereleaseId::Numeric(a), PrereleaseId::Numeric(b)) => a.cmp(b),
						// Numeric identifiers are lower than alphanumeric ones.
						(PrereleaseId::Numeric(_), PrereleaseId::Alpha(_)) => Ordering::Less,
						(PrereleaseId::Alpha(_), PrereleaseId::Numeric(_)) => Ordering::Greater,
						(PrereleaseId::Alpha(a), PrereleaseId::Alpha(b)) => a.cmp(b),
					};
					if ord != Ordering::Equal {
						return ord;
					}
				}
				self.prerelease.len().cmp(&other.prerelease.len())
			}
		}
	}
}

/// Upstream `comparePackageVersions`: `None` when either version is invalid.
pub fn compare_package_versions(left_version: &str, right_version: &str) -> Option<std::cmp::Ordering> {
	let left = SemVer::parse(left_version.trim())?;
	let right = SemVer::parse(right_version.trim())?;
	Some(left.cmp(&right))
}

/// Upstream `isNewerPackageVersion`.
pub fn is_newer_package_version(candidate_version: &str, current_version: &str) -> bool {
	if let Some(ordering) = compare_package_versions(candidate_version, current_version) {
		return ordering == std::cmp::Ordering::Greater;
	}
	candidate_version.trim() != current_version.trim()
}

fn pi_offline() -> bool {
	std::env::var("PI_OFFLINE").map(|v| !v.is_empty()).unwrap_or(false)
}

fn pi_skip_version_check() -> bool {
	std::env::var("PI_SKIP_VERSION_CHECK").map(|v| !v.is_empty()).unwrap_or(false)
}

/// Upstream `getLatestPiRelease`. The fetcher is injected (D12); pass
/// [`crate::utils::management_http::ReqwestFetcher`] in production.
pub async fn get_latest_pi_release(
	fetcher: Arc<dyn Fetcher>,
	current_version: &str,
	options: FetchCheckOptions,
) -> std::io::Result<Option<LatestPiRelease>> {
	if pi_offline() {
		return Ok(None);
	}

	let headers = vec![
		("User-Agent".to_string(), get_pi_user_agent(current_version)),
		("accept".to_string(), "application/json".to_string()),
	];
	let retry_options = FetchRetryOptions {
		max_retries: Some(if options.retry { 2 } else { 0 }),
		timeout_ms: Some(options.timeout_ms.unwrap_or(DEFAULT_VERSION_CHECK_TIMEOUT_MS)),
		..Default::default()
	};
	let response = match fetch_with_retry(fetcher, LATEST_VERSION_URL, &headers, &retry_options).await {
		Ok(response) => response,
		Err(error) => return Err(error),
	};
	if !response.ok() {
		return Ok(None);
	}

	let data: Value = serde_json::from_slice(&response.body)
		.map_err(|e| std::io::Error::other(format!("version check parse failed: {e}")))?;
	let version = match data.get("version").and_then(Value::as_str) {
		Some(version) if !version.trim().is_empty() => version.trim().to_string(),
		_ => return Ok(None),
	};
	let package_name = data
		.get("packageName")
		.and_then(Value::as_str)
		.map(str::trim)
		.filter(|s| !s.is_empty())
		.map(str::to_string);
	let note = data.get("note").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
	Ok(Some(LatestPiRelease { version, package_name, note }))
}

/// Upstream `getLatestPiRelease(currentVersion, options)` options.
#[derive(Debug, Clone, Copy, Default)]
pub struct FetchCheckOptions {
	pub timeout_ms: Option<u64>,
	pub retry: bool,
}

/// Upstream `getLatestPiVersion`.
pub async fn get_latest_pi_version(
	fetcher: Arc<dyn Fetcher>,
	current_version: &str,
	options: FetchCheckOptions,
) -> std::io::Result<Option<String>> {
	Ok(get_latest_pi_release(fetcher, current_version, options).await?.map(|release| release.version))
}

/// Upstream `checkForNewPiVersion`: never errors; `PI_SKIP_VERSION_CHECK`
/// short-circuits.
pub async fn check_for_new_pi_version(fetcher: Arc<dyn Fetcher>, current_version: &str) -> Option<LatestPiRelease> {
	if pi_skip_version_check() {
		return None;
	}
	let latest_release = get_latest_pi_release(fetcher, current_version, FetchCheckOptions::default())
		.await
		.ok()
		.flatten();
	match latest_release {
		Some(release) if is_newer_package_version(&release.version, current_version) => Some(release),
		_ => None,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use parking_lot::Mutex;

	/// Serializes tests that mutate/read PI_* env vars against tests that
	/// assert fetch behavior (cargo runs them in parallel threads).
	fn env_lock() -> parking_lot::MutexGuard<'static, ()> {
		crate::utils::test_env_lock()
	}
	use crate::utils::management_http::HttpResponse;
	use std::future::Future;

	#[test]
	fn compares_package_versions() {
		assert_eq!(compare_package_versions("0.70.6", "0.70.5"), Some(std::cmp::Ordering::Greater));
		assert_eq!(compare_package_versions("0.70.5", "0.70.5"), Some(std::cmp::Ordering::Equal));
		assert_eq!(compare_package_versions("0.70.4", "0.70.5"), Some(std::cmp::Ordering::Less));
		assert_eq!(compare_package_versions("5.0.0-beta.20", "5.0.0-beta.9"), Some(std::cmp::Ordering::Greater));
		assert!(!is_newer_package_version("0.70.5", "0.70.5"));
		assert!(is_newer_package_version("0.70.6", "0.70.5"));
	}

	#[test]
	fn invalid_versions_compare_as_none() {
		assert_eq!(compare_package_versions("abc", "1.0.0"), None);
		assert_eq!(compare_package_versions("1.0", "1.0.0"), None);
		// Fallback: unequal trims count as newer.
		assert!(is_newer_package_version("beta", "1.0.0"));
	}

	#[test]
	fn prerelease_orders_below_release() {
		assert_eq!(compare_package_versions("5.0.0-beta.1", "5.0.0"), Some(std::cmp::Ordering::Less));
		assert_eq!(compare_package_versions("5.0.0-alpha", "5.0.0-beta"), Some(std::cmp::Ordering::Less));
		assert_eq!(compare_package_versions("1.0.0+build", "1.0.0"), Some(std::cmp::Ordering::Equal));
	}

	#[test]
	fn user_agent_shape() {
		assert!(get_pi_user_agent("1.2.3").starts_with("pi/1.2.3 ("));
	}

	#[tokio::test]
	async fn returns_only_newer_versions() {
		let _env = env_lock();
		let fetcher = json_fetcher(r#"{"version":"1.2.3"}"#);
		assert_eq!(check_for_new_pi_version(fetcher.clone(), "1.2.3").await, None);
		assert_eq!(
			check_for_new_pi_version(fetcher, "1.2.2").await,
			Some(LatestPiRelease { version: "1.2.3".into(), package_name: None, note: None })
		);
	}

	#[tokio::test]
	async fn uses_the_pi_dev_version_check_api_with_a_pi_user_agent() {
		let _env = env_lock();
		let fetcher = Arc::new(CaptureFetcher::new(r#"{"version":"1.2.4"}"#));
		let version = get_latest_pi_version(fetcher.clone(), "1.2.3", FetchCheckOptions::default()).await.unwrap();
		assert_eq!(version.as_deref(), Some("1.2.4"));
		let (url, headers) = fetcher.last_request();
		assert_eq!(url, "https://pi.dev/api/latest-version");
		let user_agent = headers.iter().find(|(k, _)| k == "User-Agent").map(|(_, v)| v.clone()).unwrap();
		assert!(user_agent.starts_with("pi/1.2.3 "));
		assert!(headers.iter().any(|(k, v)| k == "accept" && v == "application/json"));
	}

	#[tokio::test]
	async fn returns_active_package_metadata_and_notes() {
		let _env = env_lock();
		let fetcher = json_fetcher(r#"{"packageName":"@new-scope/pi","version":"1.2.4"}"#);
		assert_eq!(
			get_latest_pi_release(fetcher.clone(), "1.2.3", FetchCheckOptions::default()).await.unwrap(),
			Some(LatestPiRelease { version: "1.2.4".into(), package_name: Some("@new-scope/pi".into()), note: None })
		);

		let fetcher = json_fetcher(r#"{"note":" **Read this** ","version":"1.2.4"}"#);
		assert_eq!(
			get_latest_pi_release(fetcher, "1.2.3", FetchCheckOptions::default()).await.unwrap(),
			Some(LatestPiRelease { version: "1.2.4".into(), package_name: None, note: Some("**Read this**".into()) })
		);
	}

	#[tokio::test]
	async fn skips_automatic_checks_when_disabled_but_allows_direct_calls() {
		let _env = env_lock();
		// SAFETY: tests touching distinct env keys run serialized by the
		// coordinator (single test binary; keys unique to this suite).
		std::env::set_var("PI_SKIP_VERSION_CHECK", "1");
		let fetcher = Arc::new(CaptureFetcher::new(r#"{"version":"1.2.4"}"#));
		assert_eq!(check_for_new_pi_version(fetcher.clone(), "1.2.3").await, None);
		assert_eq!(fetcher.calls(), 0);
		let version = get_latest_pi_version(fetcher.clone(), "1.2.3", FetchCheckOptions::default()).await.unwrap();
		assert_eq!(version.as_deref(), Some("1.2.4"));
		assert_eq!(fetcher.calls(), 1);
		std::env::remove_var("PI_SKIP_VERSION_CHECK");
	}

	#[tokio::test]
	async fn offline_env_disables_even_direct_calls() {
		let _env = env_lock();
		std::env::set_var("PI_OFFLINE", "1");
		let fetcher = Arc::new(CaptureFetcher::new(r#"{"version":"1.2.4"}"#));
		assert_eq!(get_latest_pi_version(fetcher.clone(), "1.2.3", FetchCheckOptions::default()).await.unwrap(), None);
		assert_eq!(fetcher.calls(), 0);
		std::env::remove_var("PI_OFFLINE");
	}

	fn json_fetcher(body: &str) -> Arc<dyn Fetcher> {
		Arc::new(CaptureFetcher::new(body))
	}

	/// Fetcher recording request details and returning a fixed JSON body.
	struct CaptureFetcher {
		body: Vec<u8>,
		requests: std::sync::Mutex<Vec<(String, Vec<(String, String)>)>>,
	}

	impl CaptureFetcher {
		fn new(body: &str) -> Self {
			CaptureFetcher { body: body.as_bytes().to_vec(), requests: std::sync::Mutex::new(Vec::new()) }
		}

		fn calls(&self) -> u32 {
			self.requests.lock().unwrap().len() as u32
		}

		fn last_request(&self) -> (String, Vec<(String, String)>) {
			self.requests.lock().unwrap().last().cloned().unwrap()
		}
	}

	impl Fetcher for CaptureFetcher {
		fn fetch(
			&self,
			url: &str,
			headers: &[(String, String)],
			_timeout_ms: Option<u64>,
		) -> std::pin::Pin<Box<dyn Future<Output = std::io::Result<HttpResponse>> + Send + '_>> {
			let url = url.to_string();
			let headers = headers.to_vec();
			Box::pin(async move {
				self.requests.lock().unwrap().push((url, headers));
				Ok(HttpResponse { status: 200, headers: Vec::new(), body: self.body.clone() })
			})
		}
	}
}
