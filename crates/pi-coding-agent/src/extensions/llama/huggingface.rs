//! Port of `packages/coding-agent/src/extensions/llama/huggingface.ts` — the
//! Hugging Face model search client: GGUF repository search, quantization
//! discovery from repository blobs, access-requirement detection, and token
//! discovery from the environment.
//!
//! # Deviations (D97/D98 range, shared with [`super`])
//!
//! - **D97** — cancellation uses `pi_ai::utils::abort::AbortSignal`; the
//!   15-second deadline is a `tokio::time::timeout` race (see
//!   [`super::client`]).
//! - **D98a** — the two upstream quantization/shard regexes become
//!   hand-rolled scanners with identical leftmost-match, greedy, case
//!   -insensitive semantics (no `regex` dependency in this crate).
//! - **D98b** — quantization sort falls back to byte-wise name comparison
//!   for equal sizes (upstream `localeCompare`; names are ASCII
//!   `A-Z0-9_`, so ordering differences are unreachable in practice).
//! - **D98c** — `homedir()` resolves via the `HOME` environment variable
//!   (the crate targets macOS/Linux; no `os.homedir()` equivalent without a
//!   new dependency).

use std::collections::BTreeMap;
use std::future::Future;
use std::time::Duration;

use pi_ai::utils::abort::AbortSignal;
use serde::Serialize;
use serde_json::Value as JsonValue;

const DEFAULT_HUGGING_FACE_URL: &str = "https://huggingface.co";

/// Deadline of a single Hugging Face request (upstream
/// `AbortSignal.timeout(15_000)`).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Upstream `HuggingFaceModel`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HuggingFaceModel {
	pub id: String,
	pub downloads: f64,
}

/// Upstream `HuggingFaceQuantization`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HuggingFaceQuantization {
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub size: Option<f64>,
}

/// Upstream `HuggingFaceModelDetails.gated` (`false | "auto" | "manual"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HuggingFaceGating {
	#[serde(rename = "false")]
	No,
	Auto,
	Manual,
}

/// Upstream `HuggingFaceModelDetails`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HuggingFaceModelDetails {
	pub id: String,
	pub gated: HuggingFaceGating,
	pub quantizations: Vec<HuggingFaceQuantization>,
}

/// Upstream `payloadError`.
fn payload_error(payload: &JsonValue, fallback: &str) -> String {
	payload
		.get("error")
		.and_then(|error| error.as_str())
		.filter(|error| !error.is_empty())
		.map(str::to_string)
		.unwrap_or_else(|| fallback.to_string())
}

/// Upstream `parseRateLimitDelay`: the `t=<seconds>` member of the
/// `ratelimit` header (`/(?:^|;)t=(\d+)/`).
fn parse_rate_limit_delay(value: Option<&str>) -> Option<f64> {
	let value = value?;
	let mut search_from = 0;
	while let Some(position) = value[search_from..].find("t=") {
		let absolute = search_from + position;
		if absolute == 0 || value[..absolute].ends_with(';') {
			let digits: &str = value[absolute + 2..]
				.split(|character: char| !character.is_ascii_digit())
				.next()
				.unwrap_or("");
			if !digits.is_empty() {
				return digits.parse::<f64>().ok();
			}
		}
		search_from = absolute + 2;
		if search_from >= value.len() {
			break;
		}
	}
	None
}

/// Read a token file, ignoring read errors (upstream `readToken`).
async fn read_token(path: &str) -> Option<String> {
	match tokio::fs::read_to_string(path).await {
		Ok(token) => {
			let token = token.trim();
			if token.is_empty() {
				None
			} else {
				Some(token.to_string())
			}
		}
		Err(_) => None,
	}
}

/// Resolve the token candidates from an environment map, deduplicated in
/// first-seen order (upstream `findHuggingFaceToken`).
async fn find_hugging_face_token_from_map(env: &BTreeMap<String, String>) -> Option<String> {
	let get = |name: &str| -> Option<String> {
		env.get(name)
			.map(|value| value.trim().to_string())
			.filter(|value| !value.is_empty())
	};
	if let Some(token) = get("HF_TOKEN") {
		return Some(token);
	}
	let home = std::env::var("HOME").unwrap_or_default();
	let mut candidates: Vec<String> = Vec::new();
	let mut push = |path: Option<String>| {
		if let Some(path) = path {
			if !candidates.contains(&path) {
				candidates.push(path);
			}
		}
	};
	push(get("HF_TOKEN_PATH"));
	push(get("HF_HOME").map(|home| format!("{home}/token")));
	push(get("XDG_CACHE_HOME").map(|cache| format!("{cache}/huggingface/token")));
	push(if home.is_empty() {
		None
	} else {
		Some(format!("{home}/.cache/huggingface/token"))
	});
	for path in &candidates {
		if let Some(token) = read_token(path).await {
			return Some(token);
		}
	}
	None
}

/// Upstream `findHuggingFaceToken()` with the default (`process.env`)
/// environment.
pub async fn find_hugging_face_token() -> Option<String> {
	let env: BTreeMap<String, String> = std::env::vars().collect();
	find_hugging_face_token_from_map(&env).await
}

/// Upstream `findHuggingFaceToken(env)` with an explicit environment map.
pub async fn find_hugging_face_token_in(env: &BTreeMap<String, String>) -> Option<String> {
	find_hugging_face_token_from_map(env).await
}

/// Hugging Face API client (upstream `HuggingFaceClient`).
#[derive(Clone, Debug)]
pub struct HuggingFaceClient {
	token: Option<String>,
	base_url: String,
	http: reqwest::Client,
}

/// Deadline race — mirrors [`crate::extensions::llama::client`] (D97).
async fn run_bounded<T>(future: impl Future<Output = T>, signal: Option<&AbortSignal>) -> Result<T, String> {
	let abort = async {
		match signal {
			Some(signal) => {
				signal.aborted_future().await;
			}
			None => std::future::pending::<()>().await,
		}
	};
	let reason = |signal: Option<&AbortSignal>| -> String {
		match signal.and_then(|signal| signal.reason()) {
			Some(reason) => reason.to_string(),
			None => "Cancelled".to_string(),
		}
	};
	tokio::select! {
		biased;
		_ = abort => Err(reason(signal)),
		output = tokio::time::timeout(REQUEST_TIMEOUT, future) => match output {
			Ok(result) => Ok(result),
			Err(_) => Err("TimeoutError: the request timed out".to_string()),
		},
	}
}

/// [`run_bounded`] over a fallible request future, flattening the transport
/// error into the message channel.
async fn run_bounded_ok<T, E: std::fmt::Display>(
	future: impl Future<Output = Result<T, E>>,
	signal: Option<&AbortSignal>,
) -> Result<T, String> {
	match run_bounded(future, signal).await {
		Ok(Ok(value)) => Ok(value),
		Ok(Err(error)) => Err(error.to_string()),
		Err(message) => Err(message),
	}
}

impl HuggingFaceClient {
	/// Upstream `new HuggingFaceClient(token?, baseUrl?)`.
	pub fn new(token: Option<&str>, base_url: Option<&str>) -> Self {
		HuggingFaceClient {
			token: token.map(str::to_string),
			base_url: base_url
				.unwrap_or(DEFAULT_HUGGING_FACE_URL)
				.trim_end_matches('/')
				.to_string(),
			http: reqwest::Client::new(),
		}
	}

	/// Upstream `request`: GET JSON with auth and the 15-second deadline.
	async fn request(&self, path: &str, signal: Option<&AbortSignal>) -> Result<JsonValue, String> {
		let mut request = self.http.get(format!("{}{}", self.base_url, path));
		if let Some(token) = &self.token {
			request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"));
		}
		let response = run_bounded_ok(request.send(), signal).await?;
		let status = response.status();
		let retry_after = response
			.headers()
			.get("retry-after")
			.and_then(|value| value.to_str().ok())
			.and_then(|value| value.parse::<f64>().ok());
		let rate_limit = response
			.headers()
			.get("ratelimit")
			.and_then(|value| value.to_str().ok())
			.map(str::to_string);
		// Upstream tolerates non-JSON bodies (`payload = undefined`).
		let payload = run_bounded_ok(response.json::<JsonValue>(), signal).await.unwrap_or(JsonValue::Null);
		if !status.is_success() {
			let fallback = format!("Hugging Face returned HTTP {}", status.as_u16());
			if status.as_u16() == 429 {
				return Err(match retry_after.or_else(|| parse_rate_limit_delay(rate_limit.as_deref())) {
					Some(delay) if delay > 0.0 => {
						format!("Hugging Face rate limit reached; retry in {}s", format_delay(delay))
					}
					_ => "Hugging Face rate limit reached".to_string(),
				});
			}
			return Err(payload_error(&payload, &fallback));
		}
		Ok(payload)
	}

	/// Upstream `search`: GGUF repositories ordered by downloads.
	pub async fn search(&self, query: &str, signal: Option<&AbortSignal>) -> Result<Vec<HuggingFaceModel>, String> {
		let params = url::form_urlencoded::Serializer::new(String::new())
			.append_pair("search", query)
			.append_pair("filter", "gguf")
			.append_pair("sort", "downloads")
			.append_pair("direction", "-1")
			.append_pair("limit", "20")
			.finish();
		let payload = self.request(&format!("/api/models?{params}"), signal).await?;
		let Some(entries) = payload.as_array() else {
			return Err("Hugging Face returned invalid search results".to_string());
		};
		Ok(entries
			.iter()
			.filter_map(|value| {
				let id = value.get("id")?.as_str()?.to_string();
				let downloads = value.get("downloads").and_then(|downloads| downloads.as_f64()).unwrap_or(0.0);
				Some(HuggingFaceModel { id, downloads })
			})
			.collect())
	}

	/// Upstream `details`: model access requirements and quantization sizes.
	pub async fn details(&self, id: &str, signal: Option<&AbortSignal>) -> Result<HuggingFaceModelDetails, String> {
		let encoded_id = id
			.split('/')
			.map(encode_uri_component)
			.collect::<Vec<_>>()
			.join("/");
		let payload = self.request(&format!("/api/models/{encoded_id}?blobs=true"), signal).await?;
		if !payload.is_object() {
			return Err("Hugging Face returned invalid model details".to_string());
		}
		let mut sizes: BTreeMap<String, (f64, bool)> = BTreeMap::new();
		if let Some(siblings) = payload.get("siblings").and_then(|siblings| siblings.as_array()) {
			for value in siblings {
				let Some(file) = value.get("rfilename").and_then(|name| name.as_str()) else {
					continue;
				};
				if !file.to_lowercase().ends_with(".gguf") {
					continue;
				}
				let filename = file.rsplit('/').next().unwrap_or(file);
				if filename.to_lowercase().starts_with("mmproj") {
					continue;
				}
				let stem = &filename[..filename.len() - ".gguf".len()];
				let stem = strip_shard_suffix(stem);
				let Some(quantization) = match_quantization(stem) else {
					continue;
				};
				let current = sizes.entry(quantization.to_uppercase()).or_insert((0.0, true));
				match value.get("size").and_then(|size| size.as_f64()) {
					Some(size) => current.0 += size,
					None => current.1 = false,
				}
			}
		}
		let mut quantizations: Vec<HuggingFaceQuantization> = sizes
			.into_iter()
			.map(|(name, (total, complete))| HuggingFaceQuantization {
				name,
				size: complete.then_some(total),
			})
			.collect();
		quantizations.sort_by(|left, right| {
			if left.name == "Q4_K_M" {
				return std::cmp::Ordering::Less;
			}
			if right.name == "Q4_K_M" {
				return std::cmp::Ordering::Greater;
			}
			let left_size = left.size.unwrap_or(MAX_SAFE_INTEGER);
			let right_size = right.size.unwrap_or(MAX_SAFE_INTEGER);
			left_size
				.partial_cmp(&right_size)
				.unwrap_or(std::cmp::Ordering::Equal)
				// D98b: byte-wise stand-in for `localeCompare`.
				.then_with(|| left.name.cmp(&right.name))
		});
		Ok(HuggingFaceModelDetails {
			id: payload
				.get("id")
				.and_then(|value| value.as_str())
				.map(str::to_string)
				.unwrap_or_else(|| id.to_string()),
			gated: match payload.get("gated").and_then(|value| value.as_str()) {
				Some("auto") => HuggingFaceGating::Auto,
				Some("manual") => HuggingFaceGating::Manual,
				_ => HuggingFaceGating::No,
			},
			quantizations,
		})
	}
}

/// JS `Number.MAX_SAFE_INTEGER` (missing sizes sort last).
const MAX_SAFE_INTEGER: f64 = 9007199254740991.0;

/// JS `Number` rendering of integral delays.
fn format_delay(delay: f64) -> String {
	if delay.fract() == 0.0 {
		format!("{}", delay as i64)
	} else {
		format!("{delay}")
	}
}

/// `encodeURIComponent` — unreserved characters per ECMA-262.
fn encode_uri_component(value: &str) -> String {
	let mut encoded = String::with_capacity(value.len());
	for byte in value.bytes() {
		match byte {
			b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'('
			| b')' => encoded.push(byte as char),
			other => encoded.push_str(&format!("%{other:02X}")),
		}
	}
	encoded
}

/// Upstream `SHARD_SUFFIX_PATTERN` (`-\d{5}-of-\d{5}$`).
fn strip_shard_suffix(stem: &str) -> &str {
	let bytes = stem.as_bytes();
	if bytes.len() < "-00000-of-00000".len() {
		return stem;
	}
	let head = &bytes[bytes.len() - "-00000-of-00000".len()..];
	let digits = |range: std::ops::Range<usize>| range.clone().all(|index| head[index].is_ascii_digit());
	let offset = bytes.len() - head.len();
	if head[0] == b'-'
		&& digits(1..6)
		&& head[6] == b'-'
		&& head[7] == b'o'
		&& head[8] == b'f'
		&& head[9] == b'-'
		&& digits(10..15)
	{
		&stem[..offset]
	} else {
		stem
	}
}

/// Whether `rest` matches one quantization alternative of upstream
/// `QUANTIZATION_PATTERN` (case-insensitive) and consumes it entirely.
fn matches_quantization_alternative(rest: &str) -> bool {
	let bytes = rest.as_bytes();
	let is_alnum = |byte: u8| byte.is_ascii_alphanumeric();
	// `(?:IQ\d(?:_[A-Z0-9]+)+|Q\d(?:_[A-Z0-9]+)+|BF16|F16|F32|MXFP\d(?:_[A-Z0-9]+)*)`
	let after_prefix = |index: usize| -> bool {
		let rest = &bytes[index..];
		let segment = |start: usize| -> bool {
			// `_[A-Z0-9]+`, greedy; the whole alternative must span to the end.
			let mut cursor = start;
			while cursor < rest.len() && rest[cursor] == b'_' {
				cursor += 1;
				let content_start = cursor;
				while cursor < rest.len() && is_alnum(rest[cursor]) {
					cursor += 1;
				}
				if cursor == content_start {
					return false;
				}
			}
			cursor == rest.len()
		};
		if rest.len() > 3 && rest[..2].eq_ignore_ascii_case(b"IQ") && rest[2].is_ascii_digit() {
			if segment(3) {
				return true;
			}
		}
		if rest.len() > 2 && rest[0].eq_ignore_ascii_case(&b'Q') && rest[1].is_ascii_digit() {
			if segment(2) {
				return true;
			}
		}
		if (rest.len() == 4 && rest.eq_ignore_ascii_case(b"BF16"))
			|| (rest.len() == 3 && (rest.eq_ignore_ascii_case(b"F16") || rest.eq_ignore_ascii_case(b"F32")))
		{
			return true;
		}
		if rest.len() > 4 && rest[..4].eq_ignore_ascii_case(b"MXFP") && rest[4].is_ascii_digit() {
			return segment(5);
		}
		false
	};
	if after_prefix(0) {
		return true;
	}
	// `(?:UD-)?` prefix.
	bytes.len() > 3
		&& bytes[..2].eq_ignore_ascii_case(b"UD")
		&& bytes[2] == b'-'
		&& after_prefix(3)
}

/// Upstream `QUANTIZATION_PATTERN` capture: leftmost `(?:^|[-_.])`-anchored
/// quantization suffix (D98a).
fn match_quantization(stem: &str) -> Option<&str> {
	let bytes = stem.as_bytes();
	for index in 0..bytes.len() {
		let anchored = index == 0 || matches!(bytes[index - 1], b'-' | b'_' | b'.');
		if anchored && matches_quantization_alternative(&stem[index..]) {
			return Some(&stem[index..]);
		}
	}
	None
}

// ============================================================================
// Tests — port of the huggingface-related cases from
// `test/llama-extension.test.ts` (see the shared test-support server).
// ============================================================================

#[cfg(test)]
mod huggingface_tests {
	use super::*;
	use crate::extensions::llama::test_support::{listen, TestResponse};
	use serde_json::json;
	use std::sync::Arc;

	#[test]
	fn quantization_scanner_matches_upstream_patterns() {
		assert_eq!(match_quantization("model-Q4_K_M"), Some("Q4_K_M"));
		assert_eq!(match_quantization("model-IQ4_XS"), Some("IQ4_XS"));
		assert_eq!(match_quantization("model-ud-q4_k_m"), Some("ud-q4_k_m"));
		assert_eq!(match_quantization("some-model-F16-F32"), Some("F32"));
		assert_eq!(match_quantization("mxfp4_abc"), Some("mxfp4_abc"));
		assert_eq!(match_quantization("plain"), None);
		assert_eq!(match_quantization("x.q8_0"), Some("q8_0"));
	}

	#[test]
	fn shard_suffix_stripper_matches_upstream_pattern() {
		assert_eq!(strip_shard_suffix("model-Q4_K_M-00001-of-00002"), "model-Q4_K_M");
		assert_eq!(strip_shard_suffix("model-Q4_K_M"), "model-Q4_K_M");
		assert_eq!(strip_shard_suffix("model-0001-of-0002"), "model-0001-of-0002");
	}

	#[tokio::test]
	async fn searches_hugging_face_and_reads_quantizations_plus_access_requirements() {
		let server = listen(Arc::new(|request| {
			let authorization = request.authorization.clone();
			let target = request.target.clone();
			Box::pin(async move {
				assert_eq!(authorization.as_deref(), Some("Bearer hf-secret"));
				if target.starts_with("/api/models?") {
					assert!(target.contains("search=qwen+coder"));
					assert!(target.contains("filter=gguf"));
					assert!(target.contains("sort=downloads"));
					return TestResponse::Json(200, json!([{ "id": "owner/model-GGUF", "downloads": 1200 }]));
				}
				if target == "/api/models/owner/model-GGUF?blobs=true" {
					return TestResponse::Json(
						200,
						json!({
							"id": "owner/model-GGUF",
							"gated": "manual",
							"siblings": [
								{ "rfilename": "model-Q5_K_M.gguf", "size": 6000 },
								{ "rfilename": "model-Q4_K_M-00001-of-00002.gguf", "size": 2000 },
								{ "rfilename": "model-Q4_K_M-00002-of-00002.gguf", "size": 3000 },
								{ "rfilename": "mmproj-F16.gguf", "size": 1000 },
							],
						}),
					);
				}
				TestResponse::Empty(404)
			})
		}))
		.await;
		let client = HuggingFaceClient::new(Some("hf-secret"), Some(&server.url));

		assert_eq!(
			client.search("qwen coder", None).await.unwrap(),
			vec![HuggingFaceModel { id: "owner/model-GGUF".to_string(), downloads: 1200.0 }]
		);
		assert_eq!(
			client.details("owner/model-GGUF", None).await.unwrap(),
			HuggingFaceModelDetails {
				id: "owner/model-GGUF".to_string(),
				gated: HuggingFaceGating::Manual,
				quantizations: vec![
					HuggingFaceQuantization { name: "Q4_K_M".to_string(), size: Some(5000.0) },
					HuggingFaceQuantization { name: "Q5_K_M".to_string(), size: Some(6000.0) },
				],
			}
		);

		let env = BTreeMap::from([("HF_TOKEN".to_string(), " hf-secret ".to_string())]);
		assert_eq!(find_hugging_face_token_in(&env).await.as_deref(), Some("hf-secret"));
	}
}
