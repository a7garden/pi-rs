//! Port of `packages/ai/src/api/bedrock-converse-stream.ts`.
//!
//! Upstream drives the AWS SDK (`@aws-sdk/client-bedrock-runtime`,
//! `BedrockRuntimeClient.send(ConverseStreamCommand)` over the default
//! credential chain) and iterates the binary AWS event-stream body. The Rust
//! port hand-rolls the same observable behavior over the
//! [`crate::http::FetchFn`] seam with no new dependencies:
//!
//! * **SigV4** (`sign_request`) implemented privately in this module
//!   (HMAC-SHA256 over `sha2`, standard construction; no `hmac` crate).
//!   Requests are `POST {endpoint}/model/{modelId}/converse-stream` with the
//!   `ConverseStreamInput` JSON body byte-compatible with the SDK's
//!   serialization.
//! * **Credentials** (`resolve_client_config` + `shared_profile_credentials`)
//!   reproduce the SDK's observable resolution: env vars
//!   (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`), the
//!   shared credentials/config files (`~/.aws/credentials`,
//!   `~/.aws/config`) with `AWS_PROFILE`/`profile` option selection, the
//!   bearer-token path (`options.apiKey`, `AWS_BEARER_TOKEN_BEDROCK`), and
//!   `AWS_BEDROCK_SKIP_AUTH`.
//! * **Endpoint resolution** mirrors `getStandardBedrockEndpointRegion` /
//!   `shouldUseExplicitBedrockEndpoint` (standard runtime URLs are only pinned
//!   when no region/profile is configured; custom endpoints always pass
//!   through; ARN-embedded regions win).
//! * **Response** decoding (`EventStreamDecoder`) implements the AWS
//!   event-stream framing per spec: 8-byte prelude (total length, headers
//!   length, prelude CRC32), header map (all nine value types), payload, and
//!   message CRC32. CRC mismatches are errors, like the SDK.
//!
//! # Deviations (per-file ledger)
//!
//! - **D15** — the SDK client/middleware stack has no Rust equivalent.
//!   `addCustomHeadersMiddleware` becomes direct header injection (reserved
//!   `authorization`/`host`/`x-amz-*` keys still skipped case-insensitively);
//!   `addResponseHeadersMiddleware` becomes invoking `onResponse` with the raw
//!   response as soon as it arrives (the SDK's modeled-metadata fallback after
//!   `send()` is unobservable here because the raw response is always
//!   available). The structural-guard half of
//!   `bedrock-custom-headers.test.ts` VC3 (requests without a headers object)
//!   has no equivalent.
//! - **D16** — SigV4 signing and the AWS event-stream decoder replace the
//!   SDK transport. The signed header set is
//!   `content-type` + caller headers + `host` + `x-amz-date` +
//!   `x-amz-content-sha256` (+ `x-amz-security-token`); the SDK signs the same
//!   set plus SDK-internal headers (`amz-sdk-*`, `content-length`,
//!   `accept-encoding`) that no test observes.
//! - **D17** — per-API options cannot ride `StreamOptions` (`types.rs` is
//!   frozen; deviation follows crate D2 pattern): [`BedrockOptions`] lives in
//!   this module with the upstream-exact extra fields, `pub fn stream` /
//!   `pub fn stream_simple` build its `Default` and delegate to
//!   `stream_with_options` / `stream_simple_with_options`; final integration
//!   should add the nested `StreamOptions` group (reported in reexports).
//! - **D18** — retry goes through `utils/provider_retry.rs` (crate D20
//!   pattern). Upstream relies on the SDK default retry strategy (standard
//!   mode, 3 total attempts for throttling/transient failures); the port
//!   defaults to two retries via `provider_retry`'s OpenAI/Anthropic-style
//!   classification (408/409/429/5xx) — close but not identical. Caller
//!   `maxRetries` overrides the default (upstream ignores it for Bedrock).
//! - **D19** — HTTP/2 vs HTTP/1.1 request-handler selection
//!   (`AWS_BEDROCK_FORCE_HTTP1`, `NodeHttpHandler` + proxy agents from
//!   `utils/node-http-proxy.ts`, deferred in the core wave) has no Rust
//!   equivalent; the env var is accepted but has no effect, and proxying is
//!   out of scope.
//! - **D20** — the browser fallback branch (`config.region` without file
//!   resolution) is dropped; the Rust process always resolves like Node.
//! - **D21** — non-Error throws: upstream's catch handles arbitrary thrown
//!   values; the Rust failure domain is [`StreamFailure`], which carries the
//!   SDK-exception shape (name/message/metadata/body) or a plain message.
//! - **D22** — upstream test cases that feed `{type: "unknown"}` content
//!   blocks (bedrock-convert-messages) or lone surrogates (D5:
//!   `sanitizeSurrogates` is the identity on valid UTF-8) are unrepresentable
//!   against the crate's closed content enums and are omitted; the observable
//!   placeholder/blank-filtering behavior is covered by the representable
//!   cases.
//! - **D23** — tests assert on captured [`crate::http::HttpRequest`]s and on
//!   the resolved client config (returned by `resolve_client_config`) where
//!   upstream asserted on mocked SDK client constructors/middleware.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use base64::Engine as _;
use futures_util::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::http::{BodyStream, FetchFn, HttpBody, HttpRequest, HttpResponse};
use crate::models::{calculate_cost, ProviderStreams};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    AssistantMessageEventStream, CacheRetention, Message, Model, ModelThinkingLevel,
    ProviderResponse, SimpleStreamOptions, StopReason, StreamDoneReason, StreamErrorReason,
    StreamOptions, ThinkingBudgets, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolChoice,
    TranscriptContext, UserContent, UserContentField,
};
use crate::utils::abort::AbortSignal;
use crate::utils::diagnostics::{append_assistant_message_diagnostic, now_ms};
use crate::utils::event_stream::{done_event, error_event};
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::get_system_message_text;
use crate::utils::transcript::{
    collapse_system_messages, get_current_tools, get_initial_system_message,
    without_initial_system_message,
};

// ---------------------------------------------------------------------------
// Options (deviation D17: upstream `BedrockOptions extends StreamOptions`)
// ---------------------------------------------------------------------------

/// How thinking content is returned in responses (`BedrockThinkingDisplay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockThinkingDisplay {
    Summarized,
    Omitted,
}

/// Bedrock tool choice (`"auto" | "any" | "none" | { type: "tool", name }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BedrockToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Upstream `BedrockOptions` extra fields (base `StreamOptions` fields stay on
/// [`StreamOptions`]; deviation D17).
#[derive(Debug, Clone, Default)]
pub struct BedrockOptions {
    pub region: Option<String>,
    pub profile: Option<String>,
    pub tool_choice: Option<BedrockToolChoice>,
    /// See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html for supported models.
    pub reasoning: Option<ThinkingLevel>,
    /// Custom token budgets per thinking level. Overrides default budgets.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Only supported by Claude 4.x models, see
    /// https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html#claude-messages-extended-thinking-tool-use-interleaved
    pub interleaved_thinking: Option<bool>,
    /// Controls how Claude's thinking content is returned in responses.
    /// Default: `Summarized` (upstream comment preserved).
    pub thinking_display: Option<BedrockThinkingDisplay>,
    /// Key-value pairs attached to the inference request for cost allocation
    /// tagging (`requestMetadata`).
    pub request_metadata: Option<BTreeMap<String, String>>,
    /// Bearer token for Bedrock API key authentication. When set, bypasses
    /// SigV4 signing and sends `Authorization: Bearer <token>` instead.
    pub bearer_token: Option<String>,
}

// ---------------------------------------------------------------------------
// Constants and shared helpers
// ---------------------------------------------------------------------------

const EMPTY_TEXT_PLACEHOLDER: &str = "<empty>";

const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";

/// Matches the placeholder the Anthropic API path uses for redacted thinking.
const REDACTED_THINKING_PLACEHOLDER: &str = "[Reasoning redacted]";

/// Human-readable prefixes for Bedrock SDK exception names
/// (`BEDROCK_ERROR_PREFIXES`).
const BEDROCK_ERROR_PREFIXES: &[(&str, &str)] = &[
    ("InternalServerException", "Internal server error"),
    ("ModelStreamErrorException", "Model stream error"),
    ("ValidationException", "Validation error"),
    ("ThrottlingException", "Throttling error"),
    ("ServiceUnavailableException", "Service unavailable"),
];

/// Some models reject the account/profile's configured Bedrock data retention
/// mode; point users at the AWS docs (`BEDROCK_DATA_RETENTION_DOCS_URL`).
const BEDROCK_DATA_RETENTION_DOCS_URL: &str =
    "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html";

/// Modeled mid-stream exceptions (`item.internalServerException` & co.).
const MODELED_STREAM_EXCEPTIONS: &[&str] = &[
    "InternalServerException",
    "ModelStreamErrorException",
    "ValidationException",
    "ThrottlingException",
    "ServiceUnavailableException",
];

/// Resolve cache retention preference (`resolveCacheRetention`). Defaults to
/// "short" and uses `PI_CACHE_RETENTION` for backward compatibility.
fn resolve_cache_retention(
    cache_retention: Option<CacheRetention>,
    env: Option<&crate::types::ProviderEnv>,
) -> CacheRetention {
    if let Some(retention) = cache_retention {
        return retention;
    }
    if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// `CacheTTL.ONE_HOUR` on the wire.
const CACHE_TTL_ONE_HOUR: &str = "1h";

// ---------------------------------------------------------------------------
// CRC32 (AWS event-stream framing)
// ---------------------------------------------------------------------------

/// IEEE CRC-32 (reflected, polynomial `0xEDB88320`), as required by the AWS
/// event-stream prelude/message trailer.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: LazyLock<[u32; 256]> = LazyLock::new(|| {
        let mut table = [0u32; 256];
        for (index, slot) in table.iter_mut().enumerate() {
            let mut crc = index as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
            *slot = crc;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc = (crc >> 8) ^ TABLE[((crc ^ u32::from(*byte)) & 0xFF) as usize];
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 (standard construction over sha2)
// ---------------------------------------------------------------------------

fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Standard HMAC over SHA-256 (RFC 2104; block size 64).
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&sha256(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = [0x36u8; 64];
    let mut outer = [0x5cu8; 64];
    for index in 0..64 {
        inner[index] ^= block[index];
        outer[index] ^= block[index];
    }
    let mut inner_hasher_input = Vec::with_capacity(64 + data.len());
    inner_hasher_input.extend_from_slice(&inner);
    inner_hasher_input.extend_from_slice(data);
    let inner_hash = sha256(&inner_hasher_input);
    let mut outer_input = [0u8; 96];
    outer_input[..64].copy_from_slice(&outer);
    outer_input[64..].copy_from_slice(&inner_hash);
    sha256(&outer_input)
}

// ---------------------------------------------------------------------------
// SigV4 (deviation D16)
// ---------------------------------------------------------------------------

/// Resolved AWS credentials (`BedrockRuntimeClientConfig["credentials"]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

/// `extendedEncodeURIComponent`: every byte outside `A-Za-z0-9-_.~` becomes
/// `%XX` (uppercase hex).
fn extended_encode_uri_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// Current time as `(YYYYMMDD'T'HHMMSS'Z', YYYYMMDD)` in UTC.
fn amz_dates(epoch_ms: u64) -> (String, String) {
    let seconds = (epoch_ms / 1000) as i64;
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    let (hour, minute, second) = (
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60,
    );
    (
        format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z"),
        format!("{year:04}{month:02}{day:02}"),
    )
}

/// Header keys that must never be overwritten by caller-supplied headers
/// (`RESERVED_HEADER_EXACT` + the `x-amz-` prefix).
fn is_reserved_header(key: &str) -> bool {
    let lower = key.to_lowercase();
    lower.starts_with("x-amz-") || lower == "authorization" || lower == "host"
}

/// Sign a POST request with SigV4 (`service: "bedrock"`), adding the signed
/// headers to the request. Payload is hashed as `x-amz-content-sha256`.
fn sign_request(
    request: &mut HttpRequest,
    region: &str,
    credentials: &AwsCredentials,
    payload: &[u8],
) {
    sign_request_at(
        request,
        region,
        credentials,
        payload,
        now_ms().max(0) as u64,
    );
}

/// [`sign_request`] with an explicit epoch (test seam).
fn sign_request_at(
    request: &mut HttpRequest,
    region: &str,
    credentials: &AwsCredentials,
    payload: &[u8],
    epoch_ms: u64,
) {
    let url = url::Url::parse(&request.url).expect("request URL parses");
    let host = match (url.port(), url.port_or_known_default()) {
        (Some(port), Some(default)) if port != default => {
            format!("{}:{port}", url.host_str().expect("URL has host"))
        }
        _ => url.host_str().expect("URL has host").to_string(),
    };
    let payload_hash = hex::encode(sha256(payload));
    let (amz_date, amz_day) = amz_dates(epoch_ms);

    // Canonical headers: everything we send that participates in the signature.
    let mut canonical: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in &request.headers {
        if name.eq_ignore_ascii_case("authorization") {
            continue;
        }
        canonical.insert(name.to_lowercase(), value.trim().to_string());
    }
    canonical.insert("host".to_string(), host.clone());
    canonical.insert("x-amz-date".to_string(), amz_date.clone());
    canonical.insert("x-amz-content-sha256".to_string(), payload_hash.clone());
    if let Some(session_token) = &credentials.session_token {
        canonical.insert("x-amz-security-token".to_string(), session_token.clone());
    }

    let signed_headers = canonical.keys().cloned().collect::<Vec<_>>().join(";");
    let canonical_headers = canonical
        .iter()
        .map(|(name, value)| format!("{name}:{value}\n"))
        .collect::<String>();
    let canonical_request = format!(
        "POST\n{}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        url.path()
    );

    let scope = format!("{amz_day}/{region}/bedrock/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex::encode(sha256(canonical_request.as_bytes()))
    );
    let k_date = hmac_sha256(credentials.secret_access_key.as_bytes(), amz_day.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"bedrock");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    request.headers.push(("host".to_string(), host));
    request.headers.push(("x-amz-date".to_string(), amz_date));
    request
        .headers
        .push(("x-amz-content-sha256".to_string(), payload_hash));
    if let Some(session_token) = &credentials.session_token {
        request
            .headers
            .push(("x-amz-security-token".to_string(), session_token.clone()));
    }
    request.headers.push((
        "authorization".to_string(),
        format!(
			"AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={signed_headers}, Signature={signature}",
			credentials.access_key_id, scope
		),
    ));
}

// ---------------------------------------------------------------------------
// AWS event-stream framing (deviation D16)
// ---------------------------------------------------------------------------

/// One event-stream header value (all nine spec types).
#[derive(Debug, Clone, PartialEq)]
enum HeaderValue {
    True,
    False,
    Byte(u8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    ByteBuf(Vec<u8>),
    String(String),
    Timestamp(i64),
}

impl HeaderValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            HeaderValue::String(value) => Some(value),
            _ => None,
        }
    }
}

/// One decoded event-stream message (headers + payload).
struct EventStreamMessage {
    headers: Vec<(String, HeaderValue)>,
    payload: Vec<u8>,
}

impl EventStreamMessage {
    fn header(&self, name: &str) -> Option<&HeaderValue> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }
}

/// Incremental AWS event-stream decoder (`response.stream` iteration).
struct EventStreamDecoder {
    buffer: Vec<u8>,
}

impl EventStreamDecoder {
    fn new() -> EventStreamDecoder {
        EventStreamDecoder { buffer: Vec::new() }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.buffer.extend_from_slice(chunk);
    }

    /// Decode the next complete message, or `None` when more bytes are needed.
    fn next_message(&mut self) -> Result<Option<EventStreamMessage>, String> {
        if self.buffer.len() < 12 {
            return Ok(None);
        }
        let total_length = u32::from_be_bytes([
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ]) as usize;
        let headers_length = u32::from_be_bytes([
            self.buffer[4],
            self.buffer[5],
            self.buffer[6],
            self.buffer[7],
        ]) as usize;
        if total_length < 16 {
            return Err(format!("Invalid event-stream message: total length {total_length} is below the 16-byte minimum"));
        }
        if self.buffer.len() < total_length {
            return Ok(None);
        }
        let prelude_crc = u32::from_be_bytes([
            self.buffer[8],
            self.buffer[9],
            self.buffer[10],
            self.buffer[11],
        ]);
        if crc32(&self.buffer[..8]) != prelude_crc {
            return Err("Invalid event-stream message: prelude CRC mismatch".to_string());
        }
        let headers_end = 12 + headers_length;
        if headers_end + 4 > total_length {
            return Err(
                "Invalid event-stream message: headers length exceeds message bounds".to_string(),
            );
        }
        let headers = parse_headers(&self.buffer[12..headers_end])?;
        let payload = self.buffer[headers_end..total_length - 4].to_vec();
        let message_crc = u32::from_be_bytes([
            self.buffer[total_length - 4],
            self.buffer[total_length - 3],
            self.buffer[total_length - 2],
            self.buffer[total_length - 1],
        ]);
        if crc32(&self.buffer[..total_length - 4]) != message_crc {
            return Err("Invalid event-stream message: message CRC mismatch".to_string());
        }
        self.buffer.drain(..total_length);
        Ok(Some(EventStreamMessage { headers, payload }))
    }
}

/// Parse a complete header map (`name len u8, name, value type u8, value`).
fn parse_headers(mut bytes: &[u8]) -> Result<Vec<(String, HeaderValue)>, String> {
    let mut headers = Vec::new();
    while !bytes.is_empty() {
        let (&name_length, rest) = bytes.split_first().ok_or("truncated header name length")?;
        if rest.len() < name_length as usize + 1 {
            return Err("truncated header name".to_string());
        }
        let name = String::from_utf8_lossy(&rest[..name_length as usize]).to_string();
        bytes = &rest[name_length as usize..];
        let (&value_type, rest) = bytes.split_first().ok_or("truncated header value type")?;
        let (value, consumed) = match value_type {
            0 => (HeaderValue::True, 0usize),
            1 => (HeaderValue::False, 0),
            2 => {
                if rest.is_empty() {
                    return Err("truncated header byte value".to_string());
                }
                (HeaderValue::Byte(rest[0]), 1)
            }
            3 => {
                if rest.len() < 2 {
                    return Err("truncated header int16 value".to_string());
                }
                (
                    HeaderValue::Int16(i16::from_be_bytes([rest[0], rest[1]])),
                    2,
                )
            }
            4 => {
                if rest.len() < 4 {
                    return Err("truncated header int32 value".to_string());
                }
                (
                    HeaderValue::Int32(i32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]])),
                    4,
                )
            }
            5 => {
                if rest.len() < 8 {
                    return Err("truncated header int64 value".to_string());
                }
                (
                    HeaderValue::Int64(i64::from_be_bytes([
                        rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7],
                    ])),
                    8,
                )
            }
            6 => {
                if rest.len() < 2 {
                    return Err("truncated header byte-buffer length".to_string());
                }
                let length = u16::from_be_bytes([rest[0], rest[1]]) as usize;
                if rest.len() < 2 + length {
                    return Err("truncated header byte-buffer value".to_string());
                }
                (
                    HeaderValue::ByteBuf(rest[2..2 + length].to_vec()),
                    2 + length,
                )
            }
            7 => {
                if rest.len() < 2 {
                    return Err("truncated header string length".to_string());
                }
                let length = u16::from_be_bytes([rest[0], rest[1]]) as usize;
                if rest.len() < 2 + length {
                    return Err("truncated header string value".to_string());
                }
                (
                    HeaderValue::String(String::from_utf8_lossy(&rest[2..2 + length]).to_string()),
                    2 + length,
                )
            }
            8 => {
                if rest.len() < 8 {
                    return Err("truncated header timestamp value".to_string());
                }
                (
                    HeaderValue::Timestamp(i64::from_be_bytes([
                        rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7],
                    ])),
                    8,
                )
            }
            other => return Err(format!("unknown event-stream header value type {other}")),
        };
        bytes = &rest[consumed..];
        headers.push((name, value));
    }
    Ok(headers)
}

// ---------------------------------------------------------------------------
// Wire types: ConverseStreamInput (byte-compatible with the SDK serializer)
// ---------------------------------------------------------------------------

/// Upstream `ConverseStreamInput` (marshaled JSON shape).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConverseStreamInput {
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    messages: Option<Vec<BedrockMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemContentBlock>>,
    inference_config: InferenceConfiguration,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    additional_model_request_fields: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_metadata: Option<BTreeMap<String, String>>,
}

/// Upstream `Message` (`role` + `content`).
#[derive(Debug, Serialize)]
struct BedrockMessage {
    role: &'static str,
    content: Vec<ContentBlock>,
}

/// Upstream `ContentBlock` union (single-member objects on the wire).
#[derive(Debug, Serialize)]
enum ContentBlock {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "image")]
    Image(ImageBlock),
    #[serde(rename = "toolUse")]
    ToolUse(ToolUseBlock),
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultBlock),
    #[serde(rename = "reasoningContent")]
    ReasoningContent(ReasoningContentBlock),
    #[serde(rename = "cachePoint")]
    CachePoint(CachePointBlock),
}

/// Upstream `ImageBlock` (`format` + `source.bytes` base64 blob).
#[derive(Debug, Serialize)]
struct ImageBlock {
    format: &'static str,
    source: ImageSource,
}

#[derive(Debug, Serialize)]
struct ImageSource {
    /// Base64-encoded image bytes (SDK blob serialization).
    bytes: String,
}

/// Upstream `ToolUseBlock`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolUseBlock {
    tool_use_id: String,
    name: String,
    input: Value,
}

/// Upstream `ToolResultBlock`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolResultBlock {
    tool_use_id: String,
    content: Vec<ToolResultContentBlock>,
    status: &'static str,
}

/// Upstream `ToolResultContentBlock` union.
#[derive(Debug, Serialize)]
enum ToolResultContentBlock {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "image")]
    Image(ImageBlock),
}

/// Upstream `ReasoningContentBlock`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReasoningContentBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_text: Option<ReasoningText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    redacted_content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReasoningText {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<String>,
}

/// Upstream `CachePointBlock` (`CachePointType.DEFAULT`, optional
/// `CacheTTL.ONE_HOUR`).
#[derive(Debug, Serialize)]
struct CachePointBlock {
    #[serde(rename = "type")]
    cache_point_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<&'static str>,
}

/// Upstream `SystemContentBlock` union.
#[derive(Debug, Serialize)]
enum SystemContentBlock {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "cachePoint")]
    CachePoint(CachePointBlock),
}

/// Upstream `InferenceConfiguration`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

/// Upstream `ToolConfiguration`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolConfiguration {
    tools: Vec<BedrockTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoiceBlock>,
}

/// Upstream `Tool` (`toolSpec` wrapper).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BedrockTool {
    tool_spec: ToolSpec,
}

/// Upstream `ToolSpec`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolSpec {
    name: String,
    description: String,
    input_schema: ToolInputSchema,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// Upstream `ToolSpecification.inputSchema` (`{ json: document }`).
#[derive(Debug, Serialize)]
struct ToolInputSchema {
    json: Value,
}

/// Upstream `ToolChoice` union.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoiceBlock {
    Auto { auto: EmptyObject },
    Any { any: EmptyObject },
    Tool { tool: ToolChoiceName },
}

#[derive(Debug, Serialize)]
struct EmptyObject {}

#[derive(Debug, Serialize)]
struct ToolChoiceName {
    name: String,
}

// ---------------------------------------------------------------------------
// Wire types: ConverseStreamOutput event payloads
// ---------------------------------------------------------------------------

/// Upstream `MessageStartEvent`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageStartEvent {
    role: Option<String>,
}

/// Upstream `ContentBlockStartEvent`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockStartEvent {
    content_block_index: Option<u32>,
    start: Option<BlockStart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockStart {
    tool_use: Option<ToolUseStart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolUseStart {
    tool_use_id: Option<String>,
    name: Option<String>,
}

/// Upstream `ContentBlockDeltaEvent`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockDeltaEvent {
    content_block_index: Option<u32>,
    delta: Option<BlockDelta>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockDelta {
    text: Option<String>,
    tool_use: Option<ToolUseDelta>,
    reasoning_content: Option<ReasoningContentDelta>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolUseDelta {
    input: Option<String>,
}

/// Upstream `ReasoningContentBlockDelta` (`redactedContent` arrives as a base64
/// blob on the wire and is decoded to bytes, like the SDK's `Uint8Array`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReasoningContentDelta {
    text: Option<String>,
    signature: Option<String>,
    redacted_content: Option<String>,
}

/// Upstream `ContentBlockStopEvent`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentBlockStopEvent {
    content_block_index: Option<u32>,
}

/// Upstream `MessageStopEvent`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageStopEvent {
    stop_reason: Option<String>,
}

/// Upstream `ConverseStreamMetadataEvent`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetadataEvent {
    usage: Option<MetadataUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetadataUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_write_input_tokens: Option<u64>,
    #[serde(default)]
    cache_details: Vec<CacheDetail>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheDetail {
    ttl: Option<String>,
    input_tokens: Option<u64>,
}

/// One decoded output item: either a ConverseStream event or an exception
/// frame (`:message-type: exception`).
enum BedrockStreamItem {
    Event {
        event_type: String,
        payload: Value,
        event: ConverseStreamEvent,
    },
    Exception {
        member: String,
        name: String,
        message: String,
        modeled: bool,
    },
}

/// Typed ConverseStream events the adapter consumes.
enum ConverseStreamEvent {
    MessageStart(MessageStartEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    MessageStop(MessageStopEvent),
    Metadata(MetadataEvent),
}

/// `ThrottlingException` → `throttlingException` (first letter lowered).
fn exception_member_name(exception_type: &str) -> String {
    let mut chars = exception_type.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Decode one framed message into an output item (upstream the SDK
/// unmarshaller).
fn frame_to_item(message: &EventStreamMessage) -> Result<Option<BedrockStreamItem>, String> {
    let message_type = message
        .header(":message-type")
        .and_then(HeaderValue::as_str)
        .unwrap_or_default();
    if message_type == "exception" {
        let exception_type = message
            .header(":exception-type")
            .and_then(HeaderValue::as_str)
            .or_else(|| message.header(":error-code").and_then(HeaderValue::as_str))
            .unwrap_or("Unknown");
        let payload: Value = serde_json::from_slice(&message.payload).unwrap_or_else(|_| json!({}));
        let message_text = payload
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| payload.get("Message").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        let normalized = exception_type
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        let modeled = MODELED_STREAM_EXCEPTIONS
            .iter()
            .find(|name| name.to_lowercase() == normalized)
            .copied();
        let (member, name) = match modeled {
            Some(name) => (exception_member_name(name), name.to_string()),
            None => (
                exception_member_name(exception_type),
                exception_type.to_string(),
            ),
        };
        return Ok(Some(BedrockStreamItem::Exception {
            member,
            name,
            message: message_text,
            modeled: modeled.is_some(),
        }));
    }
    if message_type != "event" {
        // Unknown message types are skipped (the SDK ignores non-event frames).
        return Ok(None);
    }
    let Some(event_type) = message.header(":event-type").and_then(HeaderValue::as_str) else {
        return Ok(None);
    };
    let payload: Value = serde_json::from_slice(&message.payload)
        .map_err(|error| format!("Failed to parse {event_type} event payload: {error}"))?;
    let event = match event_type {
        "messageStart" => ConverseStreamEvent::MessageStart(
            serde_json::from_value(payload.clone())
                .map_err(|error| format!("Failed to parse messageStart: {error}"))?,
        ),
        "contentBlockStart" => ConverseStreamEvent::ContentBlockStart(
            serde_json::from_value(payload.clone())
                .map_err(|error| format!("Failed to parse contentBlockStart: {error}"))?,
        ),
        "contentBlockDelta" => ConverseStreamEvent::ContentBlockDelta(
            serde_json::from_value(payload.clone())
                .map_err(|error| format!("Failed to parse contentBlockDelta: {error}"))?,
        ),
        "contentBlockStop" => ConverseStreamEvent::ContentBlockStop(
            serde_json::from_value(payload.clone())
                .map_err(|error| format!("Failed to parse contentBlockStop: {error}"))?,
        ),
        "messageStop" => ConverseStreamEvent::MessageStop(
            serde_json::from_value(payload.clone())
                .map_err(|error| format!("Failed to parse messageStop: {error}"))?,
        ),
        "metadata" => ConverseStreamEvent::Metadata(
            serde_json::from_value(payload.clone())
                .map_err(|error| format!("Failed to parse metadata: {error}"))?,
        ),
        // Unknown event types are skipped (upstream the union is closed).
        _ => return Ok(None),
    };
    Ok(Some(BedrockStreamItem::Event {
        event_type: event_type.to_string(),
        payload,
        event,
    }))
}

// ---------------------------------------------------------------------------
// Error formatting (`formatBedrockError` + failure diagnostics)
// ---------------------------------------------------------------------------

/// A service-shaped failure (modeled or unmodeled Bedrock exception, or a
/// transport failure with its HTTP context).
#[derive(Debug, Clone)]
struct BedrockSdkError {
    /// `error.name` (`None` for non-`Error` throws and transport failures).
    name: Option<String>,
    /// `error.message`.
    message: String,
    /// `$metadata.httpStatusCode`.
    status: Option<u16>,
    /// `$metadata.requestId`.
    request_id: Option<String>,
    /// Raw HTTP body, when not folded into `message` (deviation note: the Rust
    /// seam folds the deserialized `message` field and keeps the raw body only
    /// when no message field exists).
    body: Option<String>,
    /// `error instanceof BedrockRuntimeServiceException`.
    is_service_exception: bool,
}

/// Everything the `catch` block can observe (deviation D21).
enum StreamFailure {
    Sdk(BedrockSdkError),
    Message(String),
}

impl StreamFailure {
    fn from_message(message: impl Into<String>) -> StreamFailure {
        StreamFailure::Message(message.into())
    }

    fn sdk_error(&self) -> Option<&BedrockSdkError> {
        match self {
            StreamFailure::Sdk(error) => Some(error),
            StreamFailure::Message(_) => None,
        }
    }
}

impl From<String> for StreamFailure {
    fn from(message: String) -> StreamFailure {
        StreamFailure::Message(message)
    }
}

/// Upstream `formatBedrockError`.
fn format_bedrock_error(failure: &StreamFailure) -> String {
    let Some(error) = failure.sdk_error() else {
        return match failure {
            StreamFailure::Message(message) => {
                let core = apply_data_retention_hint(message);
                core
            }
            StreamFailure::Sdk(_) => unreachable!("sdk_error() returned Some for Sdk"),
        };
    };
    let norm = normalize_error(error);
    // Surface the raw HTTP body (with status) when the SDK did not fold it into
    // the message; otherwise fall back to the message. This is what stops a
    // gateway 403 from collapsing to `Unknown: UnknownError`.
    let core = if !norm.1 && norm.0.is_some() && error.body.is_some() {
        apply_data_retention_hint(&format!(
            "{}: {}",
            norm.0.expect("checked"),
            error.body.clone().expect("checked")
        ))
    } else {
        apply_data_retention_hint(&error.message)
    };
    if error.is_service_exception {
        let name = error.name.as_deref().unwrap_or_default();
        let prefix = BEDROCK_ERROR_PREFIXES
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, prefix)| prefix.to_string())
            .unwrap_or_else(|| name.to_string());
        format!("{prefix}: {core}")
    } else {
        core
    }
}

/// `/data retention mode/i` → docs hint (upstream suffix).
fn apply_data_retention_hint(core: &str) -> String {
    if core.to_lowercase().contains("data retention mode") {
        format!("{core} See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes.")
    } else {
        core.to_string()
    }
}

/// `(status, message_carries_body)` from [`crate::utils::error_body`]'s
/// normalization, applied to [`BedrockSdkError`].
fn normalize_error(error: &BedrockSdkError) -> (Option<u16>, bool) {
    let body = error
        .body
        .as_deref()
        .map(str::trim)
        .filter(|body| !body.is_empty() && *body != "{}")
        .map(|body| {
            crate::utils::error_body::truncate_error_text(
                body,
                crate::utils::error_body::MAX_PROVIDER_ERROR_BODY_CHARS,
            )
            .to_string()
        });
    match body {
        Some(body) => (error.status, error.message.contains(&body)),
        None => (error.status, true),
    }
}

/// Over-long header values are dropped rather than truncated: a truncated
/// request id is not a request id (`normalizeDiagnosticValue`).
fn normalize_diagnostic_value(value: Option<&str>) -> Option<String> {
    let value = value?;
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 200 {
        return None;
    }
    Some(trimmed.to_string())
}

/// The SDK puts the modeled code on `error.name` for service exceptions and
/// unmodeled stream errors alike (`extractBedrockErrorCode`).
fn extract_bedrock_error_code(error: &BedrockSdkError) -> Option<String> {
    let name = error.name.as_deref()?;
    if !name.ends_with("Exception") {
        return None;
    }
    normalize_diagnostic_value(Some(name))
}

/// Structured metadata alongside `errorMessage`
/// (`appendBedrockFailureDiagnostic`).
fn append_bedrock_failure_diagnostic(
    output: &mut AssistantMessage,
    failure: &StreamFailure,
    fallback_request_id: Option<&str>,
) {
    let Some(error) = failure.sdk_error() else {
        // Plain messages carry no provider metadata; only the fallback id.
        let request_id = normalize_diagnostic_value(fallback_request_id);
        let Some(request_id) = request_id else { return };
        let mut details = crate::types::JsonObject::new();
        details.insert("requestId".to_string(), Value::String(request_id));
        append_assistant_message_diagnostic(
            output,
            AssistantMessageDiagnostic {
                diagnostic_type: "bedrock_response_failure".to_string(),
                timestamp: now_ms(),
                error: None,
                details: Some(details),
            },
        );
        return;
    };
    let mut details = crate::types::JsonObject::new();
    if let Some(status) = error.status {
        details.insert("status".to_string(), json!(status));
    }
    if let Some(error_code) = extract_bedrock_error_code(error) {
        details.insert("errorCode".to_string(), Value::String(error_code));
    }
    let request_id = normalize_diagnostic_value(error.request_id.as_deref())
        .or_else(|| normalize_diagnostic_value(fallback_request_id));
    if let Some(request_id) = request_id {
        details.insert("requestId".to_string(), Value::String(request_id));
    }
    if details.is_empty() {
        return;
    }
    append_assistant_message_diagnostic(
        output,
        AssistantMessageDiagnostic {
            diagnostic_type: "bedrock_response_failure".to_string(),
            timestamp: now_ms(),
            error: None,
            details: Some(details),
        },
    )
}

// ---------------------------------------------------------------------------
// Client config resolution (credentials, region, endpoint)
// ---------------------------------------------------------------------------

/// The observable `BedrockRuntimeClientConfig` the port builds (deviation D23:
/// asserted directly by the credential/endpoint-resolution test ports).
#[derive(Debug, Clone, Default, PartialEq)]
struct BedrockClientConfig {
    profile: Option<String>,
    region: Option<String>,
    endpoint: Option<String>,
    credentials: Option<AwsCredentials>,
    /// Bearer token (`config.token`); set ⇒ `authSchemePreference` is
    /// `["httpBearerAuth"]`.
    token: Option<String>,
}

/// `getConfiguredBedrockRegion`: option > `AWS_REGION` > `AWS_DEFAULT_REGION`.
fn get_configured_bedrock_region(
    base: &crate::types::ProviderRequestOptions,
    group: &BedrockOptions,
) -> Option<String> {
    group
        .region
        .clone()
        .or_else(|| get_provider_env_value("AWS_REGION", base.env.as_ref()))
        .or_else(|| get_provider_env_value("AWS_DEFAULT_REGION", base.env.as_ref()))
}

/// `getConfiguredBedrockCredentials`: env access keys (+ optional session
/// token).
fn get_configured_bedrock_credentials(
    env: Option<&crate::types::ProviderEnv>,
) -> Option<AwsCredentials> {
    let access_key_id = get_provider_env_value("AWS_ACCESS_KEY_ID", env)?;
    let secret_access_key = get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)?;
    let session_token = get_provider_env_value("AWS_SESSION_TOKEN", env);
    Some(AwsCredentials {
        access_key_id,
        secret_access_key,
        session_token,
    })
}

static ENDPOINT_REGION_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$")
        .expect("endpoint pattern")
});

/// `getStandardBedrockEndpointRegion`.
fn get_standard_bedrock_endpoint_region(base_url: Option<&str>) -> Option<String> {
    let url = url::Url::parse(base_url?).ok()?;
    let hostname = url.host_str()?.to_lowercase();
    ENDPOINT_REGION_PATTERN
        .captures(&hostname)
        .and_then(|captures| captures.get(1))
        .map(|region| region.as_str().to_string())
}

static ARN_REGION_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):").expect("arn pattern")
});

/// `shouldUseExplicitBedrockEndpoint`.
fn should_use_explicit_bedrock_endpoint(
    base_url: &str,
    configured_region: Option<&str>,
    has_ambient_configured_profile: bool,
) -> bool {
    let Some(endpoint_region) = get_standard_bedrock_endpoint_region(Some(base_url)) else {
        return true;
    };
    let _ = endpoint_region;
    configured_region.is_none() && !has_ambient_configured_profile
}

/// Minimal INI parse of the AWS shared files: `[section]` headers and
/// `key = value` pairs; comments (`;`, `#`) and blanks skipped.
fn parse_shared_config(text: &str) -> Vec<(String, Vec<(String, String)>)> {
    let mut sections: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            sections.push((name.trim().to_string(), Vec::new()));
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if let Some((_, entries)) = sections.last_mut() {
            entries.push((key.trim().to_string(), value.trim().to_string()));
        }
    }
    sections
}

/// Expand a leading `~` like `os.homedir()` (crate `auth/context.rs` pattern:
/// `HOME` environment variable).
fn expand_home(path: &str) -> std::path::PathBuf {
    if let Some(rest) = path.strip_prefix('~') {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home).join(rest.trim_start_matches('/'));
        }
    }
    std::path::PathBuf::from(path)
}

fn read_shared_file(path: &std::path::Path) -> Option<Vec<(String, Vec<(String, String)>)>> {
    std::fs::read_to_string(path)
        .ok()
        .map(|text| parse_shared_config(&text))
}

/// Credentials for a named profile from `~/.aws/credentials` (+ `~/.aws/config`
/// for fallback keys and `region`). The credentials file wins, like the SDK.
fn shared_profile(profile: &str) -> Option<(AwsCredentials, Option<String>)> {
    let home = expand_home("~/.aws");
    let credential_sections = read_shared_file(&home.join("credentials")).unwrap_or_default();
    let config_sections = read_shared_file(&home.join("config")).unwrap_or_default();

    let mut lookup: BTreeMap<&str, String> = BTreeMap::new();
    let config_section_name = if profile == "default" {
        "default".to_string()
    } else {
        format!("profile {profile}")
    };
    for (section, entries) in config_sections
        .iter()
        .filter(|(section, _)| *section == config_section_name)
    {
        let _ = section;
        for (key, value) in entries {
            lookup.entry(key.as_str()).or_insert_with(|| value.clone());
        }
    }
    for (section, entries) in credential_sections
        .iter()
        .filter(|(section, _)| section == profile)
    {
        let _ = section;
        for (key, value) in entries {
            lookup.insert(key.as_str(), value.clone());
        }
    }
    let access_key_id = lookup.get("aws_access_key_id")?;
    let secret_access_key = lookup.get("aws_secret_access_key")?;
    let credentials = AwsCredentials {
        access_key_id: access_key_id.clone(),
        secret_access_key: secret_access_key.clone(),
        session_token: lookup.get("aws_session_token").cloned(),
    };
    Some((credentials, lookup.get("region").cloned()))
}

/// Build the observable client config (upstream the `BedrockRuntimeClient`
/// constructor argument; see the module docs for the resolution rules).
fn resolve_client_config(
    model: &Model,
    base: &crate::types::ProviderRequestOptions,
    group: &BedrockOptions,
) -> BedrockClientConfig {
    // A profile explicitly configured through pi's auth flow (the `profile`
    // option or scoped `AWS_PROFILE` on the stored credential's env) must win
    // over ambient AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (upstream #6957).
    let scoped_profile = base
        .env
        .as_ref()
        .and_then(|env| env.get("AWS_PROFILE"))
        .filter(|value| !value.is_empty())
        .cloned();
    let options_profile = group.profile.clone().or(scoped_profile);
    let mut config = BedrockClientConfig {
        profile: options_profile
            .clone()
            .or_else(|| get_provider_env_value("AWS_PROFILE", base.env.as_ref())),
        ..Default::default()
    };
    let configured_region = get_configured_bedrock_region(base, group);
    // Upstream `getProviderEnvValue("AWS_PROFILE")`: ambient means the process
    // environment only — a scoped `AWS_PROFILE` on the stored credential's env
    // is an explicit choice, not ambient.
    let has_ambient_configured_profile = std::env::var("AWS_PROFILE")
        .map(|value| !value.is_empty())
        .unwrap_or(false);
    let endpoint_region = get_standard_bedrock_endpoint_region(Some(&model.base.base_url));
    let use_explicit_endpoint = should_use_explicit_bedrock_endpoint(
        &model.base.base_url,
        configured_region.as_deref(),
        has_ambient_configured_profile,
    );

    // Only pin standard AWS Bedrock runtime endpoints when no region or ambient
    // AWS_PROFILE is configured (upstream #3402).
    if use_explicit_endpoint {
        config.endpoint = Some(model.base.base_url.clone());
    }

    // Resolve bearer token for Bedrock API key auth.
    let skip_auth =
        get_provider_env_value("AWS_BEDROCK_SKIP_AUTH", base.env.as_ref()).as_deref() == Some("1");
    let bearer_token = group
        .bearer_token
        .clone()
        .or_else(|| base.api_key.clone().filter(|key| !key.is_empty()))
        .or_else(|| get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", base.env.as_ref()));
    let use_bearer_token = bearer_token.is_some() && !skip_auth;

    // Region resolution: ARN-embedded > explicit option > env vars > SDK default chain.
    let arn_region = ARN_REGION_PATTERN
        .captures(&model.base.id)
        .and_then(|captures| captures.get(1))
        .map(|region| region.as_str().to_string());
    config.region = if let Some(region) = arn_region {
        Some(region)
    } else if let Some(region) = configured_region.clone() {
        Some(region)
    } else if let (Some(region), true) = (endpoint_region.clone(), use_explicit_endpoint) {
        Some(region)
    } else if !has_ambient_configured_profile {
        Some("us-east-1".to_string())
    } else {
        None
    };

    if skip_auth {
        config.credentials = Some(AwsCredentials {
            access_key_id: "dummy-access-key".to_string(),
            secret_access_key: "dummy-secret-key".to_string(),
            session_token: None,
        });
    }

    let credentials = get_configured_bedrock_credentials(base.env.as_ref());
    if !skip_auth && credentials.is_some() && options_profile.is_none() {
        config.credentials = credentials;
    }

    if use_bearer_token {
        config.token = bearer_token;
    }

    config
}

/// The region SigV4 signs for: resolved config region, else the shared-config
/// profile region, else the SDK default.
fn signing_region(config: &BedrockClientConfig) -> String {
    config
        .region
        .clone()
        .or_else(|| {
            config
                .profile
                .as_deref()
                .and_then(shared_profile)
                .and_then(|(_, region)| region)
        })
        .unwrap_or_else(|| "us-east-1".to_string())
}

/// Effective credentials for signing (env keys / profile files / skip-auth
/// dummies).
fn signing_credentials(config: &BedrockClientConfig) -> Result<AwsCredentials, String> {
    if let Some(credentials) = &config.credentials {
        return Ok(credentials.clone());
    }
    if let Some(profile) = &config.profile {
        if let Some((credentials, _)) = shared_profile(profile) {
            return Ok(credentials);
        }
    }
    Err("Could not load credentials from any providers".to_string())
}

/// Build the streaming HTTP request for one attempt (SDK request assembly +
/// signing).
#[allow(clippy::too_many_arguments)]
fn build_signed_request(
    model: &Model,
    payload: &[u8],
    config: &BedrockClientConfig,
    custom_headers: Option<&BTreeMap<String, String>>,
    signal: Option<AbortSignal>,
    timeout_ms: Option<u64>,
) -> Result<HttpRequest, StreamFailure> {
    let region = signing_region(config);
    let endpoint = config
        .endpoint
        .clone()
        .unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com"));
    let url = format!(
        "{}/model/{}/converse-stream",
        endpoint.trim_end_matches('/'),
        extended_encode_uri_component(&model.base.id)
    );
    let mut request = HttpRequest::new("POST", url);
    request
        .headers
        .push(("content-type".to_string(), "application/json".to_string()));
    if let Some(custom_headers) = custom_headers {
        for (key, value) in custom_headers {
            if !is_reserved_header(key) {
                request.headers.push((key.clone(), value.clone()));
            }
        }
    }
    request.body = HttpBody::Bytes(payload.to_vec());
    request.timeout_ms = timeout_ms;
    request.signal = signal;
    if let Some(token) = &config.token {
        request
            .headers
            .push(("authorization".to_string(), format!("Bearer {token}")));
    } else {
        let credentials = signing_credentials(config).map_err(|message| {
            StreamFailure::Sdk(BedrockSdkError {
                name: None,
                message,
                status: None,
                request_id: None,
                body: None,
                is_service_exception: false,
            })
        })?;
        sign_request(&mut request, &region, &credentials, payload);
    }
    Ok(request)
}

/// Non-2xx response → the service exception `client.send()` throws
/// (`handleError`): name from `x-amzn-errortype` (default `Unknown`), message
/// folded from the body.
async fn response_to_failure(response: HttpResponse) -> StreamFailure {
    let status = response.status;
    let request_id = response.header("x-amzn-requestid").map(str::to_string);
    let errortype = response.header("x-amzn-errortype").map(str::to_string);
    let body_text = match response.body.collect().await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        Err(_) => String::new(),
    };
    // `handleError`: the code is the `x-amzn-errortype` header's first segment;
    // without one the SDK names the exception `UnknownError`.
    let name = errortype
        .as_deref()
        .map(|value| {
            value
                .split(':')
                .next()
                .unwrap_or_default()
                .trim()
                .to_string()
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "UnknownError".to_string());
    // Modeled exceptions fold the body's message field into `error.message`;
    // unmodeled ("UnknownError") exceptions stay body-blind — the raw body
    // travels on the error so `formatBedrockError` can surface `status: body`
    // instead of collapsing to `UnknownError: UnknownError` (upstream
    // provider-error-body-regression).
    let parsed: Option<Value> = serde_json::from_str(body_text.trim()).ok();
    let body_message = parsed.as_ref().and_then(|value| {
        value
            .get("message")
            .or_else(|| value.get("Message"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let (message, body) = if errortype.is_some() {
        match body_message {
            Some(message) => (message, None),
            None if !body_text.trim().is_empty() => (body_text.clone(), None),
            None => (String::new(), None),
        }
    } else {
        (
            "UnknownError".to_string(),
            Some(body_text).filter(|body| !body.trim().is_empty()),
        )
    };
    StreamFailure::Sdk(BedrockSdkError {
        name: Some(name),
        message,
        status: Some(status),
        request_id,
        body,
        is_service_exception: true,
    })
}

/// Retryable attempt error (`ProviderRequestError` for `provider_retry`).
/// Carries the final [`StreamFailure`] through the retry loop.
struct BedrockAttemptError {
    message: String,
    status: Option<u16>,
    headers: Vec<(String, String)>,
    failure: StreamFailure,
}

impl crate::utils::provider_retry::ProviderRequestError for BedrockAttemptError {
    fn status(&self) -> Option<u16> {
        self.status
    }

    fn header(&self, name: &str) -> Option<String> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    }

    fn message(&self) -> &str {
        &self.message
    }

    fn from_abort() -> Self {
        BedrockAttemptError {
            message: "Request aborted".to_string(),
            status: None,
            headers: Vec::new(),
            failure: StreamFailure::from_message("Request aborted"),
        }
    }
}

impl From<StreamFailure> for BedrockAttemptError {
    fn from(failure: StreamFailure) -> Self {
        BedrockAttemptError {
            message: match &failure {
                StreamFailure::Sdk(error) => error.message.clone(),
                StreamFailure::Message(message) => message.clone(),
            },
            status: failure.sdk_error().and_then(|error| error.status),
            headers: Vec::new(),
            failure,
        }
    }
}

impl From<BedrockAttemptError> for StreamFailure {
    fn from(error: BedrockAttemptError) -> StreamFailure {
        error.failure
    }
}

// ---------------------------------------------------------------------------
// Model capability checks
// ---------------------------------------------------------------------------

/// `getModelMatchCandidates`: model ID and name, lowercased, plus the
/// separator-normalized form.
fn get_model_match_candidates(model_id: &str, model_name: Option<&str>) -> Vec<String> {
    let values: Vec<&str> = match model_name {
        Some(name) => vec![model_id, name],
        None => vec![model_id],
    };
    values
        .iter()
        .flat_map(|value| {
            let lower = value.to_lowercase();
            let normalized: String = lower
                .chars()
                .map(|c| {
                    if c.is_whitespace() || matches!(c, '.' | '_' | ':') {
                        '-'
                    } else {
                        c
                    }
                })
                .collect();
            vec![lower, normalized]
        })
        .collect()
}

/// `supportsAdaptiveThinking` (Opus 4.6+, Sonnet 4.6+).
fn supports_adaptive_thinking(model_id: &str, model_name: Option<&str>) -> bool {
    get_model_match_candidates(model_id, model_name)
        .iter()
        .any(|s| {
            s.contains("opus-4-6")
                || s.contains("opus-4-7")
                || s.contains("opus-4-8")
                || s.contains("opus-5")
                || s.contains("sonnet-4-6")
                || s.contains("sonnet-5")
                || s.contains("haiku-5")
                || s.contains("fable-5")
        })
}

/// `supportsNativeXhighEffort`.
fn supports_native_xhigh_effort(model: &Model) -> bool {
    get_model_match_candidates(&model.base.id, Some(&model.base.name))
        .iter()
        .any(|s| {
            s.contains("opus-4-7")
                || s.contains("opus-4-8")
                || s.contains("opus-5")
                || s.contains("sonnet-5")
                || s.contains("haiku-5")
                || s.contains("fable-5")
        })
}

/// `supportsThinkingBlockBinding`: Opus 4.6 and Sonnet 4.6 reject it with
/// "thinking.adaptive.block_binding: Extra inputs are not permitted".
fn supports_thinking_block_binding(model: &Model) -> bool {
    supports_native_xhigh_effort(model)
}

/// `isAnthropicClaudeModel` (checks both model ID and model name).
fn is_anthropic_claude_model(model: &Model) -> bool {
    let id = model.base.id.to_lowercase();
    let name = model.base.name.to_lowercase();
    id.contains("anthropic.claude")
        || id.contains("anthropic/claude")
        || name.contains("anthropic.claude")
        || name.contains("anthropic/claude")
        || name.contains("claude")
}

/// `supportsPromptCaching`.
fn supports_prompt_caching(model: &Model, env: Option<&crate::types::ProviderEnv>) -> bool {
    let candidates = get_model_match_candidates(&model.base.id, Some(&model.base.name));
    let has_claude_ref = candidates.iter().any(|s| s.contains("claude"));
    if !has_claude_ref {
        // Application inference profiles don't contain the model name in the
        // ARN. Allow users to force cache points via environment variable.
        return get_provider_env_value("AWS_BEDROCK_FORCE_CACHE", env).as_deref() == Some("1");
    }
    // Claude 5 models (fable-5, opus-5, sonnet-5, haiku-5)
    if candidates.iter().any(|s| {
        s.contains("fable-5")
            || s.contains("opus-5")
            || s.contains("sonnet-5")
            || s.contains("haiku-5")
    }) {
        return true;
    }
    // Claude 4.x models (opus-4, sonnet-4, haiku-4)
    if candidates.iter().any(|s| s.contains("-4-")) {
        return true;
    }
    // Claude 3.7 Sonnet
    if candidates.iter().any(|s| s.contains("claude-3-7-sonnet")) {
        return true;
    }
    // Claude 3.5 Haiku
    if candidates.iter().any(|s| s.contains("claude-3-5-haiku")) {
        return true;
    }
    false
}

/// `supportsThinkingSignature`: only Anthropic Claude models support the
/// signature field in reasoningContent.
fn supports_thinking_signature(model: &Model) -> bool {
    is_anthropic_claude_model(model)
}

/// `isGovCloudBedrockTarget`.
fn is_gov_cloud_bedrock_target(
    model: &Model,
    base: &crate::types::ProviderRequestOptions,
    group: &BedrockOptions,
) -> bool {
    if get_configured_bedrock_region(base, group)
        .map(|region| region.to_lowercase().starts_with("us-gov-"))
        .unwrap_or(false)
    {
        return true;
    }
    let model_id = model.base.id.to_lowercase();
    model_id.starts_with("us-gov.") || model_id.starts_with("arn:aws-us-gov:")
}

/// `mapThinkingLevelToEffort`.
fn map_thinking_level_to_effort(model: &Model, level: Option<ThinkingLevel>) -> String {
    if level == Some(ThinkingLevel::Xhigh) && supports_native_xhigh_effort(model) {
        return "xhigh".to_string();
    }
    if let Some(level) = level {
        let mapped = model
            .thinking_level_map
            .as_ref()
            .and_then(|map| map.get(&ModelThinkingLevel::from(level)))
            .cloned()
            .flatten();
        if let Some(mapped) = mapped {
            return mapped;
        }
    }
    match level {
        Some(ThinkingLevel::Minimal) | Some(ThinkingLevel::Low) => "low".to_string(),
        Some(ThinkingLevel::Medium) => "medium".to_string(),
        _ => "high".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Payload conversion
// ---------------------------------------------------------------------------

/// `normalizeToolCallId`.
fn normalize_tool_call_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    sanitized.chars().take(64).collect()
}

/// `createNonBlankTextBlock`: sanitized text, `None` when blank.
fn create_non_blank_text_block(text: &str) -> Option<ContentBlock> {
    let sanitized = sanitize_surrogates(text);
    (!sanitized.trim().is_empty()).then_some(ContentBlock::Text(sanitized))
}

/// `createRequiredTextBlock`: blank text becomes the `<empty>` placeholder.
fn create_required_text_block(text: &str) -> ContentBlock {
    create_non_blank_text_block(text)
        .unwrap_or_else(|| ContentBlock::Text(EMPTY_TEXT_PLACEHOLDER.to_string()))
}

/// `sanitizeBedrockDocument`: drops empty property names recursively.
fn sanitize_bedrock_document(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(sanitize_bedrock_document).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| !key.is_empty())
                .map(|(key, nested)| (key.clone(), sanitize_bedrock_document(nested)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `createImageBlock` (`Unknown image type: <mime>` on unsupported types).
/// OpenAI GPT models (GPT-5.x, GPT-6) take a nested `reasoning.effort` and
/// reject `minimal` (`OPENAI_GPT_EFFORT`).
fn openai_gpt_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal => "low",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

/// gpt-oss takes a flat `reasoning_effort` and only accepts low, medium and
/// high (`OPENAI_GPT_OSS_EFFORT`).
fn openai_gpt_oss_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal => "low",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "high",
        ThinkingLevel::Max => "high",
    }
}

/// v1.1.0: reasoning effort reaches OpenAI models on Bedrock — gpt-oss gets a
/// flat `reasoning_effort` clamped to low/medium/high, other GPT models get
/// `reasoning.effort` with `minimal` sent as `low` (#9331, #10142).
fn openai_reasoning_fields(model: &Model, group: &BedrockOptions) -> Option<Value> {
    let reasoning = group.reasoning?;
    if !model.reasoning {
        return None;
    }
    let candidates = get_model_match_candidates(&model.base.id, Some(&model.base.name));
    if candidates.iter().any(|s| s.contains("gpt-oss")) {
        return Some(serde_json::json!({ "reasoning_effort": openai_gpt_oss_effort(reasoning) }));
    }
    if candidates.iter().any(|s| s.contains("gpt-")) {
        let mapped = model
            .thinking_level_map
            .as_ref()
            .and_then(|map| map.get(&ModelThinkingLevel::from(reasoning)))
            .cloned()
            .flatten();
        let effort = match mapped {
            Some(effort) => effort,
            None => openai_gpt_effort(reasoning).to_string(),
        };
        return Some(serde_json::json!({ "reasoning": { "effort": effort } }));
    }
    None
}

fn create_image_block(mime_type: &str, data: &str) -> Result<ImageBlock, String> {
    let format = match mime_type {
        "image/jpeg" | "image/jpg" => "jpeg",
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        other => return Err(format!("Unknown image type: {other}")),
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|error| format!("Failed to decode image data: {error}"))?;
    Ok(ImageBlock {
        format,
        source: ImageSource {
            bytes: base64::engine::general_purpose::STANDARD.encode(bytes),
        },
    })
}

/// `convertToolResultContent`.
fn convert_tool_result_content(
    content: &[UserContent],
) -> Result<Vec<ToolResultContentBlock>, String> {
    let mut result = Vec::new();
    for block in content {
        match block {
            UserContent::Image(image) => result.push(ToolResultContentBlock::Image(
                create_image_block(&image.mime_type, &image.data)?,
            )),
            UserContent::Text(text) => {
                if let Some(text_block) = create_non_blank_text_block(&text.text) {
                    result.push(ToolResultContentBlock::Text(match text_block {
                        ContentBlock::Text(text) => text,
                        _ => unreachable!("text block"),
                    }));
                }
            }
        }
    }
    if result.is_empty() {
        result.push(ToolResultContentBlock::Text(
            EMPTY_TEXT_PLACEHOLDER.to_string(),
        ));
    }
    Ok(result)
}

/// `decodeRedactedContent`: stored base64 payload → bytes; invalid base64 is
/// dropped instead of failing the request.
fn decode_redacted_content(signature: Option<&str>) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(signature?)
        .ok()
        .filter(|bytes| !bytes.is_empty())
}

/// `convertMessages`.
fn convert_messages(
    context: &TranscriptContext,
    model: &Model,
    cache_retention: CacheRetention,
    env: Option<&crate::types::ProviderEnv>,
) -> Result<Vec<BedrockMessage>, String> {
    let mut result: Vec<BedrockMessage> = Vec::new();
    let messages = context.messages();
    let transformed = crate::api::transform_messages::transform_messages(
        &without_initial_system_message(messages),
        model,
        Some(&|id: &str, _model: &Model, _message: &AssistantMessage| normalize_tool_call_id(id)),
    );

    let mut index = 0;
    while index < transformed.len() {
        let message = &transformed[index];
        match message {
            Message::User(user) => {
                let mut content: Vec<ContentBlock> = Vec::new();
                match &user.content {
                    UserContentField::Text(text) => {
                        content.push(create_required_text_block(text));
                    }
                    UserContentField::Blocks(blocks) => {
                        for block in blocks {
                            match block {
                                UserContent::Text(text) => {
                                    if let Some(text_block) =
                                        create_non_blank_text_block(&text.text)
                                    {
                                        content.push(text_block);
                                    }
                                }
                                UserContent::Image(image) => {
                                    content.push(ContentBlock::Image(create_image_block(
                                        &image.mime_type,
                                        &image.data,
                                    )?));
                                }
                            }
                        }
                        if content.is_empty() {
                            content.push(ContentBlock::Text(EMPTY_TEXT_PLACEHOLDER.to_string()));
                        }
                    }
                }
                result.push(BedrockMessage {
                    role: "user",
                    content,
                });
            }
            Message::Assistant(assistant) => {
                // Skip assistant messages with empty content (e.g. from aborted
                // requests); Bedrock rejects empty content arrays.
                if assistant.content.is_empty() {
                    index += 1;
                    continue;
                }
                let mut content_blocks: Vec<ContentBlock> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) => {
                            // Skip empty text blocks.
                            if let Some(text_block) = create_non_blank_text_block(&text.text) {
                                content_blocks.push(text_block);
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            content_blocks.push(ContentBlock::ToolUse(ToolUseBlock {
                                tool_use_id: tool_call.id.clone(),
                                name: tool_call.name.clone(),
                                input: sanitize_bedrock_document(&Value::Object(
                                    tool_call.arguments.clone(),
                                )),
                            }));
                        }
                        AssistantContent::Thinking(thinking) => {
                            // Encrypted reasoning is opaque: replay the stored payload
                            // as the `redactedContent` member.
                            if thinking.redacted == Some(true) {
                                if let Some(redacted_content) =
                                    decode_redacted_content(thinking.thinking_signature.as_deref())
                                {
                                    content_blocks.push(ContentBlock::ReasoningContent(
                                        ReasoningContentBlock {
                                            reasoning_text: None,
                                            redacted_content: Some(
                                                base64::engine::general_purpose::STANDARD
                                                    .encode(redacted_content),
                                            ),
                                        },
                                    ));
                                }
                                continue;
                            }
                            // Skip empty thinking blocks.
                            let thinking_text = sanitize_surrogates(&thinking.thinking);
                            if thinking_text.trim().is_empty() {
                                continue;
                            }
                            // Only Anthropic models support the signature field in
                            // reasoningText.
                            if supports_thinking_signature(model) {
                                let signature =
                                    thinking.thinking_signature.as_deref().unwrap_or_default();
                                if signature.trim().is_empty() {
                                    // Partial or externally persisted messages without a
                                    // signature fall back to plain text, matching Anthropic.
                                    content_blocks.push(ContentBlock::Text(thinking_text));
                                } else {
                                    content_blocks.push(ContentBlock::ReasoningContent(
                                        ReasoningContentBlock {
                                            reasoning_text: Some(ReasoningText {
                                                text: thinking_text,
                                                signature: Some(signature.to_string()),
                                            }),
                                            redacted_content: None,
                                        },
                                    ));
                                }
                            } else {
                                content_blocks.push(ContentBlock::ReasoningContent(
                                    ReasoningContentBlock {
                                        reasoning_text: Some(ReasoningText {
                                            text: thinking_text,
                                            signature: None,
                                        }),
                                        redacted_content: None,
                                    },
                                ));
                            }
                        }
                    }
                }
                // Skip if all content blocks were filtered out.
                if content_blocks.is_empty() {
                    index += 1;
                    continue;
                }
                result.push(BedrockMessage {
                    role: "assistant",
                    content: content_blocks,
                });
            }
            Message::ToolResult(tool_result) => {
                // Collect all consecutive toolResult messages into a single user
                // message; Bedrock requires all tool results in one message.
                let mut tool_results: Vec<ContentBlock> = Vec::new();
                tool_results.push(ContentBlock::ToolResult(ToolResultBlock {
                    tool_use_id: tool_result.tool_call_id.clone(),
                    content: convert_tool_result_content(&tool_result.content)?,
                    status: if tool_result.is_error {
                        "error"
                    } else {
                        "success"
                    },
                }));
                let mut cursor = index + 1;
                while cursor < transformed.len() {
                    let Message::ToolResult(next) = &transformed[cursor] else {
                        break;
                    };
                    tool_results.push(ContentBlock::ToolResult(ToolResultBlock {
                        tool_use_id: next.tool_call_id.clone(),
                        content: convert_tool_result_content(&next.content)?,
                        status: if next.is_error { "error" } else { "success" },
                    }));
                    cursor += 1;
                }
                // Skip the messages we've already processed.
                index = cursor - 1;
                result.push(BedrockMessage {
                    role: "user",
                    content: tool_results,
                });
            }
            Message::System(_) => {}
        }
        index += 1;
    }

    // Add cache point to the last user message for supported Claude models when
    // caching is enabled.
    if cache_retention != CacheRetention::None
        && supports_prompt_caching(model, env)
        && !result.is_empty()
    {
        let last = result.last_mut().expect("non-empty");
        if last.role == "user" {
            last.content
                .push(ContentBlock::CachePoint(cache_point_block(cache_retention)));
        }
    }

    Ok(result)
}

/// `cachePoint: { type: default, ttl? }`.
fn cache_point_block(cache_retention: CacheRetention) -> CachePointBlock {
    CachePointBlock {
        cache_point_type: "default",
        ttl: (cache_retention == CacheRetention::Long).then_some(CACHE_TTL_ONE_HOUR),
    }
}

/// `buildSystemPrompt`.
fn build_system_prompt(
    system_prompt: Option<&str>,
    model: &Model,
    cache_retention: CacheRetention,
    env: Option<&crate::types::ProviderEnv>,
) -> Option<Vec<SystemContentBlock>> {
    let system_prompt = system_prompt.filter(|prompt| !prompt.is_empty())?;
    let mut blocks = vec![SystemContentBlock::Text(sanitize_surrogates(system_prompt))];
    // Add cache point for supported Claude models when caching is enabled.
    if cache_retention != CacheRetention::None && supports_prompt_caching(model, env) {
        blocks.push(SystemContentBlock::CachePoint(cache_point_block(
            cache_retention,
        )));
    }
    Some(blocks)
}

/// `convertToolConfig`.
fn convert_tool_config(
    tools: &[Tool],
    tool_choice: Option<&BedrockToolChoice>,
    supports_strict_mode: bool,
) -> Result<Option<ToolConfiguration>, String> {
    if tools.is_empty() {
        return Ok(None);
    }
    if tool_choice == Some(&BedrockToolChoice::None) {
        return Ok(None);
    }
    let mut bedrock_tools = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = crate::api::constrained_sampling::resolve_json_schema_strict_sampling(
            tool,
            supports_strict_mode,
        )?;
        bedrock_tools.push(BedrockTool {
            tool_spec: ToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: ToolInputSchema {
                    json: crate::api::constrained_sampling::get_json_schema_tool_parameters(
                        tool,
                        strict.unwrap_or(false),
                    )
                    .map_err(|error| error.to_string())?,
                },
                strict: (strict == Some(true)).then_some(true),
            },
        });
    }
    let bedrock_tool_choice = match tool_choice {
        Some(BedrockToolChoice::Auto) => Some(ToolChoiceBlock::Auto {
            auto: EmptyObject {},
        }),
        Some(BedrockToolChoice::Any) => Some(ToolChoiceBlock::Any {
            any: EmptyObject {},
        }),
        Some(BedrockToolChoice::Tool { name }) => Some(ToolChoiceBlock::Tool {
            tool: ToolChoiceName { name: name.clone() },
        }),
        _ => None,
    };
    Ok(Some(ToolConfiguration {
        tools: bedrock_tools,
        tool_choice: bedrock_tool_choice,
    }))
}

/// `buildAdditionalModelRequestFields`.
fn build_additional_model_request_fields(
    model: &Model,
    base: &crate::types::ProviderRequestOptions,
    group: &BedrockOptions,
) -> Option<Value> {
    let reasoning = group.reasoning?;
    if !model.reasoning {
        return None;
    }
    if !is_anthropic_claude_model(model) {
        return openai_reasoning_fields(model, group);
    }

    // GovCloud Bedrock currently rejects the Claude thinking.display field.
    let is_gov_cloud = is_gov_cloud_bedrock_target(model, base, group);
    let display = if is_gov_cloud {
        None
    } else {
        Some(
            match group
                .thinking_display
                .unwrap_or(BedrockThinkingDisplay::Summarized)
            {
                BedrockThinkingDisplay::Summarized => "summarized",
                BedrockThinkingDisplay::Omitted => "omitted",
            },
        )
    };
    // Replayed signed thinking blocks are bound to the system prompt and tools
    // they were created with; stale blocks are dropped on replay. Skipped on
    // GovCloud like display.
    let use_block_binding = !is_gov_cloud && supports_thinking_block_binding(model);
    let adaptive = supports_adaptive_thinking(&model.base.id, Some(&model.base.name));
    let mut result = crate::types::JsonObject::new();
    if adaptive {
        let mut thinking = crate::types::JsonObject::new();
        thinking.insert("type".to_string(), Value::String("adaptive".to_string()));
        if let Some(display) = display {
            thinking.insert("display".to_string(), Value::String(display.to_string()));
        }
        if use_block_binding {
            thinking.insert(
                "block_binding".to_string(),
                json!({ "prefix_mismatch_behavior": "drop_block" }),
            );
        }
        result.insert("thinking".to_string(), Value::Object(thinking));
        result.insert(
            "output_config".to_string(),
            json!({ "effort": map_thinking_level_to_effort(model, group.reasoning) }),
        );
        if use_block_binding {
            result.insert(
                "anthropic_beta".to_string(),
                json!([THINKING_BINDING_CONTROLS_BETA]),
            );
        }
    } else {
        let default_budgets: [(ThinkingLevel, u64); 6] = [
            (ThinkingLevel::Minimal, 1024),
            (ThinkingLevel::Low, 2048),
            (ThinkingLevel::Medium, 8192),
            (ThinkingLevel::High, 16384),
            (ThinkingLevel::Xhigh, 16384), // Budget-based Claude clamps extended levels to high
            (ThinkingLevel::Max, 16384),
        ];
        // Custom budgets only cover token-based levels through high.
        let level = match reasoning {
            ThinkingLevel::Xhigh | ThinkingLevel::Max => ThinkingLevel::High,
            other => other,
        };
        let budget = group
            .thinking_budgets
            .as_ref()
            .and_then(|budgets| match level {
                ThinkingLevel::Minimal => budgets.minimal,
                ThinkingLevel::Low => budgets.low,
                ThinkingLevel::Medium => budgets.medium,
                ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => budgets.high,
            })
            .unwrap_or_else(|| {
                default_budgets
                    .iter()
                    .find(|(key, _)| *key == reasoning)
                    .map(|(_, value)| *value)
                    .unwrap_or(16384)
            });
        let mut thinking = crate::types::JsonObject::new();
        thinking.insert("type".to_string(), Value::String("enabled".to_string()));
        thinking.insert("budget_tokens".to_string(), json!(budget));
        if let Some(display) = display {
            thinking.insert("display".to_string(), Value::String(display.to_string()));
        }
        result.insert("thinking".to_string(), Value::Object(thinking));
    }

    if !adaptive && group.interleaved_thinking.unwrap_or(true) {
        result.insert(
            "anthropic_beta".to_string(),
            json!(["interleaved-thinking-2025-05-14"]),
        );
    }

    Some(Value::Object(result))
}

/// `mapStopReason`.
fn map_stop_reason(reason: Option<&str>) -> (StopReason, Option<String>) {
    match reason {
        Some("end_turn") | Some("stop_sequence") => (StopReason::Stop, None),
        Some("max_tokens") | Some("model_context_window_exceeded") => (StopReason::Length, None),
        Some("tool_use") => (StopReason::ToolUse, None),
        Some(other) => (
            StopReason::Error,
            Some(format!("Provider stopped with: {other}")),
        ),
        None => (StopReason::Error, None),
    }
}

// ---------------------------------------------------------------------------
// Streaming block handlers
// ---------------------------------------------------------------------------

/// Parallel scratch state for the blocks in `output.content` (upstream stores
/// `index`/`partialJson`/`redactedChunks` directly on the block).
struct BlockScratch {
    bedrock_index: u32,
    partial_json: String,
    /// Scratch buffer for encrypted reasoning deltas, joined into
    /// `thinkingSignature`.
    redacted_chunks: Vec<Vec<u8>>,
    /// Set once `content_block_stop` finalized the block (upstream deletes
    /// `index`, so later events no longer match the block).
    stopped: bool,
}

/// `flushRedactedContent`: encode buffered encrypted reasoning into
/// `thinkingSignature` and drop the scratch buffer.
fn flush_redacted_content(
    position: usize,
    blocks: &mut [BlockScratch],
    output: &mut AssistantMessage,
) {
    if blocks[position].redacted_chunks.is_empty() {
        return;
    }
    let joined = blocks[position].redacted_chunks.concat();
    let AssistantContent::Thinking(block) = &mut output.content[position] else {
        return;
    };
    block.thinking_signature = Some(base64::engine::general_purpose::STANDARD.encode(joined));
    blocks[position].redacted_chunks.clear();
}

/// `handleContentBlockStart`.
fn handle_content_block_start(
    event: ContentBlockStartEvent,
    blocks: &mut Vec<BlockScratch>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) {
    let index = event.content_block_index.unwrap_or(0);
    let Some(tool_use) = event.start.and_then(|start| start.tool_use) else {
        return;
    };
    output.content.push(AssistantContent::ToolCall(ToolCall {
        id: tool_use.tool_use_id.unwrap_or_default(),
        name: tool_use.name.unwrap_or_default(),
        arguments: crate::types::JsonObject::new(),
        thought_signature: None,
        namespace: None,
    }));
    blocks.push(BlockScratch {
        bedrock_index: index,
        partial_json: String::new(),
        redacted_chunks: Vec::new(),
        stopped: false,
    });
    stream.push(AssistantMessageEvent::ToolcallStart {
        content_index: (output.content.len() - 1) as u32,
        partial: output.clone(),
    });
}

/// `handleContentBlockDelta`.
fn handle_content_block_delta(
    event: ContentBlockDeltaEvent,
    blocks: &mut Vec<BlockScratch>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) {
    let content_block_index = event.content_block_index.unwrap_or(0);
    let Some(delta) = event.delta else {
        return;
    };
    let position = blocks
        .iter()
        .position(|b| b.bedrock_index == content_block_index && !b.stopped);

    if let Some(text) = &delta.text {
        // If no text block exists yet, create one: `contentBlockStart` is not
        // sent for text blocks.
        let position = match position {
            Some(position) => position,
            None => {
                output
                    .content
                    .push(AssistantContent::Text(crate::types::TextContent {
                        text: String::new(),
                        text_signature: None,
                    }));
                blocks.push(BlockScratch {
                    bedrock_index: content_block_index,
                    partial_json: String::new(),
                    redacted_chunks: Vec::new(),
                    stopped: false,
                });
                let index = output.content.len() - 1;
                stream.push(AssistantMessageEvent::TextStart {
                    content_index: index as u32,
                    partial: output.clone(),
                });
                index
            }
        };
        if let Some(AssistantContent::Text(block)) = output.content.get_mut(position) {
            block.text += text;
            stream.push(AssistantMessageEvent::TextDelta {
                content_index: position as u32,
                delta: text.clone(),
                partial: output.clone(),
            });
        }
        return;
    }

    if let Some(tool_use) = &delta.tool_use {
        let Some(position) = position else {
            return;
        };
        if !matches!(
            output.content.get(position),
            Some(AssistantContent::ToolCall(_))
        ) {
            return;
        }
        let input = tool_use.input.clone().unwrap_or_default();
        blocks[position].partial_json += &input;
        let arguments = parse_streaming_json(Some(&blocks[position].partial_json));
        if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(position) {
            block.arguments = arguments.as_object().cloned().unwrap_or_default();
        }
        stream.push(AssistantMessageEvent::ToolcallDelta {
            content_index: position as u32,
            delta: input,
            partial: output.clone(),
        });
        return;
    }

    if let Some(reasoning) = &delta.reasoning_content {
        let position = match position {
            Some(position) => position,
            None => {
                output
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: Some(String::new()),
                        redacted: None,
                    }));
                blocks.push(BlockScratch {
                    bedrock_index: content_block_index,
                    partial_json: String::new(),
                    redacted_chunks: Vec::new(),
                    stopped: false,
                });
                let index = output.content.len() - 1;
                stream.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index as u32,
                    partial: output.clone(),
                });
                index
            }
        };
        let is_thinking_block = matches!(
            output.content.get(position),
            Some(AssistantContent::Thinking(_))
        );
        if !is_thinking_block {
            return;
        }
        if let Some(text) = reasoning.text.as_deref().filter(|text| !text.is_empty()) {
            if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(position) {
                block.thinking += text;
            }
            stream.push(AssistantMessageEvent::ThinkingDelta {
                content_index: position as u32,
                delta: text.to_string(),
                partial: output.clone(),
            });
        }
        // `thinkingSignature` holds either an Anthropic signature or an opaque
        // redacted payload, never both.
        if let Some(signature) = reasoning.signature.as_deref().filter(|s| !s.is_empty()) {
            if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(position) {
                if block.redacted != Some(true) {
                    let slot = block.thinking_signature.get_or_insert_with(String::new);
                    slot.push_str(signature);
                }
            }
        }
        if let Some(redacted_content) = reasoning
            .redacted_content
            .as_deref()
            .and_then(|payload| {
                base64::engine::general_purpose::STANDARD
                    .decode(payload)
                    .ok()
            })
            .filter(|bytes| !bytes.is_empty())
        {
            // Encrypted reasoning from non-Anthropic models on Bedrock. The
            // payload is opaque, so keep it verbatim in `thinkingSignature` and
            // replay it on the next turn.
            let mut placeholder: Option<String> = None;
            if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(position) {
                if block.redacted != Some(true) {
                    block.redacted = Some(true);
                    block.thinking_signature = Some(String::new());
                    block.thinking += REDACTED_THINKING_PLACEHOLDER;
                    placeholder = Some(REDACTED_THINKING_PLACEHOLDER.to_string());
                }
            }
            if let Some(placeholder) = placeholder {
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: position as u32,
                    delta: placeholder,
                    partial: output.clone(),
                });
            }
            blocks[position].redacted_chunks.push(redacted_content);
        }
    }
}

/// `handleContentBlockStop`.
fn handle_content_block_stop(
    event: ContentBlockStopEvent,
    blocks: &mut Vec<BlockScratch>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) {
    let content_block_index = event.content_block_index.unwrap_or(0);
    let Some(position) = blocks
        .iter()
        .position(|b| b.bedrock_index == content_block_index && !b.stopped)
    else {
        return;
    };
    blocks[position].stopped = true;
    flush_redacted_content(position, blocks, output);
    match output.content.get(position).cloned() {
        Some(AssistantContent::Text(block)) => {
            stream.push(AssistantMessageEvent::TextEnd {
                content_index: position as u32,
                content: block.text,
                partial: output.clone(),
            });
        }
        Some(AssistantContent::Thinking(block)) => {
            stream.push(AssistantMessageEvent::ThinkingEnd {
                content_index: position as u32,
                content: block.thinking,
                partial: output.clone(),
            });
        }
        Some(AssistantContent::ToolCall(mut block)) => {
            let arguments = parse_streaming_json(Some(&blocks[position].partial_json));
            block.arguments = arguments.as_object().cloned().unwrap_or_default();
            // Finalize in-place and strip the scratch buffer so replay only
            // carries parsed arguments.
            blocks[position].partial_json.clear();
            output.content[position] = AssistantContent::ToolCall(block.clone());
            stream.push(AssistantMessageEvent::ToolcallEnd {
                content_index: position as u32,
                tool_call: block,
                partial: output.clone(),
            });
        }
        None => {}
    }
}

/// `handleMetadata`.
fn handle_metadata(event: MetadataEvent, model: &Model, output: &mut AssistantMessage) {
    let Some(usage) = event.usage else {
        return;
    };
    output.usage.input = usage.input_tokens.unwrap_or(0);
    output.usage.output = usage.output_tokens.unwrap_or(0);
    output.usage.cache_read = usage.cache_read_input_tokens.unwrap_or(0);
    output.usage.cache_write = usage.cache_write_input_tokens.unwrap_or(0);
    // `cacheDetails?.reduce(...)`: absent cacheDetails stays unset (undefined);
    // a present-but-empty array sums to 0.
    output.usage.cache_write_1h = if usage.cache_details.is_empty() {
        None
    } else {
        Some(
            usage
                .cache_details
                .iter()
                .filter(|detail| detail.ttl.as_deref() == Some(CACHE_TTL_ONE_HOUR))
                .map(|detail| detail.input_tokens.unwrap_or(0))
                .sum(),
        )
    };
    output.usage.total_tokens = match usage.total_tokens {
        Some(total) if total > 0 => total,
        _ => output.usage.input + output.usage.output,
    };
    calculate_cost(
        &crate::types::AnyModel::Chat(model.clone()),
        &mut output.usage,
    );
}

// ---------------------------------------------------------------------------
// stream()
// ---------------------------------------------------------------------------

/// Upstream `stream` export for the bedrock-converse-stream API.
pub fn stream(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
) -> AssistantMessageEventStream {
    stream_with_options(model, context, options, &BedrockOptions::default())
}

/// [`stream`] with the upstream `BedrockOptions` group (deviation D17).
pub fn stream_with_options(
    model: &Model,
    context: &TranscriptContext,
    options: &StreamOptions,
    group: &BedrockOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    // Bedrock has no mid-conversation system messages; fold them into the
    // leading prompt.
    let normalized_context = collapse_system_messages(context);
    let model = model.clone();
    let options = options.clone();
    let group = group.clone();
    let inner_stream = stream.clone();
    tokio::spawn(async move {
        run_stream(model, normalized_context, options, group, inner_stream).await;
    });
    stream
}

fn done_reason(stop_reason: StopReason) -> StreamDoneReason {
    match stop_reason {
        StopReason::Stop => StreamDoneReason::Stop,
        StopReason::Length => StreamDoneReason::Length,
        StopReason::ToolUse => StreamDoneReason::ToolUse,
        StopReason::Deferred => StreamDoneReason::Deferred,
        StopReason::Pending | StopReason::Error | StopReason::Aborted => StreamDoneReason::Stop,
    }
}

#[allow(clippy::too_many_lines)]
async fn run_stream(
    model: Model,
    normalized_context: TranscriptContext,
    options: StreamOptions,
    group: BedrockOptions,
    stream: AssistantMessageEventStream,
) {
    let mut output = AssistantMessage {
        content: Vec::new(),
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
        duration_ms: None,
    };

    // Parallel scratch state for the blocks in `output.content`, and the
    // response's request id — both must survive into the catch-equivalent
    // (upstream declares them outside the `try` for the same reason).
    let mut blocks: Vec<BlockScratch> = Vec::new();
    let mut response_request_id: Option<String> = None;

    let result: Result<(), StreamFailure> = async {
        let config = resolve_client_config(&model, &options.base, &group);
        let cache_retention =
            resolve_cache_retention(options.cache_retention, options.base.env.as_ref());
        let inference_max_tokens = options
            .max_tokens
            .or_else(|| is_anthropic_claude_model(&model).then_some(model.max_tokens));
        let initial_system_message = get_initial_system_message(normalized_context.messages());
        let initial_system_prompt = initial_system_message.map(get_system_message_text);
        let supports_strict_mode = model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_strict_mode)
            .unwrap_or(false);
        let input = ConverseStreamInput {
            model_id: model.base.id.clone(),
            messages: Some(convert_messages(
                &normalized_context,
                &model,
                cache_retention,
                options.base.env.as_ref(),
            )?),
            system: build_system_prompt(
                initial_system_prompt.as_deref(),
                &model,
                cache_retention,
                options.base.env.as_ref(),
            ),
            inference_config: InferenceConfiguration {
                max_tokens: inference_max_tokens,
                temperature: options.temperature,
            },
            tool_config: convert_tool_config(
                &get_current_tools(normalized_context.messages()),
                group.tool_choice.as_ref(),
                supports_strict_mode,
            )?,
            additional_model_request_fields: build_additional_model_request_fields(
                &model,
                &options.base,
                &group,
            ),
            request_metadata: group.request_metadata.clone(),
        };
        let mut payload_value = serde_json::to_value(&input)
            .map_err(|error| StreamFailure::from_message(error.to_string()))?;
        if let Some(hook) = &options.base.on_payload {
            if let Some(next) = hook(payload_value.clone(), model.clone()).await {
                payload_value = next;
            }
        }
        let payload = serde_json::to_vec(&payload_value)
            .map_err(|error| StreamFailure::from_message(error.to_string()))?;

        // The SDK rejects before sending when the signal is already aborted.
        if let Some(signal) = &options.base.signal {
            if signal.aborted() {
                return Err(StreamFailure::from_message("Request was aborted"));
            }
        }

        let custom_headers =
            crate::utils::headers::provider_headers_to_record(&[options.base.headers.as_ref()]);
        let fetch: FetchFn = options.base.fetch.clone().unwrap_or_default();
        let response = crate::utils::provider_retry::retry_provider_request(
            || {
                let request = build_signed_request(
                    &model,
                    &payload,
                    &config,
                    custom_headers.as_ref(),
                    options.base.signal.clone(),
                    options.base.timeout_ms,
                );
                let fetch = fetch.clone();
                async move {
                    let request = request?;
                    match fetch.send(request).await {
                        Ok(response) if response.ok() => Ok(response),
                        Ok(response) => {
                            let status = response.status;
                            let headers = response.headers.clone();
                            let failure = response_to_failure(response).await;
                            Err(BedrockAttemptError {
                                message: match &failure {
                                    StreamFailure::Sdk(error) => error.message.clone(),
                                    StreamFailure::Message(message) => message.clone(),
                                },
                                status: failure
                                    .sdk_error()
                                    .and_then(|error| error.status)
                                    .or(Some(status)),
                                headers,
                                failure,
                            })
                        }
                        Err(error) => Err(BedrockAttemptError {
                            message: error.message.clone(),
                            status: error.status,
                            headers: error.headers,
                            failure: StreamFailure::Sdk(BedrockSdkError {
                                name: None,
                                message: error.message,
                                status: error.status,
                                request_id: None,
                                body: None,
                                is_service_exception: false,
                            }),
                        }),
                    }
                }
            },
            crate::utils::provider_retry::ProviderRetryOptions {
                max_retries: Some(options.base.max_retries.unwrap_or(2)),
                max_retry_delay_ms: options.base.max_retry_delay_ms,
                signal: options.base.signal.clone(),
                no_retry_statuses: None,
            },
        )
        .await
        .map_err(StreamFailure::from)?;

        // Kept outside the try so the catch can still correlate a mid-stream
        // failure: exceptions delivered as stream events carry no HTTP metadata.
        response_request_id =
            normalize_diagnostic_value(response.header("x-amzn-requestid").as_deref());
        if let Some(hook) = &options.base.on_response {
            hook(
                ProviderResponse {
                    status: response.status,
                    headers: crate::utils::headers::headers_to_record(&response.headers),
                },
                model.clone(),
            )
            .await;
        }

        let events_result: Result<(), StreamFailure> = 'events: {
            let body = match response.body {
                HttpBody::Stream(body_stream) => body_stream,
                other => Box::pin(futures_util::stream::once(
                    async move { other.collect().await },
                )) as BodyStream,
            };
            let mut decoder = EventStreamDecoder::new();
            futures_util::pin_mut!(body);
            loop {
                // Drain every complete message already buffered.
                loop {
                    let message = match decoder.next_message() {
                        Ok(Some(message)) => message,
                        Ok(None) => break,
                        Err(message) => break 'events Err(StreamFailure::from_message(message)),
                    };
                    let Some(item) =
                        frame_to_item(&message).map_err(StreamFailure::from_message)?
                    else {
                        continue;
                    };
                    if let Some(hook) = &options.on_provider_stream_event {
                        let payload = match &item {
                            BedrockStreamItem::Event {
                                event_type,
                                payload,
                                ..
                            } => {
                                json!({ event_type.clone(): payload })
                            }
                            BedrockStreamItem::Exception {
                                member,
                                name,
                                message,
                                ..
                            } => {
                                json!({ member.clone(): { "message": message, "name": name } })
                            }
                        };
                        hook(payload, model.clone()).await;
                    }
                    match item {
                        BedrockStreamItem::Event { event, .. } => match event {
                            ConverseStreamEvent::MessageStart(start) => {
                                if start.role.as_deref() != Some("assistant") {
                                    break 'events Err(StreamFailure::from_message(
										"Unexpected assistant message start but got user message start instead",
									));
                                }
                                stream.push(AssistantMessageEvent::Start {
                                    partial: output.clone(),
                                });
                            }
                            ConverseStreamEvent::ContentBlockStart(start) => {
                                handle_content_block_start(
                                    start,
                                    &mut blocks,
                                    &mut output,
                                    &stream,
                                );
                            }
                            ConverseStreamEvent::ContentBlockDelta(delta) => {
                                handle_content_block_delta(
                                    delta,
                                    &mut blocks,
                                    &mut output,
                                    &stream,
                                );
                            }
                            ConverseStreamEvent::ContentBlockStop(stop) => {
                                handle_content_block_stop(stop, &mut blocks, &mut output, &stream);
                            }
                            ConverseStreamEvent::MessageStop(stop) => {
                                output.raw_stop_reason = stop.stop_reason.clone();
                                let (stop_reason, error_message) =
                                    map_stop_reason(stop.stop_reason.as_deref());
                                output.stop_reason = stop_reason;
                                if let Some(error_message) = error_message {
                                    output.error_message = Some(error_message);
                                }
                            }
                            ConverseStreamEvent::Metadata(metadata) => {
                                handle_metadata(metadata, &model, &mut output);
                            }
                        },
                        BedrockStreamItem::Exception {
                            name,
                            message,
                            modeled,
                            ..
                        } => {
                            break 'events Err(StreamFailure::Sdk(BedrockSdkError {
                                name: Some(name),
                                message,
                                status: None,
                                request_id: None,
                                body: None,
                                is_service_exception: modeled,
                            }));
                        }
                    }
                }
                match body.as_mut().next().await {
                    Some(Ok(chunk)) => decoder.push(&chunk),
                    Some(Err(error)) => {
                        break 'events Err(StreamFailure::from_message(error.message))
                    }
                    None => break 'events Ok(()),
                }
            }
        };
        events_result?;

        if options
            .base
            .signal
            .as_ref()
            .map(|signal| signal.aborted())
            .unwrap_or(false)
        {
            return Err(StreamFailure::from_message("Request was aborted"));
        }
        if output.stop_reason == StopReason::Pending {
            return Err(StreamFailure::from_message(
                "Bedrock stream ended without a stop reason",
            ));
        }
        if output.stop_reason == StopReason::Error || output.stop_reason == StopReason::Aborted {
            return Err(StreamFailure::from_message(
                output
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "An unknown error occurred".to_string()),
            ));
        }

        // A stream can settle without stopping every block, so finalize here too.
        for position in 0..output.content.len() {
            flush_redacted_content(position, &mut blocks, &mut output);
        }
        let reason = done_reason(output.stop_reason);
        stream.push(done_event(output.clone(), reason));
        stream.end(Some(output.clone()));
        Ok(())
    }
    .await;

    if let Err(failure) = result {
        for position in 0..output.content.len() {
            flush_redacted_content(position, &mut blocks, &mut output);
        }
        let aborted = options
            .base
            .signal
            .as_ref()
            .map(|signal| signal.aborted())
            .unwrap_or(false);
        output.stop_reason = if aborted {
            StopReason::Aborted
        } else {
            StopReason::Error
        };
        output.error_message = Some(format_bedrock_error(&failure));
        if output.stop_reason == StopReason::Error {
            append_bedrock_failure_diagnostic(
                &mut output,
                &failure,
                response_request_id.as_deref(),
            );
        }
        let reason = if aborted {
            StreamErrorReason::Aborted
        } else {
            StreamErrorReason::Error
        };
        stream.push(error_event(output, reason));
        stream.end(None);
    }
}

// ---------------------------------------------------------------------------
// streamSimple()
// ---------------------------------------------------------------------------

/// Upstream `streamSimple` export.
pub fn stream_simple(
    model: &Model,
    context: &TranscriptContext,
    options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
    stream_simple_with_options(model, context, options, &BedrockOptions::default())
}

/// [`stream_simple`] with the upstream `BedrockOptions` group (deviation D17).
pub fn stream_simple_with_options(
    model: &Model,
    context: &TranscriptContext,
    options: &SimpleStreamOptions,
    group: &BedrockOptions,
) -> AssistantMessageEventStream {
    let mut base = crate::api::simple_options::build_base_options(model, context, options, None);
    let mut group = group.clone();
    group.tool_choice = options.tool_choice.map(|tool_choice| match tool_choice {
        ToolChoice::Auto => BedrockToolChoice::Auto,
        ToolChoice::None => BedrockToolChoice::None,
    });

    let Some(reasoning) = options.reasoning else {
        group.reasoning = None;
        return stream_with_options(model, context, &base, &group);
    };
    group.reasoning = Some(reasoning);

    if is_anthropic_claude_model(model) {
        if supports_adaptive_thinking(&model.base.id, Some(&model.base.name)) {
            group.thinking_budgets = options.thinking_budgets.clone();
            return stream_with_options(model, context, &base, &group);
        }

        // Undefined means the caller did not request an output cap; let the
        // helper use the model cap. Do not coerce to 0 here, or the thinking
        // budget would become the entire maxTokens value.
        let adjusted = crate::api::simple_options::adjust_max_tokens_for_thinking(
            base.max_tokens,
            model.max_tokens,
            reasoning,
            options.thinking_budgets.as_ref(),
        );
        let max_tokens = crate::api::simple_options::clamp_max_tokens_to_context(
            model,
            context,
            adjusted.max_tokens,
        );
        base.max_tokens = Some(max_tokens);
        let mut budgets = options.thinking_budgets.clone().unwrap_or_default();
        let value = adjusted
            .thinking_budget
            .min(max_tokens.saturating_sub(1024));
        match crate::api::simple_options::clamp_reasoning(Some(reasoning))
            .expect("reasoning is Some")
        {
            ThinkingLevel::Minimal => budgets.minimal = Some(value),
            ThinkingLevel::Low => budgets.low = Some(value),
            ThinkingLevel::Medium => budgets.medium = Some(value),
            ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => {
                budgets.high = Some(value)
            }
        }
        group.thinking_budgets = Some(budgets);
        return stream_with_options(model, context, &base, &group);
    }

    group.thinking_budgets = options.thinking_budgets.clone();
    stream_with_options(model, context, &base, &group)
}

// ---------------------------------------------------------------------------
// ProviderStreams export
// ---------------------------------------------------------------------------

/// The bedrock-converse-stream API implementation (`export const
/// bedrockConverseStreamApi` upstream object). No deferred methods are
/// exported upstream, so both presence probes stay `false`.
pub struct BedrockConverseStreamApi;

impl ProviderStreams for BedrockConverseStreamApi {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        stream_simple(model, context, options)
    }
}

/// Upstream `bedrockConverseStreamApi()`.
pub fn bedrock_converse_stream_api() -> Arc<dyn ProviderStreams> {
    Arc::new(BedrockConverseStreamApi)
}

// ---------------------------------------------------------------------------
// Tests (ports of test/bedrock-*.test.ts)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::ProviderHttpError;
    use crate::types::{
        BaseModelFields, ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context, Message,
        ModelCompat, ModelCost, ModelCostRates, ModelInput, StrictSchemaMode, ToolResultMessage,
        UserContentField, UserMessage,
    };
    use crate::utils::abort::AbortController;
    use futures_util::StreamExt;
    use parking_lot::Mutex;

    // -- environment guard ---------------------------------------------------

    const GUARDED_VARS: &[&str] = &[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_BEDROCK_SKIP_AUTH",
        "AWS_BEDROCK_FORCE_CACHE",
        "AWS_BEDROCK_FORCE_HTTP1",
        "PI_CACHE_RETENTION",
    ];

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Serializes AWS env mutation across tests (upstream `vi.stubEnv`).
    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
        _lock: parking_lot::MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn set(&self, name: &str, value: &str) {
            std::env::set_var(name, value);
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    fn env_guard() -> EnvGuard {
        let lock = ENV_LOCK.lock();
        let mut saved = Vec::new();
        for name in GUARDED_VARS {
            saved.push((name.to_string(), std::env::var(name).ok()));
            std::env::remove_var(name);
        }
        EnvGuard { saved, _lock: lock }
    }

    // -- fixtures -------------------------------------------------------------

    fn aws_test_env() -> crate::types::ProviderEnv {
        [
            ("AWS_ACCESS_KEY_ID".to_string(), "AKIAEXAMPLE".to_string()),
            (
                "AWS_SECRET_ACCESS_KEY".to_string(),
                "secretexample".to_string(),
            ),
        ]
        .into_iter()
        .collect()
    }

    fn builtin_model(id: &str) -> Model {
        crate::providers::amazon_bedrock_models::amazon_bedrock_models()
            .into_iter()
            .find(|model| model.base.id == id)
            .unwrap_or_else(|| panic!("catalog model not found: {id}"))
    }

    fn override_model(base: &Model, id: &str, name: &str) -> Model {
        let mut model = base.clone();
        model.base.id = id.to_string();
        model.base.name = name.to_string();
        model
    }

    fn with_base_url(model: &Model, base_url: &str) -> Model {
        let mut model = model.clone();
        model.base.base_url = base_url.to_string();
        model
    }

    /// Upstream `baseModel` fixture of bedrock-convert-messages.test.ts.
    fn convert_messages_base_model() -> Model {
        Model {
            base: BaseModelFields {
                id: "us.anthropic.claude-sonnet-4-5-20250929-v1:0".to_string(),
                name: "Claude Sonnet 4.5 (US)".to_string(),
                api: "bedrock-converse-stream".to_string(),
                provider: "amazon-bedrock".to_string(),
                base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
                input: vec![ModelInput::Text, ModelInput::Image],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 3.0,
                        output: 15.0,
                        cache_read: 0.3,
                        cache_write: 3.75,
                    },
                    tiers: None,
                },
                headers: None,
            },
            model_type: None,
            reasoning: true,
            thinking_level_map: None,
            prompt_cache: None,
            context_window: 200_000,
            max_tokens: 64_000,
            sampling_params: None,
            sampling_params_by_thinking_level: None,
            compat: Some(ModelCompat {
                supports_strict_mode: Some(true),
                ..Default::default()
            }),
        }
    }

    fn nova_model() -> Model {
        let mut model = convert_messages_base_model();
        model.base.id = "amazon.nova-lite-v1:0".to_string();
        model.base.name = "Nova Lite".to_string();
        model.reasoning = false;
        model.compat = None;
        model
    }

    /// Upstream `gptModel` fixture of bedrock-redacted-reasoning.test.ts.
    fn gpt_model() -> Model {
        Model {
            base: BaseModelFields {
                id: "global.openai.gpt-5.6-terra".to_string(),
                name: "GPT-5.6 Terra (Global)".to_string(),
                api: "bedrock-converse-stream".to_string(),
                provider: "amazon-bedrock".to_string(),
                base_url: "https://bedrock-runtime.ap-northeast-1.amazonaws.com".to_string(),
                input: vec![ModelInput::Text],
                input_limits: None,
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 1.25,
                        output: 10.0,
                        cache_read: 0.125,
                        cache_write: 0.0,
                    },
                    tiers: None,
                },
                headers: None,
            },
            model_type: None,
            reasoning: true,
            thinking_level_map: None,
            prompt_cache: None,
            context_window: 400_000,
            max_tokens: 128_000,
            sampling_params: None,
            sampling_params_by_thinking_level: None,
            compat: None,
        }
    }

    fn user_context(messages: Vec<Message>) -> Context {
        Context {
            system_prompt: None,
            messages,
            tools: None,
        }
    }

    fn hello_context() -> Context {
        user_context(vec![Message::User(UserMessage {
            content: UserContentField::Text("hello".to_string()),
            timestamp: 1,
        })])
    }

    fn normalize(context: &Context) -> TranscriptContext {
        crate::utils::transcript::normalize_context(context)
    }

    // -- event-stream frame builder ------------------------------------------

    fn es_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut header_bytes = Vec::new();
        for (name, value) in headers {
            header_bytes.push(name.len() as u8);
            header_bytes.extend_from_slice(name.as_bytes());
            header_bytes.push(7); // string value
            header_bytes.extend_from_slice(&(value.len() as u16).to_be_bytes());
            header_bytes.extend_from_slice(value.as_bytes());
        }
        let total = 12 + header_bytes.len() + payload.len() + 4;
        let mut frame = Vec::with_capacity(total);
        frame.extend_from_slice(&(total as u32).to_be_bytes());
        frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
        let prelude_crc = crc32(&frame);
        frame.extend_from_slice(&prelude_crc.to_be_bytes());
        frame.extend_from_slice(&header_bytes);
        frame.extend_from_slice(payload);
        let message_crc = crc32(&frame);
        frame.extend_from_slice(&message_crc.to_be_bytes());
        frame
    }

    fn event_frame(event_type: &str, payload: &Value) -> Vec<u8> {
        es_frame(
            &[(":message-type", "event"), (":event-type", event_type)],
            &serde_json::to_vec(payload).expect("payload serializes"),
        )
    }

    fn exception_frame(exception_type: &str, payload: &Value) -> Vec<u8> {
        es_frame(
            &[
                (":message-type", "exception"),
                (":exception-type", exception_type),
            ],
            &serde_json::to_vec(payload).expect("payload serializes"),
        )
    }

    fn es_body(frames: Vec<Vec<u8>>) -> Vec<u8> {
        frames.concat()
    }

    fn message_start_frame() -> Vec<u8> {
        event_frame("messageStart", &json!({ "role": "assistant" }))
    }

    fn message_stop_frame(stop_reason: &str) -> Vec<u8> {
        event_frame("messageStop", &json!({ "stopReason": stop_reason }))
    }

    // -- fetch fakes -----------------------------------------------------------

    #[derive(Debug, Clone)]
    struct RecordedRequest {
        method: String,
        url: String,
        headers: Vec<(String, String)>,
        body: Value,
    }

    type Recording = Arc<Mutex<Vec<RecordedRequest>>>;

    fn recording_fetch(
        build_response: impl Fn() -> HttpResponse + Send + Sync + 'static,
    ) -> (FetchFn, Recording) {
        let recorded: Recording = Arc::new(Mutex::new(Vec::new()));
        let sink = recorded.clone();
        let build_response = Arc::new(build_response);
        let fetch = FetchFn::new(move |request: HttpRequest| {
            let sink = sink.clone();
            let build_response = build_response.clone();
            Box::pin(async move {
                let body = match request.body {
                    HttpBody::Bytes(bytes) => {
                        serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null)
                    }
                    _ => Value::Null,
                };
                sink.lock().push(RecordedRequest {
                    method: request.method,
                    url: request.url,
                    headers: request.headers,
                    body,
                });
                Ok(build_response())
            })
        });
        (fetch, recorded)
    }

    fn event_stream_fetch(frames: &[u8], extra_headers: &[(&str, &str)]) -> (FetchFn, Recording) {
        let frames = frames.to_vec();
        let extra_headers = extra_headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect::<Vec<_>>();
        recording_fetch(move || {
            let mut headers = extra_headers.clone();
            headers.push((
                "content-type".to_string(),
                "application/vnd.amazon.eventstream".to_string(),
            ));
            HttpResponse {
                status: 200,
                headers,
                body: HttpBody::Bytes(frames.clone()),
            }
        })
    }

    fn status_fetch(status: u16, headers: &[(&str, &str)], body: &str) -> (FetchFn, Recording) {
        let headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect::<Vec<_>>();
        let body = body.to_string();
        recording_fetch(move || HttpResponse {
            status,
            headers: headers.clone(),
            body: HttpBody::Bytes(body.clone().into_bytes()),
        })
    }

    fn failing_fetch(message: &str) -> (FetchFn, Recording) {
        let message = message.to_string();
        recording_fetch(move || {
            let _ = &message;
            HttpResponse {
                status: 400,
                headers: Vec::new(),
                body: HttpBody::Bytes(b"{\"message\":\"nope\"}".to_vec()),
            }
        })
    }

    fn rejecting_fetch(message: &str) -> (FetchFn, Recording) {
        let message = message.to_string();
        let recorded: Recording = Arc::new(Mutex::new(Vec::new()));
        let sink = recorded.clone();
        let fetch = FetchFn::new(move |request: HttpRequest| {
            let sink = sink.clone();
            let message = message.clone();
            Box::pin(async move {
                let body = match request.body {
                    HttpBody::Bytes(bytes) => {
                        serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null)
                    }
                    _ => Value::Null,
                };
                sink.lock().push(RecordedRequest {
                    method: request.method,
                    url: request.url,
                    headers: request.headers,
                    body,
                });
                Err(ProviderHttpError::new(message))
            })
        });
        (fetch, recorded)
    }

    // -- stream drivers ---------------------------------------------------------

    fn base_options() -> StreamOptions {
        let mut options = StreamOptions::default();
        options.base.env = Some(aws_test_env());
        options.cache_retention = Some(CacheRetention::None);
        options
    }

    async fn run_to_result(
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        group: &BedrockOptions,
    ) -> AssistantMessage {
        stream_with_options(model, &normalize(context), options, group)
            .into_result()
            .await
    }

    /// Consume the stream until the `error` event (upstream `capturePayload`).
    async fn capture_payload_with(
        model: &Model,
        context: &Context,
        group: &BedrockOptions,
        signal: Option<AbortSignal>,
        cache_retention: Option<CacheRetention>,
    ) -> Value {
        let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let mut options = base_options();
        options.cache_retention = cache_retention;
        if let Some(signal) = signal {
            options.base.signal = Some(signal);
        }
        let sink = captured.clone();
        options.base.on_payload = Some(Arc::new(move |payload: Value, _model: Model| {
            let sink = sink.clone();
            Box::pin(async move {
                *sink.lock() = Some(payload);
                None
            })
        }));
        let (fetch, _recording) = failing_fetch("payload captured");
        options.base.fetch = Some(fetch);
        let event_stream = stream_with_options(model, &normalize(context), &options, group);
        futures_util::pin_mut!(event_stream);
        while let Some(event) = event_stream.next().await {
            if matches!(event, AssistantMessageEvent::Error { .. }) {
                break;
            }
        }
        let payload = captured
            .lock()
            .clone()
            .expect("payload captured before request abort");
        payload
    }

    fn pre_aborted_signal() -> AbortSignal {
        let controller = AbortController::new();
        controller.abort();
        controller.signal()
    }

    fn find_diagnostic(message: &AssistantMessage) -> Option<&AssistantMessageDiagnostic> {
        message.diagnostics.as_ref().and_then(|diagnostics| {
            diagnostics
                .iter()
                .find(|d| d.diagnostic_type == "bedrock_response_failure")
        })
    }

    fn assert_diagnostic_keys(diagnostic: &AssistantMessageDiagnostic) {
        let value = serde_json::to_value(diagnostic).expect("diagnostic serializes");
        let mut keys: Vec<String> = value.as_object().expect("object").keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec!["details", "timestamp", "type"]);
    }

    fn get_header<'a>(request: &'a RecordedRequest, name: &str) -> Option<&'a str> {
        request
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn header_values<'a>(request: &'a RecordedRequest, name: &str) -> Vec<&'a str> {
        request
            .headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
            .collect()
    }

    // ===========================================================================
    // Port of test/bedrock-credentials.test.ts (asserts on the resolved client
    // config; upstream captured the mocked SDK client constructor argument).
    // ===========================================================================

    #[test]
    fn prefers_explicit_and_scoped_profiles_over_ambient_aws_access_keys() {
        let _guard = env_guard();
        guard_set(
            &_guard,
            &[
                ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
                ("AWS_SECRET_ACCESS_KEY", "secretexample"),
            ],
        );
        let model = builtin_model("us.anthropic.claude-opus-4-8");

        let mut group = BedrockOptions::default();
        group.profile = Some("explicit-profile".to_string());
        let config = resolve_client_config(&model, &StreamOptions::default().base, &group);
        assert_eq!(config.profile.as_deref(), Some("explicit-profile"));
        assert!(config.credentials.is_none());

        let mut options = StreamOptions::default();
        options.base.env = Some(
            [("AWS_PROFILE".to_string(), "scoped-profile".to_string())]
                .into_iter()
                .collect(),
        );
        let config = resolve_client_config(&model, &options.base, &BedrockOptions::default());
        assert_eq!(config.profile.as_deref(), Some("scoped-profile"));
        assert!(config.credentials.is_none());
    }

    #[test]
    fn uses_ambient_aws_access_keys_when_no_profile_is_configured() {
        let _guard = env_guard();
        guard_set(
            &_guard,
            &[
                ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
                ("AWS_SECRET_ACCESS_KEY", "secretexample"),
            ],
        );
        let model = builtin_model("us.anthropic.claude-opus-4-8");

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.profile, None);
        assert_eq!(
            config.credentials,
            Some(AwsCredentials {
                access_key_id: "AKIAEXAMPLE".to_string(),
                secret_access_key: "secretexample".to_string(),
                session_token: None,
            })
        );
    }

    #[test]
    fn uses_ambient_aws_access_keys_when_only_an_ambient_profile_is_set() {
        let _guard = env_guard();
        guard_set(
            &_guard,
            &[
                ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
                ("AWS_SECRET_ACCESS_KEY", "secretexample"),
                ("AWS_PROFILE", "ambient-profile"),
            ],
        );
        let model = builtin_model("us.anthropic.claude-opus-4-8");

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.profile.as_deref(), Some("ambient-profile"));
        assert_eq!(
            config.credentials,
            Some(AwsCredentials {
                access_key_id: "AKIAEXAMPLE".to_string(),
                secret_access_key: "secretexample".to_string(),
                session_token: None,
            })
        );
    }

    fn guard_set(guard: &EnvGuard, vars: &[(&str, &str)]) {
        for (name, value) in vars {
            guard.set(name, value);
        }
    }

    // ===========================================================================
    // Port of test/bedrock-endpoint-resolution.test.ts
    // ===========================================================================

    #[test]
    fn assigns_eu_central_1_runtime_urls_to_builtin_eu_inference_profiles() {
        let model = builtin_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");
        assert_eq!(
            model.base.base_url,
            "https://bedrock-runtime.eu-central-1.amazonaws.com"
        );
    }

    #[test]
    fn does_not_pin_standard_aws_endpoints_when_aws_region_is_configured() {
        let _guard = env_guard();
        guard_set(&_guard, &[("AWS_REGION", "us-east-2")]);
        let model = builtin_model("us.anthropic.claude-opus-4-8");

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.region.as_deref(), Some("us-east-2"));
        assert_eq!(config.endpoint, None);
    }

    #[test]
    fn derives_region_from_a_builtin_eu_endpoint_when_no_region_or_profile_is_configured() {
        let _guard = env_guard();
        let model = builtin_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(
            config.endpoint.as_deref(),
            Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
        );
        assert_eq!(config.region.as_deref(), Some("eu-central-1"));
    }

    #[test]
    fn handles_missing_regions_for_explicit_scoped_and_ambient_profiles() {
        let _guard = env_guard();
        let model = builtin_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");

        let mut group = BedrockOptions::default();
        group.profile = Some("bedrock-profile".to_string());
        let config = resolve_client_config(&model, &StreamOptions::default().base, &group);
        assert_eq!(config.profile.as_deref(), Some("bedrock-profile"));
        assert_eq!(
            config.endpoint.as_deref(),
            Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
        );
        assert_eq!(config.region.as_deref(), Some("eu-central-1"));

        let mut options = StreamOptions::default();
        options.base.env = Some(
            [(
                "AWS_PROFILE".to_string(),
                "scoped-bedrock-profile".to_string(),
            )]
            .into_iter()
            .collect(),
        );
        let config = resolve_client_config(&model, &options.base, &BedrockOptions::default());
        assert_eq!(config.profile.as_deref(), Some("scoped-bedrock-profile"));
        assert_eq!(
            config.endpoint.as_deref(),
            Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
        );
        assert_eq!(config.region.as_deref(), Some("eu-central-1"));

        guard_set(&_guard, &[("AWS_PROFILE", "ambient-bedrock-profile")]);
        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.profile.as_deref(), Some("ambient-bedrock-profile"));
        assert_eq!(config.endpoint, None);
        assert_eq!(config.region, None);
    }

    #[test]
    fn still_passes_custom_bedrock_endpoints_through_to_the_sdk_client() {
        let _guard = env_guard();
        guard_set(&_guard, &[("AWS_REGION", "us-west-2")]);
        let model = with_base_url(
            &builtin_model("us.anthropic.claude-opus-4-8"),
            "https://bedrock-vpc.example.com",
        );

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(
            config.endpoint.as_deref(),
            Some("https://bedrock-vpc.example.com")
        );
        assert_eq!(config.region.as_deref(), Some("us-west-2"));
    }

    #[test]
    fn extracts_region_from_inference_profile_arn_regardless_of_aws_region() {
        let _guard = env_guard();
        guard_set(&_guard, &[("AWS_REGION", "us-east-1")]);
        let model = override_model(
            &builtin_model("us.anthropic.claude-opus-4-8"),
            "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123",
            "us.anthropic.claude-opus-4-8",
        );

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.region.as_deref(), Some("us-west-2"));
    }

    #[test]
    fn extracts_region_from_govcloud_inference_profile_arn() {
        let _guard = env_guard();
        guard_set(&_guard, &[("AWS_REGION", "us-east-1")]);
        let model = override_model(
			&builtin_model("us.anthropic.claude-opus-4-8"),
			"arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:application-inference-profile/abc123",
			"us.anthropic.claude-opus-4-8",
		);

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.region.as_deref(), Some("us-gov-west-1"));
    }

    #[test]
    fn preserves_ambient_aws_auth_for_custom_model_ids_through_compat_dispatch() {
        let _guard = env_guard();
        guard_set(&_guard, &[("AWS_PROFILE", "bedrock-profile")]);
        let model = override_model(
            &builtin_model("us.anthropic.claude-opus-4-8"),
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example",
            "us.anthropic.claude-opus-4-8",
        );

        let config = resolve_client_config(
            &model,
            &StreamOptions::default().base,
            &BedrockOptions::default(),
        );
        assert_eq!(config.profile.as_deref(), Some("bedrock-profile"));
        assert_eq!(config.token, None);
    }

    #[test]
    fn uses_the_generic_api_key_option_as_a_bedrock_bearer_token() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let mut options = StreamOptions::default();
        options.base.api_key = Some("bedrock-api-key".to_string());

        let config = resolve_client_config(&model, &options.base, &BedrockOptions::default());
        assert_eq!(config.token.as_deref(), Some("bedrock-api-key"));
    }

    // ===========================================================================
    // Port of test/bedrock-custom-headers.test.ts (upstream registered a Smithy
    // build-step middleware and drove it with fake args; the port asserts on the
    // captured HttpRequest — deviation D15/D23. The VC3 structural guard
    // (requests without a headers object) has no equivalent here.)
    // ===========================================================================

    async fn drive_bedrock(options: &StreamOptions, group: &BedrockOptions) -> AssistantMessage {
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        run_to_result(&model, &hello_context(), options, group).await
    }

    #[tokio::test]
    async fn custom_headers_vc1_injects_the_caller_header() {
        let _guard = env_guard();
        let mut options = base_options();
        options.base.headers = Some(
            [("x-custom".to_string(), Some("v".to_string()))]
                .into_iter()
                .collect(),
        );
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        options.base.fetch = Some(fetch);

        let result = drive_bedrock(&options, &BedrockOptions::default()).await;
        assert_eq!(result.stop_reason, StopReason::Error);

        let requests = recorded.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(get_header(&requests[0], "x-custom"), Some("v"));
        assert_eq!(
            get_header(&requests[0], "content-type"),
            Some("application/json")
        );
        assert!(get_header(&requests[0], "authorization")
            .expect("sigv4 auth")
            .starts_with("AWS4-HMAC-SHA256"));
    }

    #[tokio::test]
    async fn custom_headers_vc2_skips_reserved_headers_case_insensitively() {
        let _guard = env_guard();
        let mut options = base_options();
        options.base.headers = Some(
            [
                ("authorization".to_string(), Some("evil".to_string())),
                ("x-amz-date".to_string(), Some("evil".to_string())),
                ("x-allowed".to_string(), Some("ok".to_string())),
                ("Authorization".to_string(), Some("evil2".to_string())),
                ("X-Amz-Date".to_string(), Some("evil2".to_string())),
                ("HOST".to_string(), Some("evil3".to_string())),
            ]
            .into_iter()
            .collect(),
        );
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        options.base.fetch = Some(fetch);

        let _ = drive_bedrock(&options, &BedrockOptions::default()).await;

        let requests = recorded.lock();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        // Reserved SigV4/auth headers keep their real values.
        assert!(get_header(request, "authorization")
            .expect("real auth")
            .starts_with("AWS4-HMAC-SHA256"));
        let amz_date = get_header(request, "x-amz-date").expect("real date");
        assert!(
            amz_date.ends_with('Z') && amz_date.len() == 16 && amz_date.contains('T'),
            "real x-amz-date: {amz_date}"
        );
        assert_eq!(
            get_header(request, "host"),
            Some("bedrock-runtime.us-east-1.amazonaws.com")
        );
        assert_eq!(get_header(request, "x-allowed"), Some("ok"));
        // No case-variant leak of the reserved keys: only the single lowercase
        // SigV4 `authorization`/`x-amz-date`/`host` entries exist.
        assert!(header_values(request, "Authorization").len() == 1);
        assert!(!request
            .headers
            .iter()
            .any(|(key, _)| key == "Authorization" || key == "X-Amz-Date" || key == "HOST"));
        let mut names: Vec<String> = request
            .headers
            .iter()
            .map(|(key, _)| key.to_lowercase())
            .collect();
        names.sort();
        names.dedup();
        assert_eq!(
            names,
            vec![
                "authorization",
                "content-type",
                "host",
                "x-allowed",
                "x-amz-content-sha256",
                "x-amz-date"
            ]
        );
    }

    #[tokio::test]
    async fn custom_headers_vc3_sends_no_caller_headers_when_headers_is_undefined() {
        let _guard = env_guard();
        let mut options = base_options();
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        options.base.fetch = Some(fetch);

        let _ = drive_bedrock(&options, &BedrockOptions::default()).await;

        let requests = recorded.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(get_header(&requests[0], "x-custom"), None);
    }

    #[tokio::test]
    async fn custom_headers_vc3_sends_no_caller_headers_when_headers_is_empty() {
        let _guard = env_guard();
        let mut options = base_options();
        options.base.headers = Some(Default::default());
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        options.base.fetch = Some(fetch);

        let _ = drive_bedrock(&options, &BedrockOptions::default()).await;

        let requests = recorded.lock();
        assert_eq!(get_header(&requests[0], "x-custom"), None);
    }

    #[tokio::test]
    async fn custom_headers_vc4_stream_simple_forwards_headers_end_to_end() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let mut simple = SimpleStreamOptions::default();
        simple.base.base.headers = Some(
            [("x-custom".to_string(), Some("v".to_string()))]
                .into_iter()
                .collect(),
        );
        simple.base.base.env = Some(aws_test_env());
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        simple.base.base.fetch = Some(fetch);

        let _ = stream_simple(&model, &normalize(&hello_context()), &simple)
            .into_result()
            .await;

        let requests = recorded.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(get_header(&requests[0], "x-custom"), Some("v"));
    }

    // ===========================================================================
    // Port of test/bedrock-error-metadata.test.ts
    // ===========================================================================

    const VALIDATION_MESSAGE: &str = "The provided model identifier is invalid.";
    const REQUEST_ID: &str = "11111111-2222-3333-4444-555555555555";

    async fn run_bedrock_error_metadata(options: &StreamOptions) -> AssistantMessage {
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        run_to_result(
            &model,
            &hello_context(),
            options,
            &BedrockOptions::default(),
        )
        .await
    }

    #[tokio::test]
    async fn error_metadata_records_status_error_code_and_request_id_for_non_2xx() {
        let _guard = env_guard();
        let mut options = base_options();
        let (fetch, _recording) = status_fetch(
            400,
            &[
                ("x-amzn-errortype", "ValidationException"),
                ("x-amzn-requestid", REQUEST_ID),
            ],
            &json!({ "message": VALIDATION_MESSAGE }).to_string(),
        );
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(message.stop_reason, StopReason::Error);
        let diagnostic = find_diagnostic(&message).expect("diagnostic present");
        assert_eq!(
            diagnostic.details,
            Some(
                json!({"status": 400, "errorCode": "ValidationException", "requestId": REQUEST_ID})
                    .as_object()
                    .expect("object")
                    .clone()
            )
        );
        assert!(diagnostic.error.is_none());
        assert_diagnostic_keys(diagnostic);
    }

    #[tokio::test]
    async fn error_metadata_leaves_error_message_untouched_for_retry_classification() {
        let _guard = env_guard();
        let mut options = base_options();
        let (fetch, _recording) = status_fetch(
            400,
            &[
                ("x-amzn-errortype", "ValidationException"),
                ("x-amzn-requestid", REQUEST_ID),
            ],
            &json!({ "message": VALIDATION_MESSAGE }).to_string(),
        );
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(
            message.error_message.as_deref(),
            Some("Validation error: The provided model identifier is invalid.")
        );
    }

    #[tokio::test]
    async fn error_metadata_reports_only_the_request_id_for_a_modeled_mid_stream_exception() {
        let _guard = env_guard();
        let mut options = base_options();
        // Upstream's mock throws a bare object literal, so the error code is
        // genuinely unavailable; the wire-equivalent is an exception frame with
        // no :exception-type header.
        let frames = es_body(vec![
            message_start_frame(),
            es_frame(
                &[(":message-type", "exception")],
                b"{\"message\":\"Too many requests, please wait.\"}",
            ),
        ]);
        let (fetch, _recording) = event_stream_fetch(&frames, &[("x-amzn-requestid", REQUEST_ID)]);
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            find_diagnostic(&message)
                .expect("diagnostic present")
                .details,
            Some(
                json!({ "requestId": REQUEST_ID })
                    .as_object()
                    .expect("object")
                    .clone()
            )
        );
    }

    #[tokio::test]
    async fn error_metadata_captures_the_error_code_for_an_unmodeled_mid_stream_error() {
        let _guard = env_guard();
        let mut options = base_options();
        let frames = es_body(vec![
            message_start_frame(),
            exception_frame(
                "ModelStreamErrorException",
                &json!({ "message": "Model stream terminated unexpectedly." }),
            ),
        ]);
        let (fetch, _recording) = event_stream_fetch(&frames, &[("x-amzn-requestid", REQUEST_ID)]);
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(
            find_diagnostic(&message)
                .expect("diagnostic present")
                .details,
            Some(
                json!({"errorCode": "ModelStreamErrorException", "requestId": REQUEST_ID})
                    .as_object()
                    .expect("object")
                    .clone()
            )
        );
    }

    #[tokio::test]
    async fn error_metadata_does_not_report_a_transport_failure_name_as_a_provider_error_code() {
        let _guard = env_guard();
        let mut options = base_options();
        let frames = es_body(vec![
            message_start_frame(),
            exception_frame(
                "TimeoutError",
                &json!({ "message": "Connection timed out after 1000 ms" }),
            ),
        ]);
        let (fetch, _recording) = event_stream_fetch(&frames, &[("x-amzn-requestid", REQUEST_ID)]);
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(
            find_diagnostic(&message)
                .expect("diagnostic present")
                .details,
            Some(
                json!({ "requestId": REQUEST_ID })
                    .as_object()
                    .expect("object")
                    .clone()
            )
        );
    }

    #[tokio::test]
    async fn error_metadata_emits_no_diagnostic_when_the_failure_carries_no_provider_metadata() {
        let _guard = env_guard();
        let mut options = base_options();
        options.base.max_retries = Some(0);
        let (fetch, _recording) = rejecting_fetch("socket hang up");
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(message.error_message.as_deref(), Some("socket hang up"));
        assert!(find_diagnostic(&message).is_none());
    }

    #[tokio::test]
    async fn error_metadata_emits_no_diagnostic_for_an_aborted_turn() {
        let _guard = env_guard();
        let mut options = base_options();
        options.base.signal = Some(pre_aborted_signal());
        let (fetch, _recording) = status_fetch(
            400,
            &[
                ("x-amzn-errortype", "ValidationException"),
                ("x-amzn-requestid", REQUEST_ID),
            ],
            &json!({ "message": VALIDATION_MESSAGE }).to_string(),
        );
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(message.stop_reason, StopReason::Aborted);
        assert!(find_diagnostic(&message).is_none());
    }

    #[tokio::test]
    async fn error_metadata_drops_header_derived_values_that_exceed_the_length_bound() {
        let _guard = env_guard();
        let mut options = base_options();
        let (fetch, _recording) = status_fetch(
            400,
            &[
                (
                    "x-amzn-errortype",
                    &format!("{}Exception", "E".repeat(5000)),
                ),
                ("x-amzn-requestid", &"R".repeat(5000)),
            ],
            "{}",
        );
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(
            find_diagnostic(&message)
                .expect("diagnostic present")
                .details,
            Some(
                json!({ "status": 400 })
                    .as_object()
                    .expect("object")
                    .clone()
            )
        );
    }

    #[tokio::test]
    async fn error_metadata_omits_the_sdks_unknown_placeholder_instead_of_reporting_it_as_a_code() {
        let _guard = env_guard();
        let mut options = base_options();
        let (fetch, _recording) = status_fetch(
            403,
            &[
                ("x-amzn-errortype", "Unknown"),
                ("x-amzn-requestid", REQUEST_ID),
            ],
            "{\"ResponseMetadata\":{}}",
        );
        options.base.fetch = Some(fetch);

        let message = run_bedrock_error_metadata(&options).await;
        assert_eq!(
            find_diagnostic(&message)
                .expect("diagnostic present")
                .details,
            Some(
                json!({"status": 403, "requestId": REQUEST_ID})
                    .as_object()
                    .expect("object")
                    .clone()
            )
        );
    }

    // ===========================================================================
    // Port of test/bedrock-raw-stop-reason.test.ts
    // ===========================================================================

    #[tokio::test]
    async fn forwards_sdk_stream_items_in_order_before_normalizing_them() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let frames = es_body(vec![
            message_start_frame(),
            event_frame(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"text": "hello"}}),
            ),
            event_frame(
                "messageStop",
                &json!({"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}),
            ),
            event_frame(
                "metadata",
                &json!({"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}),
            ),
        ]);
        let received: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let event_models: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));

        let mut options = base_options();
        let received_hook = received.clone();
        let models_hook = event_models.clone();
        options.on_provider_stream_event =
            Some(Arc::new(move |item: Value, event_model: Model| {
                let received_hook = received_hook.clone();
                let models_hook = models_hook.clone();
                Box::pin(async move {
                    received_hook.lock().push(item);
                    models_hook.lock().push((
                        event_model.provider().to_string(),
                        event_model.id().to_string(),
                    ));
                })
            }));
        let (fetch, _recording) = event_stream_fetch(&frames, &[]);
        options.base.fetch = Some(fetch);

        let result = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        let received = received.lock();
        assert_eq!(
            received.clone(),
            vec![
                json!({"messageStart": {"role": "assistant"}}),
                json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}}),
                json!({"messageStop": {"stopReason": "end_turn", "additionalModelResponseFields": {"source": "test"}}}),
                json!({"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}}),
            ]
        );
        let models = event_models.lock();
        assert!(models
            .iter()
            .all(|(provider, id)| provider == "amazon-bedrock" && id.as_str() == model.base.id));
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(crate::types::TextContent {
                text: "hello".to_string(),
                text_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn forwards_sdk_error_items_before_reporting_them() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let frames = es_body(vec![
            message_start_frame(),
            exception_frame(
                "InternalServerException",
                &json!({"message": "bedrock stream failed"}),
            ),
        ]);
        let received: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let mut options = base_options();
        let received_hook = received.clone();
        options.on_provider_stream_event =
            Some(Arc::new(move |item: Value, _event_model: Model| {
                let received_hook = received_hook.clone();
                Box::pin(async move {
                    received_hook.lock().push(item);
                })
            }));
        let (fetch, _recording) = event_stream_fetch(&frames, &[]);
        options.base.fetch = Some(fetch);

        let result = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        let received = received.lock();
        assert_eq!(received[0], json!({"messageStart": {"role": "assistant"}}));
        assert_eq!(
            received[1]["internalServerException"]["message"],
            json!("bedrock stream failed")
        );
        assert_eq!(result.stop_reason, StopReason::Error);
        // Upstream's mock injects a plain `Error` (no service-exception prefix);
        // a real modeled exception frame carries the `Internal server error`
        // prefix from `BEDROCK_ERROR_PREFIXES`.
        assert_eq!(
            result.error_message.as_deref(),
            Some("Internal server error: bedrock stream failed")
        );
    }

    #[tokio::test]
    async fn preserves_raw_bedrock_stop_reasons_for_successful_stops() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let frames = es_body(vec![message_start_frame(), message_stop_frame("end_turn")]);
        let mut options = base_options();
        let (fetch, _recording) = event_stream_fetch(&frames, &[]);
        options.base.fetch = Some(fetch);

        let message = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(message.raw_stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(message.error_message, None);
    }

    #[tokio::test]
    async fn preserves_raw_bedrock_stop_reasons_for_provider_error_stops() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let frames = es_body(vec![
            message_start_frame(),
            message_stop_frame("guardrail_intervened"),
        ]);
        let mut options = base_options();
        let (fetch, _recording) = event_stream_fetch(&frames, &[]);
        options.base.fetch = Some(fetch);

        let message = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.raw_stop_reason.as_deref(),
            Some("guardrail_intervened")
        );
        assert_eq!(
            message.error_message.as_deref(),
            Some("Provider stopped with: guardrail_intervened")
        );
    }

    // ===========================================================================
    // Port of test/bedrock-redacted-reasoning.test.ts
    // ===========================================================================

    const REDACTED_BASE64: &str = "cnNuXzVaVnJpZjRKMGJYSXFtV2RsZWRqN1FJRmVOaWtSUWJF";

    fn redacted_bytes() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(REDACTED_BASE64)
            .expect("fixture base64")
    }

    fn redacted_delta_frame(bytes: &[u8]) -> Vec<u8> {
        let payload = base64::engine::general_purpose::STANDARD.encode(bytes);
        event_frame(
            "contentBlockDelta",
            &json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": payload}}}),
        )
    }

    fn redacted_reasoning_events() -> Vec<u8> {
        es_body(vec![
            message_start_frame(),
            redacted_delta_frame(&redacted_bytes()),
            event_frame("contentBlockStop", &json!({"contentBlockIndex": 0})),
            event_frame(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 1, "delta": {"text": "done"}}),
            ),
            event_frame("contentBlockStop", &json!({"contentBlockIndex": 1})),
            message_stop_frame("end_turn"),
        ])
    }

    async fn run_gpt_stream(frames: &[u8]) -> AssistantMessage {
        let _guard = env_guard();
        let mut options = base_options();
        let (fetch, _recording) = event_stream_fetch(frames, &[]);
        options.base.fetch = Some(fetch);
        let model = gpt_model();
        run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await
    }

    #[tokio::test]
    async fn redacted_reasoning_does_not_fail_the_stream() {
        let response = run_gpt_stream(&redacted_reasoning_events()).await;
        assert_ne!(
            response.stop_reason,
            StopReason::Error,
            "error: {:?}",
            response.error_message
        );
        let types: Vec<&str> = response
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(_) => "text",
                AssistantContent::Thinking(_) => "thinking",
                AssistantContent::ToolCall(_) => "toolCall",
            })
            .collect();
        // Reasoning precedes the answer, matching the order Bedrock streamed it.
        assert_eq!(types, vec!["thinking", "text"]);
        assert_eq!(
            response.content[1],
            AssistantContent::Text(crate::types::TextContent {
                text: "done".to_string(),
                text_signature: None
            })
        );
    }

    fn find_thinking(message: &AssistantMessage) -> &ThinkingContent {
        message
            .content
            .iter()
            .find_map(|block| match block {
                AssistantContent::Thinking(thinking) => Some(thinking),
                _ => None,
            })
            .expect("thinking block present")
    }

    #[tokio::test]
    async fn redacted_reasoning_preserves_the_encrypted_payload_on_the_assistant_message() {
        let response = run_gpt_stream(&redacted_reasoning_events()).await;
        let thinking = find_thinking(&response);
        assert_eq!(thinking.redacted, Some(true));
        assert_eq!(
            thinking.thinking_signature.as_deref(),
            Some(REDACTED_BASE64)
        );
    }

    #[tokio::test]
    async fn redacted_reasoning_encodes_the_payload_when_the_stream_never_sends_content_block_stop()
    {
        let frames = es_body(vec![
            message_start_frame(),
            redacted_delta_frame(&redacted_bytes()),
            message_stop_frame("end_turn"),
        ]);
        let response = run_gpt_stream(&frames).await;
        let thinking = find_thinking(&response);
        assert_eq!(
            thinking.thinking_signature.as_deref(),
            Some(REDACTED_BASE64)
        );
    }

    #[tokio::test]
    async fn redacted_reasoning_joins_encrypted_reasoning_split_across_deltas() {
        let bytes = redacted_bytes();
        let (head, tail) = (&bytes[..7], &bytes[7..]);
        let frames = es_body(vec![
            message_start_frame(),
            redacted_delta_frame(head),
            redacted_delta_frame(tail),
            event_frame("contentBlockStop", &json!({"contentBlockIndex": 0})),
            message_stop_frame("end_turn"),
        ]);
        let response = run_gpt_stream(&frames).await;
        let thinking = find_thinking(&response);
        assert_eq!(
            thinking.thinking_signature.as_deref(),
            Some(REDACTED_BASE64)
        );
        // The placeholder marks the block once, not once per delta.
        assert_eq!(thinking.thinking, "[Reasoning redacted]");
    }

    async fn capture_gpt_payload(context: &Context) -> Value {
        let model = gpt_model();
        capture_payload_with(
            &model,
            context,
            &BedrockOptions::default(),
            Some(pre_aborted_signal()),
            Some(CacheRetention::None),
        )
        .await
    }

    fn empty_usage_json() -> Value {
        json!({
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        })
    }

    fn gpt_assistant_message(content: Value) -> Message {
        Message::Assistant(AssistantMessage {
            content: serde_json::from_value(content).expect("assistant content"),
            api: "bedrock-converse-stream".to_string(),
            provider: "amazon-bedrock".to_string(),
            model: "global.openai.gpt-5.6-terra".to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: None,
            usage: serde_json::from_value(empty_usage_json()).expect("usage"),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
            duration_ms: None,
        })
    }

    #[tokio::test]
    async fn redacted_reasoning_replays_as_reasoning_content_redacted_content() {
        let context = user_context(vec![
            Message::User(UserMessage {
                content: UserContentField::Text("hello".to_string()),
                timestamp: 1,
            }),
            gpt_assistant_message(json!([
                {"type": "thinking", "thinking": "", "thinkingSignature": REDACTED_BASE64, "redacted": true},
                {"type": "text", "text": "done"},
            ])),
            Message::User(UserMessage {
                content: UserContentField::Text("continue".to_string()),
                timestamp: 1,
            }),
        ]);
        let payload = capture_gpt_payload(&context).await;
        let assistant = payload["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message");
        assert_eq!(
            assistant["content"],
            json!([
                {"reasoningContent": {"redactedContent": REDACTED_BASE64}},
                {"text": "done"},
            ])
        );
    }

    #[tokio::test]
    async fn redacted_reasoning_replays_before_the_tool_use_block_it_belongs_to() {
        let context = user_context(vec![
            Message::User(UserMessage {
                content: UserContentField::Text("read the file".to_string()),
                timestamp: 1,
            }),
            gpt_assistant_message(json!([
                {"type": "thinking", "thinking": "", "thinkingSignature": REDACTED_BASE64, "redacted": true},
                {"type": "toolCall", "id": "tool-1", "name": "read", "arguments": {"path": "/tmp/a.txt"}},
            ])),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "tool-1".to_string(),
                tool_name: "read".to_string(),
                content: vec![crate::types::UserContent::Text(
                    crate::types::TextContent::new("file body"),
                )],
                details: None,
                usage: None,
                nested_calls: None,
                is_error: false,
                duration_ms: None,
                timestamp: 1,
            }),
        ]);
        let payload = capture_gpt_payload(&context).await;
        let assistant = payload["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message");
        assert_eq!(
            assistant["content"],
            json!([
                {"reasoningContent": {"redactedContent": REDACTED_BASE64}},
                {"toolUse": {"toolUseId": "tool-1", "name": "read", "input": {"path": "/tmp/a.txt"}}},
            ])
        );
    }

    // ===========================================================================
    // Port of test/bedrock-response-headers.test.ts (the upstream suite starts a
    // real local HTTP server; the port injects a fake FetchFn carrying the same
    // headers — deviation D23.)
    // ===========================================================================

    #[tokio::test]
    async fn forwards_raw_smithy_response_headers_to_on_response() {
        let _guard = env_guard();
        const MODEL_ID: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
        let model = with_base_url(&builtin_model(MODEL_ID), "http://127.0.0.1:9");
        let responses: Arc<Mutex<Vec<ProviderResponse>>> = Arc::new(Mutex::new(Vec::new()));

        let mut options = StreamOptions::default();
        options.cache_retention = Some(CacheRetention::None);
        options.base.env = Some(
            [
                ("AWS_BEDROCK_FORCE_HTTP1".to_string(), "1".to_string()),
                ("AWS_BEDROCK_SKIP_AUTH".to_string(), "1".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let responses_hook = responses.clone();
        options.base.on_response = Some(Arc::new(
            move |response: ProviderResponse, _model: Model| {
                let responses_hook = responses_hook.clone();
                Box::pin(async move {
                    responses_hook.lock().push(response);
                })
            },
        ));
        // The fake response carries an empty event stream; this documents that
        // the header callback still fires before stream consumption.
        let (fetch, _recording) = event_stream_fetch(
            &[],
            &[
                ("x-bifrost-provider", "bedrock"),
                ("x-bifrost-resolved-model", MODEL_ID),
                ("x-amzn-requestid", "req-123"),
            ],
        );
        options.base.fetch = Some(fetch);

        let result = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Bedrock stream ended without a stop reason")
        );
        let responses = responses.lock();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].status, 200);
        assert_eq!(
            responses[0]
                .headers
                .get("x-amzn-requestid")
                .map(String::as_str),
            Some("req-123")
        );
        assert_eq!(
            responses[0]
                .headers
                .get("x-bifrost-provider")
                .map(String::as_str),
            Some("bedrock")
        );
        assert_eq!(
            responses[0]
                .headers
                .get("x-bifrost-resolved-model")
                .map(String::as_str),
            Some(MODEL_ID)
        );
    }

    // ===========================================================================
    // Port of test/bedrock-convert-messages.test.ts. Cases feeding
    // `{type: "unknown"}` content blocks and lone surrogates are unrepresentable
    // against the crate's closed content enums (deviation D22); the
    // assistant-empty-content variant of the skip case is ported instead.
    // ===========================================================================

    async fn capture_convert_payload(model: &Model, context: &Context) -> Value {
        capture_payload_with(
            model,
            context,
            &BedrockOptions::default(),
            Some(pre_aborted_signal()),
            Some(CacheRetention::None),
        )
        .await
    }

    fn lookup_tool(strict: StrictSchemaMode) -> Tool {
        Tool {
            name: "lookup".to_string(),
            description: "Look up a value".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }),
            constrained_sampling: Some(ConstrainedSamplingSetting::Config(
                ConstrainedSamplingConfig::JsonSchema { strict },
            )),
        }
    }

    #[tokio::test]
    async fn constrained_sampling_gates_native_strict_tool_use_by_model_capability() {
        let base_model = convert_messages_base_model();
        let mut context = user_context(vec![Message::User(UserMessage {
            content: UserContentField::Text("Use the tool".to_string()),
            timestamp: 1,
        })]);
        context.tools = Some(vec![lookup_tool(StrictSchemaMode::Require)]);

        let payload = capture_convert_payload(&base_model, &context).await;
        assert_eq!(
            payload["toolConfig"]["tools"][0]["toolSpec"]["strict"],
            json!(true)
        );

        context.tools = Some(vec![lookup_tool(StrictSchemaMode::Prefer)]);
        let nova = nova_model();
        let payload = capture_convert_payload(&nova, &context).await;
        assert!(payload["toolConfig"]["tools"][0]["toolSpec"]
            .get("strict")
            .is_none());
    }

    #[tokio::test]
    async fn tool_arguments_preserve_empty_property_names_in_streamed_arguments() {
        let _guard = env_guard();
        let base_model = convert_messages_base_model();
        let frames = es_body(vec![
            message_start_frame(),
            event_frame(
                "contentBlockStart",
                &json!({"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "tool-1", "name": "edit"}}}),
            ),
            event_frame(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": "{\"path\":\"/workspace/foobar/file.js\",\"edits\":[{\"oldText\":\"first\",\"newText\":\"updated first\"},{\"oldText\":\"second\",\"newText\":\"updated second\",\"\":\"\"}]}"}},
                }),
            ),
            event_frame("contentBlockStop", &json!({"contentBlockIndex": 0})),
            message_stop_frame("tool_use"),
        ]);
        let mut options = base_options();
        let (fetch, _recording) = event_stream_fetch(&frames, &[]);
        options.base.fetch = Some(fetch);

        let message = run_to_result(
            &base_model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        assert_eq!(
            message.content[0],
            AssistantContent::ToolCall(ToolCall {
                id: "tool-1".to_string(),
                name: "edit".to_string(),
                arguments: json!({
                    "path": "/workspace/foobar/file.js",
                    "edits": [
                        {"oldText": "first", "newText": "updated first"},
                        {"oldText": "second", "newText": "updated second", "": ""},
                    ],
                })
                .as_object()
                .expect("object")
                .clone(),
                thought_signature: None,
                namespace: None,
            })
        );
    }

    #[tokio::test]
    async fn convert_messages_replaces_blank_user_string_content_with_a_placeholder() {
        let base_model = convert_messages_base_model();
        let context = user_context(vec![Message::User(UserMessage {
            content: UserContentField::Text("   ".to_string()),
            timestamp: 1,
        })]);
        let payload = capture_convert_payload(&base_model, &context).await;
        assert_eq!(payload["messages"].as_array().expect("messages").len(), 1);
        assert_eq!(
            payload["messages"][0]["content"],
            json!([{ "text": "<empty>" }])
        );
    }

    #[tokio::test]
    async fn convert_messages_filters_blank_user_text_blocks_when_other_content_remains() {
        let base_model = convert_messages_base_model();
        let context = user_context(vec![Message::User(UserMessage {
            content: UserContentField::Blocks(vec![
                crate::types::UserContent::Text(crate::types::TextContent::new("")),
                crate::types::UserContent::Text(crate::types::TextContent::new("hello")),
            ]),
            timestamp: 1,
        })]);
        let payload = capture_convert_payload(&base_model, &context).await;
        assert_eq!(
            payload["messages"][0]["content"],
            json!([{ "text": "hello" }])
        );
    }

    #[tokio::test]
    async fn convert_messages_replaces_blank_tool_result_content_with_a_placeholder() {
        let base_model = convert_messages_base_model();
        let context = user_context(vec![Message::ToolResult(ToolResultMessage {
            tool_call_id: "tool-1".to_string(),
            tool_name: "tool".to_string(),
            content: vec![crate::types::UserContent::Text(
                crate::types::TextContent::new(""),
            )],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            duration_ms: None,
            timestamp: 1,
        })]);
        let payload = capture_convert_payload(&base_model, &context).await;
        assert_eq!(payload["messages"].as_array().expect("messages").len(), 1);
        assert_eq!(
            payload["messages"][0]["content"][0]["toolResult"]["content"],
            json!([{ "text": "<empty>" }])
        );
    }

    #[tokio::test]
    async fn convert_messages_skips_assistant_messages_with_empty_content() {
        // Rust port of "skips assistant messages with only unknown content
        // blocks": the empty-content array is the representable form (deviation
        // D22), and Bedrock rejects empty assistant content arrays.
        let base_model = convert_messages_base_model();
        let context = user_context(vec![gpt_assistant_message(json!([]))]);
        let payload = capture_convert_payload(&base_model, &context).await;
        assert_eq!(payload["messages"].as_array().expect("messages").len(), 0);
    }

    #[tokio::test]
    async fn convert_messages_removes_empty_property_names_only_from_replayed_input() {
        let base_model = convert_messages_base_model();
        let arguments = json!({
            "path": "/workspace/foobar/file.js",
            "edits": [
                {"oldText": "first", "newText": "updated first"},
                {"oldText": "second", "newText": "updated second", "": ""},
            ],
        });
        let context = user_context(vec![
            gpt_assistant_message(json!([
                {"type": "toolCall", "id": "tool-1", "name": "edit", "arguments": arguments},
            ])),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "tool-1".to_string(),
                tool_name: "edit".to_string(),
                content: vec![crate::types::UserContent::Text(
                    crate::types::TextContent::new("done"),
                )],
                details: None,
                usage: None,
                nested_calls: None,
                is_error: false,
                duration_ms: None,
                timestamp: 1,
            }),
            Message::User(UserMessage {
                content: UserContentField::Text("Continue".to_string()),
                timestamp: 1,
            }),
        ]);

        let payload = capture_convert_payload(&base_model, &context).await;
        assert_eq!(
            payload["messages"][0]["content"][0]["toolUse"]["input"],
            json!({
                "path": "/workspace/foobar/file.js",
                "edits": [
                    {"oldText": "first", "newText": "updated first"},
                    {"oldText": "second", "newText": "updated second"},
                ],
            })
        );
        // The original message object is untouched.
        let original = &context.messages[0];
        let Message::Assistant(assistant) = original else {
            panic!("assistant message");
        };
        let AssistantContent::ToolCall(tool_call) = &assistant.content[0] else {
            panic!("tool call");
        };
        assert!(tool_call
            .arguments
            .get("edits")
            .and_then(|edits| edits.get(1))
            .and_then(|edit| edit.get(""))
            .is_some());
    }

    // ===========================================================================
    // Port of test/bedrock-thinking-payload.test.ts (the max-tokens E2E
    // describe is credentials-gated upstream and stays deferred). Upstream
    // throws from `onPayload` to stop the request; the port captures the payload
    // through the hook and lets the pre-aborted signal end the stream (the hook
    // cannot throw — deviation D17 note).
    // ===========================================================================

    fn adaptive_with_binding() -> Value {
        json!({
            "type": "adaptive",
            "display": "summarized",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        })
    }

    async fn capture_thinking_payload(model: &Model, group: &BedrockOptions) -> Value {
        // No explicit cache retention: upstream's capturePayload relies on the
        // default ("short"), which matters for the cache-point assertions.
        capture_payload_with(
            model,
            &hello_context(),
            group,
            Some(pre_aborted_signal()),
            None,
        )
        .await
    }

    fn thinking_fields(payload: &Value) -> (&Value, &Value, &Value) {
        let fields = &payload["additionalModelRequestFields"];
        (
            &fields["thinking"],
            &fields["output_config"],
            &fields["anthropic_beta"],
        )
    }

    #[tokio::test]
    async fn thinking_uses_adaptive_thinking_for_claude_opus_4_8() {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("global.anthropic.claude-opus-4-6-v1"),
            "global.anthropic.claude-opus-4-8-v1",
            "Claude Opus 4.8 (Global)",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "high"}));
        assert_eq!(beta, &json!(["thinking-binding-controls-2026-08-01"]));
    }

    // Port of test/bedrock-thinking-payload.test.ts "Bedrock OpenAI reasoning
    // payload" (v1.1.0, #9331/#10142).
    #[tokio::test]
    async fn sends_reasoning_effort_for_openai_gpt_models_on_bedrock() {
        let _guard = env_guard();
        for (level, effort) in [
            (ThinkingLevel::Minimal, "low"),
            (ThinkingLevel::Low, "low"),
            (ThinkingLevel::Medium, "medium"),
            (ThinkingLevel::High, "high"),
            (ThinkingLevel::Xhigh, "xhigh"),
            (ThinkingLevel::Max, "max"),
        ] {
            for id in [
                "global.openai.gpt-6-sol",
                "us.openai.gpt-6-luna",
                "global.openai.gpt-5.6-sol",
            ] {
                let model = builtin_model(id);
                let group = BedrockOptions {
                    reasoning: Some(level),
                    ..Default::default()
                };
                let payload = capture_thinking_payload(&model, &group).await;
                assert_eq!(
                    payload["additionalModelRequestFields"],
                    json!({ "reasoning": { "effort": effort } }),
                    "{id} at {effort}"
                );
            }
        }
    }

    #[tokio::test]
    async fn sends_reasoning_effort_when_only_the_model_name_identifies_a_gpt_model() {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("global.openai.gpt-6-sol"),
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
            "GPT-6 Sol",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        assert_eq!(
            payload["additionalModelRequestFields"],
            json!({ "reasoning": { "effort": "medium" } })
        );
    }

    #[tokio::test]
    async fn sends_flat_reasoning_effort_for_gpt_oss_clamped_to_high() {
        let _guard = env_guard();
        let model = builtin_model("openai.gpt-oss-120b-1:0");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Minimal),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        assert_eq!(
            payload["additionalModelRequestFields"],
            json!({ "reasoning_effort": "low" })
        );

        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        assert_eq!(
            payload["additionalModelRequestFields"],
            json!({ "reasoning_effort": "medium" })
        );

        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Xhigh),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        assert_eq!(
            payload["additionalModelRequestFields"],
            json!({ "reasoning_effort": "high" })
        );
    }

    #[tokio::test]
    async fn sends_no_reasoning_fields_for_gpt_models_when_reasoning_is_off() {
        let _guard = env_guard();
        let model = builtin_model("global.openai.gpt-6-sol");
        let group = BedrockOptions::default();
        let payload = capture_thinking_payload(&model, &group).await;
        assert!(payload.get("additionalModelRequestFields").is_none());
    }

    #[tokio::test]
    async fn thinking_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8() {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("global.anthropic.claude-opus-4-6-v1"),
            "global.anthropic.claude-opus-4-8-v1",
            "Claude Opus 4.8 (Global)",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Xhigh),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "xhigh"}));
        assert_eq!(beta, &json!(["thinking-binding-controls-2026-08-01"]));
    }

    #[tokio::test]
    async fn thinking_uses_adaptive_thinking_for_claude_fable_5() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-fable-5");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "high"}));
        assert_eq!(beta, &json!(["thinking-binding-controls-2026-08-01"]));
    }

    #[tokio::test]
    async fn thinking_uses_adaptive_thinking_for_claude_sonnet_5() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-sonnet-5");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "high"}));
        assert_eq!(beta, &json!(["thinking-binding-controls-2026-08-01"]));
    }

    #[tokio::test]
    async fn thinking_uses_adaptive_thinking_for_claude_opus_5() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-opus-5");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "high"}));
        assert_eq!(beta, &json!(["thinking-binding-controls-2026-08-01"]));
    }

    #[tokio::test]
    async fn thinking_maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_5() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-opus-5");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Xhigh),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, _beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "xhigh"}));
    }

    #[tokio::test]
    async fn thinking_maps_xhigh_reasoning_to_effort_xhigh_for_claude_fable_5() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-fable-5");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::Xhigh),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, _beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(output_config, &json!({"effort": "xhigh"}));
    }

    // https://github.com/earendil-works/pi/issues/10324
    #[tokio::test]
    async fn thinking_sends_block_binding_and_the_binding_beta_for_claude_opus_5_5() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-opus-5-5");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, _output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &adaptive_with_binding());
        assert_eq!(beta, &json!(["thinking-binding-controls-2026-08-01"]));
    }

    // Bedrock rejects block_binding on 4.6 models: "Extra inputs are not permitted" (#10324)
    #[tokio::test]
    async fn thinking_omits_block_binding_for_claude_opus_4_6() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-opus-4-6-v1");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, _output_config, beta) = thinking_fields(&payload);
        assert_eq!(
            thinking,
            &json!({"type": "adaptive", "display": "summarized"})
        );
        assert!(beta.is_null());
    }

    #[tokio::test]
    async fn thinking_omits_block_binding_for_claude_sonnet_4_6() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-sonnet-4-6");
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, _output_config, beta) = thinking_fields(&payload);
        assert_eq!(
            thinking,
            &json!({"type": "adaptive", "display": "summarized"})
        );
        assert!(beta.is_null());
    }

    #[tokio::test]
    async fn thinking_omits_display_for_govcloud_model_ids_on_non_adaptive_claude_thinking() {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude Sonnet 4.5 (GovCloud)",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, _output_config, beta) = thinking_fields(&payload);
        assert_eq!(
            thinking,
            &json!({"type": "enabled", "budget_tokens": 16384})
        );
        assert_eq!(beta, &json!(["interleaved-thinking-2025-05-14"]));
    }

    #[tokio::test]
    async fn thinking_omits_display_for_govcloud_regions_on_adaptive_claude_thinking() {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("global.anthropic.claude-opus-4-6-v1"),
            "global.anthropic.claude-opus-4-8-v1",
            "Claude Opus 4.8 (Global)",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            region: Some("us-gov-west-1".to_string()),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, beta) = thinking_fields(&payload);
        assert_eq!(thinking, &json!({"type": "adaptive"}));
        assert_eq!(output_config, &json!({"effort": "high"}));
        assert!(beta.is_null());
    }

    // ==== Application inference profile support ====

    #[tokio::test]
    async fn profile_support_uses_adaptive_thinking_when_model_name_contains_the_model_name() {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("global.anthropic.claude-opus-4-6-v1"),
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
            "Claude Opus 4.6",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let (thinking, output_config, _beta) = thinking_fields(&payload);
        assert_eq!(
            thinking,
            &json!({"type": "adaptive", "display": "summarized"})
        );
        assert_eq!(output_config, &json!({"effort": "high"}));
    }

    #[tokio::test]
    async fn profile_support_injects_cache_points_when_model_name_identifies_a_supported_claude_model(
    ) {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("global.anthropic.claude-opus-4-6-v1"),
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
            "Claude Sonnet 4.6",
        );
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::User(UserMessage {
                content: UserContentField::Text("Hello".to_string()),
                timestamp: 1,
            })],
            tools: None,
        };
        let payload = capture_payload_with(
            &model,
            &context,
            &BedrockOptions::default(),
            Some(pre_aborted_signal()),
            None,
        )
        .await;

        // System prompt should have a cache point.
        let system = payload["system"].as_array().expect("system blocks");
        assert_eq!(system.len(), 2);
        assert!(system[1].get("cachePoint").is_some());

        // Last user message should have a cache point.
        let messages = payload["messages"].as_array().expect("messages");
        let last = messages.last().expect("last message");
        let content = last["content"].as_array().expect("content blocks");
        assert!(content
            .last()
            .expect("last block")
            .get("cachePoint")
            .is_some());
    }

    #[tokio::test]
    async fn profile_support_falls_back_to_fixed_budget_thinking_for_non_adaptive_claude_via_model_name(
    ) {
        let _guard = env_guard();
        let model = override_model(
            &builtin_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile",
            "Claude Sonnet 4.5",
        );
        let group = BedrockOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let payload = capture_thinking_payload(&model, &group).await;
        let thinking = &payload["additionalModelRequestFields"]["thinking"];
        assert_eq!(thinking["type"], json!("enabled"));
        assert!(thinking["budget_tokens"].is_number());
        assert_eq!(
            payload["additionalModelRequestFields"]["anthropic_beta"],
            json!(["interleaved-thinking-2025-05-14"])
        );
    }

    // ===========================================================================
    // Port of test/bedrock-cache-write-1h-cost.test.ts (regression for
    // earendil-works/pi#9457)
    // ===========================================================================

    #[tokio::test]
    async fn cache_write_1h_cost_prices_the_1h_cache_details_at_2x() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let frames = es_body(vec![
            message_start_frame(),
            event_frame(
                "metadata",
                &json!({
                    "usage": {
                        "inputTokens": 100,
                        "outputTokens": 5,
                        "totalTokens": 1_000_105,
                        "cacheWriteInputTokens": 1_000_000,
                        "cacheDetails": [
                            {"ttl": "1h", "inputTokens": 150_000},
                            {"ttl": "5m", "inputTokens": 600_000},
                            {"ttl": "1h", "inputTokens": 250_000},
                        ],
                    },
                }),
            ),
            message_stop_frame("end_turn"),
        ]);
        let mut options = base_options();
        let (fetch, _recording) = event_stream_fetch(&frames, &[]);
        options.base.fetch = Some(fetch);

        let result = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        assert_eq!(result.usage.cache_write, 1_000_000);
        assert_eq!(result.usage.cache_write_1h, Some(400_000));
        let rates = &model.base.cost.rates;
        let expected_cache_write_cost =
            (600_000.0 * rates.cache_write + 400_000.0 * rates.input * 2.0) / 1_000_000.0;
        assert!(
            (result.usage.cost.cache_write - expected_cache_write_cost).abs() < 1e-10,
            "cache write cost {} != {expected_cache_write_cost}",
            result.usage.cost.cache_write
        );
    }

    // ===========================================================================
    // SigV4 + event-stream decoder unit checks (wire machinery introduced by
    // deviation D16; upstream relies on the AWS SDK for these).
    // ===========================================================================

    #[test]
    fn event_stream_decoder_rejects_crc_mismatches() {
        let mut frame = es_frame(
            &[(":message-type", "event"), (":event-type", "messageStart")],
            b"{}",
        );
        // Corrupt one payload byte: the message CRC no longer matches.
        let last = frame.len() - 1;
        frame[last - 1] ^= 0xFF;
        let mut decoder = EventStreamDecoder::new();
        decoder.push(&frame);
        assert!(decoder.next_message().is_err());
    }

    #[test]
    fn event_stream_decoder_buffers_partial_frames() {
        let frame = es_frame(
            &[(":message-type", "event"), (":event-type", "messageStop")],
            b"{\"stopReason\":\"end_turn\"}",
        );
        let (head, tail) = frame.split_at(9);
        let mut decoder = EventStreamDecoder::new();
        decoder.push(head);
        assert!(decoder.next_message().expect("no error").is_none());
        decoder.push(tail);
        let message = decoder
            .next_message()
            .expect("no error")
            .expect("complete frame");
        assert_eq!(
            message.header(":event-type").and_then(HeaderValue::as_str),
            Some("messageStop")
        );
    }

    #[test]
    fn sigv4_known_vector_signs_deterministically() {
        // Frozen-time SigV4 over a known canonical request (construction
        // verified against the AWS SigV4 reference algorithm).
        let credentials = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let mut request = HttpRequest::new(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse-stream",
        );
        request
            .headers
            .push(("content-type".to_string(), "application/json".to_string()));
        sign_request_at(&mut request, "us-east-1", &credentials, b"{}", 0);
        let authorization = request
            .headers
            .iter()
            .find(|(key, _)| key == "authorization")
            .map(|(_, value)| value.clone())
            .expect("authorization header");
        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/19700101/us-east-1/bedrock/aws4_request, \
			 SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, \
			 Signature=2b6bc37622e6fa423754b8f19659c2a337fad77b05ef5f3bdebaac498ebe7945"
        );
        assert_eq!(
            get_header(
                &RecordedRequest {
                    method: "POST".to_string(),
                    url: request.url.clone(),
                    headers: request.headers.clone(),
                    body: Value::Null,
                },
                "x-amz-date"
            ),
            Some("19700101T000000Z")
        );
    }

    #[test]
    fn hmac_sha256_rfc4231_case_1() {
        // RFC 4231 test case 1: key = 0x0b * 20, data "Hi There".
        let expected = "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7";
        assert_eq!(
            hex::encode(hmac_sha256(&[0x0bu8; 20], b"Hi There")),
            expected
        );
    }

    #[test]
    fn extended_encode_uri_component_matches_the_sdk_encoding() {
        assert_eq!(
            extended_encode_uri_component("us.anthropic.claude-opus-4-8"),
            "us.anthropic.claude-opus-4-8"
        );
        assert_eq!(
            extended_encode_uri_component("eu.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "eu.anthropic.claude-sonnet-4-5-20250929-v1%3A0"
        );
        assert_eq!(extended_encode_uri_component("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn crc32_matches_known_vectors() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[tokio::test]
    async fn bearer_token_requests_skip_sigv4_and_sign_with_bearer() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let mut options = StreamOptions::default();
        options.cache_retention = Some(CacheRetention::None);
        options.base.api_key = Some("bedrock-api-key".to_string());
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        options.base.fetch = Some(fetch);

        let _ = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        let requests = recorded.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            get_header(&requests[0], "authorization"),
            Some("Bearer bedrock-api-key")
        );
        // No SigV4 machinery on the bearer path.
        assert_eq!(get_header(&requests[0], "x-amz-date"), None);
        assert_eq!(get_header(&requests[0], "x-amz-content-sha256"), None);
        let url = &requests[0].url;
        assert_eq!(
			url,
			"https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-opus-4-8/converse-stream"
		);
    }

    #[tokio::test]
    async fn signed_request_targets_the_encoded_model_path() {
        let _guard = env_guard();
        let model = builtin_model("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");
        let mut options = base_options();
        let (fetch, recorded) =
            status_fetch(403, &[("x-amzn-errortype", "ForbiddenException")], "{}");
        options.base.fetch = Some(fetch);

        let _ = run_to_result(
            &model,
            &hello_context(),
            &options,
            &BedrockOptions::default(),
        )
        .await;

        let requests = recorded.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(
			requests[0].url,
			"https://bedrock-runtime.eu-central-1.amazonaws.com/model/eu.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream"
		);
        assert_eq!(requests[0].method, "POST");
        assert_eq!(
            requests[0].body["modelId"],
            json!("eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
        );
        assert_eq!(
            requests[0].body["inferenceConfig"]["maxTokens"],
            json!(model.max_tokens)
        );
        assert!(get_header(&requests[0], "x-amz-date").is_some());
        assert!(get_header(&requests[0], "authorization")
            .expect("sigv4 auth")
            .starts_with("AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/"));
    }

    // ===========================================================================
    // Port of the bedrock cases of test/provider-error-body-regression.test.ts
    // (the openai-completions/openai-responses cases travel with those modules).
    // ===========================================================================

    /// "bedrock (body-blind) surfaces the gateway body instead of Unknown: UnknownError".
    #[tokio::test]
    async fn bedrock_surfaces_the_gateway_body_instead_of_unknown_unknownerror() {
        let _guard = env_guard();
        let model = builtin_model("us.anthropic.claude-opus-4-8");
        let (fetch, _recording) =
            status_fetch(403, &[], "{\"message\":\"blocked by gateway WAF\"}");
        let mut simple = SimpleStreamOptions::default();
        simple.base.base.env = Some(aws_test_env());
        simple.base.base.fetch = Some(fetch);

        let context = Context {
            system_prompt: Some(String::new()),
            messages: vec![Message::User(UserMessage {
                content: UserContentField::Blocks(vec![crate::types::UserContent::Text(
                    crate::types::TextContent::new("hi"),
                )]),
                timestamp: 0,
            })],
            tools: Some(Vec::new()),
        };
        let output = stream_simple(&model, &normalize(&context), &simple)
            .into_result()
            .await;

        assert_eq!(output.stop_reason, StopReason::Error);
        let message = output.error_message.expect("error message");
        assert!(message.contains("403"), "message: {message}");
        assert!(
            message.contains("blocked by gateway WAF"),
            "message: {message}"
        );
        assert!(!message.contains("Unknown: UnknownError"));
    }

    /// "bedrock preserves the SDK validation message when the response body is a stream".
    #[tokio::test]
    async fn bedrock_preserves_the_sdk_validation_message() {
        let _guard = env_guard();
        let model = builtin_model("global.anthropic.claude-opus-5");
        let (fetch, _recording) = status_fetch(
			400,
			&[("x-amzn-errortype", "ValidationException")],
			"{\"message\":\"Invocation of model ID anthropic.claude-opus-5 with on-demand throughput isn't supported. Retry with an inference profile.\"}",
		);
        let mut simple = SimpleStreamOptions::default();
        simple.base.base.env = Some(aws_test_env());
        simple.base.base.fetch = Some(fetch);

        let context = Context {
            system_prompt: Some(String::new()),
            messages: vec![Message::User(UserMessage {
                content: UserContentField::Blocks(vec![crate::types::UserContent::Text(
                    crate::types::TextContent::new("hi"),
                )]),
                timestamp: 0,
            })],
            tools: Some(Vec::new()),
        };
        let output = stream_simple(&model, &normalize(&context), &simple)
            .into_result()
            .await;

        assert_eq!(output.stop_reason, StopReason::Error);
        let message = output.error_message.expect("error message");
        assert!(
            message.contains("on-demand throughput isn't supported"),
            "message: {message}"
        );
        assert!(message.contains("inference profile"), "message: {message}");
        assert!(!message.contains("_readableState"));
    }
}
