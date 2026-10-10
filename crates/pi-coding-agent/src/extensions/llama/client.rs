//! Port of `packages/coding-agent/src/extensions/llama/client.ts` — the
//! llama.cpp router management client: model catalog listing, server props,
//! load/unload/download, SSE progress watching, and the URL/byte helpers.
//!
//! # Deviations (D97 range, shared with [`super`])
//!
//! - **D97** — cancellation uses `pi_ai::utils::abort::AbortSignal` (the
//!   crate-wide signal type) instead of the DOM `AbortSignal`; the 15-second
//!   per-request deadline of upstream's `AbortSignal.timeout(15_000)` becomes
//!   a `tokio::time::timeout` race, surfacing `"TimeoutError: the request
//!   timed out"` so [`super::is_connection_error`] treats it like upstream.
//! - **D97a** — `LlamaModelStatus` adds an `Unknown` fallback for status
//!   strings the enum does not know (upstream passes any string through;
//!   consumers only test for the known values).
//! - **D97b** — upstream's structural `isModelInfo` guard becomes typed
//!   `serde` parsing with the same two error messages ("llama.cpp returned an
//!   invalid model catalog" / "Server is not running in llama.cpp router
//!   mode").
//! - **D97c** — `formatBytes` prints non-integral byte counts with Rust's
//!   `Display` (upstream JS number formatting); server payloads only carry
//!   integral byte sizes in practice.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use pi_ai::utils::abort::AbortSignal;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// How long the model may take to answer a management request (upstream
/// `AbortSignal.timeout(15_000)` inside `request`).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Upstream `LlamaModelStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LlamaModelStatus {
    #[default]
    Unloaded,
    Loading,
    Loaded,
    Downloading,
    Sleeping,
    /// Unknown status string — see D97a.
    #[serde(other)]
    Unknown,
}

/// Upstream `LlamaModelInfo`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlamaModelInfo {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
    pub status: LlamaModelStatusInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<LlamaModelArchitecture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<LlamaModelMeta>,
}

/// Upstream `LlamaModelInfo.status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlamaModelStatusInfo {
    pub value: LlamaModelStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed: Option<bool>,
    /// Upstream `exit_code`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<std::collections::BTreeMap<String, LlamaProgressEntry>>,
}

/// Value of `status.progress` (`{ done: number; total: number }`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LlamaProgressEntry {
    pub done: f64,
    pub total: f64,
}

/// Upstream `LlamaModelInfo.architecture`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlamaModelArchitecture {
    /// Upstream `input_modalities`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_modalities: Option<Vec<String>>,
    /// Upstream `output_modalities`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<String>>,
}

/// Upstream `LlamaModelInfo.meta`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlamaModelMeta {
    /// Upstream `n_ctx`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u64>,
    /// Upstream `n_ctx_train`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx_train: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ftype: Option<f64>,
}

/// Upstream `LlamaModelsResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlamaModelsResponse {
    pub data: Vec<LlamaModelInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
}

/// Upstream `LlamaServerProps`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlamaServerProps {
    /// Upstream `models_autoload`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models_autoload: Option<bool>,
    /// Upstream `chat_template`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}

/// Upstream `LlamaModelEvent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaModelEvent {
    pub model: String,
    pub event: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

/// Upstream `LlamaProgress`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaProgress {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Upstream `errorMessage`: the payload's `error.message` when it is a
/// non-empty string.
fn json_error_message(payload: &JsonValue, fallback: &str) -> String {
    payload
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(|message| message.as_str())
        .filter(|message| !message.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

/// Human-readable cancel message for an aborted signal (upstream rejections
/// carry `signal.reason ?? new Error("Cancelled")`).
fn abort_message(signal: Option<&AbortSignal>) -> String {
    match signal.and_then(|signal| signal.reason()) {
        Some(reason) => reason.to_string(),
        None => "Cancelled".to_string(),
    }
}

/// Resolve when the signal aborts; never when `None`.
async fn abort_future(signal: Option<&AbortSignal>) {
    match signal {
        Some(signal) => {
            signal.aborted_future().await;
        }
        None => std::future::pending::<()>().await,
    }
}

/// Upstream `sleep(ms, signal)`.
async fn sleep_with_signal(duration: Duration, signal: Option<&AbortSignal>) -> Result<(), String> {
    if signal.is_some_and(|signal| signal.aborted()) {
        return Err(abort_message(signal));
    }
    tokio::select! {
        biased;
        _ = abort_future(signal) => Err(abort_message(signal)),
        _ = tokio::time::sleep(duration) => Ok(()),
    }
}

/// Run a request future under the optional abort signal and the 15-second
/// deadline (upstream `AbortSignal.any([signal, AbortSignal.timeout(15_000)])`
/// — D97).
async fn run_bounded<T>(
    future: impl Future<Output = T>,
    signal: Option<&AbortSignal>,
) -> Result<T, String> {
    tokio::select! {
        biased;
        _ = abort_future(signal) => Err(abort_message(signal)),
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

/// Upstream `parseLoadProgress`.
fn parse_load_progress(data: &JsonValue) -> Option<LlamaProgress> {
    let progress = data.get("progress")?.as_object()?;
    let stage = progress
        .get("current")
        .and_then(|value| value.as_str())
        .or_else(|| progress.get("stage").and_then(|value| value.as_str()));
    let stages: Vec<&str> = progress
        .get("stages")
        .and_then(|value| value.as_array())
        .map(|stages| stages.iter().filter_map(|entry| entry.as_str()).collect())
        .unwrap_or_default();
    let stage_ratio = progress
        .get("value")
        .and_then(|value| value.as_f64())
        .map(|value| value.clamp(0.0, 1.0));
    let mut ratio = stage_ratio;
    if let Some(stage) = stage {
        if !stages.is_empty() {
            if let Some(index) = stages.iter().position(|entry| *entry == stage) {
                ratio = Some((index as f64 + stage_ratio.unwrap_or(0.0)) / stages.len() as f64);
            }
        }
    }
    Some(LlamaProgress {
        message: stage
            .map(|stage| format!("Loading {}", stage.replace('_', " ")))
            .unwrap_or_else(|| "Loading model".to_string()),
        ratio,
        detail: None,
    })
}

/// Upstream `parseDownloadProgress`.
fn parse_download_progress(data: &JsonValue) -> Option<LlamaProgress> {
    let nested = data.get("progress");
    let files = match nested {
        Some(value) if value.is_object() => value,
        _ => data,
    };
    let mut done = 0.0;
    let mut total = 0.0;
    for (_, value) in files.as_object()?.iter() {
        let Some(entry_done) = value.get("done").and_then(|value| value.as_f64()) else {
            continue;
        };
        let Some(entry_total) = value.get("total").and_then(|value| value.as_f64()) else {
            continue;
        };
        done += entry_done;
        total += entry_total;
    }
    download_progress(done, total)
}

/// Shared tail of the download-progress parsers.
fn download_progress(done: f64, total: f64) -> Option<LlamaProgress> {
    if total <= 0.0 {
        return None;
    }
    Some(LlamaProgress {
        message: "Downloading model".to_string(),
        ratio: Some(done / total),
        detail: Some(format!("{} / {}", format_bytes(done), format_bytes(total))),
    })
}

/// Upstream `formatBytes` (D97c for non-integral input).
pub fn format_bytes(bytes: f64) -> String {
    if bytes < 1024.0 {
        return format!("{} B", format_plain_number(bytes));
    }
    let units = ["KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes / 1024.0;
    let mut unit = units[0];
    let mut index = 1;
    while index < units.len() && value >= 1024.0 {
        value /= 1024.0;
        unit = units[index];
        index += 1;
    }
    let rendered = if value >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    };
    format!("{rendered} {unit}")
}

/// JS `String(number)` for the values that reach byte formatting.
fn format_plain_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Upstream `normalizeLlamaServerUrl` (D97: invalid URLs report
/// `"Invalid URL"` where JS throws a TypeError).
pub fn normalize_llama_server_url(value: &str) -> Result<String, String> {
    let mut url = url::Url::parse(value.trim()).map_err(|_| "Invalid URL".to_string())?;
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err("Server URL must use http or https".to_string());
    }
    url.set_fragment(None);
    url.set_query(None);
    let mut path = url.path().trim_end_matches('/').to_string();
    if path.ends_with("/v1") {
        path.truncate(path.len() - "/v1".len());
    }
    if path.is_empty() {
        path.push('/');
    }
    url.set_path(&path);
    let mut normalized = url.to_string();
    while normalized.ends_with('/') {
        normalized.pop();
    }
    Ok(normalized)
}

/// Upstream `llamaInferenceUrl`.
pub fn llama_inference_url(server_url: &str) -> Result<String, String> {
    Ok(format!("{}/v1", normalize_llama_server_url(server_url)?))
}

/// Infallible inference URL for inputs that are already normalized or
/// derived from [`normalize_llama_server_url`] output (internal; D98).
pub(crate) fn inference_url_of(server_url: &str) -> String {
    match normalize_llama_server_url(server_url) {
        Ok(normalized) => format!("{normalized}/v1"),
        Err(_) => format!("{}/v1", server_url.trim().trim_end_matches('/')),
    }
}

/// Client for the llama.cpp router management API (upstream `LlamaClient`).
#[derive(Clone)]
pub struct LlamaClient {
    /// Normalized management URL (upstream `readonly serverUrl`).
    pub server_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

impl std::fmt::Debug for LlamaClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlamaClient")
            .field("server_url", &self.server_url)
            .finish_non_exhaustive()
    }
}

impl LlamaClient {
    /// Upstream `new LlamaClient(serverUrl, apiKey?)`; normalizes the URL and
    /// fails with the [`normalize_llama_server_url`] error.
    pub fn new(server_url: &str, api_key: Option<&str>) -> Result<Self, String> {
        Ok(LlamaClient {
            server_url: normalize_llama_server_url(server_url)?,
            api_key: api_key.map(str::to_string),
            http: reqwest::Client::new(),
        })
    }

    fn authorization(&self) -> Option<String> {
        self.api_key.as_ref().map(|key| format!("Bearer {key}"))
    }

    /// Upstream `request`: JSON in, JSON out, errors as the payload's
    /// `error.message` or the fallback.
    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&JsonValue>,
        signal: Option<&AbortSignal>,
    ) -> Result<JsonValue, String> {
        let mut request = self
            .http
            .request(method, format!("{}{}", self.server_url, path));
        if let Some(authorization) = self.authorization() {
            request = request.header(reqwest::header::AUTHORIZATION, authorization);
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = run_bounded_ok(request.send(), signal).await?;
        let status = response.status();
        // Upstream tolerates non-JSON bodies (`payload = undefined`).
        let payload = run_bounded_ok(response.json::<JsonValue>(), signal)
            .await
            .unwrap_or(JsonValue::Null);
        if !status.is_success() {
            return Err(json_error_message(
                &payload,
                &format!("llama.cpp returned HTTP {}", status.as_u16()),
            ));
        }
        Ok(payload)
    }

    /// Upstream `list`.
    pub async fn list(
        &self,
        reload: bool,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<LlamaModelInfo>, String> {
        let path = if reload {
            "/models?reload=1"
        } else {
            "/models"
        };
        let payload = self
            .request(reqwest::Method::GET, path, None, signal)
            .await?;
        let Some(data) = payload.get("data").and_then(|data| data.as_array()) else {
            return Err("llama.cpp returned an invalid model catalog".to_string());
        };
        data.iter()
            .map(|entry| {
                let is_model_info = entry.get("id").and_then(|value| value.as_str()).is_some()
                    && entry
                        .get("status")
                        .and_then(|status| status.get("value"))
                        .and_then(|value| value.as_str())
                        .is_some();
                if !is_model_info {
                    return Err("Server is not running in llama.cpp router mode".to_string());
                }
                serde_json::from_value::<LlamaModelInfo>(entry.clone())
                    .map_err(|_| "Server is not running in llama.cpp router mode".to_string())
            })
            .collect()
    }

    /// Upstream `props`.
    pub async fn props(
        &self,
        model: Option<&str>,
        signal: Option<&AbortSignal>,
    ) -> Result<LlamaServerProps, String> {
        let path = match model {
            Some(model) => format!(
                "/props?{}",
                url::form_urlencoded::Serializer::new(String::new())
                    .append_pair("model", model)
                    .append_pair("autoload", "false")
                    .finish()
            ),
            None => "/props".to_string(),
        };
        let payload = self
            .request(reqwest::Method::GET, &path, None, signal)
            .await?;
        Ok(LlamaServerProps {
            models_autoload: payload
                .get("models_autoload")
                .and_then(|value| value.as_bool()),
            chat_template: payload
                .get("chat_template")
                .and_then(|value| value.as_str())
                .map(str::to_string),
        })
    }

    /// Upstream `load`.
    pub async fn load(&self, model: &str, signal: Option<&AbortSignal>) -> Result<(), String> {
        self.request(
            reqwest::Method::POST,
            "/models/load",
            Some(&serde_json::json!({ "model": model })),
            signal,
        )
        .await
        .map(|_| ())
    }

    /// Upstream `unload`.
    pub async fn unload(&self, model: &str, signal: Option<&AbortSignal>) -> Result<(), String> {
        self.request(
            reqwest::Method::POST,
            "/models/unload",
            Some(&serde_json::json!({ "model": model })),
            signal,
        )
        .await
        .map(|_| ())
    }

    /// Upstream `unloadAndWait`.
    pub async fn unload_and_wait(
        &self,
        model: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<(), String> {
        self.unload(model, signal).await?;
        loop {
            let models = self.list(false, signal).await?;
            match models.iter().find(|candidate| candidate.id == model) {
                None => return Ok(()),
                Some(entry) if entry.status.value == LlamaModelStatus::Unloaded => return Ok(()),
                Some(_) => {}
            }
            sleep_with_signal(Duration::from_millis(100), signal).await?;
        }
    }

    /// Upstream `download`.
    pub async fn download(&self, model: &str, signal: Option<&AbortSignal>) -> Result<(), String> {
        self.request(
            reqwest::Method::POST,
            "/models",
            Some(&serde_json::json!({ "model": model })),
            signal,
        )
        .await
        .map(|_| ())
    }

    /// Upstream `watch`: consume `/models/sse` frames, invoking `on_event`
    /// for well-formed events.
    pub async fn watch(
        &self,
        on_event: Arc<dyn Fn(LlamaModelEvent) + Send + Sync>,
        signal: Option<&AbortSignal>,
    ) -> Result<(), String> {
        let mut request = self.http.get(format!("{}/models/sse", self.server_url));
        if let Some(authorization) = self.authorization() {
            request = request.header(reqwest::header::AUTHORIZATION, authorization);
        }
        // No 15s deadline here: the stream lives until the watcher aborts.
        let response = {
            tokio::select! {
                biased;
                _ = abort_future(signal) => return Err(abort_message(signal)),
                response = request.send() => response.map_err(|error| error.to_string())?,
            }
        };
        if !response.status().is_success() {
            return Err(format!(
                "llama.cpp SSE returned HTTP {}",
                response.status().as_u16()
            ));
        }
        let mut response = response;
        let mut buffer = String::new();
        loop {
            let chunk = tokio::select! {
                biased;
                _ = abort_future(signal) => return Err(abort_message(signal)),
                chunk = response.chunk() => chunk.map_err(|error| error.to_string())?,
            };
            let Some(bytes) = chunk else {
                break;
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            buffer = buffer.replace("\r\n", "\n");
            while let Some(boundary) = buffer.find("\n\n") {
                let frame: String = buffer.drain(..boundary + 2).collect();
                let data = frame[..boundary]
                    .split('\n')
                    .filter(|line| line.starts_with("data:"))
                    .map(|line| line["data:".len()..].trim_start())
                    .collect::<Vec<_>>()
                    .join("\n");
                if data.is_empty() {
                    continue;
                }
                // Ignore malformed events; catalog polling remains authoritative.
                if let Ok(event) = serde_json::from_str::<LlamaModelEvent>(&data) {
                    on_event(event);
                }
            }
        }
        Ok(())
    }

    /// Upstream `loadAndWait`: load the model, stream SSE progress, and poll
    /// until the catalog reports it loaded (or the events do, or it fails).
    pub async fn load_and_wait(
        &self,
        model: &str,
        on_progress: Arc<dyn Fn(LlamaProgress) + Send + Sync>,
        signal: Option<&AbortSignal>,
    ) -> Result<LlamaModelInfo, String> {
        struct WatchState {
            event_loaded: bool,
            event_error: Option<String>,
        }
        let watcher = pi_ai::utils::abort::AbortController::new();
        // Upstream linkSignal: the watcher dies with the outer signal.
        let link_task = signal.map(|signal| {
            let watcher = watcher.clone();
            let signal = signal.clone();
            tokio::spawn(async move {
                signal.aborted_future().await;
                watcher.abort();
            })
        });
        let state = Arc::new(parking_lot::Mutex::new(WatchState {
            event_loaded: false,
            event_error: None,
        }));
        let watch_signal = watcher.signal().clone();
        let watch_client = self.clone();
        let watch_model = model.to_string();
        let handler_state = Arc::clone(&state);
        let handler_progress = Arc::clone(&on_progress);
        let watch_task = tokio::spawn(async move {
            let handler = move |event: LlamaModelEvent| {
                if event.model != watch_model {
                    return;
                }
                if event.event != "model_status" && event.event != "status_change" {
                    return;
                }
                let status = event
                    .data
                    .as_ref()
                    .and_then(|data| data.get("status"))
                    .and_then(|status| status.as_str());
                {
                    let mut guard = handler_state.lock();
                    if status == Some("loaded") {
                        guard.event_loaded = true;
                    }
                    if status == Some("unloaded") {
                        guard.event_error = Some("Model failed to load".to_string());
                    }
                }
                if let Some(progress) = event.data.as_ref().and_then(parse_load_progress) {
                    handler_progress(progress);
                }
            };
            let _ = watch_client
                .watch(Arc::new(handler), Some(&watch_signal))
                .await;
        });
        let result = async {
            self.load(model, signal).await?;
            on_progress(LlamaProgress {
                message: "Loading model".to_string(),
                ratio: None,
                detail: None,
            });
            loop {
                if let Some(signal) = signal {
                    signal.check().map_err(|_| abort_message(Some(signal)))?;
                }
                let models = self.list(false, signal).await?;
                let entry = models.iter().find(|candidate| candidate.id == model);
                if entry.map(|entry| entry.status.value) == Some(LlamaModelStatus::Loaded) {
                    return Ok(entry.cloned().unwrap());
                }
                let (event_loaded, event_error) = {
                    let guard = state.lock();
                    (guard.event_loaded, guard.event_error.clone())
                };
                if event_loaded && entry.is_none() {
                    return Ok(LlamaModelInfo {
                        id: model.to_string(),
                        status: LlamaModelStatusInfo {
                            value: LlamaModelStatus::Loaded,
                            ..Default::default()
                        },
                        ..Default::default()
                    });
                }
                if entry.map(|entry| entry.status.failed) == Some(Some(true))
                    || event_error.is_some()
                {
                    let exit_code = entry.and_then(|entry| entry.status.exit_code);
                    return Err(match exit_code {
                        None => event_error.unwrap_or_else(|| "Model failed to load".to_string()),
                        Some(code) => format!("Model exited with code {code}"),
                    });
                }
                sleep_with_signal(Duration::from_millis(250), signal).await?;
            }
        }
        .await;
        // Upstream `finally`: unlink and stop the watcher.
        watcher.abort();
        if let Some(link_task) = link_task {
            link_task.abort();
        }
        let _ = watch_task.await;
        result
    }

    /// Upstream `downloadAndWait`: download the model with byte progress and
    /// return the refreshed (reload) catalog.
    pub async fn download_and_wait(
        &self,
        model: &str,
        on_progress: Arc<dyn Fn(LlamaProgress) + Send + Sync>,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<LlamaModelInfo>, String> {
        struct WatchState {
            finished: bool,
            failure: Option<String>,
            saw_downloading: bool,
        }
        let watcher = pi_ai::utils::abort::AbortController::new();
        let link_task = signal.map(|signal| {
            let watcher = watcher.clone();
            let signal = signal.clone();
            tokio::spawn(async move {
                signal.aborted_future().await;
                watcher.abort();
            })
        });
        let state = Arc::new(parking_lot::Mutex::new(WatchState {
            finished: false,
            failure: None,
            saw_downloading: false,
        }));
        let watch_signal = watcher.signal().clone();
        let watch_client = self.clone();
        let watch_model = model.to_string();
        let handler_state = Arc::clone(&state);
        let handler_progress = Arc::clone(&on_progress);
        let watch_task = tokio::spawn(async move {
            let handler = move |event: LlamaModelEvent| {
                if event.model != watch_model {
                    return;
                }
                let mut guard = handler_state.lock();
                if event.event == "download_finished" {
                    guard.finished = true;
                }
                if event.event == "download_failed" {
                    guard.failure = Some(
                        event
                            .data
                            .as_ref()
                            .map(|data| json_error_message(data, "Download failed"))
                            .unwrap_or_else(|| "Download failed".to_string()),
                    );
                }
                if event.event == "download_progress" {
                    guard.saw_downloading = true;
                    if let Some(progress) = event.data.as_ref().and_then(parse_download_progress) {
                        // Progress callbacks fire outside the state lock.
                        drop(guard);
                        handler_progress(progress);
                    }
                }
            };
            let _ = watch_client
                .watch(Arc::new(handler), Some(&watch_signal))
                .await;
        });
        let result = async {
            self.download(model, signal).await?;
            on_progress(LlamaProgress {
                message: "Downloading model".to_string(),
                ratio: None,
                detail: None,
            });
            let mut polls = 0;
            loop {
                if let Some(signal) = signal {
                    signal.check().map_err(|_| abort_message(Some(signal)))?;
                }
                let (finished, failure, saw_downloading) = {
                    let guard = state.lock();
                    (guard.finished, guard.failure.clone(), guard.saw_downloading)
                };
                if let Some(failure) = failure {
                    return Err(failure);
                }
                let models = self.list(false, signal).await?;
                polls += 1;
                let entry = models.iter().find(|candidate| candidate.id == model);
                if entry.map(|entry| entry.status.value) == Some(LlamaModelStatus::Downloading) {
                    {
                        let mut guard = state.lock();
                        guard.saw_downloading = true;
                    }
                    if let Some(progress) = entry
                        .and_then(|entry| entry.status.progress.as_ref())
                        .and_then(|entries| {
                            let done: f64 = entries.values().map(|value| value.done).sum();
                            let total: f64 = entries.values().map(|value| value.total).sum();
                            download_progress(done, total)
                        })
                    {
                        on_progress(progress);
                    }
                } else if finished || entry.is_some() && (saw_downloading || polls >= 2) {
                    return self.list(true, signal).await;
                }
                sleep_with_signal(Duration::from_millis(500), signal).await?;
            }
        }
        .await;
        watcher.abort();
        if let Some(link_task) = link_task {
            link_task.abort();
        }
        let _ = watch_task.await;
        result
    }
}

// ============================================================================
// Tests — port of the client cases from `test/llama-extension.test.ts`.
// ============================================================================

#[cfg(test)]
mod client_tests {
    use super::*;
    use crate::extensions::llama::test_support::{
        listen, Handler, SseHub, TestRequest, TestResponse,
    };
    use pi_agent::BoxFuture;
    use serde_json::json;
    use std::sync::Arc;

    fn handler(
        function: impl Fn(TestRequest) -> BoxFuture<'static, TestResponse> + Send + Sync + 'static,
    ) -> Handler {
        Arc::new(function)
    }

    #[test]
    fn normalizes_management_and_inference_urls() {
        assert_eq!(
            normalize_llama_server_url("http://127.0.0.1:8080/v1/").unwrap(),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            normalize_llama_server_url("https://example.com/prefix/v1").unwrap(),
            "https://example.com/prefix"
        );
        let error = normalize_llama_server_url("file:///tmp/llama").unwrap_err();
        assert!(error.contains("http or https"), "unexpected error: {error}");
        assert_eq!(
            llama_inference_url("http://127.0.0.1:8080").unwrap(),
            "http://127.0.0.1:8080/v1"
        );
    }

    #[tokio::test]
    async fn loads_with_sse_progress_and_waits_for_the_loaded_catalog_state() {
        let status: Arc<parking_lot::Mutex<&'static str>> =
            Arc::new(parking_lot::Mutex::new("unloaded"));
        let hub = SseHub::new();
        let hub_for_handler = Arc::clone(&hub);
        let server = listen(handler(move |request| {
			let status = Arc::clone(&status);
			let hub = Arc::clone(&hub_for_handler);
			Box::pin(async move {
				match (request.method.as_str(), request.target.as_str()) {
					("GET", "/models/sse") => TestResponse::Sse(hub),
					("POST", "/models/load") => {
						*status.lock() = "loading";
						let status_clone = Arc::clone(&status);
						let hub_clone = Arc::clone(&hub);
						tokio::spawn(async move {
							tokio::time::sleep(Duration::from_millis(20)).await;
							hub_clone.send(&json!({
								"model": "test-model",
								"event": "status_change",
								"data": {
									"status": "loading",
									"progress": { "stages": ["text_model", "mmproj_model"], "current": "text_model", "value": 0.5 },
								},
							}));
							*status_clone.lock() = "loaded";
							hub_clone.send(&json!({ "model": "test-model", "event": "status_change", "data": { "status": "loaded" } }));
						});
						TestResponse::Json(200, json!({ "success": true }))
					}
					("GET", "/models") => {
						let value = *status.lock();
						TestResponse::Json(200, json!({ "data": [{ "id": "test-model", "status": { "value": value } }] }))
					}
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

        let progress: Arc<parking_lot::Mutex<Vec<String>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let progress_sink = Arc::clone(&progress);
        let model = LlamaClient::new(&server.url, None)
            .unwrap()
            .load_and_wait(
                "test-model",
                Arc::new(move |entry: LlamaProgress| progress_sink.lock().push(entry.message)),
                None,
            )
            .await
            .expect("load completes");
        assert_eq!(model.status.value, LlamaModelStatus::Loaded);
        assert!(progress
            .lock()
            .iter()
            .any(|message| message == "Loading text model"));
        hub.close_all();
    }

    #[tokio::test]
    async fn downloads_with_byte_progress_and_returns_the_refreshed_catalog() {
        let status: Arc<parking_lot::Mutex<&'static str>> =
            Arc::new(parking_lot::Mutex::new("missing"));
        let hub = SseHub::new();
        let hub_for_handler = Arc::clone(&hub);
        let status_for_handler = Arc::clone(&status);
        let server = listen(handler(move |request| {
			let status = Arc::clone(&status_for_handler);
			let hub = Arc::clone(&hub_for_handler);
			Box::pin(async move {
				match (request.method.as_str(), request.target.as_str()) {
					("GET", "/models/sse") => TestResponse::Sse(hub),
					("POST", "/models") => {
						*status.lock() = "downloading";
						let status_clone = Arc::clone(&status);
						let hub_clone = Arc::clone(&hub);
						tokio::spawn(async move {
							tokio::time::sleep(Duration::from_millis(20)).await;
							hub_clone.send(&json!({
								"model": "owner/repo:Q4_K_M",
								"event": "download_progress",
								"data": { "progress": { "https://example/model.gguf": { "done": 512, "total": 1024 } } },
							}));
							*status_clone.lock() = "unloaded";
							hub_clone.send(&json!({ "model": "owner/repo:Q4_K_M", "event": "download_finished", "data": {} }));
						});
						TestResponse::Json(200, json!({ "success": true }))
					}
					("GET", target) if target.starts_with("/models") => {
						let value = *status.lock();
						let data = if value == "missing" {
							json!([])
						} else {
							json!([{ "id": "owner/repo:Q4_K_M", "status": { "value": value } }])
						};
						TestResponse::Json(200, json!({ "data": data }))
					}
					_ => TestResponse::Empty(404),
				}
			})
		}))
		.await;

        let progress: Arc<parking_lot::Mutex<Vec<LlamaProgress>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let progress_sink = Arc::clone(&progress);
        let models = LlamaClient::new(&server.url, None)
            .unwrap()
            .download_and_wait(
                "owner/repo:Q4_K_M",
                Arc::new(move |entry: LlamaProgress| progress_sink.lock().push(entry)),
                None,
            )
            .await
            .expect("download completes");
        assert_eq!(
            serde_json::to_value(&models).unwrap(),
            json!([{ "id": "owner/repo:Q4_K_M", "status": { "value": "unloaded" } }])
        );
        assert!(progress.lock().contains(&LlamaProgress {
            message: "Downloading model".to_string(),
            ratio: Some(0.5),
            detail: Some("512 B / 1.00 KiB".to_string()),
        }));
        hub.close_all();
    }

    #[test]
    fn format_bytes_matches_upstream_rendering() {
        assert_eq!(format_bytes(512.0), "512 B");
        assert_eq!(format_bytes(1024.0), "1.00 KiB");
        assert_eq!(format_bytes(1536.0), "1.50 KiB");
        assert_eq!(format_bytes(1024.0 * 1024.0 * 10.0), "10.0 MiB");
    }
}
