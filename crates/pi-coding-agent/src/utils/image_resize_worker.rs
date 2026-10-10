//! Port of `packages/coding-agent/src/utils/image-resize-worker.ts` — the
//! image resize worker entry point.
//!
//! Upstream is a Node `worker_threads` entry that receives one
//! `{ inputBytes, mimeType, options }` message and posts back
//! `{ result }`/`{ error }`. The Rust port (crate docs, deviation D15) runs
//! the same request/response contract on `tokio::task::spawn_blocking`
//! instead of a subprocess worker: [`handle_resize_worker_request`] validates
//! a deserialized request and produces the response, and
//! [`run_resize_worker`] is the spawn_blocking body. The message shapes are
//! serde-compatible with upstream's structured-clone payloads.

use serde::{Deserialize, Serialize};

use super::image_resize_core::{resize_image_in_process, ImageResizeOptions, ResizedImage};

/// Upstream `ResizeImageWorkerRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResizeImageWorkerRequest {
    pub input_bytes: Vec<u8>,
    pub mime_type: String,
    pub options: Option<ImageResizeOptions>,
}

/// Upstream `ResizeImageWorkerResponse` — `result: null` and `error` both map
/// to `Option` fields; an `error` string is fatal. Replies carry the
/// [`super::image_resize_core::IMAGE_RESIZE_WORKER_RESPONSE_TYPE`] tag so
/// foreign channel messages are recognizable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResizeImageWorkerResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub result: Option<ResizedImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ResizeImageWorkerResponse {
    fn new(result: Option<ResizedImage>, error: Option<String>) -> Self {
        ResizeImageWorkerResponse {
            response_type: super::image_resize_core::IMAGE_RESIZE_WORKER_RESPONSE_TYPE.to_string(),
            result,
            error,
        }
    }
}

/// `isResizeImageWorkerRequest` — the serde payload must carry image bytes and
/// a mime type.
pub fn is_resize_image_worker_request(value: &serde_json::Value) -> bool {
    serde_json::from_value::<ResizeImageWorkerRequest>(value.clone()).is_ok()
}

/// Worker message loop body (`port.once("message", ...)`): run one resize
/// request, mapping failure to the `{ error }` response.
pub async fn handle_resize_worker_request(
    request: ResizeImageWorkerRequest,
) -> ResizeImageWorkerResponse {
    match resize_image_in_process(
        &request.input_bytes,
        &request.mime_type,
        request.options.as_ref(),
    )
    .await
    {
        Some(result) => ResizeImageWorkerResponse::new(Some(result), None),
        None => ResizeImageWorkerResponse::new(None, None),
    }
}

/// Run the worker body on the blocking thread pool (`resizeImageInWorker`'s
/// worker side). Never panics on failure — errors surface as the response's
/// `error` field, like upstream's catch path.
pub async fn run_resize_worker(payload: serde_json::Value) -> serde_json::Value {
    let request: Result<ResizeImageWorkerRequest, _> = serde_json::from_value(payload);
    match request {
        Err(_) => {
            serde_json::json!({ "type": super::image_resize_core::IMAGE_RESIZE_WORKER_RESPONSE_TYPE, "error": "Invalid image resize worker request" })
        }
        Ok(request) => {
            let response = tokio::task::spawn_blocking(move || {
                tokio::runtime::Handle::current().block_on(handle_resize_worker_request(request))
            })
            .await;
            match response {
                Ok(response) => serde_json::to_value(response).unwrap_or_default(),
                Err(error) => {
                    serde_json::json!({ "type": super::image_resize_core::IMAGE_RESIZE_WORKER_RESPONSE_TYPE, "error": error.to_string() })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_requests() {
        assert!(!is_resize_image_worker_request(
            &serde_json::json!({ "mimeType": "image/png" })
        ));
        assert!(!is_resize_image_worker_request(
            &serde_json::json!({ "inputBytes": [] })
        ));
        assert!(!is_resize_image_worker_request(&serde_json::json!("nope")));
    }

    #[test]
    fn accepts_well_formed_requests() {
        assert!(is_resize_image_worker_request(&serde_json::json!({
            "inputBytes": [1, 2, 3],
            "mimeType": "image/png",
            "options": { "maxWidth": 100 }
        })));
    }

    #[tokio::test]
    async fn invalid_payload_yields_error_response() {
        let response = run_resize_worker(serde_json::json!({ "nope": true })).await;
        assert_eq!(response["error"], "Invalid image resize worker request");
    }

    #[tokio::test]
    async fn valid_payload_without_engine_yields_null_result() {
        let response =
            run_resize_worker(serde_json::json!({ "inputBytes": [], "mimeType": "image/png" }))
                .await;
        assert!(response.get("error").is_none());
        assert!(response["result"].is_null());
    }
}
