//! Port of `packages/coding-agent/src/core/bug-report-upload.ts` — upload a
//! bug report bundle to the Radius gateway as multipart form data, either
//! anonymously or attributed to a Radius account.
//!
//! Deviation (crate docs):
//! - **D128 — status-text fallback.** Status-text fallbacks use the
//!   standard reason phrase (`status.canonical_reason`), so unknown codes
//!   fall through to the numeric status where JS would echo the wire reason.
//!   (The Radius gateway resolution upstream imports from `core/radius.ts`
//!   is taken from [`crate::core::radius`], as upstream does.)
//!
//! Cancellation follows the crate-wide [`AbortSignal`] convention (cf.
//! `extensions::llama`): an aborted signal returns `Err` with the abort
//! reason instead of an upstream `AbortError`.

use std::future::Future;

use serde_json::Value;
use url::Url;

use pi_ai::utils::abort::AbortSignal;

use crate::core::bug_report::{bug_report_files, BugReportBundle};
pub use crate::core::radius::{get_radius_gateway_url, ENV_RADIUS_GATEWAY};

/// Upstream `UploadBugReportOptions`.
#[derive(Debug, Clone, Default)]
pub struct UploadBugReportOptions {
    pub token: Option<String>,
    pub signal: Option<AbortSignal>,
    pub gateway_url: Option<String>,
}

/// Upstream `{ id: string }` return shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadedBugReport {
    pub id: String,
}

/// Abort-aware future race without an overall deadline (upstream `fetch`
/// carries the signal; there is no timeout).
async fn raced<T, E>(
    future: impl Future<Output = Result<T, E>>,
    signal: Option<&AbortSignal>,
) -> Result<T, String>
where
    E: std::fmt::Display,
{
    match signal {
        Some(signal) => {
            tokio::select! {
                biased;
                _ = signal.aborted_future() => Err(signal
                    .reason()
                    .map(|reason| reason.to_string())
                    .unwrap_or_else(|| "Cancelled".to_string())),
                output = future => output.map_err(|error| error.to_string()),
            }
        }
        None => future.await.map_err(|error| error.to_string()),
    }
}

/// Upload a report as multipart form data, anonymously or attributed to a
/// Radius account (`uploadBugReport`). Success requires
/// `response.ok && json.ok === true && typeof json.bug_report?.id === "string"`;
/// every other outcome composes the upstream error message.
pub async fn upload_bug_report(
    bundle: &BugReportBundle,
    options: UploadBugReportOptions,
) -> Result<UploadedBugReport, String> {
    let gateway = options.gateway_url.unwrap_or_else(get_radius_gateway_url);
    let url = Url::parse(&gateway)
        .and_then(|base| base.join("/v1/bug-reports"))
        .map_err(|error| error.to_string())?;
    let mut form = reqwest::multipart::Form::new();
    for file in bug_report_files(bundle) {
        let part = reqwest::multipart::Part::text(file.data)
            .file_name(file.name.clone())
            .mime_str(&file.content_type)
            .map_err(|error| error.to_string())?;
        form = form.part(file.name, part);
    }
    let mut request = reqwest::Client::new().post(url).multipart(form);
    if let Some(token) = &options.token {
        request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = raced(request.send(), options.signal.as_ref()).await?;
    let status = response.status();
    // `response.statusText` is the reason phrase; HTTP/2 responses carry none.
    let status_text = status.canonical_reason().unwrap_or_default().to_string();
    // `await response.json().catch(() => null)`
    let json: Option<Value> = raced(response.json::<Value>(), options.signal.as_ref())
        .await
        .ok();
    let bug_report_id = json
        .as_ref()
        .and_then(|json| json.get("bug_report"))
        .and_then(|bug_report| bug_report.get("id"))
        .and_then(Value::as_str);
    if status.is_success()
        && json
            .as_ref()
            .and_then(|json| json.get("ok"))
            .and_then(Value::as_bool)
            == Some(true)
        && bug_report_id.is_some()
    {
        return Ok(UploadedBugReport {
            id: bug_report_id.expect("checked above").to_string(),
        });
    }
    // `json && !json.ok ? json.description || json.error : undefined`
    let detail = match &json {
        Some(json) if json.get("ok").and_then(Value::as_bool) != Some(true) => json
            .get("description")
            .and_then(Value::as_str)
            .or_else(|| json.get("error").and_then(Value::as_str))
            .map(str::to_string),
        _ => None,
    };
    // `Bug report upload failed: ${detail || response.statusText || response.status}`
    let fallback = if status_text.is_empty() {
        status.as_u16().to_string()
    } else {
        status_text
    };
    Err(format!(
        "Bug report upload failed: {}",
        detail
            .filter(|detail| !detail.is_empty())
            .unwrap_or(fallback)
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use pi_ai::utils::abort::AbortController;
    use serde_json::json;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    use super::*;
    use crate::core::bug_report::sample_bug_report_bundle;

    type Captured = Arc<Mutex<Vec<(Option<String>, String)>>>;
    type Script = Arc<Mutex<VecDeque<(u16, String)>>>;

    /// Ephemeral HTTP server responding with scripted `(status, body)` pairs
    /// per request and capturing `(authorization, body)`.
    async fn serve(script: Script, captured: Captured) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let captured = Arc::clone(&captured);
                let script = Arc::clone(&script);
                tokio::spawn(async move {
                    let mut buffer: Vec<u8> = Vec::new();
                    let mut chunk = [0u8; 8192];
                    let head_end = loop {
                        if let Some(at) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
                        {
                            break at;
                        }
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                        }
                    };
                    let head = String::from_utf8_lossy(&buffer[..head_end]).to_string();
                    let head_lower = head.to_ascii_lowercase();
                    let mut content_length = 0usize;
                    let mut authorization: Option<String> = None;
                    for line in head.split("\r\n").skip(1) {
                        if let Some((name, value)) = line.split_once(':') {
                            let name = name.trim().to_ascii_lowercase();
                            if name == "content-length" {
                                content_length = value.trim().parse().unwrap_or(0);
                            } else if name == "authorization" {
                                authorization = Some(value.trim().to_string());
                            }
                        }
                    }
                    let body_start = head_end + 4;
                    // reqwest streams multipart bodies with chunked encoding.
                    let body: Vec<u8> = if head_lower.contains("transfer-encoding: chunked") {
                        let mut body = Vec::new();
                        let mut pos = body_start;
                        loop {
                            let line_end = loop {
                                if let Some(at) = buffer[pos..]
                                    .windows(2)
                                    .position(|window| window == b"\r\n")
                                {
                                    break pos + at;
                                }
                                match stream.read(&mut chunk).await {
                                    Ok(0) | Err(_) => return,
                                    Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                                }
                            };
                            let size = usize::from_str_radix(
                                String::from_utf8_lossy(&buffer[pos..line_end]).trim(),
                                16,
                            )
                            .unwrap_or(0);
                            pos = line_end + 2;
                            if size == 0 {
                                break;
                            }
                            while buffer.len() < pos + size {
                                match stream.read(&mut chunk).await {
                                    Ok(0) | Err(_) => return,
                                    Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                                }
                            }
                            body.extend_from_slice(&buffer[pos..pos + size]);
                            pos += size + 2;
                        }
                        body
                    } else {
                        while buffer.len() < body_start + content_length {
                            match stream.read(&mut chunk).await {
                                Ok(0) | Err(_) => return,
                                Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                            }
                        }
                        buffer[body_start..body_start + content_length].to_vec()
                    };
                    captured
                        .lock()
                        .push((authorization, String::from_utf8_lossy(&body).into_owned()));
                    let (status, payload) =
                        script.lock().pop_front().unwrap_or((500, "{}".to_string()));
                    let reason = match status {
                        200 => "OK",
                        500 => "Internal Server Error",
                        _ => "Custom",
                    };
                    let response = format!(
						"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
						payload.len()
					);
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn uploads_files_as_multipart_and_returns_id() {
        let script: Script = Arc::new(Mutex::new(VecDeque::from(vec![(
            200,
            json!({ "ok": true, "bug_report": { "id": "br_123" } }).to_string(),
        )])));
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let gateway = serve(Arc::clone(&script), Arc::clone(&captured)).await;

        let result = upload_bug_report(
            &sample_bug_report_bundle(),
            UploadBugReportOptions {
                token: Some("tok".to_string()),
                signal: None,
                gateway_url: Some(gateway),
            },
        )
        .await;

        assert_eq!(
            result,
            Ok(UploadedBugReport {
                id: "br_123".to_string()
            })
        );
        let (authorization, body) = &captured.lock()[0];
        assert_eq!(authorization.as_deref(), Some("Bearer tok"));
        for expected in [
            "report.json",
            "diagnostics.json",
            "session.jsonl",
            "summary.md",
        ] {
            assert!(
                body.contains(&format!("name=\"{expected}\"")),
                "missing part {expected}: {body}"
            );
            assert!(
                body.contains(&format!("filename=\"{expected}\"")),
                "missing filename {expected}"
            );
        }
        assert!(
            body.contains("content-type: application/json")
                || body.contains("Content-Type: application/json"),
            "missing json part type: {body}"
        );
        assert!(
            body.contains("content-type: application/x-ndjson")
                || body.contains("Content-Type: application/x-ndjson"),
            "missing ndjson part type: {body}"
        );
        assert!(
            body.contains("content-type: text/markdown")
                || body.contains("Content-Type: text/markdown"),
            "missing markdown part type: {body}"
        );
        assert!(
            body.contains("{\"type\":\"message\"}"),
            "missing session content: {body}"
        );
    }

    #[tokio::test]
    async fn composes_error_message_exactly() {
        let script: Script = Arc::new(Mutex::new(VecDeque::from(vec![
            (
                200,
                json!({ "ok": false, "description": "quota exceeded" }).to_string(),
            ),
            (200, json!({ "error": "boom" }).to_string()),
            (500, "not json".to_string()),
            (599, "{}".to_string()),
        ])));
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let gateway = serve(Arc::clone(&script), Arc::clone(&captured)).await;

        let options_for = || UploadBugReportOptions {
            token: None,
            signal: None,
            gateway_url: Some(gateway.clone()),
        };

        assert_eq!(
            upload_bug_report(&sample_bug_report_bundle(), options_for()).await,
            Err("Bug report upload failed: quota exceeded".to_string())
        );
        // `description || error` falls through to `error`.
        assert_eq!(
            upload_bug_report(&sample_bug_report_bundle(), options_for()).await,
            Err("Bug report upload failed: boom".to_string())
        );
        // Non-JSON body: statusText fallback.
        assert_eq!(
            upload_bug_report(&sample_bug_report_bundle(), options_for()).await,
            Err("Bug report upload failed: Internal Server Error".to_string())
        );
        // Unknown status without a standard reason: numeric status fallback.
        assert_eq!(
            upload_bug_report(&sample_bug_report_bundle(), options_for()).await,
            Err("Bug report upload failed: 599".to_string())
        );
        assert_eq!(captured.lock().len(), 4);
        assert!(captured
            .lock()
            .iter()
            .all(|(authorization, _)| authorization.is_none()));
    }

    #[tokio::test]
    async fn abort_returns_err_without_sending() {
        let script: Script = Arc::new(Mutex::new(VecDeque::from(vec![(200, "{}".to_string())])));
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let gateway = serve(Arc::clone(&script), Arc::clone(&captured)).await;
        let controller = AbortController::new();
        let signal = controller.signal();
        controller.abort();

        let result = upload_bug_report(
            &sample_bug_report_bundle(),
            UploadBugReportOptions {
                token: None,
                signal: Some(signal),
                gateway_url: Some(gateway),
            },
        )
        .await;

        assert!(result.is_err());
        assert!(
            captured.lock().is_empty(),
            "aborted request must not reach the wire"
        );
    }
}
