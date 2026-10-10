//! Port of `test/helpers.ts`: a loopback HTTP test server. Upstream builds on
//! `node:http`; here a minimal HTTP/1.1 server runs on `tokio::TcpListener` and
//! speaks the crate's [`HttpRequest`]/[`HttpResponse`] values directly.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::future::BoxFuture;
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use pi_mcp::auth_provider::{ByteStream, HttpHeaders, HttpRequest, HttpResponse};
use pi_mcp::McpError;

pub type TestHandler = Arc<dyn Fn(HttpRequest) -> BoxFuture<'static, HttpResponse> + Send + Sync>;

pub struct TestHttpServer {
    pub origin: String,
    listener: Arc<TcpListener>,
    shutdown: Arc<tokio::sync::Notify>,
}

/// Upstream `listen(handler)`: starts a loopback HTTP server and returns its origin.
pub async fn listen(handler: TestHandler) -> TestHttpServer {
    let listener = Arc::new(TcpListener::bind(("127.0.0.1", 0)).await.expect("bind"));
    let origin = format!(
        "http://127.0.0.1:{}",
        listener.local_addr().expect("addr").port()
    );
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let accept_shutdown = Arc::clone(&shutdown);
    let accept_listener = Arc::clone(&listener);
    tokio::spawn(async move {
        loop {
            let next = tokio::select! {
                accepted = accept_listener.accept() => accepted,
                () = accept_shutdown.notified() => break,
            };
            let Ok((stream, _)) = next else { break };
            let handler = Arc::clone(&handler);
            tokio::spawn(async move {
                serve_connection(stream, handler).await;
            });
        }
    });
    TestHttpServer {
        origin,
        listener,
        shutdown,
    }
}

impl TestHttpServer {
    /// Upstream `closeServers`.
    pub async fn close(self) {
        self.shutdown.notify_waiters();
        drop(self.listener);
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

async fn serve_connection(mut stream: tokio::net::TcpStream, handler: TestHandler) {
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                buffer.extend_from_slice(&chunk[..n]);
                if let Some(position) = find_subsequence(&buffer, b"\r\n\r\n") {
                    break position;
                }
                if buffer.len() > 1024 * 1024 {
                    return;
                }
            }
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut headers: HttpHeaders = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    let content_length: usize = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse().ok())
        .unwrap_or(0);
    let mut body = buffer.split_off(head_end + 4);
    while body.len() < content_length {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }
    body.truncate(content_length);
    let origin = stream
        .local_addr()
        .map(|addr| format!("http://{addr}"))
        .unwrap_or_default();
    let url = url::Url::parse(&format!("{origin}{target}")).expect("request target");
    let request = HttpRequest {
        method,
        url,
        headers,
        body: Some(body),
        signal: None,
    };
    let response = handler(request).await;
    let head = format!(
        "HTTP/1.1 {} {}\r\nconnection: close\r\n{}\r\n",
        response.status,
        reason_phrase(response.status),
        response
            .headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>(),
    );
    if stream.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    let _ = stream.flush().await;
    let mut body = response.body;
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(chunk) => {
                if stream.write_all(&chunk).await.is_err() || stream.flush().await.is_err() {
                    return;
                }
            }
            Err(_) => break,
        }
    }
    let _ = stream.shutdown().await;
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Response",
    }
}

/// Upstream `readBody`.
pub async fn read_body(request: &HttpRequest) -> String {
    String::from_utf8_lossy(request.body.as_deref().unwrap_or_default()).into_owned()
}

/// A completed response body as a one-shot stream.
pub fn body_from(text: impl Into<Bytes> + Send + 'static) -> ByteStream {
    Box::pin(futures_util::stream::once(async move {
        Ok::<Bytes, McpError>(text.into())
    }))
}

/// A response body that stays open until the sender side is dropped.
pub fn open_body() -> (
    tokio::sync::mpsc::Sender<Result<Bytes, McpError>>,
    ByteStream,
) {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, McpError>>(4);
    let stream: ByteStream = Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
        let chunk = rx.recv().await?;
        Some((chunk, rx))
    }));
    (tx, stream)
}

/// Wait until `condition` holds (upstream tests poll with `setTimeout` loops).
pub async fn wait_for(mut condition: impl FnMut() -> bool, attempts: usize) -> bool {
    for _ in 0..attempts {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    condition()
}
