//! Test-only HTTP mock server for the llama extension tests — the Rust
//! stand-in for the upstream suite's `node:http` servers. Supports JSON
//! responses and registerable SSE streams (`data: …\n\n` frames).

#![cfg(test)]

use std::io::Write;
use std::sync::Arc;

use pi_agent::BoxFuture;
use serde_json::Value as JsonValue;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// A parsed test request (`{ method, url, headers, body }` upstream).
#[derive(Debug, Clone)]
pub(crate) struct TestRequest {
    pub method: String,
    /// Path plus query, e.g. `/models?reload=1`.
    pub target: String,
    pub authorization: Option<String>,
    pub body: JsonValue,
}

/// Handler response.
pub(crate) enum TestResponse {
    /// `json(response, value)`.
    Json(u16, JsonValue),
    /// Bare status (`response.writeHead(404).end()`).
    Empty(u16),
    /// `text/event-stream` connection registered with a hub.
    Sse(Arc<SseHub>),
}

pub(crate) type Handler =
    Arc<dyn Fn(TestRequest) -> BoxFuture<'static, TestResponse> + Send + Sync>;

/// Broadcasts `data:` frames to every connected SSE client (`send` upstream).
#[derive(Default)]
pub(crate) struct SseHub {
    streams: parking_lot::Mutex<Vec<std::net::TcpStream>>,
    closed: tokio::sync::Notify,
}

impl SseHub {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(SseHub::default())
    }

    fn register(&self, stream: std::net::TcpStream) {
        self.streams.lock().push(stream);
    }

    /// `send(event)`: `data: {json}\n\n` to every stream; broken streams are
    /// dropped.
    pub(crate) fn send(&self, event: &JsonValue) {
        let frame = format!("data: {event}\n\n");
        let mut streams = self.streams.lock();
        streams.retain_mut(|stream| stream.write_all(frame.as_bytes()).is_ok());
    }

    /// End every SSE connection.
    pub(crate) fn close_all(&self) {
        let mut streams = self.streams.lock();
        for stream in streams.drain(..) {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

/// A running mock server; dropping it leaks the accept loop into the test
/// runtime's teardown, like the upstream `afterEach` cleanup.
pub(crate) struct TestServer {
    pub url: String,
}

/// `listen(handler)`: bind an ephemeral loopback port and serve until the
/// runtime ends.
pub(crate) async fn listen(handler: Handler) -> TestServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let handler = Arc::clone(&handler);
            tokio::spawn(async move {
                serve_connection(stream, handler).await;
            });
        }
    });
    TestServer {
        url: format!("http://127.0.0.1:{port}"),
    }
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

async fn serve_connection(mut stream: TcpStream, handler: Handler) {
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        // Read the head.
        let head_end = loop {
            if let Some(head_end) = find_head_end(&buffer) {
                break head_end;
            }
            match stream.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(read) => buffer.extend_from_slice(&chunk[..read]),
            }
        };
        let head = String::from_utf8_lossy(&buffer[..head_end]).to_string();
        let mut lines = head.split("\r\n");
        let mut parts = lines.next().unwrap_or("").split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let target = parts.next().unwrap_or("").to_string();
        let mut content_length = 0usize;
        let mut authorization = None;
        for line in lines {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            match name.as_str() {
                "content-length" => content_length = value.parse().unwrap_or(0),
                "authorization" => authorization = Some(value.to_string()),
                _ => {}
            }
        }
        // Read the body.
        let body_start = head_end + 4;
        let body_end = body_start + content_length;
        while buffer.len() < body_end {
            match stream.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(read) => buffer.extend_from_slice(&chunk[..read]),
            }
        }
        let body: JsonValue =
            serde_json::from_slice(&buffer[body_start..body_end]).unwrap_or(JsonValue::Null);
        buffer.drain(..body_end);

        let request = TestRequest {
            method,
            target,
            authorization,
            body,
        };
        match handler(request).await {
            TestResponse::Json(status, value) => {
                let payload = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
                let response = format!(
					"HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{payload}",
					payload.len()
				);
                if stream.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
            }
            TestResponse::Empty(status) => {
                let response =
					format!("HTTP/1.1 {status} Response\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n");
                if stream.write_all(response.as_bytes()).await.is_err() {
                    return;
                }
            }
            TestResponse::Sse(hub) => {
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n";
                if stream.write_all(head.as_bytes()).await.is_err() {
                    return;
                }
                let _ = stream.flush().await;
                // Register a duplicated descriptor; the connection stays open
                // through the hub's copy, and SSE connections carry no
                // further requests, so the accept task ends here.
                match stream.into_std() {
                    Ok(std_stream) => {
                        let _ = std_stream.set_nonblocking(false);
                        if let Ok(registered) = std_stream.try_clone() {
                            hub.register(registered);
                        }
                    }
                    Err(_) => return,
                }
                return;
            }
        }
    }
}
