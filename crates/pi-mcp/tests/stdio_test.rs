//! Port of `test/stdio.test.ts`. Upstream spawns `node fixtures/stdio-server.mjs`;
//! here the fixture is the crate's own `mcp-stdio-fixture` bin (and
//! `mcp-stubborn-fixture` for the shutdown case), located via `CARGO_BIN_EXE_*`.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use pi_mcp::client::{ClientState, McpClient, McpClientOptions};
use pi_mcp::transports::stdio::{StdioTransport, StdioTransportOptions};

fn fixture() -> String {
	env!("CARGO_BIN_EXE_mcp-stdio-fixture").to_string()
}

#[cfg(unix)]
fn stubborn() -> String {
	env!("CARGO_BIN_EXE_mcp-stubborn-fixture").to_string()
}

fn stderr_text(stderr: &Arc<StdMutex<String>>) -> String {
	stderr.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

async fn wait_for_stderr(stderr: &Arc<StdMutex<String>>, needle: &str) {
	for _ in 0..500 {
		if stderr_text(stderr).contains(needle) {
			return;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
	panic!("stderr never contained {needle:?}: {:?}", stderr_text(stderr));
}

// Port of stdio.test.ts "connects to a newline-delimited MCP server and captures stderr".
#[tokio::test]
async fn connects_to_a_newline_delimited_mcp_server_and_captures_stderr() {
	let stderr: Arc<StdMutex<String>> = Arc::new(StdMutex::new(String::new()));
	let stderr_sink = Arc::clone(&stderr);
	let stdio = Arc::new(StdioTransport::new(StdioTransportOptions {
		command: fixture(),
		on_stderr: Some(Arc::new(move |chunk| {
			stderr_sink
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner)
				.push_str(chunk);
		})),
		..Default::default()
	}));
	let client = McpClient::new(McpClientOptions::new("stdio-test", "1.0.0"));
	client.connect(Arc::clone(&stdio) as Arc<dyn pi_mcp::McpTransport>).await.expect("connect");
	let tools = client.list_tools(&Default::default()).await.expect("list tools");
	assert_eq!(
		serde_json::to_value(&tools).unwrap(),
		serde_json::json!([{ "name": "echo", "inputSchema": { "type": "object" } }])
	);
	let mut args = serde_json::Map::new();
	args.insert("text".to_string(), serde_json::json!("hello"));
	let result = client.call_tool("echo", Some(args), &Default::default()).await.expect("call");
	assert_eq!(
		serde_json::to_value(&result).unwrap(),
		serde_json::json!({ "content": [{ "type": "text", "text": "hello" }] })
	);
	assert!(stdio.pid().is_some(), "transport exposes a pid");
	wait_for_stderr(&stderr, "stdio fixture ready").await;
	assert!(stderr_text(&stderr).contains("stdio fixture ready"));
	client.close().await.expect("close");
	assert_eq!(client.connection_state(), ClientState::Closed);
}

// Port of stdio.test.ts "kills a server that ignores shutdown, including its children"
// (upstream skips on win32; this port relies on Unix process groups).
#[cfg(unix)]
#[tokio::test]
async fn kills_a_server_that_ignores_shutdown_including_its_children() {
	let stderr: Arc<StdMutex<String>> = Arc::new(StdMutex::new(String::new()));
	let stderr_sink = Arc::clone(&stderr);
	let stdio = Arc::new(StdioTransport::new(StdioTransportOptions {
		command: stubborn(),
		close_timeout_ms: Some(100),
		on_stderr: Some(Arc::new(move |chunk| {
			stderr_sink
				.lock()
				.unwrap_or_else(std::sync::PoisonError::into_inner)
				.push_str(chunk);
		})),
		..Default::default()
	}));
	let client = McpClient::new(McpClientOptions::new("stdio-test", "1.0.0"));
	client.connect(Arc::clone(&stdio) as Arc<dyn pi_mcp::McpTransport>).await.expect("connect");

	// The fixture reports its grandchild as `grandchild <pid>` on stderr.
	let grandchild_pid = loop {
		let text = stderr_text(&stderr);
		if let Some(pid) = text
			.split("grandchild ")
			.nth(1)
			.and_then(|rest| rest.split_whitespace().next())
			.and_then(|token| token.parse::<i32>().ok())
		{
			break pid;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	};

	let started_at = Instant::now();
	client.close().await.expect("close");
	assert!(
		started_at.elapsed() < Duration::from_secs(5),
		"close must not wait for the default 2s SIGKILL budget"
	);

	// The grandchild must be gone once the group is killed.
	let mut alive = true;
	for _ in 0..100 {
		if !process_alive(grandchild_pid) {
			alive = false;
			break;
		}
		tokio::time::sleep(Duration::from_millis(20)).await;
	}
	assert!(!alive, "grandchild should be killed with the server's process group");
}

#[cfg(unix)]
fn process_alive(pid: i32) -> bool {
	// kill(pid, 0): ESRCH means the process is gone.
	unsafe { libc::kill(pid, 0) == 0 }
}
