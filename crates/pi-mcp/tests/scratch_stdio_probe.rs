//! Scratch probe: reproduce the extensions::mcp stdio hang against /bin/sh.
use pi_mcp::client::{McpClient, McpClientOptions};
use pi_mcp::protocol::jsonrpc::JsonRpcId;
use pi_mcp::transports::stdio::{StdioMode, StdioTransport, StdioTransportOptions};
use pi_mcp::McpTransport;
use serde_json::json;
use std::process::Stdio;
use std::time::Duration;

const SCRIPT: &str = r#"while IFS= read -r line; do
	id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
	[ -z "$id" ] && continue
	case "$line" in
	*initialize*) result='{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"cwd","version":"1"}}';;
	*tools/list*) result='{"tools":[{"name":"cwd","inputSchema":{"type":"object"}}]}';;
	*) result "{\"content\":[{\"type\":\"text\",\"text\":\"$PWD\"}]}";;
	esac
	printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$result"
done"#;

#[tokio::test(flavor = "multi_thread")]
async fn probe_stdio_client_roundtrip() {
    let transport = StdioTransport::new(StdioTransportOptions {
        command: "/bin/sh".to_string(),
        args: vec!["-c".to_string(), SCRIPT.to_string()],
        cwd: Some("/tmp".to_string()),
        stderr: StdioMode::Pipe,
        ..Default::default()
    });
    transport.on_error(Arc::new(|e| eprintln!("transport error: {e}")));
    // Mirror the runtime's client options (roots + 60s request timeout).
    let cwd = std::env::temp_dir().to_string_lossy().into_owned();
    let file_url = url::Url::from_file_path(&cwd).map(|u| u.to_string()).unwrap_or_else(|_| format!("file://{cwd}/"));
    let client = McpClient::new(McpClientOptions {
        name: "pi".to_string(),
        version: "1.0.3".to_string(),
        request_timeout_ms: Some(60000),
        roots: Some(pi_mcp::client::Roots::Static(vec![pi_mcp::protocol::types::Root {
            uri: file_url,
            name: Some("tmp".to_string()),
        }])),
        ..McpClientOptions::new("pi", "1.0.3")
    });
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(15), client.connect(Arc::new(transport))).await;
    match result {
        Err(_) => panic!("connect timed out after 15s"),
        Ok(Err(e)) => panic!("connect failed: {e}"),
        Ok(Ok(init)) => println!("connected in {:?}: {}", started.elapsed(), init.server_info.name),
    }
    let tools = tokio::time::timeout(Duration::from_secs(15), client.list_tools(&Default::default())).await;
    match tools {
        Err(_) => panic!("list_tools timed out"),
        Ok(Err(e)) => panic!("list_tools failed: {e}"),
        Ok(Ok(t)) => println!("tools: {:?}", t.iter().map(|t| t.name.clone()).collect::<Vec<_>>()),
    }
    let _ = JsonRpcId::number(0);
    let _ = json!({});
    let _ = Stdio::piped();
    let _ = client.close().await;
}

use std::sync::Arc;
