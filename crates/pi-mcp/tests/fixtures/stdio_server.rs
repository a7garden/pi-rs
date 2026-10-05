//! Rust port of `test/fixtures/stdio-server.mjs`: a newline-delimited MCP server
//! fixture answering initialize/tools list/call/ping on stdin/stdout.

use std::io::{BufRead, Write};

fn main() {
	eprintln!("stdio fixture ready");
	let stdin = std::io::stdin();
	let stdout = std::io::stdout();
	let mut out = stdout.lock();
	for line in stdin.lock().lines() {
		let Ok(line) = line else { break };
		let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
		let Some(id) = message.get("id") else { continue };
		let method = message.get("method").and_then(serde_json::Value::as_str).unwrap_or_default();
		let result = match method {
			"initialize" => serde_json::json!({
				"protocolVersion": "2025-06-18",
				"capabilities": { "tools": {} },
				"serverInfo": { "name": "stdio-fixture", "version": "1.0.0" },
			}),
			"tools/list" => {
				serde_json::json!({ "tools": [{ "name": "echo", "inputSchema": { "type": "object" } }] })
			}
			"tools/call" => {
				let text = message
					.pointer("/params/arguments/text")
					.and_then(serde_json::Value::as_str)
					.unwrap_or_default();
				serde_json::json!({ "content": [{ "type": "text", "text": text }] })
			}
			"ping" => serde_json::json!({}),
			_ => {
				let reply = serde_json::json!({
					"jsonrpc": "2.0",
					"id": id,
					"error": { "code": -32601, "message": "not found" },
				});
				writeln!(out, "{reply}").ok();
				continue;
			}
		};
		writeln!(out, "{}", serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })).ok();
	}
}
