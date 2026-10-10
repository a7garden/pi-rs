//! Rust port of `test/fixtures/stubborn-server.mjs`: answers initialize, spawns a
//! grandchild that outlives stdin, and ignores stdin EOF and SIGTERM.

use std::io::{BufRead, Write};
use std::process::{Child, Command, Stdio};

#[cfg(unix)]
fn ignore_sigterm() {
    unsafe {
        libc::signal(libc::SIGTERM, libc::SIG_IGN);
    }
}

#[cfg(not(unix))]
fn ignore_sigterm() {}

fn spawn_grandchild() -> Option<Child> {
    // Re-invoke this binary as a grandchild that ignores SIGTERM and idles.
    let exe = std::env::current_exe().ok()?;
    Command::new(exe)
        .arg("--stubborn-grandchild")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()
}

fn main() {
    let mut args = std::env::args().skip(1);
    if matches!(args.next().as_deref(), Some("--stubborn-grandchild")) {
        ignore_sigterm();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    ignore_sigterm();
    if let Some(grandchild) = spawn_grandchild() {
        let pid = grandchild.id();
        eprintln!("grandchild {pid}");
        // Upstream leaks the grandchild deliberately; park a handle so it is not reaped.
        std::mem::forget(grandchild);
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if message.get("method").and_then(serde_json::Value::as_str) != Some("initialize") {
            continue;
        }
        let reply = serde_json::json!({
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": { "name": "stubborn-fixture", "version": "1.0.0" },
            },
        });
        writeln!(out, "{reply}").ok();
    }
    // Ignore stdin EOF: keep running until killed.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
