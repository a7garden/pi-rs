//! Port of `packages/coding-agent/src/modes/rpc/` — headless RPC mode:
//! JSON commands on stdin, events and responses on stdout.
//!
//! Upstream transports this over Node streams; the Rust port keeps the same
//! wire protocol on plain stdio plus the [`super::ModeStdio`] seam (D92). The
//! upstream `pi-server`/`pi-chord` socket protocol is a different, unrelated
//! surface — `modes/rpc` is a direct JSONL protocol, so no pi-protocol/
//! pi-chord types participate (mapping note, PORT_MAP rule 3).
//!
//! Tests: the upstream `test/rpc-client-*.test.ts` suites live in the
//! module tests ([`rpc_client`], [`jsonl`]); the `test/rpc.test.ts`
//! live-CLI suite (18 cases, credential-gated upstream via
//! `describe.skipIf`) is ported 1:1 in `tests/rpc_live_cli_tests.rs`
//! against the built `pi` binary (D94/D109/D110).

pub mod jsonl;
pub mod rpc_client;
pub mod rpc_mode;
pub mod rpc_types;

pub use jsonl::{read_jsonl_lines, serialize_json_line, JsonlLineReader};
pub use rpc_client::{ModelInfo, RpcClient, RpcClientOptions, RpcEventListener};
pub use rpc_mode::run_rpc_mode;
pub use rpc_types::{RpcCommandType, RpcExtensionUIResponse, RpcResponse, RpcSessionState};
