//! `rpc-entry.ts` — RPC-mode entry point.
//!
//! Process titling is Node-specific; the `--mode rpc` prefixing and env
//! markers carry over.

use crate::cli::main_dispatch::{main, MainOptions};

/// `rpc-entry` main: re-runs [`main`] with `--mode rpc`.
pub async fn rpc_main(args: Vec<String>) {
    std::env::set_var("PI_CODING_AGENT", "true");
    std::env::set_var("AI_AGENT", "pi");

    let mut prefixed = vec!["--mode".to_string(), "rpc".to_string()];
    prefixed.extend(args);
    main(prefixed, Some(MainOptions {})).await;
}
