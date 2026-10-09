//! Port of `experimental/cli.ts` — the development-only entrypoint that tries
//! the experimental `server`/`client` dispatch before falling back to the
//! main CLI (`#!/usr/bin/env node` script becomes a library entry, D180).
//!
//! Deviation **D180**: upstream ends the process (`process.exit` for the
//! client command, natural script end otherwise, with `process.exitCode`
//! carrying failures); here [`run`] returns the exit code and exits
//! immediately on the client command to match the upstream short-circuit.
//! The fallback `main(args)` maps to
//! [`crate::cli::main_dispatch::main`], and wiring this entry into the `pi`
//! bin stays with the bin slice (this module only exposes the seam).

use crate::cli::main_dispatch::{self, MainOptions};
use crate::cli::setup::setup_cli;
use crate::experimental::commands::{run_experimental_command, ExperimentalDispatch};

/// Upstream `setupCli(); … if (await runExperimentalCommand(args)) …`.
pub async fn run(args: Vec<String>) -> i32 {
	setup_cli();
	match run_experimental_command(&args).await {
		ExperimentalDispatch::Handled { exit_code } => {
			if args.first().map(String::as_str) == Some("client") {
				std::process::exit(exit_code);
			}
			exit_code
		}
		ExperimentalDispatch::NotHandled => {
			main_dispatch::main(args, None::<MainOptions>).await;
			0
		}
	}
}
