//! The `pi` CLI binary (upstream `packages/coding-agent` `bin` entry,
//! `src/cli.ts`: `setupCli(); main(process.argv.slice(2))`).
//!
//! Installs the modes-module dispatcher (**D102** seam) over the print and
//! RPC runners and hands control to [`cli::main_dispatch::main`]. Interactive
//! mode stays excluded with the TUI: the dispatcher only ever receives the
//! print/json/rpc modes (the interactive branch errors in `main` itself).
//!
//! Deviation **D109** — the dispatcher receives the `AgentSessionRuntime` by
//! value: the production [`RuntimeHost`] handle owns a shared
//! `Arc<parking_lot::Mutex<AgentSessionRuntime>>`, which a `&`-borrow
//! cannot provide. Print mode's exit code is returned to `main` for
//! `process.exitCode` parity; RPC mode never returns (it exits the process
//! from the mode loop, D92).

use std::sync::Arc;

use pi_coding_agent::cli::main_dispatch::{self, MainOptions, ModeDispatcher, PrintModeOptions};
use pi_coding_agent::cli::setup::setup_cli;
use pi_coding_agent::core::agent_session_runtime::AgentSessionRuntime;
use pi_coding_agent::modes::print_mode::{PrintOutputMode, PrintModeOptions as ModePrintOptions};
use pi_coding_agent::modes::{AgentSessionRuntimeHandle, RuntimeHost};

/// Print/rpc runners installed for the binary (D102/D109).
struct ModeRunners;

impl ModeDispatcher for ModeRunners {
	fn run_print_mode(&self, runtime: AgentSessionRuntime, options: PrintModeOptions) -> i32 {
		let host = runtime_host(runtime);
		let options = ModePrintOptions {
			mode: if options.json { PrintOutputMode::Json } else { PrintOutputMode::Text },
			messages: options.messages,
			initial_message: options.initial_message,
			initial_images: if options.initial_images.is_empty() { None } else { Some(options.initial_images) },
		};
		tokio::task::block_in_place(|| {
			tokio::runtime::Handle::current()
				.block_on(pi_coding_agent::modes::print_mode::run_print_mode(host, options))
		})
	}

	fn run_rpc_mode(&self, runtime: AgentSessionRuntime) {
		let host = runtime_host(runtime);
		tokio::task::block_in_place(|| {
			tokio::runtime::Handle::current().block_on(async move {
				pi_coding_agent::modes::rpc::run_rpc_mode(host).await
			})
		})
	}
}

/// Wrap the runtime in the production [`RuntimeHost`] handle (D90).
fn runtime_host(runtime: AgentSessionRuntime) -> Arc<dyn RuntimeHost> {
	Arc::new(AgentSessionRuntimeHandle(Arc::new(parking_lot::Mutex::new(runtime))))
}

fn main() {
	// Internal-process dispatch (D138/D140/D161/D162): spawned
	// coordinator/server/session-worker roles run their entrypoints instead
	// of the CLI (upstream each entry file's `isDirectInternalProcessEntry`
	// block).
	if let Some(role) = pi_coding_agent::experimental::process::get_internal_process_role().ok().flatten() {
		let args: Vec<String> = std::env::args().skip(1).collect();
		let runtime = tokio::runtime::Builder::new_multi_thread()
			.enable_all()
			.build()
			.expect("failed to build the async runtime");
		let result = runtime.block_on(async move {
			match role {
				pi_coding_agent::experimental::process::InternalProcessRole::Coordinator => {
					pi_coding_agent::experimental::coordinator_entry::run_coordinator_entry(&args).await
				}
				pi_coding_agent::experimental::process::InternalProcessRole::Server => {
					pi_coding_agent::experimental::server::run_server_entry(&args).await
				}
				pi_coding_agent::experimental::process::InternalProcessRole::SessionWorker => {
					pi_coding_agent::experimental::session_worker::run_session_worker_entry(&args).await
				}
			}
		});
		if let Err(error) = result {
			eprintln!("{error}");
			std::process::exit(1);
		}
		return;
	}
	main_dispatch::set_mode_dispatcher(Box::new(ModeRunners));
	setup_cli();
	// The modes drive `!Send` runtime-switch futures to completion on the
	// calling worker (block_in_place), which requires the multi-thread flavor.
	let runtime = tokio::runtime::Builder::new_multi_thread()
		.enable_all()
		.build()
		.expect("failed to build the async runtime");
	let args: Vec<String> = std::env::args().skip(1).collect();
	runtime.block_on(main_dispatch::main(args, Some(MainOptions)));
}
