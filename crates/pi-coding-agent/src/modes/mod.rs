//! Port of `packages/coding-agent/src/modes/` — the non-interactive run
//! modes. `interactive/` stays excluded with the TUI (2026-10-05 decision);
//! print/JSON/RPC modes never touch it.
//!
//! Deviations:
//! - **D90 — runtime-host seam.** Upstream passes the concrete
//!   `AgentSessionRuntime` (tests abuse structural typing with casted object
//!   literals). Rust modes are generic over [`RuntimeHost`]; the production
//!   impl wraps `Arc<tokio::sync::Mutex<AgentSessionRuntime>>`
//!   ([`AgentSessionRuntimeHandle`]) and tests inject fakes.
//! - **D91 — stdio seam.** `core/output-guard.ts` has no Rust counterpart for
//!   overridable `process.stdout`; modes take a [`ModeStdio`] implementation
//!   ([`StdioModeIo`] in production) so tests capture protocol output where
//!   upstream `vi.mock`s the module.
//! - **D92 — bounded mode loop.** Upstream modes terminate with
//!   `process.exit`; the Rust loops return the exit code and the production
//!   wrappers perform the exit, so tests can drive a mode in-process.
//!
//! Production wiring: the `pi` binary target (D109) installs a
//! `ModeDispatcher` (cli D102 seam) over [`print_mode::run_print_mode`] and
//! [`rpc::run_rpc_mode`], wrapping the runtime in
//! [`AgentSessionRuntimeHandle`]. The upstream `test/rpc.test.ts` live-CLI
//! suite runs the real binary through [`rpc::RpcClient`]
//! (`tests/rpc_live_cli_tests.rs`, D110).

pub mod json_event;
pub mod print_mode;
pub mod rpc;

#[cfg(test)]
mod test_support;

use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::core::agent_session::{AgentSession, ExtensionBindings};
use crate::core::agent_session_runtime::{
    AgentSessionRuntime, ForkOptions, ForkOutcome, NewSessionOptions, SwitchOutcome,
    SwitchSessionOptions as RuntimeSwitchSessionOptions,
};
use crate::core::extensions::types::{
    ExtensionCommandContextActions, ForkCommandOptions, NavigateTreeOptions,
    NewSessionCommandOptions, ReplacedSessionContext, SessionSwitchOutcome, SwitchSessionOptions,
};
use crate::core::output_guard;

/// `setRebindSession` hook (upstream `() => Promise<void>`; invoked with the
/// session the runtime switched to).
pub type RebindSessionHook = Arc<dyn Fn(&AgentSession) -> BoxFuture<'static, ()> + Send + Sync>;

/// The subset of `AgentSessionRuntime` the modes consume (D90).
pub trait RuntimeHost: Send + Sync + 'static {
    /// Current session (`runtimeHost.session`).
    fn session(&self) -> Arc<AgentSession>;

    /// `newSession`.
    fn new_session(
        &self,
        options: Option<NewSessionOptions>,
    ) -> BoxFuture<'static, Result<SwitchOutcome, String>>;

    /// `fork`.
    fn fork(
        &self,
        entry_id: &str,
        options: Option<ForkOptions>,
    ) -> BoxFuture<'static, Result<ForkOutcome, String>>;

    /// `switchSession`.
    fn switch_session(
        &self,
        session_path: &str,
        options: Option<RuntimeSwitchSessionOptions>,
    ) -> BoxFuture<'static, Result<SwitchOutcome, String>>;

    /// `dispose`.
    fn dispose(&self) -> BoxFuture<'static, ()>;

    /// `setRebindSession`.
    fn set_rebind_session(&self, rebind_session: Option<RebindSessionHook>);
}

/// Production [`RuntimeHost`] over [`AgentSessionRuntime`] (D90). The
/// runtime's switch futures are `!Send` (landed code holds parking-lot
/// guards across internal awaits), so the handle drives each switch to
/// completion on the calling worker via `block_in_place` + `block_on`
/// (multi-thread runtime; the modes always run on one).
#[derive(Clone)]
pub struct AgentSessionRuntimeHandle(pub Arc<parking_lot::Mutex<AgentSessionRuntime>>);

/// Drive a `!Send` runtime-switch future on the current worker (D90).
impl RuntimeHost for AgentSessionRuntimeHandle {
    fn session(&self) -> Arc<AgentSession> {
        self.0.lock().session_arc()
    }

    fn new_session(
        &self,
        options: Option<NewSessionOptions>,
    ) -> BoxFuture<'static, Result<SwitchOutcome, String>> {
        let runtime = Arc::clone(&self.0);
        Box::pin(async move {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(async move { runtime.lock().new_session(options).await })
            })
        })
    }

    fn fork(
        &self,
        entry_id: &str,
        options: Option<ForkOptions>,
    ) -> BoxFuture<'static, Result<ForkOutcome, String>> {
        let runtime = Arc::clone(&self.0);
        let entry_id = entry_id.to_string();
        Box::pin(async move {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(async move { runtime.lock().fork(&entry_id, options).await })
            })
        })
    }

    fn switch_session(
        &self,
        session_path: &str,
        options: Option<RuntimeSwitchSessionOptions>,
    ) -> BoxFuture<'static, Result<SwitchOutcome, String>> {
        let runtime = Arc::clone(&self.0);
        let session_path = session_path.to_string();
        Box::pin(async move {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    runtime.lock().switch_session(&session_path, options).await
                })
            })
        })
    }

    fn dispose(&self) -> BoxFuture<'static, ()> {
        let runtime = Arc::clone(&self.0);
        Box::pin(async move {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(async move { runtime.lock().dispose().await })
            })
        })
    }

    fn set_rebind_session(&self, rebind_session: Option<RebindSessionHook>) {
        self.0.lock().set_rebind_session(rebind_session);
    }
}

/// Stdio seam of the modes (D91): raw protocol stdout, diagnostics stderr,
/// backpressure, and process signal handling. Production: [`StdioModeIo`];
/// tests capture lines where upstream `vi.mock`s `core/output-guard.ts`.
pub trait ModeStdio: Send + Sync + 'static {
    /// `writeRawStdout`.
    fn write_raw_stdout(&self, text: &str);

    /// `console.error`-family diagnostics channel.
    fn write_stderr(&self, text: &str);

    /// `waitForRawStdoutBackpressure`.
    fn wait_for_raw_stdout_backpressure(&self) -> BoxFuture<'static, ()>;

    /// `flushRawStdout`.
    fn flush_raw_stdout(&self) -> BoxFuture<'static, ()>;

    /// Install SIGTERM (and SIGHUP off-Windows) handlers dispatching the
    /// exit code (143 / 129); returns the uninstall closure
    /// (`registerSignalHandlers`).
    fn register_signal_handlers(
        &self,
        on_signal: Arc<dyn Fn(i32) -> BoxFuture<'static, ()> + Send + Sync>,
    ) -> Box<dyn FnOnce() + Send>;

    /// `process.exit(code)` — production terminates the process (D92).
    fn exit(&self, code: i32) -> !;
}

/// Production [`ModeStdio`] over [`output_guard`] and tokio signal streams.
pub struct StdioModeIo;

impl Default for StdioModeIo {
    fn default() -> Self {
        Self::new()
    }
}

impl StdioModeIo {
    pub fn new() -> Self {
        Self
    }

    /// `takeOverStdout`.
    pub fn take_over_stdout(&self) {
        output_guard::take_over_stdout();
    }
}

impl ModeStdio for StdioModeIo {
    fn write_raw_stdout(&self, text: &str) {
        output_guard::write_raw_stdout(text);
    }

    fn write_stderr(&self, text: &str) {
        eprint!("{text}");
        if !text.ends_with('\n') {
            eprintln!();
        }
    }

    fn wait_for_raw_stdout_backpressure(&self) -> BoxFuture<'static, ()> {
        Box::pin(output_guard::wait_for_raw_stdout_backpressure())
    }

    fn flush_raw_stdout(&self) -> BoxFuture<'static, ()> {
        Box::pin(output_guard::flush_raw_stdout())
    }

    fn register_signal_handlers(
        &self,
        on_signal: Arc<dyn Fn(i32) -> BoxFuture<'static, ()> + Send + Sync>,
    ) -> Box<dyn FnOnce() + Send> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM stream");
            let mut sighup = signal(SignalKind::hangup()).expect("SIGHUP stream");
            let term_task = {
                let on_signal = Arc::clone(&on_signal);
                tokio::spawn(async move {
                    if sigterm.recv().await.is_some() {
                        on_signal(143).await;
                    }
                })
            };
            let hup_task = {
                let on_signal = Arc::clone(&on_signal);
                tokio::spawn(async move {
                    if sighup.recv().await.is_some() {
                        on_signal(129).await;
                    }
                })
            };
            Box::new(move || {
                term_task.abort();
                hup_task.abort();
            })
        }
        #[cfg(windows)]
        {
            let mut ctrl_close = tokio::signal::windows::ctrl_close().expect("ctrl_close stream");
            let task = tokio::spawn(async move {
                if ctrl_close.recv().await.is_some() {
                    on_signal(143).await;
                }
            });
            Box::new(move || {
                task.abort();
            })
        }
    }

    fn exit(&self, code: i32) -> ! {
        std::process::exit(code)
    }
}

/// Base extension bindings both modes install (`bindExtensions({ mode })`).
pub fn extension_bindings(
    mode: crate::core::extensions::types::ExtensionMode,
) -> ExtensionBindings {
    ExtensionBindings {
        ui_context: None,
        mode: Some(mode),
        ..Default::default()
    }
}

/// Host + stdio pair the mode closures capture (D90/D91).
pub(crate) struct ModeCore {
    pub host: Arc<dyn RuntimeHost>,
    pub io: Arc<dyn ModeStdio>,
}

/// Map an extension `with_session` callback to the runtime's `FnOnce` seam.
fn map_with_session(
    with_session: Option<crate::core::extensions::types::WithSessionCallback>,
) -> Option<Box<dyn FnOnce(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send>> {
    with_session.map(|with_session| {
        Box::new(move |context: ReplacedSessionContext| with_session(context))
            as Box<dyn FnOnce(ReplacedSessionContext) -> BoxFuture<'static, ()> + Send>
    })
}

/// `commandContextActions` of the modes: session-control actions routed
/// through the runtime host and the live session.
pub(crate) fn command_context_actions(
    core: &ModeCore,
    session: &Arc<AgentSession>,
) -> ExtensionCommandContextActions {
    let host_new = Arc::clone(&core.host);
    let host_fork = Arc::clone(&core.host);
    let host_switch = Arc::clone(&core.host);
    let io_new = Arc::clone(&core.io);
    let io_fork = Arc::clone(&core.io);
    let io_switch = Arc::clone(&core.io);
    let io_navigate = Arc::clone(&core.io);
    let idle_session = Arc::clone(session);
    let navigate_session = Arc::clone(session);
    let reload_session = Arc::clone(session);
    ExtensionCommandContextActions {
        wait_for_idle: Some(Arc::new(move || {
            let session = Arc::clone(&idle_session);
            Box::pin(async move { session.wait_for_idle().await })
        })),
        new_session: Some(Arc::new(
            move |options: Option<NewSessionCommandOptions>| {
                let host = Arc::clone(&host_new);
                let io = Arc::clone(&io_new);
                Box::pin(async move {
                    let options = options.map(|options| NewSessionOptions {
                        parent_session: options.parent_session,
                        // D93b: the extension `setup` hook is not forwarded.
                        setup: None,
                        with_session: map_with_session(options.with_session),
                    });
                    match host.new_session(options).await {
                        Ok(outcome) => SessionSwitchOutcome {
                            cancelled: outcome.cancelled,
                        },
                        Err(error) => {
                            io.write_stderr(&error);
                            SessionSwitchOutcome { cancelled: false }
                        }
                    }
                })
            },
        )),
        fork: Some(Arc::new(
            move |entry_id: String, options: Option<ForkCommandOptions>| {
                let host = Arc::clone(&host_fork);
                let io = Arc::clone(&io_fork);
                Box::pin(async move {
                    let options = options.map(|options| ForkOptions {
                        position: options.position,
                        with_session: map_with_session(options.with_session),
                    });
                    match host.fork(&entry_id, options).await {
                        Ok(result) => SessionSwitchOutcome {
                            cancelled: result.cancelled,
                        },
                        Err(error) => {
                            io.write_stderr(&error);
                            SessionSwitchOutcome { cancelled: false }
                        }
                    }
                })
            },
        )),
        navigate_tree: Some(Arc::new(
            move |target_id: String, options: Option<NavigateTreeOptions>| {
                let session = Arc::clone(&navigate_session);
                let io = Arc::clone(&io_navigate);
                Box::pin(async move {
                    // The command-side and session-side `NavigateTreeOptions` are
                    // distinct landed types with the same fields; map field-wise.
                    let options =
                        options.map(|options| crate::core::agent_session::NavigateTreeOptions {
                            summarize: options.summarize,
                            custom_instructions: options.custom_instructions,
                            replace_instructions: options.replace_instructions,
                            label: options.label,
                        });
                    match session.navigate_tree(&target_id, options).await {
                        Ok(result) => SessionSwitchOutcome {
                            cancelled: result.cancelled,
                        },
                        Err(error) => {
                            io.write_stderr(&error);
                            SessionSwitchOutcome { cancelled: false }
                        }
                    }
                })
            },
        )),
        switch_session: Some(Arc::new(
            move |session_path: String, options: Option<SwitchSessionOptions>| {
                let host = Arc::clone(&host_switch);
                let io = Arc::clone(&io_switch);
                Box::pin(async move {
                    let options = options.map(|options| RuntimeSwitchSessionOptions {
                        cwd_override: None,
                        with_session: map_with_session(options.with_session),
                        project_trust_context_factory: None,
                    });
                    match host.switch_session(&session_path, options).await {
                        Ok(result) => SessionSwitchOutcome {
                            cancelled: result.cancelled,
                        },
                        Err(error) => {
                            io.write_stderr(&error);
                            SessionSwitchOutcome { cancelled: false }
                        }
                    }
                })
            },
        )),
        reload: Some(Arc::new(move || {
            let session = Arc::clone(&reload_session);
            Box::pin(async move {
                let _ = session.reload(None).await;
            })
        })),
    }
}
