//! Port of `experimental/vacation/runtime.ts` — the headless vacation
//! runtime.
//!
//! Deviations:
//! - **D174 — the shared `openDurable` machinery lives in
//!   `durable::runtime::open_harness_view`**; this module injects the
//!   vacation registry and the vacation-only root agent (`extensions:
//!   [Vacation]`, no `cwd`, no execution environments — upstream creates
//!   none here). The view/controller types re-export so the module keeps
//!   its upstream surface (`vacation/runtime.ts` declares the same shapes;
//!   the port keeps one implementation per D174).

use std::sync::Arc;

use pi_durable::harness::types::{AgentChange, ExtensionsChange, SetOrClear};

use crate::experimental::durable::harness_setup::InitialModel;
use crate::experimental::durable::runtime::{
    open_harness_view, ClusterOptions, OpenDurableOptions, OpenDurableResult,
};
use crate::experimental::durable::sessions::select_session;
use crate::experimental::vacation::harness_setup::create_vacation_registry;
use crate::experimental::vacation::vacation::vacation_extension;

// Shared surface (upstream `vacation/runtime.ts` declares the same shapes).
pub use crate::experimental::durable::runtime::{
    agent_of, Closing, ConversationSummary, DurableController, DurableView, DurableViewSource,
    ModelSummary, Notice, NoticeLevel, SessionSummary,
};

/// The vacation cluster's `openDurable`.
pub async fn open_durable(options: OpenDurableOptions) -> Result<OpenDurableResult, String> {
    let cwd = match options.cwd {
        Some(cwd) => cwd,
        None => std::env::current_dir()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|_| ".".to_string()),
    };
    let location = select_session(&cwd, options.continue_session).await?;
    open_harness_view(
        location,
        ClusterOptions {
            registry: create_vacation_registry(),
            // Only the vacation planner; the research subagent selects search
            // itself (upstream the root agent change).
            root_agent: Arc::new(|initial: Option<InitialModel>| {
                let initial = initial.unwrap_or_default();
                AgentChange {
                    extensions: Some(SetOrClear::Set(ExtensionsChange::List(vec![
                        vacation_extension(),
                    ]))),
                    model: initial.model.map(SetOrClear::Set),
                    thinking_level: initial.thinking_level.map(SetOrClear::Set),
                    ..AgentChange::default()
                }
            }),
            env: None,
            cleanup: None,
        },
    )
    .await
}
