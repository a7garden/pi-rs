//! Port of `experimental/vacation/vacation.ts` — the vacation planner's
//! tools, its background research task, and the `Search`/`Vacation`
//! extensions.
//!
//! Deviations:
//! - **D176 — typebox schemas become JSON-Schema `serde_json` values and
//!   task phases become erased [`pi_durable::tasks::PhaseHandler`]s** (the
//!   crate-wide erased-tool/task deviation; see `pi_durable::tasks` D1).
//!   `setTimeout` sleeps become `tokio::time::sleep` raced against the
//!   context's abort signal; tool failures take the erased error-result
//!   convention (`is_error` + `tool_error` diagnostic).
//! - The assistant-entry text projection reuses
//!   `durable::subagent::assistant_text` (same erased `EntryRecord` shape).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pi_chord::context::Context;
use pi_durable::harness::agent::configure;
use pi_durable::harness::api::{
    define_tool, ToolDiagnostic, ToolDiagnosticSeverity, ToolExecute, ToolExecutionApi,
    ToolExecutionResult, ToolRegistration, ToolReplay,
};
use pi_durable::harness::define::section;
use pi_durable::harness::types::{
    AgentChange, AnyTask, ConversationOwnership, ExtensionsChange, PromptSection, SectionRender,
    SetOrClear,
};
use pi_durable::session::Tx;
use pi_durable::tasks::{commit_change, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime};
use pi_durable::types::{ConversationId, ConversationQuery, TaskOptions, TaskOwnership};
use serde_json::{json, Value};

use crate::experimental::durable::subagent::assistant_text;

// ─── search: slow, fake, and safe to rerun ──────────────────────────────────

/// Canned results; each topic takes a different time, so a crash can land
/// between them (upstream `SEARCHES`).
static SEARCHES: [(&str, u64, &[&str]); 3] = [
    (
        "weather",
        6,
        &[
            "Saturday: 24°C and sunny",
            "Sunday: 21°C, a short shower around 3 pm",
        ],
    ),
    (
        "museums",
        10,
        &[
            "Kunsthistorisches Museum: open 10-18, book a time slot",
            "Belvedere: Klimt's The Kiss, quietest before 11",
            "Albertina: Monet to Picasso, open until 21 on Friday",
        ],
    ),
    (
        "trains",
        30,
        &[
            "Railjet from Salzburg: 2h 22m, every 30 min",
            "Nightjet from Munich: arrives 06:20",
        ],
    ),
];

fn searches_for(topic: &str) -> Option<(u64, &'static [&'static str])> {
    SEARCHES
        .iter()
        .find(|(name, _, _)| *name == topic)
        .map(|(_, seconds, results)| (*seconds, *results))
}

/// One 2 s step that yields to the context's abort signal (upstream
/// `sleep(2000, undefined, { signal: context.abortSignal })`).
async fn search_step(context: &Context) -> Result<(), String> {
    let sleep = tokio::time::sleep(Duration::from_millis(2_000));
    match context.abort_signal() {
        Some(signal) => {
            tokio::select! {
                () = sleep => Ok(()),
                _reason = signal.aborted_future() => Err("This operation was aborted".into()),
            }
        }
        None => {
            sleep.await;
            Ok(())
        }
    }
}

fn search_tool() -> ToolRegistration {
    let execute: ToolExecute = Arc::new(
        |args: Value, api: Arc<dyn ToolExecutionApi>, context: Context| {
            Box::pin(async move {
                match run_search(args, api.as_ref(), &context).await {
                    Ok(result) => result,
                    Err(message) => ToolExecutionResult {
                        is_error: Some(true),
                        diagnostics: Some(vec![ToolDiagnostic {
                            severity: ToolDiagnosticSeverity::Error,
                            code: Some("tool_error".to_string()),
                            message,
                        }]),
                        ..Default::default()
                    },
                }
            })
        },
    );
    define_tool(ToolRegistration {
		tool: pi_ai::Tool {
			name: "search".into(),
			description: "Search travel information about a city. Topics: weather, museums, trains. Slow; call \
				several in parallel."
				.into(),
			parameters: json!({
				"type": "object",
				"properties": {
					"topic": { "enum": ["weather", "museums", "trains"] },
					"city": { "type": "string" },
				},
				"required": ["topic", "city"],
			}),
			constrained_sampling: None,
		},
		// Only reads, so a call cut off by a crash simply runs again.
		replay: Some(ToolReplay::Safe),
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute,
	})
}

async fn run_search(
    args: Value,
    api: &dyn ToolExecutionApi,
    context: &Context,
) -> Result<ToolExecutionResult, String> {
    let topic = args
        .get("topic")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let city = args
        .get("city")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let Some((seconds, results)) = searches_for(&topic) else {
        return Err(format!("Unknown search topic: {topic}"));
    };
    let steps = seconds.div_ceil(2);
    for step in 1..=steps {
        api.output(
            format!("searching {topic} in {city}: source {step}/{steps}\n").as_bytes(),
            None,
        );
        search_step(context).await?;
    }
    for result in results {
        api.output(format!("- {result}\n").as_bytes(), None);
    }
    Ok(ToolExecutionResult::default())
}

/// What the research subagent gets: only `search` (upstream `Search`).
pub fn search_extension() -> pi_durable::harness::types::Extension {
    pi_durable::harness::types::Extension::new("vacation-search")
        .with_tools(vec![Arc::new(search_tool())])
}

// ─── research: a background subagent that reports back ──────────────────────

/// Delivers the task to its subagent (upstream the `deliver` phase). The
/// subagent's conversation is owned by this task; no Session call may run
/// inside a commit.
async fn deliver_phase(
    task: pi_durable::types::TaskRecord,
    runtime: Arc<dyn TaskRuntime>,
    context: Context,
) -> Result<(), String> {
    let owned: parking_lot::Mutex<Option<ConversationId>> = parking_lot::Mutex::new(None);
    runtime
        .commit(
            commit_change(|tx: &mut Tx<'_>, _running| {
                let found = tx
                    .scan_conversations(
                        &ConversationQuery {
                            owner_task_id: Some(task.id),
                            ..Default::default()
                        },
                        1,
                        None,
                    )
                    .map_err(|error| error.to_string())?
                    .items
                    .into_iter()
                    .next()
                    .map(|record| record.id);
                *owned.lock() = found;
                Ok(None)
            }),
            &context,
        )
        .await?;
    let owned = owned
        .into_inner()
        .ok_or_else(|| "Research task owns no conversation".to_string())?;
    let child = runtime
        .conversation(owned, &context)
        .await?
        .ok_or_else(|| format!("Research conversation {owned} does not exist"))?;
    // A rerun after a crash gets the same submission back.
    let request = pi_durable::harness::types::InputSubmissionDraft {
        request_id: Some(format!("research:{}", task.id)),
        content: pi_durable::harness::types::UserInput::Text(
            task.input
                .get("task")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        when_busy: None,
    };
    let settled = child
        .submit(request, &context)
        .await?
        .wait(&context)
        .await?;
    runtime
        .commit(
            commit_change(move |tx: &mut Tx<'_>, _running| {
                let report = if settled.status == pi_durable::types::SubmissionStatus::Done
                    && settled.submission_type == pi_durable::types::SubmissionType::Input
                {
                    let entry = settled
                        .answer
                        .map(|answer| tx.entry(answer).map_err(|error| error.to_string()))
                        .transpose()?
                        .flatten();
                    format!(
                        "[research report] {}",
                        entry.as_ref().map(assistant_text).unwrap_or_default()
                    )
                } else {
                    let reason =
                        if settled.status == pi_durable::types::SubmissionStatus::Unanswered {
                            settled.reason.clone().unwrap_or_else(|| "?".to_string())
                        } else {
                            "?".to_string()
                        };
                    format!("[research report] The research failed: {reason}")
                };
                Ok(Some(NextTaskState::Running {
                    checkpoint: json!({ "phase": "report", "report": report }),
                }))
            }),
            &context,
        )
        .await?;
    Ok(())
}

/// Posts the report to the main conversation as a new message (upstream the
/// `report` phase).
async fn report_phase(
    task: pi_durable::types::TaskRecord,
    runtime: Arc<dyn TaskRuntime>,
    context: Context,
) -> Result<(), String> {
    let report = task
        .state
        .checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.get("report"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let main = runtime
        .conversation(runtime.conversation_id(), &context)
        .await?
        .ok_or_else(|| "Research task's conversation does not exist".to_string())?;
    main.submit(
        pi_durable::harness::types::InputSubmissionDraft {
            request_id: Some(format!("research-report:{}", task.id)),
            content: pi_durable::harness::types::UserInput::Text(report),
            when_busy: Some(pi_durable::harness::types::WhenBusy::FollowUp),
        },
        &context,
    )
    .await?;
    runtime
        .commit(
            commit_change(|_tx: &mut Tx<'_>, _running| {
                Ok(Some(NextTaskState::Terminal {
                    outcome: pi_durable::types::TaskOutcome::completed(Value::Null),
                }))
            }),
            &context,
        )
        .await?;
    Ok(())
}

async fn abort_phase(
    _task: pi_durable::types::TaskRecord,
    runtime: Arc<dyn TaskRuntime>,
    context: Context,
) -> Result<(), String> {
    runtime
        .commit(
            commit_change(|_tx: &mut Tx<'_>, _running| {
                Ok(Some(NextTaskState::Terminal {
                    outcome: pi_durable::types::TaskOutcome::aborted(None),
                }))
            }),
            &context,
        )
        .await?;
    Ok(())
}

/// The upstream `Research` task definition (`vacation.research`).
pub fn research_task() -> AnyTask {
    let deliver: PhaseHandler =
        Arc::new(move |task, runtime, context| Box::pin(deliver_phase(task, runtime, context)));
    let report: PhaseHandler =
        Arc::new(move |task, runtime, context| Box::pin(report_phase(task, runtime, context)));
    let abort: PhaseHandler =
        Arc::new(move |task, runtime, context| Box::pin(abort_phase(task, runtime, context)));
    AnyTask::new(TaskDefinition {
        name: "vacation.research".into(),
        version: 1,
        initial: Arc::new(|_input: &Value| json!({ "phase": "deliver" })),
        phases: Arc::new(BTreeMap::from([
            ("deliver".to_string(), deliver),
            ("report".to_string(), report),
        ])),
        abort,
        migrate: None,
        hooks: None,
    })
}

const RESEARCH_INSTRUCTIONS: &str =
    "You are a research subagent for a vacation planner. Search weather, museums, \
	and trains in parallel, in one step, then report the findings in a few short bullet points.";

fn research_tool(research: AnyTask) -> ToolRegistration {
    let execute: ToolExecute = Arc::new(
        move |args: Value, api: Arc<dyn ToolExecutionApi>, context: Context| {
            let research = research.clone();
            Box::pin(async move {
                match start_research(args, research, api.as_ref(), &context).await {
                    Ok(result) => result,
                    Err(message) => ToolExecutionResult {
                        is_error: Some(true),
                        diagnostics: Some(vec![ToolDiagnostic {
                            severity: ToolDiagnosticSeverity::Error,
                            code: Some("tool_error".to_string()),
                            message,
                        }]),
                        ..Default::default()
                    },
                }
            })
        },
    );
    define_tool(ToolRegistration {
		tool: pi_ai::Tool {
			name: "research".into(),
			description: "Start a research subagent in the background. It searches while you keep talking with the \
				user; its report arrives later as a message starting with [research report]."
				.into(),
			parameters: json!({
				"type": "object",
				"properties": {
					"task": { "type": "string", "description": "What to research, with the city and dates" },
				},
				"required": ["task"],
			}),
			constrained_sampling: None,
		},
		replay: None,
		execution_mode: None,
		prepare_arguments: None,
		output_limits: None,
		execute,
	})
}

async fn start_research(
    args: Value,
    research: AnyTask,
    api: &dyn ToolExecutionApi,
    context: &Context,
) -> Result<ToolExecutionResult, String> {
    let task = args
        .get("task")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    api.commit(
        &mut Box::new(move |tx: &mut Tx<'_>| {
            // Background: the main conversation stays free while the research
            // runs.
            let owner = tx
                .create_task(
                    &research,
                    json!({ "task": task }),
                    TaskOptions {
                        ownership: TaskOwnership::Conversation,
                        conversation_id: None,
                        background: Some(true),
                    },
                )
                .map_err(|error| error.to_string())?;
            let child = tx
                .create_conversation(ConversationOwnership::Task { task_id: owner })
                .map_err(|error| error.to_string())?;
            configure(
                tx,
                child.id,
                &AgentChange {
                    extensions: Some(SetOrClear::Set(ExtensionsChange::List(vec![
                        search_extension(),
                    ]))),
                    instructions: Some(SetOrClear::Set(RESEARCH_INSTRUCTIONS.to_string())),
                    ..AgentChange::default()
                },
            )?;
            Ok(Value::Null)
        }),
        context,
    )
    .await?;
    Ok(ToolExecutionResult {
        content: Some(json!([{ "type": "text", "text": "Research started in the background." }])),
        ..Default::default()
    })
}

fn preamble_section() -> PromptSection {
    let render: SectionRender = Arc::new(|_input, _context| {
        Box::pin(async move {
            Ok(Some(
				"You are a friendly vacation planning assistant. Keep answers short. Hand all research to the \
				 research tool; while it runs, keep chatting with the user. When a [research report] arrives, turn \
				 it into a short plan."
					.to_string(),
			))
        })
    });
    section("preamble", render, Some(false))
}

/// What the main conversation gets: the prompt, `research`, and the task
/// that delivers its report (upstream `Vacation`).
pub fn vacation_extension() -> pi_durable::harness::types::Extension {
    pi_durable::harness::types::Extension::new("vacation")
        .with_tools(vec![Arc::new(research_tool(research_task()))])
        .with_tasks(vec![research_task()])
        .with_sections(vec![preamble_section()])
}
