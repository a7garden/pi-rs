//! Port of `packages/agent/test/e2e.test.ts` (faux-provider integration —
//! the upstream file is entirely offline; no live cases exist to defer).

mod common;

use common::*;

use std::sync::Arc;

use parking_lot::Mutex;
use pi_agent::{
    Agent, AgentEvent, AgentInitialState, AgentMessage, AgentOptions, AgentTool, AgentToolResult,
};
use pi_ai::compat::{register_faux_provider, stream_simple, FauxProviderRegistrationCompat};
use pi_ai::providers::faux::{
    faux_assistant_message, faux_text, faux_thinking, faux_tool_call, FauxContent,
    FauxContentBlock, FauxMessageOptions, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions,
};
use pi_ai::types::{
    AssistantContent, AssistantMessage, Message, StopReason, ToolCall, ToolResultMessage,
    TranscriptContext, Usage, UserContent,
};
use serde_json::{json, Value};

// ==== test/utils/calculate.ts port ====

/// Minimal arithmetic evaluator standing in for upstream `new Function(...)`
/// (tests only exercise `+`/`*` expressions).
fn calculate_expression(expression: &str) -> Result<f64, String> {
    fn parse_expr(tokens: &[String], index: &mut usize) -> Result<f64, String> {
        let mut left = parse_term(tokens, index)?;
        while *index < tokens.len() && (tokens[*index] == "+" || tokens[*index] == "-") {
            let operator = tokens[*index].clone();
            *index += 1;
            let right = parse_term(tokens, index)?;
            left = if operator == "+" {
                left + right
            } else {
                left - right
            };
        }
        Ok(left)
    }
    fn parse_term(tokens: &[String], index: &mut usize) -> Result<f64, String> {
        let mut left = parse_atom(tokens, index)?;
        while *index < tokens.len() && (tokens[*index] == "*" || tokens[*index] == "/") {
            let operator = tokens[*index].clone();
            *index += 1;
            let right = parse_atom(tokens, index)?;
            left = if operator == "*" {
                left * right
            } else {
                left / right
            };
        }
        Ok(left)
    }
    fn parse_atom(tokens: &[String], index: &mut usize) -> Result<f64, String> {
        if *index >= tokens.len() {
            return Err("unexpected end of expression".into());
        }
        if tokens[*index] == "(" {
            *index += 1;
            let value = parse_expr(tokens, index)?;
            if *index >= tokens.len() || tokens[*index] != ")" {
                return Err("missing closing paren".into());
            }
            *index += 1;
            return Ok(value);
        }
        tokens[*index]
            .parse::<f64>()
            .map_err(|_| format!("unexpected token: {}", tokens[*index]))
            .inspect(|_| *index += 1)
    }
    fn tokenize(expression: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut number = String::new();
        for character in expression.chars() {
            if character.is_ascii_digit() || character == '.' {
                number.push(character);
            } else if "+-*/()".contains(character) {
                if !number.is_empty() {
                    tokens.push(std::mem::take(&mut number));
                }
                tokens.push(character.to_string());
            } else if !character.is_whitespace() {
                return Vec::new();
            }
        }
        if !number.is_empty() {
            tokens.push(number);
        }
        tokens
    }

    let tokens = tokenize(expression);
    if tokens.is_empty() {
        return Err(format!("Invalid expression: {expression}"));
    }
    let mut index = 0;
    parse_expr(&tokens, &mut index)
}

/// `calculateTool` (test/utils/calculate.ts).
pub fn calculate_tool() -> AgentTool {
    AgentTool::new(
        "calculate",
        "Evaluate mathematical expressions",
        json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string", "description": "The mathematical expression to evaluate" }
            },
            "required": ["expression"]
        }),
        "Calculator",
        tool_execute(|_id, params| {
            let expression = params
                .get("expression")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let result = calculate_expression(expression)?;
            Ok(AgentToolResult::new(
                vec![UserContent::Text(pi_ai::types::TextContent::new(format!(
                    "{expression} = {}",
                    format_number(result)
                )))],
                Value::Null,
            ))
        }),
    )
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

// ==== faux registration harness ====

fn create_faux_registration(
    options: RegisterFauxProviderOptions,
) -> FauxProviderRegistrationCompat {
    register_faux_provider(options)
}

fn default_registration() -> FauxProviderRegistrationCompat {
    register_faux_provider(RegisterFauxProviderOptions::default())
}

fn registration_model(registration: &FauxProviderRegistrationCompat) -> pi_ai::types::Model {
    registration
        .core
        .get_model("faux-1")
        .expect("faux default model")
}

/// Upstream `streamFn: streamSimple` through the crate's StreamFn seam.
fn stream_simple_fn() -> pi_agent::StreamFn {
    Arc::new(|model, context: TranscriptContext, options| {
        Box::pin(async move {
            let plain = pi_ai::types::Context {
                system_prompt: None,
                messages: context.into_messages(),
                tools: None,
            };
            Ok(stream_simple(&model, &plain, options))
        })
    })
}

fn text_of_message(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn text_of_tool_result(message: &ToolResultMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn new_agent(
    _registration: &FauxProviderRegistrationCompat,
    initial_state: AgentInitialState,
) -> Arc<Agent> {
    Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(stream_simple_fn()),
        initial_state,
        ..Default::default()
    }))
}

fn with_model(
    registration: &FauxProviderRegistrationCompat,
    mut state: AgentInitialState,
) -> AgentInitialState {
    state.model = Some(registration_model(registration));
    state
}

// ==== shared scenario bodies ====

async fn basic_prompt(registration: &FauxProviderRegistrationCompat) {
    let agent = new_agent(
        registration,
        with_model(
            registration,
            AgentInitialState {
                system_prompt: Some(
                    "You are a helpful assistant. Keep your responses concise.".into(),
                ),
                thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
                tools: vec![],
                ..Default::default()
            },
        ),
    );

    agent
        .clone()
        .prompt("What is 2+2? Answer with just the number.")
        .await
        .expect("prompt");

    assert!(!agent.is_streaming());
    assert_eq!(agent.messages().len(), 3);
    assert_eq!(agent.messages()[0].role(), "system");
    assert_eq!(agent.messages()[1].role(), "user");
    assert_eq!(agent.messages()[2].role(), "assistant");

    let AgentMessage::Llm(Message::Assistant(assistant)) = &agent.messages()[2] else {
        panic!("expected assistant message");
    };
    assert!(
        text_of_message(assistant).contains('4'),
        "got: {}",
        text_of_message(assistant)
    );
}

async fn tool_execution(registration: &FauxProviderRegistrationCompat) {
    let agent = new_agent(
        registration,
        with_model(
            registration,
            AgentInitialState {
                system_prompt: Some(
                    "You are a helpful assistant. Always use the calculator tool for math.".into(),
                ),
                thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
                tools: vec![calculate_tool()],
                ..Default::default()
            },
        ),
    );

    let pending_during_events: Arc<Mutex<Vec<(&'static str, Vec<String>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    {
        let pending_during_events = pending_during_events.clone();
        let agent_for_listener = agent.clone();
        let agent_weak = Arc::downgrade(&agent);
        let _ = agent_for_listener;
        agent.subscribe(move |event, _signal| {
            let pending_during_events = pending_during_events.clone();
            let agent_weak = agent_weak.clone();
            Box::pin(async move {
                match &event {
                    AgentEvent::ToolExecutionStart { .. } | AgentEvent::ToolExecutionEnd { .. } => {
                        if let Some(agent) = agent_weak.upgrade() {
                            let event_type = event.event_type();
                            pending_during_events.lock().push((
                                event_type,
                                agent.pending_tool_calls().into_iter().collect(),
                            ));
                        }
                    }
                    _ => {}
                }
            })
        });
    }

    agent
        .clone()
        .prompt("Calculate 123 * 456 using the calculator tool.")
        .await
        .expect("prompt");

    assert!(!agent.is_streaming());
    assert!(agent.messages().len() >= 4);
    let tool_result = agent.messages().iter().find_map(|message| match message {
        AgentMessage::Llm(Message::ToolResult(tool_result)) => Some(tool_result.clone()),
        _ => None,
    });
    let tool_result = tool_result.expect("tool result message");
    assert!(text_of_tool_result(&tool_result).contains("123 * 456 = 56088"));

    let final_message = agent.messages().last().cloned().unwrap();
    let AgentMessage::Llm(Message::Assistant(assistant)) = final_message else {
        panic!("expected final assistant message");
    };
    assert!(
        text_of_message(&assistant).contains("56088"),
        "got: {}",
        text_of_message(&assistant)
    );
    assert_eq!(agent.pending_tool_calls().len(), 0);

    let mut observed = pending_during_events.lock();
    observed.iter_mut().for_each(|(_, ids)| ids.sort());
    assert_eq!(
        *observed,
        vec![
            ("tool_execution_start", vec!["calc-1".to_string()]),
            ("tool_execution_end", vec![]),
        ]
    );
}

async fn abort_execution(registration: &FauxProviderRegistrationCompat) {
    let agent = new_agent(
        registration,
        with_model(
            registration,
            AgentInitialState {
                system_prompt: Some("You are a helpful assistant.".into()),
                thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
                tools: vec![],
                ..Default::default()
            },
        ),
    );

    let agent_for_prompt = agent.clone();
    let prompt_task =
        tokio::spawn(async move { agent_for_prompt.prompt("Count slowly from 1 to 20.").await });
    let agent_for_abort = agent.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        agent_for_abort.abort();
    });

    prompt_task
        .await
        .expect("prompt task")
        .expect("aborted runs settle");

    assert!(!agent.is_streaming());
    assert!(agent.messages().len() >= 2);

    let last = agent.messages().last().cloned().unwrap();
    let AgentMessage::Llm(Message::Assistant(assistant)) = last else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert!(assistant.error_message.is_some());
    assert_eq!(agent.error_message(), assistant.error_message);
}

async fn state_updates(registration: &FauxProviderRegistrationCompat) {
    let agent = new_agent(
        registration,
        with_model(
            registration,
            AgentInitialState {
                system_prompt: Some("You are a helpful assistant.".into()),
                thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
                tools: vec![],
                ..Default::default()
            },
        ),
    );

    let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let events = events.clone();
        agent.subscribe(move |event, _signal| {
            let events = events.clone();
            Box::pin(async move {
                events.lock().push(event.event_type());
            })
        });
    }

    agent
        .clone()
        .prompt("Count from 1 to 5.")
        .await
        .expect("prompt");

    let events = events.lock();
    assert!(events.contains(&"agent_start"));
    assert!(events.contains(&"turn_start"));
    assert!(events.contains(&"message_start"));
    assert!(events.contains(&"message_update"));
    assert!(events.contains(&"message_end"));
    assert!(events.contains(&"turn_end"));
    assert!(events.contains(&"agent_end"));
    let index_of = |needle: &str| {
        events
            .iter()
            .position(|event| *event == needle)
            .expect("event present")
    };
    let last_index_of = |needle: &str| {
        events
            .iter()
            .rposition(|event| *event == needle)
            .expect("event present")
    };
    assert!(index_of("agent_start") < index_of("message_start"));
    assert!(index_of("message_start") < index_of("message_end"));
    assert!(index_of("message_end") < last_index_of("agent_end"));

    assert!(!agent.is_streaming());
    assert_eq!(agent.messages().len(), 3);
}

async fn multi_turn_conversation(registration: &FauxProviderRegistrationCompat) {
    let agent = new_agent(
        registration,
        with_model(
            registration,
            AgentInitialState {
                system_prompt: Some("You are a helpful assistant.".into()),
                thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
                tools: vec![],
                ..Default::default()
            },
        ),
    );

    agent
        .clone()
        .prompt("My name is Alice.")
        .await
        .expect("prompt");
    assert_eq!(agent.messages().len(), 3);

    agent
        .clone()
        .prompt("What is my name?")
        .await
        .expect("prompt");
    assert_eq!(agent.messages().len(), 5);

    let AgentMessage::Llm(Message::Assistant(assistant)) = &agent.messages()[4] else {
        panic!("expected assistant message");
    };
    assert!(text_of_message(assistant).to_lowercase().contains("alice"));
}

#[tokio::test]
async fn handles_a_basic_text_prompt() {
    let faux = default_registration();
    faux.core
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxContent::from("4"),
            FauxMessageOptions::default(),
        ))]);
    basic_prompt(&faux).await;
    faux.unregister();
}

#[tokio::test]
async fn executes_tools_and_tracks_pending_tool_calls() {
    let faux = default_registration();
    faux.core.set_responses(vec![
        FauxResponseStep::Message(faux_assistant_message(
            FauxContent::Blocks(vec![
                FauxContentBlock::Text(faux_text("Let me calculate that.")),
                FauxContentBlock::ToolCall(faux_tool_call(
                    "calculate",
                    json!({ "expression": "123 * 456" })
                        .as_object()
                        .cloned()
                        .unwrap(),
                    Some("calc-1"),
                )),
            ]),
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )),
        FauxResponseStep::Message(faux_assistant_message(
            FauxContent::from("The result is 56088."),
            FauxMessageOptions::default(),
        )),
    ]);
    tool_execution(&faux).await;
    faux.unregister();
}

#[tokio::test]
async fn handles_abort_during_streaming() {
    let faux = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(20.0),
        token_size: Some(FauxTokenSize {
            min: Some(2),
            max: Some(2),
        }),
        ..Default::default()
    });
    faux.core.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
		FauxContent::from("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"),
		FauxMessageOptions::default(),
	))]);
    abort_execution(&faux).await;
    faux.unregister();
}

#[tokio::test]
async fn emits_lifecycle_updates_while_streaming() {
    let faux = register_faux_provider(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..Default::default()
    });
    faux.core
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxContent::from("1 2 3 4 5"),
            FauxMessageOptions::default(),
        ))]);
    state_updates(&faux).await;
    faux.unregister();
}

#[tokio::test]
async fn maintains_context_across_multiple_turns() {
    let faux = default_registration();
    faux.core.set_responses(vec![
        FauxResponseStep::Message(faux_assistant_message(
            FauxContent::from("Nice to meet you, Alice."),
            FauxMessageOptions::default(),
        )),
        FauxResponseStep::sync(|context: &TranscriptContext, _options, _state, _model| {
            let has_alice = context.messages().iter().any(|message| {
                match message {
				Message::User(user) => match &user.content {
					pi_ai::types::UserContentField::Text(text) => text.contains("Alice"),
					pi_ai::types::UserContentField::Blocks(blocks) => blocks.iter().any(|block| {
						matches!(block, UserContent::Text(text) if text.text.contains("Alice"))
					}),
				},
				_ => false,
			}
            });
            faux_assistant_message(
                FauxContent::from(if has_alice {
                    "Your name is Alice."
                } else {
                    "I do not know your name."
                }),
                FauxMessageOptions::default(),
            )
        }),
    ]);
    multi_turn_conversation(&faux).await;
    faux.unregister();
}

#[tokio::test]
async fn preserves_thinking_content_blocks() {
    let faux = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![pi_ai::providers::faux::FauxModelDefinition {
            id: "faux-reasoning".into(),
            reasoning: Some(true),
            ..Default::default()
        }],
        ..Default::default()
    });
    faux.core
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxContent::Blocks(vec![
                FauxContentBlock::Thinking(faux_thinking("step by step")),
                FauxContentBlock::Text(faux_text("4")),
            ]),
            FauxMessageOptions::default(),
        ))]);

    let model = faux
        .core
        .get_model("faux-reasoning")
        .expect("faux-reasoning model");
    let agent = Arc::new(Agent::new(AgentOptions {
        stream_fn: Some(stream_simple_fn()),
        initial_state: AgentInitialState {
            system_prompt: Some("You are a helpful assistant.".into()),
            model: Some(model),
            thinking_level: Some(pi_agent::AgentThinkingLevel::Low),
            tools: vec![],
            ..Default::default()
        },
        ..Default::default()
    }));

    agent.clone().prompt("What is 2+2?").await.expect("prompt");

    let AgentMessage::Llm(Message::Assistant(assistant)) = &agent.messages()[2] else {
        panic!("expected assistant message");
    };
    assert_eq!(
        assistant.content,
        vec![
            AssistantContent::Thinking(pi_ai::types::ThinkingContent::new("step by step")),
            AssistantContent::Text(pi_ai::types::TextContent::new("4")),
        ]
    );
    faux.unregister();
}

// ==== Agent.continue() with faux provider ====

#[tokio::test]
async fn continue_throws_when_no_messages_in_context() {
    let faux = default_registration();
    let agent = new_agent(
        &faux,
        with_model(
            &faux,
            AgentInitialState {
                system_prompt: Some("Test".into()),
                ..Default::default()
            },
        ),
    );

    let error = agent
        .clone()
        .continue_()
        .await
        .expect_err("continue rejects");
    assert_eq!(error.0, "No messages to continue from");
    faux.unregister();
}

#[tokio::test]
async fn continue_throws_when_last_message_is_assistant() {
    let faux = default_registration();
    let model = registration_model(&faux);
    let agent = new_agent(
        &faux,
        with_model(
            &faux,
            AgentInitialState {
                system_prompt: Some("Test".into()),
                ..Default::default()
            },
        ),
    );

    let assistant_message = AssistantMessage {
        content: vec![AssistantContent::Text(pi_ai::types::TextContent::new(
            "Hello",
        ))],
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
        timestamp: pi_agent::now_ms(),
    };
    agent.set_messages(vec![AgentMessage::assistant(assistant_message)]);

    let error = agent
        .clone()
        .continue_()
        .await
        .expect_err("continue rejects");
    assert_eq!(error.0, "Cannot continue from message role: assistant");
    faux.unregister();
}

#[tokio::test]
async fn continues_and_gets_a_response_when_last_message_is_user() {
    let faux = default_registration();
    faux.core
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxContent::from("HELLO WORLD"),
            FauxMessageOptions::default(),
        ))]);
    let agent = new_agent(
        &faux,
        with_model(
            &faux,
            AgentInitialState {
                system_prompt: Some(
                    "You are a helpful assistant. Follow instructions exactly.".into(),
                ),
                thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
                tools: vec![],
                ..Default::default()
            },
        ),
    );

    let user_message = pi_ai::types::UserMessage {
        content: pi_ai::types::UserContentField::Blocks(vec![UserContent::Text(
            pi_ai::types::TextContent::new("Say exactly: HELLO WORLD"),
        )]),
        timestamp: pi_agent::now_ms(),
    };
    agent.set_messages(vec![AgentMessage::user(user_message)]);

    agent.clone().continue_().await.expect("continue succeeds");

    assert!(!agent.is_streaming());
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[0].role(), "user");
    assert_eq!(agent.messages()[1].role(), "assistant");

    let AgentMessage::Llm(Message::Assistant(assistant)) = &agent.messages()[1] else {
        panic!("expected assistant message");
    };
    assert!(text_of_message(assistant)
        .to_uppercase()
        .contains("HELLO WORLD"));
    faux.unregister();
}

#[tokio::test]
async fn continues_and_processes_tool_results() {
    let faux = default_registration();
    let model = registration_model(&faux);
    faux.core
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxContent::from("The answer is 8."),
            FauxMessageOptions::default(),
        ))]);
    let agent = new_agent(
		&faux,
		with_model(
			&faux,
			AgentInitialState {
				system_prompt: Some(
					"You are a helpful assistant. After getting a calculation result, state the answer clearly.".into(),
				),
				thinking_level: Some(pi_agent::AgentThinkingLevel::Off),
				tools: vec![calculate_tool()],
				..Default::default()
			},
		),
	);

    let user_message = pi_ai::types::UserMessage {
        content: pi_ai::types::UserContentField::Blocks(vec![UserContent::Text(
            pi_ai::types::TextContent::new("What is 5 + 3?"),
        )]),
        timestamp: pi_agent::now_ms(),
    };

    let assistant_message = AssistantMessage {
        content: vec![
            AssistantContent::Text(pi_ai::types::TextContent::new("Let me calculate that.")),
            AssistantContent::ToolCall(ToolCall {
                id: "calc-1".into(),
                name: "calculate".into(),
                arguments: json!({ "expression": "5 + 3" })
                    .as_object()
                    .cloned()
                    .unwrap(),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: model.api().to_string(),
        provider: model.provider().to_string(),
        model: model.id().to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        duration_ms: None,
        timestamp: pi_agent::now_ms(),
    };

    let tool_result = ToolResultMessage {
        tool_call_id: "calc-1".into(),
        tool_name: "calculate".into(),
        content: vec![UserContent::Text(pi_ai::types::TextContent::new(
            "5 + 3 = 8",
        ))],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        duration_ms: None,
        timestamp: pi_agent::now_ms(),
    };

    agent.set_messages(vec![
        AgentMessage::user(user_message),
        AgentMessage::assistant(assistant_message),
        AgentMessage::tool_result(tool_result),
    ]);

    agent.clone().continue_().await.expect("continue succeeds");

    assert!(!agent.is_streaming());
    assert!(agent.messages().len() >= 4);

    let last = agent.messages().last().cloned().unwrap();
    let AgentMessage::Llm(Message::Assistant(assistant)) = last else {
        panic!("expected assistant message");
    };
    assert!(
        text_of_message(&assistant).contains('8'),
        "got: {}",
        text_of_message(&assistant)
    );
    faux.unregister();
}

// Unused-import guards for types exercised through helper signatures.
#[allow(unused)]
fn _type_guards(_: Vec<ToolCall>, _: Option<pi_ai::providers::faux::FauxCore>, _: Vec<String>) {}
