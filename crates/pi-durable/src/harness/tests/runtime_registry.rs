use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::{ModelThinkingLevel, Tool};
use serde_json::json;

use crate::harness::agent::{agent_hooks, resolve_agent, resolve_settings};
use crate::harness::api::ToolExecutionResult;
use crate::harness::define::{define_tool_of, hook, section, wrap_section, wrap_tool};
use crate::harness::registry::{builtin_tasks, create_registry};
use crate::harness::types::{
    Agent, AgentState, AnyTask, DocumentReader, Extension, ExtensionSelection, HarnessSettings,
    ModelRef, PromptInput, Registry, RegistryReader, RegistrySnapshot, ToolRegistration,
    ToolsSelection,
};
use crate::tasks::{HookHandler, PhaseHandler, TaskDefinition};

fn tool(name: &str, description: &str) -> Arc<ToolRegistration> {
    Arc::new(define_tool_of(ToolRegistration {
        tool: Tool {
            name: name.to_string(),
            description: description.to_string(),
            parameters: json!({"type":"object","properties":{}}),
            constrained_sampling: None,
        },
        replay: None,
        execution_mode: None,
        prepare_arguments: None,
        output_limits: None,
        execute: Arc::new(|_, _, _| Box::pin(async { ToolExecutionResult::default() })),
    }))
}

fn section_text(key: &str, text: &str, tag: Option<bool>) -> crate::harness::types::PromptSection {
    let text = text.to_string();
    section(
        key,
        Arc::new(move |_, _| {
            let text = text.clone();
            Box::pin(async move { Ok(Some(text)) })
        }),
        tag,
    )
}

fn task(name: &str, version: u64) -> AnyTask {
    let phase: PhaseHandler = Arc::new(|_, _, _| Box::pin(async { Ok(()) }));
    AnyTask::new(TaskDefinition {
        name: name.to_string(),
        version,
        initial: Arc::new(|_| json!({"phase":"run"})),
        phases: Arc::new(BTreeMap::from([("run".into(), phase.clone())])),
        abort: phase,
        migrate: None,
        hooks: None,
    })
}

fn extension_names(extensions: &[Arc<Extension>]) -> Vec<&str> {
    extensions
        .iter()
        .map(|extension| extension.name.as_str())
        .collect()
}

fn task_names(tasks: &[AnyTask]) -> Vec<&str> {
    tasks.iter().map(AnyTask::name).collect()
}

fn resolve(
    state: Option<&AgentState>,
    snapshot: &dyn RegistrySnapshot,
    settings: Option<&HarnessSettings>,
    reports: &mut Vec<String>,
) -> Agent {
    let settings = resolve_settings(settings);
    let reports = std::cell::RefCell::new(reports);
    resolve_agent(state, snapshot, &settings, &|error| {
        reports.borrow_mut().push(error)
    })
}

fn panic_message(result: std::thread::Result<()>) -> String {
    let payload = result.expect_err("registry install should reject this extension");
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        panic!("unexpected panic payload")
    }
}

#[test]
fn invalid_registry_states_publish_nothing_and_replacement_revalidates_the_resulting_state() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let registry = create_registry();
    registry.install(Extension::new("tasks").with_tasks(vec![task("app.index", 1)]));
    let published = Arc::new(AtomicUsize::new(0));
    let published_from_listener = published.clone();
    let _unsubscribe = registry.subscribe(Arc::new(move || {
        published_from_listener.fetch_add(1, Ordering::SeqCst);
    }));
    let before = registry.snapshot();
    let read = tool("read", "read tool");
    let invalid = [
        (
            Extension::new("x").with_tools(vec![read.clone(), tool("read", "duplicate read")]),
            "Extension x has two tools named read",
        ),
        (
            Extension::new("x").with_sections(vec![
                section_text("a", "first", None),
                section_text("a", "second", None),
            ]),
            "Extension x has two sections with key a",
        ),
        (
            Extension::new("x").with_sections(vec![section_text("Bad Key", "", None)]),
            "Section key \"Bad Key\" must match /^[a-z][a-z0-9_-]*$/",
        ),
        (
            Extension::new("x").with_sections(vec![section_text("instructions", "", None)]),
            "Section key instructions is reserved for the agent's instructions",
        ),
        (
            Extension::new("x").with_tasks(vec![task("pi.generation", 1)]),
            "Task pi.generation of extension x is already installed",
        ),
        (
            Extension::new("x").with_tasks(vec![task("app.index", 1)]),
            "Task app.index of extension x is already installed",
        ),
    ];
    for (extension, expected) in invalid {
        let message = panic_message(catch_unwind(AssertUnwindSafe(|| {
            registry.install(extension)
        })));
        assert_eq!(message, expected);
    }
    assert!(Arc::ptr_eq(&before, &registry.snapshot()));
    assert_eq!(extension_names(&registry.snapshot().installed()), ["tasks"]);
    assert_eq!(
        registry
            .snapshot()
            .task("app.index")
            .unwrap()
            .definition
            .version,
        1
    );
    assert_eq!(published.load(Ordering::SeqCst), 0);

    registry.install(Extension::new("tasks").with_tasks(vec![task("app.index", 2)]));
    assert_eq!(extension_names(&registry.snapshot().installed()), ["tasks"]);
    assert_eq!(
        registry
            .snapshot()
            .task("app.index")
            .unwrap()
            .definition
            .version,
        2
    );
    assert_eq!(before.task("app.index").unwrap().definition.version, 1);
    assert_eq!(published.load(Ordering::SeqCst), 1);
}

#[test]
fn builtins_exist_without_being_installed_as_extensions() {
    let registry = create_registry();
    let snapshot = registry.snapshot();
    assert!(snapshot.installed().is_empty());
    for builtin in builtin_tasks() {
        let registered = snapshot
            .task(builtin.name())
            .expect("builtin task is registered");
        assert_eq!(registered.name(), builtin.name());
        assert!(Arc::ptr_eq(&registered.definition, &builtin.definition));
    }
}

#[test]
fn builtin_tasks_are_listed_in_the_upstream_order() {
    let tasks = create_registry().snapshot().tasks();
    assert_eq!(
        task_names(&tasks),
        ["pi.generation", "pi.tool", "pi.compaction"]
    );
}

#[test]
fn custom_tasks_are_available_only_while_their_extension_is_installed() {
    let registry = create_registry();
    let custom = task("app.custom", 1);
    registry.install(Extension::new("custom").with_tasks(vec![custom.clone()]));
    let installed = registry.snapshot();
    let resolved = installed
        .task("app.custom")
        .expect("custom task is registered");
    assert!(Arc::ptr_eq(&resolved.definition, &custom.definition));
    assert_eq!(
        task_names(&installed.tasks()),
        ["pi.generation", "pi.tool", "pi.compaction", "app.custom"]
    );

    registry.uninstall(&Extension::new("custom"));
    let uninstalled = registry.snapshot();
    assert!(uninstalled.task("app.custom").is_none());
    assert!(uninstalled.task("pi.generation").is_some());
    assert_eq!(
        extension_names(&uninstalled.installed()),
        Vec::<&str>::new()
    );
}

#[test]
fn extension_defaults_and_edits_keep_order_and_deduplicate_by_first_name() {
    let registry = create_registry();
    let coding = Extension::new("coding");
    let skills = Extension::new("skills");
    let reviewer = Extension::new("reviewer");
    for extension in [&coding, &skills, &reviewer] {
        registry.install(extension.clone());
    }
    let snapshot = registry.snapshot();
    let mut reports = Vec::new();

    assert_eq!(
        extension_names(&resolve(None, snapshot.as_ref(), None, &mut reports).extensions),
        ["coding", "skills", "reviewer"]
    );

    let settings = HarnessSettings {
        extensions: Some(vec![Arc::new(coding.clone()), Arc::new(skills.clone())]),
        ..HarnessSettings::default()
    };
    assert_eq!(
        extension_names(
            &resolve(None, snapshot.as_ref(), Some(&settings), &mut reports).extensions
        ),
        ["coding", "skills"]
    );
    let explicit = AgentState {
        extensions: Some(ExtensionSelection::List(vec![
            "reviewer".into(),
            "coding".into(),
        ])),
        ..AgentState::default()
    };
    assert_eq!(
        extension_names(
            &resolve(
                Some(&explicit),
                snapshot.as_ref(),
                Some(&settings),
                &mut reports
            )
            .extensions
        ),
        ["reviewer", "coding"]
    );

    let edited = AgentState {
        extensions: Some(ExtensionSelection::Edit {
            add: Some(vec!["reviewer".into(), "coding".into(), "gone".into()]),
            remove: Some(vec!["skills".into()]),
        }),
        ..AgentState::default()
    };
    assert_eq!(
        extension_names(
            &resolve(
                Some(&edited),
                snapshot.as_ref(),
                Some(&settings),
                &mut reports
            )
            .extensions
        ),
        ["coding", "reviewer"]
    );

    let stale_settings = HarnessSettings {
        extensions: Some(vec![Arc::new(Extension::new("skills")), Arc::new(skills)]),
        ..HarnessSettings::default()
    };
    let selected = resolve(None, snapshot.as_ref(), Some(&stale_settings), &mut reports);
    assert_eq!(extension_names(&selected.extensions), ["skills"]);
    assert!(Arc::ptr_eq(
        &selected.extensions[0],
        &snapshot.extension("skills").unwrap()
    ));
    assert!(reports.is_empty());
}

#[test]
fn an_agent_name_resolves_again_after_live_install_and_uninstall() {
    let registry = create_registry();
    registry.install(Extension::new("coding"));
    let state = AgentState {
        extensions: Some(ExtensionSelection::List(vec![
            "coding".into(),
            "skills".into(),
        ])),
        ..AgentState::default()
    };
    let before_install = registry.snapshot();
    let mut reports = Vec::new();
    assert_eq!(
        extension_names(
            &resolve(Some(&state), before_install.as_ref(), None, &mut reports).extensions
        ),
        ["coding"]
    );

    registry.install(Extension::new("skills"));
    let after_install = registry.snapshot();
    assert_eq!(
        extension_names(
            &resolve(Some(&state), after_install.as_ref(), None, &mut reports).extensions
        ),
        ["coding", "skills"]
    );
    assert_eq!(
        extension_names(
            &resolve(Some(&state), before_install.as_ref(), None, &mut reports).extensions
        ),
        ["coding"]
    );

    registry.uninstall(&Extension::new("coding"));
    assert_eq!(
        extension_names(
            &resolve(
                Some(&state),
                registry.snapshot().as_ref(),
                None,
                &mut reports
            )
            .extensions
        ),
        ["skills"]
    );
    registry.install(Extension::new("coding"));
    assert_eq!(
        extension_names(
            &resolve(
                Some(&state),
                registry.snapshot().as_ref(),
                None,
                &mut reports
            )
            .extensions
        ),
        ["coding", "skills"]
    );
    assert!(reports.is_empty());
}

#[test]
fn tools_replace_in_place_then_receive_ordered_wrappers_before_filters() {
    use parking_lot::Mutex;

    let registry = create_registry();
    let read = tool("read", "read tool");
    let bash = tool("bash", "bash tool");
    let edit = tool("edit", "edit tool");
    let venv_bash = tool("bash", "venv bash");
    let missing = tool("grep", "grep tool");
    let fallback = missing.clone();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let called = calls.clone();
    let timing = Extension::new("timing").with_wraps(vec![
        wrap_tool(bash.as_ref(), |inner| {
            let mut wrapped = inner.as_ref().clone();
            wrapped.tool.description = format!("{} (timed)", wrapped.tool.description);
            Arc::new(wrapped)
        }),
        wrap_tool(bash.as_ref(), |inner| {
            let mut wrapped = inner.as_ref().clone();
            wrapped.tool.description = format!("{} [2]", wrapped.tool.description);
            Arc::new(wrapped)
        }),
        wrap_tool(missing.as_ref(), move |_| {
            called.lock().push("grep");
            fallback.clone()
        }),
    ]);
    registry.install(Extension::new("coding").with_tools(vec![read, bash, edit]));
    registry.install(Extension::new("venv").with_tools(vec![venv_bash]));
    registry.install(timing);
    let snapshot = registry.snapshot();
    let mut reports = Vec::new();
    let agent = resolve(None, snapshot.as_ref(), None, &mut reports);
    assert_eq!(
        agent
            .tools
            .iter()
            .map(|tool| (tool.tool.name.as_str(), tool.tool.description.as_str()))
            .collect::<Vec<_>>(),
        [
            ("read", "read tool"),
            ("bash", "venv bash (timed) [2]"),
            ("edit", "edit tool")
        ]
    );
    assert!(reports.is_empty());
    assert!(calls.lock().is_empty());

    let listed = AgentState {
        tools: Some(ToolsSelection::List(vec![
            "edit".into(),
            "missing".into(),
            "read".into(),
            "edit".into(),
        ])),
        ..AgentState::default()
    };
    assert_eq!(
        agent_tool_names(&resolve(
            Some(&listed),
            snapshot.as_ref(),
            None,
            &mut reports
        )),
        ["edit", "read"]
    );
    let removed = AgentState {
        tools: Some(ToolsSelection::Remove {
            remove: vec!["bash".into()],
        }),
        ..AgentState::default()
    };
    assert_eq!(
        agent_tool_names(&resolve(
            Some(&removed),
            snapshot.as_ref(),
            None,
            &mut reports
        )),
        ["read", "edit"]
    );
    assert!(reports.is_empty());
}

fn agent_tool_names(agent: &Agent) -> Vec<&str> {
    agent
        .tools
        .iter()
        .map(|tool| tool.tool.name.as_str())
        .collect()
}

#[test]
fn a_tool_wrapper_that_renames_its_target_drops_it_and_reports_the_exact_failure() {
    let registry = create_registry();
    let edit = tool("edit", "edit tool");
    let broken = Extension::new("renamer").with_wraps(vec![wrap_tool(edit.as_ref(), |inner| {
        let mut renamed = inner.as_ref().clone();
        renamed.tool.name = "renamed".into();
        Arc::new(renamed)
    })]);
    registry.install(
        Extension::new("coding")
            .with_tools(vec![tool("read", "read tool"), tool("edit", "edit tool")]),
    );
    registry.install(broken);
    let mut reports = Vec::new();
    let agent = resolve(None, registry.snapshot().as_ref(), None, &mut reports);
    assert_eq!(agent_tool_names(&agent), ["read"]);
    assert_eq!(reports, ["Wrapper renamed edit to renamed"]);
}

#[test]
fn a_panicking_tool_wrapper_is_dropped_and_reported() {
    let registry = create_registry();
    let read = tool("read", "read tool");
    registry.install(
        Extension::new("coding").with_tools(vec![read.clone(), tool("bash", "bash tool")]),
    );
    registry.install(
        Extension::new("broken")
            .with_wraps(vec![wrap_tool(read.as_ref(), |_| panic!("wrapper failed"))]),
    );
    let mut reports = Vec::new();
    let agent = resolve(None, registry.snapshot().as_ref(), None, &mut reports);
    assert_eq!(agent_tool_names(&agent), ["bash"]);
    assert_eq!(reports, ["wrapper failed"]);
}

struct NoDocuments;

#[async_trait::async_trait]
impl DocumentReader for NoDocuments {
    async fn snapshot(
        &self,
        _: &crate::documents::DocToken,
        _: Option<u64>,
        _: Option<&str>,
        _: &pi_chord::context::Context,
    ) -> Result<Option<Arc<serde_json::Map<String, serde_json::Value>>>, String> {
        Ok(None)
    }

    async fn snapshot_as_of(
        &self,
        _: &crate::documents::DocToken,
        _: crate::harness::types::ConversationId,
        _: Option<&str>,
        _: crate::harness::types::EntryId,
        _: &pi_chord::context::Context,
    ) -> Result<Option<serde_json::Map<String, serde_json::Value>>, String> {
        Ok(None)
    }
}

fn prompt_input(agent: &Agent) -> PromptInput {
    PromptInput {
        conversation_id: crate::harness::types::ConversationId(1),
        agent: agent.clone(),
        env: None,
        shown: serde_json::Map::new(),
        read: Arc::new(NoDocuments),
    }
}

#[tokio::test]
async fn sections_replace_in_place_apply_wrappers_in_order_and_leave_instructions_last_unwrapped() {
    let registry = create_registry();
    let coding = Extension::new("coding").with_sections(vec![
        section_text("preamble", "You code.", Some(false)),
        section_text("cwd", "/repo", None),
    ]);
    let skills = Extension::new("skills").with_sections(vec![section_text("skills", "S", None)]);
    let review = Extension::new("override")
        .with_sections(vec![section_text("preamble", "You review.", Some(false))])
        .with_wraps(vec![
            wrap_section("cwd", |inner| {
                let render = inner.render.clone();
                crate::harness::define::section(
                    inner.key,
                    Arc::new(move |input, context| {
                        let rendered = render.clone();
                        Box::pin(async move {
                            Ok(rendered(input, context)
                                .await?
                                .map(|text| format!("{text}!")))
                        })
                    }),
                    inner.tag,
                )
            }),
            wrap_section("instructions", |_| {
                panic!("instructions must not be wrapped")
            }),
        ]);
    registry.install(coding);
    registry.install(skills);
    registry.install(review);
    let mut reports = Vec::new();
    let agent = resolve(
        Some(&AgentState {
            instructions: Some("Be terse.".into()),
            ..AgentState::default()
        }),
        registry.snapshot().as_ref(),
        None,
        &mut reports,
    );
    let mut rendered = Vec::new();
    for item in &agent.sections {
        let text = (item.render)(prompt_input(&agent), super::support::context())
            .await
            .expect("section render");
        rendered.push((item.key.as_str(), text));
    }
    assert_eq!(
        rendered,
        [
            ("preamble", Some("You review.".into())),
            ("cwd", Some("/repo!".into())),
            ("skills", Some("S".into())),
            ("instructions", Some("Be terse.".into()))
        ]
    );
    assert_eq!(agent.sections.last().unwrap().tag, None);
    assert!(reports.is_empty());
}

#[test]
fn a_panicking_section_wrapper_drops_its_target_and_reports_the_exact_failure() {
    let registry = create_registry();
    registry
        .install(Extension::new("coding").with_sections(vec![section_text("cwd", "/repo", None)]));
    registry.install(
        Extension::new("broken").with_wraps(vec![wrap_section("cwd", |_| {
            panic!("section wrapper failed")
        })]),
    );
    let mut reports = Vec::new();
    let agent = resolve(None, registry.snapshot().as_ref(), None, &mut reports);
    assert!(agent.sections.is_empty());
    assert_eq!(reports, ["section wrapper failed"]);
}

#[test]
fn hooks_follow_selected_extension_order_filter_by_task_and_agent_fields_keep_defaults() {
    let registry = create_registry();
    let before_a: Arc<dyn crate::tasks::HookHandler> = Arc::new("before-a");
    let before_b: Arc<dyn crate::tasks::HookHandler> = Arc::new("before-b");
    let on_yield: Arc<dyn crate::tasks::HookHandler> = Arc::new("on-yield");
    registry.install(Extension::new("a").with_hooks(vec![
        hook(&crate::harness::tool::tool_task(), before_a),
        hook(&crate::harness::generation::generation_task(), on_yield),
    ]));
    registry.install(
        Extension::new("b").with_hooks(vec![hook(&crate::harness::tool::tool_task(), before_b)]),
    );
    let snapshot = registry.snapshot();
    let mut reports = Vec::new();
    let default = resolve(None, snapshot.as_ref(), None, &mut reports);
    let hook_names = |handlers: Vec<Arc<dyn crate::tasks::HookHandler>>| {
        handlers
            .iter()
            .map(|handler| {
                *((**handler)
                    .as_any()
                    .downcast_ref::<&'static str>()
                    .expect("hook marker"))
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        hook_names(agent_hooks(&default, "pi.tool")),
        ["before-a", "before-b"]
    );
    let reversed = AgentState {
        extensions: Some(ExtensionSelection::List(vec!["b".into(), "a".into()])),
        ..AgentState::default()
    };
    let reversed = resolve(Some(&reversed), snapshot.as_ref(), None, &mut reports);
    assert_eq!(
        hook_names(agent_hooks(&reversed, "pi.tool")),
        ["before-b", "before-a"]
    );
    let only_b = AgentState {
        extensions: Some(ExtensionSelection::List(vec!["b".into()])),
        ..AgentState::default()
    };
    let only_b = resolve(Some(&only_b), snapshot.as_ref(), None, &mut reports);
    assert!(agent_hooks(&only_b, "pi.generation").is_empty());

    assert!(default.model.is_none());
    assert_eq!(default.thinking_level, ModelThinkingLevel::Off);
    assert!(default.cwd.is_none());
    let configured = resolve(
        Some(&AgentState {
            model: Some(ModelRef {
                provider: "p".into(),
                model_id: "m".into(),
            }),
            thinking_level: Some(ModelThinkingLevel::High),
            cwd: Some("/w".into()),
            ..AgentState::default()
        }),
        snapshot.as_ref(),
        None,
        &mut reports,
    );
    assert_eq!(
        configured.model,
        Some(ModelRef {
            provider: "p".into(),
            model_id: "m".into()
        })
    );
    assert_eq!(configured.thinking_level, ModelThinkingLevel::High);
    assert_eq!(configured.cwd.as_deref(), Some("/w"));
    assert!(reports.is_empty());
}
