//! Port of `packages/coding-agent/src/extensions/codemode/tool.ts` — the
//! `codemode` tool definition: schema, model-facing description (the
//! declaration catalog with the OpenCode-style inline budget), loadout
//! presentation (`on`/`only` modes), and the tool/`AgentTool` factories.
//!
//! Tests: upstream has no dedicated `tool.test.ts`; the behavioral coverage
//! lives in `suite/agent-session-codemode.test.ts` (deferred with the
//! AgentSession integration, see [`super`] docs) and in the tool-search
//! port's tests, which exercise [`create_codemode_description`].

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::Arc;

use pi_agent::AgentTool;
use pi_ai::types::{ConstrainedSamplingConfig, ConstrainedSamplingSetting, GrammarFormat};
use pi_codemode::declarations::{
    mcp_structured_content_schema, render_tool_output_type, render_tool_sample, SampleTool,
};
use pi_codemode::{to_codemode_identifier, CodemodeJsonSchema, CODEMODE_SOURCE_GRAMMAR};
use serde::Serialize;
use serde_json::{json, Value};

use crate::core::extensions::types::{
    PrepareLoadoutFn, ToolDefinition, ToolExposure, ToolInfo, ToolLoadout, ToolLoadoutChanges,
    ToolNamespace,
};
use crate::core::settings_manager::CodemodeMode;
use crate::core::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::extensions::codemode::execute::execute_codemode;
use crate::utils::paths::join;

pub const CODEMODE_TOOL_NAME: &str = "codemode";

/// Custom entry type holding one script's `store()` writes:
/// [`CodemodeStoreEntryData`].
pub const CODEMODE_STORE_ENTRY_TYPE: &str = "codemode-store";

/// `store(key, value)`/`load(key)` persistence payload of one script
/// (`CodemodeStoreEntryData`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeStoreEntryData {
    pub set: serde_json::Map<String, Value>,
    pub delete: Vec<String>,
}

/// The part of the model registry that scripts reach through `models`
/// (upstream `Pick<ModelRegistry, ...>`). The Rust registry already exposes
/// exactly that surface, so the alias carries the real type.
pub type CodemodeModelRuntime = crate::core::model_registry::ModelRegistry;

/// Overrides and hooks of the codemode tool (`CodemodeToolOptions`).
#[derive(Clone, Default)]
pub struct CodemodeToolOptions {
    /// Namespace of a tool, for `searchTools()` ranking and its `namespace`
    /// filter.
    pub get_tool_namespace: Option<Arc<dyn Fn(&str) -> Option<ToolNamespace> + Send + Sync>>,
    /// Prompt guidelines of every tool, by tool name, shown with declarations
    /// by `describeTool()` and `ALL_TOOLS`.
    pub get_tool_guidelines: Option<Arc<dyn Fn() -> BTreeMap<String, Vec<String>> + Send + Sync>>,
    /// Expose the `models` namespace to scripts, backed by the session's
    /// model registry (`ctx.model_registry`). Without it, `models` is not
    /// declared.
    pub models: Option<bool>,
    /// Persists `store()` writes as a session custom entry. Without it,
    /// writes last only for the current script; `load()` still reads entries
    /// already on the branch.
    pub append_entry: Option<Arc<dyn Fn(&str, Option<Value>) + Send + Sync>>,
    /// How the tool presents the loadout while active (the `codemode.mode`
    /// setting). Default: [`CodemodeMode::On`].
    pub get_mode: Option<Arc<dyn Fn() -> CodemodeMode + Send + Sync>>,
    /// Token budget for tool declarations in the description. Default:
    /// [`DEFAULT_CODEMODE_INLINE_BUDGET`].
    pub get_inline_budget: Option<Arc<dyn Fn() -> Option<f64> + Send + Sync>>,
    /// D91: QuickJS wasm loader injected by pi-runtime (upstream
    /// `loadQuickJSWasm(getQuickJSWasmPath())`). Default: the pi-codemode
    /// unavailable loader (sandbox error, like a missing wasm file).
    pub wasm_loader: Option<Arc<dyn pi_codemode::wasm::WasmLoader>>,
    /// D91: per-execution worker spawner injected by pi-runtime (upstream
    /// `getCodemodeWorkerSpecifier()`). Default: the pi-codemode no-worker
    /// factory (sandbox error, like a missing worker file).
    pub worker_factory: Option<Arc<dyn pi_codemode::runtime::host::CodeModeWorkerFactory>>,
}

impl std::fmt::Debug for CodemodeToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodemodeToolOptions")
            .field("models", &self.models)
            .finish_non_exhaustive()
    }
}

/// Parameter schema of the codemode tool (`codemodeSchema`; upstream TypeBox,
/// carried as JSON Schema — core extensions D27). Upstream `isCodemodeTool`
/// compares schemas by reference; here they compare by value.
pub fn codemode_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "code": { "type": "string", "description": "Raw JavaScript source." }
        },
        "required": ["code"]
    })
}

/// Validated input of the codemode tool (`CodemodeToolInput`).
#[derive(Debug, Clone)]
pub struct CodemodeToolInput {
    pub code: String,
}

/// Whether a registered tool is this package's `codemode` tool rather than
/// another extension's tool with the same name (`isCodemodeTool`). Compares
/// the parameter schema (upstream: by reference; here: by value).
pub fn is_codemode_tool(tool: &ToolInfo) -> bool {
    tool.name == CODEMODE_TOOL_NAME && tool.parameters == codemode_schema()
}

/// Status of one nested call on the tool result (`CodemodeNestedCallStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CodemodeNestedCallStatus {
    Running,
    Ok,
    Error,
    Cancelled,
}

/// One nested tool or `models.*` call a script made
/// (`CodemodeNestedCall`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeNestedCall {
    /// Tool call id of the nested call, `<codemode call id>/<n>`.
    pub id: String,
    pub name: String,
    /// Compact JSON of the arguments, truncated for display.
    pub args: String,
    pub status: CodemodeNestedCallStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    /// Error text, truncated for display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Cost in USD of a `models.*` call that reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

/// Structured details of a codemode tool result (`CodemodeToolDetails`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodemodeToolDetails {
    pub calls: Vec<CodemodeNestedCall>,
    /// Temp file with the full text output, when the output was truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// Snippet and guidelines the codemode tool contributes to the default
/// system prompt (`codemodeToolSystemPromptContribution`).
pub const CODEMODE_TOOL_SYSTEM_PROMPT_CONTRIBUTION: CodemodeToolSystemPromptContribution =
	CodemodeToolSystemPromptContribution {
		snippet: "Run JavaScript that calls other tools",
		guidelines: &[
			"Use codemode to batch independent tool calls (Promise.allSettled), chain them, or filter large output, instead of many separate calls.",
		],
	};

/// Type of [`CODEMODE_TOOL_SYSTEM_PROMPT_CONTRIBUTION`].
pub struct CodemodeToolSystemPromptContribution {
    pub snippet: &'static str,
    pub guidelines: &'static [&'static str],
}

/// The reference for scripts: globals, tool results, `store()`, the `models`
/// API, and limits (`CODEMODE_DOCS_PATH`).
pub fn codemode_docs_path() -> String {
    join(&crate::config::get_docs_path(), "codemode.md")
}

const DESCRIPTION_INTRO: &str = "Run JavaScript that calls other tools. The input is raw JavaScript (not JSON, no code fence), run as an async function body in a QuickJS sandbox: top-level `await` and `return` work. No Node, file system, network, or timers.\n- `await tools.<name>({ ...args })` resolves to a string, or an object if the tool's declaration says so, and rejects with an Error on failure. Calls still running when the script ends are cancelled.\n- Optional first line: `// @options: {\"max_output_tokens\": 10000, \"timeout_ms\": 60000}`";

/// One line per global. The details live in [`codemode_docs_path`].
fn describe_globals(models: bool) -> String {
    let mut lines = vec![
		"Globals:".to_string(),
		"- `text(value)`, `image(dataUrlOrImageBlock)`, `console.log(...)`, and top-level `return` add output; `exit()` ends the script. With several text items, each starts with a `==> text N/M <==` line, and `console` lines follow the other output in one `<console_output>` block. `image()` also saves the image to a temp file and the result names its path.".to_string(),
		"- `store(key, value)` and `load(key)` keep JSON values across codemode calls.".to_string(),
		"- `ALL_TOOLS`, `await searchTools(query, { limit?, namespace? })`, `await describeTool(name)`, `await describeNamespace(name)`: find unlisted tools, such as MCP tools.".to_string(),
	];
    if models {
        lines.push(format!(
            "- `models`: classifiers and image generation. Read {} first.",
            codemode_docs_path()
        ));
    }
    lines.join("\n")
}

/// Default inline budget, in estimated tokens
/// (`DEFAULT_CODEMODE_INLINE_BUDGET`).
pub const DEFAULT_CODEMODE_INLINE_BUDGET: usize = 3000;
/// Characters per token when estimating the cost of a tool section.
const CHARS_PER_TOKEN: usize = 4;

const TEXT_OUTPUT_SCHEMA: &str = r#"{"type":"string"}"#;

/// What a script sees of a tool (`CodemodeTool` minus `execute`). Tools
/// without an output schema resolve to their text output.
#[derive(Debug, Clone, PartialEq)]
pub struct CodemodeDeclaration {
    pub name: String,
    pub description: String,
    pub input_schema: CodemodeJsonSchema,
    pub output_schema: CodemodeJsonSchema,
}

/// What a script sees of a tool: its description followed by its prompt
/// guidelines, which the system prompt only has for declared tools. Tools
/// without an output schema resolve to their text output
/// (`toCodemodeDeclaration`).
pub fn to_codemode_declaration(tool: &AgentTool, guidelines: &[String]) -> CodemodeDeclaration {
    let bullets: Vec<String> = guidelines
        .iter()
        .filter_map(|guideline| {
            let trimmed = guideline.trim();
            (!trimmed.is_empty()).then(|| format!("- {trimmed}"))
        })
        .collect();
    CodemodeDeclaration {
        name: tool.tool.name.clone(),
        description: if bullets.is_empty() {
            tool.tool.description.clone()
        } else {
            format!("{}\n\n{}", tool.tool.description.trim(), bullets.join("\n"))
        },
        input_schema: tool.tool.parameters.clone(),
        output_schema: tool.output_schema.clone().unwrap_or_else(|| {
            serde_json::from_str(TEXT_OUTPUT_SCHEMA).expect("text output schema parses")
        }),
    }
}

/// Tools a script may call: every given tool except the codemode tool itself
/// (`getCodemodeCallableTools`).
pub fn get_codemode_callable_tools(tools: &[AgentTool]) -> Vec<AgentTool> {
    tools
        .iter()
        .filter(|tool| tool.name() != CODEMODE_TOOL_NAME)
        .cloned()
        .collect()
}

/// Options of [`create_codemode_description`]
/// (`CodemodeDescriptionOptions`).
#[derive(Debug, Clone, Default)]
pub struct CreateCodemodeDescriptionOptions {
    /// Declare the `models` namespace; only for tools created with model
    /// access.
    pub models: bool,
    /// Namespace of each tool, by tool name. Tools of one namespace are
    /// listed under one heading.
    pub namespaces: BTreeMap<String, ToolNamespace>,
    /// Prompt guidelines of each tool, by tool name, listed after its
    /// description.
    pub guidelines: BTreeMap<String, Vec<String>>,
    /// Tools that are callable but never listed with their declaration
    /// (`deferred` exposure).
    pub deferred: BTreeSet<String>,
    /// Estimated tokens (characters / 4) the tool sections may use. Tools
    /// that do not fit are left out, like deferred tools. `None` lists every
    /// tool that is not deferred.
    pub inline_budget: Option<usize>,
}

/// `### \`id\` (\`raw name\`)` followed by the tool's description and
/// declaration (`renderToolSection`).
fn render_tool_section(declaration: &CodemodeDeclaration) -> String {
    let id = to_codemode_identifier(&declaration.name);
    let heading = if id == declaration.name {
        format!("### `{id}`")
    } else {
        format!("### `{id}` (`{}`)", declaration.name)
    };
    let sample = render_tool_sample(&SampleTool {
        name: &declaration.name,
        description: Some(&declaration.description),
        input_schema: Some(&declaration.input_schema),
        output_schema: Some(&declaration.output_schema),
        input_max_chars: None,
    });
    format!("{heading}\n{}", sample.trim())
}

#[derive(Clone)]
struct CatalogEntry {
    name: String,
    section: String,
    cost: usize,
}

struct CatalogGroup {
    namespace: Option<ToolNamespace>,
    entries: Vec<CatalogEntry>,
}

/// Pick the tool sections that fit the budget, like OpenCode's catalog: in
/// each round every group (tools without a namespace first, then namespaces
/// by name) places its cheapest remaining tool; a group whose next tool does
/// not fit drops out while the others continue. Every namespace is
/// represented before any namespace is complete (`selectCatalog`).
fn select_catalog(groups: &[CatalogGroup], budget: Option<usize>) -> HashSet<String> {
    let Some(budget) = budget else {
        return groups
            .iter()
            .flat_map(|group| group.entries.iter().map(|entry| entry.name.clone()))
            .collect();
    };
    let mut queues: Vec<VecDeque<CatalogEntry>> = groups
        .iter()
        .map(|group| {
            let mut entries = group.entries.clone();
            // JS sort is stable, so equal costs keep declaration order.
            entries.sort_by_key(|entry| entry.cost);
            entries.into()
        })
        .collect();
    let mut shown = HashSet::new();
    let mut remaining = budget;
    let mut active: Vec<VecDeque<CatalogEntry>> =
        queues.drain(..).filter(|queue| !queue.is_empty()).collect();
    loop {
        active.retain_mut(|queue| {
            let next = queue.front().expect("queue is non-empty");
            if next.cost > remaining {
                return false;
            }
            remaining -= next.cost;
            shown.insert(next.name.clone());
            queue.pop_front();
            !queue.is_empty()
        });
        if active.is_empty() {
            break;
        }
    }
    shown
}

/// Model-facing description: the helper list, guidance for finding tools
/// that are not listed, the shared MCP types when listed tools need them,
/// the `models` API, and one section per listed tool, grouped by namespace.
/// Deferred tools are never listed and do not affect the description at all,
/// so it stays the same while MCP servers connect or change their tools. Tool
/// sections are limited to `inline_budget` (`createCodemodeDescription`).
pub fn create_codemode_description(
    tools: &[AgentTool],
    options: CreateCodemodeDescriptionOptions,
) -> String {
    let declarations: Vec<CodemodeDeclaration> = get_codemode_callable_tools(tools)
        .into_iter()
        .filter(|tool| !options.deferred.contains(tool.name()))
        .map(|tool| {
            let guidelines = options
                .guidelines
                .get(tool.name())
                .cloned()
                .unwrap_or_default();
            to_codemode_declaration(&tool, &guidelines)
        })
        .collect();
    // Insertion order of the upstream Map is irrelevant: `ordered` re-sorts
    // fully (no-namespace group first, then namespace name), and names are
    // unique per group key.
    let mut groups: BTreeMap<String, CatalogGroup> = BTreeMap::from([(
        String::new(),
        CatalogGroup {
            namespace: None,
            entries: Vec::new(),
        },
    )]);
    for declaration in &declarations {
        let namespace = options.namespaces.get(&declaration.name).cloned();
        let key = namespace
            .as_ref()
            .map_or(String::new(), |ns| format!("ns:{}", ns.name));
        let group = groups.entry(key).or_insert_with(|| CatalogGroup {
            namespace: namespace.clone(),
            entries: Vec::new(),
        });
        let section = render_tool_section(declaration);
        group.entries.push(CatalogEntry {
            name: declaration.name.clone(),
            cost: section.chars().count().div_ceil(CHARS_PER_TOKEN),
            section,
        });
    }
    let mut ordered: Vec<CatalogGroup> = groups.into_values().collect();
    ordered.sort_by(|a, b| match (&a.namespace, &b.namespace) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        // Upstream `localeCompare`; the port compares by Unicode scalar values
        // (deviation D90: no locale collation).
        (Some(a), Some(b)) => a.name.cmp(&b.name),
    });
    let shown = select_catalog(&ordered, options.inline_budget);

    let mut sections = vec![
        DESCRIPTION_INTRO.to_string(),
        describe_globals(options.models),
    ];
    if declarations.iter().any(|declaration| {
        shown.contains(&declaration.name)
            && mcp_structured_content_schema(Some(&declaration.output_schema)).is_some()
    }) {
        sections.push(format!(
            "Shared MCP Types:\n```ts\n{}\n```",
            pi_codemode::MCP_TYPESCRIPT_PREAMBLE
        ));
    }
    if declarations.is_empty() {
        return sections.join("\n\n");
    }

    let mut tool_sections = vec!["Nested tools:".to_string()];
    for group in &ordered {
        let visible: Vec<&CatalogEntry> = group
            .entries
            .iter()
            .filter(|entry| shown.contains(&entry.name))
            .collect();
        if let Some(namespace) = &group.namespace {
            // Only tools that did not fit the budget are counted as not listed here.
            let listing = if visible.len() == group.entries.len() {
                ""
            } else if visible.is_empty() {
                " (tools not listed)"
            } else {
                " (some tools not listed)"
            };
            let description = namespace
                .description
                .as_deref()
                .map(str::trim)
                .filter(|d| !d.is_empty());
            tool_sections.push(match description {
                Some(description) => format!("## {}{listing}\n{description}", namespace.name),
                None => format!("## {}{listing}", namespace.name),
            });
        }
        for entry in visible {
            tool_sections.push(entry.section.clone());
        }
    }
    sections.push(tool_sections.join("\n\n"));
    sections.join("\n\n")
}

/// What a script call resolves to, in one line: `a string`, the field names
/// of an object (`{ output, exit_code, full_output_path? }`), or the rendered
/// type for anything else (`describeOutput`).
fn describe_output(schema: Option<&CodemodeJsonSchema>) -> String {
    let type_ = render_tool_output_type(schema);
    if type_ == "string" {
        return "a string".to_string();
    }
    let object = schema.filter(|schema| schema.is_object());
    let properties = object
        .and_then(|object| object.get("properties"))
        .and_then(Value::as_object);
    let is_object_schema = object
        .and_then(|object| object.get("type"))
        .and_then(Value::as_str)
        == Some("object");
    if is_object_schema && properties.is_some() && mcp_structured_content_schema(schema).is_none() {
        let properties = properties.expect("checked above");
        let required: HashSet<&str> = object
            .expect("checked above")
            .get("required")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let fields: Vec<String> = properties
            .keys()
            .map(|name| {
                if required.contains(name.as_str()) {
                    name.clone()
                } else {
                    format!("{name}?")
                }
            })
            .collect();
        return format!("`{{ {} }}`", fields.join(", "));
    }
    format!(
        "`{}`",
        type_.split_whitespace().collect::<Vec<_>>().join(" ")
    )
}

/// A declared tool's description followed by how scripts call it and what
/// the call resolves to (`describeScriptCall`). The arguments are the tool's
/// declared parameters, so they are not repeated.
fn describe_script_call(tool: &AgentTool) -> String {
    let declaration = to_codemode_declaration(tool, &[]);
    format!(
        "{}\n\nCodemode: `tools.{}(args)` resolves to {}.",
        tool.tool.description.trim(),
        to_codemode_identifier(&tool.tool.name),
        describe_output(Some(&declaration.output_schema))
    )
}

/// How the codemode tool presents tools that are both declared and callable
/// from scripts (`prepareCodemodeLoadout`):
///
/// - `on`: their descriptions say how scripts call them, and the codemode
///   description lists only the callable tools without `direct` exposure.
/// - `only`: the codemode description lists every callable tool, and requests
///   leave out the declarations of active `direct` tools.
///
/// Listing by exposure, not by the active set, keeps the codemode description
/// unchanged when `tool_search` loads a tool, so loads do not redeclare
/// codemode.
pub fn prepare_codemode_loadout(
    loadout: &ToolLoadout,
    options: &CodemodeToolOptions,
) -> ToolLoadoutChanges {
    let mode = options
        .get_mode
        .as_ref()
        .map_or(CodemodeMode::On, |get_mode| get_mode());
    let is_direct = |tool: &AgentTool| loadout.get_exposure(tool.name()) == ToolExposure::Direct;
    let callable = get_codemode_callable_tools(&loadout.callable);
    let callable_names: HashSet<&str> = callable.iter().map(|tool| tool.name()).collect();
    let mut descriptions: BTreeMap<String, String> = BTreeMap::new();
    if mode == CodemodeMode::On {
        for tool in &loadout.declared {
            if callable_names.contains(tool.name()) {
                descriptions.insert(tool.name().to_string(), describe_script_call(tool));
            }
        }
    }
    let listed: Vec<AgentTool> = match mode {
        CodemodeMode::Only => callable.clone(),
        _ => callable
            .iter()
            .filter(|tool| !is_direct(tool))
            .cloned()
            .collect(),
    };
    let namespaces: BTreeMap<String, ToolNamespace> = listed
        .iter()
        .filter_map(|tool| {
            loadout
                .get_namespace(tool.name())
                .map(|ns| (tool.name().to_string(), ns))
        })
        .collect();
    // Listed tools carry their prompt guidelines, which the system prompt
    // only has for declared tools.
    let guidelines: BTreeMap<String, Vec<String>> = listed
        .iter()
        .map(|tool| {
            (
                tool.name().to_string(),
                loadout.get_prompt_guidelines(tool.name()),
            )
        })
        .collect();
    descriptions.insert(
        CODEMODE_TOOL_NAME.to_string(),
        create_codemode_description(
            &listed,
            CreateCodemodeDescriptionOptions {
                models: options.models == Some(true),
                namespaces,
                guidelines,
                deferred: listed
                    .iter()
                    .filter(|tool| loadout.get_exposure(tool.name()) == ToolExposure::Deferred)
                    .map(|tool| tool.name().to_string())
                    .collect(),
                inline_budget: Some(
                    options
                        .get_inline_budget
                        .as_ref()
                        .and_then(|get| get())
                        // Fractional budgets floor to whole tokens; identical
                        // selection because section costs are integers.
                        .map(|budget| budget.floor() as usize)
                        .unwrap_or(DEFAULT_CODEMODE_INLINE_BUDGET),
                ),
            },
        ),
    );
    let declared_names: HashSet<&str> = loadout.declared.iter().map(|tool| tool.name()).collect();
    let hidden_declarations = match mode {
        CodemodeMode::Only => callable
            .iter()
            .filter(|tool| is_direct(tool) && declared_names.contains(tool.name()))
            .map(|tool| tool.name().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::new(),
    };
    ToolLoadoutChanges {
        descriptions: Some(descriptions),
        hidden_declarations: Some(hidden_declarations),
    }
}

/// The codemode tool definition (`createCodemodeToolDefinition`).
pub fn create_codemode_tool_definition(options: CodemodeToolOptions) -> ToolDefinition {
    let mut definition = ToolDefinition::new(
        CODEMODE_TOOL_NAME,
        CODEMODE_TOOL_NAME,
        // Replaced with the declarations of the callable tools when the tool
        // is activated.
        create_codemode_description(
            &[],
            CreateCodemodeDescriptionOptions {
                models: options.models == Some(true),
                ..Default::default()
            },
        ),
        codemode_schema(),
    );
    definition.prompt_snippet = Some(CODEMODE_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_string());
    definition.prompt_guidelines = Some(
        CODEMODE_TOOL_SYSTEM_PROMPT_CONTRIBUTION
            .guidelines
            .iter()
            .map(|guideline| guideline.to_string())
            .collect(),
    );
    // Scripts must not start other scripts.
    definition.exposure = Some(ToolExposure::ModelOnly);
    definition.prepare_loadout = {
        let options = options.clone();
        let prepare_loadout: PrepareLoadoutFn = Arc::new(move |loadout: &ToolLoadout| {
            Some(prepare_codemode_loadout(loadout, &options))
        });
        Some(prepare_loadout)
    };
    // Capable models write the script as raw text instead of a JSON-escaped
    // string.
    definition.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
        ConstrainedSamplingConfig::Grammar {
            variants: [(
                GrammarFormat::OpenaiLark,
                CODEMODE_SOURCE_GRAMMAR.to_string(),
            )]
            .into_iter()
            .collect(),
        },
    ));
    definition.execute = {
        let options = options.clone();
        Arc::new(move |tool_call_id, params, signal, on_update, ctx| {
            let options = options.clone();
            // The sandbox (worker, QuickJS wasm) loads on the first call; the
            // upstream `execute.lazy.ts` dynamic import is a plain module in
            // Rust (deviation D91).
            let code = params
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Box::pin(async move {
                execute_codemode(
                    tool_call_id,
                    CodemodeToolInput { code },
                    signal,
                    on_update,
                    ctx,
                    options,
                )
                .await
            })
        })
    };
    definition
}

/// Create the codemode tool as an [`AgentTool`] (`createCodemodeTool`). The
/// description lists the given tools; the script can call whatever tools the
/// agent loop provides at execution time.
pub fn create_codemode_tool(tools: &[AgentTool], options: CodemodeToolOptions) -> AgentTool {
    let models = options.models == Some(true);
    let definition = create_codemode_tool_definition(options);
    let mut tool = wrap_tool_definition(definition, None);
    tool.tool.description = create_codemode_description(
        tools,
        CreateCodemodeDescriptionOptions {
            models,
            ..Default::default()
        },
    );
    // Upstream also assigns `promptSnippet`/`promptGuidelines` onto the
    // AgentTool; the Rust AgentTool carries neither (they are
    // `ToolDefinition`-only in the erased port), and `create_codemode_tool`'s
    // callers reach the model through `create_codemode_tool_definition` via
    // `register_tool`.
    tool
}
