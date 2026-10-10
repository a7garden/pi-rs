//! Port of `packages/coding-agent/src/extensions/tool-search/tool.ts` — tool
//! discovery: a BM25 ranker over tool metadata, shared by `searchTools()` in
//! codemode scripts and the optional `tool_search` tool.
//!
//! `tool_search` searches tools that are not declared to the model
//! ([`ToolExposure::Codemode`] and [`ToolExposure::Deferred`]) and loads the
//! matches, so they are declared for the next model call. Loading goes through
//! the active tool set, so it is recorded in the transcript like any other
//! tool change and survives `/tree`, resume, and fork on that branch.
//!
//! # Deviations (D95 range, shared with [`super`])
//!
//! - **D95** — TypeBox `toolSearchSchema` becomes a static JSON Schema
//!   document ([`tool_search_schema`]). Upstream's `isToolSearchTool`
//!   identifies the tool by schema *identity* (`tool.parameters ===
//!   toolSearchSchema`); the port compares the schema *structurally*,
//!   indistinguishable for JSON data.
//! - **D95a** — the `limit` option is `usize`: the runner validates JSON
//!   numbers before `execute` sees them, so the `Number.isInteger` guard
//!   checks the raw `f64` and the ranker receives a `usize` (`limit <= 0`
//!   collapses to the empty `usize == 0` case, same observable result).
//! - **D95b** — `ToolRanker` is a trait like upstream; [`Bm25Ranker::rank`]
//!   takes `&self` (no interior mutation) and scores are `f64`.
//!
//! Tests: the upstream `test/tool-search.test.ts` suite is ported 1:1 in the
//! inline module at the bottom; the `codemode description catalog` block
//! drives the codemode slice's `create_codemode_description` per the D90–D94
//! contract.

use std::collections::HashSet;
use std::sync::LazyLock;

use pi_agent::types::{AgentTool, AgentToolResult};
use pi_ai::types::{TextContent, UserContent};
use serde_json::{json, Value as JsonValue};

use crate::core::extensions::types::{ToolDefinition, ToolExposure, ToolInfo, ToolNamespace};

pub const TOOL_SEARCH_TOOL_NAME: &str = "tool_search";
pub const DEFAULT_TOOL_SEARCH_LIMIT: usize = 8;

/// A tool as the ranker sees it: its name and the text built by
/// [`create_tool_search_document`] (`ToolSearchDocument`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSearchDocument {
    pub name: String,
    pub text: String,
}

/// One ranked result (`ToolSearchMatch`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSearchMatch {
    pub name: String,
    pub score: f64,
}

/// Ranks tools for a query. BM25 today; a hybrid ranker with embeddings can
/// replace it (`ToolRanker`).
pub trait ToolRanker: Send + Sync {
    fn rank(
        &self,
        query: &str,
        documents: &[ToolSearchDocument],
        limit: usize,
    ) -> Vec<ToolSearchMatch>;
}

/// Upstream `STOP_WORDS`.
static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
        "on", "or", "that", "the", "this", "to", "with",
    ]
    .into_iter()
    .collect()
});

/// Naive singular form, so `issues` matches `issue` and `searches` matches
/// `search` (upstream `stem`).
fn stem(term: &str) -> String {
    let len = term.len();
    if len > 4 && term.ends_with("ies") {
        return format!("{}y", &term[..len - 3]);
    }
    if len > 4
        && ["ches", "shes", "sses", "xes", "zes"]
            .iter()
            .any(|suffix| term.ends_with(suffix))
    {
        return term[..len - 2].to_string();
    }
    if len > 3 && term.ends_with('s') && !term.ends_with("ss") {
        return term[..len - 1].to_string();
    }
    term.to_string()
}

/// Lowercase terms, split at camelCase boundaries and non-alphanumerics,
/// without stop words (upstream `tokenize`; the two upstream camelCase
/// regexes become boundary detection over the same ASCII character classes).
pub fn tokenize(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut spaced = String::with_capacity(text.len() + 8);
    for (index, &character) in characters.iter().enumerate() {
        if character.is_ascii_uppercase() {
            let previous = if index > 0 {
                Some(characters[index - 1])
            } else {
                None
            };
            let next = characters.get(index + 1).copied();
            match previous {
                Some(previous) if previous.is_ascii_lowercase() || previous.is_ascii_digit() => {
                    spaced.push(' ')
                }
                Some(previous) if previous.is_ascii_uppercase() => {
                    if next.is_some_and(|next| next.is_ascii_lowercase()) {
                        spaced.push(' ');
                    }
                }
                _ => {}
            }
        }
        spaced.push(character);
    }
    spaced
        .to_lowercase()
        .split(|character: char| !character.is_ascii_lowercase() && !character.is_ascii_digit())
        .filter(|term| !term.is_empty() && !STOP_WORDS.contains(term))
        .map(stem)
        .collect()
}

/// Schema descriptions and property names, recursively (upstream
/// `schemaText`; insertion order is preserved by `serde_json`'s
/// `preserve_order` feature).
fn schema_text(schema: &JsonValue, parts: &mut Vec<String>) {
    let Some(object) = schema.as_object() else {
        return;
    };
    if let Some(description) = object
        .get("description")
        .and_then(|description| description.as_str())
    {
        parts.push(description.to_string());
    }
    if let Some(properties) = object
        .get("properties")
        .and_then(|properties| properties.as_object())
    {
        for (name, property) in properties {
            parts.push(name.clone());
            schema_text(property, parts);
        }
    }
    schema_text(object.get("items").unwrap_or(&JsonValue::Null), parts);
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = object.get(key).and_then(|variants| variants.as_array()) {
            for variant in variants {
                schema_text(variant, parts);
            }
        }
    }
}

/// Search text of a tool: the name, the name with `_` as spaces, the
/// description, schema descriptions and property names, and the namespace
/// with its description and instructions (`createToolSearchDocument`).
fn tool_search_document_from_parts(
    name: &str,
    description: &str,
    parameters: &JsonValue,
    namespace: Option<&ToolNamespace>,
) -> ToolSearchDocument {
    let mut parts = vec![
        name.to_string(),
        name.replace('_', " "),
        description.to_string(),
    ];
    schema_text(parameters, &mut parts);
    if let Some(namespace) = namespace {
        parts.push(namespace.name.clone());
        parts.push(namespace.description.clone().unwrap_or_default());
        parts.push(namespace.instructions.clone().unwrap_or_default());
    }
    let text = parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    ToolSearchDocument {
        name: name.to_string(),
        text,
    }
}

/// [`tool_search_document_from_parts`] over an extension-api tool
/// (upstream's structural `Pick<ToolInfo, …>` argument — agents and codemode
/// pass [`AgentTool`]).
pub fn create_tool_search_document(
    tool: &AgentTool,
    namespace: Option<&ToolNamespace>,
) -> ToolSearchDocument {
    tool_search_document_from_parts(
        &tool.tool.name,
        &tool.tool.description,
        &tool.tool.parameters,
        namespace,
    )
}

/// Same document over a [`ToolInfo`] (the `getAllTools` shape
/// `search_and_load` ranks).
pub fn create_tool_search_document_for_info(
    tool: &ToolInfo,
    namespace: Option<&ToolNamespace>,
) -> ToolSearchDocument {
    tool_search_document_from_parts(&tool.name, &tool.description, &tool.parameters, namespace)
}

/// Okapi BM25 with the usual parameters. Ties keep document order
/// (`Bm25Ranker`; Rust's stable sort matches `Array.prototype.sort`).
#[derive(Debug, Clone)]
pub struct Bm25Ranker {
    k1: f64,
    b: f64,
}

impl Default for Bm25Ranker {
    fn default() -> Self {
        Bm25Ranker::new()
    }
}

impl Bm25Ranker {
    pub fn new() -> Self {
        Bm25Ranker { k1: 1.2, b: 0.75 }
    }

    /// Upstream `new({ k1, b })`.
    pub fn with_parameters(k1: f64, b: f64) -> Self {
        Bm25Ranker { k1, b }
    }

    /// Inherent passthrough to [`ToolRanker::rank`] so callers that hold the
    /// concrete ranker need not import the trait.
    pub fn rank(
        &self,
        query: &str,
        documents: &[ToolSearchDocument],
        limit: usize,
    ) -> Vec<ToolSearchMatch> {
        ToolRanker::rank(self, query, documents, limit)
    }
}

impl ToolRanker for Bm25Ranker {
    fn rank(
        &self,
        query: &str,
        documents: &[ToolSearchDocument],
        limit: usize,
    ) -> Vec<ToolSearchMatch> {
        // Upstream spreads a Set: unique terms in first-seen order.
        let mut seen = HashSet::new();
        let query_terms: Vec<String> = tokenize(query)
            .into_iter()
            .filter(|term| seen.insert(term.clone()))
            .collect();
        if query_terms.is_empty() || documents.is_empty() || limit == 0 {
            return Vec::new();
        }
        let term_counts: Vec<std::collections::HashMap<String, usize>> = documents
            .iter()
            .map(|document| {
                let mut counts = std::collections::HashMap::new();
                for term in tokenize(&document.text) {
                    *counts.entry(term).or_insert(0usize) += 1;
                }
                counts
            })
            .collect();
        let lengths: Vec<usize> = term_counts
            .iter()
            .map(|counts| counts.values().sum())
            .collect();
        let total_length: usize = lengths.iter().sum();
        let average_length = {
            let average = total_length as f64 / documents.len() as f64;
            // `|| 1` upstream: an all-empty corpus ranks nothing anyway.
            if average == 0.0 {
                1.0
            } else {
                average
            }
        };
        let idf: std::collections::HashMap<String, f64> = query_terms
            .iter()
            .map(|term| {
                let frequency = term_counts
                    .iter()
                    .filter(|counts| counts.contains_key(term))
                    .count();
                let value = (1.0
                    + (documents.len() as f64 - frequency as f64 + 0.5) / (frequency as f64 + 0.5))
                    .ln();
                (term.clone(), value)
            })
            .collect();
        let mut matches: Vec<ToolSearchMatch> = Vec::new();
        for (index, document) in documents.iter().enumerate() {
            let mut score = 0.0;
            for term in &query_terms {
                let Some(&count) = term_counts[index].get(term.as_str()) else {
                    continue;
                };
                if count == 0 {
                    continue;
                }
                let norm =
                    self.k1 * (1.0 - self.b + (self.b * lengths[index] as f64) / average_length);
                score += idf.get(term).copied().unwrap_or(0.0)
                    * ((count as f64 * (self.k1 + 1.0)) / (count as f64 + norm));
            }
            if score > 0.0 {
                matches.push(ToolSearchMatch {
                    name: document.name.clone(),
                    score,
                });
            }
        }
        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        matches
    }
}

/// Parameter schema of the tool (upstream TypeBox `toolSearchSchema`; D95).
pub fn tool_search_schema() -> &'static JsonValue {
    static SCHEMA: LazyLock<JsonValue> = LazyLock::new(|| {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for deferred tools.",
                },
                "limit": {
                    "type": "number",
                    "description": format!("Maximum number of tools to return. Defaults to {DEFAULT_TOOL_SEARCH_LIMIT}."),
                },
            },
            "required": ["query"],
        })
    });
    &SCHEMA
}

/// Whether the tool is this `tool_search`, not another extension's tool of
/// the same name (`isToolSearchTool`; structural comparison — D95).
pub fn is_tool_search_tool(tool: &ToolInfo) -> bool {
    tool.name == TOOL_SEARCH_TOOL_NAME && tool.parameters == *tool_search_schema()
}

/// A loaded tool as reported in the result text (`ToolSearchResultTool`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSearchResultTool {
    pub name: String,
    pub description: String,
}

/// Options of [`create_tool_search_tool_definition`] (`ToolSearchToolOptions`).
#[derive(Clone, Default)]
pub struct ToolSearchToolOptions {
    /// The session's tools. `tool_search` searches the tools that are not
    /// declared to the model and activates the matches. Without it, the tool
    /// finds nothing. An `ExtensionAPI` fits — see [`ApiToolSource`].
    pub tools: Option<std::sync::Arc<dyn ToolSearchSource>>,
}

/// The `Pick<ExtensionAPI, "getAllTools" | "getActiveTools" |
/// "setActiveTools">` seam (`ToolSearchSource`).
pub trait ToolSearchSource: Send + Sync {
    fn get_active_tools(&self) -> Vec<String>;
    fn get_all_tools(&self) -> Vec<ToolInfo>;
    fn set_active_tools(&self, tool_names: Vec<String>);
}

/// [`ToolSearchSource`] adapter over the extension API handle the factory
/// receives.
#[derive(Clone)]
pub struct ApiToolSource {
    api: crate::core::extensions::types::ExtensionApiHandle,
}

impl ApiToolSource {
    pub fn new(api: crate::core::extensions::types::ExtensionApiHandle) -> Self {
        ApiToolSource { api }
    }
}

impl ToolSearchSource for ApiToolSource {
    fn get_active_tools(&self) -> Vec<String> {
        self.api.get_active_tools()
    }

    fn get_all_tools(&self) -> Vec<ToolInfo> {
        self.api.get_all_tools()
    }

    fn set_active_tools(&self, tool_names: Vec<String>) {
        self.api.set_active_tools(tool_names);
    }
}

/// Whether `tool_search` can load a tool with this exposure (`isSearchable`).
fn is_searchable(exposure: ToolExposure) -> bool {
    matches!(exposure, ToolExposure::Codemode | ToolExposure::Deferred)
}

/// Rank the searchable tools that are not active yet and activate the
/// matches, so the next model call declares them. Activation is recorded in
/// the transcript like any tool change (`searchAndLoad`).
fn search_and_load(
    source: &dyn ToolSearchSource,
    query: &str,
    limit: usize,
) -> Vec<ToolSearchResultTool> {
    let active = source.get_active_tools();
    let candidates: Vec<ToolInfo> = source
        .get_all_tools()
        .into_iter()
        .filter(|tool| {
            is_searchable(tool.exposure) && !active.iter().any(|name| name == &tool.name)
        })
        .collect();
    let documents: Vec<ToolSearchDocument> = candidates
        .iter()
        .map(|tool| create_tool_search_document_for_info(tool, tool.namespace.as_ref()))
        .collect();
    let matches = Bm25Ranker::new().rank(query, &documents, limit);
    if !matches.is_empty() {
        let mut next = active;
        next.extend(matches.iter().map(|match_| match_.name.clone()));
        source.set_active_tools(next);
    }
    matches
        .into_iter()
        .map(|match_| ToolSearchResultTool {
            name: match_.name.clone(),
            description: candidates
                .iter()
                .find(|tool| tool.name == match_.name)
                .map(|tool| tool.description.clone())
                .unwrap_or_default(),
        })
        .collect()
}

/// The `tool_search` description. It does not list the searchable tools or
/// their namespaces, so it stays the same while tools are registered, for
/// example when MCP servers connect (`TOOL_SEARCH_DESCRIPTION`).
pub fn tool_search_description() -> &'static str {
    TOOL_SEARCH_DESCRIPTION
}

pub static TOOL_SEARCH_DESCRIPTION: &str = "# Tool discovery\n\nSearches over deferred tool metadata with BM25 and exposes matching tools for the next model call.\n\nSome of the tools, such as tools of MCP servers, may not have been provided to you upfront, and you should use this tool (`tool_search`) to search for the required tools. For MCP tool discovery, always use `tool_search`.";

/// First line of a trimmed description (result-text rendering in upstream's
/// `execute`).
fn description_first_line(description: &str) -> &str {
    let trimmed = description.trim();
    trimmed
        .split('\n')
        .next()
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .unwrap_or("")
}

/// The `tool_search` tool definition (`createToolSearchToolDefinition`).
pub fn create_tool_search_tool_definition(options: ToolSearchToolOptions) -> ToolDefinition {
    let mut definition = ToolDefinition::new(
        TOOL_SEARCH_TOOL_NAME,
        TOOL_SEARCH_TOOL_NAME,
        TOOL_SEARCH_DESCRIPTION,
        tool_search_schema().clone(),
    );
    definition.prompt_snippet =
        Some("Search for tools that are not loaded yet and load the matches".to_string());
    // Searching is not something scripts need; it changes what the model sees.
    definition.exposure = Some(ToolExposure::ModelOnly);
    let tools = options.tools;
    definition.execute = std::sync::Arc::new(
        move |_tool_call_id: String,
              args: JsonValue,
              _signal: Option<crate::utils::abort::AbortSignal>,
              _update: Option<pi_agent::types::AgentToolUpdateCallback>,
              _context: Option<crate::core::extensions::types::ExtensionToolContext>| {
            let tools = tools.clone();
            Box::pin(async move {
                let query = args
                    .get("query")
                    .and_then(|query| query.as_str())
                    .map(str::trim)
                    .unwrap_or("");
                if query.is_empty() {
                    return Err("query must not be empty".to_string());
                }
                let max = match args.get("limit") {
                    None | Some(JsonValue::Null) => DEFAULT_TOOL_SEARCH_LIMIT,
                    Some(value) => {
                        let number = value.as_f64().ok_or("limit must be a positive integer")?;
                        if !number.is_finite() || number.fract() != 0.0 || number <= 0.0 {
                            return Err("limit must be a positive integer".to_string());
                        }
                        number as usize
                    }
                };
                let loaded = match tools.as_deref() {
                    Some(source) => search_and_load(source, query, max),
                    None => Vec::new(),
                };
                let text = if loaded.is_empty() {
                    "No matching tools found.".to_string()
                } else {
                    format!(
                        "Loaded {} tool{}. They are available from your next call:\n{}",
                        loaded.len(),
                        if loaded.len() == 1 { "" } else { "s" },
                        loaded
                            .iter()
                            .map(|tool| format!(
                                "- {}: {}",
                                tool.name,
                                description_first_line(&tool.description)
                            ))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                };
                let details = json!({ "loaded": loaded.iter().map(|tool| tool.name.clone()).collect::<Vec<_>>() });
                Ok(AgentToolResult::new(
                    vec![UserContent::Text(TextContent {
                        text,
                        text_signature: None,
                    })],
                    details,
                ))
            })
        },
    );
    definition
}

// ============================================================================
// Tests — port of `test/tool-search.test.ts`.
// ============================================================================

#[cfg(test)]
mod tool_tests {
    use super::*;
    use pi_agent::types::AgentTool;
    use pi_ai::types::Tool;
    use std::sync::Arc;

    fn tool(name: &str, description: &str, properties: JsonValue) -> AgentTool {
        AgentTool {
            tool: Tool {
                name: name.to_string(),
                description: description.to_string(),
                parameters: json!({ "type": "object", "properties": properties }),
                constrained_sampling: None,
            },
            label: name.to_string(),
            prepare_arguments: None,
            output_schema: None,
            execute: Arc::new(|_, _, _, _| {
                Box::pin(async { Ok(AgentToolResult::new(Vec::new(), JsonValue::Null)) })
                    as pi_agent::BoxFuture<'static, Result<AgentToolResult, String>>
            }),
            replay: None,
            execution_mode: None,
        }
    }

    // -- tokenize ----------------------------------------------------------

    #[test]
    fn tokenize_splits_camel_and_snake_case_drops_stop_words_and_folds_plurals() {
        assert_eq!(
            tokenize("listIssues for the GitHub_repo"),
            vec!["list", "issue", "git", "hub", "repo"]
        );
        assert_eq!(
            tokenize("searches queries HTTPServer"),
            vec!["search", "query", "http", "server"]
        );
    }

    // -- Bm25Ranker --------------------------------------------------------

    fn ranker_documents() -> Vec<ToolSearchDocument> {
        let tools = vec![
            tool(
                "mcp__github__list_issues",
                "List issues in a repository.",
                json!({ "state": { "type": "string", "description": "open or closed" } }),
            ),
            tool(
                "mcp__github__create_pull_request",
                "Open a pull request.",
                json!({}),
            ),
            tool(
                "mcp__linear__search_issues",
                "Search Linear issues by text.",
                json!({}),
            ),
            tool("mcp__docs__search", "Search the documentation.", json!({})),
        ];
        tools
            .iter()
            .map(|entry| create_tool_search_document(entry, None))
            .collect()
    }

    #[test]
    fn ranks_by_term_relevance_and_respects_the_limit() {
        let ranker = Bm25Ranker::new();
        let documents = ranker_documents();
        let names: Vec<String> = ranker
            .rank("issue", &documents, 8)
            .into_iter()
            .map(|match_| match_.name)
            .collect();
        assert_eq!(
            names,
            vec!["mcp__linear__search_issues", "mcp__github__list_issues"]
        );
        assert_eq!(
            ranker.rank("pull requests", &documents, 8)[0].name,
            "mcp__github__create_pull_request"
        );
        assert_eq!(ranker.rank("search", &documents, 1).len(), 1);
        // Property names and descriptions are searchable.
        let closed: Vec<String> = ranker
            .rank("closed", &documents, 8)
            .into_iter()
            .map(|match_| match_.name)
            .collect();
        assert_eq!(closed, vec!["mcp__github__list_issues"]);
    }

    #[test]
    fn returns_nothing_for_unknown_or_empty_queries() {
        let ranker = Bm25Ranker::new();
        let documents = ranker_documents();
        assert!(ranker.rank("kubernetes", &documents, 8).is_empty());
        assert!(ranker.rank("the", &documents, 8).is_empty());
        // Known v1 limit: no synonyms, so "tickets" does not find "issues".
        assert!(ranker.rank("tickets", &documents, 8).is_empty());
    }

    #[test]
    fn includes_the_namespace_in_the_search_text() {
        let namespace = ToolNamespace {
            name: "mcp__x".to_string(),
            description: Some("Kubernetes cluster tools".to_string()),
            instructions: None,
        };
        let document = create_tool_search_document(
            &tool("mcp__x__run", "Run it.", json!({})),
            Some(&namespace),
        );
        let matches = Bm25Ranker::new().rank("kubernetes", &[document], 8);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "mcp__x__run");
    }

    // -- codemode description catalog ---------------------------------------
    //
    // Ported from the `codemode description catalog` block; drives the
    // codemode slice's catalog per the D90–D94 contract (CodemodePort).

    use crate::extensions::codemode::tool::{
        create_codemode_description, CreateCodemodeDescriptionOptions,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn catalog_tools() -> (AgentTool, Vec<AgentTool>, Vec<AgentTool>) {
        let plain = tool("read_notes", "Read notes.", json!({}));
        let github: Vec<AgentTool> = ["a", "b", "c"]
            .iter()
            .map(|suffix| {
                tool(
                    &format!("mcp__github__{suffix}"),
                    &format!("GitHub {suffix}."),
                    json!({}),
                )
            })
            .collect();
        let docs = vec![
            tool("mcp__docs__search", "Search docs.", json!({})),
            tool(
                "mcp__docs__long",
                &format!("Long {}", "Long ".repeat(200)),
                json!({}),
            ),
        ];
        (plain, github, docs)
    }

    fn catalog_options(
        namespaces: BTreeMap<String, ToolNamespace>,
        deferred: BTreeSet<String>,
        inline_budget: Option<usize>,
    ) -> CreateCodemodeDescriptionOptions {
        CreateCodemodeDescriptionOptions {
            models: false,
            namespaces,
            guidelines: BTreeMap::new(),
            deferred,
            inline_budget,
        }
    }

    #[test]
    fn catalog_lists_everything_without_a_budget() {
        let (plain, github, docs) = catalog_tools();
        let mut all = vec![plain.clone()];
        all.extend(github.iter().cloned());
        all.extend(docs.iter().cloned());
        let mut namespaces = BTreeMap::new();
        for entry in &github {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__github".to_string(),
                    description: Some("GitHub server".to_string()),
                    instructions: None,
                },
            );
        }
        for entry in &docs {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__docs".to_string(),
                    description: None,
                    instructions: None,
                },
            );
        }
        let description =
            create_codemode_description(&all, catalog_options(namespaces, BTreeSet::new(), None));
        assert!(description.contains("Nested tools:"));
        assert!(description.contains("## mcp__github\nGitHub server"));
        assert!(description.contains("## mcp__docs\n\n### `mcp__docs"));
        // The search guidance is always there, so tools that appear later do not change it.
        assert!(description.contains("find unlisted tools, such as MCP tools"));
        // #10555: the lookup helpers are async; without `await` scripts serialize the promise as {}.
        assert!(description.contains("`await searchTools(query, { limit?, namespace? })`"));
        assert!(description.contains("`await describeTool(name)`"));
        assert!(description.contains("`await describeNamespace(name)`"));
    }

    #[test]
    fn catalog_fills_the_budget_round_robin_cheapest_first_and_says_what_is_missing() {
        let (plain, github, docs) = catalog_tools();
        let mut all = vec![plain.clone()];
        all.extend(github.iter().cloned());
        all.extend(docs.iter().cloned());
        let mut namespaces = BTreeMap::new();
        for entry in &github {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__github".to_string(),
                    description: Some("GitHub server".to_string()),
                    instructions: None,
                },
            );
        }
        for entry in &docs {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__docs".to_string(),
                    description: None,
                    instructions: None,
                },
            );
        }
        // Each small section costs about 42 tokens: one tool per group, then one more.
        let description = create_codemode_description(
            &all,
            catalog_options(namespaces.clone(), BTreeSet::new(), Some(170)),
        );
        assert!(description.contains("### `read_notes`"));
        assert!(description.contains("## mcp__docs (some tools not listed)"));
        assert!(description.contains("### `mcp__docs__search`"));
        assert!(!description.contains("### `mcp__docs__long`"));
        assert!(description.contains("## mcp__github (some tools not listed)"));
        assert!(description.contains("find unlisted tools, such as MCP tools"));
        // Deterministic: the same input gives the same description.
        assert_eq!(
            create_codemode_description(
                &all,
                catalog_options(namespaces, BTreeSet::new(), Some(170))
            ),
            description
        );
    }

    #[test]
    fn catalog_leaves_deferred_tools_and_their_namespaces_out_entirely() {
        let (plain, github, docs) = catalog_tools();
        let mut all = vec![plain.clone()];
        all.extend(github.iter().cloned());
        all.extend(docs.iter().cloned());
        let deferred: BTreeSet<String> =
            github.iter().map(|entry| entry.tool.name.clone()).collect();
        let mut namespaces = BTreeMap::new();
        for entry in &github {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__github".to_string(),
                    description: Some("GitHub server".to_string()),
                    instructions: None,
                },
            );
        }
        for entry in &docs {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__docs".to_string(),
                    description: None,
                    instructions: None,
                },
            );
        }
        let description =
            create_codemode_description(&all, catalog_options(namespaces.clone(), deferred, None));
        assert!(!description.contains("mcp__github"));
        // Deferred tools, such as those of a server that connects later, do not change the description.
        let mut without_github = vec![plain];
        without_github.extend(docs.iter().cloned());
        assert_eq!(
            create_codemode_description(
                &without_github,
                catalog_options(namespaces, BTreeSet::new(), None)
            ),
            description
        );
    }

    #[test]
    fn catalog_leaves_namespace_instructions_out() {
        let (_, github, _) = catalog_tools();
        let mut namespaces = BTreeMap::new();
        for entry in &github {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__github".to_string(),
                    description: None,
                    instructions: Some("Long usage guide.".to_string()),
                },
            );
        }
        let description = create_codemode_description(
            &github,
            catalog_options(namespaces, BTreeSet::new(), None),
        );
        assert!(description.contains("## mcp__github\n\n### `mcp__github__a`"));
        assert!(!description.contains("Long usage guide."));
    }

    #[test]
    fn catalog_lists_only_namespaces_with_a_zero_budget() {
        let (plain, github, docs) = catalog_tools();
        let mut all = vec![plain];
        all.extend(github.iter().cloned());
        all.extend(docs.iter().cloned());
        let mut namespaces = BTreeMap::new();
        for entry in &github {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__github".to_string(),
                    description: Some("GitHub server".to_string()),
                    instructions: None,
                },
            );
        }
        for entry in &docs {
            namespaces.insert(
                entry.tool.name.clone(),
                ToolNamespace {
                    name: "mcp__docs".to_string(),
                    description: None,
                    instructions: None,
                },
            );
        }
        let description = create_codemode_description(
            &all,
            catalog_options(namespaces, BTreeSet::new(), Some(0)),
        );
        assert!(description.contains("## mcp__docs (tools not listed)"));
        assert!(!description.contains("codemode tool declaration:"));
    }

    // -- schema / guard ------------------------------------------------------

    #[test]
    fn schema_carries_query_and_optional_limit() {
        let schema = tool_search_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        assert_eq!(schema["properties"]["limit"]["type"], "number");
        assert_eq!(schema["required"], json!(["query"]));
    }

    #[test]
    fn is_tool_search_tool_matches_name_and_schema() {
        let source_info = crate::core::source_info::create_synthetic_source_info(
            "builtin:tool-search",
            "builtin",
            None,
            None,
            None,
        );
        let mut info = ToolInfo {
            name: TOOL_SEARCH_TOOL_NAME.to_string(),
            description: String::new(),
            parameters: tool_search_schema().clone(),
            prompt_guidelines: None,
            exposure: ToolExposure::ModelOnly,
            namespace: None,
            annotations: None,
            source_info,
        };
        assert!(is_tool_search_tool(&info));
        info.parameters = json!({ "type": "object" });
        assert!(!is_tool_search_tool(&info));
    }
}
