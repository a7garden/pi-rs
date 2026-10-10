//! Port of `packages/coding-agent/src/extensions/mcp/tools.ts`: adapts MCP
//! tools to pi tool definitions.
//!
//! # Port deviations
//!
//! - **D89.** Model-facing content keeps [`pi_mcp::protocol::content::LlmContent`]
//!   (the `pi-ai` text/image shapes) instead of re-declaring upstream's
//!   `TextContent | ImageContent` union; `AgentToolResult.content` carries
//!   them directly, and `McpToolDetails` is erased JSON (core extensions D27).
//! - **D81.** The TUI renderers (`createMcpToolRenderers`) defer with the
//!   `pi-tui` exclusion; registered tools carry no `renderCall`/`renderResult`.

use std::sync::Arc;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use pi_mcp::client::McpRequestOptions;
use pi_mcp::protocol::content::{
    CallToolResult, ContentBlock, LlmContent, ResourceContents, ResourceLinkContent,
};
use pi_mcp::protocol::jsonrpc::McpError;
use pi_mcp::protocol::types::Tool;

use crate::core::extensions::types::{
    ToolAnnotations, ToolDefinition, ToolExposure, ToolNamespace,
};
use crate::core::tools::truncate::truncate_middle;
use crate::extensions::mcp::config::{get_mcp_tool_exposure, McpExposure};

/// Tool exposure of an MCP exposure. `codemode` and `deferred` both leave
/// tools out of the codemode description; they differ only in which tool the
/// MCP extension activates to reach them (`toToolExposure`).
pub fn to_tool_exposure(exposure: McpExposure) -> ToolExposure {
    match exposure {
        McpExposure::Codemode => ToolExposure::Deferred,
        McpExposure::Deferred => ToolExposure::Deferred,
        McpExposure::Direct => ToolExposure::Direct,
        McpExposure::Hidden => ToolExposure::Hidden,
    }
}

/// Provider tool names are limited to 64 characters of `[A-Za-z0-9_-]`.
const MAX_TOOL_NAME_LENGTH: usize = 64;
/// Model-facing text of an MCP result beyond this is cut in the middle.
pub const MCP_OUTPUT_MAX_BYTES: usize = 20 * 1024;
/// Tool that reads the resources named by resource links; owned by
/// `core/mcp-servers.ts` and re-exported here.
pub use crate::core::mcp_servers::READ_MCP_RESOURCE_TOOL;

/// Upstream `McpToolDetails`, as erased JSON (`ToolDefinition` details — D27
/// in core extensions).
pub fn tool_details(server: &str, tool: &str, full_output_path: Option<&str>) -> Value {
    let mut details = json!({ "server": server, "tool": tool });
    if let Some(path) = full_output_path {
        details["fullOutputPath"] = json!(path);
    }
    details
}

/// Saves the full text of a truncated result, or a binary resource, and
/// returns the file path. `extension` includes the dot, for example `.txt`
/// (upstream `McpOutputSaver`).
pub type McpOutputSaver = Arc<dyn Fn(&[u8], &str) -> Result<String, String> + Send + Sync>;

/// Upstream `saveToTempFile`: `writeOutputFile("pi-mcp", extension, data)`.
pub fn save_to_temp_file(data: &[u8], extension: &str) -> Result<String, String> {
    write_output_file("pi-mcp", extension, data)
}

/// Upstream `utils/output-files.ts` `writeOutputFile`: a uniquely named file
/// under the temp directory, in a folder per tool.
fn write_output_file(tool: &str, extension: &str, data: &[u8]) -> Result<String, String> {
    let dir = std::env::temp_dir()
        .join(format!("pi-{}-{}", tool, std::process::id()))
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    let path = format!("{dir}/output-{unique}{extension}");
    std::fs::write(&path, data).map_err(|error| error.to_string())?;
    Ok(path)
}

/// `mcp__<server>__<tool>`, sanitized and shortened with a hash suffix when
/// too long (`createMcpToolName`). Like Codex, everything but
/// `[A-Za-z0-9_]` becomes `_`, so the name is also the identifier codemode
/// scripts call it by. `is_taken` reports names used by a different MCP tool:
/// sanitizing can map two tools to one name (`a-b` and `a_b`), which then get
/// the hash suffix.
pub fn create_mcp_tool_name(server: &str, tool: &str, is_taken: impl Fn(&str) -> bool) -> String {
    let raw = format!("mcp__{server}__{tool}");
    let name: String = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if name.len() <= MAX_TOOL_NAME_LENGTH && !is_taken(&name) {
        return name;
    }
    let hash = hex_encode(&Sha256::digest(format!("{server}\0{tool}").as_bytes())[..4]);
    let keep = MAX_TOOL_NAME_LENGTH - hash.len() - 1;
    let mut truncated: String = name.chars().take(keep).collect();
    // Upstream `slice` cuts by UTF-16 units; the sanitized name is ASCII, so a
    // byte cut is equivalent here.
    while truncated.len() > keep {
        truncated.pop();
    }
    format!("{truncated}_{hash}")
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn text_of(content: &[LlmContent]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            LlmContent::Text { text } => Some(text.as_str()),
            LlmContent::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Output schema of every MCP tool: the `CallToolResult` scripts receive, with
/// the tool's own output schema as `structuredContent`. Codemode detects this
/// shape to render `CallToolResult<T>` declarations (`createMcpResultSchema`).
pub fn create_mcp_result_schema(structured_content_schema: Option<&Value>) -> Value {
    let mut properties = serde_json::Map::new();
    properties.insert(
        "content".to_string(),
        json!({ "type": "array", "items": { "type": "object" } }),
    );
    if let Some(schema) = structured_content_schema {
        properties.insert("structuredContent".to_string(), schema.clone());
    }
    properties.insert("isError".to_string(), json!({ "type": "boolean" }));
    properties.insert("_meta".to_string(), json!({ "type": "object" }));
    json!({
        "type": "object",
        "properties": Value::Object(properties),
        "required": ["content"],
    })
}

/// Keep model-facing text within [`MCP_OUTPUT_MAX_BYTES`]. Longer text becomes
/// one text block in Codex's truncation format, followed by the path of the
/// file with the full text; images follow it (`limitMcpContent`).
pub fn limit_mcp_content(
    content: Vec<LlmContent>,
    save_output: &McpOutputSaver,
) -> Result<(Vec<LlmContent>, Option<String>), String> {
    let combined = text_of(&content);
    let truncation = truncate_middle(&combined, MCP_OUTPUT_MAX_BYTES);
    if !truncation.truncated {
        return Ok((content, None));
    }
    let (full_output_path, where_note) = match save_output.as_ref()(combined.as_bytes(), ".txt") {
        Ok(path) => {
            let note = format!("[Full output: {path} (read it with offset/limit)]");
            (Some(path), note)
        }
        Err(error) => (None, format!("[Could not save the full output: {error}]")),
    };
    let tokens = truncation.total_bytes.div_ceil(4);
    let text = format!(
		"Warning: truncated output (original token count: {tokens})\nTotal output lines: {}\n\n{}\n\n{}",
		truncation.total_lines, truncation.content, where_note
	);
    let mut limited = vec![LlmContent::Text { text }];
    limited.extend(
        content
            .into_iter()
            .filter(|block| matches!(block, LlmContent::Image { .. })),
    );
    Ok((limited, full_output_path))
}

/// Upstream `ConvertMcpResultOptions`.
#[derive(Clone, Default)]
pub struct ConvertMcpResultOptions {
    /// Saves truncated text and binary resources. Default: a temp file.
    pub save_output: Option<McpOutputSaver>,
    /// Whether the server's resources can be read with
    /// `read_mcp_resource`, which resource links then name.
    pub readable_resources: bool,
}

/// File extension for a saved binary resource: the one its URI ends in, else
/// `.bin` (`extensionOf`).
fn extension_of(uri: &str) -> String {
    let path = url::Url::parse(uri)
        .map(|url| url.path().to_string())
        .unwrap_or_else(|_| uri.to_string());
    let last = path.rsplit('/').next().unwrap_or("");
    match last.rsplit_once('.') {
        Some((_, extension))
            if !extension.is_empty()
                && extension.len() <= 8
                && extension.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            format!(".{extension}")
        }
        _ => ".bin".to_string(),
    }
}

/// Blobs of these types are shown as text (`isTextMimeType`).
fn is_text_mime_type(mime_type: Option<&str>) -> bool {
    let Some(mime_type) = mime_type else {
        return false;
    };
    let type_part = mime_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    type_part.starts_with("text/")
        || type_part == "application/json"
        || type_part.ends_with("+json")
        || type_part.ends_with("+xml")
}

/// `formatSize` (core tools): binary sizes with one decimal and a unit.
fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes}B")
    } else {
        format!("{value:.1}{}", UNITS[unit])
    }
}

/// Model-facing content of one block of `server`'s result (`blockToContent`).
async fn block_to_content(
    server: &str,
    block: &ContentBlock,
    options: &ConvertMcpResultOptions,
) -> Vec<LlmContent> {
    match block {
        ContentBlock::ResourceLink(link) => {
            let ResourceLinkContent {
                uri,
                name,
                title,
                description,
                mime_type,
                size,
                ..
            } = link;
            let mut details: Vec<String> = Vec::new();
            if let Some(mime_type) = mime_type {
                details.push(mime_type.clone());
            }
            if let Some(size) = size {
                details.push(format_size(*size as u64));
            }
            let read = if options.readable_resources {
                format!(". Read it with {READ_MCP_RESOURCE_TOOL} (server \"{server}\")")
            } else {
                String::new()
            };
            let description = description
                .as_ref()
                .map(|description| format!(": {description}"))
                .unwrap_or_default();
            let details_text = if details.is_empty() {
                String::new()
            } else {
                format!(" ({})", details.join(", "))
            };
            let title = title.clone().unwrap_or_else(|| name.clone());
            vec![LlmContent::Text {
                text: format!("[Resource {uri} \"{title}\"{details_text}{description}{read}]"),
            }]
        }
        ContentBlock::EmbeddedResource(embedded) => {
            let ResourceContents::Blob(resource) = &embedded.resource else {
                return to_llm_block(block);
            };
            if resource
                .mime_type
                .as_deref()
                .unwrap_or_default()
                .starts_with("image/")
            {
                return to_llm_block(block);
            }
            let pi_mcp::protocol::content::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } = resource;
            let Ok(data) = decode_base64(blob) else {
                return vec![LlmContent::Text {
                    text: format!("[Binary resource {uri} could not be decoded]"),
                }];
            };
            if is_text_mime_type(mime_type.as_deref()) {
                return vec![LlmContent::Text {
                    text: String::from_utf8_lossy(&data).into_owned(),
                }];
            }
            let kind = format!(
                "{}, {}",
                mime_type
                    .clone()
                    .unwrap_or_else(|| "unknown type".to_string()),
                format_size(data.len() as u64)
            );
            let save = options
                .save_output
                .clone()
                .unwrap_or_else(|| Arc::new(save_to_temp_file));
            match save.as_ref()(&data, &extension_of(uri)) {
                Ok(path) => vec![LlmContent::Text {
                    text: format!("[Binary resource {uri} ({kind}) saved to {path}]"),
                }],
                Err(reason) => vec![LlmContent::Text {
                    text: format!("[Binary resource {uri} ({kind}) could not be saved: {reason}]"),
                }],
            }
        }
        other => to_llm_block(other),
    }
}

fn to_llm_block(block: &ContentBlock) -> Vec<LlmContent> {
    // Upstream `toLlmContent({ content: [block] })`.
    match block {
        ContentBlock::Text(text) => vec![LlmContent::Text {
            text: text.text.clone(),
        }],
        ContentBlock::Image(image) => vec![LlmContent::Image {
            data: image.data.clone(),
            mime_type: image.mime_type.clone(),
        }],
        ContentBlock::Audio(audio) => vec![LlmContent::Text {
            text: format!("[audio {} omitted]", audio.mime_type),
        }],
        ContentBlock::ResourceLink(link) => vec![LlmContent::Text {
            text: format!("{}: {}", link.name, link.uri),
        }],
        ContentBlock::EmbeddedResource(embedded) => match &embedded.resource {
            ResourceContents::Text(resource) => vec![LlmContent::Text {
                text: resource.text.clone(),
            }],
            ResourceContents::Blob(resource) => {
                let mime = resource.mime_type.as_deref().unwrap_or_default();
                if mime.starts_with("image/") {
                    vec![LlmContent::Image {
                        data: resource.blob.clone(),
                        mime_type: mime.to_string(),
                    }]
                } else {
                    vec![LlmContent::Text {
                        text: format!(
                            "[binary resource {} ({}) omitted]",
                            resource.uri,
                            resource.mime_type.as_deref().unwrap_or("unknown type")
                        ),
                    }]
                }
            }
        },
    }
}

/// Model-facing content of `server`'s content blocks, before the output limit
/// (`toModelContent`).
pub async fn to_model_content(
    server: &str,
    blocks: &[ContentBlock],
    options: &ConvertMcpResultOptions,
) -> Vec<LlmContent> {
    let mut content = Vec::new();
    for block in blocks {
        content.extend(block_to_content(server, block, options).await);
    }
    content
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    fn value_of(byte: u8) -> Result<u32, String> {
        match byte {
            b'A'..=b'Z' => Ok(u32::from(byte - b'A')),
            b'a'..=b'z' => Ok(u32::from(byte - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(byte - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 byte {byte}")),
        }
    }
    let cleaned: Vec<u8> = input
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace() && *byte != b'=')
        .collect();
    let mut output = Vec::new();
    for chunk in cleaned.chunks(4) {
        let mut value = 0u32;
        for (index, byte) in chunk.iter().enumerate() {
            value |= value_of(*byte)? << (18 - 6 * index);
        }
        // Trailing '=' padding shortens the tail: 3 chars encode 2 bytes,
        // 2 chars encode 1 byte (atob semantics).
        match chunk.len() {
            4 => {
                output.push((value >> 16) as u8);
                output.push((value >> 8) as u8);
                output.push(value as u8);
            }
            3 => {
                output.push((value >> 16) as u8);
                output.push((value >> 8) as u8);
            }
            2 => {
                output.push((value >> 16) as u8);
            }
            _ => {}
        }
    }
    Ok(output)
}

/// Convert an MCP result. `isError` results become error results that keep the
/// structured result (`convertMcpResult`).
pub async fn convert_mcp_result(
    server: &str,
    tool: &str,
    result: &CallToolResult,
    options: &ConvertMcpResultOptions,
) -> Result<pi_agent::AgentToolResult, String> {
    // Without content blocks, toLlmContent falls back to the structured
    // content as JSON.
    let converted: Vec<LlmContent> = if !result.content.is_empty() {
        to_model_content(server, &result.content, options).await
    } else {
        pi_mcp::protocol::content::to_llm_content(result)
    };
    let mut converted = converted;
    if result.is_error == Some(true) && text_of(&converted).is_empty() {
        converted.push(LlmContent::Text {
            text: format!("MCP tool {server}/{tool} returned an error"),
        });
    }
    let default_saver: McpOutputSaver = Arc::new(save_to_temp_file);
    let (content, full_output_path) = limit_mcp_content(
        converted,
        options.save_output.as_ref().unwrap_or(&default_saver),
    )?;
    let model_content: Vec<pi_ai::types::UserContent> = content
        .into_iter()
        .map(|block| match block {
            LlmContent::Text { text } => {
                pi_ai::types::UserContent::Text(pi_ai::types::TextContent::new(text))
            }
            LlmContent::Image { data, mime_type } => {
                pi_ai::types::UserContent::Image(pi_ai::types::ImageContent::new(data, mime_type))
            }
        })
        .collect();
    // Scripts get the server's result as sent, without `_meta`.
    let mut script_result = serde_json::Map::new();
    script_result.insert(
        "content".to_string(),
        serde_json::to_value(&result.content).map_err(|error| error.to_string())?,
    );
    if let Some(structured) = &result.structured_content {
        script_result.insert(
            "structuredContent".to_string(),
            Value::Object(structured.clone()),
        );
    }
    if let Some(is_error) = result.is_error {
        script_result.insert("isError".to_string(), json!(is_error));
    }
    let script_result = Value::Object(script_result);
    let mut agent_result = pi_agent::AgentToolResult::new(
        model_content,
        tool_details(server, tool, full_output_path.as_deref()),
    );
    agent_result.structured_content = Some(script_result);
    if result.is_error == Some(true) {
        agent_result.is_error = Some(true);
    }
    Ok(agent_result)
}

/// Upstream `createMcpToolDefinition`.
pub struct McpToolDefinitionOptions {
    pub server: String,
    pub tool: Tool,
    pub name: String,
    pub exposure: McpExposure,
    pub namespace: ToolNamespace,
    pub timeout_ms: u64,
    /// The connected server (`getClient`).
    pub get_client: Arc<
        dyn Fn() -> futures_util::future::BoxFuture<'static, Result<Arc<dyn McpToolCaller>, String>>
            + Send
            + Sync,
    >,
    /// Whether `read_mcp_resource` can read the server's resources.
    pub readable_resources: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

/// The part of a connected server tool calls go through (upstream
/// `McpToolCaller`).
pub trait McpToolCaller: Send + Sync {
    fn call_tool(
        &self,
        name: &str,
        args: Value,
        options: McpRequestOptions,
    ) -> futures_util::future::BoxFuture<'static, Result<CallToolResult, McpError>>;
}

/// Tool input schemas must be objects. MCP servers may omit `type`, and some
/// providers reject object schemas without `properties` (`toParameters`).
fn to_parameters(schema: &Value) -> Value {
    let mut parameters = schema.clone();
    if !parameters.is_object() {
        return json!({ "type": "object", "properties": {} });
    }
    let object = parameters.as_object_mut().expect("checked object");
    object
        .entry("type".to_string())
        .or_insert_with(|| json!("object"));
    object
        .entry("properties".to_string())
        .or_insert_with(|| json!({}));
    parameters
}

/// The boolean hints of an MCP tool's annotations, or `None` when it has none
/// (`toToolAnnotations`).
fn to_tool_annotations(tool: &Tool) -> Option<ToolAnnotations> {
    let Some(annotations) = &tool.annotations else {
        return None;
    };
    let mapped = ToolAnnotations {
        read_only_hint: annotations.read_only_hint,
        destructive_hint: annotations.destructive_hint,
        idempotent_hint: annotations.idempotent_hint,
        open_world_hint: annotations.open_world_hint,
    };
    if mapped.read_only_hint.is_none()
        && mapped.destructive_hint.is_none()
        && mapped.idempotent_hint.is_none()
        && mapped.open_world_hint.is_none()
    {
        None
    } else {
        Some(mapped)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn create_mcp_tool_definition(options: McpToolDefinitionOptions) -> ToolDefinition {
    let McpToolDefinitionOptions {
        server,
        tool,
        name,
        exposure,
        namespace,
        timeout_ms,
        get_client,
        readable_resources,
    } = options;
    let title = tool.title.clone().or_else(|| {
        tool.annotations
            .as_ref()
            .and_then(|annotations| annotations.title.clone())
    });
    let annotations = to_tool_annotations(&tool);
    let label = format!("{}/{}", server, tool.name);
    let description = tool
        .description
        .clone()
        .map(|description| description.trim().to_string())
        .filter(|description| !description.is_empty())
        .or(title)
        .unwrap_or_else(|| format!("MCP tool {} from server {}", tool.name, server));
    let mut definition = ToolDefinition::new(
        name,
        label,
        description,
        to_parameters(&Value::Object(tool.input_schema.clone())),
    );
    definition.output_schema = Some(create_mcp_result_schema(
        tool.output_schema
            .as_ref()
            .map(|schema| Value::Object(schema.clone()))
            .as_ref(),
    ));
    definition.exposure = Some(to_tool_exposure(exposure));
    definition.namespace = Some(namespace);
    definition.annotations = annotations;
    let tool_name = tool.name.clone();
    let server_for_execute = server.clone();
    let tool_for_execute = tool_name.clone();
    definition.execute = Arc::new(move |_tool_call_id, params, signal, on_update, _ctx| {
        let get_client = Arc::clone(&get_client);
        let server = server_for_execute.clone();
        let tool = tool_for_execute.clone();
        let readable_resources = readable_resources.clone();
        Box::pin(async move {
            let client = (get_client)().await?;
            let mut request_options = McpRequestOptions::default().timeout_ms(timeout_ms);
            if let Some(signal) = signal {
                request_options = request_options.signal(adapt_abort_signal(signal));
            }
            if let Some(on_update) = on_update {
                let server = server.clone();
                let tool = tool.clone();
                request_options = request_options.on_progress(Arc::new(move |progress| {
                    let total = progress
                        .total
                        .as_ref()
                        .map(|total| format!("/{total}"))
                        .unwrap_or_default();
                    let text = progress
                        .message
                        .clone()
                        .unwrap_or_else(|| format!("Progress {}{}", progress.progress, total));
                    on_update(pi_agent::AgentToolResult::new(
                        vec![pi_ai::types::UserContent::Text(
                            pi_ai::types::TextContent::new(text),
                        )],
                        tool_details(&server, &tool, None),
                    ));
                }));
            }
            let arguments = match params {
                Value::Object(object) => object,
                Value::Null => serde_json::Map::new(),
                other => return Err(format!("tool arguments must be an object, got {other}")),
            };
            let result = client
                .call_tool(&tool, Value::Object(arguments), request_options)
                .await
                .map_err(|error| error.to_string())?;
            convert_mcp_result(
                &server,
                &tool,
                &result,
                &ConvertMcpResultOptions {
                    save_output: None,
                    readable_resources: readable_resources
                        .map(|readable| readable())
                        .unwrap_or(false),
                },
            )
            .await
        })
    });
    definition
}

/// pi-ai `AbortSignal` → `pi-mcp` `AbortSignal` (the crates keep separate
/// signal types; see pi-mcp lib docs D-deviations).
fn adapt_abort_signal(signal: crate::utils::abort::AbortSignal) -> pi_mcp::client::AbortSignal {
    let controller = pi_mcp::client::AbortController::new();
    let forward = controller.signal();
    tokio::spawn(async move {
        signal.aborted_future().await;
        controller.abort(Some("request aborted".to_string()));
    });
    forward
}

#[cfg(test)]
mod tools_tests {
    //! Port of `test/mcp-extension.test.ts` `describe("MCP tools")` and the
    //! `truncateMiddle` boundary test.

    use super::*;
    use serde_json::json;

    /// Port of "creates provider-safe tool names".
    #[test]
    fn creates_provider_safe_tool_names() {
        assert_eq!(
            create_mcp_tool_name("docs", "search", |_| false),
            "mcp__docs__search"
        );
        assert_eq!(
            create_mcp_tool_name("my-server", "get.item/v2", |_| false),
            "mcp__my_server__get_item_v2"
        );
        let long = create_mcp_tool_name("server", &"x".repeat(100), |_| false);
        assert_eq!(long.len(), 64);
        assert!(
            long.starts_with("mcp__server__x") && long.ends_with(&long[long.len() - 8..]),
            "unexpected shape: {long}"
        );
        assert!(long
            .rsplit_once('_')
            .map(|(_, hash)| hash.len() == 8 && hash.chars().all(|c| c.is_ascii_hexdigit()))
            .unwrap_or(false));
        assert_ne!(
            create_mcp_tool_name("server", &format!("{}y", "x".repeat(100)), |_| false),
            long
        );
        // Names that sanitize to one already taken by another tool get a hash suffix.
        let taken = create_mcp_tool_name("s", "a_b", |_| false);
        let second = create_mcp_tool_name("s", "a-b", |name| name == &taken);
        assert!(second.starts_with("mcp__s__a_b_"));
        assert!(second
            .rsplit_once('_')
            .map(|(_, hash)| hash.len() == 8 && hash.chars().all(|c| c.is_ascii_hexdigit()))
            .unwrap_or(false));
    }

    fn llm_content(result: &pi_agent::AgentToolResult) -> Vec<LlmContent> {
        result
            .content
            .iter()
            .map(|block| match block {
                pi_ai::types::UserContent::Text(text) => LlmContent::Text {
                    text: text.text.clone(),
                },
                pi_ai::types::UserContent::Image(image) => LlmContent::Image {
                    data: image.data.clone(),
                    mime_type: image.mime_type.clone(),
                },
            })
            .collect()
    }

    fn parse_blocks(blocks: Value) -> Vec<ContentBlock> {
        serde_json::from_value(blocks).expect("valid content blocks")
    }

    fn call_tool_result(blocks: Value, extra: Value) -> CallToolResult {
        let mut value = json!({ "content": blocks });
        if let Value::Object(extra) = extra {
            for (key, entry) in extra {
                value[key] = entry;
            }
        }
        serde_json::from_value(value).expect("valid CallToolResult")
    }

    /// Port of "converts results, passing the CallToolResult to scripts and
    /// flagging errors".
    #[tokio::test]
    async fn converts_results_passing_the_result_to_scripts_and_flagging_errors() {
        let blocks = parse_blocks(json!([
            { "type": "resource_link", "uri": "file:///a", "name": "a" },
            { "type": "resource", "resource": { "uri": "file:///b", "text": "b text" } },
            { "type": "audio", "data": "", "mimeType": "audio/wav" }
        ]));
        let result = call_tool_result(
            json!(blocks.clone()),
            json!({ "structuredContent": { "ok": true }, "_meta": { "trace": "x" } }),
        );
        let converted =
            convert_mcp_result("docs", "t", &result, &ConvertMcpResultOptions::default())
                .await
                .expect("convert");
        assert_eq!(
            llm_content(&converted),
            vec![
                LlmContent::Text {
                    text: "[Resource file:///a \"a\"]".to_string()
                },
                LlmContent::Text {
                    text: "b text".to_string()
                },
                LlmContent::Text {
                    text: "[audio audio/wav omitted]".to_string()
                },
            ]
        );
        assert_eq!(converted.details, json!({ "server": "docs", "tool": "t" }));
        // Scripts get the server's blocks as sent, without `_meta`.
        assert_eq!(
            converted.structured_content,
            Some(json!({ "content": blocks, "structuredContent": { "ok": true } }))
        );

        let empty = call_tool_result(json!([]), json!({ "structuredContent": { "n": 1 } }));
        let converted =
            convert_mcp_result("docs", "t", &empty, &ConvertMcpResultOptions::default())
                .await
                .expect("convert");
        assert_eq!(
            llm_content(&converted),
            vec![LlmContent::Text {
                text: "{\n  \"n\": 1\n}".to_string()
            }]
        );

        let error_result = call_tool_result(
            json!([{ "type": "text", "text": "nope" }]),
            json!({ "isError": true }),
        );
        let converted = convert_mcp_result(
            "docs",
            "t",
            &error_result,
            &ConvertMcpResultOptions::default(),
        )
        .await
        .expect("convert");
        assert_eq!(
            llm_content(&converted),
            vec![LlmContent::Text {
                text: "nope".to_string()
            }]
        );
        assert_eq!(converted.details, json!({ "server": "docs", "tool": "t" }));
        assert_eq!(
            converted.structured_content,
            Some(json!({ "content": [{ "type": "text", "text": "nope" }], "isError": true }))
        );
        assert_eq!(converted.is_error, Some(true));

        let error_empty = call_tool_result(json!([]), json!({ "isError": true }));
        let converted = convert_mcp_result(
            "docs",
            "t",
            &error_empty,
            &ConvertMcpResultOptions::default(),
        )
        .await
        .expect("convert");
        assert_eq!(
            llm_content(&converted),
            vec![LlmContent::Text {
                text: "MCP tool docs/t returned an error".to_string()
            }]
        );
    }

    /// Port of "points resource links to read_mcp_resource and saves binary
    /// resources".
    #[tokio::test]
    async fn points_resource_links_to_read_mcp_resource_and_saves_binary_resources() {
        let saved: Arc<Mutex<Vec<(Vec<u8>, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let saved_saver = Arc::clone(&saved);
        let save_output: McpOutputSaver = Arc::new(move |data, extension| {
            saved_saver
                .lock()
                .push((data.to_vec(), extension.to_string()));
            Ok(format!("/tmp/saved{extension}"))
        });
        let blocks = parse_blocks(json!([
            {
                "type": "resource_link",
                "uri": "docs://guide",
                "name": "guide",
                "title": "The Guide",
                "mimeType": "text/markdown",
                "size": 2048,
                "description": "How to use it"
            },
            {
                "type": "resource",
                "resource": { "uri": "file:///r/report.pdf", "mimeType": "application/pdf", "blob": "JVBERg==" }
            },
            { "type": "resource", "resource": { "uri": "docs://logo", "mimeType": "image/png", "blob": "AAAA" } }
        ]));
        let result = call_tool_result(json!(blocks), json!({}));
        let converted = convert_mcp_result(
            "docs",
            "t",
            &result,
            &ConvertMcpResultOptions {
                save_output: Some(save_output),
                readable_resources: true,
            },
        )
        .await
        .expect("convert");
        assert_eq!(
			llm_content(&converted),
			vec![
				LlmContent::Text {
					text: "[Resource docs://guide \"The Guide\" (text/markdown, 2.0KB): How to use it. Read it with read_mcp_resource (server \"docs\")]".to_string()
				},
				LlmContent::Text {
					text: "[Binary resource file:///r/report.pdf (application/pdf, 4B) saved to /tmp/saved.pdf]"
						.to_string()
				},
				LlmContent::Image { data: "AAAA".to_string(), mime_type: "image/png".to_string() },
			]
		);
        assert_eq!(*saved.lock(), vec![(b"%PDF".to_vec(), ".pdf".to_string())]);
    }

    /// Port of "cuts the middle of model-facing text over 20KB and keeps the
    /// full result for scripts".
    #[tokio::test]
    async fn cuts_the_middle_of_model_facing_text_over_20kb() {
        let saved: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let saved_saver = Arc::clone(&saved);
        let save_output: McpOutputSaver = Arc::new(move |data, _extension| {
            saved_saver.lock().push(data.to_vec());
            Ok("/tmp/full.txt".to_string())
        });
        let lines: Vec<String> = (0..3000)
            .map(|index| format!("line {}", index + 1))
            .collect();
        let full = lines.join("\n");
        let image = LlmContent::Image {
            data: "AAAA".to_string(),
            mime_type: "image/png".to_string(),
        };
        let result = call_tool_result(
            json!([
                { "type": "text", "text": full },
                { "type": "image", "data": "AAAA", "mimeType": "image/png" }
            ]),
            json!({}),
        );
        let converted = convert_mcp_result(
            "docs",
            "snapshot",
            &result,
            &ConvertMcpResultOptions {
                save_output: Some(Arc::clone(&save_output)),
                readable_resources: false,
            },
        )
        .await
        .expect("convert");
        assert_eq!(converted.content.len(), 2);
        let pi_ai::types::UserContent::Text(text) = &converted.content[0] else {
            panic!("expected text content");
        };
        let pi_ai::types::UserContent::Image(converted_image) = &converted.content[1] else {
            panic!("expected image content");
        };
        // Codex's format: a header, the start and end of the text, then the
        // file with the full text.
        let header = format!(
			"Warning: truncated output (original token count: {})\nTotal output lines: 3000\n\nline 1\nline 2\n",
			full.len().div_ceil(4)
		);
        assert!(
            text.text.starts_with(&header),
            "unexpected header: {}",
            &text.text[..header.len().min(200)]
        );
        assert!(text.text.contains(" chars truncated…"));
        assert!(text
            .text
            .ends_with("line 3000\n\n[Full output: /tmp/full.txt (read it with offset/limit)]"));
        assert!(text.text.len() < 21 * 1024);
        assert_eq!(converted_image.data, "AAAA");
        assert_eq!(
            converted.details,
            json!({ "server": "docs", "tool": "snapshot", "fullOutputPath": "/tmp/full.txt" })
        );
        assert_eq!(saved.lock().as_slice(), [full.as_bytes().to_vec()]);
        assert_eq!(
            converted.structured_content,
            Some(json!({
                "content": [
                    { "type": "text", "text": full },
                    { "type": "image", "data": "AAAA", "mimeType": "image/png" }
                ]
            }))
        );

        // Text within the limit is not saved.
        let small = call_tool_result(json!([{ "type": "text", "text": "ok" }]), json!({}));
        convert_mcp_result(
            "docs",
            "small",
            &small,
            &ConvertMcpResultOptions {
                save_output: Some(save_output),
                readable_resources: false,
            },
        )
        .await
        .expect("convert");
        assert_eq!(saved.lock().len(), 1);
    }

    /// Port of "cuts multi-byte text only at character boundaries" (the
    /// `truncateMiddle` boundary assertions).
    #[test]
    fn cuts_multi_byte_text_only_at_character_boundaries() {
        let text = format!("{}end", "\u{e9}".repeat(20_000));
        let result = truncate_middle(&text, 1001);
        assert!(result.truncated);
        assert!(!result.content.contains('\u{FFFD}'));
        assert!(result.content.ends_with("end"));
        let parts: Vec<&str> = result.content.split('\u{2026}').collect();
        assert_eq!(
            parts.len(),
            3,
            "expected head … marker … tail: {:?}",
            parts.len()
        );
        assert!(parts[0].len() <= 500, "head too long: {}", parts[0].len());
        assert!(parts[2].len() <= 501, "tail too long: {}", parts[2].len());
        let head_chars = parts[0].chars().count();
        let tail_chars = parts[2].chars().count();
        assert_eq!(
            head_chars + tail_chars + result.removed_chars,
            text.chars().count()
        );
    }

    type Mutex<T> = parking_lot::Mutex<T>;
}
