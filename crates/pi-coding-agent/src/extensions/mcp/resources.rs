//! Port of `packages/coding-agent/src/extensions/mcp/resources.ts`: MCP
//! resources, through the tools Codex and opencode use —
//! `list_mcp_resources`, `list_mcp_resource_templates`, and
//! `read_mcp_resource`.

use std::sync::Arc;

use serde_json::{json, Value};

use pi_mcp::client::McpRequestOptions;
use pi_mcp::protocol::content::ContentBlock;
use pi_mcp::protocol::types::{
	ListResourcesResult, ListResourceTemplatesResult, ReadResourceResult, Resource, ResourceTemplate,
};

use crate::core::extensions::types::{ToolAnnotations, ToolDefinition};
use crate::extensions::mcp::config::McpExposure;
use crate::extensions::mcp::tools::{
	limit_mcp_content, to_model_content, to_tool_exposure, ConvertMcpResultOptions, McpOutputSaver,
};

pub const LIST_MCP_RESOURCES_TOOL: &str = "list_mcp_resources";
pub const LIST_MCP_RESOURCE_TEMPLATES_TOOL: &str = "list_mcp_resource_templates";
pub use crate::extensions::mcp::tools::READ_MCP_RESOURCE_TOOL;

/// A connected server that offers resources (upstream `McpResourceServer`).
#[async_trait::async_trait]
pub trait McpResourceServer: Send + Sync {
	fn name(&self) -> &str;
	fn timeout_ms(&self) -> u64;
	async fn resources_page(
		&self,
		cursor: Option<&str>,
		options: McpRequestOptions,
	) -> Result<ListResourcesResult, String>;
	async fn resource_templates_page(
		&self,
		cursor: Option<&str>,
		options: McpRequestOptions,
	) -> Result<ListResourceTemplatesResult, String>;
	async fn all_resources(&self, options: McpRequestOptions) -> Result<Vec<Resource>, String>;
	async fn all_resource_templates(&self, options: McpRequestOptions) -> Result<Vec<ResourceTemplate>, String>;
	async fn read_resource(&self, uri: &str, options: McpRequestOptions) -> Result<ReadResourceResult, String>;
}

/// MCP App user interfaces, which only hosts that render them can use
/// (`isMcpAppResource`).
pub fn is_mcp_app_resource(item: &Value) -> bool {
	let uri = item
		.get("uri")
		.or_else(|| item.get("uriTemplate"))
		.and_then(|value| value.as_str())
		.unwrap_or_default();
	let mime_type = item.get("mimeType").and_then(|value| value.as_str()).unwrap_or_default();
	uri.starts_with("ui://") || {
		// `/;\s*profile\s*=\s*"?mcp-app"?/i`
		let lower = mime_type.to_lowercase();
		lower.contains("profile") && {
			let mut rest = lower.as_str();
			while let Some(index) = rest.find(';') {
				rest = &rest[index + 1..];
				let trimmed = rest.trim_start();
				if let Some(after) = trimmed.strip_prefix("profile") {
					let after = after.trim_start();
					if let Some(after) = after.strip_prefix('=') {
						let after = after.trim_start();
						if after.starts_with("\"mcp-app") || after.starts_with("mcp-app") {
							return true;
						}
					}
				}
			}
			false
		}
	}
}

/// A listed resource or template without `_meta` and icons, tagged with its
/// server (`listed`).
fn listed(server: &str, item: &Value) -> Value {
	let mut object = match item {
		Value::Object(object) => object.clone(),
		other => {
			let mut wrapper = serde_json::Map::new();
			wrapper.insert("value".to_string(), other.clone());
			wrapper
		}
	};
	object.remove("_meta");
	object.remove("icons");
	let mut result = serde_json::Map::new();
	result.insert("server".to_string(), json!(server));
	for (key, value) in object {
		result.insert(key, value);
	}
	Value::Object(result)
}

fn string_property(description: &str) -> Value {
	json!({ "type": "string", "description": description })
}

fn list_parameters() -> Value {
	json!({
		"type": "object",
		"properties": {
			"server": string_property("MCP server name. Omit to list every server with resources."),
			"cursor": string_property("Opaque cursor from a previous call with the same server; omit for the first page."),
		},
		"additionalProperties": false,
	})
}

fn read_parameters() -> Value {
	json!({
		"type": "object",
		"properties": {
			"server": string_property(
				"MCP server name exactly as configured. Must match the 'server' field returned by list_mcp_resources.",
			),
			"uri": string_property("Resource URI to read. Must be one of the URIs returned by list_mcp_resources."),
		},
		"required": ["server", "uri"],
		"additionalProperties": false,
	})
}

fn optional_string() -> Value {
	json!({ "type": "string" })
}

fn listing_errors() -> Value {
	json!({
		"type": "array",
		"description": "Servers that could not be listed",
		"items": {
			"type": "object",
			"properties": { "server": { "type": "string" }, "error": { "type": "string" } },
			"required": ["server", "error"],
		},
	})
}

fn list_output_schema() -> Value {
	json!({
		"type": "object",
		"properties": {
			"server": optional_string(),
			"resources": {
				"type": "array",
				"items": {
					"type": "object",
					"properties": {
						"server": { "type": "string" },
						"uri": { "type": "string" },
						"name": { "type": "string" },
						"title": optional_string(),
						"description": optional_string(),
						"mimeType": optional_string(),
						"size": { "type": "number" },
					},
					"required": ["server", "uri", "name"],
				},
			},
			"nextCursor": optional_string(),
			"errors": listing_errors(),
		},
		"required": ["resources"],
	})
}

fn list_templates_output_schema() -> Value {
	json!({
		"type": "object",
		"properties": {
			"server": optional_string(),
			"resourceTemplates": {
				"type": "array",
				"items": {
					"type": "object",
					"properties": {
						"server": { "type": "string" },
						"uriTemplate": { "type": "string", "description": "RFC 6570 URI template" },
						"name": { "type": "string" },
						"title": optional_string(),
						"description": optional_string(),
						"mimeType": optional_string(),
					},
					"required": ["server", "uriTemplate", "name"],
				},
			},
			"nextCursor": optional_string(),
			"errors": listing_errors(),
		},
		"required": ["resourceTemplates"],
	})
}

fn read_output_schema() -> Value {
	json!({
		"type": "object",
		"properties": {
			"server": { "type": "string" },
			"uri": { "type": "string" },
			"contents": {
				"type": "array",
				"items": {
					"anyOf": [
						{
							"type": "object",
							"properties": { "uri": { "type": "string" }, "mimeType": optional_string(), "text": { "type": "string" } },
							"required": ["uri", "text"],
						},
						{
							"type": "object",
							"properties": {
								"uri": { "type": "string" },
								"mimeType": optional_string(),
								"blob": { "type": "string", "description": "base64" },
							},
							"required": ["uri", "blob"],
						},
					],
				},
			},
		},
		"required": ["server", "uri", "contents"],
	})
}

fn string_argument(params: &Value, key: &str) -> Result<Option<String>, String> {
	let Some(value) = params.get(key) else {
		return Ok(None);
	};
	if value.is_null() {
		return Ok(None);
	}
	let Some(text) = value.as_str() else {
		return Err(format!("{key} must be a string"));
	};
	Ok(if text.trim().is_empty() { None } else { Some(text.trim().to_string()) })
}

async fn json_result(
	tool: &str,
	server: Option<&str>,
	payload: Value,
) -> Result<pi_agent::AgentToolResult, String> {
	let text = serde_json::to_string(&payload).map_err(|error| error.to_string())?;
	let default_saver: McpOutputSaver =
		Arc::new(crate::extensions::mcp::tools::save_to_temp_file);
	let (content, full_output_path) =
		limit_mcp_content(vec![pi_mcp::protocol::content::LlmContent::Text { text }], &default_saver)?;
	let details = crate::extensions::mcp::tools::tool_details(
		server.unwrap_or(""),
		tool,
		full_output_path.as_deref(),
	);
	let model_content: Vec<pi_ai::types::UserContent> = content
		.into_iter()
		.map(|block| match block {
			pi_mcp::protocol::content::LlmContent::Text { text } => {
				pi_ai::types::UserContent::Text(pi_ai::types::TextContent::new(text))
			}
			pi_mcp::protocol::content::LlmContent::Image { data, mime_type } => {
				pi_ai::types::UserContent::Image(pi_ai::types::ImageContent::new(data, mime_type))
			}
		})
		.collect();
	let mut result = pi_agent::AgentToolResult::new(model_content, details);
	result.structured_content = Some(payload);
	Ok(result)
}

/// One page of one server, or every page of every server (upstream's inner
/// `list`).
async fn list_payload(
	servers: Arc<dyn Fn() -> Vec<Arc<dyn McpResourceServer>> + Send + Sync>,
	params: &Value,
	key: &'static str,
	page: Arc<
		dyn Fn(Arc<dyn McpResourceServer>, Option<String>, McpRequestOptions) -> futures_util::future::BoxFuture<
			'static,
			Result<(Vec<Value>, Option<String>), String>,
		> + Send
		+ Sync,
	>,
	all: Arc<
		dyn Fn(Arc<dyn McpResourceServer>, McpRequestOptions) -> futures_util::future::BoxFuture<
			'static,
			Result<Vec<Value>, String>,
		> + Send
		+ Sync,
	>,
) -> Result<Value, String> {
	let server_name = string_argument(params, "server")?;
	let cursor = string_argument(params, "cursor")?;
	if let Some(server_name) = server_name {
		let server = find_server_fn(&servers, &server_name)?;
		let options = McpRequestOptions::default().timeout_ms(server.timeout_ms());
		let (items, next_cursor) = page(Arc::clone(&server), cursor, options).await?;
		let visible: Vec<Value> = items
			.into_iter()
			.filter(|item| !is_mcp_app_resource(item))
			.map(|item| listed(server.name(), &item))
			.collect();
		let mut payload = json!({ "server": server.name(), key: visible });
		if let Some(next_cursor) = next_cursor {
			payload["nextCursor"] = json!(next_cursor);
		}
		return Ok(payload);
	}
	if cursor.is_some() {
		return Err("cursor can only be used when a server is specified".to_string());
	}
	let mut candidates = servers();
	candidates.sort_by(|a, b| a.name().to_lowercase().cmp(&b.name().to_lowercase()));
	let mut items: Vec<Value> = Vec::new();
	let mut errors: Vec<Value> = Vec::new();
	for server in candidates {
		let options = McpRequestOptions::default().timeout_ms(server.timeout_ms());
		match all(Arc::clone(&server), options).await {
			Ok(listed_items) => {
				items.extend(
					listed_items
						.into_iter()
						.filter(|item| !is_mcp_app_resource(item))
						.map(|item| listed(server.name(), &item)),
				);
			}
			Err(error) => errors.push(json!({ "server": server.name(), "error": error })),
		}
	}
	let mut payload = json!({ key: items });
	if !errors.is_empty() {
		payload["errors"] = Value::Array(errors);
	}
	Ok(payload)
}

/// The three resource tools. `servers` returns the servers whose resources
/// they reach, at call time (`createMcpResourceToolDefinitions`).
pub fn create_mcp_resource_tool_definitions(
	exposure: McpExposure,
	servers: Arc<dyn Fn() -> Vec<Arc<dyn McpResourceServer>> + Send + Sync>,
) -> Vec<ToolDefinition> {
	let read_only = ToolAnnotations {
		read_only_hint: Some(true),
		..Default::default()
	};

	let list_servers = Arc::clone(&servers);
	let list_resources = {
		let page = Arc::new(|server: Arc<dyn McpResourceServer>, cursor: Option<String>, options: McpRequestOptions| {
			Box::pin(async move {
				let result = server.resources_page(cursor.as_deref(), options).await?;
				Ok((
					result
						.resources
						.iter()
						.map(|resource| serde_json::to_value(resource).expect("resource serialization is infallible"))
						.collect::<Vec<_>>(),
					result.next_cursor,
				))
			}) as futures_util::future::BoxFuture<'static, Result<(Vec<Value>, Option<String>), String>>
		});
		let all = Arc::new(|server: Arc<dyn McpResourceServer>, options: McpRequestOptions| {
			Box::pin(async move {
				let resources = server.all_resources(options).await?;
				Ok(resources
					.iter()
					.map(|resource| serde_json::to_value(resource).expect("resource serialization is infallible"))
					.collect())
			}) as futures_util::future::BoxFuture<'static, Result<Vec<Value>, String>>
		});
		let mut definition = ToolDefinition::new(
			LIST_MCP_RESOURCES_TOOL,
			LIST_MCP_RESOURCES_TOOL,
			"Lists resources provided by MCP servers. Resources allow servers to share data that provides context to language models, such as files, database schemas, or application-specific information. Prefer resources over web search when possible.",
			list_parameters(),
		);
		definition.output_schema = Some(list_output_schema());
		definition.exposure = Some(to_tool_exposure(exposure));
		definition.annotations = Some(read_only);
		definition.execute = {
			let page = Arc::clone(&page);
			let all = Arc::clone(&all);
			Arc::new(move |_tool_call_id, params, _signal, _on_update, _ctx| {
				let page = Arc::clone(&page);
				let all = Arc::clone(&all);
				let servers = Arc::clone(&list_servers);
				Box::pin(async move {
					let payload = list_payload(servers, &params, "resources", page, all).await?;
					let server = string_argument(&params, "server")?;
					json_result(LIST_MCP_RESOURCES_TOOL, server.as_deref(), payload).await
				})
			})
		};
		definition
	};

	let list_templates = {
		let page = Arc::new(|server: Arc<dyn McpResourceServer>, cursor: Option<String>, options: McpRequestOptions| {
			Box::pin(async move {
				let result = server.resource_templates_page(cursor.as_deref(), options).await?;
				Ok((
					result
						.resource_templates
						.iter()
						.map(|template| serde_json::to_value(template).expect("template serialization is infallible"))
						.collect::<Vec<_>>(),
					result.next_cursor,
				))
			}) as futures_util::future::BoxFuture<'static, Result<(Vec<Value>, Option<String>), String>>
		});
		let all = Arc::new(|server: Arc<dyn McpResourceServer>, options: McpRequestOptions| {
			Box::pin(async move {
				let templates = server.all_resource_templates(options).await?;
				Ok(templates
					.iter()
					.map(|template| serde_json::to_value(template).expect("template serialization is infallible"))
					.collect())
			}) as futures_util::future::BoxFuture<'static, Result<Vec<Value>, String>>
		});
		let mut definition = ToolDefinition::new(
			LIST_MCP_RESOURCE_TEMPLATES_TOOL,
			LIST_MCP_RESOURCE_TEMPLATES_TOOL,
			"Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to share data that takes parameters and provides context to language models, such as files, database schemas, or application-specific information. Prefer resource templates over web search when possible.",
			list_parameters(),
		);
		definition.output_schema = Some(list_templates_output_schema());
		definition.exposure = Some(to_tool_exposure(exposure));
		definition.annotations = Some(read_only);
		let servers_for_execute = Arc::clone(&servers);
		definition.execute = Arc::new(move |_tool_call_id, params, _signal, _on_update, _ctx| {
			let page = Arc::clone(&page);
			let all = Arc::clone(&all);
			let servers = Arc::clone(&servers_for_execute);
			Box::pin(async move {
				let payload = list_payload(servers, &params, "resourceTemplates", page, all).await?;
				let server = string_argument(&params, "server")?;
				json_result(LIST_MCP_RESOURCE_TEMPLATES_TOOL, server.as_deref(), payload).await
			})
		});
		definition
	};

	let read_resource = {
		let mut definition = ToolDefinition::new(
			READ_MCP_RESOURCE_TOOL,
			READ_MCP_RESOURCE_TOOL,
			"Read a specific resource from an MCP server given the server name and resource URI.",
			read_parameters(),
		);
		definition.output_schema = Some(read_output_schema());
		definition.exposure = Some(to_tool_exposure(exposure));
		definition.annotations = Some(read_only);
		let servers_for_execute = Arc::clone(&servers);
		let find_servers = Arc::clone(&servers);
		definition.execute = Arc::new(move |_tool_call_id, params, _signal, _on_update, _ctx| {
			let servers = Arc::clone(&servers_for_execute);
			let find_servers = Arc::clone(&find_servers);
			Box::pin(async move {
				let find_servers = Arc::clone(&find_servers);
				let _ = &find_servers;
				let server_name = string_argument(&params, "server")?
					.ok_or_else(|| "server must be provided".to_string())?;
				let uri = string_argument(&params, "uri")?.ok_or_else(|| "uri must be provided".to_string())?;
				let server = find_server_fn(&find_servers, &server_name)?;
				let mut options = McpRequestOptions::default().timeout_ms(server.timeout_ms());
				let _ = &mut options;
				let result = server.read_resource(&uri, options).await?;
				// Several contents (for example a directory) are labeled with
				// their URIs.
				let contents_json: Vec<Value> = result
					.contents
					.iter()
					.map(|contents| serde_json::to_value(contents).expect("contents serialization is infallible"))
					.collect();
				let mut blocks: Vec<ContentBlock> = Vec::new();
				for (index, contents) in contents_json.iter().enumerate() {
					if result.contents.len() > 1 {
						let label_uri = contents
							.get("uri")
							.and_then(|value| value.as_str())
							.unwrap_or_default()
							.to_string();
						blocks.push(
							serde_json::from_value(json!({ "type": "text", "text": format!("{label_uri}:") }))
								.expect("valid text block"),
						);
					}
					let _ = index;
					blocks.push(
						serde_json::from_value(json!({ "type": "resource", "resource": contents }))
							.expect("valid resource block"),
					);
				}
				let converted = to_model_content(
					server.name(),
					&blocks,
					&ConvertMcpResultOptions { save_output: None, readable_resources: true },
				)
				.await;
				let default_saver: McpOutputSaver =
					Arc::new(crate::extensions::mcp::tools::save_to_temp_file);
				let input = if converted.is_empty() {
					vec![pi_mcp::protocol::content::LlmContent::Text {
						text: format!("Resource {uri} is empty."),
					}]
				} else {
					converted
				};
				let (content, full_output_path) = limit_mcp_content(input, &default_saver)?;
				let contents_without_meta: Vec<Value> = contents_json
					.iter()
					.map(|contents| {
						let mut object = contents.clone();
						if let Value::Object(map) = &mut object {
							map.remove("_meta");
						}
						object
					})
					.collect();
				let model_content: Vec<pi_ai::types::UserContent> = content
					.into_iter()
					.map(|block| match block {
						pi_mcp::protocol::content::LlmContent::Text { text } => {
							pi_ai::types::UserContent::Text(pi_ai::types::TextContent::new(text))
						}
						pi_mcp::protocol::content::LlmContent::Image { data, mime_type } => {
							pi_ai::types::UserContent::Image(pi_ai::types::ImageContent::new(data, mime_type))
						}
					})
					.collect();
				let mut tool_result = pi_agent::AgentToolResult::new(
					model_content,
					crate::extensions::mcp::tools::tool_details(
						server.name(),
						READ_MCP_RESOURCE_TOOL,
						full_output_path.as_deref(),
					),
				);
				tool_result.structured_content = Some(json!({
					"server": server.name(),
					"uri": uri,
					"contents": contents_without_meta,
				}));
				Ok(tool_result)
			})
		});
		definition
	};

	vec![list_resources, list_templates, read_resource]
}

fn find_server_fn(
	servers: &Arc<dyn Fn() -> Vec<Arc<dyn McpResourceServer>> + Send + Sync>,
	name: &str,
) -> Result<Arc<dyn McpResourceServer>, String> {
	let candidates = servers();
	if let Some(server) = candidates.iter().find(|candidate| candidate.name() == name) {
		return Ok(Arc::clone(server));
	}
	let available = candidates
		.iter()
		.map(|candidate| candidate.name().to_string())
		.collect::<Vec<_>>()
		.join(", ");
	Err(format!(
		"MCP server \"{name}\" has no resources{}",
		if available.is_empty() {
			String::new()
		} else {
			format!(". Servers with resources: {available}")
		}
	))
}

#[cfg(test)]
mod resources_tests {
	use super::*;

	/// Port of the `isMcpAppResource` behavior the listing tools rely on.
	#[test]
	fn detects_mcp_app_resources() {
		assert!(is_mcp_app_resource(&json!({ "uri": "ui://widget" })));
		assert!(is_mcp_app_resource(&json!({ "uriTemplate": "ui://widget/{id}" })));
		assert!(is_mcp_app_resource(&json!({ "mimeType": "text/html; profile=mcp-app" })));
		assert!(is_mcp_app_resource(&json!({ "mimeType": "text/html;profile=\"MCP-APP\"" })));
		assert!(!is_mcp_app_resource(&json!({ "uri": "file:///a" })));
		assert!(!is_mcp_app_resource(&json!({ "mimeType": "text/plain" })));
	}

	/// Port of the `listed` shape: `_meta` and icons are dropped, `server` is
	/// prepended.
	#[test]
	fn lists_items_tagged_with_their_server() {
		let listed = listed("docs", &json!({ "uri": "file:///a", "name": "a", "_meta": { "x": 1 }, "icons": [] }));
		assert_eq!(listed, json!({ "server": "docs", "uri": "file:///a", "name": "a" }));
	}

	/// The resource tools refuse cursors without a server and unknown servers.
	#[tokio::test]
	async fn read_resource_rejects_unknown_servers() {
		let definitions = create_mcp_resource_tool_definitions(McpExposure::Codemode, Arc::new(|| Vec::new()));
		let read = definitions
			.iter()
			.find(|definition| definition.name == READ_MCP_RESOURCE_TOOL)
			.expect("read tool");
		let error = (read.execute)(
			"id".to_string(),
			json!({ "server": "docs", "uri": "file:///a" }),
			None,
			None,
			None,
		)
		.await
		.expect_err("must fail");
		assert!(error.contains("has no resources"), "unexpected: {error}");
	}
}
