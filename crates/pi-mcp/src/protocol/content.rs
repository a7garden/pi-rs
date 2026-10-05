//! Port of `packages/mcp/src/protocol/content.ts`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Wire alias for the `_meta` extension field.
pub type Meta = Map<String, Value>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentAnnotations {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub audience: Option<Vec<ContentAudience>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub priority: Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub last_modified: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentAudience {
	User,
	Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub annotations: Option<ContentAnnotations>,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
	pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
	pub data: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub annotations: Option<ContentAnnotations>,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
	pub mime_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
	pub data: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub annotations: Option<ContentAnnotations>,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
	pub mime_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceLinkContent {
	pub uri: String,
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub title: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub mime_type: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub size: Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub annotations: Option<ContentAnnotations>,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextResourceContents {
	pub uri: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub mime_type: Option<String>,
	pub text: String,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlobResourceContents {
	pub uri: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub mime_type: Option<String>,
	pub blob: String,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceContents {
	Text(TextResourceContents),
	Blob(BlobResourceContents),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedResourceContent {
	pub resource: ResourceContents,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub annotations: Option<ContentAnnotations>,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
	#[serde(rename = "text")]
	Text(TextContent),
	#[serde(rename = "image")]
	Image(ImageContent),
	#[serde(rename = "audio")]
	Audio(AudioContent),
	#[serde(rename = "resource_link")]
	ResourceLink(ResourceLinkContent),
	#[serde(rename = "resource")]
	EmbeddedResource(EmbeddedResourceContent),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
	pub content: Vec<ContentBlock>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub structured_content: Option<Map<String, Value>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub is_error: Option<bool>,
	#[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
	pub meta: Option<Meta>,
}

/// Tool result content in the shape LLM APIs accept: text and base64 images. Matches the
/// `TextContent` and `ImageContent` types of `@earendil-works/pi-ai`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LlmContent {
	#[serde(rename = "text")]
	Text { text: String },
	#[serde(rename = "image")]
	Image { data: String, mime_type: String },
}

fn block_to_llm_content(block: &ContentBlock) -> LlmContent {
	match block {
		ContentBlock::Text(text) => LlmContent::Text { text: text.text.clone() },
		ContentBlock::Image(image) => LlmContent::Image {
			data: image.data.clone(),
			mime_type: image.mime_type.clone(),
		},
		ContentBlock::Audio(audio) => LlmContent::Text {
			text: format!("[audio {} omitted]", audio.mime_type),
		},
		ContentBlock::ResourceLink(link) => LlmContent::Text {
			text: format!("{}: {}", link.name, link.uri),
		},
		ContentBlock::EmbeddedResource(embedded) => match &embedded.resource {
			ResourceContents::Text(resource) => LlmContent::Text { text: resource.text.clone() },
			ResourceContents::Blob(resource) => {
				let mime = resource.mime_type.as_deref().unwrap_or_default();
				if mime.starts_with("image/") {
					LlmContent::Image {
						data: resource.blob.clone(),
						mime_type: mime.to_string(),
					}
				} else {
					LlmContent::Text {
						text: format!(
							"[binary resource {} ({}) omitted]",
							resource.uri,
							resource.mime_type.as_deref().unwrap_or("unknown type")
						),
					}
				}
			}
		},
	}
}

/// Convert a tool result to text and image content for a model. Text and images pass through,
/// embedded text resources become text, embedded image resources become images, and other blocks
/// (audio, resource links, binary resources) become a short text placeholder. A result without
/// content blocks but with `structuredContent` becomes its JSON, since servers should, but do not
/// always, mirror structured results as text.
pub fn to_llm_content(result: &CallToolResult) -> Vec<LlmContent> {
	let mut content: Vec<LlmContent> = result.content.iter().map(block_to_llm_content).collect();
	if content.is_empty() {
		if let Some(structured) = &result.structured_content {
			let pretty = Value::Object(structured.clone());
			content.push(LlmContent::Text {
				text: serde_json::to_string_pretty(&pretty).unwrap_or_else(|_| "{}".to_string()),
			});
		}
	}
	content
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	fn parse(value: Value) -> ContentBlock {
		serde_json::from_value(value).expect("valid content block")
	}

	fn llm_text(text: &str) -> LlmContent {
		LlmContent::Text { text: text.to_string() }
	}

	// Port of content.test.ts "passes text and images through and replaces other blocks with text".
	#[test]
	fn passes_text_and_images_through_and_replaces_other_blocks_with_text() {
		let blocks = vec![
			parse(json!({"type": "text", "text": "hello", "annotations": {"priority": 1}})),
			parse(json!({"type": "image", "data": "aW1n", "mimeType": "image/png", "_meta": {"x": 1}})),
			parse(json!({"type": "audio", "data": "YXVk", "mimeType": "audio/wav"})),
			parse(json!({"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"})),
			parse(json!({"type": "resource", "resource": {"uri": "file:///b.txt", "text": "inline"}})),
			parse(json!({"type": "resource", "resource": {"uri": "file:///c.png", "mimeType": "image/png", "blob": "Yw=="}})),
			parse(json!({"type": "resource", "resource": {"uri": "file:///d.bin", "blob": "ZA=="}})),
		];
		let result = CallToolResult {
			content: blocks,
			structured_content: None,
			is_error: None,
			meta: None,
		};
		assert_eq!(
			to_llm_content(&result),
			vec![
				llm_text("hello"),
				LlmContent::Image { data: "aW1n".into(), mime_type: "image/png".into() },
				llm_text("[audio audio/wav omitted]"),
				llm_text("a.txt: file:///a.txt"),
				llm_text("inline"),
				LlmContent::Image { data: "Yw==".into(), mime_type: "image/png".into() },
				llm_text("[binary resource file:///d.bin (unknown type) omitted]"),
			]
		);
	}

	// Port of content.test.ts "falls back to the structured content as JSON when there are no blocks".
	#[test]
	fn falls_back_to_the_structured_content_as_json_when_there_are_no_blocks() {
		let structured: Map<String, Value> = serde_json::from_value(json!({"n": 1})).unwrap();
		let empty = CallToolResult {
			content: vec![],
			structured_content: Some(structured.clone()),
			is_error: None,
			meta: None,
		};
		assert_eq!(to_llm_content(&empty), vec![llm_text("{\n  \"n\": 1\n}")]);

		let both = CallToolResult {
			content: vec![parse(json!({"type": "text", "text": "n=1"}))],
			structured_content: Some(structured),
			is_error: None,
			meta: None,
		};
		assert_eq!(to_llm_content(&both), vec![llm_text("n=1")]);
	}

	#[test]
	fn round_trips_wire_shapes() {
		let value = json!({
			"content": [{"type": "text", "text": "ok"}],
			"structuredContent": {"n": 1},
			"isError": false,
			"_meta": {"x": 1}
		});
		let result: CallToolResult = serde_json::from_value(value.clone()).unwrap();
		assert_eq!(serde_json::to_value(&result).unwrap(), value);
	}
}
