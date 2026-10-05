//! Port of `packages/ai/src/utils/typebox-helpers.ts`.
//!
//! Upstream builds TypeBox schemas (`TSchema`). Rust carries tool parameters
//! as JSON Schema documents (`serde_json::Value`, crate docs deviation D6), so
//! the helpers construct the same JSON Schema objects.

use serde_json::{json, Value};

/// Creates a string enum schema compatible with Google's API and other
/// providers that don't support anyOf/const patterns (`StringEnum`).
pub fn string_enum(values: &[&str], options: Option<StringEnumOptions>) -> Value {
	let mut schema = serde_json::Map::new();
	schema.insert("type".to_string(), Value::String("string".to_string()));
	schema.insert(
		"enum".to_string(),
		Value::Array(values.iter().map(|value| Value::String((*value).to_string())).collect()),
	);
	if let Some(options) = options {
		if let Some(description) = options.description {
			schema.insert("description".to_string(), Value::String(description));
		}
		if let Some(default) = options.default {
			schema.insert("default".to_string(), Value::String(default));
		}
	}
	Value::Object(schema)
}

/// Options of [`string_enum`].
#[derive(Debug, Clone, Default)]
pub struct StringEnumOptions {
	pub description: Option<String>,
	pub default: Option<String>,
}

/// Convenience for building an object schema (`Type.Object` equivalent).
pub fn object_schema(properties: serde_json::Map<String, Value>, required: &[&str]) -> Value {
	json!({
		"type": "object",
		"properties": Value::Object(properties),
		"required": required.iter().map(|name| Value::String((*name).to_string())).collect::<Vec<_>>(),
	})
}
