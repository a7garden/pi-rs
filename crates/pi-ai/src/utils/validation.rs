//! Port of `packages/ai/src/utils/validation.ts`.
//!
//! Finds a tool by name and validates tool-call arguments against its JSON
//! Schema, coercing serialized primitives the way AJV/TypeBox `Value.Convert`
//! do and treating `null` as omission for optional non-nullable properties.
//!
//! Deviations (crate docs ledger):
//! - **D20 — TypeBox `Compile`/`Value.Convert` are replaced by a native
//!   JSON-Schema checker and one coercion pass** (crate docs D6: tool
//!   parameters are JSON Schema documents). The
//!   `Object.getOwnPropertySymbols(parameters).includes(TYPEBOX_KIND)` branch
//!   is always false for plain JSON documents, so the plain-schema coercion
//!   path is the only path. `$ref` is resolved within the tool's schema
//!   document via JSON Pointer (`#/$defs/...`).
//! - **D20a — validators are built per call**; the upstream `WeakMap` compile
//!   cache keys off object identity, which has no Rust equivalent.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::types::{Tool, ToolCall};

/// Validation failure surface (upstream `throw new Error(...)`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ValidationError(pub String);

/// Finds a tool by name and validates the tool call arguments against its
/// schema. Returns the validated arguments; `Err` carries the exact upstream
/// error strings (`Tool "<name>" not found`, `Validation failed ...`).
///
/// Upstream `validateToolCall`.
pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<serde_json::Map<String, Value>, ValidationError> {
	let tool = tools.iter().find(|tool| tool.name == tool_call.name);
	let Some(tool) = tool else {
		return Err(ValidationError(format!("Tool \"{}\" not found", tool_call.name)));
	};
	validate_tool_arguments(tool, tool_call)
}

/// Validates tool call arguments against the tool's schema, returning the
/// validated (and potentially coerced) arguments.
///
/// Upstream `validateToolArguments`.
pub fn validate_tool_arguments(
	tool: &Tool,
	tool_call: &ToolCall,
) -> Result<serde_json::Map<String, Value>, ValidationError> {
	let schema = &tool.parameters;
	let mut args = Value::Object(tool_call.arguments.clone());
	normalize_optional_nulls(&mut args, schema, schema);
	coerce_with_json_schema(&mut args, schema, schema);

	if check_value(schema, schema, &args) == Some(true) {
		return Ok(args.as_object().cloned().unwrap_or_default());
	}

	let mut errors = Vec::new();
	collect_errors(schema, schema, &args, "", &mut errors);
	let formatted = if errors.is_empty() {
		"Unknown validation error".to_string()
	} else {
		errors
			.iter()
			.map(|(path, message)| format!("  - {}: {}", path, message))
			.collect::<Vec<_>>()
			.join("\n")
	};

	Err(ValidationError(format!(
		"Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{}",
		tool_call.name,
		formatted,
		serde_json::to_string_pretty(&Value::Object(tool_call.arguments.clone()))
			.expect("arguments serialize")
	)))
}

/// Delete optional non-nullable `null` properties (upstream
/// `normalizeOptionalNulls`).
fn normalize_optional_nulls(value: &mut Value, schema: &Value, root: &Value) {
	if let Value::Array(items) = value {
		match schema.get("items") {
			Some(Value::Array(item_schemas)) => {
				for (index, item) in items.iter_mut().enumerate() {
					if let Some(item_schema) = item_schemas.get(index) {
						normalize_optional_nulls(item, item_schema, root);
					}
				}
			}
			Some(item_schema) if item_schema.is_object() => {
				for item in items.iter_mut() {
					normalize_optional_nulls(item, item_schema, root);
				}
			}
			_ => {}
		}
		return;
	}
	if !value.is_object() {
		return;
	}
	let Some(object) = value.as_object_mut() else {
		return;
	};
	let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
		return;
	};
	let required: BTreeSet<String> = schema
		.get("required")
		.and_then(Value::as_array)
		.map(|required| {
			required
				.iter()
				.filter_map(|name| name.as_str().map(str::to_string))
				.collect()
		})
		.unwrap_or_default();

	for (key, property_schema) in properties {
		let Some(property_value) = object.get(key) else {
			continue;
		};
		let is_optional_null = property_value.is_null()
			&& !required.contains(key)
			&& !property_schema.get("$ref").map(Value::is_string).unwrap_or(false)
			&& check_value(root, property_schema, &Value::Null) == Some(false);
		if is_optional_null {
			object.remove(key);
		} else {
			normalize_optional_nulls(object.get_mut(key).expect("checked above"), property_schema, root);
		}
	}
}

fn get_schema_types(schema: &Value) -> Vec<&str> {
	match schema.get("type") {
		Some(Value::String(type_name)) => vec![type_name.as_str()],
		Some(Value::Array(type_names)) => type_names.iter().filter_map(|name| name.as_str()).collect(),
		_ => Vec::new(),
	}
}

fn matches_json_type(value: &Value, type_name: &str) -> bool {
	match type_name {
		"number" => value.is_number(),
		"integer" => value.as_f64().map(|number| number.fract() == 0.0).unwrap_or(false),
		"boolean" => value.is_boolean(),
		"string" => value.is_string(),
		"null" => value.is_null(),
		"array" => value.is_array(),
		"object" => value.is_object(),
		_ => false,
	}
}

/// JS `Number` parity: an integral float serializes without a fraction, so
/// represent it as an integer (`serde_json` distinguishes 42 from 42.0).
fn number_from_f64(value: f64) -> Value {
	if value.fract() == 0.0 && value.is_finite() && value.abs() <= 9.007_199_254_740_992e15 {
		Value::from(value as i64)
	} else {
		Value::from(value)
	}
}

/// Coerce primitives per schema type (upstream `coercePrimitiveByType`).
fn coerce_primitive_by_type(value: &Value, type_name: &str) -> Value {
	match type_name {
		"number" => match value {
			Value::Null => Value::from(0),
			Value::String(text) if !text.trim().is_empty() => text
				.parse::<f64>()
				.ok()
				.filter(|parsed| parsed.is_finite())
				.map(number_from_f64)
				.unwrap_or_else(|| value.clone()),
			Value::Bool(flag) => Value::from(if *flag { 1 } else { 0 }),
			_ => value.clone(),
		},
		"integer" => match value {
			Value::Null => Value::from(0),
			Value::String(text) if !text.trim().is_empty() => text
				.parse::<f64>()
				.ok()
				.filter(|parsed| parsed.is_finite() && parsed.fract() == 0.0)
				.map(number_from_f64)
				.unwrap_or_else(|| value.clone()),
			Value::Bool(flag) => Value::from(if *flag { 1 } else { 0 }),
			_ => value.clone(),
		},
		"boolean" => match value {
			Value::Null => Value::Bool(false),
			Value::String(text) if text == "true" => Value::Bool(true),
			Value::String(text) if text == "false" => Value::Bool(false),
			Value::Number(number) if number.as_f64() == Some(1.0) => Value::Bool(true),
			Value::Number(number) if number.as_f64() == Some(0.0) => Value::Bool(false),
			_ => value.clone(),
		},
		"string" => match value {
			Value::Null => Value::String(String::new()),
			Value::Number(number) => Value::String(number.to_string()),
			Value::Bool(flag) => Value::String(flag.to_string()),
			_ => value.clone(),
		},
		"null" => match value {
			Value::String(text) if text.is_empty() => Value::Null,
			Value::Number(number) if number.as_f64() == Some(0.0) => Value::Null,
			Value::Bool(false) => Value::Null,
			_ => value.clone(),
		},
		_ => value.clone(),
	}
}

fn apply_schema_object_coercion(object: &mut serde_json::Map<String, Value>, schema: &Value, root: &Value) {
	let properties = schema.get("properties").and_then(Value::as_object);
	let defined_keys: BTreeSet<&String> = properties
		.map(|properties| properties.keys().collect())
		.unwrap_or_default();

	if let Some(properties) = properties {
		for (key, property_schema) in properties {
			if let Some(property_value) = object.get_mut(key) {
				coerce_with_json_schema(property_value, property_schema, root);
			}
		}
	}

	if let Some(additional) = schema.get("additionalProperties").filter(|additional| additional.is_object()) {
		for (key, property_value) in object.iter_mut() {
			if defined_keys.contains(&key) {
				continue;
			}
			coerce_with_json_schema(property_value, additional, root);
		}
	}
}

fn apply_schema_array_coercion(items: &mut Vec<Value>, schema: &Value, root: &Value) {
	match schema.get("items") {
		Some(Value::Array(item_schemas)) => {
			for (index, item) in items.iter_mut().enumerate() {
				let Some(item_schema) = item_schemas.get(index) else {
					continue;
				};
				coerce_with_json_schema(item, item_schema, root);
			}
		}
		Some(item_schema) if item_schema.is_object() => {
			for item in items.iter_mut() {
				coerce_with_json_schema(item, item_schema, root);
			}
		}
		_ => {}
	}
}

fn coerce_with_union_schema(value: &mut Value, schemas: &[Value], root: &Value) {
	for schema in schemas {
		if check_value(root, schema, value) == Some(true) {
			return;
		}
	}

	for schema in schemas {
		let mut candidate = value.clone();
		coerce_with_json_schema(&mut candidate, schema, root);
		if check_value(root, schema, &candidate) == Some(true) {
			*value = candidate;
			return;
		}
	}
}

/// Upstream `coerceWithJsonSchema`: mutates `value` in place.
fn coerce_with_json_schema(value: &mut Value, schema: &Value, root: &Value) {
	if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
		for nested in all_of {
			coerce_with_json_schema(value, nested, root);
		}
	}
	if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
		coerce_with_union_schema(value, any_of, root);
	}
	if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
		coerce_with_union_schema(value, one_of, root);
	}

	let schema_types = get_schema_types(schema);
	let matches_union_member = schema_types.len() > 1 && schema_types.iter().any(|schema_type| matches_json_type(value, schema_type));
	if !schema_types.is_empty() && !matches_union_member {
		for schema_type in &schema_types {
			let candidate = coerce_primitive_by_type(value, schema_type);
			if candidate != *value {
				*value = candidate;
				break;
			}
		}
	}

	if schema_types.contains(&"object") {
		if let Value::Object(object) = value {
			apply_schema_object_coercion(object, schema, root);
		}
	}

	if schema_types.contains(&"array") {
		if let Value::Array(items) = value {
			apply_schema_array_coercion(items, schema, root);
		}
	}
}

/// Resolve a local `$ref` JSON Pointer (`#/...`) within the root document.
fn resolve_ref<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
	let pointer = reference.strip_prefix('#')?;
	let mut current = root;
	for segment in pointer.split('/').filter(|segment| !segment.is_empty()) {
		let segment = segment.replace("~1", "/").replace("~0", "~");
		current = match current {
			Value::Object(object) => object.get(&segment)?,
			Value::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
			_ => return None,
		};
	}
	Some(current)
}

/// Check `value` against `schema`; `None` mirrors upstream
/// `getSubSchemaValidator` returning `undefined` (schema cannot compile).
fn check_value(root: &Value, schema: &Value, value: &Value) -> Option<bool> {
	let schema_object = schema.as_object()?;
	if let Some(reference) = schema_object.get("$ref").and_then(Value::as_str) {
		let resolved = resolve_ref(root, reference)?;
		return check_value(root, resolved, value);
	}

	if let Some(all_of) = schema_object.get("allOf").and_then(Value::as_array) {
		if !all_of.iter().all(|nested| check_value(root, nested, value).unwrap_or(false)) {
			return Some(false);
		}
	}
	if let Some(any_of) = schema_object.get("anyOf").and_then(Value::as_array) {
		if !any_of.iter().any(|nested| check_value(root, nested, value).unwrap_or(false)) {
			return Some(false);
		}
	}
	if let Some(one_of) = schema_object.get("oneOf").and_then(Value::as_array) {
		let matches = one_of
			.iter()
			.filter(|nested| check_value(root, nested, value).unwrap_or(false))
			.count();
		if matches != 1 {
			return Some(false);
		}
	}

	let schema_types = get_schema_types(schema);
	if !schema_types.is_empty() && !schema_types.iter().any(|schema_type| matches_json_type(value, schema_type)) {
		return Some(false);
	}

	if schema_types.contains(&"object") {
		if let Value::Object(object) = value {
			let properties = schema_object.get("properties").and_then(Value::as_object);
			let required: Vec<&str> = schema_object
				.get("required")
				.and_then(Value::as_array)
				.map(|required| required.iter().filter_map(Value::as_str).collect())
				.unwrap_or_default();
			for name in &required {
				if !object.contains_key(*name) {
					return Some(false);
				}
			}
			if let Some(properties) = properties {
				for (key, property_schema) in properties {
					if let Some(property_value) = object.get(key) {
						if !check_value(root, property_schema, property_value).unwrap_or(false) {
							return Some(false);
						}
					}
				}
			}
			match schema_object.get("additionalProperties") {
				Some(Value::Bool(false)) => {
					let defined = properties.map(|properties| properties.keys().cloned().collect::<BTreeSet<_>>()).unwrap_or_default();
					if object.keys().any(|key| !defined.contains(key)) {
						return Some(false);
					}
				}
				Some(additional @ Value::Object(_)) => {
					let defined = properties.map(|properties| properties.keys().cloned().collect::<BTreeSet<_>>()).unwrap_or_default();
					for (key, property_value) in object {
						if defined.contains(key) {
							continue;
						}
						if !check_value(root, additional, property_value).unwrap_or(false) {
							return Some(false);
						}
					}
				}
				_ => {}
			}
		}
	}

	if schema_types.contains(&"array") {
		if let Value::Array(items) = value {
			match schema_object.get("items") {
				Some(Value::Array(item_schemas)) => {
					for (index, item) in items.iter().enumerate() {
						let Some(item_schema) = item_schemas.get(index) else {
							continue;
						};
						if !check_value(root, item_schema, item).unwrap_or(false) {
							return Some(false);
						}
					}
				}
				Some(item_schema) if item_schema.is_object() => {
					if !items.iter().all(|item| check_value(root, item_schema, item).unwrap_or(false)) {
						return Some(false);
					}
				}
				_ => {}
			}
		}
	}

	Some(true)
}

fn instance_path_to_name(instance_path: &str) -> String {
	let path = instance_path.strip_prefix('/').unwrap_or(instance_path).replace('/', ".");
	if path.is_empty() {
		"root".to_string()
	} else {
		path
	}
}

/// Upstream `formatValidationPath`.
fn format_validation_path(keyword: Option<&str>, instance_path: &str, required_property: Option<&str>) -> String {
	if keyword == Some("required") {
		if let Some(required_property) = required_property {
			let base_path = instance_path.strip_prefix('/').unwrap_or(instance_path).replace('/', ".");
			return if base_path.is_empty() {
				required_property.to_string()
			} else {
				format!("{}.{}", base_path, required_property)
			};
		}
	}
	instance_path_to_name(instance_path)
}

/// Collect `(formatted path, message)` pairs (upstream `validator.Errors`).
fn collect_errors(root: &Value, schema: &Value, value: &Value, instance_path: &str, errors: &mut Vec<(String, String)>) {
	let Some(schema_object) = schema.as_object() else {
		return;
	};
	if let Some(reference) = schema_object.get("$ref").and_then(Value::as_str) {
		if let Some(resolved) = resolve_ref(root, reference) {
			collect_errors(root, resolved, value, instance_path, errors);
		}
		return;
	}

	if let Some(all_of) = schema_object.get("allOf").and_then(Value::as_array) {
		for nested in all_of {
			collect_errors(root, nested, value, instance_path, errors);
		}
	}
	if let Some(any_of) = schema_object.get("anyOf").and_then(Value::as_array) {
		if !any_of.iter().any(|nested| check_value(root, nested, value).unwrap_or(false)) {
			errors.push((instance_path_to_name(instance_path), "Expected union value".to_string()));
			return;
		}
	}
	if let Some(one_of) = schema_object.get("oneOf").and_then(Value::as_array) {
		let matches = one_of
			.iter()
			.filter(|nested| check_value(root, nested, value).unwrap_or(false))
			.count();
		if matches != 1 {
			errors.push((instance_path_to_name(instance_path), "Expected union value".to_string()));
			return;
		}
	}

	let schema_types = get_schema_types(schema);
	if !schema_types.is_empty() && !schema_types.iter().any(|schema_type| matches_json_type(value, schema_type)) {
		let message = if schema_types.len() == 1 {
			format!("Expected {}", schema_types[0])
		} else {
			"Expected union value".to_string()
		};
		errors.push((instance_path_to_name(instance_path), message));
		return;
	}

	if schema_types.contains(&"object") {
		if let Value::Object(object) = value {
			let properties = schema_object.get("properties").and_then(Value::as_object);
			for name in schema_object
				.get("required")
				.and_then(Value::as_array)
				.map(|required| required.iter().filter_map(Value::as_str).collect::<Vec<_>>())
				.unwrap_or_default()
			{
				if !object.contains_key(name) {
					errors.push((
						format_validation_path(Some("required"), instance_path, Some(name)),
						"Required property".to_string(),
					));
				}
			}
			if let Some(properties) = properties {
				for (key, property_schema) in properties {
					if let Some(property_value) = object.get(key) {
						collect_errors(
							root,
							property_schema,
							property_value,
							&format!("{}/{}", instance_path, key),
							errors,
						);
					}
				}
			}
		}
	}

	if schema_types.contains(&"array") {
		if let Value::Array(items) = value {
			match schema_object.get("items") {
				Some(Value::Array(item_schemas)) => {
					for (index, item) in items.iter().enumerate() {
						let Some(item_schema) = item_schemas.get(index) else {
							continue;
						};
						collect_errors(root, item_schema, item, &format!("{}/{}", instance_path, index), errors);
					}
				}
				Some(item_schema) if item_schema.is_object() => {
					for (index, item) in items.iter().enumerate() {
						collect_errors(root, item_schema, item, &format!("{}/{}", instance_path, index), errors);
					}
				}
				_ => {}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	fn tool_with_parameters(parameters: Value) -> Tool {
		Tool {
			name: "echo".to_string(),
			description: "Echo tool".to_string(),
			parameters,
			constrained_sampling: None,
		}
	}

	fn tool_call_with_arguments(arguments: Value) -> ToolCall {
		ToolCall {
			id: "tool-1".to_string(),
			name: "echo".to_string(),
			arguments: arguments.as_object().cloned().expect("object arguments"),
			thought_signature: None,
			namespace: None,
		}
	}

	fn create_tool_call_with_plain_schema(schema: Value, value: Value) -> (Tool, ToolCall) {
		let tool = tool_with_parameters(json!({
			"type": "object",
			"properties": { "value": schema },
			"required": ["value"],
		}));
		let tool_call = tool_call_with_arguments(json!({ "value": value }));
		(tool, tool_call)
	}

	fn validated_arguments(tool: &Tool, tool_call: &ToolCall) -> serde_json::Map<String, Value> {
		validate_tool_arguments(tool, tool_call).expect("validation succeeds")
	}

	#[test]
	fn still_validates_when_the_function_constructor_is_unavailable() {
		// The upstream case also mocks `globalThis.Function` away (CSP); the Rust
		// checker never uses codegen, so only the interpreted validation path exists.
		let tool = tool_with_parameters(json!({
			"type": "object",
			"properties": { "count": { "type": "number" } },
			"required": ["count"],
		}));
		let tool_call = tool_call_with_arguments(json!({ "count": "42" }));

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(arguments, json!({ "count": 42 }).as_object().cloned().expect("object"));
	}

	#[test]
	fn coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules() {
		let passing_cases: Vec<(Value, Value, Value)> = vec![
			(json!({ "type": "number" }), json!("42"), json!(42)),
			(json!({ "type": "number" }), json!(true), json!(1)),
			(json!({ "type": "number" }), Value::Null, json!(0)),
			(json!({ "type": "integer" }), json!("42"), json!(42)),
			(json!({ "type": "boolean" }), json!("true"), json!(true)),
			(json!({ "type": "boolean" }), json!("false"), json!(false)),
			(json!({ "type": "boolean" }), json!(1), json!(true)),
			(json!({ "type": "boolean" }), json!(0), json!(false)),
			(json!({ "type": "string" }), Value::Null, json!("")),
			(json!({ "type": "string" }), json!(true), json!("true")),
			(json!({ "type": "null" }), json!(""), Value::Null),
			(json!({ "type": "null" }), json!(0), Value::Null),
			(json!({ "type": "null" }), json!(false), Value::Null),
			(json!({ "type": ["number", "string"] }), json!("1"), json!("1")),
			(json!({ "type": ["boolean", "number"] }), json!("1"), json!(1)),
		];

		for (schema, input, expected) in passing_cases {
			let (tool, tool_call) = create_tool_call_with_plain_schema(schema, input);
			let arguments = validated_arguments(&tool, &tool_call);
			assert_eq!(arguments, json!({ "value": expected }).as_object().cloned().expect("object"));
		}
	}

	#[test]
	fn treats_null_as_omission_for_optional_non_nullable_properties() {
		let tool = tool_with_parameters(json!({
			"type": "object",
			"properties": {
				"path": { "type": "string" },
				"offset": { "type": "number" },
				"nullable": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
				"metadata": {
					"type": "object",
					"properties": { "enabled": { "type": "boolean" } },
				},
			},
			"required": ["path"],
		}));
		let tool_call = tool_call_with_arguments(json!({
			"path": "file.txt",
			"offset": null,
			"nullable": null,
			"metadata": { "enabled": null },
		}));

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(
			arguments,
			json!({ "path": "file.txt", "nullable": null, "metadata": {} })
				.as_object()
				.cloned()
				.expect("object")
		);
	}

	#[test]
	fn preserves_optional_nulls_whose_referenced_schema_is_nullable() {
		let tool = tool_with_parameters(json!({
			"type": "object",
			"properties": { "value": { "$ref": "#/$defs/value" } },
			"$defs": { "value": { "anyOf": [{ "type": "number" }, { "type": "null" }] } },
		}));
		let tool_call = tool_call_with_arguments(json!({ "value": null }));

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(arguments, json!({ "value": null }).as_object().cloned().expect("object"));
	}

	#[test]
	fn preserves_a_value_that_already_matches_a_nullable_union_arm() {
		let tool = tool_with_parameters(json!({
			"type": "object",
			"properties": { "value": { "anyOf": [{ "type": "number" }, { "type": "null" }] } },
			"required": ["value"],
		}));
		let tool_call = tool_call_with_arguments(json!({ "value": null }));

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(arguments, json!({ "value": null }).as_object().cloned().expect("object"));
	}

	#[test]
	fn preserves_a_value_that_already_matches_a_one_of_nullable_union_arm() {
		let (tool, tool_call) = create_tool_call_with_plain_schema(
			json!({ "oneOf": [{ "type": "number" }, { "type": "null" }] }),
			Value::Null,
		);

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(arguments, json!({ "value": null }).as_object().cloned().expect("object"));
	}

	#[test]
	fn still_coerces_nullable_unions_when_the_original_value_does_not_match_any_arm() {
		let (tool, tool_call) = create_tool_call_with_plain_schema(
			json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] }),
			json!("42"),
		);

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(arguments, json!({ "value": 42 }).as_object().cloned().expect("object"));
	}

	#[test]
	fn accepts_null_for_nullable_array_schemas_with_items() {
		// The upstream case additionally compiles and runs TypeBox's generated
		// validator via `new Function(...)`; the Rust checker has no codegen path,
		// so only the interpreted assertions apply.
		let (tool, tool_call) = create_tool_call_with_plain_schema(
			json!({ "type": ["array", "null"], "items": { "type": "string" } }),
			Value::Null,
		);

		let arguments = validated_arguments(&tool, &tool_call);
		assert_eq!(arguments, json!({ "value": null }).as_object().cloned().expect("object"));
	}

	#[test]
	fn rejects_invalid_coercions_for_serialized_plain_json_schemas() {
		let failing_cases: Vec<(Value, Value)> = vec![
			(json!({ "type": "boolean" }), json!("1")),
			(json!({ "type": "boolean" }), json!("0")),
			(json!({ "type": "null" }), json!("null")),
			(json!({ "type": "integer" }), json!("42.1")),
		];

		for (schema, input) in failing_cases {
			let (tool, tool_call) = create_tool_call_with_plain_schema(schema, input);
			let error = validate_tool_arguments(&tool, &tool_call).expect_err("validation fails");
			assert!(error.to_string().starts_with("Validation failed for tool \"echo\""), "{error}");
			assert!(error.to_string().contains("Received arguments:"), "{error}");
		}
	}

	#[test]
	fn reports_a_missing_tool_by_name() {
		let tool = tool_with_parameters(json!({ "type": "object", "properties": {} }));
		let mut tool_call = tool_call_with_arguments(json!({}));
		tool_call.name = "missing".to_string();

		let error = validate_tool_call(&[tool], &tool_call).expect_err("tool lookup fails");
		assert_eq!(error.to_string(), "Tool \"missing\" not found");
	}
}
