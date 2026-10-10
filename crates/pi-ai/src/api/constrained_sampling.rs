//! Port of `packages/ai/src/api/constrained-sampling.ts` — JSON-schema strict
//! conversion and grammar-tool input buffering over JSON Schema
//! [`serde_json::Value`]s (upstream TypeBox schemas; crate deviation D6).
//!
//! Deviations:
//! - D22: upstream `UnsupportedStrictJsonSchemaError` / thrown `Error`s become
//!   typed [`UnsupportedStrictJsonSchemaError`] for schema-conversion failures
//!   and `Result<_, String>` (Err carries the upstream message) for the plain
//!   `Error` throws. The `instanceof UnsupportedStrictJsonSchemaError`
//!   discrimination inside `resolveJsonSchemaStrictSampling` is expressed
//!   structurally via the typed [`make_strict_json_schema`] result.
//! - D23: `GrammarConstrainedSampling.format` is the wire string
//!   (`"lark" | "regex"`) so adapters can copy it verbatim into payloads.

use std::collections::{BTreeMap, HashSet};

use crate::types::{
    ConstrainedSamplingConfig, ConstrainedSamplingSetting, GrammarFormat, StrictSchemaMode, Tool,
};
use serde_json::{Map, Value};

/// Upstream `UnsupportedStrictJsonSchemaError`: a schema cannot be converted to
/// the provider strict subset.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct UnsupportedStrictJsonSchemaError {
    pub message: String,
}

impl UnsupportedStrictJsonSchemaError {
    fn new(message: impl Into<String>) -> UnsupportedStrictJsonSchemaError {
        UnsupportedStrictJsonSchemaError {
            message: message.into(),
        }
    }
}

/// Returns true when a provider's strict mode rejects this schema keyword with
/// this value (`UnsupportedStrictSchemaKeywordCheck`).
pub type UnsupportedStrictSchemaKeywordCheck<'a> = &'a dyn Fn(&str, &Value) -> bool;

const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

fn is_json_schema_object(value: &Value) -> bool {
    value.is_object()
}

fn is_structured_schema(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    let types: Vec<&str> = match object.get("type") {
        Some(Value::String(value)) => vec![value.as_str()],
        Some(Value::Array(values)) => values.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    types.contains(&"object")
        || types.contains(&"array")
        || object.contains_key("properties")
        || object.contains_key("items")
}

fn schema_allows_null(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) == Some("null") {
        return true;
    }
    if let Some(Value::Array(types)) = object.get("type") {
        if types.iter().any(|value| value == "null") {
            return true;
        }
    }
    if object.get("const").map(Value::is_null).unwrap_or(false) {
        return true;
    }
    if let Some(Value::Array(values)) = object.get("enum") {
        if values.iter().any(Value::is_null) {
            return true;
        }
    }
    if let Some(Value::Array(variants)) = object.get("anyOf") {
        return variants.iter().any(schema_allows_null);
    }
    false
}

fn make_json_schema_node_strict(
    schema: &mut Value,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck<'_>>,
) -> Result<(), UnsupportedStrictJsonSchemaError> {
    if !is_json_schema_object(schema) {
        return Err(UnsupportedStrictJsonSchemaError::new(
            "boolean schemas are unsupported",
        ));
    }
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if schema.get(key).is_some() {
            return Err(UnsupportedStrictJsonSchemaError::new(format!(
                "{key} schemas are unsupported"
            )));
        }
    }
    if let Some(check) = is_unsupported_keyword {
        for (key, value) in schema.as_object().expect("checked object above") {
            if check(key, value) {
                let serialized = serde_json::to_string(value).expect("JSON value serializes");
                return Err(UnsupportedStrictJsonSchemaError::new(format!(
                    "{key}: {serialized} is unsupported"
                )));
            }
        }
    }

    if schema.get("anyOf").is_some() {
        let Value::Array(variants) = schema.get("anyOf").expect("checked above") else {
            return Err(UnsupportedStrictJsonSchemaError::new(
                "anyOf must contain at least one schema",
            ));
        };
        if variants.is_empty() {
            return Err(UnsupportedStrictJsonSchemaError::new(
                "anyOf must contain at least one schema",
            ));
        }
        for index in 0..variants.len() {
            if is_structured_schema(&schema["anyOf"][index]) {
                return Err(UnsupportedStrictJsonSchemaError::new(
                    "object and array unions are unsupported",
                ));
            }
            make_json_schema_node_strict(&mut schema["anyOf"][index], is_unsupported_keyword)?;
        }
    }

    if schema.get("items").is_some() {
        if schema["items"].is_array() {
            return Err(UnsupportedStrictJsonSchemaError::new(
                "tuple schemas are unsupported",
            ));
        }
        make_json_schema_node_strict(&mut schema["items"], is_unsupported_keyword)?;
    }

    let is_object_schema = schema.get("type").and_then(Value::as_str) == Some("object");
    if schema.get("properties").is_some() && !is_object_schema {
        return Err(UnsupportedStrictJsonSchemaError::new(
            "properties require type object",
        ));
    }
    if !is_object_schema {
        return Ok(());
    }
    if let Some(additional_properties) = schema.get("additionalProperties") {
        if *additional_properties != Value::Bool(false) {
            return Err(UnsupportedStrictJsonSchemaError::new(
                "schema-valued or true additionalProperties is unsupported",
            ));
        }
    }
    if schema.get("properties").is_some() && !schema["properties"].is_object() {
        return Err(UnsupportedStrictJsonSchemaError::new(
            "object properties must be a schema map",
        ));
    }
    if let Some(required) = schema.get("required") {
        let valid = match required {
            Value::Array(entries) => entries.iter().all(Value::is_string),
            _ => false,
        };
        if !valid {
            return Err(UnsupportedStrictJsonSchemaError::new(
                "object required must be a string array",
            ));
        }
    }

    let empty_properties = Map::new();
    let properties = match schema.get("properties") {
        Some(Value::Object(map)) => map,
        _ => &empty_properties,
    };
    let property_names: Vec<String> = properties.keys().cloned().collect();
    let required: HashSet<String> = match schema.get("required") {
        Some(Value::Array(entries)) => entries
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => HashSet::new(),
    };
    if required
        .iter()
        .any(|key| !properties.contains_key(key.as_str()))
    {
        return Err(UnsupportedStrictJsonSchemaError::new(
            "required contains an unknown property",
        ));
    }

    if let Some(Value::Object(properties)) = schema.get_mut("properties") {
        for key in properties.keys().cloned().collect::<Vec<String>>() {
            // Recurse into the original property first (upstream order), then
            // wrap optional non-nullable properties.
            make_json_schema_node_strict(&mut properties[&key], is_unsupported_keyword)?;
            if !required.contains(key.as_str()) {
                let allows_null = properties
                    .get(&key)
                    .map(schema_allows_null)
                    .unwrap_or(false);
                if !allows_null {
                    let property = properties.remove(&key).expect("key from owned snapshot");
                    properties.insert(
                        key,
                        serde_json::json!({ "anyOf": [property, { "type": "null" }] }),
                    );
                }
            }
        }
    }
    schema["required"] = Value::Array(property_names.into_iter().map(Value::String).collect());
    schema["additionalProperties"] = Value::Bool(false);
    Ok(())
}

/// Convert a tool schema to the strict subset expected by provider constrained
/// sampling (`makeStrictJsonSchema`).
pub fn make_strict_json_schema(
    schema: &Value,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck<'_>>,
) -> Result<Map<String, Value>, UnsupportedStrictJsonSchemaError> {
    let mut cloned = schema.clone();
    if !is_json_schema_object(&cloned) {
        return Err(UnsupportedStrictJsonSchemaError::new(
            "root schema must have type object",
        ));
    }
    make_json_schema_node_strict(&mut cloned, is_unsupported_keyword)?;
    if cloned.get("type").and_then(Value::as_str) != Some("object") {
        return Err(UnsupportedStrictJsonSchemaError::new(
            "root schema must have type object",
        ));
    }
    match cloned {
        Value::Object(map) => Ok(map),
        _ => Err(UnsupportedStrictJsonSchemaError::new(
            "root schema must have type object",
        )),
    }
}

pub fn get_json_schema_tool_parameters(
    tool: &Tool,
    strict: bool,
) -> Result<Value, UnsupportedStrictJsonSchemaError> {
    if strict {
        Ok(Value::Object(make_strict_json_schema(
            &tool.parameters,
            None,
        )?))
    } else {
        Ok(tool.parameters.clone())
    }
}

/// Upstream `GrammarConstrainedSampling`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarConstrainedSampling {
    /// `"lark" | "regex"` (deviation D23).
    pub format: String,
    pub definition: String,
    pub input_property: String,
}

/// Upstream `GrammarToolInputJsonBuffer`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GrammarToolInputJsonBuffer {
    pub input: String,
    pub started: bool,
    pub closed: bool,
}

pub fn get_grammar_tool_input(
    tool_name: &str,
    arguments: &Map<String, Value>,
    input_property: &str,
) -> Result<String, String> {
    match arguments.get(input_property) {
        Some(Value::String(input)) => Ok(input.clone()),
        _ => Err(format!(
			"Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
		)),
    }
}

pub fn append_grammar_tool_input_json_delta(
    buffer: &mut GrammarToolInputJsonBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<Option<String>, String> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(None);
        }
        return Err(format!(
            "grammar tool input for property \"{input_property}\" changed after it was closed"
        ));
    }
    if !next_input.starts_with(&buffer.input) {
        return Err(format!(
            "grammar tool input for property \"{input_property}\" changed non-monotonically"
        ));
    }

    let input_delta = &next_input[buffer.input.len()..];
    if !close && input_delta.is_empty() {
        return Ok(None);
    }

    let mut delta = String::new();
    if !buffer.started {
        delta.push('{');
        delta.push_str(
            &serde_json::to_string(input_property)
                .expect("JSON string serialization is infallible"),
        );
        delta.push_str(":\"");
        buffer.started = true;
    }
    // `JSON.stringify(inputDelta).slice(1, -1)`: the escaped body without the
    // surrounding quotes.
    let escaped =
        serde_json::to_string(input_delta).expect("JSON string serialization is infallible");
    delta.push_str(&escaped[1..escaped.len() - 1]);
    buffer.input = next_input.to_string();

    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(Some(delta))
}

fn infer_grammar_input_property(tool: &Tool) -> Result<String, String> {
    let schema = &tool.parameters;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("grammar constrained sampling requires an object parameter schema".to_string());
    }
    let required = schema.get("required").and_then(Value::as_array);
    let Some(required) = required else {
        return Err(
            "grammar constrained sampling requires exactly one required string property"
                .to_string(),
        );
    };
    if required.len() != 1 {
        return Err(
            "grammar constrained sampling requires exactly one required string property"
                .to_string(),
        );
    }
    let Some(input_property) = required[0].as_str() else {
        return Err(
            "grammar constrained sampling requires exactly one required string property"
                .to_string(),
        );
    };
    let property = schema
        .get("properties")
        .and_then(|properties| properties.get(input_property));
    if property.is_none() {
        return Err(format!(
            "grammar constrained sampling requires a properties entry for {input_property}"
        ));
    }
    if property
        .and_then(|property| property.get("type"))
        .and_then(Value::as_str)
        != Some("string")
    {
        return Err(format!(
            "grammar constrained sampling property {input_property} must have type string"
        ));
    }
    Ok(input_property.to_string())
}

/// Decide whether a JSON-schema tool is sent in strict mode.
/// `is_unsupported_keyword` lets a provider reject extra keywords its strict
/// mode does not accept, so "prefer" tools fall back to non-strict.
pub fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
) -> Result<Option<bool>, String> {
    resolve_json_schema_strict_sampling_with(tool, supports_strict_mode, None)
}

/// [`resolve_json_schema_strict_sampling`] with a provider keyword check
/// (upstream's optional third argument).
pub fn resolve_json_schema_strict_sampling_with(
    tool: &Tool,
    supports_strict_mode: bool,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck<'_>>,
) -> Result<Option<bool>, String> {
    let Some(ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::JsonSchema { strict })) =
        tool.constrained_sampling.as_ref()
    else {
        return Ok(None);
    };

    if supports_strict_mode {
        match make_strict_json_schema(&tool.parameters, is_unsupported_keyword) {
            Ok(_) => return Ok(Some(true)),
            Err(error) => {
                if *strict != StrictSchemaMode::Require {
                    return Ok(None);
                }
                return Err(format!(
                    "Tool \"{}\" requires JSON-schema constrained sampling, but {error}.",
                    tool.name
                ));
            }
        }
    }
    if *strict == StrictSchemaMode::Require {
        return Err(format!(
			"Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
			tool.name
		));
    }
    Ok(None)
}

pub fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>, String> {
    let Some(ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::Grammar { variants })) =
        tool.constrained_sampling.as_ref()
    else {
        return Ok(None);
    };

    if !supports_openai_grammar_tools {
        return Ok(None);
    }

    let lark_definition = variants.get(&GrammarFormat::OpenaiLark);
    let regex_definition = variants.get(&GrammarFormat::OpenaiRegex);
    let has_lark_definition = lark_definition
        .map(|definition| !definition.trim().is_empty())
        .unwrap_or(false);
    let has_regex_definition = regex_definition
        .map(|definition| !definition.trim().is_empty())
        .unwrap_or(false);
    if !has_lark_definition && !has_regex_definition {
        return Err(format!(
			"Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
			tool.name
		));
    }

    let input_property = infer_grammar_input_property(tool).map_err(|message| {
        format!(
            "Tool \"{}\" cannot use grammar constrained sampling: {message}.",
            tool.name
        )
    })?;

    Ok(Some(GrammarConstrainedSampling {
        format: if has_lark_definition {
            "lark".to_string()
        } else {
            "regex".to_string()
        },
        definition: if has_lark_definition {
            lark_definition.expect("checked above").clone()
        } else {
            regex_definition.expect("checked above").clone()
        },
        input_property,
    }))
}

pub fn create_grammar_tool_input_properties(
    tools: &[Tool],
    supports_openai_grammar_tools: bool,
) -> Result<BTreeMap<String, String>, String> {
    let mut properties = BTreeMap::new();
    for tool in tools {
        if let Some(grammar) =
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?
        {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    Ok(properties)
}

#[cfg(test)]
mod tests {
    // Port of the parts of test/constrained-sampling.test.ts that belong to
    // this module. The `convertResponsesTools` / `convertResponsesMessages` /
    // `processResponsesStream` assertions ("converts supported constraints and
    // falls back when unsupported", "replays grammar calls as custom Responses
    // items", "drops foreign item ids when replaying grammar calls", "starts
    // custom Responses tool calls with their initial input") belong to
    // api/openai-responses-shared.ts and travel with that module; the strict
    // schema and grammar-buffer assertions are decided here and are ported
    // against this module directly.
    use super::*;

    fn object_entries(value: serde_json::Value) -> Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn derives_strict_provider_schemas_without_changing_tool_definitions() {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "number" },
                "metadata": { "type": "object", "properties": { "enabled": { "type": "boolean" } } },
                "nullable": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["path", "metadata"]
        });

        let strict = make_strict_json_schema(&parameters, None).expect("converts");

        // The original tool definition is untouched.
        assert!(parameters.get("additionalProperties").is_none());
        assert_eq!(
            parameters.get("required"),
            Some(&serde_json::json!(["path", "metadata"]))
        );

        assert_eq!(
            strict.get("additionalProperties"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            strict.get("required"),
            Some(&serde_json::json!([
                "path", "offset", "metadata", "nullable"
            ]))
        );
        let properties = strict
            .get("properties")
            .and_then(Value::as_object)
            .expect("properties");
        assert_eq!(
            properties.get("offset"),
            Some(&serde_json::json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] }))
        );
        assert_eq!(
            properties.get("metadata"),
            Some(&serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["enabled"],
                "properties": { "enabled": { "anyOf": [{ "type": "boolean" }, { "type": "null" }] } }
            }))
        );
        assert_eq!(
            properties.get("nullable"),
            Some(&serde_json::json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] }))
        );
    }

    #[test]
    fn falls_back_or_rejects_schemas_that_cannot_be_safely_converted() {
        let cases: Vec<(Value, &str)> = vec![
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { "metadata": { "type": "object", "additionalProperties": { "type": "string" } } },
                    "required": ["metadata"]
                }),
                "additionalProperties is unsupported",
            ),
            (
                serde_json::json!({
                    "allOf": [
                        { "type": "object", "properties": { "a": { "type": "string" } }, "required": ["a"] },
                        { "type": "object", "properties": { "b": { "type": "number" } }, "required": ["b"] }
                    ]
                }),
                "allOf schemas are unsupported",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": { "anyOf": [
                            { "type": "object", "properties": { "nested": { "type": "string" } }, "required": ["nested"] },
                            { "type": "null" }
                        ] }
                    },
                    "required": ["value"]
                }),
                "object and array unions are unsupported",
            ),
            (
                serde_json::json!({
                    "type": "object",
                    "properties": { "child": { "$ref": "https://example.com/child.json" } },
                    "required": ["child"]
                }),
                "$ref schemas are unsupported",
            ),
        ];

        for (parameters, error) in cases {
            let mut tool = make_tool();
            tool.parameters = parameters.clone();
            tool.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
                ConstrainedSamplingConfig::JsonSchema {
                    strict: StrictSchemaMode::Prefer,
                },
            ));

            let failure = make_strict_json_schema(&parameters, None).expect_err("conversion fails");
            assert!(
                failure.message.contains(error),
                "expected {error} in {}",
                failure.message
            );
            // "prefer" tools fall back to non-strict.
            assert_eq!(resolve_json_schema_strict_sampling(&tool, true), Ok(None));
            // (The matching `convertResponsesTools` fallback assertions belong to
            // openai-responses-shared and travel with that module.)

            tool.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
                ConstrainedSamplingConfig::JsonSchema {
                    strict: StrictSchemaMode::Require,
                },
            ));
            let rejection =
                resolve_json_schema_strict_sampling(&tool, true).expect_err("require rejects");
            assert!(rejection.contains(error), "expected {error} in {rejection}");
        }
    }

    fn make_tool() -> Tool {
        Tool {
            name: "sample_tool".to_string(),
            description: "Sample tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "payload": { "type": "string" } },
                "additionalProperties": false,
                "required": ["payload"]
            }),
            constrained_sampling: None,
        }
    }

    #[test]
    fn keeps_grammar_input_json_deltas_append_only() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        let first = append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"", false)
            .expect("append succeeds")
            .expect("first delta emitted");
        let second = append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"\nb", true)
            .expect("append succeeds")
            .expect("closing delta emitted");

        let combined: Value =
            serde_json::from_str(&format!("{first}{second}")).expect("valid JSON object");
        assert_eq!(combined, serde_json::json!({ "payload": "a\"\nb" }));
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"\nb", true)
                .expect("idempotent close"),
            None
        );
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "payload", "changed", true)
                .expect_err("closed buffer rejects changes"),
            "grammar tool input for property \"payload\" changed after it was closed"
        );
    }

    #[test]
    fn resolves_grammar_constraints_and_reports_unsupported_variants() {
        let mut tool = make_tool();
        tool.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: [(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_string())]
                    .into_iter()
                    .collect(),
            },
        ));

        let grammar = resolve_grammar_constrained_sampling(&tool, true)
            .expect("grammar resolves")
            .expect("grammar produced");
        assert_eq!(
            grammar,
            GrammarConstrainedSampling {
                format: "lark".to_string(),
                definition: "start: /[a-z]+/".to_string(),
                input_property: "payload".to_string(),
            }
        );

        // Grammar tools are skipped when the provider cannot serve them.
        assert_eq!(resolve_grammar_constrained_sampling(&tool, false), Ok(None));

        tool.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: BTreeMap::new(),
            },
        ));
        assert_eq!(
			resolve_grammar_constrained_sampling(&tool, true).expect_err("no variants"),
			"Tool \"sample_tool\" cannot use grammar constrained sampling: no supported grammar variant was provided."
		);

        // Missing or non-string grammar arguments are reported with the
        // upstream message (asserted upstream via the adapter tests).
        let arguments = serde_json::json!({}).as_object().expect("object").clone();
        assert_eq!(
            get_grammar_tool_input("sample_tool", &arguments, "payload")
                .expect_err("missing argument"),
            "Grammar tool call \"sample_tool\" requires argument \"payload\" to be a string."
        );

        let mut tool = make_tool();
        tool.parameters = serde_json::json!({
            "type": "object",
            "properties": { "payload": { "type": "string" } },
            "required": ["payload"]
        });
        tool.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: [(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_string())]
                    .into_iter()
                    .collect(),
            },
        ));
        let properties = create_grammar_tool_input_properties(std::slice::from_ref(&tool), true)
            .expect("properties built");
        assert_eq!(
            properties.get("sample_tool").map(String::as_str),
            Some("payload")
        );

        // `constrainedSampling: false` disables the constraint entirely.
        let mut disabled = make_tool();
        disabled.constrained_sampling = Some(ConstrainedSamplingSetting::Disabled(false));
        assert_eq!(
            resolve_json_schema_strict_sampling(&disabled, true),
            Ok(None)
        );
        assert_eq!(
            resolve_grammar_constrained_sampling(&disabled, true),
            Ok(None)
        );
    }
}
