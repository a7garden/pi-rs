//! Mirrors upstream `src/declarations.ts`: tool-schema → TypeScript declaration string
//! generation.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use crate::types::CodemodeJsonSchema;

const IDENTIFIER_START: fn(char) -> bool = |c| c.is_ascii_alphabetic() || c == '_' || c == '$';
const INDENT: &str = "  ";
/// Largest rendered input type, in characters, before it becomes `unknown`.
pub const DEFAULT_INPUT_SCHEMA_MAX_CHARS: usize = 16_000;
/// Local `$ref` expansions per rendered schema, so shared definitions cannot blow up the output.
const MAX_REF_EXPANSIONS: usize = 32;

/// TypeScript types for MCP results, from the MCP `CallToolResult` schema, so `CallToolResult<T>`
/// declarations can refer to them.
pub const MCP_TYPESCRIPT_PREAMBLE: &str = include_str!("mcp_typescript_preamble.txt");

pub use crate::identifier::to_codemode_identifier;

#[derive(Debug, Clone, Default)]
pub struct RenderDeclarationsOptions<'a> {
    pub tools: &'a [crate::types::CodemodeTool],
    pub globals: &'a [crate::types::CodemodeTool],
}

/// Render TypeScript declarations for the script-visible API. Tools become members of
/// `declare const tools`, globals become `declare function` statements, and `ns.member` globals
/// members of `declare const ns`. Descriptions become doc comments; schemas become types.
pub fn render_declarations(options: RenderDeclarationsOptions<'_>) -> String {
    let mut sections: Vec<String> = Vec::new();
    let tools = options.tools;
    if !tools.is_empty() {
        let members: Vec<String> = tools
            .iter()
            .map(|tool| {
                format!(
                    "{}{}{}",
                    doc_comment(tool.description.as_deref(), INDENT),
                    INDENT,
                    render_tool_signature_opts(tool, None)
                )
            })
            .collect();
        sections.push(format!(
            "declare const tools: {{\n{}\n}};",
            members.join("\n")
        ));
    }
    let mut namespaces: HashMap<&str, Vec<String>> = HashMap::new();
    let mut namespace_order: Vec<&str> = Vec::new();
    for global in options.globals {
        match global.name.find('.') {
            None => {
                sections.push(render_global(
                    &format!("declare function {}", global.name),
                    global,
                    "",
                ));
            }
            Some(dot) => {
                let namespace = &global.name[..dot];
                let members = namespaces.entry(namespace).or_insert_with(|| {
                    namespace_order.push(namespace);
                    Vec::new()
                });
                members.push(render_global(&global.name[dot + 1..], global, INDENT));
            }
        }
    }
    for namespace in namespace_order {
        let members = &namespaces[namespace];
        sections.push(format!(
            "declare const {namespace}: {{\n{}\n}};",
            members.join("\n")
        ));
    }
    sections.join("\n\n")
}

/// One tool as a member of the `tools` object: `name(args: T): Promise<R>;` with the
/// name as the identifier scripts use. Input types longer than `input_max_chars` render as
/// `unknown`. Tools whose output schema is an MCP `CallToolResult` render as
/// `Promise<CallToolResult<T>>`, which needs [`MCP_TYPESCRIPT_PREAMBLE`].
pub fn render_tool_signature(tool: &SignatureTool<'_>) -> String {
    render_tool_signature_opts(&tool.as_tool(), tool.input_max_chars)
}

fn render_tool_signature_opts(tool: &impl SchemaAccess, input_max_chars: Option<usize>) -> String {
    let input = match tool.input_schema() {
        None => "unknown".to_string(),
        Some(schema) => schema_to_type_with(
            schema,
            Some(input_max_chars.unwrap_or(DEFAULT_INPUT_SCHEMA_MAX_CHARS)),
        ),
    };
    format!(
        "{}(args: {}): Promise<{}>;",
        to_codemode_identifier(tool.name()),
        input,
        render_tool_output_type(tool.output_schema())
    )
}

/// A tool's sample: the description followed by the tool's declaration. Used for tool
/// listings and `ALL_TOOLS` entries.
pub fn render_tool_sample(tool: &SampleTool<'_>) -> String {
    let tool_ref = tool.as_tool();
    let declaration = format!(
        "declare const tools: {{ {} }};",
        render_tool_signature_opts(&tool_ref, tool.input_max_chars)
    );
    format!(
        "{}\n\ncodemode tool declaration:\n```ts\n{}\n```",
        tool.description.as_deref().unwrap_or("").trim(),
        declaration
    )
}

/// Borrowed view over the fields [`render_tool_signature`] needs
/// (upstream `Pick<CodemodeTool, "name" | "inputSchema" | "outputSchema">`).
pub struct SignatureTool<'a> {
    pub name: &'a str,
    pub input_schema: Option<&'a CodemodeJsonSchema>,
    pub output_schema: Option<&'a CodemodeJsonSchema>,
    pub input_max_chars: Option<usize>,
}

impl<'a> SignatureTool<'a> {
    fn as_tool(&self) -> SignatureTool<'a> {
        SignatureTool {
            name: self.name,
            input_schema: self.input_schema,
            output_schema: self.output_schema,
            input_max_chars: self.input_max_chars,
        }
    }
}

impl SchemaAccess for SignatureTool<'_> {
    fn name(&self) -> &str {
        self.name
    }
    fn input_schema(&self) -> Option<&CodemodeJsonSchema> {
        self.input_schema
    }
    fn output_schema(&self) -> Option<&CodemodeJsonSchema> {
        self.output_schema
    }
}

/// Borrowed view over the fields [`render_tool_sample`] needs
/// (upstream `Pick<CodemodeTool, "name" | "description" | "inputSchema" | "outputSchema">`).
pub struct SampleTool<'a> {
    pub name: &'a str,
    pub description: Option<&'a str>,
    pub input_schema: Option<&'a CodemodeJsonSchema>,
    pub output_schema: Option<&'a CodemodeJsonSchema>,
    pub input_max_chars: Option<usize>,
}

impl SampleTool<'_> {
    fn as_tool(&self) -> SampleTool<'_> {
        SampleTool {
            name: self.name,
            description: self.description,
            input_schema: self.input_schema,
            output_schema: self.output_schema,
            input_max_chars: self.input_max_chars,
        }
    }
}

impl SchemaAccess for SampleTool<'_> {
    fn name(&self) -> &str {
        self.name
    }
    fn input_schema(&self) -> Option<&CodemodeJsonSchema> {
        self.input_schema
    }
    fn output_schema(&self) -> Option<&CodemodeJsonSchema> {
        self.output_schema
    }
}

trait SchemaAccess {
    fn name(&self) -> &str;
    fn input_schema(&self) -> Option<&CodemodeJsonSchema>;
    fn output_schema(&self) -> Option<&CodemodeJsonSchema>;
}

impl SchemaAccess for crate::types::CodemodeTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn input_schema(&self) -> Option<&CodemodeJsonSchema> {
        self.input_schema.as_ref()
    }
    fn output_schema(&self) -> Option<&CodemodeJsonSchema> {
        self.output_schema.as_ref()
    }
}

/// The `structuredContent` schema of an MCP `CallToolResult` output schema (detected by a
/// `content` array of objects, boolean `isError`, and object `_meta`), `Some(Value::Bool(true))`
/// when it declares none, or `None` when the schema is not a `CallToolResult`.
pub fn mcp_structured_content_schema(
    schema: Option<&CodemodeJsonSchema>,
) -> Option<CodemodeJsonSchema> {
    let schema = schema?;
    let properties = schema.get("properties")?.as_object()?;
    let content = properties.get("content")?;
    let content = content.as_object()?;
    if content.get("type")?.as_str()? != "array" {
        return None;
    }
    let items = content.get("items")?.as_object()?;
    if items.get("type")?.as_str()? != "object" {
        return None;
    }
    let is_error = properties.get("isError")?;
    if is_error.get("type")?.as_str()? != "boolean" {
        return None;
    }
    let meta = properties.get("_meta")?;
    if meta.get("type")?.as_str()? != "object" {
        return None;
    }
    match properties.get("structuredContent") {
        Some(sc) if sc.is_object() || sc.is_boolean() => Some(sc.clone()),
        _ => Some(Value::Bool(true)),
    }
}

/// The type a tool call resolves to: `CallToolResult<T>` for MCP output schemas (needs
/// [`MCP_TYPESCRIPT_PREAMBLE`]), the schema's type otherwise, and `unknown` without a schema.
pub fn render_tool_output_type(schema: Option<&CodemodeJsonSchema>) -> String {
    if let Some(structured) = mcp_structured_content_schema(schema) {
        let ty = schema_to_type_with(&structured, None);
        return if ty == "unknown" {
            "CallToolResult".to_string()
        } else {
            format!("CallToolResult<{ty}>")
        };
    }
    match schema {
        None => "unknown".to_string(),
        Some(schema) => schema_to_type_with(schema, None),
    }
}

fn render_global(head: &str, global: &crate::types::CodemodeTool, indent: &str) -> String {
    if let Some(signature) = &global.signature {
        return format!(
            "{}{}{}{signature};",
            doc_comment(global.description.as_deref(), indent),
            indent,
            head
        );
    }
    let input = match &global.input_schema {
        None => "unknown".to_string(),
        Some(schema) => schema_to_type_with(schema, None),
    };
    let output = match &global.output_schema {
        None => "unknown".to_string(),
        Some(schema) => schema_to_type_with(schema, None),
    };
    format!(
        "{}{}{}(args: {input}): Promise<{output}>;",
        doc_comment(global.description.as_deref(), indent),
        indent,
        head
    )
}

fn doc_comment(description: Option<&str>, indent: &str) -> String {
    let text = match description.map(str::trim) {
        Some(text) if !text.is_empty() => text,
        _ => return String::new(),
    };
    let text = text.replace("*/", "*\\/");
    let lines: Vec<&str> = split_crlf(&text);
    if lines.len() == 1 {
        return format!("{indent}/** {} */\n", lines[0]);
    }
    let body: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.is_empty() {
                format!("{indent} *")
            } else {
                format!("{indent} * {line}")
            }
        })
        .collect();
    format!("{indent}/**\n{}\n{indent} */\n", body.join("\n"))
}

/// JS `text.split(/\r?\n/)`.
fn split_crlf(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut current = text;
    while let Some(pos) = current.find('\n') {
        let (line, rest) = current.split_at(pos);
        lines.push(line.strip_suffix('\r').unwrap_or(line));
        current = &rest[1..];
    }
    lines.push(current);
    lines
}

fn property_key(name: &str) -> String {
    if is_identifier(name) {
        name.to_string()
    } else {
        Value::String(name.to_string()).to_string()
    }
}

/// JS `/^[A-Za-z_$][A-Za-z0-9_$]*$/`.
fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if IDENTIFIER_START(c) => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

fn union(types: Vec<String>) -> String {
    let mut unique: Vec<String> = Vec::new();
    for ty in types {
        if !unique.contains(&ty) {
            unique.push(ty);
        }
    }
    if unique.iter().any(|ty| ty == "unknown") {
        return "unknown".to_string();
    }
    if unique.is_empty() {
        "never".to_string()
    } else {
        unique.join(" | ")
    }
}

/// Convert a JSON Schema to a TypeScript type expression: objects on one line
/// (`{ a: string; b?: number; }`) with properties sorted by name, or one property per line
/// with `//` comments when a property has a description; `Array<T>` for arrays. Local
/// references (`#/$defs/...`, `#/definitions/...`) resolve against `schema`; recursive and
/// remote references render as `unknown`. A result longer than `max_chars` renders as `unknown`.
pub fn schema_to_type(schema: &CodemodeJsonSchema) -> String {
    schema_to_type_with(schema, None)
}

/// Like [`schema_to_type`] with an explicit character budget (`None` disables the cap).
pub fn schema_to_type_with(schema: &CodemodeJsonSchema, max_chars: Option<usize>) -> String {
    let mut context = SchemaContext {
        root: schema,
        resolving: HashSet::new(),
        expansions: 0,
    };
    let ty = to_type(schema, &mut context);
    match max_chars {
        Some(max) if ty.chars().count() > max => "unknown".to_string(),
        _ => ty,
    }
}

struct SchemaContext<'a> {
    root: &'a CodemodeJsonSchema,
    /// References being expanded on the current path, to stop at recursive types.
    resolving: HashSet<String>,
    expansions: usize,
}

fn resolve_ref(reference: &str, root: &CodemodeJsonSchema) -> Option<CodemodeJsonSchema> {
    if reference != "#" && !reference.starts_with("#/") {
        return None;
    }
    let mut current = root.clone();
    for segment in reference[2..].split('/').filter(|s| !s.is_empty()) {
        let key = percent_decode(segment)
            .replace("~1", "/")
            .replace("~0", "~");
        let next = current.get(&key)?;
        current = next.clone();
    }
    if current.is_boolean() || current.is_object() {
        Some(current)
    } else {
        None
    }
}

/// `decodeURIComponent` for a single JSON-pointer segment (ASCII %XX escapes).
fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&segment[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn to_type(schema: &CodemodeJsonSchema, context: &mut SchemaContext<'_>) -> String {
    match schema {
        Value::Bool(true) => return "unknown".to_string(),
        Value::Bool(false) => return "never".to_string(),
        Value::Object(_) => {}
        _ => return "unknown".to_string(),
    }
    if let Some(Value::String(reference)) = schema.get("$ref") {
        let reference = reference.clone();
        if context.resolving.contains(&reference) || context.expansions >= MAX_REF_EXPANSIONS {
            return "unknown".to_string();
        }
        let target = match resolve_ref(&reference, context.root) {
            Some(target) => target,
            None => return "unknown".to_string(),
        };
        context.expansions += 1;
        context.resolving.insert(reference.clone());
        let result = to_type(&target, context);
        context.resolving.remove(&reference);
        return result;
    }

    if let Some(const_value) = schema.get("const") {
        return const_value.to_string();
    }
    if let Some(Value::Array(enum_values)) = schema.get("enum") {
        return union(enum_values.iter().map(|v| v.to_string()).collect());
    }

    let variants = schema
        .get("anyOf")
        .filter(|v| v.is_array())
        .or_else(|| schema.get("oneOf").filter(|v| v.is_array()));
    if let Some(Value::Array(variants)) = variants {
        return union(variants.iter().map(|v| to_type(v, context)).collect());
    }
    if let Some(Value::Array(all_of)) = schema.get("allOf") {
        let parts: Vec<String> = all_of
            .iter()
            .map(|part| to_type(part, context))
            .filter(|part| part != "unknown")
            .collect();
        if parts.is_empty() {
            return "unknown".to_string();
        }
        return parts
            .into_iter()
            .map(|part| {
                if part.contains(" | ") {
                    format!("({part})")
                } else {
                    part
                }
            })
            .collect::<Vec<_>>()
            .join(" & ");
    }

    match schema.get("type") {
        Some(Value::Array(types)) => {
            let rendered: Vec<String> = types
                .iter()
                .map(|entry| {
                    let mut single = schema.as_object().cloned().unwrap_or_else(Map::new);
                    single.insert("type".to_string(), entry.clone());
                    to_type(&Value::Object(single), context)
                })
                .collect();
            union(rendered)
        }
        Some(Value::String(kind)) => match kind.as_str() {
            "string" => "string".to_string(),
            "number" | "integer" => "number".to_string(),
            "boolean" => "boolean".to_string(),
            "null" => "null".to_string(),
            "array" => array_type(schema.as_object().unwrap(), context),
            "object" => object_type(schema.as_object().unwrap(), context),
            _ => "unknown".to_string(),
        },
        Some(_) => "unknown".to_string(),
        None => {
            let obj = schema.as_object().unwrap();
            if obj.contains_key("properties")
                || obj.contains_key("additionalProperties")
                || obj.contains_key("required")
            {
                return object_type(obj, context);
            }
            if obj.contains_key("items") || obj.contains_key("prefixItems") {
                return array_type(obj, context);
            }
            "unknown".to_string()
        }
    }
}

fn array_type(schema: &Map<String, Value>, context: &mut SchemaContext<'_>) -> String {
    let items = schema.get("items");
    if let Some(items) = items {
        if !items.is_array() {
            return format!("Array<{}>", to_type(items, context));
        }
    }
    let tuple: Vec<&Value> = if let Some(Value::Array(prefix)) = schema.get("prefixItems") {
        prefix.iter().collect()
    } else if let Some(Value::Array(items)) = items {
        items.iter().collect()
    } else {
        Vec::new()
    };
    if !tuple.is_empty() {
        return format!(
            "[{}]",
            tuple
                .iter()
                .map(|item| to_type(item, context))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    "unknown[]".to_string()
}

fn description_of(property: &Value) -> String {
    match property.get("description") {
        Some(Value::String(text)) => text.trim().to_string(),
        _ => String::new(),
    }
}

fn object_type(schema: &Map<String, Value>, context: &mut SchemaContext<'_>) -> String {
    let properties: &Map<String, Value> = schema
        .get("properties")
        .and_then(|p| p.as_object())
        .unwrap_or_else(|| empty_map());
    let required: HashSet<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|items| items.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut names: Vec<&String> = properties.keys().collect();
    names.sort();
    let mut members: Vec<String> = names
        .iter()
        .map(|name| {
            let optional = if required.contains(name.as_str()) {
                ""
            } else {
                "?"
            };
            format!(
                "{}{}: {};",
                property_key(name),
                optional,
                to_type(&properties[name.as_str()], context)
            )
        })
        .collect();
    let additional = schema.get("additionalProperties");
    match additional {
        Some(additional) if !additional.is_boolean() || additional == &Value::Bool(true) => {
            let ty = if additional == &Value::Bool(true) {
                "unknown".to_string()
            } else {
                to_type(additional, context)
            };
            members.push(format!("[key: string]: {ty};"));
        }
        None if names.is_empty() => members.push("[key: string]: unknown;".to_string()),
        _ => {}
    }
    if members.is_empty() {
        return "{}".to_string();
    }
    let has_descriptions = names
        .iter()
        .any(|name| !description_of(&properties[name.as_str()]).is_empty());
    if !has_descriptions {
        return format!("{{ {} }}", members.join(" "));
    }

    let mut lines = vec!["{".to_string()];
    for (index, name) in names.iter().enumerate() {
        for line in split_crlf(&description_of(&properties[name.as_str()])) {
            let line = line.trim();
            if !line.is_empty() {
                lines.push(format!("{INDENT}// {line}"));
            }
        }
        lines.push(format!(
            "{INDENT}{}",
            members[index].replace('\n', &format!("\n{INDENT}"))
        ));
    }
    for member in members.iter().skip(names.len()) {
        lines.push(format!("{INDENT}{member}"));
    }
    lines.push("}".to_string());
    lines.join("\n")
}

fn empty_map() -> &'static Map<String, Value> {
    static EMPTY: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);
    &EMPTY
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mcp_result_schema(structured_content: Option<Value>) -> Value {
        let mut properties = serde_json::Map::new();
        properties.insert(
            "content".into(),
            json!({ "type": "array", "items": { "type": "object" } }),
        );
        if let Some(sc) = structured_content {
            properties.insert("structuredContent".into(), sc);
        }
        properties.insert("isError".into(), json!({ "type": "boolean" }));
        properties.insert("_meta".into(), json!({ "type": "object" }));
        json!({ "type": "object", "properties": properties, "required": ["content"] })
    }

    #[test]
    fn renders_primitives_literals_and_unions() {
        assert_eq!(schema_to_type(&json!({ "type": "string" })), "string");
        assert_eq!(schema_to_type(&json!({ "type": "integer" })), "number");
        assert_eq!(
            schema_to_type(&json!({ "type": ["string", "null"] })),
            "string | null"
        );
        assert_eq!(schema_to_type(&json!({ "const": "a" })), "\"a\"");
        assert_eq!(
            schema_to_type(&json!({ "enum": ["a", 1, null] })),
            "\"a\" | 1 | null"
        );
        assert_eq!(
            schema_to_type(&json!({ "anyOf": [{ "type": "string" }, { "type": "number" }] })),
            "string | number"
        );
        assert_eq!(
            schema_to_type(&json!({ "anyOf": [{ "type": "string" }, {}] })),
            "unknown"
        );
        assert_eq!(
            schema_to_type(
                &json!({ "allOf": [{ "anyOf": [{ "type": "string" }, { "type": "number" }] }, { "const": 1 }] })
            ),
            "(string | number) & 1"
        );
        assert_eq!(schema_to_type(&json!({ "$ref": "#/defs/x" })), "unknown");
        assert_eq!(schema_to_type(&Value::Bool(true)), "unknown");
        assert_eq!(schema_to_type(&Value::Bool(false)), "never");
    }

    #[test]
    fn renders_objects_on_one_line_with_sorted_properties() {
        assert_eq!(
            schema_to_type(&json!({
                "type": "object",
                "properties": { "city": { "type": "string" }, "max-lines": { "type": "number" } },
                "required": ["city"],
                "additionalProperties": false,
            })),
            "{ city: string; \"max-lines\"?: number; }"
        );
        assert_eq!(
            schema_to_type(
                &json!({ "type": "object", "additionalProperties": { "type": "number" } })
            ),
            "{ [key: string]: number; }"
        );
        assert_eq!(
            schema_to_type(&json!({ "type": "object" })),
            "{ [key: string]: unknown; }"
        );
        assert_eq!(
            schema_to_type(
                &json!({ "type": "object", "properties": {}, "additionalProperties": false })
            ),
            "{}"
        );
    }

    #[test]
    fn puts_property_descriptions_on_comment_lines() {
        assert_eq!(
            schema_to_type(&json!({
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "array",
                        "description": "look up weather for a given list of locations",
                        "items": { "type": "object", "properties": { "location": { "type": "string" } }, "required": ["location"] },
                    },
                },
                "required": ["weather"],
            })),
            "{\n  // look up weather for a given list of locations\n  weather: Array<{ location: string; }>;\n}"
        );
        assert_eq!(
            schema_to_type(&json!({
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "description": "Outer",
                        "properties": { "inner": { "type": "string", "description": "Inner" } },
                    },
                },
            })),
            "{\n  // Outer\n  outer?: {\n    // Inner\n    inner?: string;\n  };\n}"
        );
    }

    #[test]
    fn resolves_local_references_and_stops_at_recursive_ones() {
        let schema = json!({
            "type": "object",
            "properties": {
                "item": { "$ref": "#/$defs/Item" },
                "legacy": { "$ref": "#/definitions/Legacy" },
                "remote": { "$ref": "https://example.com/schema.json" },
            },
            "required": ["item"],
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": { "id": { "type": "string" }, "parent": { "$ref": "#/$defs/Item" } },
                    "required": ["id"],
                },
            },
            "definitions": { "Legacy": { "enum": ["a", "b"] } },
        });
        assert_eq!(
            schema_to_type(&schema),
            "{ item: { id: string; parent?: unknown; }; legacy?: \"a\" | \"b\"; remote?: unknown; }"
        );
    }

    #[test]
    fn renders_arrays_and_tuples() {
        assert_eq!(
            schema_to_type(&json!({ "type": "array", "items": { "type": "string" } })),
            "Array<string>"
        );
        assert_eq!(
            schema_to_type(
                &json!({ "type": "array", "prefixItems": [{ "type": "string" }, { "type": "number" }] })
            ),
            "[string, number]"
        );
        assert_eq!(schema_to_type(&json!({ "type": "array" })), "unknown[]");
    }

    #[test]
    fn renders_types_over_the_budget_as_unknown() {
        let mut properties = serde_json::Map::new();
        for i in 0..50 {
            properties.insert(format!("field{i}"), json!({ "type": "string" }));
        }
        let schema = json!({ "type": "object", "properties": properties });
        assert_eq!(schema_to_type_with(&schema, Some(100)), "unknown");
        assert!(schema_to_type(&schema).contains("field49?: string;"));
    }

    #[test]
    fn renders_signatures_with_normalized_identifiers() {
        assert_eq!(
            render_tool_signature(&SignatureTool {
                name: "hidden-dynamic-tool",
                input_schema: Some(&json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                    "additionalProperties": false,
                })),
                output_schema: Some(&json!({
                    "type": "object",
                    "properties": { "ok": { "type": "boolean" } },
                    "required": ["ok"],
                })),
                input_max_chars: None,
            }),
            "hidden_dynamic_tool(args: { city: string; }): Promise<{ ok: boolean; }>;"
        );
        assert_eq!(
            render_tool_signature(&SignatureTool {
                name: "free",
                input_schema: None,
                output_schema: None,
                input_max_chars: None
            }),
            "free(args: unknown): Promise<unknown>;"
        );
    }

    #[test]
    fn renders_mcp_call_tool_result_output_schemas() {
        let input_schema =
            json!({ "type": "object", "properties": {}, "additionalProperties": false });
        assert_eq!(
            render_tool_signature(&SignatureTool {
                name: "mcp__sample__search",
                input_schema: Some(&input_schema),
                output_schema: Some(&mcp_result_schema(Some(json!({
                    "type": "object",
                    "properties": { "results": { "type": "array", "items": { "$ref": "#/definitions/Result~1item~0v1" } } },
                    "required": ["results"],
                    "additionalProperties": false,
                    "definitions": {
                        "Result/item~v1": {
                            "type": "object",
                            "properties": { "id": { "type": "string" }, "score": { "type": "number" } },
                            "required": ["id", "score"],
                            "additionalProperties": false,
                        },
                    },
                })))),
                input_max_chars: None,
            }),
            "mcp__sample__search(args: {}): Promise<CallToolResult<{ results: Array<{ id: string; score: number; }>; }>>;"
        );
        assert_eq!(
            render_tool_signature(&SignatureTool {
                name: "plain",
                input_schema: Some(&input_schema),
                output_schema: Some(&mcp_result_schema(None)),
                input_max_chars: None,
            }),
            "plain(args: {}): Promise<CallToolResult>;"
        );
        assert_eq!(
            mcp_structured_content_schema(Some(
                &json!({ "type": "object", "properties": { "content": { "type": "array" } } })
            )),
            None
        );
    }

    #[test]
    fn renders_the_per_tool_sample() {
        assert_eq!(
            render_tool_sample(&SampleTool {
                name: "foo",
                description: Some("bar"),
                input_schema: Some(&json!({ "type": "string" })),
                output_schema: None,
                input_max_chars: None,
            }),
            "bar\n\ncodemode tool declaration:\n```ts\ndeclare const tools: { foo(args: string): Promise<unknown>; };\n```"
        );
    }

    #[test]
    fn renders_tools_and_globals() {
        let read = crate::types::CodemodeTool::new("read", |_, _| async { Ok(None) })
            .with_description("Read a file.\nSecond line.")
            .with_input_schema(json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }))
            .with_output_schema(json!({ "type": "string" }));
        let remote = crate::types::CodemodeTool::new("remote-api", |_, _| async { Ok(None) });
        let attach = crate::types::CodemodeTool::new("attach", |_, _| async { Ok(None) })
            .with_description("Attach it.")
            .with_input_schema(json!({ "type": "string" }));
        let text = render_declarations(RenderDeclarationsOptions {
            tools: &[read, remote],
            globals: &[attach],
        });
        assert_eq!(
            text,
            [
                "declare const tools: {",
                "  /**",
                "   * Read a file.",
                "   * Second line.",
                "   */",
                "  read(args: { path: string; }): Promise<string>;",
                "  remote_api(args: unknown): Promise<unknown>;",
                "};",
                "",
                "/** Attach it. */",
                "declare function attach(args: string): Promise<unknown>;",
            ]
            .join("\n")
        );
    }

    #[test]
    fn renders_namespaced_globals_and_explicit_signatures() {
        let list = crate::types::CodemodeTool::new("models.list", |_, _| async { Ok(None) })
            .with_description("List models.")
            .with_signature("(type: string): Promise<string[]>");
        let get = crate::types::CodemodeTool::new("models.get", |_, _| async { Ok(None) })
            .with_input_schema(json!({ "type": "string" }));
        let plain = crate::types::CodemodeTool::new("plain", |_, _| async { Ok(None) })
            .with_signature("(): void");
        let text = render_declarations(RenderDeclarationsOptions {
            tools: &[],
            globals: &[list, get, plain],
        });
        assert_eq!(
            text,
            [
                "declare function plain(): void;",
                "",
                "declare const models: {",
                "  /** List models. */",
                "  list(type: string): Promise<string[]>;",
                "  get(args: string): Promise<unknown>;",
                "};",
            ]
            .join("\n")
        );
    }

    #[test]
    fn escapes_comment_terminators_in_descriptions() {
        let tool = crate::types::CodemodeTool::new("x", |_, _| async { Ok(None) })
            .with_description("a */ b");
        let text = render_declarations(RenderDeclarationsOptions {
            tools: &[tool],
            globals: &[],
        });
        assert!(text.contains("/** a *\\/ b */"));
    }
}
