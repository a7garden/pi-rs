//! Mirrors upstream `src/identifier.ts`.

/// The identifier a script uses for a tool: characters that are not valid in a JavaScript
/// identifier become `_`. `mcp__docs__search` stays as is, `my-tool` becomes `my_tool`.
pub fn to_codemode_identifier(name: &str) -> String {
    let mut identifier = String::new();
    for ch in name.chars() {
        let valid = if identifier.is_empty() {
            ch.is_ascii_alphabetic() || ch == '_' || ch == '$'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
        };
        identifier.push(if valid { ch } else { '_' });
    }
    if identifier.is_empty() {
        "_".to_string()
    } else {
        identifier
    }
}

#[cfg(test)]
mod tests {
    use super::to_codemode_identifier;

    #[test]
    fn normalizes_invalid_characters() {
        assert_eq!(
            to_codemode_identifier("mcp__docs__search"),
            "mcp__docs__search"
        );
        assert_eq!(to_codemode_identifier("my-tool"), "my_tool");
        assert_eq!(to_codemode_identifier(""), "_");
        assert_eq!(to_codemode_identifier("9lives"), "_lives");
        assert_eq!(to_codemode_identifier("$ok"), "$ok");
    }
}
