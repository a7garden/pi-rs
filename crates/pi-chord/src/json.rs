//! Port of `packages/chord/src/json.ts`.
//!
//! Upstream validates arbitrary JS values and copies them into alias-free strict-JSON
//! trees (no `undefined`, no cycles, finite numbers, plain objects, dense arrays).
//! In Rust the value type is [`serde_json::Value`], which upholds those invariants by
//! construction: numbers are finite (`Number::from_f64` rejects NaN/±∞), trees are
//! acyclic (ownership), object keys are strings, arrays are dense.
//! Upstream's `omitUndefinedProperties` option has no Rust counterpart (`Value` has no
//! `undefined` variant); upstream `json.test.ts` cases exercising descriptor/symbol/
//! prototype edge cases are not constructible in Rust and are intentionally not ported.
//! This deviation is recorded in PORT_MAP.md.

/// Canonical strict-JSON tree shared across the workspace. Wire-format compatible with
/// upstream's `JsonValue` union.
pub type JsonValue = serde_json::Value;

/// Copy a value into an alias-free strict-JSON tree owned by the caller.
///
/// Mirrors upstream `copyJson(value)` without options (the `omitUndefinedProperties`
/// option is meaningless for [`JsonValue`]).
pub fn copy_json(value: &JsonValue) -> JsonValue {
    value.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn copies_nested_tree() {
        let v = json!({"a": [1, "x", true, null], "b": {"c": 2.5}});
        let c = copy_json(&v);
        assert_eq!(v, c);
    }

    #[test]
    fn copy_is_owned() {
        let v = json!({"a": [1]});
        let mut c = copy_json(&v);
        c["a"][0] = json!(2);
        assert_eq!(v["a"][0], json!(1));
    }
}
