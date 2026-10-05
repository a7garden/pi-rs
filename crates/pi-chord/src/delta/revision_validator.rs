//! Port of `packages/chord/src/delta/revision-validator.ts`.
//!
//! Upstream validates that a replica revision is strict JSON before it enters
//! replicated state, caching validated containers in a `WeakSet` so later
//! revisions skip already-checked subtrees. The checks, and why each one is
//! structurally impossible for [`JsonValue`]:
//!
//! - **Cycles** (`Replicated state cannot contain cycles`): [`JsonValue`] trees
//!   are owned; a value cannot contain itself.
//! - **Non-plain objects** (`Replicated state containers must be plain objects
//!   or arrays`): `Value` has exactly `Null`, `Bool`, `Number`, `String`,
//!   `Array`, `Object` — there are no prototypes, exotic objects, maps, or
//!   class instances.
//! - **Sparse arrays / non-index entries** (`Replicated state arrays must be
//!   dense ...`): `Value::Array` is a dense `Vec`; holes and extra properties
//!   cannot exist.
//! - **Symbol properties / non-enumerable or accessor descriptors**
//!   (`Replicated state objects must contain enumerable data properties`):
//!   object members are `String` keys mapping to `Value`s; descriptors,
//!   symbols, and getters do not exist.
//! - **Non-finite numbers** (`Replicated state values must be strict JSON`):
//!   `serde_json::Number` cannot represent NaN or ±∞ (`Number::from_f64`
//!   rejects them and JSON grammar cannot carry them).
//!
//! The type is therefore stateful in signature only: `validate` returns its
//! input unchanged, and no validation cache is needed. Keeping the struct and
//! method keeps every upstream call site identical.

use crate::json::JsonValue;

/// Validates immutable replica revisions while skipping containers validated in
/// earlier revisions. In Rust, every check is upheld by construction — see the
/// module docs — so validation is the identity function.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JsonRevisionValidator {}

impl JsonRevisionValidator {
    pub fn new() -> JsonRevisionValidator {
        JsonRevisionValidator {}
    }

    /// Returns `value` unchanged. Upstream performs a full strict-JSON walk
    /// (cycle, prototype, density, descriptor and finiteness checks) and skips
    /// subtrees already validated; [`JsonValue`] upholds all of those
    /// invariants by construction, so there is nothing to check and nothing to
    /// cache.
    pub fn validate(&mut self, value: JsonValue) -> JsonValue {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn returns_strict_json_values_unchanged() {
        let mut validator = JsonRevisionValidator::new();
        let value = json!({"text": "a", "values": [1, 2.5, null, true, {"nested": []}]});
        assert_eq!(validator.validate(value.clone()), value);
        assert_eq!(validator.validate(json!(null)), json!(null));
        assert_eq!(validator.validate(json!("scalar")), json!("scalar"));
    }

    #[test]
    fn accepts_repeated_validation_of_the_same_revision() {
        let mut validator = JsonRevisionValidator::new();
        let value = json!({"rows": [1, 2, 3]});
        assert_eq!(validator.validate(value.clone()), value);
        assert_eq!(validator.validate(value.clone()), value);
    }
}
