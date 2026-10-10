//! Port of `packages/durable/src/ids.ts`.
//!
//! Upstream brands plain numbers via a type-level symbol; Rust uses distinct
//! newtypes over `u64` (see [`crate::types`]). The trusted-boundary conversion
//! helpers map to [`DurableId::from_number`] / [`Seq::from_number`]; the
//! functions here exist to mirror the upstream surface.
#![allow(dead_code)]

use crate::types::{DurableId, Seq};

/// Apply an erased ID brand at a trusted numeric allocation or decoding boundary.
pub fn id_from_number<I: DurableId>(value: u64) -> I {
    I::from_number(value)
}

/// Apply the erased commit-sequence brand at a trusted storage boundary.
pub fn seq_from_number(value: u64) -> Seq {
    Seq::from_number(value)
}

#[cfg(test)]
mod tests {
    use crate::types::{ConversationId, TaskId};

    use super::{id_from_number, seq_from_number};

    #[test]
    fn durable_ids_keep_nominal_types_and_serialize_as_numbers() {
        let conversation = id_from_number::<ConversationId>(1);
        let task = id_from_number::<TaskId>(4);
        assert_eq!(
            serde_json::to_value(conversation).unwrap(),
            serde_json::json!(1)
        );
        assert_eq!(serde_json::to_string(&task).unwrap(), "4");
        assert_eq!(u64::from(seq_from_number(1)), 1);
    }
}
