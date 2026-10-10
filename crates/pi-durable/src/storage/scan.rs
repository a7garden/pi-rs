//! Port of `packages/durable/src/storage/scan.ts` — shared scan-order plumbing
//! for the built-in storage backends.
//!
//! Deviations (module ledger):
//! - D1: upstream `scanStart` rejects a malformed requested order at runtime
//!   (`Invalid scan order: …`); Rust `ScanOrder` is a closed serde enum, so an
//!   invalid query order fails query deserialization before reaching this
//!   helper.
//! - D2: upstream throws `TypeError` on invalid cursors and cursor/order
//!   mismatches; the shared [`Storage`](crate::types::Storage) read surface is
//!   infallible, so [`scan_start`] panics with the upstream message (same
//!   convention as `cursor_id` in `memory.rs`).

use serde_json::{json, Map, Value};

use crate::types::{Cursor, ScanOrder};

/// Where a built-in storage scan starts: its order and the last ID a previous page returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScanStart {
    /// Order the scan continues in.
    pub order: ScanOrder,
    /// Last ID a previous page returned, when continuing.
    pub after: Option<u64>,
}

/// Resolve a scan's order and position. A cursor continues in the order it was created with, whether the query repeats
/// that order or omits it; a different `order` panics. A cursor without an order, written before scans had one,
/// continues in the scan's default order.
pub(super) fn scan_start(
    requested: Option<ScanOrder>,
    cursor: Option<&Cursor>,
    fallback: ScanOrder,
) -> ScanStart {
    let Some(cursor) = cursor else {
        return ScanStart {
            order: requested.unwrap_or(fallback),
            after: None,
        };
    };
    let after = match cursor.get("after") {
        Some(after) => after
            .as_u64()
            .unwrap_or_else(|| panic!("Invalid storage cursor")),
        None => panic!("Invalid storage cursor"),
    };
    let stored = match cursor.get("order") {
        None => None,
        Some(Value::String(order)) if order == "ascending" => Some(ScanOrder::Ascending),
        Some(Value::String(order)) if order == "descending" => Some(ScanOrder::Descending),
        Some(_) => panic!("Invalid storage cursor"),
    };
    let order = stored.unwrap_or(fallback);
    if let Some(requested) = requested {
        assert!(
            requested == order,
            "The cursor continues a {} scan; the query asks for {}",
            order_name(order),
            order_name(requested)
        );
    }
    ScanStart {
        order,
        after: Some(after),
    }
}

/// The continuation of a scan whose last returned item has `id`.
pub(super) fn next_cursor(id: u64, order: ScanOrder) -> Cursor {
    let mut cursor = Map::new();
    cursor.insert("after".into(), json!(id));
    cursor.insert("order".into(), json!(order));
    cursor
}

/// The serde spelling of an order, for upstream error messages.
pub(super) fn order_name(order: ScanOrder) -> &'static str {
    match order {
        ScanOrder::Ascending => "ascending",
        ScanOrder::Descending => "descending",
    }
}
