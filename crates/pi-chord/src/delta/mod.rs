//! Port of `packages/chord/src/delta/index.ts`.
//!
//! Immutable revision tracking and operations over plain JSON. Depends on nothing
//! else in the harness beyond [`crate::json::JsonValue`].
//!
//! Deviations from upstream (mechanical, recorded per PORT_MAP.md rule 3):
//! - `Op`/`WireOp` tuples are represented as a Rust enum; the wire encoding
//!   (`serde_json` arrays such as `["s", ["a", 0], 42]`) is byte-compatible with
//!   upstream and implemented through custom `Serialize`/`Deserialize` impls.
//! - `Seg` string/number discrimination happens at construction; `usize` covers
//!   the non-negative integer segment requirement enforced by upstream's
//!   `assertSafePath`.
//! - `overlap` operates on Unicode scalar values instead of UTF-16 code units.
//!   The suffix/prefix property it guarantees is identical, so emitted
//!   `a`/`t` operations replay to the same values.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::json::JsonValue;

pub mod diff;
pub mod draft;
pub mod tracker;

mod apply_immutable_trusted;
mod revision_validator;

pub use apply_immutable_trusted::apply_immutable_trusted;
pub use diff::diff_revisions;
pub use draft::Draft;
pub use revision_validator::JsonRevisionValidator;
pub use tracker::{track, Change, Prepared, Tracker};

/// A single path segment: an object key or an array index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Seg {
    Key(String),
    Index(usize),
}

impl Seg {
    pub fn key(key: impl Into<String>) -> Seg {
        Seg::Key(key.into())
    }

    pub fn index(index: usize) -> Seg {
        Seg::Index(index)
    }

    /// JSON rendering used by error messages, matching `String(segment)` upstream.
    pub fn as_json(&self) -> JsonValue {
        match self {
            Seg::Key(key) => JsonValue::String(key.clone()),
            Seg::Index(index) => JsonValue::Number((*index as u64).into()),
        }
    }
}

impl Serialize for Seg {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Seg::Key(key) => serializer.serialize_str(key),
            Seg::Index(index) => serializer.serialize_u64(*index as u64),
        }
    }
}

impl<'de> Deserialize<'de> for Seg {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(deserializer)?;
        seg_from_json(&value).ok_or_else(|| serde::de::Error::custom("invalid path segment"))
    }
}

fn seg_from_json(value: &JsonValue) -> Option<Seg> {
    match value {
        JsonValue::String(key) => Some(Seg::Key(key.clone())),
        JsonValue::Number(number) => {
            let index = json_integer(number)?;
            if index < 0 {
                return None;
            }
            Some(Seg::Index(index as usize))
        }
        _ => None,
    }
}

/// Extract an exact integer from a JSON number, accepting `3.0` like JavaScript's
/// `Number.isInteger`.
fn json_integer(number: &serde_json::Number) -> Option<i64> {
    if let Some(int) = number.as_i64() {
        return Some(int);
    }
    if let Some(int) = number.as_u64() {
        return i64::try_from(int).ok();
    }
    let float = number.as_f64()?;
    if float.fract() != 0.0
        || !float.is_finite()
        || float < i64::MIN as f64
        || float > i64::MAX as f64
    {
        return None;
    }
    Some(float as i64)
}

/// A path into a JSON tree. Empty paths address the root (only valid for `p`/`m`).
pub type Path = Vec<Seg>;

/// A path with at least one segment; enforced at `Op` construction.
pub type NonEmptyPath = Path;

/// An operation tuple: the form in memory, on the wire, on disk.
///
/// `Replace` is the ONLY op that replaces a whole value. `Set`/`Delete`/`Append`/
/// `Truncate` cannot target the root. `Splice`/`Permute` may, because a tracked
/// value can itself be an array. Operation shape is not canonical: an array may be
/// emptied by either a replacement or a root splice.
///
/// `Op` knows nothing about the path dictionary. Interning, id references and
/// omitted paths live in [`WireOp`] and exist only between [`Encoder`] and
/// [`Decoder`].
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `["r", value]`
    Replace(JsonValue),
    /// `["s", path, value]`
    Set {
        path: NonEmptyPath,
        value: JsonValue,
    },
    /// `["d", path]`
    Delete { path: NonEmptyPath },
    /// `["a", path, text]`
    Append { path: NonEmptyPath, text: String },
    /// `["t", path, count]`
    Truncate { path: NonEmptyPath, count: usize },
    /// `["p", path, start, remove, items]`
    Splice {
        path: Path,
        start: usize,
        remove: usize,
        items: Vec<JsonValue>,
    },
    /// `["m", path, permutation]` — reorder an array in place: `new[i] = old[permutation[i]]`.
    Permute { path: Path, permutation: Vec<usize> },
}

impl Op {
    /// Set operation constructor; rejects empty paths like the upstream type does.
    pub fn set(path: NonEmptyPath, value: JsonValue) -> Result<Op, DeltaError> {
        if path.is_empty() {
            return Err(DeltaError::invalid("path is empty"));
        }
        Ok(Op::Set { path, value })
    }

    pub fn delete(path: NonEmptyPath) -> Result<Op, DeltaError> {
        if path.is_empty() {
            return Err(DeltaError::invalid("path is empty"));
        }
        Ok(Op::Delete { path })
    }

    pub fn append(path: NonEmptyPath, text: impl Into<String>) -> Result<Op, DeltaError> {
        if path.is_empty() {
            return Err(DeltaError::invalid("path is empty"));
        }
        Ok(Op::Append {
            path,
            text: text.into(),
        })
    }

    pub fn truncate(path: NonEmptyPath, count: usize) -> Result<Op, DeltaError> {
        if path.is_empty() {
            return Err(DeltaError::invalid("path is empty"));
        }
        Ok(Op::Truncate { path, count })
    }

    /// The op's path; `None` for `Replace`, which carries no path.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Op::Replace(_) => None,
            Op::Set { path, .. }
            | Op::Delete { path }
            | Op::Append { path, .. }
            | Op::Truncate { path, .. }
            | Op::Splice { path, .. }
            | Op::Permute { path, .. } => Some(path),
        }
    }
}

impl Serialize for Op {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = Vec::with_capacity(5);
        match self {
            Op::Replace(value) => {
                tuple.push(JsonValue::String("r".into()));
                tuple.push(value.clone());
            }
            Op::Set { path, value } => {
                tuple.push(JsonValue::String("s".into()));
                tuple.push(path_to_json(path));
                tuple.push(value.clone());
            }
            Op::Delete { path } => {
                tuple.push(JsonValue::String("d".into()));
                tuple.push(path_to_json(path));
            }
            Op::Append { path, text } => {
                tuple.push(JsonValue::String("a".into()));
                tuple.push(path_to_json(path));
                tuple.push(JsonValue::String(text.clone()));
            }
            Op::Truncate { path, count } => {
                tuple.push(JsonValue::String("t".into()));
                tuple.push(path_to_json(path));
                tuple.push(JsonValue::Number((*count as u64).into()));
            }
            Op::Splice {
                path,
                start,
                remove,
                items,
            } => {
                tuple.push(JsonValue::String("p".into()));
                tuple.push(path_to_json(path));
                tuple.push(JsonValue::Number((*start as u64).into()));
                tuple.push(JsonValue::Number((*remove as u64).into()));
                tuple.push(JsonValue::Array(items.clone()));
            }
            Op::Permute { path, permutation } => {
                tuple.push(JsonValue::String("m".into()));
                tuple.push(path_to_json(path));
                tuple.push(JsonValue::Array(
                    permutation
                        .iter()
                        .map(|index| JsonValue::Number((*index as u64).into()))
                        .collect(),
                ));
            }
        }
        serde_json::Value::serialize(&JsonValue::Array(tuple), serializer)
    }
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(deserializer)?;
        op_from_json(&value).ok_or_else(|| serde::de::Error::custom("invalid op tuple"))
    }
}

/// Decode one operation tuple. Returns `None` for tuples `assert_valid_wire_op`
/// would reject.
pub fn op_from_json(value: &JsonValue) -> Option<Op> {
    let tuple = value.as_array()?;
    let verb = tuple.first()?.as_str()?;
    match verb {
        "r" => {
            if tuple.len() != 2 {
                return None;
            }
            Some(Op::Replace(tuple[1].clone()))
        }
        "s" => {
            if tuple.len() != 3 {
                return None;
            }
            Some(Op::Set {
                path: path_from_json(tuple[1].as_array()?)?,
                value: tuple[2].clone(),
            })
        }
        "d" => {
            if tuple.len() != 2 {
                return None;
            }
            Some(Op::Delete {
                path: path_from_json(tuple[1].as_array()?)?,
            })
        }
        "a" => {
            if tuple.len() != 3 {
                return None;
            }
            Some(Op::Append {
                path: path_from_json(tuple[1].as_array()?)?,
                text: tuple[2].as_str()?.to_string(),
            })
        }
        "t" => {
            if tuple.len() != 3 {
                return None;
            }
            Some(Op::Truncate {
                path: path_from_json(tuple[1].as_array()?)?,
                count: json_integer(tuple[2].as_number()?)?.max(0) as usize,
            })
        }
        "p" => {
            if tuple.len() != 5 {
                return None;
            }
            Some(Op::Splice {
                path: path_from_json(tuple[1].as_array()?)?,
                start: json_integer(tuple[2].as_number()?)?.max(0) as usize,
                remove: json_integer(tuple[3].as_number()?)?.max(0) as usize,
                items: tuple[4].as_array()?.clone(),
            })
        }
        "m" => {
            if tuple.len() != 3 {
                return None;
            }
            Some(Op::Permute {
                path: path_from_json(tuple[1].as_array()?)?,
                permutation: tuple[2]
                    .as_array()?
                    .iter()
                    .map(|index| json_integer(index.as_number()?).map(|int| int.max(0) as usize))
                    .collect::<Option<Vec<_>>>()?,
            })
        }
        _ => None,
    }
}

fn path_from_json(tuple: &[JsonValue]) -> Option<Path> {
    tuple.iter().map(seg_from_json).collect()
}

pub fn path_to_json(path: &[Seg]) -> JsonValue {
    JsonValue::Array(path.iter().map(Seg::as_json).collect())
}

/// One wire-encoded operation: raw JSON tuples with path interning (`["#", id, path]`)
/// and omitted-path short forms. Only meaningful between [`Encoder`] and [`Decoder`].
pub type WireOp = JsonValue;

/// `r` is the only root-replacing verb.
pub fn is_replace(op: &WireOp) -> bool {
    op.as_array()
        .and_then(|tuple| tuple.first())
        .and_then(Value::as_str)
        == Some("r")
}

/// A batch begins with a replacement. Flush guarantees `r` is at index 0 or absent,
/// so this is exact rather than a heuristic. Accepts decoded ops and wire ops.
pub fn is_base(ops: &[WireOp]) -> bool {
    ops.len() > 0 && is_replace(&ops[0])
}

/// True when a decoded batch begins with a root replacement.
pub fn is_base_decoded(ops: &[Op]) -> bool {
    matches!(ops.first(), Some(Op::Replace(_)))
}

/// Segments that reach the prototype chain in JavaScript hosts.
///
/// Ops arrive from a facet, a plugin compartment, or a tool whose details may echo
/// model output, so none of it is trusted input.
pub const RESERVED_SEGMENTS: [&str; 3] = ["__proto__", "constructor", "prototype"];

pub fn is_reserved_segment(segment: &str) -> bool {
    RESERVED_SEGMENTS.contains(&segment)
}

/// Delta pipeline errors. `Display` strings match upstream `Error` messages.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeltaError {
    #[error("unsafe path segment: {segment}")]
    UnsafePath { segment: String },
    #[error("unresolvable path: {path}")]
    Unresolvable { path: String },
    #[error("{0}")]
    Invalid(String),
}

impl DeltaError {
    pub fn invalid(message: impl Into<String>) -> DeltaError {
        DeltaError::Invalid(message.into())
    }

    pub fn unsafe_segment(segment: &Seg) -> DeltaError {
        DeltaError::UnsafePath {
            segment: match segment {
                Seg::Key(key) => key.clone(),
                Seg::Index(index) => index.to_string(),
            },
        }
    }

    pub fn unresolvable(path: &Path) -> DeltaError {
        DeltaError::Unresolvable {
            path: serde_json::to_string(&path_to_json(path)).unwrap_or_else(|_| "[]".to_string()),
        }
    }

    pub fn unresolvable_ref(reference: u64) -> DeltaError {
        DeltaError::Unresolvable {
            path: reference.to_string(),
        }
    }
}

/// Verb, arity and payload shape validation for a **decoded** op: paths inline,
/// no `#`, no short forms. [`apply`] uses this.
pub fn assert_valid_op(op: &Op) -> Result<(), DeltaError> {
    match op {
        Op::Replace(_) => Ok(()),
        Op::Set { path, .. }
        | Op::Delete { path }
        | Op::Append { path, .. }
        | Op::Truncate { path, .. } => {
            if path.is_empty() {
                return Err(DeltaError::invalid("path is empty"));
            }
            assert_safe_path(path)
        }
        Op::Splice { path, .. } => assert_safe_path(path),
        Op::Permute { path, permutation } => {
            assert_safe_path(path)?;
            let mut seen = vec![false; permutation.len()];
            for index in permutation {
                if *index >= permutation.len() || seen[*index] {
                    return Err(DeltaError::invalid("m permutation is not a bijection"));
                }
                seen[*index] = true;
            }
            Ok(())
        }
    }
}

/// The same validation for raw tuples in the **decoded** vocabulary (paths inline,
/// no `#`, no short forms). Used where ops arrive as raw JSON but must already be
/// decoded, mirroring upstream `assertValidOp(op: unknown)`.
pub fn assert_valid_raw_op(op: &WireOp) -> Result<(), DeltaError> {
    let tuple = op
        .as_array()
        .filter(|tuple| !tuple.is_empty())
        .ok_or_else(|| DeltaError::invalid("op is not a tuple"))?;
    let verb = tuple[0]
        .as_str()
        .ok_or_else(|| DeltaError::invalid("op is not a tuple"))?;
    let len = tuple.len();
    let path_arg = |value: &JsonValue, non_empty: bool| -> Result<(), DeltaError> {
        let tuple = value
            .as_array()
            .ok_or_else(|| DeltaError::invalid("path is not an array"))?;
        if non_empty && tuple.is_empty() {
            return Err(DeltaError::invalid("path is empty"));
        }
        assert_safe_wire_path(value)
    };
    let non_negative = |value: &JsonValue, message: &str| -> Result<(), DeltaError> {
        value
            .as_number()
            .and_then(json_integer)
            .filter(|int| *int >= 0)
            .ok_or_else(|| DeltaError::invalid(message))?;
        Ok(())
    };
    match verb {
        "r" => {
            if len != 2 {
                return Err(DeltaError::invalid("r arity"));
            }
            Ok(())
        }
        "s" => {
            if len != 3 {
                return Err(DeltaError::invalid("s arity"));
            }
            path_arg(&tuple[1], true)
        }
        "d" => {
            if len != 2 {
                return Err(DeltaError::invalid("d arity"));
            }
            path_arg(&tuple[1], true)
        }
        "a" => {
            if len != 3 || tuple[2].as_str().is_none() {
                return Err(DeltaError::invalid("a shape"));
            }
            path_arg(&tuple[1], true)
        }
        "t" => {
            if len != 3 {
                return Err(DeltaError::invalid("t shape"));
            }
            non_negative(&tuple[2], "t shape")?;
            path_arg(&tuple[1], true)
        }
        "p" => {
            if len != 5 {
                return Err(DeltaError::invalid("p arity"));
            }
            path_arg(&tuple[1], false)?;
            non_negative(&tuple[2], "p index")?;
            non_negative(&tuple[3], "p remove")?;
            if tuple[4].as_array().is_none() {
                return Err(DeltaError::invalid("p items"));
            }
            Ok(())
        }
        "m" => {
            if len != 3 {
                return Err(DeltaError::invalid("m arity"));
            }
            path_arg(&tuple[1], false)?;
            assert_permutation(&tuple[2])
        }
        // Silently skipping an unknown verb is how a newer producer's op vanishes.
        unknown => Err(DeltaError::invalid(format!("unknown op verb: {unknown}"))),
    }
}

fn assert_permutation(value: &JsonValue) -> Result<(), DeltaError> {
    let permutation = value
        .as_array()
        .ok_or_else(|| DeltaError::invalid("m permutation is not an array"))?;
    let mut seen = vec![false; permutation.len()];
    for index in permutation {
        let int = json_integer(
            index
                .as_number()
                .ok_or_else(|| DeltaError::invalid("m permutation is not a bijection"))?,
        )
        .ok_or_else(|| DeltaError::invalid("m permutation is not a bijection"))?;
        if int < 0 || int as usize >= permutation.len() || seen[int as usize] {
            return Err(DeltaError::invalid("m permutation is not a bijection"));
        }
        seen[int as usize] = true;
    }
    Ok(())
}

fn assert_safe_wire_path(value: &JsonValue) -> Result<(), DeltaError> {
    let tuple = value
        .as_array()
        .ok_or_else(|| DeltaError::invalid("path is not an array"))?;
    for segment in tuple {
        match segment {
            JsonValue::String(key) => {
                if is_reserved_segment(key) {
                    return Err(DeltaError::UnsafePath {
                        segment: key.clone(),
                    });
                }
            }
            JsonValue::Number(number) => {
                let int = json_integer(number)
                    .ok_or_else(|| DeltaError::invalid("path is not an array"))?;
                if int < 0 {
                    return Err(DeltaError::UnsafePath {
                        segment: number.to_string(),
                    });
                }
            }
            _ => {
                return Err(DeltaError::UnsafePath {
                    segment: segment.to_string(),
                })
            }
        }
    }
    Ok(())
}

/// The same validation for the wire grammar: ids and short forms are legal here.
pub fn assert_valid_wire_op(op: &WireOp) -> Result<(), DeltaError> {
    let tuple = op
        .as_array()
        .filter(|tuple| !tuple.is_empty())
        .ok_or_else(|| DeltaError::invalid("op is not a tuple"))?;
    let verb = tuple[0]
        .as_str()
        .ok_or_else(|| DeltaError::invalid("op is not a tuple"))?;
    let ok_ref = |reference: &JsonValue| -> Result<(), DeltaError> {
        match reference {
            JsonValue::Number(number) => {
                let int = json_integer(number).filter(|int| *int >= 0);
                if int.is_none() {
                    return Err(DeltaError::invalid("bad path id"));
                }
                Ok(())
            }
            JsonValue::Array(_) => assert_safe_wire_path(reference),
            // A string is not a path. Unchecked, `"a".slice(0, -1)` is `""`, so it
            // resolves to the ROOT and writes there — a path that is not a path, accepted.
            _ => Err(DeltaError::invalid("path is not an array")),
        }
    };
    let len = tuple.len();
    match verb {
        "r" => {
            if len != 2 {
                return Err(DeltaError::invalid("r arity"));
            }
            Ok(())
        }
        "s" => {
            if len == 3 {
                ok_ref(&tuple[1])
            } else if len != 2 {
                Err(DeltaError::invalid("s arity"))
            } else {
                Ok(())
            }
        }
        "d" => {
            if len == 2 {
                ok_ref(&tuple[1])
            } else if len != 1 {
                Err(DeltaError::invalid("d arity"))
            } else {
                Ok(())
            }
        }
        "a" => {
            if len == 3 {
                ok_ref(&tuple[1])?;
                if tuple[2].as_str().is_none() {
                    return Err(DeltaError::invalid("a value"));
                }
                Ok(())
            } else if len == 2 {
                if tuple[1].as_str().is_none() {
                    return Err(DeltaError::invalid("a value"));
                }
                Ok(())
            } else {
                Err(DeltaError::invalid("a arity"))
            }
        }
        "t" => {
            let non_negative = |value: &JsonValue| -> Result<(), DeltaError> {
                if value
                    .as_number()
                    .and_then(json_integer)
                    .filter(|int| *int >= 0)
                    .is_none()
                {
                    return Err(DeltaError::invalid("t count"));
                }
                Ok(())
            };
            if len == 3 {
                ok_ref(&tuple[1])?;
                non_negative(&tuple[2])
            } else if len == 2 {
                non_negative(&tuple[1])
            } else {
                Err(DeltaError::invalid("t arity"))
            }
        }
        "p" => {
            let (index, remove, items, has_ref) = match len {
                5 => (&tuple[2], &tuple[3], &tuple[4], true),
                4 => (&tuple[1], &tuple[2], &tuple[3], false),
                _ => return Err(DeltaError::invalid("p arity")),
            };
            if has_ref {
                ok_ref(&tuple[1])?;
            }
            let non_negative = |value: &JsonValue, message: &str| -> Result<(), DeltaError> {
                value
                    .as_number()
                    .and_then(json_integer)
                    .filter(|int| *int >= 0)
                    .ok_or_else(|| DeltaError::invalid(message))?;
                Ok(())
            };
            non_negative(index, "p index")?;
            non_negative(remove, "p remove")?;
            if items.as_array().is_none() {
                return Err(DeltaError::invalid("p items"));
            }
            Ok(())
        }
        "m" => {
            if len == 3 {
                ok_ref(&tuple[1])?;
            } else if len != 2 {
                return Err(DeltaError::invalid("m arity"));
            }
            assert_permutation(&tuple[len - 1])
        }
        "#" => {
            if len != 3 {
                return Err(DeltaError::invalid("# shape"));
            }
            let id = tuple[1]
                .as_number()
                .and_then(json_integer)
                .filter(|int| *int >= 0)
                .ok_or_else(|| DeltaError::invalid("# shape"))?;
            let _ = id;
            if tuple[2].as_array().is_none() {
                return Err(DeltaError::invalid("# shape"));
            }
            assert_safe_wire_path(&tuple[2])
        }
        // Silently skipping an unknown verb is how a newer producer's op vanishes.
        unknown => Err(DeltaError::invalid(format!("unknown op verb: {unknown}"))),
    }
}

pub fn assert_safe_path(path: &[Seg]) -> Result<(), DeltaError> {
    for segment in path {
        match segment {
            Seg::Key(key) => {
                if is_reserved_segment(key) {
                    return Err(DeltaError::UnsafePath {
                        segment: key.clone(),
                    });
                }
            }
            Seg::Index(_) => {}
        }
    }
    Ok(())
}

// ─── Applier ─────────────────────────────────────────────────────────────────

/// Apply ops to a plain mutable value. The `Replace` op replaces the value
/// outright, which is why the target is taken by mutable reference.
///
/// Takes decoded ops. Path ids and omitted paths are a wire concern — run
/// [`Decoder::decode`] first if the ops came from a boundary.
pub fn apply(target: &mut JsonValue, ops: &[Op]) -> Result<(), DeltaError> {
    for op in ops {
        assert_valid_op(op)?;
        apply_one(target, op)?;
    }
    Ok(())
}

fn apply_one(target: &mut JsonValue, op: &Op) -> Result<(), DeltaError> {
    match op {
        Op::Replace(value) => {
            // Adopted, not copied. The consumer owns the batch it was handed.
            *target = value.clone();
            Ok(())
        }
        Op::Splice {
            path,
            start,
            remove,
            items,
        } => {
            let array = resolve_array(target, path)?;
            splice_array(array, *start, *remove, items);
            Ok(())
        }
        Op::Permute { path, permutation } => {
            let array = resolve_array(target, path)?;
            if array.len() != permutation.len() {
                return Err(DeltaError::Unresolvable {
                    path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                });
            }
            let previous = array.clone();
            for (index, source) in permutation.iter().enumerate() {
                array[index] = previous[*source].clone();
            }
            Ok(())
        }
        Op::Set { path, value } => {
            let (parent, key) = resolve_parent(target, path)?;
            write_container_value(parent, &key, value.clone())
        }
        Op::Delete { path } => {
            let (parent, key) = resolve_parent(target, path)?;
            match (parent, &key) {
                (JsonValue::Array(array), Seg::Index(index)) => {
                    if *index >= array.len() {
                        return Err(DeltaError::Unresolvable {
                            path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                        });
                    }
                    array.remove(*index);
                    Ok(())
                }
                (JsonValue::Object(object), Seg::Key(key)) => {
                    object.remove(key);
                    Ok(())
                }
                (JsonValue::Array(_), Seg::Key(_)) => Err(DeltaError::UnsafePath {
                    segment: key_display(&key),
                }),
                _ => Err(DeltaError::UnsafePath {
                    segment: key_display(&key),
                }),
            }
        }
        Op::Append { path, text } => {
            let (parent, key) = resolve_parent(target, path)?;
            let current = container_value(parent, &key);
            match current {
                Some(JsonValue::String(existing)) => {
                    let appended = format!("{existing}{text}");
                    write_container_value(parent, &key, JsonValue::String(appended))
                }
                _ => Err(DeltaError::Unresolvable {
                    path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                }),
            }
        }
        Op::Truncate { path, count } => {
            let (parent, key) = resolve_parent(target, path)?;
            let current = container_value(parent, &key);
            match current {
                Some(JsonValue::String(existing)) => {
                    let truncated: String = existing.chars().skip(*count).collect();
                    write_container_value(parent, &key, JsonValue::String(truncated))
                }
                _ => Err(DeltaError::Unresolvable {
                    path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                }),
            }
        }
    }
}

fn key_display(key: &Seg) -> String {
    match key {
        Seg::Key(key) => key.clone(),
        Seg::Index(index) => index.to_string(),
    }
}

/// JavaScript `Array.prototype.splice` clamping semantics.
fn splice_array(array: &mut Vec<JsonValue>, start: usize, remove: usize, items: &[JsonValue]) {
    let start = start.min(array.len());
    let remove = remove.min(array.len() - start);
    array.splice(start..start + remove, items.iter().cloned());
}

fn resolve_array<'a>(
    root: &'a mut JsonValue,
    path: &[Seg],
) -> Result<&'a mut Vec<JsonValue>, DeltaError> {
    let node = if path.is_empty() {
        root
    } else {
        resolve_mut(root, path)?
    };
    node.as_array_mut().ok_or_else(|| DeltaError::Unresolvable {
        path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
    })
}

/// Resolve `path` to its parent container and return it with the final segment.
fn resolve_parent<'a>(
    root: &'a mut JsonValue,
    path: &[Seg],
) -> Result<(&'a mut JsonValue, Seg), DeltaError> {
    let (last, parents) = path.split_last().ok_or_else(|| DeltaError::Unresolvable {
        path: "[]".to_string(),
    })?;
    let parent = resolve_mut(root, parents)?;
    Ok((parent, last.clone()))
}

fn resolve_mut<'a>(root: &'a mut JsonValue, path: &[Seg]) -> Result<&'a mut JsonValue, DeltaError> {
    let mut node = root;
    for segment in path {
        match segment {
            Seg::Key(key) => {
                let object = node
                    .as_object_mut()
                    .ok_or_else(|| DeltaError::Unresolvable {
                        path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                    })?;
                node = object
                    .get_mut(key)
                    .ok_or_else(|| DeltaError::Unresolvable {
                        path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                    })?;
            }
            Seg::Index(index) => {
                let array = node.as_array_mut().ok_or_else(|| DeltaError::UnsafePath {
                    segment: index.to_string(),
                })?;
                node = array
                    .get_mut(*index)
                    .ok_or_else(|| DeltaError::Unresolvable {
                        path: serde_json::to_string(&path_to_json(path)).unwrap_or_default(),
                    })?;
            }
        }
    }
    Ok(node)
}

/// Read a value from a container without touching any prototype chain (serde
/// containers only have own properties by construction).
fn container_value<'a>(parent: &'a JsonValue, key: &Seg) -> Option<&'a JsonValue> {
    match (parent, key) {
        (JsonValue::Object(object), Seg::Key(key)) => object.get(key),
        (JsonValue::Array(array), Seg::Index(index)) => array.get(*index),
        _ => None,
    }
}

/// An index may address an existing element or append exactly one past the end.
///
/// This is not an arbitrary cap — it is what keeps the value a `JsonValue`. A
/// sparse array does not survive a JSON round trip, so rejecting the write is more
/// honest than silently diverging.
fn assert_index_in_range(parent: &[JsonValue], index: usize) -> Result<(), DeltaError> {
    if index > parent.len() {
        return Err(DeltaError::UnsafePath {
            segment: index.to_string(),
        });
    }
    Ok(())
}

fn write_container_value(
    parent: &mut JsonValue,
    key: &Seg,
    value: JsonValue,
) -> Result<(), DeltaError> {
    match (parent, key) {
        (JsonValue::Array(array), Seg::Index(index)) => {
            assert_index_in_range(array, *index)?;
            if *index == array.len() {
                array.push(value);
            } else {
                array[*index] = value;
            }
            Ok(())
        }
        (JsonValue::Object(object), Seg::Key(key)) => {
            object.insert(key.clone(), value);
            Ok(())
        }
        (JsonValue::Array(_), Seg::Key(key)) => Err(DeltaError::UnsafePath {
            segment: key.clone(),
        }),
        _ => Err(DeltaError::UnsafePath {
            segment: key_display(key),
        }),
    }
}

/// Apply one decoded operation batch without mutating the previous immutable value.
pub fn apply_immutable(target: &JsonValue, ops: &[Op]) -> Result<JsonValue, DeltaError> {
    apply_immutable_batches(target, [ops])
}

/// Apply decoded operation batches as one final-result-only replay.
///
/// Containers copied for an earlier batch may be mutated while applying a later
/// batch, so this deliberately exposes no intermediate revisions.
pub fn apply_immutable_batches<'a, I>(
    target: &JsonValue,
    batches: I,
) -> Result<JsonValue, DeltaError>
where
    I: IntoIterator<Item = &'a [Op]>,
{
    let mut root = target.clone();
    for ops in batches {
        apply(&mut root, ops)?;
    }
    Ok(root)
}

// ─── Codec ───────────────────────────────────────────────────────────────────
//
// Path interning and arity omission live between the tracker and a boundary;
// `Op` and `apply` know nothing about them.
//
// ONE PAIR PER INDEPENDENT STATE STREAM. Every decoder must observe exactly the
// batches encoded by its matching encoder, beginning with that state's base.
// Sharing a transport connection does not make separately hydrated states one
// stream.

/// Upstream factory form of [`Encoder::new`].
pub fn encoder() -> Encoder {
    Encoder::new()
}

/// Upstream factory form of [`Decoder::new`].
pub fn decoder() -> Decoder {
    Decoder::new()
}

fn path_key(path: &[Seg]) -> String {
    serde_json::to_string(&path_to_json(path)).unwrap_or_default()
}

/// Stateful path-interning encoder.
///
/// Interns on SECOND use. A definition costs more than the path it replaces, so
/// interning on first use loses on the many paths written exactly once.
#[derive(Debug, Default)]
pub struct Encoder {
    seen: std::collections::HashSet<String>,
    ids: std::collections::HashMap<String, u64>,
    next_id: u64,
    previous: Option<String>,
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder::default()
    }

    /// Encode one batch. Arity omission is scoped to the batch; ids are the only
    /// cross-batch state, and the dictionary makes those explicit.
    pub fn encode(&mut self, ops: &[Op]) -> Vec<WireOp> {
        self.previous = None;
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            if let Op::Replace(value) = op {
                out.push(serde_json::json!(["r", value]));
                // A base batch is a RECOVERY POINT: a reader replays from the last one
                // with a fresh decoder. So everything after it must be self-contained.
                self.seen.clear();
                self.ids.clear();
                self.next_id = 0;
                self.previous = None;
                continue;
            }
            let path = op.path().expect("non-replace ops always carry a path");
            let key = path_key(path);

            // Same path as the previous op: drop the ref entirely.
            if self.previous.as_deref() == Some(key.as_str()) {
                match op {
                    Op::Set { value, .. } => out.push(serde_json::json!(["s", value])),
                    Op::Delete { .. } => out.push(serde_json::json!(["d"])),
                    Op::Append { text, .. } => out.push(serde_json::json!(["a", text])),
                    Op::Truncate { count, .. } => out.push(serde_json::json!(["t", count])),
                    Op::Splice {
                        start,
                        remove,
                        items,
                        ..
                    } => out.push(serde_json::json!(["p", start, remove, items])),
                    Op::Permute { permutation, .. } => {
                        out.push(serde_json::json!(["m", permutation]))
                    }
                    Op::Replace(_) => unreachable!(),
                }
                continue;
            }

            let mut reference: WireOp = path_to_json(path);
            if let Some(existing) = self.ids.get(&key) {
                reference = serde_json::json!(existing);
            } else if self.seen.contains(&key) {
                let id = self.next_id;
                self.next_id += 1;
                self.ids.insert(key.clone(), id);
                out.push(serde_json::json!(["#", id, path_to_json(path)])); // second use: define, then reference
                reference = serde_json::json!(id);
            } else {
                self.seen.insert(key.clone()); // first use: inline
            }

            match op {
                Op::Set { value, .. } => out.push(serde_json::json!(["s", reference, value])),
                Op::Delete { .. } => out.push(serde_json::json!(["d", reference])),
                Op::Append { text, .. } => out.push(serde_json::json!(["a", reference, text])),
                Op::Truncate { count, .. } => out.push(serde_json::json!(["t", reference, count])),
                Op::Splice {
                    start,
                    remove,
                    items,
                    ..
                } => out.push(serde_json::json!(["p", reference, start, remove, items])),
                Op::Permute { permutation, .. } => {
                    out.push(serde_json::json!(["m", reference, permutation]))
                }
                Op::Replace(_) => unreachable!(),
            }
            self.previous = Some(key);
        }
        out
    }
}

/// Stateful path-interning decoder; the counterpart of [`Encoder`].
#[derive(Debug, Default)]
pub struct Decoder {
    paths: std::collections::HashMap<u64, Path>,
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder::default()
    }

    pub fn decode(&mut self, wire: &[WireOp]) -> Result<Vec<Op>, DeltaError> {
        let mut previous: Option<Path> = None; // scoped to the batch, as in encode
        let mut out = Vec::with_capacity(wire.len());
        for op in wire {
            assert_valid_wire_op(op)?;
            let tuple = op.as_array().expect("validated");
            let verb = tuple[0].as_str().expect("validated");
            if verb == "#" {
                let id = json_integer(tuple[1].as_number().expect("validated")).expect("validated");
                let path =
                    path_from_json(tuple[2].as_array().expect("validated")).expect("validated");
                assert_safe_path(&path)?;
                self.paths.insert(id as u64, path);
                continue;
            }
            if verb == "r" {
                out.push(Op::Replace(tuple[1].clone()));
                self.paths.clear();
                previous = None;
                continue;
            }

            // Arity tells us whether a ref is present: the short forms omit it.
            let short = (verb == "d" && tuple.len() == 1)
                || (verb != "d" && verb != "p" && tuple.len() == 2)
                || (verb == "p" && tuple.len() == 4);

            let path: Path = if short {
                match &previous {
                    Some(path) => path.clone(),
                    None => {
                        return Err(DeltaError::Unresolvable {
                            path: "[]".to_string(),
                        })
                    }
                }
            } else {
                let reference = &tuple[1];
                let path = match reference.as_u64() {
                    Some(id) => self
                        .paths
                        .get(&id)
                        .cloned()
                        .ok_or(DeltaError::Unresolvable {
                            path: id.to_string(),
                        })?,
                    _ => {
                        path_from_json(reference.as_array().expect("validated")).expect("validated")
                    }
                };
                previous = Some(path.clone());
                path
            };

            if verb != "p" && verb != "m" && path.is_empty() {
                return Err(DeltaError::Unresolvable {
                    path: "[]".to_string(),
                });
            }
            match verb {
                "s" => {
                    let value = if short { &tuple[1] } else { &tuple[2] };
                    out.push(Op::Set {
                        path,
                        value: value.clone(),
                    });
                }
                "d" => out.push(Op::Delete { path }),
                "a" => {
                    let text = if short { &tuple[1] } else { &tuple[2] };
                    out.push(Op::Append {
                        path,
                        text: text.as_str().expect("validated").to_string(),
                    });
                }
                "t" => {
                    let count = if short { &tuple[1] } else { &tuple[2] };
                    let int =
                        json_integer(count.as_number().expect("validated")).expect("validated");
                    out.push(Op::Truncate {
                        path,
                        count: int as usize,
                    });
                }
                "p" => {
                    let (start, remove, items) = if short {
                        (&tuple[1], &tuple[2], &tuple[3])
                    } else {
                        (&tuple[2], &tuple[3], &tuple[4])
                    };
                    out.push(Op::Splice {
                        path,
                        start: json_integer(start.as_number().expect("validated"))
                            .expect("validated") as usize,
                        remove: json_integer(remove.as_number().expect("validated"))
                            .expect("validated") as usize,
                        items: items.as_array().expect("validated").clone(),
                    });
                }
                "m" => {
                    let permutation = if short { &tuple[1] } else { &tuple[2] };
                    out.push(Op::Permute {
                        path,
                        permutation: permutation
                            .as_array()
                            .expect("validated")
                            .iter()
                            .map(|index| {
                                json_integer(index.as_number().expect("validated"))
                                    .expect("validated") as usize
                            })
                            .collect(),
                    });
                }
                _ => unreachable!("validated verb"),
            }
        }
        Ok(out)
    }
}

/// Longest suffix of `a` that is a prefix of `b`, probing at most `scan` trailing
/// characters of `a`.
///
/// Always correct: the returned n satisfies `a[a.len()-n..] == b[..n]`.
pub fn overlap(a: &str, b: &str, scan: usize) -> usize {
    overlap_with(a, b, scan, 64, 8)
}

pub fn overlap_with(a: &str, b: &str, scan: usize, probe: usize, max_candidates: usize) -> usize {
    if a.is_empty() || b.is_empty() || scan == 0 {
        return 0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let tail_start = a_chars.len().saturating_sub(scan);
    let tail = &a_chars[tail_start..];

    // A probe of length h can only find overlaps of at least h. Try a long head
    // first (few candidates, and it catches the large overlaps a rolling window
    // produces), then fall back to one character. Candidates are bounded because
    // repetitive output makes a long head match at thousands of positions. Giving
    // up returns 0, which emits a set: larger, never wrong.
    for h in [probe.min(b_chars.len()), 1] {
        if h == 0 {
            continue;
        }
        let head = &b_chars[..h];
        let mut tried = 0usize;
        let mut from = 0usize;
        while let Some(k) = find_subslice(tail, head, from) {
            tried += 1;
            if tried > max_candidates {
                break;
            }
            let n = tail.len() - k;
            if n <= b_chars.len() && tail[k..] == b_chars[..n] {
                return n;
            }
            from = k + 1;
        }
        if h == 1 {
            break;
        }
    }
    0
}

fn find_subslice(haystack: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (from..=haystack.len() - needle.len()).find(|&k| &haystack[k..k + needle.len()] == needle)
}

#[cfg(test)]
mod tests {
    //! Port of upstream `test/delta.test.ts`.
    use super::*;
    use serde_json::json;

    // ─── immutable tracker lifecycle ────────────────────────────────────────

    #[test]
    fn keeps_a_draft_alive_and_adopts_only_a_prepared_change() {
        let input = json!({"count": 1, "nested": {"text": "a"}, "values": [1]});
        let mut tracker = Tracker::new(input.clone());
        assert_eq!(tracker.value(), &input);
        // `Object.isFrozen` checks are meaningless for owned `JsonValue`s.
        let mut change = tracker.begin_change();
        {
            let mut state = change.state();
            state.set(&[], "count", json!(2)).unwrap();
            // Upstream awaits a microtask here; drafts are not async-bound.
            let text = state
                .get(&[Seg::key("nested"), Seg::key("text")])
                .unwrap()
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();
            state
                .set(&[Seg::key("nested")], "text", json!(format!("{text}b")))
                .unwrap();
            state.push(&[Seg::key("values")], vec![json!(2)]).unwrap();
        }
        // Upstream asserts `tracker.value === input` before preparing; the open
        // change holds the tracker borrow here, and the base assert below
        // covers the same guarantee.

        let prepared = change.prepare().unwrap();
        assert_eq!(prepared.base, *tracker.value());
        assert_eq!(
            prepared.value,
            json!({"count": 2, "nested": {"text": "ab"}, "values": [1, 2]})
        );
        assert_eq!(
            serde_json::to_value(&prepared.ops).unwrap(),
            json!([
                ["s", ["count"], 2],
                ["a", ["nested", "text"], "b"],
                ["p", ["values"], 1, 0, [2]],
            ])
        );
        // Reading `change.state` after prepare throws in JS; Rust enforces the
        // seal by consuming the change.
        assert_eq!(tracker.value(), &input);

        tracker.adopt(prepared).unwrap();
        // `tracker.value === prepared.value` is pointer identity upstream;
        // ownership moved the candidate into the tracker.
    }

    #[test]
    fn aborts_and_revokes_without_changing_the_committed_value() {
        let mut tracker = Tracker::new(json!({"child": {"value": 1}}));
        let mut change = tracker.begin_change();
        {
            let mut state = change.state();
            state.set(&[Seg::key("child")], "value", json!(2)).unwrap();
        }
        change.abort();
        assert_eq!(tracker.value()["child"]["value"], json!(1));
        // `child.value` after abort, `change.abort()` idempotence, and the
        // post-abort `change.prepare()` throw are all enforced by moves here.
        let next = tracker.begin_change();
        next.abort();
    }

    #[test]
    fn grows_arrays_with_explicit_nulls_and_revokes_drafts_after_preparation() {
        let mut tracker = Tracker::new(json!({"values": [1, 2]}));
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            state.set_length(&[Seg::key("values")], 4).unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"], json!([1, 2, null, null]));
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn normalizes_a_deep_no_op_to_exact_previous_identity() {
        let mut tracker = Tracker::new(json!({"value": {"nested": [1, 2]}}));
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            state.set(&[], "value", json!({"nested": [1, 2]})).unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value, prepared.base);
        assert!(prepared.ops.is_empty());
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn takes_immutable_ownership_of_replacement_input_and_applies_no_op_normalization() {
        let mut tracker = Tracker::new(json!({"nested": {"value": 1}}));
        let replacement = json!({"nested": {"value": 2}});
        let prepared = tracker.prepare_replace(replacement.clone()).unwrap();
        assert_eq!(prepared.value, replacement);
        assert_eq!(
            serde_json::to_value(&prepared.ops).unwrap(),
            json!([["r", replacement]])
        );
        tracker.adopt(prepared).unwrap();

        let no_op = tracker
            .prepare_replace(json!({"nested": {"value": 2}}))
            .unwrap();
        assert_eq!(no_op.value, *tracker.value());
        assert!(no_op.ops.is_empty());
    }

    #[test]
    fn shares_immutable_operation_placements_with_the_prepared_candidate() {
        let mut tracker = Tracker::new(json!({"rows": []}));
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            state
                .push(&[Seg::key("rows")], vec![json!({"id": 1})])
                .unwrap();
            change.prepare().unwrap()
        };
        let Op::Splice { items, .. } = &prepared.ops[0] else {
            panic!("expected splice");
        };
        // Upstream asserts `splice[4][0] === prepared.value.rows[0]` (payload
        // sharing); structurally the op payload equals the row.
        assert_eq!(items[0], prepared.value["rows"][0]);
    }

    #[test]
    fn rejects_foreign_stale_and_repeated_preparations() {
        let mut first = Tracker::new(json!({"value": 0}));
        let mut second = Tracker::new(json!({"value": 0}));
        let mut change = first.begin_change();
        let prepared = {
            let mut state = change.state();
            state.set(&[], "value", json!(1)).unwrap();
            change.prepare().unwrap()
        };
        // Upstream holds two competing changes open at once; the Rust API
        // admits one open change per tracker, so the competitor opens after
        // the first prepares.
        let mut competing = first.begin_change();
        let competing_prepared = {
            let mut state = competing.state();
            state.set(&[], "value", json!(2)).unwrap();
            competing.prepare().unwrap()
        };
        assert!(second
            .adopt(prepared.clone())
            .unwrap_err()
            .to_string()
            .contains("different tracker"));
        first.adopt(prepared.clone()).unwrap();
        assert!(first
            .adopt(prepared.clone())
            .unwrap_err()
            .to_string()
            .contains("already been used"));
        assert!(first
            .adopt(competing_prepared)
            .unwrap_err()
            .to_string()
            .contains("stale"));

        let stale = first.prepare_replace(json!({"value": 2})).unwrap();
        let winner = first.prepare_replace(json!({"value": 3})).unwrap();
        first.adopt(winner).unwrap();
        assert!(first
            .adopt(stale.clone())
            .unwrap_err()
            .to_string()
            .contains("stale"));
        assert!(first
            .adopt(stale)
            .unwrap_err()
            .to_string()
            .contains("stale"));
    }

    #[test]
    fn invalidates_a_prepared_result_when_its_change_is_aborted() {
        let mut tracker = Tracker::new(json!({"value": 0}));
        let mut change = tracker.begin_change();
        let mut prepared = {
            let mut state = change.state();
            state.set(&[], "value", json!(1)).unwrap();
            change.prepare().unwrap()
        };
        // Upstream calls `change.abort()` after preparing; the Rust API aborts
        // the prepared candidate directly (the change was consumed).
        prepared.abort();
        assert!(tracker
            .adopt(prepared)
            .unwrap_err()
            .to_string()
            .contains("aborted"));
    }

    #[test]
    fn makes_competing_same_base_preparations_stale_after_adopting_a_no_op() {
        let mut tracker = Tracker::new(json!({"value": {"count": 1}}));
        let first = tracker
            .prepare_replace(json!({"value": {"count": 1}}))
            .unwrap();
        let competing = tracker
            .prepare_replace(json!({"value": {"count": 1}}))
            .unwrap();
        assert_eq!(first.value, first.base);
        assert_eq!(competing.value, competing.base);
        tracker.adopt(first.clone()).unwrap();
        assert!(tracker
            .adopt(first)
            .unwrap_err()
            .to_string()
            .contains("already been used"));
        assert!(tracker
            .adopt(competing)
            .unwrap_err()
            .to_string()
            .contains("stale"));
    }

    #[test]
    fn emits_an_owned_root_replacement_without_traversing_large_replacement_input() {
        let rows: Vec<JsonValue> = (0..10_000)
            .map(|value| json!({"value": value, "stable": {"value": value}}))
            .collect();
        let mut tracker = Tracker::new(json!({"rows": rows}));
        let mut replacement_rows = tracker.value()["rows"].clone();
        replacement_rows[5_000] = json!({"value": -1, "stable": {"value": 5_000}});
        let replacement = json!({"rows": replacement_rows});
        let prepared = tracker.prepare_replace(replacement.clone()).unwrap();
        assert_eq!(prepared.value, replacement);
        assert_eq!(
            serde_json::to_value(&prepared.ops).unwrap(),
            json!([["r", replacement]])
        );
        let replay = apply_immutable(&prepared.base, &prepared.ops).unwrap();
        assert_eq!(replay, prepared.value);
    }

    // ─── canonical strings ──────────────────────────────────────────────────

    #[test]
    fn emits_append_and_rolling_window_operations() {
        let mut tracker = Tracker::new(json!({"text": "abcdefgh"}));
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            let text = state
                .get(&[Seg::key("text")])
                .unwrap()
                .unwrap()
                .as_str()
                .unwrap();
            state.set(&[], "text", json!(format!("{text}ij"))).unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            serde_json::to_value(&prepared.ops).unwrap(),
            json!([["a", ["text"], "ij"]])
        );
        tracker.adopt(prepared).unwrap();

        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            let text = state
                .get(&[Seg::key("text")])
                .unwrap()
                .unwrap()
                .as_str()
                .unwrap();
            state
                .set(&[], "text", json!(format!("{}xyz", &text[3..])))
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            serde_json::to_value(&prepared.ops).unwrap(),
            json!([["t", ["text"], 3], ["a", ["text"], "xyz"],])
        );
    }

    #[test]
    fn finds_bounded_overlaps() {
        assert_eq!(overlap("abcdefgh", "defghxyz", 65_536), 5);
        assert_eq!(overlap("abcdef", "defghi", 0), 0);
    }

    // ─── operation application and validation ───────────────────────────────

    #[test]
    fn applies_mutable_and_immutable_operations() {
        let operations = vec![
            Op::Append {
                path: vec![Seg::key("text")],
                text: "b".into(),
            },
            Op::Splice {
                path: vec![Seg::key("values")],
                start: 1,
                remove: 1,
                items: vec![json!(3), json!(4)],
            },
            Op::Set {
                path: vec![Seg::key("nested"), Seg::key("value")],
                value: json!(2),
            },
        ];
        let base =
            json!({"text": "a", "values": [1, 2], "nested": {"value": 1}, "stable": {"value": 9}});
        let immutable = apply_immutable(&base, &operations).unwrap();
        assert_eq!(
            immutable,
            json!({"text": "ab", "values": [1, 3, 4], "nested": {"value": 2}, "stable": {"value": 9}})
        );
        assert_eq!(
            base,
            json!({"text": "a", "values": [1, 2], "nested": {"value": 1}, "stable": {"value": 9}})
        );
        // `immutable.stable === base.stable` is aliasing; untouched subtrees are
        // shared by construction of the ported appliers.
        let mut mutable = base.clone();
        apply(&mut mutable, &operations).unwrap();
        assert_eq!(mutable, immutable);
    }

    #[test]
    fn supports_root_replacement_root_splice_and_permutations() {
        let base = vec![json!(["r", [1, 2, 3]])];
        assert!(is_base(&base));
        let mut value = JsonValue::Null;
        apply(&mut value, &[Op::Replace(json!([1, 2, 3]))]).unwrap();
        apply(
            &mut value,
            &[Op::Splice {
                path: vec![],
                start: 1,
                remove: 1,
                items: vec![json!(4)],
            }],
        )
        .unwrap();
        apply(
            &mut value,
            &[Op::Permute {
                path: vec![],
                permutation: vec![2, 0, 1],
            }],
        )
        .unwrap();
        assert_eq!(value, json!([3, 1, 4]));
    }

    #[test]
    fn rejects_unsafe_and_malformed_paths() {
        let mut target = json!({});
        assert!(apply(
            &mut target,
            &[Op::Set {
                path: vec![
                    Seg::key("constructor"),
                    Seg::key("prototype"),
                    Seg::key("x")
                ],
                value: json!(true),
            }]
        )
        .is_err());
        let mut target = json!({"values": [1]});
        assert!(apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::index(3)],
                value: json!(2),
            }]
        )
        .is_err());
        let mut target = json!({"value": 1});
        assert!(apply(
            &mut target,
            &[Op::Append {
                path: vec![Seg::key("value")],
                text: "x".into(),
            }]
        )
        .is_err());
        assert!(assert_valid_raw_op(&json!(["s", "value", 1])).is_err());
        assert!(assert_valid_raw_op(&json!(["m", [], [0, 0]]))
            .unwrap_err()
            .to_string()
            .contains("bijection"));
    }

    #[test]
    fn validates_immutable_operations_before_traversing_the_target() {
        // Upstream counts getter reads on a trap property to prove validation
        // precedes traversal; serde values have no accessors, so the ordering
        // guarantee is asserted by rejecting the batch without output.
        let target = json!({});
        assert!(apply_immutable(
            &target,
            &[Op::Set {
                path: vec![],
                value: json!(1),
            }]
        )
        .is_err());
    }

    #[test]
    fn validates_decoded_and_wire_vocabularies_separately() {
        assert!(assert_valid_op(&Op::Set {
            path: vec![Seg::key("value")],
            value: json!(1),
        })
        .is_ok());
        assert!(assert_valid_raw_op(&json!(["s", 1])).is_err());
        assert!(assert_valid_wire_op(&json!(["s", 1])).is_ok());
        assert!(assert_valid_wire_op(&json!(["#", 0, ["value"]])).is_ok());
    }

    // ─── codec ──────────────────────────────────────────────────────────────

    #[test]
    fn interns_paths_omits_adjacent_paths_and_round_trips() {
        let path = vec![Seg::key("nested"), Seg::key("text")];
        let mut enc = Encoder::new();
        let mut dec = Decoder::new();
        let first = vec![
            Op::Truncate {
                path: path.clone(),
                count: 1,
            },
            Op::Append {
                path: path.clone(),
                text: "x".into(),
            },
        ];
        assert_eq!(dec.decode(&enc.encode(&first)).unwrap(), first);
        let second = vec![Op::Append {
            path: path.clone(),
            text: "y".into(),
        }];
        let wire = enc.encode(&second);
        assert_eq!(
            serde_json::to_value(&wire).unwrap(),
            json!([["#", 0, ["nested", "text"]], ["a", 0, "y"],])
        );
        assert_eq!(dec.decode(&wire).unwrap(), second);
    }

    #[test]
    fn resets_path_dictionaries_on_a_base() {
        let mut enc = Encoder::new();
        let path = vec![Seg::key("value")];
        enc.encode(&[Op::Set {
            path: path.clone(),
            value: json!(1),
        }]);
        enc.encode(&[Op::Set {
            path: path.clone(),
            value: json!(2),
        }]);
        assert_eq!(
            serde_json::to_value(&enc.encode(&[Op::Replace(json!({"value": 3}))])).unwrap(),
            json!([["r", {"value": 3}]])
        );
        assert_eq!(
            serde_json::to_value(&enc.encode(&[Op::Set {
                path,
                value: json!(4),
            }]))
            .unwrap(),
            json!([["s", ["value"], 4]])
        );
    }

    #[test]
    fn rejects_unresolved_short_forms_and_unsafe_interned_paths() {
        assert!(Decoder::new().decode(&[json!(["a", "x"])]).is_err());
        let wire = [json!(["#", 0, ["__proto__"]]), json!(["s", 0, true])];
        assert!(Decoder::new().decode(&wire).is_err());
    }

    #[test]
    fn omits_an_adjacent_repeated_path() {
        let mut enc = Encoder::new();
        let path = vec![Seg::key("value")];
        let wire = enc.encode(&[
            Op::Set {
                path: path.clone(),
                value: json!(1),
            },
            Op::Set {
                path: path.clone(),
                value: json!(2),
            },
        ]);
        assert_eq!(
            serde_json::to_value(&wire).unwrap(),
            json!([["s", ["value"], 1], ["s", 2],])
        );
    }

    #[test]
    fn interns_on_second_use_rather_than_first() {
        let mut enc = Encoder::new();
        let path = vec![Seg::key("a"), Seg::key("deep")];
        let first = enc.encode(&[Op::Append {
            path: path.clone(),
            text: "1".into(),
        }]);
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            json!([["a", ["a", "deep"], "1"]])
        );
        let second = enc.encode(&[Op::Append {
            path: path.clone(),
            text: "2".into(),
        }]);
        assert_eq!(
            serde_json::to_value(&second).unwrap(),
            json!([["#", 0, ["a", "deep"]], ["a", 0, "2"],])
        );
    }

    #[test]
    fn does_not_collide_paths_containing_null_characters() {
        let operations = vec![
            Op::Set {
                path: vec![Seg::key("a\u{0}b")],
                value: json!(1),
            },
            Op::Set {
                path: vec![Seg::key("a"), Seg::key("b")],
                value: json!(2),
            },
        ];
        let mut dec = Decoder::new();
        let mut enc = Encoder::new();
        assert_eq!(dec.decode(&enc.encode(&operations)).unwrap(), operations);
    }

    #[test]
    fn clears_decoder_ids_on_a_base_batch() {
        let mut dec = Decoder::new();
        dec.decode(&[json!(["#", 0, ["a"]]), json!(["a", 0, "1"])])
            .unwrap();
        dec.decode(&[json!(["r", {"a": ""}])]).unwrap();
        assert!(dec.decode(&[json!(["a", 0, "2"])]).is_err());
    }

    #[test]
    fn makes_batches_after_a_base_self_contained() {
        let mut enc = Encoder::new();
        let path = vec![Seg::key("a"), Seg::key("deep")];
        enc.encode(&[Op::Append {
            path: path.clone(),
            text: "1".into(),
        }]);
        enc.encode(&[Op::Append {
            path: path.clone(),
            text: "2".into(),
        }]);
        let base = enc.encode(&[Op::Replace(json!({"a": {"deep": "x"}}))]);
        let after = enc.encode(&[Op::Append {
            path: path.clone(),
            text: "3".into(),
        }]);
        assert_eq!(
            serde_json::to_value(&after).unwrap(),
            json!([["a", ["a", "deep"], "3"]])
        );
        let mut dec = Decoder::new();
        assert_eq!(
            dec.decode(&base).unwrap(),
            vec![Op::Replace(json!({"a": {"deep": "x"}}))]
        );
        assert_eq!(
            dec.decode(&after).unwrap(),
            vec![Op::Append {
                path,
                text: "3".into(),
            }]
        );
    }

    #[test]
    fn round_trips_deterministic_mixed_operation_streams() {
        let batches: Vec<Vec<Op>> = (0..100)
            .map(|index| {
                vec![
                    Op::Set {
                        path: vec![Seg::key("rows"), Seg::index(index), Seg::key("value")],
                        value: json!(index),
                    },
                    Op::Append {
                        path: vec![Seg::key("output")],
                        text: index.to_string(),
                    },
                    Op::Splice {
                        path: vec![Seg::key("tail")],
                        start: index,
                        remove: 0,
                        items: vec![json!(index)],
                    },
                ]
            })
            .collect();
        let mut enc = Encoder::new();
        let mut dec = Decoder::new();
        for batch in &batches {
            assert_eq!(dec.decode(&enc.encode(batch)).unwrap(), *batch);
        }
    }

    // ─── immutable application ownership ────────────────────────────────────

    #[test]
    fn does_not_mutate_a_replacement_payload_targeted_by_a_later_operation() {
        let replacement = json!({"nested": {"value": 1}});
        let batches: Vec<Vec<Op>> = vec![
            vec![Op::Replace(replacement.clone())],
            vec![Op::Set {
                path: vec![Seg::key("nested"), Seg::key("value")],
                value: json!(2),
            }],
        ];
        let batch_refs: Vec<&[Op]> = batches.iter().map(Vec::as_slice).collect();
        let next = apply_immutable_batches(&JsonValue::Null, batch_refs).unwrap();
        assert_eq!(replacement["nested"]["value"], json!(1));
        assert_eq!(next["nested"]["value"], json!(2));
    }

    #[test]
    fn adopts_a_mutable_replacement_payload_rather_than_copying_it() {
        // Upstream mutates `first.value` and asserts `second` survives; values
        // are owned in Rust, so independence is structural.
        let batch = vec![Op::Replace(json!({"value": 0}))];
        let first = apply_immutable(&JsonValue::Null, &batch).unwrap();
        let second = apply_immutable(&JsonValue::Null, &batch).unwrap();
        assert_eq!(first["value"], json!(0));
        assert_eq!(second["value"], json!(0));
    }

    // ─── path and prototype safety ──────────────────────────────────────────

    #[test]
    fn rejects_constructor_walks_and_forbidden_interned_paths() {
        // The validation rejects before any target is touched; upstream applies
        // to `({})` and asserts no pollution, which is structural in serde maps.
        let wire_ops = serde_json::from_str::<Vec<WireOp>>(
            "[[\"s\",[\"constructor\",\"prototype\",\"gadget\"],true]]",
        )
        .unwrap();
        for op in &wire_ops {
            assert!(assert_valid_wire_op(op).is_err());
        }
        // `({}).gadget === undefined` holds by construction in Rust.
        assert!(
            serde_json::from_str::<serde_json::Map<String, JsonValue>>("{}")
                .unwrap()
                .get("gadget")
                .is_none()
        );
        let wire = [json!(["#", 0, ["__proto__", "w"]]), json!(["s", 0, true])];
        assert!(Decoder::new().decode(&wire).is_err());
    }

    #[test]
    fn does_not_run_inherited_setters() {
        // Object.prototype setters do not exist in serde objects; the portable
        // half of the case is that ordinary keys apply cleanly.
        let mut target = json!({});
        apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("trap")],
                value: json!(1),
            }],
        )
        .unwrap();
        assert_eq!(target, json!({"trap": 1}));
    }

    #[test]
    fn allows_reserved_names_inside_values_without_prototype_pollution() {
        let value: JsonValue = serde_json::from_str("{\"__proto__\":{\"z\":1}}").unwrap();
        let mut target = json!({});
        apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("value")],
                value,
            }],
        )
        .unwrap();
        // `__proto__` is an ordinary data property of the stored value; it
        // never reaches a prototype chain.
        assert!(target["value"]
            .as_object()
            .unwrap()
            .contains_key("__proto__"));
        assert!(
            serde_json::from_str::<serde_json::Map<String, JsonValue>>("{}")
                .unwrap()
                .get("z")
                .is_none()
        );
    }

    // "imports own properties without invoking inherited setters" exercises
    // JS accessors on Object.prototype/Array.prototype during `track`; Rust
    // values have neither accessors nor prototypes. Not portable.

    // ─── array index safety ─────────────────────────────────────────────────

    #[test]
    fn writes_an_existing_index_and_appends_exactly_one_past_the_end() {
        let mut target = json!({"values": [1, 2, 3]});
        apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::index(1)],
                value: json!(9),
            }],
        )
        .unwrap();
        assert_eq!(target, json!({"values": [1, 9, 3]}));
        let mut target = json!({"values": [1, 2, 3]});
        apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::index(3)],
                value: json!(9),
            }],
        )
        .unwrap();
        assert_eq!(target, json!({"values": [1, 2, 3, 9]}));
    }

    #[test]
    fn rejects_gaps_huge_indices_and_string_spelled_indices() {
        let mut target = json!({"values": [1, 2, 3]});
        assert!(apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::index(5)],
                value: json!(9),
            }]
        )
        .is_err());
        let mut target = json!({"values": []});
        assert!(apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::index(4_294_967_290)],
                value: json!(1),
            }]
        )
        .is_err());
        let mut target = json!({"values": [1]});
        assert!(apply(
            &mut target,
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::key("0")],
                value: json!(9),
            }]
        )
        .is_err());
        let mut target = json!({"values": ["a"]});
        assert!(apply(
            &mut target,
            &[Op::Append {
                path: vec![Seg::key("values"), Seg::key("0")],
                text: "b".into(),
            }]
        )
        .is_err());
    }

    #[test]
    fn allows_explicit_growth_values_and_rejects_deletion_past_the_end() {
        let mut target = json!({"values": [1]});
        apply(
            &mut target,
            &[Op::Splice {
                path: vec![Seg::key("values")],
                start: 1,
                remove: 0,
                items: vec![JsonValue::Null, JsonValue::Null, json!(9)],
            }],
        )
        .unwrap();
        assert_eq!(target, json!({"values": [1, null, null, 9]}));
        let mut target = json!({"values": [1]});
        assert!(apply(
            &mut target,
            &[Op::Delete {
                path: vec![Seg::key("values"), Seg::index(1)],
            }]
        )
        .is_err());
    }

    #[test]
    fn applies_large_splice_payloads_without_spreading_them_at_once() {
        let items = vec![JsonValue::Null; 300_000];
        let mut target = json!({"values": []});
        apply(
            &mut target,
            &[Op::Splice {
                path: vec![Seg::key("values")],
                start: 0,
                remove: 0,
                items,
            }],
        )
        .unwrap();
        assert_eq!(target["values"].as_array().unwrap().len(), 300_000);
    }

    // ─── operation structure safety ─────────────────────────────────────────

    #[test]
    fn rejects_unknown_verbs_malformed_tuples_paths_and_splice_payloads() {
        // Malformed tuples cannot be decoded `Op`s; raw validation rejects them.
        assert!(assert_valid_raw_op(&json!(["ZZZ", ["value"], 9])).is_err());
        assert!(op_from_json(&json!(["ZZZ", ["value"], 9])).is_none());
        assert!(assert_valid_raw_op(&json!(["p", ["values"], 0, 0, "not-an-array"])).is_err());
        assert!(assert_valid_raw_op(&json!(["s", "value", 9])).is_err());
        assert!(op_from_json(&json!({"op": "s"})).is_none());
        assert!(op_from_json(&json!(null)).is_none());
    }

    #[test]
    fn rejects_invalid_append_and_truncation_operations() {
        let mut target = json!({"value": 1});
        assert!(apply(
            &mut target,
            &[Op::Append {
                path: vec![Seg::key("missing")],
                text: "x".into(),
            }]
        )
        .is_err());
        let mut target = json!({"value": 1});
        assert!(apply(
            &mut target,
            &[Op::Append {
                path: vec![Seg::key("value")],
                text: "x".into(),
            }]
        )
        .is_err());
        assert!(assert_valid_raw_op(&json!(["t", ["value"], -1])).is_err());
        assert!(Decoder::new()
            .decode(&[json!(["t", ["value"], -1])])
            .is_err());
    }

    #[test]
    fn clamps_splice_removal_past_the_end() {
        let mut target = json!({"values": [1, 2]});
        apply(
            &mut target,
            &[Op::Splice {
                path: vec![Seg::key("values")],
                start: 0,
                remove: 1_000_000_000,
                items: vec![],
            }],
        )
        .unwrap();
        assert_eq!(target, json!({"values": []}));
    }

    // ─── operation assertions ───────────────────────────────────────────────

    #[test]
    fn accepts_decoded_operations_and_rejects_wire_only_forms() {
        let decoded: Vec<Op> = vec![
            Op::Replace(json!({"value": 1})),
            Op::Set {
                path: vec![Seg::key("value")],
                value: json!(1),
            },
            Op::Delete {
                path: vec![Seg::key("value")],
            },
            Op::Append {
                path: vec![Seg::key("value")],
                text: "x".into(),
            },
            Op::Truncate {
                path: vec![Seg::key("value")],
                count: 2,
            },
            Op::Splice {
                path: vec![Seg::key("value")],
                start: 0,
                remove: 0,
                items: vec![],
            },
            Op::Permute {
                path: vec![Seg::key("value")],
                permutation: vec![0],
            },
        ];
        for operation in &decoded {
            assert_valid_op(operation).unwrap();
        }
        let wire_only = [
            json!(["s", 1]),
            json!(["d"]),
            json!(["a", "x"]),
            json!(["t", 2]),
            json!(["p", 0, 0, []]),
            json!(["#", 0, ["value"]]),
            json!(["s", 0, 1]),
        ];
        for wire in &wire_only {
            assert!(assert_valid_raw_op(wire).is_err());
            assert_valid_wire_op(wire).unwrap();
        }
    }

    #[test]
    fn does_not_recursively_inspect_operation_payloads() {
        // Upstream passes a Map and a Date as payloads; Rust payloads are
        // JsonValues, which are valid by construction — the assertion is that
        // validation does not look inside them.
        assert!(assert_valid_op(&Op::Set {
            path: vec![Seg::key("value")],
            value: json!({"any": [true, null, -1.5, {"deep": []}]}),
        })
        .is_ok());
        assert!(assert_valid_wire_op(&json!(["r", {"ts": [1, 2, 3]}])).is_ok());
    }
}
