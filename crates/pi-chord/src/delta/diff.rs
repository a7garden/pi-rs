//! Port of `packages/chord/src/delta/diff.ts`.
//!
//! Computes a compact operation batch from two immutable JSON revisions.
//!
//! Deviations from upstream (mechanical, recorded per PORT_MAP.md rule 3):
//! - Upstream guards `MAX_DELTA_OPERATIONS` with a `WeakSet` keyed on the ops
//!   array (`overflowedBatches`). Reference identity of a `Vec` is meaningless
//!   in Rust, so the emitter is a small [`Emitter`] struct carrying the batch
//!   plus an overflow flag threaded through every emit site. Behavior is
//!   identical: once the batch overflows, further emissions are skipped and
//!   `diff_revisions` falls back to a root replacement.
//! - `sameValue` upstream is `left === right || equalJson(left, right)`.
//!   Pointer equality does not exist for [`JsonValue`]; `same_value` is
//!   structural equality only (`serde_json::Value` equality). Upstream reaches
//!   for the pointer check only as a fast path in front of `equalJson`, so the
//!   observable diff is unchanged.
//! - Maps keyed by `JsonValue` (`permutation`, `identityAnchors`) upstream key
//!   on object identity. They are keyed structurally here. For element values
//!   this yields a different — but equally valid — bijection only when a
//!   revision contains several structurally equal objects; every upstream test
//!   case produces the same operations.
//! - `semanticallyAligned`'s container-child check upstream asks whether the
//!   two revisions share a child container *by reference*. Rust has no
//!   references to compare, so the check asks whether an equally-keyed child is
//!   *structurally equal*; the LCS walk then resolves ties to the same matches
//!   upstream's pointer check produces (last compatible element wins).
//! - Costs count Unicode scalar values instead of UTF-16 code units, matching
//!   the [`crate::delta::overlap`] deviation.

use std::collections::HashMap;

use super::{Op, Seg};
use crate::json::JsonValue;

const DEFAULT_OVERLAP_SCAN: usize = 65_536;
const MAX_DELTA_OPERATIONS: usize = 4_096;

/// Operation sink for one batch: the ops plus the upstream `overflowedBatches`
/// WeakSet bit, which marks the batch as overflowed once it reaches
/// [`MAX_DELTA_OPERATIONS`] entries.
#[derive(Debug, Default)]
struct Emitter {
    operations: Vec<Op>,
    overflowed: bool,
}

impl Emitter {
    /// Upstream `emitOperation`: silently skip once the batch has overflowed;
    /// the push that would exceed the cap marks the overflow instead.
    fn emit(&mut self, operation: Op) {
        if self.overflowed {
            return;
        }
        if self.operations.len() >= MAX_DELTA_OPERATIONS {
            self.overflowed = true;
            return;
        }
        self.operations.push(operation);
    }

    fn overflowed(&self) -> bool {
        self.overflowed
    }
}

fn is_container(value: &JsonValue) -> bool {
    value.is_object() || value.is_array()
}

fn emit_set(path: &[Seg], value: &JsonValue, emitter: &mut Emitter) {
    if path.is_empty() {
        emitter.emit(Op::Replace(value.clone()));
    } else {
        emitter.emit(Op::Set {
            path: path.to_vec(),
            value: value.clone(),
        });
    }
}

/// Upstream `equalJson`: structural comparison with container-kind agreement.
/// `serde_json::Value` equality already means exactly this.
fn equal_json(left: &JsonValue, right: &JsonValue) -> bool {
    left == right
}

fn permutation(before: &[JsonValue], after: &[JsonValue]) -> Option<Vec<usize>> {
    if before.len() != after.len() {
        return None;
    }
    let mut positions: HashMap<&JsonValue, Vec<usize>> = HashMap::new();
    for (index, value) in before.iter().enumerate() {
        positions.entry(value).or_default().push(index);
    }
    let mut used: HashMap<&JsonValue, usize> = HashMap::new();
    let mut result = Vec::with_capacity(after.len());
    for value in after {
        let indices = positions.get(value)?;
        let used_at = used.entry(value).or_insert(0);
        if *used_at == indices.len() {
            return None;
        }
        result.push(indices[*used_at]);
        *used_at += 1;
    }
    Some(result)
}

fn emit_string(before: &str, after: &str, path: &[Seg], emitter: &mut Emitter) {
    if before == after {
        return;
    }
    if after.len() > before.len() && after.starts_with(before) {
        emitter.emit(Op::Append {
            path: path.to_vec(),
            text: after[before.len()..].to_string(),
        });
        return;
    }
    let shared = super::overlap(before, after, DEFAULT_OVERLAP_SCAN);
    if shared == 0 {
        emitter.emit(Op::Set {
            path: path.to_vec(),
            value: JsonValue::String(after.to_string()),
        });
        return;
    }
    let before_chars = before.chars().count();
    emitter.emit(Op::Truncate {
        path: path.to_vec(),
        count: before_chars - shared,
    });
    if shared < after.chars().count() {
        emitter.emit(Op::Append {
            path: path.to_vec(),
            text: after.chars().skip(shared).collect(),
        });
    }
}

/// `[beforeIndex, afterIndex]` element match between two array revisions.
type ArrayMatch = (usize, usize);

#[derive(Debug, Clone, Copy)]
struct MatchCandidate {
    before: usize,
    after: usize,
    previous: isize,
}

const MAX_IDENTITY_CANDIDATES: usize = 200_000;
const MAX_SEMANTIC_CELLS: usize = 65_536;

/// Upstream `sameValue`: pointer-or-equal, reduced to structural equality —
/// see the module deviation notes.
fn same_value(left: &JsonValue, right: &JsonValue) -> bool {
    left == right
}

fn lcs_matches(
    before: &[JsonValue],
    after: &[JsonValue],
    equal: fn(&JsonValue, &JsonValue) -> bool,
    max_cells: usize,
) -> Option<Vec<ArrayMatch>> {
    if before.is_empty() || after.is_empty() {
        return Some(Vec::new());
    }
    if before.len() * after.len() > max_cells {
        return None;
    }
    let width = after.len() + 1;
    let mut lengths = vec![0u32; (before.len() + 1) * width];
    for left in (0..before.len()).rev() {
        for right in (0..after.len()).rev() {
            let at = left * width + right;
            lengths[at] = if equal(&before[left], &after[right]) {
                lengths[(left + 1) * width + right + 1] + 1
            } else {
                lengths[(left + 1) * width + right].max(lengths[left * width + right + 1])
            };
        }
    }
    let mut matches = Vec::new();
    let mut left = 0;
    let mut right = 0;
    while left < before.len() && right < after.len() {
        if equal(&before[left], &after[right])
            && lengths[left * width + right] == lengths[(left + 1) * width + right + 1] + 1
        {
            matches.push((left, right));
            left += 1;
            right += 1;
        } else if lengths[(left + 1) * width + right] >= lengths[left * width + right + 1] {
            left += 1;
        } else {
            right += 1;
        }
    }
    Some(matches)
}

fn semantically_aligned(left: &JsonValue, right: &JsonValue) -> bool {
    if same_value(left, right) {
        return true;
    }
    // Upstream aligns a pair when the revisions share a child container BY
    // REFERENCE (`value === right[index]` / `value === rightObject[key]`).
    // Identity does not exist here, so the substitute requires the child
    // subtree to be structurally equal AND the remaining (scalar) content to
    // agree — otherwise unrelated rows whose child objects are coincidentally
    // equal (empty objects, in particular) would align where upstream's
    // per-object identity keeps them apart.
    match (left, right) {
        (JsonValue::Array(left_array), JsonValue::Array(right_array)) => {
            if left_array.len() != right_array.len() {
                return false;
            }
            let mut shared_child = false;
            for (value, other) in left_array.iter().zip(right_array) {
                if is_container(value) {
                    if same_value(value, other) {
                        shared_child = true;
                    }
                } else if is_container(other) || value != other {
                    return false;
                }
            }
            shared_child
        }
        (JsonValue::Object(left_object), JsonValue::Object(right_object)) => {
            let mut shared_child = false;
            for (key, value) in left_object {
                let Some(other) = right_object.get(key) else {
                    return false;
                };
                if is_container(value) {
                    if same_value(value, other) {
                        shared_child = true;
                    }
                } else if is_container(other) || value != other {
                    return false;
                }
            }
            shared_child
        }
        _ => false,
    }
}

fn lower_bound(values: &[usize], value: usize) -> usize {
    let mut low = 0;
    let mut high = values.len();
    while low < high {
        let middle = (low + high) / 2;
        if values[middle] < value {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn identity_subsequence(
    before: &[JsonValue],
    before_start: usize,
    before_end: usize,
    after: &[JsonValue],
    after_start: usize,
    after_end: usize,
) -> Option<Vec<ArrayMatch>> {
    let before_count = before_end - before_start;
    let after_count = after_end - after_start;
    let mut matches = Vec::new();
    if after_count < before_count {
        let mut before_index = before_start;
        for after_index in after_start..after_end {
            while before_index < before_end && !same_value(&before[before_index], &after[after_index]) {
                before_index += 1;
            }
            if before_index == before_end {
                return None;
            }
            matches.push((before_index, after_index));
            before_index += 1;
        }
        return Some(matches);
    }
    if before_count < after_count {
        let mut after_index = after_start;
        for before_index in before_start..before_end {
            while after_index < after_end && !same_value(&before[before_index], &after[after_index]) {
                after_index += 1;
            }
            if after_index == after_end {
                return None;
            }
            matches.push((before_index, after_index));
            after_index += 1;
        }
        return Some(matches);
    }
    None
}

fn greedy_identity_anchors(
    positions: &HashMap<&JsonValue, Vec<usize>>,
    after: &[JsonValue],
    after_start: usize,
    after_end: usize,
) -> Vec<ArrayMatch> {
    let mut matches = Vec::new();
    let mut previous: isize = -1;
    for after_index in after_start..after_end {
        let Some(candidates) = positions.get(&after[after_index]) else {
            continue;
        };
        let at = lower_bound(candidates, (previous + 1).max(0) as usize);
        let Some(before_index) = candidates.get(at) else {
            continue;
        };
        matches.push((*before_index, after_index));
        previous = *before_index as isize;
    }
    matches
}

fn identity_anchors(
    before: &[JsonValue],
    before_start: usize,
    before_end: usize,
    after: &[JsonValue],
    after_start: usize,
    after_end: usize,
) -> Vec<ArrayMatch> {
    let mut positions: HashMap<&JsonValue, Vec<usize>> = HashMap::new();
    for index in before_start..before_end {
        positions.entry(&before[index]).or_default().push(index);
    }
    let mut candidate_count = 0;
    for index in after_start..after_end {
        candidate_count += positions.get(&after[index]).map_or(0, Vec::len);
        if candidate_count > MAX_IDENTITY_CANDIDATES {
            return greedy_identity_anchors(&positions, after, after_start, after_end);
        }
    }
    if candidate_count == 0 {
        return Vec::new();
    }

    let mut candidates: Vec<MatchCandidate> = Vec::new();
    let mut tails: Vec<usize> = Vec::new();
    let mut tail_values: Vec<usize> = Vec::new();
    for after_index in after_start..after_end {
        let Some(before_positions) = positions.get(&after[after_index]) else {
            continue;
        };
        for &before_index in before_positions.iter().rev() {
            let at = lower_bound(&tail_values, before_index);
            let candidate_index = candidates.len();
            candidates.push(MatchCandidate {
                before: before_index,
                after: after_index,
                previous: if at == 0 { -1 } else { tails[at - 1] as isize },
            });
            if at == tails.len() {
                tails.push(candidate_index);
                tail_values.push(before_index);
            } else {
                tails[at] = candidate_index;
                tail_values[at] = before_index;
            }
        }
    }
    let mut matches = Vec::new();
    let mut candidate_index = tails.last().map_or(-1isize, |index| *index as isize);
    while candidate_index >= 0 {
        let candidate = &candidates[candidate_index as usize];
        matches.push((candidate.before, candidate.after));
        candidate_index = candidate.previous;
    }
    matches.reverse();
    matches
}

fn process_array_matches(
    before: &[JsonValue],
    after: &[JsonValue],
    path: &[Seg],
    emitter: &mut Emitter,
    before_start: usize,
    before_end: usize,
    after_start: usize,
    after_end: usize,
    output_start: usize,
    matches: &[ArrayMatch],
) {
    let mut before_at = before_start;
    let mut after_at = after_start;
    let mut output_at = output_start;
    for &(before_match, after_match) in matches {
        if emitter.overflowed() {
            return;
        }
        diff_array_region(
            before, after, path, emitter, before_at, before_match, after_at, after_match, output_at,
        );
        output_at += after_match - after_at;
        if !same_value(&before[before_match], &after[after_match]) {
            let mut child = path.to_vec();
            child.push(Seg::index(output_at));
            diff_value(&before[before_match], &after[after_match], &child, emitter);
        }
        output_at += 1;
        before_at = before_match + 1;
        after_at = after_match + 1;
    }
    if !emitter.overflowed() {
        diff_array_region(before, after, path, emitter, before_at, before_end, after_at, after_end, output_at);
    }
}

fn diff_array_region(
    before: &[JsonValue],
    after: &[JsonValue],
    path: &[Seg],
    emitter: &mut Emitter,
    before_start: usize,
    before_end: usize,
    after_start: usize,
    after_end: usize,
    output_start: usize,
) {
    if emitter.overflowed() {
        return;
    }
    let mut before_start = before_start;
    let mut before_end = before_end;
    let mut after_start = after_start;
    let mut after_end = after_end;
    let mut output_start = output_start;
    while before_start < before_end
        && after_start < after_end
        && same_value(&before[before_start], &after[after_start])
    {
        before_start += 1;
        after_start += 1;
        output_start += 1;
    }
    while before_start < before_end
        && after_start < after_end
        && same_value(&before[before_end - 1], &after[after_end - 1])
    {
        before_end -= 1;
        after_end -= 1;
    }
    let before_count = before_end - before_start;
    let after_count = after_end - after_start;
    if before_count == 0 && after_count == 0 {
        return;
    }
    if before_count == 0 || after_count == 0 {
        emitter.emit(Op::Splice {
            path: path.to_vec(),
            start: output_start,
            remove: before_count,
            items: after[after_start..after_end].to_vec(),
        });
        return;
    }

    if before_count == after_count {
        let positional: Vec<ArrayMatch> = (0..before_count)
            .filter(|&offset| same_value(&before[before_start + offset], &after[after_start + offset]))
            .map(|offset| (before_start + offset, after_start + offset))
            .collect();
        if !positional.is_empty() {
            process_array_matches(
                before,
                after,
                path,
                emitter,
                before_start,
                before_end,
                after_start,
                after_end,
                output_start,
                &positional,
            );
            return;
        }
    }

    if let Some(subsequence) = identity_subsequence(before, before_start, before_end, after, after_start, after_end) {
        if !subsequence.is_empty() {
            process_array_matches(
                before,
                after,
                path,
                emitter,
                before_start,
                before_end,
                after_start,
                after_end,
                output_start,
                &subsequence,
            );
            return;
        }
    }

    let identity = identity_anchors(before, before_start, before_end, after, after_start, after_end);
    if !identity.is_empty() {
        process_array_matches(
            before,
            after,
            path,
            emitter,
            before_start,
            before_end,
            after_start,
            after_end,
            output_start,
            &identity,
        );
        return;
    }

    let semantic = lcs_matches(
        &before[before_start..before_end],
        &after[after_start..after_end],
        semantically_aligned,
        MAX_SEMANTIC_CELLS,
    );
    if let Some(semantic) = semantic {
        if !semantic.is_empty() {
            let absolute: Vec<ArrayMatch> = semantic
                .into_iter()
                .map(|(before_index, after_index)| (before_start + before_index, after_start + after_index))
                .collect();
            process_array_matches(
                before,
                after,
                path,
                emitter,
                before_start,
                before_end,
                after_start,
                after_end,
                output_start,
                &absolute,
            );
            return;
        }
    }

    if before_count == 1 && after_count == 1 {
        let mut child = path.to_vec();
        child.push(Seg::index(output_start));
        diff_value(&before[before_start], &after[after_start], &child, emitter);
        return;
    }
    emitter.emit(Op::Splice {
        path: path.to_vec(),
        start: output_start,
        remove: before_count,
        items: after[after_start..after_end].to_vec(),
    });
}

fn diff_array(before: &[JsonValue], after: &[JsonValue], path: &[Seg], emitter: &mut Emitter) {
    if equal_json(&JsonValue::Array(before.to_vec()), &JsonValue::Array(after.to_vec())) {
        return;
    }
    if before.len() == after.len()
        && before.len() > 1
        && !same_value(&before[0], &after[0])
        && !same_value(&before[before.len() - 1], &after[after.len() - 1])
    {
        if let Some(order) = permutation(before, after) {
            emitter.emit(Op::Permute {
                path: path.to_vec(),
                permutation: order,
            });
            return;
        }
    }
    diff_array_region(before, after, path, emitter, 0, before.len(), 0, after.len(), 0);
}

fn diff_object(
    before: &serde_json::Map<String, JsonValue>,
    after: &serde_json::Map<String, JsonValue>,
    path: &[Seg],
    emitter: &mut Emitter,
) {
    let before_keys: Vec<&String> = before.keys().collect();
    let after_keys: Vec<&String> = after.keys().collect();
    if before_keys
        .iter()
        .chain(after_keys.iter())
        .any(|key| super::is_reserved_segment(key))
    {
        if !equal_json(&JsonValue::Object(before.clone()), &JsonValue::Object(after.clone())) {
            emit_set(path, &JsonValue::Object(after.clone()), emitter);
        }
        return;
    }
    for key in &after_keys {
        if emitter.overflowed() {
            return;
        }
        let mut child = path.to_vec();
        child.push(Seg::key((*key).clone()));
        if let Some(before_value) = before.get(*key) {
            diff_value(before_value, &after[*key], &child, emitter);
        } else {
            emit_set(&child, &after[*key], emitter);
        }
    }
    for key in &before_keys {
        if emitter.overflowed() {
            return;
        }
        if !after.contains_key(*key) {
            let mut child = path.to_vec();
            child.push(Seg::key((*key).clone()));
            emitter.emit(Op::Delete { path: child });
        }
    }
}

fn diff_value(before: &JsonValue, after: &JsonValue, path: &[Seg], emitter: &mut Emitter) {
    if same_value(before, after) || emitter.overflowed() {
        return;
    }
    if let (Some(before_text), Some(after_text)) = (before.as_str(), after.as_str()) {
        if !path.is_empty() {
            emit_string(before_text, after_text, path, emitter);
            return;
        }
    }
    if let (Some(before_array), Some(after_array)) = (before.as_array(), after.as_array()) {
        diff_array(before_array, after_array, path, emitter);
        return;
    }
    if let (Some(before_object), Some(after_object)) = (before.as_object(), after.as_object()) {
        diff_object(before_object, after_object, path, emitter);
        return;
    }
    emit_set(path, after, emitter);
}

fn json_cost(value: &JsonValue) -> usize {
    match value {
        JsonValue::Null => 4,
        JsonValue::String(text) => text.chars().count() + 2,
        JsonValue::Number(number) => number.to_string().len(),
        JsonValue::Bool(true) => 4,
        JsonValue::Bool(false) => 5,
        JsonValue::Array(items) => {
            let mut cost = 2;
            for (index, item) in items.iter().enumerate() {
                cost += json_cost(item) + if index == 0 { 0 } else { 1 };
            }
            cost
        }
        JsonValue::Object(object) => {
            let mut cost = 2;
            for (index, (key, value)) in object.iter().enumerate() {
                cost += key.chars().count() + 3 + json_cost(value) + if index == 0 { 0 } else { 1 };
            }
            cost
        }
    }
}

fn segment_cost(segment: &Seg) -> usize {
    match segment {
        Seg::Key(key) => key.chars().count() + 2,
        Seg::Index(index) => index.to_string().len(),
    }
}

fn path_cost(path: &[Seg]) -> usize {
    let mut cost = 2;
    for (index, segment) in path.iter().enumerate() {
        cost += segment_cost(segment) + if index == 0 { 0 } else { 1 };
    }
    cost
}

fn number_cost(value: usize) -> usize {
    value.to_string().len()
}

fn permutation_json_cost(permutation: &[usize]) -> usize {
    let mut cost = 2;
    for (index, value) in permutation.iter().enumerate() {
        cost += number_cost(*value) + if index == 0 { 0 } else { 1 };
    }
    cost
}

fn operation_cost(operation: &Op) -> usize {
    match operation {
        Op::Replace(value) => 6 + json_cost(value),
        Op::Set { path, value } => 7 + path_cost(path) + json_cost(value),
        Op::Delete { path } => 6 + path_cost(path),
        Op::Append { path, text } => 7 + path_cost(path) + text.chars().count() + 2,
        Op::Truncate { path, count } => 7 + path_cost(path) + number_cost(*count),
        Op::Splice {
            path,
            start,
            remove,
            items,
        } => 10 + path_cost(path) + number_cost(*start) + number_cost(*remove) + json_cost(&JsonValue::Array(items.clone())),
        Op::Permute { path, permutation } => {
            7 + path_cost(path) + permutation_json_cost(permutation)
        }
    }
}

/// Compute a compact operation batch from two immutable JSON revisions.
pub fn diff_revisions(before: &JsonValue, after: &JsonValue) -> Vec<Op> {
    let mut emitter = Emitter::default();
    diff_value(before, after, &[], &mut emitter);
    if emitter.overflowed() {
        return vec![Op::Replace(after.clone())];
    }
    let operations = emitter.operations;
    if operations.is_empty() || matches!(operations.first(), Some(Op::Replace(_))) {
        return operations;
    }
    let mut delta_cost = 2;
    for operation in &operations {
        delta_cost += operation_cost(operation) + 1;
    }
    if delta_cost < 65_536 {
        return operations;
    }
    let snapshot_cost = json_cost(after) + 6;
    if delta_cost >= snapshot_cost {
        vec![Op::Replace(after.clone())]
    } else {
        operations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{apply_immutable, assert_valid_op, assert_valid_wire_op, op_from_json, Decoder, Encoder};
    use serde_json::json;

    fn expect_diff(before: &JsonValue, after: &JsonValue, expected: JsonValue) {
        let operations = diff_revisions(before, after);
        assert_eq!(
            serde_json::to_value(&operations).unwrap(),
            expected,
            "ops for {before} -> {after}"
        );
        let replay = apply_immutable(before, &operations).unwrap();
        assert_eq!(replay, *after, "replay of ops for {before} -> {after}");
    }

    #[test]
    fn emits_sets_and_deletes() {
        expect_diff(
            &json!({"keep": 1, "change": 1, "remove": true}),
            &json!({"keep": 1, "change": 2, "add": 3}),
            json!([
                ["s", ["change"], 2],
                ["s", ["add"], 3],
                ["d", ["remove"]]
            ]),
        );
    }

    #[test]
    fn emits_string_append_and_front_truncation() {
        expect_diff(&json!({"text": "hello"}), &json!({"text": "hello world"}), json!([["a", ["text"], " world"]]));
        expect_diff(&json!({"text": "hello world"}), &json!({"text": "world"}), json!([["t", ["text"], 6]]));
        expect_diff(
            &json!({"text": "abcdefgh"}),
            &json!({"text": "defghxyz"}),
            json!([["t", ["text"], 3], ["a", ["text"], "xyz"]]),
        );
    }

    #[test]
    fn represents_array_insertion_removal_and_shift_with_splices() {
        let a = json!({"id": "a"});
        let b = json!({"id": "b"});
        let c = json!({"id": "c"});
        expect_diff(
            &json!({"values": [a, b]}),
            &json!({"values": [a, c, b]}),
            json!([["p", ["values"], 1, 0, [c]]]),
        );
        expect_diff(
            &json!({"values": [a, b, c]}),
            &json!({"values": [b, c]}),
            json!([["p", ["values"], 0, 1, []]]),
        );
    }

    #[test]
    fn collapses_a_same_length_queue_update_to_two_splices() {
        let a = json!({"id": "a"});
        let b = json!({"id": "b"});
        let c = json!({"id": "c"});
        let d = json!({"id": "d"});
        expect_diff(
            &json!({"values": [a, b, c]}),
            &json!({"values": [b, c, d]}),
            json!([["p", ["values"], 0, 1, []], ["p", ["values"], 2, 0, [d]]]),
        );
    }

    #[test]
    fn emits_a_permutation_for_a_pure_reorder() {
        let a = json!({"id": "a"});
        let b = json!({"id": "b"});
        let c = json!({"id": "c"});
        expect_diff(
            &json!({"values": [a, b, c]}),
            &json!({"values": [c, a, b]}),
            json!([["m", ["values"], [2, 0, 1]]]),
        );
    }

    #[test]
    fn normalizes_reordered_distinct_deeply_equal_objects_to_a_no_op() {
        let first = json!({"nested": {"value": 1}});
        let second = json!({"nested": {"value": 1}});
        assert_ne!(&first as *const JsonValue, &second as *const JsonValue);
        assert_eq!(
            diff_revisions(&json!({"values": [first, second]}), &json!({"values": [second, first]})),
            Vec::<Op>::new()
        );
    }

    #[test]
    fn validates_and_encodes_permutations() {
        let operations = vec![
            Op::Permute {
                path: vec![Seg::key("values")],
                permutation: vec![2, 0, 1],
            },
            Op::Permute {
                path: vec![Seg::key("values")],
                permutation: vec![1, 2, 0],
            },
        ];
        for operation in &operations {
            assert_valid_op(operation).unwrap();
        }
        let wire = Encoder::new().encode(&operations);
        assert_eq!(
            serde_json::to_value(&wire).unwrap(),
            json!([
                ["m", ["values"], [2, 0, 1]],
                ["m", [1, 2, 0]]
            ])
        );
        for operation in &wire {
            assert_valid_wire_op(operation).unwrap();
        }
        assert_eq!(Decoder::new().decode(&wire).unwrap(), operations);
        assert!(assert_valid_op(&Op::Permute {
            path: vec![Seg::key("values")],
            permutation: vec![0, 0],
        })
        .unwrap_err()
        .to_string()
        .contains("bijection"));
    }

    #[test]
    fn emits_nothing_for_deeply_equal_reconstructed_values() {
        assert_eq!(
            diff_revisions(&json!({"value": {"nested": [1, 2]}}), &json!({"value": {"nested": [1, 2]}})),
            Vec::<Op>::new()
        );
        assert_eq!(
            diff_revisions(
                &json!({"values": [{"id": 1}, {"id": 2}]}),
                &json!({"values": [{"id": 1}, {"id": 2}]})
            ),
            Vec::<Op>::new()
        );
        assert_eq!(
            diff_revisions(&json!({"values": [true, true, true]}), &json!({"values": [true, true, true]})),
            Vec::<Op>::new()
        );
    }

    #[test]
    fn keeps_a_leaf_edit_inside_a_reconstructed_array_narrow() {
        expect_diff(
            &json!({"values": [
                {"id": 1, "label": "one"},
                {"id": 2, "label": "two"}
            ]}),
            &json!({"values": [
                {"id": 1, "label": "one"},
                {"id": 2, "label": "changed"}
            ]}),
            json!([["s", ["values", 1, "label"], "changed"]]),
        );
    }

    #[test]
    fn emits_payload_free_splices_for_scattered_removals() {
        let values: Vec<JsonValue> = ["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|id| json!({"id": id}))
            .collect();
        expect_diff(
            &json!({"values": values}),
            &json!({"values": [values[0].clone(), values[2].clone(), values[4].clone()]}),
            json!([["p", ["values"], 1, 1, []], ["p", ["values"], 2, 1, []]]),
        );
    }

    #[test]
    fn encodes_removal_positions_canonically() {
        let cases: Vec<(&str, JsonValue, JsonValue, JsonValue)> = vec![
            ("front", json!([1, 2, 3, 4]), json!([2, 3, 4]), json!([["p", ["values"], 0, 1, []]])),
            ("tail", json!([1, 2, 3, 4]), json!([1, 2, 3]), json!([["p", ["values"], 3, 1, []]])),
            ("middle", json!([1, 2, 3, 4]), json!([1, 3, 4]), json!([["p", ["values"], 1, 1, []]])),
            ("all", json!([1, 2, 3, 4]), json!([]), json!([["p", ["values"], 0, 4, []]])),
            ("none", json!([1, 2, 3, 4]), json!([1, 2, 3, 4]), json!([])),
        ];
        for (name, before, after, expected) in cases {
            expect_diff(&json!({"values": before}), &json!({"values": after}), expected, );
            let _ = name;
        }
    }

    #[test]
    fn does_not_field_diff_unrelated_shifted_objects_with_common_fields() {
        let before: Vec<JsonValue> = [1, 2, 3].iter().map(|value| json!({"type": "row", "value": value})).collect();
        let after: Vec<JsonValue> = [2, 3, 4].iter().map(|value| json!({"type": "row", "value": value})).collect();
        expect_diff(
            &json!({"values": before}),
            &json!({"values": after}),
            json!([
                ["p", ["values"], 0, 1, []],
                ["p", ["values"], 2, 0, [{"type": "row", "value": 4}]]
            ]),
        );
    }

    #[test]
    fn does_not_treat_coincidental_id_or_key_fields_as_structural_identity() {
        let left = json!({"value": "left"});
        let right = json!({"value": "right"});
        let before = json!({"values": [
            left,
            {"id": 1, "key": "a", "value": "first"},
            {"id": 2, "key": "b", "value": "second"},
            right
        ]});
        let replacements = json!([
            {"id": 2, "key": "b", "value": "edited-second"},
            {"id": 1, "key": "a", "value": "edited-first"}
        ]);
        let after = json!({"values": [json!({"value": "left"}), replacements[0].clone(), replacements[1].clone(), json!({"value": "right"})]});
        expect_diff(&before, &after, json!([["p", ["values"], 1, 2, replacements]]));
    }

    #[test]
    fn combines_removals_append_and_a_shared_subtree_survivor_edit() {
        let values = vec![
            json!({"id": "a", "stable": {}, "detail": {"text": "a"}}),
            json!({"id": "b", "stable": {}, "detail": {"text": "b"}}),
            json!({"id": "c", "stable": {}, "detail": {"text": "c"}}),
            json!({"id": "d", "stable": {}, "detail": {"text": "d"}}),
        ];
        let appended = json!({"id": "e", "stable": {}, "detail": {"text": "e"}});
        // Upstream aliases `changedC.stable` to `c.stable`, which the identity
        // child-check in `semanticallyAligned` sees. Identity does not exist in
        // Rust, so the check requires a structurally equal child subtree PLUS
        // scalar agreement — which selects `c` (matching `id`) and keeps `b`,
        // whose child objects are only coincidentally equal, unaligned.
        let changed_c = json!({"id": "c", "stable": {}, "detail": {"text": "changed"}});
        expect_diff(
            &json!({"values": values}),
            &json!({"values": [values[0].clone(), changed_c, values[3].clone(), appended]}),
            json!([
                ["p", ["values"], 1, 1, []],
                ["a", ["values", 1, "detail", "text"], "hanged"],
                ["p", ["values"], 3, 0, [{"id": "e", "stable": {}, "detail": {"text": "e"}}]]
            ]),
        );
    }

    #[test]
    fn does_not_retain_removed_neighbor_payloads() {
        for size in [256 * 1024, 1024 * 1024] {
            let payload = "x".repeat(size);
            let retained: Vec<JsonValue> = ["a", "b", "c", "d", "e"]
                .into_iter()
                .map(|id| json!({"id": id, "payload": payload}))
                .collect();
            let before = json!({"values": retained});
            let after = json!({"values": [retained[0].clone(), retained[2].clone(), retained[4].clone()]});
            let operations = diff_revisions(&before, &after);
            assert_eq!(
                serde_json::to_value(&operations).unwrap(),
                json!([["p", ["values"], 1, 1, []], ["p", ["values"], 2, 1, []]]),
                "size {size}"
            );
            assert!(serde_json::to_string(&operations).unwrap().len() < 100);
            assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
        }
    }

    #[test]
    fn keeps_push_pop_and_middle_removal_narrow() {
        for size in [1_001usize, 10_000] {
            let values: Vec<JsonValue> = (0..size).map(|value| json!({"value": value})).collect();
            let appended = json!({"value": size});
            let mut with_append = values.clone();
            with_append.push(appended.clone());
            expect_diff(
                &json!({"values": values}),
                &json!({"values": with_append}),
                json!([["p", ["values"], size, 0, [appended]]]),
            );
            let mut without_last = values.clone();
            without_last.pop();
            expect_diff(
                &json!({"values": values}),
                &json!({"values": without_last}),
                json!([["p", ["values"], size - 1, 1, []]]),
            );
            let middle = size / 2;
            let mut without_middle = values.clone();
            without_middle.remove(middle);
            expect_diff(
                &json!({"values": values}),
                &json!({"values": without_middle}),
                json!([["p", ["values"], middle, 1, []]]),
            );
        }
    }

    #[test]
    fn keeps_forty_thousand_row_sparse_edits_narrow() {
        let values: Vec<JsonValue> = (0..40_000)
            .map(|value| json!({"value": value, "stable": {"value": value}}))
            .collect();
        let mut after = values.clone();
        let mut changed = Vec::new();
        let mut index = 100;
        while index < after.len() {
            after[index] = json!({"value": -(index as i64), "stable": {"value": index}});
            changed.push(index);
            index += 400;
        }
        let operations = diff_revisions(&json!({"values": values}), &json!({"values": after}));
        let expected: Vec<JsonValue> = changed
            .iter()
            .map(|index| json!(["s", ["values", index, "value"], -( *index as i64)]))
            .collect();
        assert_eq!(serde_json::to_value(&operations).unwrap(), JsonValue::Array(expected));
        assert!(serde_json::to_string(&operations).unwrap().len() < 7_500);
        let replay = apply_immutable(&json!({"values": values}), &operations).unwrap();
        assert_eq!(replay, json!({"values": after}));
    }

    #[test]
    fn keeps_a_reconstructed_large_array_leaf_edit_narrow() {
        let before: Vec<JsonValue> = (0..1_000)
            .map(|value| json!({"value": value, "label": format!("row-{value}")}))
            .collect();
        let mut after = before.clone();
        after[700]["label"] = json!("changed");
        let operations = diff_revisions(&json!({"values": before}), &json!({"values": after}));
        assert_eq!(
            serde_json::to_value(&operations).unwrap(),
            json!([["s", ["values", 700, "label"], "changed"]])
        );
        assert!(serde_json::to_string(&operations).unwrap().len() < 100);
        let replay = apply_immutable(&json!({"values": before}), &operations).unwrap();
        assert_eq!(replay, json!({"values": after}));
    }

    #[test]
    fn splices_an_ambiguous_equal_count_moved_and_edited_gap() {
        let before = json!({"values": [
            {"value": "left"},
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"value": "right"}
        ]});
        let replacements = json!([
            {"id": 2, "value": "edited"},
            {"id": 1, "value": "also-edited"}
        ]);
        let after = json!({"values": [
            {"value": "left"},
            replacements[0],
            replacements[1],
            {"value": "right"}
        ]});
        expect_diff(&before, &after, json!([["p", ["values"], 1, 2, replacements]]));
    }

    #[test]
    fn keeps_a_large_rotation_payload_free() {
        let values: Vec<JsonValue> = (0..10_000).map(|value| json!({"value": value})).collect();
        let rotated: Vec<JsonValue> = values[1_000..]
            .iter()
            .chain(values[..1_000].iter())
            .cloned()
            .collect();
        let operations = diff_revisions(&json!({"values": values}), &json!({"values": rotated}));
        assert_eq!(operations.len(), 1);
        assert!(matches!(operations[0], Op::Permute { .. }));
        assert!(serde_json::to_string(&operations).unwrap().len() < 60_000);
        let replay = apply_immutable(&json!({"values": values}), &operations).unwrap();
        assert_eq!(replay, json!({"values": rotated}));
    }

    #[test]
    fn encodes_five_hundred_unshifts_without_snapshotting_retained_rows() {
        let retained: Vec<JsonValue> = (0..10_000)
            .map(|value| json!({"value": value, "payload": "x".repeat(100)}))
            .collect();
        let inserted: Vec<JsonValue> = (0..500).map(|value| json!({"value": -(value as i64) - 1})).collect();
        let after: Vec<JsonValue> = inserted.iter().chain(retained.iter()).cloned().collect();
        let operations = diff_revisions(&json!({"values": retained}), &json!({"values": after}));
        assert_eq!(
            serde_json::to_value(&operations).unwrap(),
            json!([["p", ["values"], 0, 0, inserted]])
        );
        assert!(serde_json::to_string(&operations).unwrap().len() < 20_000);
    }

    #[test]
    fn bounds_wide_object_operation_emission_with_a_root_replacement() {
        let mut before = serde_json::Map::new();
        let mut after = serde_json::Map::new();
        for index in 0..20_000 {
            before.insert(format!("field{index}"), json!(0));
            after.insert(format!("field{index}"), json!(1));
        }
        assert_eq!(
            diff_revisions(&JsonValue::Object(before), &JsonValue::Object(after.clone())),
            vec![Op::Replace(JsonValue::Object(after))]
        );
    }

    #[test]
    fn bounds_a_wide_normalized_array_fallback_with_a_root_replacement() {
        let before = json!({"values": Vec::<JsonValue>::from_iter((0..40_000).map(|_| json!(0)))});
        let after = json!({"values": Vec::<JsonValue>::from_iter((0..40_000).map(|_| json!(1)))});
        let operations = diff_revisions(&before, &after);
        assert_eq!(operations, vec![Op::Replace(after.clone())]);
        assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
    }

    #[test]
    fn decode_rejects_unbuildable_tuples_from_the_structure_safety_table() {
        // Upstream builds malformed tuples directly; Rust reaches the same
        // validators through `op_from_json`/`assert_valid_raw_op`.
        for tuple in [
            json!(["ZZZ", ["value"], 9]),
            json!(["p", ["values"], 0, 0, "not-an-array"]),
            json!(["s", "value", 9]),
            json!({"op": "s"}),
            json!(null),
            json!(["t", ["value"], -1]),
        ] {
            assert!(op_from_json(&tuple).is_none() || assert_valid_wire_op(&tuple).is_err());
            assert!(assert_valid_raw_op_shim(&tuple).is_err());
        }
    }

    fn assert_valid_raw_op_shim(tuple: &JsonValue) -> Result<(), crate::delta::DeltaError> {
        crate::delta::assert_valid_raw_op(tuple)
    }
}

