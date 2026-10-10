//! Port of `packages/chord/src/delta/apply-immutable-trusted.ts`.
//!
//! Applies self-produced trusted operations to an immutable revision. Upstream
//! copies each touched container once, tracked through a `WeakSet` of owned
//! objects, and adopts untouched subtrees by reference. In Rust the input is a
//! shared [`JsonValue`], so the port deep-clones the root up front and mutates
//! the private clone in place — clone-on-write is unnecessary when nothing can
//! alias the clone. `Replace` swaps the root without re-cloning (the clone is
//! already private), matching upstream's `root = operation[1]` payload adoption.
//!
//! Error strings match the upstream `TypeError` messages exactly; they surface
//! as [`DeltaError::Invalid`] so `Display` carries the same text.
//!
//! One behavioral note: upstream `defineData` writes array indices directly and
//! would happily create sparse arrays for out-of-range indices. Trusted batches
//! come from the tracker and never do that; out-of-range writes reuse the
//! checked [`super::write_container_value`] guard so the impossible case fails
//! loudly instead of materializing a non-JSON value.

use super::{DeltaError, Op, Seg};
use crate::json::JsonValue;

/// Apply self-produced trusted operations, returning the new immutable value.
pub fn apply_immutable_trusted(
    target: &JsonValue,
    operations: &[Op],
) -> Result<JsonValue, DeltaError> {
    let mut root = target.clone();
    for operation in operations {
        match operation {
            Op::Replace(value) => {
                // Adopted, not copied: the clone is already private to this call.
                root = value.clone();
            }
            Op::Splice {
                path,
                start,
                remove,
                items,
            } => {
                let container = copy_path(&mut root, path, path.len())?;
                let Some(array) = container.as_array_mut() else {
                    return Err(trusted_error(
                        "Trusted array operation target is not an array",
                    ));
                };
                super::splice_array(array, *start, *remove, items);
            }
            Op::Permute { path, permutation } => {
                let container = copy_path(&mut root, path, path.len())?;
                let Some(array) = container.as_array_mut() else {
                    return Err(trusted_error(
                        "Trusted array operation target is not an array",
                    ));
                };
                if array.len() != permutation.len() {
                    return Err(trusted_error("Trusted permutation length mismatch"));
                }
                let previous = array.clone();
                for (index, source) in permutation.iter().enumerate() {
                    array[index] = previous[*source].clone();
                }
            }
            Op::Set { path, value } => {
                let (container, key) = resolve_parent_trusted(&mut root, path)?;
                define_data(container, &key, value.clone())?;
            }
            Op::Delete { path } => {
                let (container, key) = resolve_parent_trusted(&mut root, path)?;
                match (&mut *container, &key) {
                    (JsonValue::Array(array), Seg::Index(index)) => {
                        super::splice_array(array, *index, 1, &[]);
                    }
                    (JsonValue::Object(object), Seg::Key(key)) => {
                        object.remove(key);
                    }
                    // Upstream: string keys on arrays reach the numeric check and
                    // throw; everything else deletes nothing.
                    (JsonValue::Array(_), Seg::Key(_)) => {
                        return Err(trusted_error("Trusted array deletion key is not numeric"));
                    }
                    (JsonValue::Object(_), Seg::Index(_)) => {}
                    _ => return Err(trusted_error("Trusted operation path is not a container")),
                }
            }
            Op::Append { path, text } => {
                let (container, key) = resolve_parent_trusted(&mut root, path)?;
                let Some(JsonValue::String(existing)) = read(container, &key) else {
                    return Err(trusted_error("Trusted append target is not a string"));
                };
                let appended = format!("{existing}{text}");
                define_data(container, &key, JsonValue::String(appended))?;
            }
            Op::Truncate { path, count } => {
                let (container, key) = resolve_parent_trusted(&mut root, path)?;
                let Some(JsonValue::String(existing)) = read(container, &key) else {
                    return Err(trusted_error("Trusted truncate target is not a string"));
                };
                let truncated: String = existing.chars().skip(*count).collect();
                define_data(container, &key, JsonValue::String(truncated))?;
            }
        }
    }
    Ok(root)
}

fn trusted_error(message: &str) -> DeltaError {
    DeltaError::Invalid(message.to_string())
}

/// Upstream `copyPath`: walk `length` segments from the root, requiring a
/// container at every step. The deep clone replaces upstream's owned-copy
/// bookkeeping — every container is already private.
fn copy_path<'a>(
    root: &'a mut JsonValue,
    path: &[Seg],
    length: usize,
) -> Result<&'a mut JsonValue, DeltaError> {
    if !is_container(root) {
        return Err(trusted_error("Trusted operation root is not a container"));
    }
    let mut destination = root;
    for segment in path.iter().take(length) {
        let child_key = segment_for_lookup(destination, segment);
        let Some(child) = container_value(destination, &child_key) else {
            return Err(trusted_error("Trusted operation path is not a container"));
        };
        if !is_container(child) {
            return Err(trusted_error("Trusted operation path is not a container"));
        }
        let child_key = segment_for_lookup(destination, segment);
        let Some(nested) = container_value_mut(destination, &child_key) else {
            return Err(trusted_error("Trusted operation path is not a container"));
        };
        destination = nested;
    }
    Ok(destination)
}

/// Resolve the parent container of `path` and return it with the final segment.
fn resolve_parent_trusted<'a>(
    root: &'a mut JsonValue,
    path: &[Seg],
) -> Result<(&'a mut JsonValue, Seg), DeltaError> {
    let Some((key, parents)) = path.split_last() else {
        return Err(trusted_error("Trusted operation path is not a container"));
    };
    let container = copy_path(root, parents, parents.len())?;
    Ok((container, key.clone()))
}

/// Upstream `read`: `Seg` lookup with JavaScript key coercion.
fn read<'a>(target: &'a JsonValue, key: &Seg) -> Option<&'a JsonValue> {
    container_value(target, &segment_for_lookup(target, key))
}

/// Upstream `defineData`: an unconditional writable enumerable data write.
fn define_data(target: &mut JsonValue, key: &Seg, value: JsonValue) -> Result<(), DeltaError> {
    let key = segment_for_lookup(target, key);
    match (&mut *target, &key) {
        (JsonValue::Array(array), Seg::Index(index)) => {
            if *index > array.len() {
                // Unreachable for tracker-produced batches; see module docs.
                return Err(super::write_container_value(target, &key, value).unwrap_err());
            }
            super::write_container_value(target, &key, value)
        }
        (JsonValue::Object(_), _) => super::write_container_value(target, &key, value),
        _ => Err(trusted_error("Trusted operation path is not a container")),
    }
}

fn is_container(value: &JsonValue) -> bool {
    value.is_object() || value.is_array()
}

fn container_value<'a>(parent: &'a JsonValue, key: &Seg) -> Option<&'a JsonValue> {
    match (parent, key) {
        (JsonValue::Object(object), Seg::Key(key)) => object.get(key),
        (JsonValue::Array(array), Seg::Index(index)) => array.get(*index),
        _ => None,
    }
}

fn container_value_mut<'a>(parent: &'a mut JsonValue, key: &Seg) -> Option<&'a mut JsonValue> {
    match (parent, key) {
        (JsonValue::Object(object), Seg::Key(key)) => object.get_mut(key),
        (JsonValue::Array(array), Seg::Index(index)) => array.get_mut(*index),
        _ => None,
    }
}

/// JavaScript coerces array lookup keys to canonical numeric indices; objects
/// stringify numeric segments (`obj[5]` is property `"5"`).
fn segment_for_lookup(container: &JsonValue, segment: &Seg) -> Seg {
    match (container, segment) {
        (JsonValue::Array(_), Seg::Key(key)) => match key.parse::<usize>() {
            Ok(index) if *key == index.to_string() => Seg::Index(index),
            _ => Seg::Key(key.clone()),
        },
        (JsonValue::Object(_), Seg::Index(index)) => Seg::Key(index.to_string()),
        _ => segment.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{apply, apply_immutable, apply_immutable_batches, Tracker};
    use serde_json::json;

    fn freeze_deep(value: JsonValue) -> JsonValue {
        value
    }

    #[test]
    fn copies_each_touched_container_once_while_preserving_input_and_payload_ownership() {
        let shared = freeze_deep(json!({"nested": {"value": 1}}));
        let untouched = freeze_deep(json!({"value": 9}));
        let row_payload = freeze_deep(json!({"id": 4, "label": "placed"}));
        let base = freeze_deep(json!({
            "text": "abcdef",
            "stable": {"value": 7},
            "branch": {"value": 1},
            "copy": null,
            "placed": null,
            "untouched": null,
            "left": null,
            "right": null,
            "meta": {"count": 0, "obsolete": true},
            "rows": [
                {"id": 1, "label": "one"},
                {"id": 2, "label": "two"},
                {"id": 3, "label": "three"}
            ]
        }));
        let base_before = base.clone();
        let operations: Vec<Op> = vec![
            Op::Truncate {
                path: vec![Seg::key("text")],
                count: 2,
            },
            Op::Append {
                path: vec![Seg::key("text")],
                text: "!".into(),
            },
            Op::Set {
                path: vec![Seg::key("meta"), Seg::key("count")],
                value: json!(1),
            },
            Op::Set {
                path: vec![Seg::key("meta"), Seg::key("count")],
                value: json!(2),
            },
            Op::Delete {
                path: vec![Seg::key("meta"), Seg::key("obsolete")],
            },
            Op::Set {
                path: vec![Seg::key("copy")],
                value: json!({"value": 1}),
            },
            Op::Set {
                path: vec![Seg::key("copy"), Seg::key("value")],
                value: json!(2),
            },
            Op::Set {
                path: vec![Seg::key("placed")],
                value: shared.clone(),
            },
            Op::Set {
                path: vec![Seg::key("placed"), Seg::key("nested"), Seg::key("value")],
                value: json!(2),
            },
            Op::Set {
                path: vec![Seg::key("untouched")],
                value: untouched.clone(),
            },
            Op::Set {
                path: vec![Seg::key("left")],
                value: shared.clone(),
            },
            Op::Set {
                path: vec![Seg::key("right")],
                value: shared.clone(),
            },
            Op::Set {
                path: vec![Seg::key("left"), Seg::key("nested"), Seg::key("value")],
                value: json!(3),
            },
            Op::Splice {
                path: vec![Seg::key("rows")],
                start: 1,
                remove: 1,
                items: vec![row_payload.clone()],
            },
            Op::Set {
                path: vec![Seg::key("rows"), Seg::index(1), Seg::key("label")],
                value: json!("edited"),
            },
            Op::Permute {
                path: vec![Seg::key("rows")],
                permutation: vec![1, 0, 2],
            },
            Op::Set {
                path: vec![Seg::key("rows"), Seg::index(0), Seg::key("label")],
                value: json!("moved"),
            },
        ];

        let result = apply_immutable(&base, &operations).unwrap();
        // Upstream also runs the batch through the mutable applier on a clone
        // and requires identical output; identity/aliasing assertions are
        // pointer-based and have no Rust counterpart.
        let mutable_base = base.clone();
        let mutable_operations = operations.clone();
        let mutable_result = {
            let mut target = mutable_base;
            apply(&mut target, &mutable_operations).unwrap();
            target
        };

        assert_eq!(result, mutable_result);
        assert_eq!(result["text"], json!("cdef!"));
        assert_eq!(result["untouched"], untouched);
        assert_eq!(result["copy"], json!({"value": 2}));
        assert_ne!(result["copy"], base["branch"]);
        assert_eq!(result["placed"], json!({"nested": {"value": 2}}));
        assert_eq!(result["left"], json!({"nested": {"value": 3}}));
        // `right` was never traversed, so upstream shares `shared` by reference;
        // structurally it keeps the untouched payload content.
        assert_eq!(result["right"], shared);
        assert_eq!(result["rows"][0], json!({"id": 4, "label": "moved"}));
        assert_ne!(result["rows"][0], row_payload);
        assert_eq!(result["rows"][0], json!({"id": 4, "label": "moved"}));
        // Inputs are borrowed immutably, so the freeze/copy guarantees are
        // enforced by the type system; assert them anyway for regression cover.
        assert_eq!(base, base_before);
        assert_eq!(shared["nested"]["value"], json!(1));
        assert_eq!(row_payload["label"], json!("placed"));
        assert_eq!(result["stable"], json!({"value": 7}));
    }

    #[test]
    fn protects_root_replacement_payloads_before_later_object_and_array_edits() {
        let replacement = freeze_deep(json!({"nested": {"value": 1}, "values": [1, 2, 3]}));
        let replacement_before = replacement.clone();
        let batches: Vec<Vec<Op>> = vec![
            vec![Op::Replace(replacement.clone())],
            vec![Op::Set {
                path: vec![Seg::key("nested"), Seg::key("value")],
                value: json!(2),
            }],
            vec![
                Op::Splice {
                    path: vec![Seg::key("values")],
                    start: 1,
                    remove: 1,
                    items: vec![json!(4), json!(5)],
                },
                Op::Permute {
                    path: vec![Seg::key("values")],
                    permutation: vec![3, 0, 1, 2],
                },
            ],
        ];
        let batch_refs: Vec<&[Op]> = batches.iter().map(Vec::as_slice).collect();
        let result = apply_immutable_batches(&JsonValue::Null, batch_refs).unwrap();
        assert_eq!(
            result,
            json!({"nested": {"value": 2}, "values": [3, 1, 4, 5]})
        );
        assert_eq!(replacement, replacement_before);

        let array = freeze_deep(json!([1, 2, 3]));
        let array_before = array.clone();
        let array_result = apply_immutable(
            &array,
            &[
                Op::Splice {
                    path: vec![],
                    start: 1,
                    remove: 1,
                    items: vec![json!(4), json!(5)],
                },
                Op::Permute {
                    path: vec![],
                    permutation: vec![3, 0, 1, 2],
                },
                Op::Delete {
                    path: vec![Seg::index(1)],
                },
            ],
        )
        .unwrap();
        assert_eq!(array_result, json!([3, 4, 5]));
        assert_eq!(array, array_before);
    }

    #[test]
    fn shares_one_private_copy_on_write_scope_across_batch_partitions() {
        let base = freeze_deep(json!({
            "text": "abcdef",
            "meta": {"count": 0},
            "values": [
                {"id": 1, "value": 1},
                {"id": 2, "value": 2},
                {"id": 3, "value": 3}
            ]
        }));
        let batches: Vec<Vec<Op>> = vec![
            vec![
                Op::Set {
                    path: vec![Seg::key("meta"), Seg::key("count")],
                    value: json!(1),
                },
                Op::Splice {
                    path: vec![Seg::key("values")],
                    start: 1,
                    remove: 1,
                    items: vec![json!({"id": 4, "value": 4})],
                },
            ],
            vec![],
            vec![
                Op::Permute {
                    path: vec![Seg::key("values")],
                    permutation: vec![2, 0, 1],
                },
                Op::Set {
                    path: vec![Seg::key("values"), Seg::index(2), Seg::key("value")],
                    value: json!(40),
                },
            ],
            vec![
                Op::Truncate {
                    path: vec![Seg::key("text")],
                    count: 2,
                },
                Op::Append {
                    path: vec![Seg::key("text")],
                    text: "!".into(),
                },
            ],
        ];
        let batch_refs: Vec<&[Op]> = batches.iter().map(Vec::as_slice).collect();

        let intermediate = apply_immutable(&base, &batches[0]).unwrap();
        let intermediate_snapshot = intermediate.clone();
        let mut sequential = intermediate.clone();
        for batch in &batches[1..] {
            sequential = apply_immutable(&sequential, batch).unwrap();
        }
        let streamed = apply_immutable_batches(&base, batch_refs.clone()).unwrap();
        let flattened = apply_immutable(&base, &batches.concat()).unwrap();

        assert_eq!(streamed, sequential);
        assert_eq!(streamed, flattened);
        assert_eq!(intermediate, intermediate_snapshot);
        assert_eq!(base["meta"]["count"], json!(0));
        assert_eq!(base["values"][1]["id"], json!(2));
    }

    #[test]
    fn handles_tracker_produced_batches_across_arbitrary_revision_boundaries() {
        let initial = json!({
            "text": "start",
            "values": (0..8).map(|id| json!({"id": id, "score": 0})).collect::<Vec<_>>(),
            "revision": 0
        });
        let mut tracker = Tracker::new(initial.clone());
        let mut batches: Vec<Vec<Op>> = Vec::new();
        for revision in 1..=40u64 {
            let mut change = tracker.begin_change();
            let prepared = {
                let mut state = change.state();
                state.set(&[], "revision", json!(revision)).unwrap();
                let text = state
                    .get(&[Seg::key("text")])
                    .unwrap()
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                state
                    .set(&[], "text", json!(format!("{}{revision}", &text[1..])))
                    .unwrap();
                match revision % 5 {
                    0 => {
                        state.reverse(&[Seg::key("values")]).unwrap();
                    }
                    1 => {
                        state
                            .push(
                                &[Seg::key("values")],
                                vec![json!({"id": 100 + revision, "score": revision})],
                            )
                            .unwrap();
                    }
                    2 => {
                        state.shift(&[Seg::key("values")]).unwrap();
                    }
                    3 => {
                        let length = state.length(&[Seg::key("values")]).unwrap();
                        let index = (revision as usize) % length;
                        let mut element =
                            state.at(&[Seg::key("values"), Seg::index(index)]).unwrap();
                        element.set(&[], "score", json!(revision)).unwrap();
                    }
                    _ => {
                        state
                            .splice(
                                &[Seg::key("values")],
                                1,
                                Some(1),
                                vec![json!({"id": 200 + revision, "score": revision})],
                            )
                            .unwrap();
                    }
                }
                change.prepare().unwrap()
            };
            batches.push(prepared.ops.clone());
            tracker.adopt(prepared).unwrap();
        }

        let batch_refs: Vec<&[Op]> = batches.iter().map(Vec::as_slice).collect();
        let replayed = apply_immutable_batches(&initial, batch_refs).unwrap();
        assert_eq!(replayed, *tracker.value());
    }

    #[test]
    fn does_not_expose_partial_application_when_validation_fails() {
        let base = freeze_deep(json!({"nested": {"value": 1}}));
        let base_before = base.clone();
        let invalid_batches: Vec<Vec<Op>> = vec![
            vec![Op::Set {
                path: vec![Seg::key("nested"), Seg::key("value")],
                value: json!(2),
            }],
            vec![Op::Set {
                path: vec![
                    Seg::key("constructor"),
                    Seg::key("prototype"),
                    Seg::key("polluted"),
                ],
                value: json!(true),
            }],
        ];
        let batch_refs: Vec<&[Op]> = invalid_batches.iter().map(Vec::as_slice).collect();
        assert!(apply_immutable_batches(&base, batch_refs).is_err());
        assert_eq!(base, base_before);
        assert!(
            serde_json::from_str::<serde_json::Map<String, JsonValue>>("{}")
                .unwrap()
                .get("polluted")
                .is_none()
        );

        // Upstream also fails a batch stream whose iterator throws mid-iteration
        // and asserts generator-closure bookkeeping; Rust batch inputs are
        // borrowed slices with no failure mode, so those cases have no port.
        let string_segment = apply_immutable(
            &json!({"values": []}),
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::key("missing"), Seg::key("value")],
                value: json!(1),
            }],
        )
        .unwrap_err();
        assert!(string_segment.to_string().contains("unresolvable path"));
        let string_spelled_index = apply_immutable(
            &json!({"values": [{}]}),
            &[Op::Set {
                path: vec![Seg::key("values"), Seg::key("0"), Seg::key("value")],
                value: json!(1),
            }],
        )
        .unwrap_err();
        assert!(string_spelled_index
            .to_string()
            .contains("unresolvable path"));
    }

    #[test]
    fn allows_one_immutable_batch_to_fan_out_without_mutating_shared_payloads() {
        let payload = freeze_deep(json!({"nested": {"value": 1}}));
        let operations: Vec<Op> = vec![
            Op::Set {
                path: vec![Seg::key("placed")],
                value: payload.clone(),
            },
            Op::Set {
                path: vec![Seg::key("placed"), Seg::key("nested"), Seg::key("value")],
                value: json!(2),
            },
        ];
        let base = freeze_deep(json!({"placed": null}));
        let first = apply_immutable(&base, &operations).unwrap();
        let second = apply_immutable(&base, &operations).unwrap();
        assert_eq!(first, second);
        // `first !== second` and `first.placed !== second.placed` are pointer
        // assertions; each call returns a fresh value by construction here.
        assert_eq!(payload["nested"]["value"], json!(1));
    }

    #[test]
    fn copies_a_wide_object_once_rather_than_once_per_repeated_write() {
        let mut base = serde_json::Map::new();
        for index in 0..20_000 {
            base.insert(format!("field{index}"), json!(index));
        }
        let base = JsonValue::Object(base);
        let operations: Vec<Op> = (0..1_000)
            .map(|index| Op::Set {
                path: vec![Seg::key(format!("field{index}"))],
                value: json!(-(index as i64)),
            })
            .collect();
        let result = apply_immutable(&base, &operations).unwrap();
        // The `shallowCopies === 1` assertion instruments `Object.keys`; the
        // behavioral half of the case is that repeated writes to one wide object
        // produce the correct result without disturbing the base.
        assert_eq!(result["field999"], json!(-999));
        assert_eq!(base["field999"], json!(999));
    }

    #[test]
    fn trusted_apply_matches_the_checked_applier_for_tracker_batches() {
        // Direct coverage for this module's function (upstream exercises it
        // inside tracker materialization): identical results on well-formed
        // batches, distinct error strings on malformed ones.
        let base = json!({"text": "abcdef", "rows": [{"id": 1}, {"id": 2}], "meta": {"count": 0}});
        let operations: Vec<Op> = vec![
            Op::Truncate {
                path: vec![Seg::key("text")],
                count: 3,
            },
            Op::Append {
                path: vec![Seg::key("text")],
                text: "xyz".into(),
            },
            Op::Splice {
                path: vec![Seg::key("rows")],
                start: 0,
                remove: 1,
                items: vec![json!({"id": 9})],
            },
            Op::Permute {
                path: vec![Seg::key("rows")],
                permutation: vec![1, 0],
            },
            Op::Set {
                path: vec![Seg::key("meta"), Seg::key("count")],
                value: json!(5),
            },
            Op::Delete {
                path: vec![Seg::key("rows"), Seg::index(1)],
            },
        ];
        let trusted = apply_immutable_trusted(&base, &operations).unwrap();
        let checked = apply_immutable(&base, &operations).unwrap();
        assert_eq!(trusted, checked);
        assert_eq!(base["meta"]["count"], json!(0));

        let err = apply_immutable_trusted(
            &json!("scalar"),
            &[Op::Set {
                path: vec![Seg::key("value")],
                value: json!(1),
            }],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "Trusted operation root is not a container");

        let err = apply_immutable_trusted(
            &json!({"rows": {"not": "an array"}}),
            &[Op::Splice {
                path: vec![Seg::key("rows")],
                start: 0,
                remove: 0,
                items: vec![],
            }],
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Trusted array operation target is not an array"
        );

        let err = apply_immutable_trusted(
            &json!({"rows": []}),
            &[Op::Permute {
                path: vec![Seg::key("rows")],
                permutation: vec![0, 1],
            }],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "Trusted permutation length mismatch");

        let err = apply_immutable_trusted(
            &json!({"rows": [1]}),
            &[Op::Delete {
                path: vec![Seg::key("rows"), Seg::key("zero")],
            }],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "Trusted array deletion key is not numeric");

        let err = apply_immutable_trusted(
            &json!({"a": {"b": 1}}),
            &[Op::Set {
                path: vec![Seg::key("a"), Seg::key("b"), Seg::key("c")],
                value: json!(1),
            }],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "Trusted operation path is not a container");

        let err = apply_immutable_trusted(
            &json!({"text": 1}),
            &[Op::Append {
                path: vec![Seg::key("text")],
                text: "x".into(),
            }],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "Trusted append target is not a string");

        let err = apply_immutable_trusted(
            &json!({"text": 1}),
            &[Op::Truncate {
                path: vec![Seg::key("text")],
                count: 1,
            }],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "Trusted truncate target is not a string");
    }
}
