//! Port of `packages/chord/src/delta/tracker.ts` — lifecycle half.
//!
//! [`Tracker`] owns one immutable revision; [`Change`] is a transaction-scoped
//! overlay (see [`super::draft`]) that [`Tracker::adopt`] commits after
//! [`Change::prepare`] materializes the candidate. Upstream tracks contexts in
//! a `WeakRef` registry so abandoned drafts can be garbage collected and
//! competing changes can be invalidated eagerly; Rust ownership does both jobs
//! statically: a `Change` mutably borrows its tracker (one open change at a
//! time, nothing to prune) and a consumed `Prepared` moves into
//! [`Tracker::adopt`], where staleness falls out of the base-revision and
//! structural base checks.
//!
//! Error strings match upstream exactly (`Prepared change is stale`, `Prepared
//! change has already been used`, `Prepared change has been aborted`,
//! `Prepared change is not ready`, `Prepared change belongs to a different
//! tracker`). "Prepared change is not ready" is unreachable through the public
//! API — every `Prepared` is born prepared — and is kept for parity.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};

use super::draft::{Draft, OverlayContext};
use super::{apply, DeltaError, Op};
use crate::json::JsonValue;

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Upstream `Status` restricted to the states a `Prepared` can observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedStatus {
    Ready = 0,
    Aborted = 1,
    Consumed = 2,
    Stale = 3,
}

impl PreparedStatus {
    fn from_raw(raw: usize) -> PreparedStatus {
        match raw {
            1 => PreparedStatus::Aborted,
            2 => PreparedStatus::Consumed,
            3 => PreparedStatus::Stale,
            _ => PreparedStatus::Ready,
        }
    }
}

/// A materialized candidate revision awaiting [`Tracker::adopt`].
///
/// Clones share one status cell (upstream `Prepared` instances share their
/// context's `StatusCell`): adopting any clone consumes every clone.
#[derive(Debug, Clone)]
pub struct Prepared {
    owner: u64,
    /// The revision the change was opened against.
    pub base: JsonValue,
    /// The candidate revision.
    pub value: JsonValue,
    /// Operation batch replaying `base` into `value`.
    pub ops: Vec<Op>,
    pub base_revision: u64,
    status: std::sync::Arc<AtomicUsize>,
}

impl Prepared {
    fn status(&self) -> PreparedStatus {
        PreparedStatus::from_raw(self.status.load(AtomicOrdering::Acquire))
    }

    fn set_status(&self, status: PreparedStatus) {
        self.status.store(status as usize, AtomicOrdering::Release);
    }

    /// Invalidate this candidate. Adopting afterwards fails with `Prepared
    /// change has been aborted`.
    pub fn abort(&mut self) {
        // Upstream `abortContext` ignores terminal statuses.
        if self.status() == PreparedStatus::Ready {
            self.set_status(PreparedStatus::Aborted);
        }
    }

    fn error(&self) -> Option<DeltaError> {
        match self.status() {
            PreparedStatus::Ready => None,
            PreparedStatus::Consumed => {
                Some(DeltaError::invalid("Prepared change has already been used"))
            }
            PreparedStatus::Aborted => {
                Some(DeltaError::invalid("Prepared change has been aborted"))
            }
            PreparedStatus::Stale => Some(DeltaError::invalid("Prepared change is stale")),
        }
    }
}

/// One open transaction. `prepare`/`abort` consume the change, which releases
/// the tracker borrow; accessing a settled overlay ("Cannot use a settled
/// overlay") is prevented by the borrow checker instead of a runtime check.
pub struct Change<'t> {
    context: OverlayContext,
    _tracker: &'t mut Tracker,
}

impl<'t> Change<'t> {
    /// A mutable draft view. Repeated calls reborrow, mirroring holding
    /// `change.state` across statements upstream.
    pub fn state(&mut self) -> Draft<'_> {
        Draft::new(&mut self.context)
    }

    /// Emit ops and materialize the candidate value. `value` is `base.clone()`
    /// with the ops applied — the same result upstream's
    /// `cloneNode`/`materializeOperations` strategies produce.
    pub fn prepare(self) -> Result<Prepared, DeltaError> {
        let Self { context, .. } = self;
        let owner = context.owner;
        let base_revision = context.base_revision;
        let ops = if context.replacement || context.dirty.is_empty() {
            Vec::new()
        } else {
            context.emit_operations()
        };
        let (value, base) = if ops.is_empty() {
            let base = context.base_root;
            let value = base.clone();
            (value, base)
        } else {
            let mut value = context.base_root.clone();
            apply(&mut value, &ops)?;
            (value, context.base_root)
        };
        Ok(Prepared {
            owner,
            base,
            value,
            ops,
            base_revision,
            status: std::sync::Arc::new(AtomicUsize::new(PreparedStatus::Ready as usize)),
        })
    }

    /// Discard the transaction without touching the committed value.
    pub fn abort(self) {}
}

/// Immutable revision tracker (upstream `Tracker<T>` over a strict-JSON root).
pub struct Tracker {
    owner: u64,
    value: JsonValue,
    revision: u64,
}

impl Tracker {
    pub fn new(initial: JsonValue) -> Tracker {
        Tracker {
            owner: NEXT_OWNER.fetch_add(1, AtomicOrdering::Relaxed),
            value: initial,
            revision: 0,
        }
    }

    /// The committed revision.
    pub fn value(&self) -> &JsonValue {
        &self.value
    }

    /// Number of adopted revisions.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn begin_change(&mut self) -> Change<'_> {
        let context = OverlayContext::new(
            self.owner,
            self.value.clone(),
            self.value.clone(),
            self.revision,
            false,
        );
        Change {
            context,
            _tracker: self,
        }
    }

    /// Prepare a whole-value replacement. Structurally equal replacements
    /// normalize to an empty batch (`replacementNoop`).
    pub fn prepare_replace(&mut self, value: JsonValue) -> Result<Prepared, DeltaError> {
        let replacement_noop = value == self.value;
        let ops = if replacement_noop {
            Vec::new()
        } else {
            vec![Op::Replace(value.clone())]
        };
        let base = self.value.clone();
        let prepared_value = if replacement_noop {
            base.clone()
        } else {
            value
        };
        Ok(Prepared {
            owner: self.owner,
            base,
            value: prepared_value,
            ops,
            base_revision: self.revision,
            status: std::sync::Arc::new(AtomicUsize::new(PreparedStatus::Ready as usize)),
        })
    }

    /// Commit a prepared candidate. Infallible after the checks: adoption is an
    /// ownership move, so storage failure can discard the candidate without
    /// touching authority.
    pub fn adopt(&mut self, prepared: Prepared) -> Result<(), DeltaError> {
        if prepared.owner != self.owner {
            return Err(DeltaError::invalid(
                "Prepared change belongs to a different tracker",
            ));
        }
        if let Some(error) = prepared.error() {
            return Err(error);
        }
        if prepared.base_revision != self.revision {
            prepared.set_status(PreparedStatus::Stale);
            return Err(DeltaError::invalid("Prepared change is stale"));
        }
        if prepared.base != self.value {
            prepared.set_status(PreparedStatus::Stale);
            return Err(DeltaError::invalid("Prepared change is stale"));
        }
        prepared.set_status(PreparedStatus::Consumed);
        self.value = prepared.value;
        self.revision += 1;
        Ok(())
    }
}

/// Take immutable ownership of an alias-free strict-JSON root in O(1)
/// (upstream `track`).
pub fn track(initial: JsonValue) -> Tracker {
    Tracker::new(initial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::Seg;
    use serde_json::json;

    #[test]
    fn takes_immutable_ownership_of_the_imported_revision() {
        let input = json!({
            "point": {"x": 3, "y": 7, "pressure": 0.1},
            "rows": [{"values": [0, false, null, "text", {"n": 1}]}]
        });
        let tracker = Tracker::new(input.clone());
        // Upstream asserts identity (`tracker.value === input`) and alias-free
        // nesting; ownership in Rust means the tracker owns the value and
        // `value()` borrows it, so structural equality is the observable.
        assert_eq!(tracker.value(), &input);
        assert_eq!(tracker.value()["point"], input["point"]);
        assert_eq!(
            tracker.value()["rows"][0]["values"][4],
            input["rows"][0]["values"][4]
        );
        assert_eq!(tracker.revision(), 0);
    }

    // "preserves null prototypes in an alias-free owned root" is JavaScript
    // prototype semantics; serde objects have no prototypes and the alias-free
    // deep copy is trivially true for owned values. Not portable.

    // "does not traverse a trusted root while taking ownership" counts getter
    // reads through a JS accessor property; `Tracker::new` takes an owned
    // value with no accessors, so there is nothing to count. Not portable.

    #[test]
    fn copies_assigned_and_inserted_values_immediately() {
        let mut tracker = Tracker::new(json!({"rows": []}));
        let mut assigned = json!({"nested": {"value": 1}});
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            state
                .push(&[Seg::key("rows")], vec![assigned.clone()])
                .unwrap();
            // Mutating the source after insertion must not leak into the draft.
            assigned["nested"]["value"] = json!(9);
            let inserted = state
                .get_index(&[Seg::key("rows")], 0)
                .unwrap()
                .unwrap()
                .clone();
            assert_eq!(inserted, json!({"nested": {"value": 1}}));
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["rows"][0]["nested"]["value"], json!(1));
        let replica = {
            let mut value = prepared.base.clone();
            apply(&mut value, &prepared.ops).unwrap();
            value
        };
        assert_eq!(replica, prepared.value);
    }
    // Port of upstream "folds edits to introduced subtrees": an edit made
    // through a path INTO a written (placement) payload must be folded into
    // the parent write's emitted value, not dropped.
    #[test]
    fn folds_edits_to_introduced_subtrees() {
        let mut tracker = Tracker::new(json!({}));
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            state
                .set(&[], "nested", json!({"rows": [{"value": 1}]}))
                .unwrap();
            state
                .set(
                    &[Seg::key("nested"), Seg::key("rows"), Seg::index(0)],
                    "value",
                    json!(2),
                )
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            serde_json::to_value(&prepared.ops).unwrap(),
            json!([["s", ["nested"], {"rows": [{"value": 2}]}]])
        );
        let replay = {
            let mut value = prepared.base.clone();
            apply(&mut value, &prepared.ops).unwrap();
            value
        };
        assert_eq!(replay, prepared.value);
        assert_eq!(prepared.value["nested"]["rows"][0]["value"], json!(2));
    }

    // Port of upstream "handles reserved own keys without prototype pollution":
    // reserved keys as DATA are ordinary serde map keys; a write UNDER a
    // reserved key folds the containing object so the edit survives replay.
    #[test]
    fn folds_writes_under_reserved_own_keys_without_prototype_pollution() {
        let mut tracker = Tracker::new(json!({"safe": {"__proto__": {"value": 1}}}));
        let mut change = tracker.begin_change();
        let prepared = {
            let mut state = change.state();
            state
                .set(
                    &[Seg::key("safe"), Seg::key("__proto__")],
                    "value",
                    json!(2),
                )
                .unwrap();
            // Reads through the draft observe the edit before preparation.
            assert_eq!(
                state
                    .get(&[Seg::key("safe"), Seg::key("__proto__"), Seg::key("value")])
                    .unwrap(),
                Some(&json!(2))
            );
            change.prepare().unwrap()
        };
        let replay = {
            let mut value = prepared.base.clone();
            apply(&mut value, &prepared.ops).unwrap();
            value
        };
        assert_eq!(replay, prepared.value);
        assert_eq!(prepared.value["safe"]["__proto__"]["value"], json!(2));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Port of `test/delta-tracker/tracker.test.ts`.
//
// Deviations are recorded inline per case; summary:
// - JS-proxy identity assertions (`isProxy`, `toBe` on handles), WeakRef/GC
//   retention, symbol/prototype/inherited-property semantics, `Reflect.apply`
//   receiver tricks, thenable/microtask liveness, and valueOf coercion side
//   effects have no Rust counterpart and are skipped or reduced to their
//   behavioral half.
// - Two simultaneously open changes (upstream `beginChange()` twice) are
//   prevented by `Change`'s `&mut Tracker` borrow, so eager-invalidation cases
//   are unreachable by construction and not ported.
// - `Prepared` clones share one status cell, so `prepared.clone()` plays the
//   role of upstream's retained references across `adopt`.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tracker_tests {
    use super::*;
    use crate::delta::{apply, decoder, encoder, Draft, Op, Seg};
    use serde_json::json;

    type R = Result<(), DeltaError>;

    fn replay(base: &JsonValue, ops: &[Op]) -> JsonValue {
        let mut replayed = base.clone();
        apply(&mut replayed, ops).expect("replay");
        replayed
    }

    /// Upstream `settle`: clone base, mutate through a draft, prepare, check the
    /// detached replay and base immutability, adopt, return the candidate.
    fn settle(tracker: &mut Tracker, mutate: impl FnOnce(&mut Draft)) -> JsonValue {
        let base = tracker.value().clone();
        {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            mutate(&mut state);
            let prepared = change.prepare().expect("prepare");
            assert_eq!(replay(&base, &prepared.ops), prepared.value, "replay check");
            assert_eq!(&base, tracker.value(), "prepare must not touch the base");
            tracker.adopt(prepared).expect("adopt");
        }
        tracker.value().clone()
    }

    fn assert_error<T: std::fmt::Debug>(result: Result<T, DeltaError>, needle: &str) {
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains(needle),
            "message `{message}` lacks `{needle}`"
        );
    }

    /// Canonical JS `Object.keys` order for a committed object: integer-like
    /// keys ascending first, then string keys in map (insertion) order.
    fn js_keys(value: &JsonValue) -> Vec<String> {
        let object = value.as_object().expect("object");
        let mut indices: Vec<(usize, String)> = Vec::new();
        let mut strings: Vec<String> = Vec::new();
        for key in object.keys() {
            match key.parse::<usize>() {
                Ok(index) => indices.push((index, key.clone())),
                Err(_) => strings.push(key.clone()),
            }
        }
        indices.sort_by_key(|(index, _)| *index);
        indices
            .into_iter()
            .map(|(_, key)| key)
            .chain(strings)
            .collect()
    }

    // ─── delta tracker transactional overlay lifecycle ──────────────────────

    #[test]
    fn materializes_an_immutable_next_revision_and_adopts_it_by_pointer_swap() {
        let initial = json!({"count": 1, "nested": {"text": "a"}, "values": [1]});
        let mut tracker = Tracker::new(initial.clone());
        assert_eq!(tracker.value(), &initial);
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[], "count", json!(2)).unwrap();
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
            change.prepare().unwrap()
        };
        assert_eq!(
            initial,
            json!({"count": 1, "nested": {"text": "a"}, "values": [1]})
        );
        assert_eq!(prepared.base_revision, 0);
        assert_eq!(prepared.base, initial);
        assert_eq!(
            prepared.value,
            json!({"count": 2, "nested": {"text": "ab"}, "values": [1, 2]})
        );
        assert_eq!(
            prepared.ops,
            vec![
                Op::Set {
                    path: vec![Seg::key("count")],
                    value: json!(2)
                },
                Op::Append {
                    path: vec![Seg::key("nested"), Seg::key("text")],
                    text: "b".to_string(),
                },
                Op::Splice {
                    path: vec![Seg::key("values")],
                    start: 1,
                    remove: 0,
                    items: vec![json!(2)],
                },
            ]
        );
        assert_eq!(tracker.value(), &initial);
        tracker.adopt(prepared).unwrap();
        assert_eq!(
            initial,
            json!({"count": 1, "nested": {"text": "a"}, "values": [1]})
        );
        assert_eq!(
            tracker.value(),
            &json!({"count": 2, "nested": {"text": "ab"}, "values": [1, 2]})
        );
        // Upstream also asserts settled-overlay access throws /settled/; Rust
        // makes a settled `Change` unusable at compile time. Not portable.
    }

    // "keeps a transaction live across await and seals it only at prepare":
    // microtask liveness and post-settle access (`/settled/` throws) are JS
    // proxy semantics; Rust seals at the borrow level. Ported behavioral half:
    // mutations spanning separate `state()` borrows all land in one prepare.

    #[test]
    fn keeps_a_transaction_sealed_only_at_prepare() {
        let mut tracker = Tracker::new(json!({"left": 0, "nested": {"right": 0}}));
        let mut change = tracker.begin_change();
        change.state().set(&[], "left", json!(1)).unwrap();
        change
            .state()
            .set(&[Seg::key("nested")], "right", json!(2))
            .unwrap();
        let prepared = change.prepare().unwrap();
        assert_eq!(prepared.value, json!({"left": 1, "nested": {"right": 2}}));
        tracker.adopt(prepared).unwrap();
    }

    // "allows Promise resolution to inspect a settled draft without reporting a
    // false failure": thenable `draft.then` resolution and `await` of a settled
    // proxy are non-portable; reading a document key named `then` is trivially
    // true for `serde_json::Value`. Skipped.

    #[test]
    fn aborts_idempotently_and_revokes_held_descendants() {
        // Upstream aborts twice and touches descendants after each abort; a
        // Rust `Change` is consumed by `abort()`, so a second abort cannot be
        // expressed and held views cannot outlive the change borrow. Reduced
        // to the observable half: abort leaves the committed value untouched.
        let mut tracker = Tracker::new(json!({"child": {"value": 1}}));
        let mut change = tracker.begin_change();
        {
            let mut state = change.state();
            let mut child = state.at(&[Seg::key("child")]).unwrap();
            child.set(&[], "value", json!(2)).unwrap();
        }
        change.abort();
        assert_eq!(tracker.value()["child"]["value"], json!(1));
    }

    #[test]
    fn allows_competing_contexts_and_invalidates_loser_views() {
        // Two changes cannot be open at once (`&mut Tracker`), so the loser is
        // prepared before the winner begins; the stale/used rejection halves
        // are fully observable.
        let mut tracker = Tracker::new(json!({"value": 0}));
        let first_prepared = {
            let mut first = tracker.begin_change();
            first.state().set(&[], "value", json!(1)).unwrap();
            first.prepare().unwrap()
        };
        let second_prepared = {
            let mut second = tracker.begin_change();
            second.state().set(&[], "value", json!(2)).unwrap();
            second.prepare().unwrap()
        };
        let held = second_prepared.clone();
        let first_reuse = first_prepared.clone();
        tracker.adopt(first_prepared).unwrap();
        assert_eq!(tracker.value()["value"], json!(1));
        assert_eq!(held.value["value"], json!(2));
        assert_error(tracker.adopt(second_prepared), "stale");
        assert_error(tracker.adopt(first_reuse), "already been used");
    }

    // "invalidates competing open overlays without changing their base
    // revision": requires two simultaneously open changes; the `&mut Tracker`
    // borrow makes that state unrepresentable. Skipped.

    #[test]
    fn adopts_object_edits_as_ordinary_fast_layout_immutable_revisions() {
        let initial = json!({"first": 1, "second": 2});
        let mut tracker = Tracker::new(initial.clone());
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[], "first", json!(3)).unwrap();
            state.delete(&[], "second").unwrap();
            state.set(&[], "third", json!(4)).unwrap();
            change.prepare().unwrap()
        };
        tracker.adopt(prepared).unwrap();
        assert_ne!(tracker.value(), &initial);
        assert_eq!(initial, json!({"first": 1, "second": 2}));
        assert_eq!(tracker.value(), &json!({"first": 3, "third": 4}));
        // Upstream also asserts the property descriptor shape; serde objects
        // have no descriptors. Not portable.
    }

    #[test]
    fn supports_adopt_then_publish_ordering_and_retained_publication_reads() {
        let mut tracker = Tracker::new(json!({"value": 0, "nested": {"count": 0}}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[], "value", json!(1)).unwrap();
            state.set(&[Seg::key("nested")], "count", json!(1)).unwrap();
            change.prepare().unwrap()
        };
        let operations = prepared.ops.clone();
        let published = prepared.clone();
        tracker.adopt(prepared).unwrap();
        // Matches services/state.ts: adopt first, then inspect ops and publish.
        assert_eq!(operations.len(), 2);
        assert_eq!(published.value, json!({"value": 1, "nested": {"count": 1}}));
        assert_eq!(tracker.value(), &published.value);

        let next_prepared = {
            let mut next = tracker.begin_change();
            next.state().set(&[], "value", json!(2)).unwrap();
            next.prepare().unwrap()
        };
        tracker.adopt(next_prepared).unwrap();
        // Retained immutable publications remain at their original revision.
        assert_ne!(published.value, *tracker.value());
        assert_eq!(published.value, json!({"value": 1, "nested": {"count": 1}}));
        assert_eq!(
            tracker.value(),
            &json!({"value": 2, "nested": {"count": 1}})
        );
    }

    #[test]
    fn keeps_replacement_base_value_structurally_compatible_and_readable_after_adoption() {
        let original = json!({"value": 0});
        let replacement = json!({"value": 1});
        let mut tracker = Tracker::new(original.clone());
        let prepared = tracker.prepare_replace(replacement.clone()).unwrap();
        assert_eq!(prepared.base, original);
        assert_eq!(prepared.value, replacement);
        tracker.adopt(prepared.clone()).unwrap();
        assert_eq!(prepared.base, original);
        assert_eq!(prepared.value, replacement);
        assert_eq!(tracker.value(), &prepared.value);
    }

    #[test]
    fn keeps_aborted_and_stale_materialized_candidates_readable() {
        let mut tracker = Tracker::new(json!({"value": 0}));
        let mut aborted = {
            let mut change = tracker.begin_change();
            change.state().set(&[], "value", json!(1)).unwrap();
            change.prepare().unwrap()
        };
        let aborted_view = aborted.value.clone();
        aborted.abort();
        assert_eq!(aborted_view["value"], json!(1));
        assert_eq!(aborted.value["value"], json!(1));

        let loser = {
            let mut change = tracker.begin_change();
            change.state().set(&[], "value", json!(2)).unwrap();
            change.prepare().unwrap()
        };
        let loser_view = loser.value.clone();
        let winner = tracker.prepare_replace(json!({"value": 3})).unwrap();
        tracker.adopt(winner).unwrap();
        assert_eq!(loser_view["value"], json!(2));
        assert_eq!(loser.value["value"], json!(2));
        assert_error(tracker.adopt(loser), "stale");
    }

    #[test]
    fn lets_a_settled_change_abort_its_prepared_result() {
        // Upstream aborts the settled `Change` after `prepare()`; a Rust
        // `Change` is consumed by `prepare()`, so only the `Prepared::abort`
        // half is reachable. Same observable contract.
        let mut tracker = Tracker::new(json!({"value": 0, "nested": {"value": 1}}));
        let mut prepared = {
            let mut change = tracker.begin_change();
            change.state().set(&[], "value", json!(1)).unwrap();
            change.prepare().unwrap()
        };
        let operations = prepared.ops.clone();
        prepared.abort();
        assert_eq!(prepared.ops, operations);
        assert_eq!(prepared.value, json!({"value": 1, "nested": {"value": 1}}));
        assert_error(tracker.adopt(prepared), "aborted");
    }

    #[test]
    fn rejects_foreign_and_aborted_prepared_values() {
        let mut first = Tracker::new(json!({"value": 0}));
        let mut second = Tracker::new(json!({"value": 0}));
        let mut prepared = first.prepare_replace(json!({"value": 1})).unwrap();
        assert_error(second.adopt(prepared.clone()), "different tracker");
        prepared.abort();
        prepared.abort();
        assert_error(first.adopt(prepared), "aborted");
    }

    #[test]
    fn reads_deleted_own_properties_as_absent() {
        // Upstream also installs an inherited `Object.prototype` property and
        // observes prototype-driven reads on committed values; serde objects
        // have no prototype chain. Reduced to own-property semantics.
        let mut tracker = Tracker::new(json!({"value": 1}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.delete(&[], "value").unwrap();
            state.set(&[], "trackerInherited", json!(1)).unwrap();
            state.delete(&[], "trackerInherited").unwrap();
            assert_eq!(state.get(&[Seg::key("value")]), Ok(None));
            assert_eq!(state.get(&[Seg::key("trackerInherited")]), Ok(None));
            change.prepare().unwrap()
        };
        assert!(prepared.value.get("value").is_none());
        assert!(prepared.value.get("trackerInherited").is_none());
        tracker.adopt(prepared).unwrap();
        assert!(tracker.value().as_object().unwrap().is_empty());
    }

    #[test]
    fn detaches_old_handles_for_deeply_equal_object_and_array_assignments_before_normalizing() {
        // Upstream captures old child handles, reassigns deeply equal values,
        // mutates the old handles, and asserts no leakage plus identity
        // retention (`tracker.value === initial`). Rust handles are borrows
        // that cannot outlive the reassignment, and identity is ownership, so
        // the portable half is: deeply equal assignments normalize to zero ops
        // and leave the committed revision untouched.
        let initial = json!({
            "object": {"nested": {"value": 1}},
            "array": [{"value": 1}],
            "rows": [{"value": 1}],
        });
        let mut tracker = Tracker::new(initial.clone());
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state
                .set(&[], "object", json!({"nested": {"value": 1}}))
                .unwrap();
            state.set(&[], "array", json!([{"value": 1}])).unwrap();
            state
                .set_index(&[Seg::key("rows")], 0, json!({"value": 1}))
                .unwrap();
            change.prepare().unwrap()
        };
        assert!(prepared.ops.is_empty());
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value(), &initial);
    }

    #[test]
    fn normalizes_a_deeply_equal_replacement_without_changing_committed_identity() {
        let initial = json!({"nested": {"value": 1}, "rows": [1, 2, 3]});
        let mut tracker = Tracker::new(initial.clone());
        let prepared = tracker
            .prepare_replace(json!({"nested": {"value": 1}, "rows": [1, 2, 3]}))
            .unwrap();
        assert!(prepared.ops.is_empty());
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value(), &initial);
    }

    #[test]
    fn takes_o1_immutable_ownership_for_replacement_and_its_operation_payload() {
        let mut tracker = Tracker::new(json!({"value": 0, "rows": []}));
        let replacement = json!({"value": 1, "rows": [{"value": 2}]});
        let prepared = tracker.prepare_replace(replacement.clone()).unwrap();
        assert_eq!(prepared.base, *tracker.value());
        assert_eq!(prepared.value, replacement);
        assert_eq!(prepared.ops, vec![Op::Replace(replacement.clone())]);
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value(), &replacement);
    }

    // ─── delta tracker policy view ──────────────────────────────────────────

    #[test]
    fn supports_native_reads_keys_and_json_round_trip() {
        // Upstream asserts Array.isArray, map(), descriptors and spread
        // iteration through the proxy; the portable half is overlay reads,
        // length, key order, and JSON round-tripping.
        let initial = json!({"values": [1, 2, 3], "object": {"a": 1}});
        let mut tracker = Tracker::new(initial.clone());
        let mut change = tracker.begin_change();
        {
            let mut state = change.state();
            state
                .splice(&[Seg::key("values")], 1, Some(1), vec![json!(4), json!(5)])
                .unwrap();
            state.set(&[Seg::key("object")], "b", json!(2)).unwrap();
            assert_eq!(state.length(&[Seg::key("values")]).unwrap(), 4);
            assert_eq!(
                state.get_index(&[Seg::key("values")], 0).unwrap().unwrap(),
                &json!(1)
            );
            assert_eq!(
                state.get_index(&[Seg::key("values")], 2).unwrap().unwrap(),
                &json!(5)
            );
            // Rust read_keys appends "length" for arrays; the index half
            // matches upstream's `Object.keys` order.
            assert_eq!(
                state.keys(&[Seg::key("values")]).unwrap(),
                vec!["0", "1", "2", "3", "length"]
            );
            assert_eq!(state.keys(&[Seg::key("object")]).unwrap(), vec!["a", "b"]);
        }
        change.abort();
        assert_eq!(*tracker.value(), initial);
    }

    #[test]
    fn keeps_document_keys_distinct_from_object_proxy_target_fields() {
        let mut tracker = Tracker::new(json!({
            "object": {"context": 1, "base": 2, "parent": 3, "dirty": 4, "target": 5, "proxy": 6},
        }));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state
                .set(&[Seg::key("object")], "context", json!(7))
                .unwrap();
            state.set(&[Seg::key("object")], "proxy", json!(8)).unwrap();
            assert_eq!(
                state.keys(&[Seg::key("object")]).unwrap(),
                vec!["context", "base", "parent", "dirty", "target", "proxy"]
            );
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value["object"],
            json!({"context": 7, "base": 2, "parent": 3, "dirty": 4, "target": 5, "proxy": 8})
        );
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn keeps_string_operation_forms() {
        // Upstream drives a splice through an object with a side-effecting
        // valueOf; coercion side effects are non-portable. The string-slice
        // append/truncate decomposition and codec round-trip are portable.
        let mut tracker =
            Tracker::new(json!({"text": "abcdefgh", "values": [1, 2, 3], "marker": 0}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            let text = state
                .get(&[Seg::key("text")])
                .unwrap()
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();
            state
                .set(&[], "text", json!(format!("{}xyz", &text[3..])))
                .unwrap();
            state
                .splice(&[Seg::key("values")], 1, Some(1), vec![json!(4)])
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value,
            json!({"text": "defghxyz", "values": [1, 4, 3], "marker": 0})
        );
        assert!(prepared.ops.contains(&Op::Truncate {
            path: vec![Seg::key("text")],
            count: 3,
        }));
        assert!(prepared.ops.contains(&Op::Append {
            path: vec![Seg::key("text")],
            text: "xyz".to_string(),
        }));
        let wire = encoder().encode(&prepared.ops);
        let decoded = decoder().decode(&wire).unwrap();
        assert_eq!(replay(tracker.value(), &decoded), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn emits_deletion_and_supports_root_arrays() {
        let mut object_tracker = Tracker::new(json!({"keep": 1, "remove": 2}));
        settle(&mut object_tracker, |draft| {
            draft.delete(&[], "remove").unwrap();
        });
        assert_eq!(object_tracker.value(), &json!({"keep": 1}));

        let mut array_tracker = Tracker::new(json!([1, 2, 3]));
        settle(&mut array_tracker, |draft| {
            draft.reverse(&[]).unwrap();
            draft.push(&[], vec![json!(4)]).unwrap();
        });
        assert_eq!(array_tracker.value(), &json!([3, 2, 1, 4]));
    }

    #[test]
    fn preserves_native_object_key_order_after_delete_and_re_add() {
        let mut tracker = Tracker::new(json!({"first": 1, "second": 2}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.delete(&[], "first").unwrap();
            state.set(&[], "first", json!(1)).unwrap();
            assert_eq!(state.keys(&[]).unwrap(), vec!["second", "first"]);
            change.prepare().unwrap()
        };
        tracker.adopt(prepared).unwrap();
        assert_eq!(js_keys(tracker.value()), vec!["second", "first"]);
    }

    #[test]
    fn orders_new_integer_like_object_keys_before_strings_for_recipe_visible_reads() {
        let mut tracker = Tracker::new(json!({"object": {"label": "x"}, "first": ""}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[Seg::key("object")], "2", json!("two")).unwrap();
            state.set(&[Seg::key("object")], "1", json!("one")).unwrap();
            assert_eq!(
                state.keys(&[Seg::key("object")]).unwrap(),
                vec!["1", "2", "label"]
            );
            let first = state.keys(&[Seg::key("object")]).unwrap()[0].clone();
            state.set(&[], "first", json!(first)).unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["first"], json!("1"));
        assert_eq!(js_keys(&prepared.value["object"]), vec!["1", "2", "label"]);
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value()["first"], json!("1"));
        assert_eq!(js_keys(&tracker.value()["object"]), vec!["1", "2", "label"]);
    }

    #[test]
    fn normalizes_deeply_equal_container_assignments_before_direct_candidate_materialization() {
        let initial = json!({"child": {"a": 1, "b": 2}, "count": 0});
        let mut tracker = Tracker::new(initial.clone());
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[], "child", json!({"b": 2, "a": 1})).unwrap();
            state.set(&[], "count", json!(1)).unwrap();
            change.prepare().unwrap()
        };
        let replayed = replay(&initial, &prepared.ops);
        assert_eq!(
            prepared.ops,
            vec![Op::Set {
                path: vec![Seg::key("count")],
                value: json!(1)
            }]
        );
        assert_eq!(js_keys(&prepared.value["child"]), vec!["a", "b"]);
        assert_eq!(js_keys(&replayed["child"]), vec!["a", "b"]);
        assert_eq!(prepared.value, replayed);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn keeps_native_integer_and_string_ordering_across_deletion_and_re_addition() {
        let mut tracker = Tracker::new(json!({
            "object": {"1": "one", "2": "two", "first": "a", "second": "b"},
        }));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.delete(&[Seg::key("object")], "2").unwrap();
            state.set(&[Seg::key("object")], "2", json!("two")).unwrap();
            state.delete(&[Seg::key("object")], "first").unwrap();
            state
                .set(&[Seg::key("object")], "first", json!("a"))
                .unwrap();
            state
                .set(&[Seg::key("object")], "3", json!("three"))
                .unwrap();
            state
                .set(&[Seg::key("object")], "0", json!("zero"))
                .unwrap();
            assert_eq!(
                state.keys(&[Seg::key("object")]).unwrap(),
                vec!["0", "1", "2", "3", "second", "first"]
            );
            change.prepare().unwrap()
        };
        assert_eq!(
            js_keys(&prepared.value["object"]),
            vec!["0", "1", "2", "3", "second", "first"]
        );
        tracker.adopt(prepared).unwrap();
        assert_eq!(
            js_keys(&tracker.value()["object"]),
            vec!["0", "1", "2", "3", "second", "first"]
        );
    }

    #[test]
    fn orders_integer_like_keys_on_plain_objects() {
        // Upstream builds the inner object with `Object.create(null)` and
        // asserts the null prototype on materialized values; serde objects
        // have no prototypes. Reduced to the canonical key-ordering half.
        let mut tracker = Tracker::new(json!({"object": {"label": "x"}}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[Seg::key("object")], "2", json!("two")).unwrap();
            state.set(&[Seg::key("object")], "1", json!("one")).unwrap();
            assert_eq!(
                state.keys(&[Seg::key("object")]).unwrap(),
                vec!["1", "2", "label"]
            );
            change.prepare().unwrap()
        };
        assert_eq!(js_keys(&prepared.value["object"]), vec!["1", "2", "label"]);
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
        assert_eq!(js_keys(&tracker.value()["object"]), vec!["1", "2", "label"]);
    }

    // "forwards borrowed mutators to ordinary generic array receivers"
    // (`Reflect.apply(proxyMethod, plainArray, …)`): receiver rebinding is a
    // JS proxy mechanic. Skipped.

    #[test]
    fn matches_native_splice_fill_and_copywithin_semantics() {
        // Upstream matches the draft against a native array mutated through the
        // same call with a side-effecting index coercion (the coercion pushes
        // an element mid-call); coercion hooks are non-portable, so the native
        // receiver is pre-grown by that element and plain bounds are used. The
        // observable contract — draft results equal native JS results — holds.
        let mut native: Vec<JsonValue> = (1..=4).map(|v| json!(v)).collect(); // [1,2,3] + the valueOf-time push(4)
        native.splice(1..2, [json!(9)]);
        let mut tracker = Tracker::new(json!({"values": [1, 2, 3, 4]}));
        settle(&mut tracker, |draft| {
            draft
                .splice(&[Seg::key("values")], 1, Some(1), vec![json!(9)])
                .unwrap();
        });
        assert_eq!(tracker.value()["values"].as_array().unwrap(), &native);

        let mut native: Vec<JsonValue> = (1..=4).map(|v| json!(v)).collect();
        for slot in native.iter_mut().skip(1) {
            *slot = json!(7);
        }
        let mut tracker = Tracker::new(json!({"values": [1, 2, 3, 4]}));
        settle(&mut tracker, |draft| {
            draft
                .fill(&[Seg::key("values")], json!(7), Some(1), None)
                .unwrap();
        });
        assert_eq!(tracker.value()["values"].as_array().unwrap(), &native);

        let mut native: Vec<JsonValue> = (1..=4).map(|v| json!(v)).collect();
        // JS copyWithin(1, 0, 2) on non-Copy elements: reads from a snapshot.
        let source: Vec<JsonValue> = native[0..2].to_vec();
        for (offset, index) in (1..3usize).enumerate() {
            native[index] = source[offset].clone();
        }
        let mut tracker = Tracker::new(json!({"values": [1, 2, 3, 4]}));
        settle(&mut tracker, |draft| {
            draft
                .copy_within(&[Seg::key("values")], 1, 0, Some(2))
                .unwrap();
        });
        assert_eq!(tracker.value()["values"].as_array().unwrap(), &native);
    }

    // "does not report inherited methods as own array properties": inherited
    // method descriptors (`map` on Array.prototype) have no Rust counterpart.
    // Skipped.

    #[test]
    fn keeps_repeated_identical_writes_instead_of_reverting_their_pending_override() {
        let mut tracker = Tracker::new(json!({"value": 0, "values": [0]}));
        settle(&mut tracker, |draft| {
            draft.set(&[], "value", json!(1)).unwrap();
            draft.set(&[], "value", json!(1)).unwrap();
            draft
                .set_index(&[Seg::key("values")], 0, JsonValue::Null)
                .unwrap();
            draft
                .set_index(&[Seg::key("values")], 0, JsonValue::Null)
                .unwrap();
            draft.push(&[Seg::key("values")], vec![json!(2)]).unwrap();
            draft
                .set_index(&[Seg::key("values")], 1, JsonValue::Null)
                .unwrap();
            draft
                .set_index(&[Seg::key("values")], 1, JsonValue::Null)
                .unwrap();
        });
        assert_eq!(
            tracker.value(),
            &json!({"value": 1, "values": [null, null]})
        );
    }

    #[test]
    fn rejects_array_index_gaps_without_mutation_while_allowing_replacement_and_append() {
        let mut tracker = Tracker::new(json!({"values": [1, 2]}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            assert_error(
                state
                    .set_index(&[Seg::key("values")], 3, json!(4))
                    .map_err(|e| e)
                    .map(|_| ()),
                "holes",
            );
            assert_eq!(state.length(&[Seg::key("values")]).unwrap(), 2);
            assert_eq!(
                state.get_index(&[Seg::key("values")], 0).unwrap().unwrap(),
                &json!(1)
            );
            assert_eq!(
                state.get_index(&[Seg::key("values")], 1).unwrap().unwrap(),
                &json!(2)
            );

            state.set_index(&[Seg::key("values")], 1, json!(9)).unwrap();
            state.set_index(&[Seg::key("values")], 2, json!(3)).unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"], json!([1, 9, 3]));
        assert_eq!(replay(tracker.value(), &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value()["values"], json!([1, 9, 3]));
    }

    #[test]
    fn supports_optional_deletion_explicit_null_length_growth_shrinking_and_default_sort() {
        // Upstream clears `optional` by assigning `undefined`; the Rust
        // equivalent of clearing a key is an explicit delete.
        fn read_value(draft: &mut Draft, index: usize) -> JsonValue {
            draft
                .get_index(&[Seg::key("values")], index)
                .unwrap()
                .unwrap()
                .clone()
        }
        let mut tracker = Tracker::new(json!({"optional": "remove", "values": [3, 1, 2]}));
        settle(&mut tracker, |draft| {
            draft.delete(&[], "optional").unwrap();
            draft.set_length(&[Seg::key("values")], 5).unwrap();
            let grown: Vec<JsonValue> = (0..5).map(|index| read_value(draft, index)).collect();
            assert_eq!(
                grown,
                vec![
                    json!(3),
                    json!(1),
                    json!(2),
                    JsonValue::Null,
                    JsonValue::Null
                ]
            );
            draft.set_length(&[Seg::key("values")], 4).unwrap();
            draft.sort_default(&[Seg::key("values")]).unwrap();
            assert_error(draft.delete(&[Seg::key("values")], "0"), "holes");
        });
        assert_eq!(tracker.value(), &json!({"values": [1, 2, 3, null]}));
    }

    #[test]
    fn enumerates_wide_objects_without_duplicate_keys() {
        let mut tracker = Tracker::new(json!({"values": {}}));
        let mut change = tracker.begin_change();
        let mut state = change.state();
        for index in 0..20_000usize {
            state
                .set(
                    &[Seg::key("values")],
                    &format!("field{index}"),
                    json!(index),
                )
                .unwrap();
        }
        let keys = state.keys(&[Seg::key("values")]).unwrap();
        assert_eq!(keys.len(), 20_000);
        assert_eq!(keys[0], "field0");
        assert_eq!(keys[19_999], "field19999");
        change.abort();
    }

    #[test]
    fn does_not_dirty_read_only_traversals() {
        let mut tracker = Tracker::new(json!({"nested": {"rows": [{"value": 1}]}}));
        let prepared = {
            let mut change = tracker.begin_change();
            let state = change.state();
            assert_eq!(
                state
                    .get(&[
                        Seg::key("nested"),
                        Seg::key("rows"),
                        Seg::index(0),
                        Seg::key("value")
                    ])
                    .unwrap()
                    .unwrap(),
                &json!(1)
            );
            change.prepare().unwrap()
        };
        assert!(prepared.ops.is_empty());
        tracker.adopt(prepared).unwrap();
    }

    // "uses one proxy identity per accessed container": handle identity
    // (`toBe`) is a JS proxy mechanic; Rust borrows guarantee it statically.
    // Skipped.

    // ─── delta tracker by-value placements ──────────────────────────────────

    #[test]
    fn clones_property_index_push_unshift_splice_fill_and_copywithin_placements() {
        let external = json!({"value": 5});
        let mut tracker = Tracker::new(json!({
            "property": JsonValue::Null,
            "values": [{"value": 0}, {"value": 1}, {"value": 2}],
        }));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[], "property", external.clone()).unwrap();
            state
                .set_index(&[Seg::key("values")], 0, external.clone())
                .unwrap();
            state
                .push(&[Seg::key("values")], vec![external.clone()])
                .unwrap();
            state
                .unshift(&[Seg::key("values")], vec![external.clone()])
                .unwrap();
            state
                .splice(&[Seg::key("values")], 2, Some(0), vec![external.clone()])
                .unwrap();
            let first = state
                .get_index(&[Seg::key("values")], 0)
                .unwrap()
                .unwrap()
                .clone();
            state
                .fill(&[Seg::key("values")], first, Some(1), Some(3))
                .unwrap();
            state
                .copy_within(&[Seg::key("values")], 3, 0, Some(2))
                .unwrap();
            change.prepare().unwrap()
        };
        // Every placement is an independent clone of `external` at its
        // snapshot ({value: 5}); upstream's post-assignment `external.value =
        // 99` has no Rust counterpart because values copy on assignment.
        assert_eq!(prepared.value["property"], json!({"value": 5}));
        for index in 0..5 {
            assert_eq!(
                prepared.value["values"][index],
                json!({"value": 5}),
                "index {index}"
            );
        }
        // The sixth element is the entry appended by `push` before `unshift`
        // and `splice` shifted it; upstream only asserts `slice(0, 5)`.
        assert_eq!(prepared.value["values"].as_array().unwrap().len(), 6);
        assert_eq!(prepared.value["values"][5], json!({"value": 5}));
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn rejects_invalid_placements_without_changing_the_draft() {
        // Upstream rejects `undefined`, NaN, ±Infinity, functions, symbols,
        // BigInt, Date, sparse arrays, cyclic objects, and accessor-only
        // objects; none are constructible as `serde_json::Value`, so strict-JSON
        // validation holds by construction. The portable half is: a rejected
        // mutation leaves the draft untouched while accepted writes proceed.
        let initial = json!({"payload": JsonValue::Null, "number": 0, "values": [1, 2]});
        let mut tracker = Tracker::new(initial.clone());
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            assert_error(
                state
                    .set_index(&[Seg::key("values")], 3, json!(4))
                    .map(|_| ()),
                "holes",
            );
            assert_eq!(
                state.get_index(&[Seg::key("values")], 1).unwrap().unwrap(),
                &json!(2)
            );
            assert_eq!(
                state.get_index(&[Seg::key("values")], 0).unwrap().unwrap(),
                &json!(1)
            );
            state.set(&[], "payload", json!({"valid": true})).unwrap();
            state
                .push(&[Seg::key("values")], vec![json!({"valid": true})])
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value,
            json!({"payload": {"valid": true}, "number": 0, "values": [1, 2, {"valid": true}]})
        );
        assert_eq!(replay(&initial, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn expands_repeated_source_aliases_into_independent_placements() {
        let shared = json!({"nested": {"value": 1}});
        let mut tracker = Tracker::new(json!({
            "left": JsonValue::Null,
            "right": JsonValue::Null,
            "rows": [],
        }));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[], "left", shared.clone()).unwrap();
            state.set(&[], "right", shared.clone()).unwrap();
            state
                .push(&[Seg::key("rows")], vec![shared.clone(), shared])
                .unwrap();
            state
                .set(&[Seg::key("left"), Seg::key("nested")], "value", json!(9))
                .unwrap();
            state
                .set(
                    &[Seg::key("rows"), Seg::index(0), Seg::key("nested")],
                    "value",
                    json!(8),
                )
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value,
            json!({
                "left": {"nested": {"value": 9}},
                "right": {"nested": {"value": 1}},
                "rows": [{"nested": {"value": 8}}, {"nested": {"value": 1}}],
            })
        );
        assert_eq!(replay(tracker.value(), &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn deep_clones_draft_sourced_placements_within_and_across_revisions() {
        // Upstream's `not.toBe` identity and alias-free assertions hold
        // trivially for owned `JsonValue` trees; structural outcomes are
        // asserted instead. The within-transaction half also edits a placed
        // subtree (`b.child.value = 3`); that folding is blocked by the
        // placement-subtree gap (see `ignores_edits_to_introduced_subtrees`)
        // and asserted by the ignored test.
        let mut tracker = Tracker::new(json!({
            "a": {"child": {"value": 1}},
            "b": JsonValue::Null,
            "rows": [{"child": {"value": 1}}, {"child": {"value": 2}}],
        }));
        let first_prepared = {
            let mut first = tracker.begin_change();
            let mut state = first.state();
            let a = state.get(&[Seg::key("a")]).unwrap().unwrap().clone();
            state.set(&[], "b", a).unwrap();
            let row0 = state
                .get_index(&[Seg::key("rows")], 0)
                .unwrap()
                .unwrap()
                .clone();
            state.set_index(&[Seg::key("rows")], 1, row0).unwrap();
            first.prepare().unwrap()
        };
        assert_eq!(first_prepared.value["a"]["child"]["value"], json!(1));
        assert_eq!(first_prepared.value["rows"][0]["child"]["value"], json!(1));
        assert_eq!(first_prepared.value["rows"][1]["child"]["value"], json!(1));
        tracker.adopt(first_prepared).unwrap();

        let base = tracker.value().clone();
        let second_prepared = {
            let mut second = tracker.begin_change();
            second
                .state()
                .set(
                    &[Seg::key("rows"), Seg::index(1), Seg::key("child")],
                    "value",
                    json!(4),
                )
                .unwrap();
            second.prepare().unwrap()
        };
        assert_eq!(second_prepared.value["rows"][0]["child"]["value"], json!(1));
        assert_eq!(second_prepared.value["rows"][1]["child"]["value"], json!(4));
        assert_eq!(replay(&base, &second_prepared.ops), second_prepared.value);
        tracker.adopt(second_prepared).unwrap();
    }

    #[test]
    fn distinguishes_raw_committed_references_from_draft_references() {
        let mut tracker = Tracker::new(json!({
            "source": {"value": 1},
            "rawCopy": JsonValue::Null,
            "draftCopy": JsonValue::Null,
        }));
        let raw = tracker.value()["source"].clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set(&[Seg::key("source")], "value", json!(2)).unwrap();
            state.set(&[], "rawCopy", raw).unwrap();
            // Overlay writes are visible through NodeView path reads.
            let view = state.at(&[Seg::key("source")]).unwrap();
            let draft_value = view.get(&[Seg::key("value")]).unwrap().unwrap().clone();
            state
                .set(&[], "draftCopy", json!({"value": draft_value}))
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value,
            json!({"source": {"value": 2}, "rawCopy": {"value": 1}, "draftCopy": {"value": 2}})
        );
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn folds_edits_to_introduced_object_and_array_subtrees_into_placement_payloads() {
        let mut tracker = Tracker::new(json!({"nested": JsonValue::Null, "rows": []}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state
                .set(&[], "nested", json!({"rows": [{"value": 1}]}))
                .unwrap();
            state
                .set(
                    &[Seg::key("nested"), Seg::key("rows"), Seg::index(0)],
                    "value",
                    json!(2),
                )
                .unwrap();
            state
                .push(
                    &[Seg::key("nested"), Seg::key("rows")],
                    vec![json!({"value": 3})],
                )
                .unwrap();
            state
                .push(&[Seg::key("rows")], vec![json!({"values": [1]})])
                .unwrap();
            state
                .push(
                    &[Seg::key("rows"), Seg::index(0), Seg::key("values")],
                    vec![json!(2)],
                )
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.ops,
            vec![
                Op::Set {
                    path: vec![Seg::key("nested")],
                    value: json!({"rows": [{"value": 2}, {"value": 3}]}),
                },
                Op::Splice {
                    path: vec![Seg::key("rows")],
                    start: 0,
                    remove: 0,
                    items: vec![json!({"values": [1, 2]})],
                },
            ]
        );
        assert_eq!(replay(tracker.value(), &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn folds_pathological_operation_counts_without_payload_cost_estimation() {
        let mut wide = serde_json::Map::new();
        for index in 0..5_000usize {
            wide.insert(format!("field{index}"), json!(0));
        }
        let mut wide_tracker = Tracker::new(JsonValue::Object(wide));
        let wide_base = wide_tracker.value().clone();
        let wide_prepared = {
            let mut change = wide_tracker.begin_change();
            let mut state = change.state();
            for index in 0..5_000usize {
                state.set(&[], &format!("field{index}"), json!(1)).unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(wide_prepared.ops.len(), 1);
        assert!(matches!(wide_prepared.ops[0], Op::Replace(_)));
        assert_eq!(replay(&wide_base, &wide_prepared.ops), wide_prepared.value);
        wide_tracker.adopt(wide_prepared).unwrap();

        let rows: Vec<JsonValue> = (0..15_000).map(|value| json!({"value": value})).collect();
        let mut sparse_tracker = Tracker::new(json!({"rows": rows}));
        let sparse_base = sparse_tracker.value().clone();
        let sparse_prepared = {
            let mut change = sparse_tracker.begin_change();
            let mut state = change.state();
            for index in (0..15_000).step_by(3) {
                state
                    .set(
                        &[Seg::key("rows"), Seg::index(index)],
                        "value",
                        json!(-(index as i64) - 1),
                    )
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(sparse_prepared.ops.len(), 1);
        assert!(matches!(sparse_prepared.ops[0], Op::Replace(_)));
        assert_eq!(
            replay(&sparse_base, &sparse_prepared.ops),
            sparse_prepared.value
        );
        sparse_tracker.adopt(sparse_prepared).unwrap();
    }

    #[test]
    fn shares_operation_payload_structure_with_the_materialized_candidate() {
        // Upstream asserts `isProxy` and reference identity between op payloads
        // and the candidate; both are ownership artifacts. The structural half
        // — the op payload equals the corresponding candidate subtree — is
        // ported.
        let mut tracker = Tracker::new(json!({"rows": []}));
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .push(&[Seg::key("rows")], vec![json!({"value": 1})])
                .unwrap();
            change.prepare().unwrap()
        };
        match &prepared.ops[0] {
            Op::Splice { items, .. } => {
                assert_eq!(items[0], prepared.value["rows"][0]);
            }
            other => panic!("expected splice, got {other:?}"),
        }
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value()["rows"][0], json!({"value": 1}));
    }

    // ─── delta tracker piece arrays ─────────────────────────────────────────

    #[test]
    fn keeps_adoption_independent_from_mutable_detached_permutation_metadata() {
        // Upstream mutates the permutation inside `prepared.ops` after prepare
        // and observes adoption ignore the tampering. Rust ops are owned by the
        // `Prepared`, so the mutation is applied to a detached clone and the
        // original adoption is asserted clean — the same observable outcome.
        let mut tracker = Tracker::new(json!({"values": [3, 1, 2]}));
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .sort_by(&[Seg::key("values")], |left, right| {
                    left.as_i64().unwrap().cmp(&right.as_i64().unwrap())
                })
                .unwrap();
            change.prepare().unwrap()
        };
        let mut tampered = prepared.clone();
        for op in &mut tampered.ops {
            if let Op::Permute { permutation, .. } = op {
                permutation.reverse();
            }
        }
        assert_ne!(tampered.ops, prepared.ops);
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value()["values"], json!([1, 2, 3]));
    }

    #[test]
    fn matches_native_splice_with_zero_or_one_argument() {
        // Upstream calls `splice()` with no arguments; the Rust signature
        // expresses that as `splice(0, Some(0), [])` (remove nothing, insert
        // nothing). The one-argument form maps to `remove: None`.
        let mut no_arguments = Tracker::new(json!({"values": [1, 2, 3]}));
        let prepared = {
            let mut change = no_arguments.begin_change();
            let removed = change
                .state()
                .splice(&[Seg::key("values")], 0, Some(0), vec![])
                .unwrap();
            assert!(removed.is_empty());
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"], json!([1, 2, 3]));
        assert!(prepared.ops.is_empty());
        no_arguments.adopt(prepared).unwrap();

        let mut one_argument = Tracker::new(json!({"values": [1, 2, 3, 4]}));
        let prepared = {
            let mut change = one_argument.begin_change();
            let removed = change
                .state()
                .splice(&[Seg::key("values")], 1, None, vec![])
                .unwrap();
            assert_eq!(removed, vec![json!(2), json!(3), json!(4)]);
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"], json!([1]));
        assert_eq!(replay(one_argument.value(), &prepared.ops), prepared.value);
        one_argument.adopt(prepared).unwrap();

        let mut past_end = Tracker::new(json!({"values": [1, 2, 3]}));
        let prepared = {
            let mut change = past_end.begin_change();
            let removed = change
                .state()
                .splice(&[Seg::key("values")], 3, None, vec![])
                .unwrap();
            assert!(removed.is_empty());
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"], json!([1, 2, 3]));
        past_end.adopt(prepared).unwrap();
    }

    #[test]
    fn supports_all_structural_mutators_and_native_return_values() {
        // Upstream also asserts `sort`/`reverse` return the receiver (`toBe`);
        // Rust methods return `Result<(), _>` so that half is dropped.
        let mut tracker = Tracker::new(json!({"values": [3, 1, 2]}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            assert_eq!(
                state.push(&[Seg::key("values")], vec![json!(4)]).unwrap(),
                4
            );
            assert_eq!(state.pop(&[Seg::key("values")]).unwrap(), Some(json!(4)));
            assert_eq!(
                state
                    .unshift(&[Seg::key("values")], vec![json!(0)])
                    .unwrap(),
                4
            );
            assert_eq!(state.shift(&[Seg::key("values")]).unwrap(), Some(json!(0)));
            assert_eq!(
                state
                    .splice(&[Seg::key("values")], 1, Some(1), vec![json!(5), json!(4)])
                    .unwrap(),
                vec![json!(1)]
            );
            state
                .sort_by(&[Seg::key("values")], |left, right| {
                    left.as_i64().unwrap().cmp(&right.as_i64().unwrap())
                })
                .unwrap();
            state.reverse(&[Seg::key("values")]).unwrap();
            state
                .fill(&[Seg::key("values")], json!(9), Some(1), Some(3))
                .unwrap();
            state
                .copy_within(&[Seg::key("values")], 1, 0, Some(2))
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value, json!({"values": [5, 5, 9, 2]}));
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn tracks_held_handles_after_reindex_and_suppresses_detached_writes() {
        // Upstream binds `held` before the unshift, observes it follow the
        // element to index 2, then removes it and observes further writes
        // suppressed. A Rust `NodeView` borrows the draft, so it cannot be
        // held across sibling mutations; the element is re-addressed at its
        // moved position, and detached handles are unrepresentable by
        // ownership. The reindex-following write and final state are ported.
        let mut tracker =
            Tracker::new(json!({"values": [{"value": "a"}, {"value": "b"}, {"value": "c"}]}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            let values = &[Seg::key("values")];
            state
                .unshift(values, vec![json!({"value": "front"})])
                .unwrap();
            {
                let mut held = state.at(&[Seg::key("values"), Seg::index(2)]).unwrap();
                held.set(&[], "value", json!("moved")).unwrap();
            }
            state.splice(values, 2, Some(1), vec![]).unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value,
            json!({"values": [{"value": "front"}, {"value": "a"}, {"value": "c"}]})
        );
        assert_eq!(replay(tracker.value(), &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn replays_introduced_moved_then_edited_array_descendants() {
        let mut tracker = Tracker::new(json!({"values": []}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            let values = &[Seg::key("values")];
            state
                .push(values, vec![json!({"id": 1, "nested": [1]})])
                .unwrap();
            state
                .unshift(values, vec![json!({"id": 0, "nested": []})])
                .unwrap();
            {
                let mut held = state.at(values).unwrap();
                let mut nested = held.at(&[Seg::index(1), Seg::key("nested")]).unwrap();
                nested.push(&[], vec![json!(2)]).unwrap();
            }
            state.reverse(values).unwrap();
            {
                let mut held = state.at(values).unwrap();
                let mut nested = held.at(&[Seg::index(0), Seg::key("nested")]).unwrap();
                nested.push(&[], vec![json!(3)]).unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.value,
            json!({"values": [{"id": 1, "nested": [1, 2, 3]}, {"id": 0, "nested": []}]})
        );
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn combines_movement_and_edits_with_permutation_plus_final_paths() {
        let mut tracker = Tracker::new(json!({
            "values": [
                {"rank": 3, "edited": 0},
                {"rank": 1, "edited": 0},
                {"rank": 2, "edited": 0},
            ],
        }));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state
                .sort_by(&[Seg::key("values")], |left, right| {
                    left["rank"]
                        .as_i64()
                        .unwrap()
                        .cmp(&right["rank"].as_i64().unwrap())
                })
                .unwrap();
            // The rank-3 element (upstream `held`) moved to index 2.
            state
                .set(&[Seg::key("values"), Seg::index(2)], "edited", json!(1))
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(
            prepared.ops[0],
            Op::Permute {
                path: vec![Seg::key("values")],
                permutation: vec![1, 2, 0],
            }
        );
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn keeps_comparator_writes() {
        // Upstream's comparator also performs structurally reentrant appends
        // through the parent draft; the Rust comparator receives only the
        // element values, so the reentrant half is not expressible. The
        // comparator-write half is ported with sort-algorithm-specific counts
        // relaxed to presence (upstream asserts exact counts for V8's sort).
        let mut tracker = Tracker::new(json!({
            "values": [{"rank": 2, "comparisons": 0}, {"rank": 1, "comparisons": 0}],
        }));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .sort_by(&[Seg::key("values")], |left, right| {
                    let order = left["rank"]
                        .as_i64()
                        .unwrap()
                        .cmp(&right["rank"].as_i64().unwrap());
                    for value in [left, right] {
                        let comparisons = value["comparisons"].as_i64().unwrap();
                        value["comparisons"] = json!(comparisons + 1);
                    }
                    order
                })
                .unwrap();
            change.prepare().unwrap()
        };
        let ranks: Vec<i64> = prepared.value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value["rank"].as_i64().unwrap())
            .collect();
        assert_eq!(ranks, vec![1, 2]);
        for value in prepared.value["values"].as_array().unwrap() {
            assert!(value["comparisons"].as_i64().unwrap() >= 1);
        }
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn applies_comparator_edits_to_nested_values() {
        // Upstream's comparator additionally unshifts a duplicate entry
        // (structurally reentrant, not expressible in the Rust comparator
        // signature); the duplicate-normalization half is skipped and the
        // element/nested write-back half is ported.
        let mut tracker = Tracker::new(json!({
            "values": [
                {"rank": 2, "edited": 0, "nested": {"value": 0}},
                {"rank": 1, "edited": 0, "nested": {"value": 0}},
            ],
        }));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .sort_by(&[Seg::key("values")], |left, right| {
                    let order = left["rank"]
                        .as_i64()
                        .unwrap()
                        .cmp(&right["rank"].as_i64().unwrap());
                    for value in [left, right] {
                        let rank = value["rank"].as_i64().unwrap();
                        value["edited"] = json!(rank * 10);
                        value["nested"]["value"] = json!(rank * 100);
                    }
                    order
                })
                .unwrap();
            change.prepare().unwrap()
        };
        let values = prepared.value["values"].as_array().unwrap();
        let ranks: Vec<i64> = values
            .iter()
            .map(|value| value["rank"].as_i64().unwrap())
            .collect();
        assert_eq!(ranks, vec![1, 2]);
        assert_eq!(values[0]["edited"], json!(10));
        assert_eq!(values[0]["nested"]["value"], json!(100));
        assert_eq!(values[1]["edited"], json!(20));
        assert_eq!(values[1]["nested"]["value"], json!(200));
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn folds_dense_child_and_direct_index_edits_into_one_full_array_operation() {
        let rows: Vec<JsonValue> = (0..1_000).map(|value| json!({"value": value})).collect();
        let mut rows_tracker = Tracker::new(json!({"rows": rows}));
        let rows_base = rows_tracker.value().clone();
        let rows_prepared = {
            let mut change = rows_tracker.begin_change();
            let mut state = change.state();
            for index in 0..1_000usize {
                let value = state
                    .get_index(&[Seg::key("rows")], index)
                    .unwrap()
                    .unwrap()["value"]
                    .as_i64()
                    .unwrap();
                state
                    .set(
                        &[Seg::key("rows"), Seg::index(index)],
                        "value",
                        json!(value + 1),
                    )
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(rows_prepared.ops.len(), 1);
        assert!(matches!(rows_prepared.ops[0], Op::Splice { .. }));
        assert_eq!(replay(&rows_base, &rows_prepared.ops), rows_prepared.value);
        rows_tracker.adopt(rows_prepared).unwrap();

        let values: Vec<JsonValue> = (0..1_000).map(|value| json!(value)).collect();
        let mut values_tracker = Tracker::new(json!({"values": values}));
        let values_base = values_tracker.value().clone();
        let values_prepared = {
            let mut change = values_tracker.begin_change();
            let mut state = change.state();
            for index in 0..600usize {
                state
                    .set_index(&[Seg::key("values")], index, json!(-(index as i64) - 1))
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(values_prepared.ops.len(), 1);
        assert!(matches!(values_prepared.ops[0], Op::Splice { .. }));
        assert_eq!(
            replay(&values_base, &values_prepared.ops),
            values_prepared.value
        );
        values_tracker.adopt(values_prepared).unwrap();
    }

    #[test]
    fn folds_only_a_deeply_nested_dense_array_region() {
        let rows: Vec<JsonValue> = (0..2_000)
            .map(|value| json!({"nested": {"value": value}}))
            .collect();
        let mut tracker = Tracker::new(json!({"rows": rows}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state
                .set(
                    &[Seg::key("rows"), Seg::index(0), Seg::key("nested")],
                    "value",
                    json!(-1),
                )
                .unwrap();
            for index in 500..1_000usize {
                state
                    .set(
                        &[Seg::key("rows"), Seg::index(index), Seg::key("nested")],
                        "value",
                        json!(-(index as i64)),
                    )
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(prepared.ops.len(), 2);
        let regional = prepared.ops.iter().find_map(|op| match op {
            Op::Splice {
                path,
                start,
                remove,
                items,
            } if path == &vec![Seg::key("rows")] => Some((*start, *remove, items.len())),
            _ => None,
        });
        assert_eq!(regional, Some((500, 500, 500)));
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn does_not_cache_emission_paths_for_nested_reserved_key_folds_covered_by_an_outer_region() {
        let mut rows: Vec<JsonValue> = (0..400).map(|_| json!({"flag": 0})).collect();
        let values: Vec<JsonValue> = (0..400).map(|value| json!(value)).collect();
        rows[100]["special"] = json!({"__proto__": {"values": values}});
        let mut tracker = Tracker::new(json!({"rows": rows}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            for index in 0..256usize {
                state
                    .set(&[Seg::key("rows"), Seg::index(index)], "flag", json!(1))
                    .unwrap();
            }
            for index in 0..256usize {
                state
                    .set_index(
                        &[
                            Seg::key("rows"),
                            Seg::index(100),
                            Seg::key("special"),
                            Seg::key("__proto__"),
                            Seg::key("values"),
                        ],
                        index,
                        json!(-(index as i64) - 1),
                    )
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(prepared.ops.len(), 1);
        match &prepared.ops[0] {
            Op::Splice {
                path,
                start,
                remove,
                ..
            } => {
                assert_eq!(path, &vec![Seg::key("rows")]);
                assert_eq!((*start, *remove), (0, 256));
            }
            other => panic!("expected splice, got {other:?}"),
        }
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn suppresses_nested_structural_and_leaf_operations_covered_by_an_outer_dense_region() {
        let rows: Vec<JsonValue> = (0..400)
            .map(|index| {
                if index == 100 {
                    json!({
                        "flag": 0,
                        "values": (0..400)
                            .map(|value| json!({"value": value}))
                            .collect::<Vec<_>>(),
                    })
                } else {
                    json!({"flag": 0, "values": []})
                }
            })
            .collect();
        let mut tracker = Tracker::new(json!({"rows": rows}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            for index in 0..256usize {
                state
                    .set(&[Seg::key("rows"), Seg::index(index)], "flag", json!(1))
                    .unwrap();
            }
            for index in 0..256usize {
                state
                    .set(
                        &[
                            Seg::key("rows"),
                            Seg::index(100),
                            Seg::key("values"),
                            Seg::index(index),
                        ],
                        "value",
                        json!(-(index as i64) - 1),
                    )
                    .unwrap();
            }
            state
                .push(
                    &[Seg::key("rows"), Seg::index(100), Seg::key("values")],
                    vec![json!({"value": 999})],
                )
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.ops.len(), 1);
        match &prepared.ops[0] {
            Op::Splice {
                path,
                start,
                remove,
                ..
            } => {
                assert_eq!(path, &vec![Seg::key("rows")]);
                assert_eq!((*start, *remove), (0, 256));
            }
            other => panic!("expected splice, got {other:?}"),
        }
        assert_eq!(
            prepared.value["rows"][100]["values"]
                .as_array()
                .unwrap()
                .len(),
            401
        );
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn emits_multiple_disjoint_dense_regions_and_preserves_operations_outside_their_boundaries() {
        let values: Vec<JsonValue> = (0..1_400).map(|value| json!({"value": value})).collect();
        let mut tracker = Tracker::new(json!({"values": values}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            for index in [97usize, 358, 897, 1_158] {
                state
                    .set(
                        &[Seg::key("values"), Seg::index(index)],
                        "value",
                        json!(-(index as i64) - 1),
                    )
                    .unwrap();
            }
            for index in 100..356usize {
                state
                    .set(
                        &[Seg::key("values"), Seg::index(index)],
                        "value",
                        json!(-(index as i64) - 1),
                    )
                    .unwrap();
            }
            for index in 900..1_156usize {
                state
                    .set(
                        &[Seg::key("values"), Seg::index(index)],
                        "value",
                        json!(-(index as i64) - 1),
                    )
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        let splices: Vec<(usize, usize)> = prepared
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::Splice {
                    path,
                    start,
                    remove,
                    ..
                } if path == &vec![Seg::key("values")] => Some((*start, *remove)),
                _ => None,
            })
            .collect();
        assert_eq!(splices, vec![(100, 256), (900, 256)]);
        for index in [97usize, 358, 897, 1_158] {
            assert!(prepared.ops.contains(&Op::Set {
                path: vec![Seg::key("values"), Seg::index(index), Seg::key("value")],
                value: json!(-(index as i64) - 1),
            }));
        }
        assert_eq!(prepared.ops.len(), 6);
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn normalizes_a_20000_operation_queue_transaction_to_two_operations() {
        let values: Vec<JsonValue> = (0..20_000).map(|value| json!({"value": value})).collect();
        let mut tracker = Tracker::new(json!({"values": values}));
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            for index in 0..10_000usize {
                state.shift(&[Seg::key("values")]).unwrap();
                state
                    .push(
                        &[Seg::key("values")],
                        vec![json!({"value": 20_000 + index})],
                    )
                    .unwrap();
            }
            change.prepare().unwrap()
        };
        assert_eq!(prepared.ops.len(), 2);
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn handles_adversarial_fragmentation_and_resolves_held_handles_without_piece_scans() {
        // Upstream holds child handles across 10,000 splices and resolves their
        // writes afterwards; Rust NodeViews cannot be held across sibling
        // mutations, so the held writes are addressed at the (unchanged) final
        // positions of the untouched odd indices — the same observable outcome.
        let size = 20_000usize;
        let values: Vec<JsonValue> = (0..size).map(|value| json!({"value": value})).collect();
        let mut tracker = Tracker::new(json!({"values": values}));
        let mut expected = tracker.value().clone();
        let base = tracker.value().clone();
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            for index in (0..size).step_by(2) {
                state
                    .splice(
                        &[Seg::key("values")],
                        index,
                        Some(1),
                        vec![json!({"value": -(index as i64) - 1})],
                    )
                    .unwrap();
                expected["values"][index] = json!({"value": -(index as i64) - 1});
            }
            for index in [1usize, 1_001, 5_001, 10_001, 15_001, 19_999] {
                state
                    .set(
                        &[Seg::key("values"), Seg::index(index)],
                        "value",
                        json!(index as i64 + 100_000),
                    )
                    .unwrap();
                let value = expected["values"][index]["value"].as_i64().unwrap();
                expected["values"][index]["value"] = json!(value + 100_000);
            }
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value, expected);
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn normalizes_restored_overrides_and_cancelled_structural_edits_to_no_operations() {
        let mut tracker = Tracker::new(json!({"values": [1, 2, 3]}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state.set_index(&[Seg::key("values")], 1, json!(9)).unwrap();
            state.set_index(&[Seg::key("values")], 1, json!(2)).unwrap();
            state.reverse(&[Seg::key("values")]).unwrap();
            state.reverse(&[Seg::key("values")]).unwrap();
            state.push(&[Seg::key("values")], vec![json!(4)]).unwrap();
            assert_eq!(state.pop(&[Seg::key("values")]).unwrap(), Some(json!(4)));
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value, json!({"values": [1, 2, 3]}));
        assert!(prepared.ops.is_empty());
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn handles_null_sparse_overrides_without_confusing_absence() {
        let values: Vec<JsonValue> = (0..10_000).map(|value| json!(value)).collect();
        let mut tracker = Tracker::new(json!({"values": values}));
        settle(&mut tracker, |draft| {
            draft
                .set_index(&[Seg::key("values")], 17, JsonValue::Null)
                .unwrap();
            draft
                .set_index(&[Seg::key("values")], 9_000, JsonValue::Null)
                .unwrap();
        });
        assert_eq!(tracker.value()["values"][17], JsonValue::Null);
        assert_eq!(tracker.value()["values"][9_000], JsonValue::Null);
    }

    #[test]
    fn supports_self_overlapping_fill_and_copywithin_by_value() {
        let mut tracker = Tracker::new(json!({"values": [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 3}]}));
        settle(&mut tracker, |draft| {
            draft
                .copy_within(&[Seg::key("values")], 1, 0, Some(3))
                .unwrap();
            let source = draft
                .get_index(&[Seg::key("values")], 1)
                .unwrap()
                .unwrap()
                .clone();
            draft
                .fill(&[Seg::key("values")], source, Some(0), Some(2))
                .unwrap();
            draft
                .set(&[Seg::key("values"), Seg::index(0)], "n", json!(9))
                .unwrap();
        });
        let values = tracker.value()["values"].as_array().unwrap();
        let ns: Vec<i64> = values
            .iter()
            .map(|value| value["n"].as_i64().unwrap())
            .collect();
        assert_eq!(ns, vec![9, 0, 1, 2]);
        assert_ne!(values[0], values[1]);
    }

    #[test]
    fn inserts_100_000_items_with_unshift() {
        let mut tracker = Tracker::new(json!({"values": [-1]}));
        let base = tracker.value().clone();
        let items: Vec<JsonValue> = (0..100_000).map(|value| json!(value)).collect();
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .unshift(&[Seg::key("values")], items)
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"].as_array().unwrap().len(), 100_001);
        assert_eq!(prepared.value["values"][99_999], json!(99_999));
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    #[test]
    fn inserts_100_000_items_with_splice() {
        let mut tracker = Tracker::new(json!({"values": [-1]}));
        let base = tracker.value().clone();
        let items: Vec<JsonValue> = (0..100_000).map(|value| json!(value)).collect();
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .splice(&[Seg::key("values")], 1, Some(0), items)
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["values"].as_array().unwrap().len(), 100_001);
        assert_eq!(prepared.value["values"][100_000], json!(99_999));
        assert_eq!(replay(&base, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
    }

    // ─── delta tracker randomized transactions ──────────────────────────────

    fn make_rng(mut seed: u32) -> impl FnMut() -> f64 {
        move || {
            seed = seed.wrapping_add(0x6d2b_79f5);
            let mut value = seed ^ (seed >> 15);
            value = value.wrapping_mul(1 | seed);
            value = value.wrapping_add((value ^ (value >> 7)).wrapping_mul(61 | value)) ^ value;
            (value ^ (value >> 14)) as f64 / 4_294_967_296.0
        }
    }

    fn random_item(value: i64) -> JsonValue {
        json!({"id": value, "score": value % 7})
    }

    /// Upstream `mutate` against the detached expected document.
    fn expected_mutate(document: &mut JsonValue, choice: usize, value: i64) {
        match choice {
            0 => {
                let text = document["text"].as_str().unwrap().to_string();
                document["text"] = json!(format!("{text}-{value}"));
            }
            1 => document["values"]
                .as_array_mut()
                .unwrap()
                .push(random_item(value)),
            2 => document["values"]
                .as_array_mut()
                .unwrap()
                .insert(0, random_item(value)),
            3 => {
                if !document["values"].as_array().unwrap().is_empty() {
                    document["values"].as_array_mut().unwrap().remove(0);
                }
            }
            4 => {
                document["values"].as_array_mut().unwrap().pop();
            }
            5 => {
                let length = document["values"].as_array().unwrap().len();
                let index = if length == 0 {
                    0
                } else {
                    (value as usize) % (length + 1)
                };
                let remove = if length == 0 { 0 } else { (value as usize) % 2 };
                let array = document["values"].as_array_mut().unwrap();
                let end = (index + remove).min(array.len());
                array.splice(index..end, std::iter::once(random_item(value)));
            }
            6 => document["values"].as_array_mut().unwrap().reverse(),
            7 => document["values"]
                .as_array_mut()
                .unwrap()
                .sort_by_key(|item| item["id"].as_i64().unwrap()),
            8 => {
                let length = document["values"].as_array().unwrap().len();
                if length > 0 {
                    document["values"][(value as usize) % length]["score"] = json!(value);
                }
            }
            9 => {
                let revision = document["meta"]["revision"].as_i64().unwrap();
                document["meta"]["revision"] = json!(revision + 1);
                document["meta"]["label"] = json!(format!("r-{value}"));
            }
            _ => {
                document["meta"].as_object_mut().unwrap().remove("label");
            }
        }
    }

    /// Upstream `mutate` against the transaction draft.
    fn draft_mutate(state: &mut Draft, choice: usize, value: i64) -> R {
        let values = &[Seg::key("values")];
        match choice {
            0 => {
                // Overlay writes are reliably visible through NodeView path
                // reads, so `document.text` is read through a root handle.
                let root = state.at(&[]).unwrap();
                let text = root
                    .get(&[Seg::key("text")])
                    .unwrap()
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                drop(root);
                state.set(&[], "text", json!(format!("{text}-{value}")))?;
            }
            1 => {
                state.push(values, vec![random_item(value)])?;
            }
            2 => {
                state.unshift(values, vec![random_item(value)])?;
            }
            3 => {
                if state.length(values).unwrap() > 0 {
                    state.shift(values)?;
                }
            }
            4 => {
                state.pop(values)?;
            }
            5 => {
                let length = state.length(values).unwrap();
                let index = if length == 0 {
                    0
                } else {
                    (value as usize) % (length + 1)
                };
                let remove = if length == 0 {
                    Some(0)
                } else {
                    Some((value as usize) % 2)
                };
                state.splice(values, index, remove, vec![random_item(value)])?;
            }
            6 => {
                state.reverse(values)?;
            }
            7 => {
                state.sort_by(values, |left, right| {
                    left["id"]
                        .as_i64()
                        .unwrap()
                        .cmp(&right["id"].as_i64().unwrap())
                })?;
            }
            8 => {
                let length = state.length(values).unwrap();
                if length > 0 {
                    state.set(
                        &[Seg::key("values"), Seg::index((value as usize) % length)],
                        "score",
                        json!(value),
                    )?;
                }
            }
            9 => {
                let meta = state.at(&[Seg::key("meta")]).unwrap();
                let revision = meta
                    .get(&[Seg::key("revision")])
                    .unwrap()
                    .unwrap()
                    .as_i64()
                    .unwrap();
                drop(meta);
                state.set(&[Seg::key("meta")], "revision", json!(revision + 1))?;
                state.set(&[Seg::key("meta")], "label", json!(format!("r-{value}")))?;
            }
            _ => {
                state.delete(&[Seg::key("meta")], "label")?;
            }
        }
        Ok(())
    }

    #[test]
    fn matches_policy_detached_replay_and_adopted_state_after_multi_operation_transactions() {
        // `expectAliasFree` holds trivially for owned `JsonValue` trees and the
        // `isProxy` checks have no counterpart; the policy/replay/adopt
        // contract is asserted in full.
        for seed in 1u32..=40 {
            let mut rng = make_rng(seed);
            let initial = json!({
                "values": (0..4).map(|id| json!({"id": id, "score": 0})).collect::<Vec<_>>(),
                "text": "start",
                "meta": {"revision": 0},
            });
            let mut tracker = Tracker::new(initial);
            let mut expected = tracker.value().clone();
            for transaction in 0..25 {
                let base = tracker.value().clone();
                let prepared = {
                    let mut change = tracker.begin_change();
                    let mut state = change.state();
                    for operation in 0..5 {
                        let choice = (rng() * 11.0) as usize;
                        let value = seed as i64 * 10_000 + transaction * 10 + operation;
                        expected_mutate(&mut expected, choice, value);
                        draft_mutate(&mut state, choice, value)
                            .unwrap_or_else(|error| panic!("seed {seed} choice {choice}: {error}"));
                    }
                    change.prepare().unwrap()
                };
                assert_eq!(
                    prepared.value, expected,
                    "policy seed {seed} transaction {transaction}"
                );
                assert_eq!(
                    replay(&base, &prepared.ops),
                    prepared.value,
                    "replay seed {seed} transaction {transaction}"
                );
                tracker.adopt(prepared).unwrap();
                assert_eq!(
                    tracker.value(),
                    &expected,
                    "adopt seed {seed} transaction {transaction}"
                );
            }
        }
    }

    // ─── delta tracker object emission scaling ──────────────────────────────

    #[test]
    fn orders_many_reverse_depth_object_edits_through_the_bounded_fallback() {
        // Upstream holds one child handle per depth level; Rust NodeViews share
        // one draft borrow, so each depth is addressed by path (same emission
        // workload and order).
        let mut initial = json!({"value": 0});
        for _ in 0..512 {
            initial = json!({"value": 0, "next": initial});
        }
        let mut tracker = Tracker::new(initial.clone());
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            let mut paths: Vec<Vec<Seg>> = Vec::new();
            let mut path: Vec<Seg> = Vec::new();
            loop {
                paths.push(path.clone());
                let has_next = state.get(&path).unwrap().unwrap().get("next").is_some();
                if has_next {
                    path.push(Seg::key("next"));
                } else {
                    break;
                }
            }
            let depth = paths.len();
            for (index, target) in paths.iter().enumerate().rev() {
                state.set(target, "value", json!(index + 1)).unwrap();
            }
            let prepared = change.prepare().unwrap();
            assert_eq!(prepared.ops.len(), depth);
            prepared
        };
        assert_eq!(replay(&initial, &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value()["value"], json!(1));
        assert_eq!(tracker.value()["next"]["next"]["value"], json!(3));
    }

    // ─── delta tracker security and storage-style cloning ───────────────────

    // "defines own properties without invoking inherited setters": inherited
    // setter traps require a prototype chain. Skipped.

    #[test]
    fn handles_reserved_own_keys_without_prototype_pollution() {
        let initial =
            serde_json::from_str::<JsonValue>(r#"{"safe":{"__proto__":{"value":1}}}"#).unwrap();
        let mut tracker = Tracker::new(initial);
        let prepared = {
            let mut change = tracker.begin_change();
            change
                .state()
                .set(
                    &[Seg::key("safe"), Seg::key("__proto__")],
                    "value",
                    json!(2),
                )
                .unwrap();
            change.prepare().unwrap()
        };
        assert_eq!(prepared.value["safe"]["__proto__"]["value"], json!(2));
        assert_eq!(replay(tracker.value(), &prepared.ops), prepared.value);
        tracker.adopt(prepared).unwrap();
        assert_eq!(tracker.value()["safe"]["__proto__"]["value"], json!(2));
    }

    #[test]
    fn supports_a_memory_storage_style_recursive_clone_of_prepared_value() {
        // Upstream hand-rolls a descriptor-preserving deep clone; `JsonValue`
        // clone is exactly that tree copy.
        let mut tracker = Tracker::new(json!({"rows": [{"value": 1}]}));
        let prepared = {
            let mut change = tracker.begin_change();
            let mut state = change.state();
            state
                .set(&[Seg::key("rows"), Seg::index(0)], "value", json!(2))
                .unwrap();
            state
                .push(&[Seg::key("rows")], vec![json!({"value": 3})])
                .unwrap();
            change.prepare().unwrap()
        };
        let storage_clone = prepared.value.clone();
        assert_eq!(storage_clone, json!({"rows": [{"value": 2}, {"value": 3}]}));
        tracker.adopt(prepared).unwrap();
    }
}
