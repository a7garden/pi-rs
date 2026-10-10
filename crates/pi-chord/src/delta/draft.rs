//! Port of `packages/chord/src/delta/tracker.ts` — overlay half.
//!
//! Upstream exposes a transaction-scoped mutable view through JS `Proxy`
//! objects (`Draft<T>`): property access on a proxy resolves through the
//! overlay and returns child proxies. Rust has no proxies, so the overlay is
//! driven through an explicit path-addressed API ([`Draft`]) plus bound child
//! handles ([`NodeView`], upstream's mid-transaction child proxies). This file
//! holds the overlay machinery — node graph, array entry plan, and the
//! `emitOperations` port; [`crate::delta::tracker`] holds the
//! [`crate::delta::Tracker`] / [`crate::delta::Change`] /
//! [`crate::delta::Prepared`] lifecycle built on top.
//!
//! Upstream semantics preserved here:
//! - Nodes record how they were reached (base route, write slot, base entry,
//!   inserted entry). Emission recomputes each dirty node's root-relative path
//!   and **skips nodes whose final overlay state no longer routes to them**
//!   (deleted parents, shadowed writes, spliced-out entries). Writes through
//!   such detached handles still mutate the node's own overlay state and reads
//!   through the handle still return it — only emission drops them.
//! - First-touch dirty order, reserved-segment forced folds (including folding
//!   the reserved-path ancestor), dense-region folding (>= 256 overridden
//!   indices in a dense run with gap <= 1 and count * 2 >= length), readded-key
//!   `d`-then-write emission, string append/truncate minimization through
//!   [`crate::delta::overlap`], remove runs (descending), retention permutation,
//!   insert runs (ascending, reading CURRENT entry values so mutations under
//!   inserted elements are included), the simple-object fast path (plain
//!   objects only, no arrays in any ancestor chain, no reserved segments, <=
//!   128 nodes, stable sort by path depth), and the 4_096-operation overflow
//!   fallback to a root replacement.
//!
//! Documented deviations (mechanical; PORT_MAP.md rule 3):
//! - Pointer identity does not exist for [`JsonValue`]; every upstream
//!   `===`/`WeakMap` comparison becomes structural equality. Value-replacing
//!   writes capture a private clone at node-creation time so detached handles
//!   keep reading the value they were bound to (upstream keeps the same object
//!   via `context.bases`).
//! - The prepared `value` is always computed as `base.clone()` with the ops
//!   applied in place. Upstream picks between `cloneNode(root)` and
//!   `materializeOperations(base, ops)`; both replay to exactly the ops'
//!   result, and `apply` over the base clone is that result by construction.
//! - The array piece-tree becomes a symbolic entry list
//!   ([`Entry::Base`] | [`Entry::Insert`]) plus an inserted-payload arena and
//!   override slots. Per-operation asymptotics differ; every observable op
//!   sequence matches.
//! - Inserted runs are never tail-merged into the previous source (a pure
//!   allocation optimization upstream) and the post-sort
//!   `deduplicateArrayEntries` pass is unreachable: the comparator here
//!   receives extracted element values, so it cannot structurally mutate the
//!   array mid-sort. Comparator edits land by writing mutated elements back to
//!   their slots before reordering (upstream keeps them via child-proxy
//!   writes folded in at emission).
//! - Upstream allows several competing `Change`s per tracker; the Rust API
//!   admits one open `Change` at a time
//!   (`begin_change(&mut self) -> Change<'_>`), so competing-context staleness
//!   reduces to the `adopt` checks.
//! - Reading a property that does not exist on a clean route returns
//!   `Ok(None)` for a final segment and [`DeltaError::Unresolvable`] for an
//!   intermediate one (upstream throws on the intermediate case when the next
//!   property is read). `"length"` on array overlays is read through
//!   [`Draft::length`], mirroring the upstream `length` getter.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::{is_reserved_segment, DeltaError, Op, Seg};
use crate::json::JsonValue;

pub(super) const MAX_DELTA_OPERATIONS: usize = 4_096;
const MAX_SIMPLE_OBJECT_NODES: usize = 128;
const DEFAULT_OVERLAP_SCAN: usize = 65_536;

pub(super) type NodeId = usize;

/// How a node's wrapped container was reached. Doubles as upstream's
/// `parentKind`/`parentKey`/`parentSource`/`parentPlacement` bundle plus the
/// captured value for routes that a later write can replace.
#[derive(Debug, Clone)]
pub(super) enum NodeOrigin {
    /// The context root.
    Root,
    /// Object child reached through the parent's raw value at `key`.
    BaseKey { parent: NodeId, key: String },
    /// Object child reached through a parent write slot at `key`.
    WriteSlot {
        parent: NodeId,
        key: String,
        captured: JsonValue,
    },
    /// Array child over `base_index`: wraps the base element (`captured: None`)
    /// or the override that was current when the node was created.
    BaseEntry {
        parent: NodeId,
        base_index: usize,
        captured: Option<JsonValue>,
    },
    /// Array child over an inserted payload (or its current override).
    InsertEntry {
        parent: NodeId,
        source: usize,
        index: usize,
        captured: JsonValue,
    },
}

impl NodeOrigin {
    fn parent(&self) -> Option<NodeId> {
        match self {
            NodeOrigin::Root => None,
            NodeOrigin::BaseKey { parent, .. }
            | NodeOrigin::WriteSlot { parent, .. }
            | NodeOrigin::BaseEntry { parent, .. }
            | NodeOrigin::InsertEntry { parent, .. } => Some(*parent),
        }
    }

    /// Upstream `parentPlacement`: the route delivered a placement (a written
    /// or inserted value) rather than untouched base structure.
    fn is_placement(&self) -> bool {
        matches!(
            self,
            NodeOrigin::WriteSlot { .. }
                | NodeOrigin::InsertEntry { .. }
                | NodeOrigin::BaseEntry {
                    captured: Some(_),
                    ..
                }
        )
    }
}

/// A written value plus the node (if any) currently bound to it. `node` is
/// cleared whenever the value is replaced or the slot is removed, detaching the
/// old node from emission routing.
#[derive(Debug, Clone)]
pub(super) struct Slot {
    pub(super) value: JsonValue,
    pub(super) node: Option<NodeId>,
}

#[derive(Debug, Default)]
pub(super) struct ObjectOverlay {
    /// Insertion-ordered writes (upstream `writeKey` + `writes` Map).
    pub(super) writes: Vec<(String, Slot)>,
    /// Insertion-ordered deletes (upstream `deleteKey` + `deletes` Set).
    pub(super) deletes: Vec<String>,
    /// Keys deleted and then re-written while present in the base (upstream
    /// `readded` Set); emitted as an explicit `d` before the write so replay
    /// preserves string-key insertion order.
    pub(super) readded: Vec<String>,
    /// Key -> node currently bound to that route (base or write).
    child_nodes: HashMap<String, NodeId>,
}

/// One symbolic array position: a retained base element or an inserted value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Entry {
    Base(usize),
    Insert { source: usize, index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ChildRoute {
    Base(usize),
    Insert(usize, usize),
}

#[derive(Debug, Default)]
pub(super) struct ArrayOverlay {
    /// Symbolic entry list: the overlay's logical order.
    pub(super) entries: Vec<Entry>,
    /// Inserted payloads by source id (upstream `InsertSource.refs`).
    pub(super) sources: Vec<Vec<JsonValue>>,
    /// Base-entry overrides, insertion-ordered (upstream `baseOverrides` Map).
    pub(super) base_overrides: Vec<(usize, Slot)>,
    /// Inserted-entry overrides by source, insertion-ordered (upstream
    /// `insertOverrides`).
    pub(super) insert_overrides: Vec<(usize, Vec<(usize, Slot)>)>,
    /// True once entries were inserted, removed, or reordered.
    pub(super) structural: bool,
    child_nodes: HashMap<ChildRoute, NodeId>,
}

impl ArrayOverlay {
    /// A fresh overlay over a base array covers all of it (upstream
    /// `arrayOverlay` seeds the piece tree with the whole base).
    fn seeded(raw: &JsonValue) -> Option<ArrayOverlay> {
        let array = raw.as_array()?;
        Some(ArrayOverlay {
            entries: (0..array.len()).map(Entry::Base).collect(),
            ..ArrayOverlay::default()
        })
    }
}

#[derive(Debug)]
pub(super) struct OverlayNode {
    pub(super) origin: NodeOrigin,
    pub(super) object: Option<ObjectOverlay>,
    pub(super) array: Option<ArrayOverlay>,
    pub(super) dirty: bool,
    pub(super) subtree_dirty: bool,
}

/// Dense-region candidate set (upstream `DenseCandidates`).
#[derive(Debug)]
struct DenseCandidates {
    indices: Vec<usize>,
    bits: Option<Vec<u8>>,
    length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DenseRegion {
    start: usize,
    length: usize,
}

/// The transaction-scoped overlay: base value, node arena, first-touch dirty
/// order, and the `emitOperations` port.
pub(super) struct OverlayContext {
    pub(super) owner: u64,
    /// `prepared.base`: the value the change was opened against.
    pub(super) base_root: JsonValue,
    /// The value the root node wraps (the base, or the replacement payload).
    root_value: JsonValue,
    pub(super) root: NodeId,
    pub(super) nodes: Vec<OverlayNode>,
    pub(super) dirty: Vec<NodeId>,
    pub(super) base_revision: u64,
    pub(super) replacement: bool,
}

impl OverlayContext {
    pub(super) fn new(
        owner: u64,
        base: JsonValue,
        root_value: JsonValue,
        base_revision: u64,
        replacement: bool,
    ) -> Self {
        let mut context = OverlayContext {
            owner,
            base_root: base,
            root_value,
            root: 0,
            nodes: Vec::new(),
            dirty: Vec::new(),
            base_revision,
            replacement,
        };
        let root_object = context.root_value.is_object().then(ObjectOverlay::default);
        let root_array = ArrayOverlay::seeded(&context.root_value);
        context.root = context.push_node(OverlayNode {
            origin: NodeOrigin::Root,
            object: root_object,
            array: root_array,
            dirty: false,
            subtree_dirty: false,
        });
        context
    }

    fn push_node(&mut self, node: OverlayNode) -> NodeId {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    // ─── Raw value access ───────────────────────────────────────────────────

    /// The immutable value the node wraps: the base element it was created for,
    /// or the captured write/insert/override payload.
    fn node_raw(&self, nid: NodeId) -> &JsonValue {
        match &self.nodes[nid].origin {
            NodeOrigin::Root => &self.root_value,
            NodeOrigin::BaseKey { parent, key } => self
                .node_raw(*parent)
                .get(key)
                .expect("node route existed at creation"),
            NodeOrigin::BaseEntry {
                parent,
                base_index,
                captured: None,
            } => &self.node_raw(*parent).as_array().expect("array route")[*base_index],
            NodeOrigin::WriteSlot { captured, .. }
            | NodeOrigin::BaseEntry {
                captured: Some(captured),
                ..
            }
            | NodeOrigin::InsertEntry { captured, .. } => captured,
        }
    }

    // ─── Object overlay helpers ─────────────────────────────────────────────

    fn object(&self, nid: NodeId) -> &ObjectOverlay {
        self.nodes[nid].object.as_ref().expect("object node")
    }

    fn is_array_node(&self, nid: NodeId) -> bool {
        self.nodes[nid].array.is_some()
    }

    fn write_slot(&self, nid: NodeId, key: &str) -> Option<&Slot> {
        self.object(nid)
            .writes
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, slot)| slot)
    }

    fn object_deleted(&self, nid: NodeId, key: &str) -> bool {
        self.object(nid)
            .deletes
            .iter()
            .any(|existing| existing == key)
    }

    /// Upstream `objectHas`: overlay presence. serde objects have own
    /// properties only, so no prototype chain is reachable.
    fn object_has(&self, nid: NodeId, key: &str) -> bool {
        !self.object_deleted(nid, key)
            && (self.write_slot(nid, key).is_some() || self.object_base_has(nid, key))
    }

    fn object_base_has(&self, nid: NodeId, key: &str) -> bool {
        self.node_raw(nid).get(key).is_some()
    }

    /// Overlay-aware key ordering (upstream `ownKeys` object half): numeric
    /// indices ascending first, then string keys; readded keys surface at their
    /// re-write position rather than their base position. JS canonicalizes
    /// integer-like keys to numeric-first order at property creation; serde
    /// maps preserve text order, so the partition always runs.
    fn object_keys(&self, nid: NodeId) -> Vec<String> {
        let base_keys: Vec<String> = match self.node_raw(nid) {
            JsonValue::Object(object) => object.keys().cloned().collect(),
            _ => Vec::new(),
        };
        let overlay = self.object(nid);
        let mut keys: Vec<String> = base_keys
            .into_iter()
            .filter(|key| {
                !self.object_deleted(nid, key) && !overlay.readded.iter().any(|k| k == key)
            })
            .collect();
        for (key, _) in &overlay.writes {
            if !keys.contains(key) {
                keys.push(key.clone());
            }
        }
        let mut indices: Vec<(usize, String)> = Vec::new();
        let mut strings: Vec<String> = Vec::new();
        for key in keys {
            match array_index(&key) {
                Some(index) => indices.push((index, key)),
                None => strings.push(key),
            }
        }
        indices.sort_by_key(|(index, _)| *index);
        indices
            .into_iter()
            .map(|(_, key)| key)
            .chain(strings)
            .collect()
    }

    // ─── Array overlay helpers ──────────────────────────────────────────────

    fn array(&self, nid: NodeId) -> &ArrayOverlay {
        self.nodes[nid].array.as_ref().expect("array node")
    }

    fn array_len(&self, nid: NodeId) -> usize {
        self.array(nid).entries.len()
    }

    fn base_override_slot(&self, nid: NodeId, base_index: usize) -> Option<&Slot> {
        self.array(nid)
            .base_overrides
            .iter()
            .find(|(index, _)| *index == base_index)
            .map(|(_, slot)| slot)
    }

    fn insert_override_slot(&self, nid: NodeId, source: usize, index: usize) -> Option<&Slot> {
        self.array(nid)
            .insert_overrides
            .iter()
            .find(|(s, _)| *s == source)
            .and_then(|(_, overrides)| overrides.iter().find(|(i, _)| *i == index))
            .map(|(_, slot)| slot)
    }

    /// Current raw value behind one entry: override slot or base/inserted value,
    /// with the node bound to that value (if any).
    fn entry_value(&self, nid: NodeId, entry: &Entry) -> (&JsonValue, Option<NodeId>) {
        match entry {
            Entry::Base(base_index) => match self.base_override_slot(nid, *base_index) {
                Some(slot) => (&slot.value, slot.node),
                None => (
                    &self.node_raw(nid).as_array().expect("array route")[*base_index],
                    self.array(nid)
                        .child_nodes
                        .get(&ChildRoute::Base(*base_index))
                        .copied(),
                ),
            },
            Entry::Insert { source, index } => {
                match self.insert_override_slot(nid, *source, *index) {
                    Some(slot) => (&slot.value, slot.node),
                    None => (
                        &self.array(nid).sources[*source][*index],
                        self.array(nid)
                            .child_nodes
                            .get(&ChildRoute::Insert(*source, *index))
                            .copied(),
                    ),
                }
            }
        }
    }

    /// Current entry value with descendant node writes folded in (upstream
    /// `entryValueAt` + `cloneStored`).
    fn entry_value_folded(&self, nid: NodeId, entry: &Entry) -> JsonValue {
        let (value, child) = self.entry_value(nid, entry);
        self.fold_value(value, child)
    }

    fn fold_value(&self, value: &JsonValue, child: Option<NodeId>) -> JsonValue {
        match child {
            Some(child) if self.needs_materialize(child) => self.materialize_node(child),
            _ => value.clone(),
        }
    }

    fn needs_materialize(&self, nid: NodeId) -> bool {
        let node = &self.nodes[nid];
        node.dirty || node.subtree_dirty
    }

    fn logical_index_for_base(&self, nid: NodeId, base_index: usize) -> Option<usize> {
        self.array(nid)
            .entries
            .iter()
            .position(|entry| matches!(entry, Entry::Base(index) if *index == base_index))
    }

    fn logical_index_for_insert(&self, nid: NodeId, source: usize, index: usize) -> Option<usize> {
        self.array(nid)
            .entries
            .iter()
            .position(|entry| matches!(entry, Entry::Insert { source: s, index: i } if *s == source && *i == index))
    }

    // ─── Mutation bookkeeping ───────────────────────────────────────────────

    /// Upstream `markDirty`: first-touch order for the node, `subtreeDirty` up
    /// the parent chain.
    fn mark_dirty(&mut self, nid: NodeId) {
        if !self.nodes[nid].dirty {
            self.nodes[nid].dirty = true;
            self.dirty.push(nid);
        }
        let mut current = self.nodes[nid].origin.parent();
        while let Some(parent) = current {
            self.nodes[parent].subtree_dirty = true;
            current = self.nodes[parent].origin.parent();
        }
    }

    fn set_write_slot(&mut self, nid: NodeId, key: String, value: JsonValue) {
        let overlay = self.nodes[nid].object.as_mut().expect("object node");
        // Replacing the payload detaches any node bound to the old value.
        overlay.child_nodes.remove(&key);
        if let Some(position) = overlay
            .writes
            .iter()
            .position(|(existing, _)| *existing == key)
        {
            overlay.writes[position].1 = Slot { value, node: None };
        } else {
            overlay.writes.push((key, Slot { value, node: None }));
        }
    }

    fn remove_write_slot(&mut self, nid: NodeId, key: &str) {
        let overlay = self.nodes[nid].object.as_mut().expect("object node");
        if let Some(position) = overlay
            .writes
            .iter()
            .position(|(existing, _)| existing == key)
        {
            overlay.writes.remove(position);
        }
        overlay.child_nodes.remove(key);
    }

    fn record_deletion(&mut self, nid: NodeId, key: &str) {
        let overlay = self.nodes[nid].object.as_mut().expect("object node");
        if !overlay.deletes.iter().any(|existing| existing == key) {
            overlay.deletes.push(key.to_string());
        }
    }

    fn upsert_base_override(&mut self, nid: NodeId, base_index: usize, value: JsonValue) {
        let overlay = self.nodes[nid].array.as_mut().expect("array node");
        overlay.child_nodes.remove(&ChildRoute::Base(base_index));
        if let Some(position) = overlay
            .base_overrides
            .iter()
            .position(|(index, _)| *index == base_index)
        {
            overlay.base_overrides[position].1 = Slot { value, node: None };
        } else {
            overlay
                .base_overrides
                .push((base_index, Slot { value, node: None }));
        }
    }

    fn remove_base_override(&mut self, nid: NodeId, base_index: usize) {
        let overlay = self.nodes[nid].array.as_mut().expect("array node");
        if let Some(position) = overlay
            .base_overrides
            .iter()
            .position(|(index, _)| *index == base_index)
        {
            overlay.base_overrides.remove(position);
        }
    }

    fn upsert_insert_override(
        &mut self,
        nid: NodeId,
        source: usize,
        index: usize,
        value: JsonValue,
    ) {
        let overlay = self.nodes[nid].array.as_mut().expect("array node");
        overlay
            .child_nodes
            .remove(&ChildRoute::Insert(source, index));
        if !overlay.insert_overrides.iter().any(|(s, _)| *s == source) {
            overlay.insert_overrides.push((source, Vec::new()));
        }
        let overrides = &mut overlay
            .insert_overrides
            .iter_mut()
            .find(|(s, _)| *s == source)
            .expect("just ensured")
            .1;
        if let Some(position) = overrides
            .iter()
            .position(|(existing, _)| *existing == index)
        {
            overrides[position].1 = Slot { value, node: None };
        } else {
            overrides.push((index, Slot { value, node: None }));
        }
    }

    fn remove_insert_override(&mut self, nid: NodeId, source: usize, index: usize) {
        let overlay = self.nodes[nid].array.as_mut().expect("array node");
        let Some(position) = overlay
            .insert_overrides
            .iter()
            .position(|(s, _)| *s == source)
        else {
            return;
        };
        let overrides = &mut overlay.insert_overrides[position].1;
        if let Some(at) = overrides
            .iter()
            .position(|(existing, _)| *existing == index)
        {
            overrides.remove(at);
        }
        if overrides.is_empty() {
            overlay.insert_overrides.remove(position);
        }
    }

    fn new_source(&mut self, nid: NodeId, items: Vec<JsonValue>) -> usize {
        let overlay = self.nodes[nid].array.as_mut().expect("array node");
        let source = overlay.sources.len();
        overlay.sources.push(items);
        source
    }

    fn mark_structural(&mut self, nid: NodeId) {
        self.nodes[nid]
            .array
            .as_mut()
            .expect("array node")
            .structural = true;
    }

    fn push_insert_entries(&mut self, nid: NodeId, at: usize, source: usize, count: usize) {
        let overlay = self.nodes[nid].array.as_mut().expect("array node");
        let entries: Vec<Entry> = (0..count)
            .map(|index| Entry::Insert { source, index })
            .collect();
        overlay.entries.splice(at..at, entries);
    }

    // ─── Node creation / resolution ─────────────────────────────────────────

    /// Resolve `path` from `start`, creating nodes for every traversed
    /// container (upstream proxies are created on property access).
    pub(super) fn node_for(&mut self, start: NodeId, path: &[Seg]) -> Result<NodeId, DeltaError> {
        let mut current = start;
        for (depth, segment) in path.iter().enumerate() {
            current = self
                .step_to_node(current, segment)
                .map_err(|error| match error {
                    StepError::Missing => DeltaError::Unresolvable {
                        path: serde_json::to_string(&super::path_to_json(&path[..=depth]))
                            .unwrap_or_else(|_| "[]".to_string()),
                    },
                    StepError::Other(error) => error,
                })?;
        }
        Ok(current)
    }

    fn step_to_node(&mut self, current: NodeId, segment: &Seg) -> Result<NodeId, StepError> {
        if self.nodes[current].object.is_some() {
            return self.step_to_object_child(current, segment);
        }
        if self.nodes[current].array.is_some() {
            return self.step_to_array_child(current, segment);
        }
        // Reading a property through a non-container.
        Err(StepError::Missing)
    }

    fn step_to_object_child(
        &mut self,
        current: NodeId,
        segment: &Seg,
    ) -> Result<NodeId, StepError> {
        let key = object_key_form(segment);
        // Route through a write slot when one exists.
        let written = self
            .object(current)
            .writes
            .iter()
            .find(|(existing, _)| *existing == key)
            .map(|(_, slot)| (slot.value.clone(), slot.node));
        if let Some((value, bound)) = written {
            if let Some(existing) = self.object(current).child_nodes.get(&key).copied() {
                if bound == Some(existing) {
                    return Ok(existing);
                }
            }
            let node_object = value.is_object().then(ObjectOverlay::default);
            let node_array = ArrayOverlay::seeded(&value);
            let nid = self.push_node(OverlayNode {
                origin: NodeOrigin::WriteSlot {
                    parent: current,
                    key: key.clone(),
                    captured: value,
                },
                object: node_object,
                array: node_array,
                dirty: false,
                subtree_dirty: false,
            });
            let overlay = self.nodes[current].object.as_mut().expect("object node");
            overlay.child_nodes.insert(key.clone(), nid);
            let slot = &mut overlay
                .writes
                .iter_mut()
                .find(|(existing, _)| *existing == key)
                .expect("write slot exists")
                .1;
            slot.node = Some(nid);
            return Ok(nid);
        }
        if self.object_deleted(current, &key) {
            return Err(StepError::Missing);
        }
        let is_container = self
            .node_raw(current)
            .get(&key)
            .is_some_and(is_value_container);
        if !is_container {
            return Err(StepError::Missing);
        }
        if let Some(existing) = self.object(current).child_nodes.get(&key).copied() {
            return Ok(existing);
        }
        let child_raw = self
            .node_raw(current)
            .get(&key)
            .expect("container checked above");
        let child_object = child_raw.is_object().then(ObjectOverlay::default);
        let child_array = ArrayOverlay::seeded(child_raw);
        let nid = self.push_node(OverlayNode {
            origin: NodeOrigin::BaseKey {
                parent: current,
                key: key.clone(),
            },
            object: child_object,
            array: child_array,
            dirty: false,
            subtree_dirty: false,
        });
        self.nodes[current]
            .object
            .as_mut()
            .expect("object node")
            .child_nodes
            .insert(key, nid);
        Ok(nid)
    }

    fn step_to_array_child(&mut self, current: NodeId, segment: &Seg) -> Result<NodeId, StepError> {
        let entry = {
            let overlay = self.array(current);
            resolve_entry(overlay, segment).ok_or(StepError::Missing)?
        };
        let (value, bound) = self.entry_value(current, &entry);
        if !is_value_container(value) {
            return Err(StepError::Missing);
        }
        // `bound` is Some exactly when the route's node is still current.
        if let Some(existing) = bound {
            return Ok(existing);
        }
        let captured = value.clone();
        let child_is_object = value.is_object();
        let child_array = ArrayOverlay::seeded(value);
        let route = child_route(&entry);
        let origin = match entry {
            Entry::Base(base_index) => NodeOrigin::BaseEntry {
                parent: current,
                base_index,
                captured: self
                    .base_override_slot(current, base_index)
                    .map(|slot| slot.value.clone()),
            },
            Entry::Insert { source, index } => NodeOrigin::InsertEntry {
                parent: current,
                source,
                index,
                captured,
            },
        };
        let nid = self.push_node(OverlayNode {
            origin,
            object: child_is_object.then(ObjectOverlay::default),
            array: child_array,
            dirty: false,
            subtree_dirty: false,
        });
        // Bind the node to whichever slot it wraps.
        match entry {
            Entry::Base(base_index) => {
                if let Some(slot) = self.nodes[current]
                    .array
                    .as_mut()
                    .expect("array node")
                    .base_overrides
                    .iter_mut()
                    .find(|(index, _)| *index == base_index)
                    .map(|(_, slot)| slot)
                {
                    slot.node = Some(nid);
                }
            }
            Entry::Insert { source, index } => {
                if let Some(slot) = self.nodes[current]
                    .array
                    .as_mut()
                    .expect("array node")
                    .insert_overrides
                    .iter_mut()
                    .find(|(s, _)| *s == source)
                    .and_then(|(_, overrides)| overrides.iter_mut().find(|(i, _)| *i == index))
                    .map(|(_, slot)| slot)
                {
                    slot.node = Some(nid);
                }
            }
        }
        self.nodes[current]
            .array
            .as_mut()
            .expect("array node")
            .child_nodes
            .insert(route, nid);
        Ok(nid)
    }

    // ─── Reads ──────────────────────────────────────────────────────────────

    /// Overlay-aware read of `path` from `start` (upstream proxy property
    /// walk). A missing final segment is `Ok(None)`; a missing intermediate
    /// segment is [`DeltaError::Unresolvable`].
    pub(super) fn read_value<'a>(
        &'a self,
        start: NodeId,
        path: &[Seg],
    ) -> Result<Option<&'a JsonValue>, DeltaError> {
        let mut node = Some(start);
        let mut raw: &JsonValue = self.node_raw(start);
        for (depth, segment) in path.iter().enumerate() {
            if let Seg::Key(key) = segment {
                if key == "length" && node.is_some_and(|nid| self.is_array_node(nid)) {
                    return Err(DeltaError::invalid("use length() for array length"));
                }
            }
            let resolved = self.read_step(raw, node, segment);
            let Some((value, child_node)) = resolved else {
                if depth + 1 == path.len() {
                    return Ok(None);
                }
                return Err(DeltaError::Unresolvable {
                    path: serde_json::to_string(&super::path_to_json(&path[..=depth]))
                        .unwrap_or_else(|_| "[]".to_string()),
                });
            };
            raw = value;
            node = child_node;
        }
        Ok(Some(raw))
    }

    /// Immutable resolution of `path` to `(raw value, bound node)`. Nodes are
    /// only created by writes, so a `None` node means the container has no
    /// overlay state and reads answer from the raw value — no mutable borrow
    /// needed for any read API.
    fn resolve_read_target<'a>(
        &'a self,
        start: NodeId,
        path: &[Seg],
    ) -> Result<(&'a JsonValue, Option<NodeId>), DeltaError> {
        let mut node = Some(start);
        let mut raw: &JsonValue = self.node_raw(start);
        for (depth, segment) in path.iter().enumerate() {
            let Some((value, child_node)) = self.read_step(raw, node, segment) else {
                return Err(DeltaError::Unresolvable {
                    path: serde_json::to_string(&super::path_to_json(&path[..=depth]))
                        .unwrap_or_else(|_| "[]".to_string()),
                });
            };
            raw = value;
            node = child_node;
        }
        Ok((raw, node))
    }

    /// One overlay-aware step: node routes check writes/deletes/entries first,
    /// clean routes read the raw value directly.
    fn read_step<'a>(
        &'a self,
        raw: &'a JsonValue,
        node: Option<NodeId>,
        segment: &Seg,
    ) -> Option<(&'a JsonValue, Option<NodeId>)> {
        if let Some(nid) = node {
            if self.nodes[nid].object.is_some() {
                let key = object_key_form(segment);
                if self.object_deleted(nid, &key) {
                    return None;
                }
                if let Some(slot) = self.write_slot(nid, &key) {
                    return Some((&slot.value, slot.node));
                }
                let child = self.object(nid).child_nodes.get(&key).copied();
                return raw.get(&key).map(|value| (value, child));
            }
            if self.nodes[nid].array.is_some() {
                let entry = resolve_entry(self.array(nid), segment)?;
                return Some(self.entry_value(nid, &entry));
            }
        }
        match (raw, segment) {
            (JsonValue::Object(object), Seg::Key(key)) => {
                object.get(key).map(|value| (value, None))
            }
            (JsonValue::Array(array), Seg::Index(index)) => {
                array.get(*index).map(|value| (value, None))
            }
            (JsonValue::Array(array), Seg::Key(key)) => {
                array.get(array_index(key)?).map(|value| (value, None))
            }
            _ => None,
        }
    }

    // ─── Object writes ──────────────────────────────────────────────────────

    pub(super) fn object_set(
        &mut self,
        nid: NodeId,
        key: &str,
        value: JsonValue,
    ) -> Result<(), DeltaError> {
        let key = key.to_string();
        let was_deleted = self.object_deleted(nid, &key);
        if was_deleted && self.object_base_has(nid, &key) {
            let overlay = self.nodes[nid].object.as_mut().expect("object node");
            if !overlay.readded.iter().any(|existing| existing == &key) {
                overlay.readded.push(key.clone());
            }
        }
        // Identical primitive assignment is a no-op (upstream `current === stored`).
        let exists = self.write_slot(nid, &key).is_some() || self.object_base_has(nid, &key);
        if !was_deleted && exists && !is_value_container(&value) {
            let current = match self.write_slot(nid, &key) {
                Some(slot) => slot.value.clone(),
                None => self
                    .node_raw(nid)
                    .get(&key)
                    .cloned()
                    .expect("presence checked above"),
            };
            if current == value {
                return Ok(());
            }
        }
        self.set_write_slot(nid, key.clone(), value);
        {
            let overlay = self.nodes[nid].object.as_mut().expect("object node");
            overlay.deletes.retain(|existing| existing != &key);
        }
        self.mark_dirty(nid);
        Ok(())
    }

    pub(super) fn object_delete(&mut self, nid: NodeId, key: &str) -> Result<(), DeltaError> {
        if !self.object_has(nid, key) {
            return Ok(());
        }
        self.remove_write_slot(nid, key);
        {
            let overlay = self.nodes[nid].object.as_mut().expect("object node");
            overlay.readded.retain(|existing| existing != key);
        }
        self.record_deletion(nid, key);
        self.mark_dirty(nid);
        Ok(())
    }

    // ─── Array mutators ─────────────────────────────────────────────────────

    fn require_array(&self, nid: NodeId) -> Result<(), DeltaError> {
        if self.nodes[nid].array.is_none() {
            return Err(DeltaError::invalid(
                "Array mutator called on incompatible receiver",
            ));
        }
        Ok(())
    }

    pub(super) fn array_push(
        &mut self,
        nid: NodeId,
        items: Vec<JsonValue>,
    ) -> Result<usize, DeltaError> {
        self.require_array(nid)?;
        let count = items.len();
        let source = self.new_source(nid, items);
        let at = self.array_len(nid);
        self.push_insert_entries(nid, at, source, count);
        self.mark_structural(nid);
        self.mark_dirty(nid);
        Ok(self.array_len(nid))
    }

    pub(super) fn array_pop(&mut self, nid: NodeId) -> Result<Option<JsonValue>, DeltaError> {
        self.require_array(nid)?;
        let length = self.array_len(nid);
        if length == 0 {
            return Ok(None);
        }
        let entry = self.array(nid).entries[length - 1];
        let value = self.entry_value_folded(nid, &entry);
        self.nodes[nid]
            .array
            .as_mut()
            .expect("array node")
            .entries
            .pop();
        self.mark_structural(nid);
        self.mark_dirty(nid);
        Ok(Some(value))
    }

    pub(super) fn array_shift(&mut self, nid: NodeId) -> Result<Option<JsonValue>, DeltaError> {
        self.require_array(nid)?;
        let length = self.array_len(nid);
        if length == 0 {
            return Ok(None);
        }
        let entry = self.array(nid).entries[0];
        let value = self.entry_value_folded(nid, &entry);
        self.nodes[nid]
            .array
            .as_mut()
            .expect("array node")
            .entries
            .remove(0);
        self.mark_structural(nid);
        self.mark_dirty(nid);
        Ok(Some(value))
    }

    pub(super) fn array_unshift(
        &mut self,
        nid: NodeId,
        items: Vec<JsonValue>,
    ) -> Result<usize, DeltaError> {
        self.require_array(nid)?;
        let count = items.len();
        let source = self.new_source(nid, items);
        self.push_insert_entries(nid, 0, source, count);
        self.mark_structural(nid);
        self.mark_dirty(nid);
        Ok(self.array_len(nid))
    }

    pub(super) fn array_splice(
        &mut self,
        nid: NodeId,
        start: usize,
        remove: Option<usize>,
        items: Vec<JsonValue>,
    ) -> Result<Vec<JsonValue>, DeltaError> {
        self.require_array(nid)?;
        let length = self.array_len(nid);
        let start = start.min(length);
        let remove = remove.unwrap_or(length - start).min(length - start);
        if remove == 0 && items.is_empty() {
            return Ok(Vec::new());
        }
        let removed: Vec<JsonValue> = self.array(nid).entries[start..start + remove]
            .iter()
            .map(|entry| self.entry_value_folded(nid, entry))
            .collect();
        let count = items.len();
        let source = self.new_source(nid, items);
        {
            let overlay = self.nodes[nid].array.as_mut().expect("array node");
            let inserted: Vec<Entry> = (0..count)
                .map(|index| Entry::Insert { source, index })
                .collect();
            overlay.entries.splice(start..start + remove, inserted);
            overlay.structural = true;
        }
        self.mark_dirty(nid);
        Ok(removed)
    }

    pub(super) fn array_set_index(
        &mut self,
        nid: NodeId,
        index: usize,
        value: JsonValue,
    ) -> Result<(), DeltaError> {
        self.require_array(nid)?;
        let length = self.array_len(nid);
        if index > length {
            return Err(DeltaError::invalid("Overlay arrays cannot contain holes"));
        }
        if index == length {
            let source = self.new_source(nid, vec![value]);
            self.push_insert_entries(nid, length, source, 1);
            self.mark_structural(nid);
            self.mark_dirty(nid);
            return Ok(());
        }
        let entry = self.array(nid).entries[index];
        // Identical primitive assignment is a no-op (upstream `current === stored`).
        if !is_value_container(&value) {
            let (current, _) = self.entry_value(nid, &entry);
            if current == &value {
                return Ok(());
            }
        }
        match entry {
            Entry::Base(base_index) => {
                // Writing the base value back removes the override (upstream's
                // primitive branch; equality is structural here).
                if !is_value_container(&value)
                    && self.node_raw(nid).as_array().expect("array route")[base_index] == value
                {
                    self.remove_base_override(nid, base_index);
                    self.mark_dirty(nid);
                    return Ok(());
                }
                self.upsert_base_override(nid, base_index, value);
            }
            Entry::Insert {
                source,
                index: item_index,
            } => {
                if !is_value_container(&value)
                    && self.array(nid).sources[source][item_index] == value
                {
                    self.remove_insert_override(nid, source, item_index);
                    self.mark_dirty(nid);
                    return Ok(());
                }
                self.upsert_insert_override(nid, source, item_index, value);
            }
        }
        self.mark_dirty(nid);
        Ok(())
    }

    pub(super) fn array_set_length(&mut self, nid: NodeId, next: usize) -> Result<(), DeltaError> {
        self.require_array(nid)?;
        let current = self.array_len(nid);
        if next == current {
            return Ok(());
        }
        if next < current {
            let overlay = self.nodes[nid].array.as_mut().expect("array node");
            overlay.entries.truncate(next);
            overlay.structural = true;
        } else {
            let grow = next - current;
            let source = self.new_source(nid, vec![JsonValue::Null; grow]);
            self.push_insert_entries(nid, current, source, grow);
            self.mark_structural(nid);
        }
        self.mark_dirty(nid);
        Ok(())
    }

    pub(super) fn array_reverse(&mut self, nid: NodeId) -> Result<(), DeltaError> {
        self.require_array(nid)?;
        if self.array_len(nid) < 2 {
            return Ok(());
        }
        {
            let overlay = self.nodes[nid].array.as_mut().expect("array node");
            overlay.entries.reverse();
            overlay.structural = true;
        }
        self.mark_dirty(nid);
        Ok(())
    }

    pub(super) fn array_sort(
        &mut self,
        nid: NodeId,
        mut comparator: impl FnMut(&mut JsonValue, &mut JsonValue) -> Ordering,
    ) -> Result<(), DeltaError> {
        self.require_array(nid)?;
        if self.array_len(nid) < 2 {
            return Ok(());
        }
        let entries = self.array(nid).entries.clone();
        let mut values: Vec<JsonValue> = entries
            .iter()
            .map(|entry| self.entry_value_folded(nid, entry))
            .collect();
        let originals = values.clone();
        let mut order: Vec<usize> = (0..values.len()).collect();
        // Stable, matching `Array.prototype.sort`.
        order.sort_by(|&left, &right| {
            let (first, second) = two_mut(&mut values, left, right);
            comparator(first, second)
        });
        // Comparator edits land: write mutated elements back to their slots.
        let mut mutated = false;
        for (position, entry) in entries.iter().enumerate() {
            if values[position] == originals[position] {
                continue;
            }
            mutated = true;
            match entry {
                Entry::Base(base_index) => {
                    self.upsert_base_override(nid, *base_index, values[position].clone())
                }
                Entry::Insert { source, index } => {
                    self.upsert_insert_override(nid, *source, *index, values[position].clone())
                }
            }
        }
        let order_changed = order
            .iter()
            .enumerate()
            .any(|(position, &token)| token != position);
        if order_changed {
            let overlay = self.nodes[nid].array.as_mut().expect("array node");
            overlay.entries = order.iter().map(|&token| entries[token]).collect();
            overlay.structural = true;
        }
        if mutated || order_changed {
            self.mark_dirty(nid);
        }
        Ok(())
    }

    pub(super) fn array_fill(
        &mut self,
        nid: NodeId,
        value: JsonValue,
        start: Option<isize>,
        end: Option<isize>,
    ) -> Result<(), DeltaError> {
        self.require_array(nid)?;
        let length = self.array_len(nid);
        let start = clamp_index(start.unwrap_or(0), length);
        let end = clamp_index(end.unwrap_or(length as isize), length);
        if end <= start {
            return Ok(());
        }
        let count = end - start;
        let items: Vec<JsonValue> = (0..count).map(|_| value.clone()).collect();
        let source = self.new_source(nid, items);
        {
            let overlay = self.nodes[nid].array.as_mut().expect("array node");
            let entries: Vec<Entry> = (0..count)
                .map(|index| Entry::Insert { source, index })
                .collect();
            overlay.entries.splice(start..end, entries);
            overlay.structural = true;
        }
        self.mark_dirty(nid);
        Ok(())
    }

    pub(super) fn array_copy_within(
        &mut self,
        nid: NodeId,
        target: isize,
        start: isize,
        end: Option<isize>,
    ) -> Result<(), DeltaError> {
        self.require_array(nid)?;
        let length = self.array_len(nid);
        let target = clamp_index(target, length);
        let start = clamp_index(start, length);
        let end = clamp_index(end.unwrap_or(length as isize), length);
        let count = end.saturating_sub(start).min(length - target);
        if count == 0 {
            return Ok(());
        }
        let values: Vec<JsonValue> = self.array(nid).entries[start..start + count]
            .iter()
            .map(|entry| self.entry_value_folded(nid, entry))
            .collect();
        let source = self.new_source(nid, values);
        {
            let overlay = self.nodes[nid].array.as_mut().expect("array node");
            let entries: Vec<Entry> = (0..count)
                .map(|index| Entry::Insert { source, index })
                .collect();
            overlay.entries.splice(target..target + count, entries);
            overlay.structural = true;
        }
        self.mark_dirty(nid);
        Ok(())
    }

    // ─── Emission ───────────────────────────────────────────────────────────

    /// Upstream `emitOperations`.
    pub(super) fn emit_operations(&self) -> Vec<Op> {
        if let Some(simple) = self.emit_simple_object_operations() {
            return simple;
        }
        let mut operations: Vec<Op> = Vec::new();
        let mut forced_folds: Vec<(NodeId, Vec<Seg>)> = Vec::new();
        let mut emission_paths: Vec<(NodeId, Vec<Seg>)> = Vec::new();
        let mut dense_candidates: HashMap<NodeId, DenseCandidates> = HashMap::new();

        for &nid in &self.dirty {
            self.record_dense_array_position(nid, &mut dense_candidates);
            if self.nodes[nid].array.is_some() {
                let overlay = self.array(nid);
                if !overlay.structural && !overlay.base_overrides.is_empty() {
                    let base_length = self.node_raw(nid).as_array().map_or(0, Vec::len);
                    let candidates =
                        dense_candidates
                            .entry(nid)
                            .or_insert_with(|| DenseCandidates {
                                indices: Vec::new(),
                                bits: None,
                                length: base_length,
                            });
                    for (index, _) in &overlay.base_overrides {
                        add_dense_candidate(candidates, *index);
                    }
                }
            }
        }

        let mut dense_regions: HashMap<NodeId, Vec<DenseRegion>> = HashMap::new();
        for (nid, candidates) in dense_candidates {
            let Some(path) = self.resolve_emission_path(nid) else {
                continue;
            };
            if path
                .iter()
                .any(|segment| matches!(segment, Seg::Key(key) if is_reserved_segment(key)))
            {
                continue;
            }
            let regions = build_dense_regions(&candidates);
            if !regions.is_empty() {
                dense_regions.insert(nid, regions);
            }
        }

        for &nid in &self.dirty {
            let Some(path) = self.resolve_emission_path(nid) else {
                continue;
            };
            if self.has_covering_dense_region(nid, &path, &dense_regions) {
                continue;
            }
            upsert_emission_path(&mut emission_paths, nid, path.clone());
            if self.nodes[nid].array.is_none() && self.has_reserved_mutation(nid) {
                upsert_emission_path(&mut forced_folds, nid, path.clone());
            }
            if let Some(reserved_at) = path
                .iter()
                .position(|segment| matches!(segment, Seg::Key(key) if is_reserved_segment(key)))
            {
                let mut ancestor = nid;
                for _ in reserved_at..path.len() {
                    ancestor = self.nodes[ancestor].origin.parent().expect("path ancestor");
                }
                upsert_emission_path(&mut forced_folds, ancestor, path[..reserved_at].to_vec());
            }
        }
        let forced_nodes: Vec<NodeId> = forced_folds.iter().map(|(nid, _)| *nid).collect();
        for (nid, path) in forced_folds {
            upsert_emission_path(&mut emission_paths, nid, path);
        }
        for nid in dense_regions.keys().copied().collect::<Vec<_>>() {
            if let Some(path) = self.resolve_emission_path(nid) {
                if !self.has_covering_dense_region(nid, &path, &dense_regions) {
                    upsert_emission_path(&mut emission_paths, nid, path);
                }
            }
        }

        let max_depth = emission_paths
            .iter()
            .map(|(_, path)| path.len())
            .max()
            .unwrap_or(0);
        let mut folded: Vec<NodeId> = Vec::new();
        for depth in 0..=max_depth {
            for (nid, path) in emission_paths
                .iter()
                .filter(|(_, path)| path.len() == depth)
            {
                let nid = *nid;
                if self.has_placement_ancestor(nid) || self.has_folded_ancestor(nid, &folded) {
                    continue;
                }
                if forced_nodes.contains(&nid) {
                    operations.push(set_op_for_path(path, self.materialize_node(nid)));
                    folded.push(nid);
                    continue;
                }
                if self.nodes[nid].array.is_some() {
                    self.emit_array_operations(nid, path, &mut operations, dense_regions.get(&nid));
                } else {
                    self.emit_object_operations(nid, path, &mut operations);
                }
                if operations.len() > MAX_DELTA_OPERATIONS {
                    return vec![Op::Replace(self.materialize_node(self.root))];
                }
            }
        }
        operations
    }

    /// Upstream `emitSimpleObjectOperations`.
    fn emit_simple_object_operations(&self) -> Option<Vec<Op>> {
        let mut nodes: Vec<(NodeId, Vec<Seg>)> = Vec::new();
        for &nid in &self.dirty {
            if self.nodes[nid].array.is_some()
                || self.has_reserved_mutation(nid)
                || self.has_placement_ancestor(nid)
            {
                return None;
            }
            let mut parent = self.nodes[nid].origin.parent();
            while let Some(ancestor) = parent {
                if self.nodes[ancestor].array.is_some() {
                    return None;
                }
                parent = self.nodes[ancestor].origin.parent();
            }
            let Some(path) = self.resolve_emission_path(nid) else {
                // Unroutable nodes are skipped, exactly as in the general path.
                continue;
            };
            if path
                .iter()
                .any(|segment| matches!(segment, Seg::Key(key) if is_reserved_segment(key)))
            {
                return None;
            }
            nodes.push((nid, path));
            if nodes.len() > MAX_SIMPLE_OBJECT_NODES {
                return None;
            }
        }
        // Stable sort by path depth keeps first-touch order within a depth.
        nodes.sort_by_key(|(_, path)| path.len());
        let mut operations = Vec::new();
        for (nid, path) in &nodes {
            self.emit_object_operations(*nid, path, &mut operations);
            if operations.len() > MAX_DELTA_OPERATIONS {
                return Some(vec![Op::Replace(self.materialize_node(self.root))]);
            }
        }
        // Upstream's `simpleObjectMaterialization` flag only selects the value
        // clone strategy; the Rust port always replays the ops over the base
        // clone, which is the same result (see module docs).
        Some(operations)
    }

    /// Upstream `resolvePath`: the node's root-relative path, `None` when the
    /// final overlay state no longer routes to the node.
    fn resolve_emission_path(&self, nid: NodeId) -> Option<Vec<Seg>> {
        let mut segments: Vec<Seg> = Vec::new();
        let mut current = nid;
        loop {
            match &self.nodes[current].origin {
                NodeOrigin::Root => break,
                NodeOrigin::BaseKey { parent, key } => {
                    if self.object_deleted(*parent, key) || self.write_slot(*parent, key).is_some()
                    {
                        return None;
                    }
                    segments.push(Seg::Key(key.clone()));
                    current = *parent;
                }
                NodeOrigin::WriteSlot { parent, key, .. } => {
                    let slot = self.write_slot(*parent, key)?;
                    if slot.node != Some(current) {
                        return None;
                    }
                    segments.push(Seg::Key(key.clone()));
                    current = *parent;
                }
                NodeOrigin::BaseEntry {
                    parent,
                    base_index,
                    captured,
                } => {
                    match (captured, self.base_override_slot(*parent, *base_index)) {
                        (Some(_), Some(slot)) if slot.node == Some(current) => {}
                        (None, None) => {}
                        _ => return None,
                    }
                    let index = self.logical_index_for_base(*parent, *base_index)?;
                    segments.push(Seg::index(index));
                    current = *parent;
                }
                NodeOrigin::InsertEntry {
                    parent,
                    source,
                    index,
                    ..
                } => {
                    match self.insert_override_slot(*parent, *source, *index) {
                        Some(slot) if slot.node == Some(current) => {}
                        Some(_) => return None,
                        None => {}
                    }
                    let logical = self.logical_index_for_insert(*parent, *source, *index)?;
                    segments.push(Seg::index(logical));
                    current = *parent;
                }
            }
        }
        segments.reverse();
        Some(segments)
    }

    /// Upstream `hasReservedMutation`.
    fn has_reserved_mutation(&self, nid: NodeId) -> bool {
        let Some(object) = &self.nodes[nid].object else {
            return false;
        };
        object
            .writes
            .iter()
            .any(|(key, _)| is_reserved_segment(key))
            || object.deletes.iter().any(|key| is_reserved_segment(key))
    }

    /// Upstream `hasPlacementAncestor`: any non-root node bound to a placement
    /// route.
    fn has_placement_ancestor(&self, nid: NodeId) -> bool {
        let mut current = nid;
        loop {
            let node = &self.nodes[current];
            if node.origin.parent().is_some() && node.origin.is_placement() {
                return true;
            }
            match node.origin.parent() {
                Some(parent) => current = parent,
                None => return false,
            }
        }
    }

    fn has_folded_ancestor(&self, nid: NodeId, folded: &[NodeId]) -> bool {
        let mut current = self.nodes[nid].origin.parent();
        while let Some(parent) = current {
            if folded.contains(&parent) {
                return true;
            }
            current = self.nodes[parent].origin.parent();
        }
        false
    }

    /// Upstream `emitObjectOperations`.
    fn emit_object_operations(&self, nid: NodeId, path: &[Seg], operations: &mut Vec<Op>) {
        // Immutable replay must encode delete-and-readd explicitly so string-key
        // insertion order matches the draft.
        let readded = self.object(nid).readded.clone();
        for key in &readded {
            if self.object_base_has(nid, key) {
                let mut child = path.to_vec();
                child.push(Seg::key(key.clone()));
                operations.push(Op::Delete { path: child });
            }
        }
        let writes = self.object(nid).writes.clone();
        for (key, slot) in &writes {
            let mut child = path.to_vec();
            child.push(Seg::key(key.clone()));
            let before = if readded.iter().any(|existing| existing == key)
                || !self.object_base_has(nid, key)
            {
                None
            } else {
                Some(self.node_raw(nid).get(key).expect("base key").clone())
            };
            let after = self.fold_slot(slot);
            emit_changed_value(operations, child, before.as_ref(), after);
        }
        let deletes = self.object(nid).deletes.clone();
        for key in &deletes {
            if self.object_base_has(nid, key) {
                let mut child = path.to_vec();
                child.push(Seg::key(key.clone()));
                operations.push(Op::Delete { path: child });
            }
        }
    }

    fn fold_slot(&self, slot: &Slot) -> JsonValue {
        self.fold_value(&slot.value, slot.node)
    }

    /// Upstream `emitArrayOperations`.
    fn emit_array_operations(
        &self,
        nid: NodeId,
        path: &[Seg],
        operations: &mut Vec<Op>,
        dense_regions: Option<&Vec<DenseRegion>>,
    ) {
        let entries = self.array(nid).entries.clone();
        if let Some(regions) = dense_regions {
            for region in regions {
                if operations.len() > MAX_DELTA_OPERATIONS {
                    return;
                }
                let items: Vec<JsonValue> = (region.start..region.start + region.length)
                    .map(|index| self.entry_value_folded(nid, &entries[index]))
                    .collect();
                operations.push(Op::Splice {
                    path: path.to_vec(),
                    start: region.start,
                    remove: region.length,
                    items,
                });
            }
        }
        if self.array(nid).structural {
            let plan = self.build_array_plan(nid);
            for run in (0..plan.remove_runs.len()).step_by(2) {
                if operations.len() > MAX_DELTA_OPERATIONS {
                    return;
                }
                operations.push(Op::Splice {
                    path: path.to_vec(),
                    start: plan.remove_runs[run],
                    remove: plan.remove_runs[run + 1],
                    items: Vec::new(),
                });
            }
            if let Some(permutation) = plan.permutation {
                operations.push(Op::Permute {
                    path: path.to_vec(),
                    permutation,
                });
            }
            for (logical_index, run_start, run_end) in plan.insert_runs {
                if operations.len() > MAX_DELTA_OPERATIONS {
                    return;
                }
                let items: Vec<JsonValue> = entries[run_start..run_end]
                    .iter()
                    .map(|entry| self.entry_value_folded(nid, entry))
                    .collect();
                operations.push(Op::Splice {
                    path: path.to_vec(),
                    start: logical_index,
                    remove: 0,
                    items,
                });
            }
        }
        let base_overrides = self.array(nid).base_overrides.clone();
        for (base_index, slot) in &base_overrides {
            if operations.len() > MAX_DELTA_OPERATIONS {
                return;
            }
            let Some(index) = self.logical_index_for_base(nid, *base_index) else {
                continue;
            };
            if let Some(regions) = dense_regions {
                if regions.iter().any(|region| region_contains(region, index)) {
                    continue;
                }
            }
            let before = self.node_raw(nid).as_array().expect("array route")[*base_index].clone();
            let after = self.fold_slot(slot);
            let mut child = path.to_vec();
            child.push(Seg::index(index));
            emit_changed_value(operations, child, Some(&before), after);
        }
    }

    /// Upstream `buildArrayPlan`.
    fn build_array_plan(&self, nid: NodeId) -> ArrayPlan {
        let overlay = self.array(nid);
        let base_length = self.node_raw(nid).as_array().map_or(0, Vec::len);
        let mut retained = vec![false; base_length];
        let mut target_base: Vec<usize> = Vec::new();
        for entry in &overlay.entries {
            if let Entry::Base(index) = entry {
                retained[*index] = true;
                target_base.push(*index);
            }
        }
        let mut remove_runs: Vec<usize> = Vec::new();
        let mut end = base_length;
        while end > 0 {
            if retained[end - 1] {
                end -= 1;
                continue;
            }
            let mut start = end - 1;
            while start > 0 && !retained[start - 1] {
                start -= 1;
            }
            remove_runs.push(start);
            remove_runs.push(end - start);
            end = start;
        }
        let retained_base: Vec<usize> = (0..base_length).filter(|&index| retained[index]).collect();
        let permutation = if target_base != retained_base {
            let positions: HashMap<usize, usize> = retained_base
                .iter()
                .enumerate()
                .map(|(position, index)| (*index, position))
                .collect();
            Some(target_base.iter().map(|index| positions[index]).collect())
        } else {
            None
        };
        let mut insert_runs: Vec<(usize, usize, usize)> = Vec::new();
        let mut logical_index = 0;
        let mut at = 0;
        while at < overlay.entries.len() {
            if matches!(overlay.entries[at], Entry::Base(_)) {
                logical_index += 1;
                at += 1;
                continue;
            }
            let run_start = at;
            while at < overlay.entries.len() && matches!(overlay.entries[at], Entry::Insert { .. })
            {
                logical_index += 1;
                at += 1;
            }
            insert_runs.push((logical_index - (at - run_start), run_start, at));
        }
        ArrayPlan {
            remove_runs,
            permutation,
            insert_runs,
        }
    }

    /// Upstream `recordDenseArrayPosition` + `locateDenseArrayPosition`.
    fn record_dense_array_position(
        &self,
        nid: NodeId,
        groups: &mut HashMap<NodeId, DenseCandidates>,
    ) {
        let mut child = nid;
        loop {
            let Some(parent) = self.nodes[child].origin.parent() else {
                return;
            };
            if self.nodes[parent].array.is_some() {
                let NodeOrigin::BaseEntry {
                    base_index,
                    captured: None,
                    ..
                } = &self.nodes[child].origin
                else {
                    return;
                };
                if self.array(parent).structural
                    || self.base_override_slot(parent, *base_index).is_some()
                {
                    return;
                }
                let base_length = self.node_raw(parent).as_array().map_or(0, Vec::len);
                let candidates = groups.entry(parent).or_insert_with(|| DenseCandidates {
                    indices: Vec::new(),
                    bits: None,
                    length: base_length,
                });
                add_dense_candidate(candidates, *base_index);
                return;
            }
            child = parent;
        }
    }

    /// Upstream `hasCoveringDenseRegion`.
    fn has_covering_dense_region(
        &self,
        nid: NodeId,
        path: &[Seg],
        dense_regions: &HashMap<NodeId, Vec<DenseRegion>>,
    ) -> bool {
        let mut parent = self.nodes[nid].origin.parent();
        while let Some(ancestor) = parent {
            if let Some(regions) = dense_regions.get(&ancestor) {
                if let Some(parent_path) = self.resolve_emission_path(ancestor) {
                    if path.len() > parent_path.len()
                        && path[..parent_path.len()] == parent_path[..]
                    {
                        if let Some(Seg::Index(index)) = path.get(parent_path.len()) {
                            if regions.iter().any(|region| region_contains(region, *index)) {
                                return true;
                            }
                        }
                    }
                }
            }
            parent = self.nodes[ancestor].origin.parent();
        }
        false
    }

    /// Upstream `cloneNode`: the node's current overlay state as a value, with
    /// descendant writes folded in. Clean nodes return their raw value.
    pub(super) fn materialize_node(&self, nid: NodeId) -> JsonValue {
        if !self.needs_materialize(nid) {
            return self.node_raw(nid).clone();
        }
        if self.nodes[nid].array.is_some() {
            let entries = self.array(nid).entries.clone();
            JsonValue::Array(
                entries
                    .iter()
                    .map(|entry| self.entry_value_folded(nid, entry))
                    .collect(),
            )
        } else {
            let keys = self.object_keys(nid);
            let mut object = serde_json::Map::new();
            for key in keys {
                if let Some(slot) = self.write_slot(nid, &key) {
                    object.insert(key, self.fold_slot(slot));
                } else if let Some(value) = self.node_raw(nid).get(&key) {
                    // A node may be bound to the raw (base) route for this key;
                    // fold its overlay state in exactly as write slots fold.
                    let child = self.nodes[nid]
                        .object
                        .as_ref()
                        .and_then(|overlay| overlay.child_nodes.get(&key).copied());
                    object.insert(key, self.fold_value(value, child));
                }
            }
            JsonValue::Object(object)
        }
    }

    /// Overlay-aware key listing for [`Draft::keys`] (arrays append
    /// `"length"`, mirroring upstream `ownKeys`).
    pub(super) fn overlay_keys(&self, nid: NodeId) -> Vec<String> {
        if self.nodes[nid].array.is_some() {
            let length = self.array_len(nid);
            return (0..length)
                .map(|index| index.to_string())
                .chain(std::iter::once("length".to_string()))
                .collect();
        }
        self.object_keys(nid)
    }

    pub(super) fn overlay_length(&self, nid: NodeId) -> Result<usize, DeltaError> {
        if self.nodes[nid].array.is_none() {
            return Err(DeltaError::invalid("length() requires an array overlay"));
        }
        Ok(self.array_len(nid))
    }

    /// Upstream `has` on array overlays: `length` and canonical indices.
    pub(super) fn array_has(&self, nid: NodeId, key: &str) -> bool {
        key == "length" || array_index(key).is_some_and(|index| index < self.array_len(nid))
    }

    // ─── Read answers (immutable; usable from `&self` Draft methods) ───────

    fn read_has(&self, start: NodeId, path: &[Seg], key: &str) -> Result<bool, DeltaError> {
        let (raw, node) = self.resolve_read_target(start, path)?;
        match node {
            Some(nid) if self.is_array_node(nid) => Ok(self.array_has(nid, key)),
            Some(nid) => Ok(self.object_has(nid, key)),
            None => Ok(match raw {
                JsonValue::Array(array) => {
                    key == "length" || array_index(key).is_some_and(|index| index < array.len())
                }
                JsonValue::Object(object) => object.get(key).is_some(),
                _ => false,
            }),
        }
    }

    fn read_keys(&self, start: NodeId, path: &[Seg]) -> Result<Vec<String>, DeltaError> {
        let (raw, node) = self.resolve_read_target(start, path)?;
        Ok(match node {
            Some(nid) => self.overlay_keys(nid),
            None => match raw {
                JsonValue::Array(array) => (0..array.len())
                    .map(|index| index.to_string())
                    .chain(std::iter::once("length".to_string()))
                    .collect(),
                JsonValue::Object(object) => object.keys().cloned().collect(),
                _ => Vec::new(),
            },
        })
    }

    fn read_length(&self, start: NodeId, path: &[Seg]) -> Result<usize, DeltaError> {
        let (raw, node) = self.resolve_read_target(start, path)?;
        match node {
            Some(nid) => self.overlay_length(nid),
            None => raw
                .as_array()
                .map(Vec::len)
                .ok_or_else(|| DeltaError::invalid("length() requires an array overlay")),
        }
    }

    fn read_get_index<'a>(
        &'a self,
        start: NodeId,
        path: &[Seg],
        index: usize,
    ) -> Result<Option<&'a JsonValue>, DeltaError> {
        let (raw, node) = self.resolve_read_target(start, path)?;
        match node {
            Some(nid) => {
                if !self.is_array_node(nid) {
                    return Err(DeltaError::invalid("get_index() requires an array overlay"));
                }
                if index >= self.array_len(nid) {
                    return Ok(None);
                }
                let entry = self.array(nid).entries[index];
                let (value, _) = self.entry_value(nid, &entry);
                Ok(Some(value))
            }
            None => {
                let array = raw
                    .as_array()
                    .ok_or_else(|| DeltaError::invalid("get_index() requires an array overlay"))?;
                Ok(array.get(index))
            }
        }
    }
}

struct ArrayPlan {
    remove_runs: Vec<usize>,
    permutation: Option<Vec<usize>>,
    insert_runs: Vec<(usize, usize, usize)>,
}

enum StepError {
    Missing,
    Other(DeltaError),
}

impl From<DeltaError> for StepError {
    fn from(error: DeltaError) -> Self {
        StepError::Other(error)
    }
}

fn is_value_container(value: &JsonValue) -> bool {
    value.is_object() || value.is_array()
}

fn object_key_form(segment: &Seg) -> String {
    match segment {
        Seg::Key(key) => key.clone(),
        Seg::Index(index) => index.to_string(),
    }
}

fn child_route(entry: &Entry) -> ChildRoute {
    match entry {
        Entry::Base(index) => ChildRoute::Base(*index),
        Entry::Insert { source, index } => ChildRoute::Insert(*source, *index),
    }
}

/// Resolve a segment against an array overlay's symbolic entries. String keys
/// use JavaScript's canonical-index rules; `"length"` is handled by callers.
fn resolve_entry(overlay: &ArrayOverlay, segment: &Seg) -> Option<Entry> {
    let index = match segment {
        Seg::Index(index) => *index,
        Seg::Key(key) => array_index(key)?,
    };
    overlay.entries.get(index).copied()
}

/// Upstream `arrayIndex`: canonical array-index strings (`"0"`, `"1"`, …, no
/// leading zeros, below 2^32 - 1).
fn array_index(key: &str) -> Option<usize> {
    if key.is_empty() || key.len() > 10 {
        return None;
    }
    if key == "0" {
        return Some(0);
    }
    let first = key.as_bytes()[0];
    if !(b'1'..=b'9').contains(&first) {
        return None;
    }
    let mut index: usize = (first - b'0') as usize;
    for digit in key.as_bytes()[1..].iter() {
        if !digit.is_ascii_digit() {
            return None;
        }
        index = index * 10 + (digit - b'0') as usize;
        if index >= 4_294_967_295 {
            return None;
        }
    }
    Some(index)
}

/// Upstream `clampIndex` for signed arguments (already truncated to integers).
fn clamp_index(value: isize, length: usize) -> usize {
    if value < 0 {
        (length as isize + value).max(0) as usize
    } else {
        (value as usize).min(length)
    }
}

/// Two distinct mutable elements of one slice.
fn two_mut(
    values: &mut [JsonValue],
    left: usize,
    right: usize,
) -> (&mut JsonValue, &mut JsonValue) {
    debug_assert_ne!(left, right, "sort compares distinct elements");
    if left < right {
        let (head, tail) = values.split_at_mut(right);
        (&mut head[left], &mut tail[0])
    } else {
        let (head, tail) = values.split_at_mut(left);
        (&mut tail[0], &mut head[right])
    }
}

/// JavaScript default-sort string coercion (`String(value)`).
pub(super) fn js_string(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(true) => "true".to_string(),
        JsonValue::Bool(false) => "false".to_string(),
        JsonValue::Number(number) => js_number(number),
        JsonValue::String(text) => text.clone(),
        // Array.prototype.toString === join(","), with null slots rendered as
        // empty strings; nested arrays recurse.
        JsonValue::Array(items) => items
            .iter()
            .map(|item| match item {
                JsonValue::Null => String::new(),
                other => js_string(other),
            })
            .collect::<Vec<String>>()
            .join(","),
        JsonValue::Object(_) => "[object Object]".to_string(),
    }
}

fn js_number(number: &serde_json::Number) -> String {
    if let Some(int) = number.as_i64() {
        return int.to_string();
    }
    if let Some(int) = number.as_u64() {
        return int.to_string();
    }
    let float = number.as_f64().unwrap_or(0.0);
    if float == 0.0 {
        return "0".to_string();
    }
    if float.fract() == 0.0 && float.abs() < 1e21 {
        return format!("{}", float as i128);
    }
    format!("{float}")
}

/// Upstream `emitChangedValue` (bool return dropped: only the materialization
/// strategy consumed it).
fn emit_changed_value(
    operations: &mut Vec<Op>,
    path: Vec<Seg>,
    before: Option<&JsonValue>,
    after: JsonValue,
) {
    if let Some(before) = before {
        let before_container = is_value_container(before);
        let after_container = is_value_container(&after);
        if !after_container && before == &after {
            return;
        }
        if before_container && after_container && before == &after {
            return;
        }
        if let (Some(before_text), Some(after_text)) = (before.as_str(), after.as_str()) {
            // Slice/equality stays on the rope fast path upstream; startsWith
            // scans the entire accumulated prefix on every append.
            if after_text.len() > before_text.len() && after_text.starts_with(before_text) {
                operations.push(Op::Append {
                    path,
                    text: after_text[before_text.len()..].to_string(),
                });
                return;
            }
            let shared = super::overlap(before_text, after_text, DEFAULT_OVERLAP_SCAN);
            if shared > 0 {
                operations.push(Op::Truncate {
                    path: path.clone(),
                    count: before_text.chars().count() - shared,
                });
                if shared < after_text.chars().count() {
                    operations.push(Op::Append {
                        path,
                        text: after_text.chars().skip(shared).collect(),
                    });
                }
                return;
            }
        }
    }
    operations.push(Op::Set { path, value: after });
}

fn set_op_for_path(path: &[Seg], value: JsonValue) -> Op {
    if path.is_empty() {
        Op::Replace(value)
    } else {
        Op::Set {
            path: path.to_vec(),
            value,
        }
    }
}

fn upsert_emission_path(paths: &mut Vec<(NodeId, Vec<Seg>)>, nid: NodeId, path: Vec<Seg>) {
    if let Some(position) = paths.iter().position(|(existing, _)| *existing == nid) {
        paths[position].1 = path;
    } else {
        paths.push((nid, path));
    }
}

/// Upstream `addDenseCandidate`: bitmap promotion at 256 candidates.
fn add_dense_candidate(candidates: &mut DenseCandidates, index: usize) {
    if let Some(bits) = &mut candidates.bits {
        bits[index] = 1;
        return;
    }
    candidates.indices.push(index);
    if candidates.indices.len() < 256 {
        return;
    }
    let mut bits = vec![0u8; candidates.length];
    for existing in &candidates.indices {
        bits[*existing] = 1;
    }
    candidates.indices.clear();
    candidates.bits = Some(bits);
}

/// Upstream `buildDenseRegions`.
fn build_dense_regions(candidates: &DenseCandidates) -> Vec<DenseRegion> {
    let Some(bits) = &candidates.bits else {
        return Vec::new();
    };
    let mut regions = Vec::new();
    let mut at = 0;
    while at < bits.len() {
        while at < bits.len() && bits[at] == 0 {
            at += 1;
        }
        if at == bits.len() {
            break;
        }
        let start = at;
        let mut end = at;
        let mut count = 0;
        let mut gap = 0;
        while at < bits.len() {
            if bits[at] != 0 {
                count += 1;
                end = at;
                gap = 0;
            } else {
                gap += 1;
                if gap > 1 {
                    break;
                }
            }
            at += 1;
        }
        let length = end - start + 1;
        if count >= 256 && count * 2 >= length {
            regions.push(DenseRegion { start, length });
        }
    }
    regions
}

fn region_contains(region: &DenseRegion, index: usize) -> bool {
    index >= region.start && index < region.start + region.length
}

/// A transaction-scoped, path-addressed view of the overlay (upstream's
/// `Draft<T>` proxy, driven explicitly).
pub struct Draft<'c> {
    context: &'c mut OverlayContext,
}

impl<'c> Draft<'c> {
    pub(super) fn new(context: &'c mut OverlayContext) -> Draft<'c> {
        Draft { context }
    }

    /// Overlay-aware read. The root value (empty path) reflects the base; every
    /// keyed step reflects overlay writes, deletes, and entry overrides.
    pub fn get(&self, path: &[Seg]) -> Result<Option<&JsonValue>, DeltaError> {
        let root = self.context.root;
        self.context.read_value(root, path)
    }

    /// Upstream `has`: overlay-aware presence. Arrays answer `"length"` and
    /// canonical indices; objects answer overlay presence.
    pub fn has(&self, path: &[Seg], key: &str) -> Result<bool, DeltaError> {
        let root = self.context.root;
        self.context.read_has(root, path, key)
    }

    /// Upstream `ownKeys` order: numeric indices ascending as strings first,
    /// then string keys in insertion order; arrays append `"length"`.
    pub fn keys(&self, path: &[Seg]) -> Result<Vec<String>, DeltaError> {
        let root = self.context.root;
        self.context.read_keys(root, path)
    }

    /// Overlay array length.
    pub fn length(&self, path: &[Seg]) -> Result<usize, DeltaError> {
        let root = self.context.root;
        self.context.read_length(root, path)
    }

    /// Overlay array element read.
    pub fn get_index(&self, path: &[Seg], index: usize) -> Result<Option<&JsonValue>, DeltaError> {
        let root = self.context.root;
        self.context.read_get_index(root, path, index)
    }

    /// Object key write (upstream `set` through a proxy). Reserved segments are
    /// accepted here and folded at emission (upstream `forcedFolds`); arrays
    /// accept only `"length"`.
    pub fn set(&mut self, path: &[Seg], key: &str, value: JsonValue) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        if self.context.is_array_node(node) {
            if key == "length" {
                let length = json_array_length(&value)?;
                return self.context.array_set_length(node, length);
            }
            return Err(DeltaError::invalid(
                "Only array indices and length can be written",
            ));
        }
        self.context.object_set(node, key, value)
    }

    /// Object key delete. Deleting from an array overlay is rejected
    /// (`Overlay arrays cannot contain holes`).
    pub fn delete(&mut self, path: &[Seg], key: &str) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        if self.context.is_array_node(node) {
            return Err(DeltaError::invalid("Overlay arrays cannot contain holes"));
        }
        self.context.object_delete(node, key)
    }

    pub fn push(&mut self, path: &[Seg], items: Vec<JsonValue>) -> Result<usize, DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_push(node, items)
    }

    pub fn pop(&mut self, path: &[Seg]) -> Result<Option<JsonValue>, DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_pop(node)
    }

    pub fn shift(&mut self, path: &[Seg]) -> Result<Option<JsonValue>, DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_shift(node)
    }

    pub fn unshift(&mut self, path: &[Seg], items: Vec<JsonValue>) -> Result<usize, DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_unshift(node, items)
    }

    /// JS `splice` clamping: `remove: None` removes through the end.
    pub fn splice(
        &mut self,
        path: &[Seg],
        start: usize,
        remove: Option<usize>,
        items: Vec<JsonValue>,
    ) -> Result<Vec<JsonValue>, DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_splice(node, start, remove, items)
    }

    pub fn set_index(
        &mut self,
        path: &[Seg],
        index: usize,
        value: JsonValue,
    ) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_set_index(node, index, value)
    }

    /// Grow fills with explicit nulls; shrink truncates.
    pub fn set_length(&mut self, path: &[Seg], len: usize) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_set_length(node, len)
    }

    pub fn reverse(&mut self, path: &[Seg]) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_reverse(node)
    }

    /// Stable sort with a comparator that may mutate the extracted element
    /// values; edits are written back to their slots (upstream keeps comparator
    /// edits made through child proxies).
    pub fn sort_by(
        &mut self,
        path: &[Seg],
        comparator: impl FnMut(&mut JsonValue, &mut JsonValue) -> Ordering,
    ) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_sort(node, comparator)
    }

    /// JS default sort: compare `String(value)` coercion.
    pub fn sort_default(&mut self, path: &[Seg]) -> Result<(), DeltaError> {
        self.sort_by(path, |left, right| js_string(left).cmp(&js_string(right)))
    }

    /// JS `fill` with from-end semantics for negative bounds.
    pub fn fill(
        &mut self,
        path: &[Seg],
        value: JsonValue,
        start: Option<isize>,
        end: Option<isize>,
    ) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_fill(node, value, start, end)
    }

    /// JS `copyWithin` with from-end semantics for negative bounds.
    pub fn copy_within(
        &mut self,
        path: &[Seg],
        target: isize,
        start: isize,
        end: Option<isize>,
    ) -> Result<(), DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        self.context.array_copy_within(node, target, start, end)
    }

    /// Bind a handle to the overlay node at `path` (upstream: holding a child
    /// proxy). Writes through a handle whose container was later deleted or
    /// shifted out mutate the node's own state but are skipped by emission;
    /// reads through the handle still return that state.
    pub fn at(&mut self, path: &[Seg]) -> Result<NodeView<'_>, DeltaError> {
        let root = self.context.root;
        let node = self.context.node_for(root, path)?;
        Ok(NodeView {
            context: self.context,
            node,
        })
    }
}

/// A handle bound to one overlay node (upstream: a retained child proxy). All
/// [`Draft`] operations are available relative to the node, plus nested
/// [`NodeView::at`].
pub struct NodeView<'c> {
    context: &'c mut OverlayContext,
    node: NodeId,
}

impl<'c> NodeView<'c> {
    pub fn get(&self, path: &[Seg]) -> Result<Option<&JsonValue>, DeltaError> {
        self.context.read_value(self.node, path)
    }

    pub fn has(&self, path: &[Seg], key: &str) -> Result<bool, DeltaError> {
        self.context.read_has(self.node, path, key)
    }

    pub fn keys(&self, path: &[Seg]) -> Result<Vec<String>, DeltaError> {
        self.context.read_keys(self.node, path)
    }

    pub fn length(&self, path: &[Seg]) -> Result<usize, DeltaError> {
        self.context.read_length(self.node, path)
    }

    pub fn get_index(&self, path: &[Seg], index: usize) -> Result<Option<&JsonValue>, DeltaError> {
        self.context.read_get_index(self.node, path, index)
    }

    pub fn set(&mut self, path: &[Seg], key: &str, value: JsonValue) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        if self.context.is_array_node(node) {
            if key == "length" {
                let length = json_array_length(&value)?;
                return self.context.array_set_length(node, length);
            }
            return Err(DeltaError::invalid(
                "Only array indices and length can be written",
            ));
        }
        self.context.object_set(node, key, value)
    }

    pub fn delete(&mut self, path: &[Seg], key: &str) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        if self.context.is_array_node(node) {
            return Err(DeltaError::invalid("Overlay arrays cannot contain holes"));
        }
        self.context.object_delete(node, key)
    }

    pub fn push(&mut self, path: &[Seg], items: Vec<JsonValue>) -> Result<usize, DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_push(node, items)
    }

    pub fn pop(&mut self, path: &[Seg]) -> Result<Option<JsonValue>, DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_pop(node)
    }

    pub fn shift(&mut self, path: &[Seg]) -> Result<Option<JsonValue>, DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_shift(node)
    }

    pub fn unshift(&mut self, path: &[Seg], items: Vec<JsonValue>) -> Result<usize, DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_unshift(node, items)
    }

    pub fn splice(
        &mut self,
        path: &[Seg],
        start: usize,
        remove: Option<usize>,
        items: Vec<JsonValue>,
    ) -> Result<Vec<JsonValue>, DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_splice(node, start, remove, items)
    }

    pub fn set_index(
        &mut self,
        path: &[Seg],
        index: usize,
        value: JsonValue,
    ) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_set_index(node, index, value)
    }

    pub fn set_length(&mut self, path: &[Seg], len: usize) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_set_length(node, len)
    }

    pub fn reverse(&mut self, path: &[Seg]) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_reverse(node)
    }

    pub fn sort_by(
        &mut self,
        path: &[Seg],
        comparator: impl FnMut(&mut JsonValue, &mut JsonValue) -> Ordering,
    ) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_sort(node, comparator)
    }

    pub fn sort_default(&mut self, path: &[Seg]) -> Result<(), DeltaError> {
        self.sort_by(path, |left, right| js_string(left).cmp(&js_string(right)))
    }

    pub fn fill(
        &mut self,
        path: &[Seg],
        value: JsonValue,
        start: Option<isize>,
        end: Option<isize>,
    ) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_fill(node, value, start, end)
    }

    pub fn copy_within(
        &mut self,
        path: &[Seg],
        target: isize,
        start: isize,
        end: Option<isize>,
    ) -> Result<(), DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        self.context.array_copy_within(node, target, start, end)
    }

    /// Nested handle relative to this node (upstream: `child.deeper`).
    pub fn at(&mut self, path: &[Seg]) -> Result<NodeView<'_>, DeltaError> {
        let node = self.context.node_for(self.node, path)?;
        Ok(NodeView {
            context: self.context,
            node,
        })
    }
}

/// Upstream `toArrayLength`.
fn json_array_length(value: &JsonValue) -> Result<usize, DeltaError> {
    let length = match value {
        JsonValue::Number(number) => {
            if let Some(int) = number.as_u64() {
                int
            } else if let Some(int) = number.as_i64() {
                int.max(0) as u64
            } else {
                let float = number.as_f64().unwrap_or(f64::NAN);
                if float.fract() != 0.0 || !float.is_finite() {
                    return Err(DeltaError::invalid("Invalid array length"));
                }
                float.max(0.0) as u64
            }
        }
        _ => return Err(DeltaError::invalid("Invalid array length")),
    };
    if length >= 4_294_967_296 {
        return Err(DeltaError::invalid("Invalid array length"));
    }
    Ok(length as usize)
}
