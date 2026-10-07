//! Port of `packages/durable/src/harness/json.ts`.
//!
//! Deviations: JSON values are `serde_json::Value` with `preserve_order`
//! objects; `assignJson`'s leaf-wise assignment avoids publishing whole
//! containers as one delta for the same reason as upstream.

use serde_json::Value;

/// Assign `value` at `target[key]` leaf by leaf (upstream `assignJson`).
/// Chord records a container assignment as one full set and only emits an
/// append when a string leaf is reassigned with a longer string, so writing
/// the partial whole would store and publish the complete message on every
/// flush.
pub fn assign_json(target: &mut Value, key: &str, value: Value) {
	let Some(slot) = target.get_mut(key) else {
		if let Value::Object(map) = target {
			map.insert(key.to_string(), value);
		}
		return;
	};
	assign_json_at(slot, value);
}

fn assign_json_at(current: &mut Value, value: Value) {
	if let (Value::Object(map), Value::Object(new_map)) = (&mut *current, &value) {
		let stale: Vec<String> =
			map.keys().filter(|name| !new_map.contains_key(*name)).cloned().collect();
		for name in stale {
			map.remove(&name);
		}
		for (name, child) in new_map.iter() {
			if let Some(slot) = map.get_mut(name) {
				assign_json_at(slot, child.clone());
			} else {
				map.insert(name.clone(), child.clone());
			}
		}
		return;
	}
	if let (Value::Array(items), Value::Array(new_items)) = (&mut *current, &value) {
		if items.len() <= new_items.len() {
			for (index, child) in new_items.iter().enumerate() {
				if index < items.len() {
					assign_json_at(&mut items[index], child.clone());
				} else {
					items.push(child.clone());
				}
			}
			return;
		}
	}
	if *current != value {
		*current = value;
	}
}
