//! Port of `packages/ai/src/model-catalog.ts`.
//!
//! Upstream, this module is mostly TypeScript type-level machinery:
//! `ChatModelCatalog<TGroups, TProvider>` and friends derive literal-typed
//! keyed catalogs (`MODELS["xai"]["grok-4.5"].api` is the literal
//! `"openai-responses"` at compile time). The runtime surface is
//! `flattenModelCatalog(groups, type)` plus the three exported wrappers
//! `flattenChatModelCatalog` / `flattenImageModelCatalog` /
//! `flattenClassifierModelCatalog`, which filter a grouped record
//! (`api → modelKey → model`) down to one map keyed by `model.id`, keeping
//! only models whose `type` matches, preserving iteration order.
//!
//! Rust has no literal-type machinery to port, so the catalogs are plain
//! insertion-ordered keyed maps ([`ModelCatalog`], [`OrderedMap`]);
//! type-level guarantees are carried by the concrete [`AnyModel`] variants
//! (deviation D15). The embedded all-provider catalogs built from these
//! helpers live in [`crate::models_generated`].
//!
//! # Deviations
//!
//! - **D15 — type-level catalog typing is not portable.** Upstream derives
//!   literal key/api/provider types from the group records; the Rust port
//!   returns an insertion-ordered keyed map (first-appearance order kept,
//!   later duplicate ids overwrite in place, like `Object.fromEntries`).
//!   Consumers narrow with `AnyModel` accessors.
//! - **D16 — non-object groups and models are skipped.** Upstream
//!   `Object.values` tolerates non-object group values (yields `[]`); the
//!   port skips non-object groups/models and models without a string `id`
//!   instead of relying on JS coercion quirks.

use std::ops::Index;

use serde_json::Value;

use crate::types::{AnyModel, Model, ModelType};

/// Grouped model data (`ModelGroups = Record<string, Record<string, object>>
/// upstream`): api name → model key → raw model object. Group and model
/// values must be JSON objects; see deviation D16.
pub type ModelGroups = serde_json::Map<String, Value>;

/// Insertion-ordered string-keyed map backing the flattened catalogs.
/// `serde_json::Map`'s inherent API is concrete to `Value`, so catalogs of
/// [`AnyModel`] carry this tiny index-map stand-in: `insert` replaces an
/// existing key in place (JS object semantics) or appends.
pub struct OrderedMap<V> {
	entries: Vec<(String, V)>,
}

impl<V> OrderedMap<V> {
	pub fn new() -> OrderedMap<V> {
		OrderedMap { entries: Vec::new() }
	}

	/// Upstream `record[key] = value`: replace in place, else append.
	pub fn insert(&mut self, key: impl Into<String>, value: V) {
		let key = key.into();
		match self.entries.iter_mut().find(|(existing, _)| *existing == key) {
			Some((existing, slot)) => {
				*existing = key;
				*slot = value;
			}
			None => self.entries.push((key, value)),
		}
	}

	pub fn get(&self, key: &str) -> Option<&V> {
		self.entries.iter().find(|(existing, _)| existing == key).map(|(_, value)| value)
	}

	pub fn contains_key(&self, key: &str) -> bool {
		self.get(key).is_some()
	}

	/// Remove and return the value stored under `key`.
	pub fn remove(&mut self, key: &str) -> Option<V> {
		let index = self.entries.iter().position(|(existing, _)| existing == key)?;
		Some(self.entries.remove(index).1)
	}

	/// Remove every entry, keeping the map usable (`Map.clear` upstream).
	pub fn clear(&mut self) {
		self.entries.clear();
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	pub fn keys(&self) -> impl Iterator<Item = &str> {
		self.entries.iter().map(|(key, _)| key.as_str())
	}

	pub fn values(&self) -> impl Iterator<Item = &V> {
		self.entries.iter().map(|(_, value)| value)
	}

	pub fn iter(&self) -> impl Iterator<Item = (&str, &V)> {
		self.entries.iter().map(|(key, value)| (key.as_str(), value))
	}

	/// Drop the keys, keeping order (`Object.values(catalog)` upstream).
	pub fn into_models(self) -> Vec<V>
	where
		V: Clone,
	{
		self.entries.into_iter().map(|(_, value)| value).collect()
	}
}

impl<V> Index<&str> for OrderedMap<V> {
	type Output = V;

	fn index(&self, key: &str) -> &V {
		self.get(key).unwrap_or_else(|| panic!("catalog key not found: {key}"))
	}
}

impl<V> Default for OrderedMap<V> {
	fn default() -> OrderedMap<V> {
		OrderedMap::new()
	}
}

impl<'a, V> IntoIterator for &'a OrderedMap<V> {
	type Item = (&'a str, &'a V);
	type IntoIter = Box<dyn Iterator<Item = (&'a str, &'a V)> + 'a>;

	fn into_iter(self) -> Self::IntoIter {
		Box::new(self.entries.iter().map(|(key, value)| (key.as_str(), value)))
	}
}

/// One flattened model catalog keyed by `model.id` (`ChatModelCatalog` /
/// `ImageModelCatalog` / `ClassifierModelCatalog`). Insertion order follows
/// group order then model order, like `Object.fromEntries` upstream.
pub type ModelCatalog = OrderedMap<AnyModel>;

/// `type` tag of a raw model object (`(model as { type?: unknown }).type`).
fn model_type_tag(model: &Value) -> Option<&str> {
	model.get("type").and_then(Value::as_str)
}

fn type_tag(type_: ModelType) -> &'static str {
	match type_ {
		ModelType::Chat => "chat",
		ModelType::Image => "image",
		ModelType::Classifier => "classifier",
	}
}

/// Upstream `flattenModelCatalog`: filter every grouped model by `type` and
/// key the survivors by their `id`. Later duplicate ids overwrite earlier
/// ones in place (`Object.fromEntries` semantics); first-appearance order is
/// preserved.
pub fn flatten_model_catalog(groups: &Value, type_: ModelType) -> ModelCatalog {
	let expected = type_tag(type_);
	let mut catalog = ModelCatalog::new();
	let Some(groups_object) = groups.as_object() else {
		return catalog;
	};
	for (_api, group) in groups_object {
		let Some(models) = group.as_object() else {
			continue;
		};
		for (_key, model) in models {
			if model_type_tag(model) != Some(expected) {
				continue;
			}
			let Some(id) = model.get("id").and_then(Value::as_str) else {
				continue;
			};
			match AnyModel::from_json(model) {
				Ok(entry) => {
					catalog.insert(id, entry);
				}
				// A model this version cannot represent is dropped, matching
				// the unknown-type tolerance of `withKnownModelTypes`.
				Err(_) => continue,
			}
		}
	}
	catalog
}

/// Upstream `flattenChatModelCatalog`. The `provider` parameter is unused,
/// exactly like upstream's `_provider` (models already carry `provider`).
pub fn flatten_chat_model_catalog(provider: &str, groups: &Value) -> ModelCatalog {
	let _ = provider;
	flatten_model_catalog(groups, ModelType::Chat)
}

/// Upstream `flattenImageModelCatalog`.
pub fn flatten_image_model_catalog(provider: &str, groups: &Value) -> ModelCatalog {
	let _ = provider;
	flatten_model_catalog(groups, ModelType::Image)
}

/// Upstream `flattenClassifierModelCatalog`.
pub fn flatten_classifier_model_catalog(provider: &str, groups: &Value) -> ModelCatalog {
	let _ = provider;
	flatten_model_catalog(groups, ModelType::Classifier)
}

/// Chat-model view over a [`ModelCatalog`] (drop the keys).
pub fn chat_models(catalog: &ModelCatalog) -> Vec<Model> {
	catalog
		.values()
		.filter_map(|model| match model {
			AnyModel::Chat(model) => Some(model.clone()),
			_ => None,
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::models_generated;
	use crate::types::ModelThinkingLevel;
	use serde_json::json;

	fn sample_groups() -> Value {
		json!({
			"openai-completions": {
				"chat:model-a": {
					"type": "chat", "id": "model-a", "name": "Model A", "api": "openai-completions",
					"provider": "test", "baseUrl": "https://example.test/v1",
					"reasoning": false, "input": ["text"],
					"cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 1000, "maxTokens": 100
				},
				"chat:model-b": {
					"type": "chat", "id": "model-b", "name": "Model B", "api": "openai-completions",
					"provider": "test", "baseUrl": "https://example.test/v1",
					"reasoning": true, "input": ["text"],
					"cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 1000, "maxTokens": 100
				}
			},
			"anthropic-messages": {
				"chat:model-c": {
					"type": "chat", "id": "model-c", "name": "Model C", "api": "anthropic-messages",
					"provider": "test", "baseUrl": "https://example.test/v1",
					"reasoning": false, "input": ["text"],
					"cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 1000, "maxTokens": 100
				},
				"image:model-a": {
					"type": "image", "id": "model-a", "name": "Model A Image", "api": "openrouter-images",
					"provider": "test", "baseUrl": "https://example.test/v1",
					"input": ["text"], "output": ["image"],
					"cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 }
				},
				"classifier:model-a": {
					"type": "classifier", "id": "model-a", "name": "Model A Classifier", "api": "llama-cpp-classify",
					"provider": "test", "baseUrl": "https://example.test/v1",
					"input": ["text"],
					"cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 1000
				}
			}
		})
	}

	#[test]
	fn flattens_by_type_keying_by_model_id_in_group_order() {
		let groups = sample_groups();
		let chat = flatten_chat_model_catalog("test", &groups);
		let chat_ids: Vec<&str> = chat.keys().collect();
		// Group order (openai-completions, then anthropic-messages), then model order.
		assert_eq!(chat_ids, ["model-a", "model-b", "model-c"]);
		assert!(chat.values().all(|model| model.model_type() == ModelType::Chat));
		assert_eq!(chat["model-c"].api(), "anthropic-messages");

		let image = flatten_image_model_catalog("test", &groups);
		assert_eq!(image.len(), 1);
		let AnyModel::Image(image_model) = &image["model-a"] else {
			panic!("image model expected");
		};
		assert_eq!(image_model.output, vec![crate::types::ModelOutput::Image]);

		let classifier = flatten_classifier_model_catalog("test", &groups);
		assert_eq!(classifier.len(), 1);
		assert_eq!(classifier["model-a"].api(), "llama-cpp-classify");
	}

	#[test]
	fn duplicate_ids_from_later_groups_overwrite() {
		let groups = json!({
			"openai-completions": {
				"chat:shared": {
					"type": "chat", "id": "shared", "name": "First", "api": "openai-completions",
					"provider": "test", "baseUrl": "https://example.test/v1", "reasoning": false,
					"input": ["text"], "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 1, "maxTokens": 1
				}
			},
			"anthropic-messages": {
				"chat:shared": {
					"type": "chat", "id": "shared", "name": "Second", "api": "anthropic-messages",
					"provider": "test", "baseUrl": "https://example.test/v1", "reasoning": false,
					"input": ["text"], "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
					"contextWindow": 2, "maxTokens": 2
				}
			}
		});
		let chat = flatten_chat_model_catalog("test", &groups);
		assert_eq!(chat.len(), 1);
		assert_eq!(chat["shared"].api(), "anthropic-messages");
		let AnyModel::Chat(shared) = &chat["shared"] else {
			panic!("shared is a chat model");
		};
		assert_eq!(shared.base.name, "Second");
	}

	#[test]
	fn skips_models_without_a_matching_type_or_string_id() {
		let groups = json!({
			"openai-completions": {
				"a": { "id": "no-type", "api": "openai-completions" },
				"b": { "type": "chat", "api": "openai-completions" },
				"c": { "type": "image", "id": 7 }
			}
		});
		assert!(flatten_chat_model_catalog("test", &groups).is_empty());
	}

	/// Runtime assertions from upstream `test/model-catalog-types.test.ts`
	/// (the `expectTypeOf` halves are compile-time checks with no Rust
	/// equivalent — deviation D15 — and the second case's runtime assertion
	/// is subsumed by the full gpt sweep below).
	#[test]
	fn generated_catalog_routes_xai_and_github_copilot_through_responses() {
		let models = models_generated::models();

		// "derives model API, ID, and provider literals from grouped model data"
		assert_eq!(models["xai"]["grok-4.5"].api(), "openai-responses");
		assert_eq!(models["xai"]["grok-4.5"].id(), "grok-4.5");
		assert_eq!(models["xai"]["grok-4.5"].provider(), "xai");
		assert_eq!(models["xai"]["grok-4.6"].api(), "openai-responses");
		assert_eq!(models["xai"]["grok-4.6"].id(), "grok-4.6");
		assert_eq!(models["xai"]["grok-4.7"].api(), "openai-responses");
		assert_eq!(models["xai"]["grok-4.7"].id(), "grok-4.7");
		assert_eq!(models["xai"]["grok-4.3"].api(), "openai-responses");

		// "routes GitHub Copilot Grok 4.5 through the Responses API"
		assert_eq!(models["github-copilot"]["grok-4.5"].api(), "openai-responses");

		// "routes all GitHub Copilot GPT models through the Responses API"
		// (regression for earendil-works/pi#9209)
		let gpt_models: Vec<&AnyModel> = models["github-copilot"]
			.values()
			.filter(|model| model.id().starts_with("gpt-"))
			.collect();
		assert!(!gpt_models.is_empty());
		assert!(gpt_models.iter().all(|model| model.api() == "openai-responses"));
		for model_id in ["gpt-6-sol", "gpt-6-luna"] {
			let AnyModel::Chat(model) = &models["github-copilot"][model_id] else {
				panic!("{model_id} is a chat model");
			};
			assert_eq!(model.base.api, "openai-responses");
			assert_eq!(model.context_window, 1_000_000);
			assert_eq!(model.max_tokens, 128_000);
			let map = model.thinking_level_map.as_ref().expect("thinkingLevelMap present");
			assert_eq!(map.get(&ModelThinkingLevel::Off), Some(&Some("none".to_string())));
			assert_eq!(map.get(&ModelThinkingLevel::Max), Some(&Some("max".to_string())));
		}
	}

	/// Upstream `model-catalog-types.test.ts` also pins
	/// `GITHUB_COPILOT_MODELS["gpt-6-astra"].api` as a literal; asserted here
	/// at runtime against the embedded catalog.
	#[test]
	fn generated_catalog_includes_gpt_6_astra_through_responses() {
		assert_eq!(models_generated::models()["github-copilot"]["gpt-6-astra"].api(), "openai-responses");
	}
}
