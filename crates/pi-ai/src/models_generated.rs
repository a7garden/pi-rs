//! Port of `packages/ai/src/models.generated.ts` — the all-provider model
//! catalogs assembled from every `providers/*.models.ts` shard.
//!
//! # Data embedding choice
//!
//! Upstream `models.generated.ts` is a **pure data aggregation**: 42 imports
//! of per-provider shards, re-exported through
//! `flatten{Chat,Image,Classifier}ModelCatalog` into three insertion-ordered
//! records `MODELS`, `IMAGE_MODELS`, `CLASSIFIER_MODELS`
//! (`providerId → modelId → model`). It contains no logic worth translating.
//!
//! The shard data itself is also generated (`providers/data/*.json` imported
//! by `providers/*.models.ts`), so the mechanically faithful move is to embed
//! the aggregated catalogs as JSON:
//! `models_generated.data.json` is produced from the upstream generator
//! output by importing `models.generated.ts` and `JSON.stringify`-ing the
//! three exported records (42 providers each, byte-identical key order).
//! This module `include_str!`s that file and parses every leaf into
//! [`AnyModel`] once, lazily. Generating a Rust source module with struct
//! literals instead would balloon compile times for zero fidelity gain, and
//! parsing preserves upstream key order exactly (`serde_json`
//! `preserve_order`), so lookups and iteration match upstream objects.
//!
//! Regeneration (when upstream data changes): re-run the upstream
//! `npm run generate-models` inside `upstream/pi/packages/ai`, then dump
//! `{ models, imageModels, classifierModels }` from `models.generated.ts` to
//! `models_generated.data.json`.
//!
//! All 42 providers are embedded — `models.generated.ts` re-exports every
//! shard, and the data is trivially embeddable, so out-of-scope providers
//! (only `anthropic`, `openai`, `typesafe`, `meta` have Rust provider
//! factories this wave) still round-trip for catalog consumers and tests.
//!
//! # Deviations
//!
//! - **D17 — lazy parsed statics instead of object literals.** Upstream
//!   builds plain objects at module load; the port parses the embedded JSON
//!   on first access into `&'static` maps. A malformed embedded file panics
//!   at first access — a build-time data bug caught by the tests below, never
//!   a runtime input path.
//! - **D18 — provider-private extra fields are not represented.** The
//!   embedded data carries fields the upstream `BaseModel` interface does not
//!   declare (e.g. the `radius` shard's `lab`/`enabled`/`providers` keys).
//!   The typed [`AnyModel`] view drops them, like upstream's typed access;
//!   no upstream code path outside the radius provider reads them.

use std::sync::LazyLock;

use serde_json::Value;

use crate::model_catalog::OrderedMap;
use crate::types::AnyModel;

/// Embedded output of the upstream model-data generator (see module docs).
static CATALOG_DATA: &str = include_str!("models_generated.data.json");

/// One provider's flattened catalog: model id → model
/// (`typeof PROVIDER_MODELS` upstream). Insertion order = upstream key order.
pub type ProviderCatalog = crate::model_catalog::ModelCatalog;

/// All providers' flattened catalogs (`typeof MODELS` upstream):
/// provider id → model id → model.
pub type GeneratedCatalog = OrderedMap<ProviderCatalog>;

struct GeneratedCatalogs {
	models: GeneratedCatalog,
	image_models: GeneratedCatalog,
	classifier_models: GeneratedCatalog,
}

fn parse_provider_catalog(value: &Value) -> ProviderCatalog {
	let mut catalog = ProviderCatalog::new();
	let Some(object) = value.as_object() else {
		return catalog;
	};
	for (id, model) in object {
		if let Ok(entry) = AnyModel::from_json(model) {
			catalog.insert(id.clone(), entry);
		}
	}
	catalog
}

fn parse_catalogs() -> GeneratedCatalogs {
	let root: Value = serde_json::from_str(CATALOG_DATA).expect("embedded model catalog parses");
	let parse = |key: &str| -> GeneratedCatalog {
		let mut providers = GeneratedCatalog::new();
		if let Some(object) = root.get(key).and_then(Value::as_object) {
			for (provider, models) in object {
				providers.insert(provider.clone(), parse_provider_catalog(models));
			}
		}
		providers
	};
	GeneratedCatalogs {
		models: parse("models"),
		image_models: parse("imageModels"),
		classifier_models: parse("classifierModels"),
	}
}

static CATALOGS: LazyLock<GeneratedCatalogs> = LazyLock::new(parse_catalogs);

/// Upstream `MODELS`: flattened chat catalogs for every provider.
pub fn models() -> &'static GeneratedCatalog {
	&CATALOGS.models
}

/// Upstream `IMAGE_MODELS`.
pub fn image_models() -> &'static GeneratedCatalog {
	&CATALOGS.image_models
}

/// Upstream `CLASSIFIER_MODELS`.
pub fn classifier_models() -> &'static GeneratedCatalog {
	&CATALOGS.classifier_models
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::ModelType;

	const PROVIDER_COUNT: usize = 42;

	#[test]
	fn embeds_every_provider_shard_in_all_three_catalogs() {
		for catalog in [models(), image_models(), classifier_models()] {
			assert_eq!(catalog.len(), PROVIDER_COUNT, "provider count");
		}
		// Insertion order matches the upstream aggregation order (the first
		// and last shards of models.generated.ts's import list).
		let providers: Vec<&str> = models().keys().collect();
		assert_eq!(providers.first(), Some(&"amazon-bedrock"));
		assert_eq!(providers.last(), Some(&"zai-coding-cn"));
	}

	#[test]
	fn every_entry_keys_by_its_own_id_and_carries_its_provider() {
		for catalog in [models(), image_models(), classifier_models()] {
			for (provider, entries) in catalog.iter() {
				assert!(!provider.is_empty());
				for (id, model) in entries {
					assert_eq!(model.id(), id, "{provider}/{id} keyed by id");
					assert_eq!(model.provider(), provider, "{provider}/{id} carries provider");
					assert!(!model.api().is_empty(), "{provider}/{id} has an api");
				}
			}
		}
	}

	/// The four in-scope providers this wave, as a data regression pin
	/// (counts from the upstream generator output at porting time).
	#[test]
	fn in_scope_provider_counts() {
		assert_eq!(models()["anthropic"].len(), 16);
		assert_eq!(models()["openai"].len(), 44);
		assert_eq!(models()["meta"].len(), 5);
		// typesafe routes every chat model through the shared system-one
		// pipeline; its static shard currently lists no chat models.
		assert_eq!(models()["typesafe"].len(), 0);
		assert_eq!(classifier_models()["typesafe"].len(), 1);
	}

	#[test]
	fn image_and_classifier_catalogs_carry_the_multi_type_shards() {
		assert_eq!(image_models()["openrouter"].len(), 59);
		assert_eq!(image_models()["amazon-bedrock"].len(), 0);
		assert_eq!(classifier_models()["cloudflare-workers-ai"].len(), 3);
		assert_eq!(classifier_models()["opencode"].len(), 2);
		assert_eq!(classifier_models()["openrouter"].len(), 13);
		assert_eq!(classifier_models()["vercel-ai-gateway"].len(), 4);
	}

	/// Chat and image models may share an id; they key separately per type
	/// (upstream model-data invariant, `model-data-validation.test.ts`).
	#[test]
	fn chat_and_image_entries_share_ids_without_collision() {
		let shared: Vec<&str> = models()["openrouter"]
			.keys()
			.filter(|id| image_models()["openrouter"].contains_key(id))
			.collect();
		assert!(!shared.is_empty(), "openrouter has chat+image models sharing ids");
		for id in shared {
			assert_eq!(models()["openrouter"][id].model_type(), ModelType::Chat);
			assert_eq!(image_models()["openrouter"][id].model_type(), ModelType::Image);
		}
	}

	/// Every embedded model must deserialize into a known [`AnyModel`]
	/// variant — the runtime counterpart of upstream
	/// `withKnownModelTypes`/`validateModelDataDirectory` ("rejects a model
	/// without a known type").
	#[test]
	fn parses_every_model_into_a_known_type() {
		let mut total = 0usize;
		for catalog in [models(), image_models(), classifier_models()] {
			for entries in catalog.values() {
				for model in entries.values() {
					// AnyModel parsing is strict; re-parse the JSON form as a
					// round-trip check.
					let json = model.to_json();
					AnyModel::from_json(&json).unwrap_or_else(|error| {
						panic!("model {}/{} re-parses: {error}", model.provider(), model.id())
					});
					total += 1;
				}
			}
		}
		assert!(total > 500, "embedded catalogs are populated ({total} models)");
	}
}
