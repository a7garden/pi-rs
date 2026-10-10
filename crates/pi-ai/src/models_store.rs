//! Port of `packages/ai/src/models-store.ts`.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::types::AnyModel;
use crate::utils::abort::AbortSignal;
use crate::utils::thrown::ThrownValue;

/// Persisted model catalog entry (`ModelsStoreEntry`).
///
/// Serialization matches the lax reader below and the upstream JSON: camelCase
/// field names, absent for `None` optionals (JS `undefined` keys are dropped
/// by `JSON.stringify`).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsStoreEntry {
    /// Persisted models of every type.
    pub models: Vec<AnyModel>,
    /// Unix timestamp from the remote catalog's Last-Modified header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<i64>,
    /// Unix timestamp of the last completed remote check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<i64>,
    /// Opaque validator from the remote catalog's ETag header, stored verbatim
    /// (quotes included) and echoed back as If-None-Match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

/// Lax read (upstream `withKnownModelTypes`): entries persisted by a newer
/// version can carry model types this version does not know; those models are
/// dropped instead of failing the whole entry read.
impl<'de> Deserialize<'de> for ModelsStoreEntry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("store entry must be an object"))?;
        let mut models = Vec::new();
        if let Some(list) = object.get("models").and_then(|models| models.as_array()) {
            for model in list {
                if let Ok(model) = AnyModel::from_json(model) {
                    models.push(model);
                }
            }
        }
        Ok(ModelsStoreEntry {
            models,
            last_modified: object
                .get("lastModified")
                .and_then(serde_json::Value::as_i64),
            checked_at: object.get("checkedAt").and_then(serde_json::Value::as_i64),
            etag: object
                .get("etag")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        })
    }
}

/// Operation options for [`ModelsStore`] (`ModelsStoreOperationOptions`).
#[derive(Debug, Clone, Default)]
pub struct ModelsStoreOperationOptions {
    pub signal: Option<AbortSignal>,
}

/// Persistent model catalogs keyed by provider ID (`ModelsStore`).
#[async_trait]
pub trait ModelsStore: Send + Sync {
    async fn read(
        &self,
        provider_id: &str,
        options: Option<ModelsStoreOperationOptions>,
    ) -> Result<Option<ModelsStoreEntry>, ThrownValue>;

    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        options: Option<ModelsStoreOperationOptions>,
    ) -> Result<(), ThrownValue>;

    async fn delete(
        &self,
        provider_id: &str,
        options: Option<ModelsStoreOperationOptions>,
    ) -> Result<(), ThrownValue>;
}

/// In-memory [`ModelsStore`] (`InMemoryModelsStore`). Entries are deep-cloned
/// on read and write, matching upstream `structuredClone`.
#[derive(Default)]
pub struct InMemoryModelsStore {
    entries: Mutex<BTreeMap<String, ModelsStoreEntry>>,
}

impl InMemoryModelsStore {
    pub fn new() -> InMemoryModelsStore {
        InMemoryModelsStore {
            entries: Mutex::new(BTreeMap::new()),
        }
    }
}

fn check_aborted(options: &Option<ModelsStoreOperationOptions>) -> Result<(), ThrownValue> {
    if let Some(signal) = options.as_ref().and_then(|options| options.signal.as_ref()) {
        return signal.check().map_err(|reason| ThrownValue::error(reason));
    }
    Ok(())
}

#[async_trait]
impl ModelsStore for InMemoryModelsStore {
    async fn read(
        &self,
        provider_id: &str,
        options: Option<ModelsStoreOperationOptions>,
    ) -> Result<Option<ModelsStoreEntry>, ThrownValue> {
        check_aborted(&options)?;
        let entries = self.entries.lock();
        Ok(entries.get(provider_id).cloned())
    }

    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        options: Option<ModelsStoreOperationOptions>,
    ) -> Result<(), ThrownValue> {
        check_aborted(&options)?;
        let mut entries = self.entries.lock();
        entries.insert(provider_id.to_string(), entry);
        Ok(())
    }

    async fn delete(
        &self,
        provider_id: &str,
        options: Option<ModelsStoreOperationOptions>,
    ) -> Result<(), ThrownValue> {
        check_aborted(&options)?;
        let mut entries = self.entries.lock();
        entries.remove(provider_id);
        Ok(())
    }
}

/// Arc helper used by `create_models`.
pub fn in_memory_models_store() -> Arc<dyn ModelsStore> {
    Arc::new(InMemoryModelsStore::new())
}
