//! Port of `packages/coding-agent/src/core/runtime-credentials.ts`.
//!
//! Async credential store overlay for non-persistent runtime API keys.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthOperationOptions, Credential, CredentialInfo, CredentialModifyFn,
    CredentialStore,
};
use pi_ai::utils::thrown::ThrownValue;

/// Upstream `RuntimeCredentials`: an in-memory API-key overlay over an
/// underlying [`CredentialStore`]. Overrides win over stored credentials on
/// `read` and appear in `list`.
// D16-style ordering note: `list` merges via `BTreeMap`, so entries are
// ordered by provider id rather than upstream insertion order (same
// deviation as pi-ai's `InMemoryCredentialStore`).
pub struct RuntimeCredentials {
    store: Arc<dyn CredentialStore>,
    overrides: Mutex<BTreeMap<String, String>>,
}

impl RuntimeCredentials {
    pub fn new(store: Arc<dyn CredentialStore>) -> RuntimeCredentials {
        RuntimeCredentials {
            store,
            overrides: Mutex::new(BTreeMap::new()),
        }
    }

    /// `setRuntimeApiKey` — non-persistent override.
    pub fn set_runtime_api_key(&self, provider_id: &str, api_key: &str) {
        self.overrides
            .lock()
            .insert(provider_id.to_string(), api_key.to_string());
    }

    /// `removeRuntimeApiKey`.
    pub fn remove_runtime_api_key(&self, provider_id: &str) {
        self.overrides.lock().remove(provider_id);
    }

    /// `hasRuntimeApiKey`.
    pub fn has_runtime_api_key(&self, provider_id: &str) -> bool {
        self.overrides.lock().contains_key(provider_id)
    }
}

#[async_trait::async_trait]
impl CredentialStore for RuntimeCredentials {
    async fn read(
        &self,
        provider_id: &str,
        options: Option<AuthOperationOptions>,
    ) -> Result<Option<Credential>, ThrownValue> {
        if let Some(signal) = options.as_ref().and_then(|options| options.signal.as_ref()) {
            signal.check().map_err(ThrownValue::error)?;
        }
        if let Some(key) = self.overrides.lock().get(provider_id) {
            return Ok(Some(Credential::ApiKey(ApiKeyCredential {
                key: Some(key.clone()),
                env: None,
            })));
        }
        self.store.read(provider_id, options).await
    }

    async fn list(
        &self,
        options: Option<AuthOperationOptions>,
    ) -> Result<Vec<CredentialInfo>, ThrownValue> {
        let mut entries: BTreeMap<String, CredentialInfo> = self
            .store
            .list(options.clone())
            .await?
            .into_iter()
            .map(|entry| (entry.provider_id.clone(), entry))
            .collect();
        if let Some(signal) = options.as_ref().and_then(|options| options.signal.as_ref()) {
            signal.check().map_err(ThrownValue::error)?;
        }
        for provider_id in self.overrides.lock().keys() {
            entries.insert(
                provider_id.clone(),
                CredentialInfo {
                    provider_id: provider_id.clone(),
                    credential_type: "api_key".to_string(),
                },
            );
        }
        Ok(entries.into_values().collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        modify_fn: CredentialModifyFn,
        options: Option<AuthOperationOptions>,
    ) -> Result<Option<Credential>, ThrownValue> {
        self.store.modify(provider_id, modify_fn, options).await
    }

    async fn delete(
        &self,
        provider_id: &str,
        options: Option<AuthOperationOptions>,
    ) -> Result<(), ThrownValue> {
        if let Some(signal) = options.as_ref().and_then(|options| options.signal.as_ref()) {
            signal.check().map_err(ThrownValue::error)?;
        }
        self.store.delete(provider_id, options).await?;
        self.overrides.lock().remove(provider_id);
        Ok(())
    }
}
