//! Port of `packages/ai/src/auth/credential-store.ts`.
//!
//! Default in-memory credential store. Apps inject persistent stores. Keyed
//! by provider id, one credential per provider; mutations serialize per
//! provider through a per-id async mutex (upstream: a promise chain).
//!
//! Deviations:
//! - D16: `list` returns entries sorted by provider id (`BTreeMap`) instead of
//!   JS `Map` insertion order.
//! - D17: upstream races the whole queued task with the operation signal, so a
//!   signal aborting mid-task resolves the caller while the task finishes in
//!   the background. The Rust port races only the lock wait (mirroring the
//!   `throwIfAborted` before the task) and re-checks the signal at the
//!   explicit points; an in-flight mutation is otherwise awaited, and is
//!   dropped mid-await only when the caller's future itself is dropped (the
//!   crate-wide abort convention — see `utils/abort.rs`).

use std::collections::BTreeMap;
use std::sync::{Arc};
use {parking_lot::Mutex};

use crate::auth::types::{
	AuthOperationOptions, Credential, CredentialInfo, CredentialModifyFn, CredentialStore,
};
use crate::utils::abort::{operation_signal, race_with_abort_signal};
use crate::utils::thrown::ThrownValue;

/// Default in-memory credential store (`InMemoryCredentialStore`).
#[derive(Default)]
pub struct InMemoryCredentialStore {
	credentials: Mutex<BTreeMap<String, Credential>>,
	chains: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl InMemoryCredentialStore {
	pub fn new() -> InMemoryCredentialStore {
		InMemoryCredentialStore::default()
	}

	/// Per-provider serialization chain (upstream `enqueue`'s chain map).
	fn chain_for(&self, provider_id: &str) -> Arc<tokio::sync::Mutex<()>> {
		let mut chains = self.chains.lock();
		chains
			.entry(provider_id.to_string())
			.or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
			.clone()
	}
}

#[async_trait::async_trait]
impl CredentialStore for InMemoryCredentialStore {
	async fn read(
		&self,
		provider_id: &str,
		options: Option<AuthOperationOptions>,
	) -> Result<Option<Credential>, ThrownValue> {
		if let Some(signal) = options.as_ref().and_then(|options| options.signal.as_ref()) {
			signal.check().map_err(ThrownValue::error)?;
		}
		Ok(self.credentials.lock().get(provider_id).cloned())
	}

	async fn list(&self, options: Option<AuthOperationOptions>) -> Result<Vec<CredentialInfo>, ThrownValue> {
		if let Some(signal) = options.as_ref().and_then(|options| options.signal.as_ref()) {
			signal.check().map_err(ThrownValue::error)?;
		}
		Ok(self
			.credentials
			.lock()
			.iter()
			.map(|(provider_id, credential)| CredentialInfo {
				provider_id: provider_id.clone(),
				credential_type: credential.type_name().to_string(),
			})
			.collect())
	}

	async fn modify(
		&self,
		provider_id: &str,
		modify_fn: CredentialModifyFn,
		options: Option<AuthOperationOptions>,
	) -> Result<Option<Credential>, ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|options| options.signal.as_ref()));
		let chain = self.chain_for(provider_id);
		// Wait for the previous task, aborting the wait when the signal fires.
		let _guard =
			race_with_abort_signal(chain.lock(), signal.clone()).await.map_err(ThrownValue::error)?;
		signal.check().map_err(ThrownValue::error)?;
		let current = self.credentials.lock().get(provider_id).cloned();
		let next = modify_fn(current.as_ref()).await?;
		signal.check().map_err(ThrownValue::error)?;
		if let Some(next) = &next {
			self.credentials
				.lock()
				.insert(provider_id.to_string(), next.clone());
		}
		// Upstream: `return next ?? current`.
		Ok(next.or(current))
	}

	async fn delete(&self, provider_id: &str, options: Option<AuthOperationOptions>) -> Result<(), ThrownValue> {
		let signal = operation_signal(options.as_ref().and_then(|options| options.signal.as_ref()));
		let chain = self.chain_for(provider_id);
		let _guard =
			race_with_abort_signal(chain.lock(), signal.clone()).await.map_err(ThrownValue::error)?;
		signal.check().map_err(ThrownValue::error)?;
		self.credentials.lock().remove(provider_id);
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::types::{ApiKeyCredential, OAuthCredential, OAuthCredentials};
	use crate::utils::abort::AbortController;

	fn api_credential(key: &str) -> Credential {
		Credential::ApiKey(ApiKeyCredential { key: Some(key.to_string()), env: None })
	}

	fn oauth_credential(expires_in_ms: i64) -> Credential {
		Credential::OAuth(OAuthCredential {
			credentials: OAuthCredentials {
				refresh: "r".to_string(),
				access: "a".to_string(),
				expires: unix_now_ms() + expires_in_ms,
				extra: BTreeMap::new(),
			},
		})
	}

	fn unix_now_ms() -> i64 {
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|duration| duration.as_millis() as i64)
			.unwrap_or(0)
	}

	#[tokio::test]
	async fn read_missing_returns_none_and_modify_stores() {
		let store = InMemoryCredentialStore::new();
		assert_eq!(store.read("p", None).await.unwrap(), None);
		let post = store
			.modify(
				"p",
				Box::new(|_| Box::pin(async { Ok(Some(api_credential("k"))) })),
				None,
			)
			.await
			.unwrap();
		assert_eq!(post, Some(api_credential("k")));
		assert_eq!(store.read("p", None).await.unwrap(), Some(api_credential("k")));
	}

	#[tokio::test]
	async fn modify_returning_none_keeps_current_and_returns_it() {
		let store = InMemoryCredentialStore::new();
		store
			.modify("p", Box::new(|_| Box::pin(async { Ok(Some(api_credential("first"))) })), None)
			.await
			.unwrap();
		let post = store
			.modify("p", Box::new(|_| Box::pin(async { Ok(None) })), None)
			.await
			.unwrap();
		assert_eq!(post, Some(api_credential("first")));
		assert_eq!(store.read("p", None).await.unwrap(), Some(api_credential("first")));
	}

	#[tokio::test]
	async fn modify_sees_current_credential() {
		let store = InMemoryCredentialStore::new();
		store
			.modify("p", Box::new(|_| Box::pin(async { Ok(Some(api_credential("first"))) })), None)
			.await
			.unwrap();
		let post = store
			.modify(
				"p",
				Box::new(|current| {
					Box::pin(async move {
						assert_eq!(current, Some(&api_credential("first")));
						Ok(Some(api_credential("second")))
					})
				}),
				None,
			)
			.await
			.unwrap();
		assert_eq!(post, Some(api_credential("second")));
	}

	#[tokio::test]
	async fn delete_removes_entry() {
		let store = InMemoryCredentialStore::new();
		store
			.modify("p", Box::new(|_| Box::pin(async { Ok(Some(api_credential("k"))) })), None)
			.await
			.unwrap();
		store.delete("p", None).await.unwrap();
		assert_eq!(store.read("p", None).await.unwrap(), None);
		assert!(store.delete("p", None).await.is_ok());
	}

	#[tokio::test]
	async fn list_reports_metadata() {
		let store = InMemoryCredentialStore::new();
		store
			.modify("b", Box::new(|_| Box::pin(async { Ok(Some(oauth_credential(0))) })), None)
			.await
			.unwrap();
		store
			.modify("a", Box::new(|_| Box::pin(async { Ok(Some(api_credential("k"))) })), None)
			.await
			.unwrap();
		let listed = store.list(None).await.unwrap();
		assert_eq!(
			listed,
			vec![
				CredentialInfo { provider_id: "a".to_string(), credential_type: "api_key".to_string() },
				CredentialInfo { provider_id: "b".to_string(), credential_type: "oauth".to_string() },
			]
		);
	}

	#[tokio::test]
	async fn modifies_serialize_per_provider() {
		let store = Arc::new(InMemoryCredentialStore::new());
		let first = {
			let store = store.clone();
			tokio::spawn(async move {
				store
					.modify(
						"p",
						Box::new(|_| {
							Box::pin(async {
								tokio::time::sleep(std::time::Duration::from_millis(30)).await;
								Ok(Some(api_credential("first")))
							})
						}),
						None,
					)
					.await
			})
		};
		tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		// Must queue behind the in-flight modify and observe its write.
		let second = store
			.modify(
				"p",
				Box::new(|current| {
					Box::pin(async move {
						assert_eq!(current, Some(&api_credential("first")));
						Ok(Some(api_credential("second")))
					})
				}),
				None,
			)
			.await
			.unwrap();
		assert_eq!(second, Some(api_credential("second")));
		assert_eq!(first.await.unwrap().unwrap(), Some(api_credential("first")));
	}

	#[tokio::test]
	async fn aborted_signal_fails_operations() {
		let store = InMemoryCredentialStore::new();
		let controller = AbortController::new();
		controller.abort();
		let options = AuthOperationOptions { signal: Some(controller.signal()) };
		assert!(store.read("p", Some(options.clone())).await.is_err());
		assert!(store.list(Some(options.clone())).await.is_err());
		assert!(
			store
				.modify("p", Box::new(|_| Box::pin(async { Ok(None) })), Some(options.clone()))
				.await
			.is_err()
		);
		assert!(store.delete("p", Some(options)).await.is_err());
		assert_eq!(store.read("p", None).await.unwrap(), None);
	}

	#[tokio::test]
	async fn aborted_signal_cancels_lock_wait() {
		let store = Arc::new(InMemoryCredentialStore::new());
		// Hold the per-provider chain with a slow modify.
		let holder = {
			let store = store.clone();
			tokio::spawn(async move {
				store
					.modify(
						"p",
						Box::new(|_| {
							Box::pin(async {
								tokio::time::sleep(std::time::Duration::from_millis(60)).await;
								Ok(Some(api_credential("held")))
							})
						}),
						None,
					)
					.await
			})
		};
		tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		let controller = AbortController::new();
		let waiter = {
			let store = store.clone();
			let signal = controller.signal();
			tokio::spawn(async move {
				store
					.modify("p", Box::new(|_| Box::pin(async { Ok(None) })), Some(AuthOperationOptions { signal: Some(signal) }))
					.await
			})
		};
		tokio::time::sleep(std::time::Duration::from_millis(10)).await;
		controller.abort();
		assert!(waiter.await.unwrap().is_err());
		// The holder still completes and the entry is stored.
		assert_eq!(holder.await.unwrap().unwrap(), Some(api_credential("held")));
		assert_eq!(store.read("p", None).await.unwrap(), Some(api_credential("held")));
	}

	#[tokio::test]
	async fn modify_fn_error_propagates_without_storing() {
		let store = InMemoryCredentialStore::new();
		let error = store
			.modify(
				"p",
				Box::new(|_| Box::pin(async { Err(ThrownValue::message("boom")) })),
				None,
			)
			.await
			.unwrap_err();
		assert!(matches!(&error, ThrownValue::Text(message) if message == "boom"));
		assert_eq!(store.read("p", None).await.unwrap(), None);
	}
}
