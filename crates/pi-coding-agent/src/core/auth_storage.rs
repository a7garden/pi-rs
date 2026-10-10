//! Port of `packages/coding-agent/src/core/auth-storage.ts` — the
//! `CredentialStore` implementation backed by `auth.json`.
//!
//! Surface: [`AuthStorageBackend`] (upstream `AuthStorageBackend` interface),
//! [`FileAuthStorageBackend`], [`InMemoryAuthStorageBackend`],
//! [`ReadOnlyAuthStorage`], [`AuthStorage`] (upstream `create` /
//! `fromStorage` / `inMemory` statics) and [`read_stored_credential`].
//! `resolveConfigValue`/`isCommandConfigValue` reuse
//! [`crate::core::resolve_config_value`]; `stripBom` lives in
//! [`crate::utils::text`]; `getAgentDir`/`getAuthPath` in
//! [`crate::config`]; `getFileRevision` is
//! [`crate::utils::paths::get_file_revision`].
//!
//! # Deviations
//!
//! - **D70 — file locking.** npm `proper-lockfile` becomes std's advisory
//!   file locks ([`std::fs::File::try_lock`] — `flock`/`LockFileEx` held on
//!   `auth.json` itself), which the kernel releases on process death, so the
//!   30 s staleness tracking and `onCompromised` machinery have no
//!   counterpart here. The sync acquisition keeps the upstream retry budget
//!   (10 attempts, 20 ms apart — sleeping instead of busy-waiting); async
//!   acquisition keeps exponential backoff with random jitter up to a 30 s
//!   deadline, then fails with a timeout message where upstream rethrows the
//!   underlying `ELOCKED` error. `mkdirSync(..., { mode: 0o700 })` becomes
//!   [`std::fs::create_dir_all`] (umask defaults); `mode: 0o600` still
//!   applies to `auth.json` on creation only (unix).
//! - **D70a — backend callbacks are typed, not erased.** `withLock`'s
//!   `current` argument is `Option<String>` (owned) and the async lock
//!   closure is fallible (`Result<LockResult<T>, ThrownValue>`) so parse
//!   errors reject like JS throws. The trait is deliberately not
//!   object-safe (generic methods, async-fn-in-trait); consumers hold
//!   concrete backends (`models-store` keeps an
//!   `Arc<FileAuthStorageBackend>`) and [`AuthStorage`] dispatches an
//!   internal enum. Upstream `AuthStorage.fromStorage` narrows to
//!   [`AuthStorage::from_file_backend`] (no shared file read state, like
//!   upstream's `authPath = undefined`).
//! - **D70b — coalesced reloads run to completion.** Upstream tracks
//!   `readers` and aborts the reload controller when the last reader
//!   leaves. The port spawns the shared reload as a tokio task whose result
//!   lands in a watch slot; waiters race the slot with their own signal.
//!   A reload abandoned by all readers finishes in the background instead
//!   of being cancelled (crate-wide D9 convention).
//! - **D70c — `parseStorageData` validates.** Upstream's unchecked TS cast
//!   becomes strict `Credential` deserialization; `auth.json` entries that
//!   are not valid credentials error (with the upstream read-only wording
//!   in [`ReadOnlyAuthStorage`], serde wording in [`AuthStorage`]) instead
//!   of passing through as inert objects.
//! - **D70d — in-memory promise chain.** The in-memory backend serializes
//!   async operations through a tokio mutex raced only on the lock wait
//!   (pi-ai D17 convention) instead of racing the whole operation with the
//!   signal.
//! - **D70e — sync `reload()` stays sync.** Upstream `reload()` is sync
//!   (proper-lockfile `lockSync`); the port keeps that with std locks, so
//!   it must not be called from an async context that cannot tolerate the
//!   10 × 20 ms retry window.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use parking_lot::Mutex;

use pi_ai::auth::types::{
    ApiKeyCredential, AuthOperationOptions, BoxFuture, Credential, CredentialInfo,
    CredentialModifyFn, CredentialStore, OAuthCredential, OAuthCredentials,
};
use pi_ai::types::ProviderEnv;
use pi_ai::utils::abort::{operation_signal, race_with_abort_signal, AbortSignal};
use pi_ai::utils::thrown::ThrownValue;

use crate::config::get_auth_path;
use crate::core::resolve_config_value::{is_command_config_value, resolve_config_value};
use crate::utils::paths::{get_file_revision, normalize_path, PathInputOptions};
use crate::utils::text::strip_bom;

/// Upstream `AuthStorageData` — provider id → credential. [`IndexMap`]
/// preserves JSON key insertion order (JS object semantics).
pub type AuthStorageData = IndexMap<String, Credential>;

/// Upstream `LockResult<T>` — the callback's value plus the next file
/// content to persist (`None` leaves the stored content untouched).
#[derive(Debug, Clone, PartialEq)]
pub struct LockResult<T> {
    pub result: T,
    pub next: Option<String>,
}

/// Upstream `AuthOperationOptions.signal` → `throwIfAborted`.
fn ensure_not_aborted(signal: Option<&AbortSignal>) -> Result<(), ThrownValue> {
    match signal {
        Some(signal) => signal.check().map_err(ThrownValue::error),
        None => Ok(()),
    }
}

/// `Math.random()` stand-in via OS randomness (getrandom; utils D13).
fn random_unit() -> f64 {
    let mut bytes = [0u8; 8];
    if getrandom::getrandom(&mut bytes).is_err() {
        return 0.5;
    }
    f64::from_le_bytes(bytes) / u64::MAX as f64
}

/// `JSON.stringify(data, null, 2)` — credential serialization is infallible.
fn json_pretty(data: &AuthStorageData) -> String {
    serde_json::to_string_pretty(data).expect("credential serialization is infallible")
}

/// Upstream `AuthStorage.parseStorageData` — `undefined`/empty content is an
/// empty store; otherwise strict JSON (D70c: unchecked TS cast becomes
/// deserialization).
fn parse_storage_data(content: Option<&str>) -> Result<AuthStorageData, ThrownValue> {
    let Some(content) = content else {
        return Ok(AuthStorageData::new());
    };
    if content.is_empty() {
        return Ok(AuthStorageData::new());
    }
    serde_json::from_str::<AuthStorageData>(strip_bom(content))
        .map_err(|error| ThrownValue::message(error.to_string()))
}

/// Upstream `AuthStorageBackend` interface. Not object-safe (D70a); callers
/// hold concrete backends. Async lock callbacks must own their captures
/// (`'static`) because the in-lock future is boxed.
pub trait AuthStorageBackend: Send + Sync {
    /// Upstream `withLock`: run `f` with the current stored content under an
    /// exclusive lock, persisting `next` when given.
    fn with_lock<T, F>(&self, f: F) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> LockResult<T> + Send,
        T: Send;

    /// Upstream `withLockAsync`: async lock acquisition (retry/backoff) with
    /// a fallible closure (D70a) and cancellation via `options.signal`.
    async fn with_lock_async<T, F>(
        &self,
        f: F,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> Result<LockResult<T>, ThrownValue> + Send + 'static,
        T: Send;
}

/// A future-returning lock callback (AuthStorage's in-lock awaits; D70a).
pub type AsyncLockCallback<T> = Box<
    dyn FnOnce(Option<String>) -> BoxFuture<'static, Result<LockResult<T>, ThrownValue>> + Send,
>;

/// Advisory lock on the open `auth.json` handle; the kernel releases it on
/// drop/process exit (D70).
struct FileLockGuard(File);

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn read_locked_content(file: &mut File) -> std::io::Result<String> {
    file.seek(SeekFrom::Start(0))?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

fn write_locked_content(file: &mut File, next: &str) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    file.write_all(next.as_bytes())
}

/// Upstream `FileAuthStorageBackend` — advisory-locked JSON file (D70).
pub struct FileAuthStorageBackend {
    auth_path: String,
}

impl FileAuthStorageBackend {
    pub fn new(auth_path: &str) -> FileAuthStorageBackend {
        FileAuthStorageBackend {
            auth_path: normalize_path(auth_path, &PathInputOptions::default()),
        }
    }

    /// The normalized `auth.json` path this backend locks and persists.
    pub fn auth_path(&self) -> &str {
        &self.auth_path
    }

    fn ensure_parent_dir(&self) -> std::io::Result<()> {
        if let Some(parent) = Path::new(&self.auth_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(())
    }

    fn ensure_file_exists(&self) -> std::io::Result<()> {
        if Path::new(&self.auth_path).exists() {
            return Ok(());
        }
        // The mode applies only on creation so administrator-managed modes
        // and ACLs remain intact (upstream `writeFileSync(..., 0o600)`).
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&self.auth_path) {
            Ok(mut file) => file.write_all(b"{}"),
            // Lost a creation race; the winner wrote `{}`.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// Upstream `acquireLockSyncWithRetry`: 10 attempts, 20 ms apart
    /// (sleeping — D70); non-lock errors throw immediately.
    fn acquire_lock_sync_with_retry(&self) -> std::io::Result<FileLockGuard> {
        const MAX_ATTEMPTS: usize = 10;
        const DELAY_MS: u64 = 20;
        let mut last_error: Option<std::io::Error> = None;
        for attempt in 1..=MAX_ATTEMPTS {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.auth_path)?;
            match file.try_lock() {
                Ok(()) => return Ok(FileLockGuard(file)),
                Err(std::fs::TryLockError::WouldBlock) => {
                    last_error = Some(std::io::Error::from(std::io::ErrorKind::WouldBlock));
                }
                Err(std::fs::TryLockError::Error(error)) => return Err(error),
            }
            if attempt < MAX_ATTEMPTS {
                std::thread::sleep(Duration::from_millis(DELAY_MS));
            }
        }
        Err(last_error
            .unwrap_or_else(|| std::io::Error::other("Failed to acquire auth storage lock")))
    }

    /// Upstream `acquireLockAsync`: exponential backoff with jitter up to a
    /// 30 s deadline (the upstream staleness bound — advisory locks cannot
    /// go stale, D70).
    async fn acquire_lock_async(
        &self,
        signal: Option<&AbortSignal>,
    ) -> Result<FileLockGuard, ThrownValue> {
        const LOCK_DEADLINE_MS: u64 = 30_000;
        const MAX_DELAY_MS: u64 = 2_000;
        let deadline = Instant::now() + Duration::from_millis(LOCK_DEADLINE_MS);
        let mut retry = 0u32;
        loop {
            ensure_not_aborted(signal)?;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.auth_path)
                .map_err(ThrownValue::error)?;
            match file.try_lock() {
                Ok(()) => {
                    // Upstream releases and rethrows when the signal fired
                    // during acquisition.
                    if let Err(reason) = ensure_not_aborted(signal) {
                        drop(FileLockGuard(file));
                        return Err(reason);
                    }
                    return Ok(FileLockGuard(file));
                }
                Err(std::fs::TryLockError::Error(error)) => return Err(ThrownValue::error(error)),
                Err(std::fs::TryLockError::WouldBlock) => {
                    drop(file);
                    ensure_not_aborted(signal)?;
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        return Err(ThrownValue::message(
                            "Could not lock auth.json: lock acquisition timed out",
                        ));
                    };
                    // base = min(10 * 2^retry, 1000); delay = min(round(base *
                    // (1 + random)), remaining).
                    let base_delay_ms = 10u64
                        .saturating_mul(1u64 << retry.min(20))
                        .min(MAX_DELAY_MS / 2);
                    retry += 1;
                    let delay_ms = ((base_delay_ms as f64 * (1.0 + random_unit())).round() as u64)
                        .min(remaining.as_millis() as u64)
                        .max(1);
                    race_with_abort_signal(
                        tokio::time::sleep(Duration::from_millis(delay_ms)),
                        operation_signal(signal),
                    )
                    .await
                    .map_err(ThrownValue::error)?;
                }
            }
        }
    }

    /// Async core shared by [`Self::with_lock_async`] and AuthStorage's
    /// in-lock awaits (D70a: future-returning callbacks).
    pub async fn with_lock_async_future<T>(
        &self,
        f: AsyncLockCallback<T>,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        T: Send,
    {
        let signal = options.as_ref().and_then(|options| options.signal.as_ref());
        ensure_not_aborted(signal)?;
        self.ensure_parent_dir().map_err(ThrownValue::error)?;
        self.ensure_file_exists().map_err(ThrownValue::error)?;
        let mut guard = self.acquire_lock_async(signal).await?;
        ensure_not_aborted(signal)?;
        let current = read_locked_content(&mut guard.0).map_err(ThrownValue::error)?;
        let outcome = f(Some(current)).await?;
        ensure_not_aborted(signal)?;
        if let Some(next) = outcome.next {
            write_locked_content(&mut guard.0, &next).map_err(ThrownValue::error)?;
        }
        ensure_not_aborted(signal)?;
        Ok(outcome.result)
    }
}

impl FileAuthStorageBackend {
    /// Sync variant consumers use directly (models-store D60, tests).
    pub fn with_lock<T, F>(&self, f: F) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> LockResult<T> + Send,
        T: Send,
    {
        self.ensure_parent_dir().map_err(ThrownValue::error)?;
        self.ensure_file_exists().map_err(ThrownValue::error)?;
        let mut guard = self
            .acquire_lock_sync_with_retry()
            .map_err(ThrownValue::error)?;
        let current = read_locked_content(&mut guard.0).map_err(ThrownValue::error)?;
        let outcome = f(Some(current));
        if let Some(next) = outcome.next {
            write_locked_content(&mut guard.0, &next).map_err(ThrownValue::error)?;
        }
        Ok(outcome.result)
    }

    /// Fallible sync closure under the async lock (models-store D60 shape).
    pub async fn with_lock_async<T, F>(
        &self,
        f: F,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> Result<LockResult<T>, ThrownValue> + Send + 'static,
        T: Send,
    {
        self.with_lock_async_future(
            Box::new(move |content| Box::pin(async move { f(content) })),
            options,
        )
        .await
    }
}

impl AuthStorageBackend for FileAuthStorageBackend {
    fn with_lock<T, F>(&self, f: F) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> LockResult<T> + Send,
        T: Send,
    {
        FileAuthStorageBackend::with_lock(self, f)
    }

    async fn with_lock_async<T, F>(
        &self,
        f: F,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> Result<LockResult<T>, ThrownValue> + Send + 'static,
        T: Send,
    {
        FileAuthStorageBackend::with_lock_async(self, f, options).await
    }
}

/// Upstream `InMemoryAuthStorageBackend` — an optional JSON string guarded
/// by a serialization chain (D70d).
pub struct InMemoryAuthStorageBackend {
    value: Mutex<Option<String>>,
    chain: tokio::sync::Mutex<()>,
}

impl InMemoryAuthStorageBackend {
    pub fn new() -> InMemoryAuthStorageBackend {
        InMemoryAuthStorageBackend {
            value: Mutex::new(None),
            chain: tokio::sync::Mutex::new(()),
        }
    }

    pub fn with_lock<T, F>(&self, f: F) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> LockResult<T> + Send,
        T: Send,
    {
        let current = self.value.lock().clone();
        let outcome = f(current);
        if let Some(next) = outcome.next {
            *self.value.lock() = Some(next);
        }
        Ok(outcome.result)
    }

    /// Future-returning callback core (D70a).
    pub async fn with_lock_async_future<T>(
        &self,
        f: AsyncLockCallback<T>,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        T: Send,
    {
        let signal = operation_signal(options.as_ref().and_then(|options| options.signal.as_ref()));
        // Upstream awaits the previous task through `.catch(() => {})`; the
        // chain lock releases regardless of outcome. Only the wait races the
        // signal (D70d).
        let _guard = race_with_abort_signal(self.chain.lock(), signal.clone())
            .await
            .map_err(ThrownValue::error)?;
        signal.check().map_err(ThrownValue::error)?;
        let current = self.value.lock().clone();
        let outcome = f(current).await?;
        signal.check().map_err(ThrownValue::error)?;
        if let Some(next) = outcome.next {
            *self.value.lock() = Some(next);
        }
        Ok(outcome.result)
    }

    pub async fn with_lock_async<T, F>(
        &self,
        f: F,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> Result<LockResult<T>, ThrownValue> + Send + 'static,
        T: Send,
    {
        self.with_lock_async_future(
            Box::new(move |content| Box::pin(async move { f(content) })),
            options,
        )
        .await
    }
}

impl Default for InMemoryAuthStorageBackend {
    fn default() -> InMemoryAuthStorageBackend {
        InMemoryAuthStorageBackend::new()
    }
}

impl AuthStorageBackend for InMemoryAuthStorageBackend {
    fn with_lock<T, F>(&self, f: F) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> LockResult<T> + Send,
        T: Send,
    {
        InMemoryAuthStorageBackend::with_lock(self, f)
    }

    async fn with_lock_async<T, F>(
        &self,
        f: F,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> Result<LockResult<T>, ThrownValue> + Send + 'static,
        T: Send,
    {
        InMemoryAuthStorageBackend::with_lock_async(self, f, options).await
    }
}

/// Upstream `ReadOnlyAuthStorage` — strict validating reader (D70c).
pub struct ReadOnlyAuthStorage {
    auth_path: String,
    data: Mutex<Option<AuthStorageData>>,
}

impl ReadOnlyAuthStorage {
    pub fn new(auth_path: &str) -> ReadOnlyAuthStorage {
        ReadOnlyAuthStorage {
            auth_path: normalize_path(auth_path, &PathInputOptions::default()),
            data: Mutex::new(None),
        }
    }

    /// Upstream `load()` — one strict parse, cached; ENOENT reads as empty.
    fn load(&self) -> Result<AuthStorageData, ThrownValue> {
        if let Some(data) = self.data.lock().as_ref() {
            return Ok(data.clone());
        }
        let parsed: serde_json::Value = match std::fs::read_to_string(&self.auth_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let data = AuthStorageData::new();
                *self.data.lock() = Some(data.clone());
                return Ok(data);
            }
            Err(error) => {
                return Err(ThrownValue::message(format!(
                    "Failed to read auth.json: {error}"
                )))
            }
            Ok(content) => match serde_json::from_str(strip_bom(&content)) {
                Ok(value) => value,
                Err(error) => {
                    return Err(ThrownValue::message(format!(
                        "Failed to read auth.json: {error}"
                    )))
                }
            },
        };
        let Some(object) = parsed.as_object() else {
            return Err(ThrownValue::message(
                "Invalid auth.json: expected an object",
            ));
        };
        let mut data = AuthStorageData::new();
        for (provider_id, value) in object {
            data.insert(
                provider_id.clone(),
                parse_stored_credential(provider_id, value)?,
            );
        }
        *self.data.lock() = Some(data.clone());
        Ok(data)
    }

    fn read_credential(&self, provider_id: &str) -> Result<Option<Credential>, ThrownValue> {
        let Some(credential) = self.load()?.get(provider_id).cloned() else {
            return Ok(None);
        };
        // Upstream: api_key credentials with a concrete (non-command) key get
        // their value resolved; everything else is returned as stored.
        if let Credential::ApiKey(api_key) = &credential {
            if let Some(key) = &api_key.key {
                if !key.is_empty() && !is_command_config_value(key) {
                    return Ok(Some(Credential::ApiKey(ApiKeyCredential {
                        key: resolve_config_value(key, api_key.env.as_ref()),
                        env: api_key.env.clone(),
                    })));
                }
            }
        }
        Ok(Some(credential))
    }
}

/// Upstream's per-credential shape validation in `ReadOnlyAuthStorage.load`
/// (`api_key`: optional string `key`, optional env map of strings; `oauth`:
/// string `access`/`refresh`, finite number `expires`; unknown fields kept).
fn parse_stored_credential(
    provider_id: &str,
    value: &serde_json::Value,
) -> Result<Credential, ThrownValue> {
    let invalid = || {
        ThrownValue::message(format!(
            "Invalid auth.json credential for provider \"{provider_id}\""
        ))
    };
    let Some(object) = value.as_object() else {
        return Err(invalid());
    };
    match object.get("type").and_then(serde_json::Value::as_str) {
        Some("api_key") => {
            let key = match object.get("key") {
                None => None,
                Some(value) if value.is_string() => {
                    Some(value.as_str().expect("checked string").to_string())
                }
                Some(_) => return Err(invalid()),
            };
            let env = match object.get("env") {
                None => None,
                Some(value) => {
                    let Some(entries) = value.as_object() else {
                        return Err(invalid());
                    };
                    let mut env = ProviderEnv::new();
                    for (name, entry) in entries {
                        let Some(text) = entry.as_str() else {
                            return Err(invalid());
                        };
                        env.insert(name.clone(), text.to_string());
                    }
                    Some(env)
                }
            };
            Ok(Credential::ApiKey(ApiKeyCredential { key, env }))
        }
        Some("oauth") => {
            let mut fields = (None, None, None);
            for (name, value) in object {
                match name.as_str() {
                    "access" => fields.0 = Some(value.as_str().ok_or_else(invalid)?.to_string()),
                    "refresh" => fields.1 = Some(value.as_str().ok_or_else(invalid)?.to_string()),
                    "expires" => {
                        let expires = value
                            .as_f64()
                            .filter(|expires| expires.is_finite())
                            .ok_or_else(invalid)?;
                        fields.2 = Some(expires as i64);
                    }
                    _ => {}
                }
            }
            let (Some(access), Some(refresh), Some(expires)) = fields else {
                return Err(invalid());
            };
            let mut extra = std::collections::BTreeMap::new();
            for (name, value) in object {
                if !matches!(name.as_str(), "type" | "access" | "refresh" | "expires") {
                    extra.insert(name.clone(), value.clone());
                }
            }
            Ok(Credential::OAuth(OAuthCredential {
                credentials: OAuthCredentials {
                    refresh,
                    access,
                    expires,
                    extra,
                },
            }))
        }
        _ => Err(invalid()),
    }
}

#[async_trait::async_trait]
impl CredentialStore for ReadOnlyAuthStorage {
    async fn read(
        &self,
        provider_id: &str,
        options: Option<AuthOperationOptions>,
    ) -> Result<Option<Credential>, ThrownValue> {
        let signal = options.as_ref().and_then(|options| options.signal.as_ref());
        ensure_not_aborted(signal)?;
        let credential = self.read_credential(provider_id)?;
        ensure_not_aborted(signal)?;
        Ok(credential)
    }

    async fn list(
        &self,
        options: Option<AuthOperationOptions>,
    ) -> Result<Vec<CredentialInfo>, ThrownValue> {
        let signal = options.as_ref().and_then(|options| options.signal.as_ref());
        ensure_not_aborted(signal)?;
        let credentials = self
            .load()?
            .into_iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id,
                credential_type: credential.type_name().to_string(),
            })
            .collect::<Vec<_>>();
        ensure_not_aborted(signal)?;
        Ok(credentials)
    }

    async fn modify(
        &self,
        _provider_id: &str,
        _modify_fn: CredentialModifyFn,
        _options: Option<AuthOperationOptions>,
    ) -> Result<Option<Credential>, ThrownValue> {
        Err(ThrownValue::message(
            "Read-only credential storage cannot modify auth.json",
        ))
    }

    async fn delete(
        &self,
        _provider_id: &str,
        _options: Option<AuthOperationOptions>,
    ) -> Result<(), ThrownValue> {
        Err(ThrownValue::message(
            "Read-only credential storage cannot modify auth.json",
        ))
    }
}

/// Concrete backends an [`AuthStorage`] can drive (D70a: the upstream
/// duck-typed backend parameter becomes this internal enum).
#[derive(Clone)]
enum StorageBackend {
    File(Arc<FileAuthStorageBackend>),
    InMemory(Arc<InMemoryAuthStorageBackend>),
}

impl StorageBackend {
    fn with_lock<T, F>(&self, f: F) -> Result<T, ThrownValue>
    where
        F: FnOnce(Option<String>) -> LockResult<T> + Send,
        T: Send,
    {
        match self {
            StorageBackend::File(backend) => backend.with_lock(f),
            StorageBackend::InMemory(backend) => backend.with_lock(f),
        }
    }

    async fn with_lock_async_future<T>(
        &self,
        f: AsyncLockCallback<T>,
        options: Option<AuthOperationOptions>,
    ) -> Result<T, ThrownValue>
    where
        T: Send,
    {
        match self {
            StorageBackend::File(backend) => backend.with_lock_async_future(f, options).await,
            StorageBackend::InMemory(backend) => backend.with_lock_async_future(f, options).await,
        }
    }
}

#[derive(Default)]
struct ReadStateCore {
    data: AuthStorageData,
    revision: Option<String>,
}

/// A coalesced reload: one spawned task whose result lands in a watch slot
/// (D70b).
struct ReloadHandle {
    receiver: tokio::sync::watch::Receiver<Option<Result<AuthStorageData, ThrownValue>>>,
}

impl ReloadHandle {
    fn settled(&self) -> bool {
        self.receiver.borrow().is_some()
    }

    async fn result(&self) -> Option<Result<AuthStorageData, ThrownValue>> {
        let mut receiver = self.receiver.clone();
        // Bind first: the watch guard borrows `receiver`, so the match must
        // not be the block's final expression.
        let settled = receiver.wait_for(|value| value.is_some()).await;
        match settled {
            Ok(value) => value.clone(),
            Err(_) => None,
        }
    }
}

#[derive(Default)]
struct SharedReadState {
    core: Mutex<ReadStateCore>,
    reload: Mutex<Option<Arc<ReloadHandle>>>,
}

/// Upstream module-level `sharedAuthFileReadState` — a single-entry cache
/// keyed by normalized path, set only while absent.
static SHARED_AUTH_FILE_READ_STATE: LazyLock<Mutex<Option<(String, Arc<SharedReadState>)>>> =
    LazyLock::new(|| Mutex::new(None));

/// The reload callback shared by the in-memory path and the coalesced file
/// reload: parse under the lock, refresh the read state, return the data.
fn reload_callback(
    state: Arc<SharedReadState>,
    auth_path: Option<String>,
) -> Box<
    dyn FnOnce(
            Option<String>,
        ) -> BoxFuture<'static, Result<LockResult<AuthStorageData>, ThrownValue>>
        + Send,
> {
    Box::new(move |content| {
        let state = state.clone();
        let auth_path = auth_path.clone();
        Box::pin(async move {
            let data = parse_storage_data(content.as_deref())?;
            let revision = auth_path.as_deref().and_then(get_file_revision);
            let mut core = state.core.lock();
            core.data = data.clone();
            core.revision = revision;
            Ok(LockResult {
                result: data,
                next: None,
            })
        })
    })
}

/// Credential storage backed by a JSON file (upstream `AuthStorage`).
pub struct AuthStorage {
    backend: StorageBackend,
    auth_path: Option<String>,
    state: Arc<SharedReadState>,
}

impl AuthStorage {
    /// Upstream `AuthStorage.create(authPath = join(getAgentDir(), "auth.json"))`.
    pub fn new(auth_path: Option<String>) -> AuthStorage {
        let auth_path = normalize_path(
            auth_path.unwrap_or_else(get_auth_path).as_str(),
            &PathInputOptions::default(),
        );
        AuthStorage::with_backend(
            StorageBackend::File(Arc::new(FileAuthStorageBackend::new(&auth_path))),
            Some(auth_path),
        )
    }

    /// Upstream `AuthStorage.fromStorage(new FileAuthStorageBackend(path))`
    /// — no shared file read state (upstream passes `authPath = undefined`),
    /// so reads always go through the backend lock (D70a).
    pub fn from_file_backend(backend: Arc<FileAuthStorageBackend>) -> AuthStorage {
        AuthStorage::with_backend(StorageBackend::File(backend), None)
    }

    /// Upstream `AuthStorage.inMemory(data = {})`.
    pub fn in_memory(data: AuthStorageData) -> AuthStorage {
        let backend = Arc::new(InMemoryAuthStorageBackend::new());
        let next = json_pretty(&data);
        let seeded = backend.with_lock(|_current| LockResult {
            result: (),
            next: Some(next),
        });
        debug_assert!(seeded.is_ok(), "in-memory seeding never fails");
        AuthStorage::with_backend(StorageBackend::InMemory(backend), None)
    }

    fn with_backend(backend: StorageBackend, auth_path: Option<String>) -> AuthStorage {
        let state = match &auth_path {
            Some(path) => {
                let mut shared = SHARED_AUTH_FILE_READ_STATE.lock();
                match shared.as_ref() {
                    Some((shared_path, state)) if shared_path == path => state.clone(),
                    _ => {
                        let state = Arc::new(SharedReadState::default());
                        if shared.is_none() {
                            *shared = Some((path.clone(), state.clone()));
                        }
                        state
                    }
                }
            }
            None => Arc::new(SharedReadState::default()),
        };
        let storage = AuthStorage {
            backend,
            auth_path,
            state,
        };
        if let Some(path) = &storage.auth_path {
            let revision = get_file_revision(path);
            if revision.is_some() && storage.state.core.lock().revision == revision {
                return storage;
            }
        }
        storage.reload();
        storage
    }

    fn update_read_state(&self, data: AuthStorageData, revision: Option<String>) {
        let mut core = self.state.core.lock();
        core.data = data;
        core.revision = revision;
    }

    /// Reload credentials from storage (upstream `reload()` — sync, D70e).
    /// Storage and parse failures preserve the last valid snapshot.
    pub fn reload(&self) {
        let slot: Mutex<Option<(Option<String>, Option<String>)>> = Mutex::new(None);
        let auth_path = self.auth_path.clone();
        let read = self.backend.with_lock(|current| {
            *slot.lock() = Some((current, auth_path.as_deref().and_then(get_file_revision)));
            LockResult {
                result: (),
                next: None,
            }
        });
        if read.is_ok() {
            if let Some((content, revision)) = slot.lock().take() {
                if let Ok(data) = parse_storage_data(content.as_deref()) {
                    self.update_read_state(data, revision);
                }
            }
        }
    }

    async fn reload_from_storage_async(
        &self,
        options: Option<AuthOperationOptions>,
    ) -> Result<AuthStorageData, ThrownValue> {
        self.backend
            .with_lock_async_future(
                reload_callback(self.state.clone(), self.auth_path.clone()),
                options,
            )
            .await
    }

    fn spawn_reload(&self) -> Arc<ReloadHandle> {
        let backend = self.backend.clone();
        let state = self.state.clone();
        let auth_path = self.auth_path.clone();
        let (sender, receiver) = tokio::sync::watch::channel(None);
        tokio::spawn(async move {
            let result = backend
                .with_lock_async_future(reload_callback(state, auth_path), None)
                .await;
            let _ = sender.send(Some(result));
        });
        Arc::new(ReloadHandle { receiver })
    }

    /// Upstream `readLatestData` — revision-validated cache with one
    /// coalesced reload in flight (D70b). Without a signal, reload failures
    /// fall back to the cached snapshot like upstream's `.catch`.
    async fn read_latest_data(
        &self,
        options: Option<&AuthOperationOptions>,
    ) -> Result<AuthStorageData, ThrownValue> {
        let signal = options.as_ref().and_then(|options| options.signal.as_ref());
        ensure_not_aborted(signal)?;
        if self.auth_path.is_none() {
            return match self.reload_from_storage_async(options.cloned()).await {
                Ok(data) => Ok(data),
                Err(error) if signal.is_some() => Err(error),
                Err(_) => Ok(self.state.core.lock().data.clone()),
            };
        }
        let auth_path = self
            .auth_path
            .as_deref()
            .expect("file-backed storage has an auth path");
        let revision = get_file_revision(auth_path);
        {
            let core = self.state.core.lock();
            if revision.is_some() && core.revision == revision {
                return Ok(core.data.clone());
            }
        }
        let handle = {
            let mut reload = self.state.reload.lock();
            if !reload.as_ref().is_some_and(|handle| !handle.settled()) {
                *reload = Some(self.spawn_reload());
            }
            reload.as_ref().expect("reload handle just created").clone()
        };
        match race_with_abort_signal(handle.result(), operation_signal(signal)).await {
            Ok(Some(Ok(data))) => Ok(data),
            Ok(Some(Err(error))) if signal.is_some() => Err(error),
            Ok(_) if signal.is_some() => Err(ThrownValue::message("Failed to reload auth.json")),
            Err(reason) => Err(ThrownValue::error(reason)),
            Ok(_) => Ok(self.state.core.lock().data.clone()),
        }
    }
}

#[async_trait::async_trait]
impl CredentialStore for AuthStorage {
    async fn read(
        &self,
        provider_id: &str,
        options: Option<AuthOperationOptions>,
    ) -> Result<Option<Credential>, ThrownValue> {
        let data = self.read_latest_data(options.as_ref()).await?;
        let credential = match data.get(provider_id) {
            Some(Credential::ApiKey(api_key)) => {
                // Upstream spreads the credential and resolves the key value
                // (command references included); `undefined` keys pass
                // through.
                match api_key.key.clone() {
                    Some(key) => Credential::ApiKey(ApiKeyCredential {
                        key: resolve_config_value(&key, api_key.env.as_ref()),
                        env: api_key.env.clone(),
                    }),
                    None => Credential::ApiKey(api_key.clone()),
                }
            }
            Some(credential) => credential.clone(),
            None => return Ok(None),
        };
        Ok(Some(credential))
    }

    async fn list(
        &self,
        options: Option<AuthOperationOptions>,
    ) -> Result<Vec<CredentialInfo>, ThrownValue> {
        let data = self.read_latest_data(options.as_ref()).await?;
        ensure_not_aborted(options.as_ref().and_then(|options| options.signal.as_ref()))?;
        Ok(data
            .into_iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id,
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
        let provider = provider_id.to_string();
        let auth_path = self.auth_path.clone();
        let latest: Arc<Mutex<Option<AuthStorageData>>> = Arc::new(Mutex::new(None));
        let revision: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let latest_callback = latest.clone();
        let revision_callback = revision.clone();
        let outcome = self
            .backend
            .with_lock_async_future(
                Box::new(move |content| {
                    let provider = provider.clone();
                    let auth_path = auth_path.clone();
                    let latest = latest_callback;
                    let revision = revision_callback;
                    Box::pin(async move {
                        let current_data = parse_storage_data(content.as_deref())?;
                        let current = current_data.get(&provider).cloned();
                        let next = modify_fn(current.as_ref()).await?;
                        Ok(match next {
                            None => {
                                // Upstream keeps the entry and reports the
                                // current credential; the revision is
                                // refreshed so caches revalidate.
                                *latest.lock() = Some(current_data);
                                *revision.lock() = auth_path.as_deref().and_then(get_file_revision);
                                LockResult {
                                    result: current,
                                    next: None,
                                }
                            }
                            Some(next) => {
                                let mut merged = current_data;
                                merged.insert(provider.clone(), next.clone());
                                *latest.lock() = Some(merged.clone());
                                LockResult {
                                    result: Some(next),
                                    next: Some(json_pretty(&merged)),
                                }
                            }
                        })
                    })
                }),
                options,
            )
            .await?;
        let latest_data = latest.lock().take().unwrap_or_default();
        let revision_value = revision.lock().take();
        self.update_read_state(latest_data, revision_value);
        Ok(outcome)
    }

    async fn delete(
        &self,
        provider_id: &str,
        options: Option<AuthOperationOptions>,
    ) -> Result<(), ThrownValue> {
        let provider = provider_id.to_string();
        let latest: Arc<Mutex<Option<AuthStorageData>>> = Arc::new(Mutex::new(None));
        let latest_callback = latest.clone();
        self.backend
            .with_lock_async_future(
                Box::new(move |content| {
                    let provider = provider.clone();
                    let latest = latest_callback;
                    Box::pin(async move {
                        let mut current_data = parse_storage_data(content.as_deref())?;
                        current_data.shift_remove(&provider);
                        *latest.lock() = Some(current_data.clone());
                        Ok(LockResult {
                            result: (),
                            next: Some(json_pretty(&current_data)),
                        })
                    })
                }),
                options,
            )
            .await?;
        let latest_data = latest.lock().take().unwrap_or_default();
        self.update_read_state(latest_data, None);
        Ok(())
    }
}

/// One-off synchronous read of a stored credential from an `auth.json` file,
/// without instantiating a store or resolving configured key values
/// (upstream `readStoredCredential`; `auth_path` defaults to
/// [`get_auth_path`]).
pub fn read_stored_credential(provider_id: &str, auth_path: Option<&str>) -> Option<Credential> {
    let path = match auth_path {
        Some(path) => path.to_string(),
        None => get_auth_path(),
    };
    let path = normalize_path(&path, &PathInputOptions::default());
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str::<AuthStorageData>(strip_bom(&content))
        .ok()?
        .get(provider_id)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    static DIR_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_dir(label: &str) -> String {
        let dir = std::env::temp_dir().join(format!(
            "pi-auth-storage-{label}-{}-{}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_string_lossy().into_owned()
    }

    fn auth_path(dir: &str) -> String {
        crate::utils::paths::join(dir, "auth.json")
    }

    fn api_credential(key: &str) -> Credential {
        Credential::ApiKey(ApiKeyCredential {
            key: Some(key.to_string()),
            env: None,
        })
    }

    fn oauth_credential(refresh: &str, access: &str, expires: i64) -> Credential {
        Credential::OAuth(OAuthCredential {
            credentials: OAuthCredentials {
                refresh: refresh.to_string(),
                access: access.to_string(),
                expires,
                extra: BTreeMap::new(),
            },
        })
    }

    async fn set_credential(storage: &AuthStorage, provider: &str, credential: Credential) {
        storage
            .modify(
                provider,
                Box::new(move |_current| {
                    let credential = credential.clone();
                    Box::pin(async move { Ok(Some(credential)) })
                }),
                None,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn modify_writes_pretty_json_and_reads_back() {
        let dir = temp_dir("round-trip");
        let path = auth_path(&dir);
        let storage = AuthStorage::new(Some(path.clone()));
        let modified = storage
            .modify(
                "anthropic",
                Box::new(|_| Box::pin(async { Ok(Some(api_credential("sk-test"))) })),
                None,
            )
            .await
            .unwrap();
        assert_eq!(modified, Some(api_credential("sk-test")));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\n  \"anthropic\": {\n    \"type\": \"api_key\",\n    \"key\": \"sk-test\"\n  }\n}"
        );
        assert_eq!(
            storage.read("anthropic", None).await.unwrap(),
            Some(api_credential("sk-test"))
        );
        assert_eq!(storage.read("missing", None).await.unwrap(), None);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "auth.json is created 0o600");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_resolves_env_templates_from_the_credential_env() {
        let dir = temp_dir("env-template");
        let storage = AuthStorage::new(Some(auth_path(&dir)));
        let env = ProviderEnv::from([(
            "PI_RS_AUTH_STORAGE_TEST_TOKEN".to_string(),
            "resolved-value".to_string(),
        )]);
        let stored_env = env.clone();
        storage
            .modify(
                "openai",
                Box::new(move |_| {
                    let env = stored_env.clone();
                    Box::pin(async move {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("$PI_RS_AUTH_STORAGE_TEST_TOKEN".to_string()),
                            env: Some(env),
                        })))
                    })
                }),
                None,
            )
            .await
            .unwrap();
        // Upstream `{...credential, key: resolveConfigValue(...)}` keeps the
        // credential's env map on the returned credential.
        assert_eq!(
            storage.read("openai", None).await.unwrap(),
            Some(Credential::ApiKey(ApiKeyCredential {
                key: Some("resolved-value".to_string()),
                env: Some(env)
            }))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_resolves_command_values_through_the_shell() {
        let dir = temp_dir("command-value");
        let storage = AuthStorage::new(Some(auth_path(&dir)));
        set_credential(
            &storage,
            "p",
            api_credential("!echo pi-rs-auth-command-value"),
        )
        .await;
        assert_eq!(
            storage.read("p", None).await.unwrap(),
            Some(api_credential("pi-rs-auth-command-value"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn modify_with_none_keeps_the_entry_and_returns_the_current_credential() {
        let dir = temp_dir("modify-undefined");
        let path = auth_path(&dir);
        let storage = AuthStorage::new(Some(path.clone()));
        set_credential(&storage, "p", api_credential("v1")).await;
        let before = std::fs::read_to_string(&path).unwrap();
        let result = storage
            .modify("p", Box::new(|_| Box::pin(async { Ok(None) })), None)
            .await
            .unwrap();
        assert_eq!(result, Some(api_credential("v1")));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            before,
            "file untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn delete_removes_the_entry_and_persists() {
        let dir = temp_dir("delete");
        let path = auth_path(&dir);
        let storage = AuthStorage::new(Some(path.clone()));
        set_credential(&storage, "keep", api_credential("k")).await;
        set_credential(&storage, "drop", oauth_credential("r", "a", 5)).await;
        storage.delete("drop", None).await.unwrap();
        assert_eq!(storage.read("drop", None).await.unwrap(), None);
        assert_eq!(
            storage.read("keep", None).await.unwrap(),
            Some(api_credential("k"))
        );
        let infos = storage.list(None).await.unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].provider_id, "keep");
        assert_eq!(infos[0].credential_type, "api_key");
        let stored: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(stored.get("drop").is_none());
        // Deleting an absent provider is a no-op that still rewrites valid JSON.
        storage.delete("absent", None).await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn list_reports_providers_in_insertion_order() {
        let dir = temp_dir("list-order");
        let storage = AuthStorage::new(Some(auth_path(&dir)));
        set_credential(&storage, "zeta", api_credential("z")).await;
        set_credential(&storage, "alpha", oauth_credential("r", "a", 1)).await;
        let infos = storage.list(None).await.unwrap();
        let ids: Vec<_> = infos.iter().map(|info| info.provider_id.as_str()).collect();
        assert_eq!(ids, ["zeta", "alpha"]);
        let types: Vec<_> = infos
            .iter()
            .map(|info| info.credential_type.as_str())
            .collect();
        assert_eq!(types, ["api_key", "oauth"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn shared_read_state_picks_up_external_writes_across_instances() {
        let dir = temp_dir("shared-state");
        let path = auth_path(&dir);
        let first = AuthStorage::new(Some(path.clone()));
        set_credential(&first, "p", api_credential("v1")).await;
        let second = AuthStorage::new(Some(path.clone()));
        assert_eq!(
            second.read("p", None).await.unwrap(),
            Some(api_credential("v1"))
        );
        let backend = FileAuthStorageBackend::new(&path);
        backend
            .with_lock(|_current| LockResult {
                result: (),
                next: Some("{\"p\": {\"type\": \"api_key\", \"key\": \"v2\"}}".to_string()),
            })
            .unwrap();
        assert_eq!(
            first.read("p", None).await.unwrap(),
            Some(api_credential("v2"))
        );
        assert_eq!(
            second.read("p", None).await.unwrap(),
            Some(api_credential("v2"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn reload_keeps_the_last_valid_snapshot_on_corrupt_content() {
        let dir = temp_dir("reload-corrupt");
        let path = auth_path(&dir);
        let storage = AuthStorage::new(Some(path.clone()));
        set_credential(&storage, "p", api_credential("v1")).await;
        std::fs::write(&path, "not json {").unwrap();
        storage.reload();
        assert_eq!(
            storage.read("p", None).await.unwrap(),
            Some(api_credential("v1"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn in_memory_storage_seeds_and_round_trips_without_a_file() {
        let storage = AuthStorage::in_memory(AuthStorageData::from([
            ("seeded".to_string(), api_credential("seed-key")),
            ("oauth".to_string(), oauth_credential("r", "a", 9)),
        ]));
        assert_eq!(
            storage.read("seeded", None).await.unwrap(),
            Some(api_credential("seed-key"))
        );
        set_credential(&storage, "added", api_credential("added-key")).await;
        assert_eq!(
            storage.read("added", None).await.unwrap(),
            Some(api_credential("added-key"))
        );
        assert_eq!(storage.list(None).await.unwrap().len(), 3);
        let _ = std::fs::remove_dir_all(temp_dir("in-memory-noop"));
    }

    #[test]
    fn parse_storage_data_handles_missing_empty_and_bom_content() {
        assert!(parse_storage_data(None).unwrap().is_empty());
        assert!(parse_storage_data(Some("")).unwrap().is_empty());
        let parsed = parse_storage_data(Some(
            "\u{feff}{\n  \"p\": {\"type\": \"api_key\", \"key\": \"k\"}\n}",
        ))
        .unwrap();
        assert_eq!(parsed.get("p"), Some(&api_credential("k")));
        assert!(parse_storage_data(Some("not json")).is_err());
        assert!(
            parse_storage_data(Some("[]")).is_err(),
            "top-level arrays are rejected (D70c)"
        );
    }

    #[tokio::test]
    async fn read_only_storage_reads_but_rejects_mutations() {
        let dir = temp_dir("read-only");
        let path = auth_path(&dir);
        std::fs::write(
            &path,
            "{\"p\": {\"type\": \"api_key\", \"key\": \"sk-1\"}, \"o\": {\"type\": \"oauth\", \
			 \"refresh\": \"r\", \"access\": \"a\", \"expires\": 123}}",
        )
        .unwrap();
        let storage = ReadOnlyAuthStorage::new(&path);
        assert_eq!(
            storage.read("p", None).await.unwrap(),
            Some(api_credential("sk-1"))
        );
        assert_eq!(
            storage.read("o", None).await.unwrap(),
            Some(oauth_credential("r", "a", 123))
        );
        assert_eq!(storage.read("missing", None).await.unwrap(), None);
        assert_eq!(storage.list(None).await.unwrap().len(), 2);
        let modify_error = storage
            .modify("p", Box::new(|_| Box::pin(async { Ok(None) })), None)
            .await
            .unwrap_err();
        assert!(
            matches!(modify_error, ThrownValue::Text(text) if text == "Read-only credential storage cannot modify auth.json")
        );
        let delete_error = storage.delete("p", None).await.unwrap_err();
        assert!(
            matches!(delete_error, ThrownValue::Text(text) if text == "Read-only credential storage cannot modify auth.json")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_only_storage_resolves_plain_keys_but_keeps_commands_raw() {
        let dir = temp_dir("read-only-resolve");
        let path = auth_path(&dir);
        std::fs::write(
            &path,
            "{\"plain\": {\"type\": \"api_key\", \"key\": \"$PI_RS_AUTH_RO_TEST\", \
			 \"env\": {\"PI_RS_AUTH_RO_TEST\": \"resolved\"}}, \
			 \"command\": {\"type\": \"api_key\", \"key\": \"!echo untouched\"}, \
			 \"empty\": {\"type\": \"api_key\", \"key\": \"\"}}",
        )
        .unwrap();
        let storage = ReadOnlyAuthStorage::new(&path);
        assert_eq!(
            storage.read("plain", None).await.unwrap(),
            Some(Credential::ApiKey(ApiKeyCredential {
                key: Some("resolved".to_string()),
                env: Some(ProviderEnv::from([(
                    "PI_RS_AUTH_RO_TEST".to_string(),
                    "resolved".to_string()
                )])),
            }))
        );
        assert_eq!(
            storage.read("command", None).await.unwrap(),
            Some(api_credential("!echo untouched"))
        );
        assert_eq!(
            storage.read("empty", None).await.unwrap(),
            Some(Credential::ApiKey(ApiKeyCredential {
                key: Some(String::new()),
                env: None
            }))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_only_storage_validates_credential_shapes() {
        let dir = temp_dir("read-only-invalid");
        let path = auth_path(&dir);
        // Missing file reads as empty.
        let missing = ReadOnlyAuthStorage::new(&path);
        assert_eq!(missing.read("p", None).await.unwrap(), None);
        // Top-level non-object.
        std::fs::write(&path, "[1, 2]").unwrap();
        let error = ReadOnlyAuthStorage::new(&path)
            .read("p", None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, ThrownValue::Text(text) if text == "Invalid auth.json: expected an object")
        );
        // Invalid credential entries carry the upstream wording.
        for content in [
			"{\"p\": null}",
			"{\"p\": {\"type\": \"unknown\"}}",
			"{\"p\": {\"type\": \"api_key\", \"key\": 5}}",
			"{\"p\": {\"type\": \"api_key\", \"env\": {\"a\": 1}}}",
			"{\"p\": {\"type\": \"oauth\", \"access\": \"a\"}}",
			"{\"p\": {\"type\": \"oauth\", \"access\": \"a\", \"refresh\": \"r\", \"expires\": \"soon\"}}",
		] {
			std::fs::write(&path, content).unwrap();
			let error = ReadOnlyAuthStorage::new(&path).read("p", None).await.unwrap_err();
			assert!(
				matches!(error, ThrownValue::Text(text) if text == "Invalid auth.json credential for provider \"p\""),
				"content {content} should be invalid"
			);
		}
        // Syntactic corruption wraps the upstream read error.
        std::fs::write(&path, "{").unwrap();
        let error = ReadOnlyAuthStorage::new(&path)
            .read("p", None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, ThrownValue::Text(text) if text.starts_with("Failed to read auth.json:"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_stored_credential_reads_one_off_without_a_store() {
        let dir = temp_dir("one-off");
        let path = auth_path(&dir);
        std::fs::write(
            &path,
            "{\"p\": {\"type\": \"api_key\", \"key\": \"$UNRESOLVED_VAR\"}}",
        )
        .unwrap();
        assert_eq!(
            read_stored_credential("p", Some(&path)),
            Some(api_credential("$UNRESOLVED_VAR"))
        );
        assert_eq!(read_stored_credential("missing", Some(&path)), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn file_backend_lock_round_trip_and_error_propagation() {
        let dir = temp_dir("backend-lock");
        let path = auth_path(&dir);
        let backend = FileAuthStorageBackend::new(&path);
        // The file is created on first use.
        let created = backend
            .with_lock(|current| LockResult {
                result: current,
                next: None,
            })
            .unwrap();
        assert_eq!(created, Some("{}".to_string()));
        // Sync write, then async read-modify-write with a fallible closure.
        backend
            .with_lock(|_current| LockResult {
                result: (),
                next: Some("{\"p\": {\"type\": \"api_key\", \"key\": \"k\"}}".to_string()),
            })
            .unwrap();
        let outcome = backend
            .with_lock_async(
                |current| {
                    let mut data = parse_storage_data(current.as_deref()).unwrap();
                    data.shift_remove("p");
                    Ok::<_, ThrownValue>(LockResult {
                        result: data.len(),
                        next: Some(json_pretty(&data)),
                    })
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(outcome, 0);
        // Closure errors reject like JS throws (D70a).
        let error = backend
            .with_lock_async(
                |_| Err::<LockResult<()>, _>(ThrownValue::message("boom")),
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, ThrownValue::Text(text) if text == "boom"));
        // Missing parent directories are created (upstream ensureParentDir).
        let nested = auth_path(&format!("{}/nested/deep", dir));
        let nested_backend = FileAuthStorageBackend::new(&nested);
        nested_backend
            .with_lock(|current| LockResult {
                result: current.is_some(),
                next: None,
            })
            .unwrap();
        assert!(Path::new(&nested).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
