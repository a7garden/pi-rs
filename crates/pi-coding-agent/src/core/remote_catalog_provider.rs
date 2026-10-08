//! Port of `packages/coding-agent/src/core/remote-catalog-provider.ts` — a
//! dynamic-catalog overlay for a static built-in provider: persisted pi.dev
//! catalog entries are restored into the provider's model lists and
//! revalidated against `/api/models/providers/<id>` on refresh.
//!
//! Deviation (**D61**):
//! - **D61a — `AbortSignal` during fetch** (upstream passes `context.signal`
//!   into `fetchWithRetry`) becomes
//!   [`pi_ai::utils::abort::race_with_abort_signal`]: an abort during the
//!   request surfaces as an `Err` from [`Provider::refresh_models`], matching
//!   upstream's rejected promise.
//! - **D61b — catalog entry parsing is strict.** Upstream spreads JSON entries
//!   loosely (`{...model, provider}`); Rust deserializes each entry into
//!   [`AnyModel`] and drops entries that fail to parse. The unknown-type
//!   filter (`REMOTE_CATALOG_MODEL_TYPES`) runs first, so entries from newer
//!   clients are skipped, not fatal.
//! - **D61c — `Last-Modified` parsing.** JS `Date.parse` accepts many formats;
//!   the port only parses IMF-fixdate (`Tue, 15 Nov 1994 12:45:26 GMT`) and
//!   falls back to `0` otherwise (upstream's `Number.isNaN` fallback).
//! - **D61d — `getBuiltinModelDataGeneratedAt`** has no pi-ai equivalent: the
//!   embedded generated catalog (`pi_ai::models_generated`) carries no
//!   `generatedAt` manifest field, so callers (model-runtime, when ported)
//!   pass `None` until pi-ai grows the accessor.
//! - **D61e — response headers.** The crate's [`Fetcher`] seam previously
//!   returned only `status` + `body`; `HttpResponse` gained a `headers` list
//!   so `etag` / `last-modified` revalidation ports 1:1.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use pi_ai::models::{ModelsPublication, ModelsStoreWrite, Provider, RefreshModelsContext};
use pi_ai::models_store::ModelsStoreEntry;
use pi_ai::types::{AnyModel, ClassifierModel, Model, ModelType};
use pi_ai::utils::abort::race_with_abort_signal;
use pi_ai::utils::thrown::ThrownValue;
use serde_json::Value as JsonValue;

use crate::config::VERSION;
use crate::utils::management_http::{fetch_with_retry, FetchRetryOptions, Fetcher, ReqwestFetcher};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::time::now_ms;

const DEFAULT_CATALOG_BASE_URL: &str = "https://pi.dev";
const REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS: u64 = 4_000;
pub const REMOTE_CATALOG_REFRESH_INTERVAL_MS: u64 = 4 * 60 * 60 * 1000;

/// Model types this client can consume (`REMOTE_CATALOG_MODEL_TYPES`). Sent as
/// `?types=` so the catalog server returns the full-type shard instead of the
/// chat-only one served to clients that predate model types. A server that
/// ignores the parameter still returns the chat-only shard, which this client
/// handles unchanged.
pub const REMOTE_CATALOG_MODEL_TYPES: [ModelType; 3] = [ModelType::Chat, ModelType::Image, ModelType::Classifier];

fn model_type_slug(model_type: ModelType) -> &'static str {
	match model_type {
		ModelType::Chat => "chat",
		ModelType::Image => "image",
		ModelType::Classifier => "classifier",
	}
}

fn is_supported_model_type(entry: &JsonValue) -> bool {
	match entry.get("type") {
		None => true,
		Some(value) => matches!(value.as_str(), Some("chat") | Some("image") | Some("classifier")),
	}
}

/// Merge baseline and dynamic models by `(type, id)`; dynamic wins
/// (`mergeModels`).
fn merge_models(baseline: Vec<AnyModel>, dynamic: Vec<AnyModel>) -> Vec<AnyModel> {
	let mut merged = IndexMap::new();
	for model in baseline.into_iter().chain(dynamic) {
		merged.insert(format!("{}\0{}", model_type_slug(model.model_type()), model.id()), model);
	}
	merged.into_values().collect()
}

/// Parse a fetched catalog payload (`parseCatalog`): an array of models, a
/// `{models: [...]}` envelope, or a map of models. Entries without `id`, of
/// unsupported types, or unparseable as models are dropped (D61b).
fn parse_catalog(provider_id: &str, value: &JsonValue) -> Result<Vec<AnyModel>, ThrownValue> {
	let entries: Vec<JsonValue> = if let Some(list) = value.as_array() {
		list.clone()
	} else if let Some(envelope) = value.as_object().and_then(|object| object.get("models")) {
		match envelope.as_array() {
			Some(list) => list.clone(),
			None => return Err(invalid_catalog(provider_id)),
		}
	} else if let Some(map) = value.as_object() {
		map.values().cloned().collect()
	} else {
		return Err(invalid_catalog(provider_id));
	};
	Ok(entries
		.into_iter()
		.filter(|entry| entry.is_object() && entry.get("id").is_some())
		.filter(is_supported_model_type)
		.filter_map(|mut model| {
			if let Some(object) = model.as_object_mut() {
				object.insert("provider".to_string(), JsonValue::String(provider_id.to_string()));
			}
			AnyModel::from_json(&model).ok()
		})
		.collect())
}

fn invalid_catalog(provider_id: &str) -> ThrownValue {
	ThrownValue::message(format!("Invalid model catalog for provider \"{provider_id}\""))
}

/// Stored-overlay gating (`remoteModels`): empty without an entry or when the
/// stored snapshot is not newer than the local generated catalog.
fn remote_models(entry: Option<&ModelsStoreEntry>, local_generated_at: Option<u64>) -> Vec<AnyModel> {
	let Some(entry) = entry else { return Vec::new() };
	if let Some(local) = local_generated_at {
		if entry.last_modified.is_none_or(|last_modified| (last_modified as u64) <= local) {
			return Vec::new();
		}
	}
	entry.models.clone()
}

/// Add a persisted pi.dev catalog overlay to a static built-in provider
/// (`withRemoteCatalog`).
pub fn with_remote_catalog(
	provider: Arc<dyn Provider>,
	catalog_base_url: Option<&str>,
	generated_at: Option<u64>,
) -> Arc<dyn Provider> {
	with_remote_catalog_fetcher(provider, catalog_base_url, generated_at, Arc::new(ReqwestFetcher::default()))
}

/// Injection seam for tests: the same wrapper with an explicit HTTP fetcher.
pub(crate) fn with_remote_catalog_fetcher(
	provider: Arc<dyn Provider>,
	catalog_base_url: Option<&str>,
	generated_at: Option<u64>,
	fetcher: Arc<dyn Fetcher>,
) -> Arc<dyn Provider> {
	Arc::new(RemoteCatalogProvider {
		inner: provider,
		catalog_base_url: catalog_base_url
			.unwrap_or(DEFAULT_CATALOG_BASE_URL)
			.trim_end_matches('/')
			.to_string(),
		local_generated_at: generated_at,
		dynamic: Arc::new(Mutex::new(Vec::new())),
		fetcher,
	})
}

type DynamicModels = Arc<Mutex<Vec<AnyModel>>>;

struct RemoteCatalogProvider {
	inner: Arc<dyn Provider>,
	catalog_base_url: String,
	local_generated_at: Option<u64>,
	dynamic: DynamicModels,
	fetcher: Arc<dyn Fetcher>,
}

impl RemoteCatalogProvider {
	fn current_models(&self) -> Vec<AnyModel> {
		self.dynamic.lock().clone()
	}

	async fn refresh_impl(&self, context: RefreshModelsContext) -> Result<(), ThrownValue> {
		let provider_id = self.inner.id().to_string();
		let stored = context.stored.clone();
		let restored: Vec<AnyModel> = remote_models(stored.as_ref(), self.local_generated_at)
			.into_iter()
			.filter(|model| model.provider() == provider_id)
			.collect();

		// Publish the restored overlay first; a superseded publication ends the
		// refresh (upstream returns when `context.publish` resolves false).
		let dynamic = self.dynamic.clone();
		let published = context
			.publish
			.publish(ModelsPublication {
				persist: None,
				update: Some(Box::new(move || *dynamic.lock() = restored)),
			})
			.await;
		if !published {
			return Ok(());
		}
		if !context.allow_network || context.signal.aborted() {
			return Ok(());
		}
		let force = context.force.unwrap_or(false);
		if !force {
			let fresh = matches!(
				stored.as_ref(),
				Some(entry)
					if entry.checked_at.is_some()
						&& entry.last_modified.is_some()
						&& (now_ms().saturating_sub(entry.checked_at.unwrap()) as u64)
							< REMOTE_CATALOG_REFRESH_INTERVAL_MS
			);
			if fresh {
				return Ok(());
			}
		}

		// Only revalidate when a cached body backs the validator, so a 304 can
		// never leave the overlay empty.
		let validator =
			stored.as_ref().filter(|entry| !entry.models.is_empty()).and_then(|entry| entry.etag.clone());
		let url = format!(
			"{}/api/models/providers/{}?types={}",
			self.catalog_base_url,
			percent_encode(&provider_id),
			REMOTE_CATALOG_MODEL_TYPES.iter().map(|t| model_type_slug(*t)).collect::<Vec<_>>().join(",")
		);
		let mut headers: Vec<(String, String)> = vec![
			("accept".to_string(), "application/json".to_string()),
			("User-Agent".to_string(), get_pi_user_agent(VERSION)),
		];
		if let Some(etag) = &validator {
			headers.push(("if-none-match".to_string(), etag.clone()));
		}
		let response = race_with_abort_signal(
			fetch_with_retry(
				self.fetcher.clone(),
				&url,
				&headers,
				&FetchRetryOptions {
					attempt_timeout_ms: Some(REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS),
					..Default::default()
				},
			),
			context.signal.clone(),
		)
		.await
		.map_err(|reason| ThrownValue::message(reason.to_string()))?
		.map_err(ThrownValue::error)?;
		if context.signal.aborted() {
			return Ok(());
		}
		let checked_at = now_ms();
		// Unchanged: dynamicModels already holds the stored overlay, so only
		// the freshness window moves.
		if response.status == 304 {
			if let Some(mut entry) = stored {
				entry.checked_at = Some(checked_at);
				context
					.publish
					.publish(ModelsPublication { persist: Some(ModelsStoreWrite::Write(entry)), update: None })
					.await;
				return Ok(());
			}
			// Upstream falls through to the not-ok branch (which never fires
			// for 304), then tries to parse the empty body.
			return Err(fetch_failed(&provider_id, response.status));
		}
		if response.status == 404 || response.status == 501 {
			let mut entry = stored.unwrap_or_default();
			entry.checked_at = Some(checked_at);
			entry.last_modified = Some(0);
			entry.etag = None;
			context
				.publish
				.publish(ModelsPublication { persist: Some(ModelsStoreWrite::Write(entry)), update: None })
				.await;
			return Ok(());
		}
		if !response.ok() {
			// Transient failure: the cached body and its validator stay valid,
			// so keep the etag and let the next refresh revalidate instead of
			// downloading the catalog.
			let mut entry = stored.unwrap_or_default();
			entry.checked_at = Some(checked_at);
			context
				.publish
				.publish(ModelsPublication { persist: Some(ModelsStoreWrite::Write(entry)), update: None })
				.await;
			return Err(fetch_failed(&provider_id, response.status));
		}
		let payload: JsonValue = serde_json::from_slice(&response.body).map_err(ThrownValue::error)?;
		let refreshed = parse_catalog(&provider_id, &payload)?;
		let last_modified = response.header("last-modified").and_then(parse_http_date);
		if context.signal.aborted() {
			return Ok(());
		}
		let entry = ModelsStoreEntry {
			checked_at: Some(checked_at),
			last_modified: Some(last_modified.unwrap_or(0)),
			etag: response.header("etag").map(str::to_string),
			models: refreshed,
		};
		let published_models = remote_models(Some(&entry), self.local_generated_at);
		let dynamic = self.dynamic.clone();
		context
			.publish
			.publish(ModelsPublication {
				persist: Some(ModelsStoreWrite::Write(entry)),
				update: Some(Box::new(move || *dynamic.lock() = published_models)),
			})
			.await;
		Ok(())
	}
}

fn fetch_failed(provider_id: &str, status: u16) -> ThrownValue {
	ThrownValue::message(format!("Model catalog request failed for {provider_id}: {status}"))
}

/// Percent-encode every byte outside the RFC 3986 unreserved-and-marked set
/// (`encodeURIComponent` semantics for provider ids).
fn percent_encode(value: &str) -> String {
	let mut encoded = String::with_capacity(value.len());
	for byte in value.bytes() {
		if byte.is_ascii_alphanumeric()
			|| matches!(byte, b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')')
		{
			encoded.push(byte as char);
		} else {
			encoded.push_str(&format!("%{byte:02X}"));
		}
	}
	encoded
}

/// Parse an IMF-fixdate `Last-Modified` header to Unix milliseconds (D61c).
fn parse_http_date(value: &str) -> Option<i64> {
	let parts: Vec<&str> = value.split_whitespace().collect();
	if parts.len() < 5 || !parts[0].ends_with(',') {
		return None;
	}
	let day: i64 = parts[1].parse().ok()?;
	let month = match parts[2] {
		"Jan" => 1,
		"Feb" => 2,
		"Mar" => 3,
		"Apr" => 4,
		"May" => 5,
		"Jun" => 6,
		"Jul" => 7,
		"Aug" => 8,
		"Sep" => 9,
		"Oct" => 10,
		"Nov" => 11,
		"Dec" => 12,
		_ => return None,
	};
	let year: i64 = parts[3].parse().ok()?;
	let time: Vec<i64> =
		parts[4].split(':').map(|part| part.parse().ok()).collect::<Option<Vec<i64>>>()?;
	if time.len() != 3 {
		return None;
	}
	let (hour, minute, second) = (time[0], time[1], time[2]);
	// Days-from-civil (Howard Hinnant's algorithm), proleptic Gregorian UTC.
	let mut y = year;
	if month <= 2 {
		y -= 1;
	}
	let era = if y >= 0 { y } else { y - 399 } / 400;
	let yoe = y - era * 400;
	let mp = (month + 9) % 12;
	let doy = (153 * mp + 2) / 5 + day - 1;
	let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
	let days = era * 146_097 + doe - 719_468;
	Some((days * 86_400 + hour * 3_600 + minute * 60 + second) * 1_000)
}

impl Provider for RemoteCatalogProvider {
	fn id(&self) -> &str {
		self.inner.id()
	}

	fn name(&self) -> &str {
		self.inner.name()
	}

	fn base_url(&self) -> Option<&str> {
		self.inner.base_url()
	}

	fn headers(&self) -> Option<&pi_ai::types::ProviderHeaders> {
		self.inner.headers()
	}

	fn auth(&self) -> &pi_ai::auth::types::ProviderAuth {
		self.inner.auth()
	}

	fn get_models(&self) -> Vec<Model> {
		merge_models(
			self.inner.get_models().into_iter().map(AnyModel::Chat).collect(),
			self.current_models()
				.into_iter()
				.filter(|model| model.model_type() == ModelType::Chat)
				.collect(),
		)
		.into_iter()
		.map(|model| match model {
			AnyModel::Chat(model) => model,
			_ => unreachable!("chat-only merge produced a non-chat model"),
		})
		.collect()
	}

	fn get_all_models(&self) -> Vec<AnyModel> {
		merge_models(self.inner.get_all_models(), self.current_models())
	}

	fn refresh_models(
		&self,
		context: RefreshModelsContext,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		Some(Box::pin(self.refresh_impl(context)))
	}

	fn filter_models(
		&self,
		models: &[Model],
		credential: Option<&pi_ai::auth::types::Credential>,
	) -> Option<Vec<Model>> {
		self.inner.filter_models(models, credential)
	}

	fn filter_all_models(
		&self,
		models: &[AnyModel],
		credential: Option<&pi_ai::auth::types::Credential>,
	) -> Option<Vec<AnyModel>> {
		self.inner.filter_all_models(models, credential)
	}

	fn stream(
		&self,
		model: &Model,
		context: &pi_ai::types::TranscriptContext,
		options: &pi_ai::types::StreamOptions,
	) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
		self.inner.stream(model, context, options)
	}

	fn stream_simple(
		&self,
		model: &Model,
		context: &pi_ai::types::TranscriptContext,
		options: &pi_ai::types::SimpleStreamOptions,
	) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
		self.inner.stream_simple(model, context, options)
	}

	fn fetch_deferred(
		&self,
		model: &Model,
		handle: &pi_ai::types::DeferredHandle,
		options: &pi_ai::types::DeferredFetchOptions,
	) -> Option<pi_ai::utils::event_stream::AssistantMessageEventStream> {
		self.inner.fetch_deferred(model, handle, options)
	}

	fn cancel_deferred(
		&self,
		model: &Model,
		handle: &pi_ai::types::DeferredHandle,
		options: &pi_ai::types::DeferredCancelOptions,
	) -> Option<Pin<Box<dyn Future<Output = Result<(), ThrownValue>> + Send + '_>>> {
		self.inner.cancel_deferred(model, handle, options)
	}

	fn generate_images<'a>(
		&'a self,
		model: &'a pi_ai::types::ImageModel,
		context: &'a pi_ai::types::ImagesContext,
		options: &'a pi_ai::types::ImagesOptions,
	) -> Option<Pin<Box<dyn Future<Output = pi_ai::types::AssistantImages> + Send + 'a>>> {
		self.inner.generate_images(model, context, options)
	}

	fn classify<'a>(
		&'a self,
		model: &'a ClassifierModel,
		context: &'a pi_ai::types::ClassifierContext,
		options: &'a pi_ai::types::ClassifierOptions,
	) -> Option<Pin<Box<dyn Future<Output = pi_ai::types::ClassifierResult> + Send + 'a>>> {
		self.inner.classify(model, context, options)
	}
}

#[cfg(test)]
#[cfg(test)]
#[path = "remote_catalog_provider_tests.rs"]
mod tests;
