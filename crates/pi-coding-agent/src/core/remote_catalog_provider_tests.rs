//! Tests for [`super::remote_catalog_provider`] — offline logic ported from
//! the upstream wrapper behavior (upstream has no dedicated test file; the
//! suites here mirror the documented `withRemoteCatalog` semantics).

use std::future::Future;
use std::sync::Arc;

use parking_lot::Mutex as ParkingMutex;
use pi_ai::models::{ModelsStoreWrite, PublishHandle, RefreshModelsContext};
use pi_ai::models_store::ModelsStoreEntry;
use pi_ai::providers::faux::{faux_provider, FauxModelDefinition, RegisterFauxProviderOptions};
use pi_ai::types::AnyModel;
use pi_ai::utils::abort::AbortSignal;
use serde_json::{json, Value as JsonValue};

use super::{merge_models, parse_catalog, parse_http_date, percent_encode, remote_models, with_remote_catalog_fetcher};
use crate::utils::management_http::{Fetcher, HttpResponse};

fn chat_model_json(id: &str) -> JsonValue {
	json!({
		"id": id, "name": id, "api": "anthropic-messages",
		"provider": "testprov", "baseUrl": "https://api.example.com",
		"input": ["text"], "reasoning": false,
		"contextWindow": 200000, "maxTokens": 64000,
		"cost": { "input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75 }
	})
}

fn chat_entry(id: &str) -> AnyModel {
	AnyModel::from_json(&chat_model_json(id)).expect("chat model parses")
}

type Persisted = Arc<ParkingMutex<Vec<ModelsStoreEntry>>>;

struct MockFetcher {
	urls: Arc<ParkingMutex<Vec<String>>>,
	queue: ParkingMutex<Vec<HttpResponse>>,
}

impl Fetcher for MockFetcher {
	fn fetch(
		&self,
		url: &str,
		_headers: &[(String, String)],
		_timeout_ms: Option<u64>,
	) -> std::pin::Pin<Box<dyn Future<Output = std::io::Result<HttpResponse>> + Send + '_>> {
		let url = url.to_string();
		Box::pin(async move {
			self.urls.lock().push(url);
			match self.queue.lock().pop() {
				Some(response) => Ok(response),
				None => Ok(HttpResponse { status: 200, headers: Vec::new(), body: b"[]".to_vec() }),
			}
		})
	}
}

fn mock_fetcher(responses: Vec<HttpResponse>) -> (Arc<dyn Fetcher>, Arc<ParkingMutex<Vec<String>>>) {
	let urls = Arc::new(ParkingMutex::new(Vec::new()));
	let fetcher: Arc<dyn Fetcher> =
		Arc::new(MockFetcher { urls: urls.clone(), queue: ParkingMutex::new(responses) });
	(fetcher, urls)
}

struct Harness {
	wrapper: Arc<dyn pi_ai::models::Provider>,
	persisted: Persisted,
	urls: Arc<ParkingMutex<Vec<String>>>,
	inner_id: String,
}

fn harness(responses: Vec<HttpResponse>, generated_at: Option<u64>) -> Harness {
	let faux = faux_provider(RegisterFauxProviderOptions {
		provider: Some("testprov".to_string()),
		models: vec![FauxModelDefinition { id: "base".to_string(), ..Default::default() }],
		..Default::default()
	});
	let inner_id = faux.provider.id().to_string();
	let (fetcher, urls) = mock_fetcher(responses);
	let wrapper = with_remote_catalog_fetcher(faux.provider.clone(), None, generated_at, fetcher);
	Harness { wrapper, persisted: Arc::new(ParkingMutex::new(Vec::new())), urls, inner_id }
}

fn context(harness: &Harness, stored: Option<ModelsStoreEntry>, allow_network: bool) -> RefreshModelsContext {
	let log: Persisted = harness.persisted.clone();
	RefreshModelsContext {
		credential: None,
		stored,
		publish: PublishHandle::new(move |publication| {
			let log = log.clone();
			Box::pin(async move {
				if let Some(update) = publication.update {
					update();
				}
				if let Some(ModelsStoreWrite::Write(entry)) = publication.persist {
					log.lock().push(entry);
				}
				true
			})
		}),
		allow_network,
		force: None,
		signal: AbortSignal::new(),
	}
}

fn stored_entry(
	models: Vec<AnyModel>,
	last_modified: Option<i64>,
	checked_at: Option<i64>,
	etag: Option<&str>,
) -> ModelsStoreEntry {
	ModelsStoreEntry { models, last_modified, checked_at, etag: etag.map(str::to_string) }
}

fn chat_entry_from_provider(provider: &str, id: &str) -> AnyModel {
	let mut model = chat_model_json(id);
	model["provider"] = json!(provider);
	AnyModel::from_json(&model).expect("chat model parses")
}

#[test]
fn merge_models_dynamic_wins_and_preserves_order() {
	let mut updated_b = chat_model_json("b");
	updated_b["name"] = json!("Updated B");
	let merged = merge_models(
		vec![chat_entry("a"), chat_entry("b")],
		vec![chat_entry("b2"), AnyModel::from_json(&updated_b).expect("parses")],
	);
	let ids: Vec<&str> = merged.iter().map(|model| model.id()).collect();
	assert_eq!(ids, vec!["a", "b", "b2"]);
	assert_eq!(merged[1].id(), "b");
	assert_eq!(merged[1].to_json()["name"], json!("Updated B"));
}

#[test]
fn parse_catalog_accepts_array_envelope_and_map_shapes() {
	let mut array = chat_model_json("m1");
	array["provider"] = json!("ignored");
	let parsed = parse_catalog("p", &json!([array, { "type": "video", "id": "bad" }, { "nope": true }])).expect("parses");
	assert_eq!(parsed.len(), 1);
	assert_eq!(parsed[0].id(), "m1");
	assert_eq!(parsed[0].provider(), "p");

	let mut envelope = json!({ "models": [chat_model_json("m2")] });
	envelope["models"][0]["provider"] = json!("ignored");
	assert_eq!(parse_catalog("p", &envelope).expect("parses")[0].id(), "m2");

	let mut map = json!({ "m3": chat_model_json("m3") });
	map["m3"]["provider"] = json!("ignored");
	assert_eq!(parse_catalog("p", &map).expect("parses")[0].id(), "m3");
}

#[test]
fn parse_catalog_rejects_invalid_payloads() {
	assert!(parse_catalog("p", &json!(42)).is_err());
	assert!(parse_catalog("p", &json!({ "models": 5 })).is_err());
}

#[test]
fn remote_models_gates_on_local_generated_at() {
	let entry = stored_entry(vec![chat_entry("m")], Some(100), Some(0), None);
	assert!(remote_models(Some(&entry), None).len() == 1);
	assert!(remote_models(Some(&entry), Some(100)).is_empty());
	assert!(remote_models(Some(&entry), Some(50)).len() == 1);
	assert!(remote_models(None, None).is_empty());
	let stale = stored_entry(vec![chat_entry("m")], None, Some(0), None);
	assert!(remote_models(Some(&stale), Some(1)).is_empty());
}

#[tokio::test]
async fn offline_refresh_restores_stored_overlay_without_network() {
	let h = harness(Vec::new(), None);
	let stored = stored_entry(
		vec![chat_entry("dynamic"), chat_entry_from_provider("other-provider", "foreign")],
		Some(1000),
		Some(0),
		None,
	);
	h.wrapper.refresh_models(context(&h, Some(stored), false)).unwrap().await.expect("refresh ok");
	let models = h.wrapper.get_all_models();
	let ids: Vec<&str> = models.iter().map(|m| m.id()).collect();
	assert_eq!(ids, vec!["base", "dynamic"]);
	assert!(h.persisted.lock().is_empty());
	assert!(h.urls.lock().is_empty());
}

#[tokio::test]
async fn fresh_stored_snapshot_skips_the_network() {
	let h = harness(Vec::new(), None);
	let now = crate::utils::time::now_ms();
	let stored = stored_entry(vec![chat_entry("dynamic")], Some(1000), Some(now), None);
	h.wrapper.refresh_models(context(&h, Some(stored), true)).unwrap().await.expect("refresh ok");
	assert!(h.urls.lock().is_empty());
	assert!(h.wrapper.get_all_models().iter().any(|m| m.id() == "dynamic"));
}

#[tokio::test]
async fn forced_refresh_bypasses_the_freshness_window() {
	let h = harness(vec![HttpResponse { status: 200, headers: Vec::new(), body: b"[]".to_vec() }], None);
	let now = crate::utils::time::now_ms();
	let stored = stored_entry(vec![chat_entry("dynamic")], Some(1000), Some(now), None);
	let mut ctx = context(&h, Some(stored), true);
	ctx.force = Some(true);
	h.wrapper.refresh_models(ctx).unwrap().await.expect("refresh ok");
	assert_eq!(h.urls.lock().len(), 1);
}

#[tokio::test]
async fn missing_catalog_resets_the_entry() {
	let h = harness(vec![HttpResponse { status: 404, headers: Vec::new(), body: Vec::new() }], None);
	let stored = stored_entry(vec![chat_entry("dynamic")], Some(1000), Some(0), Some("\"old\"".into()));
	h.wrapper.refresh_models(context(&h, Some(stored), true)).unwrap().await.expect("refresh ok");
	let entry = &h.persisted.lock()[0];
	assert_eq!(entry.last_modified, Some(0));
	assert_eq!(entry.etag, None);
	assert!(entry.checked_at.is_some());
}

#[tokio::test]
async fn transient_failure_keeps_the_validator_and_throws() {
	// fetch_with_retry retries transient statuses twice: keep the last
	// response a 503 so the not-ok branch runs.
	let h = harness(
		vec![
			HttpResponse { status: 503, headers: Vec::new(), body: Vec::new() },
			HttpResponse { status: 503, headers: Vec::new(), body: Vec::new() },
			HttpResponse { status: 503, headers: Vec::new(), body: Vec::new() },
		],
		None,
	);
	let stored = stored_entry(vec![chat_entry("dynamic")], Some(1000), Some(0), Some("\"old\"".into()));
	let result = h.wrapper.refresh_models(context(&h, Some(stored), true)).unwrap().await;
	assert!(result.is_err());
	let entry = &h.persisted.lock()[0];
	assert_eq!(entry.etag.as_deref(), Some("\"old\""));
	assert_eq!(entry.last_modified, Some(1000));
}

#[tokio::test]
async fn not_modified_only_moves_the_freshness_window() {
	let h = harness(
		vec![HttpResponse {
			status: 304,
			headers: Vec::new(),
			body: Vec::new(),
		}],
		None,
	);
	let stored = stored_entry(vec![chat_entry("dynamic")], Some(1000), Some(0), Some("\"old\"".into()));
	h.wrapper.refresh_models(context(&h, Some(stored), true)).unwrap().await.expect("refresh ok");
	let entry = &h.persisted.lock()[0];
	assert_eq!(entry.etag.as_deref(), Some("\"old\""));
	assert_eq!(entry.last_modified, Some(1000));
	assert!(entry.checked_at.unwrap() >= h.urls.lock().len() as i64 - 1);
}

#[tokio::test]
async fn successful_fetch_publishes_the_catalog_and_updates_models() {
	let body = json!([
		chat_model_json("fresh"),
		{ "type": "video", "id": "unsupported" },
	]);
	let h = harness(
		vec![HttpResponse {
			status: 200,
			headers: vec![
				("last-modified".to_string(), "Tue, 15 Nov 1994 12:45:26 GMT".to_string()),
				("etag".to_string(), "\"v2\"".to_string()),
			],
			body: serde_json::to_vec(&body).unwrap(),
		}],
		None,
	);
	h.wrapper.refresh_models(context(&h, None, true)).unwrap().await.expect("refresh ok");
	let entry = &h.persisted.lock()[0];
	assert_eq!(entry.etag.as_deref(), Some("\"v2\""));
	assert_eq!(entry.last_modified, Some(784_903_526_000));
	assert_eq!(entry.models.len(), 1);
	assert_eq!(entry.models[0].id(), "fresh");
	assert_eq!(entry.models[0].provider(), h.inner_id);
	let models = h.wrapper.get_all_models();
	let ids: Vec<&str> = models.iter().map(|m| m.id()).collect();
	assert_eq!(ids, vec!["base", "fresh"]);
	// Chat-only `get_models` mirrors upstream's `isModelType(model, "chat")`
	// filter once non-chat overlays exist.
	let chat_models = h.wrapper.get_models();
	assert_eq!(chat_models.iter().map(|m| m.id()).collect::<Vec<_>>(), vec!["base", "fresh"]);
}

#[test]
fn percent_encode_matches_encode_uri_component_for_provider_ids() {
	assert_eq!(percent_encode("openai"), "openai");
	assert_eq!(percent_encode("custom/id 1"), "custom%2Fid%201");
}

#[test]
fn parse_http_date_handles_imf_fixdate_and_rejects_other_formats() {
	assert_eq!(parse_http_date("Tue, 15 Nov 1994 12:45:26 GMT"), Some(784_903_526_000));
	assert_eq!(parse_http_date("invalid"), None);
}
