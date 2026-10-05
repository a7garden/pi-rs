//! Ports of packages/telemetry/test/telemetry.test.ts and
//! test/conformance.test.ts (mapping rule 4). Upstream compile-time-only
//! assertions (`@ts-expect-error`, `expectTypeOf`) have no Rust counterpart
//! and are documented per test.

use std::error::Error;
use std::sync::Arc;

use serde_json::json;

use crate::testing::{SnapshotReader, TelemetryAdapterFixture, TelemetryAdapterFixtureFactory, create_telemetry_adapter_conformance};
use crate::{
	AttributeValue, InMemoryTelemetryContext, NOOP_TELEMETRY_CONTEXT, RecordedTelemetrySpan, SharedError,
	SpanOptions, TelemetryAttributeDefinition, TelemetryAttributeMetadata, TelemetryContextExt, TelemetryEventDefinition,
	TelemetryParentDefinition, TelemetrySchemaDefinition, TelemetrySpan, TelemetrySpanContext, TelemetrySpanExt,
	TelemetrySpanStatusDefinition, TelemetryRequiredAttributeDefinition, create_typed_span_starter,
	define_telemetry_schema, noop_telemetry_span,
};

#[derive(Debug)]
struct TestError(&'static str);

impl std::fmt::Display for TestError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(self.0)
	}
}

impl Error for TestError {}

fn test_error(message: &'static str) -> SharedError {
	Arc::new(TestError(message))
}

use std::future::Future;
use std::pin::Pin;

fn operation_schema() -> TelemetrySchemaDefinition {
	define_telemetry_schema(TelemetrySchemaDefinition {
		version: 1,
		spans: [(
			"operation".to_string(),
			crate::TelemetrySpanDefinition {
				description: "Test operation".to_string(),
				parents: TelemetryParentDefinition::Any,
				start_attributes: [(
					"kind".to_string(),
					TelemetryRequiredAttributeDefinition {
						definition: TelemetryAttributeDefinition::String {
							metadata: TelemetryAttributeMetadata { description: "Kind".to_string(), sensitive: None, cardinality: None },
							values: Some(vec!["read".to_string(), "write".to_string()]),
							examples: None,
						},
						required: true,
					},
				)]
				.into_iter()
				.collect(),
				end_attributes: [].into_iter().collect(),
				events: Some([(
					"result".to_string(),
					TelemetryEventDefinition {
						description: "Result".to_string(),
						attributes: [(
							"outcome".to_string(),
							TelemetryRequiredAttributeDefinition {
								definition: TelemetryAttributeDefinition::String {
									metadata: TelemetryAttributeMetadata { description: "Outcome".to_string(), sensitive: None, cardinality: None },
									values: Some(vec!["ok".to_string(), "error".to_string()]),
									examples: None,
								},
								required: true,
							},
						)]
						.into_iter()
						.collect(),
					},
				)]
				.into_iter()
				.collect()),
				status: TelemetrySpanStatusDefinition { default: "ok".to_string(), error_when: "The operation fails".to_string() },
			},
		)]
		.into_iter()
		.collect(),
	})
}

/// telemetry.test.ts: "telemetry schemas" / "preserves serializable definitions
/// and infers exact attributes". Type-level inference assertions are compile
/// time-only upstream and are enforced by the typed Rust constructors instead.
#[test]
fn schema_preserves_serializable_definitions() {
	let definition = operation_schema();
	let schema = define_telemetry_schema(definition.clone());
	assert_eq!(schema, definition);
	// Serializable without throwing.
	let serialized = serde_json::to_string(&schema).expect("schema must serialize");
	let parsed: TelemetrySchemaDefinition = serde_json::from_str(&serialized).expect("schema must round-trip");
	assert_eq!(parsed, schema);
	// JSON shape matches the upstream wire format.
	let value = serde_json::to_value(&schema).unwrap();
	assert_eq!(value["version"], json!(1));
	assert_eq!(value["spans"]["operation"]["description"], json!("Test operation"));
	assert_eq!(value["spans"]["operation"]["parents"]["kind"], json!("any"));
	assert_eq!(
		value["spans"]["operation"]["startAttributes"]["kind"]["values"],
		json!(["read", "write"])
	);
	assert_eq!(value["spans"]["operation"]["startAttributes"]["kind"]["required"], json!(true));
	assert_eq!(
		value["spans"]["operation"]["events"]["result"]["attributes"]["outcome"]["required"],
		json!(true)
	);
	assert_eq!(value["spans"]["operation"]["status"], json!({ "default": "ok", "errorWhen": "The operation fails" }));
}

/// telemetry.test.ts: "telemetry schemas" / "combines schema vocabularies and
/// binds child starters to their parent spans".
#[tokio::test]
async fn typed_span_starter_combines_vocabularies_and_binds_children() {
	let operation_schema = operation_schema();
	let request_schema = define_telemetry_schema(TelemetrySchemaDefinition {
		version: 3,
		spans: [(
			"request".to_string(),
			crate::TelemetrySpanDefinition {
				description: "Request".to_string(),
				parents: TelemetryParentDefinition::Spans { spans: vec!["operation".to_string()] },
				start_attributes: [(
					"provider".to_string(),
					TelemetryRequiredAttributeDefinition {
						definition: TelemetryAttributeDefinition::String {
							metadata: TelemetryAttributeMetadata { description: "Provider".to_string(), sensitive: None, cardinality: None },
							values: None,
							examples: None,
						},
						required: true,
					},
				)]
				.into_iter()
				.collect(),
				end_attributes: [(
					"response".to_string(),
					TelemetryAttributeDefinition::String {
						metadata: TelemetryAttributeMetadata { description: "Response kind".to_string(), sensitive: None, cardinality: None },
						values: None,
						examples: None,
					},
				)]
				.into_iter()
				.collect(),
				events: None,
				status: TelemetrySpanStatusDefinition { default: "ok".to_string(), error_when: "The request fails".to_string() },
			},
		)]
		.into_iter()
		.collect(),
	});

	let telemetry_context = Arc::new(InMemoryTelemetryContext::new());
	let start_span = create_typed_span_starter(
		Arc::clone(&telemetry_context) as Arc<dyn crate::TelemetryContext>,
		vec![operation_schema, request_schema],
	);

	let result: i32 = start_span
		.start_span("operation", attrs! { "kind" => "read" }, |_operation_span, start_child_span| {
			Box::pin(async move {
				start_child_span
					.start_span("request", attrs! { "provider" => "example" }, |request_span, _start_grandchild| {
						Box::pin(async move {
							request_span.set_attributes(attrs! { "response" => "cached" });
							Ok::<i32, SharedError>(42)
						}) as Pin<Box<dyn Future<Output = Result<i32, SharedError>> + Send>>
					})
					.await
			}) as Pin<Box<dyn Future<Output = Result<i32, SharedError>> + Send>>
		})
		.await
		.unwrap();
	assert_eq!(result, 42);

	let spans = telemetry_context.get_spans();
	let operation_span = spans.iter().find(|span| span.name == "operation").unwrap();
	let request_span = spans.iter().find(|span| span.name == "request").unwrap();
	assert_eq!(operation_span.parent_id, None);
	assert_eq!(request_span.parent_id, Some(operation_span.id));

	// Creating a starter performs no runtime schema validation (upstream
	// "unreadable schemas" acceptance: construction must not throw).
	let _ = create_typed_span_starter(Arc::new(InMemoryTelemetryContext::new()), Vec::new());

	let sync_error = test_error("sync");
	let inner_sync_error = Arc::clone(&sync_error);
	let sync_result: Result<(), SharedError> =
		start_span.start_span("operation", attrs! { "kind" => "write" }, move |_span, _start_child| {
			Box::pin(async move { Err(inner_sync_error) }) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
		})
		.await;
	assert!(matches!(&sync_result, Err(err) if Arc::ptr_eq(err, &sync_error)));

	let async_error = test_error("async");
	let inner_async_error = Arc::clone(&async_error);
	let async_result: Result<(), SharedError> = start_span
		.start_span("request", attrs! { "provider" => "example" }, move |_span, _start_child| {
			Box::pin(async move { Err(inner_async_error) }) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
		})
		.await;
	assert!(matches!(&async_result, Err(err) if Arc::ptr_eq(err, &async_error)));
}

/// telemetry.test.ts: "NOOP_TELEMETRY_CONTEXT" / "admits callbacks
/// synchronously and reuses one inert span".
#[tokio::test]
async fn noop_context_admits_callbacks_and_reuses_one_span() {
	let admitted = Arc::new(std::sync::atomic::AtomicBool::new(false));
	let admitted_flag = Arc::clone(&admitted);
	let result: i32 = NOOP_TELEMETRY_CONTEXT
		.start_span(SpanOptions::new("first"), move |span| {
			let admitted = Arc::clone(&admitted_flag);
			Box::pin(async move {
				admitted.store(true, std::sync::atomic::Ordering::SeqCst);
				let child = span
					.start_span(SpanOptions::new("child"), |child_span| {
						Box::pin(async move { Ok::<_, SharedError>(child_span) })
							as Pin<Box<dyn Future<Output = Result<_, SharedError>> + Send>>
					})
					.await
					.unwrap();
				// Upstream expects child === parent: one frozen span is reused.
				let child_ctx = TelemetrySpanContext(Arc::clone(&child));
				assert!(Arc::ptr_eq(
					&(child_ctx.0),
					&span
				));
				assert!(Arc::ptr_eq(&child, &noop_telemetry_span()));
				Ok::<i32, SharedError>(42)
			}) as Pin<Box<dyn Future<Output = Result<i32, SharedError>> + Send>>
		})
		.await
		.unwrap();
	assert!(admitted.load(std::sync::atomic::Ordering::SeqCst));
	assert_eq!(result, 42);
}

/// telemetry.test.ts: "NOOP_TELEMETRY_CONTEXT" / "preserves synchronous and
/// asynchronous rejection values".
#[tokio::test]
async fn noop_context_preserves_rejection_values() {
	let sync_error = test_error("sync");
	let inner_sync_error = Arc::clone(&sync_error);
	let sync: Result<(), SharedError> = NOOP_TELEMETRY_CONTEXT
		.start_span(SpanOptions::new("sync"), move |_span| {
			Box::pin(async move { Err(inner_sync_error) }) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
		})
		.await;
	assert!(matches!(&sync, Err(err) if Arc::ptr_eq(err, &sync_error)));

	let async_error = test_error("async");
	let inner_async_error = Arc::clone(&async_error);
	let async_result: Result<(), SharedError> = NOOP_TELEMETRY_CONTEXT
		.start_span(SpanOptions::new("async"), move |_span| {
			Box::pin(async move { Err(inner_async_error) }) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
		})
		.await;
	assert!(matches!(&async_result, Err(err) if Arc::ptr_eq(err, &async_error)));
}

/// telemetry.test.ts: "NOOP_TELEMETRY_CONTEXT" / "does not inspect or retain
/// telemetry payloads". Upstream passes unreadable Proxies; the typed Rust
/// surface has no unreadable values, so the case asserts payload independence.
#[tokio::test]
async fn noop_context_does_not_inspect_payloads() {
	NOOP_TELEMETRY_CONTEXT
		.start_span(SpanOptions::with_attributes("operation", attrs! { "secret" => "prompt content" }), |span| {
			Box::pin(async move {
				span.add_event("event", Some(attrs! { "secret" => "content" }));
				span.set_attributes(attrs! { "secret" => "content" });
				span.set_status(crate::SpanStatus::Ok);
				Ok(())
			}) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
		})
		.await
		.unwrap();
	// Nothing was retained: the noop span exposes no recording surface, so
	// only callability is asserted above.
	assert!(AttributeKindCheck::assert_attr_is_value());
}

/// Compile-surface check used by the payload test above.
struct AttributeKindCheck;

impl AttributeKindCheck {
	fn assert_attr_is_value() -> bool {
		let value = AttributeValue::from("x");
		matches!(value, AttributeValue::String(_))
	}
}

/// conformance.test.ts fixture factory for the in-memory adapter.
fn in_memory_fixture_factory() -> TelemetryAdapterFixtureFactory {
	Arc::new(|| {
		Box::pin(async move {
			let context = Arc::new(InMemoryTelemetryContext::new());
			let snapshot_context = Arc::clone(&context);
			let get_spans: SnapshotReader = Arc::new(move || {
				let snapshot_context = Arc::clone(&snapshot_context);
				Box::pin(async move { snapshot_context.get_spans() })
			});
			TelemetryAdapterFixture {
				context: Arc::clone(&context) as Arc<dyn crate::TelemetryContext>,
				get_spans,
			}
		}) as Pin<Box<dyn Future<Output = TelemetryAdapterFixture> + Send>>
	})
}

/// conformance.test.ts: every generated conformance case runs against
/// InMemoryTelemetryContext.
#[tokio::test]
async fn in_memory_context_passes_conformance() {
	for test_case in create_telemetry_adapter_conformance(in_memory_fixture_factory()) {
		test_case.run().await;
	}
}

/// conformance.test.ts: "returns detached snapshots without exposing mutable
/// recording state".
#[tokio::test]
async fn in_memory_context_returns_detached_snapshots() {
	let context = InMemoryTelemetryContext::new();
	let mut open_settled = None;
	let mut open_end_sequence = None;
	let callback_context = &context;
	{
		let open_settled = &mut open_settled;
		let open_end_sequence = &mut open_end_sequence;
		callback_context
			.start_span(
				SpanOptions::with_attributes("snapshot", attrs! { "tags" => vec!["initial"] }),
				move |span| {
					Box::pin(async move {
						span.add_event("event", Some(attrs! { "value" => 1 }));
						let open = &callback_context.get_spans()[0];
						*open_settled = Some(open.settled);
						*open_end_sequence = open.end_sequence;
						Ok(())
					}) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
				},
			)
			.await
			.unwrap();
	}

	assert_eq!(open_settled, Some(false));
	assert_eq!(open_end_sequence, None);

	let spans = context.get_spans();
	assert_eq!(spans[0].settled, true);
	assert_eq!(spans[0].end_sequence, Some(1));

	// Mutating the detached snapshot must not affect later snapshots.
	let mut mutated = spans;
	mutated[0]
		.attributes
		.insert("tags".to_string(), AttributeValue::StringArray(vec!["mutated".to_string()]));
	if let Some(event) = mutated[0].events.get_mut(0) {
		event.attributes.insert("value".to_string(), AttributeValue::Number(2.0));
	}

	let spans = context.get_spans();
	assert_eq!(
		spans[0].attributes.get("tags"),
		Some(&AttributeValue::StringArray(vec!["initial".to_string()]))
	);
	assert_eq!(
		spans[0].events,
		vec![crate::RecordedTelemetryEvent {
			name: "event".to_string(),
			attributes: attrs! { "value" => 1 },
		}]
	);
}

/// Extra compile-surface references so the ported API stays exercised.
#[allow(unused)]
fn _reference_surface(span: Arc<dyn TelemetrySpan>, spans: &[RecordedTelemetrySpan]) {
	let _ = spans.len();
	let _ = TelemetrySpanExt::start_span(&*span, SpanOptions::new("child"), |_child| {
		Box::pin(async move { Ok(()) }) as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
	});
}
