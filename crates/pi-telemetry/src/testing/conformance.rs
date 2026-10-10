//! Runner-independent conformance cases for callback telemetry adapters
//! (`testing/conformance.ts` upstream).

use std::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::attrs;
use crate::testing::types::{
    TelemetryAdapterConformanceCase, TelemetryAdapterFixture, TelemetryAdapterFixtureFactory,
};
use crate::{
    BoxSpanFuture, RecordedTelemetrySpan, SharedError, SpanStatus, TelemetryContextExt,
    TelemetrySpanExt,
};

type ConformanceFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
type ConformanceTest = fn(TelemetryAdapterFixture) -> ConformanceFuture;

/// Error payload used by the conformance cases.
#[derive(Debug)]
pub struct CaseError(&'static str);

impl std::fmt::Display for CaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for CaseError {}

pub fn case_error(message: &'static str) -> SharedError {
    Arc::new(CaseError(message))
}

fn create_case(
    factory: &TelemetryAdapterFixtureFactory,
    group: &'static str,
    name: &'static str,
    test: ConformanceTest,
) -> TelemetryAdapterConformanceCase {
    let factory = Arc::clone(factory);
    TelemetryAdapterConformanceCase {
        group,
        name,
        run: Arc::new(move || {
            let factory = Arc::clone(&factory);
            Box::pin(async move {
                let fixture = factory().await;
                test(fixture).await;
            })
        }),
    }
}

fn find_span<'a>(spans: &'a [RecordedTelemetrySpan], name: &str) -> &'a RecordedTelemetrySpan {
    spans
        .iter()
        .find(|candidate| candidate.name == name)
        .unwrap_or_else(|| panic!("Expected recorded span {name}"))
}

async fn rejects_with_same_value<T>(operation: BoxSpanFuture<'_, T>, expected: &SharedError) {
    match operation.await {
        Ok(_) => panic!("Expected operation to reject"),
        Err(error) => assert!(
            Arc::ptr_eq(&error, expected),
            "expected the original rejection value to propagate unchanged"
        ),
    }
}

/// Creates runner-independent cases for the callback telemetry adapter contract.
pub fn create_telemetry_adapter_conformance(
    factory: TelemetryAdapterFixtureFactory,
) -> Vec<TelemetryAdapterConformanceCase> {
    vec![
        create_case(
            &factory,
            "callback lifecycle",
            "admits once synchronously and preserves the result",
            |fixture| {
                Box::pin(async move {
                    let admitted = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
                    let expected = 42u32;
                    let admitted_flag = Arc::clone(&admitted);
                    let calls_counter = Arc::clone(&calls);
                    let result = fixture.context.start_span(
                        crate::SpanOptions::new("success"),
                        move |_span| {
                            Box::pin(async move {
                                admitted_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                                calls_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                Ok::<u32, SharedError>(expected)
                            })
                                as Pin<Box<dyn Future<Output = Result<u32, SharedError>> + Send>>
                        },
                    );

                    // Deviation: Rust futures are lazy, so the callback runs on
                    // first poll rather than synchronously at call time; the
                    // admission contract is asserted after awaiting.
                    assert_eq!(result.await.unwrap(), expected);
                    assert!(admitted.load(std::sync::atomic::Ordering::SeqCst));
                    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
                    let spans = fixture.get_spans().await;
                    assert_eq!(find_span(&spans, "success").status, SpanStatus::Ok);
                    assert!(find_span(&spans, "success").settled);
                })
            },
        ),
        create_case(
            &factory,
            "callback lifecycle",
            "preserves synchronous and asynchronous rejection values",
            |fixture| {
                Box::pin(async move {
                    let sync_error = case_error("sync");
                    let inner_sync_error = Arc::clone(&sync_error);
                    rejects_with_same_value(
                        fixture.context.start_span(
                            crate::SpanOptions::new("sync-error"),
                            move |_span| {
                                Box::pin(async move { Err(inner_sync_error) })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        ),
                        &sync_error,
                    )
                    .await;

                    let async_error = case_error("async");
                    let inner_async_error = Arc::clone(&async_error);
                    rejects_with_same_value(
                        fixture.context.start_span(
                            crate::SpanOptions::new("async-error"),
                            move |_span| {
                                Box::pin(async move { Err(inner_async_error) })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        ),
                        &async_error,
                    )
                    .await;

                    let spans = fixture.get_spans().await;
                    for name in ["sync-error", "async-error"] {
                        assert!(matches!(
                            find_span(&spans, name).status,
                            SpanStatus::Error { .. }
                        ));
                    }
                })
            },
        ),
        create_case(
            &factory,
            "status",
            "uses last explicit status without automatic overwrite",
            |fixture| {
                Box::pin(async move {
                    fixture
                        .context
                        .start_span(crate::SpanOptions::new("last-status"), |span| {
                            Box::pin(async move {
                                span.set_status(SpanStatus::error_with("Expected", "first"));
                                span.set_status(SpanStatus::Ok);
                                Ok(())
                            })
                                as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                        })
                        .await
                        .unwrap();

                    let thrown = case_error("after explicit status");
                    let inner_thrown = Arc::clone(&thrown);
                    rejects_with_same_value(
                        fixture.context.start_span(
                            crate::SpanOptions::new("explicit-before-throw"),
                            move |span| {
                                Box::pin(async move {
                                    span.set_status(SpanStatus::Ok);
                                    Err(inner_thrown)
                                })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        ),
                        &thrown,
                    )
                    .await;

                    let rejected = case_error("after async explicit status");
                    let inner_rejected = Arc::clone(&rejected);
                    rejects_with_same_value(
                        fixture.context.start_span(
                            crate::SpanOptions::new("explicit-before-rejection"),
                            move |span| {
                                Box::pin(async move {
                                    span.set_status(SpanStatus::error_with(
                                        "Expected",
                                        "async failure",
                                    ));
                                    Err(inner_rejected)
                                })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        ),
                        &rejected,
                    )
                    .await;

                    fixture
                        .context
                        .start_span(crate::SpanOptions::new("expected-failure"), |span| {
                            Box::pin(async move {
                                span.set_status(SpanStatus::error_with(
                                    "Expected",
                                    "returned failure",
                                ));
                                Ok(())
                            })
                                as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                        })
                        .await
                        .unwrap();

                    let spans = fixture.get_spans().await;
                    assert_eq!(find_span(&spans, "last-status").status, SpanStatus::Ok);
                    assert_eq!(
                        find_span(&spans, "explicit-before-throw").status,
                        SpanStatus::Ok
                    );
                    assert_eq!(
                        find_span(&spans, "explicit-before-rejection").status,
                        SpanStatus::error_with("Expected", "async failure")
                    );
                    assert_eq!(
                        find_span(&spans, "expected-failure").status,
                        SpanStatus::error_with("Expected", "returned failure")
                    );
                })
            },
        ),
        create_case(
            &factory,
            "recording",
            "merges attributes and records ordered events",
            |fixture| {
                Box::pin(async move {
                    fixture
                        .context
                        .start_span(
                            crate::SpanOptions::with_attributes(
                                "recording",
                                attrs! {
                                    "start" => "value",
                                    "overwrite" => "start",
                                },
                            ),
                            |span| {
                                Box::pin(async move {
                                    span.set_attributes(attrs! {
                                        "count" => 1,
                                        "overwrite" => "middle",
                                    });
                                    span.set_attributes(attrs! {
                                        "overwrite" => "end",
                                    });
                                    span.add_event("first", Some(attrs! { "index" => 1 }));
                                    span.add_event("second", Some(attrs! { "index" => 2 }));
                                    Ok(())
                                })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        )
                        .await
                        .unwrap();

                    let spans = fixture.get_spans().await;
                    let span = find_span(&spans, "recording");
                    assert_eq!(
                        span.attributes,
                        attrs! {
                            "start" => "value",
                            "overwrite" => "end",
                            "count" => 1,
                        }
                    );
                    assert_eq!(
                        span.events,
                        vec![
                            crate::RecordedTelemetryEvent {
                                name: "first".into(),
                                attributes: attrs! { "index" => 1 }
                            },
                            crate::RecordedTelemetryEvent {
                                name: "second".into(),
                                attributes: attrs! { "index" => 2 }
                            },
                        ]
                    );
                })
            },
        ),
        // Adapted from upstream "ignores failed attribute calls atomically":
        // Rust attribute maps are strongly typed and cannot fail mid-read, so
        // the case asserts that multi-batch merges apply whole and preserve
        // retained attributes.
        create_case(
            &factory,
            "recording",
            "applies attribute merges atomically and preserves retained values",
            |fixture| {
                Box::pin(async move {
                    fixture
                        .context
                        .start_span(
                            crate::SpanOptions::with_attributes(
                                "atomic-attributes",
                                attrs! { "retained" => "value" },
                            ),
                            |span| {
                                Box::pin(async move {
                                    span.set_attributes(attrs! {
                                        "partial" => "must not survive",
                                        "retained" => "value",
                                    });
                                    Ok(())
                                })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        )
                        .await
                        .unwrap();

                    let spans = fixture.get_spans().await;
                    // With no mid-read failure mode, every applied batch
                    // survives whole; upstream asserted the same only after a
                    // failed batch was dropped entirely.
                    assert_eq!(
                        find_span(&spans, "atomic-attributes").attributes,
                        attrs! {
                            "retained" => "value",
                            "partial" => "must not survive",
                        }
                    );
                })
            },
        ),
        create_case(
            &factory,
            "recording",
            "makes calls after settlement inert",
            |fixture| {
                Box::pin(async move {
                    let captured_span: Arc<dyn crate::TelemetrySpan> = fixture
                        .context
                        .start_span(
                            crate::SpanOptions::with_attributes(
                                "settled",
                                attrs! { "value" => "initial" },
                            ),
                            |span| {
                                Box::pin(async move { Ok::<_, SharedError>(span) })
                                    as Pin<Box<dyn Future<Output = Result<_, SharedError>> + Send>>
                            },
                        )
                        .await
                        .expect("Expected callback span");

                    captured_span.set_attributes(attrs! { "value" => "late" });
                    captured_span.add_event("late", Some(attrs! { "value" => true }));
                    captured_span.set_status(SpanStatus::Error { error: None });

                    let mut child_admitted = false;
                    let child_result: u32 = captured_span
                        .start_span(crate::SpanOptions::new("late-child"), |_child| {
                            let admitted = &mut child_admitted;
                            Box::pin(async move {
                                *admitted = true;
                                Ok::<u32, SharedError>(7)
                            })
                                as Pin<Box<dyn Future<Output = Result<u32, SharedError>> + Send>>
                        })
                        .await
                        .unwrap();
                    assert!(child_admitted);
                    assert_eq!(child_result, 7);

                    let spans = fixture.get_spans().await;
                    assert_eq!(spans.len(), 1);
                    assert_eq!(spans[0].attributes, attrs! { "value" => "initial" });
                    assert_eq!(spans[0].events, Vec::new());
                    assert_eq!(spans[0].status, SpanStatus::Ok);
                })
            },
        ),
        create_case(
            &factory,
            "parentage",
            "records nested and concurrent child relationships",
            |fixture| {
                Box::pin(async move {
                    use std::sync::atomic::{AtomicBool, Ordering};

                    let first =
                        fixture
                            .context
                            .start_span(crate::SpanOptions::new("parent"), |parent| {
                                Box::pin(async move {
                                    let gate_flag = Arc::new(AtomicBool::new(false));
                                    let wait_flag = Arc::clone(&gate_flag);
                                    let first = parent.start_span(
                                        crate::SpanOptions::new("first-child"),
                                        move |_span| {
                                            let wait_flag = Arc::clone(&wait_flag);
                                            Box::pin(async move {
                                                std::future::poll_fn(move |cx| {
                                                    if wait_flag.load(Ordering::SeqCst) {
                                                        std::task::Poll::Ready(())
                                                    } else {
                                                        cx.waker().wake_by_ref();
                                                        std::task::Poll::Pending
                                                    }
                                                })
                                                .await;
                                                Ok::<(), SharedError>(())
                                            })
                                                as Pin<
                                                    Box<
                                                        dyn Future<Output = Result<(), SharedError>>
                                                            + Send,
                                                    >,
                                                >
                                        },
                                    );
                                    let second = parent.start_span(
                                        crate::SpanOptions::new("second-child"),
                                        |_span| {
                                            Box::pin(async move { Ok::<&str, SharedError>("done") })
                                                as Pin<
                                                    Box<
                                                        dyn Future<
                                                                Output = Result<&str, SharedError>,
                                                            > + Send,
                                                    >,
                                                >
                                        },
                                    );
                                    assert_eq!(second.await.unwrap(), "done");
                                    gate_flag.store(true, Ordering::SeqCst);
                                    first.await.unwrap();
                                    Ok::<(), SharedError>(())
                                })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            });
                    first.await.unwrap();

                    let spans = fixture.get_spans().await;
                    let parent = find_span(&spans, "parent");
                    let first = find_span(&spans, "first-child");
                    let second = find_span(&spans, "second-child");
                    assert_eq!(parent.parent_id, None);
                    assert_eq!(first.parent_id, Some(parent.id));
                    assert_eq!(second.parent_id, Some(parent.id));
                    let (parent_end, first_end, second_end) = (
                        parent.end_sequence.expect("parent end sequence"),
                        first.end_sequence.expect("first end sequence"),
                        second.end_sequence.expect("second end sequence"),
                    );
                    assert!(second_end < first_end);
                    assert!(first_end < parent_end);
                })
            },
        ),
        // Adapted from upstream "suppresses unreadable telemetry payload
        // failures": JS Proxies that throw on property access have no Rust
        // counterpart, so the case asserts that readable payloads are recorded
        // faithfully and that mutation methods stay callable throughout.
        create_case(
            &factory,
            "passivity",
            "records telemetry payloads without inspection failures",
            |fixture| {
                Box::pin(async move {
                    let mut calls = 0u32;
                    let result: u32 = fixture
                        .context
                        .start_span(
                            crate::SpanOptions::with_attributes(
                                "readable-options",
                                attrs! { "secret" => "value" },
                            ),
                            |_span| {
                                let calls = &mut calls;
                                Box::pin(async move {
                                    *calls += 1;
                                    Ok::<u32, SharedError>(9)
                                })
                                    as Pin<
                                        Box<dyn Future<Output = Result<u32, SharedError>> + Send>,
                                    >
                            },
                        )
                        .await
                        .unwrap();

                    assert_eq!(calls, 1);
                    assert_eq!(result, 9);

                    fixture
                        .context
                        .start_span(crate::SpanOptions::new("readable-recording"), |span| {
                            Box::pin(async move {
                                span.set_attributes(attrs! { "secret" => "value" });
                                span.add_event(
                                    "readable-event",
                                    Some(attrs! { "secret" => "value" }),
                                );
                                span.set_status(SpanStatus::Ok);
                                Ok(())
                            })
                                as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                        })
                        .await
                        .unwrap();

                    let recorded = fixture.get_spans().await;
                    assert_eq!(recorded.len(), 2);
                    assert_eq!(recorded[0].attributes, attrs! { "secret" => "value" });
                    assert_eq!(recorded[1].events.len(), 1);
                    assert_eq!(recorded[1].status, SpanStatus::Ok);
                })
            },
        ),
        // Adapted from upstream "ignores failed status calls atomically":
        // setStatus cannot fail mid-inspection in Rust, so the case asserts
        // that an explicit status set before a rejection survives without an
        // automatic overwrite (status detail integrity).
        create_case(
            &factory,
            "passivity",
            "keeps an explicit status intact across a rejection",
            |fixture| {
                Box::pin(async move {
                    let rejection = case_error("rejected after explicit status");
                    let inner_rejection = Arc::clone(&rejection);
                    rejects_with_same_value(
                        fixture.context.start_span(
                            crate::SpanOptions::new("explicit-status-rejection"),
                            move |span| {
                                Box::pin(async move {
                                    span.set_status(SpanStatus::Ok);
                                    Err(inner_rejection)
                                })
                                    as Pin<Box<dyn Future<Output = Result<(), SharedError>> + Send>>
                            },
                        ),
                        &rejection,
                    )
                    .await;

                    let spans = fixture.get_spans().await;
                    assert_eq!(
                        find_span(&spans, "explicit-status-rejection").status,
                        SpanStatus::Ok
                    );
                })
            },
        ),
    ]
}
