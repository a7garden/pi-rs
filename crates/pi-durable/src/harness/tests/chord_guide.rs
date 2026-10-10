use std::{collections::BTreeMap, sync::Arc, time::Duration};

use parking_lot::Mutex;
use pi_chord::context::Context;
use pi_chord::delta::Seg;
use pi_chord::services::provider::RemoteServiceObject;
use pi_chord::types::{
    BoxFuture, FacetHost, JsonValue, RemoteServiceBindingOptions, RemoteServiceImpl,
    RemoteServiceTransport, RemoteServices, ReplicatedState, ServiceCall, ServiceMode,
    ServiceSubscription, StateListener,
};
use pi_chord::{
    create_facet_host, create_remote_service_binding, define_facet, AnyError, Facet, FacetOptions,
};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use super::support::{context, open_default, root};
use crate::documents::{DocDefinition, DocToken, ScopeKind};
use crate::harness::types::{AnyTask, Extension};
use crate::session::observation::{ObservedDocumentValue, WatchListener};
use crate::tasks::{
    commit_change, define_task, NextTaskState, PhaseHandler, TaskDefinition, TaskRuntime,
};
use crate::types::{
    DocumentFork, DocumentHistory, JsonObject, TaskOptions, TaskOwnership, WatchEnd,
};

const CANVAS_SERVICE: &str = "app.canvas";
const REVIEWS_SERVICE: &str = "app.diff-reviews";

fn chord_error(message: impl Into<String>) -> AnyError {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        message.into(),
    ))
}

fn object(value: Value) -> JsonObject {
    value.as_object().expect("JSON object").clone()
}

async fn eventually(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("condition becomes true");
}

struct HostTransport(Arc<dyn FacetHost>);

#[async_trait::async_trait]
impl RemoteServiceTransport for HostTransport {
    async fn invoke(
        &self,
        call: &ServiceCall,
        context: &Context,
    ) -> Result<Option<JsonValue>, AnyError> {
        self.0.services().invoke(call, context).await
    }

    async fn subscribe(
        &self,
        service_id: &str,
        mode: ServiceMode,
        listener: pi_chord::types::UpdateListener,
        _context: &Context,
    ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
        self.0.services().subscribe(service_id, mode, listener)
    }
}

fn binding(host: &Arc<dyn FacetHost>, services: &[&str]) -> Arc<pi_chord::RemoteServiceBinding> {
    create_remote_service_binding(RemoteServiceBindingOptions {
        services: services
            .iter()
            .map(|service| (*service).to_owned())
            .collect(),
        transport: Arc::new(HostTransport(host.clone())),
        bound: true,
        on_error: None,
        assert_access: None,
    })
    .expect("create remote service binding")
}

#[tokio::test]
async fn session_canvas_survives_client_and_facet_host_disposal_until_harness_close() {
    let (_storage, _registry, harness) = open_default().await;
    let context = context();
    let session = harness.session.clone();
    let initial = json!({ "strokes": [] });
    let token = DocToken::new(DocDefinition::singleton(
        "app.canvas",
        1,
        ScopeKind::Session,
        move || object(initial.clone()),
    ));
    let create_token = token.clone();
    session
        .commit(
            move |tx| {
                tx.doc(&create_token, None, None, None)
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
            &context,
        )
        .await
        .expect("create session canvas");
    let state = session
        .document_state(&token, None, None, &context)
        .await
        .expect("attach canvas state")
        .expect("canvas exists");

    let method_session = session.clone();
    let method_token = token.clone();
    let implementation: Arc<dyn RemoteServiceImpl> = RemoteServiceObject::builder()
        .state("state", state.internals())
        .method("addStroke", move |arguments, call_context| {
            let session = method_session.clone();
            let token = method_token.clone();
            Box::pin(async move {
                let stroke = arguments
                    .into_iter()
                    .next()
                    .ok_or_else(|| chord_error("missing stroke"))?;
                session
                    .commit(
                        move |tx| {
                            let draft = tx
                                .doc(&token, None, None, None)
                                .map_err(|error| error.to_string())?;
                            draft
                                .push(&[Seg::Key("strokes".into())], vec![stroke])
                                .map(|_| ())
                        },
                        &call_context,
                    )
                    .await
                    .map_err(|error| chord_error(error.to_string()))?;
                Ok(None)
            }) as BoxFuture<Result<Option<JsonValue>, AnyError>>
        })
        .build();
    let owned_state = state.clone();
    let provider = define_facet(Facet::new("app.canvas/session", move |env| {
        let implementation = implementation.clone();
        env.provide_remote(CANVAS_SERVICE, ServiceMode::Singleton, implementation)?;
        let state = owned_state.clone();
        env.own(Arc::new(move || {
            let state = state.clone();
            Box::pin(async move {
                state.dispose();
                Ok(())
            })
        }))?;
        Ok(())
    }));
    let host = create_facet_host(FacetOptions {
        facets: vec![provider],
        service_sources: Vec::new(),
        on_error: None,
    })
    .await
    .expect("create canvas host");
    let remote = binding(&host, &[CANVAS_SERVICE]);
    let canvas = remote
        .use_service(CANVAS_SERVICE)
        .expect("use canvas service");
    let remote_state = canvas
        .state("state")
        .expect("canvas state member")
        .expect("canvas state exists");
    let updates = Arc::new(Mutex::new(Vec::new()));
    let captured = updates.clone();
    let listener: StateListener = Arc::new(move |value, _context, delivery| {
        let strokes = value
            .get("strokes")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        captured
            .lock()
            .push((delivery.kind, delivery.sequence, strokes));
        Box::pin(async {})
    });
    let unsubscribe = remote_state.subscribe(listener);
    remote
        .ready(&context)
        .await
        .expect("remote canvas snapshot");
    canvas
        .invoke(
            "addStroke",
            vec![json!({"color":"black","points":[{"x":10,"y":20},{"x":30,"y":40}]})],
            &context,
        )
        .await
        .expect("add stroke through remote client");
    eventually(|| updates.lock().len() == 2).await;
    assert_eq!(
        *updates.lock(),
        [
            (pi_chord::types::ReplicatedStateDeliveryKind::Hydrate, 0, 0),
            (pi_chord::types::ReplicatedStateDeliveryKind::Update, 1, 1),
        ]
    );
    assert_eq!(
        session
            .snapshot(&token, None, None, &context)
            .await
            .unwrap()
            .unwrap()["strokes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    unsubscribe();
    remote
        .dispose(&context)
        .await
        .expect("detach remote client");
    host.dispose().await.expect("dispose canvas host");
    assert!(host.services().use_singleton(CANVAS_SERVICE).is_err());
    harness.close(&context).await.expect("close harness");
    assert!(session
        .snapshot(&token, None, None, &context)
        .await
        .is_err());
}

#[tokio::test]
async fn keyed_diff_review_service_routes_by_key_and_persists_comments() {
    let (_storage, _registry, harness) = open_default().await;
    let context = context();
    let root = root(&harness).await;
    let session = harness.session.clone();
    let definition = DocDefinition::conversation_family(
        "app.diff-review",
        1,
        DocumentHistory::Latest,
        DocumentFork::Current,
        |seed| {
            let mut initial = seed.as_object().cloned().unwrap_or_default();
            initial.insert("comments".into(), json!([]));
            initial
        },
    );
    let token = DocToken::new(definition);
    let key = "review-7".to_owned();
    let seed = json!({"path":"a.ts","patch":"-old\n+new"});
    let create_token = token.clone();
    let create_key = key.clone();
    let create_seed = seed.clone();
    session
        .commit(
            move |tx| {
                tx.doc(
                    &create_token,
                    Some(root.id.0),
                    Some(&create_key),
                    Some(create_seed),
                )
                .map(|_| ())
                .map_err(|error| error.to_string())
            },
            &context,
        )
        .await
        .expect("create keyed review document");
    let state = session
        .document_state(&token, Some(root.id.0), Some(&key), &context)
        .await
        .expect("attach review state")
        .expect("review exists");

    let identity_key = key.clone();
    let identity_conversation = root.id;
    let method_session = session.clone();
    let method_token = token.clone();
    let method_key = key.clone();
    let method_seed = seed.clone();
    let implementation: Arc<dyn RemoteServiceImpl> = RemoteServiceObject::builder()
        .state("state", state.internals())
        .method("identity", move |_arguments, _context| {
            let key = identity_key.clone();
            Box::pin(async move {
                Ok(Some(
                    json!({"conversationId": identity_conversation.0, "key": key}),
                ))
            })
        })
        .method("addComment", move |arguments, call_context| {
            let session = method_session.clone();
            let token = method_token.clone();
            let key = method_key.clone();
            let seed = method_seed.clone();
            Box::pin(async move {
                let comment = arguments
                    .into_iter()
                    .next()
                    .ok_or_else(|| chord_error("missing comment"))?;
                session
                    .commit(
                        move |tx| {
                            let draft = tx
                                .doc(
                                    &token,
                                    Some(identity_conversation.0),
                                    Some(&key),
                                    Some(seed),
                                )
                                .map_err(|error| error.to_string())?;
                            draft
                                .push(&[Seg::Key("comments".into())], vec![comment])
                                .map(|_| ())
                        },
                        &call_context,
                    )
                    .await
                    .map_err(|error| chord_error(error.to_string()))?;
                Ok(None)
            }) as BoxFuture<Result<Option<JsonValue>, AnyError>>
        })
        .build();
    let provider_key = key.clone();
    let provider_impl = implementation.clone();
    let owned_state = state.clone();
    let provider = define_facet(Facet::new("app.diff-reviews/session", move |env| {
        let spawner = env.provide_many_remote(REVIEWS_SERVICE, ServiceMode::Keyed)?;
        let key = provider_key.clone();
        let implementation = provider_impl.clone();
        env.on_activate(Arc::new(move || {
            let key = key.clone();
            let implementation = implementation.clone();
            let spawner = spawner.clone();
            Box::pin(async move { spawner.spawn(&key, implementation).map(|_| ()) })
        }))?;
        let state = owned_state.clone();
        env.own(Arc::new(move || {
            let state = state.clone();
            Box::pin(async move {
                state.dispose();
                Ok(())
            })
        }))?;
        Ok(())
    }));
    let host = create_facet_host(FacetOptions {
        facets: vec![provider],
        service_sources: Vec::new(),
        on_error: None,
    })
    .await
    .expect("create review host");
    let remote = binding(&host, &[REVIEWS_SERVICE]);
    let client_slot = Arc::new(Mutex::new(None));
    let captured = client_slot.clone();
    let observed = remote
        .observe(
            REVIEWS_SERVICE,
            Arc::new(move |client, _context| {
                let captured = captured.clone();
                Box::pin(async move { *captured.lock() = Some(client) })
            }),
        )
        .expect("observe keyed reviews");
    remote.ready(&context).await.expect("review service ready");
    let client = client_slot.lock().clone().expect("keyed review client");
    assert_eq!(
        client
            .invoke("identity", Vec::new(), &context)
            .await
            .unwrap(),
        Some(json!({"conversationId": root.id.0, "key": "review-7"}))
    );
    let remote_state = client.state("state").unwrap().unwrap();
    assert_eq!(remote_state.value().unwrap()["comments"], json!([]));
    client
        .invoke(
            "addComment",
            vec![json!({"id":"c1","line":1,"text":"why?"})],
            &context,
        )
        .await
        .expect("add review comment");
    eventually(|| {
        remote_state
            .value()
            .and_then(|value| value["comments"].as_array().map(Vec::len))
            == Some(1)
    })
    .await;
    let stored = session
        .snapshot(&token, Some(root.id.0), Some(&key), &context)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored["path"], "a.ts");
    assert_eq!(stored["comments"][0]["id"], "c1");
    observed();
    remote
        .dispose(&context)
        .await
        .expect("detach review client");
    host.dispose().await.expect("dispose review host");
    harness.close(&context).await.expect("close harness");
}

#[tokio::test]
async fn task_output_watch_delivers_committed_text_then_retirement() {
    let (_storage, registry, harness) = open_default().await;
    let context = context();
    let conversation = root(&harness).await;
    let output = DocToken::new(DocDefinition::singleton(
        "app.job-output",
        1,
        ScopeKind::Task,
        || object(json!({"stdout":"","chunks":0})),
    ));
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));
    let output_for_task = output.clone();
    let running: PhaseHandler =
        Arc::new(move |task, runtime: Arc<dyn TaskRuntime>, task_context| {
            let task_id = task.id;
            let first_output = output_for_task.clone();
            let second_output = output_for_task.clone();
            let entered = entered_tx.clone();
            let release = release_rx.clone();
            Box::pin(async move {
                runtime
                    .commit(
                        commit_change(move |tx, _| {
                            let draft = tx
                                .doc(&first_output, Some(task_id.0), None, None)
                                .map_err(|error| error.to_string())?;
                            draft.set(&[Seg::Key("stdout".into())], json!("$ make\n"))?;
                            draft.set(&[Seg::Key("chunks".into())], json!(1))?;
                            Ok(None)
                        }),
                        &task_context,
                    )
                    .await?;
                if let Some(entered) = entered.lock().take() {
                    let _ = entered.send(());
                }
                let release = { release.lock().take() };
                if let Some(release) = release {
                    let _ = release.await;
                }
                runtime
                    .commit(
                        commit_change(move |tx, _| {
                            let draft = tx
                                .doc(&second_output, Some(task_id.0), None, None)
                                .map_err(|error| error.to_string())?;
                            draft.set(&[Seg::Key("stdout".into())], json!("$ make\nok\n"))?;
                            draft.set(&[Seg::Key("chunks".into())], json!(2))?;
                            Ok(None)
                        }),
                        &task_context,
                    )
                    .await?;
                // Upstream commits the outcome separately from the final output
                // write, so the content publication precedes terminal settlement.
                runtime
                    .commit(
                        commit_change(|_, _| Ok(Some(NextTaskState::completed(Value::Null)))),
                        &task_context,
                    )
                    .await?;
                Ok(())
            })
        });
    let abort: PhaseHandler = Arc::new(|_, runtime, task_context| {
        Box::pin(async move {
            runtime
                .commit(
                    commit_change(|_, _| Ok(Some(NextTaskState::aborted_with("aborted")))),
                    &task_context,
                )
                .await?;
            Ok(())
        })
    });
    let task = define_task(TaskDefinition {
        name: "app.job".into(),
        version: 1,
        initial: Arc::new(|_| json!({"phase":"running"})),
        phases: Arc::new(BTreeMap::from([("running".into(), running)])),
        abort,
        migrate: None,
        hooks: None,
    });
    crate::harness::types::Registry::install(
        registry.as_ref(),
        Extension::new("chord-guide-job").with_tasks(vec![AnyTask {
            definition: task.definition.clone(),
        }]),
    );
    let definition = task.definition.clone();
    let id = conversation
        .commit(
            move |tx| {
                tx.create_task(
                    &AnyTask { definition },
                    json!({"command":"make"}),
                    TaskOptions {
                        ownership: TaskOwnership::Conversation,
                        conversation_id: None,
                        background: None,
                    },
                )
                .map_err(|error| error.to_string())
            },
            &context,
        )
        .await
        .expect("create job task");
    harness.resume();
    entered_rx.await.expect("job wrote first chunk");
    let watch = harness
        .session
        .watch_doc(&output, Some(id.0), None, &context)
        .await
        .expect("acquire job output watch")
        .expect("job output exists");
    let initial = watch.value().expect("initial output value");
    assert_eq!(initial["stdout"], "$ make\n");
    let rendered = Arc::new(Mutex::new(vec![initial["stdout"]
        .as_str()
        .unwrap()
        .to_owned()]));
    let captured = rendered.clone();
    let listener: WatchListener<ObservedDocumentValue> = Arc::new(move |value, _ops, _context| {
        let captured = captured.clone();
        Box::pin(async move {
            captured.lock().push(
                value
                    .as_ref()
                    .and_then(|object| object.get("stdout"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| "retired".into()),
            )
        })
    });
    watch.start(listener).expect("start job output watch");
    release_tx.send(()).expect("release job producer");
    harness
        .wait_for_task(id, &context)
        .await
        .expect("job completes");
    assert_eq!(watch.closed().await, WatchEnd::Retired);
    assert_eq!(&*rendered.lock(), &["$ make\n", "$ make\nok\n", "retired"]);
    assert!(harness
        .session
        .watch_doc(&output, Some(id.0), None, &context)
        .await
        .unwrap()
        .is_none());
    harness.close(&context).await.expect("close harness");
}
