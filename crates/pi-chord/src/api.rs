//! Port of `packages/chord/src/api.ts`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::facets::host::FacetKernel;
use crate::facets::loader::dispose_loaded_facets;
use crate::services::consumer::RemoteServiceBinding;
use crate::services::errors::{type_error, AggregateError, AnyError};
use crate::services::provider::RemoteServiceProvider;
use crate::services::state::{
    attach_replicated_state_source, AttachedReplicatedState, MutableReplicatedState,
};
use crate::types::{
    BoxFuture, Facet, FacetHost, FacetLoader, FacetOptions, JsonValue, LoadedFacets,
    RemoteServiceBindingOptions, ReplicatedStateSource, ReplicatedStateSourceOptions, Service,
};

/// Create an active host for one complete set of facets.
pub async fn create_facet_host(options: FacetOptions) -> Result<Arc<dyn FacetHost>, AnyError> {
    let mut kernel = FacetKernel::new(options)?;
    kernel.activate().await?;
    let provider = kernel.provider()?;
    Ok(Arc::new(FacetHostFacade {
        kernel: Arc::new(tokio::sync::Mutex::new(kernel)),
        provider,
    }))
}

/// Host facade over the kernel; `reload`/`dispose` serialize behind a mutex so
/// the shared [`FacetHost`] trait can take `&self`.
struct FacetHostFacade {
    kernel: Arc<tokio::sync::Mutex<FacetKernel>>,
    provider: Arc<RemoteServiceProvider>,
}

#[async_trait]
impl FacetHost for FacetHostFacade {
    fn services(&self) -> &RemoteServiceProvider {
        &self.provider
    }

    async fn reload(&self, facets: Vec<Facet>) -> Result<(), AnyError> {
        self.kernel.lock().await.reload(facets).await
    }

    async fn dispose(&self) -> Result<(), AnyError> {
        self.kernel.lock().await.dispose().await
    }
}

/// A loader returning one fixed facet set (upstream `createStaticFacetLoader`).
pub fn create_static_facet_loader(facets: Vec<Facet>) -> Arc<dyn FacetLoader> {
    Arc::new(StaticFacetLoader { facets })
}

struct StaticFacetLoader {
    facets: Vec<Facet>,
}

impl FacetLoader for StaticFacetLoader {
    fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>> {
        let facets = self.facets.clone();
        Box::pin(async move { Ok(LoadedFacets::new(facets)) })
    }
}

/// Chain several loaders into one; a failed load disposes the generations
/// loaded so far in reverse (upstream `combineFacetLoaders`).
pub fn combine_facet_loaders(loaders: Vec<Arc<dyn FacetLoader>>) -> Arc<dyn FacetLoader> {
    Arc::new(CombinedFacetLoader { loaders })
}

struct CombinedFacetLoader {
    loaders: Vec<Arc<dyn FacetLoader>>,
}

impl FacetLoader for CombinedFacetLoader {
    fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>> {
        let loaders = self.loaders.clone();
        Box::pin(async move {
            let mut loaded: Vec<LoadedFacets> = Vec::new();
            let load: Result<(), AnyError> = async {
                for loader in &loaders {
                    loaded.push(loader.load().await?);
                }
                Ok(())
            }
            .await;
            if let Err(load_error) = load {
                loaded.reverse();
                let cleanup_errors = dispose_loaded_facets(&mut loaded).await;
                if !cleanup_errors.is_empty() {
                    let mut errors = Vec::with_capacity(cleanup_errors.len() + 1);
                    errors.push(load_error);
                    errors.extend(cleanup_errors);
                    return Err(AggregateError::new(
                        "Facet loading and cleanup failed",
                        errors,
                    ));
                }
                return Err(load_error);
            }
            let facets: Vec<Facet> = loaded
                .iter()
                .flat_map(|entry| entry.facets.iter().cloned())
                .collect();
            let mut combined = CombinedLoadedFacets {
                loaded,
                disposed: false,
            };
            Ok(LoadedFacets {
                facets,
                dispose: Some(Box::new(move || {
                    Box::pin(async move { combined.dispose().await })
                })),
            })
        })
    }
}

struct CombinedLoadedFacets {
    loaded: Vec<LoadedFacets>,
    disposed: bool,
}

impl CombinedLoadedFacets {
    async fn dispose(&mut self) -> Result<(), AnyError> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        self.loaded.reverse();
        let errors = dispose_loaded_facets(&mut self.loaded).await;
        if errors.len() == 1 {
            return Err(errors.into_iter().next().expect("one error"));
        }
        if errors.len() > 1 {
            return Err(AggregateError::new(
                "Failed to dispose loaded facets",
                errors,
            ));
        }
        Ok(())
    }
}

/// Identity helper marking a facet definition (upstream `defineFacet`).
pub fn define_facet(facet: Facet) -> Facet {
    facet
}

/// Define one service identity, rejecting empty and reserved IDs (upstream
/// `defineService`).
pub fn define_service<T>(id: &str, local: bool) -> Result<Service<T>, AnyError> {
    if id.is_empty() {
        return Err(type_error("Service ID must not be empty"));
    }
    // TODO: check if the reserved namespace should be part of Chord.
    if id.starts_with("$chord.") {
        return Err(type_error(
            "Service IDs beginning with $chord. are reserved",
        ));
    }
    Ok(Service::new(id, local))
}

/// Build one consumer-side remote service binding (upstream
/// `createRemoteServiceBinding`). The upstream constructor cannot throw; the
/// Rust binding validates its allowlist, so construction is fallible.
pub fn create_remote_service_binding(
    options: RemoteServiceBindingOptions,
) -> Result<Arc<RemoteServiceBinding>, AnyError> {
    Ok(Arc::new(RemoteServiceBinding::new(options)?))
}

/// Create authoritative state by taking immutable ownership of a strict-JSON
/// root (upstream `replicatedState(initial)`).
pub fn replicated_state(initial: JsonValue) -> Arc<MutableReplicatedState> {
    MutableReplicatedState::new(initial)
}

/// Attach a publication-only replicated state to one authoritative source
/// stream (upstream `replicatedState(source, options)`).
pub fn attach_replicated_state(
    source: Arc<dyn ReplicatedStateSource>,
    options: ReplicatedStateSourceOptions,
) -> Result<Arc<AttachedReplicatedState>, AnyError> {
    attach_replicated_state_source(source, options)
}

#[cfg(test)]
mod tests {
    //! Port of upstream `test/facet-loader.test.ts`.
    //!
    //! Non-portable cases and their reasons:
    //! - "terminates the host when replacement publication fails after cutover":
    //!   upstream injects failure through a provider-subscription listener that
    //!   throws; the Rust `UpdateListener` returns `()`, so the publication
    //!   error path is unreachable from tests.
    //! - consumer-handle identity (`toBe` on `use_service` results) ports to
    //!   `Arc::ptr_eq`; extracted JS member methods collapse into late-binding
    //!   requirement calls.

    use super::*;
    use crate::context::{Context, BACKGROUND_CONTEXT};
    use crate::facets::host::FacetEnvironment;
    use crate::services::errors::error;
    use crate::services::loopback::create_loopback_service_transport;
    use crate::types::{
        FacetOptions, RemoteServiceImpl, RemoteServices, ServiceMember, ServiceMode,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, LazyLock};
    use {parking_lot::Mutex};

    type Trace = Arc<Mutex<Vec<String>>>;

    struct GenerationValue {
        value: String,
    }

    impl GenerationValue {
        fn read(&self) -> &str {
            &self.value
        }
    }

    struct RemoteGenerationSvc;

    static LOCAL_GENERATION_VALUE: LazyLock<Service<GenerationValue>> = LazyLock::new(|| {
        define_service("test.experimental.local-generation-value", true).expect("define service")
    });
    static REMOTE_GENERATION_VALUE: LazyLock<Service<RemoteGenerationSvc>> = LazyLock::new(|| {
        define_service("test.experimental.remote-generation-value", false).expect("define service")
    });

    fn first_facet() -> Facet {
        Facet::new("first", |_env: &mut FacetEnvironment| Ok(()))
    }

    fn second_facet() -> Facet {
        Facet::new("second", |_env: &mut FacetEnvironment| Ok(()))
    }

    fn push(trace: &Trace, entry: &str) {
        trace.lock().push(entry.to_string());
    }

    fn snapshot(trace: &Trace) -> Vec<String> {
        trace.lock().clone()
    }

    fn trace() -> Trace {
        Arc::new(Mutex::new(Vec::new()))
    }

    /// Minimal [`RemoteServiceImpl`] standing in for upstream plain-object
    /// implementations.
    struct TestRemoteObject {
        methods: Vec<(&'static str, crate::types::RemoteMethod)>,
    }

    impl RemoteServiceImpl for TestRemoteObject {
        fn members(&self) -> Vec<(String, ServiceMember)> {
            self.methods
                .iter()
                .map(|(name, method)| ((*name).to_string(), ServiceMember::Method(method.clone())))
                .collect()
        }
    }

    fn read_impl(value: impl Into<String>) -> Arc<dyn RemoteServiceImpl> {
        let value = Arc::new(value.into());
        Arc::new(TestRemoteObject {
            methods: vec![(
                "read",
                Arc::new(move |_args: Vec<JsonValue>, _context: Context| {
                    let value = value.clone();
                    Box::pin(async move { Ok(Some(json!(&*value))) }) as BoxFuture<_>
                }),
            )],
        })
    }

    /// Loader that traces loads and disposals; optionally fails to load.
    struct ScriptedLoader {
        load_trace: &'static str,
        dispose_trace: &'static str,
        fail: bool,
        facets: Vec<Facet>,
        trace: Trace,
        dispose_called: Arc<AtomicBool>,
    }

    impl FacetLoader for ScriptedLoader {
        fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>> {
            push(&self.trace, self.load_trace);
            if self.fail {
                return Box::pin(async { Err(error("load failed")) });
            }
            let facets = self.facets.clone();
            let trace = self.trace.clone();
            let dispose_trace = self.dispose_trace;
            let dispose_called = self.dispose_called.clone();
            Box::pin(async move {
                Ok(LoadedFacets {
                    facets,
                    dispose: Some(Box::new(move || {
                        dispose_called.store(true, Ordering::SeqCst);
                        push(&trace, dispose_trace);
                        Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
                    })),
                })
            })
        }
    }

    fn scripted_loader(
        load_trace: &'static str,
        dispose_trace: &'static str,
        facets: Vec<Facet>,
        trace: &Trace,
        fail: bool,
    ) -> Arc<ScriptedLoader> {
        Arc::new(ScriptedLoader {
            load_trace,
            dispose_trace,
            fail,
            facets,
            trace: trace.clone(),
            dispose_called: Arc::new(AtomicBool::new(false)),
        })
    }

    #[tokio::test]
    async fn combines_loaded_facets_in_loader_order_and_disposes_generations_in_reverse() {
        let trace = trace();
        let first = scripted_loader(
            "load first",
            "dispose first",
            vec![first_facet()],
            &trace,
            false,
        );
        let second = scripted_loader(
            "load second",
            "dispose second",
            vec![second_facet()],
            &trace,
            false,
        );

        let mut loaded = combine_facet_loaders(vec![first, second])
            .load()
            .await
            .expect("combined load");
        assert_eq!(
            loaded
                .facets
                .iter()
                .map(|facet| facet.id().to_string())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        loaded.dispose().await.expect("dispose");
        loaded.dispose().await.expect("dispose again");
        assert_eq!(
            snapshot(&trace),
            [
                "load first",
                "load second",
                "dispose second",
                "dispose first"
            ]
        );
    }

    #[tokio::test]
    async fn cleans_up_loaded_facets_when_a_later_loader_fails() {
        let trace = trace();
        let first = scripted_loader(
            "load first",
            "dispose first",
            vec![first_facet()],
            &trace,
            false,
        );
        let second = {
            let loader = ScriptedLoader {
                load_trace: "load second",
                dispose_trace: "dispose second",
                fail: true,
                facets: Vec::new(),
                trace: trace.clone(),
                dispose_called: Arc::new(AtomicBool::new(false)),
            };
            assert!(!loader.dispose_called.load(Ordering::SeqCst));
            Arc::new(loader)
        };

        let error = match combine_facet_loaders(vec![first.clone(), second])
            .load()
            .await
        {
            Ok(_) => panic!("second loader should fail"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), "load failed");
        assert!(first.dispose_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn keeps_local_and_rpc_service_handles_stable_when_their_provider_facet_reloads() {
        let trace = trace();
        let local_value: Arc<
            Mutex<Option<crate::facets::host::LocalRequirement<GenerationValue>>>,
        > = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let (continue_tx, continue_rx) = tokio::sync::oneshot::channel::<()>();
        let (reload_done_tx, reload_done_rx) =
            tokio::sync::oneshot::channel::<Result<(), AnyError>>();

        let consumer_local = local_value.clone();
        let consumer_trace = trace.clone();
        let consumer = Facet::new("consumer", move |env: &mut FacetEnvironment| {
            push(&consumer_trace, "setup consumer");
            let requirement = env.use_local(&*LOCAL_GENERATION_VALUE)?;
            *consumer_local.lock() = Some(requirement.clone());
            let activation_trace = consumer_trace.clone();
            let activation_local = requirement.clone();
            env.on_activate(Arc::new(move || {
                let activation_trace = activation_trace.clone();
                let activation_local = activation_local.clone();
                Box::pin(async move {
                    let value = activation_local
                        .with(|value: &GenerationValue| value.read().to_string())
                        .expect("local read on activate");
                    push(&activation_trace, &format!("activate consumer:{value}"));
                    Ok(())
                })
            }))?;
            env.on_deactivate(trace_disposal(&consumer_trace, "deactivate consumer"))?;
            Ok(())
        });

        let loader: Arc<SwitchingLoader> = Arc::new(SwitchingLoader {
            generation: generation.clone(),
            trace: trace.clone(),
            started_tx: Arc::new(Mutex::new(Some(started_tx))),
            continue_rx: Arc::new(Mutex::new(Some(continue_rx))),
        });

        let loaded_a = loader.load().await.expect("load A");
        let kernel = Arc::new(tokio::sync::Mutex::new(
            FacetKernel::new(FacetOptions {
                facets: [vec![consumer], loaded_a.facets.clone()].concat(),
                service_sources: Vec::new(),
                on_error: None,
            })
            .expect("kernel"),
        ));
        kernel.lock().await.activate().await.expect("activate");
        let provider = kernel.lock().await.provider().expect("provider");
        let original_local = local_value
            .lock()
            .clone()
            .unwrap();

        let remote_services = create_remote_service_binding(RemoteServiceBindingOptions {
            services: vec![REMOTE_GENERATION_VALUE.id().to_string()],
            transport: create_loopback_service_transport(provider),
            bound: true,
            on_error: None,
            assert_access: None,
        })
        .expect("binding");
        let original_remote = remote_services
            .use_service(REMOTE_GENERATION_VALUE.id())
            .expect("remote use");
        remote_services
            .ready(&BACKGROUND_CONTEXT)
            .await
            .expect("remote ready");
        let read_local = || original_local.with(|value: &GenerationValue| value.read().to_string());
        assert_eq!(read_local().expect("local read A"), "A");
        let value = original_remote
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("remote read A");
        assert_eq!(value, Some(json!("A")));

        // Shape-violating replacement is rejected before cutover.
        let invalid = Facet::new("provider", |env: &mut FacetEnvironment| {
            env.provide_local(
                &*LOCAL_GENERATION_VALUE,
                Arc::new(GenerationValue {
                    value: "invalid".to_string(),
                }),
            )?;
            Ok(())
        });
        let error = kernel
            .lock()
            .await
            .reload(vec![invalid])
            .await
            .expect_err("shape violation rejected");
        assert!(
            error.to_string().contains(
                "Reloaded facet provider must preserve its service requirements and provisions"
            ),
            "{error}"
        );
        assert_eq!(read_local().expect("local read still A"), "A");
        let value = original_remote
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("remote read still A");
        assert_eq!(value, Some(json!("A")));

        let loaded_b = loader.load().await.expect("load B");
        {
            let kernel = kernel.clone();
            let loaded_b_facets = loaded_b.facets.clone();
            tokio::spawn(async move {
                let result = kernel.lock().await.reload(loaded_b_facets).await;
                let _ = reload_done_tx.send(result);
            });
        }
        started_rx.await.expect("replacement started signal");
        // While provider B activation is parked, the old handles still read "A".
        assert_eq!(read_local().expect("local read during replacement"), "A");
        let value = original_remote
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("remote read during replacement");
        assert_eq!(value, Some(json!("A")));

        // Release the parked activation.
        continue_tx.send(()).expect("continue replacement");
        let reload_result = reload_done_rx.await.expect("reload completes");
        reload_result.expect("reload succeeds");
        let mut loaded_a = loaded_a;
        loaded_a.dispose().await.expect("loaded A dispose");

        assert_eq!(read_local().expect("local read B"), "B");
        let replacement_remote = remote_services
            .use_service(REMOTE_GENERATION_VALUE.id())
            .expect("remote use after reload");
        assert!(Arc::ptr_eq(&replacement_remote, &original_remote));
        let value = original_remote
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("remote read B");
        assert_eq!(value, Some(json!("B")));

        remote_services
            .dispose(&BACKGROUND_CONTEXT)
            .await
            .expect("remote dispose");
        kernel.lock().await.dispose().await.expect("kernel dispose");
        let mut loaded_b = loaded_b;
        loaded_b.dispose().await.expect("loaded B dispose");

        assert_eq!(
            snapshot(&trace),
            [
                "load A",
                "setup consumer",
                "setup provider A",
                "activate provider A",
                "activate consumer:A",
                "load B",
                "setup provider B",
                "activate provider B",
                "deactivate provider A",
                "unload A",
                "deactivate consumer",
                "deactivate provider B",
                "unload B"
            ]
        );
    }

    /// Loader producing the two-generation provider facet with a parked
    /// activation for generation B (upstream's promise-gated loader).
    struct SwitchingLoader {
        generation: Arc<AtomicU64>,
        trace: Trace,
        started_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        continue_rx: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    }

    impl FacetLoader for SwitchingLoader {
        fn load(&self) -> BoxFuture<Result<LoadedFacets, AnyError>> {
            let name: &'static str = if self.generation.fetch_add(1, Ordering::SeqCst) == 0 {
                "A"
            } else {
                "B"
            };
            push(&self.trace, &format!("load {name}"));
            let trace = self.trace.clone();
            let trace_for_dispose = trace.clone();
            let started_tx = self.started_tx.clone();
            let continue_rx = self.continue_rx.clone();
            Box::pin(async move {
                let provider = Facet::new("provider", move |env: &mut FacetEnvironment| {
                    push(&trace, &format!("setup provider {name}"));
                    env.provide_local(
                        &*LOCAL_GENERATION_VALUE,
                        Arc::new(GenerationValue {
                            value: name.to_string(),
                        }),
                    )?;
                    env.provide_remote(
                        REMOTE_GENERATION_VALUE.id(),
                        ServiceMode::Singleton,
                        read_impl(name),
                    )?;
                    let activation_trace = trace.clone();
                    let started_tx = started_tx.clone();
                    let continue_rx = continue_rx.clone();
                    env.on_activate(Arc::new(move || {
                        let activation_trace = activation_trace.clone();
                        let started_tx = started_tx.clone();
                        let continue_rx = continue_rx.clone();
                        Box::pin(async move {
                            push(&activation_trace, &format!("activate provider {name}"));
                            if name == "B" {
                                if let Some(started) =
                                    started_tx.lock().take()
                                {
                                    let _ = started.send(());
                                }
                                let continue_signal = {
                                    // The std guard must drop before the await.
                                    continue_rx.lock().take()
                                };
                                if let Some(continue_signal) = continue_signal {
                                    let _ = continue_signal.await;
                                }
                            }
                            Ok(())
                        })
                    }))?;
                    env.on_deactivate(trace_disposal(
                        &trace,
                        &format!("deactivate provider {name}"),
                    ))?;
                    Ok(())
                });
                Ok(LoadedFacets {
                    facets: vec![provider],
                    dispose: Some(Box::new(move || {
                        push(&trace_for_dispose, &format!("unload {name}"));
                        Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
                    })),
                })
            })
        }
    }

    fn trace_disposal(trace: &Trace, entry: &str) -> crate::facets::host::Disposal {
        let trace = trace.clone();
        let entry = entry.to_string();
        Arc::new(move || {
            push(&trace, &entry);
            Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
        })
    }

    #[tokio::test]
    async fn rejects_remote_singleton_member_shape_changes_before_reload_cutover() {
        let retained: Arc<Mutex<Option<crate::facets::host::RemoteRequirement>>> =
            Arc::new(Mutex::new(None));
        let provider_disposed = Arc::new(AtomicBool::new(false));
        let retained_for_consumer = retained.clone();
        let consumer = Facet::new("shape-consumer", move |env: &mut FacetEnvironment| {
            let requirement = env.use_remote(REMOTE_GENERATION_VALUE.id())?;
            *retained_for_consumer.lock() = Some(requirement);
            Ok(())
        });
        let provider = {
            let disposed_for_provider = provider_disposed.clone();
            Facet::new("shape-provider", move |env: &mut FacetEnvironment| {
                env.provide_remote(
                    REMOTE_GENERATION_VALUE.id(),
                    ServiceMode::Singleton,
                    read_impl("A"),
                )?;
                let disposed = disposed_for_provider.clone();
                env.on_deactivate(Arc::new(move || {
                    disposed.store(true, Ordering::SeqCst);
                    Box::pin(async { Ok(()) }) as BoxFuture<Result<(), AnyError>>
                }))?;
                Ok(())
            })
        };
        let host = create_facet_host(FacetOptions {
            facets: vec![consumer, provider],
            service_sources: Vec::new(),
            on_error: None,
        })
        .await
        .expect("host activates");

        let renamed = Facet::new("shape-provider", |env: &mut FacetEnvironment| {
            let implementation: Arc<dyn RemoteServiceImpl> = Arc::new(TestRemoteObject {
                methods: vec![(
                    "renamed",
                    Arc::new(|_args: Vec<JsonValue>, _context: Context| {
                        Box::pin(async { Ok(Some(json!("B"))) }) as BoxFuture<_>
                    }),
                )],
            });
            env.provide_remote(
                REMOTE_GENERATION_VALUE.id(),
                ServiceMode::Singleton,
                implementation,
            )?;
            Ok(())
        });
        let error = host
            .reload(vec![renamed])
            .await
            .expect_err("shape change rejected");
        assert!(
            error
                .to_string()
                .contains("replacement must preserve its member shape"),
            "{error}"
        );
        assert!(!provider_disposed.load(Ordering::SeqCst));
        let retained_requirement = retained.lock().clone().unwrap();
        let client = retained_requirement.client().expect("retained client");
        let value = client
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("retained read");
        assert_eq!(value, Some(json!("A")));

        host.dispose().await.expect("dispose");
        assert!(provider_disposed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn terminates_the_host_when_old_cleanup_fails_after_cutover() {
        let provider = |name: &'static str, fail_cleanup: bool| {
            Facet::new("cleanup-provider", move |env: &mut FacetEnvironment| {
                env.provide_remote(
                    REMOTE_GENERATION_VALUE.id(),
                    ServiceMode::Singleton,
                    read_impl(name),
                )?;
                env.on_deactivate(Arc::new(move || {
                    Box::pin(async move {
                        if fail_cleanup {
                            return Err(error("cleanup failed"));
                        }
                        Ok(())
                    })
                }))?;
                Ok(())
            })
        };
        let host = create_facet_host(FacetOptions {
            facets: vec![provider("A", true)],
            service_sources: Vec::new(),
            on_error: None,
        })
        .await
        .expect("host activates");

        let error = host
            .reload(vec![provider("B", false)])
            .await
            .expect_err("reload fails after cutover");
        assert!(
            error
                .to_string()
                .contains("Facet reload failed after cutover"),
            "{error}"
        );
        let error = host.reload(Vec::new()).await.expect_err("dead host");
        assert_eq!(error.to_string(), "Facet host cannot reload while dead");
        host.dispose()
            .await
            .expect("dispose is idempotent when dead");
    }

    #[tokio::test]
    async fn cleans_failed_candidate_activation_in_reverse_dependency_order() {
        let trace = trace();
        let provider = |name: &'static str| {
            let trace = trace.clone();
            Facet::new("ordered-provider", move |env: &mut FacetEnvironment| {
                env.provide_remote(
                    REMOTE_GENERATION_VALUE.id(),
                    ServiceMode::Singleton,
                    read_impl(name),
                )?;
                env.on_activate(trace_disposal(&trace, &format!("activate provider {name}")))?;
                env.on_deactivate(trace_disposal(
                    &trace,
                    &format!("deactivate provider {name}"),
                ))?;
                Ok(())
            })
        };
        let consumer = |name: &'static str, fail: bool| {
            let trace = trace.clone();
            Facet::new("ordered-consumer", move |env: &mut FacetEnvironment| {
                env.use_remote(REMOTE_GENERATION_VALUE.id())?;
                let activation_trace = trace.clone();
                env.on_activate(Arc::new(move || {
                    let activation_trace = activation_trace.clone();
                    Box::pin(async move {
                        push(&activation_trace, &format!("activate consumer {name}"));
                        if fail {
                            return Err(error("consumer activation failed"));
                        }
                        Ok(())
                    })
                }))?;
                env.on_deactivate(trace_disposal(
                    &trace,
                    &format!("deactivate consumer {name}"),
                ))?;
                Ok(())
            })
        };

        let host = create_facet_host(FacetOptions {
            facets: vec![consumer("A", false), provider("A")],
            service_sources: Vec::new(),
            on_error: None,
        })
        .await
        .expect("host activates");
        // Clear the activation trace for the replacement generation.
        trace.lock().clear();

        let error = host
            .reload(vec![consumer("B", true), provider("B")])
            .await
            .expect_err("candidate activation fails");
        assert_eq!(error.to_string(), "consumer activation failed");
        assert_eq!(
            snapshot(&trace),
            [
                "activate provider B",
                "activate consumer B",
                "deactivate consumer B",
                "deactivate provider B"
            ]
        );
        host.dispose().await.expect("dispose");
    }

    #[tokio::test]
    async fn keeps_the_old_generation_active_when_replacement_activation_fails_before_cutover() {
        let trace = trace();
        let retained: Arc<Mutex<Option<crate::facets::host::RemoteRequirement>>> =
            Arc::new(Mutex::new(None));
        let retained_for_consumer = retained.clone();
        let consumer = {
            let trace = trace.clone();
            let retained = retained_for_consumer.clone();
            Facet::new("terminal-consumer", move |env: &mut FacetEnvironment| {
                let requirement = env.use_remote(REMOTE_GENERATION_VALUE.id())?;
                *retained.lock() = Some(requirement);
                env.on_deactivate(trace_disposal(&trace, "deactivate consumer"))?;
                Ok(())
            })
        };
        let provider = |name: &'static str, fail: bool| {
            let trace = trace.clone();
            Facet::new("terminal-provider", move |env: &mut FacetEnvironment| {
                env.provide_remote(
                    REMOTE_GENERATION_VALUE.id(),
                    ServiceMode::Singleton,
                    read_impl(name),
                )?;
                let activation_trace = trace.clone();
                env.on_activate(Arc::new(move || {
                    let activation_trace = activation_trace.clone();
                    Box::pin(async move {
                        push(&activation_trace, &format!("activate {name}"));
                        if fail {
                            return Err(error("replacement activation failed"));
                        }
                        Ok(())
                    })
                }))?;
                env.on_deactivate(trace_disposal(&trace, &format!("deactivate {name}")))?;
                Ok(())
            })
        };

        let host = create_facet_host(FacetOptions {
            facets: vec![consumer, provider("A", false)],
            service_sources: Vec::new(),
            on_error: None,
        })
        .await
        .expect("host activates");
        let retained_requirement = retained.lock().clone().unwrap();
        let client = retained_requirement.client().expect("retained client");
        let value = client
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("retained read");
        assert_eq!(value, Some(json!("A")));

        let error = host
            .reload(vec![provider("B", true)])
            .await
            .expect_err("replacement activation fails");
        assert_eq!(error.to_string(), "replacement activation failed");
        assert_eq!(
            snapshot(&trace),
            ["activate A", "activate B", "deactivate B"]
        );
        let client = retained_requirement.client().expect("retained client");
        let value = client
            .invoke("read", Vec::new(), &BACKGROUND_CONTEXT)
            .await
            .expect("retained read after failure");
        assert_eq!(value, Some(json!("A")));
        host.reload(Vec::new()).await.expect("empty reload");
        host.dispose().await.expect("dispose");
        assert_eq!(
            snapshot(&trace),
            [
                "activate A",
                "activate B",
                "deactivate B",
                "deactivate consumer",
                "deactivate A"
            ]
        );
    }

    #[tokio::test]
    async fn creates_a_reusable_static_loader() {
        let loader = create_static_facet_loader(vec![first_facet()]);
        let mut first = loader.load().await.expect("first load");
        let mut second = loader.load().await.expect("second load");
        assert_eq!(
            first
                .facets
                .iter()
                .map(|facet| facet.id().to_string())
                .collect::<Vec<_>>(),
            ["first"]
        );
        assert_eq!(
            second
                .facets
                .iter()
                .map(|facet| facet.id().to_string())
                .collect::<Vec<_>>(),
            ["first"]
        );
        first.dispose().await.expect("first dispose");
        second.dispose().await.expect("second dispose");
    }

    #[test]
    fn define_service_rejects_empty_and_reserved_ids() {
        let error = define_service::<RemoteGenerationSvc>("", false).unwrap_err();
        assert_eq!(error.to_string(), "Service ID must not be empty");
        let error = define_service::<RemoteGenerationSvc>("$chord.service", false).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Service IDs beginning with $chord. are reserved"
        );
    }
}
