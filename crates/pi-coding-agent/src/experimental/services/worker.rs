//! Port of `experimental/services/worker.ts` — assembles the Session worker's
//! service facets (agent controller, session plugins, models, transcript) into
//! one facet host with scoped remote endpoints.
//!
//! Deviations (D144):
//! - `FacetHost` is an async trait object from chord; one
//!   [`pi_chord::RemoteServiceEndpoint`] per attachment scope fronts the
//!   host's service provider (cloned, chord providers share the same inner
//!   registrations).
//! - Upstream promise-tail serialization (`reloadTail`) maps to a tokio mutex.
//! - `serviceScopeKey` joins the scope with a NUL byte like upstream.
//! - `AggregateError` maps to [`pi_chord::AggregateError`].

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use pi_chord::api::{create_facet_host, create_static_facet_loader};
use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use pi_chord::json::JsonValue;
use pi_chord::services::errors::{error, AggregateError, AnyError};
use pi_chord::services::provider::{
    create_remote_service_endpoint, RemoteServiceEndpoint, RemoteServiceProvider,
    ServiceUpdatePublisher,
};
use pi_chord::types::ServiceCall;
use pi_chord::types::{Facet, FacetOptions, LoadedFacets};
use pi_chord::types::{FacetHost, FacetLoader};
use pi_durable::harness::harness::{Conversation, Harness};

use crate::core::model_runtime::ModelRuntime;
use crate::experimental::services::agent_controller::AGENT_CONTROLLER;
use crate::experimental::services::agent_controller_provider::{
    agent_controller_object, AgentControllerService,
};
use crate::experimental::services::models_provider::{
    create_models_service_facet, ModelsServiceFacetOptions, SharedSettingsManager,
};
use crate::experimental::services::plugins::{SessionPluginsService, SESSION_PLUGINS};
use crate::experimental::services::transcript_provider::create_transcript_service_facet;

/// Upstream `SessionWorkerRuntime`.
pub struct SessionWorkerRuntime {
    pub harness: Arc<Harness>,
    /// The root conversation the services expose.
    pub conversation: Arc<Conversation>,
    pub model_runtime: Option<Arc<ModelRuntime>>,
    pub settings_manager: Option<SharedSettingsManager>,
    pub facet_loader: Option<Arc<dyn FacetLoader>>,
    /// Release resources the Harness does not own, such as execution
    /// environments, after it closed.
    pub cleanup: Option<
        Arc<
            dyn Fn() -> futures_util::future::BoxFuture<'static, Result<(), AnyError>>
                + Send
                + Sync,
        >,
    >,
}

/// Upstream `WorkerServiceScope`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerServiceScope {
    pub server_connection_id: String,
    pub attachment_id: String,
}

/// Upstream `SessionWorkerServices`.
pub struct SessionWorkerServices {
    host: Arc<dyn FacetHost>,
    endpoints: Mutex<HashMap<String, ScopedEndpoint>>,
    reload: Reload,
}

struct ScopedEndpoint {
    scope: WorkerServiceScope,
    endpoint: Arc<RemoteServiceEndpoint>,
}

struct Reload {
    /// Serializes reloads (upstream `reloadTail`).
    lock: Arc<tokio::sync::Mutex<()>>,
    loader: Arc<dyn FacetLoader>,
    loaded: tokio::sync::Mutex<LoadedFacets>,
}

impl SessionWorkerServices {
    /// Upstream `invoke(call, scope, context)`; `publish` forwards service
    /// updates for the scope's subscriptions.
    pub async fn invoke(
        &self,
        call: &ServiceCall,
        scope: &WorkerServiceScope,
        publish: ServiceUpdatePublisher,
        context: &Context,
    ) -> Result<Option<JsonValue>, AnyError> {
        let endpoint = {
            let mut endpoints = self.endpoints.lock();
            let key = service_scope_key(scope);
            endpoints
                .entry(key)
                .or_insert_with(|| ScopedEndpoint {
                    scope: scope.clone(),
                    endpoint: Arc::new(create_remote_service_endpoint(provider_arc(&self.host))),
                })
                .endpoint
                .clone()
        };
        endpoint.invoke(call, &publish, context).await
    }

    /// Upstream `removeSubscriptions(matches)`.
    pub fn remove_subscriptions(&self, matches: impl Fn(&WorkerServiceScope) -> bool) {
        let mut endpoints = self.endpoints.lock();
        endpoints.retain(|_, entry| {
            if matches(&entry.scope) {
                let endpoint = Arc::clone(&entry.endpoint);
                tokio::spawn(async move {
                    endpoint.dispose().await;
                });
                false
            } else {
                true
            }
        });
    }

    /// The `SessionPlugins.reload()` handoff — reload plugin facets atomically.
    pub async fn reload_plugins(&self) -> Result<(), AnyError> {
        let _guard = self.reload.lock.lock().await;
        let mut candidate = self.reload.loader.load().await?;
        if let Err(reload_error) = self.host.reload(candidate.facets.clone()).await {
            let cleanup_error = candidate.dispose().await.err();
            return match cleanup_error {
                Some(cleanup_error) => Err(AggregateError::new(
                    "Session plugin reload and cleanup failed",
                    vec![reload_error, cleanup_error],
                )),
                None => Err(reload_error),
            };
        }
        let mut loaded = self.reload.loaded.lock().await;
        let retired_result = loaded.dispose().await;
        *loaded = candidate;
        retired_result
    }

    /// Upstream `dispose()`.
    pub async fn dispose(&self) -> Result<(), AnyError> {
        self.remove_subscriptions(|_| true);
        let _guard = self.reload.lock.lock().await;
        let mut errors: Vec<AnyError> = Vec::new();
        if let Some(host_error) = self.host.dispose().await.err() {
            errors.push(host_error);
        }
        if let Err(loaded_error) = self.reload.loaded.lock().await.dispose().await {
            errors.push(loaded_error);
        }
        finish(errors, "Failed to dispose Session facets")
    }
}

/// The live provider behind the host, shared through the host's `Arc` handle
/// (pi-chord addition D147: `FacetHost::provider_arc`).
fn provider_arc(host: &Arc<dyn FacetHost>) -> Arc<RemoteServiceProvider> {
    host.provider_arc().expect("facet host shares its provider")
}

fn service_scope_key(scope: &WorkerServiceScope) -> String {
    format!("{}\0{}", scope.server_connection_id, scope.attachment_id)
}

fn finish(errors: Vec<AnyError>, message: &str) -> Result<(), AnyError> {
    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.into_iter().next().expect("one error")),
        _ => Err(AggregateError::new(message, errors)),
    }
}

/// Upstream `createSessionWorkerServices` options.
pub struct CreateSessionWorkerServicesOptions {
    pub harness: Arc<Harness>,
    pub conversation: Arc<Conversation>,
    pub model_runtime: Option<Arc<ModelRuntime>>,
    pub settings_manager: Option<SharedSettingsManager>,
    pub facet_loader: Option<Arc<dyn FacetLoader>>,
    pub publish: Arc<
        dyn Fn(
                WorkerServiceScope,
                String,
                pi_chord::ServiceProviderUpdate,
            ) -> futures_util::future::BoxFuture<'static, ()>
            + Send
            + Sync,
    >,
}

/// `createSessionWorkerServices(options)`.
pub async fn create_session_worker_services(
    options: CreateSessionWorkerServicesOptions,
) -> Result<Arc<SessionWorkerServices>, AnyError> {
    let options = Arc::new(options);

    // `@pi/agent-controller-runtime`: provide the AgentController facade.
    let controller = Arc::new(AgentControllerService::new(
        Arc::clone(&options.harness),
        Arc::clone(&options.conversation),
    ));
    let agent_controller_runtime_facet = agent_controller_runtime_facet(controller);

    // `@pi/session-plugins-runtime`: the reload closure delegates to the
    // assembled services once available.
    let reload_cell: Arc<tokio::sync::Mutex<Option<Arc<SessionWorkerServices>>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let session_plugins_runtime_facet = session_plugins_runtime_facet(Arc::clone(&reload_cell));

    let models_facet = create_models_service_facet(ModelsServiceFacetOptions {
        harness: Arc::clone(&options.harness),
        conversation: Arc::clone(&options.conversation),
        model_runtime: options.model_runtime.clone(),
        settings_manager: options.settings_manager.clone(),
        context: BACKGROUND_CONTEXT.clone(),
    })
    .await
    .map_err(|facet_error| error(facet_error))?;
    let transcript_facet =
        create_transcript_service_facet(&options.conversation, &BACKGROUND_CONTEXT)
            .await
            .map_err(|facet_error| error(facet_error))?;

    let mut builtins = create_static_facet_loader(vec![
        agent_controller_runtime_facet,
        session_plugins_runtime_facet,
        models_facet,
        transcript_facet,
    ])
    .load()
    .await?;
    let plugin_loader: Arc<dyn FacetLoader> = match &options.facet_loader {
        Some(loader) => Arc::clone(loader),
        None => create_static_facet_loader(Vec::new()),
    };
    let mut loaded_plugins = plugin_loader.load().await?;

    let mut facets = builtins.facets.clone();
    facets.extend(loaded_plugins.facets.clone());
    let facet_host = match create_facet_host(FacetOptions {
        facets,
        service_sources: Vec::new(),
        on_error: None,
    })
    .await
    {
        Ok(host) => host,
        // Upstream: clean up both generations when the host fails to start.
        Err(start_error) => {
            let (plugin_cleanup, builtin_cleanup) =
                futures_util::future::join(loaded_plugins.dispose(), builtins.dispose()).await;
            let mut errors = vec![start_error];
            if let Err(plugin_cleanup) = plugin_cleanup {
                errors.push(plugin_cleanup);
            }
            if let Err(builtin_cleanup) = builtin_cleanup {
                errors.push(builtin_cleanup);
            }
            return match finish(errors, "Session facets failed to start and clean up") {
                Ok(()) => unreachable!("finish reports the collected errors"),
                Err(assembled_error) => Err(assembled_error),
            };
        }
    };

    let services = Arc::new(SessionWorkerServices {
        host: Arc::clone(&facet_host),
        endpoints: Mutex::new(HashMap::new()),
        reload: Reload {
            lock: Arc::new(tokio::sync::Mutex::new(())),
            loader: plugin_loader,
            loaded: tokio::sync::Mutex::new(LoadedFacets::new(Vec::new())),
        },
    });
    *services.reload.loaded.lock().await = loaded_plugins;
    *reload_cell.lock().await = Some(Arc::clone(&services));
    Ok(services)
}

/// `@pi/agent-controller-runtime` — provide the AgentController remote facade.
fn agent_controller_runtime_facet(controller: Arc<AgentControllerService>) -> Facet {
    use pi_chord::types::ServiceMode;
    let object = agent_controller_object(controller);
    pi_chord::api::define_facet(pi_chord::types::Facet::new(
        "@pi/agent-controller-runtime",
        move |env| {
            env.provide_remote(
                AGENT_CONTROLLER.id(),
                ServiceMode::Singleton,
                object.clone(),
            )?;
            Ok(())
        },
    ))
}

/// `@pi/session-plugins-runtime` — provide SessionPlugins; reloads delegate to
/// the assembled services once available (upstream `reloadPlugins` indirection).
fn session_plugins_runtime_facet(
    reload_cell: Arc<tokio::sync::Mutex<Option<Arc<SessionWorkerServices>>>>,
) -> Facet {
    pi_chord::api::define_facet(pi_chord::types::Facet::new(
        "@pi/session-plugins-runtime",
        move |env| {
            let cell = Arc::clone(&reload_cell);
            env.provide_local(
                &SESSION_PLUGINS,
                Arc::new(SessionPluginsService::new(move |_context| {
                    let cell = Arc::clone(&cell);
                    Box::pin(async move {
                        let services = cell
                            .lock()
                            .await
                            .clone()
                            .ok_or_else(|| "Session plugins are not ready".to_string())?;
                        services
                            .reload_plugins()
                            .await
                            .map_err(|reload_error| reload_error.to_string())
                    })
                })),
            )?;
            Ok(())
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_key_joins_with_nul() {
        let scope = WorkerServiceScope {
            server_connection_id: "conn".into(),
            attachment_id: "att".into(),
        };
        assert_eq!(service_scope_key(&scope), "conn\0att");
    }

    #[tokio::test]
    async fn reload_before_assembly_reports_not_ready() {
        let cell: Arc<tokio::sync::Mutex<Option<Arc<SessionWorkerServices>>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let facet = session_plugins_runtime_facet(Arc::clone(&cell));
        assert_eq!(facet.id(), "@pi/session-plugins-runtime");
    }
}
