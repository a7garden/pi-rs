//! Port of `experimental/services/server.ts` — the experimental server-side
//! service host: session directory, session management, and presentation
//! plugin services routed per client.
//!
//! Deviations (D144):
//! - `RemoteServiceProvider` singletons are provided as
//!   [`pi_chord::RemoteServiceObject`]s with JSON members; `SessionDirectory`
//!   exposes its replicated state through the `state` member.
//! - `RoutedServerServiceHost`/`RoutedServerServiceAttachment` come from
//!   [`pi_server::types`]; `attachClient` builds one provider + endpoint per
//!   presentation and returns the routed attachment. The chord publisher
//!   callback is adapted to pi-server's [`ServicePublish`] shape.
//! - Upstream `mutationTail` promise serialization maps to a tokio mutex over
//!   the serialized mutation section; directory revisions advance under it.
//! - Upstream's `refresh()`/`dispose()` object methods map to inherent async
//!   methods on [`ExperimentalServerServices`].

use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_chord::context::Context;
use pi_chord::json::JsonValue;
use pi_chord::services::errors::{error, AnyError};
use pi_chord::services::provider::{
	create_remote_service_endpoint, ProviderDefinition, RemoteServiceEndpoint, RemoteServiceObject,
	RemoteServiceProvider,
};
use pi_chord::services::state::MutableReplicatedState;
use pi_chord::types::{ServiceCall, ServiceMode};
use pi_server::types::{
	MaybeFuture, RoutedServerPresentation, RoutedServerServiceAttachment, RoutedServerServiceHost,
	ServicePublish,
};
use serde::{Deserialize, Serialize};

use super::plugins::{members as plugin_members, PrepareSessionRequest, PRESENTATION_PLUGINS};
use super::sessions::{SessionCreateOptions, SessionDirectoryState, SessionSummary, SESSION_DIRECTORY, SESSION_MANAGEMENT};

/// pi-server's shared error type (`Arc<dyn Error + Send + Sync>`, unlike
/// chord's boxed [`pi_chord::services::errors::AnyError`]).
type SrvError = pi_server::types::AnyError;

/// Convert a chord error into pi-server's `Arc`-shared error.
fn to_srv_error(err: AnyError) -> SrvError {
	Arc::from(err)
}

/// Upstream `ExperimentalServerServices`.
pub struct ExperimentalServerServices {
	pub host: Arc<dyn RoutedServerServiceHost>,
	state: Arc<ServerState>,
}

impl ExperimentalServerServices {
	/// Upstream `refresh(context?)`.
	pub async fn refresh(&self, context: Context) -> Result<(), AnyError> {
		let state = &self.state;
		let _guard = state.mutation_lock.lock().await;
		let sessions = (state.options.list)(context.clone()).await?;
		let revision = read_revision(&state.directory)? + 1;
		state.directory.replace(
			&context,
			serde_json::to_value(SessionDirectoryState { revision, sessions })
				.expect("SessionDirectoryState serializes"),
		)?;
		Ok(())
	}

	/// Upstream `dispose()`.
	pub async fn dispose(&self) -> Result<(), AnyError> {
		let attachments: Vec<Arc<ReleasedAttachment>> = self
			.state
			.attachments
			.lock()
			.drain(..)
			.filter_map(|weak| weak.0.upgrade())
			.collect();
		for attachment in &attachments {
			attachment.begin_release();
		}
		// Hold the mutation lock until in-flight mutations settle.
		let _guard = self.state.mutation_lock.lock().await;
		for attachment in attachments {
			attachment.finish_release().await;
		}
		Ok(())
	}
}

/// Upstream `createExperimentalServerServices` options.
pub struct ExperimentalServerServicesOptions {
	pub list: Arc<
		dyn Fn(Context) -> futures_util::future::BoxFuture<'static, Result<Vec<SessionSummary>, AnyError>>
			+ Send
			+ Sync,
	>,
	pub create: Arc<
		dyn Fn(
				SessionCreateOptions,
				Context,
			) -> futures_util::future::BoxFuture<'static, Result<SessionSummary, AnyError>>
			+ Send
			+ Sync,
	>,
	pub remove: Arc<
		dyn Fn(String, Context) -> futures_util::future::BoxFuture<'static, Result<(), AnyError>> + Send + Sync,
	>,
	pub prepare_session_plugins: Arc<
		dyn Fn(
				String,
				Option<Vec<String>>,
				Context,
			) -> futures_util::future::BoxFuture<'static, Result<PreparedSessionPlugins, AnyError>>
			+ Send
			+ Sync,
	>,
	pub reload_presentation_plugins: Arc<
		dyn Fn(Vec<String>, Context) -> futures_util::future::BoxFuture<'static, Result<JsonValue, AnyError>>
			+ Send
			+ Sync,
	>,
}

/// Upstream return of `options.prepareSessionPlugins`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedSessionPlugins {
	pub package_paths: Vec<String>,
	pub presentation_plugins: JsonValue,
}

struct ServerState {
	directory: Arc<MutableReplicatedState>,
	options: Arc<ExperimentalServerServicesOptions>,
	/// Serializes mutations (upstream `mutationTail`).
	mutation_lock: Arc<tokio::sync::Mutex<()>>,
	attachments: Mutex<Vec<WeakAttachment>>,
}

/// Weak handle so released attachments drop out of the set (upstream
/// `attachments.delete(attachment)` in `onRelease`).
struct WeakAttachment(Weak<ReleasedAttachment>);

struct ReleasedAttachment {
	endpoint: RemoteServiceEndpoint,
	provider: Arc<RemoteServiceProvider>,
	released: Mutex<bool>,
}

impl ReleasedAttachment {
	fn begin_release(&self) -> bool {
		let mut released = self.released.lock();
		let previous = *released;
		*released = true;
		previous
	}

	async fn finish_release(&self) {
		self.endpoint.dispose().await;
		let _ = self.provider.dispose();
	}
}

/// `createExperimentalServerServices(options)`.
pub async fn create_experimental_server_services(
	options: ExperimentalServerServicesOptions,
) -> Result<ExperimentalServerServices, AnyError> {
	let options = Arc::new(options);
	let sessions = (options.list)(BACKGROUND_CONTEXT.clone()).await?;
	let directory = pi_chord::api::replicated_state(
		serde_json::to_value(SessionDirectoryState { revision: 1, sessions })
			.expect("SessionDirectoryState serializes"),
	);
	let state = Arc::new(ServerState {
		directory,
		options,
		mutation_lock: Arc::new(tokio::sync::Mutex::new(())),
		attachments: Mutex::new(Vec::new()),
	});
	Ok(ExperimentalServerServices {
		host: Arc::new(ServerHost(Arc::clone(&state))),
		state,
	})
}

struct ServerHost(Arc<ServerState>);

/// Serialize one mutation through the host's mutation lock (upstream
/// `serialize(operation)`).
async fn serialize<T>(state: &ServerState, operation: impl std::future::Future<Output = T>) -> T {
	let _guard = state.mutation_lock.lock().await;
	operation.await
}

async fn refresh_revision(state: &ServerState, context: &Context) -> Result<(), AnyError> {
	let sessions = (state.options.list)(context.clone()).await?;
	let revision = read_revision(&state.directory)? + 1;
	state.directory.replace(
		context,
		serde_json::to_value(SessionDirectoryState { revision, sessions })
			.expect("SessionDirectoryState serializes"),
	)?;
	Ok(())
}

fn read_revision(directory: &MutableReplicatedState) -> Result<u64, AnyError> {
	let state: SessionDirectoryState = serde_json::from_value(directory.value().clone())
		.map_err(|serde_error| error(serde_error.to_string()))?;
	Ok(state.revision)
}

/// Adapt a pi-server publisher to the chord update-listener shape.
fn adapt_publish(publish: ServicePublish) -> pi_chord::services::provider::ServiceUpdatePublisher {
	Arc::new(move |subscription_id: String, update: &pi_chord::ServiceProviderUpdate, context: &Context| {
		let publish = Arc::clone(&publish);
		let update = update.clone();
		let context = context.clone();
		Box::pin(async move {
			if let Err(publish_error) = publish(subscription_id, update, context).await {
				let _ = publish_error;
			}
		})
	})
}

#[async_trait::async_trait]
impl RoutedServerServiceHost for ServerHost {
	async fn attach_client(
		&self,
		presentation: Arc<dyn RoutedServerPresentation>,
		_context: Context,
	) -> Result<Arc<dyn RoutedServerServiceAttachment>, SrvError> {
		attach_client_inner(Arc::clone(&self.0), presentation)
			.await
			.map_err(to_srv_error)
	}
}

async fn attach_client_inner(
	state: Arc<ServerState>,
	presentation: Arc<dyn RoutedServerPresentation>,
) -> Result<Arc<dyn RoutedServerServiceAttachment>, AnyError> {
	{
		let provider = Arc::new(RemoteServiceProvider::new(vec![
			ProviderDefinition::new(SESSION_DIRECTORY.id(), ServiceMode::Singleton),
			ProviderDefinition::new(SESSION_MANAGEMENT.id(), ServiceMode::Singleton),
			ProviderDefinition::new(PRESENTATION_PLUGINS.id(), ServiceMode::Singleton),
		])?);

		provider.provide(
			SESSION_DIRECTORY.id(),
			RemoteServiceObject::builder()
				// The shared authoritative directory state instance.
				.state("state", state.directory.internals())
				.build(),
		)?;

		let prepared: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
		let reload_prepared = Arc::clone(&prepared);
		let reload_state = Arc::clone(&state);

		let prepare_state = Arc::clone(&state);
		let prepare_prepared = Arc::clone(&prepared);
		provider.provide(
			PRESENTATION_PLUGINS.id(),
			RemoteServiceObject::builder()
				.method(
					plugin_members::PREPARE_SESSION,
					move |args, context| {
						let state = Arc::clone(&prepare_state);
						let prepared = Arc::clone(&prepare_prepared);
						Box::pin(async move {
							let request: PrepareSessionRequest = args
								.first()
								.cloned()
								.map(serde_json::from_value)
								.transpose()
								.map_err(|serde_error| error(serde_error.to_string()))?
								.ok_or_else(|| error("prepareSession requires a request"))?;
							let selected = serialize(&state, async {
								(state.options.prepare_session_plugins)(
									request.session_id,
									request.package_paths,
									context.clone(),
								)
								.await
							})
							.await?;
							*prepared.lock() = Some(selected.package_paths.clone());
							Ok(Some(selected.presentation_plugins))
						})
					},
				)
				.method(plugin_members::RELOAD, move |_args, context| {
					let state = Arc::clone(&reload_state);
					let prepared = Arc::clone(&reload_prepared);
					Box::pin(async move {
						let package_paths = serialize(&state, async { prepared.lock().clone() }).await;
						let package_paths = package_paths
							.ok_or_else(|| error("No Session plugin selection is prepared"))?;
						Ok(Some(
							(state.options.reload_presentation_plugins)(package_paths, context.clone())
								.await?,
						))
					})
				})
				.build(),
		)?;

		let create_state = Arc::clone(&state);
		let remove_state = Arc::clone(&state);
		let attach_state = Arc::clone(&state);
		let detach_state = Arc::clone(&state);
		let detach_prepared = Arc::clone(&prepared);
		let presentation_for_remove = Arc::clone(&presentation);
		let presentation_for_attach = Arc::clone(&presentation);
		let presentation_for_detach = Arc::clone(&presentation);
		provider.provide(
			SESSION_MANAGEMENT.id(),
			RemoteServiceObject::builder()
				.method("create", move |args, context| {
					let state = Arc::clone(&create_state);
					Box::pin(async move {
						let create_options: SessionCreateOptions = args
							.first()
							.cloned()
							.map(serde_json::from_value)
							.transpose()
							.map_err(|serde_error| error(serde_error.to_string()))?
							.unwrap_or_default();
						let created = serialize(&state, async {
							let created = (state.options.create)(create_options, context.clone()).await?;
							refresh_revision(&state, &context).await?;
							Ok::<_, AnyError>(created)
						})
						.await?;
						Ok(Some(serde_json::to_value(&created).expect("SessionSummary serializes")))
					})
				})
				.method("remove", move |args, context| {
					let state = Arc::clone(&remove_state);
					let presentation = Arc::clone(&presentation_for_remove);
					Box::pin(async move {
						let session_id = args
							.first()
							.and_then(|value| value.as_str())
							.ok_or_else(|| error("remove requires a session id"))?
							.to_string();
						serialize(&state, async {
							presentation
								.prepare_session_removal(&session_id, context.clone())
								.await?;
							(state.options.remove)(session_id, context.clone()).await?;
							refresh_revision(&state, &context).await?;
							Ok::<_, AnyError>(())
						})
						.await?;
						Ok(None)
					})
				})
				.method("attach", move |args, context| {
					let state = Arc::clone(&attach_state);
					let presentation = Arc::clone(&presentation_for_attach);
					Box::pin(async move {
						let session_id = args
							.first()
							.and_then(|value| value.as_str())
							.ok_or_else(|| error("attach requires a session id"))?
							.to_string();
						serialize(&state, async {
							presentation.attach_session(&session_id, context.clone()).await?;
							Ok::<_, AnyError>(())
						})
						.await?;
						Ok(None)
					})
				})
				.method("detach", move |_args, context| {
					let state = Arc::clone(&detach_state);
					let presentation = Arc::clone(&presentation_for_detach);
					let prepared = Arc::clone(&detach_prepared);
					Box::pin(async move {
						serialize(&state, async {
							presentation.detach_session(context.clone()).await?;
							*prepared.lock() = None;
							Ok::<_, AnyError>(())
						})
						.await?;
						Ok(None)
					})
				})
				.build(),
		)?;

		let endpoint = create_remote_service_endpoint(Arc::clone(&provider));
		let released = Arc::new(ReleasedAttachment {
			endpoint,
			provider,
			released: Mutex::new(false),
		});
		state.attachments.lock().push(WeakAttachment(Arc::downgrade(&released)));

		struct ProviderAttachment(Arc<ReleasedAttachment>);

		#[async_trait::async_trait]
		impl RoutedServerServiceAttachment for ProviderAttachment {
			fn invoke_service(
				&self,
				call: ServiceCall,
				publish: ServicePublish,
				context: Context,
			) -> MaybeFuture<Result<Option<JsonValue>, SrvError>> {
				let attachment = Arc::clone(&self.0);
				Box::pin(async move {
					if *attachment.released.lock() {
						return Err(to_srv_error(error("Server service attachment is released")));
					}
					let result = attachment
						.endpoint
						.invoke(&call, &adapt_publish(publish), &context)
						.await;
					result.map_err(to_srv_error)
				})
			}

			fn release(&self, _context: Context) -> MaybeFuture<Result<(), SrvError>> {
				let attachment = Arc::clone(&self.0);
				Box::pin(async move {
					if attachment.begin_release() {
						return Ok(());
					}
					attachment.finish_release().await;
					Ok(())
				})
			}
		}

		Ok(Arc::new(ProviderAttachment(released)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn prepared_session_plugins_serializes_camel_case() {
		let prepared = PreparedSessionPlugins {
			package_paths: vec!["/p".into()],
			presentation_plugins: serde_json::json!({"k": 1}),
		};
		let value = serde_json::to_value(&prepared).unwrap();
		assert_eq!(value["packagePaths"][0], "/p");
		assert_eq!(value["presentationPlugins"]["k"], 1);
	}
}
