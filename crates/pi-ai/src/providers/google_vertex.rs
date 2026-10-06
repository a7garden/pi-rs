//! Port of `packages/ai/src/providers/google-vertex.ts`.
//!
//! Deviations:
//! - D15: upstream factories return a non-failing `Provider`; Rust
//!   [`crate::models::create_provider`] returns `Result`, unwrapped with
//!   `expect` because the configuration here is static and valid.
//! - D16: upstream imports the API via the `.lazy.ts` twin; the Rust port
//!   links statically (crate deviation D10), so the factory calls
//!   [`crate::api::google_vertex::google_vertex_api`] directly.
//! - D17: `login`'s `interaction.notify({ type: "info", ... })` maps to
//!   [`crate::auth::types::AuthEvent::Info`] unchanged; the prompts map to
//!   [`crate::auth::types::AuthPromptKind`] variants with upstream strings.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;

use crate::auth::types::{
	ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthEvent, AuthInfoLink, AuthPrompt, AuthPromptKind,
	AuthResult, AuthSelectOption, ModelAuth, ProviderAuth, ProviderAuthInteraction,
};
use crate::models::{create_provider, ApiDispatch, CreateProviderOptions, Provider};
use crate::utils::thrown::ThrownValue;

/// Upstream `VERTEX_ADC_PATH`.
const VERTEX_ADC_PATH: &str = "~/.config/gcloud/application_default_credentials.json";

/// Upstream `signal.throwIfAborted()` (`Err(reason)` when aborted).
fn check_aborted(signal: &crate::utils::abort::AbortSignal) -> Result<(), ThrownValue> {
	signal.check().map_err(ThrownValue::error)
}

/// The `login` of [`vertex_auth`].
fn login_vertex(interaction: ProviderAuthInteraction) -> BoxFuture<'static, Result<ApiKeyCredential, ThrownValue>> {
	Box::pin(async move {
		interaction.signal.check().map_err(ThrownValue::error)?;
		let method = (interaction.prompt)(AuthPrompt {
			signal: Some(interaction.signal.clone()),
			kind: AuthPromptKind::Select {
				message: "Select Google Vertex AI authentication method:".to_string(),
				options: vec![
					AuthSelectOption {
						id: "api-key".to_string(),
						label: "Google Cloud API key".to_string(),
						description: None,
					},
					AuthSelectOption {
						id: "adc".to_string(),
						label: "Application Default Credentials".to_string(),
						description: None,
					},
					AuthSelectOption {
						id: "service-account".to_string(),
						label: "Service account credentials file".to_string(),
						description: None,
					},
				],
			},
		})
		.await?;
		interaction.signal.check().map_err(ThrownValue::error)?;
		if method == "api-key" {
			let key = (interaction.prompt)(AuthPrompt {
				signal: Some(interaction.signal.clone()),
				kind: AuthPromptKind::Secret {
					message: "Enter Google Cloud API key".to_string(),
					placeholder: None,
				},
			})
			.await?;
			return Ok(ApiKeyCredential { key: Some(key), env: None });
		}
		if method != "adc" && method != "service-account" {
			return Err(ThrownValue::message(format!("Unknown Google Vertex AI auth method: {method}")));
		}
		(interaction.notify)(AuthEvent::Info {
			message: if method == "adc" {
				"Run `gcloud auth application-default login`, then provide the project and location.".to_string()
			} else {
				"Provide a service account credentials file, project, and location.".to_string()
			},
			links: vec![AuthInfoLink {
				label: Some("Application Default Credentials".to_string()),
				url: "https://cloud.google.com/docs/authentication/provide-credentials-adc".to_string(),
			}],
		});
		let project = (interaction.prompt)(AuthPrompt {
			signal: Some(interaction.signal.clone()),
			kind: AuthPromptKind::Text {
				message: "Enter Google Cloud project ID".to_string(),
				placeholder: None,
			},
		})
		.await?;
		let location = (interaction.prompt)(AuthPrompt {
			signal: Some(interaction.signal.clone()),
			kind: AuthPromptKind::Text {
				message: "Enter Google Cloud location".to_string(),
				placeholder: None,
			},
		})
		.await?;
		let credentials_path = if method == "service-account" {
			Some(
				(interaction.prompt)(AuthPrompt {
					signal: Some(interaction.signal.clone()),
					kind: AuthPromptKind::Text {
						message: "Enter service account credentials file path".to_string(),
						placeholder: None,
					},
				})
				.await?,
			)
		} else {
			None
		};
		let mut env: BTreeMap<String, String> = BTreeMap::new();
		env.insert("GOOGLE_CLOUD_PROJECT".to_string(), project);
		env.insert("GOOGLE_CLOUD_LOCATION".to_string(), location);
		if let Some(credentials_path) = credentials_path {
			env.insert("GOOGLE_APPLICATION_CREDENTIALS".to_string(), credentials_path);
		}
		Ok(ApiKeyCredential { key: None, env: Some(env) })
	})
}

/// The `resolve` entry of [`vertex_auth`].
fn resolve_vertex_entry<'a>(
	input: ApiKeyAuthInput<'a>,
) -> futures_util::future::BoxFuture<'a, Result<Option<AuthResult>, ThrownValue>> {
	Box::pin(async move { resolve_vertex(&input).await })
}

/// Upstream `vertexAuth` (`ApiKeyAuth` object).
fn vertex_auth() -> ApiKeyAuth {
	ApiKeyAuth {
		name: "Google Cloud credentials".to_string(),
		login: Some(Box::new(login_vertex)),
		check: None,
		resolve: Box::new(resolve_vertex_entry),
	}
}

/// The `resolve` body of [`vertex_auth`], kept as a free function so the
/// lifetime-generic closure type stays nameable.
///
/// Vertex accepts an explicit API key or Application Default Credentials
/// (`gcloud auth application-default login`). ADC additionally requires
/// project and location env vars, which the implementation reads itself.
async fn resolve_vertex(input: &ApiKeyAuthInput<'_>) -> Result<Option<AuthResult>, ThrownValue> {
	let ApiKeyAuthInput { ctx, credential, signal } = input;
	let env = |name: &'static str| {
		let ctx = Arc::clone(ctx);
		let signal = signal.clone();
		async move {
			check_aborted(&signal)?;
			let value = ctx.env(name).await;
			check_aborted(&signal)?;
			Ok::<Option<String>, ThrownValue>(value)
		}
	};

	let credential_env = |name: &str| -> Option<String> {
		credential
			.and_then(|credential| credential.env.as_ref())
			.and_then(|env| env.get(name).cloned())
	};

	// `credential?.key ?? await env("GOOGLE_CLOUD_API_KEY")`; `if (key)` skips
	// empty strings.
	let key = match credential.and_then(|credential| credential.key.as_deref()).filter(|key| !key.is_empty()) {
		Some(key) => Some(key.to_string()),
		None => env("GOOGLE_CLOUD_API_KEY")
			.await?
			.filter(|value| !value.is_empty()),
	};
	if let Some(key) = key {
		let stored = credential.and_then(|credential| credential.key.as_deref()).is_some_and(|key| !key.is_empty());
		return Ok(Some(AuthResult {
			auth: ModelAuth { api_key: Some(key), ..Default::default() },
			env: credential.and_then(|credential| credential.env.clone()),
			source: Some(if stored { "stored credential" } else { "GOOGLE_CLOUD_API_KEY" }.to_string()),
		}));
	}

	let adc_path = match credential_env("GOOGLE_APPLICATION_CREDENTIALS") {
		Some(path) => Some(path),
		// Upstream `??`: an empty string is set, only null/undefined fall through.
		None => env("GOOGLE_APPLICATION_CREDENTIALS").await?,
	};
	check_aborted(signal)?;
	let has_credentials = ctx.file_exists(adc_path.as_deref().unwrap_or(VERTEX_ADC_PATH)).await;
	check_aborted(signal)?;
	let project = match credential_env("GOOGLE_CLOUD_PROJECT") {
		Some(project) => Some(project),
		None => match env("GOOGLE_CLOUD_PROJECT").await? {
			Some(project) => Some(project),
			None => env("GCLOUD_PROJECT").await?,
		},
	};
	let location = match credential_env("GOOGLE_CLOUD_LOCATION") {
		Some(location) => Some(location),
		None => env("GOOGLE_CLOUD_LOCATION").await?,
	};
	if has_credentials && project.is_some() && location.is_some() {
		return Ok(Some(AuthResult {
			auth: ModelAuth::default(),
			env: credential.and_then(|credential| credential.env.clone()),
			source: Some(
				if credential.is_some() {
					"stored credential"
				} else {
					"gcloud application default credentials"
				}
				.to_string(),
			),
		}));
	}
	Ok(None)
}

/// Upstream `googleVertexProvider()`.
pub fn google_vertex_provider() -> Arc<dyn Provider> {
	create_provider(CreateProviderOptions {
		id: "google-vertex".to_string(),
		name: Some("Google Vertex AI".to_string()),
		base_url: None,
		headers: None,
		auth: ProviderAuth { api_key: Some(vertex_auth()), oauth: None },
		models: super::google_vertex_models::google_vertex_any_models(),
		fetch_models: None,
		filter_models: None,
		filter_all_models: None,
		api: Some(ApiDispatch::Single(crate::api::google_vertex::google_vertex_api())),
		images: Default::default(),
		classifiers: Default::default(),
	})
	.expect("static google-vertex provider configuration is valid")
}

// ---------------------------------------------------------------------------
// Tests (ports of the providers.test.ts google-vertex cases)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::auth::types::{AuthContext, AuthEvent, ApiKeyAuthInput, ProviderAuthInteraction, AuthPrompt};
	use crate::models::{create_models, CreateModelsOptions, Models as ModelsTrait, MutableModels};
	use crate::utils::abort::AbortController;
	use std::collections::{BTreeMap, VecDeque};
	use std::sync::{Arc, Mutex};

	const NEVER_ABORTED_SIGNAL_CHECK: &str = "signal never aborts in these tests";

	fn never_aborted() -> crate::utils::abort::AbortSignal {
		AbortController::new().signal()
	}

	fn fake_auth_context(env: BTreeMap<String, String>, files: Vec<&str>) -> Arc<dyn AuthContext> {
		struct FakeContext {
			env: BTreeMap<String, String>,
			files: Vec<String>,
		}
		impl AuthContext for FakeContext {
			fn env<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Option<String>> {
				Box::pin(async move { self.env.get(name).cloned() })
			}
			fn file_exists<'a>(&'a self, path: &'a str) -> BoxFuture<'a, bool> {
				Box::pin(async move { self.files.iter().any(|file| file == path) })
			}
		}
		Arc::new(FakeContext { env, files: files.into_iter().map(str::to_string).collect() })
	}

	fn env_map(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
		entries.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect()
	}

	/// `login` interaction with queued prompt answers and recorded events.
	fn interaction(
		answers: Arc<Mutex<VecDeque<String>>>,
		events: Arc<Mutex<Vec<AuthEvent>>>,
	) -> ProviderAuthInteraction {
		ProviderAuthInteraction {
			signal: never_aborted(),
			prompt: Box::new(move |_prompt: AuthPrompt| {
				let answers = answers.clone();
				Box::pin(async move {
					answers
						.lock()
						.expect("lock")
						.pop_front()
						.ok_or_else(|| ThrownValue::message(NEVER_ABORTED_SIGNAL_CHECK))
				})
			}),
			notify: Box::new(move |event: AuthEvent| {
				events.lock().expect("lock").push(event);
			}),
		}
	}

	#[tokio::test]
	async fn runs_provider_owned_vertex_api_key_and_adc_login_flows() {
		let key_answers = Arc::new(Mutex::new(VecDeque::from([
			"api-key".to_string(),
			"vertex-key".to_string(),
		])));
		let key_events = Arc::new(Mutex::new(Vec::new()));
		let key_credential = login_vertex(interaction(key_answers, key_events)).await.expect("api-key login");
		assert_eq!(key_credential, ApiKeyCredential { key: Some("vertex-key".to_string()), env: None });

		let adc_answers = Arc::new(Mutex::new(VecDeque::from([
			"adc".to_string(),
			"project-id".to_string(),
			"us-central1".to_string(),
		])));
		let adc_events = Arc::new(Mutex::new(Vec::new()));
		let adc_credential = login_vertex(interaction(adc_answers, adc_events.clone())).await.expect("adc login");
		assert_eq!(
			adc_credential,
			ApiKeyCredential {
				key: None,
				env: Some(env_map(&[
					("GOOGLE_CLOUD_PROJECT", "project-id"),
					("GOOGLE_CLOUD_LOCATION", "us-central1"),
				])),
			}
		);
		let events = adc_events.lock().expect("lock").clone();
		assert_eq!(events.len(), 1);
		let AuthEvent::Info { links, .. } = &events[0] else {
			panic!("expected info event, got {:?}", events[0]);
		};
		assert!(
			links
				.iter()
				.any(|link| link.label.as_deref() == Some("Application Default Credentials")),
			"expected ADC link: {links:?}"
		);
	}

	#[tokio::test]
	async fn rejects_unknown_login_methods() {
		let answers = Arc::new(Mutex::new(VecDeque::from(["mystery".to_string()])));
		let error = login_vertex(interaction(answers, Arc::new(Mutex::new(Vec::new()))))
			.await
			.expect_err("unknown method");
		assert!(
			format!("{error:?}").contains("Unknown Google Vertex AI auth method: mystery"),
			"unexpected error: {error:?}"
		);
	}

	#[tokio::test]
	async fn resolves_adc_credential_with_project_and_location() {
		let auth = vertex_auth();
		let credential = ApiKeyCredential {
			key: None,
			env: Some(env_map(&[
				("GOOGLE_CLOUD_PROJECT", "project-id"),
				("GOOGLE_CLOUD_LOCATION", "us-central1"),
			])),
		};
		let ctx = fake_auth_context(
			BTreeMap::new(),
			vec!["~/.config/gcloud/application_default_credentials.json"],
		);
		let result = (auth.resolve)(ApiKeyAuthInput { ctx, credential: Some(&credential), signal: never_aborted() })
			.await
			.expect("resolve")
			.expect("configured");
		assert_eq!(result.auth, ModelAuth::default());
		assert_eq!(result.env, credential.env);
	}

	#[tokio::test]
	async fn resolves_vertex_via_adc_file_plus_project_and_location() {
		let adc = "~/.config/gcloud/application_default_credentials.json";
		let configured = create_models(Some(CreateModelsOptions {
			auth_context: Some(fake_auth_context(
				env_map(&[("GOOGLE_CLOUD_PROJECT", "proj"), ("GOOGLE_CLOUD_LOCATION", "us-central1")]),
				vec![adc],
			)),
			..Default::default()
		}));
		configured.set_provider(google_vertex_provider());
		let model = configured
			.get_models(Some("google-vertex"))
			.first()
			.cloned()
			.expect("vertex model");

		let result = configured
			.get_auth_by_provider(model.provider(), None)
			.await
			.expect("get_auth")
			.expect("configured");
		assert_eq!(result.auth, ModelAuth::default());
		assert_eq!(result.source.as_deref(), Some("gcloud application default credentials"));

		// ADC without project/location is not configured.
		let partial = create_models(Some(CreateModelsOptions {
			auth_context: Some(fake_auth_context(env_map(&[("GOOGLE_CLOUD_PROJECT", "proj")]), vec![adc])),
			..Default::default()
		}));
		partial.set_provider(google_vertex_provider());
		assert!(
			partial
				.get_auth_by_provider(model.provider(), None)
				.await
				.expect("get_auth")
				.is_none()
		);

		// Explicit key wins over ADC.
		let keyed = create_models(Some(CreateModelsOptions {
			auth_context: Some(fake_auth_context(env_map(&[("GOOGLE_CLOUD_API_KEY", "vertex-key")]), vec![])),
			..Default::default()
		}));
		keyed.set_provider(google_vertex_provider());
		let result = keyed
			.get_auth_by_provider(model.provider(), None)
			.await
			.expect("get_auth")
			.expect("configured");
		assert_eq!(result.auth.api_key.as_deref(), Some("vertex-key"));
	}

	#[test]
	fn uses_models_dev_effort_levels_for_vertex_thinking_models() {
		let find = |id: &str| {
			super::super::google_vertex_models::google_vertex_models()
				.into_iter()
				.find(|model| model.id() == id)
				.unwrap_or_else(|| panic!("catalog model google-vertex/{id}"))
		};
		assert!(crate::models::get_supported_thinking_levels(&find("gemini-3.6-flash"))
			.contains(&crate::types::ModelThinkingLevel::Minimal));
		assert_eq!(
			crate::models::get_supported_thinking_levels(&find("gemini-3.8-flash")),
			vec![
				crate::types::ModelThinkingLevel::Low,
				crate::types::ModelThinkingLevel::Medium,
				crate::types::ModelThinkingLevel::High,
			]
		);
		assert_eq!(
			crate::models::get_supported_thinking_levels(&find("gemini-3.1-pro-preview")),
			vec![
				crate::types::ModelThinkingLevel::Low,
				crate::types::ModelThinkingLevel::Medium,
				crate::types::ModelThinkingLevel::High,
			]
		);
	}

	#[tokio::test]
	async fn google_vertex_provider_registers_models_and_api() {
		let provider = google_vertex_provider();
		assert_eq!(provider.id(), "google-vertex");
		assert_eq!(provider.name(), "Google Vertex AI");
		let models = provider.get_models();
		assert!(!models.is_empty());
		assert!(models.iter().all(|model| model.provider() == "google-vertex"));
		assert!(models.iter().all(|model| model.api() == "google-vertex"));
	}
}
