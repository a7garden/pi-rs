//! Port of `packages/ai/src/providers/cloudflare-stream.ts`.
//!
//! Wraps API implementations so Cloudflare account/gateway endpoint
//! placeholders materialize from the resolved provider env before dispatch.
//!
//! Deviations (per-file ledger):
//! - **D15** — upstream `resolveCloudflareModel<T extends { baseUrl: string }>`
//!   is generic over any model shape; Rust gets one concrete helper per model
//!   type actually wrapped upstream ([`resolve_cloudflare_model`] for chat
//!   models, [`resolve_cloudflare_classifier_model`] for classifiers).
//! - **D16** — upstream returns the original object when the URL is
//!   unchanged; Rust clones (crate deviation D12 — owned hook/request
//!   arguments), which is unobservable.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::models::{ProviderClassifier, ProviderStreams};
use crate::types::{
    ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult, Model, ProviderEnv,
    SimpleStreamOptions, StreamOptions, TranscriptContext,
};
use crate::utils::event_stream::AssistantMessageEventStream;

pub const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
pub const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

/// `replaceAll("{CLOUDFLARE_*}", env[...] ?? "{CLOUDFLARE_*}")` — unresolvable
/// placeholders are kept verbatim.
fn materialize_base_url(base_url: &str, env: &ProviderEnv) -> String {
    let account_placeholder = format!("{{{CLOUDFLARE_ACCOUNT_ID}}}");
    let gateway_placeholder = format!("{{{CLOUDFLARE_GATEWAY_ID}}}");
    let account = env
        .get(CLOUDFLARE_ACCOUNT_ID)
        .cloned()
        .unwrap_or_else(|| account_placeholder.clone());
    let gateway = env
        .get(CLOUDFLARE_GATEWAY_ID)
        .cloned()
        .unwrap_or_else(|| gateway_placeholder.clone());
    base_url
        .replace(&account_placeholder, &account)
        .replace(&gateway_placeholder, &gateway)
}

/// Upstream `resolveCloudflareModel` for chat models.
pub fn resolve_cloudflare_model(model: &Model, env: Option<&ProviderEnv>) -> Model {
    let Some(env) = env else { return model.clone() };
    let base_url = materialize_base_url(&model.base.base_url, env);
    if base_url == model.base.base_url {
        return model.clone();
    }
    let mut next = model.clone();
    next.base.base_url = base_url;
    next
}

/// Upstream `resolveCloudflareModel` for classifier models.
pub fn resolve_cloudflare_classifier_model(
    model: &ClassifierModel,
    env: Option<&ProviderEnv>,
) -> ClassifierModel {
    let Some(env) = env else { return model.clone() };
    let base_url = materialize_base_url(&model.base.base_url, env);
    if base_url == model.base.base_url {
        return model.clone();
    }
    let mut next = model.clone();
    next.base.base_url = base_url;
    next
}

/// `cloudflareStreams`: the [`ProviderStreams`] wrapper resolving Cloudflare
/// placeholders before dispatch. Deferred methods are absent upstream, so the
/// trait defaults keep the presence probes `false`.
struct CloudflareStreams {
    inner: Arc<dyn ProviderStreams>,
}

impl ProviderStreams for CloudflareStreams {
    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        let request_model = resolve_cloudflare_model(model, options.base.env.as_ref());
        self.inner.stream(&request_model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let request_model = resolve_cloudflare_model(model, options.base.base.env.as_ref());
        self.inner.stream_simple(&request_model, context, options)
    }
}

/// Wrap an API implementation so Cloudflare account/gateway endpoint
/// placeholders materialize from the resolved provider env before dispatch.
pub fn cloudflare_streams(streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
    Arc::new(CloudflareStreams { inner: streams })
}

/// `cloudflareClassifier`: the [`ProviderClassifier`] counterpart of
/// [`cloudflare_streams`].
struct CloudflareClassifier {
    inner: Arc<dyn ProviderClassifier>,
}

impl ProviderClassifier for CloudflareClassifier {
    fn classify<'a>(
        &'a self,
        model: &'a ClassifierModel,
        context: &'a ClassifierContext,
        options: &'a ClassifierOptions,
    ) -> Pin<Box<dyn Future<Output = ClassifierResult> + Send + 'a>> {
        let inner = self.inner.clone();
        let request_model = resolve_cloudflare_classifier_model(model, options.base.env.as_ref());
        Box::pin(async move { inner.classify(&request_model, context, options).await })
    }
}

/// Classifier counterpart of [`cloudflare_streams`].
pub fn cloudflare_classifier(
    classifier: Arc<dyn ProviderClassifier>,
) -> Arc<dyn ProviderClassifier> {
    Arc::new(CloudflareClassifier { inner: classifier })
}

#[cfg(test)]
mod tests {
    // Port of test/cloudflare-stream.test.ts.
    use parking_lot::Mutex;

    use super::*;
    use crate::types::{BaseModelFields, Context, ModelCost, ModelCostRates, ModelInput};

    fn test_model() -> Model {
        Model {
			base: BaseModelFields {
				id: "model".to_string(),
				name: "model".to_string(),
				api: "openai-completions".to_string(),
				provider: "cloudflare-ai-gateway".to_string(),
				base_url:
					"https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai"
						.to_string(),
				input: vec![ModelInput::Text],
				input_limits: None,
				cost: ModelCost {
					rates: ModelCostRates { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 },
					tiers: None,
				},
				headers: None,
			},
			model_type: None,
			reasoning: false,
			thinking_level_map: None,
			prompt_cache: None,
			context_window: 1000,
			max_tokens: 100,
			sampling_params: None,
			sampling_params_by_thinking_level: None,
			compat: None,
		}
    }

    fn empty_context() -> TranscriptContext {
        crate::utils::transcript::normalize_context(&Context::default())
    }

    /// Capturing API implementation (`captured` in the upstream fakes).
    struct CapturingStreams {
        captured: Arc<Mutex<Vec<String>>>,
    }

    impl ProviderStreams for CapturingStreams {
        fn stream(
            &self,
            model: &Model,
            _context: &TranscriptContext,
            _options: &StreamOptions,
        ) -> AssistantMessageEventStream {
            self.captured.lock().push(model.base.base_url.clone());
            AssistantMessageEventStream::new()
        }

        fn stream_simple(
            &self,
            model: &Model,
            _context: &TranscriptContext,
            _options: &SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            self.captured.lock().push(model.base.base_url.clone());
            AssistantMessageEventStream::new()
        }
    }

    fn env_fixture() -> ProviderEnv {
        [
            (CLOUDFLARE_ACCOUNT_ID.to_string(), "account".to_string()),
            (CLOUDFLARE_GATEWAY_ID.to_string(), "gateway".to_string()),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn materializes_the_model_endpoint_before_dispatch() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let streams = cloudflare_streams(Arc::new(CapturingStreams {
            captured: captured.clone(),
        }));
        let env = env_fixture();

        let mut stream_options = StreamOptions::default();
        stream_options.base.env = Some(env.clone());
        streams.stream(&test_model(), &empty_context(), &stream_options);

        let mut simple_options = SimpleStreamOptions::default();
        simple_options.base.base.env = Some(env);
        streams.stream_simple(&test_model(), &empty_context(), &simple_options);

        assert_eq!(
            *captured.lock(),
            vec![
                "https://gateway.ai.cloudflare.com/v1/account/gateway/openai".to_string(),
                "https://gateway.ai.cloudflare.com/v1/account/gateway/openai".to_string(),
            ]
        );
    }

    #[test]
    fn keeps_placeholders_when_the_provider_env_does_not_resolve_them() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let streams = cloudflare_streams(Arc::new(CapturingStreams {
            captured: captured.clone(),
        }));

        let options = SimpleStreamOptions::default();
        streams.stream_simple(&test_model(), &empty_context(), &options);

        let captured = captured.lock();
        assert_eq!(
			captured.first().map(String::as_str),
			Some("https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai")
		);
    }
}
