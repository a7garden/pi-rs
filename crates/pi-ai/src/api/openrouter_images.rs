//! Port of `packages/ai/src/api/openrouter-images.ts` — image generation over
//! OpenRouter's chat completions endpoint.
//!
//! Upstream uses the OpenAI SDK client; the Rust port posts the identical
//! chat-completions payload through the crate's fetch seam with
//! `retryProviderRequest` (maxRetries: 0) and the shared error normalization,
//! like the other API adapters.
//!
//! # Deviations
//!
//! - D37: the crate's `onPayload`/`onResponse` hooks are typed over chat
//!   `Model`s (types.rs); the upstream image flow passes `ImageModel`. The
//!   image path therefore does not invoke the generic hooks — final
//!   integration can widen the hook payload types if the image flows need
//!   them.

use std::future::Future;
use std::sync::Arc;

use crate::http::{HttpBody, HttpRequest, ProviderHttpError};
use crate::images_api_registry::{ImagesApiFunction, ImagesApiProvider};
use crate::models::ProviderImages;
use crate::utils::provider_retry::{retry_provider_request, ProviderRetryOptions};
use crate::types::{
	AssistantImages, ImageContent, ImageModel, ImagesContext, ImagesOptions, ImagesStopReason, TextContent, Usage,
	UsageCost,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::headers::{headers_to_record, provider_headers_to_record};
use crate::utils::sanitize_unicode::sanitize_surrogates;

/// Upstream `generateImages`.
pub async fn generate_images(
	model: &ImageModel,
	context: &ImagesContext,
	options: &ImagesOptions,
) -> AssistantImages {
	let mut output = AssistantImages {
		api: model.api().to_string(),
		provider: model.provider().to_string(),
		model: model.id().to_string(),
		output: Vec::new(),
		response_id: None,
		usage: None,
		stop_reason: ImagesStopReason::Stop,
		error_message: None,
		timestamp: crate::utils::diagnostics::now_ms(),
	};

	if let Err(error) = run_request(model, context, options, &mut output).await {
		output.stop_reason = if options.signal.as_ref().is_some_and(|signal| signal.aborted()) {
			ImagesStopReason::Aborted
		} else {
			ImagesStopReason::Error
		};
		let normalized = normalize_provider_error(&error);
		output.error_message = Some(format_provider_error(&normalized, None));
	}
	output
}

async fn run_request(
	model: &ImageModel,
	context: &ImagesContext,
	options: &ImagesOptions,
	output: &mut AssistantImages,
) -> Result<(), ProviderHttpError> {
	let Some(api_key) = options.api_key.clone().filter(|key| !key.trim().is_empty()) else {
		return Err(ProviderHttpError::new(format!("No API key for provider: {}", model.provider())));
	};

	let params = build_params(model, context);

	let fetch = options.fetch.clone().unwrap_or_default();
	let base_url = model.base.base_url.trim_end_matches('/').to_string();
	let headers = {
		let mut headers: Vec<(String, String)> =
			vec![("Authorization".to_string(), format!("Bearer {api_key}"))];
		// model headers carry plain string values (upstream spreads both
		// records); caller headers win per key.
		if let Some(model_headers) = &model.base.headers {
			for (name, value) in model_headers {
				headers.push((name.clone(), value.clone()));
			}
		}
		if let Some(merged) = provider_headers_to_record(&[options.headers.as_ref()]) {
			for (name, value) in merged {
				headers.retain(|(existing, _)| existing != &name);
				headers.push((name, value));
			}
		}
		headers.push(("content-type".to_string(), "application/json".to_string()));
		headers
	};
	let body: std::sync::Arc<Vec<u8>> = std::sync::Arc::new(serde_json::to_vec(&params).expect("request body serializes"));
	let timeout_ms = options.timeout_ms;
	let signal = options.signal.clone();

	let url = format!("{base_url}/chat/completions");
	let response = retry_provider_request(
		|| {
			let mut request = HttpRequest::new("POST", url.clone());
			request.headers = headers.clone();
			request.body = HttpBody::bytes((*body).clone());
			request.timeout_ms = timeout_ms;
			request.signal = signal.clone();
			let fetch = fetch.clone();
			async move { fetch.send(request).await }
		},
		ProviderRetryOptions {
			max_retries: Some(0),
			max_retry_delay_ms: options.max_retry_delay_ms,
			signal: options.signal.clone(),
		},
	)
	.await?;

	let bytes = response.body.collect().await?;
	let response_value: serde_json::Value = serde_json::from_slice(&bytes)
		.map_err(|error| ProviderHttpError::new(format!("OpenRouter images returned invalid JSON: {error}")))?;

	output.response_id = response_value.get("id").and_then(serde_json::Value::as_str).map(str::to_string);
	if let Some(usage) = response_value.get("usage").filter(|usage| usage.is_object()) {
		output.usage = Some(parse_usage(usage, model));
	}

	if let Some(choice) = response_value
		.get("choices")
		.and_then(serde_json::Value::as_array)
		.and_then(|choices| choices.first())
	{
		if let Some(content) = choice.pointer("/message/content").and_then(serde_json::Value::as_str) {
			if !content.is_empty() {
				output.output.push(crate::types::UserContent::Text(TextContent::new(content)));
			}
		}
		if let Some(images) = choice.pointer("/message/images").and_then(serde_json::Value::as_array) {
			for image in images {
				let image_url = match image.get("image_url") {
					Some(serde_json::Value::String(url)) => Some(url.clone()),
					Some(serde_json::Value::Object(object)) => {
						object.get("url").and_then(serde_json::Value::as_str).map(str::to_string)
					}
					_ => None,
				};
				let Some(image_url) = image_url.filter(|url| url.starts_with("data:")) else {
					continue;
				};
				let Some((mime_type, data)) = parse_data_url(&image_url) else {
					continue;
				};
				output
					.output
					.push(crate::types::UserContent::Image(ImageContent::new(data, mime_type)));
			}
		}
	}

	Ok(())
}

/// Upstream `OpenRouterImagesCreateParams::buildParams`.
fn build_params(model: &ImageModel, context: &ImagesContext) -> serde_json::Value {
	let content: Vec<serde_json::Value> = context
		.input
		.iter()
		.map(|item| match item {
			crate::types::UserContent::Text(text) => serde_json::json!({
				"type": "text",
				"text": sanitize_surrogates(&text.text),
			}),
			crate::types::UserContent::Image(image) => serde_json::json!({
				"type": "image_url",
				"image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
			}),
		})
		.collect();

	let modalities = if model.output.iter().any(|output| matches!(output, crate::types::ModelOutput::Text)) {
		serde_json::json!(["image", "text"])
	} else {
		serde_json::json!(["image"])
	};

	serde_json::json!({
		"model": model.id(),
		"messages": [
			{
				"role": "user",
				"content": content,
			}
		],
		"stream": false,
		"modalities": modalities,
	})
}

fn parse_data_url(url: &str) -> Option<(String, String)> {
	let rest = url.strip_prefix("data:")?;
	let (meta, data) = rest.split_once(',')?;
	let mime_type = meta.strip_suffix(";base64")?.to_string();
	Some((mime_type, data.to_string()))
}

/// Upstream `parseUsage`.
fn parse_usage(raw_usage: &serde_json::Value, model: &ImageModel) -> Usage {
	let prompt_tokens = raw_usage.get("prompt_tokens").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
	let details = raw_usage.get("prompt_tokens_details");
	let reported_cached_tokens = details
		.and_then(|details| details.get("cached_tokens"))
		.and_then(serde_json::Value::as_f64)
		.unwrap_or(0.0);
	let cache_write_tokens = details
		.and_then(|details| details.get("cache_write_tokens"))
		.and_then(serde_json::Value::as_f64)
		.unwrap_or(0.0);
	let cache_read_tokens = if cache_write_tokens > 0.0 {
		(reported_cached_tokens - cache_write_tokens).max(0.0)
	} else {
		reported_cached_tokens
	};
	let input = (prompt_tokens - cache_read_tokens - cache_write_tokens).max(0.0);
	let output_tokens = raw_usage.get("completion_tokens").and_then(serde_json::Value::as_f64).unwrap_or(0.0);

	let cost = &model.base.cost.rates;
	let cost_input = (cost.input / 1_000_000.0) * input;
	let cost_output = (cost.output / 1_000_000.0) * output_tokens;
	let cost_cache_read = (cost.cache_read / 1_000_000.0) * cache_read_tokens;
	let cost_cache_write = (cost.cache_write / 1_000_000.0) * cache_write_tokens;

	Usage {
		input: input as u64,
		output: output_tokens as u64,
		cache_read: cache_read_tokens as u64,
		cache_write: cache_write_tokens as u64,
		cache_write_1h: None,
		reasoning: None,
		total_tokens: (input + output_tokens + cache_read_tokens + cache_write_tokens) as u64,
		cost: UsageCost {
			input: cost_input,
			output: cost_output,
			cache_read: cost_cache_read,
			cache_write: cost_cache_write,
			total: cost_input + cost_output + cost_cache_read + cost_cache_write,
		},
	}
}

/// Upstream `openrouterImagesApi()` — the provider-facing adapter consumed by
/// `providers/openrouter.rs`.
pub fn openrouter_images_api() -> Arc<dyn ProviderImages> {
	ImagesApiProvider {
		api: "openrouter-images".to_string(),
		generate_images: generate_images_function(),
	}
	.into_provider_images()
}

/// The function form handed to [`crate::images_api_registry`] by
/// `providers/images/register-builtins` at final integration.
pub fn generate_images_function() -> ImagesApiFunction {
	Arc::new(move |model, context, options| {
		Box::pin(async move { generate_images(&model, &context, &options).await })
	})
}

impl ImagesApiProvider {
	/// Registry-input → trait-object adapter.
	pub fn into_provider_images(self) -> Arc<dyn ProviderImages> {
		struct Adapter {
			inner: ImagesApiFunction,
		}
		impl ProviderImages for Adapter {
			fn generate_images<'a>(
				&'a self,
				model: &'a ImageModel,
				context: &'a ImagesContext,
				options: &'a ImagesOptions,
			) -> Pin<Box<dyn Future<Output = AssistantImages> + Send + 'a>> {
				Box::pin((self.inner)(model.clone(), context.clone(), options.clone()))
			}
		}
		Arc::new(Adapter { inner: self.generate_images })
	}
}

use std::pin::Pin;
