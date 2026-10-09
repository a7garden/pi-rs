//! Portable logic of upstream `extensions/llama/ui.ts`.
//!
//! Upstream `ui.ts` is the `/llama` TUI catalog flow; the component surface
//! (pi-tui `Container`/`SelectList`/`Input`, the theme, keybindings, the
//! `HuggingFaceSearch` debounce/cache widget, `LlamaView`, `showLlamaUi`,
//! `runWithProgress`) is **deferred with the crate-wide TUI port decision**
//! (see [`super`] D99/D160). This module ports only the pure, TUI-free
//! helpers the view uses to compose its model rows:
//!
//! - [`context_label`] — upstream `contextLabel`
//! - [`model_description`] — upstream `modelDescription`
//! - [`compact_count`] — upstream `compactCount`
//!
//! This mirrors the D99 precedent of porting the UI-flow helpers that
//! upstream defines in `index.ts` as the shared logic of the deferred flow.
//!
//! # Deviation D160
//!
//! `ui.ts`'s interactive surface (menus, search widget, progress screens)
//! stays deferred (D99); only the string helpers above are ported, 1:1.
//! JavaScript `Number.isFinite` on the parsed `--ctx-size` argument maps to
//! `str::parse::<f64>` success, and upstream's `Math.round` maps to
//! `f64::round` before the k-suffix formatting.

use super::client::LlamaModelInfo;

/// Upstream `contextLabel`: the model's context size as a display label
/// (`"32k"` / `"8192"`), from the loaded metadata or the launch flags.
pub fn context_label(model: &LlamaModelInfo) -> Option<String> {
	if let Some(meta) = &model.meta {
		let context = meta.n_ctx.or(meta.n_ctx_train);
		if let Some(context) = context.filter(|context| *context > 0) {
			return Some(format_context(context));
		}
	}
	let args = model.status.args.as_deref()?.windows(2).find_map(|pair| {
		let flag = pair[0].as_str();
		if flag != "--ctx-size" && flag != "-c" && flag != "-ctx" {
			return None;
		}
		// Upstream `Number(value)` + `Number.isFinite` + `value > 0`.
		pair[1]
			.parse::<f64>()
			.ok()
			.filter(|value| value.is_finite() && *value > 0.0)
			.map(|value| format_context(value.round() as u64))
	})?;
	Some(args)
}

fn format_context(context: u64) -> String {
	if context >= 1000 {
		format!("{}k", (context as f64 / 1000.0).round() as u64)
	} else {
		context.to_string()
	}
}

/// Upstream `modelDescription`: the one-line status description
/// (`"loaded · 32k context"`).
pub fn model_description(model: &LlamaModelInfo) -> String {
	let mut details: Vec<String> = Vec::new();
	let loaded = matches!(model.status.value, super::client::LlamaModelStatus::Loaded | super::client::LlamaModelStatus::Sleeping);
	if loaded {
		details.push("loaded".to_string());
	} else if model.status.value != super::client::LlamaModelStatus::Unloaded {
		details.push(status_label(model.status.value).to_string());
	}
	let context = if loaded { context_label(model) } else { None };
	if let Some(context) = context {
		details.push(format!("{context} context"));
	}
	details.join(" · ")
}

/// Upstream pushes the raw `model.status.value` string; the Rust enum (D97a)
/// erases unknown status strings, so [`LlamaModelStatus::Unknown`] labels
/// `"unknown"`.
fn status_label(value: super::client::LlamaModelStatus) -> &'static str {
	use super::client::LlamaModelStatus as Status;
	match value {
		Status::Unloaded => "unloaded",
		Status::Loading => "loading",
		Status::Loaded => "loaded",
		Status::Downloading => "downloading",
		Status::Sleeping => "sleeping",
		Status::Unknown => "unknown",
	}
}

/// Upstream `compactCount`: `1_234_567` → `"1.2M"`, `12_345` → `"12k"`.
pub fn compact_count(value: u64) -> String {
	if value >= 1_000_000 {
		let digits = if value >= 10_000_000 { 0 } else { 1 };
		return format!("{:.*}M", digits, value as f64 / 1_000_000.0);
	}
	if value >= 1_000 {
		let digits = if value >= 100_000 { 0 } else { 1 };
		return format!("{:.*}k", digits, value as f64 / 1_000.0);
	}
	value.to_string()
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::extensions::llama::client::{LlamaModelStatus, LlamaModelStatusInfo};
	fn status(value: LlamaModelStatus, args: Option<Vec<String>>) -> LlamaModelStatusInfo {
		LlamaModelStatusInfo { value, args, ..Default::default() }
	}

	fn loaded_model(meta: Option<super::super::client::LlamaModelMeta>, args: Option<Vec<String>>) -> LlamaModelInfo {
		LlamaModelInfo {
			id: "m".to_string(),
			aliases: None,
			status: status(LlamaModelStatus::Loaded, args),
			architecture: None,
			source: None,
			meta,
		}
	}

	fn meta_with(n_ctx: Option<u64>) -> super::super::client::LlamaModelMeta {
		super::super::client::LlamaModelMeta { n_ctx, ..Default::default() }
	}

	#[test]
	fn context_label_prefers_metadata_over_flags() {
		let model = loaded_model(Some(meta_with(Some(32768))), Some(vec!["--ctx-size".into(), "4096".into()]));
		assert_eq!(context_label(&model).as_deref(), Some("33k")); // Math.round(32.768)

		let rounded = loaded_model(Some(meta_with(Some(40960))), None);
		assert_eq!(context_label(&rounded).as_deref(), Some("41k")); // Math.round(40.96)

		let small = loaded_model(Some(meta_with(Some(8192))), None);
		assert_eq!(context_label(&small).as_deref(), Some("8k"));
	}

	#[test]
	fn context_label_falls_back_to_launch_flags() {
		for flag in ["--ctx-size", "-c", "-ctx"] {
			let model = loaded_model(Some(meta_with(None)), Some(vec!["llama-server".into(), flag.into(), "16384".into()]));
			assert_eq!(context_label(&model).as_deref(), Some("16k"), "flag {flag}"); // Math.round(16.384)
		}
		let model = loaded_model(None, Some(vec!["-c".into(), "4096".into()]));
		assert_eq!(context_label(&model).as_deref(), Some("4k")); // Math.round(4.096)
	}

	#[test]
	fn context_label_ignores_invalid_flags() {
		for value in ["abc", "0", "-1"] {
			let model = loaded_model(Some(meta_with(None)), Some(vec!["-c".into(), value.into()]));
			assert_eq!(context_label(&model), None, "value {value}");
		}
		let model = loaded_model(None, Some(vec!["-c".into()]));
		assert_eq!(context_label(&model), None);
	}

	#[test]
	fn model_description_summarizes_status_and_context() {
		let mut model = loaded_model(Some(meta_with(Some(32768))), None);
		assert_eq!(model_description(&model), "loaded · 33k context");

		model.status = status(LlamaModelStatus::Sleeping, None);
		assert_eq!(model_description(&model), "loaded · 33k context");

		model.status = status(LlamaModelStatus::Unloaded, None);
		assert_eq!(model_description(&model), "");

		model.status = status(LlamaModelStatus::Loading, None);
		assert_eq!(model_description(&model), "loading");

		model.status = status(LlamaModelStatus::Unknown, None);
		assert_eq!(model_description(&model), "unknown");
	}

	#[test]
	fn compact_count_formats_counts() {
		assert_eq!(compact_count(999), "999");
		assert_eq!(compact_count(1_000), "1.0k");
		assert_eq!(compact_count(12_345), "12.3k");
		assert_eq!(compact_count(150_000), "150k");
		assert_eq!(compact_count(1_000_000), "1.0M");
		assert_eq!(compact_count(12_345_678), "12M");
	}

	#[test]
	fn status_value_string_matches_upstream_labels() {
		assert_eq!(status_label(LlamaModelStatus::Unloaded), "unloaded");
		assert_eq!(status_label(LlamaModelStatus::Downloading), "downloading");
		assert_eq!(status_label(LlamaModelStatus::Unknown), "unknown");
	}
}
