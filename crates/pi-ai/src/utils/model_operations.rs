//! Port of `packages/ai/src/utils/model-operations.ts`.

use crate::types::{AnyModel, AssistantImages, ClassifierModel, ClassifierResult, ClassifierStopReason, ImageModel, ImagesStopReason, Model, ModelType};
use crate::utils::diagnostics::now_ms;
use crate::utils::models_error::{ModelsError, ModelsErrorCode};
use crate::utils::thrown::ThrownValue;

/// The type of a model (`getModelType`). Chat is the default.
pub fn get_model_type(model: &AnyModel) -> ModelType {
	model.model_type()
}

/// Runtime-checked model type narrowing, including legacy chat models without
/// `type` (`isModelType`). Returns the chat model when it matches.
pub fn is_chat_model(model: &AnyModel, type_: ModelType) -> Option<&Model> {
	match model {
		AnyModel::Chat(model) if type_ == ModelType::Chat => Some(model),
		_ => None,
	}
}

/// `assertChatModel`: `Err` when the model is not a chat model.
pub fn assert_chat_model(model: &AnyModel) -> Result<&Model, ModelsError> {
	match model {
		AnyModel::Chat(model) => Ok(model),
		other => Err(not_a_model_error(other, "chat")),
	}
}

/// `assertImageModel`: `Err` when the model is not an image model.
pub fn assert_image_model(model: &AnyModel) -> Result<&ImageModel, ModelsError> {
	match model {
		AnyModel::Image(model) => Ok(model),
		other => Err(not_a_model_error(other, "image")),
	}
}

/// `assertClassifierModel`: `Err` when the model is not a classifier model.
pub fn assert_classifier_model(model: &AnyModel) -> Result<&ClassifierModel, ModelsError> {
	match model {
		AnyModel::Classifier(model) => Ok(model),
		other => Err(not_a_model_error(other, "classifier")),
	}
}

fn not_a_model_error(model: &AnyModel, expected: &str) -> ModelsError {
	ModelsError::new(
		ModelsErrorCode::Provider,
		format!("Model {}/{} is not a {} model", model.provider(), model.id(), expected),
	)
}

/// `imageErrorResult`: error-shaped [`AssistantImages`].
pub fn image_error_result(model: &ImageModel, error: &ThrownValue, aborted: bool) -> AssistantImages {
	AssistantImages {
		api: model.base.api.clone(),
		provider: model.base.provider.clone(),
		model: model.base.id.clone(),
		output: Vec::new(),
		response_id: None,
		usage: None,
		stop_reason: if aborted { ImagesStopReason::Aborted } else { ImagesStopReason::Error },
		error_message: Some(crate::utils::diagnostics::format_thrown_value(error)),
		timestamp: now_ms(),
	}
}

/// `classifierErrorResult`: error-shaped [`ClassifierResult`].
pub fn classifier_error_result(model: &ClassifierModel, error: &ThrownValue, aborted: bool) -> ClassifierResult {
	ClassifierResult {
		api: model.base.api.clone(),
		provider: model.base.provider.clone(),
		model: model.base.id.clone(),
		answers: Default::default(),
		usage: None,
		stop_reason: if aborted { ClassifierStopReason::Aborted } else { ClassifierStopReason::Error },
		error_message: Some(crate::utils::diagnostics::format_thrown_value(error)),
		timestamp: now_ms(),
	}
}
