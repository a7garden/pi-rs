//! Port of @earendil-works/pi-telemetry (upstream: packages/telemetry).

mod memory;
mod noop;
pub mod testing;

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use memory::{InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan};
pub use noop::{noop_telemetry_span, NoopTelemetryContext, NOOP_TELEMETRY_CONTEXT};

/// Builds a [`SpanAttributes`] map: `attrs! { "name" => value, ... }`.
#[macro_export]
macro_rules! attrs {
	($($name:expr => $value:expr),* $(,)?) => {{
		let mut attributes = $crate::SpanAttributes::new();
		$( attributes.insert($name.to_string(), $crate::AttributeValue::from($value)); )*
		attributes
	}};
}

/// A single telemetry attribute value (`AttributeValue` upstream).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    String(String),
    Number(f64),
    Boolean(bool),
    StringArray(Vec<String>),
    NumberArray(Vec<f64>),
    BoolArray(Vec<bool>),
}

impl From<&str> for AttributeValue {
    fn from(value: &str) -> Self {
        AttributeValue::String(value.to_string())
    }
}

impl From<String> for AttributeValue {
    fn from(value: String) -> Self {
        AttributeValue::String(value)
    }
}

impl From<f64> for AttributeValue {
    fn from(value: f64) -> Self {
        AttributeValue::Number(value)
    }
}

impl From<i64> for AttributeValue {
    fn from(value: i64) -> Self {
        AttributeValue::Number(value as f64)
    }
}

impl From<i32> for AttributeValue {
    fn from(value: i32) -> Self {
        AttributeValue::Number(f64::from(value))
    }
}

impl From<bool> for AttributeValue {
    fn from(value: bool) -> Self {
        AttributeValue::Boolean(value)
    }
}

impl From<Vec<String>> for AttributeValue {
    fn from(value: Vec<String>) -> Self {
        AttributeValue::StringArray(value)
    }
}

impl From<Vec<&str>> for AttributeValue {
    fn from(value: Vec<&str>) -> Self {
        AttributeValue::StringArray(value.into_iter().map(str::to_string).collect())
    }
}

impl From<Vec<f64>> for AttributeValue {
    fn from(value: Vec<f64>) -> Self {
        AttributeValue::NumberArray(value)
    }
}

impl From<Vec<bool>> for AttributeValue {
    fn from(value: Vec<bool>) -> Self {
        AttributeValue::BoolArray(value)
    }
}

/// `SpanAttributes` upstream. Upstream allows `undefined` values that recording
/// skips; the typed Rust surface expresses that by omitting the entry entirely.
pub type SpanAttributes = BTreeMap<String, AttributeValue>;

/// Options accepted by [`TelemetryContext::start_span_box`].
#[derive(Debug, Clone)]
pub struct SpanOptions {
    pub name: String,
    pub attributes: Option<SpanAttributes>,
}

impl SpanOptions {
    pub fn new(name: impl Into<String>) -> Self {
        SpanOptions {
            name: name.into(),
            attributes: None,
        }
    }

    pub fn with_attributes(name: impl Into<String>, attributes: SpanAttributes) -> Self {
        SpanOptions {
            name: name.into(),
            attributes: Some(attributes),
        }
    }
}

/// Detailed error information attached to an error status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanErrorDetail {
    pub name: String,
    pub message: String,
}

/// `SpanStatus` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum SpanStatus {
    #[serde(rename = "ok")]
    Ok,
    #[serde(rename = "error")]
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<SpanErrorDetail>,
    },
}

impl SpanStatus {
    pub fn ok() -> Self {
        SpanStatus::Ok
    }

    pub fn error() -> Self {
        SpanStatus::Error { error: None }
    }

    pub fn error_with(name: impl Into<String>, message: impl Into<String>) -> Self {
        SpanStatus::Error {
            error: Some(SpanErrorDetail {
                name: name.into(),
                message: message.into(),
            }),
        }
    }
}

/// Error surface carried by span callbacks. Mirrors the upstream contract that a
/// rejection value propagates unchanged while spans derive an automatic error
/// status from the error's name and message.
pub trait SpanError: std::error::Error + Send + Sync + 'static {
    /// Upstream `error.name`.
    fn telemetry_name(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }

    /// Upstream `error.message`.
    fn telemetry_message(&self) -> String {
        ToString::to_string(self)
    }
}

impl<T: std::error::Error + Send + Sync + 'static> SpanError for T {
    fn telemetry_name(&self) -> String {
        std::any::type_name::<T>().to_string()
    }

    fn telemetry_message(&self) -> String {
        ToString::to_string(self)
    }
}

/// Shared error value passed through span lifecycles.
pub type SharedError = Arc<dyn SpanError>;

/// Boxed future returned by span callbacks and `start_span`.
pub type BoxSpanFuture<'a, T = ()> =
    Pin<Box<dyn Future<Output = Result<T, SharedError>> + Send + 'a>>;

/// Object-safe callback shape handed to `start_span_box`.
pub type SpanCallback<'a> =
    Box<dyn FnOnce(Arc<dyn TelemetrySpan>) -> BoxSpanFuture<'a> + Send + 'a>;

/// `TelemetrySpan` upstream. The callback-based `startSpan` is exposed through
/// the object-safe `start_span_box` plus the generic [`TelemetrySpanExt`].
pub trait TelemetrySpan: Send + Sync {
    fn start_span_box<'a>(
        &'a self,
        options: SpanOptions,
        callback: SpanCallback<'a>,
    ) -> BoxSpanFuture<'a>;
    fn add_event(&self, name: &str, attributes: Option<SpanAttributes>);
    fn set_attributes(&self, attributes: SpanAttributes);
    fn set_status(&self, status: SpanStatus);
}

/// `TelemetryContext` upstream.
pub trait TelemetryContext: Send + Sync {
    fn start_span_box<'a>(
        &'a self,
        options: SpanOptions,
        callback: SpanCallback<'a>,
    ) -> BoxSpanFuture<'a>;
}

impl<T: TelemetryContext + ?Sized> TelemetryContext for Arc<T> {
    fn start_span_box<'a>(
        &'a self,
        options: SpanOptions,
        callback: SpanCallback<'a>,
    ) -> BoxSpanFuture<'a> {
        (**self).start_span_box(options, callback)
    }
}

impl<T: TelemetrySpan + ?Sized> TelemetrySpan for Arc<T> {
    fn start_span_box<'a>(
        &'a self,
        options: SpanOptions,
        callback: SpanCallback<'a>,
    ) -> BoxSpanFuture<'a> {
        (**self).start_span_box(options, callback)
    }

    fn add_event(&self, name: &str, attributes: Option<SpanAttributes>) {
        (**self).add_event(name, attributes);
    }

    fn set_attributes(&self, attributes: SpanAttributes) {
        (**self).set_attributes(attributes);
    }

    fn set_status(&self, status: SpanStatus) {
        (**self).set_status(status);
    }
}

/// Generic `startSpan<T>` surface over the object-safe core, preserving the
/// callback result value exactly as upstream does.
pub trait TelemetryContextExt: TelemetryContext {
    fn start_span<'a, T, F>(&'a self, options: SpanOptions, callback: F) -> BoxSpanFuture<'a, T>
    where
        T: Send + 'a,
        F: FnOnce(Arc<dyn TelemetrySpan>) -> BoxSpanFuture<'a, T> + Send + 'a,
    {
        let cell: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
        let inner_cell = Arc::clone(&cell);
        let inner: SpanCallback<'a> = Box::new(move |span| {
            let fut = callback(span);
            Box::pin(async move {
                match fut.await {
                    Ok(value) => {
                        *inner_cell.lock() = Some(value);
                        Ok(())
                    }
                    Err(err) => Err(err),
                }
            })
        });
        Box::pin(async move {
            self.start_span_box(options, inner).await.map(|()| {
                cell.lock()
                    .take()
                    .expect("span callback did not produce a value")
            })
        })
    }
}

impl<C: TelemetryContext + ?Sized> TelemetryContextExt for C {}

/// Generic `startSpan<T>` on an open span (child spans upstream).
pub trait TelemetrySpanExt: TelemetrySpan {
    fn start_span<'a, T, F>(&'a self, options: SpanOptions, callback: F) -> BoxSpanFuture<'a, T>
    where
        T: Send + 'a,
        F: FnOnce(Arc<dyn TelemetrySpan>) -> BoxSpanFuture<'a, T> + Send + 'a,
    {
        let cell: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
        let inner_cell = Arc::clone(&cell);
        let inner: SpanCallback<'a> = Box::new(move |span| {
            let fut = callback(span);
            Box::pin(async move {
                match fut.await {
                    Ok(value) => {
                        *inner_cell.lock() = Some(value);
                        Ok(())
                    }
                    Err(err) => Err(err),
                }
            })
        });
        Box::pin(async move {
            self.start_span_box(options, inner).await.map(|()| {
                cell.lock()
                    .take()
                    .expect("span callback did not produce a value")
            })
        })
    }
}

impl<S: TelemetrySpan + ?Sized> TelemetrySpanExt for S {}

/// `TelemetryAttributeType` upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TelemetryAttributeType {
    String,
    Number,
    Boolean,
    #[serde(rename = "string[]")]
    StringArray,
    #[serde(rename = "number[]")]
    NumberArray,
    #[serde(rename = "boolean[]")]
    BooleanArray,
}

/// Cardinality hint on an attribute definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TelemetryAttributeCardinality {
    Low,
    High,
}

/// `TelemetryAttributeMetadata` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryAttributeMetadata {
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensitive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<TelemetryAttributeCardinality>,
}

/// `TelemetryAttributeDefinition` upstream (metadata tagged by `type`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TelemetryAttributeDefinition {
    #[serde(rename = "string")]
    String {
        #[serde(flatten)]
        metadata: TelemetryAttributeMetadata,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        values: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples: Option<Vec<String>>,
    },
    #[serde(rename = "number")]
    Number {
        #[serde(flatten)]
        metadata: TelemetryAttributeMetadata,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        values: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples: Option<Vec<f64>>,
    },
    #[serde(rename = "boolean")]
    Boolean {
        #[serde(flatten)]
        metadata: TelemetryAttributeMetadata,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        values: Option<Vec<bool>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples: Option<Vec<bool>>,
    },
    #[serde(rename = "string[]")]
    StringArray {
        #[serde(flatten)]
        metadata: TelemetryAttributeMetadata,
        #[serde(
            rename = "elementValues",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        element_values: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples: Option<Vec<Vec<String>>>,
    },
    #[serde(rename = "number[]")]
    NumberArray {
        #[serde(flatten)]
        metadata: TelemetryAttributeMetadata,
        #[serde(
            rename = "elementValues",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        element_values: Option<Vec<f64>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples: Option<Vec<Vec<f64>>>,
    },
    #[serde(rename = "boolean[]")]
    BooleanArray {
        #[serde(flatten)]
        metadata: TelemetryAttributeMetadata,
        #[serde(
            rename = "elementValues",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        element_values: Option<Vec<bool>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        examples: Option<Vec<Vec<bool>>>,
    },
}

/// `TelemetryStartAttributeDefinition` / `TelemetryEventAttributeDefinition`:
/// an attribute definition plus its `required` flag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryRequiredAttributeDefinition {
    #[serde(flatten)]
    pub definition: TelemetryAttributeDefinition,
    pub required: bool,
}

/// `TelemetryEventDefinition` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryEventDefinition {
    pub description: String,
    pub attributes: BTreeMap<String, TelemetryRequiredAttributeDefinition>,
}

/// `TelemetryParentDefinition` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TelemetryParentDefinition {
    Any,
    #[serde(rename = "root_or_external")]
    RootOrExternal,
    Spans {
        spans: Vec<String>,
    },
}

/// `status: { default: "ok", errorWhen: string }` on a span definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySpanStatusDefinition {
    #[serde(rename = "default")]
    pub default: String,
    #[serde(rename = "errorWhen")]
    pub error_when: String,
}

/// `TelemetrySpanDefinition` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySpanDefinition {
    pub description: String,
    pub parents: TelemetryParentDefinition,
    #[serde(rename = "startAttributes")]
    pub start_attributes: BTreeMap<String, TelemetryRequiredAttributeDefinition>,
    #[serde(rename = "endAttributes")]
    pub end_attributes: BTreeMap<String, TelemetryAttributeDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events: Option<BTreeMap<String, TelemetryEventDefinition>>,
    pub status: TelemetrySpanStatusDefinition,
}

/// `TelemetrySchemaDefinition` upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySchemaDefinition {
    pub version: u32,
    pub spans: BTreeMap<String, TelemetrySpanDefinition>,
}

/// Typed identity helper for serializable telemetry schema data
/// (`defineTelemetrySchema` upstream; identity at runtime).
pub fn define_telemetry_schema(schema: TelemetrySchemaDefinition) -> TelemetrySchemaDefinition {
    schema
}

/// Per-span overload set bound to one explicit parent context
/// (`TypedSpanStarter` upstream). Runtime behavior is a pass-through
/// `startSpan({ name, attributes }, callback)`; schemas are used only at the
/// upstream type level, so they are accepted and ignored here.
#[derive(Clone)]
pub struct TypedSpanStarter {
    context: Arc<dyn TelemetryContext>,
}

/// Bind an explicit parent context to the combined span vocabulary of one or
/// more schemas (`createTypedSpanStarter` upstream). No runtime schema
/// validation is performed, matching upstream.
pub fn create_typed_span_starter(
    telemetry_context: Arc<dyn TelemetryContext>,
    _schemas: Vec<TelemetrySchemaDefinition>,
) -> TypedSpanStarter {
    TypedSpanStarter {
        context: telemetry_context,
    }
}

impl TypedSpanStarter {
    pub fn start_span<'a, T, F>(
        &'a self,
        name: &str,
        attributes: SpanAttributes,
        callback: F,
    ) -> BoxSpanFuture<'a, T>
    where
        T: Send + 'a,
        F: FnOnce(Arc<dyn TelemetrySpan>, TypedSpanStarter) -> BoxSpanFuture<'a, T> + Send + 'a,
    {
        self.context.start_span(
            SpanOptions::with_attributes(name, attributes),
            move |span| {
                // Upstream binds the child starter to the span itself
                // (`TelemetrySpan extends TelemetryContext`).
                let child_starter = TypedSpanStarter {
                    context: Arc::new(TelemetrySpanContext(Arc::clone(&span))),
                };
                callback(span, child_starter)
            },
        )
    }
}

/// Adapter mirroring upstream's `TelemetrySpan extends TelemetryContext`: any
/// open span can serve as the explicit parent context for child spans.
pub struct TelemetrySpanContext(pub Arc<dyn TelemetrySpan>);

impl TelemetryContext for TelemetrySpanContext {
    fn start_span_box<'a>(
        &'a self,
        options: SpanOptions,
        callback: SpanCallback<'a>,
    ) -> BoxSpanFuture<'a> {
        self.0.start_span_box(options, callback)
    }
}

#[cfg(test)]
mod tests;
