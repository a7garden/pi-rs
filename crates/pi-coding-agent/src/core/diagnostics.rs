//! Port of `packages/coding-agent/src/core/diagnostics.ts` — shared
//! diagnostics types for resource loading (extensions, skills, prompts,
//! themes).
//!
//! Wire note (crate docs, deviation D1): the payload fields are serde
//! camelCase so the serialized shape matches upstream JSON.

use serde::Serialize;

/// The kind of resource that collided (`resourceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceType {
	Extension,
	Skill,
	Prompt,
	Theme,
}

/// Severity/kind of a diagnostic (`type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticType {
	Warning,
	Error,
	Collision,
}

/// A collision between two resources claiming the same name
/// (upstream `ResourceCollision`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceCollision {
	pub resource_type: ResourceType,
	/// Skill name, command/tool/flag name, prompt name, theme name.
	pub name: String,
	pub winner_path: String,
	pub loser_path: String,
	/// e.g. `"npm:foo"`, `"git:..."`, `"local"`.
	pub winner_source: Option<String>,
	pub loser_source: Option<String>,
}

/// A warning/error/collision surfaced while loading resources
/// (upstream `ResourceDiagnostic`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceDiagnostic {
	pub r#type: DiagnosticType,
	pub message: String,
	pub path: Option<String>,
	pub collision: Option<ResourceCollision>,
}
