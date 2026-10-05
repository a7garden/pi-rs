//! Port of `packages/protocol/src/cbor/options.ts`.

use super::CborError;

pub const UINT32_BASE: u64 = 0x1_0000_0000;
pub const MAX_UINT32: usize = 0xffff_ffff;
const MAX_CONFIGURED_DEPTH: usize = 512;

/// Safe defaults for untrusted protocol payloads.
pub const DEFAULT_MAX_CBOR_BYTE_LENGTH: usize = 16 * 1024 * 1024;
pub const DEFAULT_MAX_CBOR_CONTAINER_LENGTH: usize = 1_000_000;
pub const DEFAULT_MAX_CBOR_DEPTH: usize = 64;

/// Upstream `CborOptions` (`undefined` fields become `None`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CborOptions {
	/// Maximum encoded input/output bytes and maximum byte/text string length.
	pub max_byte_length: Option<usize>,
	/// Maximum number of elements in an array or entries in a map.
	pub max_container_length: Option<usize>,
	/// Maximum recursive item depth.
	pub max_depth: Option<usize>,
}

/// Upstream `ResolvedCborOptions`.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedCborOptions {
	pub max_byte_length: usize,
	pub max_container_length: usize,
	pub max_depth: usize,
}

/// Upstream throws `RangeError` for invalid limits; Rust collapses that into
/// [`CborError`] carrying the same message (negative/NaN limits are
/// unrepresentable for `usize` fields — recorded deviation).
fn resolve_limit(name: &str, value: usize, maximum: usize) -> Result<usize, CborError> {
	if value > maximum {
		return Err(CborError::new(format!(
			"{name} must be an integer between 0 and {maximum}"
		)));
	}
	Ok(value)
}

pub fn resolve_options(options: Option<&CborOptions>) -> Result<ResolvedCborOptions, CborError> {
	Ok(ResolvedCborOptions {
		max_byte_length: resolve_limit(
			"maxByteLength",
			options.and_then(|o| o.max_byte_length).unwrap_or(DEFAULT_MAX_CBOR_BYTE_LENGTH),
			MAX_UINT32,
		)?,
		max_container_length: resolve_limit(
			"maxContainerLength",
			options
				.and_then(|o| o.max_container_length)
				.unwrap_or(DEFAULT_MAX_CBOR_CONTAINER_LENGTH),
			MAX_UINT32,
		)?,
		max_depth: resolve_limit(
			"maxDepth",
			options.and_then(|o| o.max_depth).unwrap_or(DEFAULT_MAX_CBOR_DEPTH),
			MAX_CONFIGURED_DEPTH,
		)?,
	})
}
