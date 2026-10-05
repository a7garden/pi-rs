//! Port of `packages/protocol/src/cbor/decoder.ts`.

use super::encoder::cbor_to_json;
use super::options::{resolve_options, ResolvedCborOptions, UINT32_BASE};
use super::{CborError, CborOptions, CborValue};
use pi_chord::json::JsonValue;

const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

struct CborReader<'a> {
	bytes: &'a [u8],
	offset: usize,
	options: ResolvedCborOptions,
}

impl<'a> CborReader<'a> {
	fn decode(&mut self) -> Result<CborValue, CborError> {
		let value = self.read_item(0)?;
		if self.offset != self.bytes.len() {
			return Err(CborError::new("CBOR payload contains trailing data"));
		}
		Ok(value)
	}

	fn read_item(&mut self, depth: usize) -> Result<CborValue, CborError> {
		if depth > self.options.max_depth {
			return Err(CborError::new(format!(
				"CBOR nesting depth exceeds configured limit of {}",
				self.options.max_depth
			)));
		}
		let initial = self.read_byte()?;
		let major_type = initial >> 5;
		let additional_information = (initial & 0x1f) as u32;

		match major_type {
			0 => Ok(CborValue::Integer(self.read_argument(additional_information)? as i64)),
			1 => {
				let value = -1 - self.read_argument(additional_information)? as i64;
				if !(-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&value) {
					return Err(CborError::new("Decoded CBOR integer is outside the safe range"));
				}
				Ok(CborValue::Integer(value))
			}
			2 => {
				let length =
					self.read_length(additional_information, "byte string", self.options.max_byte_length)?;
				Ok(CborValue::Bytes(self.read_bytes(length)?.to_vec()))
			}
			3 => {
				let length =
					self.read_length(additional_information, "text string", self.options.max_byte_length)?;
				let bytes = self.read_bytes(length)?;
				match std::str::from_utf8(bytes) {
					// The upstream decoder uses a BOM-preserving, fatal TextDecoder; Rust's
					// `from_utf8` never strips a BOM and rejects invalid UTF-8, matching it.
					Ok(text) => Ok(CborValue::Text(text.to_string())),
					Err(_) => Err(CborError::new("CBOR text string contains invalid UTF-8")),
				}
			}
			4 => {
				let length = self.read_length(additional_information, "array", self.options.max_container_length)?;
				let mut result = Vec::with_capacity(length.min(1024));
				for _ in 0..length {
					result.push(self.read_item(depth + 1)?);
				}
				Ok(CborValue::Array(result))
			}
			5 => {
				let length = self.read_length(additional_information, "map", self.options.max_container_length)?;
				let mut result = Vec::new();
				let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();
				for _ in 0..length {
					let key = self.read_item(depth + 1)?;
					let CborValue::Text(key) = key else {
						return Err(CborError::new("CBOR map keys must be strings"));
					};
					if keys.contains(&key) {
						return Err(CborError::new("CBOR map contains a duplicate key"));
					}
					keys.insert(key.clone());
					let value = self.read_item(depth + 1)?;
					result.push((key, value));
				}
				Ok(CborValue::Map(result))
			}
			6 => Err(CborError::new("CBOR tags are not supported")),
			7 => self.read_simple(additional_information),
			_ => Err(CborError::new("Malformed CBOR major type")),
		}
	}

	fn read_simple(&mut self, additional_information: u32) -> Result<CborValue, CborError> {
		match additional_information {
			20 => Ok(CborValue::Bool(false)),
			21 => Ok(CborValue::Bool(true)),
			22 => Ok(CborValue::Null),
			27 => {
				let bytes = self.read_bytes(8)?;
				let value = f64::from_bits(u64::from_be_bytes(bytes.try_into().unwrap()));
				if !value.is_finite() {
					return Err(CborError::new("Decoded CBOR number must be finite"));
				}
				if value.fract() == 0.0 && !(-MAX_SAFE_INTEGER as f64..=MAX_SAFE_INTEGER as f64).contains(&value) {
					return Err(CborError::new("Decoded CBOR integer is outside the safe range"));
				}
				Ok(CborValue::Float(value))
			}
			31 => Err(CborError::new("CBOR break marker is not supported")),
			_ => Err(CborError::new("Unsupported CBOR simple value or floating-point width")),
		}
	}

	fn read_length(&mut self, additional_information: u32, kind: &str, limit: usize) -> Result<usize, CborError> {
		if additional_information == 31 {
			return Err(CborError::new(format!(
				"Indefinite-length CBOR {kind}s are not supported"
			)));
		}
		let length = self.read_argument(additional_information)?;
		if length > limit as u64 {
			return Err(CborError::new(format!(
				"CBOR {kind} length exceeds configured limit of {limit}"
			)));
		}
		Ok(length as usize)
	}

	fn read_argument(&mut self, additional_information: u32) -> Result<u64, CborError> {
		if additional_information < 24 {
			return Ok(additional_information as u64);
		}
		match additional_information {
			24 => Ok(self.read_byte()? as u64),
			25 => {
				let bytes = self.read_bytes(2)?;
				Ok(u16::from_be_bytes(bytes.try_into().unwrap()) as u64)
			}
			26 => {
				let bytes = self.read_bytes(4)?;
				Ok(u32::from_be_bytes(bytes.try_into().unwrap()) as u64)
			}
			27 => {
				// Upstream reads two 32-bit halves and rejects a high half that would
				// push the result past 2^53-1 (the safe-integer range).
				let high = self.read_argument(26)?;
				let low = self.read_argument(26)?;
				if high > 0x1f_ffff {
					return Err(CborError::new("Decoded CBOR integer or length is outside the safe range"));
				}
				Ok(high * UINT32_BASE + low)
			}
			31 => Err(CborError::new("Indefinite-length CBOR items are not supported")),
			_ => Err(CborError::new("Malformed CBOR additional information")),
		}
	}

	fn read_byte(&mut self) -> Result<u8, CborError> {
		if self.offset >= self.bytes.len() {
			return Err(CborError::new("Truncated CBOR payload"));
		}
		let value = self.bytes[self.offset];
		self.offset += 1;
		Ok(value)
	}

	fn read_bytes(&mut self, length: usize) -> Result<&'a [u8], CborError> {
		if length > self.bytes.len() - self.offset {
			return Err(CborError::new("Truncated CBOR payload"));
		}
		let value = &self.bytes[self.offset..self.offset + length];
		self.offset += length;
		Ok(value)
	}
}

/// Decodes exactly one item from the protocol's strict RFC 8949 subset.
pub fn decode_cbor(bytes: &[u8], options: Option<&CborOptions>) -> Result<CborValue, CborError> {
	let resolved = resolve_options(options)?;
	if bytes.len() > resolved.max_byte_length {
		return Err(CborError::new(format!(
			"CBOR byte length exceeds configured limit of {}",
			resolved.max_byte_length
		)));
	}
	CborReader { bytes, offset: 0, options: resolved }.decode()
}

/// Convenience wrapper mirroring upstream's usage in the codec: decode straight
/// into a strict-JSON tree (byte strings are rejected, as upstream's
/// `isJsonValue` guard rejects `Uint8Array` payloads).
pub fn decode_cbor_json(bytes: &[u8], options: Option<&CborOptions>) -> Result<JsonValue, CborError> {
	cbor_to_json(decode_cbor(bytes, options)?)
}
