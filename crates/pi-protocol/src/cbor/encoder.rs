//! Port of `packages/protocol/src/cbor/encoder.ts`.

use super::options::{resolve_options, ResolvedCborOptions, MAX_UINT32};
use super::{CborError, CborOptions, CborValue};
use pi_chord::json::JsonValue;

const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

struct CborWriter {
	buffer: Vec<u8>,
	max_byte_length: usize,
}

impl CborWriter {
	fn new(max_byte_length: usize) -> Self {
		Self { buffer: Vec::with_capacity(max_byte_length.min(256)), max_byte_length }
	}

	fn write_byte(&mut self, value: u8) -> Result<(), CborError> {
		self.ensure_capacity(1)?;
		self.buffer.push(value);
		Ok(())
	}

	fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), CborError> {
		self.ensure_capacity(bytes.len())?;
		self.buffer.extend_from_slice(bytes);
		Ok(())
	}

	fn write_uint16(&mut self, value: u16) -> Result<(), CborError> {
		self.ensure_capacity(2)?;
		self.buffer.extend_from_slice(&value.to_be_bytes());
		Ok(())
	}

	fn write_uint32(&mut self, value: u32) -> Result<(), CborError> {
		self.ensure_capacity(4)?;
		self.buffer.extend_from_slice(&value.to_be_bytes());
		Ok(())
	}

	fn write_uint64(&mut self, value: u64) -> Result<(), CborError> {
		self.ensure_capacity(8)?;
		self.buffer.extend_from_slice(&value.to_be_bytes());
		Ok(())
	}

	fn write_float64(&mut self, value: f64) -> Result<(), CborError> {
		self.ensure_capacity(9)?;
		self.buffer.push(0xfb);
		self.buffer.extend_from_slice(&value.to_bits().to_be_bytes());
		Ok(())
	}

	fn finish(self) -> Vec<u8> {
		self.buffer
	}

	fn ensure_capacity(&mut self, additional_bytes: usize) -> Result<(), CborError> {
		let required = self.buffer.len() + additional_bytes;
		if required > self.max_byte_length {
			return Err(CborError::new(format!(
				"CBOR byte length exceeds configured limit of {}",
				self.max_byte_length
			)));
		}
		if required > self.buffer.capacity() {
			self.buffer.reserve(required - self.buffer.len());
		}
		Ok(())
	}
}

fn write_argument(writer: &mut CborWriter, major_type: u8, value: u64) -> Result<(), CborError> {
	let prefix = major_type << 5;
	if value < 24 {
		writer.write_byte(prefix | value as u8)
	} else if value <= 0xff {
		writer.write_byte(prefix | 24)?;
		writer.write_byte(value as u8)
	} else if value <= 0xffff {
		writer.write_byte(prefix | 25)?;
		writer.write_uint16(value as u16)
	} else if value as u64 <= MAX_UINT32 as u64 {
		writer.write_byte(prefix | 26)?;
		writer.write_uint32(value as u32)
	} else {
		writer.write_byte(prefix | 27)?;
		writer.write_uint64(value)
	}
}

fn encode_text(writer: &mut CborWriter, value: &str, options: &ResolvedCborOptions) -> Result<(), CborError> {
	let bytes = value.as_bytes();
	if bytes.len() > options.max_byte_length {
		return Err(CborError::new(format!(
			"CBOR text string length exceeds configured limit of {}",
			options.max_byte_length
		)));
	}
	// Upstream re-decodes to reject lone surrogates; a Rust `&str` is always valid
	// Unicode by construction, so the round-trip check collapses into the type.
	write_argument(writer, 3, bytes.len() as u64)?;
	writer.write_bytes(bytes)
}

fn encode_value(
	writer: &mut CborWriter,
	value: &CborValue,
	options: &ResolvedCborOptions,
	depth: usize,
) -> Result<(), CborError> {
	if depth > options.max_depth {
		return Err(CborError::new(format!(
			"CBOR nesting depth exceeds configured limit of {}",
			options.max_depth
		)));
	}

	match value {
		CborValue::Null => writer.write_byte(0xf6),
		CborValue::Bool(true) => writer.write_byte(0xf5),
		CborValue::Bool(false) => writer.write_byte(0xf4),
		CborValue::Integer(value) => {
			if !(-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(value) {
				return Err(CborError::new("CBOR integers must be safe JavaScript integers"));
			}
			if *value >= 0 {
				write_argument(writer, 0, *value as u64)
			} else {
				write_argument(writer, 1, (-1 - *value) as u64)
			}
		}
		CborValue::Float(value) => {
			if !value.is_finite() {
				return Err(CborError::new("CBOR numbers must be finite"));
			}
			writer.write_float64(*value)
		}
		CborValue::Bytes(value) => {
			if value.len() > options.max_byte_length {
				return Err(CborError::new(format!(
					"CBOR byte string length exceeds configured limit of {}",
					options.max_byte_length
				)));
			}
			write_argument(writer, 2, value.len() as u64)?;
			writer.write_bytes(value)
		}
		CborValue::Text(value) => encode_text(writer, value, options),
		CborValue::Array(items) => {
			if items.len() > options.max_container_length {
				return Err(CborError::new(format!(
					"CBOR array length exceeds configured limit of {}",
					options.max_container_length
				)));
			}
			write_argument(writer, 4, items.len() as u64)?;
			for item in items {
				// Upstream rejects holes and `undefined` elements; dense owned vectors
				// cannot contain either.
				encode_value(writer, item, options, depth + 1)?;
			}
			Ok(())
		}
		CborValue::Map(entries) => {
			// Upstream skips `undefined` properties while collecting entries; strict
			// JSON trees have no `undefined` variant, so all entries encode.
			if entries.len() > options.max_container_length {
				return Err(CborError::new(format!(
					"CBOR map length exceeds configured limit of {}",
					options.max_container_length
				)));
			}
			write_argument(writer, 5, entries.len() as u64)?;
			for (key, entry_value) in entries {
				encode_text(writer, key, options)?;
				encode_value(writer, entry_value, options, depth + 1)?;
			}
			Ok(())
		}
	}
}

/// Encodes the protocol's strict, definite-length RFC 8949 subset.
pub fn encode_cbor(value: &CborValue, options: Option<&CborOptions>) -> Result<Vec<u8>, CborError> {
	let resolved = resolve_options(options)?;
	let mut writer = CborWriter::new(resolved.max_byte_length);
	encode_value(&mut writer, value, &resolved, 0)?;
	Ok(writer.finish())
}

/// Lifts a strict-JSON tree into the CBOR value model (upstream feeds plain
/// `JsonValue`s straight into `encodeCbor`; `Bytes` is unreachable from JSON).
impl From<&JsonValue> for CborValue {
	fn from(value: &JsonValue) -> Self {
		use serde_json::Value as J;
		match value {
			J::Null => CborValue::Null,
			J::Bool(b) => CborValue::Bool(*b),
			J::Number(n) => {
				if let Some(i) = n.as_i64() {
					// Upstream encodes integral f64s as integers too; serde_json keeps
					// parsed integers as i64/u64 and fractions as f64, matching output.
					if i.unsigned_abs() as u64 <= (i64::MAX as u64) {
						CborValue::Integer(i)
					} else {
						CborValue::Float(n.as_f64().unwrap_or(f64::INFINITY))
					}
				} else if let Some(u) = n.as_u64() {
					CborValue::Float(u as f64)
				} else {
					CborValue::Float(n.as_f64().unwrap_or(f64::INFINITY))
				}
			}
			J::String(s) => CborValue::Text(s.clone()),
			J::Array(items) => CborValue::Array(items.iter().map(CborValue::from).collect()),
			J::Object(map) => CborValue::Map(map.iter().map(|(k, v)| (k.clone(), CborValue::from(v))).collect()),
		}
	}
}

/// Downcasts a decoded CBOR item into a strict-JSON tree. `Bytes` has no JSON
/// counterpart, mirroring upstream where `isJsonValue` rejects `Uint8Array`s.
pub fn cbor_to_json(value: CborValue) -> Result<JsonValue, CborError> {
	use serde_json::Map as JMap;
	Ok(match value {
		CborValue::Null => JsonValue::Null,
		CborValue::Bool(b) => JsonValue::Bool(b),
		CborValue::Integer(i) => JsonValue::Number(i.into()),
		CborValue::Float(f) => {
			JsonValue::Number(serde_json::Number::from_f64(f).ok_or_else(|| CborError::new("CBOR numbers must be finite"))?)
		}
		CborValue::Text(s) => JsonValue::String(s),
		CborValue::Array(items) => {
			JsonValue::Array(items.into_iter().map(cbor_to_json).collect::<Result<_, _>>()?)
		}
		CborValue::Map(entries) => {
			let mut map = JMap::new();
			for (key, value) in entries {
				map.insert(key, cbor_to_json(value)?);
			}
			JsonValue::Object(map)
		}
		CborValue::Bytes(_) => return Err(CborError::new("CBOR byte strings are not JSON values")),
	})
}
