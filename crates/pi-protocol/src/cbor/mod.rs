//! Port of `packages/protocol/src/cbor` (upstream: `cbor/index.ts` re-exports).
//!
//! Upstream's codec operates on arbitrary JS values; the Rust port models that
//! input as [`CborValue`]. Byte-level wire format is identical. Upstream's cycle
//! detection, hole/`undefined` element checks, symbol-key checks, and lone-
//! surrogate text checks are unconstructible for owned Rust values and collapse
//! into the type system (recorded deviation).

mod decoder;
mod encoder;
mod options;

pub use decoder::{decode_cbor, decode_cbor_json};
pub use encoder::{cbor_to_json, encode_cbor};
pub use options::{
	resolve_options, CborOptions, ResolvedCborOptions, DEFAULT_MAX_CBOR_BYTE_LENGTH,
	DEFAULT_MAX_CBOR_CONTAINER_LENGTH, DEFAULT_MAX_CBOR_DEPTH, MAX_UINT32, UINT32_BASE,
};

/// Upstream `CborError`.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CborError(String);

impl CborError {
	pub fn new(message: impl Into<String>) -> Self {
		Self(message.into())
	}
}

/// The strict, definite-length RFC 8949 subset the protocol exchanges. Mirrors
/// the value space `encodeCbor`/`decodeCbor` accept upstream.
#[derive(Clone, Debug, PartialEq)]
pub enum CborValue {
	Null,
	Bool(bool),
	Integer(i64),
	Float(f64),
	Bytes(Vec<u8>),
	Text(String),
	Array(Vec<CborValue>),
	Map(Vec<(String, CborValue)>),
}

#[cfg(test)]
mod tests {
	//! Port of `packages/protocol/test/cbor/cbor.test.ts`.
	//!
	//! Upstream cases exercising values unrepresentable in strict JSON / owned
	//! Rust values are intentionally not constructible and are noted inline:
	//! `undefined` top-level/array-element/array-hole, bigint, symbol, function,
	//_ Date, Map, and lone-surrogate text.
	use super::*;

	fn from_hex(hex: &str) -> Vec<u8> {
		assert_eq!(hex.len() % 2, 0, "Hex fixture must contain whole bytes");
		(0..hex.len() / 2)
			.map(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
			.collect()
	}

	fn to_hex(bytes: &[u8]) -> String {
		bytes.iter().map(|byte| format!("{byte:02x}")).collect()
	}

	fn i(value: i64) -> CborValue {
		CborValue::Integer(value)
	}

	fn a(items: &[CborValue]) -> CborValue {
		CborValue::Array(items.to_vec())
	}

	fn m(entries: &[(&str, CborValue)]) -> CborValue {
		CborValue::Map(entries.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
	}

	#[test]
	fn encodes_and_decodes_rfc_8949_vectors() {
		const MAX_SAFE: i64 = 9_007_199_254_740_991;
		const MIN_SAFE: i64 = -9_007_199_254_740_991;
		let known_vectors: Vec<(CborValue, &str)> = vec![
			(CborValue::Null, "f6"),
			(CborValue::Bool(false), "f4"),
			(CborValue::Bool(true), "f5"),
			(i(0), "00"),
			(i(1), "01"),
			(i(10), "0a"),
			(i(23), "17"),
			(i(24), "1818"),
			(i(25), "1819"),
			(i(100), "1864"),
			(i(1000), "1903e8"),
			(i(1_000_000), "1a000f4240"),
			(i(1_000_000_000_000), "1b000000e8d4a51000"),
			(i(MAX_SAFE), "1b001fffffffffffff"),
			(i(-1), "20"),
			(i(-10), "29"),
			(i(-24), "37"),
			(i(-25), "3818"),
			(i(-100), "3863"),
			(i(-1000), "3903e7"),
			(i(-1_000_000), "3a000f423f"),
			(i(MIN_SAFE), "3b001ffffffffffffe"),
			(CborValue::Float(1.1), "fb3ff199999999999a"),
			(CborValue::Float(-0.0), "fb8000000000000000"),
			(CborValue::Bytes(vec![1, 2, 3, 4]), "4401020304"),
			(CborValue::Text(String::new()), "60"),
			(CborValue::Text("IETF".into()), "6449455446"),
			(CborValue::Text("\u{fc}".into()), "62c3bc"),
			(CborValue::Text("\u{6c34}".into()), "63e6b0b4"),
			(CborValue::Text("\u{10151}".into()), "64f0908591"),
			(a(&[]), "80"),
			(a(&[i(1), i(2), i(3)]), "83010203"),
			(a(&[i(1), a(&[i(2), i(3)]), a(&[i(4), i(5)])]), "8301820203820405"),
			(m(&[("a", i(1)), ("b", a(&[i(2), i(3)]))]), "a26161016162820203"),
		];
		for (value, wire) in known_vectors {
			let encoded = encode_cbor(&value, None).unwrap();
			assert_eq!(to_hex(&encoded), wire, "encoding {value:?}");
			let decoded = decode_cbor(&from_hex(wire), None).unwrap();
			assert_eq!(decoded, value, "decoding {wire}");
		}
		// Upstream asserts `-0` round-trips exactly (`Object.is`); check the sign bit.
		let decoded = decode_cbor(&from_hex("fb8000000000000000"), None).unwrap();
		assert_eq!(decoded, CborValue::Float(-0.0));
		assert!(matches!(&decoded, CborValue::Float(f) if f.is_sign_negative()));
	}

	// Upstream "omits undefined object properties without omitting falsey values":
	// strict JSON trees have no `undefined` variant, so the property-omission half
	// is unrepresentable; the falsey-preservation half still holds by round-trip.
	#[test]
	fn preserves_falsey_values() {
		let value = m(&[("zero", i(0)), ("empty", CborValue::Text(String::new())), ("no", CborValue::Bool(false)), ("nil", CborValue::Null)]);
		assert_eq!(decode_cbor(&encode_cbor(&value, None).unwrap(), None).unwrap(), value);
	}

	#[test]
	fn preserves_a_leading_unicode_bom_and_treats_dunder_proto_as_data() {
		assert_eq!(
			decode_cbor(&from_hex("63efbbbf"), None).unwrap(),
			CborValue::Text("\u{feff}".into())
		);
		let value = m(&[("__proto__", CborValue::Text("safe".into()))]);
		assert_eq!(decode_cbor(&encode_cbor(&value, None).unwrap(), None).unwrap(), value);
	}

	#[test]
	fn rejects_unsupported_encoder_values() {
		for value in [
			CborValue::Float(f64::NAN),
			CborValue::Float(f64::INFINITY),
			CborValue::Float(f64::NEG_INFINITY),
			// unsafe positive integer (MAX_SAFE_INTEGER + 1)
			CborValue::Integer(9_007_199_254_740_992),
			// unsafe negative integer (MIN_SAFE_INTEGER - 1)
			CborValue::Integer(-9_007_199_254_740_993),
		] {
			assert!(matches!(encode_cbor(&value, None), Err(CborError(_))), "{value:?}");
		}
		// Upstream also rejects: undefined, holes, bigint, symbol, function, Date,
		// Map, and maps with enumerable symbol keys — all unconstructible here.
	}

	#[test]
	fn rejects_cycles_and_excessive_encoder_depth() {
		// Cycles are impossible for owned values; the depth limit still fires.
		let mut too_deep = CborValue::Null;
		for _ in 0..=DEFAULT_MAX_CBOR_DEPTH {
			too_deep = a(&[too_deep]);
		}
		match encode_cbor(&too_deep, None) {
			Err(CborError(message)) => assert!(message.to_lowercase().contains("depth")),
			Ok(_) => panic!("expected depth error"),
		}
	}

	#[test]
	fn rejects_invalid_decoder_input() {
		let invalid_inputs: Vec<&str> = vec![
			"",             // empty input
			"18",           // truncated integer
			"1c",           // reserved additional information
			"5f",           // indefinite byte string
			"7f",           // indefinite text string
			"9f",           // indefinite array
			"bf",           // indefinite map
			"c000",         // tag
			"f7",           // undefined
			"e0",           // unsupported simple value
			"ff",           // break outside an indefinite item
			"f93c00",       // float16
			"fa3f800000",   // float32
			"fb7ff0000000000000", // positive infinity
			"fb7ff8000000000000", // NaN
			"fb3ff00000",   // truncated float64
			"44010203",     // truncated byte string
			"636162",       // truncated text string
			"8201",         // truncated array
			"a16161",       // truncated map
			"0000",         // trailing data
			"a10102",       // non-string map key
			"a2616101616102", // duplicate map key
			"61ff",         // invalid UTF-8 byte
			"62c080",       // overlong UTF-8
			"63eda080",     // UTF-8 surrogate
			"1b0020000000000000", // unsafe positive integer
			"3b001fffffffffffff", // unsafe negative integer
			"fb4340000000000000", // unsafe integer encoded as float64
		];
		for wire in invalid_inputs {
			assert!(
				matches!(decode_cbor(&from_hex(wire), None), Err(CborError(_))),
				"expected decode error for {wire}"
			);
		}
	}

	#[test]
	fn enforces_depth_and_declared_length_limits_before_traversing_values() {
		let mut too_deep = vec![0xf6_u8];
		for _ in 0..DEFAULT_MAX_CBOR_DEPTH + 1 {
			too_deep.insert(0, 0x81);
		}
		match decode_cbor(&too_deep, None) {
			Err(CborError(message)) => assert!(message.to_lowercase().contains("depth")),
			Ok(_) => panic!("expected depth error"),
		}

		let oversized = |major: u8, limit: usize| {
			from_hex(&format!(
				"{:02x}{:08x}",
				major,
				(limit + 1) as u64
			))
		};
		let cases = [
			oversized(0x5a, DEFAULT_MAX_CBOR_BYTE_LENGTH),
			oversized(0x7a, DEFAULT_MAX_CBOR_BYTE_LENGTH),
			oversized(0x9a, DEFAULT_MAX_CBOR_CONTAINER_LENGTH),
			oversized(0xba, DEFAULT_MAX_CBOR_CONTAINER_LENGTH),
		];
		for wire in &cases {
			match decode_cbor(wire, None) {
				Err(CborError(message)) => assert!(message.to_lowercase().contains("limit")),
				Ok(_) => panic!("expected limit error"),
			}
		}
	}

	#[test]
	fn supports_stricter_caller_provided_limits() {
		let opts = |byte: Option<usize>, container: Option<usize>| CborOptions {
			max_byte_length: byte,
			max_container_length: container,
			max_depth: None,
		};
		let err = decode_cbor(&from_hex("83010203"), Some(&opts(None, Some(2)))).unwrap_err();
		assert!(err.to_string().to_lowercase().contains("limit"));
		let err = decode_cbor(&from_hex("626162"), Some(&opts(Some(2), None))).unwrap_err();
		assert!(err.to_string().to_lowercase().contains("limit"));
		let err = encode_cbor(&a(&[i(1), i(2), i(3)]), Some(&opts(None, Some(2)))).unwrap_err();
		assert!(err.to_string().to_lowercase().contains("limit"));
		let err = encode_cbor(&CborValue::Text("ab".into()), Some(&opts(Some(2), None))).unwrap_err();
		assert!(err.to_string().to_lowercase().contains("limit"));
	}
}
