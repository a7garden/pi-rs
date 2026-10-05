//! Port of `packages/protocol/src/codec.ts`.
//!
//! Upstream gates encoding/decoding with `isJsonValue`/`copyJson` runtime guards
//! on dynamic JS values. In Rust the input is already
//! [`pi_chord::json::JsonValue`] — strict by construction — so those guards
//! collapse into the type (recorded deviation); `copyJson` corresponds to
//! cloning into an owned tree, which the owned `JsonValue` values provide
//! implicitly.

use crate::cbor::{decode_cbor_json, encode_cbor, CborOptions, CborValue};
use crate::framing::{encode_frame, FrameDecoder, FrameDecoderOptions, FrameError, DEFAULT_MAX_FRAME_LENGTH};
use crate::protocol::{parse_client_message, parse_server_message, ProtocolValidationError};
use pi_chord::json::{copy_json, JsonValue};

fn bounded_error_message(error: &(dyn std::error::Error + 'static)) -> String {
	let message = error.to_string();
	if message.chars().count() <= 500 {
		message
	} else {
		let truncated: String = message.chars().take(497).collect();
		format!("{truncated}...")
	}
}

fn max_frame_length(options: Option<&FrameDecoderOptions>) -> usize {
	options.and_then(|o| o.max_frame_length).unwrap_or(DEFAULT_MAX_FRAME_LENGTH)
}

fn encode_protocol_message(
	value: &JsonValue,
	parse: fn(JsonValue) -> Result<JsonValue, ProtocolValidationError>,
	kind: &str,
	options: Option<&FrameDecoderOptions>,
) -> Result<Vec<u8>, ProtocolValidationError> {
	let validated = parse(copy_json(value))?;
	let frame_length = max_frame_length(options);
	encode_frame(
		&encode_cbor(&CborValue::from(&validated), Some(&CborOptions { max_byte_length: Some(frame_length), ..Default::default() }))
			.map_err(|error| {
				ProtocolValidationError::new(format!(
					"Unable to encode {kind} protocol message: {}",
					bounded_error_message(&error)
				))
			})?,
	)
	.map_err(|error| {
		ProtocolValidationError::new(format!(
			"Unable to encode {kind} protocol message: {}",
			bounded_error_message(&error)
		))
	})
}

/// Validates and encodes one complete length-prefixed client message.
pub fn encode_client_message(
	message: &JsonValue,
	options: Option<&FrameDecoderOptions>,
) -> Result<Vec<u8>, ProtocolValidationError> {
	encode_protocol_message(message, parse_client_message, "client", options)
}

/// Validates and encodes one complete length-prefixed server message.
pub fn encode_server_message(
	message: &JsonValue,
	options: Option<&FrameDecoderOptions>,
) -> Result<Vec<u8>, ProtocolValidationError> {
	encode_protocol_message(message, parse_server_message, "server", options)
}

enum MessageKind {
	Client,
	Server,
}

impl MessageKind {
	fn parse(&self, value: JsonValue) -> Result<JsonValue, ProtocolValidationError> {
		match self {
			MessageKind::Client => parse_client_message(value),
			MessageKind::Server => parse_server_message(value),
		}
	}

	fn label(&self) -> &'static str {
		match self {
			MessageKind::Client => "client",
			MessageKind::Server => "server",
		}
	}
}

/// Upstream `ValidatedMessageDecoder<ClientMessage | ServerMessage>`; the
/// element type is [`JsonValue`] in either direction.
struct ValidatedMessageDecoder {
	failed: bool,
	frames: FrameDecoder,
	kind: MessageKind,
	max_frame_length: usize,
}

impl ValidatedMessageDecoder {
	fn new(kind: MessageKind, options: Option<&FrameDecoderOptions>) -> Result<Self, FrameError> {
		Ok(Self {
			failed: false,
			frames: FrameDecoder::new(options)?,
			max_frame_length: max_frame_length(options),
			kind,
		})
	}

	fn push(&mut self, chunk: &[u8]) -> Result<Vec<JsonValue>, ProtocolValidationError> {
		let kind = self.kind.label();
		if self.failed {
			return Err(ProtocolValidationError::new(format!("{kind} message decoder has failed")));
		}
		match self.decode_all(chunk) {
			Ok(messages) => Ok(messages),
			Err(error) => {
				self.failed = true;
				match error {
					DecodeError::Validation(error) => Err(error),
					DecodeError::Framing(error) => Err(ProtocolValidationError::new(format!(
						"Invalid {kind} protocol framing: {}",
						bounded_error_message(&error)
					))),
					DecodeError::Cbor(error) => Err(ProtocolValidationError::new(format!(
						"Invalid {kind} protocol frame: {}",
						bounded_error_message(&error)
					))),
				}
			}
		}
	}

	fn decode_all(&mut self, chunk: &[u8]) -> Result<Vec<JsonValue>, DecodeError> {
		let mut messages = Vec::new();
		for frame in self.frames.push(chunk).map_err(DecodeError::Framing)? {
			let payload =
				decode_cbor_json(&frame, Some(&CborOptions { max_byte_length: Some(self.max_frame_length), ..Default::default() }))
					.map_err(DecodeError::Cbor)?;
			messages.push(self.kind.parse(payload).map_err(DecodeError::Validation)?);
		}
		Ok(messages)
	}

	fn end(&mut self) -> Result<(), ProtocolValidationError> {
		let kind = self.kind.label();
		if self.failed {
			return Err(ProtocolValidationError::new(format!("{kind} message decoder has failed")));
		}
		match self.frames.end() {
			Ok(()) => Ok(()),
			Err(error) => {
				self.failed = true;
				Err(ProtocolValidationError::new(format!(
					"Invalid {kind} protocol framing: {}",
					bounded_error_message(&error)
				)))
			}
		}
	}
}

enum DecodeError {
	Framing(FrameError),
	Cbor(crate::cbor::CborError),
	Validation(ProtocolValidationError),
}

/// Incrementally decodes and validates framed client messages.
pub struct ClientMessageDecoder {
	decoder: ValidatedMessageDecoder,
}

impl ClientMessageDecoder {
	pub fn new(options: Option<&FrameDecoderOptions>) -> Result<Self, FrameError> {
		Ok(Self { decoder: ValidatedMessageDecoder::new(MessageKind::Client, options)? })
	}

	pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<JsonValue>, ProtocolValidationError> {
		self.decoder.push(chunk)
	}

	pub fn end(&mut self) -> Result<(), ProtocolValidationError> {
		self.decoder.end()
	}
}

/// Incrementally decodes and validates framed server messages.
pub struct ServerMessageDecoder {
	decoder: ValidatedMessageDecoder,
}

impl ServerMessageDecoder {
	pub fn new(options: Option<&FrameDecoderOptions>) -> Result<Self, FrameError> {
		Ok(Self { decoder: ValidatedMessageDecoder::new(MessageKind::Server, options)? })
	}

	pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<JsonValue>, ProtocolValidationError> {
		self.decoder.push(chunk)
	}

	pub fn end(&mut self) -> Result<(), ProtocolValidationError> {
		self.decoder.end()
	}
}

#[cfg(test)]
mod tests {
	//! Port of `packages/protocol/test/protocol.test.ts`
	//! ("validated framed protocol APIs" suite).
	use super::*;
	use crate::cbor::{decode_cbor_json, encode_cbor, CborValue};
	use crate::framing::{encode_frame, FrameDecoder, FrameDecoderOptions};
	use crate::protocol::{parse_client_message, parse_server_message, PROTOCOL_VERSION};
	use serde_json::json;

	const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

	fn client_hello() -> JsonValue {
		json!({ "type": "hello", "version": PROTOCOL_VERSION })
	}

	fn server_hello() -> JsonValue {
		json!({ "type": "hello", "version": PROTOCOL_VERSION, "serverId": SERVER_ID })
	}

	fn request() -> JsonValue {
		json!({
			"type": "request",
			"id": "request-1",
			"target": { "serverId": SERVER_ID },
			"call": { "serviceId": "pi.session-directory", "member": "list", "args": [] },
		})
	}

	fn opts(max_frame_length: usize) -> FrameDecoderOptions {
		FrameDecoderOptions { max_frame_length: Some(max_frame_length) }
	}

	#[test]
	fn encodes_complete_client_and_server_frames() {
		let mut decoder = FrameDecoder::new(None).unwrap();
		let client_frames = decoder.push(&encode_client_message(&client_hello(), None).unwrap()).unwrap();
		assert_eq!(
			parse_client_message(decode_cbor_json(&client_frames[0], None).unwrap()).unwrap(),
			client_hello()
		);
		let mut decoder = FrameDecoder::new(None).unwrap();
		let server_frames = decoder.push(&encode_server_message(&server_hello(), None).unwrap()).unwrap();
		assert_eq!(
			parse_server_message(decode_cbor_json(&server_frames[0], None).unwrap()).unwrap(),
			server_hello()
		);
	}

	#[test]
	fn enforces_outbound_frame_limits() {
		assert!(encode_client_message(&client_hello(), Some(&opts(8))).is_err());
		assert!(encode_server_message(&server_hello(), Some(&opts(8))).is_err());
	}

	#[test]
	fn incrementally_decodes_fragmented_and_coalesced_client_messages() {
		let first = encode_client_message(&client_hello(), None).unwrap();
		let second = encode_client_message(&request(), None).unwrap();
		let mut wire = first.clone();
		wire.extend_from_slice(&second);

		for split in 0..=wire.len() {
			let mut decoder = ClientMessageDecoder::new(None).unwrap();
			let mut messages = decoder.push(&wire[..split]).unwrap();
			messages.extend(decoder.push(&wire[split..]).unwrap());
			decoder.end().unwrap();
			assert_eq!(messages, vec![client_hello(), request()]);
		}
	}

	#[test]
	fn incrementally_decodes_fragmented_and_coalesced_server_messages() {
		let response = json!({ "type": "response", "id": "request-1", "ok": true, "result": [] });
		let first = encode_server_message(&server_hello(), None).unwrap();
		let second = encode_server_message(&response, None).unwrap();
		let mut wire = first.clone();
		wire.extend_from_slice(&second);

		let split = first.len() + second.len() / 2;
		let mut decoder = ServerMessageDecoder::new(None).unwrap();
		assert_eq!(decoder.push(&wire[..split]).unwrap(), vec![server_hello()]);
		assert_eq!(decoder.push(&wire[split..]).unwrap(), vec![response]);
		decoder.end().unwrap();
	}

	#[test]
	fn rejects_invalid_framed_input() {
		for wire in [
			encode_frame(&[]).unwrap(),                                        // empty CBOR payload
			encode_frame(&[0xff]).unwrap(),                                    // malformed CBOR
			encode_frame(&encode_cbor(&CborValue::from(&json!({ "type": "hello", "version": 1, "extra": true })), None).unwrap())
				.unwrap(), // schema-invalid CBOR
		] {
			let mut decoder = ClientMessageDecoder::new(None).unwrap();
			assert!(decoder.push(&wire).is_err());
			assert!(decoder
				.push(&encode_client_message(&client_hello(), None).unwrap())
				.unwrap_err()
				.to_string()
				.contains("failed"));
		}
	}

	#[test]
	fn rejects_truncated_and_oversized_framing() {
		let mut truncated = ServerMessageDecoder::new(None).unwrap();
		assert_eq!(truncated.push(&[0, 0, 0, 2, 1]).unwrap(), Vec::<JsonValue>::new());
		assert!(truncated.end().is_err());
		let mut oversized = ClientMessageDecoder::new(Some(&opts(3))).unwrap();
		assert!(oversized.push(&[0, 0, 0, 4]).is_err());
	}
}
