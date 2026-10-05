//! Port of `packages/protocol/src/framing.ts`.

const FRAME_HEADER_LENGTH: usize = 4;
const MAX_UINT32: usize = 0xffff_ffff;

/// Default upper bound for one framed CBOR payload.
pub const DEFAULT_MAX_FRAME_LENGTH: usize = 16 * 1024 * 1024;

/// Upstream `FrameDecoderOptions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameDecoderOptions {
	pub max_frame_length: Option<usize>,
}

/// Upstream `FrameError` (upstream `TypeError`/`RangeError` surfaces collapse
/// into this type; negative/NaN limits are unrepresentable for `usize` —
/// recorded deviation).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct FrameError(String);

impl FrameError {
	fn new(message: impl Into<String>) -> Self {
		Self(message.into())
	}
}

fn resolve_max_frame_length(options: Option<&FrameDecoderOptions>) -> Result<usize, FrameError> {
	let value = options.and_then(|o| o.max_frame_length).unwrap_or(DEFAULT_MAX_FRAME_LENGTH);
	if value > MAX_UINT32 {
		return Err(FrameError::new(format!(
			"maxFrameLength must be an integer between 0 and {MAX_UINT32}"
		)));
	}
	Ok(value)
}

/// Prefixes a payload with its unsigned 32-bit big-endian byte length.
pub fn encode_frame(payload: &[u8]) -> Result<Vec<u8>, FrameError> {
	if payload.len() > MAX_UINT32 {
		return Err(FrameError::new("Frame payload exceeds the unsigned 32-bit length limit"));
	}
	let mut frame = Vec::with_capacity(FRAME_HEADER_LENGTH + payload.len());
	frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
	frame.extend_from_slice(payload);
	Ok(frame)
}

#[derive(Clone, Copy, PartialEq)]
enum DecoderState {
	Open,
	Ended,
	Failed,
}

/// Incrementally splits arbitrary byte chunks into length-prefixed payloads.
///
/// Upstream assembles payloads through 64 KiB internal blocks to avoid
/// re-copying; the Rust port accumulates into one owned `Vec<u8>` per frame,
/// which is observationally identical (payload bytes are copied, never aliased).
pub struct FrameDecoder {
	header: [u8; FRAME_HEADER_LENGTH],
	header_length: usize,
	max_frame_length: usize,
	payload: Vec<u8>,
	expected_payload_length: Option<usize>,
	state: DecoderState,
}

impl FrameDecoder {
	pub fn new(options: Option<&FrameDecoderOptions>) -> Result<Self, FrameError> {
		Ok(Self {
			header: [0; FRAME_HEADER_LENGTH],
			header_length: 0,
			max_frame_length: resolve_max_frame_length(options)?,
			payload: Vec::new(),
			expected_payload_length: None,
			state: DecoderState::Open,
		})
	}

	pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>, FrameError> {
		match self.state {
			DecoderState::Ended => return Err(FrameError::new("Frame decoder has ended")),
			DecoderState::Failed => return Err(FrameError::new("Frame decoder has failed")),
			DecoderState::Open => {}
		}

		let mut frames = Vec::new();
		let mut chunk_offset = 0;
		while chunk_offset < chunk.len() {
			if self.expected_payload_length.is_none() {
				let header_bytes =
					(FRAME_HEADER_LENGTH - self.header_length).min(chunk.len() - chunk_offset);
				self.header[self.header_length..self.header_length + header_bytes]
					.copy_from_slice(&chunk[chunk_offset..chunk_offset + header_bytes]);
				self.header_length += header_bytes;
				chunk_offset += header_bytes;
				if self.header_length < FRAME_HEADER_LENGTH {
					continue;
				}

				let frame_length =
					u32::from_be_bytes(self.header) as usize;
				self.header_length = 0;
				if frame_length > self.max_frame_length {
					self.fail(format!(
						"Frame length {frame_length} exceeds configured limit of {}",
						self.max_frame_length
					))?;
				}
				if frame_length == 0 {
					frames.push(Vec::new());
					continue;
				}
				self.expected_payload_length = Some(frame_length);
				self.payload = Vec::with_capacity(frame_length);
			}

			let expected_payload_length =
				self.expected_payload_length.expect("set immediately above");
			let remaining = expected_payload_length - self.payload.len();
			let payload_bytes = remaining.min(chunk.len() - chunk_offset);
			self.payload.extend_from_slice(&chunk[chunk_offset..chunk_offset + payload_bytes]);
			chunk_offset += payload_bytes;

			if self.payload.len() == expected_payload_length {
				frames.push(std::mem::take(&mut self.payload));
				self.expected_payload_length = None;
			}
		}
		Ok(frames)
	}

	pub fn end(&mut self) -> Result<(), FrameError> {
		match self.state {
			DecoderState::Ended => return Err(FrameError::new("Frame decoder has ended")),
			DecoderState::Failed => return Err(FrameError::new("Frame decoder has failed")),
			DecoderState::Open => {}
		}
		if self.header_length != 0 || self.expected_payload_length.is_some() {
			return self.fail("Truncated frame at end of stream");
		}
		self.state = DecoderState::Ended;
		Ok(())
	}

	fn fail(&mut self, message: impl Into<String>) -> Result<(), FrameError> {
		self.state = DecoderState::Failed;
		self.header_length = 0;
		self.payload = Vec::new();
		self.expected_payload_length = None;
		Err(FrameError::new(message))
	}
}

#[cfg(test)]
mod tests {
	//! Port of `packages/protocol/test/framing.test.ts`.
	//!
	//! Upstream's `rejects invalid maximum frame length` cases for negative,
	//! fractional, and NaN values are unrepresentable for `usize`; the
	//! above-`MAX_UINT32` case is retained.
	use super::*;

	fn concatenate(chunks: &[&[u8]]) -> Vec<u8> {
		chunks.concat()
	}

	#[test]
	fn prefixes_payloads_with_a_four_byte_big_endian_length() {
		assert_eq!(
			encode_frame(&[0xaa, 0xbb, 0xcc]).unwrap(),
			vec![0x00, 0x00, 0x00, 0x03, 0xaa, 0xbb, 0xcc]
		);
		assert_eq!(encode_frame(&[]).unwrap(), vec![0, 0, 0, 0]);
	}

	#[test]
	fn decodes_fragmented_coalesced_and_empty_frames_in_order() {
		let wire = concatenate(&[&encode_frame(&[1, 2, 3]).unwrap(), &encode_frame(&[]).unwrap(), &encode_frame(&[4]).unwrap()]);
		let expected = vec![vec![1, 2, 3], Vec::new(), vec![4]];

		let mut decoder = FrameDecoder::new(None).unwrap();
		let mut frames = Vec::new();
		for byte in &wire {
			frames.extend(decoder.push(&[*byte]).unwrap());
		}
		decoder.end().unwrap();
		assert_eq!(frames, expected);

		let mut coalesced = FrameDecoder::new(None).unwrap();
		assert_eq!(coalesced.push(&wire).unwrap(), frames);
		coalesced.end().unwrap();
	}

	#[test]
	fn assembles_payloads_spanning_multiple_internal_blocks() {
		let payload: Vec<u8> = (0..70_000).map(|index| (index % 251) as u8).collect();
		let wire = encode_frame(&payload).unwrap();
		let mut decoder = FrameDecoder::new(None).unwrap();
		let mut frames = Vec::new();
		frames.extend(decoder.push(&wire[0..101]).unwrap());
		frames.extend(decoder.push(&wire[101..65_541]).unwrap());
		frames.extend(decoder.push(&wire[65_541..]).unwrap());
		decoder.end().unwrap();
		assert_eq!(frames, vec![payload]);
	}

	#[test]
	fn handles_every_split_point_across_a_frame() {
		let wire = encode_frame(&[10, 20, 30, 40]).unwrap();
		for split in 0..=wire.len() {
			let mut decoder = FrameDecoder::new(None).unwrap();
			let mut frames = decoder.push(&wire[..split]).unwrap();
			frames.extend(decoder.push(&wire[split..]).unwrap());
			decoder.end().unwrap();
			assert_eq!(frames, vec![vec![10, 20, 30, 40]]);
		}
	}

	#[test]
	fn copies_payload_bytes_instead_of_retaining_or_aliasing_input_chunks() {
		let mut chunk = encode_frame(&[1, 2, 3]).unwrap();
		let mut decoder = FrameDecoder::new(None).unwrap();
		let frames = decoder.push(&chunk).unwrap();
		chunk.fill(9);
		assert_eq!(frames, vec![vec![1, 2, 3]]);
	}

	#[test]
	fn accepts_empty_chunks_and_a_clean_empty_stream() {
		let mut decoder = FrameDecoder::new(None).unwrap();
		assert_eq!(decoder.push(&[]).unwrap(), Vec::<Vec<u8>>::new());
		decoder.end().unwrap();
	}

	#[test]
	fn rejects_a_truncated_stream_at_end() {
		for wire in [&[0u8, 0, 0][..], &[0, 0, 0, 2, 1][..]] {
			let mut decoder = FrameDecoder::new(None).unwrap();
			assert_eq!(decoder.push(wire).unwrap(), Vec::<Vec<u8>>::new());
			assert!(decoder.end().is_err());
		}
	}

	#[test]
	fn rejects_an_oversized_declared_length_as_soon_as_its_header_is_complete() {
		let mut decoder = FrameDecoder::new(Some(&FrameDecoderOptions { max_frame_length: Some(3) })).unwrap();
		assert!(decoder.push(&[0, 0, 0, 4]).unwrap_err().to_string().contains("limit"));
		assert!(decoder.push(&[1]).unwrap_err().to_string().contains("failed"));
	}

	#[test]
	fn accepts_a_frame_exactly_at_the_configured_maximum() {
		let mut decoder = FrameDecoder::new(Some(&FrameDecoderOptions { max_frame_length: Some(3) })).unwrap();
		assert_eq!(decoder.push(&encode_frame(&[1, 2, 3]).unwrap()).unwrap(), vec![vec![1, 2, 3]]);
		decoder.end().unwrap();
	}

	#[test]
	fn cannot_be_pushed_after_end() {
		let mut decoder = FrameDecoder::new(None).unwrap();
		decoder.end().unwrap();
		assert!(decoder.push(&[]).unwrap_err().to_string().contains("ended"));
		assert!(decoder.end().unwrap_err().to_string().contains("ended"));
	}

	#[test]
	fn rejects_invalid_maximum_frame_length() {
		// Negative, fractional, and NaN limits are unrepresentable for usize.
		let too_large = DEFAULT_MAX_FRAME_LENGTH * 1_000;
		assert!(FrameDecoder::new(Some(&FrameDecoderOptions { max_frame_length: Some(too_large) })).is_err());
	}
}
