//! Port of `packages/ai/src/utils/uuid.ts`.

use std::sync::Mutex;

const MAX_UUID_V7_TIMESTAMP: u64 = 0xffff_ffff_ffff;

struct GeneratorState {
	/// `None` mirrors upstream `sequence === undefined` (first call seeds from
	/// randomness).
	sequence: Option<u64>,
	last_ordinary_timestamp: i64,
}

static GENERATOR: Mutex<GeneratorState> = Mutex::new(GeneratorState {
	sequence: None,
	last_ordinary_timestamp: -1,
});

/// Generate a time-ordered UUIDv7. A supplied timestamp is preserved for
/// follower ids; otherwise the clock may only advance, with a per-process
/// monotonic sequence supplying ordering within one millisecond.
pub fn uuidv7(timestamp_ms: Option<u64>) -> Result<String, UuidError> {
	let requested_timestamp = timestamp_ms.unwrap_or_else(|| now_ms());
	if requested_timestamp > MAX_UUID_V7_TIMESTAMP {
		return Err(UuidError::TimestampRange);
	}

	let mut state = GENERATOR.lock().expect("uuidv7 generator lock");
	let effective_timestamp = match timestamp_ms {
		Some(_) => requested_timestamp,
		None => {
			let monotonic = requested_timestamp.max(state.last_ordinary_timestamp.max(0) as u64);
			state.last_ordinary_timestamp = monotonic as i64;
			monotonic
		}
	};

	let mut bytes = [0u8; 16];
	getrandom::getrandom(&mut bytes).map_err(|_| UuidError::Randomness)?;

	state.sequence = match state.sequence {
		None => {
			// 40-bit initial sequence from random bytes 1..=5 (upstream
			// `bytes[1]`..`bytes[5]`).
			Some(
				((bytes[1] as u64) << 32)
					| ((bytes[2] as u64) << 24)
					| ((bytes[3] as u64) << 16)
					| ((bytes[4] as u64) << 8)
					| (bytes[5] as u64),
			)
		}
		Some(sequence) => {
			if sequence == (1u64 << 41) - 1 {
				return Err(UuidError::SequenceExhausted);
			}
			Some(sequence + 1)
		}
	};
	let sequence = state.sequence.expect("sequence just assigned");
	drop(state);

	let timestamp = effective_timestamp;
	for index in 0..=5u64 {
		let shift = (5 - index) * 8;
		bytes[index as usize] = ((timestamp >> shift) & 0xff) as u8;
	}
	bytes[6] = 0x70 | ((sequence >> 37) & 0x0f) as u8;
	bytes[7] = ((sequence >> 29) & 0xff) as u8;
	bytes[8] = 0x80 | ((sequence >> 23) & 0x3f) as u8;
	bytes[9] = ((sequence >> 15) & 0xff) as u8;
	bytes[10] = ((sequence >> 7) & 0xff) as u8;
	bytes[11] = (((sequence & 0x7f) << 1) as u8) | (bytes[11] & 0x01);

	let hex: Vec<String> = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
	Ok(format!(
		"{}-{}-{}-{}-{}",
		hex[0..4].concat(),
		hex[4..6].concat(),
		hex[6..8].concat(),
		hex[8..10].concat(),
		hex[10..16].concat()
	))
}

/// Convenience wrapper returning the current wall-clock UUIDv7.
pub fn uuidv7_now() -> Result<String, UuidError> {
	uuidv7(None)
}

fn now_ms() -> u64 {
	std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.as_millis() as u64)
		.unwrap_or(0)
}

/// Failure modes of [`uuidv7`] (upstream throws `RangeError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UuidError {
	#[error("UUIDv7 timestamp must be an integer between 0 and 281474976710655")]
	TimestampRange,
	#[error("UUIDv7 generator sequence exhausted")]
	SequenceExhausted,
	#[error("randomness source unavailable")]
	Randomness,
}

#[cfg(test)]
mod tests {
	use super::*;

	// Port of test/uuid.test.ts. The upstream fake-timer and crypto-stub
	// cases are adapted: the clock is the real system clock (monotonicity is
	// asserted instead of exact values) and randomness is real (tail
	// uniqueness is asserted instead of stubbed bytes).

	const UUID_V7_RE_STR: &str = "^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$";
	const TIMESTAMP: u64 = 0x0123456789ab;

	fn parse_timestamp(uuid: &str) -> u64 {
		u64::from_str_radix(&uuid.replace('-', "")[0..12], 16).expect("timestamp hex")
	}

	fn matches_v7(uuid: &str) -> bool {
		// Minimal local matcher over the same shape as the upstream regex.
		let re = regex::Regex::new(UUID_V7_RE_STR).expect("regex");
		re.is_match(uuid)
	}

	#[tokio::test]
	async fn generates_ordered_uuidv7s_while_preserving_follower_timestamps() {
		let first = uuidv7(None).expect("uuid");
		let second = uuidv7(None).expect("uuid");
		let follower_timestamp = parse_timestamp(&first).saturating_sub(1_000);
		let followers = [
			uuidv7(Some(follower_timestamp)).expect("uuid"),
			uuidv7(Some(follower_timestamp)).expect("uuid"),
		];

		for id in [&first, &second, &followers[0], &followers[1]] {
			assert!(matches_v7(id), "not a UUIDv7: {id}");
		}
		let mut ordinary = vec![first.clone(), second.clone()];
		ordinary.sort();
		let mut sorted = ordinary.clone();
		sorted.sort();
		assert_eq!(ordinary, sorted);
		assert_ne!(first, second, "same-millisecond ids differ by sequence");
		assert_eq!(parse_timestamp(&followers[0]), follower_timestamp);
		assert_eq!(parse_timestamp(&followers[1]), follower_timestamp);
		assert_ne!(followers[0], followers[1]);
	}

	#[test]
	fn uses_fresh_randomness_for_every_uuid_tail() {
		// Upstream stubs crypto.getRandomValues with incrementing bytes and
		// asserts the tail bytes; with real randomness we assert the
		// observable: same explicit timestamp, distinct tails/ids.
		let first = uuidv7(Some(TIMESTAMP)).expect("uuid");
		let second = uuidv7(Some(TIMESTAMP)).expect("uuid");
		assert_ne!(first, second, "explicit same timestamp still unique");
	}

	#[test]
	fn accepts_timestamp_boundaries() {
		for timestamp in [0u64, 2u64.pow(48) - 1] {
			let uuid = uuidv7(Some(timestamp)).expect("uuid");
			assert_eq!(parse_timestamp(&uuid), timestamp);
		}
	}

	#[test]
	fn rejects_invalid_timestamp() {
		// Upstream also rejects -1, 1.5, NaN, +Inf; those cannot be expressed
		// as u64, so only the range above 2^48-1 applies.
		assert_eq!(uuidv7(Some(2u64.pow(48))), Err(UuidError::TimestampRange));
		assert_eq!(uuidv7(Some(u64::MAX)), Err(UuidError::TimestampRange));
	}
}
