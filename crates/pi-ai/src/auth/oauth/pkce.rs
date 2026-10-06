//! Port of `packages/ai/src/auth/oauth/pkce.ts` — PKCE utilities.
//!
//! Upstream uses the Web Crypto API (`crypto.getRandomValues`,
//! `crypto.subtle.digest`); the Rust port uses `getrandom` + `sha2` with the
//! identical base64url (unpadded) encoding.

use base64::Engine as _;
use sha2::{Digest, Sha256};

/// Encode bytes as an unpadded base64url string (upstream `base64urlEncode`).
pub(crate) fn base64url_encode(bytes: &[u8]) -> String {
	base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Random 32-byte verifier + SHA-256 challenge (upstream `generatePKCE`).
pub async fn generate_pkce() -> Result<(String, String), crate::utils::thrown::ThrownValue> {
	// Generate random verifier
	let mut verifier_bytes = [0u8; 32];
	getrandom::getrandom(&mut verifier_bytes).map_err(|_| crate::utils::thrown::ThrownValue::message("Failed to generate random bytes"))?;
	let verifier = base64url_encode(&verifier_bytes);

	// Compute SHA-256 challenge
	let challenge = base64url_encode(Sha256::digest(verifier.as_bytes()).as_slice());

	Ok((verifier, challenge))
}

#[cfg(test)]
mod tests {
	use super::*;

	// Port of the PKCE invariants exercised throughout the upstream oauth tests
	// (verifier round-trips as the state, challenge = S256(verifier)).
	#[tokio::test]
	async fn produces_base64url_verifier_and_s256_challenge() {
		let (verifier, challenge) = generate_pkce().await.expect("pkce");
		assert_eq!(verifier.len(), 43); // 32 bytes → 43 base64url chars
		assert!(!verifier.contains('+') && !verifier.contains('/') && !verifier.contains('='));
		let digest = Sha256::digest(verifier.as_bytes());
		assert_eq!(challenge, base64url_encode(&digest));
	}

	#[tokio::test]
	async fn generates_distinct_values_per_call() {
		let a = generate_pkce().await.expect("pkce a");
		let b = generate_pkce().await.expect("pkce b");
		assert_ne!(a.0, b.0);
		assert_ne!(a.1, b.1);
	}
}
