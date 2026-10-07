//! Port of `packages/durable/src/testing/assertions.ts` — adapts a host test
//! runner's expectation helper into [`ConformanceAssertions`] without
//! importing any specific test runner.
//!
//! Deviations (module ledger):
//! - D1: upstream `ExpectLike` returns a chainable matchers object; Rust has
//!   no such protocol, so `ExpectLike` is a checker that must fail the test
//!   when its condition is false, and the adapter precomputes the comparison
//!   (equality, subset match, or rejection-message check) before calling it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use super::types::{ConformanceAssertions, RejectedOperation};

/// Upstream `ExpectLike` (D1): a checker receiving a condition and a formatted
/// message; it must fail the current test when the condition is `false`.
pub type ExpectLike = Arc<dyn Fn(bool, String) + Send + Sync>;

/// Whether `actual` contains every entry of `expected` with a matching value
/// (objects recursively, everything else by equality) — upstream
/// `toMatchObject`.
pub fn partial_equal(actual: &Value, expected: &Value) -> bool {
	match (actual, expected) {
		(Value::Object(actual), Value::Object(expected)) => expected.iter().all(|(key, value)| {
			actual
				.get(key)
				.is_some_and(|nested| partial_equal(nested, value))
		}),
		_ => actual == expected,
	}
}

struct ExpectAssertions {
	expect: ExpectLike,
}

impl ExpectAssertions {
	fn check(&self, condition: bool, message: String) {
		(self.expect)(condition, message);
	}
}

impl ConformanceAssertions for ExpectAssertions {
	fn ok(&self, value: bool, message: &str) {
		self.check(value, message.to_string());
	}

	fn strict_equal(&self, actual: &Value, expected: &Value) {
		self.check(
			actual == expected,
			format!("expected {actual} to be {expected}"),
		);
	}

	fn deep_equal(&self, actual: &Value, expected: &Value) {
		self.check(
			actual == expected,
			format!("expected {actual} to equal {expected}"),
		);
	}

	fn partial_deep_equal(&self, actual: &Value, expected: &Value) {
		self.check(
			partial_equal(actual, expected),
			format!("expected {actual} to match {expected}"),
		);
	}

	fn greater_than(&self, actual: i64, expected: i64) {
		self.check(
			actual > expected,
			format!("expected {actual} to be greater than {expected}"),
		);
	}

	fn rejects(
		&self,
		operation: RejectedOperation,
		message_includes: &str,
	) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
		let message_includes = message_includes.to_string();
		Box::pin(async move {
			match operation.await {
				Ok(()) => self.check(false, "expected the operation to reject".to_string()),
				Err(message) => self.check(
					message.contains(&message_includes),
					format!("expected rejection containing {message_includes:?}, got {message:?}"),
				),
			}
		})
	}
}

/// Upstream `createExpectAssertions`: builds the suite assertions from a host
/// runner's [`ExpectLike`] checker.
pub fn create_expect_assertions(expect: ExpectLike) -> Arc<dyn ConformanceAssertions> {
	Arc::new(ExpectAssertions { expect })
}

/// A built-in [`ConformanceAssertions`] that panics — the Rust equivalent of
/// the Vitest `expect` adapter for hosts without one.
pub fn panicking_assertions() -> Arc<dyn ConformanceAssertions> {
	create_expect_assertions(Arc::new(|condition, message| {
		assert!(condition, "{}", message);
	}))
}

/// Convenience for call sites holding typed serializable records.
pub fn to_value<T: serde::Serialize>(value: &T) -> Value {
	serde_json::to_value(value).expect("record serializes")
}
