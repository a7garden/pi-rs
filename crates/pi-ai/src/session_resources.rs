//! Port of `packages/ai/src/session-resources.ts` — the process-global
//! session-resource cleanup registry.
//!
//! Upstream throws an `AggregateError` on failure; Rust has no aggregate
//! error, so the port collects every cleanup result and, on any failure,
//! returns/throws the upstream message with all sub-errors joined (deviation
//! D39).

use std::sync::{Arc, Mutex};

/// Upstream `SessionResourceCleanup` — a fire-and-forget cleanup callback.
pub type SessionResourceCleanup = Arc<dyn Fn(Option<&str>) + Send + Sync>;

fn cleanups() -> &'static Mutex<Vec<SessionResourceCleanup>> {
	static CLEANUPS: std::sync::LazyLock<Mutex<Vec<SessionResourceCleanup>>> = std::sync::LazyLock::new(|| Mutex::new(Vec::new()));
	&CLEANUPS
}

/// Upstream `registerSessionResourceCleanup` — returns an unregister closure.
pub fn register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> impl FnOnce() + Send {
	cleanups().lock().unwrap().push(cleanup);
	let registry = cleanups();
	let cleanup = registry_last(registry);
	move || {
		let mut guards = registry.lock().unwrap();
		if let Some(position) = guards.iter().position(|entry| Arc::ptr_eq(entry, &cleanup)) {
			guards.remove(position);
		}
	}
}

fn registry_last(registry: &'static Mutex<Vec<SessionResourceCleanup>>) -> Arc<dyn Fn(Option<&str>) + Send + Sync> {
	registry.lock().unwrap().last().expect("just pushed").clone()
}

/// Upstream `cleanupSessionResources` — every registered cleanup runs even if
/// one fails; all failures are reported together.
pub fn cleanup_session_resources(session_id: Option<&str>) -> Result<(), crate::utils::thrown::ThrownValue> {
	let mut errors: Vec<String> = Vec::new();
	for cleanup in cleanups().lock().unwrap().clone() {
		let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cleanup(session_id)));
		if let Err(panic) = result {
			let message = panic
				.downcast_ref::<String>()
				.cloned()
				.or_else(|| panic.downcast_ref::<&str>().map(|message| message.to_string()))
				.unwrap_or_else(|| "cleanup panicked".to_string());
			errors.push(message);
		}
	}
	if errors.is_empty() {
		Ok(())
	} else {
		Err(crate::utils::thrown::ThrownValue::message(format!(
			"Failed to cleanup session resources: {}",
			errors.join("; ")
		)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The cleanup registry is process-global; serialize the tests that
	/// register into it so the panicking fixture cannot leak into the other.
	static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

	// Port of test-derived registry behavior: register → run → unregister.
	#[test]
	fn registers_runs_and_unregisters_cleanups() {
		let _guard = REGISTRY_LOCK.lock().unwrap();
		let seen: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
		let seen_for_cleanup = seen.clone();
		let unregister = register_session_resource_cleanup(Arc::new(move |session_id| {
			seen_for_cleanup
				.lock()
				.unwrap()
				.push(session_id.map(str::to_string));
		}));

		cleanup_session_resources(Some("session-1")).expect("cleanups succeed");
		assert_eq!(seen.lock().unwrap().clone(), vec![Some("session-1".to_string())]);

		unregister();
		cleanup_session_resources(None).expect("no cleanups left");
		assert_eq!(seen.lock().unwrap().len(), 1);
	}

	#[test]
	fn aggregates_cleanup_failures() {
		let _guard = REGISTRY_LOCK.lock().unwrap();
		let unregister = register_session_resource_cleanup(Arc::new(|_| panic!("boom")));
		// Run first so the failing cleanup is the only registered one.
		let result = cleanup_session_resources(None);
		unregister();
		let error = result.expect_err("failed cleanup");
		assert!(crate::models::thrown_message(&error).starts_with("Failed to cleanup session resources"));
		assert!(crate::models::thrown_message(&error).contains("boom"));
	}
}
