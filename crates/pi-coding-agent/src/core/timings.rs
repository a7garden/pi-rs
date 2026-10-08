//! Port of `packages/coding-agent/src/core/timings.ts` — central timing
//! instrumentation for startup profiling. Enable with the `PI_TIMING=1`
//! environment variable.
//!
//! Deviation D57c — upstream defaults the namespace argument to `"main"`
//! (JS optional parameters); Rust exposes the namespace explicitly, with
//! `*_main` convenience wrappers for the default.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;

use parking_lot::Mutex;

/// Upstream reads `process.env.PI_TIMING === "1"` once at module load.
static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("PI_TIMING").is_ok_and(|value| value == "1"));

/// Default namespace (upstream `TimingLabel = "main" | "extensions"`).
pub const MAIN_NAMESPACE: &str = "main";

struct TimingNamespace {
	timings: Vec<(String, u128)>,
	start: Instant,
}

static TIMING_NAMESPACES: LazyLock<Mutex<HashMap<String, TimingNamespace>>> = LazyLock::new(Mutex::default);

/// Reset a timing namespace (upstream `resetTimings`, default `"main"`).
pub fn reset_timings(namespace: &str) {
	if !*ENABLED {
		return;
	}
	TIMING_NAMESPACES.lock().insert(
		namespace.to_string(),
		TimingNamespace { timings: Vec::new(), start: Instant::now() },
	);
}

/// `resetTimings()` — the default `"main"` namespace.
pub fn reset_timings_main() {
	reset_timings(MAIN_NAMESPACE);
}

/// Record the elapsed time since the namespace's last mark under `label`
/// (upstream `time`, default namespace `"main"`).
pub fn time(label: &str, namespace: &str) {
	if !*ENABLED {
		return;
	}
	let mut namespaces = TIMING_NAMESPACES.lock();
	if !namespaces.contains_key(namespace) {
		namespaces.insert(
			namespace.to_string(),
			TimingNamespace { timings: Vec::new(), start: Instant::now() },
		);
	}
	let timing_namespace = namespaces.get_mut(namespace).unwrap();
	let elapsed = timing_namespace.start.elapsed().as_millis();
	timing_namespace.timings.push((label.to_string(), elapsed));
	timing_namespace.start = Instant::now();
}

/// `time(label)` — the default `"main"` namespace.
pub fn time_main(label: &str) {
	time(label, MAIN_NAMESPACE);
}

fn print_timing_group(title: &str, timings: &[(String, u128)]) {
	if timings.is_empty() {
		return;
	}
	eprintln!("\n--- {title} ---");
	for (label, ms) in timings {
		eprintln!("  {label}: {ms}ms");
	}
	let total: u128 = timings.iter().map(|(_, ms)| ms).sum();
	eprintln!("  TOTAL: {total}ms");
	eprintln!("{}\n", "-".repeat(title.len() + 8));
}

/// Print all timing namespaces to stderr (upstream `printTimings`).
pub fn print_timings() {
	if !*ENABLED {
		return;
	}
	for (namespace, timing_namespace) in TIMING_NAMESPACES.lock().iter() {
		print_timing_group(&format!("Startup Timings: {namespace}"), &timing_namespace.timings);
	}
}
