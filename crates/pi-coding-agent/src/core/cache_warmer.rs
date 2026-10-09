//! Port of `packages/coding-agent/src/core/cache-warmer.ts` — keeps one
//! prompt cache entry alive by re-sending its request with a one-token
//! output cap before the entry expires. `start` replaces any previous run;
//! warm requests never extend the fixed safety windows.
//!
//! Deviations (crate docs D67):
//! - **D67 — ModelRuntime/session seams and the clock.** Upstream holds
//!   `Pick<ModelRuntime, "streamSimple">` and
//!   `Pick<SessionManager, "appendUsage" | "getBranch">`; the warmer takes
//!   the [`CacheWarmStream`] and [`CacheWarmSessionStore`] seams (a blanket
//!   impl covers `Mutex<SessionManager>`; tests inject fakes mirroring
//!   upstream's duck-typed objects, D47 precedent). The stream seam resolves
//!   to the final [`AssistantMessage`] directly — upstream only consumes
//!   `streamSimple(...).result()`. The [`crate::core::model_runtime`] port
//!   implements both seams.
//! - **D67a — `decide` cannot fail.** Upstream try/catches the extension
//!   callback; the ported `ExtensionRunner::emit_cache_warming_decision`
//!   already falls back to pi's decision on handler failure and returns the
//!   action, so the callback type returns the action without a result
//!   channel.
//! - **D67b — timers are tokio tasks.** Upstream `setTimeout` + `unref`
//!   becomes `tokio::time::sleep` in a spawned task; "now" is the tokio
//!   clock, so `nextWarmAt` values are monotonic rather than wall-clock
//!   epoch ms. Upstream's fake-timer suites run here under tokio's paused
//!   clock, driving the private [`CacheWarmer::refresh`] directly exactly
//!   where upstream reaches into `warmer.run`/`warmer.refresh`.
//! - **D67c — small wire shims.** `transformHeaders` (an async hook with no
//!   Rust counterpart yet) is carried as static `headers` in the replay
//!   test; `formatCacheWarmingUsage`'s decimal trim is a hand-rolled strip
//!   with the regex's keep-≥3-decimals semantics; abort-signal *identity* is
//!   not observable, so tests assert fresh-signal behavior through abort
//!   propagation.

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use pi_ai::types::{
	AnyModel, AssistantMessage, CacheRetention, Model, SimpleStreamOptions, StopReason, TranscriptContext, Usage,
};
use pi_ai::utils::abort::{AbortController, AbortSignal};
use pi_ai::utils::provider_env::get_provider_env_value;

use crate::core::session_manager::{SessionEntry, SessionManager, UsageEntry};
use crate::core::settings_manager::CacheWarmingMode;

/// Streaming warming never continues past this long after the real request
/// that started it (`MAX_WARMING_AGE_MS`).
pub const MAX_WARMING_AGE_MS: u64 = 60 * 60_000;
/// Idle warming uses a shorter horizon because continuation estimates become
/// less reliable with age (`MAX_IDLE_WARMING_AGE_MS`).
pub const MAX_IDLE_WARMING_AGE_MS: u64 = 30 * 60_000;
/// A refresh is sent only when it is expected to save at least this many
/// dollars (`CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS`).
pub const CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS: f64 = 0.05;
/// Chance that a real request arrives before the cache entry expires while
/// the agent sits idle. Measured from our own usage; per-session estimates
/// were not better than this constant (`IDLE_CONTINUATION_PROBABILITY`).
pub const IDLE_CONTINUATION_PROBABILITY: f64 = 0.15;

/// Monotonic "now" in milliseconds on the tokio clock (D67b). Anchored per
/// warmer (its construction instant) so paused-clock tests start at zero and
/// separate runtimes never share an epoch.
fn now_ms(epoch: tokio::time::Instant) -> u64 {
	epoch.elapsed().as_millis() as u64
}

/// Refresh at 90% of the TTL while preserving at least ten seconds of margin
/// (`getCacheWarmingDelayMs`).
pub fn get_cache_warming_delay_ms(ttl_ms: u64) -> Option<u64> {
	if ttl_ms <= 10_000 {
		return None;
	}
	Some(1.max(ttl_ms.saturating_mul(9) / 10).min(ttl_ms - 10_000))
}

/// Lifetime of the prompt cache entry a request writes, from the model's
/// `promptCache` tier for the retention the request used. `None` when the
/// model has no lifetime for that tier or caching is off
/// (`getPromptCacheTtlMs`).
pub fn get_prompt_cache_ttl_ms(model: &Model, options: Option<&SimpleStreamOptions>) -> Option<u64> {
	let retention = options
		.and_then(|options| options.cache_retention)
		.unwrap_or_else(|| {
			let env = options.and_then(|options| options.env.as_ref());
			if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
				CacheRetention::Long
			} else {
				CacheRetention::Short
			}
		});
	if retention == CacheRetention::None {
		return None;
	}
	let seconds = model
		.prompt_cache
		.as_ref()
		.and_then(|prompt_cache| match retention {
			CacheRetention::Long => prompt_cache.long,
			_ => prompt_cache.short,
		})?;
	Some(seconds.saturating_mul(1000))
}

/// Whether replaying the request with a one-token output cap leaves its cache
/// entry untouched. Anthropic's budget-based thinking (Claude models without
/// adaptive thinking) derives `budget_tokens` from `max_tokens`; the replay
/// would get a different budget, which Anthropic keys the message cache on,
/// and the model could still think for thousands of tokens (`isReplayable`).
pub fn is_replayable(model: &Model, options: Option<&SimpleStreamOptions>) -> bool {
	let Some(options) = options else { return true };
	if options.reasoning.is_none() || model.api() != "anthropic-messages" {
		return true;
	}
	model.compat.as_ref().and_then(|compat| compat.force_adaptive_thinking) == Some(true)
}

/// Prompt size of the most recent real request on the branch, as reported by
/// the provider (`lastPromptTokens`).
fn last_prompt_tokens(entries: &[SessionEntry]) -> u64 {
	for entry in entries.iter().rev() {
		if let SessionEntry::Message(entry) = entry {
			if let pi_agent::types::AgentMessage::Llm(pi_ai::types::Message::Assistant(message)) = &entry.message {
				let usage = &message.usage;
				return usage.input + usage.cache_read + usage.cache_write;
			}
		}
	}
	0
}

/// `price`: cost of the given token counts at the model's catalog prices
/// (upstream builds a zeroed `Usage` and overrides the given counts).
fn price(model: &Model, input: u64, output: u64, cache_read: u64, cache_write: u64) -> f64 {
	let mut usage = Usage {
		input,
		output,
		cache_read,
		cache_write,
		cache_write_1h: None,
		reasoning: None,
		total_tokens: 0,
		cost: Default::default(),
	};
	pi_ai::models::calculate_cost(&AnyModel::Chat(model.clone()), &mut usage).total
}

/// `CacheWarmingAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CacheWarmingAction {
	#[serde(rename = "warm")]
	Warm,
	#[serde(rename = "stop")]
	Stop,
}

impl CacheWarmingAction {
	pub fn as_str(&self) -> &'static str {
		match self {
			CacheWarmingAction::Warm => "warm",
			CacheWarmingAction::Stop => "stop",
		}
	}
}

/// `"streaming" | "idle"` phase of an active warming run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmingPhase {
	/// `"streaming"` while the agent run that sent the request is still active.
	Streaming,
	Idle,
}

/// Inputs and outcome of one warm-or-stop decision, as shown by `/session`
/// (`CacheWarmingDecision`).
#[derive(Debug, Clone, PartialEq)]
pub struct CacheWarmingDecision {
	pub phase: WarmingPhase,
	/// Price of this refresh: a cache read of the prompt plus one output token.
	pub warm_cost: f64,
	/// Extra price of the next real request if the cache entry is lost.
	pub miss_cost: f64,
	/// Estimated chance that a real request arrives before the entry expires.
	pub continuation_probability: f64,
	/// `continuationProbability * missCost - warmCost`.
	pub expected_savings: f64,
	/// False when the prompt size or the model's prices are unknown.
	pub economics_available: bool,
	/// Pi's decision: `"warm"` when `expectedSavings` is at least $0.05.
	pub action: CacheWarmingAction,
}

/// Fired before each refresh with pi's decision filled in. Everything else an
/// extension might want (model, idle state, context size) is on the context
/// (`CacheWarmingDecisionEvent`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename = "cache_warming_decision")]
pub struct CacheWarmingDecisionEvent {
	/// Price of this refresh: a cache read of the prompt plus one output token.
	#[serde(rename = "warmCost")]
	pub warm_cost: f64,
	/// Extra price of the next real request if the cache entry is lost.
	#[serde(rename = "missCost")]
	pub miss_cost: f64,
	/// Estimated chance that a real request arrives before the entry expires.
	#[serde(rename = "continuationProbability")]
	pub continuation_probability: f64,
	pub action: CacheWarmingAction,
}

/// `CacheWarmingDecisionEventResult`.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CacheWarmingDecisionEventResult {
	/// Override whether this refresh is sent; `"stop"` ends warming until the
	/// next real request.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub action: Option<CacheWarmingAction>,
}

/// `"inactive" | "scheduled" | "refreshing"` (`CacheWarmingStatus.state`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CacheWarmingState {
	#[serde(rename = "inactive")]
	Inactive,
	#[serde(rename = "scheduled")]
	Scheduled,
	#[serde(rename = "refreshing")]
	Refreshing,
}

/// `CacheWarmingStatus`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CacheWarmingStatus {
	pub state: Option<CacheWarmingState>,
	/// Why nothing is scheduled.
	pub reason: Option<String>,
	pub next_warm_at: Option<u64>,
	/// The pending decision, or the decision that stopped warming.
	pub decision: Option<CacheWarmingDecision>,
	/// True when an extension changed `decision.action`.
	pub extension_override: Option<bool>,
}

/// The request whose prompt cache entry should be kept warm, exactly as it
/// was sent (`CacheWarmRequest`).
#[derive(Clone)]
pub struct CacheWarmRequest {
	pub model: Model,
	pub context: TranscriptContext,
	pub options: SimpleStreamOptions,
}

/// The `ModelRuntime.streamSimple` surface the warmer consumes (D67): the
/// final message of a one-token replay, or a best-effort failure string.
pub trait CacheWarmStream: Send + Sync {
	fn stream_simple(
		&self,
		model: &Model,
		context: &TranscriptContext,
		options: SimpleStreamOptions,
	) -> BoxFuture<'static, Result<AssistantMessage, String>>;
}

/// The `SessionManager` surface the warmer consumes (D67): persisted
/// cache-warming usage plus the branch for prompt sizing.
pub trait CacheWarmSessionStore: Send + Sync {
	fn append_usage(
		&self,
		kind: &str,
		provider: &str,
		model: &str,
		usage: Usage,
		note: Option<String>,
	) -> Result<UsageEntry, String>;
	fn get_branch(&self) -> Vec<SessionEntry>;
}

impl CacheWarmSessionStore for Mutex<SessionManager> {
	fn append_usage(
		&self,
		kind: &str,
		provider: &str,
		model: &str,
		usage: Usage,
		note: Option<String>,
	) -> Result<UsageEntry, String> {
		self.lock()
			.append_usage(kind, provider, model, usage, note)
			.map_err(|error| error.to_string())
	}

	fn get_branch(&self) -> Vec<SessionEntry> {
		self.lock().get_branch()
	}
}

/// Extension decision callback (D67a): receives the event, returns the
/// action to act on.
pub type CacheWarmingDecideFn =
	Arc<dyn Fn(CacheWarmingDecisionEvent) -> BoxFuture<'static, CacheWarmingAction> + Send + Sync>;

/// Mutable state of an active run.
struct ActiveRunState {
	phase: WarmingPhase,
	next_warm_at: u64,
	/// Latest safe time to send this refresh, leaving half the original
	/// expiry margin.
	refresh_deadline_at: u64,
	/// Set while a refresh that an extension forced is in flight.
	extension_override: bool,
	timer: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

struct ActiveRun {
	request: CacheWarmRequest,
	/// False once the session's model or messages no longer match the request.
	is_current: Arc<dyn Fn() -> bool + Send + Sync>,
	ttl_ms: u64,
	delay_ms: u64,
	started_at: u64,
	controller: AbortController,
	state: Mutex<ActiveRunState>,
}

struct CacheWarmerInner {
	/// Construction instant; "now" is measured from here (D67b).
	epoch: tokio::time::Instant,
	run: Mutex<Option<Arc<ActiveRun>>>,
	inactive: Mutex<CacheWarmingStatus>,
	streamer: Arc<dyn CacheWarmStream>,
	session: Arc<dyn CacheWarmSessionStore>,
	get_mode: Box<dyn Fn() -> CacheWarmingMode + Send + Sync>,
	/// Lets extensions override `event.action`; failures fall back to pi's
	/// decision (guaranteed by the seam, D67a).
	decide: CacheWarmingDecideFn,
	/// Called with the persisted usage entry after each successful refresh.
	on_warmed: Mutex<Option<Box<dyn FnMut(&UsageEntry) + Send>>>,
}

/// Keeps one prompt cache entry alive by re-sending its request with a
/// one-token output cap before the entry expires (`CacheWarmer`).
#[derive(Clone)]
pub struct CacheWarmer {
	inner: Arc<CacheWarmerInner>,
}

impl CacheWarmer {
	/// Monotonic "now" in milliseconds since this warmer was constructed.
	fn now_ms(&self) -> u64 {
		now_ms(self.inner.epoch)
	}
	/// `new(models, sessionManager, getMode)` — `decide` defaults to the
	/// event's own action.
	pub fn new(
		streamer: Arc<dyn CacheWarmStream>,
		session: Arc<dyn CacheWarmSessionStore>,
		get_mode: impl Fn() -> CacheWarmingMode + Send + Sync + 'static,
	) -> CacheWarmer {
		CacheWarmer::with_decide(streamer, session, get_mode, Arc::new(|event| Box::pin(async move { event.action })))
	}

	/// `new(models, sessionManager, getMode, decide)`.
	pub fn with_decide(
		streamer: Arc<dyn CacheWarmStream>,
		session: Arc<dyn CacheWarmSessionStore>,
		get_mode: impl Fn() -> CacheWarmingMode + Send + Sync + 'static,
		decide: CacheWarmingDecideFn,
	) -> CacheWarmer {
		CacheWarmer {
			inner: Arc::new(CacheWarmerInner {
				epoch: tokio::time::Instant::now(),
				run: Mutex::new(None),
				inactive: Mutex::new(CacheWarmingStatus {
					state: Some(CacheWarmingState::Inactive),
					reason: Some("waiting for first request".to_string()),
					next_warm_at: None,
					decision: None,
					extension_override: None,
				}),
				streamer,
				session,
				get_mode: Box::new(get_mode),
				decide,
				on_warmed: Mutex::new(None),
			}),
		}
	}

	/// `onWarmed` assignment.
	pub fn set_on_warmed(&self, on_warmed: Box<dyn FnMut(&UsageEntry) + Send>) {
		*self.inner.on_warmed.lock() = Some(on_warmed);
	}

	/// `get status()`.
	pub fn status(&self) -> CacheWarmingStatus {
		if (self.inner.get_mode)() == CacheWarmingMode::Off {
			return CacheWarmingStatus {
				state: Some(CacheWarmingState::Inactive),
				reason: Some("cache warming disabled".to_string()),
				next_warm_at: None,
				decision: None,
				extension_override: None,
			};
		}
		let Some(run) = self.inner.run.lock().clone() else {
			return self.inner.inactive.lock().clone();
		};
		if !(run.is_current)() {
			return CacheWarmingStatus {
				state: Some(CacheWarmingState::Inactive),
				reason: Some("conversation context changed".to_string()),
				next_warm_at: None,
				decision: None,
				extension_override: None,
			};
		}
		let decision = self.evaluate(&run);
		let refreshing = run.state.lock().timer.lock().is_none();
		if !decision.economics_available && !refreshing {
			return CacheWarmingStatus {
				state: Some(CacheWarmingState::Inactive),
				reason: Some("cache economics unavailable".to_string()),
				next_warm_at: None,
				decision: None,
				extension_override: None,
			};
		}
		let state = run.state.lock();
		CacheWarmingStatus {
			state: Some(if refreshing { CacheWarmingState::Refreshing } else { CacheWarmingState::Scheduled }),
			reason: None,
			next_warm_at: Some(state.next_warm_at),
			decision: Some(decision),
			extension_override: Some(state.extension_override),
		}
	}

	/// Keep the prompt cache entry written by `request` warm while
	/// `is_current` holds (`start`).
	pub fn start(&self, request: CacheWarmRequest, is_current: Arc<dyn Fn() -> bool + Send + Sync>) {
		self.clear_run();
		let mode = (self.inner.get_mode)();
		if mode == CacheWarmingMode::Off {
			self.stop("cache warming disabled", None, None);
			return;
		}
		if !is_replayable(&request.model, Some(&request.options)) {
			self.stop("request cannot be replayed safely", None, None);
			return;
		}
		let Some(ttl_ms) = get_prompt_cache_ttl_ms(&request.model, Some(&request.options)) else {
			let reason = if request.options.cache_retention == Some(CacheRetention::None) {
				"request disabled prompt caching"
			} else {
				"cache lifetime unavailable"
			};
			self.stop(reason, None, None);
			return;
		};
		let Some(delay_ms) = get_cache_warming_delay_ms(ttl_ms) else {
			self.stop("cache lifetime unavailable", None, None);
			return;
		};
		let run = Arc::new(ActiveRun {
			request,
			is_current,
			ttl_ms,
			delay_ms,
			started_at: self.now_ms(),
			controller: AbortController::new(),
			state: Mutex::new(ActiveRunState {
				phase: WarmingPhase::Streaming,
				next_warm_at: 0,
				refresh_deadline_at: 0,
				extension_override: false,
				timer: Mutex::new(None),
			}),
		});
		*self.inner.run.lock() = Some(Arc::clone(&run));
		self.schedule(&run);
	}

	/// `onAgentSettled`.
	pub fn on_agent_settled(&self) {
		let Some(run) = self.inner.run.lock().clone() else { return };
		if (self.inner.get_mode)() == CacheWarmingMode::Streaming {
			self.stop("agent run settled", None, None);
			return;
		}
		run.state.lock().phase = WarmingPhase::Idle;
		let deadline = run.started_at + MAX_IDLE_WARMING_AGE_MS;
		if run.state.lock().next_warm_at > deadline || self.now_ms() >= deadline {
			self.stop("30-minute idle safety limit reached", None, None);
		}
	}

	/// Reconcile an active run after the persisted warming mode changes
	/// (`onModeChanged`).
	pub fn on_mode_changed(&self) {
		let Some(run) = self.inner.run.lock().clone() else { return };
		if let Some(reason) = self.get_mode_stop_reason(&run) {
			self.stop(&reason, None, None);
		}
	}

	/// `cancel`.
	pub fn cancel(&self) {
		self.stop("inactive", None, None);
	}

	/// `clearRun`.
	fn clear_run(&self) {
		let Some(run) = self.inner.run.lock().take() else { return };
		if let Some(timer) = run.state.lock().timer.lock().take() {
			timer.abort();
		}
		run.controller.abort();
	}

	/// `stop`.
	fn stop(&self, reason: &str, decision: Option<CacheWarmingDecision>, extension_override: Option<bool>) {
		self.clear_run();
		*self.inner.inactive.lock() = CacheWarmingStatus {
			state: Some(CacheWarmingState::Inactive),
			reason: Some(reason.to_string()),
			next_warm_at: None,
			decision,
			extension_override,
		};
	}

	/// `schedule`.
	fn schedule(&self, run: &Arc<ActiveRun>) {
		{
			let mut state = run.state.lock();
			state.extension_override = false;
			state.next_warm_at = self.now_ms() + run.delay_ms;
			// A timer can run late after sleep or event-loop blockage. Keep
			// half of the planned pre-expiry margin for that delay and
			// request dispatch; a late refresh is likely a full-price cache
			// write, not a cache warm.
			state.refresh_deadline_at = state.next_warm_at + (run.ttl_ms - run.delay_ms) / 2;
		}
		let phase = run.state.lock().phase;
		let deadline =
			run.started_at + if phase == WarmingPhase::Idle { MAX_IDLE_WARMING_AGE_MS } else { MAX_WARMING_AGE_MS };
		if run.state.lock().next_warm_at > deadline || self.now_ms() >= deadline {
			let reason = if phase == WarmingPhase::Idle {
				"30-minute idle safety limit reached"
			} else {
				"one-hour safety limit reached"
			};
			self.stop(reason, None, None);
			return;
		}
		let delay = run.state.lock().next_warm_at.saturating_sub(self.now_ms());
		let inner = Arc::clone(&self.inner);
		let run_task = Arc::clone(run);
		let timer = tokio::spawn(async move {
			tokio::time::sleep(Duration::from_millis(delay)).await;
			CacheWarmer { inner }.refresh(&run_task).await;
		});
		run.state.lock().timer.lock().replace(timer);
	}

	/// `refresh`.
	async fn refresh(&self, run: &Arc<ActiveRun>) {
		run.state.lock().timer.lock().take();
		if !self.validate_run(run) {
			return;
		}
		if self.refresh_deadline_missed(run) {
			return;
		}
		let decision = self.evaluate(run);
		let initial_action = decision.action;
		let decided = (self.inner.decide)(CacheWarmingDecisionEvent {
			warm_cost: decision.warm_cost,
			miss_cost: decision.miss_cost,
			continuation_probability: decision.continuation_probability,
			action: decision.action,
		});
		let action = decided.await;
		if !self.validate_run(run) || self.refresh_deadline_missed(run) {
			return;
		}
		let extension_override = action != initial_action;
		if action == CacheWarmingAction::Stop {
			let reason = if extension_override {
				"stopped by extension"
			} else if decision.economics_available {
				"expected savings below threshold"
			} else {
				"cache economics unavailable"
			};
			self.stop(reason, Some(decision), Some(extension_override));
			return;
		}

		run.state.lock().extension_override = extension_override;
		// Warm requests replay with a one-token output cap, no retries, and
		// their own abort signal.
		let mut options = run.request.options.clone();
		options.max_tokens = Some(1);
		options.max_retries = Some(0);
		options.signal = Some(run.controller.signal());
		// Cache warming is best-effort and must not affect the active agent
		// run: stream failures skip the usage record and only end this
		// refresh.
		let message = self
			.inner
			.streamer
			.stream_simple(&run.request.model, &run.request.context, options)
			.await;
		if let Ok(message) = message {
			if !self.validate_run(run) {
				return;
			}
			if message.stop_reason != StopReason::Error && message.stop_reason != StopReason::Aborted {
				let entry = self.inner.session.append_usage(
					"cache_warm",
					&message.provider,
					message.response_model.as_deref().unwrap_or(&message.model),
					message.usage.clone(),
					extension_override.then(|| "extension override".to_string()),
				);
				if let Ok(entry) = entry {
					if let Some(on_warmed) = self.inner.on_warmed.lock().as_mut() {
						on_warmed(&entry);
					}
				}
			}
		}
		if self.inner.run.lock().as_ref().is_some_and(|current| Arc::ptr_eq(current, run)) {
			self.schedule(run);
		}
	}

	/// `refreshDeadlineMissed`.
	fn refresh_deadline_missed(&self, run: &ActiveRun) -> bool {
		if self.now_ms() <= run.state.lock().refresh_deadline_at {
			return false;
		}
		self.stop("cache refresh deadline missed", None, None);
		true
	}

	/// `validateRun`.
	fn validate_run(&self, run: &Arc<ActiveRun>) -> bool {
		let is_current_run = self
			.inner
			.run
			.lock()
			.as_ref()
			.is_some_and(|current| Arc::ptr_eq(current, run));
		if !is_current_run {
			return false;
		}
		let reason = self
			.get_mode_stop_reason(run)
			.or_else(|| if !(run.is_current)() { Some("conversation context changed".to_string()) } else { None });
		match reason {
			Some(reason) => {
				self.stop(&reason, None, None);
				false
			}
			None => true,
		}
	}

	/// `getModeStopReason`.
	fn get_mode_stop_reason(&self, run: &ActiveRun) -> Option<String> {
		let mode = (self.inner.get_mode)();
		if mode == CacheWarmingMode::Off {
			return Some("cache warming disabled".to_string());
		}
		if mode == CacheWarmingMode::Streaming && run.state.lock().phase == WarmingPhase::Idle {
			return Some("agent run settled".to_string());
		}
		None
	}

	/// `evaluate`.
	fn evaluate(&self, run: &ActiveRun) -> CacheWarmingDecision {
		let model = &run.request.model;
		let prompt_tokens = last_prompt_tokens(&self.inner.session.get_branch());
		let cache_hit_cost = price(model, 0, 0, prompt_tokens, 0);
		let cache_miss_cost = if model.base.cost.rates.cache_write > 0.0 {
			price(model, 0, 0, 0, prompt_tokens)
		} else {
			price(model, prompt_tokens, 0, 0, 0)
		};
		let warm_cost = price(model, 0, 1, prompt_tokens, 0);
		let miss_cost = (cache_miss_cost - cache_hit_cost).max(0.0);
		let phase = run.state.lock().phase;
		let continuation_probability = if phase == WarmingPhase::Idle { IDLE_CONTINUATION_PROBABILITY } else { 1.0 };
		let economics_available = prompt_tokens > 0 && (cache_hit_cost > 0.0 || cache_miss_cost > 0.0);
		let expected_savings = continuation_probability * miss_cost - warm_cost;
		CacheWarmingDecision {
			phase,
			warm_cost,
			miss_cost,
			continuation_probability,
			expected_savings,
			economics_available,
			action: if expected_savings >= CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS {
				CacheWarmingAction::Warm
			} else {
				CacheWarmingAction::Stop
			},
		}
	}
}

/// `formatDollars`.
fn format_dollars(value: f64) -> String {
	if value < 0.0 {
		format!("-${:.3}", value.abs())
	} else {
		format!("${value:.3}")
	}
}

/// `formatCacheWarmingEconomics`.
fn format_cache_warming_economics(decision: &CacheWarmingDecision) -> String {
	if !decision.economics_available {
		return "cache economics unavailable".to_string();
	}
	let probability = (decision.continuation_probability * 100.0).round() as i64;
	let probability_text = if decision.phase == WarmingPhase::Streaming {
		format!("{probability}% continuation probability while agent is running")
	} else {
		format!("{probability}% continuation probability")
	};
	let comparison = if decision.action == CacheWarmingAction::Warm { ">=" } else { "<" };
	format!(
		"{probability_text}, expected savings {} {} ${:.3}",
		format_dollars(decision.expected_savings),
		comparison,
		CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS
	)
}

/// `formatCacheWarmingDecisionTime`.
fn format_cache_warming_decision_time(next_warm_at: Option<u64>, now: u64) -> String {
	let Some(next_warm_at) = next_warm_at else { return "Decision now".to_string() };
	if next_warm_at <= now {
		return "Decision now".to_string();
	}
	let mut remaining_seconds = (next_warm_at - now).div_ceil(1000);
	let hours = remaining_seconds / 3600;
	remaining_seconds %= 3600;
	let minutes = remaining_seconds / 60;
	let seconds = remaining_seconds % 60;
	let mut parts: Vec<String> = Vec::new();
	if hours > 0 {
		parts.push(format!("{hours}h"));
	}
	if minutes > 0 {
		parts.push(format!("{minutes}m"));
	}
	if seconds > 0 || parts.is_empty() {
		parts.push(format!("{seconds}s"));
	}
	format!("Decision in {}", parts.join(" "))
}

/// One-line status for `/session` (`formatCacheWarmingStatus`).
pub fn format_cache_warming_status(status: &CacheWarmingStatus, now: u64) -> String {
	let inactive = || format!("Inactive ({})", status.reason.as_deref().unwrap_or("unknown reason"));
	let Some(decision) = &status.decision else {
		return inactive();
	};
	// A decision is attached once pi (or an extension) acted on it; "inactive"
	// without one never got that far.
	if status.state == Some(CacheWarmingState::Inactive)
		&& !decision.economics_available
		&& !status.extension_override.unwrap_or(false)
	{
		return inactive();
	}
	let details = if status.extension_override.unwrap_or(false) {
		format!("extension override, {}", format_cache_warming_economics(decision))
	} else {
		format!("{} -> {}", format_cache_warming_economics(decision), decision.action.as_str())
	};
	match status.state {
		Some(CacheWarmingState::Inactive) => format!("Stopped ({details})"),
		Some(CacheWarmingState::Refreshing) => format!("Warming cache ({details})"),
		_ => format!("{} ({details})", format_cache_warming_decision_time(status.next_warm_at, now)),
	}
}

/// One-line transcript text for persisted cache-warming usage
/// (`formatCacheWarmingUsage`).
pub fn format_cache_warming_usage(entry: &UsageEntry) -> String {
	let note = entry.note.as_ref().map(|note| format!(" ({note})")).unwrap_or_default();
	format!("Cache warmed{note}: ${}", format_to_fixed6_trimmed(entry.usage.cost.total))
}

/// JS `value.toFixed(6).replace(/(\.\d{3}\d*?)0+$/, "$1")`: six decimals with
/// trailing zeros stripped, never below three decimals (D67c).
fn format_to_fixed6_trimmed(value: f64) -> String {
	let text = format!("{value:.6}");
	if let Some(dot) = text.find('.') {
		if text.len() - dot - 1 > 3 {
			let min_len = dot + 1 + 3;
			return text[..text.trim_end_matches('0').len().max(min_len)].to_string();
		}
	}
	text
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::core::extensions::loader::{create_extension_runtime, load_extension_from_factory};
	use crate::core::extensions::runner::ExtensionRunner;
	use crate::core::extensions::types::EventBus;
	use pi_ai::providers::all::get_builtin_model;
	use serde_json::{json, Value};
	use std::sync::atomic::{AtomicBool, Ordering};

	type RespondFn = Arc<dyn Fn(&Model) -> BoxFuture<'static, Result<AssistantMessage, String>> + Send + Sync>;
	type DecideFn = Arc<dyn Fn(CacheWarmingDecisionEvent) -> BoxFuture<'static, CacheWarmingAction> + Send + Sync>;

	fn chat_model_with_cache(provider: &str, id: &str, long: u64) -> Model {
		let mut model = get_builtin_model(provider, id).expect("builtin model exists");
		model.prompt_cache = Some(pi_ai::types::ModelPromptCache { short: Some(300), long: Some(long) });
		model
	}

	fn adaptive_model() -> Model {
		chat_model_with_cache("anthropic", "claude-opus-4-6", 3600)
	}

	fn budget_model() -> Model {
		chat_model_with_cache("anthropic", "claude-sonnet-4-5", 3600)
	}

	fn openai_model() -> Model {
		chat_model_with_cache("openai", "gpt-5", 86_400)
	}

	fn unknown_model() -> Model {
		let mut model = adaptive_model();
		model.prompt_cache = None;
		model
	}

	fn warm_usage() -> Usage {
		Usage {
			input: 0,
			output: 1,
			cache_read: 100,
			cache_write: 0,
			cache_write_1h: None,
			reasoning: None,
			total_tokens: 101,
			cost: pi_ai::types::UsageCost {
				input: 0.0,
				output: 0.0,
				cache_read: 0.01,
				cache_write: 0.0,
				total: 0.01,
			},
		}
	}

	fn response(model: &Model, stop_reason: StopReason) -> AssistantMessage {
		serde_json::from_value(json!({
			"role": "assistant",
			"content": [],
			"api": model.api(),
			"provider": model.base.provider,
			"model": model.id(),
			"usage": serde_json::to_value(warm_usage()).unwrap(),
			"stopReason": stop_reason,
			"timestamp": 0,
		}))
		.unwrap()
	}

	fn branch_with_prompt(prompt_tokens: u64) -> Vec<SessionEntry> {
		let model = adaptive_model();
		let mut usage = warm_usage();
		usage.output = 10;
		usage.cache_read = prompt_tokens;
		usage.total_tokens = prompt_tokens + 10;
		vec![serde_json::from_value(json!({
			"type": "message",
			"id": "a",
			"parentId": null,
			"timestamp": "1970-01-01T00:00:00.000Z",
			"message": {
				"role": "assistant",
				"content": [],
				"api": model.api(),
				"provider": model.base.provider,
				"model": model.id(),
				"usage": serde_json::to_value(usage).unwrap(),
				"stopReason": "length",
				"timestamp": 0,
			},
		}))
		.unwrap()]
	}

	/// Recorded `appendUsage` call: kind, provider, model, usage, note.
	type AppendCall = (String, String, String, Usage, Option<String>);

	/// Upstream `fakeRuntime`: duck-typed stream/branch/mode/decide seams.
	struct FakeRuntime {
		warmer: CacheWarmer,
		calls: Arc<Mutex<Vec<(Model, SimpleStreamOptions)>>>,
		events: Arc<Mutex<Vec<CacheWarmingDecisionEvent>>>,
		warmed_entries: Arc<Mutex<Vec<UsageEntry>>>,
		append_calls: Arc<Mutex<Vec<AppendCall>>>,
		last_entry: Arc<Mutex<Option<UsageEntry>>>,
		mode_cell: Arc<Mutex<CacheWarmingMode>>,
		signal: AbortSignal,
	}

	struct SpySession {
		manager: Mutex<SessionManager>,
		branch: Mutex<Vec<SessionEntry>>,
		append_calls: Arc<Mutex<Vec<AppendCall>>>,
		last_entry: Arc<Mutex<Option<UsageEntry>>>,
	}

	impl CacheWarmSessionStore for SpySession {
		fn append_usage(
			&self,
			kind: &str,
			provider: &str,
			model: &str,
			usage: Usage,
			note: Option<String>,
		) -> Result<UsageEntry, String> {
			self.append_calls.lock().push((
				kind.to_string(),
				provider.to_string(),
				model.to_string(),
				usage.clone(),
				note.clone(),
			));
			let entry = self
				.manager
				.lock()
				.append_usage(kind, provider, model, usage, note)
				.map_err(|error| error.to_string())?;
			*self.last_entry.lock() = Some(entry.clone());
			Ok(entry)
		}

		fn get_branch(&self) -> Vec<SessionEntry> {
			self.branch.lock().clone()
		}
	}

	struct TestStreamer {
		calls: Arc<Mutex<Vec<(Model, SimpleStreamOptions)>>>,
		respond: RespondFn,
	}

	impl CacheWarmStream for TestStreamer {
		fn stream_simple(
			&self,
			model: &Model,
			_context: &TranscriptContext,
			options: SimpleStreamOptions,
		) -> BoxFuture<'static, Result<AssistantMessage, String>> {
			self.calls.lock().push((model.clone(), options));
			let respond = Arc::clone(&self.respond);
			let model = model.clone();
			Box::pin(async move { respond(&model).await })
		}
	}

	fn default_respond() -> RespondFn {
		Arc::new(|model: &Model| {
			let model = model.clone();
			Box::pin(async move { Ok(response(&model, StopReason::Length)) })
		})
	}

	fn fake_runtime(
		respond: Option<RespondFn>,
		decide: Option<DecideFn>,
		mode: CacheWarmingMode,
		branch: Vec<SessionEntry>,
	) -> FakeRuntime {
		let calls = Arc::new(Mutex::new(Vec::new()));
		let events = Arc::new(Mutex::new(Vec::new()));
		let warmed_entries = Arc::new(Mutex::new(Vec::new()));
		let append_calls = Arc::new(Mutex::new(Vec::new()));
		let last_entry = Arc::new(Mutex::new(None));
		let mode_cell = Arc::new(Mutex::new(mode));
		let session = Arc::new(SpySession {
			manager: Mutex::new(SessionManager::in_memory(None, None, None).expect("in-memory session")),
			branch: Mutex::new(branch),
			append_calls: Arc::clone(&append_calls),
			last_entry: Arc::clone(&last_entry),
		});
		let streamer = Arc::new(TestStreamer {
			calls: Arc::clone(&calls),
			respond: respond.unwrap_or_else(default_respond),
		});
		let decide_fn: CacheWarmingDecideFn = {
			let events = Arc::clone(&events);
			let decide = decide.clone();
			Arc::new(move |event: CacheWarmingDecisionEvent| {
				let events = Arc::clone(&events);
				let decide = decide.clone();
				Box::pin(async move {
					let fallback = event.action;
					events.lock().push(event.clone());
					match decide {
						Some(decide) => decide(event).await,
						None => fallback,
					}
				})
			})
		};
		let warmer = CacheWarmer::with_decide(
			streamer,
			Arc::clone(&session) as Arc<dyn CacheWarmSessionStore>,
			{
				let mode_cell = Arc::clone(&mode_cell);
				move || *mode_cell.lock()
			},
			decide_fn,
		);
		let warmed_sink = Arc::clone(&warmed_entries);
		warmer.set_on_warmed(Box::new(move |entry| warmed_sink.lock().push(entry.clone())));
		FakeRuntime { warmer, calls, events, warmed_entries, append_calls, last_entry, mode_cell, signal: AbortSignal::new() }
	}

	fn request_for(model: &Model, options: SimpleStreamOptions) -> CacheWarmRequest {
		CacheWarmRequest {
			model: model.clone(),
			context: pi_ai::utils::transcript::normalize_context(&pi_ai::types::Context {
				system_prompt: None,
				messages: Vec::new(),
				tools: None,
			}),
			options,
		}
	}

	fn plain_request(model: &Model) -> CacheWarmRequest {
		request_for(model, SimpleStreamOptions::default())
	}

	fn current() -> Arc<dyn Fn() -> bool + Send + Sync> {
		Arc::new(|| true)
	}

	fn is_current_flag() -> (Arc<AtomicBool>, Arc<dyn Fn() -> bool + Send + Sync>) {
		let flag = Arc::new(AtomicBool::new(true));
		let probe = Arc::clone(&flag);
		(flag, Arc::new(move || probe.load(Ordering::SeqCst)))
	}

	/// `vi.clearAllTimers()`: abort the armed refresh task.
	fn clear_timers(warmer: &CacheWarmer) {
		if let Some(run) = warmer.inner.run.lock().clone() {
			if let Some(timer) = run.state.lock().timer.lock().take() {
				timer.abort();
			}
		}
	}

	/// `vi.advanceTimersByTimeAsync` also drains the microtask/macrotask
	/// queues: tokio's `advance` only fires the timer wheel, the woken tasks
	/// run when the test yields (D67b), so every advance is followed by this
	/// pump.
	async fn pump() {
		for _ in 0..25 {
			tokio::task::yield_now().await;
		}
	}

	/// `vi.advanceTimersByTimeAsync(ms)`. The pre-pump gives freshly spawned
	/// timer tasks their first poll (registering their sleep), the advance
	/// fires the timer wheel, and the post-pump runs the woken refresh tasks.
	async fn advance_timers(warmer: &CacheWarmer, ms: u64) {
		let _ = warmer;
		pump().await;
		tokio::time::advance(Duration::from_millis(ms)).await;
		pump().await;
	}

	// Upstream `derives eligibility and timing from retention and provider
	// behavior`.
	#[test]
	fn derives_eligibility_and_timing_from_retention_and_provider_behavior() {
		let adaptive = adaptive_model();
		let openai = openai_model();
		let unknown = unknown_model();
		let mut long = SimpleStreamOptions::default();
		long.cache_retention = Some(CacheRetention::Long);
		let mut none = SimpleStreamOptions::default();
		none.cache_retention = Some(CacheRetention::None);
		let mut long_env = SimpleStreamOptions::default();
		long_env.env = Some([("PI_CACHE_RETENTION".to_string(), "long".to_string())].into_iter().collect());

		assert_eq!(
			[
				get_prompt_cache_ttl_ms(&adaptive, None),
				get_prompt_cache_ttl_ms(&adaptive, Some(&long)),
				get_prompt_cache_ttl_ms(&adaptive, Some(&none)),
				get_prompt_cache_ttl_ms(&adaptive, Some(&long_env)),
				get_prompt_cache_ttl_ms(&openai, Some(&long)),
				get_prompt_cache_ttl_ms(&unknown, None),
			],
			[Some(300_000), Some(3_600_000), None, Some(3_600_000), Some(86_400_000), None]
		);
		assert_eq!(
			[get_cache_warming_delay_ms(300_000), get_cache_warming_delay_ms(60_000), get_cache_warming_delay_ms(10_000)],
			[Some(270_000), Some(50_000), None]
		);
		let mut reasoning = SimpleStreamOptions::default();
		reasoning.reasoning = Some(pi_ai::types::ThinkingLevel::Medium);
		assert_eq!(
			[
				is_replayable(&budget_model(), Some(&reasoning)),
				is_replayable(&budget_model(), None),
				is_replayable(&adaptive_model(), Some(&reasoning)),
				is_replayable(&openai_model(), Some(&reasoning)),
			],
			[false, true, true, true]
		);
	}

	// Upstream `replays profitable requests and preserves options across
	// repeated refreshes`.
	#[tokio::test(start_paused = true)]
	async fn replays_profitable_requests_and_preserves_options_across_repeated_refreshes() {
		let mut options = SimpleStreamOptions::default();
		options.reasoning = Some(pi_ai::types::ThinkingLevel::High);
		options.session_id = Some("s".to_string());
		// transformHeaders has no Rust counterpart (D67c); static headers
		// stand in for the passthrough assertion.
		options.headers = Some([("x-test".to_string(), Some("1".to_string()))].into_iter().collect());
		let runtime = fake_runtime(None, None, CacheWarmingMode::Idle, branch_with_prompt(100_000));
		let original_signal = runtime.signal.clone();

		runtime.warmer.start(request_for(&adaptive_model(), options), current());
		advance_timers(&runtime.warmer, 270_000).await;

		{
			let calls = runtime.calls.lock();
			assert_eq!(calls.len(), 1);
			let (model, call_options) = &calls[0];
			assert_eq!(model.id(), adaptive_model().id());
			assert_eq!(call_options.reasoning, Some(pi_ai::types::ThinkingLevel::High));
			assert_eq!(call_options.session_id.as_deref(), Some("s"));
			assert_eq!(call_options.max_tokens, Some(1));
			assert_eq!(call_options.max_retries, Some(0));
			assert!(call_options.headers.as_ref().unwrap().contains_key("x-test"));
			// Fresh abort signal per run (D67c): the outer signal stays live.
			assert!(call_options.signal.is_some());
			assert!(!original_signal.aborted());
		}

		{
			let events = runtime.events.lock();
			assert_eq!(events.len(), 1);
			assert_eq!(events[0].continuation_probability, 1.0);
			assert_eq!(events[0].action, CacheWarmingAction::Warm);
			assert!((events[0].miss_cost - 0.575).abs() < 1e-9);
			assert!((events[0].warm_cost - 0.050025).abs() < 1e-9);
		}

		{
			let appends = runtime.append_calls.lock();
			assert_eq!(appends.len(), 1);
			assert_eq!(appends[0].0, "cache_warm");
			assert_eq!(appends[0].1, adaptive_model().base.provider);
			assert_eq!(appends[0].2, adaptive_model().id());
			assert_eq!(appends[0].3, warm_usage());
			assert_eq!(appends[0].4, None);
		}
		// `warmedEntries` equals the appendUsage result.
		{
			let warmed = runtime.warmed_entries.lock();
			let last = runtime.last_entry.lock();
			assert_eq!(warmed.len(), 1);
			assert_eq!(last.as_ref(), Some(&warmed[0]));
		}

		advance_timers(&runtime.warmer, 270_000).await;
		assert_eq!(runtime.calls.lock().len(), 2);
		runtime.warmer.cancel();
	}

	// Upstream `does not issue refreshes after their safe deadline`.
	#[tokio::test(start_paused = true)]
	async fn does_not_issue_refreshes_after_their_safe_deadline() {
		let runtime = fake_runtime(None, None, CacheWarmingMode::Idle, branch_with_prompt(100_000));
		runtime.warmer.start(plain_request(&adaptive_model()), current());

		// A five-minute cache is scheduled for 4m30s and retains 15 seconds of
		// the 30-second expiry margin. Simulate a timer delayed by sleep:
		// drop the armed timer first (clearAllTimers), then move the clock
		// past the deadline and drive refresh directly.
		clear_timers(&runtime.warmer);
		advance_timers(&runtime.warmer, 285_001).await;
		let run = runtime.warmer.inner.run.lock().clone().expect("active cache-warming run");
		runtime.warmer.refresh(&run).await;

		assert_eq!(runtime.calls.lock().len(), 0);
		let status = runtime.warmer.status();
		assert_eq!(status.state, Some(CacheWarmingState::Inactive));
		assert_eq!(status.reason.as_deref(), Some("cache refresh deadline missed"));
	}

	// Upstream `rechecks the deadline after an extension decision`.
	#[tokio::test(start_paused = true)]
	async fn rechecks_the_deadline_after_an_extension_decision() {
		let decide: DecideFn = Arc::new(|_event| {
			// Advance the clock from within the decision, like upstream's
			// `await Promise.resolve(); vi.setSystemTime(285_001)`.
			Box::pin(async move {
				tokio::time::advance(Duration::from_millis(285_001)).await;
				CacheWarmingAction::Warm
			})
		});
		let runtime = fake_runtime(None, Some(decide), CacheWarmingMode::Idle, branch_with_prompt(100_000));
		runtime.warmer.start(plain_request(&adaptive_model()), current());
		clear_timers(&runtime.warmer);
		let run = runtime.warmer.inner.run.lock().clone().expect("active cache-warming run");
		runtime.warmer.refresh(&run).await;

		assert_eq!(runtime.calls.lock().len(), 0);
		let status = runtime.warmer.status();
		assert_eq!(status.state, Some(CacheWarmingState::Inactive));
		assert_eq!(status.reason.as_deref(), Some("cache refresh deadline missed"));
	}

	// Upstream `applies economic decisions and extension overrides`.
	#[tokio::test(start_paused = true)]
	async fn applies_economic_decisions_and_extension_overrides() {
		// Unprofitable: a 5k-token prompt at opus prices misses the $0.05 bar.
		let unprofitable = fake_runtime(None, None, CacheWarmingMode::Idle, branch_with_prompt(5_000));
		unprofitable.warmer.start(plain_request(&adaptive_model()), current());
		advance_timers(&unprofitable.warmer, 270_000).await;
		assert_eq!(unprofitable.calls.lock().len(), 0);
		let status = unprofitable.warmer.status();
		assert_eq!(status.state, Some(CacheWarmingState::Inactive));
		let decision = status.decision.as_ref().unwrap();
		assert_eq!(decision.action, CacheWarmingAction::Stop);
		assert!(decision.economics_available);
		assert_eq!(status.extension_override, Some(false));

		let forced_decide: DecideFn =
			Arc::new(|_event| Box::pin(async move { CacheWarmingAction::Warm }));
		let forced = fake_runtime(None, Some(forced_decide), CacheWarmingMode::Idle, branch_with_prompt(5_000));
		forced.warmer.start(plain_request(&adaptive_model()), current());
		advance_timers(&forced.warmer, 270_000).await;
		assert_eq!(forced.calls.lock().len(), 1);
		assert_eq!(forced.warmed_entries.lock()[0].note.as_deref(), Some("extension override"));
		forced.warmer.cancel();

		let vetoed_decide: DecideFn =
			Arc::new(|_event| Box::pin(async move { CacheWarmingAction::Stop }));
		let vetoed = fake_runtime(None, Some(vetoed_decide), CacheWarmingMode::Idle, branch_with_prompt(100_000));
		vetoed.warmer.start(plain_request(&adaptive_model()), current());
		advance_timers(&vetoed.warmer, 270_000).await;
		assert_eq!(vetoed.calls.lock().len(), 0);
		let status = vetoed.warmer.status();
		assert_eq!(status.state, Some(CacheWarmingState::Inactive));
		assert_eq!(status.extension_override, Some(true));

		let unavailable = fake_runtime(None, None, CacheWarmingMode::Idle, branch_with_prompt(0));
		unavailable.warmer.start(plain_request(&adaptive_model()), current());
		let status = unavailable.warmer.status();
		assert_eq!(status.state, Some(CacheWarmingState::Inactive));
		assert_eq!(status.reason.as_deref(), Some("cache economics unavailable"));
		advance_timers(&unavailable.warmer, 270_000).await;
		assert_eq!(unavailable.calls.lock().len(), 0);
	}

	// Upstream `stops for unsupported requests, context changes, and mode
	// changes`.
	#[tokio::test(start_paused = true)]
	async fn stops_for_unsupported_requests_context_changes_and_mode_changes() {
		let runtime = fake_runtime(None, None, CacheWarmingMode::Idle, branch_with_prompt(100_000));
		*runtime.mode_cell.lock() = CacheWarmingMode::Off;
		runtime.warmer.start(plain_request(&adaptive_model()), current());
		assert_eq!(runtime.warmer.status().reason.as_deref(), Some("cache warming disabled"));
		*runtime.mode_cell.lock() = CacheWarmingMode::Idle;
		runtime.warmer.start(plain_request(&unknown_model()), current());
		assert_eq!(runtime.warmer.status().reason.as_deref(), Some("cache lifetime unavailable"));
		let mut reasoning = SimpleStreamOptions::default();
		reasoning.reasoning = Some(pi_ai::types::ThinkingLevel::High);
		runtime.warmer.start(request_for(&budget_model(), reasoning), current());
		assert_eq!(runtime.warmer.status().reason.as_deref(), Some("request cannot be replayed safely"));

		let (flag, is_current) = is_current_flag();
		runtime.warmer.start(plain_request(&adaptive_model()), is_current);
		flag.store(false, Ordering::SeqCst);
		assert_eq!(runtime.warmer.status().reason.as_deref(), Some("conversation context changed"));
		advance_timers(&runtime.warmer, 270_000).await;
		assert_eq!(runtime.calls.lock().len(), 0);

		runtime.warmer.start(plain_request(&adaptive_model()), current());
		*runtime.mode_cell.lock() = CacheWarmingMode::Off;
		advance_timers(&runtime.warmer, 270_000).await;
		assert_eq!(runtime.calls.lock().len(), 0);

		let streaming = fake_runtime(None, None, CacheWarmingMode::Streaming, branch_with_prompt(400_000));
		streaming.warmer.start(plain_request(&adaptive_model()), current());
		streaming.warmer.on_agent_settled();
		assert_eq!(streaming.warmer.status().reason.as_deref(), Some("agent run settled"));
	}

	// Upstream `aborts replaced requests and does not record failed
	// refreshes`.
	#[tokio::test(start_paused = true)]
	async fn aborts_replaced_requests_and_does_not_record_failed_refreshes() {
		let release = Arc::new(tokio::sync::Notify::new());
		let release_handle = Arc::clone(&release);
		let pending: RespondFn = Arc::new(move |_model| {
			let release = Arc::clone(&release_handle);
			Box::pin(async move {
				release.notified().await;
				Ok(response(&adaptive_model(), StopReason::Length))
			})
		});
		let runtime = fake_runtime(Some(pending), None, CacheWarmingMode::Idle, branch_with_prompt(100_000));
		runtime.warmer.start(plain_request(&adaptive_model()), current());
		advance_timers(&runtime.warmer, 270_000).await;
		let first_signal = runtime.calls.lock()[0].1.signal.clone().expect("call signal");
		runtime.warmer.start(plain_request(&adaptive_model()), current());
		// Replacing the run aborts the in-flight refresh's signal.
		assert!(first_signal.aborted());
		release.notify_waiters();
		runtime.warmer.cancel();
		advance_timers(&runtime.warmer, 600_000).await;
		assert_eq!(runtime.calls.lock().len(), 1);

		let failed: RespondFn = Arc::new(|model: &Model| {
			let model = model.clone();
			Box::pin(async move { Ok(response(&model, StopReason::Error)) })
		});
		let failed_runtime = fake_runtime(Some(failed), None, CacheWarmingMode::Idle, branch_with_prompt(100_000));
		failed_runtime.warmer.start(plain_request(&adaptive_model()), current());
		advance_timers(&failed_runtime.warmer, 270_000).await;
		assert_eq!(failed_runtime.append_calls.lock().len(), 0);
		failed_runtime.warmer.cancel();
	}

	// Upstream `formats status and usage entries`.
	#[test]
	fn formats_status_and_usage_entries() {
		let decision = CacheWarmingDecision {
			phase: WarmingPhase::Idle,
			warm_cost: 0.013,
			miss_cost: 0.621,
			continuation_probability: 0.6,
			expected_savings: 0.36,
			economics_available: true,
			action: CacheWarmingAction::Warm,
		};
		assert_eq!(
			format_cache_warming_status(
				&CacheWarmingStatus {
					state: Some(CacheWarmingState::Scheduled),
					reason: None,
					next_warm_at: Some(222_000),
					decision: Some(decision),
					extension_override: None,
				},
				0
			),
			"Decision in 3m 42s (60% continuation probability, expected savings $0.360 >= $0.050 -> warm)"
		);
		let mut usage = warm_usage();
		usage.cost = pi_ai::types::UsageCost {
			input: 0.00004,
			output: 0.00005,
			cache_read: 0.02940725,
			cache_write: 0.0,
			total: 0.02949725,
		};
		let entry = SessionManager::in_memory(None, None, None)
			.unwrap()
			.append_usage(
				"cache_warm",
				&adaptive_model().base.provider,
				&adaptive_model().id(),
				usage,
				Some("extension override".to_string()),
			)
			.unwrap();
		assert_eq!(format_cache_warming_usage(&entry), "Cache warmed (extension override): $0.029497");
	}

	// Upstream `ExtensionRunner.emitCacheWarmingDecision — uses the last
	// extension override`.
	#[tokio::test]
	async fn emit_cache_warming_decision_uses_the_last_extension_override() {
		use crate::core::extensions::types::{ExtensionApiHandle, HandlerFn, Unsubscribe};

		struct NoopBus;
		impl EventBus for NoopBus {
			fn emit(&self, _channel: &str, _data: Value) {}
			fn on(&self, _channel: &str, _handler: crate::core::extensions::types::EventBusHandler) -> Unsubscribe {
				Box::new(|| {})
			}
		}

		fn handler_for(action: &'static str) -> HandlerFn {
			Arc::new(move |_event: Value, _ctx| Box::pin(async move { Ok(Some(json!({ "action": action }))) }))
		}

		let runtime = create_extension_runtime();
		let event_bus: Arc<dyn EventBus> = Arc::new(NoopBus);
		let mut extensions = Vec::new();
		for (name, action) in [("<warm>", "warm"), ("<stop>", "stop")] {
			extensions.push(
				load_extension_from_factory(
					Arc::new(move |pi: ExtensionApiHandle| {
						pi.on("cache_warming_decision", handler_for(action));
						Box::pin(async { Ok(()) })
					}),
					"/tmp",
					Arc::clone(&event_bus),
					Some(runtime.clone()),
					name,
				)
				.await
				.unwrap(),
			);
		}
		let runner = ExtensionRunner::new(
			extensions,
			runtime,
			"/tmp/pi-cache-warmer-test",
			Arc::new(parking_lot::Mutex::new(SessionManager::in_memory(None, None, None).unwrap())),
			crate::core::extensions::types::offline_registry_for_tests(),
		);
		let event = CacheWarmingDecisionEvent {
			warm_cost: 0.05,
			miss_cost: 0.5,
			continuation_probability: 0.15,
			action: CacheWarmingAction::Warm,
		};

		assert_eq!(runner.emit_cache_warming_decision(&event).await, CacheWarmingAction::Stop);
	}
}
