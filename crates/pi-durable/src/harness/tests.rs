mod support;

#[path = "tests/chord_guide.rs"]
mod chord_guide;
#[path = "tests/events_lifecycle.rs"]
mod events_lifecycle;
#[path = "tests/events_output.rs"]
mod events_output;
#[path = "tests/live_deltas.rs"]
mod live_deltas;
#[path = "tests/prompt_integration.rs"]
mod prompt_integration;
#[path = "tests/provider_cache.rs"]
mod provider_cache;
#[path = "tests/runtime.rs"]
mod runtime;
#[path = "tests/runtime_compaction.rs"]
mod runtime_compaction;
#[path = "tests/runtime_context.rs"]
mod runtime_context;
#[path = "tests/runtime_generation.rs"]
mod runtime_generation;
#[path = "tests/runtime_registry.rs"]
mod runtime_registry;
#[path = "tests/runtime_tools.rs"]
mod runtime_tools;
#[path = "tests/state.rs"]
mod state;
#[path = "tests/task_graph.rs"]
mod task_graph;
#[path = "tests/task_runtime.rs"]
mod task_runtime;
#[path = "tests/tasks_abort.rs"]
mod tasks_abort;
#[path = "tests/tasks_close.rs"]
mod tasks_close;
#[path = "tests/tasks_phase_runtime.rs"]
mod tasks_phase_runtime;
#[path = "tests/tasks_queue.rs"]
mod tasks_queue;
#[path = "tests/tasks_recovery.rs"]
mod tasks_recovery;
#[path = "tests/tasks_scheduling.rs"]
mod tasks_scheduling;
