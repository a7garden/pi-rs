//! Port of `packages/durable/src/testing/storage-benchmark.ts` — deterministic
//! representative datasets seeded through only the public [`Storage`] contract,
//! plus the read/write benchmark registries.
//!
//! Deviations (module ledger):
//! - D1: upstream is consumed by a Criterion-style timing harness; the Rust
//!   port is a plain function/registry API with no timing dependency — hosts
//!   measure `run` themselves.
//! - D2: read benchmarks report `i64` with `-1` as the missing-value sentinel,
//!   matching upstream's `?? -1` fallbacks.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use pi_chord::delta::{Op, Seg};
use serde_json::{json, Value};

use crate::types::{
	ConversationId, DocumentBase, DocumentContent, DocumentCreate, DocumentId, DocumentPoint,
	DocumentScope, EntryId, EntryQuery, Seq, Storage, StorageWrite, SubmissionId, SubmissionRecord,
	SubmissionStatus, SubmissionType, TaskRecord, ROOT_CONVERSATION_ID,
};

pub(crate) fn ctx() -> &'static Context {
	&BACKGROUND_CONTEXT
}

const REPLAY_TAILS: [u64; 4] = [0, 16, 128, 1_024];
const HISTORY_SEGMENT_LENGTH: u64 = 128;
const FORK_DEPTH: u64 = 8;
const ENTRIES_PER_FORK: u64 = 32;
const BATCH_SIZE: usize = 100;

/// Upstream `StorageBenchmarkScale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageBenchmarkScale {
	pub name: &'static str,
	pub entry_count: u64,
	pub task_count: u64,
	pub document_count: u64,
}

pub const STORAGE_MEMORY_SCALES: [StorageBenchmarkScale; 2] = [
	StorageBenchmarkScale {
		name: "1k",
		entry_count: 1_000,
		task_count: 200,
		document_count: 200,
	},
	StorageBenchmarkScale {
		name: "10k",
		entry_count: 10_000,
		task_count: 2_000,
		document_count: 2_000,
	},
];

pub const TIMING_SCALE: StorageBenchmarkScale = StorageBenchmarkScale {
	name: "timing",
	entry_count: 1_000,
	task_count: 300,
	document_count: 300,
};

pub fn storage_benchmark_primary_record_count(scale: &StorageBenchmarkScale) -> u64 {
	1 + scale.entry_count
		+ scale.task_count
		+ scale.document_count
		+ REPLAY_TAILS.len() as u64
		+ 1 + FORK_DEPTH * (1 + ENTRIES_PER_FORK)
}

/// Upstream `StorageBenchmarkDataset`.
#[derive(Debug, Clone)]
pub struct StorageBenchmarkDataset {
	pub first_entry_id: EntryId,
	pub filtered_task_count: u64,
	pub exact_document_id: DocumentId,
	pub exact_document_key: String,
	/// Replay document keyed by its upstream tail length.
	pub replay_document_ids: HashMap<u64, DocumentId>,
	pub historical_document_id: DocumentId,
	pub ancient_at: Seq,
	pub recent_at: Seq,
	pub deepest_conversation_id: ConversationId,
	pub ancestor_head_entry_id: EntryId,
}

fn task_json(id: u64, index: u64) -> Value {
	let status = match index % 3 {
		0 => "pending",
		1 => "running",
		_ => "terminal",
	};
	let mut task = json!({
		"id": id,
		"conversationId": ROOT_CONVERSATION_ID.0,
		"kind": if index % 4 == 0 { "benchmark.filtered" } else { "benchmark.other" },
		"version": 1,
		"input": { "index": index },
		"background": index % 5 == 0,
		"abortRequested": index % 7 == 0,
	});
	task["state"] = if status == "terminal" {
		json!({ "status": status, "outcome": { "status": "completed", "result": { "index": index } } })
	} else {
		json!({ "status": status, "checkpoint": { "index": index, "payload": "x".repeat(64) } })
	};
	task
}

fn task(id: u64, index: u64) -> TaskRecord {
	serde_json::from_value(task_json(id, index)).expect("benchmark task deserializes")
}

async fn commit(storage: &Arc<dyn Storage>, writes: Vec<StorageWrite>) -> Seq {
	storage
		.commit(&writes, ctx())
		.await
		.expect("benchmark commit succeeds")
}

fn entry_write(
	id: u64,
	conversation: ConversationId,
	kind: &str,
	data: Value,
	head: bool,
) -> StorageWrite {
	let mut value = json!({
		"id": id,
		"conversationId": conversation.0,
		"kind": kind,
		"data": data,
	});
	if head {
		value["head"] = json!(id);
	}
	serde_json::from_value(json!({ "type": "entry", "value": value }))
		.expect("entry write deserializes")
}

fn set_count_op(count: u64) -> Op {
	Op::set(vec![Seg::key("count")], json!(count)).expect("set op builds")
}

fn document_create(
	id: DocumentId,
	kind: &str,
	key: Option<String>,
	scope: DocumentScope,
	value: Value,
) -> StorageWrite {
	StorageWrite::DocumentCreate {
		record: DocumentCreate {
			id,
			kind: kind.to_string(),
			key,
			scope,
			history: None,
			fork: None,
		},
		content: DocumentBase {
			version: 1,
			value: value.as_object().expect("base object").clone(),
		},
	}
}

/// Seed deterministic representative data through only the public [`Storage`] contract.
pub async fn seed_storage_benchmark(
	storage: &Arc<dyn Storage>,
	scale: &StorageBenchmarkScale,
) -> StorageBenchmarkDataset {
	commit(
		storage,
		vec![StorageWrite::Conversation {
			value: crate::types::ConversationRecord {
				id: ROOT_CONVERSATION_ID,
				parent: None,
				owner: None,
			},
		}],
	)
	.await;

	let mut first_entry_id: Option<EntryId> = None;
	let mut start = 0;
	while start < scale.entry_count as usize {
		let end = usize::min(start + BATCH_SIZE, scale.entry_count as usize);
		let mut writes = Vec::new();
		for index in start..end {
			let id = EntryId(storage.mint_id().await);
			if index == 0 {
				first_entry_id = Some(id);
			}
			writes.push(entry_write(
				id.0,
				ROOT_CONVERSATION_ID,
				"benchmark.entry",
				json!({ "index": index, "text": format!("entry-{index}-{}", "x".repeat(96)) }),
				index == 0,
			));
		}
		commit(storage, writes).await;
		start += BATCH_SIZE;
	}

	let mut start = 0;
	while start < scale.task_count as usize {
		let end = usize::min(start + BATCH_SIZE, scale.task_count as usize);
		let mut writes: Vec<StorageWrite> = Vec::new();
		for index in start..end {
			writes.push(StorageWrite::Task {
				value: task(storage.mint_id().await, index as u64),
			});
		}
		commit(storage, writes).await;
		start += BATCH_SIZE;
	}

	let mut exact_document_id: Option<DocumentId> = None;
	let mut start = 0;
	while start < scale.document_count as usize {
		let end = usize::min(start + BATCH_SIZE, scale.document_count as usize);
		let mut writes = Vec::new();
		for index in start..end {
			let id = DocumentId(storage.mint_id().await);
			exact_document_id = Some(id);
			writes.push(document_create(
				id,
				"benchmark.family",
				Some(format!("key-{index}")),
				DocumentScope::Session,
				json!({ "index": index, "text": "x".repeat(128) }),
			));
		}
		commit(storage, writes).await;
		start += BATCH_SIZE;
	}

	let mut replay_entries: Vec<(u64, DocumentId)> = Vec::new();
	for tail in REPLAY_TAILS {
		replay_entries.push((tail, DocumentId(storage.mint_id().await)));
	}
	commit(
		storage,
		replay_entries
			.iter()
			.map(|(tail, id)| StorageWrite::DocumentCreate {
				record: DocumentCreate {
					id: *id,
					kind: "benchmark.replay".to_string(),
					key: Some(tail.to_string()),
					scope: DocumentScope::Conversation {
						conversation_id: ROOT_CONVERSATION_ID,
					},
					history: Some(crate::types::DocumentHistory::Rewindable),
					fork: Some(crate::types::DocumentFork::AsOf),
				},
				content: DocumentBase {
					version: 1,
					value: json!({ "count": 0, "text": "x".repeat(64) })
						.as_object()
						.expect("base object")
						.clone(),
				},
			})
			.collect(),
	)
	.await;
	for count in 1..=*REPLAY_TAILS.last().expect("non-empty tails") {
		commit(
			storage,
			replay_entries
				.iter()
				.filter(|(tail, _)| count <= *tail)
				.map(|(_, id)| StorageWrite::DocumentChange {
					id: *id,
					content: DocumentContent::Delta {
						version: 1,
						ops: vec![set_count_op(count)],
					},
				})
				.collect(),
		)
		.await;
	}

	let historical_document_id = DocumentId(storage.mint_id().await);
	commit(
		storage,
		vec![StorageWrite::DocumentCreate {
			record: DocumentCreate {
				id: historical_document_id,
				kind: "benchmark.history".to_string(),
				key: None,
				scope: DocumentScope::Conversation {
					conversation_id: ROOT_CONVERSATION_ID,
				},
				history: Some(crate::types::DocumentHistory::Rewindable),
				fork: Some(crate::types::DocumentFork::AsOf),
			},
			content: DocumentBase {
				version: 1,
				value: json!({ "count": 0 })
					.as_object()
					.expect("base object")
					.clone(),
			},
		}],
	)
	.await;
	let mut ancient_at = Seq(0);
	for count in 1..=HISTORY_SEGMENT_LENGTH {
		ancient_at = commit(
			storage,
			vec![StorageWrite::DocumentChange {
				id: historical_document_id,
				content: DocumentContent::Delta {
					version: 1,
					ops: vec![set_count_op(count)],
				},
			}],
		)
		.await;
	}
	commit(
		storage,
		vec![StorageWrite::DocumentChange {
			id: historical_document_id,
			content: DocumentContent::Base {
				version: 1,
				value: json!({ "count": HISTORY_SEGMENT_LENGTH })
					.as_object()
					.expect("base object")
					.clone(),
			},
		}],
	)
	.await;
	let mut recent_at = ancient_at;
	for count in HISTORY_SEGMENT_LENGTH + 1..=HISTORY_SEGMENT_LENGTH * 2 {
		recent_at = commit(
			storage,
			vec![StorageWrite::DocumentChange {
				id: historical_document_id,
				content: DocumentContent::Delta {
					version: 1,
					ops: vec![set_count_op(count)],
				},
			}],
		)
		.await;
	}

	let first_entry_id = first_entry_id.expect("benchmark scale creates entries");
	let mut parent_conversation_id = ROOT_CONVERSATION_ID;
	let mut parent_at = first_entry_id;
	let mut deepest_conversation_id = ROOT_CONVERSATION_ID;
	for depth in 0..FORK_DEPTH {
		let conversation_id = ConversationId(storage.mint_id().await);
		commit(
			storage,
			vec![StorageWrite::Conversation {
				value: crate::types::ConversationRecord {
					id: conversation_id,
					parent: Some(crate::types::ConversationParent {
						conversation_id: parent_conversation_id,
						at: parent_at,
					}),
					owner: None,
				},
			}],
		)
		.await;
		let mut ids: Vec<EntryId> = Vec::new();
		for _ in 0..ENTRIES_PER_FORK {
			ids.push(EntryId(storage.mint_id().await));
		}
		commit(
			storage,
			ids.iter()
				.enumerate()
				.map(|(index, id)| {
					entry_write(
						id.0,
						conversation_id,
						"benchmark.fork",
						json!({ "depth": depth, "index": index }),
						false,
					)
				})
				.collect(),
		)
		.await;
		parent_conversation_id = conversation_id;
		parent_at = *ids.last().expect("fork has entries");
		deepest_conversation_id = conversation_id;
	}

	let exact_document_id = exact_document_id.expect("benchmark scale creates documents");
	StorageBenchmarkDataset {
		first_entry_id,
		filtered_task_count: u64::min(50, scale.task_count.div_ceil(60)),
		exact_document_id,
		exact_document_key: format!("key-{}", scale.document_count - 1),
		replay_document_ids: replay_entries.into_iter().collect(),
		historical_document_id,
		ancient_at,
		recent_at,
		deepest_conversation_id,
		ancestor_head_entry_id: first_entry_id,
	}
}

/// Upstream `StorageReadBenchmark` (D2): `run` reports a count with `-1` for
/// missing values; `expected` computes the dataset-specific expectation.
pub struct StorageReadBenchmark {
	pub name: &'static str,
	pub run:
		fn(Arc<dyn Storage>, &StorageBenchmarkDataset) -> Pin<Box<dyn Future<Output = i64> + Send>>,
	pub expected: fn(&StorageBenchmarkDataset) -> i64,
}

fn document_count(
	storage: &Arc<dyn Storage>,
	id: DocumentId,
	at: DocumentPoint,
) -> Pin<Box<dyn Future<Output = i64> + Send>> {
	let storage = storage.clone();
	Box::pin(async move {
		storage
			.document(id, at, ctx())
			.await
			.and_then(|found| found.value.get("count").and_then(Value::as_i64))
			.unwrap_or(-1)
	})
}

pub const STORAGE_READ_BENCHMARKS: [StorageReadBenchmark; 12] = [
	StorageReadBenchmark {
		name: "exact entry lookup",
		run: |storage, dataset| {
			let storage = storage.clone();
			let id = dataset.first_entry_id;
			Box::pin(async move {
				storage
					.entry(id, ctx())
					.await
					.map(|(entry, _)| entry.id.0 as i64)
					.unwrap_or(-1)
			})
		},
		expected: |dataset| dataset.first_entry_id.0 as i64,
	},
	StorageReadBenchmark {
		name: "entry page scan (100)",
		run: |storage, _dataset| {
			let storage = storage.clone();
			Box::pin(async move {
				storage
					.scan_entries(
						&EntryQuery {
							conversation_id: ROOT_CONVERSATION_ID,
							min_entry_id: None,
							max_entry_id: None,
						},
						100,
						None,
						ctx(),
					)
					.await
					.items
					.len() as i64
			})
		},
		expected: |_dataset| 100,
	},
	StorageReadBenchmark {
		name: "filtered task scan (50)",
		run: |storage, _dataset| {
			let storage = storage.clone();
			Box::pin(async move {
				storage
					.scan_tasks(
						&crate::types::TaskQuery {
							kind: Some("benchmark.filtered".to_string()),
							status: Some(crate::types::TaskStateStatus::Pending),
							background: Some(true),
							..Default::default()
						},
						50,
						None,
						ctx(),
					)
					.await
					.items
					.len() as i64
			})
		},
		expected: |dataset| dataset.filtered_task_count as i64,
	},
	StorageReadBenchmark {
		name: "exact document address among many",
		run: |storage, dataset| {
			let storage = storage.clone();
			let key = dataset.exact_document_key.clone();
			Box::pin(async move {
				storage
					.find_document(
						&crate::types::DocumentAddress {
							kind: "benchmark.family".to_string(),
							key: Some(key),
							scope: DocumentScope::Session,
						},
						DocumentPoint::Current,
						ctx(),
					)
					.await
					.map(|record| record.id.0 as i64)
					.unwrap_or(-1)
			})
		},
		expected: |dataset| dataset.exact_document_id.0 as i64,
	},
	StorageReadBenchmark {
		name: "document replay tail (0)",
		run: |storage, dataset| {
			document_count(
				&storage,
				dataset.replay_document_ids[&0],
				DocumentPoint::Current,
			)
		},
		expected: |_dataset| 0,
	},
	StorageReadBenchmark {
		name: "document replay tail (16)",
		run: |storage, dataset| {
			document_count(
				&storage,
				dataset.replay_document_ids[&16],
				DocumentPoint::Current,
			)
		},
		expected: |_dataset| 16,
	},
	StorageReadBenchmark {
		name: "document replay tail (128)",
		run: |storage, dataset| {
			document_count(
				&storage,
				dataset.replay_document_ids[&128],
				DocumentPoint::Current,
			)
		},
		expected: |_dataset| 128,
	},
	StorageReadBenchmark {
		name: "document replay tail (1024)",
		run: |storage, dataset| {
			document_count(
				&storage,
				dataset.replay_document_ids[&1_024],
				DocumentPoint::Current,
			)
		},
		expected: |_dataset| 1_024,
	},
	StorageReadBenchmark {
		name: "ancient historical read before newer base",
		run: |storage, dataset| {
			document_count(
				&storage,
				dataset.historical_document_id,
				DocumentPoint::At(dataset.ancient_at),
			)
		},
		expected: |_dataset| HISTORY_SEGMENT_LENGTH as i64,
	},
	StorageReadBenchmark {
		name: "recent historical read after newer base",
		run: |storage, dataset| {
			document_count(
				&storage,
				dataset.historical_document_id,
				DocumentPoint::At(dataset.recent_at),
			)
		},
		expected: |_dataset| (HISTORY_SEGMENT_LENGTH * 2) as i64,
	},
	StorageReadBenchmark {
		name: "fork-depth history scan (100)",
		run: |storage, dataset| {
			let storage = storage.clone();
			let conversation = dataset.deepest_conversation_id;
			Box::pin(async move {
				storage
					.scan_entries(
						&EntryQuery {
							conversation_id: conversation,
							min_entry_id: None,
							max_entry_id: None,
						},
						100,
						None,
						ctx(),
					)
					.await
					.items
					.len() as i64
			})
		},
		expected: |_dataset| 100,
	},
	StorageReadBenchmark {
		name: "fork-depth head lookup",
		run: |storage, dataset| {
			let storage = storage.clone();
			let conversation = dataset.deepest_conversation_id;
			Box::pin(async move {
				storage
					.find_latest_head_marker(conversation, None, ctx())
					.await
					.map(|entry| entry.id.0 as i64)
					.unwrap_or(-1)
			})
		},
		expected: |dataset| dataset.ancestor_head_entry_id.0 as i64,
	},
];

/// Upstream `StorageWriteBenchmark`.
pub struct StorageWriteBenchmark {
	pub name: &'static str,
	pub expected: usize,
	pub run: fn(Arc<dyn Storage>) -> Pin<Box<dyn Future<Output = usize> + Send>>,
}

/// Seed the common state expected by every write benchmark sample.
pub async fn seed_storage_write_benchmark(storage: &Arc<dyn Storage>) {
	commit(
		storage,
		vec![StorageWrite::Conversation {
			value: crate::types::ConversationRecord {
				id: ROOT_CONVERSATION_ID,
				parent: None,
				owner: None,
			},
		}],
	)
	.await;
	let mut writes = Vec::new();
	for index in 0..100 {
		let id = storage.mint_id().await;
		writes.push(entry_write(
			id,
			ROOT_CONVERSATION_ID,
			"benchmark.baseline",
			json!({ "index": index }),
			false,
		));
	}
	commit(storage, writes).await;
}

pub const STORAGE_WRITE_BENCHMARKS: [StorageWriteBenchmark; 3] = [
	StorageWriteBenchmark {
		name: "commit one entry",
		expected: 1,
		run: |storage| {
			let storage = storage.clone();
			Box::pin(async move {
				let id = storage.mint_id().await;
				commit(
					&storage,
					vec![entry_write(
						id,
						ROOT_CONVERSATION_ID,
						"benchmark.write",
						json!({ "text": "x".repeat(128) }),
						false,
					)],
				)
				.await;
				1
			})
		},
	},
	StorageWriteBenchmark {
		name: "commit 100 entries",
		expected: 100,
		run: |storage| {
			let storage = storage.clone();
			Box::pin(async move {
				let mut writes = Vec::new();
				for index in 0..100 {
					let id = storage.mint_id().await;
					writes.push(entry_write(
						id,
						ROOT_CONVERSATION_ID,
						"benchmark.write",
						json!({ "index": index, "text": "x".repeat(128) }),
						false,
					));
				}
				let count = writes.len();
				commit(&storage, writes).await;
				count
			})
		},
	},
	StorageWriteBenchmark {
		name: "commit mixed entry/task/submission/document",
		expected: 4,
		run: |storage| {
			let storage = storage.clone();
			Box::pin(async move {
				let entry_id = storage.mint_id().await;
				let task_id = storage.mint_id().await;
				let submission_id = storage.mint_id().await;
				let document_id = storage.mint_id().await;
				let writes = vec![
					entry_write(
						entry_id,
						ROOT_CONVERSATION_ID,
						"benchmark.mixed",
						Value::Null,
						false,
					),
					StorageWrite::Task {
						value: task(task_id, task_id),
					},
					StorageWrite::Submission {
						value: SubmissionRecord {
							id: SubmissionId(submission_id),
							conversation_id: ROOT_CONVERSATION_ID,
							request_id: Some(format!("benchmark-{submission_id}")),
							submission_type: SubmissionType::Write,
							status: SubmissionStatus::Done,
							entry: Some(EntryId(entry_id)),
							answer: None,
							reason: None,
							detail: None,
						},
					},
					document_create(
						DocumentId(document_id),
						"benchmark.mixed",
						Some(document_id.to_string()),
						DocumentScope::Session,
						json!({ "entryId": entry_id, "taskId": task_id }),
					),
				];
				let count = writes.len();
				commit(&storage, writes).await;
				count
			})
		},
	},
];
