//! Port of `packages/durable/test/types.test.ts` (runtime-observable subset).
//!
//! The upstream file is dominated by `expectTypeOf` compile-time assertions
//! (branded-ID exclusivity, discriminated-union narrowing). Rust enforces the
//! same invariants statically via distinct newtypes and enums — those cases
//! have no runtime counterpart (D1). The JSON-shape expectations are ported as
//! serde assertions here; the harness-facing cases (`defineTask`, hooks,
//! typed waits) travel with the `harness` module tests.

use super::*;

#[test]
fn brands_numeric_ids_and_serializes_as_bare_numbers() {
    let conversation_id = ConversationId::from_number(1);
    let task_id = TaskId::from_number(4);
    assert_eq!(serde_json::to_string(&task_id).unwrap(), "4");
    let seq = Seq::from_number(1);
    assert_eq!(serde_json::to_string(&seq).unwrap(), "1");
    assert_eq!(ROOT_CONVERSATION_ID, ConversationId::from_number(1));
}

#[test]
fn encodes_discriminator_dependent_fields() {
    let entry_id = EntryId::from_number(2);
    let answer_id = EntryId::from_number(3);
    let conversation_id = ConversationId::from_number(1);
    let submission_id = SubmissionId::from_number(5);
    let document_id = DocumentId::from_number(6);
    let task_id = TaskId::from_number(4);

    let omit = ContextEdit {
        target: entry_id,
        action: ContextEditAction::Omit,
        messages: None,
    };
    assert_eq!(
        serde_json::to_value(&omit).unwrap(),
        serde_json::json!({ "target": 2, "action": "omit" })
    );
    let replace = ContextEdit {
        target: entry_id,
        action: ContextEditAction::Replace,
        messages: Some(Vec::new()),
    };
    assert_eq!(
        serde_json::to_value(&replace).unwrap(),
        serde_json::json!({ "target": 2, "action": "replace", "messages": [] })
    );

    let pending = TaskState {
        status: TaskStateStatus::Pending,
        checkpoint: Some(serde_json::json!({ "phase": "ready" })),
        on: None,
        policy: None,
        outcome: None,
    };
    assert_eq!(
        serde_json::to_value(&pending).unwrap(),
        serde_json::json!({ "status": "pending", "checkpoint": { "phase": "ready" } })
    );
    let terminal = TaskState {
        status: TaskStateStatus::Terminal,
        checkpoint: None,
        on: None,
        policy: None,
        outcome: Some(TaskOutcome::completed(serde_json::json!({ "value": 1 }))),
    };
    assert_eq!(
        serde_json::to_value(&terminal).unwrap(),
        serde_json::json!({
            "status": "terminal",
            "outcome": { "status": "completed", "result": { "value": 1 } }
        })
    );

    let completed_input = SubmissionRecord {
        id: submission_id,
        conversation_id,
        request_id: None,
        submission_type: SubmissionType::Input,
        status: SubmissionStatus::Done,
        entry: Some(entry_id),
        answer: Some(answer_id),
        reason: None,
        detail: None,
    };
    assert_eq!(
        serde_json::to_value(&completed_input).unwrap(),
        serde_json::json!({
            "id": 5, "conversationId": 1, "type": "input",
            "status": "done", "entry": 2, "answer": 3
        })
    );
    let completed_write = SubmissionRecord {
        submission_type: SubmissionType::Write,
        answer: None,
        ..completed_input.clone()
    };
    assert_eq!(
        serde_json::to_value(&completed_write).unwrap(),
        serde_json::json!({
            "id": 5, "conversationId": 1, "type": "write",
            "status": "done", "entry": 2
        })
    );
    // Upstream `SubmissionCreate` for a queued write: no id, no entry.
    let queued_write_create = SubmissionCreate {
        conversation_id,
        request_id: None,
        submission_type: Some(SubmissionType::Write),
        status: Some(SubmissionStatus::Queued),
        entry: None,
        answer: None,
        reason: None,
        detail: None,
    };
    assert_eq!(
        serde_json::to_value(&queued_write_create).unwrap(),
        serde_json::json!({ "conversationId": 1, "type": "write", "status": "queued" })
    );

    let base_content = DocumentContent::Base {
        version: 1,
        value: serde_json::json!({ "count": 1 })
            .as_object()
            .unwrap()
            .clone(),
    };
    assert_eq!(
        serde_json::to_value(&base_content).unwrap(),
        serde_json::json!({ "kind": "base", "version": 1, "value": { "count": 1 } })
    );
    let delta_content = DocumentContent::Delta {
        version: 1,
        ops: vec![serde_json::from_value(serde_json::json!(["s", ["count"], 2])).unwrap()],
    };
    assert_eq!(
        serde_json::to_value(&delta_content).unwrap(),
        serde_json::json!({ "kind": "delta", "version": 1, "ops": [["s", ["count"], 2]] })
    );

    let conversation_document = DocumentCreate {
        id: document_id,
        kind: "test".into(),
        key: None,
        scope: DocumentScope::Conversation { conversation_id },
        history: Some(DocumentHistory::Rewindable),
        fork: Some(DocumentFork::AsOf),
    };
    assert_eq!(
        serde_json::to_value(&conversation_document).unwrap(),
        serde_json::json!({
            "id": 6, "kind": "test",
            "scope": { "kind": "conversation", "conversationId": 1 },
            "history": "rewindable", "fork": "asOf"
        })
    );
    let _ = task_id;
}
