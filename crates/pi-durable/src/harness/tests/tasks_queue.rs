use serde_json::json;

use super::support::{context, open_default, root};
use crate::harness::inbox::{
    apply_boundary, inbox_doc, prepare_boundary, read_inbox, BoundaryAt, QueueModes,
};
use crate::harness::live::live_doc;
use crate::harness::types::{
    EntryDraft, InputSubmissionDraft, QueueMode, Submission, SubmissionAbort, SubmissionDraft,
    SubmissionStatus, WhenBusy, WriteSubmissionDraft,
};
use crate::types::{ConversationOwnership, Storage, SubmissionQuery, TaskQuery};

fn input(content: &str, request_id: Option<&str>, when_busy: Option<WhenBusy>) -> SubmissionDraft {
    SubmissionDraft::Input(InputSubmissionDraft {
        request_id: request_id.map(str::to_string),
        content: pi_ai::types::UserContentField::Text(content.to_string()),
        when_busy,
    })
}

fn write(kind: &str, request_id: Option<&str>) -> SubmissionDraft {
    SubmissionDraft::Write(WriteSubmissionDraft {
        request_id: request_id.map(str::to_string),
        entry: EntryDraft {
            kind: Some(kind.to_string()),
            ..Default::default()
        },
    })
}

async fn inbox_items(
    conversation: &crate::harness::harness::Conversation,
) -> Vec<crate::harness::inbox::InboxItem> {
    let id = conversation.id;
    conversation
        .commit(
            move |tx| {
                let draft = tx
                    .doc(&inbox_doc(), Some(id.0), None, None)
                    .map_err(|error| error.to_string())?;
                Ok(read_inbox(&draft).items)
            },
            &context(),
        )
        .await
        .expect("read inbox")
}

async fn set_busy(conversation: &crate::harness::harness::Conversation, busy: bool) {
    let id = conversation.id;
    conversation
        .commit(
            move |tx| {
                let draft = tx
                    .doc(&live_doc(), Some(id.0), None, None)
                    .map_err(|error| error.to_string())?;
                if busy {
                    draft.set(
                        &[pi_chord::delta::Seg::key("run")],
                        json!({"taskId": 999, "inputs": []}),
                    )
                } else {
                    draft.delete(&[pi_chord::delta::Seg::key("run")])
                }
            },
            &context(),
        )
        .await
        .expect("set live run state");
}

#[tokio::test]
async fn idle_write_appends_and_settles_without_starting_a_task() {
    let (storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let submission = conversation
        .submit(write("note", None), &context())
        .await
        .expect("submit write");
    let record = submission.wait(&context()).await.expect("settle write");
    assert!(record.entry.is_some());
    let entries = conversation
        .entries(Default::default(), 10, None, &context())
        .await
        .expect("read entries");
    assert_eq!(entries.items.len(), 1);
    assert_eq!(entries.items[0].kind, "note");
    let tasks = storage
        .scan_tasks(&TaskQuery::default(), 10, None, &context())
        .await;
    assert!(
        tasks.items.is_empty(),
        "a passive write must not create a task"
    );
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn request_ids_deduplicate_before_busy_admission_and_are_conversation_scoped() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    set_busy(&conversation, true).await;
    let first = conversation
        .submit(input("first", Some("r1"), None), &context())
        .await
        .expect("submit");
    let again = conversation
        .submit(input("different", Some("r1"), None), &context())
        .await
        .expect("deduplicate");
    assert_eq!(again.id(), first.id());
    assert_eq!(
        first.status(&context()).await.expect("status").status,
        SubmissionStatus::Queued
    );
    let commits_before_rejection = first.id();
    // Upstream `admitSubmission` deduplicates request IDs before the busy
    // check ("Deduplication runs before the busy check"): a same-type
    // duplicate returns the existing submission even when busy-rejecting.
    let deduplicated = conversation
        .submit(
            input("reject", Some("r1"), Some(WhenBusy::Reject)),
            &context(),
        )
        .await
        .expect("deduplicate before busy check");
    assert_eq!(deduplicated.id(), first.id());
    assert!(conversation
        .submit(write("note", Some("r1")), &context())
        .await
        .unwrap_err()
        .contains("already identifies a submission of type input"));
    assert_eq!(
        first
            .status(&context())
            .await
            .expect("status")
            .request_id
            .as_deref(),
        Some("r1")
    );
    assert_eq!(first.id(), commits_before_rejection);
    assert_eq!(inbox_items(&conversation).await.len(), 1);

    let other = harness
        .create_conversation(
            crate::harness::types::ConversationCreateOptions {
                ownership: ConversationOwnership::Ownerless,
                agent: None,
                init: None,
            },
            &context(),
        )
        .await
        .expect("independent conversation");
    let other_write = other
        .submit(write("note", Some("r1")), &context())
        .await
        .expect("other request id");
    assert_ne!(other_write.id(), first.id());
    assert_eq!(
        other
            .submit(write("different", Some("r1")), &context())
            .await
            .expect("dedup")
            .id(),
        other_write.id()
    );
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn busy_submissions_queue_in_order_and_reject_does_not_admit() {
    let (storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    set_busy(&conversation, true).await;
    let follow_up = conversation
        .submit(input("follow", None, None), &context())
        .await
        .expect("queue follow-up");
    let passive = conversation
        .submit(write("note", None), &context())
        .await
        .expect("queue write");
    let steer = conversation
        .submit(input("steer", None, Some(WhenBusy::Steer)), &context())
        .await
        .expect("queue steer");
    for submission in [&follow_up, &passive, &steer] {
        assert_eq!(
            submission.status(&context()).await.expect("status").status,
            SubmissionStatus::Queued
        );
    }
    let queued = inbox_items(&conversation).await;
    assert_eq!(queued.len(), 3);
    assert_eq!(queued[0].id(), follow_up.id());
    assert_eq!(queued[1].id(), passive.id());
    assert_eq!(queued[2].id(), steer.id());
    let before = storage
        .scan_submissions(&SubmissionQuery::default(), 10, None, &context())
        .await
        .items
        .len();
    assert!(conversation
        .submit(input("no", None, Some(WhenBusy::Reject)), &context())
        .await
        .is_err());
    assert_eq!(
        storage
            .scan_submissions(&SubmissionQuery::default(), 10, None, &context())
            .await
            .items
            .len(),
        before
    );
    assert_eq!(inbox_items(&conversation).await.len(), 3);
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn final_boundary_places_writes_first_and_one_follow_up_while_retaining_later_items() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    set_busy(&conversation, true).await;
    let first = conversation
        .submit(input("f1", None, None), &context())
        .await
        .expect("queue first follow-up");
    let write_submission = conversation
        .submit(write("note", None), &context())
        .await
        .expect("queue write");
    let second = conversation
        .submit(input("f2", None, None), &context())
        .await
        .expect("queue second follow-up");
    set_busy(&conversation, false).await;

    let id = conversation.id;
    let result = conversation
        .commit(
            move |tx| {
                let mut boundary = prepare_boundary(
                    tx,
                    id,
                    QueueModes {
                        steering_mode: QueueMode::OneAtATime,
                        follow_up_mode: QueueMode::OneAtATime,
                    },
                )?;
                apply_boundary(tx, &mut boundary, BoundaryAt::Final, 42)
            },
            &context(),
        )
        .await
        .expect("apply final boundary");
    assert_eq!(result.users, [first.id()]);
    // Upstream placement (`applySubmissionChange`) turns a queued input
    // `placed` and a queued write `done` — the write's entry is committed
    // with the boundary, so it settles at once.
    assert_eq!(
        write_submission
            .status(&context())
            .await
            .expect("write status")
            .status,
        SubmissionStatus::Done
    );
    assert_eq!(
        first.status(&context()).await.expect("first status").status,
        SubmissionStatus::Placed
    );
    assert_eq!(
        second
            .status(&context())
            .await
            .expect("second status")
            .status,
        SubmissionStatus::Queued
    );
    assert_eq!(
        inbox_items(&conversation)
            .await
            .iter()
            .map(|item| item.id())
            .collect::<Vec<_>>(),
        [second.id()]
    );
    let entries = conversation
        .entries(Default::default(), 10, None, &context())
        .await
        .expect("entries");
    // Entry scans are newest-first (upstream `EntryQuery`); the write's entry
    // carries the smaller id — writes are placed before user items.
    assert_eq!(
        entries
            .items
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["pi.user", "note"]
    );
    assert!(
        entries.items[0].id > entries.items[1].id,
        "writes are placed first"
    );
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn abort_withdraws_only_queued_items_and_checks_conversation_scope() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    set_busy(&conversation, true).await;
    let first = conversation
        .submit(input("one", None, None), &context())
        .await
        .expect("queue first");
    let second = conversation
        .submit(input("two", None, None), &context())
        .await
        .expect("queue second");
    assert_eq!(
        first.abort(&context()).await.expect("abort queued"),
        SubmissionAbort::Aborted
    );
    assert_eq!(
        first
            .wait(&context())
            .await
            .expect("settled abort")
            .reason
            .as_deref(),
        Some("aborted")
    );
    assert_eq!(
        inbox_items(&conversation)
            .await
            .iter()
            .map(|item| item.id())
            .collect::<Vec<_>>(),
        [second.id()]
    );
    let unrelated = harness
        .create_conversation(
            crate::harness::types::ConversationCreateOptions {
                ownership: ConversationOwnership::Ownerless,
                agent: None,
                init: None,
            },
            &context(),
        )
        .await
        .expect("other conversation");
    assert_eq!(
        harness
            .abort_submission(second.id(), &context(), Some(unrelated.id))
            .await
            .expect("scoped abort"),
        None
    );
    assert_eq!(
        harness
            .abort_submission(second.id(), &context(), Some(conversation.id))
            .await
            .expect("matching abort"),
        Some(SubmissionAbort::Aborted)
    );
    assert_eq!(
        harness
            .abort_submission(crate::types::SubmissionId(999_999), &context(), None)
            .await
            .expect("missing abort"),
        None
    );
    assert!(harness
        .submission(crate::types::SubmissionId(999_999), &context())
        .await
        .expect("missing lookup")
        .is_none());
    harness.close(&context()).await.expect("close");
}

#[tokio::test]
async fn placed_input_cannot_be_withdrawn_and_idle_head_writes_are_stale() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let input_id = conversation
        .commit(
            {
                let conversation_id = conversation.id;
                move |tx| {
                    let entry = tx
                        .append_entry(
                            conversation_id,
                            EntryDraft {
                                kind: Some("pi.user".into()),
                                ..Default::default()
                            },
                        )
                        .map_err(|error| error.to_string())?;
                    tx.create_submission(crate::types::SubmissionCreate {
                        conversation_id,
                        request_id: None,
                        submission_type: Some(crate::types::SubmissionType::Input),
                        status: Some(SubmissionStatus::Placed),
                        entry: Some(entry.id),
                        answer: None,
                        reason: None,
                        detail: None,
                    })
                    .map(|record| record.id)
                    .map_err(|error| error.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("create placed input");
    let input_submission = harness
        .submission(input_id, &context())
        .await
        .expect("lookup placed input")
        .expect("submission");
    assert_eq!(
        input_submission
            .status(&context())
            .await
            .expect("status")
            .status,
        SubmissionStatus::Placed
    );
    assert_eq!(
        input_submission
            .abort(&context())
            .await
            .expect("placed abort"),
        SubmissionAbort::AlreadyPlaced
    );

    let old = conversation
        .commit(
            {
                let id = conversation.id;
                move |tx| {
                    tx.append_entry(
                        id,
                        EntryDraft {
                            kind: Some("note".into()),
                            ..Default::default()
                        },
                    )
                    .map(|record| record.id)
                    .map_err(|e| e.to_string())
                }
            },
            &context(),
        )
        .await
        .expect("append old entry");
    conversation
        .reset(None, &context())
        .await
        .expect("reset range");
    let stale = conversation
        .submit(
            SubmissionDraft::Write(WriteSubmissionDraft {
                request_id: None,
                entry: EntryDraft {
                    kind: Some("summary".into()),
                    head: Some(crate::harness::types::EntryHead::Entry(old)),
                    ..Default::default()
                },
            }),
            &context(),
        )
        .await
        .expect("stale write");
    let record = stale.wait(&context()).await.expect("stale settles");
    assert_eq!(record.status, SubmissionStatus::Unanswered);
    assert_eq!(record.reason.as_deref(), Some("stale"));
    harness.close(&context()).await.expect("close");
}
