use parking_lot::Mutex;
use std::sync::Arc;
use crate::harness::events::{watch_events, AgentEvent};
use crate::harness::types::EntryDraft;
use crate::harness::output::{bound_output, Chunk, OutputBuffer, OutputLimits, Progress, Retain};

use crate::harness::prompt::replay_sections;
use crate::harness::tests::support::{context, open_default, root};
use crate::harness::types::{ConversationId, EntryId, EntryRecord};
use pi_ai::types::{Message, SystemContent, SystemMessage};

fn limits(retain: Retain, max_lines: usize, max_bytes: usize) -> OutputLimits {
    OutputLimits { max_bytes, max_lines, retain }
}

fn measure(text: &str) -> crate::env::ShellOutputSkip {
    crate::env::ShellOutputSkip {
        bytes: text.len() as u64,
        newlines: text.bytes().filter(|byte| *byte == b'\n').count() as u64,
        ends_with_newline: text.ends_with('\n'),
    }
}

#[test]
fn output_bounds_preserve_lines_bytes_controls_and_utf8_boundaries() {
    assert_eq!(bound_output("a\nb\n", limits(Retain::Head, 2, 100)).text, "a\nb\n");
    assert_eq!(bound_output("a\nb", limits(Retain::Tail, 2, 100)).text, "a\nb");
    assert_eq!(bound_output("", limits(Retain::Tail, 2, 100)).text, "");
    assert_eq!(bound_output("a\nb\nc\n", limits(Retain::Head, 2, 100)), crate::harness::output::OutputSlice { text: "a\nb\n".into(), bytes: 4, dropped_bytes: 2, dropped_lines: 1 });
    assert_eq!(bound_output("a\nb\nc", limits(Retain::Tail, 2, 100)).text, "b\nc");
    assert_eq!(bound_output("a\nb\nc\n\n", limits(Retain::Tail, 3, 100)).text, "b\nc\n\n");
    assert_eq!(bound_output("aa\nbb\ncc\n", limits(Retain::Head, 10, 7)).text, "aa\nbb\n");
    assert_eq!(bound_output("aa\nbb\ncc\n", limits(Retain::Tail, 10, 7)).text, "bb\ncc\n");
    assert_eq!(bound_output("ééé\n", limits(Retain::Head, 10, 5)).text, "éé");
    assert_eq!(bound_output("x\néééé", limits(Retain::Tail, 10, 5)).text, "éé");
    assert_eq!(bound_output("ab\ncd\n", limits(Retain::Head, 10, 0)).text, "");
    assert_eq!(bound_output("ab\ncd\n", limits(Retain::Tail, 0, 100)).dropped_lines, 2);

}

#[test]
fn output_buffer_decodes_chunks_counts_raw_bytes_and_bounds_long_streams() {
    let mut buffer = OutputBuffer::new(limits(Retain::Tail, 2, 100));
    buffer.push(Chunk::Text("a\nb\n".into()), None);
    let smile = "😀\n".as_bytes();
    buffer.push(Chunk::Bytes(smile[..2].to_vec()), None);
    buffer.push(Chunk::Bytes(smile[2..].to_vec()), None);
    assert_eq!(buffer.snapshot().text, "b\n😀\n");

    let mut raw = OutputBuffer::new(limits(Retain::Tail, 1, 100));
    raw.push(Chunk::Text("a\u{7}\n".into()), None);
    assert_eq!(raw.snapshot().text, "a\n");
    raw.push(Chunk::Text("b\u{1b}\n".into()), None);
    let snapshot = raw.snapshot();
    assert_eq!((snapshot.text.as_str(), snapshot.dropped_bytes, snapshot.dropped_lines), ("b\n", 3, 1));

    let euro = "€".as_bytes();
    let mut split = OutputBuffer::new(limits(Retain::Tail, 10, 100));
    split.push(Chunk::Bytes(euro[..1].to_vec()), None);
    split.push(Chunk::Text("x".into()), None);
    split.push(Chunk::Bytes(euro[..2].to_vec()), None);
    split.end();
    assert_eq!(split.snapshot().text, "\u{fffd}x\u{fffd}");

    for retain in [Retain::Head, Retain::Tail] {
        for (max_lines, max_bytes) in [(3, 40), (50, 25)] {
            let caps = limits(retain, max_lines, max_bytes);
            let mut running = OutputBuffer::new(caps);
            let mut all = String::new();
            for index in 0..300 {
                let chunk = if index % 7 == 0 { format!("{}\n", "é".repeat(index % 30)) } else { format!("line {index}\n") };
                all.push_str(&chunk);
                running.push(Chunk::Text(chunk), None);
                if index % 5 == 4 {
                    let expected = bound_output(&all, caps);
                    let actual = running.snapshot();
                    assert_eq!((actual.text, actual.dropped_bytes, actual.dropped_lines), (expected.text, expected.dropped_bytes, expected.dropped_lines));
                }
            }
        }
    }
}

#[test]
fn tail_window_is_stable_across_deterministic_progress_snapshots() {
    const ALPHABET: [&str; 15] = ["a", "b", "z", " ", "\n", "\n", "\n", "\r\n", "\t", "é", "€", "😀", "\u{fffd}", "\u{1}", "\u{1b}"];
    for seed in 1u32..=3000 {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f64 / 4_294_967_296.0
        };
        let caps = limits(Retain::Tail, 1 + (next() * 5.0) as usize, 1 + (next() * 40.0) as usize);
        let mut whole = OutputBuffer::new(caps);
        let mut sampled = OutputBuffer::new(caps);
        let chunks = (next() * 30.0) as usize;
        for _ in 0..chunks {
            let length = (next() * 12.0) as usize;
            let mut chunk = String::new();
            for _ in 0..length {
                chunk.push_str(ALPHABET[(next() * ALPHABET.len() as f64) as usize]);
            }
            whole.push(Chunk::Text(chunk.clone()), None);
            sampled.push(Chunk::Text(chunk), None);
            if next() < 0.4 {
                sampled.snapshot();
            }
        }
        assert_eq!(sampled.snapshot(), whole.snapshot(), "seed {seed}");
    }
}

fn skip_random(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x6d2b79f5);
        let mut value = state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        let value = value ^ (value >> 14);
        value as f64 / 4_294_967_296.0
    }
}

fn check_skip_seed(seed: u32) {
    const ALPHABET: [&str; 15] = ["a", "b", "z", " ", "\n", "\n", "\n", "\r\n", "\t", "é", "€", "😀", "\u{fffd}", "\u{1}", "\u{1b}"];
    let mut next = skip_random(seed);
    let caps = limits(Retain::Tail, 1 + (next() * 5.0) as usize, 1 + (next() * 40.0) as usize);
    let mut full = OutputBuffer::new(caps);
    let mut skipping = OutputBuffer::new(caps);
    let mut pending = String::new();
    let flush = |pending: &mut String, full: &mut OutputBuffer, skipping: &mut OutputBuffer, next: &mut dyn FnMut() -> f64| {
        if pending.is_empty() {
            return;
        }
        let mut cuts: Vec<usize> = pending.char_indices().map(|(index, _)| index).filter(|cut| *cut > 0).collect();
        cuts.push(pending.len());
        cuts.retain(|cut| {
            let suffix = &pending[*cut..];
            suffix.len() > caps.max_bytes || suffix.bytes().filter(|byte| *byte == b'\n').count() > caps.max_lines
        });
        if !cuts.is_empty() && next() < 0.7 {
            let cut = cuts[(next() * cuts.len() as f64) as usize];
            let omitted = &pending[..cut];
            skipping.push(Chunk::Text(pending[cut..].to_owned()), Some(measure(omitted)));
        } else {
            skipping.push(Chunk::Text(pending.clone()), None);
        }
        pending.clear();
        if next() < 0.5 {
            skipping.snapshot();
        }
        assert_eq!(skipping.snapshot(), full.snapshot(), "seed {seed}");
    };
    let chunks = (next() * 40.0) as usize;
    for _ in 0..chunks {
        let length = (next() * 12.0) as usize;
        let mut chunk = String::new();
        for _ in 0..length {
            chunk.push_str(ALPHABET[(next() * ALPHABET.len() as f64) as usize]);
        }
        full.push(Chunk::Text(chunk.clone()), None);
        if next() < 0.3 {
            full.snapshot();
        }
        pending.push_str(&chunk);
        if next() < 0.3 {
            flush(&mut pending, &mut full, &mut skipping, &mut next);
        }
    }
    flush(&mut pending, &mut full, &mut skipping, &mut next);
    assert_eq!(skipping.snapshot(), full.snapshot(), "seed {seed}");
}

#[test]
fn output_skips_match_full_stream_across_legal_deterministic_patterns() {
    for seed in 1..=3000 {
        check_skip_seed(seed);
    }
}

#[test]
fn output_buffer_skip_contract_matches_full_tail_and_rejects_head() {
    let limits = limits(Retain::Tail, 2, 1000);
    let omitted = "one\ntwo\nthr";
    let kept = "ee\nfour\nfive\nsix";
    let mut full = OutputBuffer::new(limits);
    full.push(Chunk::Text(format!("{omitted}{kept}")), None);
    let mut skipped = OutputBuffer::new(limits);
    skipped.push(Chunk::Text(kept.into()), Some(measure(omitted)));
    assert_eq!(skipped.snapshot(), full.snapshot());
    let snapshot = skipped.snapshot();
    assert_eq!((snapshot.text.as_str(), snapshot.dropped_bytes, snapshot.dropped_lines), ("five\nsix", 19, 4));
    let mut head = OutputBuffer::new(self::limits(Retain::Head, 2, 10));
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        head.push(Chunk::Text("x\ny\nz\n".into()), Some(crate::env::ShellOutputSkip { bytes: 3, newlines: 1, ends_with_newline: true }));
    }));
    let message = panicked.unwrap_err();
    let message = message.downcast_ref::<String>().map(String::as_str).or_else(|| message.downcast_ref::<&str>().copied()).unwrap_or("");
    assert!(message.contains("Skipped output requires tail retention"));
}

fn section_values(view: &crate::session::ContextView, desired: &[(&str, &str)]) -> Vec<Vec<(String, Option<String>)>> {
    let desired: serde_json::Map<String, serde_json::Value> = desired.iter().map(|(key, value)| ((*key).to_owned(), (*value).into())).collect();
    crate::harness::prompt::plan_system_entries(view, &desired, &[], 7)
        .into_iter()
        .map(|draft| {
            let Some(Message::System(message)) = draft.model.and_then(|messages| messages.into_iter().next()) else {
                panic!("planned prompt entry did not contain a system message");
            };
            message.sections.unwrap_or_default().into_iter().collect()
        })
        .collect()
}

fn append_sections(view: &mut crate::session::ContextView, sections: &[(&str, Option<&str>)]) {
    let sections = sections.iter().map(|(key, value)| ((*key).to_owned(), value.map(str::to_owned))).collect();
    view.messages.push(Message::System(SystemMessage { content: SystemContent::Text(String::new()), sections: Some(sections), tools_added: None, tools_removed: None, timestamp: 7 }));
}

#[test]
fn prompt_planning_emits_minimal_patches_removals_and_order_rewrites() {
    let mut view = crate::session::ContextView::default();
    assert_eq!(section_values(&view, &[("a", "1"), ("b", "2"), ("c", "3")]), vec![vec![("a".into(), Some("1".into())), ("b".into(), Some("2".into())), ("c".into(), Some("3".into()))]]);
    append_sections(&mut view, &[("a", Some("1")), ("b", Some("2")), ("c", Some("3"))]);
    assert_eq!(section_values(&view, &[("a", "1"), ("b", "20"), ("c", "3"), ("d", "4")]), vec![vec![("b".into(), Some("20".into())), ("d".into(), Some("4".into()))]]);
    append_sections(&mut view, &[("b", Some("20")), ("d", Some("4"))]);
    assert_eq!(section_values(&view, &[("a", "1"), ("c", "3"), ("d", "4")]), vec![vec![("b".into(), None)]]);
    append_sections(&mut view, &[("b", None)]);
    assert!(section_values(&view, &[("a", "1"), ("c", "3"), ("d", "4")]).is_empty());
    assert_eq!(section_values(&view, &[("b", "2"), ("a", "1")]), vec![
        vec![("a".into(), None), ("c".into(), None), ("d".into(), None)],
        vec![("b".into(), Some("2".into())), ("a".into(), Some("1".into()))],
    ]);
}

fn entry_record(id: u64, kind: &str) -> EntryRecord {
    EntryRecord {
        id: EntryId(id),
        conversation_id: ConversationId(1),
        kind: kind.into(),
        model: None,
        data: None,
        head: None,
        edits: None,
        by_task_id: None,
    }
}

#[test]
fn prompt_planner_writes_post_head_baseline_and_omits_retained_system_entries() {
    let system = SystemMessage {
        content: SystemContent::Text(String::new()),
        sections: Some([("a".to_owned(), Some("1".to_owned()))].into_iter().collect()),
        tools_added: None,
        tools_removed: None,
        timestamp: 1,
    };
    let mut view = crate::session::ContextView::default();
    view.messages.push(Message::System(system));
    view.entries.push(entry_record(1, "pi.system"));
    view.head = Some(entry_record(2, "summary"));
    let desired = serde_json::Map::from_iter([("a".to_owned(), serde_json::Value::String("1".into()))]);
    let drafts = crate::harness::prompt::plan_system_entries(&view, &desired, &[], 7);
    assert_eq!(drafts.len(), 1, "a head marker requires a complete baseline even when replay matches");
    assert_eq!(drafts[0].edits.as_ref().unwrap()[0].target, EntryId(1));
    assert_eq!(drafts[0].edits.as_ref().unwrap()[0].action, crate::types::ContextEditAction::Omit);

    view.entries.push(entry_record(3, "pi.system"));
    assert!(crate::harness::prompt::plan_system_entries(&view, &desired, &[], 7).is_empty(), "later system entry ends the baseline-only interval");
}

#[test]
fn prompt_tool_planning_adds_removes_replaces_and_rewrites_order() {
    let declaration = |name: &str, description: &str| pi_ai::types::Tool {
        name: name.into(), description: description.into(), parameters: serde_json::json!({"type":"object","properties":{}}), constrained_sampling: None,
    };
    let a = declaration("a", "a");
    let b = declaration("b", "b");
    let c = declaration("c", "c");
    let mut view = crate::session::ContextView::default();
    let initial = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[a.clone(), b.clone()], 7);
    assert_eq!(initial.len(), 1);
    let Message::System(first) = initial[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(first.tools_added, Some(vec![a.clone(), b.clone()]));
    view.messages.push(Message::System(first));
    let added = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[a.clone(), b.clone(), c.clone()], 7);
    assert_eq!(added.len(), 1);
    let Message::System(add) = added[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(add.tools_added, Some(vec![c.clone()]));
    view.messages.push(Message::System(add));
    let removed = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[a.clone(), c.clone()], 7);
    let Message::System(remove) = removed[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(remove.tools_removed.as_ref().unwrap().iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(), ["b"]);
    view.messages.push(Message::System(remove));
    let c2 = declaration("c", "changed");
    let changed = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[a.clone(), c2.clone()], 7);
    assert_eq!(changed.len(), 1);
    let Message::System(changed) = changed[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(changed.tools_removed.as_ref().unwrap().iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(), ["c"]);
    assert_eq!(changed.tools_added, Some(vec![c2.clone()]));
    view.messages.push(Message::System(changed));
    let a2 = declaration("a", "changed");
    let middle = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[a2.clone(), c2.clone()], 7);
    let Message::System(middle) = middle[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(middle.tools_removed.as_ref().unwrap().iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(), ["a", "c"]);
    assert_eq!(middle.tools_added, Some(vec![a2.clone(), c2.clone()]));
    view.messages.push(Message::System(middle));
    let reordered = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[c2.clone(), a2.clone()], 7);
    let Message::System(reordered) = reordered[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(reordered.tools_removed.as_ref().unwrap().iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(), ["a", "c"]);
    assert_eq!(reordered.tools_added, Some(vec![c2.clone(), a2.clone()]));
    view.messages.push(Message::System(reordered));
    let empty = crate::harness::prompt::plan_system_entries(&view, &serde_json::Map::new(), &[], 7);
    let Message::System(empty) = empty[0].model.as_ref().unwrap()[0].clone() else { panic!("system entry") };
    assert_eq!(empty.tools_removed.as_ref().unwrap().iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(), ["c", "a"]);
}

#[tokio::test(start_paused = true)]
async fn progress_commits_first_change_coalesces_and_propagates_failure() {
    let commits = Arc::new(Mutex::new(Vec::new()));
    let written = Arc::new(Mutex::new(50 * 1024usize));
    let c = commits.clone();
    let w = written.clone();
    let progress = Progress::new(move || {
        let c = c.clone();
        let w = w.clone();
        Box::pin(async move { c.lock().push(tokio::time::Instant::now()); Ok(*w.lock()) })
    }, |_| {}, 100);
    progress.run();
    let first = progress.mark_and_wait().await;
    first.await.unwrap().unwrap();
    assert_eq!(commits.lock().len(), 1);
    *written.lock() = 10;
    let next = progress.mark_and_wait().await;
    progress.mark().await;
    tokio::time::advance(std::time::Duration::from_millis(499)).await;
    tokio::task::yield_now().await;
    assert_eq!(commits.lock().len(), 1);
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    next.await.unwrap().unwrap();
    assert_eq!(commits.lock().len(), 2);
    let small = progress.mark_and_wait().await;
    tokio::time::advance(std::time::Duration::from_millis(99)).await;
    tokio::task::yield_now().await;
    assert_eq!(commits.lock().len(), 2);
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    small.await.unwrap().unwrap();
    assert_eq!(commits.lock().len(), 3);
    let waiters = progress.stop().await;
    assert!(waiters.is_empty());

    let reported = Arc::new(Mutex::new(Vec::new()));
    let errors = reported.clone();
    let failed = Progress::new(|| Box::pin(async { Err("commit failed".to_owned()) }), move |error| errors.lock().push(error), 1);
    failed.run();
    failed.mark_and_wait().await.await.unwrap().unwrap_err();
    assert_eq!(&*reported.lock(), &["commit failed"]);
    failed.stop().await;
}

#[tokio::test]
async fn progress_stop_waits_for_in_flight_commit_and_returns_uncovered_waiters() {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let started = Arc::new(Mutex::new(Some(started_tx)));
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let release = Arc::new(tokio::sync::Mutex::new(Some(release_rx)));
    let started_for_write = started.clone();
    let release_for_write = release.clone();
    let progress = Progress::new(move || {
        let started = started_for_write.clone();
        let release = release_for_write.clone();
        Box::pin(async move {
            if let Some(started) = started.lock().take() {
                let _ = started.send(());
            }
            let receiver = release.lock().await.take().expect("one progress write");
            let _ = receiver.await;
            Ok(0)
        })
    }, |_| {}, 100);
    progress.run();
    let first = progress.mark_and_wait().await;
    started_rx.await.expect("write started");
    let second = progress.mark_and_wait().await;
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        let _ = release_tx.send(());
    });
    let pending = progress.stop().await;
    first.await.unwrap().unwrap();
    assert_eq!(pending.len(), 1);
    pending.into_iter().next().unwrap().send(Ok(())).unwrap();
    second.await.unwrap().unwrap();
}

#[tokio::test]
async fn event_stream_starts_with_snapshot_and_closes_with_harness() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let mut stream = watch_events(&harness, conversation.id, &context()).await.expect("attach event stream");
    let initial = stream.next().await.expect("initial batch").expect("event batch");
    assert_eq!(initial.len(), 1);
    assert!(matches!(initial[0], AgentEvent::Snapshot { .. }));
    let id = conversation.id;
    conversation.commit(move |tx| tx.append_entry(id, EntryDraft { kind: Some("note".into()), ..Default::default() }).map_err(|error| error.to_string()), &context()).await.expect("append entry");
    let changed = stream.next().await.expect("committed event batch").expect("snapshot batch");
    assert_eq!(changed.len(), 1);
    match &changed[0] {
        AgentEvent::EntryAppended { entry } => assert_eq!(entry.kind, "note"),
        event => panic!("expected entry_appended, got {event:?}"),
    }
    harness.close(&context()).await.expect("close harness");
    let closed = stream.next().await.expect("session close notification");
    assert!(closed.is_err());
}

#[test]
fn prompt_section_replay_patches_in_place_deletes_and_appends_readditions() {
    let system = |sections: &[(&str, Option<&str>)]| {
        let sections = sections.iter().map(|(key, value)| ((*key).to_owned(), value.map(str::to_owned))).collect();
        Message::System(SystemMessage { content: SystemContent::Text(String::new()), sections: Some(sections), tools_added: None, tools_removed: None, timestamp: 1 })
    };
    let replayed = replay_sections(&[
        system(&[("a", Some("1")), ("b", Some("2")), ("c", Some("3"))]),
        system(&[("b", Some("20")), ("a", None)]),
        system(&[("a", Some("10"))]),
    ]);
    let ordered: Vec<_> = replayed.into_iter().collect();
    assert_eq!(ordered, vec![("b".to_owned(), serde_json::Value::String("20".to_owned())), ("c".to_owned(), serde_json::Value::String("3".to_owned())), ("a".to_owned(), serde_json::Value::String("10".to_owned()))]);
}

#[tokio::test]
async fn event_overflow_replaces_undelivered_commits_with_one_snapshot() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let mut stream = watch_events(&harness, conversation.id, &context()).await.expect("attach stream");
    assert!(matches!(stream.next().await.unwrap().unwrap()[0], AgentEvent::Snapshot { .. }));
    for _ in 0..101 {
        let id = conversation.id;
        conversation.commit(move |tx| tx.append_entry(id, EntryDraft { kind: Some("note".into()), ..Default::default() }).map_err(|error| error.to_string()), &context()).await.expect("append entry");
    }
    let batch = stream.next().await.expect("pending batch").expect("event batch");
    assert_eq!(batch.len(), 1, "overflow should discard pending deltas and deliver one replacement snapshot");
    let AgentEvent::Snapshot { snapshot } = &batch[0] else { panic!("overflow replacement must be a snapshot") };
    assert_eq!(snapshot.entries.len(), 101);
    harness.close(&context()).await.expect("close harness");
}



#[test]
fn live_document_checkpoints_when_no_generation_or_running_tool_exists() {
    let document = crate::harness::live::live_doc();
    let checkpoint = document.0.checkpoint_when.as_ref().expect("live checkpoint predicate");
    let check = |value: serde_json::Value| {
        let object = value.as_object().expect("live object").clone();
        checkpoint(&object, &[], crate::types::CheckpointInfo { deltas_since_base: 1000 })
    };
    assert!(check(serde_json::json!({})));
    assert!(!check(serde_json::json!({"run":{"taskId":1,"inputs":[]},"generation":{"attempt":1}})));
    assert!(check(serde_json::json!({"run":{"taskId":1,"inputs":[]},"tools":[{"callId":"c","name":"n","status":"pending"},{"callId":"d","name":"m","status":"done"}]})));
    assert!(!check(serde_json::json!({"run":{"taskId":1,"inputs":[]},"tools":[{"callId":"d","name":"m","status":"done"},{"callId":"c","name":"n","status":"running"}]})));
    assert!(check(serde_json::json!({"run":{"taskId":1,"inputs":[]}})));
}

#[test]
fn output_buffer_retains_bounded_storage_while_accounting_for_dropped_text() {
    let mut head = OutputBuffer::new(limits(Retain::Head, 2, 1000));
    for index in 0..1000 {
        head.push(Chunk::Text(format!("line {index}\n")), None);
    }
    assert!(head.stored_bytes() < 20);
    let snapshot = head.snapshot();
    assert_eq!(snapshot.text, "line 0\nline 1\n");
    assert_eq!((snapshot.dropped_bytes, snapshot.dropped_lines), (8876, 998));

    let caps = limits(Retain::Tail, 3, 100);
    let mut tail = OutputBuffer::new(caps);
    let mut full = String::new();
    for index in 0..2000 {
        let chunk = format!("line {index}\n\n");
        full.push_str(&chunk);
        tail.push(Chunk::Text(chunk), None);
        let snapshot = tail.snapshot();
        assert!(tail.stored_bytes() <= 100);
        assert_eq!(snapshot.text, bound_output(&full, caps).text);
    }
}

struct NoDocuments;

#[async_trait::async_trait]
impl crate::harness::types::DocumentReader for NoDocuments {
    async fn snapshot(&self, _: &crate::documents::DocToken, _: Option<u64>, _: Option<&str>, _: &pi_chord::context::Context) -> Result<Option<Arc<serde_json::Map<String, serde_json::Value>>>, String> {
        Ok(None)
    }
    async fn snapshot_as_of(&self, _: &crate::documents::DocToken, _: crate::harness::types::ConversationId, _: Option<&str>, _: crate::harness::types::EntryId, _: &pi_chord::context::Context) -> Result<Option<serde_json::Map<String, serde_json::Value>>, String> {
        Ok(None)
    }
}

fn prompt_input() -> crate::harness::types::PromptInput {
    crate::harness::types::PromptInput {
        conversation_id: crate::harness::types::ConversationId(1),
        agent: crate::harness::types::Agent {
            model: None,
            thinking_level: pi_ai::types::ModelThinkingLevel::Off,
            extensions: Vec::new(),
            tools: Vec::new(),
            sections: Vec::new(),
            instructions: None,
            cwd: None,
        },
        env: None,
        shown: serde_json::Map::new(),
        read: Arc::new(NoDocuments),
    }
}

#[tokio::test]
async fn prompt_rendering_tags_omissions_preserves_shown_values_and_reports_errors() {
    use crate::harness::types::PromptSection;
    let section = |key: &str, tag: Option<bool>, render: crate::harness::types::SectionRender| PromptSection { key: key.into(), render, tag };
    let failing = section("failing", None, Arc::new(|_, _| Box::pin(async { Err("render failed".to_owned()) })));
    let omitted = section("omitted", None, Arc::new(|_, _| Box::pin(async { Ok(None) })));
    let failing2 = section("failing2", None, Arc::new(|_, _| Box::pin(async { Err("also failed".to_owned()) })));
    let plain = section("plain", Some(false), Arc::new(|_, _| Box::pin(async { Ok(Some("literal".to_owned())) })));
    let tagged = section("cwd", None, Arc::new(|_, _| Box::pin(async { Ok(Some("/repo".to_owned())) })));
    let shown = serde_json::Map::from_iter([
        ("failing".to_owned(), serde_json::Value::String("old".into())),
        ("failing2".to_owned(), serde_json::Value::String("old2".into())),
    ]);
    let reports = Arc::new(Mutex::new(Vec::new()));
    let reported = reports.clone();
    let desired = crate::harness::prompt::render_sections(
        &[tagged, omitted, plain, failing, failing2],
        prompt_input(),
        &shown,
        &move |error| reported.lock().push(error),
        context(),
    ).await.expect("render sections");
    assert_eq!(desired.get("cwd"), Some(&serde_json::Value::String("<cwd>\n/repo\n</cwd>".into())));
    assert_eq!(desired.get("plain"), Some(&serde_json::Value::String("literal".into())));
    assert!(desired.get("omitted").is_none());
    assert_eq!(desired.get("failing"), Some(&serde_json::Value::String("old".into())));
    assert_eq!(desired.get("failing2"), Some(&serde_json::Value::String("old2".into())));
    assert_eq!(&*reports.lock(), &["render failed", "also failed"]);

    let controller = pi_chord::context::AbortController::new();
    controller.abort("cancelled");
    let cancelled_context = pi_chord::context::with_abort_signal(controller.signal(), context());
    let cancelled = crate::harness::types::PromptSection {
        key: "cancelled".into(),
        tag: None,
        render: Arc::new(|_, _| Box::pin(async { Err("cancelled".to_owned()) })),
    };
    assert!(crate::harness::prompt::render_sections(&[cancelled], prompt_input(), &serde_json::Map::new(), &|_| {}, cancelled_context).await.is_err());
}

#[tokio::test]
async fn event_stream_emits_message_start_for_a_committed_generation_partial() {
    let (_storage, _registry, harness) = open_default().await;
    let conversation = root(&harness).await;
    let mut stream = watch_events(&harness, conversation.id, &context()).await.expect("attach stream");
    let _initial = stream.next().await.expect("initial snapshot").expect("snapshot result");
    let id = conversation.id;
    let message = pi_ai::providers::faux::faux_assistant_message("partial", pi_ai::providers::faux::FauxMessageOptions::default());
    conversation.commit(move |tx| {
        let draft = tx.doc(&crate::harness::live::live_doc(), Some(id.0), None, None).map_err(|error| error.to_string())?;
        let mut live = crate::harness::live::read_live(&draft);
        live.generation = Some(crate::harness::live::GenerationStatus {
            attempt: 1,
            message: Some(serde_json::to_value(message).map_err(|error| error.to_string())?),
            retry: None,
            deferred: None,
        });
        crate::harness::live::store_live(&draft, &live)?;
        Ok(())
    }, &context()).await.expect("commit partial");
    let events = stream.next().await.expect("partial event batch").expect("event batch");
    assert!(matches!(events.as_slice(), [AgentEvent::MessageStart { .. }]));
    harness.close(&context()).await.expect("close harness");
}
