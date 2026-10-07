use pi_ai::{
	types::{Message, SystemContent, Tool, UserContentField, UserMessage},
	utils::transcript::{get_current_tools, to_tool_declaration},
};
use serde_json::{Map, Value};

use super::support::{context, open_default, root};
use crate::{
	harness::{
		harness::Conversation,
		prompt::{plan_system_entries, replay_sections},
		types::ConversationEntryQuery,
	},
	types::{ContextEdit, ContextEditAction, EntryDraft, EntryHead, EntryId, SelfMarker},
};

#[derive(Debug, PartialEq, Eq)]
struct Planned {
	sections: Option<Vec<(String, Option<String>)>>,
	removed: Option<Vec<String>>,
	added: Option<Vec<String>>,
	omit: Option<Vec<EntryId>>,
}

fn expected(
	sections: Option<&[(&str, Option<&str>)]>,
	removed: Option<&[&str]>,
	added: Option<&[&str]>,
	omit: Option<&[EntryId]>,
) -> Planned {
	Planned {
		sections: sections.map(|items| items.iter().map(|(key, value)| ((*key).to_owned(), value.map(str::to_owned))).collect()),
		removed: removed.map(|names| names.iter().map(|name| (*name).to_owned()).collect()),
		added: added.map(|names| names.iter().map(|name| (*name).to_owned()).collect()),
		omit: omit.map(<[EntryId]>::to_vec),
	}
}

fn section_values(values: &[(&str, &str)]) -> Map<String, Value> {
	values.iter().map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned()))).collect()
}

fn tool(name: &str, description: &str) -> Tool {
	Tool {
		name: name.to_owned(),
		description: description.to_owned(),
		parameters: serde_json::json!({"type":"object","properties":{}}),
		constrained_sampling: None,
	}
}

fn user(text: &str) -> Message {
	Message::User(UserMessage { content: UserContentField::Text(text.to_owned()), timestamp: 0 })
}

fn planned(draft: &EntryDraft) -> Planned {
	let messages = draft.model.as_ref().expect("planned system entry has model messages");
	assert_eq!(messages.len(), 1);
	let Message::System(message) = &messages[0] else { panic!("planned entry is a system message") };
	assert_eq!(message.content, SystemContent::Text(String::new()));
	assert_eq!(message.timestamp, 7);
	let omit = draft.edits.as_ref().map(|edits| {
		edits
			.iter()
			.map(|edit| {
				assert_eq!(edit.action, ContextEditAction::Omit);
				edit.target
			})
			.collect()
	});
	Planned {
		sections: message.sections.as_ref().map(|sections| {
			sections.iter().map(|(key, value)| (key.clone(), value.clone())).collect()
		}),
		removed: message.tools_removed.as_ref().map(|tools| tools.iter().map(|tool| tool.name.clone()).collect()),
		added: message.tools_added.as_ref().map(|tools| tools.iter().map(|tool| tool.name.clone()).collect()),
		omit,
	}
}

/// Plan, commit each system entry, and check the state reconstructed from the committed conversation.
async fn apply(conversation: &Conversation, desired: Map<String, Value>, tools: &[Tool]) -> Vec<Planned> {
	let view = conversation.context(&context()).await.expect("read current conversation context");
	let drafts = plan_system_entries(&view, &desired, tools, 7);
	let changes = drafts.iter().map(planned).collect::<Vec<_>>();
	let edits = drafts.iter().map(|draft| draft.edits.clone()).collect::<Vec<Option<Vec<ContextEdit>>>>();
	let conversation_id = conversation.id;
	conversation
		.commit(
			move |tx| {
				for draft in drafts {
					tx.append_entry(conversation_id, draft).map_err(|error| error.to_string())?;
				}
				Ok(())
			},
			&context(),
		)
		.await
		.expect("commit planned system entries");

	// Check the actual entry records as well as replay: omission targets must survive persistence.
	let entries = conversation
		.entries(ConversationEntryQuery::default(), 100, None, &context())
		.await
		.expect("read committed entries");
	let committed_systems = entries.items.iter().filter(|entry| entry.kind == "pi.system").take(edits.len()).collect::<Vec<_>>();
	assert_eq!(committed_systems.len(), edits.len());
	for (entry, planned_edits) in committed_systems.iter().zip(edits.iter().rev()) {
		assert_eq!(&entry.edits, planned_edits, "committed system edit matches its plan");
	}

	let replayed = conversation.context(&context()).await.expect("replay committed context");
	assert_eq!(replay_sections(&replayed.messages), desired);
	assert_eq!(get_current_tools(&replayed.messages), tools.iter().map(to_tool_declaration).collect::<Vec<_>>());
	changes
}

async fn append_user(conversation: &Conversation, text: &str) {
	let conversation_id = conversation.id;
	let text = text.to_owned();
	conversation
		.commit(
			move |tx| {
				tx.append_entry(
					conversation_id,
					EntryDraft { kind: Some("pi.user".into()), model: Some(vec![user(&text)]), ..Default::default() },
				)
				.map(|_| ())
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("append user message");
}

async fn marker(conversation: &Conversation, head: EntryHead) {
	let conversation_id = conversation.id;
	conversation
		.commit(
			move |tx| {
				tx.append_entry(
					conversation_id,
					EntryDraft {
						kind: Some("summary".into()),
						head: Some(head),
						model: Some(vec![user("summary")]),
						..Default::default()
					},
				)
				.map(|_| ())
				.map_err(|error| error.to_string())
			},
			&context(),
		)
		.await
		.expect("append head marker");
}

async fn last_system_id(conversation: &Conversation) -> EntryId {
	conversation
		.entries(ConversationEntryQuery::default(), 100, None, &context())
		.await
		.expect("scan conversation entries")
		.items
		.into_iter()
		.find(|entry| entry.kind == "pi.system")
		.expect("conversation has a system entry")
		.id
}

#[tokio::test]
async fn tool_changes_share_the_last_section_entry_and_redeclare_after_a_head_cut() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;
	let (a, b) = (tool("a", "a"), tool("b", "b"));

	assert_eq!(
		apply(&conversation, section_values(&[("x", "1"), ("y", "2")]), &[a.clone()]).await,
		vec![expected(Some(&[("x", Some("1")), ("y", Some("2"))]), None, Some(&["a"]), None)],
	);

	// Reordering sections plans a clear and a re-add; the tool addition belongs on the latter.
	assert_eq!(
		apply(&conversation, section_values(&[("y", "2"), ("x", "1")]), &[a.clone(), b.clone()]).await,
		vec![
			expected(Some(&[("x", None), ("y", None)]), None, None, None),
			expected(Some(&[("y", Some("2")), ("x", Some("1"))]), None, Some(&["b"]), None),
		],
	);

	// A self-head cuts off the prior tool declarations, so the post-head baseline declares all tools again.
	marker(&conversation, EntryHead::SelfMarker(SelfMarker)).await;
	assert_eq!(
		apply(&conversation, section_values(&[("y", "2"), ("x", "1")]), &[a, b]).await,
		vec![expected(Some(&[("y", Some("2")), ("x", Some("1"))]), None, Some(&["a", "b"]), None)],
	);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn rebaselines_sections_after_multiple_head_markers_with_exact_omissions() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;

	assert_eq!(
		apply(&conversation, section_values(&[("a", "1"), ("b", "2")]), &[]).await,
		vec![expected(Some(&[("a", Some("1")), ("b", Some("2"))]), None, None, None)],
	);
	append_user(&conversation, "hi").await;
	assert_eq!(
		apply(&conversation, section_values(&[("a", "1"), ("b", "20")]), &[]).await,
		vec![expected(Some(&[("b", Some("20"))]), None, None, None)],
	);
	let delta = last_system_id(&conversation).await;
	// The marker keeps the delta but cuts away its baseline, so a full baseline omits the retained delta.
	marker(&conversation, EntryHead::Entry(delta)).await;
	assert_eq!(
		apply(&conversation, section_values(&[("a", "1"), ("b", "20")]), &[]).await,
		vec![expected(Some(&[("a", Some("1")), ("b", Some("20"))]), None, None, Some(&[delta]))],
	);
	let baseline = last_system_id(&conversation).await;
	// The newly written post-head system entry makes later changes ordinary patches.
	assert_eq!(
		apply(&conversation, section_values(&[("a", "1"), ("b", "21")]), &[]).await,
		vec![expected(Some(&[("b", Some("21"))]), None, None, None)],
	);
	let after = last_system_id(&conversation).await;

	// A later marker retains all three deltas; their omission order is transcript order.
	marker(&conversation, EntryHead::Entry(delta)).await;
	assert_eq!(
		apply(&conversation, section_values(&[("a", "1"), ("b", "21")]), &[]).await,
		vec![expected(
			Some(&[("a", Some("1")), ("b", Some("21"))]),
			None,
			None,
			Some(&[delta, baseline, after]),
		)],
	);
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn writes_a_complete_post_head_baseline_even_when_replay_matches() {
	let (_storage, _registry, harness) = open_default().await;
	let conversation = root(&harness).await;

	assert_eq!(
		apply(&conversation, section_values(&[("a", "1")]), &[]).await,
		vec![expected(Some(&[("a", Some("1"))]), None, None, None)],
	);
	let baseline = last_system_id(&conversation).await;
	marker(&conversation, EntryHead::Entry(baseline)).await;
	assert_eq!(
		apply(&conversation, section_values(&[("a", "1")]), &[]).await,
		vec![expected(Some(&[("a", Some("1"))]), None, None, Some(&[baseline]))],
	);

	marker(&conversation, EntryHead::SelfMarker(SelfMarker)).await;
	assert_eq!(apply(&conversation, Map::new(), &[]).await, vec![expected(Some(&[]), None, None, None)]);
	assert!(apply(&conversation, Map::new(), &[]).await.is_empty());
	harness.close(&context()).await.expect("close harness");
}
