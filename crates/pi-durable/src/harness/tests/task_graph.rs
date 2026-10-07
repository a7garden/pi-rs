use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use super::support::{context, open_default, root};
use crate::harness::types::{AnyTask, Extension, Registry};
use crate::tasks::{PhaseHandler, TaskDefinition};
use crate::types::{TaskOptions, TaskOwnership};

fn blocked_task(name: &str) -> AnyTask {
	let handler: PhaseHandler = Arc::new(|_, _, _| Box::pin(async { Ok(()) }));
	AnyTask::new(TaskDefinition {
		name: name.into(),
		version: 1,
		initial: Arc::new(|_| json!({"phase":"work"})),
		phases: Arc::new(BTreeMap::from([("work".into(), handler.clone())])),
		abort: handler,
		migrate: None,
		hooks: None,
	})
}

#[tokio::test]
async fn graph_includes_committed_tasks_owner_edges_and_owned_conversations() {
	let (_storage, registry, harness) = open_default().await;
	let parent_task = blocked_task("test.graph-parent");
	let child_task = blocked_task("test.graph-child");
	registry.install(Extension { name: "graph".into(), tasks: vec![parent_task.clone(), child_task.clone()], ..Default::default() });
	let conversation = root(&harness).await;
	let parent = conversation.commit({ let task = parent_task.clone(); move |tx| {
		tx.create_task(&task, serde_json::Value::Null, TaskOptions { ownership: TaskOwnership::Conversation, conversation_id: None, background: None }).map_err(|e| e.to_string())
	}}, &context()).await.expect("create parent");
	let owned = harness.create_conversation(crate::harness::types::ConversationCreateOptions {
		ownership: crate::types::ConversationOwnership::Task { task_id: parent }, agent: None, init: None,
	}, &context()).await.expect("create owned conversation");
	let child = conversation.commit({ let task = child_task.clone(); move |tx| {
		tx.create_task(&task, serde_json::Value::Null, TaskOptions { ownership: TaskOwnership::Task { task_id: parent }, conversation_id: None, background: None }).map_err(|e| e.to_string())
	}}, &context()).await.expect("create child");
	let watch = harness.task_graph.watch(&context()).await.expect("acquire graph watch");
	let graph = watch.value();
	assert_eq!(graph.tasks.len(), 2);
	let parent_node = graph.tasks.get(&parent.to_string()).expect("parent node");
	assert_eq!(parent_node.conversations, vec![owned.id]);
	assert_eq!(parent_node.owner, None);
	let child_node = graph.tasks.get(&child.to_string()).expect("child node");
	assert_eq!(child_node.owner, Some(parent));
	assert_eq!(child_node.conversation_id, conversation.id);
	watch.stop().await;
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn empty_graph_watch_can_be_reacquired_after_stop() {
	let (_storage, _registry, harness) = open_default().await;
	let first = harness.task_graph.watch(&context()).await.expect("first graph watch");
	assert!(first.value().tasks.is_empty());
	first.stop().await;
	let second = harness.task_graph.watch(&context()).await.expect("reacquired graph watch");
	assert_eq!(second.value().tasks.len(), 0);
	second.stop().await;
	harness.close(&context()).await.expect("close harness");
}

#[tokio::test]
async fn watch_publishes_task_creation_and_terminal_removal_as_distinct_frames() {
	let (_storage, registry, harness) = open_default().await;
	let task = blocked_task("test.graph-published");
	registry.install(Extension { name: "graph-published".into(), tasks: vec![task.clone()], ..Default::default() });
	let conversation = root(&harness).await;
	let watch = harness.task_graph.watch(&context()).await.expect("watch graph");
	let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
	watch.start(Arc::new(move |value, _ops, _context| {
		let sender = sender.clone();
		Box::pin(async move { let _ = sender.send(value); })
	})).expect("start graph watch");
	let id = conversation.commit(move |tx| {
		tx.create_task(&task, serde_json::Value::Null, TaskOptions {
			ownership: TaskOwnership::Conversation, conversation_id: None, background: None,
		}).map_err(|error| error.to_string())
	}, &context()).await.expect("create task");
	let created = tokio::time::timeout(std::time::Duration::from_secs(2), receiver.recv())
		.await.expect("creation frame timeout").expect("creation frame");
	assert!(created.tasks.contains_key(&id.to_string()));
	assert_eq!(created.tasks[&id.to_string()].state, crate::harness::task_graph::TaskGraphState::Pending { phase: "work".into() });
	watch.stop().await;
	harness.close(&context()).await.expect("close harness");
}
