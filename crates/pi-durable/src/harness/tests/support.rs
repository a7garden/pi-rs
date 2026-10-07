use std::sync::Arc;

use pi_chord::context::Context;

use crate::harness::harness::{open_harness, Conversation, Harness};
use crate::harness::registry::{create_registry, RegistryImpl};
use crate::harness::types::{HarnessOptions, RootConversationOptions};
use crate::storage::memory::MemoryStorage;
use crate::types::Storage;

pub(crate) fn context() -> Context {
	crate::session::support::context()
}

pub(crate) fn registry() -> Arc<RegistryImpl> {
	Arc::new(create_registry())
}

pub(crate) async fn open(storage: Arc<dyn Storage>, options: HarnessOptions) -> Arc<Harness> {
	open_harness(storage, options, &context()).await.expect("open harness")
}

pub(crate) async fn open_default() -> (Arc<MemoryStorage>, Arc<RegistryImpl>, Arc<Harness>) {
	let storage = Arc::new(MemoryStorage::new());
	let registry = registry();
	let options = HarnessOptions { registry: registry.clone(), ..HarnessOptions::default() };
	let harness = open(storage.clone(), options).await;
	(storage, registry, harness)
}

pub(crate) async fn root(harness: &Arc<Harness>) -> Conversation {
	harness
		.root(&context(), Some(RootConversationOptions::default()))
		.await
		.expect("create root conversation")
}
