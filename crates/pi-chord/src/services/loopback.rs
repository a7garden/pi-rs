//! Port of `packages/chord/src/services/loopback.ts`.
//!
//! Connects a provider to a binding without changing remote service semantics.

use std::boxed::Box;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::Context;
use crate::json::JsonValue;
use crate::services::errors::AnyError;
use crate::services::provider::RemoteServiceProvider;
use crate::types::{
    RemoteServiceTransport, ServiceCall, ServiceMode, ServiceSubscription, ServiceSubscriptionSnapshot,
    UpdateListener,
};

/// A subscription view that hides the provider-side close semantics from the
/// loopback consumer (upstream wraps the subscription with `close: () => ...`).
struct LoopbackSubscription {
    subscription: Box<dyn ServiceSubscription>,
}

#[async_trait]
impl ServiceSubscription for LoopbackSubscription {
    fn snapshot(&self) -> &ServiceSubscriptionSnapshot {
        self.subscription.snapshot()
    }

    fn activate(&self) {
        self.subscription.activate();
    }

    async fn close(&self, _context: Option<Context>) {
        self.subscription.close(None).await;
    }
}

struct LoopbackTransport {
    provider: Arc<RemoteServiceProvider>,
}

#[async_trait]
impl RemoteServiceTransport for LoopbackTransport {
    async fn invoke(&self, call: &ServiceCall, context: &Context) -> Result<Option<JsonValue>, AnyError> {
        self.provider.invoke(call, context).await
    }

    async fn subscribe(
        &self,
        service_id: &str,
        mode: ServiceMode,
        listener: UpdateListener,
        _context: &Context,
    ) -> Result<Box<dyn ServiceSubscription>, AnyError> {
        let subscription = self.provider.subscribe(service_id, mode, listener)?;
        Ok(Box::new(LoopbackSubscription { subscription }))
    }
}

/// Connects a provider to a binding without changing remote service semantics.
pub fn create_loopback_service_transport(provider: Arc<RemoteServiceProvider>) -> Arc<dyn RemoteServiceTransport> {
    Arc::new(LoopbackTransport { provider })
}
