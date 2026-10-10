//! Port of `experimental/services/presentation-ui.ts` — narrow process-local UI
//! capabilities available to presentation facets.
//!
//! Deviations (D144):
//! - Upstream's structural `PresentationUI` interface becomes the
//!   [`PresentationUi`] trait object behind [`PresentationUiHandle`], the
//!   concrete type the local service token carries.
//! - The TUI-bound implementations (alternate-screen renderer integration,
//!   selection dialogs) are deferred with interactive mode; only the contract
//!   and token port. No rendering code exists in this module.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

use pi_chord::api::define_service;
use pi_chord::context::Context;

/// Upstream `PresentationSelectItem`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationSelectItem {
    pub value: String,
    pub label: String,
    pub description: Option<String>,
}

/// Upstream `PresentationUI` shape as a trait (async select, sync
/// `showStatus`).
pub trait PresentationUi: Send + Sync {
    fn select(
        &self,
        title: &str,
        items: &[PresentationSelectItem],
        selected_value: Option<&str>,
        context: Context,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, String>> + Send + '_>>;
    fn show_status(&self, message: &str, context: Context);
}

/// Concrete local service value (upstream provides the object itself).
#[derive(Clone)]
pub struct PresentationUiHandle(pub Arc<dyn PresentationUi>);

pub static PRESENTATION_UI: LazyLock<pi_chord::Service<PresentationUiHandle>> =
    LazyLock::new(|| {
        define_service("pi.local.presentation-ui", true).expect("service id is valid")
    });

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopUi;

    impl PresentationUi for NoopUi {
        fn select(
            &self,
            _title: &str,
            items: &[PresentationSelectItem],
            _selected_value: Option<&str>,
            _context: Context,
        ) -> Pin<Box<dyn Future<Output = Result<Option<String>, String>> + Send + '_>> {
            assert_eq!(items.len(), 1);
            Box::pin(async { Ok(None) })
        }

        fn show_status(&self, _message: &str, _context: Context) {}
    }

    #[tokio::test]
    async fn handle_selects_through_trait() {
        let handle = PresentationUiHandle(Arc::new(NoopUi));
        let item = PresentationSelectItem {
            value: "a".into(),
            label: "A".into(),
            description: None,
        };
        let selected = handle
            .0
            .select(
                "Title",
                &[item],
                None,
                pi_chord::context::BACKGROUND_CONTEXT.clone(),
            )
            .await
            .unwrap();
        assert_eq!(selected, None);
    }
}
