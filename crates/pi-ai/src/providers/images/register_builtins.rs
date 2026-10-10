//! Port of `providers/images/register-builtins.ts` — built-in images API
//! provider registrations (final-integration wiring).

use crate::api::openrouter_images::generate_images_function;
use crate::images_api_registry::{register_images_api_provider, ImagesApiProvider};

/// Register the built-in images API providers. Upstream runs this at module
/// load (D10: static linking removes the module-load side effect), so Rust
/// entry points call [`ensure_built_in_images_api_providers`] instead.
pub fn register_built_in_images_api_providers() {
    register_images_api_provider(
        ImagesApiProvider {
            api: "openrouter-images".to_string(),
            generate_images: generate_images_function(),
        },
        None,
    );
}

/// Idempotent registration guard for lazy entry points. Upstream's module-load
/// side effect becomes an explicit once-only call (static linking, D10; the
/// lazy dynamic-import failure path upstream wraps cannot occur here).
pub fn ensure_built_in_images_api_providers() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(register_built_in_images_api_providers);
}
