//! Port of @earendil-works/chord (upstream: packages/chord).
//!
//! Chord is a standalone application-composition runtime for agentic applications.
//! Public re-exports mirror upstream `src/index.ts`; the Node entry re-exports
//! mirror `src/node.ts` (see [`node`]) and `src/bundler.ts` (see [`bundler`]).
//!
//! Upstream re-exports that have no Rust counterpart (documented deviation):
//! `isJsonValue`/`CopyJsonOptions` — JSON validity is enforced by construction
//! for [`serde_json::Value`]; see [`json`].

pub mod api;
pub mod bundler;
pub mod context;
pub mod delta;
pub mod facets;
pub mod json;
pub mod node;
pub mod services;
pub mod types;

// Mirrors upstream `src/index.ts` re-exports.
pub use api::{
    combine_facet_loaders, create_facet_host, create_remote_service_binding, create_static_facet_loader,
    attach_replicated_state, define_facet, define_service, replicated_state,
};
pub use context::{
    await_with_context, create_context_key, with_abort_signal, with_cancel, with_context_value,
    without_abort_signal, AbortController, AbortReason, AbortSignal, Context, ContextKey, BACKGROUND_CONTEXT,
    TODO_CONTEXT,
};
pub use delta::{
    apply, apply_immutable, apply_immutable_batches, apply_immutable_trusted, assert_safe_path, assert_valid_op,
    assert_valid_raw_op, assert_valid_wire_op, decoder, diff_revisions, encoder, is_base, is_base_decoded,
    is_replace, overlap, path_to_json, track, Decoder, DeltaError, Draft, Encoder, JsonRevisionValidator,
    NonEmptyPath, Op, Path, Prepared, Seg, Tracker, WireOp,
};
pub use json::{copy_json, JsonValue};
pub use services::errors::{
    is_remote_service_error_code, remote_service_error, AggregateError, AnyError, ChordError, RemoteServiceError,
    RemoteServiceErrorCode, TypeError, REMOTE_SERVICE_ERROR_CODES,
};
pub use services::provider::{
    create_remote_service_endpoint, validate_remote_service_implementation, ProviderDefinition, RemoteServiceEndpoint,
    RemoteServiceProvider, RemoteServiceObject, ServiceUpdatePublisher,
};
pub use services::state::{
    attach_replicated_state_source, service_delivery_context, AttachedReplicatedState, MutableReplicatedState,
    ReplicatedStateReplica,
};
pub use services::state_codec::{
    create_service_state_decoder, create_service_state_encoder, ServiceStateDecoder, ServiceStateEncoder,
};
pub use services::wire::{
    create_service_catalogue_call, create_service_subscribe_call, create_service_unsubscribe_call,
    decode_service_control_call, parse_service_call, parse_service_catalogue, parse_service_provider_update,
    parse_service_subscription_snapshot, parse_wire_service_provider_update, parse_wire_service_subscription_snapshot,
    ServiceControlCall, WireServiceInstanceSnapshot, WireServiceMemberSnapshot, WireServiceProviderUpdate,
    WireServiceSubscriptionSnapshot,
};
pub use services::consumer::RemoteServiceBinding;
pub use types::{
    Facet, FacetHost, FacetLoader, FacetOptions, LoadedFacets, ReplicatedState, ReplicatedStateDelivery,
    ReplicatedStateDeliveryKind, ReplicatedStateSource, ReplicatedStateSourceAttachment,
    ReplicatedStateSourceFrame, ReplicatedStateSourceOptions, RemoteServiceBindingOptions, RemoteServiceImpl,
    RemoteServiceSource, RemoteServiceTransport, RemoteServices, Service, ServiceCall, ServiceCatalogueEntry,
    ServiceClientHandle, ServiceInstanceAddress, ServiceInstanceSnapshot, ServiceMember, ServiceMemberSnapshot,
    ServiceMode, ServiceProviderUpdate, ServiceSubscription, ServiceSubscriptionSnapshot, StateListener,
    Unsubscribe, UpdateListener,
};

#[cfg(test)]
mod boundary_tests {
    //! Port of upstream `test/boundary.test.ts` (package boundary).
    //!
    //! Upstream scans source files for import specifiers escaping the package;
    //! the Rust equivalents are (a) the crate manifest must not depend on other
    //! `pi-*` workspace crates and (b) no source file may reference a foreign
    //! `pi_` crate path. Module privacy enforces the file-boundary half.

    use std::path::Path;

    #[test]
    fn does_not_depend_on_pi_packages_or_files_outside_chord() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("read Cargo.toml");
        let mut violations: Vec<String> = Vec::new();
        let mut section = String::new();
        for line in manifest.lines() {
            let line = line.trim();
            if line.starts_with('[') {
                section = line.trim_matches(['[', ']']).to_string();
                continue;
            }
            if section == "dependencies" || section == "dev-dependencies" {
                if let Some((name, _)) = line.split_once('=') {
                    let name = name.trim();
                    if name.starts_with("pi-") {
                        violations.push(format!("Cargo.toml: {name}"));
                    }
                }
            }
        }

        let src = manifest_dir.join("src");
        let mut stack = vec![src.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read_dir") {
                let entry = entry.expect("dir entry");
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("read source");
                // Build the needles from parts so this test's own source never
                // matches them.
                let crate_ref = format!("pi_{:}", "_::");
                let use_ref = format!("use pi_{:}", "_");
                for (index, line) in source.lines().enumerate() {
                    if line.contains(&crate_ref) || line.trim_start().starts_with(&use_ref) {
                        violations.push(format!(
                            "{}:{}: {}",
                            path.strip_prefix(manifest_dir).unwrap_or(&path).display(),
                            index + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
        assert_eq!(violations, Vec::<String>::new());
    }
}
