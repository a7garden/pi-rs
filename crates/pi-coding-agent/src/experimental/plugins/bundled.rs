//! Port of `experimental/plugins/bundled.ts` — the session facet loader over
//! built plugin bundles and the presentation-facet data codec.
//!
//! Deviations (D152):
//! - Upstream's `resolveExternal` callback (an `import.meta.url` JS resolver)
//!   becomes the pi-runtime externals registry: the
//!   `@earendil-works/pi-coding-agent/experimental/plugin` specifier resolves
//!   to a JSON encoding of the plugin barrel's service tokens (identity
//!   travels through chord service ids, not object references).
//! - `createPresentationFacetLoaders` (building loaders from server-sent
//!   artifacts) is the connected client's presentation side and defers with
//!   the headless-client slice; only the data codec and the session facet
//!   loader are ported here.
//! - Facet evaluation requires the embedded JS host
//!   ([`pi_runtime::FacetRuntimeModuleLoader`]); validation labels are read
//!   from the manifest at load time like upstream.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use pi_chord::api::combine_facet_loaders;
use pi_chord::node::bundle_loader::{
    create_facet_bundle_loader, read_facet_bundle_manifest, FacetBundleLoaderOptions,
    FacetBundleModuleLoader,
};
use pi_chord::node::manifest::FacetBundleArtifact;
use pi_chord::services::errors::{error, AnyError};
use pi_chord::types::{FacetLoader, LoadedFacets};
use serde_json::Value;

use crate::experimental::services::agent_controller::AGENT_CONTROLLER;
use crate::experimental::services::plugins::SESSION_PLUGINS;
use crate::experimental::services::presentation_ui::PRESENTATION_UI;
use crate::experimental::services::slash_commands::SLASH_COMMANDS;

const PRESENTATION_FACET_BUNDLES_KEY: &str = "presentationFacetBundles";
const PI_PLUGIN_API: &str = "@earendil-works/pi-coding-agent/experimental/plugin";

/// Upstream `createSessionPluginFacetLoader`: combine one optional session
/// loader per manifest; `None` when no manifests are configured.
pub fn create_session_plugin_facet_loader(
    manifest_paths: &[String],
) -> Option<Arc<dyn FacetLoader>> {
    if manifest_paths.is_empty() {
        return None;
    }
    Some(combine_facet_loaders(
        manifest_paths
            .iter()
            .map(|manifest_path| {
                optional_session_facet_loader(manifest_path) as Arc<dyn FacetLoader>
            })
            .collect(),
    ))
}

/// Upstream `createOptionalSessionFacetLoader`: load the bundle's `session`
/// entry when the manifest declares one; produce an empty facet set
/// otherwise.
fn optional_session_facet_loader(manifest_path: &str) -> Arc<dyn FacetLoader> {
    struct OptionalSessionLoader {
        manifest_path: PathBuf,
    }
    impl FacetLoader for OptionalSessionLoader {
        fn load(&self) -> pi_chord::types::BoxFuture<Result<LoadedFacets, AnyError>> {
            let manifest_path = self.manifest_path.clone();
            Box::pin(async move {
                let manifest = read_facet_bundle_manifest(&manifest_path)?;
                if !manifest.entries.contains_key("session") {
                    return Ok(LoadedFacets {
                        facets: Vec::new(),
                        dispose: None,
                    });
                }
                // Validation labels mirror the manifest read upstream performs.
                let plugin_id = manifest.plugin.id.clone();
                let module_loader: Arc<dyn FacetBundleModuleLoader> = Arc::new(
                    pi_runtime::FacetRuntimeModuleLoader::new(pi_runtime::FacetRuntimeOptions {
                        plugin_id,
                        entry_name: "session".to_string(),
                        externals: plugin_externals(),
                    }),
                );
                create_facet_bundle_loader(
                    FacetBundleLoaderOptions {
                        manifest_path,
                        entry: "session".to_string(),
                        verify_integrity: None,
                    },
                    module_loader,
                )
                .load()
                .await
            })
        }
    }
    Arc::new(OptionalSessionLoader {
        manifest_path: PathBuf::from(manifest_path),
    })
}

/// Upstream `createPresentationFacetData`: the server-sent plugin payload
/// carrying the selected presentation artifacts.
pub fn create_presentation_facet_data(artifacts: &[FacetBundleArtifact]) -> Value {
    serde_json::json!({
        PRESENTATION_FACET_BUNDLES_KEY: artifacts.iter().map(artifact_to_json).collect::<Vec<_>>(),
    })
}

/// Encode one artifact as its upstream plain-object shape
/// (`artifact as unknown as JsonValue`).
fn artifact_to_json(artifact: &FacetBundleArtifact) -> Value {
    let mut value = serde_json::json!({
        "plugin": serde_json::to_value(&artifact.plugin).expect("plugin serializes"),
        "entryName": artifact.entry_name,
        "entry": serde_json::to_value(&artifact.entry).expect("entry serializes"),
        "source": artifact.source,
    });
    if let Some(source_map_contents) = &artifact.source_map_contents {
        value["sourceMapContents"] = Value::from(source_map_contents.as_str());
    }
    value
}

/// The plugin API specifier resolves to the barrel's service tokens encoded
/// as JSON (D152 — identity flows through chord service ids).
fn plugin_externals() -> HashMap<String, Value> {
    HashMap::from([(
        PI_PLUGIN_API.to_string(),
        serde_json::json!({
            "AgentController": { "id": AGENT_CONTROLLER.id() },
            "SessionPlugins": { "id": SESSION_PLUGINS.id() },
            "PresentationUI": { "id": PRESENTATION_UI.id() },
            "SlashCommands": { "id": SLASH_COMMANDS.id() },
        }),
    )])
}

/// Validate a presentation-facet payload shape (upstream
/// `createPresentationFacetLoaders` guards). Deferred loader construction
/// shares this check with the future client slice.
pub fn validate_presentation_facet_data(data: &Value) -> Result<(), AnyError> {
    if !data.is_object() {
        return Err(error("Invalid presentation plugin data"));
    }
    match data.get(PRESENTATION_FACET_BUNDLES_KEY) {
        None => Ok(()),
        Some(bundles) if bundles.is_array() => Ok(()),
        Some(_) => Err(error("Invalid presentation plugin bundle list")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_manifest_paths_have_no_loader() {
        assert!(create_session_plugin_facet_loader(&[]).is_none());
        assert!(create_session_plugin_facet_loader(&["/does/not/matter.json".into()]).is_some());
    }

    #[test]
    fn presentation_data_round_trips_artifacts() {
        let artifact = FacetBundleArtifact {
            plugin: pi_chord::node::manifest::FacetBundlePlugin {
                id: "pi-demo".into(),
                version: Some("1.0.0".into()),
            },
            entry_name: "tui".into(),
            entry: pi_chord::node::manifest::FacetBundleEntry {
                file: "tui-abc.js".into(),
                integrity: "sha256-000".into(),
                external_imports: vec![PI_PLUGIN_API.into()],
                source_map: None,
            },
            source: "// bundled".into(),
            source_map_contents: None,
        };
        let data = create_presentation_facet_data(&[artifact]);
        let bundles = data[PRESENTATION_FACET_BUNDLES_KEY].as_array().unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0]["entryName"], "tui");
        assert_eq!(bundles[0]["plugin"]["id"], "pi-demo");
        assert_eq!(bundles[0]["source"], "// bundled");
        assert!(bundles[0].get("sourceMapContents").is_none());
        validate_presentation_facet_data(&data).unwrap();
        assert!(validate_presentation_facet_data(&Value::Null).is_err());
    }

    #[test]
    fn plugin_externals_carry_service_ids() {
        let externals = plugin_externals();
        let plugin = &externals[PI_PLUGIN_API];
        assert_eq!(plugin["AgentController"]["id"], AGENT_CONTROLLER.id());
        assert_eq!(plugin["PresentationUI"]["id"], PRESENTATION_UI.id());
    }
}
