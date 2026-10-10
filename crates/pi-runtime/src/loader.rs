//! Module loading for the embedded runtime: TypeScript/JSX transpile plus a
//! virtual-module registry.
//!
//! deno_core 0.412's built-in `FsModuleLoader` does **not** transpile — `.ts`
//! files are handed to V8 as-is and fail to parse. This loader mirrors the
//! deno_core `ts_module_loader` example (parse + transpile via `deno_ast`/swc)
//! and adds a virtual registry:
//!
//! - **node-module shims.** Bare specifiers (`node:fs`, npm names) resolve to
//!   registered virtual modules (JSON exports serialized as an ESM facade).
//!   Nothing is registered by default: there are no node/npm compatibility
//!   shims beyond what an embedder explicitly provides.
//! - **facet bundle generations.** The facet loader registers each generation's
//!   bundled modules under `pi-facet://` URLs (with file-path aliases so
//!   relative imports inside the bundle keep working).

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use deno_ast::{MediaType, ParseParams, SourceMapOption};
use deno_core::{
    resolve_import, ModuleLoadResponse, ModuleLoader, ModuleSource, ModuleSourceCode,
    ModuleSpecifier, ModuleType, ResolutionKind,
};
use deno_error::JsErrorBox;
use parking_lot::Mutex;

/// Source of a virtual module.
#[derive(Clone)]
pub(crate) enum VirtualSource {
    /// JavaScript source text.
    Js(String),
}

/// Module loader shared by the extension host and the facet loader.
///
/// Interior mutability uses `Mutex` so the loader is `Send + Sync`: the JS
/// thread's async loop must stay `Send` even though only that thread touches
/// the registry.
pub(crate) struct PiModuleLoader {
    /// Canonical sources: the exact specifier/URL string → source.
    virtuals: Mutex<HashMap<String, VirtualSource>>,
    /// Alias URLs → canonical specifier (`file://` path variants probing).
    aliases: Mutex<HashMap<String, String>>,
    /// Transpiled source maps per specifier (for JS stack traces).
    source_maps: Mutex<HashMap<String, Vec<u8>>>,
}

impl PiModuleLoader {
    pub(crate) fn new() -> Rc<Self> {
        Rc::new(Self {
            virtuals: Mutex::new(HashMap::new()),
            aliases: Mutex::new(HashMap::new()),
            source_maps: Mutex::new(HashMap::new()),
        })
    }

    /// Register a virtual module under an exact specifier string. Bare
    /// specifiers that are not valid URLs (npm-style names) get a synthetic
    /// `pi-virtual:` URL alias so resolution has a URL to hand around.
    pub(crate) fn register_virtual(&self, specifier: String, source: VirtualSource) {
        if ModuleSpecifier::parse(&specifier).is_err() {
            let alias = format!("pi-virtual:{specifier}");
            self.aliases.lock().insert(alias, specifier.clone());
        }
        self.virtuals.lock().insert(specifier, source);
    }

    /// Map `alias` to an already-registered canonical specifier.
    pub(crate) fn register_alias(&self, alias: String, canonical: String) {
        self.aliases.lock().insert(alias, canonical);
    }

    /// Register the alias forms a bundled module's path may be imported under:
    /// the exact file URL, the extension-less form, and common extension
    /// rewrites (`.ts` source imported as `.js`/`.mjs`).
    pub(crate) fn register_path_aliases(&self, path: &Path, canonical: &str) -> Vec<String> {
        let Ok(file_url) = ModuleSpecifier::from_file_path(path) else {
            return Vec::new();
        };
        let mut aliases = vec![file_url.to_string()];
        let as_str = file_url.to_string();
        if path.file_stem().is_some() {
            // Extension-less and cross-extension forms, e.g. `./helper` and
            // `./helper.js` when the source file is `helper.ts`.
            if let Some(stripped) = as_str.strip_suffix(&format!(
                ".{}",
                path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or_default()
            )) {
                let base = stripped.to_string();
                aliases.push(base.clone());
                for ext in ["js", "mjs", "cjs", "ts", "mts", "cts", "tsx"] {
                    if path
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|actual| actual == ext)
                    {
                        continue;
                    }
                    aliases.push(format!("{base}.{ext}"));
                }
            }
        }
        for alias in &aliases {
            self.register_alias(alias.clone(), canonical.to_string());
        }
        aliases
    }

    fn transpile_to_js(
        &self,
        specifier: &ModuleSpecifier,
        media_type: MediaType,
        code: String,
    ) -> Result<String, JsErrorBox> {
        let parsed = deno_ast::parse_module(ParseParams {
            specifier: specifier.clone(),
            text: code.into(),
            media_type,
            capture_tokens: false,
            scope_analysis: false,
            maybe_syntax: None,
        })
        .map_err(|e| JsErrorBox::generic(format!("{e}")))?;
        let transpiled = parsed
            .transpile(
                &deno_ast::TranspileOptions {
                    imports_not_used_as_values: deno_ast::ImportsNotUsedAsValues::Remove,
                    decorators: deno_ast::DecoratorsTranspileOption::Ecma,
                    ..Default::default()
                },
                &deno_ast::TranspileModuleOptions::default(),
                &deno_ast::EmitOptions {
                    source_map: SourceMapOption::Separate,
                    inline_sources: true,
                    ..Default::default()
                },
            )
            .map_err(|e| JsErrorBox::generic(format!("{e}")))?
            .into_source();
        if let Some(map) = transpiled.source_map {
            self.source_maps
                .lock()
                .insert(specifier.to_string(), map.into_bytes());
        }
        Ok(transpiled.text)
    }
}

impl ModuleLoader for PiModuleLoader {
    /// Pump the event loop during recursive loads: side-module loads
    /// (`load_side_es_module`, used for every extension after the first)
    /// progress their dependency graph through the event loop.
    fn pump_event_loop_during_load(&self) -> bool {
        true
    }

    fn resolve(
        &self,
        specifier: &str,
        referrer: &str,
        _kind: ResolutionKind,
    ) -> Result<ModuleSpecifier, JsErrorBox> {
        // Bare specifiers registered as virtual modules resolve to themselves
        // (or to their `pi-virtual:` alias when the specifier is not a URL).
        if self.virtuals.lock().contains_key(specifier) {
            if let Ok(url) = ModuleSpecifier::parse(specifier) {
                return Ok(url);
            }
            return ModuleSpecifier::parse(&format!("pi-virtual:{specifier}"))
                .map_err(|e| JsErrorBox::generic(format!("invalid specifier {specifier}: {e}")));
        }
        let resolved = resolve_import(specifier, referrer).map_err(JsErrorBox::from_err)?;
        // Fail missing file modules at resolve time: deno_core 0.412 swallows
        // a failing `load` response for a static import silently, and the
        // module then links as an empty stub.
        if resolved.scheme() == "file"
            && !self.virtuals.lock().contains_key(resolved.as_str())
            && !self.aliases.lock().contains_key(resolved.as_str())
            && resolved
                .to_file_path()
                .map(|path| !path.exists())
                .unwrap_or(false)
        {
            return Err(JsErrorBox::generic(format!(
                "Module not found: {specifier} (resolved to {resolved}, imported from {referrer})"
            )));
        }
        Ok(resolved)
    }

    fn load(
        &self,
        module_specifier: &ModuleSpecifier,
        _maybe_referrer: Option<&deno_core::ModuleLoadReferrer>,
        _options: deno_core::ModuleLoadOptions,
    ) -> ModuleLoadResponse {
        let specifier = module_specifier.clone();
        ModuleLoadResponse::Sync(
            (move || {
                // Alias → canonical virtual module.
                let canonical = self.aliases.lock().get(specifier.as_str()).cloned();
                if let Some(canonical) = canonical {
                    if let Some(source) = self.virtuals.lock().get(&canonical).cloned() {
                        return virtual_source(&specifier, source);
                    }
                }
                // Direct virtual module (bare specifier or exact URL).
                if let Some(source) = self.virtuals.lock().get(specifier.as_str()).cloned() {
                    return virtual_source(&specifier, source);
                }
                // Filesystem module, transpiling TypeScript/JSX.
                let path = specifier
                    .to_file_path()
                    .map_err(|_| JsErrorBox::generic(format!("Only file:// URLs are supported: {specifier}")))?;
                let media_type = MediaType::from_path(&path);
                let module_type = match media_type {
                    MediaType::Json => ModuleType::Json,
                    MediaType::JavaScript | MediaType::Mjs | MediaType::Cjs => ModuleType::JavaScript,
                    MediaType::TypeScript
                    | MediaType::Mts
                    | MediaType::Cts
                    | MediaType::Tsx
                    | MediaType::Jsx
                    // Declaration files are never executed; treat like TS so the
                    // error is a transpile error rather than a parse mystery.
                    | MediaType::Dts | MediaType::Dmts | MediaType::Dcts => ModuleType::JavaScript,
                    other => {
                        return Err(JsErrorBox::generic(format!(
                            "Unsupported module media type {other:?} for {specifier}"
                        )));
                    }
                };
                let code = std::fs::read_to_string(&path)
                    .map_err(|e| JsErrorBox::generic(format!("Failed to read {path:?}: {e}")))?;
                let code = if media_type == MediaType::Json {
                    code
                } else {
                    self.transpile_to_js(&specifier, media_type, code)?
                };
                Ok(ModuleSource::new(
                    module_type,
                    ModuleSourceCode::String(code.into()),
                    &specifier,
                    None,
                ))
            })()
            .map_err(JsErrorBox::from_err),
        )
    }

    fn get_source_map(&self, specifier: &str) -> Option<std::borrow::Cow<'_, [u8]>> {
        self.source_maps
            .lock()
            .get(specifier)
            .map(|map| map.clone().into())
    }
}

fn virtual_source(
    specifier: &ModuleSpecifier,
    source: VirtualSource,
) -> Result<ModuleSource, JsErrorBox> {
    let VirtualSource::Js(code) = source;
    Ok(ModuleSource::new(
        ModuleType::JavaScript,
        ModuleSourceCode::String(code.into()),
        specifier,
        None,
    ))
}
