//! [`FacetCompiler`] implementation: a boring esbuild-lite for facet entries.
//!
//! Upstream runs esbuild with `format: "cjs"` over a TypeScript entry and all
//! statically reachable modules. This compiler bundles the same material with
//! `deno_ast`/swc (the same toolchain deno_core's own TypeScript loader uses):
//!
//! 1. Walk the static import graph from the entry (relative specifiers only;
//!    everything else must be declared external, mirroring the request).
//! 2. Transpile each module TypeScript→ESM (type stripping).
//! 3. Emit one CommonJS artifact file that carries the ESM sources in a module
//!    map; the matching [`crate::facets::FacetRuntimeModuleLoader`] evaluates
//!    the artifact's CJS shell, then links the ESM modules with V8's native
//!    module system and external imports as registered facade modules.
//!
//! Deviations from esbuild (ledger in the crate docs): `define` is applied
//! only for identifier keys; `minify`, `target`, and `platform` are accepted
//! but do not lower/minify (the embedded V8 targets current ES); emitted source
//! maps list the bundled sources without mappings.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use deno_ast::{MediaType, ParseParams};
use deno_core::ModuleSpecifier;
use pi_chord::node::bundle::{
    FacetBuildMessages, FacetBuildOutput, FacetBuildRequest, FacetCompiler,
};
use pi_chord::services::errors::{error, AnyError};

use crate::host::sha256_hex;

/// The bundle format marker understood by [`crate::facets::FacetRuntimeModuleLoader`].
pub const BUNDLE_FORMAT: &str = "pi-facet-esm-bundle";
pub const BUNDLE_FORMAT_VERSION: u64 = 1;

/// esbuild-lite facet compiler; stateless.
#[derive(Debug, Default)]
pub struct FacetBundleCompiler;

#[async_trait::async_trait]
impl FacetCompiler for FacetBundleCompiler {
    async fn build(&self, request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError> {
        compile_inner(request)
    }
}

fn compile_inner(request: FacetBuildRequest) -> Result<FacetBuildOutput, AnyError> {
    if request.entry_points.len() != 1 {
    return Err(error(format!(
            "Facet builds must have exactly one entry point, got {}",
            request.entry_points.len()
    )));
    }
    for key in request.define.keys() {
            if !is_identifier(key) {
                return Err(FacetBuildMessages {
                    messages: vec![format!(
                        "define target \"{key}\" is not supported; only plain identifiers are"
                    )],
                }
                .into());
            }
    }
    let entry = &request.entry_points[0];
    let external: BTreeSet<String> = request.external.iter().cloned().collect();
    // Records of the names each external is imported by, for facade modules.
    let mut external_names: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    // Walk the static import graph.
    let mut modules: BTreeMap<String, String> = BTreeMap::new(); // abs path → ESM source
    let mut queue = vec![entry.clone()];
    while let Some(path) = queue.pop() {
            let key = path_key(&path)?;
            if modules.contains_key(&key) {
                continue;
            }
            let source = std::fs::read_to_string(&path)
                .map_err(|e| error(format!("Could not read facet entry module {path:?}: {e}")))?;
            let media_type = MediaType::from_path(&path);
            let parsed = deno_ast::parse_module(ParseParams {
                specifier: module_specifier_for(&path)?,
                text: source.clone().into(),
                media_type,
                capture_tokens: false,
                scope_analysis: false,
                maybe_syntax: None,
            })
            .map_err(|e| error(format!("Could not parse facet entry module {path:?}: {e}")))?;
            let deno_ast::ProgramRef::Module(module) = parsed.program_ref() else {
                return Err(error(format!("Facet entry module {path:?} is not an ES module")));
            };
            let mut imports: Vec<String> = Vec::new();
            for item in &module.body {
                let decl = match item {
                    deno_ast::swc::ast::ModuleItem::ModuleDecl(decl) => decl,
                    _ => continue,
                };
                match decl {
                    deno_ast::swc::ast::ModuleDecl::Import(import) => {
                        let source = import_src_value(&import.src.value);
                        imports.push(source.clone());
                        // Named-import records matter only for externals; the
                        // bundled modules resolve their own names.
                        if !is_relative(&source) {
                            let names = external_names.entry(source).or_default();
                            for specifier in &import.specifiers {
                                if let deno_ast::swc::ast::ImportSpecifier::Named(named) = specifier {
                                    // `import { x as y }` must export `x`.
                                    let exported = match &named.imported {
                                        Some(imported) => match imported {
                                            deno_ast::swc::ast::ModuleExportName::Ident(ident) => ident.sym.to_string(),
                                            deno_ast::swc::ast::ModuleExportName::Str(str) => {
                                                str.value.to_atom_lossy().to_string()
                                            }
                                        },
                                        None => named.local.sym.to_string(),
                                    };
                                    names.insert(exported);
                                }
                            }
                        }
                    }
                    deno_ast::swc::ast::ModuleDecl::ExportAll(export) => {
                        imports.push(import_src_value(&export.src.value));
                    }
                    deno_ast::swc::ast::ModuleDecl::ExportNamed(export) => {
                        if let Some(src) = &export.src {
                            imports.push(import_src_value(&src.value));
                        }
                    }
                    deno_ast::swc::ast::ModuleDecl::ExportDefaultDecl(_)
                    | deno_ast::swc::ast::ModuleDecl::ExportDefaultExpr(_) => {}
                    other => {
                        if let deno_ast::swc::ast::ModuleDecl::TsImportEquals(_) = other {
                            return Err(error(format!(
                                "Facet entry module {path:?} uses `import x = require(...)`, which cannot be bundled"
                            )));
                        }
                    }
                }
            }
            // Dynamic imports cannot be followed statically. The check is
            // textual on purpose: a rare false positive on a string containing
            // `import(` fails the build loudly instead of silently dropping a
            // dependency (documented deviation).
            if source.contains("import(") {
                return Err(error(format!(
                    "Facet entry module {path:?} uses a dynamic import(), which cannot be bundled; declare the module external or import it statically"
                )));
            }

            // Transpile TypeScript → ESM JavaScript. Identifier `define`
            // targets become top-level bindings inside each module (esbuild
            // replaces identifier occurrences; this is the boring equivalent).
            let js = if matches!(
                media_type,
                MediaType::JavaScript | MediaType::Mjs | MediaType::Cjs
            ) {
                source
            } else {
                transpile_to_esm(&path, media_type, &source)?
            };
            let mut js = js;
            for (key, value) in &request.define {
                if is_identifier(key) {
                    js = format!("const {key} = {value};\n{js}");
                }
            }
            modules.insert(key.clone(), js);

            for specifier in imports {
                if is_relative(&specifier) {
                    let resolved = resolve_relative(&path, &specifier)?;
                    queue.push(resolved);
                } else {
                    if !external.contains(&specifier) && !is_builtin(&specifier) {
                        return Err(FacetBuildMessages {
                            messages: vec![format!(
                                "Could not resolve import \"{specifier}\" in {key}: it is not in the build's `external` list and cannot be bundled"
                            )],
                        }
                        .into());
                    }
                    external_names.entry(specifier).or_default();
                }
            }
    }

    // Emit the single-file CJS artifact carrying the module map.
    let artifact = serde_json::json!({
            "format": BUNDLE_FORMAT,
            "formatVersion": BUNDLE_FORMAT_VERSION,
            "entry": path_key(entry)?,
            "modules": modules,
            "externalNames": external_names
                .iter()
                .map(|(specifier, names)| (specifier.clone(), names.iter().cloned().collect::<Vec<_>>()))
                .collect::<BTreeMap<_, _>>(),
    });
    let mut contents = String::new();
    if !request.banner.is_empty() {
            contents.push_str(request.banner.trim_end());
            contents.push('\n');
    }
    contents.push_str("module.exports = ");
    contents.push_str(&serde_json::to_string(&artifact).map_err(|e| error(e.to_string()))?);
    contents.push_str(";\n");

    let contents = contents.into_bytes();
    let output_name = render_entry_name(&request.entry_names, entry, &contents)?;
    let mut output_files = vec![(PathBuf::from(&output_name), contents.clone())];
    let source_map_path = request.source_map.then(|| format!("{output_name}.map"));
    if let Some(map_name) = &source_map_path {
            let sources: Vec<&String> = modules.keys().collect();
            let map = serde_json::json!({
                "version": 3,
                "file": output_name,
                "sources": sources,
                "sourcesContent": modules.values().collect::<Vec<_>>(),
                "mappings": "",
            });
            let map_bytes = serde_json::to_vec(&map).map_err(|e| error(e.to_string()))?;
            output_files.push((PathBuf::from(map_name), map_bytes));
    }

    Ok(FacetBuildOutput {
        output_files,
        entry_output: output_name,
        external_imports: external_names.keys().cloned().collect(),
        source_map_path,
        messages: Vec::new(),
    })
}

fn transpile_to_esm(path: &Path, media_type: MediaType, source: &str) -> Result<String, AnyError> {
    let specifier = module_specifier_for(path)?;
    let parsed = deno_ast::parse_module(ParseParams {
        specifier,
        text: source.to_owned().into(),
        media_type,
        capture_tokens: false,
        scope_analysis: false,
        maybe_syntax: None,
    })
    .map_err(|e| error(format!("Could not parse facet entry module {path:?}: {e}")))?;
    let transpiled = parsed
        .transpile(
            &deno_ast::TranspileOptions {
                imports_not_used_as_values: deno_ast::ImportsNotUsedAsValues::Remove,
                decorators: deno_ast::DecoratorsTranspileOption::Ecma,
                ..Default::default()
            },
            &deno_ast::TranspileModuleOptions::default(),
            &deno_ast::EmitOptions::default(),
        )
        .map_err(|e| error(format!("Could not transpile facet entry module {path:?}: {e}")))?;
    Ok(transpiled.into_source().text)
}

/// Canonical bundle key for a module: its normalized absolute path.
fn path_key(path: &Path) -> Result<String, AnyError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| error(format!("Could not resolve working directory: {e}")))?
            .join(path)
    };
    Ok(normalize_path(&absolute))
}

fn module_specifier_for(path: &Path) -> Result<ModuleSpecifier, AnyError> {
    let key = path_key(path)?;
    ModuleSpecifier::parse(&format!("file://{key}"))
        .map_err(|e| error(format!("Could not make a module specifier for {key}: {e}")))
}

/// Lexical path normalization (no symlink resolution): collapses `.`/`..`.
fn normalize_path(path: &Path) -> String {
    let mut parts: Vec<std::ffi::OsString> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop();
            }
            other => parts.push(other.as_os_str().to_os_string()),
        }
    }
    let mut normalized = PathBuf::new();
    for part in parts {
        normalized.push(part);
    }
    normalized.to_string_lossy().to_string()
}

/// Resolve a relative specifier against an importing file with the extension
/// probing rules shared with the loader's alias table.
fn resolve_relative(from: &Path, specifier: &str) -> Result<PathBuf, AnyError> {
    let base = from.parent().ok_or_else(|| error("module has no parent directory"))?;
    let joined = base.join(specifier);
    let candidates = extension_candidates(&joined);
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(error(format!(
        "Could not resolve import \"{specifier}\" from {from:?}"
    )))
}

fn extension_candidates(path: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![path.to_path_buf()];
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        let stripped = path.with_extension("");
        match extension {
            "js" | "mjs" | "cjs" => {
                candidates.push(stripped.with_extension("ts"));
                candidates.push(stripped.with_extension("tsx"));
            }
            _ => {}
        }
    } else {
        for ext in ["ts", "tsx", "js", "mjs"] {
            candidates.push(path.with_extension(ext));
        }
        for ext in ["ts", "js"] {
            candidates.push(path.join(format!("index.{ext}")));
        }
    }
    candidates
}

fn is_builtin(specifier: &str) -> bool {
    specifier.starts_with("node:")
}

fn is_relative(specifier: &str) -> bool {
    specifier.starts_with("./") || specifier.starts_with("../") || specifier == "." || specifier == ".."
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(first) if (first.is_alphabetic() || first == '_' || first == '$'))
        && chars.all(|rest| rest.is_alphanumeric() || rest == '_' || rest == '$')
}

/// Render the esbuild-style `entryNames` template: `[dir]`, `[name]`, `[hash]`.
fn render_entry_name(template: &str, entry: &Path, contents: &[u8]) -> Result<String, AnyError> {
    let name = entry
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| error("Facet entry has no file name"))?;
    let hash = &sha256_hex(contents)[..8];
    let rendered = template
        .replace("[dir]", "")
        .replace("[name]", name)
        .replace("[hash]", hash);
    if rendered.is_empty() {
        return Err(error("Facet entry produced an empty output name"));
    }
    Ok(format!("{rendered}.cjs"))
}

/// Import source strings are ordinary strings; read them lossily.
fn import_src_value(value: &deno_ast::swc::atoms::Wtf8Atom) -> String {
    value.to_atom_lossy().to_string()
}
