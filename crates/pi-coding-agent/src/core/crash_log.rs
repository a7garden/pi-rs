//! Port of `packages/coding-agent/src/core/crash-log.ts` — the on-disk crash
//! log (`crashes.json`) plus stack-trace attribution to loaded extensions.
//!
//! Deviations:
//! - **D117 — `recordCrash`'s `error: unknown` becomes explicit
//!   `message`/`stack` arguments** (Rust errors carry no `.stack`, so the
//!   caller extracts `error.message || error.name` and the stack). The
//!   lenient read keeps upstream's filter (records need string `timestamp`
//!   and `message`) but absent optional fields get defaults
//!   (`kind` → `uncaught_exception`, `version`/`cwd` → `""`, `stack`/
//!   `sessionFile`/`notified` → `None`) where upstream keeps `undefined`.
//! - The optional `path`/`now` parameters become explicit arguments;
//!   [`crash_log_default_path`] supplies the `getAgentDir()` default and
//!   callers pass [`crate::utils::time::now_ms`] for `now`.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::config::{get_agent_dir, VERSION};
use crate::core::extensions::types::Extension;
use crate::core::source_info::{is_synthetic_path, SourceInfo, SourceOrigin};
use crate::utils::paths::join;
use crate::utils::time::{now_ms, parse_timestamp_ms, to_iso_timestamp};

/// One persisted crash (`CrashRecord`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrashRecord {
    pub timestamp: String,
    pub version: String,
    pub kind: CrashKind,
    pub message: String,
    pub stack: Option<String>,
    pub session_file: Option<String>,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notified: Option<bool>,
}

/// Why the process crashed (`CrashRecord["kind"]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrashKind {
    #[serde(rename = "uncaught_exception")]
    UncaughtException,
    #[serde(rename = "fatal_error")]
    FatalError,
}

impl Default for CrashKind {
    fn default() -> Self {
        Self::UncaughtException
    }
}

const MAX_CRASH_RECORDS: usize = 5;
const MAX_AGE: i64 = 7 * 24 * 60 * 60 * 1000;

/// `crashLogPath(agentDir)` for an explicit agent dir.
pub fn crash_log_path(agent_dir: &str) -> String {
    join(agent_dir, "crashes.json")
}

/// `crashLogPath()` — the default path under `getAgentDir()`.
pub fn crash_log_default_path() -> String {
    crash_log_path(&get_agent_dir())
}

/// `readCrashLog` — parse `crashes.json`, keeping records that carry string
/// `timestamp` and `message` fields; any read/parse failure yields `[]`.
pub fn read_crash_log(path: &str) -> Vec<CrashRecord> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(records) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let Some(records) = records.as_array() else {
        return Vec::new();
    };
    records
        .iter()
        .filter_map(|record| {
            let object = record.as_object()?;
            let timestamp = object.get("timestamp")?.as_str()?.to_string();
            let message = object.get("message")?.as_str()?.to_string();
            let kind = object
                .get("kind")
                .and_then(|kind| serde_json::from_value::<CrashKind>(kind.clone()).ok())
                .unwrap_or_default();
            Some(CrashRecord {
                timestamp,
                version: object
                    .get("version")
                    .and_then(|version| version.as_str())
                    .unwrap_or_default()
                    .to_string(),
                kind,
                message,
                stack: object
                    .get("stack")
                    .and_then(|stack| stack.as_str())
                    .map(str::to_string),
                session_file: object
                    .get("sessionFile")
                    .and_then(|file| file.as_str())
                    .map(str::to_string),
                cwd: object
                    .get("cwd")
                    .and_then(|cwd| cwd.as_str())
                    .unwrap_or_default()
                    .to_string(),
                notified: object
                    .get("notified")
                    .and_then(|notified| notified.as_bool()),
            })
        })
        .collect()
}

/// `writeCrashLog` — recursive mkdir + pretty-printed JSON with a trailing
/// newline.
fn write_crash_log(records: &[CrashRecord], path: &str) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_string_pretty(records).map_err(std::io::Error::other)?;
    json.push('\n');
    std::fs::write(path, json)
}

/// Loaded-extension metadata a stack can be attributed against
/// (`Pick<Extension, "path" | "resolvedPath" | "sourceInfo">`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensionStackMetadata {
    pub path: String,
    pub resolved_path: String,
    pub source_info: SourceInfo,
}

impl From<&Extension> for ExtensionStackMetadata {
    fn from(extension: &Extension) -> Self {
        Self {
            path: extension.path.clone(),
            resolved_path: extension.resolved_path.clone(),
            source_info: extension.source_info.clone(),
        }
    }
}

/// `normalizeStackPath` — backslashes to slashes, trailing slashes trimmed.
fn normalize_stack_path(value: &str) -> String {
    value.replace('\\', "/").trim_end_matches('/').to_string()
}

/// `stackContainsPath`.
fn stack_contains_path(stack: &str, target_path: &str, include_descendants: bool) -> bool {
    let target = normalize_stack_path(target_path);
    if target.is_empty() || is_synthetic_path(&target) {
        return false;
    }
    // Windows drive letters compare case-insensitively (`/^[a-z]:\//iu`).
    let case_insensitive = {
        let bytes = target.as_bytes();
        bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'/'
    };
    let (haystack, needle) = if case_insensitive {
        (stack.to_lowercase(), target.to_lowercase())
    } else {
        (stack.to_string(), target)
    };
    if include_descendants {
        return haystack.contains(&format!("{needle}/"));
    }

    let mut search_from = 0;
    while let Some(index) = haystack[search_from..].find(&needle) {
        let index = search_from + index;
        let next = haystack[index + needle.len()..].chars().next();
        match next {
            None | Some(':') | Some(')') => return true,
            Some(next) if next.is_whitespace() || next == '\u{feff}' => return true,
            _ => {}
        }
        search_from = index + needle.len();
    }
    false
}

/// JS `\s` (notably includes the BOM, which [`char::is_whitespace`] lacks).
fn is_js_whitespace(character: char) -> bool {
    character.is_whitespace() || character == '\u{feff}'
}

/// `/^\s+at\s/u` — a JS stack frame line.
fn is_stack_frame_line(line: &str) -> bool {
    let after_indent = line.trim_start_matches(is_js_whitespace);
    if after_indent.len() == line.len() {
        return false;
    }
    let Some(after_at) = after_indent.strip_prefix("at") else {
        return false;
    };
    after_at.chars().next().is_some_and(is_js_whitespace)
}

/// `decodeURI` — percent-decodes everything except the reserved characters;
/// `None` on any malformed sequence (upstream lets the `URIError` fall back
/// to the original line).
fn decode_uri(value: &str) -> Option<String> {
    if !value.contains('%') {
        return Some(value.to_string());
    }
    const RESERVED: &[u8] = b";/?:@&=+$,#";
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte != b'%' {
            decoded.push(byte);
            index += 1;
            continue;
        }
        let hex = bytes.get(index + 1..index + 3)?;
        let value = u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()?;
        if RESERVED.contains(&value) {
            decoded.extend_from_slice(&bytes[index..index + 3]);
        } else {
            decoded.push(value);
        }
        index += 3;
    }
    String::from_utf8(decoded).ok()
}

/// `/^(?:npm:|git:|https?:\/\/|ssh:\/\/)/u` — registry/remote package source.
fn is_registry_or_remote_source(source: &str) -> bool {
    source.starts_with("npm:")
        || source.starts_with("git:")
        || source.starts_with("http://")
        || source.starts_with("https://")
        || source.starts_with("ssh://")
}

/// `/\.[cm]?[jt]s$/u` — a `.js`/`.ts`-family file name.
fn is_js_ts_file(path: &str) -> bool {
    [".js", ".ts", ".cjs", ".cts", ".mjs", ".mts"]
        .iter()
        .any(|extension| path.ends_with(extension))
}

/// `/\/index\.[cm]?[jt]s$/u` — a package directory entry point.
fn is_index_entry(path: &str) -> bool {
    [
        "/index.js",
        "/index.ts",
        "/index.cjs",
        "/index.cts",
        "/index.mjs",
        "/index.mts",
    ]
    .iter()
    .any(|suffix| path.ends_with(suffix))
}

/// Find loaded extensions with source files in a stack trace
/// (`findExtensionStackMatches`). The first line (the error message) is
/// skipped; frame lines must match `^\s+at\s` and are `decodeURI`d before
/// matching.
pub fn find_extension_stack_matches(
    stack: Option<&str>,
    extensions: &[ExtensionStackMetadata],
) -> Vec<String> {
    let Some(stack) = stack else {
        return Vec::new();
    };
    let normalized_stack = stack
        .split('\n')
        .skip(1)
        .filter(|line| is_stack_frame_line(line))
        .map(|line| decode_uri(line).unwrap_or_else(|| line.to_string()))
        .collect::<Vec<_>>()
        .join("\n")
        .replace('\\', "/");
    let mut matches: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for extension in extensions {
        let resolved_path = normalize_stack_path(&extension.resolved_path);
        let source_info = &extension.source_info;
        let single_file_package = source_info.origin == SourceOrigin::Package
            && !is_registry_or_remote_source(&source_info.source)
            && is_js_ts_file(&source_info.source);
        let package_root = if source_info.origin == SourceOrigin::Package && !single_file_package {
            source_info
                .base_dir
                .clone()
                .filter(|base_dir| !base_dir.is_empty())
        } else {
            None
        };
        let slash_index = resolved_path.rfind('/');
        let directory_entry = is_index_entry(&resolved_path);
        let matched = if let Some(package_root) = package_root {
            stack_contains_path(&normalized_stack, &package_root, true)
        } else if directory_entry && slash_index.is_some() {
            stack_contains_path(
                &normalized_stack,
                &resolved_path[..slash_index.unwrap()],
                true,
            )
        } else {
            stack_contains_path(&normalized_stack, &resolved_path, false)
        };
        if !matched {
            continue;
        }

        let label = if source_info.origin == SourceOrigin::Package && !source_info.source.is_empty()
        {
            source_info.source.clone()
        } else {
            extension.path.clone()
        };
        if seen.insert(label.clone()) {
            matches.push(label);
        }
    }
    matches
}

/// Best-effort persistence for callers that are already crashing
/// (`recordCrash`). `error_message` is the upstream `error.message ||
/// error.name` extraction and `error_stack` the `error.stack` capture (D117).
pub fn record_crash(
    kind: CrashKind,
    error_message: &str,
    error_stack: Option<&str>,
    session_file: Option<&str>,
    cwd: &str,
    path: &str,
) -> Option<CrashRecord> {
    let record = CrashRecord {
        timestamp: to_iso_timestamp(now_ms()),
        version: VERSION.to_string(),
        kind,
        message: error_message.to_string(),
        stack: error_stack.map(str::to_string),
        session_file: session_file.map(str::to_string),
        cwd: cwd.to_string(),
        notified: None,
    };
    let mut records = read_crash_log(path);
    records.push(record.clone());
    let start = records.len().saturating_sub(MAX_CRASH_RECORDS);
    write_crash_log(&records[start..], path).ok()?;
    Some(record)
}

/// Return the newest recent crash, marking pending records as announced
/// (`takeUnnotifiedCrash`).
pub fn take_unnotified_crash(path: &str, now: i64) -> Option<CrashRecord> {
    let records = read_crash_log(path);
    let crash = records
        .iter()
        .rev()
        .find(|record| {
            record.notified != Some(true)
                && parse_timestamp_ms(&record.timestamp)
                    .is_some_and(|timestamp| now - timestamp <= MAX_AGE)
        })?
        .clone();
    // Showing the notice again is harmless if the write fails.
    let _ = write_crash_log(
        &records
            .iter()
            .map(|record| {
                if record.notified == Some(true) {
                    record.clone()
                } else {
                    CrashRecord {
                        notified: Some(true),
                        ..record.clone()
                    }
                }
            })
            .collect::<Vec<_>>(),
        path,
    );
    Some(crash)
}

/// Delete the crash log; the records can be attached again if this fails
/// (`clearCrashLog`).
pub fn clear_crash_log(path: &str) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::source_info::SourceScope;

    fn package_extension(source: &str, base_dir: &str) -> ExtensionStackMetadata {
        let normalized_base_dir = base_dir.replace('\\', "/");
        let resolved_path = format!("{normalized_base_dir}/extensions/index.ts");
        ExtensionStackMetadata {
            path: resolved_path.clone(),
            resolved_path: resolved_path.clone(),
            source_info: SourceInfo {
                path: resolved_path,
                source: source.to_string(),
                scope: SourceScope::User,
                origin: SourceOrigin::Package,
                base_dir: Some(base_dir.to_string()),
            },
        }
    }

    fn top_level_extension(path: &str, base_dir: &str) -> ExtensionStackMetadata {
        ExtensionStackMetadata {
            path: path.to_string(),
            resolved_path: path.to_string(),
            source_info: SourceInfo {
                path: path.to_string(),
                source: "local".to_string(),
                scope: SourceScope::User,
                origin: SourceOrigin::TopLevel,
                base_dir: Some(base_dir.to_string()),
            },
        }
    }

    fn single_file_package_extension(name: &str) -> ExtensionStackMetadata {
        let path = format!("/plugins/{name}.ts");
        ExtensionStackMetadata {
            path: path.clone(),
            resolved_path: path.clone(),
            source_info: SourceInfo {
                path: path.clone(),
                source: path,
                scope: SourceScope::User,
                origin: SourceOrigin::Package,
                base_dir: Some("/plugins".to_string()),
            },
        }
    }

    // Upstream "matches stack frames beneath loaded package roots".
    #[test]
    fn matches_stack_frames_beneath_loaded_package_roots() {
        let memory = package_extension(
            "npm:pi-observational-memory",
            "/home/fedora/.pi/agent/npm/node_modules/pi-observational-memory",
        );
        let unrelated = package_extension(
            "npm:unrelated",
            "/home/fedora/.pi/agent/npm/node_modules/unrelated",
        );
        let stack = concat!(
			"TypeError: Cannot read properties of undefined (reading 'runtime')\n",
			"    at streamSimple (file:///home/fedora/.local/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/chunks/chunk-CMRUVXTE.js:1093:16944)\n",
			"    at /home/fedora/.pi/agent/npm/node_modules/pi-observational-memory/src/agents/worker-stream.ts:43:45",
		);

        assert_eq!(
            find_extension_stack_matches(Some(stack), &[memory, unrelated]),
            vec!["npm:pi-observational-memory".to_string()]
        );
    }

    // Upstream "normalizes Windows paths and deduplicates package extensions".
    #[test]
    fn normalizes_windows_paths_and_deduplicates_package_extensions() {
        let root = "C:\\Users\\reporter\\.pi\\agent\\npm\\node_modules\\@scope\\memory";
        let first = package_extension("npm:@scope/memory", root);
        let second = package_extension("npm:@scope/memory", root);
        let stack = "Error: broken\n    at run (c:\\users\\reporter\\.pi\\agent\\npm\\node_modules\\@scope\\memory\\src\\worker.ts:4:2)";

        assert_eq!(
            find_extension_stack_matches(Some(stack), &[first, second]),
            vec!["npm:@scope/memory".to_string()]
        );
    }

    // Upstream "does not attribute sibling single-file packages".
    #[test]
    fn does_not_attribute_sibling_single_file_packages() {
        let stack = "Error: broken\n    at run (file:///plugins/b.ts:4:2)";
        assert_eq!(
            find_extension_stack_matches(
                Some(stack),
                &[
                    single_file_package_extension("a"),
                    single_file_package_extension("b")
                ]
            ),
            vec!["/plugins/b.ts".to_string()]
        );
    }

    // Upstream "ignores extension paths in the error message".
    #[test]
    fn ignores_extension_paths_in_the_error_message() {
        let extension = package_extension("npm:memory", "/tmp/node_modules/memory");
        let stack = format!(
            "Error: Failed to read {}\n    at run (file:///opt/pi/dist/core.js:4:2)",
            extension.resolved_path
        );
        assert!(find_extension_stack_matches(Some(&stack), &[extension]).is_empty());
    }

    // Upstream "decodes frame paths independently from malformed error text".
    #[test]
    fn decodes_frame_paths_independently_from_malformed_error_text() {
        let path = "/Users/reporter/.pi/agent/extensions/local memory/index.ts";
        let extension = ExtensionStackMetadata {
            path: path.to_string(),
            resolved_path: path.to_string(),
            source_info: SourceInfo {
                path: path.to_string(),
                source: "local".to_string(),
                scope: SourceScope::User,
                origin: SourceOrigin::TopLevel,
                base_dir: Some("/Users/reporter/.pi/agent/extensions".to_string()),
            },
        };
        let stack = "Error: progress 100%\n    at run (file:///Users/reporter/.pi/agent/extensions/local%20memory/worker.ts:4:2)";
        assert_eq!(
            find_extension_stack_matches(Some(stack), &[extension]),
            vec![path.to_string()]
        );
    }

    #[test]
    fn synthetic_and_missing_stacks_match_nothing() {
        let extension = top_level_extension(
            "/agent/extensions/local/index.ts",
            "/agent/extensions/local",
        );
        assert!(find_extension_stack_matches(None, &[extension.clone()]).is_empty());
        assert!(find_extension_stack_matches(Some(""), &[extension]).is_empty());
    }

    // Upstream recordCrash/takeUnnotifiedCrash/clearCrashLog behavior.
    #[test]
    fn crash_log_round_trip_caps_records_and_marks_notified() {
        let dir = fresh_dir("crash-log");
        let path = crash_log_path(&dir);
        assert!(read_crash_log(&path).is_empty());

        for index in 0..7 {
            let record = record_crash(
                CrashKind::FatalError,
                &format!("boom {index}"),
                Some("stack"),
                None,
                "/w",
                &path,
            )
            .unwrap();
            assert_eq!(record.message, format!("boom {index}"));
            assert_eq!(record.version, VERSION);
        }
        let records = read_crash_log(&path);
        assert_eq!(records.len(), MAX_CRASH_RECORDS);
        assert_eq!(records[0].message, "boom 2");
        assert_eq!(records[4].message, "boom 6");

        let now = now_ms();
        let crash = take_unnotified_crash(&path, now).unwrap();
        assert_eq!(crash.message, "boom 6");
        // Second take finds nothing: every record was marked notified.
        assert!(take_unnotified_crash(&path, now).is_none());
        // Stale crashes stay hidden.
        let stale = CrashRecord {
            timestamp: to_iso_timestamp(now - MAX_AGE - 1),
            version: VERSION.to_string(),
            kind: CrashKind::UncaughtException,
            message: "old".to_string(),
            stack: None,
            session_file: None,
            cwd: "/w".to_string(),
            notified: None,
        };
        write_json_records(&[serde_json::to_value(&stale).unwrap()], &path);
        assert!(take_unnotified_crash(&path, now).is_none());

        clear_crash_log(&path);
        assert!(read_crash_log(&path).is_empty());
        assert!(!std::path::Path::new(&path).exists());
        // Clearing a missing file stays quiet.
        clear_crash_log(&path);
    }

    #[test]
    fn read_crash_log_keeps_only_timestamp_and_message_records() {
        let dir = fresh_dir("crash-log-lenient");
        let path = crash_log_path(&dir);
        write_json_records(
            &[
                serde_json::json!({ "timestamp": "2026-01-01T00:00:00.000Z", "message": "kept" }),
                serde_json::json!({ "message": "no timestamp" }),
                serde_json::json!({ "timestamp": 5, "message": "numeric timestamp" }),
                serde_json::Value::from("not an object"),
            ],
            &path,
        );
        let records = read_crash_log(&path);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].message, "kept");
        assert_eq!(records[0].kind, CrashKind::UncaughtException);
        assert_eq!(records[0].notified, None);
        // Malformed JSON and non-array roots read as empty.
        std::fs::write(&path, "{oops").unwrap();
        assert!(read_crash_log(&path).is_empty());
        std::fs::write(&path, "{}").unwrap();
        assert!(read_crash_log(&path).is_empty());
    }

    fn write_json_records(records: &[serde_json::Value], path: &str) {
        std::fs::write(path, serde_json::to_string_pretty(records).unwrap()).unwrap();
    }

    fn fresh_dir(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("crash-log-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_string_lossy().into_owned()
    }
}
