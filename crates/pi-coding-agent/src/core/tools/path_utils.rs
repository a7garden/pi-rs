//! Port of `packages/coding-agent/src/core/tools/path-utils.ts` — path
//! expansion/variant-fallback helpers for the file tools.
//!
//! `normalize("NFD")` uses the `unicode-normalization` crate; the macOS
//! screenshot-path regex becomes a hand-rolled case-insensitive scan.

use unicode_normalization::UnicodeNormalization;

use crate::utils::paths::{normalize_path, resolve_path, PathInputOptions};

const NARROW_NO_BREAK_SPACE: char = '\u{202F}';

/// Options the tool path helpers pass to `normalizePath`/`resolvePath`
/// upstream: `{ normalizeUnicodeSpaces: true, stripAtPrefix: true }`.
fn tool_path_options() -> PathInputOptions {
    PathInputOptions {
        normalize_unicode_spaces: true,
        strip_at_prefix: true,
        ..PathInputOptions::default()
    }
}

/// Replace ` AM.`/` PM.` (any case) so the space becomes a narrow no-break
/// space, matching macOS screenshot names
/// (`path.replace(/ (AM|PM)\./gi, "\u202F$1.")`).
fn try_macos_screenshot_path(file_path: &str) -> String {
    let bytes = file_path.as_bytes();
    let mut result = String::with_capacity(file_path.len());
    let mut index = 0;
    while index < bytes.len() {
        let rest = &file_path[index..];
        let is_match = rest.len() >= 5
            && rest.starts_with(' ')
            && matches!(rest.as_bytes()[1], b'A' | b'a' | b'P' | b'p')
            && matches!(rest.as_bytes()[2], b'M' | b'm')
            && rest.as_bytes()[3] == b'.';
        if is_match {
            result.push(NARROW_NO_BREAK_SPACE);
            result.push_str(&rest[1..4]);
            index += 4;
        } else {
            let ch = rest.chars().next().unwrap();
            result.push(ch);
            index += ch.len_utf8();
        }
    }
    result
}

/// macOS stores filenames in NFD (decomposed) form; try converting user input
/// to NFD (`tryNFDVariant`).
fn try_nfd_variant(file_path: &str) -> String {
    file_path.nfd().collect()
}

/// macOS uses U+2019 (right single quotation mark) in screenshot names like
/// "Capture d'écran"; users typically type U+0027 (`tryCurlyQuoteVariant`).
fn try_curly_quote_variant(file_path: &str) -> String {
    file_path.replace('\'', "\u{2019}")
}

fn file_exists(file_path: &str) -> bool {
    std::fs::metadata(file_path).is_ok()
}

/// `pathExists`.
pub async fn path_exists(file_path: &str) -> bool {
    tokio::fs::metadata(file_path).await.is_ok()
}

/// `expandPath`: normalize with unicode-space folding and `@`-prefix
/// stripping.
pub fn expand_path(file_path: &str) -> String {
    normalize_path(file_path, &tool_path_options())
}

/// Resolve a path relative to the given cwd; handles `~` expansion and
/// absolute paths (`resolveToCwd`).
pub fn resolve_to_cwd(file_path: &str, cwd: &str) -> String {
    let normalized = normalize_path(file_path, &tool_path_options());
    resolve_path(&normalized, cwd)
}

/// Resolve a read path, falling back to macOS filename variants
/// (`resolveReadPath`).
pub fn resolve_read_path(file_path: &str, cwd: &str) -> String {
    let resolved = resolve_to_cwd(file_path, cwd);

    if file_exists(&resolved) {
        return resolved;
    }

    // Try macOS AM/PM variant (narrow no-break space before AM/PM).
    let am_pm_variant = try_macos_screenshot_path(&resolved);
    if am_pm_variant != resolved && file_exists(&am_pm_variant) {
        return am_pm_variant;
    }

    // Try NFD variant (macOS stores filenames in NFD form).
    let nfd_variant = try_nfd_variant(&resolved);
    if nfd_variant != resolved && file_exists(&nfd_variant) {
        return nfd_variant;
    }

    // Try curly quote variant (macOS uses U+2019 in screenshot names).
    let curly_variant = try_curly_quote_variant(&resolved);
    if curly_variant != resolved && file_exists(&curly_variant) {
        return curly_variant;
    }

    // Try combined NFD + curly quote (for French macOS screenshots like
    // "Capture d'écran").
    let nfd_curly_variant = try_curly_quote_variant(&nfd_variant);
    if nfd_curly_variant != resolved && file_exists(&nfd_curly_variant) {
        return nfd_curly_variant;
    }

    resolved
}

/// Async variant of [`resolve_read_path`] (`resolveReadPathAsync`).
pub async fn resolve_read_path_async(file_path: &str, cwd: &str) -> String {
    let resolved = resolve_to_cwd(file_path, cwd);

    if path_exists(&resolved).await {
        return resolved;
    }

    // Try macOS AM/PM variant (narrow no-break space before AM/PM).
    let am_pm_variant = try_macos_screenshot_path(&resolved);
    if am_pm_variant != resolved && path_exists(&am_pm_variant).await {
        return am_pm_variant;
    }

    // Try NFD variant (macOS stores filenames in NFD form).
    let nfd_variant = try_nfd_variant(&resolved);
    if nfd_variant != resolved && path_exists(&nfd_variant).await {
        return nfd_variant;
    }

    // Try curly quote variant (macOS uses U+2019 in screenshot names).
    let curly_variant = try_curly_quote_variant(&resolved);
    if curly_variant != resolved && path_exists(&curly_variant).await {
        return curly_variant;
    }

    // Try combined NFD + curly quote (for French macOS screenshots like
    // "Capture d'écran").
    let nfd_curly_variant = try_curly_quote_variant(&nfd_variant);
    if nfd_curly_variant != resolved && path_exists(&nfd_curly_variant).await {
        return nfd_curly_variant;
    }

    resolved
}
