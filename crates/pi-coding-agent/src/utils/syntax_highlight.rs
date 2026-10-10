//! Port of `packages/coding-agent/src/utils/syntax-highlight.ts` — highlight.js
//! driving + the theme/label renderer.
//!
//! Deviations (crate docs, deviation D16): highlight.js sits behind the
//! [`Highlighter`] trait with a process-wide registry. The default
//! implementation is plain-text: it HTML-escapes the code and emits no markup
//! (no languages are supported), so [`render_highlighted_html`] — the
//! algorithmically interesting half, fully ported — still runs unchanged over
//! its output. `loadAllHighlightLanguages` resolves immediately (the eager
//! language set is whatever the registered highlighter supports), and the
//! HTML entity decoding reuses the shared [`super::html`] port of upstream
//! `utils/html.ts`.

use super::html::decode_html_entity_at;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Upstream `HighlightFormatter`.
pub type HighlightFormatter = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Upstream `HighlightTheme` (`Partial<Record<string, HighlightFormatter>>`).
pub type HighlightTheme = HashMap<String, HighlightFormatter>;

/// Upstream `HighlightOptions`.
#[derive(Clone, Default)]
pub struct HighlightOptions {
    pub language: Option<String>,
    pub ignore_illegals: bool,
    pub language_subset: Vec<String>,
    pub theme: HighlightTheme,
}

/// highlight.js seam (`hljs.highlight`/`highlightAuto`/`getLanguage`).
pub trait Highlighter: Send + Sync {
    /// `hljs.highlight(code, { language, ignoreIllegals }).value` — HTML with
    /// `hljs-*` span classes. `None` if the language is unknown.
    fn highlight(&self, code: &str, language: &str, ignore_illegals: bool) -> Option<String>;

    /// `hljs.highlightAuto(code, languageSubset).value`.
    fn highlight_auto(&self, code: &str, language_subset: &[String]) -> String;

    /// `hljs.getLanguage(name) !== undefined`.
    fn supports_language(&self, name: &str) -> bool;
}

/// Default seam implementation: escape HTML, no markup, no languages.
pub struct PlainTextHighlighter;

impl Highlighter for PlainTextHighlighter {
    fn highlight(&self, code: &str, _language: &str, _ignore_illegals: bool) -> Option<String> {
        Some(escape_html(code))
    }

    fn highlight_auto(&self, code: &str, _language_subset: &[String]) -> String {
        escape_html(code)
    }

    fn supports_language(&self, _name: &str) -> bool {
        false
    }
}

fn escape_html(code: &str) -> String {
    let mut out = String::with_capacity(code.len());
    for c in code.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

static HIGHLIGHTER: OnceLock<Arc<dyn Highlighter>> = OnceLock::new();

/// Register the highlight.js-backed implementation (upstream: module import).
pub fn set_highlighter(highlighter: Arc<dyn Highlighter>) {
    let _ = HIGHLIGHTER.set(highlighter);
}

fn highlighter() -> &'static Arc<dyn Highlighter> {
    HIGHLIGHTER.get_or_init(|| Arc::new(PlainTextHighlighter))
}

/// `loadAllHighlightLanguages` — the deferred full-language import. The Rust
/// port has nothing to lazy-load (the registered highlighter is fixed), so the
/// promise resolves immediately; the eager language set remains available on
/// failure upstream, which this matches.
pub async fn load_all_highlight_languages() {}

// HTML entity decoding comes from the shared [`super::html`] port of
// upstream `utils/html.ts`.

// ---- renderer ----

const SPAN_CLOSE: &str = "</span>";
const HIGHLIGHT_CLASS_PREFIX: &str = "hljs-";

fn get_scope_from_span_tag(tag: &str) -> Option<String> {
    // /\sclass\s*=\s*(?:"([^"]*)"|'([^']*)')/
    let class_index = tag.find("class")?;
    let rest = &tag[class_index + "class".len()..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let rest = rest.trim_start();
    let (class_value, _) = if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        (&stripped[..end], end)
    } else if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        (&stripped[..end], end)
    } else {
        return None;
    };

    for class_name in class_value.split_whitespace() {
        if let Some(scope) = class_name.strip_prefix(HIGHLIGHT_CLASS_PREFIX) {
            return Some(scope.to_string());
        }
    }

    None
}

fn get_scope_formatter(scope: &str, theme: &HighlightTheme) -> Option<HighlightFormatter> {
    if let Some(exact) = theme.get(scope) {
        return Some(exact.clone());
    }

    if let Some(dot_index) = scope.find('.') {
        if let Some(prefix_formatter) = theme.get(&scope[..dot_index]) {
            return Some(prefix_formatter.clone());
        }
    }

    if let Some(dash_index) = scope.find('-') {
        if let Some(prefix_formatter) = theme.get(&scope[..dash_index]) {
            return Some(prefix_formatter.clone());
        }
    }

    None
}

fn get_active_formatter(
    scopes: &[Option<String>],
    theme: &HighlightTheme,
) -> Option<HighlightFormatter> {
    for scope in scopes.iter().rev() {
        let Some(scope) = scope else { continue };
        if let Some(formatter) = get_scope_formatter(scope, theme) {
            return Some(formatter);
        }
    }
    theme.get("default").cloned()
}

fn is_span_open_tag_start(html: &str, index: usize) -> bool {
    if !html[index..].starts_with("<span") {
        return false;
    }
    let next_char = html.as_bytes().get(index + "<span".len()).copied();
    matches!(
        next_char,
        Some(b'>') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
    )
}

/// `renderHighlightedHtml` — walk highlight.js output, apply theme formatters
/// per contiguous text run (per line, so empty lines stay unstyled).
pub fn render_highlighted_html(html: &str, theme: &HighlightTheme) -> String {
    let mut output = String::new();
    let mut text_buffer = String::new();
    let mut scopes: Vec<Option<String>> = Vec::new();

    fn flush_text(
        output: &mut String,
        text_buffer: &mut String,
        scopes: &[Option<String>],
        theme: &HighlightTheme,
    ) {
        if text_buffer.is_empty() {
            return;
        }
        match get_active_formatter(scopes, theme) {
            Some(formatter) => {
                let styled = text_buffer
                    .split('\n')
                    .map(|line| {
                        if line.is_empty() {
                            line.to_string()
                        } else {
                            formatter(line)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                output.push_str(&styled);
            }
            None => output.push_str(text_buffer),
        }
        text_buffer.clear();
    }

    let bytes = html.as_bytes();
    let mut index = 0;
    while index < html.len() {
        if is_span_open_tag_start(html, index) {
            let tag_end_index = html[index + 5..].find('>').map(|i| index + 5 + i);
            if let Some(tag_end_index) = tag_end_index {
                flush_text(&mut output, &mut text_buffer, &scopes, theme);
                let tag = &html[index..tag_end_index + 1];
                let scope = get_scope_from_span_tag(tag);
                scopes.push(scope);
                index = tag_end_index + 1;
                continue;
            }
        }

        if html[index..].starts_with(SPAN_CLOSE) {
            flush_text(&mut output, &mut text_buffer, &scopes, theme);
            if !scopes.is_empty() {
                scopes.pop();
            }
            index += SPAN_CLOSE.len();
            continue;
        }

        if bytes[index] == b'&' {
            if let Some(decoded) = decode_html_entity_at(html, index) {
                text_buffer.push_str(&decoded.text);
                index += decoded.length;
                continue;
            }
        }

        let ch = html[index..].chars().next().unwrap();
        text_buffer.push(ch);
        index += ch.len_utf8();
    }

    flush_text(&mut output, &mut text_buffer, &scopes, theme);
    output
}

/// `highlight` — run highlight.js, then the theme renderer.
pub fn highlight(code: &str, options: &HighlightOptions) -> String {
    let html = match &options.language {
        Some(language) => highlighter()
            .highlight(code, language, options.ignore_illegals)
            .unwrap_or_else(|| escape_html(code)),
        None => highlighter().highlight_auto(code, &options.language_subset),
    };
    render_highlighted_html(&html, &options.theme)
}

/// `supportsLanguage`.
pub fn supports_language(name: &str) -> bool {
    highlighter().supports_language(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme_with(scope: &str, wrap: fn(&str) -> String) -> HighlightTheme {
        let mut theme = HighlightTheme::new();
        theme.insert(
            scope.to_string(),
            Arc::new(move |text: &str| wrap(text)) as HighlightFormatter,
        );
        theme
    }

    fn keyword_wrapper(text: &str) -> String {
        format!("[keyword:{text}]")
    }

    fn string_wrapper(text: &str) -> String {
        format!("[string:{text}]")
    }

    fn number_wrapper(text: &str) -> String {
        format!("[number:{text}]")
    }

    #[test]
    fn renders_highlighted_spans_with_the_provided_theme() {
        let rendered = render_highlighted_html(
            "<span class=\"hljs-keyword\">const</span> value",
            &theme_with("keyword", keyword_wrapper),
        );
        assert_eq!(rendered, "[keyword:const] value");
    }

    #[test]
    fn decodes_html_entities_emitted_by_highlight_js() {
        let rendered = render_highlighted_html(
            "&lt;tag attr=&quot;value&quot;&gt;&amp;#x41;&#65;&lt;/tag&gt;",
            &HighlightTheme::new(),
        );
        assert_eq!(rendered, "<tag attr=\"value\">&#x41;A</tag>");
    }

    #[test]
    fn inherits_parent_formatting_for_unmapped_nested_scopes() {
        let interpolation = "${x}";
        let html = format!("<span class=\"hljs-string\">a<span class=\"hljs-subst\">{interpolation}</span>b</span>");
        let rendered = render_highlighted_html(&html, &theme_with("string", string_wrapper));
        assert_eq!(
            rendered,
            format!("[string:a][string:{interpolation}][string:b]")
        );
    }

    #[test]
    fn keeps_parent_formatting_across_unscoped_nested_spans() {
        let rendered = render_highlighted_html(
            "<span class=\"hljs-string\">a<span class=\"language-xml\">b</span>c</span>",
            &theme_with("string", string_wrapper),
        );
        assert_eq!(rendered, "[string:a][string:b][string:c]");
    }

    #[test]
    fn scopes_resolve_through_dot_and_dash_prefixes() {
        let rendered = render_highlighted_html(
            "<span class=\"hljs-meta.string\">x</span><span class=\"hljs-symbol-string\">y</span>",
            &theme_with("meta", keyword_wrapper),
        );
        assert_eq!(rendered, "[keyword:x]y");
    }

    #[test]
    fn default_formatter_applies_to_unmapped_scopes() {
        let mut theme = theme_with("unused", keyword_wrapper);
        theme.insert(
            "default".to_string(),
            Arc::new(|text: &str| format!("[d:{text}]")) as HighlightFormatter,
        );
        let rendered = render_highlighted_html("plain", &theme);
        assert_eq!(rendered, "[d:plain]");
    }

    #[test]
    fn empty_lines_stay_unstyled() {
        let rendered = render_highlighted_html(
            "<span class=\"hljs-string\">a\n\nb</span>",
            &theme_with("string", string_wrapper),
        );
        assert_eq!(rendered, "[string:a]\n\n[string:b]");
    }

    #[test]
    fn plaintext_highlighter_adds_no_markup() {
        assert!(!supports_language("typescript"));
        // The plain-text seam emits no span markup; the renderer decodes the
        // escaped source back to its original characters, matching upstream's
        // highlight()-through-render pipeline.
        let rendered = highlight("const <value> = 1\n", &HighlightOptions::default());
        assert_eq!(rendered, "const <value> = 1\n");
    }

    #[test]
    fn unknown_explicit_language_falls_back_to_plain_text() {
        let options = HighlightOptions {
            language: Some("cobol".to_string()),
            ignore_illegals: true,
            language_subset: vec![],
            theme: HighlightTheme::new(),
        };
        assert_eq!(highlight("a<b", &options), "a<b");
    }

    #[tokio::test]
    async fn load_all_highlight_languages_resolves() {
        load_all_highlight_languages().await;
    }

    #[test]
    fn entity_decoder_matches_upstream() {
        assert_eq!(
            super::super::html::decode_html_entity("amp"),
            Some("&".to_string())
        );
        assert_eq!(
            super::super::html::decode_html_entity("#65"),
            Some("A".to_string())
        );
        assert_eq!(
            super::super::html::decode_html_entity("#x41"),
            Some("A".to_string())
        );
        assert_eq!(
            super::super::html::decode_html_entity("#X41"),
            Some("A".to_string())
        );
        assert_eq!(super::super::html::decode_html_entity("nosuch"), None);
        // #110000 is the valid code point U+1ADB0 (`String.fromCodePoint`
        // accepts it); lone surrogates throw upstream and read as `None` here.
        assert_eq!(
            super::super::html::decode_html_entity("#110000"),
            Some("\u{1adb0}".to_string())
        );
        assert_eq!(super::super::html::decode_html_entity("#xD800"), None);
        assert_eq!(
            decode_html_entity_at("x&#65;y", 1).map(|d| (d.text, d.length)),
            Some(("A".to_string(), 5))
        );
    }
}

// Skipped (highlight.js-dependent upstream cases): `loads the twenty most
// common languages at startup and defers the rest` and `highlights code
// through highlight.js` (real highlight.js language grammars; the seam default
// supports none). The theme-level tests in upstream's `theme syntax
// highlighting` describe live in upstream `modes/interactive/theme` and are
// out of this slice.
