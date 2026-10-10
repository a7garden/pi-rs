//! Port of upstream export-html tests:
//! `test/export-html-skill-block.test.ts`, `test/export-html-whitespace.test.ts`,
//! `test/export-html-xss.test.ts`.

use super::tool_renderer::trim_rendered_result_lines;
use super::*;

// ===========================================================================
// test/export-html-skill-block.test.ts
// ===========================================================================

/// Upstream `expect(source).toMatch(/…/)` — substring/regex checks against the
/// embedded template.js become contains-checks on the verbatim asset (D80).
fn template_js_contains(needle: &str) -> bool {
    TEMPLATE_JS.contains(needle)
}

#[test]
fn strips_skill_wrapper_xml_from_user_message_rendering() {
    // Skill commands store a structural wrapper in the raw user message:
    //   <skill name="..." location="...">\n...\n</skill>\n\nactual prompt
    // The export renderer must detect that wrapper and render only the
    // user-visible prompt, not the Pi-generated <skill>...</skill> XML tags.
    assert!(template_js_contains("parseSkillBlock"));
    assert!(template_js_contains("skillBlock.userMessage"));
}

#[test]
fn renders_skill_invocation_and_user_message_as_separate_sibling_blocks() {
    // The skill block and user message should render as separate entry-level
    // elements, matching the TUI layout where SkillInvocationMessageComponent
    // and UserMessageComponent are siblings, not nested.
    assert!(template_js_contains("skill-invocation"));

    // When a skill block has a userMessage, the user-message div must be
    // emitted as a separate block after the skill-invocation div, containing
    // the user-authored text. Verify the code checks hasUserContent so the
    // user-message div is only omitted when the skill block has no user
    // prompt and no images.
    assert!(template_js_contains("hasUserContent"));
}

#[test]
fn renders_skill_content_as_markdown_not_raw_text() {
    // The skill block body is markdown (from the SKILL.md file).
    // It should be rendered through safeMarkedParse, not escaped as raw text.
    assert!(template_js_contains("safeMarkedParse(skillBlock.content)"));
}

#[test]
fn shows_skill_name_and_user_message_in_the_sidebar_tree() {
    // The sidebar tree should display both the skill name and the user prompt,
    // not just one or the other.
    assert!(template_js_contains("tree-role-skill"));
}

// ===========================================================================
// test/export-html-whitespace.test.ts
// ===========================================================================

/// Find `needle` allowing its inner whitespace runs to match any whitespace
/// span in `haystack` (ports the `\s*` parts of the upstream regexes).
fn matches_with_flexible_whitespace(haystack: &str, needle: &str) -> bool {
    let tokens: Vec<&str> = needle.split_whitespace().collect();
    let mut rest = haystack;
    for (i, token) in tokens.iter().enumerate() {
        let Some(pos) = rest.find(token) else {
            return false;
        };
        rest = &rest[pos + token.len()..];
        let last = i + 1 == tokens.len();
        if !last && rest.is_empty() {
            return false;
        }
    }
    true
}

/// True if `openers` appears, then later `middle`, then `closer` (ports
/// `openers[\s\S]*?middle[\s\S]*?closer`).
fn ordered_all_present(haystack: &str, parts: &[&str]) -> bool {
    let mut rest = haystack;
    for part in parts {
        let Some(pos) = rest.find(part) else {
            return false;
        };
        rest = &rest[pos + part.len()..];
    }
    true
}

#[test]
fn preserves_whitespace_for_plain_text_tool_output_lines_without_preserving_template_whitespace() {
    // `.output-preview > div:not(.expand-hint), .output-full > div:not(.expand-hint) { … white-space: pre-wrap;`
    let openers = [
        ".output-preview > div:not(.expand-hint),",
        ".output-full > div:not(.expand-hint) {",
    ];
    let found = (0..TEMPLATE_CSS.len())
        .filter(|&start| TEMPLATE_CSS[start..].starts_with(openers[0]))
        .any(|start| {
            let tail = &TEMPLATE_CSS[start + openers[0].len()..];
            let Some(brace) = tail.find(openers[1]) else {
                return false;
            };
            // Skip whitespace between the selectors and `{` (upstream `\s*`).
            let between = &tail[..brace];
            let rule_start = brace + openers[1].len();
            let Some(rule_end) = tail[rule_start..].find('}') else {
                return false;
            };
            between.trim().is_empty()
                && tail[rule_start..rule_start + rule_end].contains("white-space:")
                && {
                    let prop = tail[rule_start..rule_start + rule_end]
                        .split("white-space:")
                        .nth(1)
                        .unwrap_or_default();
                    prop.trim_start().starts_with("pre-wrap")
                }
        });
    assert!(found);
    assert!(matches_with_flexible_whitespace(
        TEMPLATE_CSS,
        ".ansi-line { white-space: pre;"
    ));
    // Negated upstream regex: `.output-preview, .output-full { … white-space: pre-wrap;` absent
    assert!(!ordered_all_present(
        TEMPLATE_CSS,
        &[".output-preview, .output-full {", "white-space: pre-wrap;"]
    ));
}

#[test]
fn does_not_insert_source_whitespace_between_ansi_rendered_lines() {
    assert_eq!(
        ansi_lines_to_html(&["one".to_string(), "two".to_string()]),
        "<div class=\"ansi-line\">one</div><div class=\"ansi-line\">two</div>"
    );
}

#[test]
fn trims_tui_spacing_lines_from_custom_tool_result_html() {
    let rendered = ["\u{1b}[31mone\u{1b}[0m".to_string(), "two".to_string()];
    // Upstream feeds `["", "\x1b[31mone\x1b[0m", "two", ""]` through the
    // component renderer; the trim stage is the part under test (D81: the
    // component pipeline is deferred, so the blank-line trimming and the ANSI
    // conversion are exercised directly on the rendered lines).
    let lines: Vec<String> = [
        "".to_string(),
        rendered[0].clone(),
        rendered[1].clone(),
        String::new(),
    ]
    .to_vec();
    let expanded = ansi_lines_to_html(&trim_rendered_result_lines(&lines));
    assert_eq!(
		expanded,
		"<div class=\"ansi-line\"><span style=\"color:#800000\">one</span></div><div class=\"ansi-line\">two</div>"
	);
}

// ===========================================================================
// test/export-html-xss.test.ts
// ===========================================================================

#[test]
fn overrides_the_marked_link_renderer_to_use_scheme_allow_list_sanitization() {
    assert!(matches_with_flexible_whitespace(
        TEMPLATE_JS,
        "link ( token )"
    ));
    assert!(TEMPLATE_JS.contains("sanitizeMarkdownUrl(token.href)"));
    assert!(TEMPLATE_JS.contains("^(https?|mailto|tel|ftp)"));
}

#[test]
fn overrides_the_marked_image_renderer_to_use_scheme_allow_list_sanitization() {
    assert!(matches_with_flexible_whitespace(
        TEMPLATE_JS,
        "image ( token )"
    ));
    assert!(TEMPLATE_JS.contains("sanitizeMarkdownUrl(token.href)"));
}

#[test]
fn strips_c0_controls_before_checking_and_emitting_markdown_urls() {
    assert!(TEMPLATE_JS.contains("replace(/[\\x00-\\x1f\\x7f]/g, '')"));
    // Negated upstream regex: /^\\s*\(javascript|vbscript|data\):/i absent
    let lower = TEMPLATE_JS.to_lowercase();
    assert!(!lower.contains("\\s*(javascript|vbscript|data):"));
}

#[test]
fn escapes_href_attributes_in_the_custom_link_renderer() {
    // The link renderer must escape href values to prevent attribute breakout
    assert!(TEMPLATE_JS.contains("escapeHtml(href)"));
}

#[test]
fn escapes_image_mimetype_attributes() {
    // Image mimeType must be escaped to prevent attribute breakout
    assert!(!TEMPLATE_JS.contains("${img.mimeType}"));
    assert!(TEMPLATE_JS.contains("escapeHtml(img.mimeType"));
}

#[test]
fn escapes_image_data_attributes() {
    // Image data is embedded in src attributes and must not allow attribute breakout.
    assert!(!TEMPLATE_JS.contains(";base64,${img.data}\""));
    assert!(
        TEMPLATE_JS.contains(";base64,${escapeHtml(img.data || '')}\"")
            || TEMPLATE_JS.contains(";base64,${escapeHtml(img.data || \"\")}\"")
    );
}

#[test]
fn escapes_entry_ids_before_inserting_them_into_attributes() {
    // Session entry IDs are embedded in id and data-entry-id attributes.
    assert!(!TEMPLATE_JS.contains("id=\"${entryId}\""));
    assert!(!TEMPLATE_JS.contains("data-entry-id=\"${entryId}\""));
    assert!(TEMPLATE_JS.contains("entry-${escapeHtml(entry.id)}"));
    assert!(TEMPLATE_JS.contains("data-entry-id=\"${escapeHtml(entryId)}\""));
}

#[test]
fn escapes_tree_metadata_rendered_from_session_fields() {
    // The tree renders session metadata via innerHTML, so dynamic fields must be escaped.
    assert!(!TEMPLATE_JS.contains("[${msg.toolName || 'tool'}]"));
    assert!(!TEMPLATE_JS.contains("[${msg.role}]"));
    assert!(!TEMPLATE_JS.contains("[model: ${entry.modelId}]"));
    assert!(!TEMPLATE_JS.contains("[thinking: ${entry.thinkingLevel}]"));
    assert!(!TEMPLATE_JS.contains("[${entry.type}]"));
    assert!(TEMPLATE_JS.contains("${escapeHtml(msg.toolName || 'tool')}"));
    assert!(TEMPLATE_JS.contains("${escapeHtml(msg.role)}"));
    assert!(TEMPLATE_JS.contains("${escapeHtml(entry.modelId)}"));
    assert!(TEMPLATE_JS.contains("${escapeHtml(entry.thinkingLevel)}"));
    assert!(TEMPLATE_JS.contains("${escapeHtml(entry.type)}"));
}

#[test]
fn escapes_model_names_in_the_exported_header() {
    // Assistant message provider/model values are collected from the session
    // and rendered with innerHTML.
    assert!(!TEMPLATE_JS.contains("${globalStats.models.join(', ') || 'unknown'}"));
    assert!(TEMPLATE_JS.contains("${escapeHtml(globalStats.models.join(', ') || 'unknown')}"));
}

// ===========================================================================
// Supplemental port tests (no upstream ansi-to-html.test.ts exists; these pin
// the SGR state machine against hand-verified upstream outputs)
// ===========================================================================

#[test]
fn ansi_to_html_matches_upstream_reference_outputs() {
    assert_eq!(
        ansi_to_html("\u{1b}[31mred\u{1b}[0m plain"),
        "<span style=\"color:#800000\">red</span> plain"
    );
    // 256-color: 38;5;196
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;196mX\u{1b}[0m"),
        "<span style=\"color:#ff0000\">X</span>"
    );
    // Grayscale 256-color: 38;5;240 → 8 + 8*10 = 88 = 0x58
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;240mX"),
        "<span style=\"color:#585858\">X</span>"
    );
    // RGB true color
    assert_eq!(
        ansi_to_html("\u{1b}[38;2;1;2;3mX"),
        "<span style=\"color:rgb(1,2,3)\">X</span>"
    );
    // Bold + underline stack
    assert_eq!(
        ansi_to_html("\u{1b}[1;4mX"),
        "<span style=\"font-weight:bold;text-decoration:underline\">X</span>"
    );
    // Reset clears
    assert_eq!(
        ansi_to_html("\u{1b}[1mX\u{1b}[0mY"),
        "<span style=\"font-weight:bold\">X</span>Y"
    );
    // HTML is escaped
    assert_eq!(ansi_to_html("<b>&</b>"), "&lt;b&gt;&amp;&lt;/b&gt;");
}

#[test]
fn ansi_lines_to_html_renders_empty_lines_as_nbsp() {
    assert_eq!(
        ansi_lines_to_html(&["".to_string(), "x".to_string()]),
        "<div class=\"ansi-line\">&nbsp;</div><div class=\"ansi-line\">x</div>"
    );
}

#[test]
fn color_derivation_matches_upstream_reference_values() {
    // Dark base (upstream example values from userMessageBg #343541)
    let derived = derive_export_colors("#343541");
    assert_eq!(
        derived,
        DerivedExportColors {
            page_bg: adjust_brightness("#343541", 0.7),
            card_bg: adjust_brightness("#343541", 0.85),
            info_bg: "rgb(72, 68, 65)".to_string(),
        }
    );
    // Unparseable → fixed fallbacks
    assert_eq!(
        derive_export_colors("terminal"),
        DerivedExportColors {
            page_bg: "rgb(24, 24, 30)".to_string(),
            card_bg: "rgb(30, 30, 36)".to_string(),
            info_bg: "rgb(60, 55, 40)".to_string(),
        }
    );
}

#[test]
fn generate_html_substitutes_all_template_slots() {
    let data = SessionData {
        header: None,
        entries: vec![],
        leaf_id: None,
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };
    let html = generate_html(&data, Some("dark")).expect("generate");
    assert!(!html.contains("{{CSS}}"));
    assert!(!html.contains("{{JS}}"));
    assert!(!html.contains("{{SESSION_DATA}}"));
    assert!(!html.contains("{{MARKED_JS}}"));
    assert!(!html.contains("{{HIGHLIGHT_JS}}"));
    let expected_payload = base64_encode(&serde_json::to_vec(&data).unwrap());
    assert!(html.contains(&format!(
        "<script id=\"session-data\" type=\"application/json\">{expected_payload}"
    )));
    // Theme vars injected into CSS
    assert!(html.contains("--exportPageBg:"));
}

#[test]
fn base64_encoder_matches_known_vectors() {
    assert_eq!(base64_encode(b""), "");
    assert_eq!(base64_encode(b"f"), "Zg==");
    assert_eq!(base64_encode(b"fo"), "Zm8=");
    assert_eq!(base64_encode(b"foo"), "Zm9v");
    assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
}
