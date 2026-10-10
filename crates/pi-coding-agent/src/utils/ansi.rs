//! Port of `packages/coding-agent/src/utils/ansi.ts` — `stripAnsi`, derived
//! from chalk `ansi-regex` / `strip-ansi` (MIT, © Sindre Sorhus). The upstream
//! global regex is ported as a hand-rolled scanner with the same semantics:
//! leftmost match wins, the OSC alternative is tried before the CSI
//! alternative at each position, and CSI params backtrack greedily.
//!
//! Deviation D8 — the upstream `TypeError` guard for non-string input is not
//! representable in Rust: `stripAnsi` only accepts `&str`.

/// Valid string terminator sequences are BEL, ESC\, and 0x9c.
const ST: [char; 3] = ['\u{0007}', '\\', '\u{009C}'];

fn is_final_byte(c: char) -> bool {
    matches!(c,
		'0'..='9'
			| 'A'..='P' | 'R'..='T' | 'Z'
			| 'c' | 'f'..='n' | 'q'..='u' | 'y'
			| '=' | '>' | '<' | '~')
}

fn is_intermediate(c: char) -> bool {
    matches!(c, '[' | ']' | '(' | ')' | '#' | ';' | '?')
}

/// Length of the ANSI escape sequence starting at the beginning of `chars`.
fn match_ansi(chars: &[char]) -> Option<usize> {
    // Alternation order matters: OSC wins when it matches at this position.
    if let Some(len) = match_osc(chars) {
        return Some(len);
    }
    match_csi(chars)
}

/// OSC alternative: `ESC ] … ST` (non-greedy until the first ST).
fn match_osc(chars: &[char]) -> Option<usize> {
    if chars.len() < 2 || chars[0] != '\u{001B}' || chars[1] != ']' {
        return None;
    }
    let mut index = 2;
    while index < chars.len() {
        let c = chars[index];
        if c == '\u{0007}' || c == '\u{009C}' {
            return Some(index + 1);
        }
        if c == '\\' && ST.contains(&c) && chars[index - 1] == '\u{001B}' {
            return Some(index + 1);
        }
        index += 1;
    }
    None
}

/// CSI alternative: `ESC|0x9B`, optional intermediates, optional params
/// (`\d{1,4}([;:]\d{0,4})*`, greedy with backtracking), then a final byte.
fn match_csi(chars: &[char]) -> Option<usize> {
    if chars.is_empty() || (chars[0] != '\u{001B}' && chars[0] != '\u{009B}') {
        return None;
    }
    let mut index = 1;
    while chars.get(index).is_some_and(|&c| is_intermediate(c)) {
        index += 1;
    }
    // Greedy param run: `\d{1,4}` then as many `[;:]\d{0,4}` repeats as fit.
    let params_start = index;
    let mut params_end = params_start;
    while chars.get(params_end).is_some_and(|&c| c.is_ascii_digit())
        && params_end - params_start < 4
    {
        params_end += 1;
    }
    // Backtrack the leading `\d{1,4}` from greedy down to absent (the JS
    // regex does the same; digits are always valid final bytes, so only the
    // leading run needs backtracking).
    let mut k = params_end - params_start;
    loop {
        let mut index = params_start + k;
        // Consume `[;:]\d{0,4}` repeats greedily — only when the leading
        // `\d{1,4}` group is present (k > 0); a repeat is always valid, so
        // greedy consumption matches the regex here.
        if k > 0 {
            while matches!(chars.get(index), Some(';' | ':')) {
                index += 1;
                let mut digits = 0;
                while chars.get(index).is_some_and(|&c| c.is_ascii_digit()) && digits < 4 {
                    index += 1;
                    digits += 1;
                }
            }
        }
        if chars.get(index).is_some_and(|&c| is_final_byte(c)) {
            return Some(index + 1);
        }
        if k == 0 {
            return None;
        }
        k -= 1;
    }
}

/// Remove ANSI escape sequences from `value`.
pub fn strip_ansi(value: &str) -> String {
    // Fast path: ANSI codes require ESC (7-bit) or CSI (8-bit) introducer.
    if !value.contains('\u{001B}') && !value.contains('\u{009B}') {
        return value.to_string();
    }
    let chars: Vec<char> = value.chars().collect();
    let mut out = String::with_capacity(value.len());
    let mut index = 0;
    while index < chars.len() {
        if let Some(len) = match_ansi(&chars[index..]) {
            index += len;
            continue;
        }
        out.push(chars[index]);
        index += 1;
    }
    out
}

/// Longest unfinished sequence held back while streaming. Longer ones are
/// processed as-is (`MAX_PENDING_ANSI_LENGTH`).
const MAX_PENDING_ANSI_LENGTH: usize = 256;

/// Whether the char slice starting at `start` is an unfinished OSC sequence to
/// the end: `ESC ]`, then only characters that cannot terminate it — anything
/// but BEL, 0x9C, and ESC, plus an ESC not followed by `\` (a trailing ESC may
/// still start an `ESC\` terminator).
fn unfinished_osc_at(chars: &[char], start: usize) -> bool {
    if chars.len() < start + 2 || chars[start] != '\u{001B}' || chars[start + 1] != ']' {
        return false;
    }
    let mut index = start + 2;
    while index < chars.len() {
        let c = chars[index];
        if c == '\u{0007}' || c == '\u{009C}' {
            return false;
        }
        if c == '\u{001B}' {
            // `ESC \` would complete the sequence; any other ESC continues it.
            return chars.get(index + 1) != Some(&'\\');
        }
        index += 1;
    }
    true
}

/// Whether the char slice starting at `start` is an unfinished CSI sequence to
/// the end: introducer, intermediates, optional params (`\d{1,4}([;:]\d{0,4})*`
/// with backtracking), and no final byte before the end.
fn unfinished_csi_at(chars: &[char], start: usize) -> bool {
    let c = chars[start];
    if c != '\u{001B}' && c != '\u{009B}' {
        return false;
    }
    let mut index = start + 1;
    while chars.get(index).is_some_and(|&c| is_intermediate(c)) {
        index += 1;
    }
    // The optional params group must consume every remaining character.
    params_consume_all(chars, index)
}

/// Whether `(?:\d{1,4}(?:[;:]\d{0,4})*)?` can consume `chars[start..]` to the
/// end (greedy with backtracking).
fn params_consume_all(chars: &[char], start: usize) -> bool {
    if start == chars.len() {
        // The group matches empty.
        return true;
    }
    fn digit_run(chars: &[char], pos: usize, max: usize) -> usize {
        let mut taken = 0;
        while taken < max && chars.get(pos + taken).is_some_and(|c| c.is_ascii_digit()) {
            taken += 1;
        }
        taken
    }
    // `([;:]\d{0,4})*` from `pos`: either the end, or another repeat.
    fn repeats(chars: &[char], pos: usize) -> bool {
        if pos == chars.len() {
            return true;
        }
        if matches!(chars[pos], ';' | ':') {
            let digits = digit_run(chars, pos + 1, 4);
            for take in (0..=digits).rev() {
                if repeats(chars, pos + 1 + take) {
                    return true;
                }
            }
        }
        false
    }
    // `\d{1,4}` needs at least one digit; try the greedy lengths first.
    let leading = digit_run(chars, start, 4);
    for take in (1..=leading).rev() {
        if repeats(chars, start + take) {
            return true;
        }
    }
    false
}

/// Split streamed text into a part that is safe to pass to
/// [`strip_ansi`] now and a trailing unfinished escape sequence that should
/// be prepended to the next chunk (`splitIncompleteAnsiSuffix`).
pub fn split_incomplete_ansi_suffix(value: &str) -> (String, String) {
    if !value.contains('\u{001B}') && !value.contains('\u{009B}') {
        return (value.to_string(), String::new());
    }
    let chars: Vec<char> = value.chars().collect();
    let window_start = chars.len().saturating_sub(MAX_PENDING_ANSI_LENGTH);
    let mut split_at = None;
    for start in window_start..chars.len() {
        if unfinished_osc_at(&chars, start) || unfinished_csi_at(&chars, start) {
            split_at = Some(start);
            break;
        }
    }
    let Some(split_at) = split_at else {
        return (value.to_string(), String::new());
    };
    let complete: String = chars[..split_at].iter().collect();
    let pending: String = chars[split_at..].iter().collect();
    (complete, pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected outputs generated by running the upstream reference (the
    /// chalk `ansi-regex`-based `stripAnsi`) under Node over the same
    /// generated compatibility inputs as `test/ansi-utils.test.ts`.
    const COMPATIBILITY_PAIRS: &[(&str, &str)] = &[
        ("plain", "plain"),
        ("a\u{1b}[31mred\u{1b}[0mz", "aredz"),
        (
            "a\u{1b}]8;;https://example.com\u{7}link\u{1b}]8;;\u{7}z",
            "alinkz",
        ),
        ("a\u{1b}]unterminated", "anterminated"),
        ("a\u{1b}]funterminated", "aunterminated"),
        ("a\u{1b}Pabc\u{1b}\\z", "aabc\u{1b}\\z"),
        ("a\u{1b}^abc\u{7}z", "a\u{1b}^abc\u{7}z"),
        ("a\u{1b}_abc\u{9c}z", "a\u{1b}_abc\u{9c}z"),
        ("a\u{90}abc\u{9c}z", "a\u{90}abc\u{9c}z"),
        ("a\u{9d}abc\u{9c}z", "a\u{9d}abc\u{9c}z"),
        ("a\u{9b}31mred", "ared"),
        ("a\u{1b}(0x", "ax"),
        ("a\u{1b}*0x", "a\u{1b}*0x"),
        ("a\u{1b}+c", "a\u{1b}+c"),
        ("a\u{1b}/0x", "a\u{1b}/0x"),
        ("a\u{1b}cok", "aok"),
        ("a\u{1b}\\ok", "a\u{1b}\\ok"),
        ("x\u{1b}ay", "x\u{1b}ay"),
        ("x\u{9b}ay", "x\u{9b}ay"),
        ("x\u{1b}aay", "x\u{1b}aay"),
        ("x\u{1b}a1y", "x\u{1b}a1y"),
        ("x\u{1b}a[y", "x\u{1b}a[y"),
        ("x\u{1b}a)y", "x\u{1b}a)y"),
        ("x\u{1b}amy", "x\u{1b}amy"),
        ("x\u{1b}a\\y", "x\u{1b}a\\y"),
        ("x\u{1b}a\u{9b}y", "x\u{1b}a"),
        ("x\u{1b}a\u{9d}y", "x\u{1b}a\u{9d}y"),
        ("x\u{1b}fy", "xy"),
        ("x\u{9b}fy", "xy"),
        ("x\u{1b}fay", "xay"),
        ("x\u{1b}f1y", "x1y"),
        ("x\u{1b}f[y", "x[y"),
        ("x\u{1b}f)y", "x)y"),
        ("x\u{1b}fmy", "xmy"),
        ("x\u{1b}f\\y", "x\\y"),
        ("x\u{1b}f\u{9b}y", "x"),
        ("x\u{1b}f\u{9d}y", "x\u{9d}y"),
        ("x\u{1b}0y", "x"),
        ("x\u{9b}0y", "x"),
        ("x\u{1b}0ay", "xay"),
        ("x\u{1b}01y", "x"),
        ("x\u{1b}0[y", "x[y"),
        ("x\u{1b}0)y", "x)y"),
        ("x\u{1b}0my", "xy"),
        ("x\u{1b}0\\y", "x\\y"),
        ("x\u{1b}0\u{9b}y", "x"),
        ("x\u{1b}0\u{9d}y", "x\u{9d}y"),
        ("x\u{1b}1y", "x"),
        ("x\u{9b}1y", "x"),
        ("x\u{1b}1ay", "xay"),
        ("x\u{1b}11y", "x"),
        ("x\u{1b}1[y", "x[y"),
        ("x\u{1b}1)y", "x)y"),
        ("x\u{1b}1my", "xy"),
        ("x\u{1b}1\\y", "x\\y"),
        ("x\u{1b}1\u{9b}y", "x"),
        ("x\u{1b}1\u{9d}y", "x\u{9d}y"),
        ("x\u{1b};y", "x"),
        ("x\u{9b};y", "x"),
        ("x\u{1b};ay", "x\u{1b};ay"),
        ("x\u{1b};1y", "x"),
        ("x\u{1b};[y", "x"),
        ("x\u{1b};)y", "x"),
        ("x\u{1b};my", "xy"),
        ("x\u{1b};\\y", "x\u{1b};\\y"),
        ("x\u{1b};\u{9b}y", "x\u{1b};"),
        ("x\u{1b};\u{9d}y", "x\u{1b};\u{9d}y"),
        ("x\u{1b}:y", "x\u{1b}:y"),
        ("x\u{9b}:y", "x\u{9b}:y"),
        ("x\u{1b}:ay", "x\u{1b}:ay"),
        ("x\u{1b}:1y", "x\u{1b}:1y"),
        ("x\u{1b}:[y", "x\u{1b}:[y"),
        ("x\u{1b}:)y", "x\u{1b}:)y"),
        ("x\u{1b}:my", "x\u{1b}:my"),
        ("x\u{1b}:\\y", "x\u{1b}:\\y"),
        ("x\u{1b}:\u{9b}y", "x\u{1b}:"),
        ("x\u{1b}:\u{9d}y", "x\u{1b}:\u{9d}y"),
        ("x\u{1b}[y", "x"),
        ("x\u{9b}[y", "x"),
        ("x\u{1b}[ay", "x\u{1b}[ay"),
        ("x\u{1b}[1y", "x"),
        ("x\u{1b}[[y", "x"),
        ("x\u{1b}[)y", "x"),
        ("x\u{1b}[my", "xy"),
        ("x\u{1b}[\\y", "x\u{1b}[\\y"),
        ("x\u{1b}[\u{9b}y", "x\u{1b}["),
        ("x\u{1b}[\u{9d}y", "x\u{1b}[\u{9d}y"),
        ("x\u{1b}]y", "x"),
        ("x\u{9b}]y", "x"),
        ("x\u{1b}]ay", "x\u{1b}]ay"),
        ("x\u{1b}]1y", "x"),
        ("x\u{1b}][y", "x"),
        ("x\u{1b}])y", "x"),
        ("x\u{1b}]my", "xy"),
        ("x\u{1b}]\\y", "x\u{1b}]\\y"),
        ("x\u{1b}]\u{9b}y", "x\u{1b}]"),
        ("x\u{1b}]\u{9d}y", "x\u{1b}]\u{9d}y"),
        ("x\u{1b}(y", "x"),
        ("x\u{9b}(y", "x"),
        ("x\u{1b}(ay", "x\u{1b}(ay"),
        ("x\u{1b}(1y", "x"),
        ("x\u{1b}([y", "x"),
        ("x\u{1b}()y", "x"),
        ("x\u{1b}(my", "xy"),
        ("x\u{1b}(\\y", "x\u{1b}(\\y"),
        ("x\u{1b}(\u{9b}y", "x\u{1b}("),
        ("x\u{1b}(\u{9d}y", "x\u{1b}(\u{9d}y"),
        ("x\u{1b})y", "x"),
        ("x\u{9b})y", "x"),
        ("x\u{1b})ay", "x\u{1b})ay"),
        ("x\u{1b})1y", "x"),
        ("x\u{1b})[y", "x"),
        ("x\u{1b}))y", "x"),
        ("x\u{1b})my", "xy"),
        ("x\u{1b})\\y", "x\u{1b})\\y"),
        ("x\u{1b})\u{9b}y", "x\u{1b})"),
        ("x\u{1b})\u{9d}y", "x\u{1b})\u{9d}y"),
        ("x\u{1b}#y", "x"),
        ("x\u{9b}#y", "x"),
        ("x\u{1b}#ay", "x\u{1b}#ay"),
        ("x\u{1b}#1y", "x"),
        ("x\u{1b}#[y", "x"),
        ("x\u{1b}#)y", "x"),
        ("x\u{1b}#my", "xy"),
        ("x\u{1b}#\\y", "x\u{1b}#\\y"),
        ("x\u{1b}#\u{9b}y", "x\u{1b}#"),
        ("x\u{1b}#\u{9d}y", "x\u{1b}#\u{9d}y"),
        ("x\u{1b}?y", "x"),
        ("x\u{9b}?y", "x"),
        ("x\u{1b}?ay", "x\u{1b}?ay"),
        ("x\u{1b}?1y", "x"),
        ("x\u{1b}?[y", "x"),
        ("x\u{1b}?)y", "x"),
        ("x\u{1b}?my", "xy"),
        ("x\u{1b}?\\y", "x\u{1b}?\\y"),
        ("x\u{1b}?\u{9b}y", "x\u{1b}?"),
        ("x\u{1b}?\u{9d}y", "x\u{1b}?\u{9d}y"),
        ("x\u{1b}my", "xy"),
        ("x\u{9b}my", "xy"),
        ("x\u{1b}may", "xay"),
        ("x\u{1b}m1y", "x1y"),
        ("x\u{1b}m[y", "x[y"),
        ("x\u{1b}m)y", "x)y"),
        ("x\u{1b}mmy", "xmy"),
        ("x\u{1b}m\\y", "x\\y"),
        ("x\u{1b}m\u{9b}y", "x"),
        ("x\u{1b}m\u{9d}y", "x\u{9d}y"),
        ("x\u{1b}Py", "xy"),
        ("x\u{9b}Py", "xy"),
        ("x\u{1b}Pay", "xay"),
        ("x\u{1b}P1y", "x1y"),
        ("x\u{1b}P[y", "x[y"),
        ("x\u{1b}P)y", "x)y"),
        ("x\u{1b}Pmy", "xmy"),
        ("x\u{1b}P\\y", "x\\y"),
        ("x\u{1b}P\u{9b}y", "x"),
        ("x\u{1b}P\u{9d}y", "x\u{9d}y"),
        ("x\u{1b}_y", "x\u{1b}_y"),
        ("x\u{9b}_y", "x\u{9b}_y"),
        ("x\u{1b}_ay", "x\u{1b}_ay"),
        ("x\u{1b}_1y", "x\u{1b}_1y"),
        ("x\u{1b}_[y", "x\u{1b}_[y"),
        ("x\u{1b}_)y", "x\u{1b}_)y"),
        ("x\u{1b}_my", "x\u{1b}_my"),
        ("x\u{1b}_\\y", "x\u{1b}_\\y"),
        ("x\u{1b}_\u{9b}y", "x\u{1b}_"),
        ("x\u{1b}_\u{9d}y", "x\u{1b}_\u{9d}y"),
        ("x\u{1b}\\y", "x\u{1b}\\y"),
        ("x\u{9b}\\y", "x\u{9b}\\y"),
        ("x\u{1b}\\ay", "x\u{1b}\\ay"),
        ("x\u{1b}\\1y", "x\u{1b}\\1y"),
        ("x\u{1b}\\[y", "x\u{1b}\\[y"),
        ("x\u{1b}\\)y", "x\u{1b}\\)y"),
        ("x\u{1b}\\my", "x\u{1b}\\my"),
        ("x\u{1b}\\\\y", "x\u{1b}\\\\y"),
        ("x\u{1b}\\\u{9b}y", "x\u{1b}\\"),
        ("x\u{1b}\\\u{9d}y", "x\u{1b}\\\u{9d}y"),
        ("x\u{1b}\u{7}y", "x\u{1b}\u{7}y"),
        ("x\u{9b}\u{7}y", "x\u{9b}\u{7}y"),
        ("x\u{1b}\u{7}ay", "x\u{1b}\u{7}ay"),
        ("x\u{1b}\u{7}1y", "x\u{1b}\u{7}1y"),
        ("x\u{1b}\u{7}[y", "x\u{1b}\u{7}[y"),
        ("x\u{1b}\u{7})y", "x\u{1b}\u{7})y"),
        ("x\u{1b}\u{7}my", "x\u{1b}\u{7}my"),
        ("x\u{1b}\u{7}\\y", "x\u{1b}\u{7}\\y"),
        ("x\u{1b}\u{7}\u{9b}y", "x\u{1b}\u{7}"),
        ("x\u{1b}\u{7}\u{9d}y", "x\u{1b}\u{7}\u{9d}y"),
        ("x\u{1b}\u{1b}y", "x\u{1b}"),
        ("x\u{9b}\u{1b}y", "x\u{9b}"),
        ("x\u{1b}\u{1b}ay", "x\u{1b}\u{1b}ay"),
        ("x\u{1b}\u{1b}1y", "x\u{1b}"),
        ("x\u{1b}\u{1b}[y", "x\u{1b}"),
        ("x\u{1b}\u{1b})y", "x\u{1b}"),
        ("x\u{1b}\u{1b}my", "x\u{1b}y"),
        ("x\u{1b}\u{1b}\\y", "x\u{1b}\u{1b}\\y"),
        ("x\u{1b}\u{1b}\u{9b}y", "x\u{1b}\u{1b}"),
        ("x\u{1b}\u{1b}\u{9d}y", "x\u{1b}\u{1b}\u{9d}y"),
        ("x\u{1b}\u{9b}y", "x\u{1b}"),
        ("x\u{9b}\u{9b}y", "x\u{9b}"),
        ("x\u{1b}\u{9b}ay", "x\u{1b}\u{9b}ay"),
        ("x\u{1b}\u{9b}1y", "x\u{1b}"),
        ("x\u{1b}\u{9b}[y", "x\u{1b}"),
        ("x\u{1b}\u{9b})y", "x\u{1b}"),
        ("x\u{1b}\u{9b}my", "x\u{1b}y"),
        ("x\u{1b}\u{9b}\\y", "x\u{1b}\u{9b}\\y"),
        ("x\u{1b}\u{9b}\u{9b}y", "x\u{1b}\u{9b}"),
        ("x\u{1b}\u{9b}\u{9d}y", "x\u{1b}\u{9b}\u{9d}y"),
        ("x\u{1b}\u{9c}y", "x\u{1b}\u{9c}y"),
        ("x\u{9b}\u{9c}y", "x\u{9b}\u{9c}y"),
        ("x\u{1b}\u{9c}ay", "x\u{1b}\u{9c}ay"),
        ("x\u{1b}\u{9c}1y", "x\u{1b}\u{9c}1y"),
        ("x\u{1b}\u{9c}[y", "x\u{1b}\u{9c}[y"),
        ("x\u{1b}\u{9c})y", "x\u{1b}\u{9c})y"),
        ("x\u{1b}\u{9c}my", "x\u{1b}\u{9c}my"),
        ("x\u{1b}\u{9c}\\y", "x\u{1b}\u{9c}\\y"),
        ("x\u{1b}\u{9c}\u{9b}y", "x\u{1b}\u{9c}"),
        ("x\u{1b}\u{9c}\u{9d}y", "x\u{1b}\u{9c}\u{9d}y"),
        ("x\u{1b}\u{90}y", "x\u{1b}\u{90}y"),
        ("x\u{9b}\u{90}y", "x\u{9b}\u{90}y"),
        ("x\u{1b}\u{90}ay", "x\u{1b}\u{90}ay"),
        ("x\u{1b}\u{90}1y", "x\u{1b}\u{90}1y"),
        ("x\u{1b}\u{90}[y", "x\u{1b}\u{90}[y"),
        ("x\u{1b}\u{90})y", "x\u{1b}\u{90})y"),
        ("x\u{1b}\u{90}my", "x\u{1b}\u{90}my"),
        ("x\u{1b}\u{90}\\y", "x\u{1b}\u{90}\\y"),
        ("x\u{1b}\u{90}\u{9b}y", "x\u{1b}\u{90}"),
        ("x\u{1b}\u{90}\u{9d}y", "x\u{1b}\u{90}\u{9d}y"),
        ("x\u{1b}\u{9d}y", "x\u{1b}\u{9d}y"),
        ("x\u{9b}\u{9d}y", "x\u{9b}\u{9d}y"),
        ("x\u{1b}\u{9d}ay", "x\u{1b}\u{9d}ay"),
        ("x\u{1b}\u{9d}1y", "x\u{1b}\u{9d}1y"),
        ("x\u{1b}\u{9d}[y", "x\u{1b}\u{9d}[y"),
        ("x\u{1b}\u{9d})y", "x\u{1b}\u{9d})y"),
        ("x\u{1b}\u{9d}my", "x\u{1b}\u{9d}my"),
        ("x\u{1b}\u{9d}\\y", "x\u{1b}\u{9d}\\y"),
        ("x\u{1b}\u{9d}\u{9b}y", "x\u{1b}\u{9d}"),
        ("x\u{1b}\u{9d}\u{9d}y", "x\u{1b}\u{9d}\u{9d}y"),
    ];

    #[test]
    fn matches_upstream_reference_for_generated_compatibility_inputs() {
        for (input, expected) in COMPATIBILITY_PAIRS {
            assert_eq!(strip_ansi(input), *expected, "input: {input:?}");
        }
    }

    #[test]
    fn strips_ris_without_leaking_the_final_byte() {
        assert_eq!(strip_ansi("\u{001b}cdone"), "done");
    }

    #[test]
    fn strips_single_byte_esc_sequences_without_leaking_final_bytes() {
        for code in 'g'..='m' {
            assert_eq!(strip_ansi(&format!("\u{001b}{code}ok")), "ok");
        }
        for code in 'r'..='t' {
            assert_eq!(strip_ansi(&format!("\u{001b}{code}ok")), "ok");
        }
    }

    #[test]
    fn strips_common_ansi_sequences_used_in_tool_output() {
        let input = "a\u{001b}[31mred\u{001b}[0m\u{001b}]8;;https://example.com\u{0007}link\u{001b}]8;;\u{0007}z";
        assert_eq!(strip_ansi(input), "aredlinkz");
    }
}
