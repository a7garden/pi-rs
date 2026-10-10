//! `cli/list-models.ts` — list available models with optional fuzzy search.
//!
//! \*\*D101 — local fuzzy matcher.** Upstream imports `fuzzyFilter` from
//! pi-tui (excluded); the scoring function is ported verbatim here.

use crate::core::auth_guidance::format_no_models_available_message;
use crate::core::model_runtime::ModelRuntime;
use pi_ai::models::Models as _;

/// One fuzzy match outcome (`FuzzyMatch`).
struct FuzzyMatch {
    matches: bool,
    score: f64,
}

fn match_query(normalized_query: &str, text_lower: &str) -> FuzzyMatch {
    if normalized_query.is_empty() {
        return FuzzyMatch {
            matches: true,
            score: 0.0,
        };
    }
    if normalized_query.chars().count() > text_lower.chars().count() {
        return FuzzyMatch {
            matches: false,
            score: 0.0,
        };
    }

    let text_chars: Vec<char> = text_lower.chars().collect();
    let query_chars: Vec<char> = normalized_query.chars().collect();
    let mut query_index = 0usize;
    let mut score = 0.0f64;
    let mut last_match_index: i64 = -1;
    let mut consecutive_matches = 0usize;

    while query_index < query_chars.len() {
        let start = (last_match_index + 1) as usize;
        let found = text_chars[start..]
            .iter()
            .position(|c| *c == query_chars[query_index])
            .map(|offset| start + offset);
        let Some(i) = found else { break };

        let is_word_boundary = i == 0
            || text_chars[i - 1].is_whitespace()
            || matches!(text_chars[i - 1], '-' | '_' | '.' | '/' | ':');

        // Reward consecutive matches
        if last_match_index == i as i64 - 1 {
            consecutive_matches += 1;
            score -= (consecutive_matches * 5) as f64;
        } else {
            consecutive_matches = 0;
            // Penalize gaps
            if last_match_index >= 0 {
                score += ((i as i64 - last_match_index - 1) * 2) as f64;
            }
        }

        // Reward word boundary matches
        if is_word_boundary {
            score -= 10.0;
        }

        // Slight penalty for later matches
        score += i as f64 * 0.1;

        last_match_index = i as i64;
        query_index += 1;
    }

    if query_index < query_chars.len() {
        return FuzzyMatch {
            matches: false,
            score: 0.0,
        };
    }

    if normalized_query == text_lower {
        score -= 100.0;
    }

    FuzzyMatch {
        matches: true,
        score,
    }
}

/// Split a lowercase query into swapped alphanumeric order
/// (`gpt4` → `4gpt`) when the primary match fails.
fn swapped_query(query_lower: &str) -> Option<String> {
    let letters: String = query_lower
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    let digits: String = query_lower.chars().filter(|c| c.is_ascii_digit()).collect();
    if letters.is_empty() || digits.is_empty() {
        return None;
    }
    if format!("{letters}{digits}") == *query_lower {
        return Some(format!("{digits}{letters}"));
    }
    if format!("{digits}{letters}") == *query_lower {
        return Some(format!("{letters}{digits}"));
    }
    None
}

fn fuzzy_match(query: &str, text: &str) -> FuzzyMatch {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();

    let primary = match_query(&query_lower, &text_lower);
    if primary.matches {
        return primary;
    }

    let Some(swapped) = swapped_query(&query_lower) else {
        return primary;
    };

    let swapped_match = match_query(&swapped, &text_lower);
    if !swapped_match.matches {
        return primary;
    }

    FuzzyMatch {
        matches: true,
        score: swapped_match.score + 5.0,
    }
}

/// Filter and sort items by fuzzy match quality (`fuzzyFilter`).
pub fn fuzzy_filter<T, F: Fn(&T) -> String>(items: Vec<T>, query: &str, get_text: F) -> Vec<T> {
    if query.trim().is_empty() {
        return items;
    }
    let tokens: Vec<&str> = query
        .trim()
        .split(|c: char| c.is_whitespace() || c == '/')
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return items;
    }

    let mut results: Vec<(T, f64)> = Vec::new();
    for item in items {
        let text = get_text(&item);
        let mut total_score = 0.0f64;
        let mut all_match = true;
        for token in &tokens {
            let m = fuzzy_match(token, &text);
            if m.matches {
                total_score += m.score;
            } else {
                all_match = false;
                break;
            }
        }
        if all_match {
            results.push((item, total_score));
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.into_iter().map(|(item, _)| item).collect()
}

/// Format a number as human-readable (e.g., 200000 -> "200K",
/// 1000000 -> "1M") (`formatTokenCount`).
pub fn format_token_count(count: u64) -> String {
    if count >= 1_000_000 {
        let millions = count as f64 / 1_000_000.0;
        if millions % 1.0 == 0.0 {
            format!("{}M", millions as u64)
        } else {
            format!("{:.1}M", ((millions * 10.0).round()) / 10.0)
        }
    } else if count >= 1_000 {
        let thousands = count as f64 / 1_000.0;
        if thousands % 1.0 == 0.0 {
            format!("{}K", thousands as u64)
        } else {
            format!("{:.1}K", ((thousands * 10.0).round()) / 10.0)
        }
    } else {
        count.to_string()
    }
}

fn pad_end(value: &str, width: usize) -> String {
    let len = value.chars().count();
    if len >= width {
        value.to_string()
    } else {
        format!("{value}{}", " ".repeat(width - len))
    }
}

/// List available models, optionally filtered by search pattern
/// (`listModels`).
pub async fn list_models(
    model_runtime: &ModelRuntime,
    search_pattern: Option<&str>,
    signal: Option<pi_ai::utils::abort::AbortSignal>,
) {
    if let Some(load_error) = model_runtime.get_error() {
        eprintln!("Warning: errors loading models.json:\n{load_error}");
    }

    let models = match model_runtime
        .get_available(
            None,
            Some(pi_ai::auth::types::AuthOperationOptions {
                signal,
                ..Default::default()
            }),
        )
        .await
    {
        Ok(models) => models,
        Err(_) => Vec::new(),
    };

    if models.is_empty() {
        println!("{}", format_no_models_available_message());
        return;
    }

    // Apply fuzzy filter if search pattern provided
    let filtered_models = match search_pattern.filter(|p| !p.is_empty()) {
        Some(pattern) => fuzzy_filter(models, pattern, |m| {
            format!("{} {}", m.base.provider, m.base.id)
        }),
        None => models,
    };

    if filtered_models.is_empty() {
        println!("No models matching \"{}\"", search_pattern.unwrap_or(""));
        return;
    }

    // Sort by provider, then by model id
    let mut filtered_models = filtered_models;
    filtered_models.sort_by(|a, b| {
        a.base
            .provider
            .cmp(&b.base.provider)
            .then_with(|| a.base.id.cmp(&b.base.id))
    });

    // Calculate column widths
    struct Row {
        provider: String,
        model: String,
        context: String,
        max_out: String,
        thinking: &'static str,
        images: &'static str,
    }
    let rows: Vec<Row> = filtered_models
        .iter()
        .map(|m| Row {
            provider: m.base.provider.clone(),
            model: m.base.id.clone(),
            context: format_token_count(m.context_window),
            max_out: format_token_count(m.max_tokens),
            thinking: if m.reasoning { "yes" } else { "no" },
            images: if m
                .base
                .input
                .iter()
                .any(|i| i == &pi_ai::types::ModelInput::Image)
            {
                "yes"
            } else {
                "no"
            },
        })
        .collect();

    let max_of = |header: usize, values: &mut dyn Iterator<Item = usize>| -> usize {
        values.fold(header, usize::max)
    };
    let widths = (
        max_of("provider".len(), &mut rows.iter().map(|r| r.provider.len())),
        max_of("model".len(), &mut rows.iter().map(|r| r.model.len())),
        max_of("context".len(), &mut rows.iter().map(|r| r.context.len())),
        max_of("max-out".len(), &mut rows.iter().map(|r| r.max_out.len())),
        max_of("thinking".len(), &mut rows.iter().map(|r| r.thinking.len())),
        max_of("images".len(), &mut rows.iter().map(|r| r.images.len())),
    );

    let line = |cells: [&str; 6]| -> String {
        [
            pad_end(cells[0], widths.0),
            pad_end(cells[1], widths.1),
            pad_end(cells[2], widths.2),
            pad_end(cells[3], widths.3),
            pad_end(cells[4], widths.4),
            pad_end(cells[5], widths.5),
        ]
        .join("  ")
    };

    // Print header
    println!(
        "{}",
        line(["provider", "model", "context", "max-out", "thinking", "images"])
    );

    // Print rows
    for row in &rows {
        println!(
            "{}",
            line([
                &row.provider,
                &row.model,
                &row.context,
                &row.max_out,
                row.thinking,
                row.images
            ])
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_token_counts() {
        assert_eq!(format_token_count(200_000), "200K");
        assert_eq!(format_token_count(1_000_000), "1M");
        assert_eq!(format_token_count(1_500_000), "1.5M");
        assert_eq!(format_token_count(1_000), "1K");
        assert_eq!(format_token_count(1_200), "1.2K");
        assert_eq!(format_token_count(999), "999");
    }

    #[test]
    fn fuzzy_matches_in_order() {
        assert!(fuzzy_match("gpt4", "openai gpt-4o").matches);
        assert!(!fuzzy_match("zzz", "claude-sonnet").matches);
    }

    #[test]
    fn fuzzy_filters_and_sorts() {
        let items = vec![
            "anthropic claude-sonnet-4",
            "openai gpt-4o",
            "google gemini-pro",
        ];
        let out = fuzzy_filter(items, "gpt", |s| s.to_string());
        assert_eq!(out, vec!["openai gpt-4o"]);
    }
}
