//! Port of `packages/coding-agent/src/core/experimental.ts` — the
//! `PI_EXPERIMENTAL` feature flag probe. No deviations; the env read
//! happens at call time like upstream (`process.env`).

const ENV_EXPERIMENTAL: &str = "PI_EXPERIMENTAL";

/// Upstream `areExperimentalFeaturesEnabled`.
pub fn are_experimental_features_enabled() -> bool {
    std::env::var(ENV_EXPERIMENTAL).as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    /// Serialized env access: the upstream suite mutates
    /// `process.env.PI_EXPERIMENTAL` per case; a single sequential test
    /// walks the same case table without racing sibling tests.
    static ENV_LOCK: LazyLock<std::sync::Mutex<()>> = LazyLock::new(std::sync::Mutex::default);

    #[test]
    fn pins_upstream_case_table() {
        let _guard = ENV_LOCK.lock().expect("lock");
        let original = std::env::var(ENV_EXPERIMENTAL).ok();

        let cases: &[(&str, bool)] = &[
            ("", false),     // empty string
            ("1", true),     // set to 1
            ("0", false),    // set to 0
            ("true", false), // non-1 value
        ];

        std::env::remove_var(ENV_EXPERIMENTAL);
        assert!(!are_experimental_features_enabled(), "unset ⇒ false");

        for (value, expected) in cases {
            std::env::set_var(ENV_EXPERIMENTAL, value);
            assert_eq!(
                are_experimental_features_enabled(),
                *expected,
                "PI_EXPERIMENTAL={value:?}"
            );
        }

        match original {
            Some(value) => std::env::set_var(ENV_EXPERIMENTAL, value),
            None => std::env::remove_var(ENV_EXPERIMENTAL),
        }
    }
}
