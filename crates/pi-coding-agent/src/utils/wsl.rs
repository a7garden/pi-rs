//! Port of `packages/coding-agent/src/utils/wsl.ts` — detect Windows
//! Subsystem for Linux, where Windows executables are reachable through
//! interop.

/// Detect WSL using `std::env` for environment variables.
pub fn is_wsl() -> bool {
    is_wsl_in_env(&|name| std::env::var(name).ok())
}

/// Detect WSL against a custom environment lookup (the counterpart of the
/// upstream `env` parameter). Checks `WSL_DISTRO_NAME`/`WSLENV`, then
/// `/proc/version` for `microsoft|wsl` (case-insensitive).
pub fn is_wsl_in_env(env: &dyn Fn(&str) -> Option<String>) -> bool {
    if env("WSL_DISTRO_NAME")
        .map(|value| !value.is_empty())
        .unwrap_or(false)
        || env("WSLENV")
            .map(|value| !value.is_empty())
            .unwrap_or(false)
    {
        return true;
    }

    match std::fs::read_to_string("/proc/version") {
        Ok(release) => {
            release.to_lowercase().contains("microsoft") || release.to_lowercase().contains("wsl")
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn env_from(entries: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    #[test]
    fn detects_wsl_from_environment() {
        assert!(is_wsl_in_env(&env_from(&[("WSL_DISTRO_NAME", "Ubuntu")])));
        assert!(is_wsl_in_env(&env_from(&[("WSLENV", "u")])));
    }

    #[test]
    fn empty_environment_values_do_not_count() {
        // On a non-Linux host /proc/version is unreadable, so this is false.
        if !cfg!(target_os = "linux") {
            assert!(!is_wsl_in_env(&env_from(&[("WSL_DISTRO_NAME", "")])));
            assert!(!is_wsl_in_env(&env_from(&[])));
        }
    }

    #[test]
    fn detects_wsl_from_proc_version_on_linux() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let env = env_from(&[]);
        if std::fs::read_to_string("/proc/version")
            .map(|release| release.to_lowercase().contains("microsoft"))
            .unwrap_or(false)
        {
            assert!(is_wsl_in_env(&env));
        }
    }
}
