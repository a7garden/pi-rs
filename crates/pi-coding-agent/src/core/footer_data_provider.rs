//! Partial port of `packages/coding-agent/src/core/footer-data-provider.ts`.
//!
//! Only [`find_git_paths`] is ported here (deviation D57d). The rest of the
//! upstream module — the interactive-mode `FooterDataProvider` class with its
//! git-branch polling/watching, extension status map, and refresh timers — is
//! deferred with the interactive/TUI mode and is not ported in this pass.

use std::fs;
use std::path::Path;

/// Git metadata paths (upstream `GitPaths`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitPaths {
    /// The repository working-tree root that the search started from.
    pub repo_dir: String,
    /// The common git directory (`.git`, or the worktree's resolved common
    /// dir).
    pub common_git_dir: String,
    /// Path to the `HEAD` file inside the git dir.
    pub head_path: String,
}

/// Find git metadata paths by walking up from `cwd`. Handles both regular git
/// repos (`.git` is a directory) and worktrees (`.git` is a file). Returns
/// `None` when no repository is found (upstream `findGitPaths`).
pub fn find_git_paths(cwd: &str) -> Option<GitPaths> {
    let mut dir = String::from(cwd);
    loop {
        let git_path = Path::new(&dir).join(".git");
        if git_path.exists() {
            match fs::metadata(&git_path) {
                Ok(metadata) if metadata.is_file() => {
                    let content = fs::read_to_string(&git_path).ok()?.trim().to_string();
                    if let Some(gitdir) = content.strip_prefix("gitdir: ") {
                        let git_dir = resolve(&dir, gitdir.trim());
                        let head_path = Path::new(&git_dir).join("HEAD");
                        if !head_path.exists() {
                            return None;
                        }
                        let common_dir_path = Path::new(&git_dir).join("commondir");
                        let common_git_dir = if common_dir_path.exists() {
                            let commondir = fs::read_to_string(&common_dir_path)
                                .ok()?
                                .trim()
                                .to_string();
                            resolve(&git_dir, &commondir)
                        } else {
                            git_dir.clone()
                        };
                        return Some(GitPaths {
                            repo_dir: dir,
                            common_git_dir,
                            head_path: head_path.to_string_lossy().into_owned(),
                        });
                    }
                }
                Ok(metadata) if metadata.is_dir() => {
                    let head_path = git_path.join("HEAD");
                    if !head_path.exists() {
                        return None;
                    }
                    return Some(GitPaths {
                        repo_dir: dir,
                        common_git_dir: git_path.to_string_lossy().into_owned(),
                        head_path: head_path.to_string_lossy().into_owned(),
                    });
                }
                // Upstream wraps the stat/read block in try/catch → null.
                _ => return None,
            }
        }
        let parent = Path::new(&dir).parent()?.to_string_lossy().into_owned();
        if parent == dir {
            return None;
        }
        dir = parent;
    }
}

/// `path.resolve(from, relative)` for a relative gitdir/commondir entry:
/// absolute-ify, then lexically normalize `.`/`..` segments.
fn resolve(from: &str, relative: &str) -> String {
    // `path.resolve` has no tilde handling; disable `resolve_path`'s on both
    // inputs.
    let no_tilde = crate::utils::paths::PathInputOptions {
        expand_tilde: Some(false),
        ..Default::default()
    };
    let base = crate::utils::paths::normalize_path(from, &no_tilde);
    let input = crate::utils::paths::normalize_path(relative, &no_tilde);
    crate::utils::paths::resolve_path(&input, &base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn finds_regular_repo_by_walking_up() {
        let root = std::env::temp_dir().join(format!("pi-rs-git-paths-{}", std::process::id()));
        let repo = root.join("repo");
        let nested = repo.join("a/b");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&nested).unwrap();
        fs::write(repo.join(".git/HEAD"), "ref: refs/heads/main").unwrap();

        let paths = find_git_paths(nested.to_string_lossy().as_ref()).unwrap();
        assert_eq!(paths.repo_dir, repo.to_string_lossy());
        assert_eq!(paths.common_git_dir, repo.join(".git").to_string_lossy());
        assert_eq!(paths.head_path, repo.join(".git/HEAD").to_string_lossy());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn handles_worktree_gitfile_and_missing_head() {
        let root = std::env::temp_dir().join(format!("pi-rs-git-wt-{}", std::process::id()));
        let repo = root.join("wt");
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all("/tmp/places");
        fs::create_dir_all(&repo).unwrap();
        fs::write(
            repo.join(".git"),
            "gitdir: /tmp/places/repo/.git/worktrees/wt",
        )
        .unwrap();

        // HEAD missing in the gitdir → null.
        assert_eq!(find_git_paths(&repo.to_string_lossy()), None);

        fs::create_dir_all("/tmp/places/repo/.git/worktrees/wt").unwrap();
        fs::write(
            "/tmp/places/repo/.git/worktrees/wt/HEAD",
            "ref: refs/heads/main",
        )
        .unwrap();
        fs::write("/tmp/places/repo/.git/worktrees/wt/commondir", "../..").unwrap();

        let paths = find_git_paths(&repo.to_string_lossy()).unwrap();
        assert_eq!(paths.repo_dir, repo.to_string_lossy());
        assert_eq!(paths.head_path, "/tmp/places/repo/.git/worktrees/wt/HEAD");
        assert_eq!(paths.common_git_dir, "/tmp/places/repo/.git");

        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all("/tmp/places");
    }

    #[test]
    fn returns_none_outside_a_repo() {
        let root = std::env::temp_dir().join(format!("pi-rs-git-none-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        assert_eq!(find_git_paths(root.to_string_lossy().as_ref()), None);
        let _ = fs::remove_dir_all(&root);
    }
}
