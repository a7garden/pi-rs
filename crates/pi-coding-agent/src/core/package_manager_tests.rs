//! Port of `packages/coding-agent/test/package-manager.test.ts` (module docs
//! D40: upstream `vi.spyOn` mocks of private methods become a recording
//! [`CommandRunner`] fake).
//!
//! Deferred (land with a network/child-process seam or are Node-runtime
//! specific):
//! - "should wait for close before resolving captured stdout" — Node
//!   exit-vs-close stream semantics (upstream `spawnCaptureCommand` mock).
//! - the max-concurrency ratio assertions of "should batch npm updates per
//!   scope and run git updates in parallel..." (args/ordering assertions are
//!   ported; the timing probes need a injectable delay hook).
//! - cases that execute real `npm install`/`git clone` against the network
//!   (none upstream — all such cases are mocked and ported).

use std::sync::Arc;

use parking_lot::Mutex;

use super::*;
use crate::core::settings_manager::{Settings, SettingsManager, SettingsManagerCreateOptions};

type Handler = Box<dyn Fn(&CommandCall) -> Result<String, String> + Send + Sync>;

struct MockRunner {
    handler: Handler,
    calls: Mutex<Vec<CommandCall>>,
}

impl MockRunner {
    fn record(&self, call: &CommandCall) {
        self.calls.lock().push(call.clone());
    }

    fn calls(&self) -> Vec<CommandCall> {
        self.calls.lock().clone()
    }

    fn calls_for(&self, command: &str) -> Vec<(Vec<String>, Option<String>)> {
        self.calls()
            .into_iter()
            .filter(|call| call.command == command)
            .map(|call| (call.args, call.cwd))
            .collect()
    }

    fn call_count(&self) -> usize {
        self.calls.lock().len()
    }
}

impl CommandRunner for MockRunner {
    fn run(&self, call: &CommandCall) -> Result<(), String> {
        self.record(call);
        (self.handler)(call).map(|_| ())
    }

    fn run_capture(&self, call: &CommandCall, _timeout_ms: Option<u64>) -> Result<String, String> {
        self.record(call);
        (self.handler)(call)
    }

    fn run_sync(&self, call: &CommandCall) -> Result<String, String> {
        self.record(call);
        (self.handler)(call)
    }
}

fn mock_runner(
    handler: impl Fn(&CommandCall) -> Result<String, String> + Send + Sync + 'static,
) -> Arc<MockRunner> {
    Arc::new(MockRunner {
        handler: Box::new(handler),
        calls: Mutex::new(Vec::new()),
    })
}

fn failing_runner() -> Arc<MockRunner> {
    mock_runner(|call| Err(format!("unexpected command: {}", call.display())))
}

struct TestEnv {
    temp_dir: String,
    agent_dir: String,
    settings: Arc<Mutex<SettingsManager>>,
}

fn fresh_dir(name: &str) -> String {
    // Counter disambiguates same-tick creations under parallel load.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pm-rs-{name}-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn write_file(path: &str, content: &str) {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

fn in_memory_manager() -> Arc<Mutex<SettingsManager>> {
    Arc::new(Mutex::new(SettingsManager::in_memory(
        &Settings::new(),
        SettingsManagerCreateOptions::default(),
    )))
}

fn make_pm(
    env: &TestEnv,
    builtins: Vec<String>,
    runner: Arc<dyn CommandRunner>,
) -> DefaultPackageManager {
    DefaultPackageManager::with_runner(
        PackageManagerOptions {
            cwd: env.temp_dir.clone(),
            agent_dir: env.agent_dir.clone(),
            settings_manager: env.settings.clone(),
            builtin_extensions: builtins,
        },
        runner,
    )
}

fn setup(name: &str) -> (TestEnv, DefaultPackageManager) {
    let temp_dir = fresh_dir(name);
    let agent_dir = Path::new(&temp_dir)
        .join("agent")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let env = TestEnv {
        temp_dir,
        agent_dir,
        settings: in_memory_manager(),
    };
    let pm = make_pm(&env, Vec::new(), Arc::new(ProcessCommandRunner));
    (env, pm)
}

fn env_guard() -> parking_lot::MutexGuard<'static, ()> {
    // Shared crate-wide lock: env mutation races concurrent fork/exec in
    // other suites (e.g. child_process spawns) under parallel test runs.
    crate::utils::test_env_lock()
}

/// RAII restore of `HOME`: a failing assertion must not leak the override to
/// unrelated suites (`get_home_dir()` reads `HOME` process-wide). The caller
/// holds [`env_guard`]; this guard only restores the value.
struct HomeOverride {
    previous: Option<String>,
}

impl HomeOverride {
    fn set(value: &str) -> HomeOverride {
        let previous = std::env::var("HOME").ok();
        std::env::set_var("HOME", value);
        HomeOverride { previous }
    }
}

impl Drop for HomeOverride {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(home) => std::env::set_var("HOME", home),
            None => std::env::remove_var("HOME"),
        }
    }
}

fn ends_with_enabled(r: &ResolvedResource, suffix: &str) -> bool {
    r.path.ends_with(suffix) && r.enabled
}

fn ends_with_disabled(r: &ResolvedResource, suffix: &str) -> bool {
    r.path.ends_with(suffix) && !r.enabled
}

fn has_extension(result: &ResolvedPaths, suffix: &str) -> bool {
    result.extensions.iter().any(|r| r.path.ends_with(suffix))
}

// ============================================================================
// resolve
// ============================================================================

#[tokio::test]
async fn no_package_sources_when_none_configured() {
    let _guard = env_guard();
    let (_env, pm) = setup("resolve-empty");
    let result = pm.resolve(None).await.unwrap();
    assert!(result.extensions.is_empty());
    assert!(result.prompts.is_empty());
    assert!(result.themes.is_empty());
    assert!(result
        .skills
        .iter()
        .all(|r| r.metadata.source == "auto" && r.metadata.origin == SourceOrigin::TopLevel));
}

#[tokio::test]
async fn resolves_local_extension_paths_from_settings() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-local-ext");
    let ext_path = Path::new(&env.agent_dir)
        .join("extensions")
        .join("my-extension.ts")
        .to_string_lossy()
        .into_owned();
    write_file(&ext_path, "export default function() {}");
    env.settings
        .lock()
        .set_extension_paths(&["extensions/my-extension.ts".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == ext_path && r.enabled));
}

#[tokio::test]
async fn resolves_builtin_extensions_with_exclusions_and_overrides() {
    let _guard = env_guard();
    let (env, _pm) = setup("resolve-builtins");
    let pm = make_pm(
        &env,
        vec!["mcp".to_string(), "llama.cpp".to_string()],
        Arc::new(ProcessCommandRunner),
    );

    async fn builtins_of(pm: &DefaultPackageManager) -> Vec<(String, bool, String, SourceScope)> {
        let result = pm.resolve(None).await.unwrap();
        result
            .extensions
            .iter()
            .map(|r| {
                (
                    r.path.clone(),
                    r.enabled,
                    r.metadata.source.clone(),
                    r.metadata.scope,
                )
            })
            .collect()
    }

    assert_eq!(
        builtins_of(&pm).await,
        vec![
            (
                "builtin:mcp".to_string(),
                true,
                "builtin".to_string(),
                SourceScope::User
            ),
            (
                "builtin:llama.cpp".to_string(),
                true,
                "builtin".to_string(),
                SourceScope::User
            ),
        ]
    );

    env.settings
        .lock()
        .set_extension_paths(&["-builtin:mcp".to_string()]);
    env.settings.lock().set_project_extension_paths(&[
        "+builtin:mcp".to_string(),
        "-builtin:llama.cpp".to_string(),
    ]);
    assert_eq!(
        builtins_of(&pm).await,
        vec![
            (
                "builtin:mcp".to_string(),
                true,
                "builtin".to_string(),
                SourceScope::Project
            ),
            (
                "builtin:llama.cpp".to_string(),
                false,
                "builtin".to_string(),
                SourceScope::Project
            ),
        ]
    );

    env.settings.lock().set_project_extension_paths(&[]);
    assert_eq!(
        builtins_of(&pm).await,
        vec![
            (
                "builtin:mcp".to_string(),
                false,
                "builtin".to_string(),
                SourceScope::User
            ),
            (
                "builtin:llama.cpp".to_string(),
                true,
                "builtin".to_string(),
                SourceScope::User
            ),
        ]
    );
}

#[tokio::test]
async fn resolves_skill_paths_from_settings() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-skills");
    let skill_file = Path::new(&env.agent_dir)
        .join("skills")
        .join("my-skill")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &skill_file,
        "---\nname: test-skill\ndescription: A test skill\n---\nContent",
    );
    env.settings.lock().set_skill_paths(&["skills".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == skill_file && r.enabled));
}

#[tokio::test]
async fn auto_discovers_root_markdown_skills() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-root-md-skills");
    let skill_file = Path::new(&env.agent_dir)
        .join("skills")
        .join("single-file.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &skill_file,
        "---\nname: single-file\ndescription: A root markdown skill\n---\nContent",
    );

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == skill_file && r.enabled));
}

#[tokio::test]
async fn resolves_project_paths_relative_to_pi() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-project-ext");
    let ext_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("extensions")
        .join("project-ext.ts")
        .to_string_lossy()
        .into_owned();
    write_file(&ext_path, "export default function() {}");
    env.settings
        .lock()
        .set_project_extension_paths(&["extensions/project-ext.ts".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == ext_path && r.enabled));
}

#[tokio::test]
async fn auto_discovers_user_prompts_with_overrides() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-user-prompts");
    let prompt_path = Path::new(&env.agent_dir)
        .join("prompts")
        .join("auto.md")
        .to_string_lossy()
        .into_owned();
    write_file(&prompt_path, "Auto prompt");
    env.settings
        .lock()
        .set_prompt_template_paths(&["!prompts/auto.md".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .prompts
        .iter()
        .any(|r| r.path == prompt_path && !r.enabled));
}

#[tokio::test]
async fn resolves_symlinked_user_and_project_resources_once() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-symlink-dedupe");
    let _home = HomeOverride::set(&env.temp_dir);

    let result = {
        let shared_dir = Path::new(&env.temp_dir).join("shared-resources");
        let shared_extensions_dir = shared_dir.join("extensions");
        let shared_skills_dir = shared_dir.join("skills");
        let shared_prompts_dir = shared_dir.join("prompts");
        let shared_themes_dir = shared_dir.join("themes");
        for dir in [
            &shared_extensions_dir,
            &shared_skills_dir,
            &shared_prompts_dir,
            &shared_themes_dir,
        ] {
            std::fs::create_dir_all(dir).unwrap();
        }

        write_file(
            &shared_extensions_dir.join("shared.ts").to_string_lossy(),
            "export default function() {}",
        );
        write_file(
            &shared_skills_dir
                .join("shared-skill")
                .join("SKILL.md")
                .to_string_lossy(),
            "---\nname: shared-skill\ndescription: Shared skill\n---\nContent",
        );
        write_file(
            &shared_prompts_dir.join("shared.md").to_string_lossy(),
            "Shared prompt",
        );
        write_file(
            &shared_themes_dir.join("shared.json").to_string_lossy(),
            r#"{"name":"shared-theme"}"#,
        );

        std::os::unix::fs::symlink(
            &shared_extensions_dir,
            Path::new(&env.agent_dir).join("extensions"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&shared_skills_dir, Path::new(&env.agent_dir).join("skills"))
            .unwrap();
        std::os::unix::fs::symlink(
            &shared_prompts_dir,
            Path::new(&env.agent_dir).join("prompts"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&shared_themes_dir, Path::new(&env.agent_dir).join("themes"))
            .unwrap();
        let pi_dir = Path::new(&env.temp_dir).join(".pi");
        std::fs::create_dir_all(&pi_dir).unwrap();
        std::os::unix::fs::symlink(&shared_extensions_dir, pi_dir.join("extensions")).unwrap();
        std::os::unix::fs::symlink(&shared_skills_dir, pi_dir.join("skills")).unwrap();
        std::os::unix::fs::symlink(&shared_prompts_dir, pi_dir.join("prompts")).unwrap();
        std::os::unix::fs::symlink(&shared_themes_dir, pi_dir.join("themes")).unwrap();

        pm.resolve(None).await.unwrap()
    };

    assert_eq!(result.extensions.len(), 1);
    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.prompts.len(), 1);
    assert_eq!(result.themes.len(), 1);
    assert_eq!(result.extensions[0].metadata.scope, SourceScope::Project);
    assert_eq!(result.skills[0].metadata.scope, SourceScope::Project);
    assert_eq!(result.prompts[0].metadata.scope, SourceScope::Project);
    assert_eq!(result.themes[0].metadata.scope, SourceScope::Project);
}

#[tokio::test]
async fn auto_discovers_project_prompts_with_overrides() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-project-prompts");
    let prompt_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("prompts")
        .join("is.md")
        .to_string_lossy()
        .into_owned();
    write_file(&prompt_path, "Is prompt");
    env.settings
        .lock()
        .set_project_prompt_template_paths(&["!prompts/is.md".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .prompts
        .iter()
        .any(|r| r.path == prompt_path && !r.enabled));
}

#[tokio::test]
async fn resolves_directory_with_pi_extensions_manifest_in_settings() {
    let _guard = env_guard();
    let (env, pm) = setup("resolve-pi-manifest-dir");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("my-extensions-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"my-extensions-pkg","pi":{"extensions":["./extensions/clip.ts","./extensions/cost.ts"]}}"#,
    );
    write_file(
        &format!("{pkg_dir}/extensions/clip.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/cost.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/helper.ts"),
        "export const x = 1;",
    );

    env.settings.lock().set_extension_paths(&[pkg_dir.clone()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == format!("{pkg_dir}/extensions/clip.ts") && r.enabled));
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == format!("{pkg_dir}/extensions/cost.ts") && r.enabled));
    assert!(!has_extension(&result, "helper.ts"));
}

// ============================================================================
// auto-discovered skill metadata
// ============================================================================

#[tokio::test]
async fn user_pi_skill_basе_dir_is_agent_dir() {
    let _guard = env_guard();
    let (env, pm) = setup("meta-user-pi");
    let skill_path = Path::new(&env.agent_dir)
        .join("skills")
        .join("user-pi")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &skill_path,
        "---\nname: user-pi\ndescription: user pi\n---\n",
    );

    let result = pm.resolve(None).await.unwrap();
    let skill = result.skills.iter().find(|r| r.path == skill_path).unwrap();
    assert_eq!(skill.metadata.source, "auto");
    assert_eq!(skill.metadata.scope, SourceScope::User);
    assert_eq!(
        skill.metadata.base_dir.as_deref(),
        Some(env.agent_dir.as_str())
    );
}

#[tokio::test]
async fn project_pi_skill_base_dir_is_project_pi() {
    let _guard = env_guard();
    let (env, pm) = setup("meta-project-pi");
    let project_base_dir = Path::new(&env.temp_dir)
        .join(".pi")
        .to_string_lossy()
        .into_owned();
    let skill_path = Path::new(&project_base_dir)
        .join("skills")
        .join("project-pi")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &skill_path,
        "---\nname: project-pi\ndescription: project pi\n---\n",
    );

    let result = pm.resolve(None).await.unwrap();
    let skill = result.skills.iter().find(|r| r.path == skill_path).unwrap();
    assert_eq!(skill.metadata.source, "auto");
    assert_eq!(skill.metadata.scope, SourceScope::Project);
    assert_eq!(
        skill.metadata.base_dir.as_deref(),
        Some(project_base_dir.as_str())
    );
}

#[tokio::test]
async fn user_agents_skill_base_dir_is_home_agents() {
    let _guard = env_guard();
    let (env, pm) = setup("meta-user-agents");
    let _home = HomeOverride::set(&env.temp_dir);

    let agents_base_dir = Path::new(&env.temp_dir)
        .join(".agents")
        .to_string_lossy()
        .into_owned();
    let skill_path = Path::new(&agents_base_dir)
        .join("skills")
        .join("user-agents")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &skill_path,
        "---\nname: user-agents\ndescription: user agents\n---\n",
    );

    let result = pm.resolve(None).await.unwrap();

    let skill = result.skills.iter().find(|r| r.path == skill_path).unwrap();
    assert_eq!(skill.metadata.source, "auto");
    assert_eq!(skill.metadata.scope, SourceScope::User);
    assert_eq!(
        skill.metadata.base_dir.as_deref(),
        Some(agents_base_dir.as_str())
    );
}

#[tokio::test]
async fn each_project_agents_dir_is_base_dir() {
    let _guard = env_guard();
    let (env, _pm) = setup("meta-project-agents");
    let repo_root = Path::new(&env.temp_dir).join("repo");
    let nested_cwd = repo_root.join("packages").join("feature");
    std::fs::create_dir_all(&nested_cwd).unwrap();
    std::fs::create_dir_all(repo_root.join(".git")).unwrap();

    let repo_agents_base_dir = repo_root.join(".agents").to_string_lossy().into_owned();
    let repo_skill = Path::new(&repo_agents_base_dir)
        .join("skills")
        .join("repo")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(&repo_skill, "---\nname: repo\ndescription: repo\n---\n");

    let package_agents_base_dir = repo_root
        .join("packages")
        .join(".agents")
        .to_string_lossy()
        .into_owned();
    let package_skill = Path::new(&package_agents_base_dir)
        .join("skills")
        .join("package")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &package_skill,
        "---\nname: package\ndescription: package\n---\n",
    );

    let pm = DefaultPackageManager::new(PackageManagerOptions {
        cwd: nested_cwd.to_string_lossy().into_owned(),
        agent_dir: env.agent_dir.clone(),
        settings_manager: env.settings.clone(),
        builtin_extensions: Vec::new(),
    });

    let result = pm.resolve(None).await.unwrap();
    let resolved_repo_skill = result.skills.iter().find(|r| r.path == repo_skill).unwrap();
    let resolved_package_skill = result
        .skills
        .iter()
        .find(|r| r.path == package_skill)
        .unwrap();
    assert_eq!(resolved_repo_skill.metadata.source, "auto");
    assert_eq!(resolved_repo_skill.metadata.scope, SourceScope::Project);
    assert_eq!(
        resolved_repo_skill.metadata.base_dir.as_deref(),
        Some(repo_agents_base_dir.as_str())
    );
    assert_eq!(resolved_package_skill.metadata.source, "auto");
    assert_eq!(resolved_package_skill.metadata.scope, SourceScope::Project);
    assert_eq!(
        resolved_package_skill.metadata.base_dir.as_deref(),
        Some(package_agents_base_dir.as_str())
    );
}

// ============================================================================
// .agents/skills auto-discovery
// ============================================================================

#[tokio::test]
async fn agents_skills_scan_stops_at_git_root() {
    let _guard = env_guard();
    let (env, _pm) = setup("agents-git-root");
    let repo_root = Path::new(&env.temp_dir).join("repo");
    let nested_cwd = repo_root.join("packages").join("feature");
    std::fs::create_dir_all(&nested_cwd).unwrap();
    std::fs::create_dir_all(repo_root.join(".git")).unwrap();

    let above_repo_skill = Path::new(&env.temp_dir)
        .join(".agents")
        .join("skills")
        .join("above-repo")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &above_repo_skill,
        "---\nname: above-repo\ndescription: above\n---\n",
    );

    let repo_root_skill = repo_root
        .join(".agents")
        .join("skills")
        .join("repo-root")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &repo_root_skill,
        "---\nname: repo-root\ndescription: repo\n---\n",
    );

    let nested_skill = repo_root
        .join("packages")
        .join(".agents")
        .join("skills")
        .join("nested")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &nested_skill,
        "---\nname: nested\ndescription: nested\n---\n",
    );

    let pm = DefaultPackageManager::new(PackageManagerOptions {
        cwd: nested_cwd.to_string_lossy().into_owned(),
        agent_dir: env.agent_dir.clone(),
        settings_manager: env.settings.clone(),
        builtin_extensions: Vec::new(),
    });

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == repo_root_skill && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == nested_skill && r.enabled));
    assert!(!result.skills.iter().any(|r| r.path == above_repo_skill));
}

#[tokio::test]
async fn agents_skills_scan_to_fs_root_without_git() {
    let _guard = env_guard();
    let (env, _pm) = setup("agents-no-git");
    let non_repo_root = Path::new(&env.temp_dir).join("non-repo");
    let nested_cwd = non_repo_root.join("a").join("b");
    std::fs::create_dir_all(&nested_cwd).unwrap();

    let root_skill = non_repo_root
        .join(".agents")
        .join("skills")
        .join("root")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(&root_skill, "---\nname: root\ndescription: root\n---\n");
    let middle_skill = non_repo_root
        .join("a")
        .join(".agents")
        .join("skills")
        .join("middle")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &middle_skill,
        "---\nname: middle\ndescription: middle\n---\n",
    );

    let pm = DefaultPackageManager::new(PackageManagerOptions {
        cwd: nested_cwd.to_string_lossy().into_owned(),
        agent_dir: env.agent_dir.clone(),
        settings_manager: env.settings.clone(),
        builtin_extensions: Vec::new(),
    });

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == root_skill && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == middle_skill && r.enabled));
}

#[tokio::test]
async fn agents_skills_ignore_root_markdown_but_find_nested() {
    let _guard = env_guard();
    let (env, _pm) = setup("agents-md-modes");
    let agents_skills_dir = Path::new(&env.temp_dir)
        .join(".agents")
        .join("skills")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(Path::new(&agents_skills_dir).join("nested-skill")).unwrap();
    std::fs::create_dir_all(
        Path::new(&agents_skills_dir)
            .join("third-party")
            .join("vendor")
            .join("pack"),
    )
    .unwrap();
    let root_skill = Path::new(&agents_skills_dir)
        .join("root-file.md")
        .to_string_lossy()
        .into_owned();
    let nested_skill = Path::new(&agents_skills_dir)
        .join("nested-skill")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    let nested_markdown_skill = Path::new(&agents_skills_dir)
        .join("third-party")
        .join("child-skill.md")
        .to_string_lossy()
        .into_owned();
    let deeply_nested_markdown_skill = Path::new(&agents_skills_dir)
        .join("third-party")
        .join("vendor")
        .join("pack")
        .join("deep-skill.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &root_skill,
        "---\nname: root-file\ndescription: Root markdown file\n---\n",
    );
    write_file(
        &nested_skill,
        "---\nname: nested-skill\ndescription: Nested skill\n---\n",
    );
    write_file(
        &nested_markdown_skill,
        "---\nname: child-skill\ndescription: Nested markdown skill\n---\n",
    );
    write_file(
        &deeply_nested_markdown_skill,
        "---\nname: deep-skill\ndescription: Deep markdown skill\n---\n",
    );

    std::fs::create_dir_all(Path::new(&env.temp_dir).join("work")).unwrap();
    let pm = DefaultPackageManager::new(PackageManagerOptions {
        cwd: Path::new(&env.temp_dir)
            .join("work")
            .to_string_lossy()
            .into_owned(),
        agent_dir: env.agent_dir.clone(),
        settings_manager: env.settings.clone(),
        builtin_extensions: Vec::new(),
    });

    let result = pm.resolve(None).await.unwrap();
    assert!(!result.skills.iter().any(|r| r.path == root_skill));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == nested_skill && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == nested_markdown_skill && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == deeply_nested_markdown_skill && r.enabled));
}

#[tokio::test]
async fn home_agents_skills_stay_user_scoped() {
    let _guard = env_guard();
    let (env, _pm) = setup("agents-home-scope");
    let _home = HomeOverride::set(&env.temp_dir);

    let cwd = Path::new(&env.temp_dir).join("scratch").join("nested");
    let local_agent_dir = Path::new(&env.temp_dir).join(".pi").join("agent");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&local_agent_dir).unwrap();
    let local_settings = in_memory_manager();

    let home_skill = Path::new(&env.temp_dir)
        .join(".agents")
        .join("skills")
        .join("home-skill")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &home_skill,
        "---\nname: home-skill\ndescription: home\n---\n",
    );

    let pm = DefaultPackageManager::new(PackageManagerOptions {
        cwd: cwd.to_string_lossy().into_owned(),
        agent_dir: local_agent_dir.to_string_lossy().into_owned(),
        settings_manager: local_settings,
        builtin_extensions: Vec::new(),
    });

    let result = pm.resolve(None).await.unwrap();

    let matching: Vec<_> = result
        .skills
        .iter()
        .filter(|r| r.path == home_skill)
        .collect();
    assert_eq!(matching.len(), 1);
    assert!(matching[0].enabled);
    assert_eq!(matching[0].metadata.scope, SourceScope::User);
    assert_eq!(matching[0].metadata.source, "auto");
}

#[tokio::test]
async fn dedupes_user_skills_when_agent_skills_symlink() {
    let _guard = env_guard();
    let (env, pm) = setup("agents-symlink-dedupe");
    let _home = HomeOverride::set(&env.temp_dir);

    let agent_skills_dir = Path::new(&env.agent_dir).join("skills");
    let agents_skills_dir = Path::new(&env.temp_dir).join(".agents").join("skills");
    std::fs::create_dir_all(&agents_skills_dir).unwrap();
    std::os::unix::fs::symlink(&agents_skills_dir, &agent_skills_dir).unwrap();

    let skill_path = agents_skills_dir
        .join("foo")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(&skill_path, "---\nname: foo\ndescription: foo\n---\n");

    let result = pm.resolve(None).await.unwrap();

    let foo_skills: Vec<_> = result
        .skills
        .iter()
        .filter(|r| r.path.ends_with("foo/SKILL.md"))
        .collect();
    assert_eq!(foo_skills.len(), 1);
}

// ============================================================================
// ignore files
// ============================================================================

#[tokio::test]
async fn respects_gitignore_in_skill_directories() {
    let _guard = env_guard();
    let (env, pm) = setup("ignore-gitignore");
    // Isolate from the real $HOME (~/.agents/skills could contain matching names).
    let fake_home = Path::new(&env.temp_dir).join("home");
    std::fs::create_dir_all(&fake_home).unwrap();
    let _home = HomeOverride::set(fake_home.to_str().unwrap());
    let skills_dir = Path::new(&env.agent_dir)
        .join("skills")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{skills_dir}/.gitignore"), "venv\n__pycache__\n");
    write_file(
        &format!("{skills_dir}/good-skill/SKILL.md"),
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    );
    write_file(
        &format!("{skills_dir}/venv/bad-skill/SKILL.md"),
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    );
    env.settings.lock().set_skill_paths(&["skills".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("good-skill") && r.enabled));
    assert!(!result
        .skills
        .iter()
        .any(|r| r.path.contains("venv") && r.enabled));
}

#[tokio::test]
async fn parent_gitignore_not_applied_to_pi_autodiscovery() {
    let _guard = env_guard();
    let (env, pm) = setup("ignore-parent");
    write_file(
        &Path::new(&env.temp_dir)
            .join(".gitignore")
            .to_string_lossy(),
        ".pi\n",
    );

    let skill_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("skills")
        .join("auto-skill")
        .join("SKILL.md")
        .to_string_lossy()
        .into_owned();
    write_file(
        &skill_path,
        "---\nname: auto-skill\ndescription: Auto\n---\nContent",
    );

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == skill_path && r.enabled));
}

// ============================================================================
// resolveExtensionSources
// ============================================================================

#[tokio::test]
async fn resolve_extension_sources_local_paths() {
    let _guard = env_guard();
    let (env, pm) = setup("res-local");
    let ext_path = Path::new(&env.temp_dir)
        .join("ext.ts")
        .to_string_lossy()
        .into_owned();
    write_file(&ext_path, "export default function() {}");

    let result = pm
        .resolve_extension_sources(&[ext_path.clone()], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == ext_path && r.enabled));
}

#[tokio::test]
async fn resolve_extension_sources_pi_manifest_directory() {
    let _guard = env_guard();
    let (env, pm) = setup("res-manifest");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("my-package")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"my-package","pi":{"extensions":["./src/index.ts"],"skills":["./skills"]}}"#,
    );
    write_file(
        &format!("{pkg_dir}/src/index.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/skills/my-skill/SKILL.md"),
        "---\nname: my-skill\ndescription: Test\n---\nContent",
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir.clone()], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == format!("{pkg_dir}/src/index.ts") && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.ends_with("skills/my-skill/SKILL.md") && r.enabled));
}

#[tokio::test]
async fn keeps_pi_manifest_entries_with_leading_tilde_package_relative() {
    let _guard = env_guard();
    let (env, pm) = setup("res-tilde");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("tilde-manifest-package")
        .to_string_lossy()
        .into_owned();
    let direct_extension_path = format!("{pkg_dir}/~extensions/main.ts");
    let slash_extension_path = format!("{pkg_dir}/~/extensions/alt.ts");
    let direct_skill_path = format!("{pkg_dir}/~skills/direct-skill/SKILL.md");
    let slash_skill_path = format!("{pkg_dir}/~/skills/slash-skill/SKILL.md");

    write_file(&direct_extension_path, "export default function() {}");
    write_file(&slash_extension_path, "export default function() {}");
    write_file(
        &direct_skill_path,
        "---\nname: direct-skill\ndescription: Direct\n---\nContent",
    );
    write_file(
        &slash_skill_path,
        "---\nname: slash-skill\ndescription: Slash\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"tilde-manifest-package","pi":{"extensions":["~extensions/main.ts","~/extensions/alt.ts"],"skills":["~skills","~/skills"]}}"#,
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == direct_extension_path && r.enabled));
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == slash_extension_path && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == direct_skill_path && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == slash_skill_path && r.enabled));
}

#[tokio::test]
async fn resolve_extension_sources_auto_discovery_layout() {
    let _guard = env_guard();
    let (env, pm) = setup("res-auto");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("auto-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/main.ts"),
        "export default function() {}",
    );
    write_file(&format!("{pkg_dir}/themes/dark.json"), "{}");

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("main.ts") && r.enabled));
    assert!(result
        .themes
        .iter()
        .any(|r| r.path.ends_with("dark.json") && r.enabled));
}

#[tokio::test]
async fn stops_recursing_when_skill_dir_has_skill_md() {
    let _guard = env_guard();
    let (env, pm) = setup("res-skill-root");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("skill-root-pkg")
        .to_string_lossy()
        .into_owned();
    let root_skill = format!("{pkg_dir}/skills/root-skill/SKILL.md");
    let nested_skill = format!("{pkg_dir}/skills/root-skill/nested-skill/SKILL.md");
    write_file(
        &root_skill,
        "---\nname: root-skill\ndescription: Root skill\n---\n",
    );
    write_file(
        &nested_skill,
        "---\nname: nested-skill\ndescription: Nested skill\n---\n",
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path == root_skill && r.enabled));
    assert!(!result.skills.iter().any(|r| r.path == nested_skill));
}

// ============================================================================
// progress callback + command spawning
// ============================================================================

#[tokio::test]
async fn progress_callback_set_without_events() {
    let _guard = env_guard();
    let (env, pm) = setup("progress-local");
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = events.clone();
    pm.set_progress_callback(Some(Arc::new(move |event: &ProgressEvent| {
        sink.lock().push(event.clone())
    })));

    let ext_path = Path::new(&env.temp_dir)
        .join("ext.ts")
        .to_string_lossy()
        .into_owned();
    write_file(&ext_path, "export default function() {}");
    pm.resolve_extension_sources(&[ext_path], false, false)
        .await
        .unwrap();
    assert_eq!(events.lock().len(), 0);
}

#[tokio::test]
async fn preserves_argv_entries_containing_spaces() {
    let _guard = env_guard();
    let (_env, pm) = setup("spawn-spaces");
    let value_with_space = "C:\\Users\\A B\\.pi\\npm";
    let output = pm.runner.run_sync(&CommandCall::new(
        "sh",
        &[
            "-c".to_string(),
            "printf %s \"$1\"".to_string(),
            "sh".to_string(),
            value_with_space.to_string(),
        ],
    ));
    assert_eq!(output.unwrap(), value_with_space);
}

// ============================================================================
// npmCommand
// ============================================================================

#[tokio::test]
async fn uses_npm_command_argv_for_installs() {
    let _guard = env_guard();
    let (env, _pm) = setup("npmcmd-install");
    env.settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "node@20".to_string(),
        "--".to_string(),
        "npm".to_string(),
    ]));
    let runner = mock_runner(|_| Ok(String::new()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("npm:@scope/pkg", false).await.unwrap();

    let calls = runner.calls_for("mise");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0],
        (
            vec![
                "exec".to_string(),
                "node@20".to_string(),
                "--".to_string(),
                "npm".to_string(),
                "install".to_string(),
                "@scope/pkg".to_string(),
                "--prefix".to_string(),
                Path::new(&env.agent_dir)
                    .join("npm")
                    .to_string_lossy()
                    .into_owned(),
                "--legacy-peer-deps".to_string(),
            ],
            None
        )
    );
}

#[tokio::test]
async fn passes_legacy_peer_deps_when_uninstalling() {
    let _guard = env_guard();
    let (env, _pm) = setup("npmcmd-uninstall");
    std::fs::create_dir_all(Path::new(&env.agent_dir).join("npm")).unwrap();
    let runner = mock_runner(|_| Ok(String::new()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.remove("npm:@scope/pkg", false).await.unwrap();

    let calls = runner.calls_for("npm");
    assert_eq!(
        calls[0],
        (
            vec![
                "uninstall".to_string(),
                "@scope/pkg".to_string(),
                "--prefix".to_string(),
                Path::new(&env.agent_dir)
                    .join("npm")
                    .to_string_lossy()
                    .into_owned(),
                "--legacy-peer-deps".to_string(),
            ],
            None
        )
    );
}

#[tokio::test]
async fn uses_bun_cwd_for_installs() {
    let _guard = env_guard();
    let (env, _pm) = setup("npmcmd-bun");
    env.settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "bun@1".to_string(),
        "--".to_string(),
        "bun".to_string(),
    ]));
    let runner = mock_runner(|_| Ok(String::new()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("npm:@scope/pkg", false).await.unwrap();

    let calls = runner.calls_for("mise");
    assert_eq!(
        calls[0],
        (
            vec![
                "exec".to_string(),
                "bun@1".to_string(),
                "--".to_string(),
                "bun".to_string(),
                "install".to_string(),
                "@scope/pkg".to_string(),
                "--cwd".to_string(),
                Path::new(&env.agent_dir)
                    .join("npm")
                    .to_string_lossy()
                    .into_owned(),
                "--omit=peer".to_string(),
            ],
            None
        )
    );
}

async fn clone_failure_test(name: &str, fail_dependency_install: bool, expected_error: &str) {
    let (mut env, _pm) = setup(name);
    let agent_dir = env.agent_dir.clone();
    let runner = mock_runner(move |call| {
        if call.command == "git" && call.args.first().map(String::as_str) == Some("clone") {
            let target_dir = Path::new(&agent_dir)
                .join("git")
                .join("github.com")
                .join("user")
                .join("repo");
            std::fs::create_dir_all(&target_dir).unwrap();
            if !fail_dependency_install {
                write_file(
                    &target_dir.join("package.json").to_string_lossy(),
                    r#"{"name":"repo","version":"1.0.0"}"#,
                );
                return Err("simulated git clone failure".to_string());
            }
            write_file(
                &target_dir.join("package.json").to_string_lossy(),
                r#"{"name":"repo","version":"1.0.0"}"#,
            );
            return Ok(String::new());
        }
        if fail_dependency_install && call.command == "npm" {
            Err("simulated dependency install failure".to_string())
        } else {
            Ok(String::new())
        }
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let error = pm
        .install("git:github.com/user/repo", false)
        .await
        .unwrap_err();
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo");
    assert!(error.contains(expected_error), "got: {error}");
    assert!(!target_dir.exists());
}

#[tokio::test]
async fn removes_checkout_when_git_clone_fails() {
    let _guard = env_guard();
    clone_failure_test("clone-fail", false, "simulated git clone failure").await;
}

#[tokio::test]
async fn removes_checkout_when_dependency_install_fails() {
    let _guard = env_guard();
    clone_failure_test("dep-fail", true, "simulated dependency install failure").await;
}

#[tokio::test]
async fn installs_git_dependencies_with_default_npm_args() {
    let _guard = env_guard();
    let (env, _pm) = setup("gitdep-npm");
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    let target_dir_clone = target_dir.clone();
    let runner = mock_runner(move |call| {
        if call.command == "git" && call.args.first().map(String::as_str) == Some("clone") {
            std::fs::create_dir_all(&target_dir_clone).unwrap();
            write_file(
                &format!("{target_dir_clone}/package.json"),
                r#"{"name":"repo","version":"1.0.0"}"#,
            );
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("git:github.com/user/repo", false).await.unwrap();

    assert_eq!(
        runner.calls_for("npm")[0],
        (
            vec![
                "install".to_string(),
                "--omit=dev".to_string(),
                "--legacy-peer-deps".to_string()
            ],
            Some(target_dir.clone())
        )
    );
}

#[tokio::test]
async fn reconciles_existing_checkout_to_pinned_ref_during_install() {
    let _guard = env_guard();
    let (env, _pm) = setup("git-ref-reconcile");
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{target_dir}/package.json"),
        r#"{"name":"repo","version":"1.0.0"}"#,
    );

    let runner = mock_runner(move |call| {
        let args: Vec<&str> = call.args.iter().map(String::as_str).collect();
        if call.command == "git" && args.first() == Some(&"rev-parse") {
            if args.get(1) == Some(&"HEAD") {
                return Ok("old-head".to_string());
            }
            if args.get(1) == Some(&"FETCH_HEAD^{commit}") {
                return Ok("new-head".to_string());
            }
            return Err(format!(
                "Unexpected runCommandCapture args: {}",
                call.args.join(" ")
            ));
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("git:github.com/user/repo@v2", false)
        .await
        .unwrap();

    let git_calls = runner.calls_for("git");
    assert!(git_calls.contains(&(
        vec!["fetch".to_string(), "origin".to_string(), "v2".to_string()],
        Some(target_dir.clone())
    )));
    assert!(git_calls.contains(&(
        vec![
            "reset".to_string(),
            "--hard".to_string(),
            "FETCH_HEAD^{commit}".to_string()
        ],
        Some(target_dir.clone())
    )));
    assert!(git_calls.contains(&(
        vec!["clean".to_string(), "-fdx".to_string()],
        Some(target_dir.clone())
    )));
    assert!(runner.calls_for("npm").contains(&(
        vec![
            "install".to_string(),
            "--omit=dev".to_string(),
            "--legacy-peer-deps".to_string()
        ],
        Some(target_dir)
    )));
}

#[tokio::test]
async fn reconciles_existing_checkout_to_update_target_when_no_ref() {
    let _guard = env_guard();
    let (env, _pm) = setup("git-update-target");
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo");
    std::fs::create_dir_all(&target_dir).unwrap();

    let runner = mock_runner(move |call| {
        let args: Vec<&str> = call.args.iter().map(String::as_str).collect();
        if call.command == "git" && args.first() == Some(&"rev-parse") {
            if args.get(1) == Some(&"HEAD") {
                return Ok("old-head".to_string());
            }
            if args.get(1) == Some(&"origin/HEAD") {
                return Ok("new-head".to_string());
            }
            if args.get(1) == Some(&"origin/HEAD^{commit}") {
                return Ok("new-head".to_string());
            }
            return Err(format!(
                "Unexpected runCommandCapture args: {}",
                call.args.join(" ")
            ));
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("git:github.com/user/repo", false).await.unwrap();

    let target_dir_str = target_dir.to_string_lossy().into_owned();
    let git_calls = runner.calls_for("git");
    // Upstream mocks getLocalGitUpdateTarget; here the fallback branch
    // (symbolic-ref lookup) determines the fetch args. The abbreviated
    // upstream lookup fails, so fetch targets +HEAD:refs/remotes/origin/HEAD.
    assert!(git_calls.iter().any(|(args, cwd)| {
        args.first().map(String::as_str) == Some("fetch")
            && args.contains(&"+HEAD:refs/remotes/origin/HEAD".to_string())
            || args.contains(&"+refs/heads/main:refs/remotes/origin/main".to_string())
    } && cwd.as_deref()
        == Some(target_dir_str.as_str())));
    assert!(git_calls.contains(&(
        vec![
            "reset".to_string(),
            "--hard".to_string(),
            "origin/HEAD^{commit}".to_string()
        ],
        Some(target_dir_str.clone())
    )));
    assert!(git_calls.contains(&(
        vec!["clean".to_string(), "-fdx".to_string()],
        Some(target_dir_str)
    )));
}

#[tokio::test]
async fn prefers_package_manager_after_separator() {
    let _guard = env_guard();
    let (env, _pm) = setup("pm-separator");
    // Regression for #9863.
    env.settings.lock().set_npm_command(Some(&[
        "npm".to_string(),
        "exec".to_string(),
        "--".to_string(),
        "pnpm".to_string(),
    ]));
    let pm = make_pm(&env, Vec::new(), failing_runner());

    assert_eq!(pm.get_package_manager_name(), "pnpm");
    assert_eq!(
        pm.get_git_dependency_install_args(),
        vec![
            "install".to_string(),
            "--prod".to_string(),
            "--config.auto-install-peers=false".to_string(),
            "--config.strict-peer-dependencies=false".to_string(),
            "--config.strict-dep-builds=false".to_string(),
        ]
    );
}

#[tokio::test]
async fn detects_pnpm_through_corepack_wrapper() {
    let _guard = env_guard();
    let (env, _pm) = setup("pm-corepack");
    // Regression for #9863.
    env.settings
        .lock()
        .set_npm_command(Some(&["corepack".to_string(), "pnpm".to_string()]));
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    let target_dir_clone = target_dir.clone();
    let runner = mock_runner(move |call| {
        if call.command == "git" && call.args.first().map(String::as_str) == Some("clone") {
            std::fs::create_dir_all(&target_dir_clone).unwrap();
            write_file(
                &format!("{target_dir_clone}/package.json"),
                r#"{"name":"repo","version":"1.0.0"}"#,
            );
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("git:github.com/user/repo", false).await.unwrap();

    assert_eq!(
        runner.calls_for("corepack")[0],
        (
            vec![
                "pnpm".to_string(),
                "install".to_string(),
                "--prod".to_string(),
                "--config.auto-install-peers=false".to_string(),
                "--config.strict-peer-dependencies=false".to_string(),
                "--config.strict-dep-builds=false".to_string(),
            ],
            Some(target_dir)
        )
    );
}

#[tokio::test]
async fn disables_peer_installation_with_bun() {
    let _guard = env_guard();
    let (env, _pm) = setup("pm-bun-deps");
    env.settings
        .lock()
        .set_npm_command(Some(&["bun".to_string()]));
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    let target_dir_clone = target_dir.clone();
    let runner = mock_runner(move |call| {
        if call.command == "git" && call.args.first().map(String::as_str) == Some("clone") {
            std::fs::create_dir_all(&target_dir_clone).unwrap();
            write_file(
                &format!("{target_dir_clone}/package.json"),
                r#"{"name":"repo","version":"1.0.0"}"#,
            );
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.install("git:github.com/user/repo", false).await.unwrap();

    assert_eq!(
        runner.calls_for("bun")[0],
        (
            vec![
                "install".to_string(),
                "--omit=dev".to_string(),
                "--omit=peer".to_string()
            ],
            Some(target_dir)
        )
    );
}

#[tokio::test]
async fn updates_git_dependencies_with_npm_args() {
    let _guard = env_guard();
    let (env, _pm) = setup("gitdep-update");
    let target_dir = Path::new(&env.temp_dir)
        .join(".pi")
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{target_dir}/package.json"),
        r#"{"name":"repo","version":"1.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple(
            "git:github.com/user/repo".to_string(),
        )]);

    let runner = mock_runner(move |call| {
        let args: Vec<&str> = call.args.iter().map(String::as_str).collect();
        if call.command == "git" && args.first() == Some(&"rev-parse") {
            if args.get(1) == Some(&"--abbrev-ref") && args.get(2) == Some(&"@{upstream}") {
                return Ok("origin/main".to_string());
            }
            if args.get(1) == Some(&"@{upstream}") || args.get(1) == Some(&"@{upstream}^{commit}") {
                return Ok("remote-head".to_string());
            }
            if args.get(1) == Some(&"HEAD") {
                return Ok("local-head".to_string());
            }
            return Err(format!(
                "Unexpected runCommandCapture args: {}",
                call.args.join(" ")
            ));
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(Some("git:github.com/user/repo")).await.unwrap();

    assert!(runner.calls_for("npm").contains(&(
        vec![
            "install".to_string(),
            "--omit=dev".to_string(),
            "--legacy-peer-deps".to_string()
        ],
        Some(target_dir)
    )));
}

#[tokio::test]
async fn repairs_missing_git_dependencies_when_current() {
    let _guard = env_guard();
    let (env, _pm) = setup("gitdep-repair-current");
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{target_dir}/package.json"),
        r#"{"name":"repo","version":"1.0.0","dependencies":{"dependency":"1.0.0"}}"#,
    );
    env.settings.lock().set_packages(&[PackageSource::Simple(
        "git:github.com/user/repo".to_string(),
    )]);

    let runner = mock_runner(move |call| {
        let args: Vec<&str> = call.args.iter().map(String::as_str).collect();
        if call.command == "git" && args.first() == Some(&"rev-parse") {
            return Ok("current-head".to_string());
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(Some("git:github.com/user/repo")).await.unwrap();

    assert!(runner.calls_for("npm").contains(&(
        vec![
            "install".to_string(),
            "--omit=dev".to_string(),
            "--legacy-peer-deps".to_string()
        ],
        Some(target_dir)
    )));
    assert!(!runner
        .calls_for("git")
        .contains(&(vec!["clean".to_string(), "-fdx".to_string()], None)));
}

#[tokio::test]
async fn repairs_deleted_git_dependencies_when_cleaning_fails() {
    let _guard = env_guard();
    let (env, _pm) = setup("gitdep-clean-fail");
    let target_dir = Path::new(&env.agent_dir)
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{target_dir}/package.json"),
        r#"{"name":"repo","version":"1.0.0","dependencies":{"dependency":"1.0.0"}}"#,
    );
    env.settings.lock().set_packages(&[PackageSource::Simple(
        "git:github.com/user/repo".to_string(),
    )]);

    let runner = mock_runner(move |call| {
        let args: Vec<&str> = call.args.iter().map(String::as_str).collect();
        if call.command == "git" && args.first() == Some(&"rev-parse") {
            return Ok(if args.get(1) == Some(&"HEAD") {
                "old-head".to_string()
            } else {
                "new-head".to_string()
            });
        }
        if call.command == "git" && args.first() == Some(&"clean") {
            return Err("simulated clean failure".to_string());
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let error = pm
        .update(Some("git:github.com/user/repo"))
        .await
        .unwrap_err();
    assert!(error.contains("simulated clean failure"));

    assert!(runner.calls_for("npm").contains(&(
        vec![
            "install".to_string(),
            "--omit=dev".to_string(),
            "--legacy-peer-deps".to_string()
        ],
        Some(target_dir)
    )));
}

#[tokio::test]
async fn disables_peer_installation_through_wrapped_pnpm_on_update() {
    let _guard = env_guard();
    let (env, _pm) = setup("gitdep-pnpm-update");
    env.settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "node@20".to_string(),
        "--".to_string(),
        "pnpm".to_string(),
    ]));
    let target_dir = Path::new(&env.temp_dir)
        .join(".pi")
        .join("git")
        .join("github.com")
        .join("user")
        .join("repo")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{target_dir}/package.json"),
        r#"{"name":"repo","version":"1.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple(
            "git:github.com/user/repo".to_string(),
        )]);

    let runner = mock_runner(move |call| {
        let args: Vec<&str> = call.args.iter().map(String::as_str).collect();
        if call.command == "git" && args.first() == Some(&"rev-parse") {
            if args.get(1) == Some(&"--abbrev-ref") && args.get(2) == Some(&"@{upstream}") {
                return Ok("origin/main".to_string());
            }
            if args.get(1) == Some(&"@{upstream}") || args.get(1) == Some(&"@{upstream}^{commit}") {
                return Ok("remote-head".to_string());
            }
            if args.get(1) == Some(&"HEAD") {
                return Ok("local-head".to_string());
            }
            return Err(format!(
                "Unexpected runCommandCapture args: {}",
                call.args.join(" ")
            ));
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(Some("git:github.com/user/repo")).await.unwrap();

    assert_eq!(
        runner.calls_for("mise")[0],
        (
            vec![
                "exec".to_string(),
                "node@20".to_string(),
                "--".to_string(),
                "pnpm".to_string(),
                "install".to_string(),
                "--prod".to_string(),
                "--config.auto-install-peers=false".to_string(),
                "--config.strict-peer-dependencies=false".to_string(),
                "--config.strict-dep-builds=false".to_string(),
            ],
            Some(target_dir)
        )
    );
}

#[test]
fn uses_npm_command_argv_for_root_lookup_and_invalidates_cache() {
    let _guard = env_guard();
    let temp_dir = fresh_dir("npm-root-cache");
    let agent_dir = Path::new(&temp_dir)
        .join("agent")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let settings = in_memory_manager();
    settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "node@20".to_string(),
        "--".to_string(),
        "npm".to_string(),
    ]));

    let root20 = Path::new(&temp_dir)
        .join("node20")
        .join("lib")
        .join("node_modules")
        .to_string_lossy()
        .into_owned();
    let root22 = Path::new(&temp_dir)
        .join("node22")
        .join("lib")
        .join("node_modules")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(Path::new(&root20).join("@scope").join("pkg")).unwrap();

    let root20_clone = root20.clone();
    let root22_clone = root22.clone();
    let runner = mock_runner(move |call| {
        assert_eq!(call.command, "mise");
        if call.args.get(1).map(String::as_str) == Some("node@20") {
            return Ok(root20_clone.clone());
        }
        if call.args.get(1).map(String::as_str) == Some("node@22") {
            return Ok(root22_clone.clone());
        }
        Err(format!("unexpected args {}", call.args.join(" ")))
    });
    let pm = make_pm(
        &TestEnv {
            temp_dir: temp_dir.clone(),
            agent_dir,
            settings: settings.clone(),
        },
        Vec::new(),
        runner.clone(),
    );

    assert_eq!(
        pm.get_installed_path("npm:@scope/pkg", SourceScope::User),
        Some(
            Path::new(&root20)
                .join("@scope")
                .join("pkg")
                .to_string_lossy()
                .into_owned()
        )
    );
    assert_eq!(
        runner.calls_for("mise")[0].0,
        vec![
            "exec".to_string(),
            "node@20".to_string(),
            "--".to_string(),
            "npm".to_string(),
            "root".to_string(),
            "-g".to_string()
        ]
    );

    settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "node@22".to_string(),
        "--".to_string(),
        "npm".to_string(),
    ]));
    assert_eq!(
        pm.get_installed_path("npm:@scope/pkg", SourceScope::User),
        None
    );
    assert_eq!(
        runner.calls_for("mise")[1].0,
        vec![
            "exec".to_string(),
            "node@22".to_string(),
            "--".to_string(),
            "npm".to_string(),
            "root".to_string(),
            "-g".to_string()
        ]
    );
    let _ = temp_dir;
}

#[tokio::test]
async fn installs_user_npm_packages_into_managed_root() {
    let _guard = env_guard();
    let (env, _pm) = setup("npm-managed-root");
    env.settings
        .lock()
        .set_npm_command(Some(&["pnpm".to_string()]));
    env.settings
        .lock()
        .set_packages(&[PackageSource::Simple("npm:pnpm-pkg".to_string())]);

    let package_path = Path::new(&env.agent_dir)
        .join("npm")
        .join("node_modules")
        .join("pnpm-pkg")
        .to_string_lossy()
        .into_owned();
    let package_path_clone = package_path.clone();
    let agent_npm_root = Path::new(&env.agent_dir)
        .join("npm")
        .to_string_lossy()
        .into_owned();
    let agent_npm_root_clone = agent_npm_root.clone();
    let combined = ChainedRunner {
        install: mock_runner(move |call| {
            assert_eq!(call.command, "pnpm");
            assert_eq!(
                call.args,
                vec![
                    "install".to_string(),
                    "pnpm-pkg".to_string(),
                    "--prefix".to_string(),
                    agent_npm_root_clone.clone(),
                    "--config.auto-install-peers=false".to_string(),
                    "--config.strict-peer-dependencies=false".to_string(),
                    "--config.strict-dep-builds=false".to_string(),
                ]
            );
            std::fs::create_dir_all(Path::new(&package_path_clone).join("extensions")).unwrap();
            write_file(
                &format!("{package_path_clone}/package.json"),
                r#"{"name":"pnpm-pkg","version":"1.0.0"}"#,
            );
            write_file(
                &format!("{package_path_clone}/extensions/index.ts"),
                "export default function() {};",
            );
            Ok(String::new())
        }),
        fallback: mock_runner(|_| Err("legacy lookup unavailable".to_string())),
    };
    let pm = make_pm(&env, Vec::new(), Arc::new(combined));

    let first = pm.resolve(None).await.unwrap();
    let second = pm.resolve(None).await.unwrap();

    assert!(first
        .extensions
        .iter()
        .any(|r| r.path == format!("{package_path}/extensions/index.ts") && r.enabled));
    assert!(second
        .extensions
        .iter()
        .any(|r| r.path == format!("{package_path}/extensions/index.ts") && r.enabled));
    assert_eq!(
        pm.get_installed_path("npm:pnpm-pkg", SourceScope::User),
        Some(package_path)
    );
}

struct ChainedRunner {
    install: Arc<MockRunner>,
    fallback: Arc<MockRunner>,
}

impl ChainedRunner {
    fn route(&self, call: &CommandCall) -> &Arc<MockRunner> {
        let is_npm_install = (call.command == "npm" || call.command == "pnpm")
            && call.args.first().map(String::as_str) == Some("install");
        if is_npm_install {
            &self.install
        } else {
            &self.fallback
        }
    }
}

impl CommandRunner for ChainedRunner {
    fn run(&self, call: &CommandCall) -> Result<(), String> {
        self.route(call).run(call)
    }

    fn run_capture(&self, call: &CommandCall, timeout_ms: Option<u64>) -> Result<String, String> {
        self.route(call).run_capture(call, timeout_ms)
    }

    fn run_sync(&self, call: &CommandCall) -> Result<String, String> {
        self.route(call).run_sync(call)
    }
}

#[tokio::test]
async fn loads_legacy_pnpm_global_paths_from_list_output() {
    let _guard = env_guard();
    let (env, _pm) = setup("npm-pnpm-legacy");
    env.settings
        .lock()
        .set_npm_command(Some(&["pnpm".to_string()]));
    env.settings
        .lock()
        .set_packages(&[PackageSource::Simple("npm:pnpm-pkg".to_string())]);

    let pnpm_root = Path::new(&env.temp_dir)
        .join("pnpm")
        .join("global")
        .join("v11")
        .to_string_lossy()
        .into_owned();
    let package_path = Path::new(&pnpm_root)
        .join("20-hash")
        .join("node_modules")
        .join("pnpm-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{package_path}/package.json"),
        r#"{"name":"pnpm-pkg","version":"1.0.0"}"#,
    );
    write_file(
        &format!("{package_path}/extensions/index.ts"),
        "export default function() {};",
    );
    std::fs::create_dir_all(Path::new(&package_path).join("extensions")).unwrap();

    let package_path_clone = package_path.clone();
    let runner = mock_runner(move |call| {
        assert_eq!(call.command, "pnpm");
        if call.args.join(" ") == "list -g --depth 0 --json" {
            return Ok(format!(
                r#"[{{"path":"{pnpm_root}","dependencies":{{"pnpm-pkg":{{"version":"1.0.0","path":"{package_path_clone}"}}}}}}]"#
            ));
        }
        Err(format!("unexpected args {}", call.args.join(" ")))
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let result = pm.resolve(None).await.unwrap();

    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == format!("{package_path}/extensions/index.ts") && r.enabled));
    assert!(runner.calls().iter().all(|call| call.command == "pnpm"));
    assert_eq!(
        pm.get_installed_path("npm:pnpm-pkg", SourceScope::User),
        Some(package_path)
    );
}

#[test]
fn resolves_wrapped_pnpm_global_paths() {
    let _guard = env_guard();
    let temp_dir = fresh_dir("npm-pnpm-wrapped");
    let agent_dir = Path::new(&temp_dir)
        .join("agent")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let settings = in_memory_manager();
    settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "node@20".to_string(),
        "--".to_string(),
        "pnpm".to_string(),
    ]));

    let pnpm_root = Path::new(&temp_dir)
        .join("pnpm")
        .join("global")
        .join("v11")
        .to_string_lossy()
        .into_owned();
    let package_path = Path::new(&pnpm_root)
        .join("20-hash")
        .join("node_modules")
        .join("pnpm-pkg")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&package_path).unwrap();

    let package_path_clone = package_path.clone();
    let runner = mock_runner(move |call| {
        assert_eq!(call.command, "mise");
        if call.args.join(" ") == "exec node@20 -- pnpm list -g --depth 0 --json" {
            return Ok(format!(
                r#"[{{"path":"{pnpm_root}","dependencies":{{"pnpm-pkg":{{"path":"{package_path_clone}"}}}}}}]"#
            ));
        }
        Err(format!("unexpected args {}", call.args.join(" ")))
    });
    let pm = make_pm(
        &TestEnv {
            temp_dir,
            agent_dir,
            settings,
        },
        Vec::new(),
        runner,
    );

    assert_eq!(
        pm.get_installed_path("npm:pnpm-pkg", SourceScope::User),
        Some(package_path)
    );
}

#[test]
fn ignores_malformed_legacy_pnpm_lists() {
    let _guard = env_guard();
    let temp_dir = fresh_dir("npm-pnpm-bad");
    let agent_dir = Path::new(&temp_dir)
        .join("agent")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let settings = in_memory_manager();
    settings.lock().set_npm_command(Some(&["pnpm".to_string()]));

    let runner = mock_runner(|_| Ok("not json".to_string()));
    let pm = make_pm(
        &TestEnv {
            temp_dir,
            agent_dir,
            settings,
        },
        Vec::new(),
        runner,
    );

    assert_eq!(
        pm.get_installed_path("npm:pnpm-pkg", SourceScope::User),
        None
    );
}

// ============================================================================
// source parsing
// ============================================================================

#[tokio::test]
async fn emits_progress_events_on_install_attempt() {
    let _guard = env_guard();
    let (env, _pm) = setup("parse-progress");
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let runner = mock_runner(|_| Err("simulated npm install failure".to_string()));
    let pm = make_pm(&env, Vec::new(), runner);
    let sink = events.clone();
    pm.set_progress_callback(Some(Arc::new(move |event: &ProgressEvent| {
        sink.lock().push(event.clone())
    })));

    let error = pm
        .install("npm:nonexistent-package@1.0.0", false)
        .await
        .unwrap_err();
    assert!(error.contains("simulated npm install failure"));

    let events = events.lock();
    assert!(events
        .iter()
        .any(|e| e.event_type == ProgressEventType::Start && e.action == ProgressAction::Install));
    assert!(events
        .iter()
        .any(|e| e.event_type == ProgressEventType::Error));
}

#[tokio::test]
async fn recognizes_github_urls_without_git_prefix() {
    let _guard = env_guard();
    let (env, _pm) = setup("parse-github");
    let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let runner = mock_runner(|_| Err("simulated git clone failure".to_string()));
    let pm = make_pm(&env, Vec::new(), runner);
    let sink = events.clone();
    pm.set_progress_callback(Some(Arc::new(move |event: &ProgressEvent| {
        sink.lock().push(event.clone())
    })));

    let error = pm
        .install("https://github.com/nonexistent/repo", false)
        .await
        .unwrap_err();
    assert!(error.contains("simulated git clone failure"));

    assert!(events
        .lock()
        .iter()
        .any(|e| e.event_type == ProgressEventType::Start && e.action == ProgressAction::Install));
}

#[test]
fn parses_package_source_types_from_docs_examples() {
    let _guard = env_guard();
    let (_env, pm) = setup("parse-docs");
    let parse_npm = |source: &str| match pm.parse_source(source) {
        ParsedSource::Npm(parsed) => parsed,
        _ => panic!("Expected npm source: {source}"),
    };

    assert!(parse_npm("npm:@scope/pkg@1.2.3").pinned);
    assert!(!parse_npm("npm:@scope/pkg@^1.2.3").pinned);
    assert!(!parse_npm("npm:pkg").pinned);

    assert!(matches!(
        pm.parse_source("git:github.com/user/repo@v1"),
        ParsedSource::Git(_)
    ));
    assert!(matches!(
        pm.parse_source("https://github.com/user/repo@v1"),
        ParsedSource::Git(_)
    ));
    assert!(matches!(
        pm.parse_source("git:git@github.com:user/repo@v1"),
        ParsedSource::Git(_)
    ));
    assert!(matches!(
        pm.parse_source("ssh://git@github.com/user/repo@v1"),
        ParsedSource::Git(_)
    ));

    assert!(matches!(
        pm.parse_source("/absolute/path/to/package"),
        ParsedSource::Local(_)
    ));
    assert!(matches!(
        pm.parse_source("./relative/path/to/package"),
        ParsedSource::Local(_)
    ));
    assert!(matches!(
        pm.parse_source("../relative/path/to/package"),
        ParsedSource::Local(_)
    ));
}

#[test]
fn never_parses_dot_relative_paths_as_git() {
    let _guard = env_guard();
    let (_env, pm) = setup("parse-dotslash");
    match pm.parse_source("./packages/agent-timers") {
        ParsedSource::Local(local) => assert_eq!(local.path, "./packages/agent-timers"),
        _ => panic!("expected local"),
    }
    match pm.parse_source("../packages/agent-timers") {
        ParsedSource::Local(local) => assert_eq!(local.path, "../packages/agent-timers"),
        _ => panic!("expected local"),
    }
}

// ============================================================================
// git install paths
// ============================================================================

#[test]
fn rejects_paths_outside_git_install_roots() {
    let _guard = env_guard();
    let (_env, pm) = setup("git-traversal");
    let traversal_source = GitSource {
        repo: "git@evil.example:../../victim/repo".to_string(),
        host: "evil.example".to_string(),
        path: "../../victim/repo".to_string(),
        ref_: None,
        pinned: false,
    };

    for scope in [
        SourceScope::User,
        SourceScope::Project,
        SourceScope::Temporary,
    ] {
        let error = pm
            .get_git_install_path(&traversal_source, scope)
            .unwrap_err();
        assert!(
            error.contains("outside package install root"),
            "got: {error}"
        );
    }
}

#[test]
fn places_temporary_npm_packages_under_agent_temp_folder() {
    let _guard = env_guard();
    let (env, pm) = setup("npm-temp-path");
    let parsed = match pm.parse_source("npm:left-pad") {
        ParsedSource::Npm(parsed) => parsed,
        _ => panic!("expected npm source"),
    };

    let install_path = pm.get_npm_install_path(&parsed, SourceScope::Temporary);
    let temp_root = Path::new(&env.agent_dir)
        .join("tmp")
        .join("extensions")
        .to_string_lossy()
        .into_owned();

    assert!(install_path.ends_with("node_modules/left-pad"));
    assert!(!node_relative_path(&temp_root, &install_path).starts_with(".."));
    assert!(!install_path.contains("pi-extensions"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&temp_root).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }
}

// ============================================================================
// settings source normalization
// ============================================================================

#[test]
fn stores_global_local_packages_relative_to_agent_settings_base() {
    let _guard = env_guard();
    let (env, pm) = setup("norm-global");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("packages")
        .join("local-global-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/index.ts"),
        "export default function() {}",
    );

    assert!(pm.add_source_to_settings("./packages/local-global-pkg", false));

    let settings = env.settings.lock().get_global_settings();
    let rel = node_relative_path(&env.agent_dir, &pkg_dir);
    let expected = if rel.starts_with('.') {
        rel
    } else {
        format!("./{rel}")
    };
    assert_eq!(
        settings.get("packages").unwrap().as_array().unwrap()[0]
            .as_str()
            .unwrap(),
        expected
    );
}

#[test]
fn stores_project_local_packages_relative_to_pi_settings_base() {
    let _guard = env_guard();
    let (env, pm) = setup("norm-project");
    let project_pkg_dir = Path::new(&env.temp_dir)
        .join("project-local-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{project_pkg_dir}/extensions/index.ts"),
        "export default function() {}",
    );

    assert!(pm.add_source_to_settings("./project-local-pkg", true));

    let settings = env.settings.lock().get_project_settings();
    let rel = node_relative_path(
        &Path::new(&env.temp_dir).join(".pi").to_string_lossy(),
        &project_pkg_dir,
    );
    let expected = if rel.starts_with('.') {
        rel
    } else {
        format!("./{rel}")
    };
    assert_eq!(
        settings.get("packages").unwrap().as_array().unwrap()[0]
            .as_str()
            .unwrap(),
        expected
    );
}

#[test]
fn removes_local_package_entries_using_equivalent_path_forms() {
    let _guard = env_guard();
    let (env, pm) = setup("norm-remove");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("remove-local-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/index.ts"),
        "export default function() {}",
    );

    pm.add_source_to_settings("./remove-local-pkg", false);
    assert!(pm.remove_source_from_settings(&format!("{pkg_dir}/"), false));
    assert_eq!(
        env.settings
            .lock()
            .get_global_settings()
            .get("packages")
            .map(|v| v.as_array().unwrap().len()),
        Some(0)
    );
}

#[test]
fn returns_false_when_adding_same_git_source_with_same_ref() {
    let _guard = env_guard();
    let (env, pm) = setup("norm-same-ref");
    assert!(pm.add_source_to_settings("git:github.com/user/repo@v1", false));
    assert!(!pm.add_source_to_settings("git:github.com/user/repo@v1", false));
    let settings = env.settings.lock().get_global_settings();
    assert_eq!(
        serde_json::from_value::<Vec<PackageSource>>(settings.get("packages").unwrap().clone())
            .unwrap(),
        vec![PackageSource::Simple(
            "git:github.com/user/repo@v1".to_string()
        )]
    );
}

#[test]
fn updates_ref_when_adding_same_git_source_with_different_ref() {
    let _guard = env_guard();
    let (env, pm) = setup("norm-ref-update");
    pm.add_source_to_settings("git:github.com/user/repo@v1", false);

    assert!(pm.add_source_to_settings("git:github.com/user/repo@v2", false));
    let settings = env.settings.lock().get_global_settings();
    assert_eq!(
        serde_json::from_value::<Vec<PackageSource>>(settings.get("packages").unwrap().clone())
            .unwrap(),
        vec![PackageSource::Simple(
            "git:github.com/user/repo@v2".to_string()
        )]
    );
}

#[test]
fn preserves_package_filters_when_replacing_ref() {
    let _guard = env_guard();
    let (env, pm) = setup("norm-preserve-filters");
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: "git:github.com/user/repo@v1".to_string(),
        autoload: None,
        extensions: Some(vec!["extensions/main.ts".to_string()]),
        skills: Some(vec![]),
        prompts: Some(vec!["prompts/review.md".to_string()]),
        themes: Some(vec!["themes/dark.json".to_string()]),
    }]);

    assert!(pm.add_source_to_settings("git:github.com/user/repo@v2", false));
    let settings = env.settings.lock().get_global_settings();
    assert_eq!(
        serde_json::from_value::<Vec<PackageSource>>(settings.get("packages").unwrap().clone())
            .unwrap(),
        vec![PackageSource::Filtered {
            source: "git:github.com/user/repo@v2".to_string(),
            autoload: None,
            extensions: Some(vec!["extensions/main.ts".to_string()]),
            skills: Some(vec![]),
            prompts: Some(vec!["prompts/review.md".to_string()]),
            themes: Some(vec!["themes/dark.json".to_string()]),
        }]
    );
}

// ============================================================================
// HTTPS git URL parsing (old behavior)
// ============================================================================

#[test]
fn parses_https_git_urls() {
    let _guard = env_guard();
    let (_env, pm) = setup("https-parse");
    let parsed = |source| match pm.parse_source(source) {
        ParsedSource::Git(parsed) => parsed,
        other => panic!("expected git, got {other:?}"),
    };

    let p = parsed("https://github.com/user/repo");
    assert_eq!(p.host, "github.com");
    assert_eq!(p.path, "user/repo");
    assert!(!p.pinned);

    let p = parsed("git:https://github.com/user/repo");
    assert_eq!(p.host, "github.com");
    assert_eq!(p.path, "user/repo");

    let p = parsed("https://github.com/user/repo@v1.2.3");
    assert_eq!(p.host, "github.com");
    assert_eq!(p.path, "user/repo");
    assert_eq!(p.ref_.as_deref(), Some("v1.2.3"));
    assert!(p.pinned);

    let p = parsed("git:github.com/user/repo");
    assert_eq!(p.host, "github.com");
    assert_eq!(p.path, "user/repo");

    assert!(matches!(
        pm.parse_source("github.com/user/repo"),
        ParsedSource::Local(_)
    ));

    let p = parsed("https://github.com/user/repo.git");
    assert_eq!(p.host, "github.com");
    assert_eq!(p.path, "user/repo");

    assert_eq!(parsed("https://gitlab.com/user/repo").host, "gitlab.com");
    assert_eq!(
        parsed("https://bitbucket.org/user/repo").host,
        "bitbucket.org"
    );
    assert_eq!(
        parsed("https://codeberg.org/user/repo").host,
        "codeberg.org"
    );
}

#[test]
fn generates_identical_identity_for_url_forms() {
    let _guard = env_guard();
    let (_env, pm) = setup("https-identity");
    let identity = |source| pm.get_package_identity(source, None);
    assert_eq!(
        identity("https://github.com/user/repo"),
        "git:github.com/user/repo"
    );
    assert_eq!(
        identity("https://github.com/user/repo@v1.0.0"),
        "git:github.com/user/repo"
    );
    assert_eq!(
        identity("git:github.com/user/repo"),
        "git:github.com/user/repo"
    );
    assert_eq!(
        identity("https://github.com/user/repo.git"),
        "git:github.com/user/repo"
    );
    assert_eq!(
        identity("git:git@github.com:user/repo"),
        identity("https://github.com/user/repo")
    );
    assert_eq!(
        identity("ssh://git@github.com/user/repo"),
        identity("git:git@github.com:user/repo")
    );
    assert_ne!(
        identity("https://github.com/user/repo1"),
        identity("git:git@github.com:user/repo2")
    );
    assert_eq!(
        identity("https://github.com/user/repo@feature/branch"),
        identity("https://github.com/user/repo")
    );

    let parsed = match pm.parse_source("https://github.com/user/repo@main") {
        ParsedSource::Git(parsed) => parsed,
        _ => panic!("expected git"),
    };
    assert_eq!(parsed.ref_.as_deref(), Some("main"));
    assert!(parsed.pinned);
}

// ============================================================================
// pattern filtering in top-level arrays
// ============================================================================

#[tokio::test]
async fn excludes_extensions_with_bang_pattern() {
    let _guard = env_guard();
    let (env, pm) = setup("filter-ext-bang");
    let ext_dir = Path::new(&env.agent_dir)
        .join("extensions")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{ext_dir}/keep.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{ext_dir}/remove.ts"),
        "export default function() {}",
    );
    env.settings
        .lock()
        .set_extension_paths(&["extensions".to_string(), "!**/remove.ts".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "keep.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "remove.ts")));
}

#[tokio::test]
async fn filters_themes_with_glob_patterns() {
    let _guard = env_guard();
    let (env, pm) = setup("filter-themes");
    let themes_dir = Path::new(&env.agent_dir)
        .join("themes")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{themes_dir}/dark.json"), "{}");
    write_file(&format!("{themes_dir}/light.json"), "{}");
    write_file(&format!("{themes_dir}/funky.json"), "{}");
    env.settings
        .lock()
        .set_theme_paths(&["themes".to_string(), "!funky.json".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_enabled(r, "dark.json")));
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_enabled(r, "light.json")));
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_disabled(r, "funky.json")));
}

#[tokio::test]
async fn filters_prompts_with_exclusion_pattern() {
    let _guard = env_guard();
    let (env, pm) = setup("filter-prompts");
    let prompts_dir = Path::new(&env.agent_dir)
        .join("prompts")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{prompts_dir}/review.md"), "Review code");
    write_file(&format!("{prompts_dir}/explain.md"), "Explain code");
    env.settings
        .lock()
        .set_prompt_template_paths(&["prompts".to_string(), "!explain.md".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .prompts
        .iter()
        .any(|r| ends_with_enabled(r, "review.md")));
    assert!(result
        .prompts
        .iter()
        .any(|r| ends_with_disabled(r, "explain.md")));
}

#[tokio::test]
async fn filters_skills_with_exclusion_pattern() {
    let _guard = env_guard();
    let (env, pm) = setup("filter-skills");
    let skills_dir = Path::new(&env.agent_dir)
        .join("skills")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{skills_dir}/good-skill/SKILL.md"),
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    );
    write_file(
        &format!("{skills_dir}/bad-skill/SKILL.md"),
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    );
    env.settings
        .lock()
        .set_skill_paths(&["skills".to_string(), "!**/bad-skill".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("good-skill") && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("bad-skill") && !r.enabled));
}

#[tokio::test]
async fn works_without_patterns() {
    let _guard = env_guard();
    let (env, pm) = setup("filter-plain");
    let ext_dir = Path::new(&env.agent_dir)
        .join("extensions")
        .to_string_lossy()
        .into_owned();
    let ext_path = format!("{ext_dir}/my-ext.ts");
    write_file(&ext_path, "export default function() {}");
    env.settings
        .lock()
        .set_extension_paths(&["extensions/my-ext.ts".to_string()]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path == ext_path && r.enabled));
}

// ============================================================================
// pattern filtering in pi manifest
// ============================================================================

#[tokio::test]
async fn supports_glob_patterns_in_manifest_extensions() {
    let _guard = env_guard();
    let (env, pm) = setup("manifest-glob-ext");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("manifest-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/local.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/node_modules/dep/extensions/remote.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/node_modules/dep/extensions/skip.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"manifest-pkg","pi":{"extensions":["extensions","node_modules/dep/extensions","!**/skip.ts"]}}"#,
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "local.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "remote.ts")));
    assert!(!result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("skip.ts")));
}

#[tokio::test]
async fn supports_glob_patterns_in_manifest_skills() {
    let _guard = env_guard();
    let (env, pm) = setup("manifest-glob-skills");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("skill-manifest-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/skills/good-skill/SKILL.md"),
        "---\nname: good-skill\ndescription: Good\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/skills/bad-skill/SKILL.md"),
        "---\nname: bad-skill\ndescription: Bad\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"skill-manifest-pkg","pi":{"skills":["skills","!**/bad-skill"]}}"#,
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("good-skill") && r.enabled));
    assert!(!result.skills.iter().any(|r| r.path.contains("bad-skill")));
}

#[tokio::test]
async fn expands_positive_glob_manifest_entries_before_collecting_skills() {
    let _guard = env_guard();
    let (env, pm) = setup("manifest-glob-skill-dirs");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("skill-manifest-glob-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/plugins/pdf-to-markdown/skills/pdf-to-markdown/SKILL.md"),
        "---\nname: pdf-to-markdown\ndescription: PDF to Markdown\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/plugins/nutrient-dws/skills/document-processor-api/SKILL.md"),
        "---\nname: document-processor-api\ndescription: DWS\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"skill-manifest-glob-pkg","pi":{"skills":["./plugins/*/skills"]}}"#,
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("pdf-to-markdown") && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("document-processor-api") && r.enabled));
}

#[tokio::test]
async fn sorts_manifest_glob_matches_and_allows_exact_dot_and_symlink_entries() {
    let _guard = env_guard();
    let (env, pm) = setup("manifest-glob-order");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("manifest-glob-semantics-pkg")
        .to_string_lossy()
        .into_owned();
    let extension_files_dir = format!("{pkg_dir}/extension-files");
    let extension_group_dir = format!("{pkg_dir}/extension-groups/group");
    let linked_plugin_source = format!("{pkg_dir}/linked-plugin-source");
    std::fs::create_dir_all(format!("{extension_files_dir}/nested")).unwrap();
    std::fs::create_dir_all(&extension_group_dir).unwrap();
    std::fs::create_dir_all(format!("{pkg_dir}/plugins/local/skills/local-skill")).unwrap();
    std::fs::create_dir_all(format!("{linked_plugin_source}/skills/linked-skill")).unwrap();
    write_file(
        &format!("{extension_files_dir}/z.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{extension_files_dir}/a.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{extension_files_dir}/.ignored.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{extension_files_dir}/nested/.hidden.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{extension_group_dir}/index.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/plugins/local/skills/local-skill/SKILL.md"),
        "---\nname: local-skill\ndescription: Local\n---\n",
    );
    write_file(
        &format!("{linked_plugin_source}/skills/linked-skill/SKILL.md"),
        "---\nname: linked-skill\ndescription: Linked\n---\n",
    );
    std::os::unix::fs::symlink(&linked_plugin_source, format!("{pkg_dir}/plugins/linked")).unwrap();
    write_file(
        &format!("{pkg_dir}/package.json"),
        &format!(
            r#"{{"name":"manifest-glob-semantics-pkg","pi":{{"extensions":["./extension-files/*.ts","./extension-files/**/.ignored.ts","./extension-files/nested/.hidden.ts","./extension-groups/*/"],"skills":["./plugins/*/skills","./plugins/linked/skills"]}}}}"#
        ),
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir.clone()], false, false)
        .await
        .unwrap();
    let relative: Vec<String> = result
        .extensions
        .iter()
        .map(|resource| node_relative_path(&pkg_dir, &resource.path))
        .collect();
    assert_eq!(
        relative,
        vec![
            "extension-files/a.ts".to_string(),
            "extension-files/z.ts".to_string(),
            "extension-files/nested/.hidden.ts".to_string(),
            "extension-groups/group/index.ts".to_string(),
        ]
    );
    assert!(result
        .skills
        .iter()
        .any(|resource| resource.path.ends_with("local-skill/SKILL.md")));
    assert!(result
        .skills
        .iter()
        .any(|resource| resource.path.ends_with("linked-skill/SKILL.md")));
}

// ============================================================================
// pattern filtering in package filters
// ============================================================================

#[tokio::test]
async fn applies_user_filters_on_top_of_manifest_filters() {
    let _guard = env_guard();
    let (env, pm) = setup("pkgfilter-layered");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("layered-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/foo.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/bar.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/baz.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"layered-pkg","pi":{"extensions":["extensions","!**/baz.ts"]}}"#,
    );

    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec!["!**/bar.ts".to_string()]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "foo.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "bar.ts")));
    assert!(!has_extension(&result, "baz.ts"));
}

#[tokio::test]
async fn excludes_extensions_from_package_with_bang_pattern() {
    let _guard = env_guard();
    let (env, pm) = setup("pkgfilter-bang");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("pattern-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/foo.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/bar.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/baz.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec!["!**/baz.ts".to_string()]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "foo.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "bar.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "baz.ts")));
}

#[tokio::test]
async fn filters_themes_from_package() {
    let _guard = env_guard();
    let (env, pm) = setup("pkgfilter-themes");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("theme-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{pkg_dir}/themes/nice.json"), "{}");
    write_file(&format!("{pkg_dir}/themes/ugly.json"), "{}");
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec![]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec!["!ugly.json".to_string()]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_enabled(r, "nice.json")));
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_disabled(r, "ugly.json")));
}

#[tokio::test]
async fn combines_include_and_exclude_patterns() {
    let _guard = env_guard();
    let (env, pm) = setup("pkgfilter-combo");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("combo-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/alpha.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/beta.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/gamma.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec![
            "**/alpha.ts".to_string(),
            "**/beta.ts".to_string(),
            "!**/beta.ts".to_string(),
        ]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "alpha.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "beta.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "gamma.ts")));
}

#[tokio::test]
async fn works_with_direct_paths_no_patterns() {
    let _guard = env_guard();
    let (env, pm) = setup("pkgfilter-direct");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("direct-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/one.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/two.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec!["extensions/one.ts".to_string()]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "one.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "two.ts")));
}

#[tokio::test]
async fn resolves_autoload_disabled_project_entries_as_deltas() {
    let _guard = env_guard();
    let (env, _pm) = setup("pkgfilter-delta");
    let pkg_dir = Path::new(&env.agent_dir)
        .join("npm")
        .join("node_modules")
        .join("pi-tools")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"pi-tools","version":"1.0.0"}"#,
    );
    write_file(
        &format!("{pkg_dir}/extensions/foo.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/bar.ts"),
        "export default function() {}",
    );
    env.settings
        .lock()
        .set_packages(&[PackageSource::Simple("npm:pi-tools".to_string())]);
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Filtered {
            source: "npm:pi-tools".to_string(),
            autoload: Some(false),
            extensions: Some(vec!["-extensions/foo.ts".to_string()]),
            skills: None,
            prompts: None,
            themes: None,
        }]);

    let runner = failing_runner();
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let result = pm.resolve(None).await.unwrap();
    assert_eq!(runner.call_count(), 0);
    let foo = result
        .extensions
        .iter()
        .find(|r| r.path == format!("{pkg_dir}/extensions/foo.ts"))
        .unwrap();
    assert!(!foo.enabled);
    assert_eq!(foo.metadata.scope, SourceScope::Project);
    let bar = result
        .extensions
        .iter()
        .find(|r| r.path == format!("{pkg_dir}/extensions/bar.ts"))
        .unwrap();
    assert!(bar.enabled);
    assert_eq!(bar.metadata.scope, SourceScope::User);
}

#[tokio::test]
async fn resolves_autoload_disabled_entries_as_positive_only_without_global() {
    let _guard = env_guard();
    let (env, pm) = setup("pkgfilter-positive-only");
    let _home = HomeOverride::set(&env.temp_dir);

    let pkg_dir = Path::new(&env.temp_dir)
        .join("positive-only-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/foo.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/bar.ts"),
        "export default function() {}",
    );
    write_file(&format!("{pkg_dir}/skills/foo/SKILL.md"), "# Foo\n");
    let rel = node_relative_path(
        &Path::new(&env.temp_dir).join(".pi").to_string_lossy(),
        &pkg_dir,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Filtered {
            source: rel,
            autoload: Some(false),
            extensions: Some(vec!["+extensions/foo.ts".to_string()]),
            skills: None,
            prompts: None,
            themes: None,
        }]);

    let result = pm.resolve(None).await.unwrap();

    assert_eq!(
        result
            .extensions
            .iter()
            .map(|r| r.path.clone())
            .collect::<Vec<_>>(),
        vec![format!("{pkg_dir}/extensions/foo.ts")]
    );
    assert!(result.skills.is_empty());
}

// ============================================================================
// force-include patterns
// ============================================================================

#[tokio::test]
async fn force_includes_extensions_with_plus_pattern_after_exclusion() {
    let _guard = env_guard();
    let (env, pm) = setup("force-ext");
    let ext_dir = Path::new(&env.agent_dir)
        .join("extensions")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{ext_dir}/keep.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{ext_dir}/excluded.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{ext_dir}/force-back.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_extension_paths(&[
        "extensions".to_string(),
        "!extensions/*.ts".to_string(),
        "+extensions/force-back.ts".to_string(),
    ]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "keep.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "excluded.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "force-back.ts")));
}

#[tokio::test]
async fn force_include_overrides_exclude_in_package_filters() {
    let _guard = env_guard();
    let (env, pm) = setup("force-pkg");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("force-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/alpha.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/beta.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/gamma.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec![
            "!**/*.ts".to_string(),
            "+extensions/beta.ts".to_string(),
        ]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "alpha.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "beta.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "gamma.ts")));
}

#[tokio::test]
async fn force_includes_multiple_resources() {
    let _guard = env_guard();
    let (env, pm) = setup("force-multi");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("multi-force-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/skills/skill-a/SKILL.md"),
        "---\nname: skill-a\ndescription: A\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/skills/skill-b/SKILL.md"),
        "---\nname: skill-b\ndescription: B\n---\nContent",
    );
    write_file(
        &format!("{pkg_dir}/skills/skill-c/SKILL.md"),
        "---\nname: skill-c\ndescription: C\n---\nContent",
    );
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec![]),
        skills: Some(vec![
            "!**/*".to_string(),
            "+skills/skill-a".to_string(),
            "+skills/skill-c".to_string(),
        ]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("skill-a") && r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("skill-b") && !r.enabled));
    assert!(result
        .skills
        .iter()
        .any(|r| r.path.contains("skill-c") && r.enabled));
}

#[tokio::test]
async fn force_includes_after_specific_exclusion() {
    let _guard = env_guard();
    let (env, pm) = setup("force-specific");
    let ext_dir = Path::new(&env.agent_dir)
        .join("extensions")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{ext_dir}/a.ts"), "export default function() {}");
    write_file(&format!("{ext_dir}/b.ts"), "export default function() {}");
    env.settings.lock().set_extension_paths(&[
        "extensions".to_string(),
        "!extensions/b.ts".to_string(),
        "+extensions/b.ts".to_string(),
    ]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "a.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "b.ts")));
}

#[tokio::test]
async fn force_include_in_manifest_patterns() {
    let _guard = env_guard();
    let (env, pm) = setup("force-manifest");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("manifest-force-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/one.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/two.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/three.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/package.json"),
        r#"{"name":"manifest-force-pkg","pi":{"extensions":["extensions","!**/two.ts","+extensions/two.ts"]}}"#,
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "one.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "two.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "three.ts")));
}

#[tokio::test]
async fn force_includes_themes_and_prompts() {
    let _guard = env_guard();
    let (env, pm) = setup("force-themes-prompts");
    let themes_dir = Path::new(&env.agent_dir)
        .join("themes")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{themes_dir}/dark.json"), "{}");
    write_file(&format!("{themes_dir}/light.json"), "{}");
    write_file(&format!("{themes_dir}/special.json"), "{}");
    env.settings.lock().set_theme_paths(&[
        "themes".to_string(),
        "!themes/*.json".to_string(),
        "+themes/special.json".to_string(),
    ]);

    let prompts_dir = Path::new(&env.agent_dir)
        .join("prompts")
        .to_string_lossy()
        .into_owned();
    write_file(&format!("{prompts_dir}/review.md"), "Review");
    write_file(&format!("{prompts_dir}/explain.md"), "Explain");
    write_file(&format!("{prompts_dir}/debug.md"), "Debug");
    env.settings.lock().set_prompt_template_paths(&[
        "prompts".to_string(),
        "!prompts/*.md".to_string(),
        "+prompts/debug.md".to_string(),
    ]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_disabled(r, "dark.json")));
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_disabled(r, "light.json")));
    assert!(result
        .themes
        .iter()
        .any(|r| ends_with_enabled(r, "special.json")));
    assert!(result
        .prompts
        .iter()
        .any(|r| ends_with_disabled(r, "review.md")));
    assert!(result
        .prompts
        .iter()
        .any(|r| ends_with_disabled(r, "explain.md")));
    assert!(result
        .prompts
        .iter()
        .any(|r| ends_with_enabled(r, "debug.md")));
}

// ============================================================================
// force-exclude patterns
// ============================================================================

#[tokio::test]
async fn force_excludes_top_level_resources() {
    let _guard = env_guard();
    let (env, pm) = setup("force-excl-top");
    let ext_dir = Path::new(&env.agent_dir)
        .join("extensions")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{ext_dir}/alpha.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{ext_dir}/beta.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_extension_paths(&[
        "extensions".to_string(),
        "+extensions/alpha.ts".to_string(),
        "-extensions/alpha.ts".to_string(),
    ]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "alpha.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "beta.ts")));
}

#[tokio::test]
async fn force_excludes_in_package_filters() {
    let _guard = env_guard();
    let (env, pm) = setup("force-excl-pkg");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("force-exclude-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/alpha.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/beta.ts"),
        "export default function() {}",
    );
    env.settings.lock().set_packages(&[PackageSource::Filtered {
        source: pkg_dir,
        autoload: None,
        extensions: Some(vec![
            "extensions/*.ts".to_string(),
            "+extensions/alpha.ts".to_string(),
            "-extensions/alpha.ts".to_string(),
        ]),
        skills: Some(vec![]),
        prompts: Some(vec![]),
        themes: Some(vec![]),
    }]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_disabled(r, "alpha.ts")));
    assert!(result
        .extensions
        .iter()
        .any(|r| ends_with_enabled(r, "beta.ts")));
}

// ============================================================================
// package deduplication
// ============================================================================

#[tokio::test]
async fn dedupes_same_local_package_project_wins() {
    let _guard = env_guard();
    let (env, pm) = setup("dedupe-local");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("shared-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/shared.ts"),
        "export default function() {}",
    );

    env.settings
        .lock()
        .set_packages(&[PackageSource::Simple(pkg_dir.clone())]);
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple(pkg_dir.clone())]);

    let result = pm.resolve(None).await.unwrap();
    let shared: Vec<_> = result
        .extensions
        .iter()
        .filter(|r| r.path.contains("shared-pkg"))
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].metadata.scope, SourceScope::Project);
}

#[tokio::test]
async fn keeps_both_when_different_packages() {
    let _guard = env_guard();
    let (env, pm) = setup("dedupe-different");
    let pkg1_dir = Path::new(&env.temp_dir)
        .join("pkg1")
        .to_string_lossy()
        .into_owned();
    let pkg2_dir = Path::new(&env.temp_dir)
        .join("pkg2")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg1_dir}/extensions/from-pkg1.ts"),
        "export default function() {}",
    );
    write_file(
        &format!("{pkg2_dir}/extensions/from-pkg2.ts"),
        "export default function() {}",
    );

    env.settings
        .lock()
        .set_packages(&[PackageSource::Simple(pkg1_dir)]);
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple(pkg2_dir)]);

    let result = pm.resolve(None).await.unwrap();
    assert!(result.extensions.iter().any(|r| r.path.contains("pkg1")));
    assert!(result.extensions.iter().any(|r| r.path.contains("pkg2")));
}

// ============================================================================
// multi-file extension discovery (issue #1102)
// ============================================================================

#[tokio::test]
async fn only_loads_index_from_subdirectories() {
    let _guard = env_guard();
    let (env, pm) = setup("multifile-index");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("multifile-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
		&format!("{pkg_dir}/extensions/subagent/index.ts"),
		"import { helper } from \"./agents.ts\";\nexport default function(api) { api.registerTool({ name: \"test\", description: \"test\", execute: async () => helper() }); }",
	);
    write_file(
        &format!("{pkg_dir}/extensions/subagent/agents.ts"),
        "export function helper() { return \"helper\"; }",
    );
    write_file(
        &format!("{pkg_dir}/extensions/standalone.ts"),
        "export default function(api) {}",
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("subagent/index.ts") && r.enabled));
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("standalone.ts") && r.enabled));
    assert!(!has_extension(&result, "agents.ts"));
}

#[tokio::test]
async fn respects_pi_manifest_in_subdirectories() {
    let _guard = env_guard();
    let (env, pm) = setup("multifile-manifest-subdir");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("manifest-subdir-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/custom/package.json"),
        r#"{"pi":{"extensions":["./main.ts"]}}"#,
    );
    write_file(
        &format!("{pkg_dir}/extensions/custom/main.ts"),
        "export default function(api) {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/custom/utils.ts"),
        "export const util = 1;",
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("custom/main.ts") && r.enabled));
    assert!(!has_extension(&result, "utils.ts"));
}

#[tokio::test]
async fn handles_mixed_top_level_files_and_subdirectories() {
    let _guard = env_guard();
    let (env, pm) = setup("multifile-mixed");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("mixed-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/simple.ts"),
        "export default function(api) {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/complex/index.ts"),
        "import { a } from './a.ts'; export default function(api) {}",
    );
    write_file(
        &format!("{pkg_dir}/extensions/complex/a.ts"),
        "export const a = 1;",
    );
    write_file(
        &format!("{pkg_dir}/extensions/complex/b.ts"),
        "export const b = 2;",
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("simple.ts") && r.enabled));
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("complex/index.ts") && r.enabled));
    assert!(!has_extension(&result, "complex/a.ts"));
    assert!(!has_extension(&result, "complex/b.ts"));
    assert_eq!(result.extensions.iter().filter(|r| r.enabled).count(), 2);
}

#[tokio::test]
async fn skips_subdirectories_without_index_or_manifest() {
    let _guard = env_guard();
    let (env, pm) = setup("multifile-no-entry");
    let pkg_dir = Path::new(&env.temp_dir)
        .join("no-entry-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{pkg_dir}/extensions/broken/helper.ts"),
        "export const x = 1;",
    );
    write_file(
        &format!("{pkg_dir}/extensions/broken/another.ts"),
        "export const y = 2;",
    );
    write_file(
        &format!("{pkg_dir}/extensions/valid.ts"),
        "export default function(api) {}",
    );

    let result = pm
        .resolve_extension_sources(&[pkg_dir], false, false)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("valid.ts") && r.enabled));
    assert_eq!(result.extensions.iter().filter(|r| r.enabled).count(), 1);
}

// ============================================================================
// offline mode and network timeouts
// ============================================================================

#[tokio::test]
async fn updates_npm_range_packages_using_configured_spec() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-npm-range");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"1.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example@^1.0.0".to_string())]);

    let runner = mock_runner(|call| {
        if call.args.first().map(String::as_str) == Some("view") {
            Ok(r#"["1.0.0","1.2.0"]"#.to_string())
        } else {
            Ok(String::new())
        }
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(Some("npm:example")).await.unwrap();

    assert!(runner.calls().iter().any(|call| {
        call.command == "npm"
            && call.args
                == vec![
                    "view".to_string(),
                    "example@^1.0.0".to_string(),
                    "version".to_string(),
                    "--json".to_string(),
                ]
            && call.cwd.as_deref() == Some(env.temp_dir.as_str())
    }));
    assert!(runner.calls_for("npm").contains(&(
        vec![
            "install".to_string(),
            "example@^1.0.0".to_string(),
            "--prefix".to_string(),
            Path::new(&env.temp_dir)
                .join(".pi")
                .join("npm")
                .to_string_lossy()
                .into_owned(),
            "--legacy-peer-deps".to_string(),
        ],
        None
    )));
}

#[tokio::test]
async fn skips_project_npm_update_when_installed_matches_latest() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-npm-current");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"1.3.1"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example@^1.0.0".to_string())]);

    let runner = mock_runner(|call| {
        if call.args.first().map(String::as_str) == Some("view") {
            Ok(r#"["1.0.0","1.3.1","1.0.2"]"#.to_string())
        } else {
            Ok(String::new())
        }
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(Some("npm:example")).await.unwrap();

    assert!(runner
        .calls()
        .iter()
        .any(|call| call.args.contains(&"example@^1.0.0".to_string())));
    assert!(runner
        .calls_for("npm")
        .iter()
        .all(|(args, _)| !args.contains(&"install".to_string())));
}

#[tokio::test]
async fn skips_npm_updates_when_installed_newer_than_registry() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-npm-newer");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"2.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example".to_string())]);

    let runner = mock_runner(|_| Ok(r#""1.9.0""#.to_string()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(Some("npm:example")).await.unwrap();

    assert!(runner
        .calls()
        .iter()
        .any(|call| call.args.contains(&"example".to_string())));
    assert!(runner
        .calls_for("npm")
        .iter()
        .all(|(args, _)| !args.contains(&"install".to_string())));
}

#[tokio::test]
async fn migrates_legacy_user_npm_installs_during_update() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-npm-migrate");
    let legacy_root = Path::new(&env.temp_dir)
        .join("legacy-global")
        .join("node_modules")
        .to_string_lossy()
        .into_owned();
    let legacy_path = format!("{legacy_root}/legacy-pkg");
    let managed_path = Path::new(&env.agent_dir)
        .join("npm")
        .join("node_modules")
        .join("legacy-pkg")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{legacy_path}/package.json"),
        r#"{"name":"legacy-pkg","version":"1.0.0"}"#,
    );
    env.settings
        .lock()
        .set_packages(&[PackageSource::Simple("npm:legacy-pkg".to_string())]);

    let managed_path_clone = managed_path.clone();
    let runner = mock_runner(move |call| {
        if call.args.first().map(String::as_str) == Some("root") {
            return Ok(legacy_root.clone());
        }
        if call.command == "npm" && call.args.first().map(String::as_str) == Some("install") {
            assert_eq!(
                call.args,
                vec![
                    "install".to_string(),
                    "legacy-pkg@latest".to_string(),
                    "--prefix".to_string(),
                    Path::new(&managed_path_clone)
                        .parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    "--legacy-peer-deps".to_string(),
                ]
            );
            std::fs::create_dir_all(&managed_path_clone).unwrap();
            write_file(
                &format!("{managed_path_clone}/package.json"),
                r#"{"name":"legacy-pkg","version":"1.0.0"}"#,
            );
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    assert_eq!(
        pm.get_installed_path("npm:legacy-pkg", SourceScope::User),
        Some(legacy_path)
    );

    pm.update(Some("npm:legacy-pkg")).await.unwrap();

    assert!(runner
        .calls()
        .iter()
        .all(|call| call.args.first().map(String::as_str) != Some("view")));
    assert_eq!(
        pm.get_installed_path("npm:legacy-pkg", SourceScope::User),
        Some(managed_path)
    );
}

#[tokio::test]
async fn suggests_npm_source_prefixes_for_update_lookups() {
    let _guard = env_guard();
    let (env, pm) = setup("suggest-npm");
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example".to_string())]);

    let error = pm.update(Some("example")).await.unwrap_err();
    assert_eq!(
        error,
        "No matching package found for example. Did you mean npm:example?"
    );
}

#[tokio::test]
async fn suggests_git_source_prefixes_for_update_lookups() {
    let _guard = env_guard();
    let (env, pm) = setup("suggest-git");
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple(
            "git:github.com/example/repo".to_string(),
        )]);

    let error = pm
        .update(Some("github.com/example/repo"))
        .await
        .unwrap_err();
    assert_eq!(error, "No matching package found for github.com/example/repo. Did you mean git:github.com/example/repo?");
}

#[tokio::test]
async fn skips_installing_missing_sources_when_offline() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-missing");
    std::env::set_var("PI_OFFLINE", "1");
    env.settings.lock().set_project_packages(&[
        PackageSource::Simple("npm:missing-package".to_string()),
        PackageSource::Simple("git:github.com/example/missing-repo".to_string()),
    ]);

    let runner = failing_runner();
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let result = pm.resolve(None).await.unwrap();
    std::env::remove_var("PI_OFFLINE");
    assert!(!result
        .all()
        .any(|r| r.metadata.origin == SourceOrigin::Package));
    assert_eq!(runner.call_count(), 0);
}

#[tokio::test]
async fn skips_refreshing_temporary_git_sources_when_offline() {
    let _guard = env_guard();
    let (_env, pm) = setup("offline-temp-git");
    std::env::set_var("PI_OFFLINE", "1");
    let git_source = "git:github.com/example/repo".to_string();
    let parsed_git_source = pm.parse_source(&git_source);
    let installed_path = match &parsed_git_source {
        ParsedSource::Git(parsed) => pm
            .get_git_install_path(parsed, SourceScope::Temporary)
            .unwrap(),
        _ => panic!("expected git source"),
    };

    write_file(
        &format!("{installed_path}/extensions/index.ts"),
        "export default function() {};",
    );

    let result = pm
        .resolve_extension_sources(&[git_source], false, true)
        .await
        .unwrap();
    std::env::remove_var("PI_OFFLINE");
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("extensions/index.ts") && r.enabled));
}

#[tokio::test]
async fn loads_new_checkout_when_pinned_temporary_git_ref_changes() {
    let _guard = env_guard();
    let (env, pm) = setup("temp-git-ref-change");
    let old_source = "git:github.com/example/repo@aaaaaaa".to_string();
    let new_source = "git:github.com/example/repo@bbbbbbb".to_string();
    let old_path = match pm.parse_source(&old_source) {
        ParsedSource::Git(parsed) => pm
            .get_git_install_path(&parsed, SourceScope::Temporary)
            .unwrap(),
        _ => panic!("expected git source"),
    };
    let new_path = match pm.parse_source(&new_source) {
        ParsedSource::Git(parsed) => pm
            .get_git_install_path(&parsed, SourceScope::Temporary)
            .unwrap(),
        _ => panic!("expected git source"),
    };
    write_file(
        &format!("{old_path}/extensions/old.ts"),
        "export default function() {};",
    );

    let new_path_clone = new_path.clone();
    let runner = mock_runner(move |call| {
        if call.command == "git" && call.args.first().map(String::as_str) == Some("clone") {
            write_file(
                &format!("{new_path_clone}/extensions/new.ts"),
                "export default function() {};",
            );
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let result = pm
        .resolve_extension_sources(&[new_source], false, true)
        .await
        .unwrap();
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("extensions/new.ts") && r.enabled));
    assert!(!result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("extensions/old.ts")));
}

#[tokio::test]
async fn does_not_run_npm_view_during_resolve_for_installed_unpinned() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-resolve-no-view");
    std::env::set_var("PI_OFFLINE", "1");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"1.0.0"}"#,
    );
    write_file(
        &format!("{installed_path}/extensions/index.ts"),
        "export default function() {};",
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example@^1.0.0".to_string())]);

    let runner = failing_runner();
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let result = pm.resolve(None).await.unwrap();
    std::env::remove_var("PI_OFFLINE");
    assert!(result
        .extensions
        .iter()
        .any(|r| r.path.ends_with("extensions/index.ts") && r.enabled));
    assert!(runner
        .calls()
        .iter()
        .all(|call| call.args.first().map(String::as_str) != Some("view")));
}

#[tokio::test]
async fn reinstalls_pinned_npm_packages_when_version_mismatches() {
    let _guard = env_guard();
    let (env, _pm) = setup("pinned-reinstall");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"1.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example@2.0.0".to_string())]);

    let runner = mock_runner(|_| Ok(String::new()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.resolve(None).await.unwrap();
    assert!(runner
        .calls_for("npm")
        .iter()
        .any(|(args, _)| args.contains(&"example@2.0.0".to_string())));
}

#[tokio::test]
async fn does_not_check_updates_when_offline() {
    let _guard = env_guard();
    let (env, _pm) = setup("offline-check");
    std::env::set_var("PI_OFFLINE", "1");
    let runner = failing_runner();
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let updates = pm.check_for_available_updates().await.unwrap();
    std::env::remove_var("PI_OFFLINE");
    assert!(updates.is_empty());
    assert_eq!(runner.call_count(), 0);
}

#[tokio::test]
async fn reports_updates_for_installed_unpinned_npm_packages() {
    let _guard = env_guard();
    let (env, _pm) = setup("check-npm-update");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"1.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example".to_string())]);

    let runner = mock_runner(|_| Ok(r#""1.2.3""#.to_string()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let updates = pm.check_for_available_updates().await.unwrap();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].source, "npm:example");
    assert_eq!(updates[0].display_name, "example");
    assert_eq!(updates[0].update_type, PackageUpdateType::Npm);
    assert_eq!(updates[0].scope, SourceScope::Project);
}

#[tokio::test]
async fn does_not_report_npm_updates_when_installed_newer() {
    let _guard = env_guard();
    let (env, _pm) = setup("check-npm-newer");
    let installed_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_path}/package.json"),
        r#"{"name":"example","version":"2.0.0"}"#,
    );
    env.settings
        .lock()
        .set_project_packages(&[PackageSource::Simple("npm:example".to_string())]);

    let runner = mock_runner(|_| Ok(r#""1.9.0""#.to_string()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let updates = pm.check_for_available_updates().await.unwrap();
    assert!(updates.is_empty());
}

#[tokio::test]
async fn skips_pinned_packages_when_checking_updates() {
    let _guard = env_guard();
    let (env, pm) = setup("check-pinned");
    let installed_npm_path = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules")
        .join("example")
        .to_string_lossy()
        .into_owned();
    write_file(
        &format!("{installed_npm_path}/package.json"),
        r#"{"name":"example","version":"1.0.0"}"#,
    );
    let installed_git_path = match pm.parse_source("git:github.com/example/repo@v1") {
        ParsedSource::Git(parsed) => pm
            .get_git_install_path(&parsed, SourceScope::Project)
            .unwrap(),
        _ => panic!("expected git"),
    };
    std::fs::create_dir_all(&installed_git_path).unwrap();

    env.settings.lock().set_project_packages(&[
        PackageSource::Simple("npm:example@1.0.0".to_string()),
        PackageSource::Simple("git:github.com/example/repo@v1".to_string()),
    ]);

    let runner = failing_runner();
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let updates = pm.check_for_available_updates().await.unwrap();
    assert!(updates.is_empty());
    assert_eq!(runner.call_count(), 0);
}

#[tokio::test]
async fn uses_npm_view_to_fetch_latest_version() {
    let _guard = env_guard();
    let (env, _pm) = setup("check-npm-view");
    let runner = mock_runner(|_| Ok(r#""1.2.3""#.to_string()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let latest = pm.get_latest_npm_version("example", None).await.unwrap();
    assert_eq!(latest, "1.2.3");
    assert!(runner.calls().iter().any(|call| {
        call.command == "npm"
            && call.args
                == vec![
                    "view".to_string(),
                    "example".to_string(),
                    "version".to_string(),
                    "--json".to_string(),
                ]
            && call.cwd.as_deref() == Some(env.temp_dir.as_str())
    }));
}

#[tokio::test]
async fn uses_npm_command_argv_for_update_checks() {
    let _guard = env_guard();
    let (env, _pm) = setup("check-npm-argv");
    env.settings.lock().set_npm_command(Some(&[
        "mise".to_string(),
        "exec".to_string(),
        "node@20".to_string(),
        "--".to_string(),
        "npm".to_string(),
    ]));
    let runner = mock_runner(|_| Ok(r#""1.2.3""#.to_string()));
    let pm = make_pm(&env, Vec::new(), runner.clone());

    let latest = pm.get_latest_npm_version("@scope/pkg", None).await.unwrap();
    assert_eq!(latest, "1.2.3");
    assert!(runner.calls().iter().any(|call| {
        call.command == "mise"
            && call.args
                == vec![
                    "exec".to_string(),
                    "node@20".to_string(),
                    "--".to_string(),
                    "npm".to_string(),
                    "view".to_string(),
                    "@scope/pkg".to_string(),
                    "version".to_string(),
                    "--json".to_string(),
                ]
            && call.cwd.as_deref() == Some(env.temp_dir.as_str())
    }));
}

// ============================================================================
// batched updates (concurrency-ratio probes deferred, module docs)
// ============================================================================

#[tokio::test]
async fn batches_npm_updates_per_scope_and_skips_pinned_and_current() {
    let _guard = env_guard();
    let (env, _pm) = setup("batch-updates");
    let agent_npm = Path::new(&env.agent_dir).join("npm").join("node_modules");
    let project_npm = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .join("node_modules");
    for name in ["user-old", "user-current", "user-unknown"] {
        let path = agent_npm.join(name);
        std::fs::create_dir_all(&path).unwrap();
        write_file(
            &path.join("package.json").to_string_lossy(),
            &format!(r#"{{"name":"{name}","version":"1.0.0"}}"#),
        );
    }
    for name in ["project-old", "project-current"] {
        let path = project_npm.join(name);
        std::fs::create_dir_all(&path).unwrap();
        write_file(
            &path.join("package.json").to_string_lossy(),
            &format!(r#"{{"name":"{name}","version":"1.0.0"}}"#),
        );
    }

    env.settings.lock().set_packages(&[
        PackageSource::Simple("npm:user-old".to_string()),
        PackageSource::Simple("npm:user-current".to_string()),
        PackageSource::Simple("npm:user-unknown".to_string()),
        PackageSource::Simple("npm:user-pinned@1.0.0".to_string()),
        PackageSource::Simple("git:github.com/example/user-repo-a".to_string()),
        PackageSource::Simple("git:github.com/example/user-repo-b".to_string()),
        PackageSource::Simple("git:github.com/example/user-repo-pinned@v1".to_string()),
    ]);
    env.settings.lock().set_project_packages(&[
        PackageSource::Simple("npm:project-old".to_string()),
        PackageSource::Simple("npm:project-current".to_string()),
        PackageSource::Simple("npm:project-missing".to_string()),
        PackageSource::Simple("git:github.com/example/project-repo-a".to_string()),
    ]);
    // Pre-existing checkouts so updateGit reconciles them (upstream mocks updateGit).
    for (root, names) in [
        (
            Path::new(&env.agent_dir)
                .join("git")
                .join("github.com")
                .join("example"),
            vec!["user-repo-a", "user-repo-b", "user-repo-pinned"],
        ),
        (
            Path::new(&env.temp_dir)
                .join(".pi")
                .join("git")
                .join("github.com")
                .join("example"),
            vec!["project-repo-a"],
        ),
    ] {
        for name in names {
            std::fs::create_dir_all(root.join(name)).unwrap();
        }
    }

    let agent_npm_prefix = Path::new(&env.agent_dir)
        .join("npm")
        .to_string_lossy()
        .into_owned();
    let agent_npm_str = agent_npm_prefix.clone();
    let project_npm_prefix = Path::new(&env.temp_dir)
        .join(".pi")
        .join("npm")
        .to_string_lossy()
        .into_owned();
    let project_npm_str = project_npm_prefix.clone();

    let runner = mock_runner(move |call| {
        if call.args.first().map(String::as_str) == Some("view") {
            return match call.args.get(1).map(String::as_str) {
                Some("user-old") | Some("project-old") => Ok(r#""2.0.0""#.to_string()),
                Some("user-current") | Some("project-current") => Ok(r#""1.0.0""#.to_string()),
                Some("user-unknown") => Err("registry unavailable".to_string()),
                other => Err(format!("Unexpected package lookup: {:?}", other)),
            };
        }
        if call.command == "git" {
            // updateGit: fetch/reset flows are exercised elsewhere.
            return Ok(String::new());
        }
        Ok(String::new())
    });
    let pm = make_pm(&env, Vec::new(), runner.clone());

    pm.update(None).await.unwrap();

    let npm_installs: Vec<_> = runner
        .calls_for("npm")
        .into_iter()
        .filter(|(args, _)| args.first().map(String::as_str) == Some("install"))
        .collect();
    assert_eq!(npm_installs.len(), 2);
    assert!(npm_installs.iter().any(|(args, _)| *args
        == vec![
            "install".to_string(),
            "user-old@latest".to_string(),
            "user-unknown@latest".to_string(),
            "--prefix".to_string(),
            agent_npm_str.clone(),
            "--legacy-peer-deps".to_string(),
        ]));
    assert!(npm_installs.iter().any(|(args, _)| *args
        == vec![
            "install".to_string(),
            "project-old@latest".to_string(),
            "project-missing@latest".to_string(),
            "--prefix".to_string(),
            project_npm_str.clone(),
            "--legacy-peer-deps".to_string(),
        ]));
    // Three unpinned + one pinned-ref fetch (upstream expects 4 updateGit calls).
    let git_fetches: Vec<_> = runner
        .calls_for("git")
        .into_iter()
        .filter(|(args, _)| args.first().map(String::as_str) == Some("fetch"))
        .collect();
    assert_eq!(git_fetches.len(), 4);
}
