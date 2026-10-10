//! Tests for `package_manager_cli.rs` — arg parsing, output strings, and the
//! pure managed-install helpers (upstream `parsePackageCommand`, help text,
//! marker/version validation, listing format).

use std::sync::Arc;

use super::{
    builtin_extension_names, config_command_help_text, config_command_usage, cwd_relative_path,
    encode_uri_component, get_package_command_usage, is_managed_release_version,
    is_valid_managed_marker, parse_package_command, render_package_listing,
    update_target_includes_extensions, update_target_includes_self, PackageCommand,
    PackageCommandOptions, UpdateTarget,
};
use crate::core::extensions::types::{ExtensionApiHandle, InlineExtension};
use crate::core::package_manager::ConfiguredPackage;
use crate::core::source_info::SourceScope;

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn parse(values: &[&str]) -> PackageCommandOptions {
    parse_package_command(&args(values)).expect("parsed")
}

fn expect_none(values: &[&str]) {
    assert!(parse_package_command(&args(values)).is_none());
}

fn assert_conflicting(options: &PackageCommandOptions, message: &str) {
    assert_eq!(options.conflicting_options.as_deref(), Some(message));
}

// ============================================================================
// parsePackageCommand — command detection
// ============================================================================

#[test]
fn non_package_commands_are_not_handled() {
    expect_none(&[]);
    expect_none(&["auth"]);
    expect_none(&["mcp"]);
    expect_none(&["installs"]);
    expect_none(&["--help"]);
}

#[test]
fn command_names_and_uninstall_alias() {
    assert_eq!(parse(&["install"]).command, Some(PackageCommand::Install));
    assert_eq!(parse(&["remove"]).command, Some(PackageCommand::Remove));
    assert_eq!(parse(&["uninstall"]).command, Some(PackageCommand::Remove));
    assert_eq!(parse(&["update"]).command, Some(PackageCommand::Update));
    assert_eq!(parse(&["list"]).command, Some(PackageCommand::List));
}

#[test]
fn help_flag() {
    let options = parse(&["install", "--help"]);
    assert!(options.help);
    assert!(!options.local);
    let options = parse(&["update", "-h", "extra"]);
    assert!(options.help);
    assert_eq!(options.source.as_deref(), Some("extra"));
}

#[test]
fn install_parses_source_local_and_trust_override() {
    let options = parse(&["install", "npm:@foo/bar", "-l", "-a"]);
    assert_eq!(options.source.as_deref(), Some("npm:@foo/bar"));
    assert!(options.local);
    assert_eq!(options.project_trust_override, Some(true));
    assert!(options.invalid_option.is_none());
    assert!(options.invalid_argument.is_none());

    let options = parse(&["remove", "./local/path", "--local", "--no-approve"]);
    assert_eq!(options.source.as_deref(), Some("./local/path"));
    assert!(options.local);
    assert_eq!(options.project_trust_override, Some(false));

    let options = parse(&["uninstall", "npm:@foo/bar", "-na"]);
    assert_eq!(options.command, Some(PackageCommand::Remove));
    assert_eq!(options.project_trust_override, Some(false));
}

#[test]
fn local_flag_is_invalid_outside_install_and_remove() {
    let options = parse(&["update", "-l"]);
    assert_eq!(options.invalid_option.as_deref(), Some("-l"));
    assert!(!options.local);
    let options = parse(&["list", "--local"]);
    assert_eq!(options.invalid_option.as_deref(), Some("--local"));
}

#[test]
fn unknown_options_record_only_the_first() {
    let options = parse(&["list", "--bogus", "--worse"]);
    assert_eq!(options.invalid_option.as_deref(), Some("--bogus"));
    let options = parse(&["install", "--force"]);
    assert_eq!(options.invalid_option.as_deref(), Some("--force"));
}

#[test]
fn extra_positional_records_only_the_first() {
    let options = parse(&["install", "a", "b", "c"]);
    assert_eq!(options.source.as_deref(), Some("a"));
    assert_eq!(options.invalid_argument.as_deref(), Some("b"));
    let options = parse(&["list", "foo", "bar"]);
    assert_eq!(options.source.as_deref(), Some("foo"));
    assert_eq!(options.invalid_argument.as_deref(), Some("bar"));
}

// ============================================================================
// parsePackageCommand — --extension
// ============================================================================

#[test]
fn extension_option_takes_one_value() {
    let options = parse(&["update", "--extension", "npm:@foo/bar"]);
    assert_eq!(
        options.update_target,
        Some(UpdateTarget::Extensions {
            source: Some("npm:@foo/bar".to_string())
        })
    );
    assert!(options.missing_option_value.is_none());
}

#[test]
fn extension_option_without_value_is_reported() {
    let options = parse(&["update", "--extension"]);
    assert_eq!(options.missing_option_value.as_deref(), Some("--extension"));
    let options = parse(&["update", "--extension", "--self"]);
    assert_eq!(options.missing_option_value.as_deref(), Some("--extension"));
    // No value was consumed, so `--self` resolved as the update target.
    assert_eq!(options.update_target, Some(UpdateTarget::Self_));
}

#[test]
fn extension_option_twice_conflicts_and_consumes_value() {
    let options = parse(&["update", "--extension", "a", "--extension", "b"]);
    assert_conflicting(&options, "--extension can only be provided once");
    assert_eq!(
        options.update_target,
        Some(UpdateTarget::Extensions {
            source: Some("a".to_string())
        })
    );
    // The second value was consumed, so it is not an unexpected argument.
    assert!(options.invalid_argument.is_none());
}

#[test]
fn extension_option_is_invalid_outside_update() {
    let options = parse(&["install", "--extension", "a"]);
    assert_eq!(options.invalid_option.as_deref(), Some("--extension"));
}

// ============================================================================
// parsePackageCommand — update target resolution
// ============================================================================

#[test]
fn bare_update_defaults_to_self_with_extensions_note() {
    let options = parse(&["update"]);
    assert_eq!(options.update_target, Some(UpdateTarget::Self_));
    assert!(options.show_extensions_skipped_note);
}

#[test]
fn update_self_flag_and_positional_aliases() {
    let options = parse(&["update", "--self"]);
    assert_eq!(options.update_target, Some(UpdateTarget::Self_));
    assert!(!options.show_extensions_skipped_note);
    let options = parse(&["update", "self"]);
    assert_eq!(options.update_target, Some(UpdateTarget::Self_));
    let options = parse(&["update", "pi"]);
    assert_eq!(options.update_target, Some(UpdateTarget::Self_));
    let options = parse(&["update", "pi", "--extensions"]);
    assert_eq!(options.update_target, Some(UpdateTarget::All));
}

#[test]
fn update_extensions_flag_only() {
    let options = parse(&["update", "--extensions"]);
    assert_eq!(
        options.update_target,
        Some(UpdateTarget::Extensions { source: None })
    );
    assert!(!options.show_extensions_skipped_note);
}

#[test]
fn update_self_and_extensions_combine_to_all() {
    let options = parse(&["update", "--self", "--extensions"]);
    assert_eq!(options.update_target, Some(UpdateTarget::All));
}

#[test]
fn update_all_flag() {
    let options = parse(&["update", "--all"]);
    assert_eq!(options.update_target, Some(UpdateTarget::All));
}

#[test]
fn update_models_flag() {
    let options = parse(&["update", "--models"]);
    assert_eq!(options.update_target, Some(UpdateTarget::Models));
}

#[test]
fn update_positional_source_targets_one_extension() {
    let options = parse(&["update", "npm:@foo/bar"]);
    assert_eq!(
        options.update_target,
        Some(UpdateTarget::Extensions {
            source: Some("npm:@foo/bar".to_string())
        })
    );
}

#[test]
fn update_all_conflicts() {
    let options = parse(&["update", "--all", "--self"]);
    assert_conflicting(
        &options,
        "--all cannot be combined with --self, --extensions, --models, or --extension",
    );
    let options = parse(&["update", "--all", "npm:x"]);
    assert_conflicting(
        &options,
        "--all cannot be combined with a positional source",
    );
    // The flags conflict wins when both apply.
    let options = parse(&["update", "--all", "--models", "npm:x"]);
    assert_conflicting(
        &options,
        "--all cannot be combined with --self, --extensions, --models, or --extension",
    );
}

#[test]
fn update_models_conflicts() {
    let options = parse(&["update", "--models", "--self"]);
    assert_conflicting(
        &options,
        "--models cannot be combined with --self, --extensions, --all, or --extension",
    );
    let options = parse(&["update", "--models", "npm:x"]);
    assert_conflicting(
        &options,
        "--models cannot be combined with a positional source",
    );
}

#[test]
fn update_extension_conflicts() {
    let options = parse(&["update", "--extension", "a", "--self"]);
    assert_conflicting(
        &options,
        "--extension cannot be combined with --self, --extensions, or --all",
    );
    let options = parse(&["update", "--extension", "a", "b"]);
    assert_conflicting(
        &options,
        "--extension cannot be combined with a positional source",
    );
}

#[test]
fn update_positional_conflicts_with_flags() {
    let options = parse(&["update", "npm:x", "--self"]);
    assert_conflicting(
        &options,
        "positional update targets cannot be combined with --self, --extensions, or --all",
    );
    let options = parse(&["update", "npm:x", "--all"]);
    assert_conflicting(
        &options,
        "--all cannot be combined with a positional source",
    );
}

#[test]
fn update_target_includes_helpers() {
    assert!(update_target_includes_self(&UpdateTarget::All));
    assert!(update_target_includes_self(&UpdateTarget::Self_));
    assert!(!update_target_includes_self(&UpdateTarget::Extensions {
        source: None
    }));
    assert!(!update_target_includes_self(&UpdateTarget::Models));
    assert!(update_target_includes_extensions(&UpdateTarget::All));
    assert!(update_target_includes_extensions(
        &UpdateTarget::Extensions { source: None }
    ));
    assert!(!update_target_includes_extensions(&UpdateTarget::Self_));
    assert!(!update_target_includes_extensions(&UpdateTarget::Models));
}

// ============================================================================
// Usage and help text
// ============================================================================

#[test]
fn package_command_usage_strings() {
    assert_eq!(
        get_package_command_usage(PackageCommand::Install),
        "pi install <source> [-l] [--approve|--no-approve]"
    );
    assert_eq!(
        get_package_command_usage(PackageCommand::Remove),
        "pi remove <source> [-l] [--approve|--no-approve]"
    );
    assert_eq!(
		get_package_command_usage(PackageCommand::Update),
		"pi update [source|self|pi] [--self|--extensions|--models|--all] [--extension <source>] [--approve|--no-approve] [--force]"
	);
    assert_eq!(
        get_package_command_usage(PackageCommand::List),
        "pi list [--approve|--no-approve]"
    );
    assert_eq!(
        config_command_usage(),
        "pi config [-l] [--approve|--no-approve]"
    );
}

#[test]
fn install_help_text() {
    assert_eq!(
		super::package_command_help_text(PackageCommand::Install),
		"Usage:\n  pi install <source> [-l] [--approve|--no-approve]\n\nInstall a package and add it to settings.\n\nOptions:\n  -l, --local       Install project-locally (.pi/settings.json)\n  -a, --approve     Trust project-local files for this command\n  -na, --no-approve Ignore project-local files for this command\n\nExamples:\n  pi install npm:@foo/bar\n  pi install git:github.com/user/repo\n  pi install git:git@github.com:user/repo\n  pi install https://github.com/user/repo\n  pi install ssh://git@github.com:user/repo\n  pi install ./local/path\n"
	);
}

#[test]
fn remove_help_text() {
    assert_eq!(
		super::package_command_help_text(PackageCommand::Remove),
		"Usage:\n  pi remove <source> [-l] [--approve|--no-approve]\n\nRemove a package and its source from settings.\nAlias: pi uninstall <source> [-l]\n\nOptions:\n  -l, --local       Remove from project settings (.pi/settings.json)\n  -a, --approve     Trust project-local files for this command\n  -na, --no-approve Ignore project-local files for this command\n\nExamples:\n  pi remove npm:@foo/bar\n  pi uninstall npm:@foo/bar\n"
	);
}

#[test]
fn update_help_text() {
    assert_eq!(
		super::package_command_help_text(PackageCommand::Update),
		"Usage:\n  pi update [source|self|pi] [--self|--extensions|--models|--all] [--extension <source>] [--approve|--no-approve] [--force]\n\nUpdate pi, installed packages, or model catalogs.\n\nOptions:\n  --self                  Update pi only (default when no target is given)\n  --extensions            Update installed packages only\n  --models                Refresh model catalogs only\n  --all                   Update pi and installed packages\n  --extension <source>    Update one package only\n  -a, --approve           Trust project-local files for this command\n  -na, --no-approve       Ignore project-local files for this command\n  --force                 Reinstall pi even if the current version is latest\n\nShort forms:\n  pi update                Update pi only\n  pi update --all          Update pi and all extensions\n  pi update --models       Refresh model catalogs only\n  pi update <source>       Update one package\n  pi update pi             Update pi only (self works as alias to pi)\n"
	);
}

#[test]
fn list_help_text() {
    assert_eq!(
		super::package_command_help_text(PackageCommand::List),
		"Usage:\n  pi list [--approve|--no-approve]\n\nList installed packages from user and project settings.\n\nOptions:\n  -a, --approve      Trust project-local files for this command\n  -na, --no-approve  Ignore project-local files for this command\n"
	);
}

#[test]
fn config_help_text() {
    assert_eq!(
		config_command_help_text(),
		"Usage:\n  pi config [-l] [--approve|--no-approve]\n\nOpen the resource configuration TUI to enable or disable package resources.\nWithout -l, starts in global settings (~/.pi/agent/settings.json).\nPress Tab in the TUI to switch between global and project-local modes.\n\nOptions:\n  -l, --local       Edit project overrides (.pi/settings.json)\n  -a, --approve     Trust project-local files for this command with -l\n  -na, --no-approve Ignore project-local files for this command with -l\n"
	);
}

// ============================================================================
// Managed-install helpers
// ============================================================================

#[test]
fn managed_release_version_validation() {
    for valid in [
        "1.2.3",
        "0.0.0",
        "01.2.3",
        "1.2.3-beta.1",
        "1.2.3+build.1",
        "1.2.3-rc.1+meta.1",
        "1.2.3-a-b",
    ] {
        assert!(is_managed_release_version(valid), "{valid} should be valid");
    }
    for invalid in [
        "",
        "1",
        "1.2",
        "1.2.",
        "v1.2.3",
        "1.2.3.4",
        "1.2.3-",
        "1.2.3+",
        "1.2.3-+",
        "1.2.3 beta",
        "1..3",
    ] {
        assert!(
            !is_managed_release_version(invalid),
            "{invalid} should be invalid"
        );
    }
}

#[test]
fn managed_marker_validation() {
    assert!(is_valid_managed_marker(
        r#"{"kind":"pi-managed-install","schemaVersion":1,"layout":"releases-v1"}"#
    ));
    assert!(is_valid_managed_marker(
        r#"{"layout":"releases-v1","kind":"pi-managed-install","schemaVersion":1,"extra":true}"#
    ));
    // Wrong field values.
    assert!(!is_valid_managed_marker(
        r#"{"kind":"other","schemaVersion":1,"layout":"releases-v1"}"#
    ));
    assert!(!is_valid_managed_marker(
        r#"{"kind":"pi-managed-install","schemaVersion":2,"layout":"releases-v1"}"#
    ));
    assert!(!is_valid_managed_marker(
        r#"{"kind":"pi-managed-install","schemaVersion":1,"layout":"releases-v2"}"#
    ));
    // schemaVersion must be the number 1.
    assert!(!is_valid_managed_marker(
        r#"{"kind":"pi-managed-install","schemaVersion":"1","layout":"releases-v1"}"#
    ));
    // Not JSON / not an object.
    assert!(!is_valid_managed_marker("not json"));
    assert!(!is_valid_managed_marker("[]"));
}

#[test]
fn encode_uri_component_matches_node() {
    assert_eq!(encode_uri_component("1.2.3"), "1.2.3");
    assert_eq!(encode_uri_component("1.2.3+build.1"), "1.2.3%2Bbuild.1");
    assert_eq!(encode_uri_component("a b&c/d?e#f"), "a%20b%26c%2Fd%3Fe%23f");
    assert_eq!(encode_uri_component("-_.!~*'()"), "-_.!~*'()");
}

#[test]
fn cwd_relative_path_semantics() {
    let base = "/tmp/pi-managed-test/releases";
    assert_eq!(
        cwd_relative_path("/tmp/pi-managed-test/releases/pkg", base).as_deref(),
        Some("pkg")
    );
    assert_eq!(
        cwd_relative_path("/tmp/pi-managed-test/releases", base).as_deref(),
        Some(".")
    );
    assert_eq!(cwd_relative_path("/elsewhere/pkg", base), None);
}

// ============================================================================
// Listing output
// ============================================================================

fn configured(
    source: &str,
    scope: SourceScope,
    filtered: bool,
    installed_path: Option<&str>,
) -> ConfiguredPackage {
    ConfiguredPackage {
        source: source.to_string(),
        scope,
        filtered,
        installed_path: installed_path.map(|path| path.to_string()),
    }
}

#[test]
fn listing_renders_user_then_project_sections() {
    let packages = vec![
        configured(
            "npm:@foo/bar",
            SourceScope::User,
            false,
            Some("/agent/extensions/@foo/bar"),
        ),
        configured("git:github.com/user/repo", SourceScope::User, true, None),
        configured(
            "npm:baz",
            SourceScope::Project,
            false,
            Some("/project/.pi/extensions/baz"),
        ),
    ];
    assert_eq!(
		render_package_listing(&packages),
		"User packages:\n  npm:@foo/bar\n    /agent/extensions/@foo/bar\n  git:github.com/user/repo (filtered)\n\nProject packages:\n  npm:baz\n    /project/.pi/extensions/baz\n"
	);
}

#[test]
fn listing_renders_project_only_without_separator() {
    let packages = vec![configured("npm:baz", SourceScope::Project, false, None)];
    assert_eq!(
        render_package_listing(&packages),
        "Project packages:\n  npm:baz\n"
    );
}

#[test]
fn listing_renders_user_only() {
    let packages = vec![configured("npm:x", SourceScope::User, false, None)];
    assert_eq!(
        render_package_listing(&packages),
        "User packages:\n  npm:x\n"
    );
}

// ============================================================================
// Settings manager / builtin extensions
// ============================================================================

#[test]
fn command_settings_manager_defaults_to_trusted_and_honors_override() {
    let temp = std::env::temp_dir().join(format!("pi-pm-cli-test-{}", std::process::id()));
    let cwd = temp.join("cwd").to_string_lossy().to_string();
    let agent_dir = temp.join("agent").to_string_lossy().to_string();
    std::fs::create_dir_all(&cwd).unwrap();

    // D134: the deferred trust store defaults to trusted (crate D103 stance).
    let default_trusted = super::create_command_settings_manager(super::CommandSettingsOptions {
        cwd: &cwd,
        agent_dir: &agent_dir,
        project_trust_override: None,
        use_saved_project_trust_only: false,
    });
    assert!(default_trusted.settings_manager.lock().is_project_trusted());
    assert!(default_trusted.project_trust_warnings.is_empty());

    let approved = super::create_command_settings_manager(super::CommandSettingsOptions {
        cwd: &cwd,
        agent_dir: &agent_dir,
        project_trust_override: Some(true),
        use_saved_project_trust_only: true,
    });
    assert!(approved.settings_manager.lock().is_project_trusted());

    let denied = super::create_command_settings_manager(super::CommandSettingsOptions {
        cwd: &cwd,
        agent_dir: &agent_dir,
        project_trust_override: Some(false),
        use_saved_project_trust_only: false,
    });
    assert!(!denied.settings_manager.lock().is_project_trusted());

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn builtin_extension_names_filter() {
    let factory: crate::core::extensions::types::ExtensionFactory =
        Arc::new(|_api: ExtensionApiHandle| Box::pin(async { Ok::<(), String>(()) }) as _);
    let extensions = vec![
        InlineExtension::Named {
            name: "built-in-one".to_string(),
            factory: factory.clone(),
            hidden: None,
            replaceable: None,
            builtin: Some(true),
        },
        InlineExtension::Named {
            name: "plain-inline".to_string(),
            factory: factory.clone(),
            hidden: None,
            replaceable: None,
            builtin: None,
        },
        InlineExtension::Named {
            name: "built-in-two".to_string(),
            factory,
            hidden: None,
            replaceable: None,
            builtin: Some(true),
        },
    ];
    assert_eq!(
        builtin_extension_names(&extensions),
        vec!["built-in-one", "built-in-two"]
    );
    assert_eq!(builtin_extension_names(&[]), Vec::<String>::new());
}
