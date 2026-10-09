//! Port of `test/resolve-config-value.test.ts` (1:1; the platform-mock case
//! exercises the stdin-transport path through the shell-config override, see
//! module docs).

use std::fs;
use std::sync::LazyLock;

use parking_lot::Mutex;

use super::{
	clear_config_value_cache, resolve_config_value, resolve_config_value_uncached, set_shell_config_override,
};
use crate::utils::shell::{CommandTransport, ShellConfig};

/// Serializes tests that mutate the process environment or the command cache
/// (upstream runs them sequentially in one process).
static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(Mutex::default);

fn temp_dir(name: &str) -> String {
	let dir = std::env::temp_dir().join(format!("pi-config-value-{}-{}", name, std::process::id()));
	let _ = fs::remove_dir_all(&dir);
	fs::create_dir_all(&dir).unwrap();
	dir.to_string_lossy().to_string()
}

#[test]
fn resolves_literals_environment_templates_and_escapes() {
		let _env_guard = crate::utils::test_env_lock();
	let _guard = ENV_LOCK.lock();
	std::env::set_var("TEST_CONFIG_LEFT", "left");
	std::env::set_var("TEST_CONFIG_RIGHT", "right");
	let result = std::panic::catch_unwind(|| {
		assert_eq!(resolve_config_value("literal-key", None), Some("literal-key".to_string()));
		assert_eq!(resolve_config_value("$TEST_CONFIG_LEFT", None), Some("left".to_string()));
		assert_eq!(
			resolve_config_value("${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT", None),
			Some("left_right".to_string())
		);
		assert_eq!(resolve_config_value("$$TEST_CONFIG_LEFT", None), Some("$TEST_CONFIG_LEFT".to_string()));
		assert_eq!(resolve_config_value("$!literal-$TEST_CONFIG_RIGHT", None), Some("!literal-right".to_string()));
	});
	std::env::remove_var("TEST_CONFIG_LEFT");
	std::env::remove_var("TEST_CONFIG_RIGHT");
	result.unwrap();
}

#[test]
fn uses_credential_scoped_environment_before_process_env() {
		let _env_guard = crate::utils::test_env_lock();
	let _guard = ENV_LOCK.lock();
	std::env::set_var("TEST_CONFIG_SCOPED", "process");
	let result = std::panic::catch_unwind(|| {
		let env = [("TEST_CONFIG_SCOPED".to_string(), "credential".to_string())].into_iter().collect();
		assert_eq!(resolve_config_value("$TEST_CONFIG_SCOPED", Some(&env)), Some("credential".to_string()));
	});
	std::env::remove_var("TEST_CONFIG_SCOPED");
	result.unwrap();
}

#[test]
fn executes_shell_commands_and_trims_their_output() {
	assert_eq!(resolve_config_value("!echo '  spaced-key  '", None), Some("spaced-key".to_string()));
	assert_eq!(resolve_config_value("!printf 'line1\\nline2'", None), Some("line1\nline2".to_string()));
	assert_eq!(resolve_config_value("!echo 'hello world' | tr ' ' '-'", None), Some("hello-world".to_string()));
}

#[test]
fn returns_undefined_when_command_resolution_fails() {
	for command in ["!exit 1", "!nonexistent-command-12345", "!printf ''"] {
		assert_eq!(resolve_config_value(command, None), None, "command: {command}");
	}
}

#[test]
fn caches_successful_and_failed_commands_until_explicitly_cleared() {
	let _guard = ENV_LOCK.lock();
	let temp_dir = temp_dir("counter");
	let counter_file = format!("{temp_dir}/counter");
	fs::write(&counter_file, "0").unwrap();
	let escaped_path = counter_file.replace('\\', "/");
	let success =
		format!("!sh -c 'count=$(cat \"{escaped_path}\"); echo $((count + 1)) > \"{escaped_path}\"; echo value'");

	assert_eq!(resolve_config_value(&success, None), Some("value".to_string()));
	assert_eq!(resolve_config_value(&success, None), Some("value".to_string()));
	assert_eq!(fs::read_to_string(&counter_file).unwrap().trim(), "1");

	clear_config_value_cache();
	assert_eq!(resolve_config_value(&success, None), Some("value".to_string()));
	assert_eq!(fs::read_to_string(&counter_file).unwrap().trim(), "2");

	let failure =
		format!("!sh -c 'count=$(cat \"{escaped_path}\"); echo $((count + 1)) > \"{escaped_path}\"; exit 1'");
	assert_eq!(resolve_config_value(&failure, None), None);
	assert_eq!(resolve_config_value(&failure, None), None);
	assert_eq!(fs::read_to_string(&counter_file).unwrap().trim(), "3");

	let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn does_not_cache_environment_values() {
		let _env_guard = crate::utils::test_env_lock();
	let _guard = ENV_LOCK.lock();
	std::env::set_var("TEST_CONFIG_DYNAMIC", "first");
	let result = std::panic::catch_unwind(|| {
		assert_eq!(resolve_config_value("$TEST_CONFIG_DYNAMIC", None), Some("first".to_string()));
		std::env::set_var("TEST_CONFIG_DYNAMIC", "second");
		assert_eq!(resolve_config_value("$TEST_CONFIG_DYNAMIC", None), Some("second".to_string()));
	});
	std::env::remove_var("TEST_CONFIG_DYNAMIC");
	result.unwrap();
}

#[test]
fn uncached_resolution_executes_a_command_on_every_call() {
	let temp_dir = temp_dir("uncached-counter");
	let counter_file = format!("{temp_dir}/uncached-counter");
	fs::write(&counter_file, "0").unwrap();
	let escaped_path = counter_file.replace('\\', "/");
	let command =
		format!("!sh -c 'count=$(cat \"{escaped_path}\"); echo $((count + 1)) > \"{escaped_path}\"; echo value'");
	assert_eq!(resolve_config_value_uncached(&command, None), Some("value".to_string()));
	assert_eq!(resolve_config_value_uncached(&command, None), Some("value".to_string()));
	assert_eq!(fs::read_to_string(&counter_file).unwrap().trim(), "2");
	let _ = fs::remove_dir_all(&temp_dir);
}

/// Upstream mocks `getShellConfig` to a stdin-transport shell and fakes
/// `process.platform === "win32"`; the port exercises the configured-shell
/// path directly through the override seam (module docs).
#[test]
fn uses_stdin_when_the_configured_shell_requires_it() {
	set_shell_config_override(Some(ShellConfig {
		shell: "/bin/sh".to_string(),
		args: vec!["-s".to_string()],
		command_transport: Some(CommandTransport::Stdin),
	}));
	let result = std::panic::catch_unwind(|| {
		assert_eq!(
			resolve_config_value_uncached("!name='World'; echo \"Hello, ${name}!\"", None),
			Some("Hello, World!".to_string())
		);
	});
	set_shell_config_override(None);
	result.unwrap();
}
