//! `core/keybindings.ts` migration subset (`migrateKeybindingsConfig`).
//!
//! **D45/D100:** the full `KeybindingsManager` and default-key tables live
//! behind the pi-tui exclusion; only the legacy-name migration and config
//! ordering are ported. Ordering follows the `app.*`/`tui.*` key lists
//! below; unknown keys sort alphabetically afterwards (upstream orders by
//! the full `KEYBINDINGS` definition table, including pi-tui entries not
//! ported here). **D132:** `core/keybindings.ts` itself — the TUI-coupled
//! `KeybindingsManager` over pi-tui — is deferred with the interactive mode.

use std::collections::BTreeMap;

/// Legacy name → namespaced id (`KEYBINDING_NAME_MIGRATIONS`).
pub static KEYBINDING_NAME_MIGRATIONS: [(&str, &str); 59] = [
	("cursorUp", "tui.editor.cursorUp"),
	("cursorDown", "tui.editor.cursorDown"),
	("cursorLeft", "tui.editor.cursorLeft"),
	("cursorRight", "tui.editor.cursorRight"),
	("cursorWordLeft", "tui.editor.cursorWordLeft"),
	("cursorWordRight", "tui.editor.cursorWordRight"),
	("cursorLineStart", "tui.editor.cursorLineStart"),
	("cursorLineEnd", "tui.editor.cursorLineEnd"),
	("jumpForward", "tui.editor.jumpForward"),
	("jumpBackward", "tui.editor.jumpBackward"),
	("pageUp", "tui.editor.pageUp"),
	("pageDown", "tui.editor.pageDown"),
	("deleteCharBackward", "tui.editor.deleteCharBackward"),
	("deleteCharForward", "tui.editor.deleteCharForward"),
	("deleteWordBackward", "tui.editor.deleteWordBackward"),
	("deleteWordForward", "tui.editor.deleteWordForward"),
	("deleteToLineStart", "tui.editor.deleteToLineStart"),
	("deleteToLineEnd", "tui.editor.deleteToLineEnd"),
	("yank", "tui.editor.yank"),
	("yankPop", "tui.editor.yankPop"),
	("undo", "tui.editor.undo"),
	("newLine", "tui.input.newLine"),
	("submit", "tui.input.submit"),
	("tab", "tui.input.tab"),
	("copy", "tui.input.copy"),
	("selectUp", "tui.select.up"),
	("selectDown", "tui.select.down"),
	("selectPageUp", "tui.select.pageUp"),
	("selectPageDown", "tui.select.pageDown"),
	("selectConfirm", "tui.select.confirm"),
	("selectCancel", "tui.select.cancel"),
	("interrupt", "app.interrupt"),
	("clear", "app.clear"),
	("exit", "app.exit"),
	("suspend", "app.suspend"),
	("cycleThinkingLevel", "app.thinking.cycle"),
	("cycleModelForward", "app.model.cycleForward"),
	("cycleModelBackward", "app.model.cycleBackward"),
	("selectModel", "app.model.select"),
	("expandTools", "app.tools.expand"),
	("toggleThinking", "app.thinking.toggle"),
	("toggleSessionNamedFilter", "app.session.toggleNamedFilter"),
	("externalEditor", "app.editor.external"),
	("followUp", "app.message.followUp"),
	("dequeue", "app.message.dequeue"),
	("pasteImage", "app.clipboard.pasteImage"),
	("newSession", "app.session.new"),
	("tree", "app.session.tree"),
	("fork", "app.session.fork"),
	("resume", "app.session.resume"),
	("treeFoldOrUp", "app.tree.foldOrUp"),
	("treeUnfoldOrDown", "app.tree.unfoldOrDown"),
	("treeEditLabel", "app.tree.editLabel"),
	("treeToggleLabelTimestamp", "app.tree.toggleLabelTimestamp"),
	("toggleSessionPath", "app.session.togglePath"),
	("toggleSessionSort", "app.session.toggleSort"),
	("renameSession", "app.session.rename"),
	("deleteSession", "app.session.delete"),
	("deleteSessionNoninvasive", "app.session.deleteNoninvasive"),
];

/// Known keybinding ids for output ordering (`Object.keys(KEYBINDINGS)`).
pub static KEYBINDING_IDS: [&str; 65] = [
	// tui.editor.* / tui.input.* / tui.select.* / tui.altScreen.* (pi-tui
	// table, ordered as referenced by the migration table)
	"tui.editor.cursorUp",
	"tui.editor.cursorDown",
	"tui.editor.cursorLeft",
	"tui.editor.cursorRight",
	"tui.editor.cursorWordLeft",
	"tui.editor.cursorWordRight",
	"tui.editor.cursorLineStart",
	"tui.editor.cursorLineEnd",
	"tui.editor.jumpForward",
	"tui.editor.jumpBackward",
	"tui.editor.pageUp",
	"tui.editor.pageDown",
	"tui.editor.deleteCharBackward",
	"tui.editor.deleteCharForward",
	"tui.editor.deleteWordBackward",
	"tui.editor.deleteWordForward",
	"tui.editor.deleteToLineStart",
	"tui.editor.deleteToLineEnd",
	"tui.editor.yank",
	"tui.editor.yankPop",
	"tui.editor.undo",
	"tui.input.newLine",
	"tui.input.submit",
	"tui.input.tab",
	"tui.input.copy",
	"tui.select.up",
	"tui.select.down",
	"tui.select.pageUp",
	"tui.select.pageDown",
	"tui.select.confirm",
	"tui.select.cancel",
	// app.* (coding-agent table, definition order)
	"app.interrupt",
	"app.clear",
	"app.exit",
	"app.suspend",
	"app.thinking.cycle",
	"app.thinking.save",
	"app.model.cycleForward",
	"app.model.cycleBackward",
	"app.model.select",
	"app.tools.expand",
	"app.thinking.toggle",
	"app.session.toggleNamedFilter",
	"app.editor.external",
	"app.message.copy",
	"app.message.followUp",
	"app.message.dequeue",
	"app.clipboard.pasteImage",
	"app.session.new",
	"app.session.tree",
	"app.session.fork",
	"app.session.resume",
	"app.tree.foldOrUp",
	"app.tree.unfoldOrDown",
	"app.tree.editLabel",
	"app.tree.toggleLabelTimestamp",
	"app.session.togglePath",
	"app.session.toggleSort",
	"app.session.rename",
	"app.session.delete",
	"app.session.deleteNoninvasive",
	"app.models.save",
	"app.models.enableAll",
	"app.models.clearAll",
	"app.models.toggleProvider",
];

fn legacy_id(key: &str) -> Option<&'static str> {
	KEYBINDING_NAME_MIGRATIONS.iter().find(|(legacy, _)| *legacy == key).map(|(_, id)| *id)
}

/// `orderKeybindingsConfig` — known ids in table order, then the rest
/// sorted.
pub fn order_keybindings_config(
	config: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
	let mut ordered = serde_json::Map::new();
	for id in KEYBINDING_IDS {
		if let Some(value) = config.get(id) {
			ordered.insert(id.to_string(), value.clone());
		}
	}

	let extras: BTreeMap<String, serde_json::Value> = config
		.into_iter()
		.filter(|(key, _)| !ordered.contains_key(key))
		.collect();
	for (key, value) in extras.into_iter() {
		ordered.insert(key, value);
	}

	ordered
}

/// `migrateKeybindingsConfig` — rewrite legacy key names to namespaced ids.
/// Returns the migrated config and whether anything changed.
pub fn migrate_keybindings_config(
	raw_config: &serde_json::Map<String, serde_json::Value>,
) -> (serde_json::Map<String, serde_json::Value>, bool) {
	let mut config = serde_json::Map::new();
	let mut migrated = false;

	for (key, value) in raw_config {
		let next_key = legacy_id(key).unwrap_or(key);
		if next_key != key {
			migrated = true;
		}
		if next_key != key && raw_config.contains_key(next_key) {
			migrated = true;
			continue;
		}
		config.insert(next_key.to_string(), value.clone());
	}

	(order_keybindings_config(config), migrated)
}

#[cfg(test)]
mod tests {
	//! Ported from `test/keybindings-migration.test.ts` (the
	//! `KeybindingsManager` in-memory test is excluded with the manager
	//! itself).
	use super::*;
	use crate::migrations::run_migrations_with_agent_dir;
	use serde_json::json;

	static NEXT_AGENT_DIR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

	fn create_agent_dir(config: serde_json::Value) -> String {
		let agent_dir = std::env::temp_dir().join(format!(
			"pi-keybindings-test-{}-{}",
			std::process::id(),
			NEXT_AGENT_DIR_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
		));
		let _ = std::fs::remove_dir_all(&agent_dir);
		std::fs::create_dir_all(&agent_dir).unwrap();
		std::fs::write(
			agent_dir.join("keybindings.json"),
			format!("{}\n", serde_json::to_string_pretty(&config).unwrap()),
		)
		.unwrap();
		agent_dir.to_string_lossy().to_string()
	}

	fn read_migrated(agent_dir: &str) -> serde_json::Value {
		let content = std::fs::read_to_string(std::path::Path::new(agent_dir).join("keybindings.json")).unwrap();
		serde_json::from_str(&content).unwrap()
	}

	#[test]
	fn rewrites_old_key_names_to_namespaced_ids() {
		let agent_dir = create_agent_dir(json!({
			"cursorUp": ["up", "ctrl+p"],
			"expandTools": "ctrl+x",
		}));
		// Upstream points ENV_AGENT_DIR at the temp dir; the port passes the
		// agent dir explicitly to avoid process-global env races between
		// parallel test threads.
		run_migrations_with_agent_dir(&agent_dir, ".");

		let migrated = read_migrated(&agent_dir);
		assert_eq!(
			migrated,
			json!({
				"tui.editor.cursorUp": ["up", "ctrl+p"],
				"app.tools.expand": "ctrl+x",
			})
		);
	}

	#[test]
	fn keeps_the_namespaced_value_when_old_and_new_names_both_exist() {
		let agent_dir = create_agent_dir(json!({
			"expandTools": "ctrl+x",
			"app.tools.expand": "ctrl+y",
		}));
		run_migrations_with_agent_dir(&agent_dir, ".");

		let migrated = read_migrated(&agent_dir);
		assert_eq!(migrated, json!({ "app.tools.expand": "ctrl+y" }));
	}

	#[test]
	fn migrate_config_is_pure() {
		let raw = serde_json::json!({ "selectConfirm": "enter", "unknown": 1 });
		let (config, migrated) = migrate_keybindings_config(raw.as_object().unwrap());
		assert!(migrated);
		assert_eq!(
			serde_json::Value::Object(config),
			json!({ "tui.select.confirm": "enter", "unknown": 1 })
		);
	}
}
