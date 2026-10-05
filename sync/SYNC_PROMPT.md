# pi-rs release sync task

You are the pi-rs release-sync agent for the Rust port of earendil-works/pi in this
repository (the cwd). Read `PORT_MAP.md` first — it is the binding contract: crate
mapping, module-mirror rules, test-transport rules, deviation policy, status ledger.

## Steps

1. Run `sync/fetch-upstream.sh <NEW_TAG>` to check out the release under `upstream/pi`.
2. Scope the change: `git -C upstream/pi diff --stat <PREV_TAG>..<NEW_TAG> -- packages/`
   then read the full diff per affected file.
3. Apply each upstream change to its mapped crate following PORT_MAP.md rules:
   - Module layout mirrors upstream 1:1 (rule 1); wire types stay serde-compatible (rule 2).
   - Port behavior faithfully; list every intentional deviation in the ledger (rule 3).
   - Port/adjust tests alongside code (rule 4).
   - JS-runtime-specific surface goes behind `pi-runtime` traits, never inline (rule 5).
   - Changes to `packages/tui` are out of scope — note and skip.
   - A brand-new upstream package needs a new crate + a PORT_MAP.md row — add both.
4. Run `cargo test --workspace` and `cargo clippy --workspace --all-targets` (clippy may
   warn; tests must pass).
5. Update the PORT_MAP.md status ledger: set affected crates' "Upstream tag ported" to
   `<NEW_TAG>` only if the crate is fully in sync with that tag.
6. If ALL crates in the ledger are at `<NEW_TAG>`, bump `[workspace.package] version` to
   the tag without the leading `v` (rule 6).
7. Commit everything (including PORT_MAP.md) as one commit: `sync: upstream <NEW_TAG>`.
   Do NOT push.

## Hard rules

- Never edit `upstream/pi` contents. Never edit `sync/` scripts.
- If the diff contains something you cannot map confidently (protocol break, new JS
  runtime coupling, upstream refactor of unmapped code), STOP before committing, leave
  the worktree clean, and report exactly what is unmappable and why. Do not improvise
  replacements.
- If `cargo test` fails and you cannot fix it confidently, revert your changes
  (`git checkout -- crates/ PORT_MAP.md Cargo.toml`), leave the state file untouched,
  and report the failing tests. The watcher will retry.
