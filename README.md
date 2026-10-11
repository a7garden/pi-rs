# pi-rs

[![Release](https://github.com/a7garden/pi-rs/actions/workflows/release.yml/badge.svg)](https://github.com/a7garden/pi-rs/actions/workflows/release.yml)

Rust port of [pi](https://github.com/earendil-works/pi) — a minimal, extensible
AI coding agent. The port tracks upstream releases automatically: when a new
upstream tag lands, the diff is applied crate-by-crate, the full test suite
runs, and a release with native binaries is published here.

Current upstream baseline: **v1.1.0**.

## What is ported

- **Headless agent surface**: `--print`, `--mode json`, and `--mode rpc`
  (protocol modes for tooling), sessions (`--continue`, `--session`,
  `--session-id`, `--fork`), compaction, skills, prompt templates,
  AGENTS.md/CLAUDE.md context loading, project trust.
- **~30 LLM providers** through `pi-ai`, including OAuth flows and a virtual
  model router.
- **Extensions**: upstream TypeScript extensions load unchanged through an
  embedded [deno_core] JS host (`pi-runtime`) — the same runtime `pi` uses for
  MCP, codemode, and chord facets.
- **MCP**: servers, OAuth sign-in (`pi mcp login`), tools, resources.
- **CLI management**: `pi install/remove/uninstall/update/list`,
  `pi auth`, HTML session export (`pi --export`).
- **Durable + env**: upstream `durable` harness/session storage and the `env`
  remote-daemon package.

The interactive TUI is **excluded by design** — the port targets embedded and
headless use. `pi config`'s TUI selector and `--resume`'s interactive session
picker are the only user-visible gaps.

## Install

Grab a tarball from [Releases](https://github.com/a7garden/pi-rs/releases)
(`pi-darwin-arm64`, `pi-darwin-x64`, `pi-linux-arm64`, `pi-linux-x64`):

```sh
tar xzf pi-darwin-arm64.tar.gz
./pi --version
```

Or build from source (Rust stable + a C/C++ toolchain — the embedded V8 is
compiled by `deno_core`):

```sh
cargo build --release -p pi-coding-agent --bin pi
```

## Usage

```sh
pi -p "explain this repo"            # one-shot print mode
pi --mode json                       # JSON event protocol on stdout
pi --mode rpc                        # JSON-RPC control surface
pi mcp list                          # inspect configured MCP servers
pi mcp login <server>                # OAuth sign-in outside a session
pi --export session.jsonl out.html   # session → standalone HTML
pi install <source>                  # install an extension package
pi auth check --provider anthropic     # provider credential readiness
pi auth print-api-key --provider openai
```

Run `pi --help` for the full surface (tool selection, model patterns, thinking
levels, session flags, project-trust overrides).

## Layout

| Crate | Upstream package | Role |
|---|---|---|
| `pi-telemetry` | telemetry | tracing abstractions |
| `pi-chord` | chord | remote control-plane protocol |
| `pi-codemode` | codemode | sandboxed JS code execution |
| `pi-mcp` | mcp | Model Context Protocol client |
| `pi-protocol` | protocol | wire types (CBOR/JSON) |
| `pi-ai` | ai | ~30 providers, OAuth, streaming |
| `pi-agent` | agent | agent loop + state machine |
| `pi-server` | server | multi-session server |
| `pi-durable` | durable | durable harness/session storage |
| `pi-client` | client | server client |
| `pi-env` | env | remote environment daemon |
| `pi-runtime` | — | embedded deno_core JS host (new) |
| `pi-coding-agent` | coding-agent | `pi` CLI, core, tools, extensions |

`PORT_MAP.md` is the binding contract: module mapping, porting rules, and the
per-crate sync ledger (including every documented deviation from upstream).

## Automated release sync

An hourly watcher (`sync/check-release.sh`) polls upstream releases. On a new
tag it launches an automated porting session (`sync/SYNC_PROMPT.md`) that
applies the upstream diff under the PORT_MAP rules, runs
`cargo test --workspace` + clippy, updates the ledger, commits, pushes, and
tags — the tag triggers the release workflow that publishes the binaries.
Failed syncs leave the state untouched and retry on the next interval.

## Development

```sh
cargo test --workspace          # full suite (parity tests included)
cargo test -p pi-coding-agent   # largest crate
```

Upstream sources are checked out under `upstream/pi` (gitignored) for
diffing; never edit them.

## License

MIT — pi is © 2025 Mario Zechner ([earendil-works/pi](https://github.com/earendil-works/pi));
this port inherits the same license. See [LICENSE](LICENSE).

[deno_core]: https://github.com/denoland/deno_core
