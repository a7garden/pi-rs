//! Mirrors upstream `src/runtime/prelude-source.ts`: the JavaScript evaluated inside the
//! QuickJS VM before the script runs.
//!
//! The prelude is data, not behavior: it is embedded verbatim (interpolated constants
//! resolved) from the upstream TypeScript template literal and kept as an embedded string
//! constant. The VM is its own wasm instance, so nothing here guards a realm boundary.
//! The prelude keeps the host bridge in a closure so the script cannot call it directly, and
//! builds `tools`, `ALL_TOOLS`, the output helpers (`text`, `image`, `exit`, `console`), and
//! globals on top of it. Tool arguments and results cross as JSON strings and are parsed on
//! this side.
//!
//! Output helpers: `text(value)` appends a text item (non-strings are JSON-stringified),
//! `image(urlOrItem)` appends an image from a base64 `data:` URL, an `{ image_url }` object,
//! or an MCP `ImageContent` block, and `exit()` ends the script successfully. `console.*`
//! appends text items like `text()`.
//!
//! `store(key, value)` and `load(key)` are synchronous: they work on a snapshot of JSON text
//! passed in as `storeJson`, and the keys the script wrote are reported with a successful
//! "done".
//!
//! Evaluates to a function `(bridge, toolsJson, globalsJson, storeJson) => { settle, run,
//! stalled }`. `toolsJson` lists `{ name, jsName, description }`: `tools[jsName]` and
//! `tools[name]` call the tool, and `ALL_TOOLS` lists `{ name: jsName, description }`.
//! `stalled()` reports a script that has not finished while no host call is pending: with no
//! timers or I/O in the VM, nothing can ever resume it. `globalsJson` lists
//! `{ name, spread }`; `a.b` names are grouped into a frozen `a` object. `bridge(kind, a, b,
//! c)` with kind "call" or "global" (id, name, argsJson), "output" ("text", text) or
//! ("image", data, mimeType), or "done" (ok, valueJsonOrErrorJson, writesJson).

/// Maximum value size for a single `store()` write (characters of JSON).
pub const MAX_STORE_VALUE_CHARS: usize = 256 * 1024;
/// Maximum total size of the store snapshot (characters of JSON, keys included).
pub const MAX_STORE_TOTAL_CHARS: usize = 1024 * 1024;
/// Output one script may produce with `text()`, `image()`, and `console.*`: characters of text
/// and base64 image data. The host keeps all output until the script ends, so without a limit
/// a script that prints in a loop grows the host's memory until it crashes.
pub const MAX_OUTPUT_CHARS: usize = 16 * 1024 * 1024;
/// Item count limit, covering loops that print empty strings.
pub const MAX_OUTPUT_ITEMS: usize = 100_000;

/// The prelude source evaluated as `"codemode-prelude.js"` inside the VM. Extracted verbatim
/// from upstream `prelude-source.ts` with the constant interpolations
/// (`MAX_STORE_VALUE_CHARS`, `MAX_STORE_TOTAL_CHARS`, `MAX_OUTPUT_CHARS`, `MAX_OUTPUT_ITEMS`,
/// `IMAGE_HELPER_EXPECTS`) resolved.
pub const PRELUDE_SOURCE: &str = include_str!("prelude.js");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_upstream_values() {
        assert_eq!(MAX_STORE_VALUE_CHARS, 262_144);
        assert_eq!(MAX_STORE_TOTAL_CHARS, 1_048_576);
        assert_eq!(MAX_OUTPUT_CHARS, 16_777_216);
        assert_eq!(MAX_OUTPUT_ITEMS, 100_000);
    }

    #[test]
    fn prelude_has_interpolated_constants_resolved() {
        assert!(PRELUDE_SOURCE.contains("(function (bridge, toolsJson, globalsJson, storeJson) {"));
        assert!(PRELUDE_SOURCE.contains("16777216"));
        assert!(PRELUDE_SOURCE.contains("100000"));
        assert!(PRELUDE_SOURCE.contains("262144"));
        assert!(PRELUDE_SOURCE.contains("1048576"));
        assert!(PRELUDE_SOURCE.contains("image expects a non-empty image URL string"));
        // No unresolved template substitutions.
        assert!(!PRELUDE_SOURCE.contains("${"));
    }
}
