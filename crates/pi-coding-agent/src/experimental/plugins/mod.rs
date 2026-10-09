//! Port of `experimental/plugins/` — explicit plugin package selection for
//! the experimental server: package profile persistence (`package.ts`) and
//! the session facet loader (`bundled.ts`).
//!
//! Deviations (D152):
//! - `import { basename, dirname, join, resolve } from "node:path"` maps to
//!   [`std::path`] operations; `resolve` is `utils::paths::resolve_path`.
//! - Profile files are written with `std::fs::OpenOptions::mode(0o600)`
//!   (unix) mirroring upstream `mode: 0o600`.

pub mod bundled;
pub mod package;
