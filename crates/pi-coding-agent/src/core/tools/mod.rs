//! Port of `packages/coding-agent/src/core/tools/` — the built-in coding
//! tools. Files port 1:1; the renderer modules (`renderers/*.ts` and the
//! theme-coupled `render-utils` helpers) are deferred with the excluded
//! interactive mode (D32).

#[cfg(test)]
#[path = "tools_tests.rs"]
mod tools_tests;

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod file_mutation_queue;
pub mod find;
pub mod grep;
pub mod index;
pub mod ls;
pub mod output_accumulator;
pub mod path_utils;
pub mod powershell;
pub mod read;
pub mod render_utils;
pub mod tool_definition_wrapper;
pub mod truncate;
pub mod write;
