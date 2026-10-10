//! Port of `packages/coding-agent/src/utils/` — helpers for the ported
//! surface. Modules are added as upstream files are ported (see crate docs).
pub mod abort;
pub mod ansi;
pub mod changelog;
pub mod child_process;
pub mod clipboard;
pub mod clipboard_command;
pub mod clipboard_image;
pub mod deprecation;
pub mod exif_orientation;
pub mod frontmatter;
pub mod fs_watch;
pub mod git;
pub mod html;
pub mod image_convert;
pub mod image_process;
pub mod image_resize;
pub mod image_resize_core;
pub mod image_resize_worker;
pub mod json;
pub mod management_http;
pub mod mime;
pub mod open_browser;
pub mod output_files;
pub mod paths;
pub mod photon;
pub mod pi_user_agent;
pub mod shell;
pub mod sleep;
pub mod syntax_highlight;
pub mod text;
pub mod time;
pub mod tool_result_images;
pub mod tools_manager;
pub mod tui_compat;
pub mod version_check;
pub mod windows_self_update;
pub mod wsl;
pub mod zip;

/// Shared serialization for tests that mutate process env vars
/// (`std::env::set_var`/`remove_var` race concurrent `fork`/`exec` on macOS,
/// breaking unrelated suites like `child_process` under parallel test runs).
/// Take this guard around every env mutation in `#[cfg(test)]` code.
#[cfg(test)]
pub(crate) fn test_env_lock() -> parking_lot::MutexGuard<'static, ()> {
    static TEST_ENV_LOCK: std::sync::LazyLock<parking_lot::Mutex<()>> =
        std::sync::LazyLock::new(|| parking_lot::Mutex::new(()));
    TEST_ENV_LOCK.lock()
}
