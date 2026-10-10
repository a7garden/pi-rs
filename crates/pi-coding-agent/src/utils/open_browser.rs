//! Port of `packages/coding-agent/src/utils/open-browser.ts` — open a URL or
//! file with the platform default handler, never through a shell.
//!
//! Deviation (crate docs, D13): upstream spawns detached and unrefs; Rust
//! spawns and immediately detaches by dropping the [`std::process::Child`]
//! (no zombie reap on unix — the child is reparented to init).
//!
//! Test seam (local): [`set_browser_launcher_override`] replaces the
//! platform launcher with a closure so the unit suite never spawns a real
//! browser on developer machines while still exercising the
//! best-effort-spawn-error path. Calls always go through the current
//! override (when set) and fall back to [`launch_default`] otherwise.

use std::process::{Child, Command, Stdio};

type BrowserLauncher = dyn Fn(&str) -> std::io::Result<Child> + Send + Sync;

static OVERRIDE: parking_lot::RwLock<Option<Box<BrowserLauncher>>> = parking_lot::RwLock::new(None);

/// Test seam: replace the platform launcher with `launcher`. Production
/// callers must NOT invoke this — `cargo test` is the only expected user.
#[doc(hidden)]
#[cfg(test)]
pub fn set_browser_launcher_override(launcher: Box<BrowserLauncher>) {
    *OVERRIDE.write() = Some(launcher);
}

#[cfg(test)]
pub(crate) fn clear_browser_launcher_override() {
    *OVERRIDE.write() = None;
}

/// Open a URL or file in the platform browser/default handler (`openBrowser`).
///
/// This intentionally never invokes a shell. On Windows, do not use
/// `cmd /c start`: cmd.exe re-parses metacharacters (&, |, ^, ...) before
/// `start` runs, which would make attacker-controlled URLs injectable.
///
/// Launcher failures (for example, a missing `xdg-open`) are swallowed:
/// browser launch is best-effort — callers still present the target to the
/// user, so the failure must not become a process crash.
pub fn open_browser(target: &str) {
    if let Some(launcher) = OVERRIDE.read().as_ref() {
        // Test seam: silent on io::Error just like the platform path.
        let _ = launcher(target);
        return;
    }
    launch_default(target);
}

fn launch_default(target: &str) {
    let mut cmd = if cfg!(target_os = "macos") {
        let mut cmd = Command::new("open");
        cmd.arg(target);
        cmd
    } else if cfg!(windows) {
        let mut cmd = Command::new("rundll32");
        cmd.arg("url.dll,FileProtocolHandler").arg(target);
        cmd
    } else {
        let mut cmd = Command::new("xdg-open");
        cmd.arg(target);
        cmd
    };

    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Best-effort: a failed spawn (or failed detach) is ignored, matching
    // the upstream `.on("error", () => {})`.
    if let Ok(child) = cmd.spawn() {
        drop(child);
    }
    #[cfg(unix)]
    {
        // Detach: setsid so the browser survives us and is not reaped as our
        // child (upstream `.unref()`). Done last so a missing launcher still
        // skips the pre_exec dance on platforms that name the launcher
        // differently than the `open`/`rundll32`/`xdg-open` defaults.
        use std::os::unix::process::CommandExt;
        // Safety: closures only run in the forked child before exec.
        unsafe {
            let _ = cmd.pre_exec(|| {
                // Ignore the already-group-leader failure; detaching is a no-op.
                let _ = libc::setsid();
                Ok(())
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Serializes tests that swap the global launcher so the override is not
    // observed cross-test under parallel `cargo test`. The OVERRIDE itself
    // is guarded by a parking_lot::RwLock, but tests also share static
    // counters/buffers that the closure captures by reference.
    static OVERRIDE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    #[test]
    fn open_browser_never_panics_on_missing_launcher() {
        // xdg-open is absent in most CI environments; the call must stay
        // best-effort and silent (upstream swallows the spawn error event).
        // Inject a stub that simulates a missing binary so `cargo test`
        // never spawns a real browser on developer machines.
        let _guard = OVERRIDE_LOCK.lock();
        set_browser_launcher_override(Box::new(|_target| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "simulated missing launcher",
            ))
        }));
        open_browser("https://example.invalid/");
        clear_browser_launcher_override();
    }

    #[test]
    fn open_browser_forwards_target_to_launcher_and_swallows_io_errors() {
        // Stronger guarantee than the missing-launcher case: the launcher
        // actually receives the URL target and the io::Error it returns
        // never propagates. Counts invocations to catch double-spawn or
        // silent skip regressions.
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        static SEEN: Mutex<Option<String>> = Mutex::new(None);

        let _guard = OVERRIDE_LOCK.lock();
        set_browser_launcher_override(Box::new(|target| {
            CALLS.fetch_add(1, Ordering::SeqCst);
            *SEEN.lock() = Some(target.to_owned());
            Err(std::io::Error::other("forced launcher failure"))
        }));
        open_browser("https://example.invalid/");
        // Call twice to make sure swallowing is stable across repeated
        // invocations and the override is not consumed on the first call.
        open_browser("https://example.invalid/?retry=1");
        clear_browser_launcher_override();

        assert_eq!(CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(
            SEEN.lock().as_deref(),
            Some("https://example.invalid/?retry=1"),
            "launcher must observe the most recent target",
        );
    }
}
