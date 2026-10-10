//! Advisory file locking shared by the server-cluster slice (upstream the
//! npm `proper-lockfile` dependency).
//!
//! # Deviations
//!
//! - **D149 — proper-lockfile becomes std advisory file locks** (the D136
//!   pattern): `lock(path)` opens (creating) `<path>` and holds
//!   `std::fs::File::try_lock` until released. Ownership is kernel-managed,
//!   so a dead holder releases automatically and proper-lockfile's
//!   mtime-based `stale` handling has no counterpart; the wait budget
//!   (`retries`/`maxRetryTime` pairs) is preserved as [`LockOptions`].
//! - Upstream locks the Session directory path directly; the port locks the
//!   `<path>.lock` file next to it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Upstream proper-lockfile retry options reduced to the observable budget:
/// poll every `retry_ms` until `wait_ms` elapses (upstream uses a constant
/// `minTimeout == maxTimeout` with a `maxRetryTime` cap).
#[derive(Debug, Clone, Copy)]
pub struct LockOptions {
    pub retry_ms: u64,
    pub wait_ms: u64,
}

/// A held advisory file lock; releasing on drop (close).
pub struct FileLock {
    _file: std::fs::File,
    path: PathBuf,
    release_unlinks: bool,
}

impl FileLock {
    /// Acquire the lock at `path`, polling until the wait budget runs out.
    pub async fn acquire(path: &Path, options: LockOptions) -> Result<FileLock, String> {
        let started = Instant::now();
        loop {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .map_err(|error| format!("Could not open {}: {error}", path.display()))?;
            match file.try_lock() {
                Ok(()) => {
                    return Ok(FileLock {
                        _file: file,
                        path: path.to_path_buf(),
                        release_unlinks: true,
                    });
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    if started.elapsed() >= Duration::from_millis(options.wait_ms) {
                        return Err(format!("Could not lock {}: timed out", path.display()));
                    }
                    tokio::time::sleep(Duration::from_millis(options.retry_ms)).await;
                }
                Err(std::fs::TryLockError::Error(error)) => {
                    return Err(format!("Could not lock {}: {error}", path.display()));
                }
            }
        }
    }

    /// A held lock whose release does not unlink the file (shared lock
    /// files like `launcher-<serverId>` keep their history).
    pub fn keep_file(mut self) -> Self {
        self.release_unlinks = false;
        self
    }

    /// Release the lock (upstream `release()`).
    pub fn release(self) {
        drop(self);
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        if self.release_unlinks {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release_round_trip() {
        let directory = std::env::temp_dir().join(format!("pi-lockfile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("probe.lock");
        let lock = FileLock::acquire(
            &path,
            LockOptions {
                retry_ms: 10,
                wait_ms: 500,
            },
        )
        .await
        .unwrap();
        // A second acquisition within the wait budget fails.
        let second = FileLock::acquire(
            &path,
            LockOptions {
                retry_ms: 10,
                wait_ms: 50,
            },
        )
        .await;
        assert!(second.is_err());
        lock.release();
        // After release the lock is available again.
        let third = FileLock::acquire(
            &path,
            LockOptions {
                retry_ms: 10,
                wait_ms: 500,
            },
        )
        .await;
        assert!(third.is_ok());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[tokio::test]
    async fn release_removes_lock_file_by_default() {
        let directory =
            std::env::temp_dir().join(format!("pi-lockfile-clean-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("clean.lock");
        FileLock::acquire(
            &path,
            LockOptions {
                retry_ms: 10,
                wait_ms: 500,
            },
        )
        .await
        .unwrap()
        .release();
        assert!(!path.exists());
        let kept = FileLock::acquire(
            &path,
            LockOptions {
                retry_ms: 10,
                wait_ms: 500,
            },
        )
        .await
        .unwrap()
        .keep_file();
        drop(kept);
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
