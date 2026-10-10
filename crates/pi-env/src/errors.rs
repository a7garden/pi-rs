//! Mapping between daemon failures and Durable's `FileError` (upstream `packages/env/src/errors.ts`).
//!
//! Deviation: upstream's `toFileError` leaves `ERR_FS_EISDIR` as `unknown`, because upstream's local
//! `NodeExecutionEnv` surfaces Node's own `ERR_FS_EISDIR` there. Our local comparator
//! (`pi_durable::env::node`) reports `is_directory` for the same operation, so this mapping keeps the
//! remote environment's answers identical to the local one, which is what the differential tests assert.

use pi_durable::env::{FileError, FileErrorCode};

use crate::connection::RemoteError;

/// The `FileError` an already aborted context produces, with the operation's path when known.
pub fn abort_error(path: Option<&str>) -> FileError {
    FileError::aborted(path.map(str::to_owned))
}

/// Whether the context is already aborted; the caller returns [`abort_error`] when it is.
pub fn aborted(signal: Option<&pi_chord::context::AbortSignal>) -> bool {
    signal.is_some_and(|signal| signal.aborted())
}

/// `NodeExecutionEnv`'s mapping of Node error codes to `FileError` codes.
pub fn to_file_error(error: &RemoteError, fallback_path: Option<&str>) -> FileError {
    let path = error
        .path
        .clone()
        .or_else(|| fallback_path.map(str::to_owned));
    let code = match error.code.as_str() {
        "aborted" => FileErrorCode::Aborted,
        "ENOENT" => FileErrorCode::NotFound,
        "EACCES" | "EPERM" => FileErrorCode::PermissionDenied,
        "ENOTDIR" => FileErrorCode::NotDirectory,
        "EISDIR" | "ERR_FS_EISDIR" => FileErrorCode::IsDirectory,
        "EINVAL" | "SYMLINK" | "NOT_REGULAR" => FileErrorCode::Invalid,
        _ => FileErrorCode::Unknown,
    };
    FileError::new(code, error.message.clone(), path)
}
