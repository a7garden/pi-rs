//! Private host filesystem seam (upstream `env` `FileSystem` + `node.ts`
//! execution env, folded into the jsonl module per D1).
//!
//! The production implementation is plain `std::fs`; tests install an
//! instrumented host to inject I/O failures (upstream `InstrumentedEnv`).

use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};

/// Error codes observed by the storage core (upstream `FileError.code`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileErrorCode {
	NotFound,
	Unknown,
}

/// Host filesystem failure carrying an upstream-style code and message.
#[derive(Debug, Clone)]
pub struct FileError {
	pub code: FileErrorCode,
	pub message: String,
	pub path: PathBuf,
}

impl FileError {
	fn io(path: &Path, error: &io::Error) -> Self {
		let code = if error.kind() == io::ErrorKind::NotFound {
			FileErrorCode::NotFound
		} else {
			FileErrorCode::Unknown
		};
		Self { code, message: error.to_string(), path: path.to_path_buf() }
	}

	fn unknown(path: &Path, message: impl Into<String>) -> Self {
		Self { code: FileErrorCode::Unknown, message: message.into(), path: path.to_path_buf() }
	}
}

/// One `list_dir` entry.
#[derive(Debug, Clone)]
pub struct ListedEntry {
	pub name: String,
	pub path: PathBuf,
	pub is_file: bool,
}

/// Host operations used by the JSONL journal.
pub trait Host: Send + Sync {
	fn read_binary_file(&self, path: &Path) -> Result<Vec<u8>, FileError>;
	fn append_file(&self, path: &Path, bytes: &[u8]) -> Result<(), FileError>;
	fn flush_file(&self, path: &Path) -> Result<(), FileError>;
	fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), FileError>;
	fn rename_file(&self, source: &Path, destination: &Path) -> Result<(), FileError>;
	/// `force` ignores a missing target (upstream `remove({ force })`).
	fn remove(&self, path: &Path, force: bool) -> Result<(), FileError>;
	fn truncate_file(&self, path: &Path, length: u64) -> Result<(), FileError>;
	fn list_dir(&self, directory: &Path) -> Result<Vec<ListedEntry>, FileError>;
	fn create_dir_all(&self, directory: &Path) -> Result<(), FileError>;
}

/// The local filesystem host (the `node.ts` adapter, folded).
pub struct FsHost;

impl Host for FsHost {
	fn read_binary_file(&self, path: &Path) -> Result<Vec<u8>, FileError> {
		std::fs::read(path).map_err(|error| FileError::io(path, &error))
	}

	fn append_file(&self, path: &Path, bytes: &[u8]) -> Result<(), FileError> {
		OpenOptions::new()
			.create(true)
			.append(true)
			.open(path)
			.and_then(|mut file| file.write_all(bytes))
			.map_err(|error| FileError::io(path, &error))
	}

	fn flush_file(&self, path: &Path) -> Result<(), FileError> {
		OpenOptions::new()
			.write(true)
			.open(path)
			.and_then(|file| file.sync_all())
			.map_err(|error| FileError::io(path, &error))
	}

	fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), FileError> {
		std::fs::write(path, bytes).map_err(|error| FileError::io(path, &error))
	}

	fn rename_file(&self, source: &Path, destination: &Path) -> Result<(), FileError> {
		std::fs::rename(source, destination).map_err(|error| FileError::io(source, &error))
	}

	fn remove(&self, path: &Path, force: bool) -> Result<(), FileError> {
		match std::fs::remove_file(path) {
			Ok(()) => Ok(()),
			Err(error) if force && error.kind() == io::ErrorKind::NotFound => Ok(()),
			Err(error) => Err(FileError::io(path, &error)),
		}
	}

	fn truncate_file(&self, path: &Path, length: u64) -> Result<(), FileError> {
		OpenOptions::new()
			.write(true)
			.open(path)
			.and_then(|file| file.set_len(length))
			.map_err(|error| FileError::io(path, &error))
	}

	fn list_dir(&self, directory: &Path) -> Result<Vec<ListedEntry>, FileError> {
		let entries = std::fs::read_dir(directory).map_err(|error| FileError::io(directory, &error))?;
		let mut listed = Vec::new();
		for entry in entries {
			let entry = entry.map_err(|error| FileError::io(directory, &error))?;
			let file_type = entry.file_type().map_err(|error| FileError::io(directory, &error))?;
			listed.push(ListedEntry {
				name: entry.file_name().to_string_lossy().into_owned(),
				path: entry.path(),
				is_file: file_type.is_file(),
			});
		}
		Ok(listed)
	}

	fn create_dir_all(&self, directory: &Path) -> Result<(), FileError> {
		std::fs::create_dir_all(directory).map_err(|error| FileError::io(directory, &error))
	}
}

#[cfg(test)]
pub(crate) use self::instrumented::{Failure, FailureMode, FailureOperation, InstrumentedHost};

#[cfg(test)]
mod instrumented {
	use super::{FileError, FsHost, Host, ListedEntry};
	use std::path::Path;

	/// Which host operation a failure targets (upstream `Failure.operation`).
	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum FailureOperation {
		Append,
		Flush,
		Write,
		Rename,
		Remove,
	}

	/// Whether the failure precedes the real call, follows it, or the real call
	/// writes only half the payload first (upstream `Failure.mode`).
	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum FailureMode {
		Before,
		After,
		Short,
	}

	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub struct Failure {
		pub operation: FailureOperation,
		pub call: usize,
		pub mode: FailureMode,
	}

	/// Test host that counts operations and injects failures, mirroring the
	/// upstream `InstrumentedEnv` subclass.
	pub struct InstrumentedHost {
		inner: FsHost,
		pub operations: parking_lot::Mutex<Vec<String>>,
		failure: parking_lot::Mutex<Option<Failure>>,
		counters: parking_lot::Mutex<[usize; 5]>,
	}

	impl InstrumentedHost {
		pub fn new() -> Self {
			Self {
				inner: FsHost,
				operations: parking_lot::Mutex::new(Vec::new()),
				failure: parking_lot::Mutex::new(None),
				counters: parking_lot::Mutex::new([0; 5]),
			}
		}

		pub fn fail(&self, failure: Failure) {
			*self.failure.lock() = Some(failure);
			self.reset_observations();
		}

		pub fn clear(&self) {
			*self.failure.lock() = None;
			self.reset_observations();
		}

		fn reset_observations(&self) {
			self.operations.lock().clear();
			*self.counters.lock() = [0; 5];
		}

		fn injection(&self, operation: FailureOperation) -> Option<FailureMode> {
			let mut counters = self.counters.lock();
			let index = match operation {
				FailureOperation::Append => 0,
				FailureOperation::Flush => 1,
				FailureOperation::Write => 2,
				FailureOperation::Rename => 3,
				FailureOperation::Remove => 4,
			};
			counters[index] += 1;
			if let Some(failure) = *self.failure.lock() {
				if failure.operation == operation && failure.call == counters[index] {
					return Some(failure.mode);
				}
			}
			None
		}

		fn injected_error(&self, path: &Path, label: &str) -> FileError {
			FileError::unknown(path, format!("injected {label} failure"))
		}
	}

	impl Host for InstrumentedHost {
		fn read_binary_file(&self, path: &Path) -> Result<Vec<u8>, FileError> {
			self.inner.read_binary_file(path)
		}

		fn append_file(&self, path: &Path, bytes: &[u8]) -> Result<(), FileError> {
			self.operations.lock().push(format!("append:{}", file_name(path)));
			match self.injection(FailureOperation::Append) {
				Some(FailureMode::Before) => Err(self.injected_error(path, "append")),
				Some(FailureMode::Short) => {
					let partial = &bytes[..(bytes.len() / 2).max(1)];
					self.inner.append_file(path, partial)?;
					Err(self.injected_error(path, "short append"))
				}
				Some(FailureMode::After) => {
					self.inner.append_file(path, bytes)?;
					Err(self.injected_error(path, "post-append"))
				}
				None => self.inner.append_file(path, bytes),
			}
		}

		fn flush_file(&self, path: &Path) -> Result<(), FileError> {
			self.operations.lock().push(format!("flush:{}", file_name(path)));
			match self.injection(FailureOperation::Flush) {
				Some(FailureMode::After) => {
					self.inner.flush_file(path)?;
					Err(self.injected_error(path, "post-flush"))
				}
				Some(_) => Err(self.injected_error(path, "flush")),
				None => self.inner.flush_file(path),
			}
		}

		fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), FileError> {
			self.operations.lock().push(format!("write:{}", file_name(path)));
			match self.injection(FailureOperation::Write) {
				Some(FailureMode::Before) => Err(self.injected_error(path, "write")),
				Some(FailureMode::Short) => {
					let partial = &bytes[..(bytes.len() / 2).max(1)];
					self.inner.write_file(path, partial)?;
					Err(self.injected_error(path, "short write"))
				}
				Some(FailureMode::After) => {
					self.inner.write_file(path, bytes)?;
					Err(self.injected_error(path, "post-write"))
				}
				None => self.inner.write_file(path, bytes),
			}
		}

		fn rename_file(&self, source: &Path, destination: &Path) -> Result<(), FileError> {
			self.operations
				.lock()
				.push(format!("rename:{}->{}", file_name(source), file_name(destination)));
			match self.injection(FailureOperation::Rename) {
				Some(FailureMode::After) => {
					self.inner.rename_file(source, destination)?;
					Err(self.injected_error(source, "post-rename"))
				}
				Some(_) => Err(self.injected_error(source, "rename")),
				None => self.inner.rename_file(source, destination),
			}
		}

		fn remove(&self, path: &Path, force: bool) -> Result<(), FileError> {
			self.operations.lock().push(format!("remove:{}", file_name(path)));
			match self.injection(FailureOperation::Remove) {
				Some(FailureMode::After) => {
					self.inner.remove(path, force)?;
					Err(self.injected_error(path, "post-remove"))
				}
				Some(_) => Err(self.injected_error(path, "remove")),
				None => self.inner.remove(path, force),
			}
		}

		fn truncate_file(&self, path: &Path, length: u64) -> Result<(), FileError> {
			self.inner.truncate_file(path, length)
		}

		fn list_dir(&self, directory: &Path) -> Result<Vec<ListedEntry>, FileError> {
			self.inner.list_dir(directory)
		}

		fn create_dir_all(&self, directory: &Path) -> Result<(), FileError> {
			self.inner.create_dir_all(directory)
		}
	}

	fn file_name(path: &Path) -> String {
		path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
	}
}

use std::io::Write as _;
