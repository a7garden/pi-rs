//! Port of `packages/coding-agent/src/utils/output-files.ts` — files pi writes
//! so the model can use output it was not shown in full (truncated tool
//! output, binary MCP resources, images from codemode scripts). Every output
//! file is created here, so where they live can change in one place. Today
//! they go to the OS temp directory.
//!
//! Deviation (crate docs, D13): `randomBytes` → [`getrandom`]; the streamed
//! variant returns a [`tokio::fs::File`] instead of a node `WriteStream`.

use std::io::Write;
use std::path::PathBuf;

/// Output can carry private data, so only the user may read the files.
const OUTPUT_FILE_MODE: u32 = 0o600;

fn random_hex(n: usize) -> String {
	let mut bytes = vec![0u8; n];
	getrandom::getrandom(&mut bytes).expect("OS RNG unavailable");
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn tmp_dir() -> PathBuf {
	std::env::temp_dir()
}

/// A new, unused path: `<dir>/<prefix>-<random hex><extension>` (upstream
/// `createOutputFilePath`). `extension` includes the dot.
fn create_output_file_path(prefix: &str, extension: &str) -> String {
	tmp_dir().join(format!("{prefix}-{}{extension}", random_hex(8))).to_string_lossy().into_owned()
}

/// Write `data` to a new output file and return its path (`writeOutputFile`).
///
/// `create_new(true)` mirrors upstream's `wx` flag: never follow a link
/// someone else placed at the path.
pub async fn write_output_file(prefix: &str, extension: &str, data: &[u8]) -> std::io::Result<String> {
	let path = create_output_file_path(prefix, extension);
	let open_path = path.clone();
	let data = data.to_vec();
	tokio::task::spawn_blocking(move || -> std::io::Result<()> {
		let mut file = open_new_output_file(&open_path)?;
		file.write_all(&data)
	})
	.await
	.map_err(|e| std::io::Error::other(format!("output file join failed: {e}")))??;
	Ok(path)
}

/// Open a new output file for streamed output (`createOutputFileStream`).
pub async fn create_output_file_stream(prefix: &str, extension: &str) -> std::io::Result<(String, tokio::fs::File)> {
	let path = create_output_file_path(prefix, extension);
	let open_path = path.clone();
	let file = tokio::task::spawn_blocking(move || open_new_output_file(&open_path))
		.await
		.map_err(|e| std::io::Error::other(format!("output file join failed: {e}")))??;
	// SAFETY: from_std hands the freshly opened std file's ownership to tokio.
	let file = { tokio::fs::File::from_std(file) };
	Ok((path, file))
}

/// Synchronous variant of [`create_output_file_stream`] for output callbacks
/// that run outside async context (core/bash-executor). Same path and mode
/// conventions.
pub fn create_output_file_stream_sync(prefix: &str, extension: &str) -> std::io::Result<(String, std::fs::File)> {
	let path = create_output_file_path(prefix, extension);
	let file = open_new_output_file(&path)?;
	Ok((path, file))
}

#[cfg(unix)]
fn open_new_output_file(path: &str) -> std::io::Result<std::fs::File> {
	use std::os::unix::fs::OpenOptionsExt;
	std::fs::OpenOptions::new()
		.write(true)
		.create_new(true)
		.mode(OUTPUT_FILE_MODE)
		.open(path)
}

#[cfg(windows)]
fn open_new_output_file(path: &str) -> std::io::Result<std::fs::File> {
	std::fs::OpenOptions::new().write(true).create_new(true).open(path)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn writes_private_output_file_with_prefix_and_extension() {
		let data = b"full tool output";
		let path = write_output_file("bash-output", ".txt", data).await.unwrap();
		assert!(path.contains("bash-output-"));
		assert!(path.ends_with(".txt"));
		// mode 0600: user read/write only (unix).
		#[cfg(unix)]
		{
			use std::os::unix::fs::PermissionsExt;
			let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
			assert_eq!(mode, 0o600);
		}
		assert_eq!(std::fs::read(&path).unwrap(), data);
		std::fs::remove_file(&path).unwrap();
	}

	#[tokio::test]
	async fn stream_creates_exclusive_file() {
		use tokio::io::AsyncWriteExt;
		let (path, mut file) = create_output_file_stream("codemode", ".png").await.unwrap();
		file.write_all(b"\x89PNG").await.unwrap();
		file.sync_all().await.unwrap();
		drop(file);
		assert_eq!(std::fs::read(&path).unwrap(), b"\x89PNG");
		std::fs::remove_file(&path).unwrap();
	}

	#[test]
	fn paths_are_unique() {
		let a = create_output_file_path("x", ".log");
		let b = create_output_file_path("x", ".log");
		assert_ne!(a, b);
	}
}
