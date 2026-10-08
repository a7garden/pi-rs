//! Port of `packages/coding-agent/src/utils/zip.ts` — the small, classic ZIP
//! archives used by bug reports: local file headers + central directory +
//! EOCD, deflate-compressed entries with UTC-unaware DOS timestamps (upstream
//! uses the node process's local time via `Date`; here `std::time` local time
//! is not available portably, so DOS timestamps use the current UTC time —
//! deviation, crate docs D13).
//!
//! Dependency request (coordinator): `flate2 = "1"` — used for raw-deflate
//! compression and CRC-32, mirroring upstream `zlib.deflateRawSync`/`crc32`.

use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::write::DeflateEncoder;
use flate2::{Compression, Crc};

/// One archive entry (upstream `ZipEntry`).
pub struct ZipEntry {
	pub name: String,
	pub data: Vec<u8>,
}

impl ZipEntry {
	pub fn new(name: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
		ZipEntry { name: name.into(), data: data.into() }
	}
}

fn u16le(buf: &mut [u8], offset: usize, value: u16) {
	buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn u32le(buf: &mut [u8], offset: usize, value: u32) {
	buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// DOS date/time from the current time (upstream `dosDateTime`, local time).
fn dos_datetime() -> (u16, u16) {
	let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
	let days = (secs / 86_400) as i64;
	let secs_of_day = (secs % 86_400) as u32;
	// Days since 1970-01-01 → civil date (Howard Hinnant's algorithm).
	let z = days + 719_468;
	let era = z.div_euclid(146_097);
	let doe = z.rem_euclid(146_097);
	let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let day = (doy - (153 * mp + 2) / 5 + 1) as u16;
	let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u16;
	let year = (yoe + era * 400) as i64;
	let year = year.clamp(1980, 2107) as u16;
	let hour = (secs_of_day / 3600) as u16;
	let minute = ((secs_of_day % 3600) / 60) as u16;
	let second = (secs_of_day % 60) as u16;
	let time = (hour << 11) | (minute << 5) | (second >> 1);
	let date = ((year - 1980) << 9) | ((month as u16) << 5) | day;
	(time, date)
}

fn deflate_raw(data: &[u8]) -> Vec<u8> {
	let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
	encoder.write_all(data).expect("writing to Vec cannot fail");
	encoder.finish().expect("finishing into Vec cannot fail")
}

/// Create the archive bytes (upstream `createZipArchive`).
fn create_zip_archive(entries: &[ZipEntry]) -> Vec<u8> {
	let mut files: Vec<u8> = Vec::new();
	let mut directory: Vec<u8> = Vec::new();
	let (time, date) = dos_datetime();
	let mut offset: u32 = 0;

	for entry in entries {
		let name = entry.name.as_bytes();
		let data = &entry.data;
		let compressed = deflate_raw(data);
		let mut crc = Crc::new();
		crc.update(data);
		let checksum = crc.sum();

		let mut local = [0u8; 30];
		u32le(&mut local, 0, 0x0403_4b50);
		u16le(&mut local, 4, 20);
		u16le(&mut local, 6, 0x0800); // UTF-8 name flag
		u16le(&mut local, 8, 8); // deflate
		u16le(&mut local, 10, time);
		u16le(&mut local, 12, date);
		u32le(&mut local, 14, checksum);
		u32le(&mut local, 18, compressed.len() as u32);
		u32le(&mut local, 22, data.len() as u32);
		u16le(&mut local, 26, name.len() as u16);

		let mut central = [0u8; 46];
		u32le(&mut central, 0, 0x0201_4b50);
		u16le(&mut central, 4, 20);
		u16le(&mut central, 6, 20);
		u16le(&mut central, 8, 0x0800);
		u16le(&mut central, 10, 8);
		u16le(&mut central, 12, time);
		u16le(&mut central, 14, date);
		u32le(&mut central, 16, checksum);
		u32le(&mut central, 20, compressed.len() as u32);
		u32le(&mut central, 24, data.len() as u32);
		u16le(&mut central, 28, name.len() as u16);
		u32le(&mut central, 42, offset);

		files.extend_from_slice(&local);
		files.extend_from_slice(name);
		files.extend_from_slice(&compressed);
		directory.extend_from_slice(&central);
		directory.extend_from_slice(name);
		offset += 30 + name.len() as u32 + compressed.len() as u32;
	}

	let mut end = [0u8; 22];
	u32le(&mut end, 0, 0x0605_4b50);
	u16le(&mut end, 8, entries.len() as u16);
	u16le(&mut end, 10, entries.len() as u16);
	u32le(&mut end, 12, directory.len() as u32);
	u32le(&mut end, 16, offset);

	let mut out = files;
	out.extend_from_slice(&directory);
	out.extend_from_slice(&end);
	out
}

/// Write `entries` as a ZIP archive to `filePath` (`writeZipArchive`; the
/// upstream promise is a sync write here, see crate docs D13).
pub fn write_zip_archive(file_path: impl AsRef<Path>, entries: &[ZipEntry]) -> std::io::Result<()> {
	std::fs::write(file_path, create_zip_archive(entries))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Minimal raw-deflate inflate for tests, via flate2's decoder.
	fn inflate_raw(data: &[u8]) -> Vec<u8> {
		use flate2::read::DeflateDecoder;
		use std::io::Read;
		let mut decoder = DeflateDecoder::new(data);
		let mut out = Vec::new();
		decoder.read_to_end(&mut out).unwrap();
		out
	}

	fn read_archive(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
		// Walk local headers from the start (this writer emits no data
		// descriptors, so sizes in the local header are authoritative).
		let mut offset = 0usize;
		let mut out = Vec::new();
		while u32le_at(bytes, offset) == 0x0403_4b50 {
			let flags = u16le_at(bytes, offset + 6);
			assert_eq!(flags, 0x0800);
			let name_len = u16le_at(bytes, offset + 26) as usize;
			let extra_len = u16le_at(bytes, offset + 28) as usize;
			let compressed_size = u32le_at(bytes, offset + 18) as usize;
			let crc = u32le_at(bytes, offset + 14);
			let uncompressed_size = u32le_at(bytes, offset + 22) as usize;
			let name = String::from_utf8(bytes[offset + 30..offset + 30 + name_len].to_vec()).unwrap();
			let data_start = offset + 30 + name_len + extra_len;
			let compressed = &bytes[data_start..data_start + compressed_size];
			let data = inflate_raw(compressed);
			assert_eq!(data.len(), uncompressed_size);
			let mut crc_check = Crc::new();
			crc_check.update(&data);
			assert_eq!(crc_check.sum(), crc);
			out.push((name, data));
			offset = data_start + compressed_size;
		}
		// The central directory follows the entries; the EOCD closes the file.
		assert_eq!(u32le_at(bytes, bytes.len() - 22), 0x0605_4b50);
		out
	}

	fn u16le_at(bytes: &[u8], offset: usize) -> u16 {
		u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
	}

	fn u32le_at(bytes: &[u8], offset: usize) -> u32 {
		u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
	}

	#[test]
	fn writes_round_trippable_deflate_archive() {
		let entries = vec![
			ZipEntry::new("report.txt", "hello bug report"),
			ZipEntry::new("logs/session.log", "line\n".repeat(200)),
		];
		let bytes = create_zip_archive(&entries);
		assert_eq!(u32le_at(&bytes, 0), 0x0403_4b50);

		// Central directory + EOCD structure.
		let eocd = bytes.len() - 22;
		assert_eq!(u32le_at(&bytes, eocd), 0x0605_4b50);
		assert_eq!(u16le_at(&bytes, eocd + 8), 2);
		assert_eq!(u16le_at(&bytes, eocd + 10), 2);
		let cd_size = u32le_at(&bytes, eocd + 12) as usize;
		let cd_offset = u32le_at(&bytes, eocd + 16) as usize;
		assert_eq!(u32le_at(&bytes, cd_offset), 0x0201_4b50);
		assert_eq!(cd_offset + cd_size, eocd);

		let round = read_archive(&bytes);
		assert_eq!(round.len(), 2);
		assert_eq!(round[0].0, "report.txt");
		assert_eq!(round[0].1, b"hello bug report");
		assert_eq!(round[1].0, "logs/session.log");
		assert_eq!(round[1].1, "line\n".repeat(200).into_bytes());
	}

	#[test]
	fn write_zip_archive_to_disk() {
		let dir = std::env::temp_dir();
		let path = dir.join(format!("pi-zip-{}-{:p}.zip", std::process::id(), &dir));
		write_zip_archive(&path, &[ZipEntry::new("a.txt", "abc")]).unwrap();
		let bytes = std::fs::read(&path).unwrap();
		assert_eq!(u32le_at(&bytes, bytes.len() - 22), 0x0605_4b50);
		std::fs::remove_file(&path).ok();
	}
}
