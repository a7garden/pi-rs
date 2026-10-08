//! Port of `packages/coding-agent/src/utils/photon.ts` — the Photon image
//! processing seam.
//!
//! Upstream loads `@silvia-odwyer/photon-node` (a Rust/WASM codec) lazily and
//! returns `null` when it cannot be imported; every consumer then treats image
//! conversion/resize as unavailable. Here the codec sits behind the
//! [`PhotonEngine`] trait with a process-wide registry; no engine is
//! registered by default, so all dependent code paths take the same
//! "unavailable" route as upstream without a photon import (crate docs,
//! deviation D14). `fliph`/`flipv` and the EXIF rotation in
//! [`super::exif_orientation`] are plain pixel loops and are ported directly
//! rather than through the engine.

use parking_lot::RwLock;
use std::sync::Arc;

/// RGBA image (`photon.PhotonImage`). Pixels are row-major RGBA8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhotonImage {
	pub width: u32,
	pub height: u32,
	pub rgba: Vec<u8>,
}

impl PhotonImage {
	/// `PhotonImage.new(width, height, rgba)`.
	pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> PhotonImage {
		PhotonImage { width, height, rgba }
	}

	/// `image.get_width()`.
	pub fn get_width(&self) -> u32 {
		self.width
	}

	/// `image.get_height()`.
	pub fn get_height(&self) -> u32 {
		self.height
	}

	/// `image.get_raw_pixels()`.
	pub fn get_raw_pixels(&self) -> &[u8] {
		&self.rgba
	}

	/// `photon.fliph(image)` — horizontal flip, in place.
	pub fn fliph(&mut self) {
		let w = self.width as usize;
		let h = self.height as usize;
		for row in 0..h {
			for col in 0..w / 2 {
				let left = (row * w + col) * 4;
				let right = (row * w + (w - 1 - col)) * 4;
				self.rgba.swap(left, right);
				self.rgba.swap(left + 1, right + 1);
				self.rgba.swap(left + 2, right + 2);
				self.rgba.swap(left + 3, right + 3);
			}
		}
	}

	/// `photon.flipv(image)` — vertical flip, in place.
	pub fn flipv(&mut self) {
		let w = self.width as usize;
		let h = self.height as usize;
		for row in 0..h / 2 {
			for col in 0..w {
				let top = (row * w + col) * 4;
				let bottom = ((h - 1 - row) * w + col) * 4;
				self.rgba.swap(top, bottom);
				self.rgba.swap(top + 1, bottom + 1);
				self.rgba.swap(top + 2, bottom + 2);
				self.rgba.swap(top + 3, bottom + 3);
			}
		}
	}
}

/// Photon codec seam (upstream: the imported `@silvia-odwyer/photon-node`
/// module).
pub trait PhotonEngine: Send + Sync {
	/// `PhotonImage.new_from_byteslice(bytes)` — decode any supported format
	/// into RGBA. `None` mirrors the upstream exception path.
	fn new_from_byteslice(&self, bytes: &[u8]) -> Option<PhotonImage>;

	/// `image.get_bytes()` — encode as PNG.
	fn get_bytes(&self, image: &PhotonImage) -> Option<Vec<u8>>;

	/// `image.get_bytes_jpeg(quality)` — encode as JPEG.
	fn get_bytes_jpeg(&self, image: &PhotonImage, quality: u8) -> Option<Vec<u8>>;

	/// `photon.resize(image, width, height, SamplingFilter.Lanczos3)`.
	fn resize(&self, image: &PhotonImage, width: u32, height: u32) -> Option<PhotonImage>;
}

static ENGINE: RwLock<Option<Arc<dyn PhotonEngine>>> = RwLock::new(None);

/// Install the process-wide photon engine (upstream has no equivalent: the
/// WASM module is either importable or not).
pub fn set_photon_engine(engine: Arc<dyn PhotonEngine>) {
	*ENGINE.write() = Some(engine);
}

/// Load the photon engine asynchronously (`loadPhoton`). Returns the cached
/// engine on subsequent calls, `None` when no engine is available.
pub async fn load_photon() -> Option<Arc<dyn PhotonEngine>> {
	ENGINE.read().clone()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn fliph_mirrors_rows() {
		let mut image = PhotonImage::new(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
		image.fliph();
		assert_eq!(image.rgba, vec![5, 6, 7, 8, 1, 2, 3, 4]);
	}

	#[test]
	fn flipv_mirrors_columns() {
		let mut image = PhotonImage::new(1, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
		image.flipv();
		assert_eq!(image.rgba, vec![5, 6, 7, 8, 1, 2, 3, 4]);
	}

	#[tokio::test]
	async fn load_photon_is_none_without_a_registered_engine() {
		assert!(load_photon().await.is_none());
	}
}
