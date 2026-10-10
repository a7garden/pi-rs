//! Port of `packages/coding-agent/src/utils/clipboard-image.ts` — reading
//! images from the system clipboard across Wayland/X11/WSL/native backends.
//!
//! Deviations (crate docs, deviation D16): the pi-tui native clipboard is the
//! [`super::clipboard::NativeClipboard`] seam; image MIME sniffing reuses the
//! shared [`crate::utils::mime`] port; tests drive the process backends
//! through the [`super::clipboard_command`] override hook, standing in for
//! upstream's vi.mock.

use crate::utils::mime::detect_supported_image_mime_type;

use super::clipboard::{native_clipboard, Platform};
use crate::utils::clipboard_command::{run_clipboard_command, RunClipboardCommandOptions};

const DEFAULT_LIST_TIMEOUT_MS: Option<u64> = Some(1000);
const DEFAULT_POWERSHELL_TIMEOUT_MS: Option<u64> = Some(5000);

/// Upstream `ClipboardImage`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipboardImage {
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

const SUPPORTED_IMAGE_MIME_TYPES: [&str; 4] =
    ["image/png", "image/jpeg", "image/webp", "image/gif"];

/// `isWaylandSession`.
pub fn is_wayland_session(env: &std::collections::HashMap<String, String>) -> bool {
    env.contains_key("WAYLAND_DISPLAY")
        || env.get("XDG_SESSION_TYPE").map(String::as_str) == Some("wayland")
}

pub(crate) fn base_mime_type(mime_type: &str) -> String {
    mime_type
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or(mime_type)
        .to_lowercase()
}

/// `extensionForImageMimeType`.
pub fn extension_for_image_mime_type(mime_type: &str) -> Option<&'static str> {
    match base_mime_type(mime_type).as_str() {
        "image/png" => Some("png"),
        "image/jpeg" => Some("jpg"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        _ => None,
    }
}

fn select_preferred_image_mime_type(mime_types: &[String]) -> Option<String> {
    let normalized: Vec<(String, String)> = mime_types
        .iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .map(|t| {
            let base = base_mime_type(&t);
            (t, base)
        })
        .collect();

    for preferred in SUPPORTED_IMAGE_MIME_TYPES {
        if let Some((raw, _)) = normalized.iter().find(|(_, base)| base == preferred) {
            return Some(raw.clone());
        }
    }

    normalized
        .iter()
        .find(|(_, base)| base.starts_with("image/"))
        .map(|(raw, _)| raw.clone())
}

fn is_supported_image_mime_type(mime_type: &str) -> bool {
    let base = base_mime_type(mime_type);
    SUPPORTED_IMAGE_MIME_TYPES.contains(&base.as_str())
}

// Image MIME sniffing comes from the shared [`crate::utils::mime`] port of
// upstream `utils/mime.ts`.

// ---- clipboard backends ----

/// Convert unsupported image formats to PNG using Photon.
/// Returns `None` if conversion is unavailable or fails.
async fn convert_to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let engine = super::photon::load_photon().await?;
    let image = engine.new_from_byteslice(bytes)?;
    engine.get_bytes(&image)
}

/// Backend outcome. Upstream distinguishes `undefined` (backend failed or is
/// absent) from `null` (backend reached, no image) so an empty Wayland
/// clipboard must not fall through to stale X11 clipboard contents.
enum BackendImage {
    Found(ClipboardImage),
    Empty,
    Failed,
}

async fn read_clipboard_image_via_wl_paste() -> BackendImage {
    match try_read_clipboard_image_via_wl_paste().await {
        Ok(Some(image)) => BackendImage::Found(image),
        Ok(None) => BackendImage::Empty,
        Err(()) => BackendImage::Failed,
    }
}

async fn try_read_clipboard_image_via_wl_paste() -> Result<Option<ClipboardImage>, ()> {
    let Some(list) = run_clipboard_command(
        "wl-paste",
        &["--list-types".to_string()],
        Some(RunClipboardCommandOptions {
            timeout_ms: DEFAULT_LIST_TIMEOUT_MS,
            ..Default::default()
        }),
    ) else {
        return Err(());
    };

    let types: Vec<String> = String::from_utf8_lossy(&list)
        .lines()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let Some(selected_type) = select_preferred_image_mime_type(&types) else {
        return Ok(None);
    };

    let Some(data) = run_clipboard_command(
        "wl-paste",
        &[
            "--type".to_string(),
            selected_type.clone(),
            "--no-newline".to_string(),
        ],
        None,
    ) else {
        return Err(());
    };
    if data.is_empty() {
        return Ok(None);
    }

    Ok(Some(ClipboardImage {
        bytes: data,
        mime_type: base_mime_type(&selected_type),
    }))
}

/// On WSL, the Linux clipboard (Wayland/X11) does not receive image data from
/// Windows screenshots (Win+Shift+S). PowerShell can access the Windows
/// clipboard directly, so we use it as a fallback.
async fn read_clipboard_image_via_power_shell() -> Option<ClipboardImage> {
    let tmp_file =
        std::env::temp_dir().join(format!("pi-wsl-clip-{}.png", super::clipboard::uuid()));

    let result: Option<ClipboardImage> = async {
		let Some(win_path_result) = run_clipboard_command(
			"wslpath",
			&[tmp_file.to_string_lossy().to_string()],
			Some(RunClipboardCommandOptions { timeout_ms: DEFAULT_LIST_TIMEOUT_MS, ..Default::default() }),
		) else {
			return None;
		};

		let win_path = String::from_utf8_lossy(&win_path_result).trim().to_string();
		if win_path.is_empty() {
			return None;
		}

		let ps_quoted_win_path = win_path.replace('\'', "''");
		let ps_script = [
			"Add-Type -AssemblyName System.Windows.Forms".to_string(),
			"Add-Type -AssemblyName System.Drawing".to_string(),
			format!("$path = '{ps_quoted_win_path}'"),
			"$img = [System.Windows.Forms.Clipboard]::GetImage()".to_string(),
			"if ($img) { $img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); Write-Output 'ok' } else { Write-Output 'empty' }"
				.to_string(),
		]
		.join("; ");

		let result = run_clipboard_command(
			"powershell.exe",
			&["-NoProfile".to_string(), "-Command".to_string(), ps_script],
			Some(RunClipboardCommandOptions { timeout_ms: DEFAULT_POWERSHELL_TIMEOUT_MS, ..Default::default() }),
		)?;
		let output = String::from_utf8_lossy(&result).trim().to_string();
		if output != "ok" {
			return None;
		}

		let bytes = std::fs::read(&tmp_file).ok()?;
		if bytes.is_empty() {
			return None;
		}

		Some(ClipboardImage { bytes, mime_type: "image/png".to_string() })
	}
	.await;
    let _ = std::fs::remove_file(&tmp_file);
    result
}

async fn read_clipboard_image_via_xclip() -> BackendImage {
    match try_read_clipboard_image_via_xclip().await {
        Ok(Some(image)) => BackendImage::Found(image),
        Ok(None) => BackendImage::Empty,
        Err(()) => BackendImage::Failed,
    }
}

async fn try_read_clipboard_image_via_xclip() -> Result<Option<ClipboardImage>, ()> {
    let Some(targets) = run_clipboard_command(
        "xclip",
        &[
            "-selection".to_string(),
            "clipboard".to_string(),
            "-t".to_string(),
            "TARGETS".to_string(),
            "-o".to_string(),
        ],
        Some(RunClipboardCommandOptions {
            timeout_ms: DEFAULT_LIST_TIMEOUT_MS,
            ..Default::default()
        }),
    ) else {
        return Err(());
    };

    let candidate_types: Vec<String> = String::from_utf8_lossy(&targets)
        .lines()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    let Some(preferred) = select_preferred_image_mime_type(&candidate_types) else {
        return Ok(None);
    };

    let Some(data) = run_clipboard_command(
        "xclip",
        &[
            "-selection".to_string(),
            "clipboard".to_string(),
            "-t".to_string(),
            preferred.clone(),
            "-o".to_string(),
        ],
        None,
    ) else {
        return Err(());
    };
    if data.is_empty() {
        return Ok(None);
    }
    Ok(Some(ClipboardImage {
        bytes: data,
        mime_type: base_mime_type(&preferred),
    }))
}

/// `BackendImage::Empty` = reached, no image; `Failed` = backend failed or is
/// absent (upstream `undefined`).
fn read_clipboard_image_via_native_clipboard() -> BackendImage {
    let Some(clipboard) = native_clipboard() else {
        return BackendImage::Failed;
    };
    match clipboard.get_image() {
        None => BackendImage::Failed,
        Some(bytes) if bytes.is_empty() => BackendImage::Empty,
        Some(bytes) => BackendImage::Found(ClipboardImage {
            mime_type: detect_supported_image_mime_type(&bytes)
                .unwrap_or("application/octet-stream")
                .to_string(),
            bytes,
        }),
    }
}

/// Options mirroring upstream's `{ env?, platform? }`.
#[derive(Debug, Clone, Default)]
pub struct ReadClipboardImageOptions {
    pub env: Option<std::collections::HashMap<String, String>>,
    pub platform: Option<Platform>,
}

/// `readClipboardImage`.
pub async fn read_clipboard_image(
    options: Option<&ReadClipboardImageOptions>,
) -> Option<ClipboardImage> {
    let env = options
        .and_then(|o| o.env.clone())
        .unwrap_or_else(|| std::env::vars().collect());
    let platform = options
        .and_then(|o| o.platform)
        .unwrap_or_else(super::clipboard::current_platform);

    if env.contains_key("TERMUX_VERSION") {
        return None;
    }

    let mut image: BackendImage = BackendImage::Failed;

    if platform == Platform::Linux {
        let wsl = super::clipboard::is_wsl(&env);
        if is_wayland_session(&env) || wsl {
            image = read_clipboard_image_via_wl_paste().await;
        }
        if matches!(image, BackendImage::Failed) {
            image = read_clipboard_image_via_xclip().await;
        }
        // Preserve Linux's empty/unavailable distinction if Windows has no image.
        if matches!(image, BackendImage::Empty | BackendImage::Failed) && wsl {
            if let Some(found) = read_clipboard_image_via_power_shell().await {
                image = BackendImage::Found(found);
            }
        }
        if matches!(image, BackendImage::Failed) {
            image = read_clipboard_image_via_native_clipboard();
        }
    } else {
        image = read_clipboard_image_via_native_clipboard();
    }

    let BackendImage::Found(image) = image else {
        return None;
    };

    // Convert unsupported formats (e.g., Windows DIB data wrapped as BMP) to PNG
    if !is_supported_image_mime_type(&image.mime_type) {
        let png_bytes = convert_to_png(&image.bytes).await?;
        return Some(ClipboardImage {
            bytes: png_bytes,
            mime_type: "image/png".to_string(),
        });
    }

    Some(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    static OVERRIDE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn png_bytes() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52,
        ]
    }

    fn env_map(pairs: &[(&str, &str)]) -> std::collections::HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn listing(present: bool) -> Option<Vec<u8>> {
        Some(if present {
            b"text/plain\nimage/png\n".to_vec()
        } else {
            b"text/plain\n".to_vec()
        })
    }

    #[test]
    fn extension_lookup_matches_upstream() {
        assert_eq!(extension_for_image_mime_type("image/png"), Some("png"));
        assert_eq!(
            extension_for_image_mime_type("image/jpeg;charset=utf8"),
            Some("jpg")
        );
        assert_eq!(extension_for_image_mime_type("text/plain"), None);
    }

    #[test]
    fn wayland_session_detection_matches_upstream() {
        assert!(is_wayland_session(&env_map(&[(
            "WAYLAND_DISPLAY",
            "wayland-0"
        )])));
        assert!(is_wayland_session(&env_map(&[(
            "XDG_SESSION_TYPE",
            "wayland"
        )])));
        assert!(!is_wayland_session(&env_map(&[(
            "XDG_SESSION_TYPE",
            "x11"
        )])));
    }

    #[test]
    fn detects_the_complete_gif_signatures() {
        assert_eq!(
            detect_supported_image_mime_type(b"GIF87a"),
            Some("image/gif")
        );
        assert_eq!(
            detect_supported_image_mime_type(b"GIF89a"),
            Some("image/gif")
        );
    }

    /// 58-byte 24-bit BMP, 1x1 red pixel (`createTinyBmp1x1Red24bpp`).
    fn tiny_bmp() -> Vec<u8> {
        let mut buffer = vec![0u8; 58];
        buffer[0] = b'B';
        buffer[1] = b'M';
        buffer[2..6].copy_from_slice(&58u32.to_le_bytes());
        buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
        buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
        buffer[18..22].copy_from_slice(&1i32.to_le_bytes());
        buffer[22..26].copy_from_slice(&1i32.to_le_bytes());
        buffer[26..28].copy_from_slice(&1u16.to_le_bytes());
        buffer[28..30].copy_from_slice(&24u16.to_le_bytes());
        buffer[34..38].copy_from_slice(&4u32.to_le_bytes());
        buffer[56] = 0xff;
        buffer
    }

    #[test]
    fn detects_bmp_files_from_magic_bytes() {
        assert_eq!(
            detect_supported_image_mime_type(&tiny_bmp()),
            Some("image/bmp")
        );
    }

    #[test]
    fn rejects_undecodable_payloads() {
        assert_eq!(detect_supported_image_mime_type(b"bm90LWFuLWltYWdl"), None);
        assert_eq!(detect_supported_image_mime_type(&[]), None);
    }

    /// Wayland/X11: an advertised image type stops the fallback chain; without
    /// one, no image is read and the native seam is never probed.
    #[tokio::test]
    async fn wayland_command_image_presence_stops_fallback() {
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard_command::set_command_runner_override(Box::new(
            |command, args, _input, _t, _m| {
                assert_eq!(command, "wl-paste");
                if args.iter().any(|a| a == "--list-types") {
                    listing(true)
                } else {
                    Some(png_bytes())
                }
            },
        ));
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[("WAYLAND_DISPLAY", "1"), ("DISPLAY", ":0")])),
            platform: Some(Platform::Linux),
        }))
        .await;
        crate::utils::clipboard_command::clear_command_runner_override();
        assert_eq!(
            image,
            Some(ClipboardImage {
                bytes: png_bytes(),
                mime_type: "image/png".to_string()
            })
        );
    }

    #[tokio::test]
    async fn wayland_without_image_types_yields_none() {
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard_command::set_command_runner_override(Box::new(
            |command, args, _input, _t, _m| {
                assert_eq!(command, "wl-paste");
                assert!(args.iter().any(|a| a == "--list-types"));
                listing(false)
            },
        ));
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[("WAYLAND_DISPLAY", "1"), ("DISPLAY", ":0")])),
            platform: Some(Platform::Linux),
        }))
        .await;
        crate::utils::clipboard_command::clear_command_runner_override();
        assert_eq!(image, None);
    }

    #[tokio::test]
    async fn x11_does_not_probe_image_types_when_targets_fails() {
        // Regression test for #9786.
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard_command::set_command_runner_override(Box::new(
            |command, args, _input, _t, _m| {
                assert_eq!(command, "xclip");
                if args.iter().any(|a| a == "TARGETS") {
                    None
                } else {
                    Some(b"hello".to_vec())
                }
            },
        ));
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[("DISPLAY", ":0")])),
            platform: Some(Platform::Linux),
        }))
        .await;
        crate::utils::clipboard_command::clear_command_runner_override();
        assert_eq!(image, None);
    }

    #[tokio::test]
    async fn x11_does_not_probe_unadvertised_image_types() {
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard_command::set_command_runner_override(Box::new(
            |command, args, _input, _t, _m| {
                assert_eq!(command, "xclip");
                if args.iter().any(|a| a == "TARGETS") {
                    Some(b"image/png\n".to_vec())
                } else if args.iter().any(|a| a == "image/png") {
                    None
                } else {
                    Some(b"hello".to_vec())
                }
            },
        ));
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[("DISPLAY", ":0")])),
            platform: Some(Platform::Linux),
        }))
        .await;
        crate::utils::clipboard_command::clear_command_runner_override();
        assert_eq!(image, None);
    }

    #[tokio::test]
    async fn termux_does_not_read_image_clipboards() {
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard::clear_native_clipboard();
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[("TERMUX_VERSION", "0.119")])),
            platform: Some(Platform::Linux),
        }))
        .await;
        assert_eq!(image, None);
    }

    #[tokio::test]
    async fn darwin_reads_native_image_once() {
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard::set_native_clipboard(Arc::new(StaticImageClipboard));
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[])),
            platform: Some(Platform::Darwin),
        }))
        .await;
        crate::utils::clipboard::clear_native_clipboard();
        assert_eq!(
            image,
            Some(ClipboardImage {
                bytes: png_bytes(),
                mime_type: "image/png".to_string()
            })
        );
    }

    #[tokio::test]
    async fn returns_none_without_a_native_helper_on_windows() {
        let _guard = OVERRIDE_LOCK.lock();
        crate::utils::clipboard::clear_native_clipboard();
        let image = read_clipboard_image(Some(&ReadClipboardImageOptions {
            env: Some(env_map(&[])),
            platform: Some(Platform::Windows),
        }))
        .await;
        assert_eq!(image, None);
    }

    struct StaticImageClipboard;
    impl crate::utils::clipboard::NativeClipboard for StaticImageClipboard {
        fn get_image(&self) -> Option<Vec<u8>> {
            Some(png_bytes())
        }
    }

    // Skipped (mock-dependent upstream cases): `native X11 result … stops
    // fallback` variants beyond the ones ported (they assert exactly-once
    // native-call counts through vi.mock), `Wayland falls back to X11 after
    // missing module/unavailable display` (needs a native backend that fails —
    // the port's seam returns None, routing to Ok(None) instead of the X11
    // fallback), and `WSL: tries PowerShell before a broken native X11 bridge`
    // (writes files from the mocked powershell.exe).
}
