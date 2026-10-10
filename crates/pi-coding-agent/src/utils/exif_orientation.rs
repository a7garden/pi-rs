//! Port of `packages/coding-agent/src/utils/exif-orientation.ts` — EXIF
//! orientation parsing (JPEG/WebP/TIFF by hand, pure) and applying it to a
//! [`PhotonImage`]. Upstream delegates the flip primitives to photon; here
//! they are the [`PhotonImage::fliph`]/[`PhotonImage::flipv`] pixel loops, so
//! the whole module is engine-independent (crate docs, deviation D14).

use super::photon::PhotonImage;

fn read_orientation_from_tiff(bytes: &[u8], tiff_start: usize) -> u16 {
    if tiff_start + 8 > bytes.len() {
        return 1;
    }

    let byte_order = ((bytes[tiff_start] as u16) << 8) | bytes[tiff_start + 1] as u16;
    let le = byte_order == 0x4949;

    let read16 = |pos: usize| -> u16 {
        if le {
            return bytes[pos] as u16 | ((bytes[pos + 1] as u16) << 8);
        }
        ((bytes[pos] as u16) << 8) | bytes[pos + 1] as u16
    };

    let read32 = |pos: usize| -> u32 {
        if le {
            return bytes[pos] as u32
                | ((bytes[pos + 1] as u32) << 8)
                | ((bytes[pos + 2] as u32) << 16)
                | ((bytes[pos + 3] as u32) << 24);
        }
        ((bytes[pos] as u32) << 24)
            | ((bytes[pos + 1] as u32) << 16)
            | ((bytes[pos + 2] as u32) << 8)
            | bytes[pos + 3] as u32
    };

    let ifd_offset = read32(tiff_start + 4) as usize;
    let ifd_start = tiff_start + ifd_offset;
    if ifd_start + 2 > bytes.len() {
        return 1;
    }

    let entry_count = read16(ifd_start);
    for i in 0..entry_count as usize {
        let entry_pos = ifd_start + 2 + i * 12;
        if entry_pos + 12 > bytes.len() {
            return 1;
        }

        if read16(entry_pos) == 0x0112 {
            let value = read16(entry_pos + 8);
            return if (1..=8).contains(&value) { value } else { 1 };
        }
    }

    1
}

fn find_jpeg_tiff_offset(bytes: &[u8]) -> isize {
    let mut offset: usize = 2;
    while offset < bytes.len() - 1 {
        if bytes[offset] != 0xff {
            return -1;
        }
        let marker = bytes[offset + 1];
        if marker == 0xff {
            offset += 1;
            continue;
        }

        if marker == 0xe1 {
            if offset + 4 >= bytes.len() {
                return -1;
            }
            let segment_start = offset + 4;
            if segment_start + 6 > bytes.len() {
                return -1;
            }
            if has_exif_header(bytes, segment_start) {
                return (segment_start + 6) as isize;
            }
        }

        if offset + 4 > bytes.len() {
            return -1;
        }
        let length = ((bytes[offset + 2] as usize) << 8) | bytes[offset + 3] as usize;
        offset += 2 + length;
    }

    -1
}

fn find_webp_tiff_offset(bytes: &[u8]) -> isize {
    let mut offset: usize = 12;
    while offset + 8 <= bytes.len() {
        let chunk_id = [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ];
        let chunk_size = bytes[offset + 4] as usize
            | ((bytes[offset + 5] as usize) << 8)
            | ((bytes[offset + 6] as usize) << 16)
            | ((bytes[offset + 7] as usize) << 24);
        let data_start = offset + 8;

        if &chunk_id == b"EXIF" {
            if data_start + chunk_size > bytes.len() {
                return -1;
            }
            // Some WebP files have "Exif\0\0" prefix before the TIFF header
            let tiff_start = if chunk_size >= 6 && has_exif_header(bytes, data_start) {
                data_start + 6
            } else {
                data_start
            };
            return tiff_start as isize;
        }

        // RIFF chunks are padded to even size
        offset = data_start + chunk_size + (chunk_size % 2);
    }

    -1
}

fn has_exif_header(bytes: &[u8], offset: usize) -> bool {
    bytes[offset] == 0x45
        && bytes[offset + 1] == 0x78
        && bytes[offset + 2] == 0x69
        && bytes[offset + 3] == 0x66
        && bytes[offset + 4] == 0x00
        && bytes[offset + 5] == 0x00
}

/// `getExifOrientation` — 1 (no transform) through 8.
pub fn get_exif_orientation(bytes: &[u8]) -> u16 {
    let tiff_offset: isize;

    // JPEG: starts with FF D8
    if bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8 {
        tiff_offset = find_jpeg_tiff_offset(bytes);
    }
    // WebP: starts with RIFF....WEBP
    else if bytes.len() >= 12
        && bytes[0] == 0x52
        && bytes[1] == 0x49
        && bytes[2] == 0x46
        && bytes[3] == 0x46
        && bytes[8] == 0x57
        && bytes[9] == 0x45
        && bytes[10] == 0x42
        && bytes[11] == 0x50
    {
        tiff_offset = find_webp_tiff_offset(bytes);
    } else {
        return 1;
    }

    if tiff_offset == -1 {
        return 1;
    }
    read_orientation_from_tiff(bytes, tiff_offset as usize)
}

type DstIndexFn = fn(x: usize, y: usize, w: usize, h: usize) -> usize;

fn rotate90(image: &PhotonImage, dst_index: DstIndexFn) -> PhotonImage {
    let w = image.get_width() as usize;
    let h = image.get_height() as usize;
    let src = &image.rgba;
    let mut dst = vec![0u8; src.len()];

    for y in 0..h {
        for x in 0..w {
            let src_idx = (y * w + x) * 4;
            let dst_idx = dst_index(x, y, w, h) * 4;
            dst[dst_idx] = src[src_idx];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
            dst[dst_idx + 3] = src[src_idx + 3];
        }
    }

    PhotonImage::new(h as u32, w as u32, dst)
}

fn rotate_cw_dst_index(x: usize, y: usize, _w: usize, h: usize) -> usize {
    x * h + (h - 1 - y)
}

fn rotate_ccw_dst_index(x: usize, y: usize, w: usize, h: usize) -> usize {
    (w - 1 - x) * h + y
}

/// Flip orientations mutate in place; rotations return a new image
/// (`applyExifOrientation`). Takes and returns the image so flips and
/// rotations share one signature.
pub fn apply_exif_orientation(mut image: PhotonImage, original_bytes: &[u8]) -> PhotonImage {
    let orientation = get_exif_orientation(original_bytes);
    if orientation == 1 {
        return image;
    }

    match orientation {
        2 => {
            image.fliph();
            image
        }
        3 => {
            image.fliph();
            image.flipv();
            image
        }
        4 => {
            image.flipv();
            image
        }
        5 => {
            let mut rotated = rotate90(&image, rotate_cw_dst_index);
            rotated.fliph();
            rotated
        }
        6 => rotate90(&image, rotate_cw_dst_index),
        7 => {
            let mut rotated = rotate90(&image, rotate_ccw_dst_index);
            rotated.fliph();
            rotated
        }
        8 => rotate90(&image, rotate_ccw_dst_index),
        _ => image,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TIFF header (`II`, IFD at offset 8) with a single orientation entry.
    fn tiff_bytes_with_orientation(orientation: u16) -> Vec<u8> {
        let mut bytes = vec![
            0x49,
            0x49,
            0x2a,
            0x00,
            0x08,
            0x00,
            0x00,
            0x00, // II*, IFD @ 8
            0x01,
            0x00, // 1 entry
            0x12,
            0x01,
            0x03,
            0x00, // tag 0x0112, type SHORT
            0x01,
            0x00,
            0x00,
            0x00, // count 1
            orientation as u8,
            (orientation >> 8) as u8,
            0x00,
            0x00, // value
            0x00,
            0x00,
            0x00,
            0x00, // next IFD
        ];
        // "Exif\0\0" prefix as it appears in a JPEG APP1 segment.
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe1, 0x00, 0x00];
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&bytes);
        bytes = jpeg;
        bytes
    }

    fn jpeg_bytes_with_orientation(orientation: u16) -> Vec<u8> {
        let mut bytes = tiff_bytes_with_orientation(orientation);
        let segment_length = (bytes.len() - 4) as u16; // payload after FF E1 + length
        bytes[4] = (segment_length >> 8) as u8;
        bytes[5] = (segment_length & 0xff) as u8;
        bytes
    }

    fn webp_bytes_with_orientation(orientation: u16) -> Vec<u8> {
        let tiff = {
            let full = tiff_bytes_with_orientation(orientation);
            full[12..].to_vec() // strip the JPEG wrapper and "Exif\0\0", keep raw TIFF
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        let payload = 4 + 8 + tiff.len();
        bytes.extend_from_slice(&(payload as u32).to_le_bytes());
        bytes.extend_from_slice(b"WEBP");
        bytes.extend_from_slice(b"EXIF");
        bytes.extend_from_slice(&(tiff.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&tiff);
        bytes
    }

    #[test]
    fn reads_orientation_one_through_eight_from_jpeg() {
        for orientation in 1u16..=8 {
            assert_eq!(
                get_exif_orientation(&jpeg_bytes_with_orientation(orientation)),
                orientation
            );
        }
    }

    #[test]
    fn reads_orientation_from_webp() {
        assert_eq!(get_exif_orientation(&webp_bytes_with_orientation(6)), 6);
    }

    #[test]
    fn non_exif_images_report_orientation_one() {
        assert_eq!(get_exif_orientation(&[]), 1);
        assert_eq!(get_exif_orientation(&[0x89, 0x50]), 1);
        assert_eq!(get_exif_orientation(&[0xff, 0xd8, 0xff, 0xe0]), 1);
    }

    #[test]
    fn orientation_two_flips_horizontally() {
        let image = PhotonImage::new(2, 1, vec![1, 0, 0, 1, 2, 0, 0, 2]);
        let applied = apply_exif_orientation(image, &jpeg_bytes_with_orientation(2));
        assert_eq!(applied.rgba, vec![2, 0, 0, 2, 1, 0, 0, 1]);
        assert_eq!((applied.get_width(), applied.get_height()), (2, 1));
    }

    #[test]
    fn orientation_three_rotates_one_eighty() {
        let image = PhotonImage::new(2, 1, vec![1, 0, 0, 1, 2, 0, 0, 2]);
        let applied = apply_exif_orientation(image, &jpeg_bytes_with_orientation(3));
        assert_eq!(applied.rgba, vec![2, 0, 0, 2, 1, 0, 0, 1]);
    }

    #[test]
    fn orientation_six_rotates_dimensions() {
        // 2x1 image, pixel (0,0)=A (1,0)=B; 90° CW rotation yields 1x2, top = A.
        let image = PhotonImage::new(2, 1, vec![1, 0, 0, 1, 2, 0, 0, 2]);
        let applied = apply_exif_orientation(image, &jpeg_bytes_with_orientation(6));
        assert_eq!((applied.get_width(), applied.get_height()), (1, 2));
        assert_eq!(applied.rgba, vec![1, 0, 0, 1, 2, 0, 0, 2]);
    }

    #[test]
    fn orientation_eight_rotates_the_other_way() {
        let image = PhotonImage::new(2, 1, vec![1, 0, 0, 1, 2, 0, 0, 2]);
        let applied = apply_exif_orientation(image, &jpeg_bytes_with_orientation(8));
        assert_eq!((applied.get_width(), applied.get_height()), (1, 2));
        assert_eq!(applied.rgba, vec![2, 0, 0, 2, 1, 0, 0, 1]);
    }

    #[test]
    fn orientation_one_returns_image_unchanged() {
        let image = PhotonImage::new(1, 1, vec![9, 9, 9, 9]);
        let applied = apply_exif_orientation(image.clone(), &[]);
        assert_eq!(applied, image);
    }
}
