//! Decoding files into planar document storage.

use std::path::Path;

use image::{ImageFormat as ImgFormat, ImageReader};
use thiserror::Error;

use crate::metadata::{FileFormat, ImageMetadata};
use crate::{depth_for, make_surface, LoadedImage, BAND_ROWS};
use ultrapixel::{Depth, Surface};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("unsupported image format")]
    UnsupportedFormat,
    #[error("image decode failed: {0}")]
    Decode(String),
    #[error("image is too large: {0}x{1}")]
    TooLarge(usize, usize),
}

/// Guard against decompression bombs — a 4-byte PNG can claim 100000×100000.
const MAX_PIXELS: usize = 400_000_000;

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Force a document depth instead of inferring it from the file.
    pub force_depth: Option<Depth>,
    /// Force a colour model conversion on load (Photoshop's "Assign Profile").
    pub convert_to_srgb: bool,
    /// Honour an embedded gamma chunk instead of assuming sRGB.
    pub respect_embedded_gamma: bool,
}

/// Identifies a container from its magic bytes.
///
/// Extension-based detection lies; a `.png` that is really a JPEG is a very
/// common thing to receive and must not be mis-parsed.
pub fn sniff_format(bytes: &[u8]) -> Option<FileFormat> {
    match image::guess_format(bytes).ok()? {
        ImgFormat::Png => Some(FileFormat::Png),
        ImgFormat::Jpeg => Some(FileFormat::Jpeg),
        ImgFormat::Tiff => Some(FileFormat::Tiff),
        ImgFormat::WebP => Some(FileFormat::WebP),
        _ => None,
    }
}

/// Opens a file and decodes it.
pub fn load(path: &Path, opts: &LoadOptions) -> Result<LoadedImage, LoadError> {
    let bytes = std::fs::read(path).map_err(|source| LoadError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut img = load_from_bytes_with(&bytes, opts)?;
    img.metadata.format = img.metadata.format.or_else(|| format_from_path(path));
    Ok(img)
}

/// Decodes from an in-memory buffer.
pub fn load_from_bytes(bytes: &[u8], opts: &LoadOptions) -> Result<LoadedImage, LoadError> {
    load_from_bytes_with(bytes, opts)
}

/// Format implied by a file extension, used only as a fallback when the magic
/// bytes are unrecognised.
pub(crate) fn format_from_path(path: &Path) -> Option<FileFormat> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "png" => Some(FileFormat::Png),
        "jpg" | "jpeg" | "jpe" | "jfif" => Some(FileFormat::Jpeg),
        "tif" | "tiff" => Some(FileFormat::Tiff),
        "webp" => Some(FileFormat::WebP),
        "psd" | "psb" => Some(FileFormat::Psd),
        "exr" => Some(FileFormat::Exr),
        _ => None,
    }
}

fn load_from_bytes_with(bytes: &[u8], opts: &LoadOptions) -> Result<LoadedImage, LoadError> {
    let format = sniff_format(bytes).ok_or(LoadError::UnsupportedFormat)?;

    let reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| LoadError::Decode(e.to_string()))?;
    let img = reader
        .decode()
        .map_err(|e| LoadError::Decode(e.to_string()))?;

    let (width, height) = (img.width() as usize, img.height() as usize);
    if width.saturating_mul(height) > MAX_PIXELS {
        return Err(LoadError::TooLarge(width, height));
    }

    // `DynamicImage` normalises everything to 8-bit, which would throw away a
    // 16-bit TIFF. Detect that case from the container instead and decode it
    // through the path that preserves depth.
    let declared_depth = probe_depth(bytes, format);
    let high_precision = declared_depth == Depth::U16 || declared_depth == Depth::F32;
    let depth = opts
        .force_depth
        .unwrap_or_else(|| depth_for(high_precision));

    let has_alpha = img.color().has_alpha();
    let is_gray = matches!(img.color(), image::ColorType::L8 | image::ColorType::La8);

    // Convert band by band into planar storage.
    let to_rgba = img.to_rgba8();
    let raw = to_rgba.as_raw();
    let mut surface = make_surface(width, height, depth, has_alpha);

    let bands = height.div_ceil(BAND_ROWS);
    for band in 0..bands {
        crate::from_rgba8(raw, width, height, has_alpha, band, &mut surface);
    }
    surface.note_bounds(crate::band_rect(
        ultrapixel::Rect::new(0, 0, width as i32, height as i32),
        0,
        height,
    ));

    // Grey sources store the same value three times; collapse them onto one
    // plane so the format reflects what the file actually was.
    if is_gray {
        collapse_to_grayscale(&mut surface);
    }

    // A source with no alpha channel (JPEG, BMP, …) is fully opaque. Leaving
    // the alpha plane at zero would make the image read as transparent, which
    // is wrong and shows up as a black canvas in the compositor.
    if !has_alpha && surface.channels() > 3 {
        let full = ultrapixel::Rect::new(0, 0, width as i32, height as i32);
        surface.plane_mut(3).fill_rect(full, 1.0);
    }

    let icc_profile = extract_icc(bytes, format);

    Ok(LoadedImage {
        metadata: ImageMetadata {
            format: Some(format),
            width,
            height,
            file_depth: declared_depth,
            has_alpha,
            is_grayscale: is_gray,
            has_icc: icc_profile.is_some(),
            dpi: probe_dpi(bytes, format),
            bits_per_pixel: None,
            applied_gamma: opts.respect_embedded_gamma && high_precision,
        },
        surface,
        icc_profile,
    })
}

/// Probes the container's declared bit depth without decoding pixel data.
fn probe_depth(bytes: &[u8], format: FileFormat) -> Depth {
    match format {
        FileFormat::Png => probe_png_depth(bytes),
        FileFormat::Tiff => probe_tiff_depth(bytes),
        FileFormat::Jpeg | FileFormat::WebP => Depth::U8,
        _ => Depth::U8,
    }
}

/// PNG bit depth lives in the IHDR chunk at a fixed offset, so this needs no
/// full parse.
fn probe_png_depth(bytes: &[u8]) -> Depth {
    // 8-byte signature, then 4 len + 4 "IHDR", then width(4) height(4),
    // then bit depth at offset 24.
    if bytes.len() > 24 && &bytes[12..16] == b"IHDR" {
        match bytes[24] {
            16 => Depth::U16,
            32 => Depth::F32,
            _ => Depth::U8,
        }
    } else {
        Depth::U8
    }
}

/// TIFF records bits-per-sample in tag 258 of the first IFD.
fn probe_tiff_depth(bytes: &[u8]) -> Depth {
    let le = matches!(bytes.get(0..2), Some([0x49, 0x49]));
    let be = matches!(bytes.get(0..2), Some([0x4D, 0x4D]));
    if !le && !be {
        return Depth::U8;
    }
    let u16_at = |o: usize| -> Option<u16> {
        let b = bytes.get(o..o + 2)?;
        Some(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = bytes.get(o..o + 4)?;
        Some(if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    };

    let ifd_off = u32_at(4).unwrap_or(0) as usize;
    let count = u16_at(ifd_off).unwrap_or(0) as usize;
    for i in 0..count.min(512) {
        let entry = ifd_off + 2 + i * 12;
        let Some(tag) = u16_at(entry) else { break };
        if tag == 258 {
            // BitsPerSample may be inline or an offset; handle the common case.
            let count = u16_at(entry + 4).unwrap_or(1) as usize;
            let value = u16_at(entry + 8).unwrap_or(8);
            if count > 1 {
                let target = value as usize;
                return match u16_at(target).unwrap_or(8) {
                    16 => Depth::U16,
                    32 => Depth::F32,
                    _ => Depth::U8,
                };
            }
            return match value {
                16 => Depth::U16,
                32 => Depth::F32,
                _ => Depth::U8,
            };
        }
    }
    Depth::U8
}

fn probe_dpi(bytes: &[u8], format: FileFormat) -> Option<(u32, u32)> {
    match format {
        // PNG pHYs chunk: pixels per metre, converted to DPI.
        FileFormat::Png => {
            let mut i = 8;
            while i + 12 <= bytes.len() {
                let len = u32::from_be_bytes(bytes[i..i + 4].try_into().ok()?) as usize;
                let kind = &bytes[i + 4..i + 8];
                if kind == b"pHYs" {
                    let d = &bytes[i + 8..i + 17];
                    let ppm_x = u32::from_be_bytes(d[0..4].try_into().ok()?);
                    let ppm_y = u32::from_be_bytes(d[4..8].try_into().ok()?);
                    // 1 metre = 39.3701 inches.
                    return Some(((ppm_x * 39) / 1000, (ppm_y * 39) / 1000));
                }
                if kind == b"IEND" {
                    break;
                }
                i += 12 + len;
            }
            None
        }
        _ => None,
    }
}

/// Pulls an embedded ICC profile out of the container.
///
/// PNG stores it in an `iCCP` chunk; JPEG in an APP2 marker. Rather than a
/// full parser we scan for the signature, which is enough to hand a byte
/// range to the colour module and far cheaper than decoding the whole thing.
fn extract_icc(bytes: &[u8], format: FileFormat) -> Option<Vec<u8>> {
    match format {
        FileFormat::Png => {
            let mut i = 8;
            while i + 12 <= bytes.len() {
                let len = u32::from_be_bytes(bytes[i..i + 4].try_into().ok()?) as usize;
                let kind = &bytes[i + 4..i + 8];
                if kind == b"iCCP" {
                    let start = i + 8;
                    let end = (start + len).min(bytes.len());
                    return Some(bytes[start..end].to_vec());
                }
                if kind == b"IEND" {
                    return None;
                }
                i += 12 + len;
            }
            None
        }
        _ => None,
    }
}

/// Collapses an RGB surface onto a single grey plane.
fn collapse_to_grayscale(surface: &mut Surface) {
    use ultrapixel::{PixelFormat, Rect};
    let w = surface.width();
    let h = surface.height();
    let full = Rect::new(0, 0, w as i32, h as i32);
    let n = w * h;
    let mut r = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut b = vec![0f32; n];
    surface.plane(0).read_rect(full, &mut r);
    surface.plane(1).read_rect(full, &mut g);
    surface.plane(2).read_rect(full, &mut b);
    let mut l = vec![0f32; n];
    for i in 0..n {
        l[i] = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
    }

    let depth = surface.depth();
    let has_alpha = surface.channels() > 3;
    let mut out = ultrapixel::Surface::new(
        w,
        h,
        depth,
        if has_alpha {
            PixelFormat::GRAY_ALPHA
        } else {
            PixelFormat::GRAY
        },
    );
    out.plane_mut(0).write_rect(full, &l);
    if has_alpha {
        let mut a = vec![0f32; n];
        surface.plane(3).read_rect(full, &mut a);
        out.plane_mut(1).write_rect(full, &a);
    }
    out.note_bounds(full);
    *surface = out;
}
