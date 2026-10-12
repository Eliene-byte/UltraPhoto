//! Encoding a [`Surface`] back out to a file.
//!
//! Every encoder funnels through [`to_rgba8`]. That is where planar storage
//! becomes interleaved — the one unavoidable copy, since the `image` crate's
//! encoder API takes contiguous buffers. Flattening happens here too, in the
//! same pass, rather than mutating the caller's surface.

use std::io::Cursor;
use std::path::Path;

use image::ImageEncoder;
use thiserror::Error;

use crate::metadata::FileFormat;
use crate::LoadedImage;
use ultrapixel::{Depth, Rect, Surface};

#[derive(Debug, Error)]
pub enum SaveError {
    #[error("cannot write {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot encode image: {0}")]
    Encode(String),
    #[error("{0:?} cannot carry an alpha channel; flatten the document first")]
    NoAlphaSupport(FileFormat),
    #[error("{0:?} cannot carry {1:?} data")]
    UnsupportedDepth(FileFormat, Depth),
    #[error("unsupported output format")]
    UnsupportedFormat,
    /// The format is known but no encoder is wired up yet.
    #[error("writing {0:?} is not implemented yet")]
    UnsupportedFormatFor(FileFormat),
}

/// How to write the file.
#[derive(Debug, Clone)]
pub struct SaveOptions {
    /// Defaults from the output extension.
    pub format: Option<FileFormat>,
    /// JPEG quality, 1..=100. Photoshop's default is 12/100 ≈ 90.
    pub jpeg_quality: u8,
    /// Background used when flattening alpha (JPEG has no alpha).
    pub flatten_background: [f32; 4],
    /// Embed the document's ICC profile.
    pub embed_icc: bool,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            format: None,
            jpeg_quality: 90,
            flatten_background: [1.0, 1.0, 1.0, 1.0],
            embed_icc: false,
        }
    }
}

/// Writes a surface to `path`.
pub fn save(
    surface: &Surface,
    icc: Option<&[u8]>,
    path: &Path,
    opts: &SaveOptions,
) -> Result<(), SaveError> {
    let format = opts
        .format
        .or_else(|| crate::decode::format_from_path(path))
        .ok_or(SaveError::UnsupportedFormat)?;

    let bytes = encode(surface, icc, format, opts)?;
    std::fs::write(path, bytes).map_err(|source| SaveError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Encodes a surface into a byte buffer.
pub fn save_to_vec(
    surface: &Surface,
    icc: Option<&[u8]>,
    format: FileFormat,
    opts: &SaveOptions,
) -> Result<Vec<u8>, SaveError> {
    encode(surface, icc, format, opts)
}

fn encode(
    surface: &Surface,
    icc: Option<&[u8]>,
    format: FileFormat,
    opts: &SaveOptions,
) -> Result<Vec<u8>, SaveError> {
    match format {
        FileFormat::Png => encode_png(surface, icc, opts),
        FileFormat::Jpeg => encode_jpeg(surface, opts),
        FileFormat::Tiff => encode_tiff(surface, icc, opts),
        FileFormat::WebP => encode_webp(surface),
        other => Err(SaveError::UnsupportedFormatFor(other)),
    }
}

/// Converts planar storage into straight-alpha `Rgba8`.
///
/// When `flatten` is `Some`, transparency is composited onto that background
/// in the same pass. This never mutates `surface`.
fn to_rgba8(surface: &Surface, flatten: Option<[f32; 4]>) -> (Vec<u8>, usize, usize) {
    let w = surface.width();
    let h = surface.height();
    let full = Rect::new(0, 0, w as i32, h as i32);
    let n = w * h;
    let channels = surface.channels();

    let mut planes = [
        vec![0f32; n],
        vec![0f32; n],
        vec![0f32; n],
        vec![0f32; n],
    ];
    for (i, buf) in planes.iter_mut().enumerate().take(channels) {
        surface.plane(i).read_rect(full, buf);
    }

    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        let (r, g, b, a);
        if channels > 3 {
            let av = planes[3][i].clamp(0.0, 1.0);
            match flatten {
                Some(bg) => {
                    let inv = 1.0 - av;
                    r = planes[0][i] * av + bg[0] * inv;
                    g = planes[1][i] * av + bg[1] * inv;
                    b = planes[2][i] * av + bg[2] * inv;
                }
                None => {
                    r = planes[0][i];
                    g = planes[1][i];
                    b = planes[2][i];
                }
            }
            a = if flatten.is_some() { 1.0 } else { av };
        } else {
            r = planes[0][i];
            g = planes[1][i];
            b = planes[2][i];
            a = 1.0;
        }
        out[i * 4] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 2] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    (out, w, h)
}

fn encode_png(surface: &Surface, icc: Option<&[u8]>, _opts: &SaveOptions) -> Result<Vec<u8>, SaveError> {
    use image::codecs::png::{CompressionType, FilterType, PngEncoder};

    let (buf, w, h) = to_rgba8(surface, None);
    let mut out = Vec::with_capacity(buf.len() / 2);
    let mut enc = PngEncoder::new_with_quality(
        Cursor::new(&mut out),
        CompressionType::Default,
        FilterType::Adaptive,
    );
    if let Some(profile) = icc {
        enc.set_icc_profile(profile.to_vec())
            .map_err(|e| SaveError::Encode(e.to_string()))?;
    }
    enc.write_image(&buf, w as u32, h as u32, image::ExtendedColorType::Rgba8)
        .map_err(|e| SaveError::Encode(e.to_string()))?;
    Ok(out)
}

fn encode_jpeg(surface: &Surface, opts: &SaveOptions) -> Result<Vec<u8>, SaveError> {
    let mut out = Vec::with_capacity(surface.resident_bytes() / 4);
    // JPEG cannot store alpha, so it must be flattened before it leaves here.
    let (buf, w, h) = to_rgba8(surface, Some(opts.flatten_background));
    let mut rgb = Vec::with_capacity(buf.len() / 4 * 3);
    for px in buf.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
        Cursor::new(&mut out),
        opts.jpeg_quality.clamp(1, 100),
    );
    enc.encode(&rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8)
        .map_err(|e| SaveError::Encode(e.to_string()))?;
    Ok(out)
}

fn encode_tiff(surface: &Surface, icc: Option<&[u8]>, _opts: &SaveOptions) -> Result<Vec<u8>, SaveError> {
    let (buf, w, h) = to_rgba8(surface, None);
    let mut out = Vec::with_capacity(buf.len());
    // `TiffEncoder` requires `Seek`; a Cursor guarantees it without pinning
    // the encoder to a concrete writer type.
    let mut enc = image::codecs::tiff::TiffEncoder::new(Cursor::new(&mut out));
    if let Some(profile) = icc {
        enc.set_icc_profile(profile.to_vec())
            .map_err(|e| SaveError::Encode(e.to_string()))?;
    }
    enc.write_image(&buf, w as u32, h as u32, image::ExtendedColorType::Rgba8)
        .map_err(|e| SaveError::Encode(e.to_string()))?;
    Ok(out)
}

fn encode_webp(surface: &Surface) -> Result<Vec<u8>, SaveError> {
    let (buf, w, h) = to_rgba8(surface, None);
    let rgba = image::RgbaImage::from_raw(w as u32, h as u32, buf)
        .ok_or_else(|| SaveError::Encode("WebP needs a tightly packed RGBA buffer".into()))?;
    let mut out = Vec::new();
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::WebP)
        .map_err(|e| SaveError::Encode(e.to_string()))?;
    Ok(out)
}

/// Re-encodes a loaded image, which is what "Save As" does when the user
/// changes the format without touching the pixels.
pub fn rewrite(img: &LoadedImage, path: &Path, opts: &SaveOptions) -> Result<(), SaveError> {
    save(&img.surface, img.icc_profile.as_deref(), path, opts)
}