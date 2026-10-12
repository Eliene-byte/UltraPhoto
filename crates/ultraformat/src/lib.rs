//! # ultraformat
//!
//! Reading and writing image files into and out of [`ultrapixel::Surface`].
//!
//! Two things matter here and they pull in opposite directions:
//!
//! * **Fidelity.** A 16-bit TIFF must not be silently truncated to 8 bits, an
//!   embedded ICC profile must survive a round trip, and CMYK JPEG has to
//!   invert correctly (that format stores inverted values).
//! * **Speed.** `image` handles the codec details; what we add is the
//!   conversion into planar, tiled, normalised-`f32` storage, which is where
//!   naive editors do a full-canvas copy. We convert row-band by row-band so
//!   the working set stays small.
//!
//! Codec features are gated so a minimal build only carries what it uses.

use ultrapixel::{Depth, PixelFormat, Rect, Surface};

pub mod decode;
pub mod encode;
pub mod metadata;

#[cfg(test)]
mod tests;

pub use decode::{load, load_from_bytes, sniff_format, LoadError, LoadOptions};
pub use encode::{save, save_to_vec, SaveError, SaveOptions};
pub use metadata::{FileFormat, ImageMetadata};

/// Everything a decoder hands back.
pub struct LoadedImage {
    /// Decoded pixels, already in the document's planar storage.
    pub surface: Surface,
    pub metadata: ImageMetadata,
    /// Embedded ICC profile, kept verbatim so colour management can use it.
    pub icc_profile: Option<Vec<u8>>,
}

/// Rounds a row-band conversion into a destination rect.
///
/// Decoders emit band by band; this maps a band index to the rect it covers
/// without ever materialising the full canvas.
#[inline]
pub(crate) fn band_rect(full: Rect, band: usize, band_height: usize) -> Rect {
    let y0 = full.y0 + (band * band_height) as i32;
    let y1 = (y0 + band_height as i32).min(full.y1);
    Rect::new(full.x0, y0, full.x1, y1)
}

/// Number of rows to convert per band.
///
/// Tuned to keep the scratch buffer comfortably inside L2 while still being
/// large enough to amortise the per-row dispatch.
pub(crate) const BAND_ROWS: usize = 64;

/// Chooses the document depth that a decode should target.
///
/// The rule matches what Photoshop does: if the source carries real
/// high-precision data (16-bit integer or float) we keep it, otherwise 8-bit.
/// Alpha never affects depth â€” Photoshop has no "16-bit + 8-bit alpha" mode.
pub(crate) fn depth_for(is_high_precision: bool) -> Depth {
    if is_high_precision {
        Depth::U16
    } else {
        Depth::U8
    }
}

/// Converts an 8-bit `Rgba` slice (as produced by `image`) into a planar
/// [`Surface`], one band at a time.
pub(crate) fn from_rgba8(
    rgba: &[u8],
    width: usize,
    height: usize,
    with_alpha: bool,
    band: usize,
    out: &mut Surface,
) {
    let full = Rect::new(0, 0, width as i32, height as i32);
    let r = band_rect(full, band, BAND_ROWS);
    if r.is_empty() {
        return;
    }
    let stride = width * 4;
    let mut buf_r = vec![0f32; r.width() as usize * r.height() as usize];
    let mut buf_g = vec![0f32; buf_r.len()];
    let mut buf_b = vec![0f32; buf_r.len()];
    let mut buf_a = vec![0f32; buf_r.len()];

    for row in 0..r.height() as usize {
        let sy = r.y0 as usize + row;
        let base = sy * stride;
        let dbase = row * r.width() as usize;
        for col in 0..r.width() as usize {
            let s = base + col * 4;
            let d = dbase + col;
            buf_r[d] = rgba[s] as f32 / 255.0;
            buf_g[d] = rgba[s + 1] as f32 / 255.0;
            buf_b[d] = rgba[s + 2] as f32 / 255.0;
            buf_a[d] = rgba[s + 3] as f32 / 255.0;
        }
    }

    out.plane_mut(0).write_rect(r, &buf_r);
    out.plane_mut(1).write_rect(r, &buf_g);
    out.plane_mut(2).write_rect(r, &buf_b);
    if with_alpha && out.channels() > 3 {
        out.plane_mut(3).write_rect(r, &buf_a);
    }
}


/// Builds a surface with the right format for a decoded file.
pub(crate) fn make_surface(width: usize, height: usize, depth: Depth, with_alpha: bool) -> Surface {
    let mut fmt = if with_alpha {
        PixelFormat::RGBA
    } else {
        PixelFormat::RGB
    };
    if depth == Depth::U8 {
        // 8-bit documents always carry alpha on pixel layers, matching the
        // default "RGB Color/8" mode.
        fmt = PixelFormat::RGBA;
    }
    Surface::new(width, height, depth, fmt)
}
