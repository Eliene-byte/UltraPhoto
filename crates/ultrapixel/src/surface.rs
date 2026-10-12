//! A [`Surface`] is the pixel data of one layer: a [`PixelFormat`] plus one
//! [`Plane`] per channel, plus the layer's content bounds.
//!
//! Layers keep a `content_bounds` rect that grows as the user paints. Ops that
//! only need to look at "where the ink is" (selection expand, layer bounds
//! readout, GPU dispatch sizing) use it and never touch a full canvas walk.

use crate::depth::Depth;
use crate::format::PixelFormat;
use crate::plane::Plane;
use crate::rect::Rect;

/// Pixel storage for a single layer.
#[derive(Debug, Clone)]
pub struct Surface {
    format: PixelFormat,
    depth: Depth,
    planes: Vec<Plane>,
    width: usize,
    height: usize,
    bounds: Rect,
}

impl Surface {
    /// Allocates an empty surface. No planes are materialised until written.
    pub fn new(width: usize, height: usize, depth: Depth, format: PixelFormat) -> Self {
        let n = format.channel_count();
        let planes = (0..n).map(|_| Plane::new(width, height, depth)).collect();
        Self {
            format,
            depth,
            planes,
            width,
            height,
            bounds: Rect::EMPTY,
        }
    }

    /// Surface filled with `color`, fully resident. Used when creating layers
    /// with "Fill with foreground colour" rather than transparency.
    pub fn filled(
        width: usize,
        height: usize,
        depth: Depth,
        format: PixelFormat,
        color: &[f32],
    ) -> Self {
        let mut s = Self::new(width, height, depth, format);
        s.fill_all(color);
        s
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }
    #[inline]
    pub fn depth(&self) -> Depth {
        self.depth
    }
    #[inline]
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    #[inline]
    pub fn channels(&self) -> usize {
        self.planes.len()
    }
    #[inline]
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }
    #[inline]
    pub fn planes_mut(&mut self) -> &mut [Plane] {
        &mut self.planes
    }
    #[inline]
    pub fn plane(&self, ch: usize) -> &Plane {
        &self.planes[ch]
    }
    #[inline]
    pub fn plane_mut(&mut self, ch: usize) -> &mut Plane {
        &mut self.planes[ch]
    }

    /// Bounding box of everything painted on this surface.
    #[inline]
    pub fn content_bounds(&self) -> Rect {
        self.bounds
    }

    /// Union of the per-channel allocation bounds. Recomputed after bulk ops;
    /// cheap enough (one pass over the tile pointers) to call freely.
    pub fn recompute_bounds(&mut self) -> Rect {
        let mut b = Rect::EMPTY;
        for p in &self.planes {
            if let Some(r) = p.allocation_bounds() {
                b = b.union(r);
            }
        }
        self.bounds = b;
        b
    }

    /// Records a written region so `content_bounds` stays accurate.
    ///
    /// Call this after any op that writes pixels; the compositor uses the
    /// bounds to shrink GPU dispatches and to skip empty layers entirely.
    #[inline]
    pub fn note_bounds(&mut self, rect: Rect) {
        self.bounds = self.bounds.union(rect);
    }

    /// Fills every channel with a constant normalised value.
    pub fn fill_all(&mut self, color: &[f32]) {
        let full = Rect::new(0, 0, self.width as i32, self.height as i32);
        for (i, plane) in self.planes.iter_mut().enumerate() {
            let v = color.get(i).copied().unwrap_or(0.0);
            plane.fill_rect(full, v);
        }
        self.note_bounds(full);
    }

    /// Reads the RGBA tuple at `(x, y)` as normalised floats.
    ///
    /// Only RGB and Gray layouts are supported; other models must go through
    /// their own converters. Unpainted pixels read as transparent black.
    pub fn sample_rgba_normalized(&self, x: usize, y: usize) -> [f32; 4] {
        use crate::format::{ChannelLayout, ColorModel};
        let ch = |i: usize| self.planes[i].sample_normalized(x, y);
        match (self.format.color, self.format.layout) {
            (ColorModel::Rgb, ChannelLayout::Rgb) => [ch(0), ch(1), ch(2), 1.0],
            (ColorModel::Rgb, ChannelLayout::Rgba) => [ch(0), ch(1), ch(2), ch(3)],
            (ColorModel::Gray, ChannelLayout::Gray) => {
                let g = ch(0);
                [g, g, g, 1.0]
            }
            (ColorModel::Gray, ChannelLayout::GrayAlpha) => {
                let g = ch(0);
                [g, g, g, ch(1)]
            }
            // Fall back to the first three planes for exotic models.
            _ => [
                ch(0),
                ch(1.min(self.planes.len() - 1)),
                ch(2.min(self.planes.len() - 1)),
                1.0,
            ],
        }
    }

    /// Total resident bytes across all channels.
    pub fn resident_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.resident_bytes()).sum()
    }

    /// Releases every tile outside `rect`. Used by "purge" and by history
    /// compaction after an operation shrinks a layer.
    pub fn purge_outside(&mut self, rect: Rect) {
        for p in &mut self.planes {
            p.deallocate_outside(rect);
        }
        self.recompute_bounds();
    }

    /// Deep-copies the given channel range. Used by undo snapshots.
    pub fn copy_channels(&self, from: usize, count: usize) -> Surface {
        let mut out = Surface::new(self.width, self.height, self.depth, self.format);
        out.bounds = self.bounds;
        for i in from..(from + count).min(self.planes.len()) {
            out.planes[i] = self.planes[i].clone();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ChannelLayout, PixelFormat};

    #[test]
    fn empty_surface_costs_nothing() {
        let s = Surface::new(4096, 4096, Depth::F32, PixelFormat::RGBA);
        assert_eq!(s.resident_bytes(), 0);
        assert!(s.content_bounds().is_empty());
    }

    #[test]
    fn rgba_sample_roundtrip() {
        let mut s = Surface::new(8, 8, Depth::U8, PixelFormat::RGBA);
        for (c, v) in [1.0, 0.5, 0.25, 1.0].into_iter().enumerate() {
            s.plane_mut(c).fill_rect(Rect::new(2, 2, 5, 5), v);
        }
        let px = s.sample_rgba_normalized(3, 3);
        assert!((px[0] - 1.0).abs() < 0.01);
        assert!((px[1] - 0.5).abs() < 0.01);
        assert!((px[2] - 0.25).abs() < 0.01);
        assert!((px[3] - 1.0).abs() < 0.01);
        // Outside the painted area: transparent black.
        assert_eq!(s.sample_rgba_normalized(6, 6), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn format_helpers() {
        assert_eq!(PixelFormat::RGB.with_alpha().layout, ChannelLayout::Rgba);
        assert_eq!(PixelFormat::RGBA.without_alpha().layout, ChannelLayout::Rgb);
        assert_eq!(PixelFormat::GRAY.with_alpha().channel_count(), 2);
        // Alpha is coverage (linear); colour primaries are not.
        assert!(PixelFormat::RGBA.is_coverage_channel(3));
        assert!(!PixelFormat::RGBA.is_coverage_channel(0));
        // CMYK: the 5th channel is under-colour removal, not a colour primary.
        assert!(PixelFormat::CMYK.is_coverage_channel(4));
        assert!(!PixelFormat::CMYK.is_coverage_channel(3));
    }
}
