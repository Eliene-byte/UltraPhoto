//! # ultrapixel
//!
//! Storage primitives for UltraPhoto.
//!
//! Design notes (these are the reasons this crate exists):
//!
//! * **Planar** — every channel of a layer is its own plane. This mirrors
//!   Photoshop's internal model and maps 1:1 onto a `wgpu::Texture` per channel
//!   (or `texture_array` slices), which makes compute kernels trivial to write
//!   and removes the de-interleaving pass that dominates naive editors.
//! * **Tiled** — planes are cut into [`TILE`]×[`TILE`] tiles that are allocated
//!   lazily. A 80000×80000 canvas whose user only painted in one corner costs
//!   almost nothing. This is what keeps us *extremely light*.
//! * **Multi-depth** — a document is 8/16/32-bit as a whole (Photoshop model),
//!   never per-pixel. Inside a document every plane has the same [`Depth`].
//! * **f32 working space** — 32-bit documents keep `f32` samples so filters,
//!   blend modes and GPU kernels never lose precision.

pub mod color;
pub mod depth;
pub mod format;
pub mod plane;
pub mod rect;
pub mod surface;

pub use color::{Lab, LinearRgb, Rgb, Rgba, Srgb, Xyz};
pub use depth::Depth;
pub use format::{ChannelLayout, ColorModel, PixelFormat};
pub use plane::{Plane, PlaneView, Tile, TileData, TileIndex, TILE_SAMPLES};
pub use rect::{DirtyAccumulator, Rect};
pub use surface::Surface;

/// Edge length, in pixels, of a single tile.
///
/// 256 is the sweet spot: small enough that a brush stroke touches very few
/// tiles (so dirty-rect work stays cheap), large enough that per-tile overhead
/// is amortised over enough work to keep the GPU happy (one dispatch per tile).
pub const TILE: usize = 256;

/// Number of tiles needed to cover `len` pixels.
#[inline]
pub const fn tiles_for(len: usize) -> usize {
    let n = len.div_ceil(TILE);
    if n == 0 {
        1
    } else {
        n
    }
}
