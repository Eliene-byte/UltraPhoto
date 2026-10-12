//! Lazily-allocated tiled plane storage.
//!
//! A [`Plane`] never allocates the full canvas. It keeps a grid of
//! [`TILE`]×[`TILE`] tiles that materialise on first write and are released on
//! `deallocate`. A 60000×40000 canvas where the user scribbled in the corner
//! costs a handful of 256 KiB tiles instead of 9.6 GB.

use crate::depth::Depth;
use crate::rect::Rect;
use crate::TILE;

/// Samples per tile.
pub const TILE_SAMPLES: usize = TILE * TILE;

/// Index of a tile in the plane's tile grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileIndex {
    pub col: usize,
    pub row: usize,
}

impl TileIndex {
    #[inline]
    pub const fn flat(self, cols: usize) -> usize {
        self.row * cols + self.col
    }
}

/// Raw tile payload, one variant per depth.
#[derive(Debug, Clone, PartialEq)]
pub enum TileData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl TileData {
    #[inline]
    fn new(depth: Depth) -> Self {
        match depth {
            Depth::U8 => TileData::U8(vec![0; TILE_SAMPLES]),
            Depth::U16 => TileData::U16(vec![0; TILE_SAMPLES]),
            Depth::F32 => TileData::F32(vec![0.0f32; TILE_SAMPLES]),
        }
    }

    #[inline]
    pub fn depth(&self) -> Depth {
        match self {
            TileData::U8(_) => Depth::U8,
            TileData::U16(_) => Depth::U16,
            TileData::F32(_) => Depth::F32,
        }
    }
}

/// An allocated tile.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    pub origin: TileIndex,
    pub data: TileData,
}

impl Tile {
    #[inline]
    pub fn byte_len(&self) -> usize {
        TILE_SAMPLES * self.data.depth().sample_bytes()
    }
}

/// A read-only view over a plane, used by kernels that must not allocate.
pub struct PlaneView<'a> {
    plane: &'a Plane,
}

impl<'a> PlaneView<'a> {
    #[inline]
    pub fn new(plane: &'a Plane) -> Self {
        Self { plane }
    }
    #[inline]
    pub fn depth(&self) -> Depth {
        self.plane.depth
    }
    #[inline]
    pub fn tile(&self, idx: TileIndex) -> Option<&'a Tile> {
        self.plane.tile(idx)
    }
    /// Reads a rect into a normalised `f32` buffer, row-major.
    pub fn read_rect(&self, rect: Rect, out: &mut [f32]) {
        self.plane.read_rect(rect, out)
    }
}

/// A single channel of a layer: tiled, lazily allocated, fixed depth.
#[derive(Debug, Clone)]
pub struct Plane {
    depth: Depth,
    width: usize,
    height: usize,
    cols: usize,
    rows: usize,
    tiles: Vec<Option<Tile>>,
}

impl Plane {
    /// Creates a fully-zeroed plane. Tiles are *not* materialised.
    pub fn new(width: usize, height: usize, depth: Depth) -> Self {
        let cols = crate::tiles_for(width);
        let rows = crate::tiles_for(height);
        Self {
            depth,
            width,
            height,
            cols,
            rows,
            tiles: vec![None; cols * rows],
        }
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
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of tiles currently resident. Used by the memory HUD and by tests
    /// that assert we are not silently allocating the whole canvas.
    #[inline]
    pub fn allocated_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_some()).count()
    }

    #[inline]
    pub fn tile(&self, idx: TileIndex) -> Option<&Tile> {
        let flat = idx.flat(self.cols);
        self.tiles.get(flat)?.as_ref()
    }

    #[inline]
    pub fn tile_mut(&mut self, idx: TileIndex) -> Option<&mut Tile> {
        let flat = idx.flat(self.cols);
        self.tiles.get_mut(flat)?.as_mut()
    }

    /// Clamps a rect to the plane bounds and returns it, or `None` when empty.
    #[inline]
    pub fn clamp_rect(&self, rect: Rect) -> Option<Rect> {
        let r = rect.clamp_to(self.width as i32 - 1, self.height as i32 - 1);
        if r.is_empty() {
            None
        } else {
            Some(r)
        }
    }

    /// Tile indices covering a (already clamped) rect.
    pub fn tiles_in(&self, rect: Rect) -> impl Iterator<Item = TileIndex> + '_ {
        let c0 = (rect.x0 as usize) / TILE;
        let c1 = (rect.x1 as usize - 1) / TILE;
        let r0 = (rect.y0 as usize) / TILE;
        let r1 = (rect.y1 as usize - 1) / TILE;
        (r0..=r1).flat_map(move |row| (c0..=c1).map(move |col| TileIndex { col, row }))
    }

    /// Materialises every tile intersecting `rect`.
    pub fn allocate(&mut self, rect: Rect) {
        let Some(rect) = self.clamp_rect(rect) else {
            return;
        };
        let cols = self.cols;
        let depth = self.depth;
        for idx in self.tiles_in(rect).collect::<Vec<_>>() {
            let flat = idx.flat(cols);
            if self.tiles[flat].is_none() {
                self.tiles[flat] = Some(Tile {
                    origin: idx,
                    data: TileData::new(depth),
                });
            }
        }
    }

    /// Releases every tile that does *not* intersect `rect`.
    pub fn deallocate_outside(&mut self, rect: Rect) {
        let Some(rect) = self.clamp_rect(rect) else {
            self.tiles.iter_mut().for_each(|t| *t = None);
            return;
        };
        let keep: Vec<TileIndex> = self.tiles_in(rect).collect();
        let cols = self.cols;
        let mut keep_flags = vec![false; self.tiles.len()];
        for idx in keep {
            keep_flags[idx.flat(cols)] = true;
        }
        for (flat, slot) in self.tiles.iter_mut().enumerate() {
            if !keep_flags[flat] {
                *slot = None;
            }
        }
    }

    /// Materialises every tile in the plane.
    pub fn allocate_all(&mut self) {
        self.allocate(Rect::new(0, 0, self.width as i32, self.height as i32));
    }

    /// Drops all tiles, returning the plane to its zeroed state.
    pub fn clear(&mut self) {
        self.tiles.iter_mut().for_each(|t| *t = None);
    }

    /// Bounding box of resident tiles, or `None` when the plane is fully empty.
    ///
    /// The compositor uses this to skip layers that cannot contribute.
    pub fn allocation_bounds(&self) -> Option<Rect> {
        let mut min_c = usize::MAX;
        let mut min_r = usize::MAX;
        let mut max_c = 0usize;
        let mut max_r = 0usize;
        let mut found = false;
        for (flat, slot) in self.tiles.iter().enumerate() {
            if slot.is_some() {
                found = true;
                let col = flat % self.cols;
                let row = flat / self.cols;
                min_c = min_c.min(col);
                min_r = min_r.min(row);
                max_c = max_c.max(col);
                max_r = max_r.max(row);
            }
        }
        if !found {
            return None;
        }
        Some(Rect::new(
            (min_c * TILE) as i32,
            (min_r * TILE) as i32,
            ((max_c + 1) * TILE).min(self.width) as i32,
            ((max_r + 1) * TILE).min(self.height) as i32,
        ))
    }

    /// Writes a normalised `f32` buffer (row-major, matching `rect`) into the
    /// plane, quantising to this plane's depth. Clips to the plane bounds.
    pub fn write_rect(&mut self, rect: Rect, src: &[f32]) {
        let Some(rect) = self.clamp_rect(rect) else {
            return;
        };
        let w = rect.width() as usize;
        let h = rect.height() as usize;
        debug_assert_eq!(src.len(), w * h, "write_rect source length mismatch");
        let depth = self.depth;

        for row in 0..h {
            let y = rect.y0 as usize + row;
            let base = row * w;
            for col in 0..w {
                let x = rect.x0 as usize + col;
                self.set_sample(x, y, depth.from_normalized(src[base + col]));
            }
        }
    }

    /// Reads a rect into a normalised `f32` buffer, row-major.
    ///
    /// Unallocated tiles read as zero — that is the "nothing painted here yet"
    /// state, which matches Photoshop.
    pub fn read_rect(&self, rect: Rect, out: &mut [f32]) {
        let Some(rect) = self.clamp_rect(rect) else {
            out.fill(0.0);
            return;
        };
        let w = rect.width() as usize;
        let h = rect.height() as usize;
        debug_assert_eq!(out.len(), w * h, "read_rect destination length mismatch");
        let depth = self.depth;

        for row in 0..h {
            let y = rect.y0 as usize + row;
            let base = row * w;
            for col in 0..w {
                let x = rect.x0 as usize + col;
                out[base + col] = depth.to_normalized(self.sample(x, y));
            }
        }
    }

    /// Raw sample at `(x, y)` in native units (0..255, 0..65535, or raw f32).
    #[inline]
    pub fn sample(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = TileIndex {
            col: x / TILE,
            row: y / TILE,
        };
        let Some(tile) = self.tile(idx) else {
            return 0.0;
        };
        let lx = x % TILE;
        let ly = y % TILE;
        let off = ly * TILE + lx;
        match &tile.data {
            TileData::U8(v) => v[off] as f32,
            TileData::U16(v) => v[off] as f32,
            TileData::F32(v) => v[off],
        }
    }

    /// Normalised `0..1` sample at `(x, y)`.
    #[inline]
    pub fn sample_normalized(&self, x: usize, y: usize) -> f32 {
        self.depth.to_normalized(self.sample(x, y))
    }

    /// Normalised write at `(x, y)`; quantises to the plane's depth and
    /// materialises the tile on demand. This is the main write API.
    #[inline]
    pub fn set_sample_normalized(&mut self, x: usize, y: usize, value: f32) {
        let n = self.depth.from_normalized(value);
        self.set_sample(x, y, n);
    }

    /// Raw write at `(x, y)`. Materialises the tile on demand.
    #[inline]
    pub fn set_sample(&mut self, x: usize, y: usize, value: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = TileIndex {
            col: x / TILE,
            row: y / TILE,
        };
        let depth = self.depth;
        let lx = x % TILE;
        let ly = y % TILE;
        let off = ly * TILE + lx;
        let tile = match self.tiles.get_mut(idx.flat(self.cols)) {
            Some(slot) => slot.get_or_insert_with(|| Tile {
                origin: idx,
                data: TileData::new(depth),
            }),
            None => return,
        };
        match &mut tile.data {
            TileData::U8(v) => v[off] = value.round().clamp(0.0, 255.0) as u8,
            TileData::U16(v) => v[off] = value.round().clamp(0.0, 65535.0) as u16,
            TileData::F32(v) => v[off] = value,
        }
    }

    /// Fills a rect with a normalised constant. Counts as a write, so the
    /// tiles become resident.
    pub fn fill_rect(&mut self, rect: Rect, value: f32) {
        let Some(rect) = self.clamp_rect(rect) else {
            return;
        };
        let w = rect.width() as usize;
        let h = rect.height() as usize;
        let depth = self.depth;
        for row in 0..h {
            let y = rect.y0 as usize + row;
            for col in 0..w {
                let x = rect.x0 as usize + col;
                self.set_sample(x, y, depth.from_normalized(value));
            }
        }
    }

    /// Installs or releases a whole tile.
    ///
    /// This is the primitive undo is built on: writing `None` releases the
    /// tile, which is how undoing the first dab on an empty canvas gives the
    /// memory straight back.
    pub fn set_tile(&mut self, idx: TileIndex, data: Option<&TileData>) {
        if idx.col >= self.cols || idx.row >= self.rows {
            return;
        }
        let flat = idx.flat(self.cols);
        match data {
            Some(d) => {
                debug_assert_eq!(d.depth(), self.depth, "tile depth must match plane depth");
                self.tiles[flat] = Some(Tile {
                    origin: idx,
                    data: d.clone(),
                });
            }
            None => self.tiles[flat] = None,
        }
    }

    /// Current contents of a tile, or `None` when it is unallocated.
    #[inline]
    pub fn tile_data(&self, idx: TileIndex) -> Option<&TileData> {
        self.tile(idx).map(|t| &t.data)
    }

    /// Byte size of resident tiles — what the memory HUD reports.
    pub fn resident_bytes(&self) -> usize {
        self.tiles
            .iter()
            .filter_map(|t| t.as_ref().map(|tile| tile.byte_len()))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty_and_stays_light() {
        let p = Plane::new(8192, 8192, Depth::F32);
        assert_eq!(p.allocated_tiles(), 0);
        assert_eq!(p.resident_bytes(), 0);
        assert!(p.allocation_bounds().is_none());
    }

    #[test]
    fn lazy_allocation_only_touches_painted_tiles() {
        let mut p = Plane::new(8192, 8192, Depth::F32);
        p.set_sample(10, 10, 0.5);
        // One tile in the top-left corner only.
        assert_eq!(p.allocated_tiles(), 1);
        assert_eq!(p.resident_bytes(), TILE_SAMPLES * 4);

        p.set_sample(8000, 10, 1.0);
        assert_eq!(p.allocated_tiles(), 2);

        // Unpainted pixels read as zero, like Photoshop.
        assert_eq!(p.sample(4000, 4000), 0.0);
        assert_eq!(p.sample(10, 10), 0.5);
    }

    #[test]
    fn deallocation_reclaims() {
        let mut p = Plane::new(4096, 4096, Depth::U8);
        p.set_sample(3000, 3000, 1.0);
        assert_eq!(p.allocated_tiles(), 1);
        p.deallocate_outside(Rect::new(0, 0, 100, 100));
        assert_eq!(p.allocated_tiles(), 0);
    }

    #[test]
    fn quantization_clamps_per_depth() {
        let mut p = Plane::new(16, 16, Depth::U8);
        p.set_sample_normalized(0, 0, 2.0);
        assert_eq!(p.sample_normalized(0, 0), 1.0);
        p.set_sample_normalized(1, 1, -5.0);
        assert_eq!(p.sample_normalized(1, 1), 0.0);
        // Native units are preserved exactly for integer depths.
        assert_eq!(p.sample(0, 0), 255.0);
    }

    #[test]
    fn rect_roundtrip() {
        let mut p = Plane::new(300, 300, Depth::F32);
        let rect = Rect::new(10, 20, 40, 50);
        let data: Vec<f32> = (0..(rect.width() * rect.height()) as usize)
            .map(|i| i as f32 / (data_len(&rect)) as f32)
            .collect();
        p.write_rect(rect, &data);
        let mut out = vec![0.0; data.len()];
        p.read_rect(rect, &mut out);
        for (a, b) in data.iter().zip(&out) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    fn data_len(r: &Rect) -> usize {
        r.width() as usize * r.height() as usize
    }
}
