//! Integer rectangles used for dirty-rectangle compositing.

use std::ops::BitOr;

/// Half-open integer rectangle: `x0 <= x < x1`, `y0 <= y < y1`.
///
/// Half-open keeps tile arithmetic branch-free. Every rect in the engine is
/// intersected before it reaches a kernel, so an empty rect is the universal
/// "nothing to do" signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub const EMPTY: Rect = Rect {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    #[inline]
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Builds a rect from inclusive pixel coordinates.
    #[inline]
    pub fn from_inclusive(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self::new(x0, y0, x1 + 1, y1 + 1)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    #[inline]
    pub const fn width(self) -> i32 {
        if self.is_empty() {
            0
        } else {
            self.x1 - self.x0
        }
    }

    #[inline]
    pub const fn height(self) -> i32 {
        if self.is_empty() {
            0
        } else {
            self.y1 - self.y0
        }
    }

    #[inline]
    pub const fn area(self) -> i64 {
        self.width() as i64 * self.height() as i64
    }

    #[inline]
    pub const fn contains(self, x: i32, y: i32) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    /// Expands to include `(x, y)`.
    pub fn include_point(&mut self, x: i32, y: i32) {
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x + 1);
        self.y1 = self.y1.max(y + 1);
    }

    /// Smallest rect containing both `self` and `other`.
    pub fn union(self, other: Rect) -> Rect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Rect::new(
            self.x0.min(other.x0),
            self.y0.min(other.y0),
            self.x1.max(other.x1),
            self.y1.max(other.y1),
        )
    }

    /// Overlap of two rects; `Rect::EMPTY` when they are disjoint.
    pub fn intersect(self, other: Rect) -> Rect {
        Rect::new(
            self.x0.max(other.x0),
            self.y0.max(other.y0),
            self.x1.min(other.x1),
            self.y1.min(other.y1),
        )
    }

    /// Clamps to the inclusive bounds `0..=max_x` / `0..=max_y`.
    pub fn clamp_to(self, max_x: i32, max_y: i32) -> Rect {
        Rect::new(
            self.x0.max(0),
            self.y0.max(0),
            self.x1.min(max_x + 1),
            self.y1.min(max_y + 1),
        )
    }

    #[inline]
    pub const fn is_empty_rc(self) -> bool {
        self.is_empty()
    }
}

impl std::ops::Add for Rect {
    type Output = Rect;
    #[inline]
    fn add(self, o: Rect) -> Rect {
        self.union(o)
    }
}

impl std::ops::BitAnd for Rect {
    type Output = Rect;
    #[inline]
    fn bitand(self, o: Rect) -> Rect {
        self.intersect(o)
    }
}

impl BitOr for Rect {
    type Output = Rect;
    #[inline]
    fn bitor(self, o: Rect) -> Rect {
        self.union(o)
    }
}

/// A rect that grows as a tool is dragged, then is consumed by the layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirtyAccumulator {
    rect: Rect,
}

impl DirtyAccumulator {
    #[inline]
    pub fn new() -> Self {
        Self { rect: Rect::EMPTY }
    }

    #[inline]
    pub fn add_rect(&mut self, r: Rect) {
        self.rect = self.rect.union(r);
    }

    #[inline]
    pub fn add_point(&mut self, x: i32, y: i32) {
        if self.rect.is_empty() {
            self.rect = Rect::from_inclusive(x, y, x, y);
        } else {
            self.rect.include_point(x, y);
        }
    }

    #[inline]
    pub fn get(&self) -> Rect {
        self.rect
    }

    #[inline]
    pub fn take(&mut self) -> Rect {
        std::mem::replace(&mut self.rect, Rect::EMPTY)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rect.is_empty()
    }
}
