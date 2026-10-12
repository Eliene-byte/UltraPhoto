//! Panel docking geometry.
//!
//! Photoshop's workspace is a grid of dockable groups separated by splitters.
//! This module owns the arithmetic: given a window size and a dock layout, where
//! does every panel go, and which splitter is the user grabbing?
//!
//! It is pure arithmetic on purpose — no GPU, no window handle. "Does the
//! layout still work at 800×600" is exactly the kind of question CI should be
//! able to answer, and that needs this file to be testable on its own.

use crate::theme::Metrics;

/// An axis-aligned rectangle in window pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    #[inline]
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    #[inline]
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (
            (self.x + self.right()) * 0.5,
            (self.y + self.bottom()) * 0.5,
        )
    }

    #[inline]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    #[inline]
    pub fn contains_rect(&self, r: Rect) -> bool {
        r.x >= self.x && r.y >= self.y && r.right() <= self.right() && r.bottom() <= self.bottom()
    }

    /// Splits horizontally, `ratio` of the width going to `self`.
    pub fn split_x(&self, ratio: f32) -> (Rect, Rect) {
        let ratio = ratio.clamp(0.0, 1.0);
        let left = self.w * ratio;
        (
            Rect::new(self.x, self.y, left, self.h),
            Rect::new(self.x + left, self.y, self.w - left, self.h),
        )
    }

    /// Splits vertically, `ratio` of the height going to `self`.
    pub fn split_y(&self, ratio: f32) -> (Rect, Rect) {
        let ratio = ratio.clamp(0.0, 1.0);
        let top = self.h * ratio;
        (
            Rect::new(self.x, self.y, self.w, top),
            Rect::new(self.x, self.y + top, self.w, self.h - top),
        )
    }

    /// Shrinks by `pad` on every side, never producing a negative size.
    pub fn inset(&self, pad: f32) -> Rect {
        Rect::new(
            self.x + pad,
            self.y + pad,
            (self.w - pad * 2.0).max(0.0),
            (self.h - pad * 2.0).max(0.0),
        )
    }
}

/// The regions of the window that are not the canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkArea {
    /// Full window.
    pub window: Rect,
    /// Menu bar strip.
    pub menu_bar: Rect,
    /// Tool options strip under the menu bar.
    pub options_bar: Rect,
    /// Document tab strip.
    pub tab_bar: Rect,
    /// Canvas viewport.
    pub canvas: Rect,
    /// Left dock.
    pub left_dock: Rect,
    /// Right dock.
    pub right_dock: Rect,
    /// Status bar.
    pub status_bar: Rect,
}

/// Computes the work area from a window size and the dock split positions.
///
/// `left_ratio` and `right_ratio` are fractions of the window width. The canvas
/// takes whatever is left; if the docks are dragged together the canvas
/// collapses to zero rather than going negative, which is what stops the
/// renderer from producing inverted geometry.
pub fn layout(width: f32, height: f32, left_ratio: f32, right_ratio: f32, m: &Metrics) -> WorkArea {
    let window = Rect::new(0.0, 0.0, width.max(0.0), height.max(0.0));

    let menu_bar = Rect::new(0.0, 0.0, window.w, m.menu_bar.min(window.h));
    let options_bar = Rect::new(
        0.0,
        menu_bar.bottom(),
        window.w,
        m.options_bar.min(window.h),
    );
    let tab_bar = Rect::new(0.0, options_bar.bottom(), window.w, m.tab_bar.min(window.h));
    let status_bar = Rect::new(
        0.0,
        (window.h - m.status_bar).max(tab_bar.bottom()),
        window.w,
        m.status_bar.min(window.h),
    );

    let body_top = tab_bar.bottom();
    let body_h = (status_bar.y - body_top).max(0.0);

    // Docks get a fraction of the width, but the two together can never take
    // more than the body.
    let min_canvas = 0.0;
    let avail = window.w;
    let left_w = (avail * left_ratio.clamp(0.0, 1.0)).min(avail - min_canvas);
    let right_w = (avail * right_ratio.clamp(0.0, 1.0)).min((avail - left_w).max(0.0));

    let left_dock = Rect::new(0.0, body_top, left_w, body_h);
    let right_dock = Rect::new(avail - right_w, body_top, right_w, body_h);
    let canvas = Rect::new(
        left_dock.right(),
        body_top,
        (avail - left_w - right_w).max(0.0),
        body_h,
    );

    WorkArea {
        window,
        menu_bar,
        options_bar,
        tab_bar,
        canvas,
        left_dock,
        right_dock,
        status_bar,
    }
}

/// What a click landed on, which drives splitter dragging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DockedHit {
    /// Inside `left_dock`.
    LeftDock,
    /// Inside `right_dock`.
    RightDock,
    /// Inside the canvas viewport.
    Canvas,
    /// On the boundary between the left dock and the canvas.
    LeftSplitter,
    /// On the boundary between the canvas and the right dock.
    RightSplitter,
    /// In the menu bar.
    MenuBar,
    /// In the status bar.
    StatusBar,
    /// Nowhere we care about.
    Nothing,
}

/// Hit-tests the work area, expanding splitter detection by
/// `Metrics::splitter` so the grab target is wider than the 1 px line.
pub fn hit_test(area: &WorkArea, x: f32, y: f32, m: &Metrics) -> DockedHit {
    if area.menu_bar.contains(x, y) {
        return DockedHit::MenuBar;
    }
    if area.status_bar.contains(x, y) {
        return DockedHit::StatusBar;
    }

    let half = m.splitter * 0.5;
    let in_body = y >= area.canvas.y && y < area.canvas.bottom();

    // Splitter detection must NOT require the point to be inside the dock: the
    // dock rect is half-open, so the divider line itself falls just outside it.
    // A zero-width dock has no divider to grab.
    if in_body && area.left_dock.w > 0.0 && (x - area.left_dock.right()).abs() <= half {
        return DockedHit::LeftSplitter;
    }
    if in_body && area.right_dock.w > 0.0 && (x - area.right_dock.x).abs() <= half {
        return DockedHit::RightSplitter;
    }
    if area.left_dock.contains(x, y) {
        return DockedHit::LeftDock;
    }
    if area.right_dock.contains(x, y) {
        return DockedHit::RightDock;
    }
    if area.canvas.contains(x, y) {
        return DockedHit::Canvas;
    }
    DockedHit::Nothing
}

/// Which edge of a panel a drag is targeting, used for re-docking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropZone {
    None,
    Left,
    Right,
    Top,
    Bottom,
    Center,
}

/// The four drop zones plus centre, computed from the canvas rect.
///
/// Photoshop tints the whole canvas while a panel is dragged, with a band at
/// each edge. The centre zone is the canvas **minus** those bands — if it were
/// the whole canvas it would swallow every edge hit and re-docking would be
/// impossible.
pub fn drop_zones(area: &WorkArea, m: &Metrics) -> (Rect, Rect, Rect, Rect, Rect) {
    let c = area.canvas;
    let band = m.splitter * 4.0;
    let horizontal = Rect::new(c.x, c.center().1 - band, c.w, band * 2.0);
    let vertical = Rect::new(c.center().0 - band, c.y, band * 2.0, c.h);
    let center = Rect::new(
        c.x + band,
        c.y + band,
        (c.w - band * 2.0).max(0.0),
        (c.h - band * 2.0).max(0.0),
    );
    (
        horizontal, // top
        horizontal, // bottom
        vertical,   // left
        vertical,   // right
        center,
    )
}

/// Maps a point to a drop zone.
///
/// The edge bands overlap each other at the corners, so a fixed precedence
/// would make the corner behaviour arbitrary. Nearest-edge wins instead: the
/// corner resolves to whichever side the pointer is actually closer to, which
/// is what a drag needs to feel predictable.
pub fn drop_zone_at(area: &WorkArea, x: f32, y: f32, m: &Metrics) -> DropZone {
    if !area.window.contains(x, y) {
        return DropZone::None;
    }
    let c = area.canvas;
    let band = m.splitter * 4.0;
    let center = Rect::new(
        c.x + band,
        c.y + band,
        (c.w - band * 2.0).max(0.0),
        (c.h - band * 2.0).max(0.0),
    );
    if center.contains(x, y) {
        return DropZone::Center;
    }

    let d_left = (x - c.x).abs();
    let d_right = (c.right() - x).abs();
    let d_top = (y - c.y).abs();
    let d_bottom = (c.bottom() - y).abs();
    let nearest = d_left.min(d_right).min(d_top).min(d_bottom);
    if nearest > band {
        return DropZone::None;
    }
    if nearest == d_left {
        DropZone::Left
    } else if nearest == d_right {
        DropZone::Right
    } else if nearest == d_top {
        DropZone::Top
    } else {
        DropZone::Bottom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::PS_DARK;

    fn m() -> Metrics {
        PS_DARK.metrics
    }

    #[test]
    fn the_regions_never_overlap() {
        let a = layout(1920.0, 1080.0, 0.18, 0.22, &m());
        // Stacked strips must not overlap vertically.
        assert!(a.options_bar.y >= a.menu_bar.bottom());
        assert!(a.tab_bar.y >= a.options_bar.bottom());
        assert!(a.canvas.y >= a.tab_bar.bottom());
        assert!(a.status_bar.y >= a.canvas.bottom());
    }

    #[test]
    fn docks_and_canvas_tile_the_body_exactly() {
        let a = layout(1920.0, 1080.0, 0.18, 0.22, &m());
        assert_eq!(a.left_dock.right(), a.canvas.x);
        assert_eq!(a.canvas.right(), a.right_dock.x);
        let body = a.canvas.h;
        assert!((a.left_dock.h - body).abs() < 0.001);
        assert!((a.right_dock.h - body).abs() < 0.001);
    }

    #[test]
    fn docks_cannotsqueeze_the_canvas_negative() {
        // The pathological case: both docks dragged fully open.
        let a = layout(1000.0, 800.0, 1.0, 1.0, &m());
        assert!(a.canvas.w >= 0.0, "canvas went negative");
        assert!(a.right_dock.x >= 0.0);
        assert!(a.right_dock.right() <= 1000.0 + 0.001);
    }

    #[test]
    fn a_tiny_window_does_not_produce_negative_or_nan_geometry() {
        for (w, h) in [(1.0, 1.0), (0.0, 0.0), (10.0, 5.0), (200.0, 30.0)] {
            let a = layout(w, h, 0.2, 0.2, &m());
            for r in [
                a.menu_bar,
                a.options_bar,
                a.tab_bar,
                a.canvas,
                a.left_dock,
                a.right_dock,
                a.status_bar,
            ] {
                assert!(r.w.is_finite() && r.h.is_finite(), "{w}x{h} produced NaN");
                assert!(r.w >= 0.0 && r.h >= 0.0, "{w}x{h} produced a negative size");
            }
        }
    }

    #[test]
    fn the_splitter_grab_zone_is_wider_than_the_divider() {
        let a = layout(1920.0, 1080.0, 0.2, 0.2, &m());
        let mm = m();
        let edge = a.left_dock.right();
        let y = a.canvas.center().1;
        // Exactly on the line is a splitter hit.
        assert_eq!(hit_test(&a, edge, y, &mm), DockedHit::LeftSplitter);
        // A couple of px to the left, still within the grab band.
        assert_eq!(hit_test(&a, edge - 2.0, y, &mm), DockedHit::LeftSplitter);
        // Well inside the dock is not.
        assert_eq!(hit_test(&a, edge - 60.0, y, &mm), DockedHit::LeftDock);
    }

    #[test]
    fn hit_testing_covers_every_region() {
        let a = layout(1920.0, 1080.0, 0.2, 0.2, &m());
        let mm = m();
        assert_eq!(hit_test(&a, 500.0, 5.0, &mm), DockedHit::MenuBar);
        assert_eq!(
            hit_test(&a, 500.0, a.canvas.center().1, &mm),
            DockedHit::Canvas
        );
        assert_eq!(
            hit_test(&a, 100.0, a.canvas.center().1, &mm),
            DockedHit::LeftDock
        );
        assert_eq!(
            hit_test(&a, 1900.0, a.canvas.center().1, &mm),
            DockedHit::RightDock
        );
        assert_eq!(
            hit_test(&a, 500.0, a.status_bar.center().1, &mm),
            DockedHit::StatusBar
        );
    }

    #[test]
    fn split_helpers_partition_without_gaps() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        let (a, b) = r.split_x(0.3);
        assert!((a.right() - b.x).abs() < 1e-5);
        assert!((a.w + b.w - 100.0).abs() < 1e-5);
        let (c, d) = r.split_y(0.6);
        assert!((c.bottom() - d.y).abs() < 1e-5);
        assert!((c.h + d.h - 50.0).abs() < 1e-5);
    }

    #[test]
    fn split_ratios_are_clamped() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        let (a, _) = r.split_x(5.0);
        assert!(
            (a.w - 100.0).abs() < 1e-5,
            "ratio > 1 must clamp to the whole rect"
        );
        let (b, _) = r.split_x(-2.0);
        assert!(b.w.abs() < 1e-5, "ratio < 0 must clamp to zero");
    }

    #[test]
    fn drop_zones_resolve_to_the_expected_edge() {
        let a = layout(1920.0, 1080.0, 0.2, 0.2, &m());
        let mm = m();
        let (cx, cy) = a.canvas.center();
        assert_eq!(drop_zone_at(&a, cx, cy, &mm), DropZone::Center);
        // Near each edge but not at the centre point.
        assert_eq!(drop_zone_at(&a, cx, a.canvas.y + 1.0, &mm), DropZone::Top);
        assert_eq!(
            drop_zone_at(&a, cx, a.canvas.bottom() - 1.0, &mm),
            DropZone::Bottom
        );
        assert_eq!(drop_zone_at(&a, a.canvas.x + 1.0, cy, &mm), DropZone::Left);
        assert_eq!(
            drop_zone_at(&a, a.canvas.right() - 1.0, cy, &mm),
            DropZone::Right
        );
        // Corners resolve to the nearest edge rather than an arbitrary one.
        assert_eq!(
            drop_zone_at(&a, a.canvas.x + 1.0, a.canvas.y + 1.0, &mm),
            DropZone::Left
        );
        assert_eq!(drop_zone_at(&a, -10.0, -10.0, &mm), DropZone::None);
    }

    #[test]
    fn inset_never_inverts() {
        let r = Rect::new(0.0, 0.0, 4.0, 4.0);
        let i = r.inset(10.0);
        assert_eq!(i.w, 0.0);
        assert_eq!(i.h, 0.0);
    }
}
