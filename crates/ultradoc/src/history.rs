//! History with tile-diff undo.
//!
//! The naive approach — snapshot the whole document before every command —
//! costs gigabytes after a hundred brush strokes. Instead every entry records
//! only the tiles it actually touched, in both directions:
//!
//! ```text
//! before ──┐         ┌── after
//!   tile   ├─ Patch ─┤
//!          └─(alloc?)┘
//! ```
//!
//! Undo writes `before` back; redo writes `after`. A brush stroke over a
//! 200 px dab touches ~1-2 tiles, so an undo entry costs ~512 KiB at 32-bit
//! instead of the whole canvas.

use crate::id::{HistoryId, LayerId};
use crate::layer::LayerProps;
use std::sync::Arc;
use ultrapixel::plane::TileData;
use ultrapixel::Rect;

/// One reversible change.
#[derive(Debug, Clone)]
pub enum Patch {
    /// Pixel data for one tile of one layer — **all** of the layer's planes
    /// for that tile, not a single channel.
    ///
    /// Tools write colour and alpha together inside the same tile, so the patch
    /// has to carry the whole tile. Storing one channel per patch would silently
    /// drop the others the moment two writes hit the same tile.
    ///
    /// `before: None` for a plane means that plane had no tile allocated
    /// (fully transparent), so undoing frees it again — which is how memory
    /// stays bounded while you scribble across a huge canvas.
    Tile {
        layer: LayerId,
        col: usize,
        row: usize,
        before: Vec<Option<Box<TileData>>>,
        after: Vec<Option<Box<TileData>>>,
    },
    /// Opacity / fill / blend mode / visibility.
    Props {
        layer: LayerId,
        before: LayerProps,
        after: LayerProps,
    },
    /// Layer mask enable/invert state.
    MaskEnabled {
        layer: LayerId,
        before: bool,
        after: bool,
    },
    /// Flat ordering of the layer tree — used by add/delete/reorder/group.
    Structure {
        before: Box<StructureState>,
        after: Box<StructureState>,
    },
}

/// Flat, fully ordered description of the layer tree.
///
/// Cheap to compare and small enough to clone for every structural command,
/// and unlike a deep clone it never copies pixel data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructureState {
    /// `(id, parent, index, kind)` in bottom-to-top paint order, depth first.
    pub nodes: Vec<StructureNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructureNode {
    pub id: LayerId,
    pub parent: Option<LayerId>,
    /// Position within the parent's `children`.
    pub index: usize,
    pub is_group: bool,
}

/// One undoable step.
///
/// The patch list lives behind an [`Arc`] so `History::pop_undo` can hand a
/// cheap clone to the document while the entry stays parked in the stack ready
/// to be redone. Without that, undo would either deep-copy megabytes of tile
/// data or destroy the redo tail.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub id: HistoryId,
    /// Menu label shown in the History panel, e.g. "Brush Tool".
    pub label: String,
    /// Name of the originating command, for scripts and the action recorder.
    pub command: String,
    /// Region affected, used to drive the GPU redraw after undo.
    pub bounds: Rect,
    patches: Arc<Vec<Patch>>,
    /// Bytes this entry costs. The history panel shows it in the tooltip.
    pub memory_bytes: usize,
}

impl HistoryEntry {
    pub fn new(label: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            id: HistoryId::next(),
            label: label.into(),
            command: command.into(),
            bounds: Rect::EMPTY,
            patches: Arc::new(Vec::new()),
            memory_bytes: 0,
        }
    }

    #[inline]
    pub fn patches(&self) -> &[Patch] {
        &self.patches
    }

    #[inline]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Mutable access to the most recent patch, for in-place coalescing.
    pub fn patch_last_mut(&mut self) -> Option<&mut Patch> {
        Arc::make_mut(&mut self.patches).last_mut()
    }

    /// Pushes a patch and keeps the byte accounting up to date.
    pub fn push(&mut self, patch: Patch) {
        match &patch {
            Patch::Tile { before, after, .. } => {
                for d in before.iter().chain(after.iter()) {
                    self.memory_bytes += size_of(d.as_deref());
                }
            }
            Patch::Structure { before, after } => {
                self.memory_bytes +=
                    std::mem::size_of::<StructureNode>() * (before.nodes.len() + after.nodes.len());
            }
            _ => self.memory_bytes += 64,
        }
        // Unique during construction, so `make_mut` never deep-copies here.
        Arc::make_mut(&mut self.patches).push(patch);
    }

    /// Union of the regions every tile patch touches.
    pub fn recompute_bounds(&mut self) {
        let mut b = Rect::EMPTY;
        for p in self.patches.iter() {
            if let Patch::Tile { col, row, .. } = p {
                let r = Rect::new(
                    (col * ultrapixel::TILE) as i32,
                    (row * ultrapixel::TILE) as i32,
                    ((col + 1) * ultrapixel::TILE) as i32,
                    ((row + 1) * ultrapixel::TILE) as i32,
                );
                b = b.union(r);
            }
        }
        self.bounds = b;
    }
}

fn size_of(t: Option<&TileData>) -> usize {
    match t {
        Some(TileData::U8(v)) => v.len(),
        Some(TileData::U16(v)) => v.len() * 2,
        Some(TileData::F32(v)) => v.len() * 4,
        None => 0,
    }
}

/// Undo/redo stack with a memory budget.
///
/// `used_bytes` counts *every* entry, including the redoable tail, because
/// that memory is genuinely resident and the budget has to reflect reality.
#[derive(Debug)]
pub struct History {
    entries: Vec<HistoryEntry>,
    /// Number of entries currently applied. `entries[cursor..]` are redone.
    cursor: usize,
    /// Photoshop's default is a fixed number of states; we use bytes instead,
    /// because a brush stroke and a "flatten document" cost wildly different
    /// amounts and a count-based limit thrashes on both.
    budget_bytes: usize,
    used_bytes: usize,
    /// Steps to drop when over budget. 0 = unlimited.
    max_states: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            cursor: 0,
            budget_bytes: 8 * 1024 * 1024 * 1024,
            used_bytes: 0,
            max_states: 200,
        }
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }
    #[inline]
    pub fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    /// Entry that `undo` would reverse.
    pub fn peek_undo(&self) -> Option<&HistoryEntry> {
        if self.can_undo() {
            self.entries.get(self.cursor - 1)
        } else {
            None
        }
    }

    /// Entry that `redo` would re-apply.
    pub fn peek_redo(&self) -> Option<&HistoryEntry> {
        self.entries.get(self.cursor)
    }

    /// Pops the entry `undo` should reverse, advancing the cursor backwards.
    ///
    /// The entry stays in the stack — only the cursor moves — so the same call
    /// chain can walk backwards and forwards without losing anything.
    pub fn pop_undo(&mut self) -> Option<HistoryEntry> {
        if !self.can_undo() {
            return None;
        }
        self.cursor -= 1;
        Some(self.entries[self.cursor].clone())
    }

    /// Pops the entry `redo` should re-apply, advancing the cursor forwards.
    pub fn pop_redo(&mut self) -> Option<HistoryEntry> {
        if !self.can_redo() {
            return None;
        }
        let e = self.entries[self.cursor].clone();
        self.cursor += 1;
        Some(e)
    }

    /// Commits a new entry. Any redoable tail is discarded first, matching
    /// what Photoshop does when you paint after undoing.
    pub fn push(&mut self, mut entry: HistoryEntry) {
        if self.cursor < self.entries.len() {
            for e in &self.entries[self.cursor..] {
                self.used_bytes = self.used_bytes.saturating_sub(e.memory_bytes);
            }
            self.entries.truncate(self.cursor);
        }
        entry.recompute_bounds();
        self.used_bytes += entry.memory_bytes;
        self.entries.push(entry);
        self.cursor = self.entries.len();
        self.enforce_budget();
    }

    /// Marks the document as saved, trimming the history like Photoshop's
    /// "New Snapshot" boundary.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
        self.used_bytes = 0;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn set_budget_bytes(&mut self, b: usize) {
        self.budget_bytes = b;
        self.enforce_budget();
    }

    /// Read-only view for the History panel.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Drops the oldest entries until we fit the budget and state cap.
    fn enforce_budget(&mut self) {
        loop {
            let over_budget = self.budget_bytes > 0 && self.used_bytes > self.budget_bytes;
            let over_count = self.max_states > 0 && self.entries.len() > self.max_states;
            if self.cursor == 0 || (!over_budget && !over_count) {
                break;
            }
            let e = self.entries.remove(0);
            self.used_bytes = self.used_bytes.saturating_sub(e.memory_bytes);
            self.cursor -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultrapixel::Depth;

    fn tile(v: f32) -> Box<TileData> {
        Box::new(TileData::F32(vec![v; ultrapixel::TILE_SAMPLES]))
    }

    fn patch(layer: LayerId, col: usize, row: usize, before: f32, after: f32) -> Patch {
        Patch::Tile {
            layer,
            col,
            row,
            before: vec![Some(tile(before))],
            after: vec![Some(tile(after))],
        }
    }

    #[test]
    fn undo_redo_advance_the_cursor() {
        let mut h = History::new();
        assert!(!h.can_undo());
        h.push(HistoryEntry::new("Brush", "Brush"));
        assert!(h.can_undo());
        assert!(!h.can_redo());
        let e = h.pop_undo().unwrap();
        assert_eq!(e.label, "Brush");
        assert!(h.can_redo());
        assert!(h.pop_redo().is_some());
        assert!(h.can_undo());
    }

    #[test]
    fn pushing_after_undo_drops_the_redo_tail() {
        let mut h = History::new();
        let id = LayerId::next();
        h.push(HistoryEntry::new("A", "A"));
        h.pop_undo();
        let mut e = HistoryEntry::new("B", "B");
        e.push(patch(id, 0, 0, 0.0, 1.0));
        h.push(e);
        assert!(!h.can_redo(), "redo tail must be discarded");
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn bounds_cover_every_touched_tile() {
        let id = LayerId::next();
        let mut e = HistoryEntry::new("Brush", "Brush");
        e.push(patch(id, 0, 0, 0.0, 1.0));
        e.push(patch(id, 2, 1, 0.0, 1.0));
        e.recompute_bounds();
        let t = ultrapixel::TILE as i32;
        assert_eq!(e.bounds, Rect::new(0, 0, 3 * t, 2 * t));
    }

    #[test]
    fn budget_drops_oldest_states_first() {
        let mut h = History::new();
        // One entry costs 2 tiles worth of f32 at TILE=256 => 512 KiB.
        let entry_bytes = ultrapixel::TILE_SAMPLES * 4 * 2;
        // Room for exactly two entries.
        h.set_budget_bytes(entry_bytes * 2 + 1024);

        let id = LayerId::next();
        for i in 0..5 {
            let mut e = HistoryEntry::new(format!("step{i}"), "x");
            e.push(patch(id, 0, 0, 0.0, i as f32));
            h.push(e);
        }

        assert!(
            h.used_bytes() <= h.budget_bytes(),
            "used {} bytes",
            h.used_bytes()
        );
        assert_eq!(h.entries().len(), 2, "oldest states must be dropped first");
        // Newest states survive, oldest are gone.
        assert_eq!(h.entries()[0].label, "step3");
        assert_eq!(h.entries()[1].label, "step4");
        assert_eq!(h.entries().last().unwrap().label, "step4");
    }

    #[test]
    fn state_cap_drops_oldest_below_the_byte_budget() {
        let mut h = History::new();
        h.set_budget_bytes(usize::MAX);
        h.max_states = 3;
        let id = LayerId::next();
        for i in 0..10 {
            let mut e = HistoryEntry::new(format!("s{i}"), "x");
            e.push(Patch::Props {
                layer: id,
                before: LayerProps::default(),
                after: LayerProps {
                    opacity: i as f32 / 10.0,
                    ..LayerProps::default()
                },
            });
            h.push(e);
        }
        assert_eq!(h.entries().len(), 3);
        assert_eq!(h.entries()[0].label, "s7");
    }

    #[test]
    fn depth_sized_patches_account_correctly() {
        let mut e = HistoryEntry::new("x", "x");
        let id = LayerId::next();
        e.push(Patch::Tile {
            layer: id,
            col: 0,
            row: 0,
            before: vec![Some(Box::new(TileData::U8(vec![
                0;
                ultrapixel::TILE_SAMPLES
            ])))],
            after: vec![Some(Box::new(TileData::F32(vec![
                0.0;
                ultrapixel::TILE_SAMPLES
            ])))],
        });
        assert_eq!(e.memory_bytes, ultrapixel::TILE_SAMPLES * 5);
        let _ = Depth::U8;
    }
}
