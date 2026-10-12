//! The document: canvas geometry, layer tree, and the history that undoes it.

use std::collections::HashMap;

use crate::history::{History, HistoryEntry, Patch, StructureNode, StructureState};
use crate::id::{DocumentId, LayerId};
use crate::layer::{Layer, LayerProps};
use ultrapixel::plane::{TileData, TileIndex};
use ultrapixel::{Depth, PixelFormat, Rect, TILE};

/// Canvas-level settings that history can revert (resize, resolution).
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentMeta {
    pub width: usize,
    pub height: usize,
    pub dpi: u32,
    pub icc_profile: Option<String>,
}

/// A single image document.
pub struct Document {
    pub id: DocumentId,
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub depth: Depth,
    pub format: PixelFormat,
    pub dpi: u32,
    pub icc_profile: Option<String>,

    /// Every layer in the document, addressed by id. A flat map plus explicit
    /// `parent`/`children` links beats a `Vec<Layer>` here: the tree can be
    /// reordered without moving pixel data around.
    layers: HashMap<LayerId, Layer>,
    /// Top-level layers, bottom-to-top (index 0 paints first).
    root: Vec<LayerId>,

    /// Layers removed by a command that history can still bring back.
    /// Keeping them here is what makes "delete layer" undoable without a
    /// deep clone of the whole tree.
    graveyard: HashMap<LayerId, Layer>,

    pub history: History,
    /// False once history has entries newer than the last save.
    pub dirty: bool,
}

impl Document {
    pub fn new(
        name: impl Into<String>,
        width: usize,
        height: usize,
        depth: Depth,
        format: PixelFormat,
    ) -> Self {
        let mut doc = Self {
            id: DocumentId::next(),
            name: name.into(),
            width,
            height,
            depth,
            format,
            dpi: 72,
            icc_profile: None,
            layers: HashMap::new(),
            root: Vec::new(),
            graveyard: HashMap::new(),
            history: History::new(),
            dirty: false,
        };
        // Photoshop always starts with exactly one Background layer.
        let bg = Layer::pixel("Background", width, height, depth, format.without_alpha());
        let id = bg.id;
        doc.layers.insert(id, bg);
        doc.root.push(id);
        doc
    }

    // ---------------------------------------------------------------- access

    #[inline]
    pub fn root(&self) -> &[LayerId] {
        &self.root
    }

    #[inline]
    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.get(&id)
    }

    #[inline]
    pub fn layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.get_mut(&id)
    }

    #[inline]
    pub fn layer_ids(&self) -> Vec<LayerId> {
        self.layers.keys().copied().collect()
    }

    pub fn meta(&self) -> DocumentMeta {
        DocumentMeta {
            width: self.width,
            height: self.height,
            dpi: self.dpi,
            icc_profile: self.icc_profile.clone(),
        }
    }

    #[inline]
    pub fn canvas_rect(&self) -> Rect {
        Rect::new(0, 0, self.width as i32, self.height as i32)
    }

    // ------------------------------------------------------------------ tree

    /// Children of `parent` in paint order (bottom to top).
    pub fn children_of(&self, parent: Option<LayerId>) -> &[LayerId] {
        match parent {
            None => &self.root,
            Some(p) => self
                .layers
                .get(&p)
                .map(|l| l.children.as_slice())
                .unwrap_or(&[]),
        }
    }

    fn detach(&mut self, id: LayerId) {
        let parent = self.layers.get(&id).and_then(|l| l.parent);
        let list = match parent {
            None => &mut self.root,
            Some(p) => match self.layers.get_mut(&p) {
                Some(l) => &mut l.children,
                None => return,
            },
        };
        list.retain(|&x| x != id);
    }

    /// Inserts `layer` at `index` inside `parent` (append when `None`).
    pub fn add_layer(
        &mut self,
        layer: Layer,
        parent: Option<LayerId>,
        index: Option<usize>,
    ) -> LayerId {
        let id = layer.id;
        let mut layer = layer;
        layer.parent = parent;
        self.layers.insert(id, layer);
        let idx = index.unwrap_or(usize::MAX);
        match parent {
            None => {
                let i = idx.min(self.root.len());
                self.root.insert(i, id);
            }
            Some(p) => {
                if let Some(pl) = self.layers.get_mut(&p) {
                    let i = idx.min(pl.children.len());
                    pl.children.insert(i, id);
                }
            }
        }
        self.dirty = true;
        id
    }

    /// Moves a layer to a new parent/index. Used by drag-and-drop in the
    /// Layers panel.
    pub fn move_layer(
        &mut self,
        id: LayerId,
        new_parent: Option<LayerId>,
        new_index: usize,
    ) -> bool {
        if new_parent == Some(id) {
            return false;
        }
        // Refuse to move a group into its own subtree.
        if let Some(p) = new_parent {
            if self.is_descendant(p, id) {
                return false;
            }
        }
        self.detach(id);
        if let Some(l) = self.layers.get_mut(&id) {
            l.parent = new_parent;
        }
        match new_parent {
            None => {
                let i = new_index.min(self.root.len());
                self.root.insert(i, id);
            }
            Some(p) => {
                if let Some(pl) = self.layers.get_mut(&p) {
                    let i = new_index.min(pl.children.len());
                    pl.children.insert(i, id);
                }
            }
        }
        self.dirty = true;
        true
    }

    /// True when `candidate` is inside the subtree rooted at `ancestor`.
    pub fn is_descendant(&self, candidate: LayerId, ancestor: LayerId) -> bool {
        let mut stack = vec![ancestor];
        while let Some(id) = stack.pop() {
            if id == candidate {
                return true;
            }
            if let Some(l) = self.layers.get(&id) {
                stack.extend(l.children.iter().copied());
            }
        }
        false
    }

    /// Removes a layer and its subtree into the graveyard so undo can restore
    /// it without a deep clone. Returns a ready-to-commit history entry, or
    /// `None` if the layer did not exist.
    pub fn remove_layer(&mut self, id: LayerId) -> Option<HistoryEntry> {
        if !self.layers.contains_key(&id) {
            return None;
        }
        let before = self.structure_state();
        self.detach(id);

        // Detach the whole subtree, stashing each layer.
        let mut stack = vec![id];
        let mut removed = Vec::new();
        while let Some(cur) = stack.pop() {
            if let Some(l) = self.layers.remove(&cur) {
                stack.extend(l.children.iter().copied());
                removed.push(l);
            }
        }
        for l in &removed {
            self.graveyard.insert(l.id, clone_layer_shallow(l));
        }

        let after = self.structure_state();
        let mut e = HistoryEntry::new("Delete Layer", "Layer Delete");
        e.push(Patch::Structure {
            before: Box::new(before),
            after: Box::new(after),
        });
        self.dirty = true;
        Some(e)
    }

    /// Commits a structural history entry (add/remove/reorder).
    pub fn commit_structure(&mut self, before: StructureState, label: impl Into<String>) {
        let after = self.structure_state();
        if before == after {
            return;
        }
        let mut e = HistoryEntry::new(label, "Layer Structure");
        e.push(Patch::Structure {
            before: Box::new(before),
            after: Box::new(after),
        });
        self.dirty = true;
        self.history.push(e);
    }

    /// Depth-first list of ids in paint order, starting at the root.
    pub fn paint_order(&self) -> Vec<LayerId> {
        let mut out = Vec::with_capacity(self.layers.len());
        fn walk(doc: &Document, ids: &[LayerId], out: &mut Vec<LayerId>) {
            for &id in ids {
                out.push(id);
                if let Some(l) = doc.layers.get(&id) {
                    if !l.children.is_empty() {
                        walk(doc, &l.children, out);
                    }
                }
            }
        }
        walk(self, &self.root, &mut out);
        out
    }

    /// The layer directly underneath `id` in paint order, honouring clipping.
    /// `Alt+click` uses this to pick the clipping base.
    pub fn layer_below(&self, id: LayerId) -> Option<LayerId> {
        let order = self.paint_order();
        let pos = order.iter().position(|&x| x == id)?;
        for &cand in order[..pos].iter().rev() {
            let Some(l) = self.layers.get(&cand) else {
                continue;
            };
            if l.props.visible && !l.clipped {
                return Some(cand);
            }
        }
        None
    }

    /// Flat snapshot of the tree topology.
    pub fn structure_state(&self) -> StructureState {
        let mut nodes = Vec::with_capacity(self.layers.len());
        fn walk(
            doc: &Document,
            ids: &[LayerId],
            parent: Option<LayerId>,
            nodes: &mut Vec<StructureNode>,
        ) {
            for (index, &id) in ids.iter().enumerate() {
                let Some(l) = doc.layers.get(&id) else {
                    continue;
                };
                nodes.push(StructureNode {
                    id,
                    parent,
                    index,
                    is_group: l.is_group(),
                });
                if !l.children.is_empty() {
                    walk(doc, &l.children, Some(id), nodes);
                }
            }
        }
        walk(self, &self.root, None, &mut nodes);
        StructureState { nodes }
    }

    // --------------------------------------------------------------- history

    /// Opens a transaction. Dropping the `Txn` without calling `commit`
    /// discards it, so a tool that bails out mid-stroke leaves no history
    /// entry behind.
    pub fn begin(&mut self, label: impl Into<String>, command: impl Into<String>) -> Txn<'_> {
        Txn {
            doc: self,
            entry: HistoryEntry::new(label, command),
        }
    }

    /// Reverses the most recent history entry. Returns the region that needs
    /// repainting so the compositor can schedule exactly one GPU pass.
    pub fn undo(&mut self) -> Option<Rect> {
        let entry = self.doc_undo_pop()?;
        let bounds = self.apply_entry(&entry, Direction::Undo);
        self.dirty = true;
        Some(bounds)
    }

    /// Re-applies the most recently undone entry.
    pub fn redo(&mut self) -> Option<Rect> {
        let entry = self.history.pop_redo()?;
        let bounds = self.apply_entry(&entry, Direction::Redo);
        self.dirty = true;
        Some(bounds)
    }

    fn doc_undo_pop(&mut self) -> Option<HistoryEntry> {
        self.history.pop_undo()
    }

    fn apply_entry(&mut self, entry: &HistoryEntry, dir: Direction) -> Rect {
        let mut bounds = Rect::EMPTY;
        for patch in entry.patches() {
            match patch {
                Patch::Tile {
                    layer,
                    col,
                    row,
                    before,
                    after,
                } => {
                    let data = match dir {
                        Direction::Undo => before,
                        Direction::Redo => after,
                    };
                    if let Some(l) = self.layers.get_mut(layer) {
                        let idx = TileIndex {
                            col: *col,
                            row: *row,
                        };
                        apply_tile(&mut l.surface, idx, data);
                        bounds = bounds.union(tile_bounds(*col, *row));
                    }
                }
                Patch::Props {
                    layer,
                    before,
                    after,
                } => {
                    let props = match dir {
                        Direction::Undo => before,
                        Direction::Redo => after,
                    };
                    if let Some(l) = self.layers.get_mut(layer) {
                        l.props = props.clone();
                    }
                }
                Patch::MaskEnabled {
                    layer,
                    before,
                    after,
                } => {
                    let v = match dir {
                        Direction::Undo => *before,
                        Direction::Redo => *after,
                    };
                    if let Some(l) = self.layers.get_mut(layer) {
                        if let Some(m) = l.mask.as_mut() {
                            m.enabled = v;
                        }
                    }
                }
                Patch::Structure { before, after } => {
                    let st = match dir {
                        Direction::Undo => before,
                        Direction::Redo => after,
                    };
                    self.restore_structure(st);
                }
            }
        }
        bounds
    }

    /// Rebuilds the tree topology from a flat node list. Layer *data* is
    /// preserved — only the links move, and layers that the target state
    /// wants back are resurrected from the graveyard.
    fn restore_structure(&mut self, st: &StructureState) {
        // 1. Resurrect anything the target state needs that is currently gone.
        let wanted: Vec<LayerId> = st.nodes.iter().map(|n| n.id).collect();
        for id in &wanted {
            if !self.layers.contains_key(id) {
                if let Some(l) = self.graveyard.remove(id) {
                    self.layers.insert(*id, l);
                }
            }
        }

        // 2. Stash anything the target state no longer wants.
        let dead: Vec<LayerId> = self
            .layers
            .keys()
            .copied()
            .filter(|id| !wanted.contains(id))
            .collect();
        for id in dead {
            if let Some(l) = self.layers.remove(&id) {
                self.graveyard.insert(id, l);
            }
        }

        // 3. Rebuild the links.
        self.root.clear();
        for l in self.layers.values_mut() {
            l.children.clear();
            l.parent = None;
        }
        for n in &st.nodes {
            if let Some(l) = self.layers.get_mut(&n.id) {
                l.parent = n.parent;
            }
        }
        for n in &st.nodes {
            match n.parent {
                None => {
                    let i = n.index.min(self.root.len());
                    self.root.insert(i, n.id);
                }
                Some(p) => {
                    if let Some(pl) = self.layers.get_mut(&p) {
                        let i = n.index.min(pl.children.len());
                        pl.children.insert(i, n.id);
                    }
                }
            }
        }
    }

    /// Total resident bytes across every layer.
    pub fn resident_bytes(&self) -> usize {
        self.layers.values().map(|l| l.resident_bytes()).sum()
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Undo,
    Redo,
}

#[inline]
fn tile_bounds(col: usize, row: usize) -> Rect {
    let t = TILE as i32;
    Rect::new(
        (col as i32) * t,
        (row as i32) * t,
        (col as i32 + 1) * t,
        (row as i32 + 1) * t,
    )
}

/// Installs (or frees) a tile across every plane of a surface.
fn apply_tile(surface: &mut ultrapixel::Surface, idx: TileIndex, data: &[Option<Box<TileData>>]) {
    for (ch, plane) in surface.planes_mut().iter_mut().enumerate() {
        plane.set_tile(idx, data.get(ch).and_then(|d| d.as_deref()));
    }
}

/// Full clone used when stashing a removed layer into the graveyard, so undo
/// can put it back byte-for-byte.
fn clone_layer_shallow(l: &Layer) -> Layer {
    l.clone()
}

/// A reversible group of pixel writes.
///
/// Patches to the same tile are coalesced: only the first `before` and the
/// latest `after` survive, so a 500-dab stroke costs one patch per tile.
pub struct Txn<'a> {
    doc: &'a mut Document,
    entry: HistoryEntry,
}

impl Txn<'_> {
    /// Writes one normalised sample and records the tile it touched.
    pub fn set_sample(
        &mut self,
        layer: LayerId,
        channel: usize,
        x: usize,
        y: usize,
        value: f32,
    ) -> bool {
        let (width, height) = (self.doc.width, self.doc.height);
        if x >= width || y >= height {
            return false;
        }
        let Some(layer_ref) = self.doc.layers.get_mut(&layer) else {
            return false;
        };
        if layer_ref.locks.blocks_paint() || channel >= layer_ref.surface.channels() {
            return false;
        }

        let col = x / TILE;
        let row = y / TILE;
        let idx = TileIndex { col, row };
        let planes = layer_ref.surface.planes_mut();

        // Snapshot every plane of this tile *before* the write. A patch covers
        // the whole tile, otherwise writing colour and alpha into the same tile
        // would silently drop one of them.
        let before: Vec<Option<Box<TileData>>> = planes
            .iter()
            .map(|p| p.tile_data(idx).map(|d| Box::new(d.clone())))
            .collect();

        planes[channel].set_sample_normalized(x, y, value);

        let after: Vec<Option<Box<TileData>>> = planes
            .iter()
            .map(|p| p.tile_data(idx).map(|d| Box::new(d.clone())))
            .collect();

        layer_ref
            .surface
            .note_bounds(Rect::from_inclusive(x as i32, y as i32, x as i32, y as i32));
        self.record_tile(layer, col, row, before, after);
        true
    }

    /// Records a layer-property change.
    pub fn set_props(&mut self, layer: LayerId, props: LayerProps) {
        let Some(before) = self.doc.layers.get(&layer).map(|l| l.props.clone()) else {
            return;
        };
        if before == props {
            return;
        }
        self.entry.push(Patch::Props {
            layer,
            before,
            after: props.clone(),
        });
        if let Some(l) = self.doc.layers.get_mut(&layer) {
            l.props = props;
        }
    }

    fn record_tile(
        &mut self,
        layer: LayerId,
        col: usize,
        row: usize,
        before: Vec<Option<Box<TileData>>>,
        after: Vec<Option<Box<TileData>>>,
    ) {
        // Coalesce with the previous patch when it touched the same tile.
        if let Some(Patch::Tile {
            layer: l,
            col: c,
            row: r,
            after: prev_after,
            ..
        }) = self.entry.patch_last_mut()
        {
            if *l == layer && *c == col && *r == row {
                *prev_after = after;
                return;
            }
        }
        self.entry.push(Patch::Tile {
            layer,
            col,
            row,
            before,
            after,
        });
    }

    /// Commits the transaction to the document's history.
    pub fn commit(self) {
        if self.entry.patch_count() == 0 {
            return;
        }
        self.doc.dirty = true;
        self.doc.history.push(self.entry);
    }
}
