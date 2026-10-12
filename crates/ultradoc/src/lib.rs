//! # ultradoc
//!
//! The document model: canvas geometry, the layer tree, and history.
//!
//! Design notes:
//!
//! * **Ids, not indices.** Tools hold a [`LayerId`] for the whole stroke, so
//!   reordering the stack mid-stroke can't retarget their writes.
//! * **Tile-diff history.** Undo stores only the tiles a command touched, in
//!   both directions. See [`history`] for why a full snapshot is not an option.
//! * **Lazy structure snapshots.** Reorder/delete records the flat topology
//!   (a few dozen bytes per layer), never the pixels.

pub mod document;
pub mod history;
pub mod id;
pub mod layer;

pub use document::{Document, DocumentMeta, Txn};
pub use history::{History, HistoryEntry, Patch, StructureNode, StructureState};
pub use id::{DocumentId, HistoryId, LayerId, LayerIndex};
pub use layer::{
    AdjustmentKind, FillKind, GradientStop, Layer, LayerKind, LayerMask, LayerProps, LockState,
    TextSpec,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ultrapixel::{Depth, PixelFormat, Rect};

    fn doc() -> Document {
        Document::new("test", 512, 512, Depth::U8, PixelFormat::RGBA)
    }

    #[test]
    fn new_document_has_one_background_layer() {
        let d = doc();
        assert_eq!(d.root().len(), 1);
        let bg = d.layer(d.root()[0]).unwrap();
        assert_eq!(bg.name, "Background");
        assert!(!bg.surface.format().channel_count() > 0);
    }

    #[test]
    fn painting_then_undo_restores_transparency_and_frees_memory() {
        let mut d = doc();
        let bg = d.root()[0];
        assert_eq!(d.resident_bytes(), 0, "nothing painted yet");

        let mut tx = d.begin("Brush Tool", "Brush");
        for y in 10..40 {
            for x in 10..40 {
                assert!(tx.set_sample(bg, 0, x, y, 1.0));
            }
        }
        tx.commit();

        assert!(d.layer(bg).unwrap().surface.plane(0).sample(20, 20) > 0.0);
        let painted_bytes = d.resident_bytes();
        assert!(painted_bytes > 0);

        assert!(d.undo().is_some());
        assert_eq!(
            d.layer(bg).unwrap().surface.plane(0).sample(20, 20),
            0.0,
            "undo must restore the original pixel"
        );
        assert_eq!(d.resident_bytes(), 0, "undo must release the tiles again");
    }

    #[test]
    fn redo_reapplies_the_stroke() {
        let mut d = doc();
        let bg = d.root()[0];
        let mut tx = d.begin("Brush", "Brush");
        tx.set_sample(bg, 0, 5, 5, 1.0);
        tx.commit();
        d.undo();
        assert!(d.redo().is_some());
        assert!(d.layer(bg).unwrap().surface.plane(0).sample(5, 5) > 0.0);
    }

    #[test]
    fn a_stroke_coalesces_to_one_patch_per_tile() {
        let mut d = doc();
        let bg = d.root()[0];
        let mut tx = d.begin("Brush", "Brush");
        // Many dabs inside a single tile must not create one patch each.
        for x in 0..100 {
            tx.set_sample(bg, 0, x, 0, 1.0);
        }
        for x in 0..100 {
            tx.set_sample(bg, 0, x, 0, 0.5);
        }
        tx.commit();
        assert_eq!(d.history.peek_undo().unwrap().patch_count(), 1);
    }

    #[test]
    fn undo_bounds_match_the_touched_tiles() {
        let mut d = doc();
        let bg = d.root()[0];
        let mut tx = d.begin("Brush", "Brush");
        tx.set_sample(bg, 0, 10, 10, 1.0);
        tx.commit();
        let bounds = d.undo().unwrap();
        assert!(bounds.contains(10, 10));
        assert_eq!(bounds, Rect::new(0, 0, 256, 256));
    }

    #[test]
    fn locked_layers_reject_writes_and_leave_no_history() {
        let mut d = doc();
        let bg = d.root()[0];
        d.layer_mut(bg).unwrap().locks.pixels = true;
        let mut tx = d.begin("Brush", "Brush");
        assert!(!tx.set_sample(bg, 0, 1, 1, 1.0));
        tx.commit();
        assert!(
            !d.history.can_undo(),
            "a rejected write must not create a step"
        );
    }

    #[test]
    fn layer_property_changes_are_undoable() {
        let mut d = doc();
        let bg = d.root()[0];
        let mut props = d.layer(bg).unwrap().props.clone();
        props.opacity = 0.5;
        props.blend_mode = ultrablend::BlendMode::Multiply as u8;
        let mut tx = d.begin("Layer Style", "LayerOpacity");
        tx.set_props(bg, props);
        tx.commit();

        assert_eq!(d.layer(bg).unwrap().props.opacity, 0.5);
        d.undo();
        assert_eq!(d.layer(bg).unwrap().props.opacity, 1.0);
        d.redo();
        assert_eq!(
            d.layer(bg).unwrap().props.blend(),
            ultrablend::BlendMode::Multiply
        );
    }

    #[test]
    fn layer_insertion_and_removal_round_trips_through_undo() {
        let mut d = doc();
        let before = d.structure_state();
        let new_id = d.add_layer(
            Layer::pixel("Layer 1", 512, 512, Depth::U8, PixelFormat::RGBA),
            None,
            Some(0),
        );
        d.commit_structure(before, "New Layer");
        assert_eq!(d.root().len(), 2);
        assert_eq!(
            d.root()[0],
            new_id,
            "new layer goes to the top of the panel"
        );

        let entry = d.remove_layer(new_id).unwrap();
        d.history.push(entry);
        assert_eq!(d.root().len(), 1);

        d.undo();
        assert_eq!(d.root().len(), 2, "undo must bring the layer back");
        assert!(d.layer(new_id).is_some());
    }

    #[test]
    fn groups_reject_cycles() {
        let mut d = doc();
        let g = d.add_layer(
            Layer::group("Group 1", Depth::U8, PixelFormat::RGBA),
            None,
            None,
        );
        let inner = d.add_layer(
            Layer::pixel("Inner", 512, 512, Depth::U8, PixelFormat::RGBA),
            Some(g),
            None,
        );
        // A node is not a descendant of itself for re-parenting purposes.
        assert!(!d.is_descendant(g, inner));
        // Moving a group into itself, or into its own subtree, is refused.
        assert!(!d.move_layer(g, Some(g), 0));
        assert!(!d.move_layer(g, Some(inner), 0));
    }

    #[test]
    fn depth_survives_the_round_trip_for_16_and_32_bit() {
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            let mut d = Document::new("t", 64, 64, depth, PixelFormat::RGBA);
            let bg = d.root()[0];
            let mut tx = d.begin("Brush", "Brush");
            assert!(tx.set_sample(bg, 0, 3, 3, 0.5));
            tx.commit();
            let v = d
                .layer(bg)
                .unwrap()
                .surface
                .plane(0)
                .sample_normalized(3, 3);
            let expected = if depth == Depth::U8 {
                128.0 / 255.0
            } else {
                0.5
            };
            assert!((v - expected).abs() < 0.01, "{depth:?} gave {v}");
        }
    }
}
