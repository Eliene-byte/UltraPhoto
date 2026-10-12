//! # ultraui
//!
//! A UI toolkit built from scratch on wgpu — no immediate-mode framework, no
//! widget library.
//!
//! Why not `egui`/`iced`: those are excellent for tools, but "pixel-identical
//! to Photoshop" means controlling every pixel of a very specific and quite
//! idiosyncratic interface — dockable panel groups, a menu bar with system
//! glyph metrics, the options bar, tool flyouts, the scrubby sliders with
//! their triangle handles. Owning the renderer is what makes that reachable.
//! It also starts in milliseconds, because there is no font-atlas pass and no
//! accessibility tree to build first.
//!
//! The crate is layered so the parts that can be verified without a GPU are
//! separable from the parts that need one:
//!
//! * [`theme`] — colour, metrics and type scale. No GPU.
//! * [`layout`] — docking, splitters, panel geometry. No GPU.
//! * [`input`] — keyboard/mouse state, shortcuts. No GPU.
//! * `renderer` / `window` — the wgpu and winit halves, behind features.
//!
//! CI verifies the first three; that is where "looks exactly like Photoshop"
//! is actually decided.

pub mod input;
pub mod layout;
pub mod theme;

pub use theme::{Theme, PS_DARK};
