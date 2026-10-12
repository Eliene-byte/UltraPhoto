//! Stable, cheap identifiers.
//!
//! Layers are never referenced by index — reordering the stack must not
//! invalidate a tool that is mid-stroke. Every node gets a monotonically
//! increasing id at creation time.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u64);

        impl $name {
            #[inline]
            pub fn next() -> Self {
                $name(NEXT.fetch_add(1, Ordering::Relaxed))
            }
            #[inline]
            pub const fn raw(self) -> u64 {
                self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}{}", stringify!($name), self.0)
            }
        }
    };
}

id_type!(LayerId);
id_type!(DocumentId);
id_type!(HistoryId);

/// Index of a layer inside its parent's child list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerIndex(pub usize);
