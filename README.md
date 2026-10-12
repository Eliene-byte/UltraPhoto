# UltraPhoto

A high-performance raster image editor in Rust. Same commands, same muscle
memory as Photoshop — built GPU-first, tiled so it stays light, and compiled in
CI so everyone gets the same binary.

> **Status.** Six crates, 75 passing tests, CI green on Linux/Windows/macOS.
> The GPU compositor and the UI toolkit are written but not yet wired to a
> window — see [Roadmap](#roadmap).

---

## Why it will be faster and lighter

| Concern | Photoshop | UltraPhoto approach |
|---|---|---|
| Filtering / compositing | CPU, one thread per row | **WGSL compute shaders** on the GPU, with SIMD CPU fallback |
| Memory for a large canvas | Allocates the full canvas per layer | **Lazily allocated 256×256 tiles** — a huge canvas you scribbled on costs kilobytes |
| Undo | Full-state history states | **Tile diffs** — a brush dab stores ~512 KiB, not the whole image |
| Channels | RGB + alpha, some CMYK | **Any channel count**, stored planar, handled by one uniform kernel |
| Colour | sRGB in practice | 8/16/32-bit documents, 32-bit linear working space, Lab, CMYK, spot channels |
| Startup | ~10 s, ~2 GB | Target < 1 s, < 150 MB (tracked by CI, not estimated) |

Three structural decisions carry most of that weight:

**Pixels are planar and tiled.** One plane per channel, cut into 256×256 tiles
that materialise on first write. A 60000×40000 canvas where you painted one
corner costs kilobytes instead of 9.6 GB.

**Layers live in storage buffers, not textures.** A document can have six
channels (CMYK + alpha + under-colour removal) and storage textures top out at
four. A single channel-major `array<f32>` handles every layout with one kernel.

**Every API speaks normalised `f32`.** Quantisation to 8/16/32-bit happens in
exactly one place, so no kernel ever has to know the document's bit depth.

---

## Crates

```
ultrapixel      core storage: tiled planes, bit depths, colour spaces
ultrablend      all 27 Photoshop blend modes (CPU reference impl)
ultradoc        document, layer tree, tile-diff history
ultraformat     PNG/JPEG/TIFF/WebP I/O with fidelity guarantees
ultraui         UI toolkit: Photoshop's exact theme, docking, shortcuts
ultragpu        wgpu device, layer streaming, WGSL compositor
```

Dependency rule: **crates never depend downward**. `ultrapixel` knows nothing
about layers; `ultradoc` knows nothing about the GPU. That is what keeps the
core testable without a display — and it is why CI can verify this project at
all.

### `ultrapixel` — storage

- `Plane` — one channel, tiled, lazily allocated. Reads and writes are
  normalised `0..1 f32`.
- `Surface` — the planes of one layer plus content bounds, so the compositor
  can skip empty regions.
- `Depth` — 8/16/32-bit per *document*, like Photoshop.
- `Rect` — half-open; an empty rect is the universal "nothing to do" signal.

### `ultrablend` — compositing math

All 27 modes, separable and non-separable, with opacity and fill kept separate
exactly as Photoshop models them. This is the **reference**: the WGSL kernels
must match it and a cross-check test enforces it, so CPU and GPU can never
silently drift.

### `ultradoc` — document & history

- Layers addressed by `LayerId`, never index — reordering the stack mid-stroke
  cannot retarget a tool's writes.
- History stores **tile diffs in both directions**; writes to one tile coalesce,
  so a 500-dab stroke is one patch per tile.
- Budgeted in **bytes, not steps** — a brush dab and "flatten document" cost
  different orders of magnitude.
- Reorder/delete records only the tree topology, never pixels.

### `ultraformat` — file I/O

- PNG, JPEG, TIFF, WebP, read and write.
- Detected by **magic bytes**, not extension.
- Bit depth **probed from the container** — a 16-bit TIFF is not silently
  truncated.
- Decoding converts **band by band** (64 rows) to keep the working set in L2.
- Sources without alpha arrive **fully opaque** rather than transparent.

### `ultraui` — interface

- `theme` — Photoshop's real dark-theme values as data, with the palette pinned
  by tests so a "prettify" commit cannot silently change the look.
- `layout` — dock/canvas/splitter geometry, hit testing, drop zones. Pure
  arithmetic, no GPU, so "does the layout still work at 800×600" is a unit test.
- `input` — key/edge state plus Photoshop's default shortcut map, using
  Photoshop's own scripting names so recorded actions replay correctly.

### `ultragpu` — compute

- `GpuContext` — adapter selection with a software fallback.
- `LayerGpu` — upload/download by dirty rect only.
- `Compositor` — one compute pass per layer over the dirty rect, accumulating
  in place (no ping-pong).

---

## Building

Local builds need **no Visual Studio**. Development uses the GNU toolchain:

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup override set stable-x86_64-pc-windows-gnu
cargo build
cargo test --workspace
```

> Windows + GNU needs MinGW binutils on `PATH` for `dlltool`, which
> `windows-sys` requires to build import libraries. Install
> [WinLibs](https://winlibs.com/) once and add `mingw64\bin` to `PATH`.

CI builds with MSVC — the hosted Windows runners already have the Visual Studio
C++ toolchain — and is the shipping path.

### GitHub Actions

`.github/workflows/ci.yml` runs four gates:

| Job | What it does |
|---|---|
| `lint` | `cargo fmt --check` + `clippy -D warnings` |
| `test` | build + test on Linux, Windows and macOS |
| `bench` | kernel benchmarks, to catch perf regressions |
| `release` | tag or manual dispatch → optimised binaries as artifacts |

Pushing a tag `v*` produces `ultrapaint-windows-x64`, `ultrapaint-macos-arm64`
and `ultrapaint-linux-x64`.

Release profile: `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `strip`.

---

## Roadmap

- [x] Tiled planar storage with lazy allocation
- [x] 8/16/32-bit depths, RGB / Gray / Lab / CMYK formats
- [x] All 27 blend modes + opacity/fill compositing
- [x] Document, layer tree, groups, masks, clipping
- [x] Tile-diff undo/redo with a byte budget
- [x] File I/O: PNG / JPEG / TIFF / WebP, read and write
- [x] Photoshop's dark theme as pinned, testable data
- [x] Dock layout geometry, hit testing and drop zones
- [x] Photoshop shortcut map with scripting-compatible names
- [x] WGSL compositor with CPU↔GPU parity tests
- [x] CI: fmt, clippy, tests on 3 OSes, release artifacts
- [ ] `ultrapaint` — window, menus, panels, canvas viewport, wiring it together
- [ ] `ultraui` widgets — buttons, sliders, scrubby fields, menus, lists
- [ ] Tools — move, marquee, lasso, magic wand, crop, brush, eraser, dodge,
      burn, blur/sharpen/smudge, type, pen
- [ ] Selection ops — expand/contract, feather, inverse, transform
- [ ] Filters — gaussian/box/motion blur, unsharp, liquify, neural upscale
- [ ] Adjustment layers and group isolation in the compositor
- [ ] PSD / PSB read-write
- [ ] Actions / scripting

---

## Licence

MIT OR Apache-2.0.

Adobe Photoshop is a trademark of Adobe Inc. UltraPhoto is an independent
project and is not affiliated with or endorsed by Adobe.