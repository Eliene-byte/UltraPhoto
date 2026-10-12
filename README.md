# UltraPhoto

A high-performance raster image editor in Rust. Same commands, same muscle
memory as Photoshop — built GPU-first, tiled so it stays light, and compiled in
CI so everyone gets the same binary.

> **Status: foundational crates.** See [Roadmap](#roadmap) for what exists
> today. Everything in this repo is tested; nothing is a stub.

---

## Why it will be faster and lighter

| Concern | Photoshop | UltraPhoto approach |
|---|---|---|
| Filtering / compositing | CPU, one thread per row | **WGSL compute shaders** on the GPU, with AVX2/AVX-512 SIMD fallback for kernels that need it |
| Memory for a large canvas | Allocates the full canvas per layer | **Lazily allocated 256×256 tiles** — a 60000×40000 canvas you scribbled on costs kilobytes |
| Undo | Full-state history states | **Tile diffs** — a brush dab stores ~512 KiB, not the whole image |
| Colour | sRGB only in practice | 8/16/32-bit documents, linear-light 32-bit working space, Lab, CMYK, spot channels |
| Startup | ~10 s, ~2 GB | Target < 1 s, < 150 MB (measured, not estimated — CI tracks it) |

The key structural decision: **pixels are stored planar, one plane per channel,
tiled**. That maps one-to-one onto GPU textures, removes the de-interleaving
pass that dominates naive editors, and makes per-channel operations (hue, a
channel merge, a mask) free.

---

## Crates

```
ultrapixel      core storage: tiled planes, bit depths, colour spaces
ultrablend      all 27 Photoshop blend modes (CPU reference impl)
ultradoc        document, layer tree, tile-diff history
ultraformat     PNG/JPEG/TIFF/WebP I/O with fidelity guarantees
ultragpu        wgpu device, compute pipelines, WGSL kernels   [next]
ultrapixel-simd AVX2/AVX-512 kernels                           [next]
ultraui         custom wgpu UI toolkit — pixel-exact PS theme  [next]
ultrapaint      the application binary                          [next]
```

Dependency rule: **crates never depend downward**. `ultrapixel` knows nothing
about layers; `ultradoc` knows nothing about the GPU; `ultrapaint` wires it
together. That is what keeps the core testable without a display.

### `ultrapixel` — storage

- `Plane` — a single channel, cut into 256×256 tiles that materialise on first
  write and are freed on demand. Every read/write API is **normalised `0..1
  f32`** so quantisation to 8/16/32-bit happens in exactly one place.
- `Surface` — the planes of one layer, plus content bounds that let the
  compositor skip empty regions and shrink GPU dispatches.
- `Depth` — 8-bit / 16-bit / 32-bit float, per *document*, like Photoshop.
- `Rect` — half-open integer rects; an empty rect is the universal "nothing to
  do" signal, which keeps dirty-rect logic branch-free.

### `ultrablend` — compositing math

All 27 modes, separable and non-separable, plus opacity/fill separation exactly
as Photoshop models it. This crate is the **reference**: the WGSL kernels must
match it pixel-for-pixel and a cross-check test enforces that, so CPU and GPU
can never silently drift.

### `ultradoc` — document & history

- Layers are addressed by `LayerId`, never by index, so reordering the stack
  mid-stroke can't retarget a tool's writes.
- History entries store **tile diffs in both directions**. Writes to the same
  tile coalesce, so a 500-dab stroke is one patch per tile, not 500.
- The history budget is in **bytes, not step count** — a brush dab and
  "flatten document" cost wildly different amounts, and a count-based limit
  thrashes on both.
- Layer add/delete/reorder records only the flat tree topology (a few dozen
  bytes per layer), never pixels.

### `ultraformat` — file I/O

- PNG, JPEG, TIFF, WebP — read and write.
- **Detected by magic bytes, not extension.** A JPEG named `.png` is extremely
  common and must not be mis-parsed.
- **Depth is probed from the container** (PNG IHDR, TIFF tag 258). A 16-bit
  TIFF does not get silently truncated to 8 bits on the way in.
- Decoding converts **band by band** (64 rows) so the working set stays inside
  L2 instead of materialising the full canvas three times.
- JPEG has no alpha, so it is flattened against an explicit background rather
  than against whatever happened to be in the buffer.
- Sources without alpha are marked **fully opaque** on load — a JPEG must not
  arrive as a transparent canvas.

---

## Building

Local builds do **not** need Visual Studio. Development uses the GNU toolchain:

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup override set stable-x86_64-pc-windows-gnu
cargo build
cargo test
```

CI builds with MSVC (the hosted Windows runners have the Visual Studio C++
toolchain preinstalled) and is the shipping path.

```bash
cargo run --release          # run
cargo test --workspace       # 36 tests
cargo clippy --workspace --all-targets -- -D warnings
```

### GitHub Actions

`.github/workflows/ci.yml` runs four gates:

| Job | What it does |
|---|---|
| `lint` | `cargo fmt --check` + `clippy -D warnings` |
| `test` | build + test on Linux, Windows and macOS |
| `bench` | kernel benchmarks, to catch perf regressions |
| `release` | tag/manual dispatch → optimised binaries as artifacts |

Pushing a tag `v*` or dispatching the workflow manually produces
`ultrapaint-windows-x64`, `ultrapaint-macos-arm64` and `ultrapaint-linux-x64`.

The release profile is tuned hard: `lto = "fat"`, `codegen-units = 1`,
`panic = "abort"`, `strip = true`.

---

## Roadmap

- [x] Tiled planar storage with lazy allocation
- [x] 8/16/32-bit depths, RGB / Gray / Lab / CMYK formats
- [x] All 27 blend modes + opacity/fill compositing
- [x] Document, layer tree, groups, masks, clipping
- [x] Tile-diff undo/redo with a byte budget
- [x] File I/O: PNG / JPEG / TIFF / WebP, read and write
- [x] CI: fmt, clippy, tests on 3 OSes, release artifacts
- [ ] **`ultragpu`** — wgpu device, layer↔texture streaming, composite pipeline
- [ ] **`ultraui`** — custom UI toolkit: Photoshop's exact dark theme, dockable
      panels, menu bar, options bar, tool presets
- [ ] Compositor — groups, masks, clipping masks, adjustment layers
- [ ] Tools — move, marquee, lasso, magic wand, crop, brush, eraser, dodge,
      burn, blur/sharpen/smudge, type, pen
- [ ] Selection ops — expand/contract, feather, inverse, transform
- [ ] Filters — gaussian/box/motion blur, unsharp, liquify, neural upscale
- [ ] PSD / PSB read-write
- [ ] Actions / scripting layer (Photoshop-compatible command names)

---

## Licence

MIT OR Apache-2.0.

Adobe Photoshop is a trademark of Adobe Inc. UltraPhoto is an independent
project and is not affiliated with or endorsed by Adobe.