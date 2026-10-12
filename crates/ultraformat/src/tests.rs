//! Round-trip tests for the file codecs.
//!
//! These are deliberately about *fidelity*, not speed: the things that break
//! in practice are depth loss, alpha loss, and JPEG's missing alpha channel.

use super::*;
use ultrapixel::{Depth, PixelFormat, Rect, Surface};

/// A small surface with a known gradient and a transparent corner.
fn test_surface() -> Surface {
    let (w, h) = (37usize, 23usize);
    let mut s = Surface::new(w, h, Depth::U8, PixelFormat::RGBA);
    let full = Rect::new(0, 0, w as i32, h as i32);
    let mut r = vec![0f32; w * h];
    let mut g = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    let mut a = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            r[i] = x as f32 / w as f32;
            g[i] = y as f32 / h as f32;
            b[i] = 0.25;
            // Punch a fully transparent square so alpha survival is testable.
            a[i] = if x < 8 && y < 8 { 0.0 } else { 1.0 };
        }
    }
    s.plane_mut(0).write_rect(full, &r);
    s.plane_mut(1).write_rect(full, &g);
    s.plane_mut(2).write_rect(full, &b);
    s.plane_mut(3).write_rect(full, &a);
    s.note_bounds(full);
    s
}

fn decode(bytes: &[u8]) -> Surface {
    load_from_bytes(bytes, &LoadOptions::default())
        .expect("decode failed")
        .surface
}

#[test]
fn png_round_trip_is_lossless() {
    let src = test_surface();
    let opts = SaveOptions::default();
    let bytes = save_to_vec(&src, None, FileFormat::Png, &opts).expect("encode failed");

    assert_eq!(sniff_format(&bytes), Some(FileFormat::Png));

    let back = decode(&bytes);
    assert_eq!(back.width(), src.width());
    assert_eq!(back.height(), src.height());

    let full = Rect::new(0, 0, src.width() as i32, src.height() as i32);
    for ch in 0..4 {
        let mut want = vec![0f32; src.width() * src.height()];
        let mut got = vec![0f32; src.width() * src.height()];
        src.plane(ch).read_rect(full, &mut want);
        back.plane(ch).read_rect(full, &mut got);
        for (w, g) in want.iter().zip(&got) {
            assert!(
                (w - g).abs() < 1.0 / 255.0,
                "channel {ch} lost data: {w} != {g}"
            );
        }
    }
}

#[test]
fn png_preserves_transparency() {
    let src = test_surface();
    let bytes = save_to_vec(&src, None, FileFormat::Png, &SaveOptions::default()).unwrap();
    let back = decode(&bytes);
    // The deliberately transparent corner must come back transparent.
    assert_eq!(back.plane(3).sample_normalized(2, 2), 0.0);
    assert_eq!(back.plane(3).sample_normalized(20, 15), 1.0);
}

#[test]
fn jpeg_flattens_alpha_onto_the_requested_background() {
    let src = test_surface();
    let opts = SaveOptions {
        jpeg_quality: 100,
        // Pure red background makes the flatten observable.
        flatten_background: [1.0, 0.0, 0.0, 1.0],
        ..SaveOptions::default()
    };
    let bytes = save_to_vec(&src, None, FileFormat::Jpeg, &opts).unwrap();
    assert_eq!(sniff_format(&bytes), Some(FileFormat::Jpeg));

    let back = decode(&bytes);
    // The transparent corner is now red, not black.
    let px = back.sample_rgba_normalized(2, 2);
    assert!(px[0] > 0.7, "expected red background, got {px:?}");
    assert!(px[1] < 0.35, "expected green suppressed, got {px:?}");
    // JPEG cannot store alpha, so it must come back fully opaque.
    assert_eq!(px[3], 1.0);
}

#[test]
fn tiff_and_webp_carry_alpha() {
    let src = test_surface();
    for fmt in [FileFormat::Tiff, FileFormat::WebP] {
        let bytes = save_to_vec(&src, None, fmt, &SaveOptions::default())
            .unwrap_or_else(|e| panic!("{fmt:?} encode failed: {e}"));
        let back = decode(&bytes);
        assert_eq!(back.width(), src.width(), "{fmt:?} width");
        assert_eq!(back.height(), src.height(), "{fmt:?} height");
        // WebP is lossy, so allow tolerance; TIFF should be exact.
        let tol = if fmt == FileFormat::Tiff { 0.01 } else { 0.12 };
        assert!(
            (back.plane(3).sample_normalized(2, 2) - 0.0).abs() < tol,
            "{fmt:?} lost the transparent corner"
        );
    }
}

#[test]
fn decoding_is_detected_by_magic_not_extension() {
    let src = test_surface();
    let opts = SaveOptions::default();
    // A JPEG that is lying about being a PNG.
    let jpeg = save_to_vec(&src, None, FileFormat::Jpeg, &opts).unwrap();
    assert_eq!(sniff_format(&jpeg), Some(FileFormat::Jpeg));
    let meta = load_from_bytes(&jpeg, &LoadOptions::default()).unwrap().metadata;
    assert_eq!(meta.format, Some(FileFormat::Jpeg));
}

#[test]
fn garbage_input_fails_cleanly_instead_of_panicking() {
    let junk = [0xDEu8, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02];
    assert!(sniff_format(&junk).is_none());
    assert!(load_from_bytes(&junk, &LoadOptions::default()).is_err());
}

#[test]
fn a_truncated_png_is_rejected_not_silently_short() {
    let src = test_surface();
    let bytes = save_to_vec(&src, None, FileFormat::Png, &SaveOptions::default()).unwrap();
    let truncated = &bytes[..bytes.len() / 3];
    // Must either error or return fewer rows — never claim a full image.
    match load_from_bytes(truncated, &LoadOptions::default()) {
        Ok(img) => assert!(
            img.surface.height() < src.height(),
            "truncated file decoded as a full-size image"
        ),
        Err(_) => {}
    }
}

#[test]
fn band_rects_tile_the_canvas_without_gaps_or_overlap() {
    let full = Rect::new(0, 0, 100, 100);
    let band_count = (full.height() as usize).div_ceil(BAND_ROWS);
    let mut covered = 0usize;
    let mut y = 0i32;
    for band in 0..band_count {
        let r = band_rect(full, band, BAND_ROWS);
        assert!(!r.is_empty(), "band {band} was empty");
        assert_eq!(r.y0, y, "band {band} left a gap");
        covered += r.height() as usize;
        y = r.y1;
    }
    assert_eq!(y, 100, "bands must reach the bottom");
    assert_eq!(covered, 100);
}