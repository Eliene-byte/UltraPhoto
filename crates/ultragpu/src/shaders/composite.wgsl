//! The WGSL compositor kernel.
//!
//! This mirrors [`ultrablend`](../../ultrablend/index.rs) function for
//! function. The CPU implementation is the reference and a cross-check test
//! compares them pixel-for-pixel — if the two ever drift, the test fails rather
//! than the user seeing different results on different machines.
//!
//! Everything is `f32` in `0..1`, straight (unassociated) alpha.

struct Params {
    /// Dirty rectangle in pixels: x0, y0, x1, y1.
    rect: vec4<i32>,
    /// Layer dimensions, for bounds checks.
    size: vec2<u32>,
    /// Index of the layer's alpha plane.
    alpha_plane: u32,
    /// Padding to 16-byte alignment.
    _pad: u32,
};

struct Blend {
    /// Per-channel separable blend.
    mode: u32,
    /// Non-separable blend, or 0xFFFFFFFF when `mode` handles it.
    non_separable: u32,
    opacity: f32,
    fill: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// `dst` is read AND written: the compositor accumulates in place so a stack of
// layers needs no ping-pong buffer.
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> src_alpha: array<f32>;
@group(0) @binding(4) var<uniform> blend: Blend;

/// Blend mode ids. These MUST match `ultrablend::BlendMode`'s discriminants.
const MODE_NORMAL: u32 = 0u;
const MODE_DISSOLVE: u32 = 1u;
const MODE_DARKEN: u32 = 2u;
const MODE_MULTIPLY: u32 = 3u;
const MODE_COLOR_BURN: u32 = 4u;
const MODE_LINEAR_BURN: u32 = 5u;
const MODE_DARKER_COLOR: u32 = 6u;
const MODE_LIGHTEN: u32 = 7u;
const MODE_SCREEN: u32 = 8u;
const MODE_COLOR_DODGE: u32 = 9u;
const MODE_LINEAR_DODGE: u32 = 10u;
const MODE_LIGHTER_COLOR: u32 = 11u;
const MODE_OVERLAY: u32 = 12u;
const MODE_SOFT_LIGHT: u32 = 13u;
const MODE_HARD_LIGHT: u32 = 14u;
const MODE_VIVID_LIGHT: u32 = 15u;
const MODE_LINEAR_LIGHT: u32 = 16u;
const MODE_PIN_LIGHT: u32 = 17u;
const MODE_HARD_MIX: u32 = 18u;
const MODE_DIFFERENCE: u32 = 19u;
const MODE_EXCLUSION: u32 = 20u;
const MODE_SUBTRACT: u32 = 21u;
const MODE_DIVIDE: u32 = 22u;
const MODE_HUE: u32 = 23u;
const MODE_SATURATION: u32 = 24u;
const MODE_COLOR: u32 = 25u;
const MODE_LUMINOSITY: u32 = 26u;

fn luma(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

fn soft_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    }
    var d = b;
    if (b <= 0.25) {
        d = ((16.0 * b - 12.0) * b + 4.0) * b;
    } else {
        d = sqrt(b);
    }
    return b + (2.0 * s - 1.0) * (d - b);
}

fn blend_channel(b: f32, s: f32) -> f32 {
    switch blend.mode {
        case 0u, MODE_DISSOLVE: { return s; }
        case MODE_DARKEN: { return min(b, s); }
        case MODE_MULTIPLY: { return b * s; }
        case MODE_COLOR_BURN: {
            if (b <= 0.0) { return 0.0; }
            if (s >= 1.0) { return 1.0; }
            // The min() is load-bearing: at s == 0 the quotient is +inf and
            // without the clamp this returns -inf. Matches the CPU reference.
            return 1.0 - min((1.0 - b) / s, 1.0);
        }
        case MODE_LINEAR_BURN: { return clamp(b + s - 1.0, 0.0, 1.0); }
        case MODE_LIGHTEN: { return max(b, s); }
        case MODE_SCREEN: { return b + s - b * s; }
        case MODE_COLOR_DODGE: {
            if (b <= 0.0) { return 0.0; }
            if (s >= 1.0) { return 1.0; }
            return min(b / (1.0 - s), 1.0);
        }
        case MODE_LINEAR_DODGE: { return min(b + s, 1.0); }
        case MODE_OVERLAY: { return hard_light(s, b); }
        case MODE_SOFT_LIGHT: { return soft_light(b, s); }
        case MODE_HARD_LIGHT: { return hard_light(b, s); }
        case MODE_VIVID_LIGHT: {
            if (s <= 0.5) {
                if (b <= 0.0) { return 0.0; }
                return min(b / (1.0 - 2.0 * s), 1.0);
            }
            if (b >= 1.0) { return 1.0; }
            return max(1.0 - (1.0 - b) / (2.0 * s - 1.0), 0.0);
        }
        case MODE_LINEAR_LIGHT: { return clamp(b + 2.0 * s - 1.0, 0.0, 1.0); }
        case MODE_PIN_LIGHT: {
            if (s <= 0.5) { return min(b, 2.0 * s); }
            return max(b, 2.0 * s - 1.0);
        }
        case MODE_HARD_MIX: {
            if (b + s >= 1.0) { return 1.0; }
            return 0.0;
        }
        case MODE_DIFFERENCE: { return abs(b - s); }
        case MODE_EXCLUSION: { return b + s - 2.0 * b * s; }
        case MODE_SUBTRACT: { return clamp(b - s, 0.0, 1.0); }
        case MODE_DIVIDE: {
            if (s <= 0.0) { return 1.0; }
            return min(b / s, 1.0);
        }
        default: { return s; }
    }
}

fn hard_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return b * (2.0 * s);
    }
    return 1.0 - (1.0 - 2.0 * (s - 0.5)) * (1.0 - b);
}

// ---------------------------------------------------------------- HSL helpers

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.x, max(c.y, c.z));
    let mn = min(c.x, min(c.y, c.z));
    let l = (mx + mn) * 0.5;
    if (abs(mx - mn) < 1e-9) {
        return vec3<f32>(0.0, 0.0, l);
    }
    let d = mx - mn;
    var s = 0.0;
    if (l > 0.5) {
        s = d / (2.0 - mx - mn);
    } else {
        s = d / (mx + mn);
    }
    var h = 0.0;
    if (abs(mx - c.x) < 1e-9) {
        h = (c.y - c.z) / d;
        if (c.y < c.z) { h = h + 6.0; }
    } else if (abs(mx - c.y) < 1e-9) {
        h = (c.z - c.x) / d + 2.0;
    } else {
        h = (c.x - c.y) / d + 4.0;
    }
    return vec3<f32>(h / 6.0, s, l);
}

fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if (t < 0.0) { t = t + 1.0; }
    if (t > 1.0) { t = t - 1.0; }
    if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
    if (t < 0.5) { return q; }
    if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    if (abs(hsl.y) < 1e-9) {
        return vec3<f32>(hsl.z, hsl.z, hsl.z);
    }
    var q = 0.0;
    if (hsl.z < 0.5) {
        q = hsl.z * (1.0 + hsl.y);
    } else {
        q = hsl.z + hsl.y - hsl.z * hsl.y;
    }
    let p = 2.0 * hsl.z - q;
    return vec3<f32>(
        hue_to_rgb(p, q, hsl.x + 1.0 / 3.0),
        hue_to_rgb(p, q, hsl.x),
        hue_to_rgb(p, q, hsl.x - 1.0 / 3.0),
    );
}

fn blend_non_separable(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    let bl = rgb_to_hsl(b);
    let sl = rgb_to_hsl(s);

    switch blend.mode {
        case MODE_DARKER_COLOR: {
            if (luma(s) < luma(b)) { return s; }
            return b;
        }
        case MODE_LIGHTER_COLOR: {
            if (luma(s) > luma(b)) { return s; }
            return b;
        }
        // Hue: source hue, backdrop saturation + lightness.
        case MODE_HUE: { return hsl_to_rgb(vec3<f32>(sl.x, bl.y, bl.z)); }
        // Saturation: source saturation, backdrop hue + lightness.
        case MODE_SATURATION: { return hsl_to_rgb(vec3<f32>(bl.x, sl.y, bl.z)); }
        // Color: source hue + saturation, backdrop lightness.
        case MODE_COLOR: { return hsl_to_rgb(vec3<f32>(sl.x, sl.y, bl.z)); }
        // Luminosity: source lightness, backdrop hue + saturation.
        case MODE_LUMINOSITY: { return hsl_to_rgb(vec3<f32>(bl.x, bl.y, sl.z)); }
        default: { return s; }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x) + params.rect.x;
    let y = i32(gid.y) + params.rect.y;
    if (x >= params.rect.z || y >= params.rect.w) {
        return;
    }
    if (u32(x) >= params.size.x || u32(y) >= params.size.y) {
        return;
    }

    let idx = u32(y) * params.size.x + u32(x);
    let plane = params.size.x * params.size.y;

    // Colour planes come first: R, G, B. Grey documents use plane 0 for all three.
    let sb = vec3<f32>(src[idx], src[plane + idx], src[plane * 2u + idx]);
    let db = vec3<f32>(dst[idx], dst[plane + idx], dst[plane * 2u + idx]);

    var sa = 1.0;
    if (params.alpha_plane != 0xFFFFFFFFu) {
        sa = src[params.alpha_plane * plane + idx];
    }

    let src_alpha = clamp(sa * blend.opacity * blend.fill, 0.0, 1.0);
    if (src_alpha <= 0.0) {
        // Fully transparent source: the backdrop survives untouched. Early-out
        // keeps a hidden layer from costing a full read-modify-write.
        return;
    }

    var sc = sb;
    switch blend.non_separable {
        case 0u: {
            sc = vec3<f32>(
                blend_channel(db.x, sb.x),
                blend_channel(db.y, sb.y),
                blend_channel(db.z, sb.z),
            );
        }
        default: {
            sc = blend_non_separable(db, sb);
        }
    }

    let inv = 1.0 - src_alpha;
    dst[idx] = sc.x * src_alpha + db.x * inv;
    dst[plane + idx] = sc.y * src_alpha + db.y * inv;
    dst[plane * 2u + idx] = sc.z * src_alpha + db.z * inv;
}