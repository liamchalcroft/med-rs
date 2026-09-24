//! CDF 9/7 wavelet transform (the irreversible JPEG 2000 wavelet) on
//! Fortran-ordered 3D blocks, computed by lifting with whole-sample symmetric
//! boundary extension.

const ALPHA: f32 = -1.586_134_3;
const BETA: f32 = -0.052_980_12;
const GAMMA: f32 = 0.882_911_1;
const DELTA: f32 = 0.443_506_87;
const K: f32 = 1.230_174_1;

/// An axis is decomposed further only while its low band has at least this
/// many samples.
const MIN_EXTENT: usize = 8;

/// For each level, which axes are transformed and the block extent before
/// that level. The same rule runs at encode and decode time.
pub(super) fn schedule(shape: [usize; 3], levels: usize) -> Vec<([bool; 3], [usize; 3])> {
    let mut extent = shape;
    let mut out = Vec::new();
    for _ in 0..levels {
        let axes = extent.map(|n| n >= MIN_EXTENT);
        if !axes.contains(&true) {
            break;
        }
        out.push((axes, extent));
        for (e, &t) in extent.iter_mut().zip(&axes) {
            if t {
                *e = e.div_ceil(2);
            }
        }
    }
    out
}

/// Coefficient indices grouped by subband: the final approximation band
/// first, then the detail bands from the coarsest level to the finest. Within
/// a band, indices are in Fortran order. Serializing coefficients in this
/// order puts similar values next to each other, which compresses better.
pub(super) fn subband_order(shape: [usize; 3], levels: usize) -> Vec<u32> {
    let schedule = schedule(shape, levels);
    let mut coarsest = shape;
    for (axes, extent) in &schedule {
        for a in 0..3 {
            if axes[a] {
                coarsest[a] = extent[a].div_ceil(2);
            }
        }
    }
    let mut order = Vec::with_capacity(shape.iter().product());
    let mut push_box = |lo: [usize; 3], hi: [usize; 3]| {
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    order.push((x + shape[0] * (y + shape[1] * z)) as u32);
                }
            }
        }
    };
    push_box([0; 3], coarsest);
    for (axes, extent) in schedule.iter().rev() {
        let half: [usize; 3] = std::array::from_fn(|a| {
            if axes[a] {
                extent[a].div_ceil(2)
            } else {
                extent[a]
            }
        });
        // Every combination of low/high halves along the transformed axes,
        // except all-low (the band refined at the next level).
        for band in 1..8u8 {
            if (0..3).any(|a| band >> a & 1 == 1 && !axes[a]) {
                continue;
            }
            let lo: [usize; 3] =
                std::array::from_fn(|a| if band >> a & 1 == 1 { half[a] } else { 0 });
            let hi: [usize; 3] = std::array::from_fn(|a| {
                if band >> a & 1 == 1 {
                    extent[a]
                } else {
                    half[a]
                }
            });
            push_box(lo, hi);
        }
    }
    order
}

/// Forward transform of `data` (a block of `shape` in Fortran order).
pub(super) fn forward(data: &mut [f32], shape: [usize; 3], levels: usize) {
    let mut line = Vec::new();
    let mut tmp = Vec::new();
    for (axes, extent) in schedule(shape, levels) {
        for (axis, _) in axes.iter().enumerate().filter(|(_, &on)| on) {
            for_each_line(data, shape, extent, axis, &mut line, |l| {
                forward_line(l, &mut tmp);
            });
        }
    }
}

/// Inverse of [`forward`].
pub(super) fn inverse(data: &mut [f32], shape: [usize; 3], levels: usize) {
    let mut line = Vec::new();
    let mut tmp = Vec::new();
    for (axes, extent) in schedule(shape, levels).into_iter().rev() {
        for (axis, _) in axes.iter().enumerate().rev().filter(|(_, &on)| on) {
            for_each_line(data, shape, extent, axis, &mut line, |l| {
                inverse_line(l, &mut tmp);
            });
        }
    }
}

/// Apply `f` to every line along `axis` within the sub-block `extent`.
fn for_each_line(
    data: &mut [f32],
    shape: [usize; 3],
    extent: [usize; 3],
    axis: usize,
    line: &mut Vec<f32>,
    mut f: impl FnMut(&mut [f32]),
) {
    let stride = [1, shape[0], shape[0] * shape[1]];
    let (a, b) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let n = extent[axis];
    line.resize(n, 0.0);
    for j in 0..extent[b] {
        for i in 0..extent[a] {
            let base = i * stride[a] + j * stride[b];
            if axis == 0 {
                f(&mut data[base..base + n]);
            } else {
                let s = stride[axis];
                for (k, v) in line.iter_mut().enumerate() {
                    *v = data[base + k * s];
                }
                f(line);
                for (k, v) in line.iter().enumerate() {
                    data[base + k * s] = *v;
                }
            }
        }
    }
}

/// One level of the 1D transform: `x` becomes `[low | high]`.
fn forward_line(x: &mut [f32], tmp: &mut Vec<f32>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend(x.iter().step_by(2));
    tmp.extend(x.iter().skip(1).step_by(2));
    let (s, d) = tmp.split_at_mut(ns);
    predict(s, d, ALPHA);
    update(s, d, BETA);
    predict(s, d, GAMMA);
    update(s, d, DELTA);
    for v in s.iter_mut() {
        *v *= K;
    }
    for v in d.iter_mut() {
        *v /= K;
    }
    x.copy_from_slice(tmp);
}

fn inverse_line(x: &mut [f32], tmp: &mut Vec<f32>) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend_from_slice(x);
    let (s, d) = tmp.split_at_mut(ns);
    for v in s.iter_mut() {
        *v /= K;
    }
    for v in d.iter_mut() {
        *v *= K;
    }
    update(s, d, -DELTA);
    predict(s, d, -GAMMA);
    update(s, d, -BETA);
    predict(s, d, -ALPHA);
    for (i, v) in s.iter().enumerate() {
        x[2 * i] = *v;
    }
    for (i, v) in d.iter().enumerate() {
        x[2 * i + 1] = *v;
    }
}

/// `d[i] += c * (s[i] + s[i + 1])`, mirroring `s` at the right edge.
fn predict(s: &[f32], d: &mut [f32], c: f32) {
    let last = s.len() - 1;
    for (i, v) in d.iter_mut().enumerate() {
        *v += c * (s[i] + s[(i + 1).min(last)]);
    }
}

/// `s[i] += c * (d[i - 1] + d[i])`, mirroring `d` at both edges.
fn update(s: &mut [f32], d: &[f32], c: f32) {
    let last = d.len() - 1;
    for (i, v) in s.iter_mut().enumerate() {
        *v += c * (d[i.saturating_sub(1)] + d[i.min(last)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_reconstruction_for_awkward_shapes() {
        for shape in [
            [1, 1, 1],
            [2, 1, 1],
            [7, 3, 1],
            [9, 16, 5],
            [64, 8, 33],
            [13, 13, 13],
        ] {
            let n: usize = shape.iter().product();
            let orig: Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 1000) as f32 * 0.37 - 100.0)
                .collect();
            let mut data = orig.clone();
            forward(&mut data, shape, 5);
            inverse(&mut data, shape, 5);
            let err = orig
                .iter()
                .zip(&data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err < 1e-3, "{shape:?}: {err}");
        }
    }

    #[test]
    fn smooth_data_compacts_into_the_low_band() {
        let shape = [32, 32, 32];
        let mut data: Vec<f32> = (0..32 * 32 * 32)
            .map(|i| {
                let (x, y, z) = (i % 32, (i / 32) % 32, i / 1024);
                (x + 2 * y + 3 * z) as f32
            })
            .collect();
        forward(&mut data, shape, 3);
        let energy: f32 = data.iter().map(|v| v * v).sum();
        let low: f32 = (0..4)
            .flat_map(|z| (0..4).flat_map(move |y| (0..4).map(move |x| x + 32 * (y + 32 * z))))
            .map(|i| data[i] * data[i])
            .sum();
        assert!(low / energy > 0.99, "{}", low / energy);
    }

    #[test]
    fn subband_order_is_a_permutation() {
        for shape in [
            [64, 64, 64],
            [37, 9, 1],
            [8, 8, 8],
            [3, 3, 3],
            [100, 13, 20],
        ] {
            let mut order = subband_order(shape, 4);
            order.sort_unstable();
            let n: u32 = shape.iter().product::<usize>() as u32;
            assert_eq!(order, (0..n).collect::<Vec<_>>(), "{shape:?}");
        }
    }

    #[test]
    fn schedule_skips_short_axes() {
        let s = schedule([64, 64, 1], 4);
        assert_eq!(s.len(), 4);
        assert!(s.iter().all(|(axes, _)| axes[0] && axes[1] && !axes[2]));
        assert!(schedule([4, 4, 4], 4).is_empty());
    }
}
