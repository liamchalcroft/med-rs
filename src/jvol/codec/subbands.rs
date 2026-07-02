//! Subband management for 3D DWT-decomposed volumes.
//!
//! After L levels of 3D DWT, the volume is partitioned into 7*L + 1 subbands.
//! Each level splits the approximation subband into 8 octants:
//! - LLL (approximation, processed recursively)
//! - 7 detail subbands: LLH, LHL, LHH, HLL, HLH, HHL, HHH
//!
//! In memory (in-place DWT), subbands occupy non-overlapping regions of the original array.

use ndarray::Array3;

/// Identifies which detail subband within a DWT level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandType {
    /// Approximation (final LLL after all levels)
    LLL,
    LLH,
    LHL,
    LHH,
    HLL,
    HLH,
    HHL,
    HHH,
}

/// Describes one subband's location in the DWT-decomposed volume.
#[derive(Debug, Clone)]
pub struct SubbandInfo {
    pub level: usize,
    pub band_type: BandType,
    /// Start indices [i, j, k] in the full array.
    pub offset: [usize; 3],
    /// Shape [ni, nj, nk] of this subband.
    pub shape: [usize; 3],
}

/// Compute all subband regions for a volume of given shape after `levels` of 3D DWT.
/// Returns subbands in encoding order: coarsest detail subbands first, then approximation last.
pub fn compute_subbands(volume_shape: [usize; 3], levels: usize) -> Vec<SubbandInfo> {
    let mut result = Vec::with_capacity(7 * levels + 1);

    // Compute the extent at each level
    let mut extents = Vec::with_capacity(levels + 1);
    extents.push(volume_shape);
    for _ in 0..levels {
        let prev = *extents.last().unwrap();
        extents.push([
            prev[0].div_ceil(2),
            prev[1].div_ceil(2),
            prev[2].div_ceil(2),
        ]);
    }

    // For each level (from coarsest to finest), emit 7 detail subbands
    for level in (0..levels).rev() {
        let ext = extents[level]; // extent at this level
        let hi = extents[level + 1]; // half-sizes (= extent at next coarser level)

        let detail_bands = [
            (BandType::LLH, [0, 0, hi[2]], [hi[0], hi[1], ext[2] - hi[2]]),
            (BandType::LHL, [0, hi[1], 0], [hi[0], ext[1] - hi[1], hi[2]]),
            (
                BandType::LHH,
                [0, hi[1], hi[2]],
                [hi[0], ext[1] - hi[1], ext[2] - hi[2]],
            ),
            (BandType::HLL, [hi[0], 0, 0], [ext[0] - hi[0], hi[1], hi[2]]),
            (
                BandType::HLH,
                [hi[0], 0, hi[2]],
                [ext[0] - hi[0], hi[1], ext[2] - hi[2]],
            ),
            (
                BandType::HHL,
                [hi[0], hi[1], 0],
                [ext[0] - hi[0], ext[1] - hi[1], hi[2]],
            ),
            (
                BandType::HHH,
                [hi[0], hi[1], hi[2]],
                [ext[0] - hi[0], ext[1] - hi[1], ext[2] - hi[2]],
            ),
        ];

        for (band_type, offset, shape) in detail_bands {
            result.push(SubbandInfo {
                level,
                band_type,
                offset,
                shape,
            });
        }
    }

    // Final approximation subband (smallest LLL)
    let approx_shape = extents[levels];
    result.push(SubbandInfo {
        level: levels,
        band_type: BandType::LLL,
        offset: [0, 0, 0],
        shape: approx_shape,
    });

    result
}

/// Extract a subband from the DWT-decomposed array as a flat Vec<i32>.
/// Rounds f64 values to i32 (for lossless mode, DWT coefficients are already integers).
pub fn extract_subband_i32(array: &Array3<f64>, info: &SubbandInfo) -> Vec<i32> {
    let [oi, oj, ok] = info.offset;
    let [ni, nj, nk] = info.shape;
    let mut values = Vec::with_capacity(ni * nj * nk);
    for i in oi..oi + ni {
        for j in oj..oj + nj {
            for k in ok..ok + nk {
                values.push(array[[i, j, k]].round() as i32);
            }
        }
    }
    values
}

/// Inject flat i32 values back into the array at the subband's location.
pub fn inject_subband_i32(array: &mut Array3<f64>, info: &SubbandInfo, values: &[i32]) {
    let [oi, oj, ok] = info.offset;
    let [ni, nj, nk] = info.shape;
    debug_assert_eq!(values.len(), ni * nj * nk);
    let mut idx = 0;
    for i in oi..oi + ni {
        for j in oj..oj + nj {
            for k in ok..ok + nk {
                array[[i, j, k]] = values[idx] as f64;
                idx += 1;
            }
        }
    }
}

/// Inject flat i32 coefficients into an f32 array at the subband's location,
/// scaling each by `step` (dequantization) on the way in. Used by the f32
/// lossy-decode and downsampled-decode paths, which keep coefficients in f32.
pub fn inject_subband_i32_f32(
    array: &mut Array3<f32>,
    info: &SubbandInfo,
    values: &[i32],
    step: f32,
) {
    let [oi, oj, ok] = info.offset;
    let [ni, nj, nk] = info.shape;
    debug_assert_eq!(values.len(), ni * nj * nk);
    let mut idx = 0;
    for i in oi..oi + ni {
        for j in oj..oj + nj {
            for k in ok..ok + nk {
                array[[i, j, k]] = values[idx] as f32 * step;
                idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subband_count() {
        let subs = compute_subbands([64, 64, 64], 3);
        assert_eq!(subs.len(), 7 * 3 + 1); // 22 subbands
    }

    #[test]
    fn test_subbands_cover_volume() {
        let shape = [32, 32, 32];
        let levels = 3;
        let subs = compute_subbands(shape, levels);

        // Total voxels across all subbands should equal the volume
        let total: usize = subs
            .iter()
            .map(|s| s.shape[0] * s.shape[1] * s.shape[2])
            .sum();
        assert_eq!(total, 32 * 32 * 32);
    }

    #[test]
    fn test_subbands_cover_odd_dims() {
        let shape = [15, 17, 13];
        let levels = 2;
        let subs = compute_subbands(shape, levels);

        let total: usize = subs
            .iter()
            .map(|s| s.shape[0] * s.shape[1] * s.shape[2])
            .sum();
        assert_eq!(total, 15 * 17 * 13);
    }

    #[test]
    fn test_extract_inject_roundtrip() {
        let shape = [16, 16, 16];
        let original =
            Array3::from_shape_fn((16, 16, 16), |(i, j, k)| (i * 100 + j * 10 + k) as f64);

        let subs = compute_subbands(shape, 2);
        let mut reconstructed = Array3::zeros((16, 16, 16));

        for sub in &subs {
            let values = extract_subband_i32(&original, sub);
            inject_subband_i32(&mut reconstructed, sub, &values);
        }

        for (a, b) in original.iter().zip(reconstructed.iter()) {
            assert_eq!(*a, *b, "Extract/inject roundtrip failed");
        }
    }

    #[test]
    fn test_last_subband_is_approx() {
        let subs = compute_subbands([64, 64, 64], 3);
        let last = subs.last().unwrap();
        assert_eq!(last.band_type, BandType::LLL);
        assert_eq!(last.offset, [0, 0, 0]);
        assert_eq!(last.shape, [8, 8, 8]); // 64 / 2^3
    }
}
