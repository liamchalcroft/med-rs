//! Bit-level I/O and Rice/Golomb entropy coding for wavelet coefficients.

use super::decoding::CodecError;

// --- Zigzag encoding ---

/// Map signed i32 to unsigned u32: 0→0, -1→1, 1→2, -2→3, ...
#[inline]
pub fn zigzag_encode(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// Map unsigned u32 back to signed i32.
#[inline]
pub fn zigzag_decode(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

// --- BitWriter ---

/// Writes individual bits to a byte buffer, MSB-first within each byte.
pub struct BitWriter {
    buf: Vec<u8>,
    current: u8,
    bits_in_current: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            current: 0,
            bits_in_current: 0,
        }
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            current: 0,
            bits_in_current: 0,
        }
    }

    /// Write a single bit (0 or 1).
    #[inline]
    pub fn write_bit(&mut self, bit: u8) {
        self.current = (self.current << 1) | (bit & 1);
        self.bits_in_current += 1;
        if self.bits_in_current == 8 {
            self.buf.push(self.current);
            self.current = 0;
            self.bits_in_current = 0;
        }
    }

    /// Write `n` bits from `value` (MSB-first), where n <= 32.
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Write a unary code: `q` zeros followed by a 1.
    #[inline]
    pub fn write_unary(&mut self, q: u32) {
        for _ in 0..q {
            self.write_bit(0);
        }
        self.write_bit(1);
    }

    /// Rice-encode a single unsigned value with parameter k.
    /// Format: unary(q) + k low bits, where q = value >> k.
    #[inline]
    pub fn rice_encode(&mut self, value: u32, k: u8) {
        let q = value >> k;
        self.write_unary(q);
        if k > 0 {
            self.write_bits(value & ((1 << k) - 1), k);
        }
    }

    /// Flush any remaining bits (pad with zeros on the right).
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_current > 0 {
            self.current <<= 8 - self.bits_in_current;
            self.buf.push(self.current);
        }
        self.buf
    }
}

// --- BitReader ---

/// Reads individual bits from a byte buffer, MSB-first within each byte.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0..8, counts from MSB
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit. Returns `None` once the underlying data is
    /// exhausted, so a truncated bitstream is reported to the caller instead
    /// of indexing past the end of `data`.
    #[inline]
    pub fn read_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    /// Read `n` bits as a u32 (MSB-first). Returns `None` if `n > 32` (the
    /// value wouldn't fit) or the bitstream runs out before `n` bits are read.
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Option<u32> {
        if n > 32 {
            return None;
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Some(value)
    }

    /// Ceiling on a unary code's zero-run length. Chosen far above anything a
    /// real encoder produces (`compute_optimal_k` keeps the mean quotient
    /// small, and caps `k` at 24), so it never rejects valid data, while
    /// bounding a hostile all-zero bitstream to a fixed amount of work instead
    /// of scanning the whole buffer, and keeping the running count (a `u32`)
    /// from ever overflowing.
    const MAX_UNARY_RUN: u32 = 1 << 24;

    /// Read a unary code: count zeros until a 1 is found, return the count.
    /// Returns `None` if the bitstream is truncated before a terminating `1`
    /// bit, or the run exceeds [`Self::MAX_UNARY_RUN`] (a malformed stream).
    #[inline]
    pub fn read_unary(&mut self) -> Option<u32> {
        let mut count: u32 = 0;
        loop {
            if self.read_bit()? == 1 {
                return Some(count);
            }
            if count >= Self::MAX_UNARY_RUN {
                return None;
            }
            count += 1;
        }
    }

    /// Rice-decode a single unsigned value with parameter k. Returns `None`
    /// on a truncated/malformed bitstream, including `k >= 32` (which would
    /// overflow the returned `u32`).
    #[inline]
    pub fn rice_decode(&mut self, k: u8) -> Option<u32> {
        let q = self.read_unary()?;
        let r = if k > 0 { self.read_bits(k)? } else { 0 };
        q.checked_shl(u32::from(k)).map(|shifted| shifted | r)
    }
}

// --- Optimal Rice parameter selection ---

/// Compute the optimal Rice parameter k for a set of unsigned values.
/// Uses the formula: k = max(0, floor(log2(mean * ln(2)))).
/// Returns 0 for empty or all-zero data.
pub fn compute_optimal_k(values: &[u32]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let sum: u64 = values.iter().map(|&v| v as u64).sum();
    if sum == 0 {
        return 0;
    }
    let mean = sum as f64 / values.len() as f64;
    let k = (mean * std::f64::consts::LN_2).log2().floor();
    if k < 0.0 {
        0
    } else {
        (k as u8).min(24) // cap at 24 bits
    }
}

/// Encode a slice of i32 wavelet coefficients using zigzag + Rice coding.
/// Returns (encoded_bytes, rice_k).
pub fn rice_encode_subband(coefficients: &[i32]) -> (Vec<u8>, u8) {
    // Zigzag-encode all values
    let unsigned: Vec<u32> = coefficients.iter().map(|&v| zigzag_encode(v)).collect();
    let k = compute_optimal_k(&unsigned);

    // Estimate output size: each value takes ~(mean_q + 1 + k) bits
    let est_bits = unsigned.len() * (k as usize + 4);
    let mut writer = BitWriter::with_capacity(est_bits / 8 + 1);

    for &v in &unsigned {
        writer.rice_encode(v, k);
    }

    (writer.finish(), k)
}

/// Decode Rice-coded bytes back to i32 coefficients. `num_values` must
/// already be validated by the caller against the expected voxel count for
/// the subband (see `decoding::validated_num_values`), since it is used to
/// preallocate the result; this function only guards against the bitstream
/// itself running out or being malformed.
pub fn rice_decode_subband(data: &[u8], num_values: usize, k: u8) -> Result<Vec<i32>, CodecError> {
    let mut reader = BitReader::new(data);
    let mut result = Vec::with_capacity(num_values);
    for _ in 0..num_values {
        let unsigned = reader.rice_decode(k).ok_or_else(|| {
            CodecError("jvol subband bitstream truncated or malformed".to_string())
        })?;
        result.push(zigzag_decode(unsigned));
    }
    Ok(result)
}

// --- 3D Lorenzo predictor ---

/// Compute 3D Lorenzo prediction residuals for lossless compression.
/// The Lorenzo predictor uses 7 neighbors to predict each voxel:
///   predicted = a + b + c - d - e - f + g
/// where a,b,c are face neighbors, d,e,f are edge neighbors, g is vertex neighbor.
/// Residuals are typically very small for smooth 3D data.
pub fn lorenzo_predict_3d(data: &[i32], shape: [usize; 3]) -> Vec<i32> {
    let [ni, nj, nk] = shape;
    let mut residuals = Vec::with_capacity(ni * nj * nk);

    let idx = |i: usize, j: usize, k: usize| -> i32 { data[i * nj * nk + j * nk + k] };

    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let val = idx(i, j, k);
                let a = if i > 0 { idx(i - 1, j, k) } else { 0 };
                let b = if j > 0 { idx(i, j - 1, k) } else { 0 };
                let c = if k > 0 { idx(i, j, k - 1) } else { 0 };
                let d = if i > 0 && j > 0 {
                    idx(i - 1, j - 1, k)
                } else {
                    0
                };
                let e = if i > 0 && k > 0 {
                    idx(i - 1, j, k - 1)
                } else {
                    0
                };
                let f = if j > 0 && k > 0 {
                    idx(i, j - 1, k - 1)
                } else {
                    0
                };
                let g = if i > 0 && j > 0 && k > 0 {
                    idx(i - 1, j - 1, k - 1)
                } else {
                    0
                };
                let predicted = a + b + c - d - e - f + g;
                residuals.push(val - predicted);
            }
        }
    }
    residuals
}

/// Reconstruct data from 3D Lorenzo prediction residuals.
pub fn lorenzo_reconstruct_3d(residuals: &[i32], shape: [usize; 3]) -> Vec<i32> {
    let [ni, nj, nk] = shape;
    let mut data = vec![0i32; ni * nj * nk];

    let idx = |d: &[i32], i: usize, j: usize, k: usize| -> i32 { d[i * nj * nk + j * nk + k] };

    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let a = if i > 0 { idx(&data, i - 1, j, k) } else { 0 };
                let b = if j > 0 { idx(&data, i, j - 1, k) } else { 0 };
                let c = if k > 0 { idx(&data, i, j, k - 1) } else { 0 };
                let d_val = if i > 0 && j > 0 {
                    idx(&data, i - 1, j - 1, k)
                } else {
                    0
                };
                let e = if i > 0 && k > 0 {
                    idx(&data, i - 1, j, k - 1)
                } else {
                    0
                };
                let f = if j > 0 && k > 0 {
                    idx(&data, i, j - 1, k - 1)
                } else {
                    0
                };
                let g = if i > 0 && j > 0 && k > 0 {
                    idx(&data, i - 1, j - 1, k - 1)
                } else {
                    0
                };
                let predicted = a + b + c - d_val - e - f + g;
                let pos = i * nj * nk + j * nk + k;
                data[pos] = residuals[pos] + predicted;
            }
        }
    }
    data
}

/// Encode residuals as zigzag + varint bytes (compact for small values).
pub fn encode_varint(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let mut z = zigzag_encode(v);
        while z >= 0x80 {
            out.push((z as u8) | 0x80);
            z >>= 7;
        }
        out.push(z as u8);
    }
    out
}

/// Decode varint bytes back to i32 values.
pub fn decode_varint(bytes: &[u8], count: usize) -> Vec<i32> {
    let mut result = Vec::with_capacity(count);
    let mut i = 0;
    while result.len() < count && i < bytes.len() {
        let mut val: u32 = 0;
        let mut shift = 0;
        loop {
            let b = bytes[i];
            i += 1;
            val |= ((b & 0x7F) as u32) << shift;
            if b < 0x80 {
                break;
            }
            shift += 7;
        }
        result.push(zigzag_decode(val));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        let values = [0, 1, -1, 2, -2, 127, -128, i32::MAX, i32::MIN + 1];
        for &v in &values {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v, "Failed for {}", v);
        }
    }

    #[test]
    fn test_zigzag_mapping() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
    }

    #[test]
    fn test_bit_writer_reader_roundtrip() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b10110, 5);
        writer.write_bits(0b001, 3);
        writer.write_bits(0xFF, 8);
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(5), Some(0b10110));
        assert_eq!(reader.read_bits(3), Some(0b001));
        assert_eq!(reader.read_bits(8), Some(0xFF));
    }

    #[test]
    fn test_read_bits_rejects_too_wide() {
        let data = [0xFFu8; 8];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(33), None);
    }

    #[test]
    fn test_read_bit_truncated_returns_none() {
        let data = [0b1010_0000u8];
        let mut reader = BitReader::new(&data);
        for _ in 0..8 {
            assert!(reader.read_bit().is_some());
        }
        assert_eq!(reader.read_bit(), None);
    }

    #[test]
    fn test_unary_roundtrip() {
        let mut writer = BitWriter::new();
        writer.write_unary(0);
        writer.write_unary(3);
        writer.write_unary(7);
        writer.write_unary(1);
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_unary(), Some(0));
        assert_eq!(reader.read_unary(), Some(3));
        assert_eq!(reader.read_unary(), Some(7));
        assert_eq!(reader.read_unary(), Some(1));
    }

    #[test]
    fn test_unary_truncated_returns_none() {
        // All-zero bits with no terminating 1: a truncated unary code.
        let data = [0u8; 4];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_unary(), None);
    }

    #[test]
    fn test_rice_roundtrip() {
        for k in 0..8u8 {
            let mut writer = BitWriter::new();
            let values = [0u32, 1, 5, 15, 100, 255];
            for &v in &values {
                writer.rice_encode(v, k);
            }
            let data = writer.finish();

            let mut reader = BitReader::new(&data);
            for &expected in &values {
                let got = reader.rice_decode(k);
                assert_eq!(
                    got,
                    Some(expected),
                    "Rice roundtrip failed for k={}, v={}",
                    k,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_rice_decode_rejects_k_at_bit_width() {
        let mut writer = BitWriter::new();
        writer.rice_encode(5, 4);
        let data = writer.finish();
        let mut reader = BitReader::new(&data);
        // k == 32 would overflow the returned u32 on the final shift; must be
        // rejected rather than panicking or silently wrapping.
        assert_eq!(reader.rice_decode(32), None);
    }

    #[test]
    fn test_subband_encode_decode() {
        let coefficients: Vec<i32> = vec![0, 1, -1, 2, -2, 0, 0, 3, -5, 100, -100, 0];
        let (data, k) = rice_encode_subband(&coefficients);
        let decoded = rice_decode_subband(&data, coefficients.len(), k).unwrap();
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_subband_decode_truncated_data_is_err() {
        let coefficients: Vec<i32> = (0..200).map(|i| i * 3 - 50).collect();
        let (data, k) = rice_encode_subband(&coefficients);
        // Truncate the encoded bitstream partway through; decoding the full
        // declared value count must fail rather than panic.
        let truncated = &data[..data.len() / 4];
        assert!(rice_decode_subband(truncated, coefficients.len(), k).is_err());
    }

    #[test]
    fn test_subband_zeros() {
        let coefficients = vec![0i32; 1000];
        let (data, k) = rice_encode_subband(&coefficients);
        assert_eq!(k, 0);
        // 1000 zeros with k=0: each zero is unary(0)="1", so 1000 bits = 125 bytes
        assert_eq!(data.len(), 125);
        let decoded = rice_decode_subband(&data, 1000, k).unwrap();
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_optimal_k() {
        // All zeros → k=0
        assert_eq!(compute_optimal_k(&[0, 0, 0]), 0);
        // mean=1 → k = floor(log2(ln2)) = floor(-0.47) = 0
        assert_eq!(compute_optimal_k(&[0, 1, 2, 1]), 0);
        // Larger values should get higher k
        let large: Vec<u32> = (0..1000).map(|i| (i % 256) as u32).collect();
        let k = compute_optimal_k(&large);
        assert!(k >= 5 && k <= 8, "k={} for mean~127", k);
    }

    #[test]
    fn test_lorenzo_roundtrip() {
        let shape = [4, 5, 6];
        let data: Vec<i32> = (0..(4 * 5 * 6)).map(|i| (i * 7 + 3) % 100).collect();
        let residuals = lorenzo_predict_3d(&data, shape);
        let reconstructed = lorenzo_reconstruct_3d(&residuals, shape);
        assert_eq!(data, reconstructed);
    }

    #[test]
    fn test_lorenzo_smooth_data() {
        let shape = [8, 8, 8];
        let data: Vec<i32> = (0..(8 * 8 * 8))
            .map(|idx| {
                let i = idx / 64;
                let j = (idx / 8) % 8;
                let k = idx % 8;
                (i + j + k) as i32
            })
            .collect();
        let residuals = lorenzo_predict_3d(&data, shape);
        // For linear data, Lorenzo predictor should produce mostly zeros
        let nonzero: usize = residuals.iter().filter(|&&r| r != 0).count();
        assert!(
            nonzero < data.len() / 2,
            "Too many non-zero residuals for smooth data"
        );
    }

    #[test]
    fn test_varint_roundtrip() {
        let values: Vec<i32> = vec![0, 1, -1, 127, -128, 1000, -1000, 0, 0, 0];
        let encoded = encode_varint(&values);
        let decoded = decode_varint(&encoded, values.len());
        assert_eq!(values, decoded);
    }
}
