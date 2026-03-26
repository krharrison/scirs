//! Instant-NGP multi-resolution hash encoding (Müller et al. 2022).
//!
//! Reference: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
//! (Müller, Evans, Schied, Keller — SIGGRAPH 2022).
//!
//! Architecture summary
//! --------------------
//! L levels of resolution, each with a hash table of capacity 2^{T} entries.
//! At resolution level l the grid spacing is 1 / (N_min · b^l) where
//! b = (N_max / N_min)^{1/(L-1)}.
//!
//! For each query position:
//!  1. Determine the 8 grid-cell corners at level l.
//!  2. Hash each corner with a spatial hash function.
//!  3. Look up F feature scalars from the hash table.
//!  4. Trilinearly interpolate the 8 corner features.
//!  5. Concatenate features across all L levels → L·F-dim vector.

// ── Constants for the hash function ───────────────────────────────────────

/// Knuth-style multiplicative constants (prime) used in the spatial hash.
const PI1: u64 = 2_654_435_761;
const PI2: u64 = 805_459_861;
const PI3: u64 = 3_674_653_429;

// ── LCG PRNG ──────────────────────────────────────────────────────────────

const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-half, +half].
    fn next_uniform(&mut self, half: f64) -> f64 {
        self.next_f64() * 2.0 * half - half
    }
}

// ── Spatial hash function ──────────────────────────────────────────────────

/// Map a 3-D integer grid coordinate to a hash-table index.
///
/// Uses the XOR-multiply spatial hash from Müller et al. 2022:
/// ```text
/// h = (x·π₁ ⊕ y·π₂ ⊕ z·π₃) mod table_size
/// ```
///
/// # Arguments
///
/// * `x`, `y`, `z`  – integer voxel coordinates (may be negative).
/// * `table_size`    – capacity of the hash table (power-of-two preferred).
pub fn hash_coords(x: i32, y: i32, z: i32, table_size: usize) -> usize {
    let hx = (x as i64 as u64).wrapping_mul(PI1);
    let hy = (y as i64 as u64).wrapping_mul(PI2);
    let hz = (z as i64 as u64).wrapping_mul(PI3);
    ((hx ^ hy ^ hz) as usize) % table_size
}

// ── Multi-resolution hash encoder ─────────────────────────────────────────

/// Multi-resolution hash encoder from Instant-NGP (Müller et al. 2022).
///
/// # Fields
///
/// * `hash_tables`           – `[L]` levels, each a `[table_size × F]` array of feature scalars.
/// * `n_levels`              – number of resolution levels L.
/// * `n_features_per_level`  – number of features F per hash entry.
/// * `base_resolution`       – grid resolution at level 0.
/// * `resolution_growth`     – per-level resolution multiplier b.
/// * `table_size`            – capacity of every hash table (= 2^{log2_hashmap_size}).
pub struct HashEncoder {
    /// `[L][table_size][F]` – learnable feature vectors.
    pub hash_tables: Vec<Vec<Vec<f64>>>,
    /// Number of resolution levels L.
    pub n_levels: usize,
    /// Number of feature scalars F stored per hash-table entry.
    pub n_features_per_level: usize,
    /// Grid resolution at the coarsest level (level 0).
    pub base_resolution: usize,
    /// Per-level resolution growth factor b = (N_max/N_min)^{1/(L-1)}.
    pub resolution_growth: f64,
    /// Capacity of each hash table (= 2^{log2_hashmap_size}).
    pub table_size: usize,
}

impl HashEncoder {
    /// Construct and randomly initialise a new `HashEncoder`.
    ///
    /// Features are initialised uniformly in `[-0.0001, 0.0001]`.
    ///
    /// # Arguments
    ///
    /// * `n_levels`             – number of resolution levels.
    /// * `n_features_per_level` – features per hash entry.
    /// * `log2_hashmap_size`    – log₂ of hash-table capacity.
    /// * `base_resolution`      – coarsest grid resolution.
    /// * `finest_resolution`    – finest grid resolution.
    /// * `seed`                 – PRNG seed.
    pub fn new(
        n_levels: usize,
        n_features_per_level: usize,
        log2_hashmap_size: usize,
        base_resolution: usize,
        finest_resolution: usize,
        seed: u64,
    ) -> Self {
        let table_size = 1_usize << log2_hashmap_size.min(30);
        let resolution_growth = if n_levels > 1 {
            (finest_resolution as f64 / base_resolution as f64).powf(1.0 / (n_levels - 1) as f64)
        } else {
            1.0
        };

        let mut rng = Lcg::new(seed);
        let hash_tables: Vec<Vec<Vec<f64>>> = (0..n_levels)
            .map(|_| {
                (0..table_size)
                    .map(|_| {
                        (0..n_features_per_level)
                            .map(|_| rng.next_uniform(0.0001))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            hash_tables,
            n_levels,
            n_features_per_level,
            base_resolution,
            resolution_growth,
            table_size,
        }
    }

    /// Total output feature dimension = `n_levels × n_features_per_level`.
    #[inline]
    pub fn n_output_features(&self) -> usize {
        self.n_levels * self.n_features_per_level
    }

    /// Grid resolution at level `l`.
    fn level_resolution(&self, l: usize) -> f64 {
        self.base_resolution as f64 * self.resolution_growth.powi(l as i32)
    }

    /// Retrieve the feature vector stored at hash-table entry `idx` at level `l`.
    fn get_feature(&self, l: usize, idx: usize) -> &[f64] {
        &self.hash_tables[l][idx % self.table_size]
    }

    /// Trilinearly interpolate the 8 corner features of the voxel containing
    /// `pos` at resolution level `l`.
    ///
    /// Returns a `Vec<f64>` of length `n_features_per_level`.
    pub fn lookup_features(&self, pos: &[f64; 3], level: usize) -> Vec<f64> {
        let res = self.level_resolution(level);
        // Scale position into [0, res]
        let px = pos[0] * res;
        let py = pos[1] * res;
        let pz = pos[2] * res;

        // Integer lower corner
        let x0 = px.floor() as i32;
        let y0 = py.floor() as i32;
        let z0 = pz.floor() as i32;

        // Trilinear weights
        let wx = px - x0 as f64;
        let wy = py - y0 as f64;
        let wz = pz - z0 as f64;

        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        // 8 corners and their trilinear weights
        let corners = [
            // (dx, dy, dz, weight)
            (x0, y0, z0, (1.0 - wx) * (1.0 - wy) * (1.0 - wz)),
            (x1, y0, z0, wx * (1.0 - wy) * (1.0 - wz)),
            (x0, y1, z0, (1.0 - wx) * wy * (1.0 - wz)),
            (x1, y1, z0, wx * wy * (1.0 - wz)),
            (x0, y0, z1, (1.0 - wx) * (1.0 - wy) * wz),
            (x1, y0, z1, wx * (1.0 - wy) * wz),
            (x0, y1, z1, (1.0 - wx) * wy * wz),
            (x1, y1, z1, wx * wy * wz),
        ];

        let mut out = vec![0.0_f64; self.n_features_per_level];
        for (cx, cy, cz, w) in &corners {
            let idx = hash_coords(*cx, *cy, *cz, self.table_size);
            let feat = self.get_feature(level, idx);
            for (o, &f) in out.iter_mut().zip(feat.iter()) {
                *o += w * f;
            }
        }
        out
    }

    /// Encode a 3-D position by looking up and concatenating features from all
    /// `n_levels` resolution levels.
    ///
    /// Output length: `n_levels × n_features_per_level`.
    pub fn encode(&self, pos: &[f64; 3]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_output_features());
        for l in 0..self.n_levels {
            out.extend_from_slice(&self.lookup_features(pos, l));
        }
        out
    }
}

// ── Tiny MLP for Instant-NGP ───────────────────────────────────────────────

/// A tiny two-layer MLP used in conjunction with [`HashEncoder`].
///
/// Input:  `hash_encoding(pos)` ∥ `direction_encoding(dir)`
/// → hidden layer (64 ReLU) → output `(density, rgb)`.
pub struct InstantNgpMlp {
    w0: Vec<Vec<f64>>, // [hidden][in_dim]
    b0: Vec<f64>,
    w1: Vec<Vec<f64>>, // [1 + 3][hidden]
    b1: Vec<f64>,
    encoder: HashEncoder,
    n_freq_dir: usize,
    hidden_dim: usize,
}

impl InstantNgpMlp {
    /// Build and initialise an `InstantNgpMlp`.
    ///
    /// # Arguments
    ///
    /// * `encoder`    – pre-constructed [`HashEncoder`].
    /// * `n_freq_dir` – positional-encoding bands for view direction.
    /// * `hidden_dim` – width of the hidden layer (typically 64).
    /// * `seed`       – PRNG seed.
    pub fn new(encoder: HashEncoder, n_freq_dir: usize, hidden_dim: usize, seed: u64) -> Self {
        use super::positional_encoding::encoding_dim;

        let dir_dim = encoding_dim(n_freq_dir);
        let in_dim = encoder.n_output_features() + dir_dim;
        let out_dim = 4; // density (1) + rgb (3)

        let mut rng = Lcg::new(seed ^ 0x1234_5678);

        let scale0 = (2.0 / in_dim as f64).sqrt();
        let w0: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| rng.next_f64() * 2.0 * scale0 - scale0)
                    .collect()
            })
            .collect();
        let b0 = vec![0.0_f64; hidden_dim];

        let scale1 = (2.0 / hidden_dim as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.next_f64() * 2.0 * scale1 - scale1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0_f64; out_dim];

        Self {
            w0,
            b0,
            w1,
            b1,
            encoder,
            n_freq_dir,
            hidden_dim,
        }
    }

    /// Run a forward pass.
    ///
    /// Returns `(density, rgb)` where `density ≥ 0` and `rgb ∈ [0,1]³`.
    ///
    /// # Arguments
    ///
    /// * `pos` – 3-D world position.
    /// * `dir` – unit view direction.
    pub fn forward(&self, pos: &[f64; 3], dir: &[f64; 3]) -> (f64, [f64; 3]) {
        use super::positional_encoding::encode_direction;

        // Build input
        let hash_feat = self.encoder.encode(pos);
        let dir_feat = encode_direction(dir, self.n_freq_dir);

        let mut inp: Vec<f64> = Vec::with_capacity(hash_feat.len() + dir_feat.len());
        inp.extend_from_slice(&hash_feat);
        inp.extend_from_slice(&dir_feat);

        // Hidden layer (ReLU)
        let h: Vec<f64> = (0..self.hidden_dim)
            .map(|i| {
                let raw: f64 = self.b0[i]
                    + self.w0[i]
                        .iter()
                        .zip(inp.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>();
                raw.max(0.0)
            })
            .collect();

        // Output layer
        let raw_out: Vec<f64> = (0..4)
            .map(|i| {
                self.b1[i]
                    + self.w1[i]
                        .iter()
                        .zip(h.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>()
            })
            .collect();

        let density = raw_out[0].max(0.0); // ReLU
        let rgb = [
            sigmoid(raw_out[1]),
            sigmoid(raw_out[2]),
            sigmoid(raw_out[3]),
        ];
        (density, rgb)
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_encoder() -> HashEncoder {
        HashEncoder::new(4, 2, 12, 16, 128, 42)
    }

    #[test]
    fn test_hash_function_deterministic() {
        // Same coords must always map to the same index.
        let table_size = 1 << 16;
        let h1 = hash_coords(3, -7, 11, table_size);
        let h2 = hash_coords(3, -7, 11, table_size);
        assert_eq!(h1, h2);

        // Different coords should produce different indices (with high probability).
        let h3 = hash_coords(3, -7, 12, table_size);
        let h4 = hash_coords(4, -7, 11, table_size);
        // Very unlikely to collide — just a sanity check.
        assert!(
            h1 != h3 || h1 != h4,
            "suspicious: three distinct coords all hash to {h1}"
        );
    }

    #[test]
    fn test_hash_table_lookup_shape() {
        let enc = build_encoder();
        let feat = enc.lookup_features(&[0.25, 0.5, 0.75], 0);
        assert_eq!(feat.len(), enc.n_features_per_level);
    }

    #[test]
    fn test_hash_encoder_output_dim() {
        let enc = build_encoder();
        let out = enc.encode(&[0.1, 0.2, 0.3]);
        assert_eq!(out.len(), enc.n_output_features());
        assert_eq!(
            enc.n_output_features(),
            enc.n_levels * enc.n_features_per_level
        );
    }

    #[test]
    fn test_trilinear_interpolation_corners() {
        // When the query point is exactly at a grid corner, the trilinear
        // weights for the other 7 corners should be 0 and the result
        // should equal that corner's hash-table features.

        let enc = build_encoder();
        let level = 0;
        let res = enc.level_resolution(level);

        // Position exactly on a grid corner: (0, 0, 0) in grid space → pos = (0, 0, 0)
        let pos_corner = [0.0_f64, 0.0, 0.0];

        // Manually compute what the table holds at (0,0,0)
        let expected_idx = hash_coords(0, 0, 0, enc.table_size);
        let expected_feat = enc.get_feature(level, expected_idx).to_vec();

        let result = enc.lookup_features(&pos_corner, level);
        for (i, (&r, &e)) in result.iter().zip(expected_feat.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-12,
                "feature[{i}] mismatch at corner: got {r}, expected {e}"
            );
        }

        // Silence unused warning for `res`
        let _ = res;
    }

    #[test]
    fn test_instant_ngp_mlp_forward() {
        let encoder = HashEncoder::new(4, 2, 12, 16, 128, 77);
        let mlp = InstantNgpMlp::new(encoder, 2, 32, 99);
        let pos = [0.3, 0.4, 0.5];
        let dir = [0.0, 1.0, 0.0];
        let (density, rgb) = mlp.forward(&pos, &dir);

        assert!(density >= 0.0, "density must be >= 0, got {density}");
        for (c, &ch) in rgb.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&ch),
                "rgb[{c}] = {ch} must be in [0,1]"
            );
        }
    }
}
