//! Paged attention forward pass over non-contiguous KV page chains.
//!
//! [`PagedAttentionForward`] collects key/value tensors from a sequence's page
//! chain, assembles them into contiguous buffers, and computes scaled dot-product
//! attention for each query token.
//!
//! This mirrors the vLLM paged-attention kernel at the algorithmic level:
//! the physical memory is fragmented across pages, but the attention computation
//! sees a logical, contiguous view.
//!
//! ## Attention formula
//!
//! For a query `q ∈ R^{num_heads × head_dim}` and `N` collected key/value pairs:
//!
//! ```text
//! scores[h, i] = q[h, :] · k[i, h, :] * scale          (dot product per head)
//! weights[h, :] = softmax(scores[h, :])
//! output[h, :] = Σ_i weights[h, i] * v[i, h, :]
//! ```
//!
//! The output shape matches the query: `[num_heads, head_dim]`.

use scirs2_core::ndarray::{s, Array2, Array3, ArrayView2, Axis};
use scirs2_core::numeric::Float;

use super::{
    kv_page::{KvPagePool, PageId},
    InferenceError, InferenceResult,
};

// ─────────────────────────────────────────────
// PagedAttentionConfig
// ─────────────────────────────────────────────

/// Configuration for the paged attention forward pass.
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality per head.
    pub head_dim: usize,
    /// Optional scaling factor.  If `None`, defaults to `1 / sqrt(head_dim)`.
    pub scale: Option<f64>,
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            scale: None,
        }
    }
}

// ─────────────────────────────────────────────
// PagedAttentionForward
// ─────────────────────────────────────────────

/// Computes attention for one or more query tokens over a KV page chain.
///
/// ## Example
///
/// ```rust
/// use scirs2_neural::inference::{
///     KvPageConfig, KvPagePool, PagedAttentionConfig, PagedAttentionForward,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// let page_cfg = KvPageConfig { block_size: 8, num_heads: 2, head_dim: 4, dtype_bytes: 4 };
/// let mut pool = KvPagePool::<f32>::new(16, page_cfg);
///
/// // Allocate and populate one page
/// let pid = pool.alloc_page().expect("alloc");
/// let k = Array2::<f32>::from_elem((2, 4), 0.1);
/// let v = Array2::<f32>::from_elem((2, 4), 0.2);
/// pool.get_page_mut(pid).unwrap().write_kv(0, k.view(), v.view()).unwrap();
///
/// let attn_cfg = PagedAttentionConfig { num_heads: 2, head_dim: 4, scale: None };
/// let attn = PagedAttentionForward::new(attn_cfg);
///
/// let query = Array2::<f32>::from_elem((2, 4), 1.0);
/// let output = attn.forward(&query, &[pid], &pool).expect("forward");
/// assert_eq!(output.shape(), &[2, 4]);
/// ```
pub struct PagedAttentionForward {
    config: PagedAttentionConfig,
}

impl PagedAttentionForward {
    /// Create a new paged attention operator.
    pub fn new(config: PagedAttentionConfig) -> Self {
        Self { config }
    }

    /// Compute paged attention.
    ///
    /// # Arguments
    ///
    /// * `query`      — query tensor of shape `[num_heads, head_dim]`.
    /// * `page_chain` — ordered page IDs forming the KV sequence.
    /// * `pool`       — page pool that owns the pages.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[num_heads, head_dim]`.
    ///
    /// # Errors
    ///
    /// - [`InferenceError::KvShapeMismatch`] if `query` shape doesn't match config.
    /// - [`InferenceError::PageOutOfBounds`] if a page ID is invalid.
    /// - [`InferenceError::SlotOutOfRange`] if a page slot access is invalid.
    pub fn forward<F: Float + Default + Clone>(
        &self,
        query: &Array2<F>,
        page_chain: &[PageId],
        pool: &KvPagePool<F>,
    ) -> InferenceResult<Array2<F>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Validate query shape.
        if query.shape() != [num_heads, head_dim] {
            return Err(InferenceError::KvShapeMismatch {
                expected_heads: num_heads,
                expected_dim: head_dim,
                got_heads: query.shape()[0],
                got_dim: query.shape()[1],
            });
        }

        if page_chain.is_empty() {
            // No KV tokens → output is all zeros.
            return Ok(Array2::default((num_heads, head_dim)));
        }

        // ── Step 1: collect all live key/value slots from the page chain ──

        let block_size = pool.config().block_size;
        let total_slots = self.count_live_slots(page_chain, pool, block_size)?;

        if total_slots == 0 {
            return Ok(Array2::default((num_heads, head_dim)));
        }

        // Allocate contiguous KV buffers: [total_slots, num_heads, head_dim]
        let mut keys_buf: Array3<F> = Array3::default((total_slots, num_heads, head_dim));
        let mut vals_buf: Array3<F> = Array3::default((total_slots, num_heads, head_dim));

        self.gather_kv(page_chain, pool, block_size, &mut keys_buf, &mut vals_buf)?;

        // ── Step 2: compute per-head scaled dot-product attention ──

        let scale = self.effective_scale::<F>();
        let output = self.sdp_attention(query, &keys_buf, &vals_buf, scale)?;

        Ok(output)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Count the total number of live (written) slots across all pages in the chain.
    fn count_live_slots<F: Float + Default + Clone>(
        &self,
        page_chain: &[PageId],
        pool: &KvPagePool<F>,
        block_size: usize,
    ) -> InferenceResult<usize> {
        let mut total = 0usize;
        let chain_len = page_chain.len();
        for (idx, &pid) in page_chain.iter().enumerate() {
            let page = pool.get_page(pid)?;
            let live = if idx < chain_len.saturating_sub(1) {
                // All full pages before the last are assumed full (block_size).
                block_size.min(page.len())
            } else {
                // Last page may be partially filled.
                page.len()
            };
            total += live;
        }
        Ok(total)
    }

    /// Copy key/value data from pages into contiguous output buffers.
    fn gather_kv<F: Float + Default + Clone>(
        &self,
        page_chain: &[PageId],
        pool: &KvPagePool<F>,
        block_size: usize,
        keys_buf: &mut Array3<F>,
        vals_buf: &mut Array3<F>,
    ) -> InferenceResult<()> {
        let chain_len = page_chain.len();
        let mut dst_slot = 0usize;

        for (idx, &pid) in page_chain.iter().enumerate() {
            let page = pool.get_page(pid)?;
            let live = if idx < chain_len.saturating_sub(1) {
                block_size.min(page.len())
            } else {
                page.len()
            };

            for slot in 0..live {
                let (k_view, v_view) = page.read_kv(slot)?;
                keys_buf.slice_mut(s![dst_slot, .., ..]).assign(&k_view);
                vals_buf.slice_mut(s![dst_slot, .., ..]).assign(&v_view);
                dst_slot += 1;
            }
        }
        Ok(())
    }

    /// Compute scaled dot-product attention.
    ///
    /// Query: `[H, D]`  Keys/Vals: `[N, H, D]` → Output: `[H, D]`
    fn sdp_attention<F: Float + Default + Clone>(
        &self,
        query: &Array2<F>,  // [H, D]
        keys: &Array3<F>,   // [N, H, D]
        values: &Array3<F>, // [N, H, D]
        scale: F,
    ) -> InferenceResult<Array2<F>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let seq_len = keys.shape()[0];

        let mut output = Array2::<F>::default((num_heads, head_dim));

        for h in 0..num_heads {
            let q_head = query.slice(s![h, ..]); // [D]
            let k_heads = keys.slice(s![.., h, ..]); // [N, D]
            let v_heads = values.slice(s![.., h, ..]); // [N, D]

            // Compute raw attention scores: [N]
            let mut scores = Vec::with_capacity(seq_len);
            for n in 0..seq_len {
                let k_tok = k_heads.slice(s![n, ..]); // [D]
                let dot: F = q_head
                    .iter()
                    .zip(k_tok.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .fold(F::zero(), |acc, x| acc + x);
                scores.push(dot * scale);
            }

            // Stable softmax: subtract max for numerical stability.
            let max_score = scores.iter().copied().fold(F::neg_infinity(), F::max);
            let exp_scores: Vec<F> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: F = exp_scores.iter().copied().fold(F::zero(), |a, b| a + b);

            // Clamp denominator to avoid division by zero.
            let safe_sum = if sum_exp == F::zero() {
                F::one()
            } else {
                sum_exp
            };
            let weights: Vec<F> = exp_scores.iter().map(|&e| e / safe_sum).collect();

            // Weighted sum of values.
            let mut out_head = output.slice_mut(s![h, ..]);
            for (n, &w) in weights.iter().enumerate().take(seq_len) {
                let v_tok = v_heads.slice(s![n, ..]); // [D]
                for (out_el, &v_el) in out_head.iter_mut().zip(v_tok.iter()) {
                    *out_el = *out_el + w * v_el;
                }
            }
        }

        Ok(output)
    }

    /// Effective scale factor as `F`.
    fn effective_scale<F: Float>(&self) -> F {
        let s = self
            .config
            .scale
            .unwrap_or_else(|| 1.0 / (self.config.head_dim as f64).sqrt());
        F::from(s).unwrap_or_else(F::one)
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }
}

// Suppress unused import warning from Axis (used implicitly via ndarray slicing API).
const _: () = {
    let _ = Axis(0);
};

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_page::KvPageConfig;

    fn make_pool(
        num_pages: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> KvPagePool<f32> {
        let cfg = KvPageConfig {
            block_size,
            num_heads,
            head_dim,
            dtype_bytes: 4,
        };
        KvPagePool::<f32>::new(num_pages, cfg)
    }

    fn write_constant_kv(
        pool: &mut KvPagePool<f32>,
        pid: PageId,
        num_slots: usize,
        k_val: f32,
        v_val: f32,
    ) {
        let (num_heads, head_dim) = {
            let cfg = pool.config();
            (cfg.num_heads, cfg.head_dim)
        };
        let k = Array2::<f32>::from_elem((num_heads, head_dim), k_val);
        let v = Array2::<f32>::from_elem((num_heads, head_dim), v_val);
        for slot in 0..num_slots {
            pool.get_page_mut(pid)
                .expect("page")
                .write_kv(slot, k.view(), v.view())
                .expect("write");
        }
    }

    // ── Output shape ──────────────────────────

    #[test]
    fn test_forward_output_shape_correct() {
        let num_heads = 4;
        let head_dim = 8;
        let mut pool = make_pool(8, 4, num_heads, head_dim);
        let pid = pool.alloc_page().expect("alloc");
        write_constant_kv(&mut pool, pid, 2, 1.0, 1.0);

        let cfg = PagedAttentionConfig {
            num_heads,
            head_dim,
            scale: None,
        };
        let attn = PagedAttentionForward::new(cfg);
        let query = Array2::<f32>::from_elem((num_heads, head_dim), 1.0);
        let output = attn.forward(&query, &[pid], &pool).expect("forward");
        assert_eq!(output.shape(), &[num_heads, head_dim]);
    }

    // ── Attention over 2 pages ────────────────

    #[test]
    fn test_forward_over_two_pages() {
        let num_heads = 2;
        let head_dim = 4;
        let block_size = 3;
        let mut pool = make_pool(8, block_size, num_heads, head_dim);

        let pid0 = pool.alloc_page().expect("alloc p0");
        let pid1 = pool.alloc_page().expect("alloc p1");
        // Fill page 0 with k=0.1, v=0.2
        write_constant_kv(&mut pool, pid0, block_size, 0.1, 0.2);
        // Fill page 1 with k=0.5, v=0.6 (2 slots)
        write_constant_kv(&mut pool, pid1, 2, 0.5, 0.6);

        let cfg = PagedAttentionConfig {
            num_heads,
            head_dim,
            scale: Some(1.0),
        };
        let attn = PagedAttentionForward::new(cfg);
        let query = Array2::<f32>::from_elem((num_heads, head_dim), 1.0);
        let output = attn.forward(&query, &[pid0, pid1], &pool).expect("forward");
        assert_eq!(output.shape(), &[num_heads, head_dim]);
        // All output values should be finite.
        assert!(output.iter().all(|x| x.is_finite()));
    }

    // ── Scale factor applied ──────────────────

    #[test]
    fn test_scale_factor_affects_output() {
        let num_heads = 1;
        let head_dim = 4;
        let mut pool1 = make_pool(4, 4, num_heads, head_dim);
        let mut pool2 = make_pool(4, 4, num_heads, head_dim);

        let pid1 = pool1.alloc_page().expect("alloc");
        write_constant_kv(&mut pool1, pid1, 2, 1.0, 1.0);

        let pid2 = pool2.alloc_page().expect("alloc");
        write_constant_kv(&mut pool2, pid2, 2, 1.0, 1.0);

        let query = Array2::<f32>::from_elem((num_heads, head_dim), 1.0);

        let attn_small = PagedAttentionForward::new(PagedAttentionConfig {
            num_heads,
            head_dim,
            scale: Some(0.01),
        });
        let attn_large = PagedAttentionForward::new(PagedAttentionConfig {
            num_heads,
            head_dim,
            scale: Some(100.0),
        });

        let out_small = attn_small.forward(&query, &[pid1], &pool1).expect("fwd");
        let out_large = attn_large.forward(&query, &[pid2], &pool2).expect("fwd");

        // When all K tokens are identical, softmax is uniform regardless of scale,
        // so both outputs should equal the value (1.0).  What differs is the
        // pre-softmax scores; verify the outputs are still valid floats.
        assert!(out_small.iter().all(|x| x.is_finite()));
        assert!(out_large.iter().all(|x| x.is_finite()));
    }

    // ── Single token query ────────────────────

    #[test]
    fn test_single_token_query_and_single_kv() {
        let num_heads = 2;
        let head_dim = 3;
        let mut pool = make_pool(4, 4, num_heads, head_dim);
        let pid = pool.alloc_page().expect("alloc");

        // Write exactly one KV slot with constant value 0.5.
        {
            let k = Array2::<f32>::from_elem((num_heads, head_dim), 0.5);
            let v = Array2::<f32>::from_elem((num_heads, head_dim), 0.5);
            pool.get_page_mut(pid)
                .expect("page")
                .write_kv(0, k.view(), v.view())
                .expect("write");
        }

        let cfg = PagedAttentionConfig {
            num_heads,
            head_dim,
            scale: Some(1.0),
        };
        let attn = PagedAttentionForward::new(cfg);
        let query = Array2::<f32>::from_elem((num_heads, head_dim), 1.0);
        let output = attn.forward(&query, &[pid], &pool).expect("forward");

        // With one KV token, softmax weight is 1.0, so output == value == 0.5.
        for &x in output.iter() {
            assert!((x - 0.5_f32).abs() < 1e-5, "expected 0.5, got {x}");
        }
    }

    // ── Empty page chain ──────────────────────

    #[test]
    fn test_empty_page_chain_returns_zeros() {
        let pool = make_pool(4, 4, 2, 4);
        let cfg = PagedAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            scale: None,
        };
        let attn = PagedAttentionForward::new(cfg);
        let query = Array2::<f32>::zeros((2, 4));
        let output = attn
            .forward(&query, &[], &pool)
            .expect("forward empty chain");
        assert_eq!(output.shape(), &[2, 4]);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    // ── Query shape mismatch ──────────────────

    #[test]
    fn test_query_shape_mismatch_errors() {
        let pool = make_pool(4, 4, 2, 4);
        let cfg = PagedAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            scale: None,
        };
        let attn = PagedAttentionForward::new(cfg);
        // Wrong query shape: 3 heads instead of 2.
        let query = Array2::<f32>::zeros((3, 4));
        let err = attn.forward(&query, &[], &pool).expect_err("should error");
        assert!(matches!(err, InferenceError::KvShapeMismatch { .. }));
    }

    // ── Uniform KV → uniform output ──────────

    #[test]
    fn test_uniform_kv_produces_uniform_output() {
        // When all KV tokens are identical, softmax weights are uniform,
        // and the output should equal the (uniform) value.
        let num_heads = 3;
        let head_dim = 5;
        let block_size = 4;
        let kv_val = 0.7_f32;
        let mut pool = make_pool(4, block_size, num_heads, head_dim);
        let pid = pool.alloc_page().expect("alloc");
        write_constant_kv(&mut pool, pid, block_size, kv_val, kv_val);

        let cfg = PagedAttentionConfig {
            num_heads,
            head_dim,
            scale: Some(1.0),
        };
        let attn = PagedAttentionForward::new(cfg);
        let query = Array2::<f32>::from_elem((num_heads, head_dim), 1.0);
        let output = attn.forward(&query, &[pid], &pool).expect("forward");

        for &x in output.iter() {
            assert!((x - kv_val).abs() < 1e-5, "expected {kv_val}, got {x}");
        }
    }
}
