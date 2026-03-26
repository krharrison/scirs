//! KV page primitives: individual pages and the pre-allocated page pool.
//!
//! A [`KvPage`] holds key and value tensors for up to `block_size` tokens.
//! Shape per page: `[block_size, num_heads, head_dim]`.
//!
//! The [`KvPagePool`] pre-allocates a fixed number of pages and manages them
//! via a free list. Allocation is O(1); eviction policy is LRU-tracked.

use std::collections::VecDeque;

use scirs2_core::ndarray::{s, Array3, ArrayView2};
use scirs2_core::numeric::Float;

use super::{InferenceError, InferenceResult};

/// Opaque identifier for a page within a [`KvPagePool`].
pub type PageId = u32;

// ─────────────────────────────────────────────
// KvPageConfig
// ─────────────────────────────────────────────

/// Configuration for a single KV page.
///
/// All pages in a pool share the same configuration.
#[derive(Debug, Clone)]
pub struct KvPageConfig {
    /// Number of token slots per page (vLLM calls this the "block size").
    pub block_size: usize,
    /// Number of KV heads (may differ from query heads in GQA/MQA).
    pub num_heads: usize,
    /// Dimensionality per head.
    pub head_dim: usize,
    /// Bytes per scalar element (e.g. 4 for f32, 2 for bf16).
    pub dtype_bytes: usize,
}

impl Default for KvPageConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_heads: 8,
            head_dim: 64,
            dtype_bytes: 4,
        }
    }
}

// ─────────────────────────────────────────────
// KvPage
// ─────────────────────────────────────────────

/// A single page of key-value cache.
///
/// Holds pre-allocated `keys` and `values` arrays of shape
/// `[block_size, num_heads, head_dim]`.  Only the first `len` slots are
/// considered live; the rest contain unspecified data.
pub struct KvPage<F> {
    /// Key tensor — shape `[block_size, num_heads, head_dim]`.
    keys: Array3<F>,
    /// Value tensor — shape `[block_size, num_heads, head_dim]`.
    values: Array3<F>,
    /// Number of slots currently written (0..=block_size).
    len: usize,
    /// Block size (copied from config for quick access).
    block_size: usize,
    /// Number of heads (copied from config).
    num_heads: usize,
    /// Head dimension (copied from config).
    head_dim: usize,
}

impl<F: Float + Default + Clone> KvPage<F> {
    /// Allocate a new, empty page using the given configuration.
    pub fn new(config: &KvPageConfig) -> Self {
        let shape = (config.block_size, config.num_heads, config.head_dim);
        Self {
            keys: Array3::default(shape),
            values: Array3::default(shape),
            len: 0,
            block_size: config.block_size,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        }
    }

    /// Write a single token's key and value into slot `pos`.
    ///
    /// `key` and `value` must each have shape `[num_heads, head_dim]`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::SlotOutOfRange`] if `pos >= block_size`.
    /// Returns [`InferenceError::KvShapeMismatch`] if key/value shapes differ
    /// from the page configuration.
    pub fn write_kv(
        &mut self,
        pos: usize,
        key: ArrayView2<F>,
        value: ArrayView2<F>,
    ) -> InferenceResult<()> {
        if pos >= self.block_size {
            return Err(InferenceError::SlotOutOfRange {
                slot: pos,
                capacity: self.block_size,
            });
        }
        let (kh, kd) = (key.shape()[0], key.shape()[1]);
        if kh != self.num_heads || kd != self.head_dim {
            return Err(InferenceError::KvShapeMismatch {
                expected_heads: self.num_heads,
                expected_dim: self.head_dim,
                got_heads: kh,
                got_dim: kd,
            });
        }
        let (vh, vd) = (value.shape()[0], value.shape()[1]);
        if vh != self.num_heads || vd != self.head_dim {
            return Err(InferenceError::KvShapeMismatch {
                expected_heads: self.num_heads,
                expected_dim: self.head_dim,
                got_heads: vh,
                got_dim: vd,
            });
        }

        self.keys.slice_mut(s![pos, .., ..]).assign(&key);
        self.values.slice_mut(s![pos, .., ..]).assign(&value);

        if pos >= self.len {
            self.len = pos + 1;
        }
        Ok(())
    }

    /// Read a single token's key and value from slot `pos`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::SlotOutOfRange`] if `pos >= block_size`.
    pub fn read_kv(&self, pos: usize) -> InferenceResult<(ArrayView2<F>, ArrayView2<F>)> {
        if pos >= self.block_size {
            return Err(InferenceError::SlotOutOfRange {
                slot: pos,
                capacity: self.block_size,
            });
        }
        let k = self.keys.slice(s![pos, .., ..]);
        let v = self.values.slice(s![pos, .., ..]);
        Ok((k, v))
    }

    /// Whether all `block_size` slots have been written.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len >= self.block_size
    }

    /// Number of written slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the page has no written slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of token slots this page can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.block_size
    }

    /// Reset the page to empty (does not zero memory; just resets `len`).
    pub(crate) fn reset(&mut self) {
        self.len = 0;
    }
}

// ─────────────────────────────────────────────
// KvPagePool
// ─────────────────────────────────────────────

/// Pre-allocated pool of [`KvPage`]s managed via a free list.
///
/// ## Memory layout
///
/// All pages are allocated upfront in `pool: Vec<KvPage<F>>`.  The free list
/// (`free_list: VecDeque<PageId>`) tracks available page IDs.  An LRU order
/// deque tracks recency for future eviction support.
///
/// ## Example
///
/// ```rust
/// use scirs2_neural::inference::{KvPageConfig, KvPagePool};
///
/// let cfg = KvPageConfig { block_size: 16, num_heads: 8, head_dim: 64, dtype_bytes: 4 };
/// let mut pool = KvPagePool::<f32>::new(64, cfg);
/// assert_eq!(pool.free_count(), 64);
///
/// let pid = pool.alloc_page().expect("alloc failed");
/// assert_eq!(pool.free_count(), 63);
/// pool.free_page(pid).expect("free failed");
/// assert_eq!(pool.free_count(), 64);
/// ```
pub struct KvPagePool<F> {
    /// All pages, indexed by `PageId`.
    pool: Vec<KvPage<F>>,
    /// IDs of pages available for allocation.
    free_list: VecDeque<PageId>,
    /// Recency order for LRU eviction (most-recent at back).
    lru_order: VecDeque<PageId>,
    /// Shared configuration for every page.
    config: KvPageConfig,
}

impl<F: Float + Default + Clone> KvPagePool<F> {
    /// Create a new pool with `num_pages` pre-allocated pages.
    ///
    /// # Panics
    ///
    /// Panics if `num_pages` is 0 (meaningless pool).
    pub fn new(num_pages: usize, config: KvPageConfig) -> Self {
        assert!(num_pages > 0, "KvPagePool must have at least one page");
        let pool: Vec<KvPage<F>> = (0..num_pages).map(|_| KvPage::new(&config)).collect();
        let free_list: VecDeque<PageId> = (0..num_pages as PageId).collect();
        Self {
            pool,
            free_list,
            lru_order: VecDeque::new(),
            config,
        }
    }

    /// Allocate a free page, returning its [`PageId`].
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::Oom`] when no free pages remain.
    pub fn alloc_page(&mut self) -> InferenceResult<PageId> {
        let id = self.free_list.pop_front().ok_or(InferenceError::Oom)?;
        // Track in LRU order (newly allocated → most recent).
        self.lru_order.push_back(id);
        Ok(id)
    }

    /// Return a page to the free list.
    ///
    /// The page's `len` is reset to 0 so it can be reused cleanly.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::PageOutOfBounds`] if `id` is invalid.
    /// Returns [`InferenceError::DoubleFree`] if the page is already free.
    pub fn free_page(&mut self, id: PageId) -> InferenceResult<()> {
        let n = self.pool.len();
        let page = self
            .pool
            .get_mut(id as usize)
            .ok_or(InferenceError::PageOutOfBounds(id, n))?;

        // Detect double-free: a page on the free_list has len == 0 *and*
        // appears in free_list.  We track via the free_list membership.
        if self.free_list.contains(&id) {
            return Err(InferenceError::DoubleFree(id));
        }

        page.reset();
        self.free_list.push_back(id);
        // Remove from LRU order.
        self.lru_order.retain(|&x| x != id);
        Ok(())
    }

    /// Immutable reference to a page.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::PageOutOfBounds`] if `id` is out of range.
    pub fn get_page(&self, id: PageId) -> InferenceResult<&KvPage<F>> {
        let n = self.pool.len();
        self.pool
            .get(id as usize)
            .ok_or(InferenceError::PageOutOfBounds(id, n))
    }

    /// Mutable reference to a page.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::PageOutOfBounds`] if `id` is out of range.
    pub fn get_page_mut(&mut self, id: PageId) -> InferenceResult<&mut KvPage<F>> {
        let n = self.pool.len();
        self.pool
            .get_mut(id as usize)
            .ok_or(InferenceError::PageOutOfBounds(id, n))
    }

    /// Number of pages currently on the free list.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of pages in the pool (free + in-use).
    #[inline]
    pub fn total_count(&self) -> usize {
        self.pool.len()
    }

    /// Total memory occupied by all key+value tensors in bytes.
    ///
    /// Formula: `total_pages * 2 * block_size * num_heads * head_dim * dtype_bytes`
    pub fn memory_bytes(&self) -> usize {
        let cfg = &self.config;
        self.pool.len()
            * 2  // keys + values
            * cfg.block_size
            * cfg.num_heads
            * cfg.head_dim
            * cfg.dtype_bytes
    }

    /// Read-only access to the pool configuration.
    pub fn config(&self) -> &KvPageConfig {
        &self.config
    }

    /// Return the LRU-ordered page IDs (least-recently-used first).
    ///
    /// This is useful for cache eviction when the pool is full.
    pub fn lru_candidates(&self) -> impl Iterator<Item = PageId> + '_ {
        self.lru_order.iter().copied()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn default_config() -> KvPageConfig {
        KvPageConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 8,
            dtype_bytes: 4,
        }
    }

    // ── KvPage ────────────────────────────────

    #[test]
    fn test_page_write_read_roundtrip() {
        let cfg = default_config();
        let mut page = KvPage::<f32>::new(&cfg);

        let key = Array2::<f32>::from_elem((2, 8), 1.5_f32);
        let val = Array2::<f32>::from_elem((2, 8), 2.5_f32);

        page.write_kv(0, key.view(), val.view())
            .expect("write_kv failed");

        let (k_out, v_out) = page.read_kv(0).expect("read_kv failed");
        assert!(k_out.iter().all(|&x| (x - 1.5_f32).abs() < 1e-6));
        assert!(v_out.iter().all(|&x| (x - 2.5_f32).abs() < 1e-6));
    }

    #[test]
    fn test_page_write_multiple_slots() {
        let cfg = default_config();
        let mut page = KvPage::<f32>::new(&cfg);

        for slot in 0..cfg.block_size {
            let fill = slot as f32;
            let k = Array2::<f32>::from_elem((2, 8), fill);
            let v = Array2::<f32>::from_elem((2, 8), fill + 0.5);
            page.write_kv(slot, k.view(), v.view())
                .expect("write_kv failed");
        }
        assert!(page.is_full());
        assert_eq!(page.len(), cfg.block_size);

        for slot in 0..cfg.block_size {
            let (k, v) = page.read_kv(slot).expect("read_kv");
            let expected_k = slot as f32;
            let expected_v = slot as f32 + 0.5;
            assert!(k.iter().all(|&x| (x - expected_k).abs() < 1e-6));
            assert!(v.iter().all(|&x| (x - expected_v).abs() < 1e-6));
        }
    }

    #[test]
    fn test_page_slot_out_of_range() {
        let cfg = default_config();
        let mut page = KvPage::<f32>::new(&cfg);
        let k = Array2::<f32>::zeros((2, 8));
        let v = Array2::<f32>::zeros((2, 8));

        let err = page
            .write_kv(cfg.block_size, k.view(), v.view())
            .expect_err("should error");
        assert!(
            matches!(err, InferenceError::SlotOutOfRange { slot, .. } if slot == cfg.block_size)
        );
    }

    #[test]
    fn test_page_shape_mismatch() {
        let cfg = default_config();
        let mut page = KvPage::<f32>::new(&cfg);
        // Wrong head count
        let k = Array2::<f32>::zeros((3, 8));
        let v = Array2::<f32>::zeros((2, 8));
        let err = page
            .write_kv(0, k.view(), v.view())
            .expect_err("should error");
        assert!(matches!(err, InferenceError::KvShapeMismatch { .. }));
    }

    #[test]
    fn test_page_read_out_of_range() {
        let cfg = default_config();
        let page = KvPage::<f32>::new(&cfg);
        let err = page.read_kv(cfg.block_size).expect_err("should error");
        assert!(matches!(err, InferenceError::SlotOutOfRange { .. }));
    }

    #[test]
    fn test_page_is_empty_initially() {
        let cfg = default_config();
        let page = KvPage::<f32>::new(&cfg);
        assert!(page.is_empty());
        assert_eq!(page.len(), 0);
        assert_eq!(page.capacity(), cfg.block_size);
    }

    // ── KvPagePool ────────────────────────────

    #[test]
    fn test_pool_alloc_and_free() {
        let cfg = default_config();
        let mut pool = KvPagePool::<f32>::new(4, cfg);

        assert_eq!(pool.total_count(), 4);
        assert_eq!(pool.free_count(), 4);

        let p0 = pool.alloc_page().expect("alloc");
        let p1 = pool.alloc_page().expect("alloc");
        assert_eq!(pool.free_count(), 2);

        pool.free_page(p0).expect("free");
        pool.free_page(p1).expect("free");
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn test_pool_oom_when_exhausted() {
        let cfg = default_config();
        let mut pool = KvPagePool::<f32>::new(2, cfg);

        pool.alloc_page().expect("alloc 0");
        pool.alloc_page().expect("alloc 1");

        let err = pool.alloc_page().expect_err("should OOM");
        assert!(matches!(err, InferenceError::Oom));
    }

    #[test]
    fn test_pool_double_free_detected() {
        let cfg = default_config();
        let mut pool = KvPagePool::<f32>::new(4, cfg);
        let id = pool.alloc_page().expect("alloc");
        pool.free_page(id).expect("first free");
        let err = pool.free_page(id).expect_err("double-free should error");
        assert!(matches!(err, InferenceError::DoubleFree(_)));
    }

    #[test]
    fn test_pool_memory_bytes() {
        let cfg = KvPageConfig {
            block_size: 16,
            num_heads: 8,
            head_dim: 64,
            dtype_bytes: 4,
        };
        let pool = KvPagePool::<f32>::new(10, cfg);
        // 10 pages * 2 tensors * 16 * 8 * 64 * 4 bytes
        let expected = 10 * 2 * 16 * 8 * 64 * 4;
        assert_eq!(pool.memory_bytes(), expected);
    }

    #[test]
    fn test_pool_get_page_write_then_read() {
        let cfg = default_config();
        let mut pool = KvPagePool::<f32>::new(4, cfg);
        let id = pool.alloc_page().expect("alloc");

        {
            let page = pool.get_page_mut(id).expect("get_mut");
            let k = Array2::<f32>::from_elem((2, 8), 7.0);
            let v = Array2::<f32>::from_elem((2, 8), 8.0);
            page.write_kv(1, k.view(), v.view()).expect("write");
        }

        let page = pool.get_page(id).expect("get");
        let (k, v) = page.read_kv(1).expect("read");
        assert!(k.iter().all(|&x| (x - 7.0_f32).abs() < 1e-6));
        assert!(v.iter().all(|&x| (x - 8.0_f32).abs() < 1e-6));
    }

    #[test]
    fn test_pool_page_reset_on_free() {
        let cfg = default_config();
        let mut pool = KvPagePool::<f32>::new(4, cfg);
        let id = pool.alloc_page().expect("alloc");

        {
            let page = pool.get_page_mut(id).expect("get_mut");
            let k = Array2::<f32>::from_elem((2, 8), 3.0);
            let v = Array2::<f32>::from_elem((2, 8), 4.0);
            page.write_kv(0, k.view(), v.view()).expect("write");
            assert_eq!(page.len(), 1);
        }

        pool.free_page(id).expect("free");
        let id2 = pool.alloc_page().expect("re-alloc");
        let page = pool.get_page(id2).expect("get");
        assert_eq!(page.len(), 0, "page should be reset after free");
    }
}
