//! Block manager: tracks page chains per sequence and shared prefix cache.
//!
//! The [`BlockManager`] owns a [`KvPagePool`] and maps each active sequence
//! (identified by a [`SeqId`]) to an ordered list of [`PageId`]s.  It handles
//! per-sequence allocation, extension, and bulk deallocation.
//!
//! The [`SharedPrefixCache`] stores page chains that correspond to common token
//! prefixes (e.g. system prompts), allowing multiple sequences to read the same
//! cached KV entries without re-computation.

use std::collections::HashMap;

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::numeric::Float;

use super::{
    kv_page::{KvPagePool, PageId},
    InferenceError, InferenceResult,
};

/// Opaque identifier for a sequence (a single LLM generation request).
pub type SeqId = u64;

// ─────────────────────────────────────────────
// BlockManagerConfig
// ─────────────────────────────────────────────

/// Configuration for the [`BlockManager`].
#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    /// Maximum number of concurrently active sequences.
    pub max_sequences: usize,
    /// Maximum number of pages that can be assigned to a single sequence.
    pub max_pages_per_seq: usize,
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        Self {
            max_sequences: 64,
            max_pages_per_seq: 128,
        }
    }
}

// ─────────────────────────────────────────────
// BlockManager
// ─────────────────────────────────────────────

/// Manages page allocation across multiple concurrent sequences.
///
/// Each sequence is represented as a chain of [`PageId`]s.  Tokens are written
/// sequentially into pages; when a page is full a new one is appended.
///
/// ## Lifecycle of a sequence
///
/// 1. [`allocate_sequence`](BlockManager::allocate_sequence) — register a new
///    sequence ID and reserve its first page.
/// 2. [`write_token_kv`](BlockManager::write_token_kv) — write each new token's
///    key/value tensors.  Pages are extended automatically when full.
/// 3. [`free_sequence`](BlockManager::free_sequence) — return all pages to the
///    pool when the sequence is done.
pub struct BlockManager<F> {
    /// Backing page pool shared by all sequences.
    pool: KvPagePool<F>,
    /// Maps sequence ID → ordered list of page IDs.
    seq_pages: HashMap<SeqId, Vec<PageId>>,
    /// Configuration.
    config: BlockManagerConfig,
}

impl<F: Float + Default + Clone> BlockManager<F> {
    /// Create a new block manager wrapping the given pool.
    pub fn new(pool: KvPagePool<F>, config: BlockManagerConfig) -> Self {
        Self {
            pool,
            seq_pages: HashMap::new(),
            config,
        }
    }

    /// Register a new sequence and allocate its first page.
    ///
    /// # Errors
    ///
    /// - [`InferenceError::Oom`] — no free pages in pool.
    pub fn allocate_sequence(&mut self, seq_id: SeqId) -> InferenceResult<()> {
        let first_page = self.pool.alloc_page()?;
        self.seq_pages.insert(seq_id, vec![first_page]);
        Ok(())
    }

    /// Append a new page to the sequence's chain.
    ///
    /// Called automatically by [`write_token_kv`](BlockManager::write_token_kv)
    /// when the last page is full, but can also be called explicitly to
    /// pre-allocate capacity.
    ///
    /// # Errors
    ///
    /// - [`InferenceError::SequenceNotFound`] — `seq_id` is not registered.
    /// - [`InferenceError::MaxPagesExceeded`] — sequence already at the limit.
    /// - [`InferenceError::Oom`] — no free pages.
    pub fn extend_sequence(&mut self, seq_id: SeqId) -> InferenceResult<PageId> {
        let pages = self
            .seq_pages
            .get_mut(&seq_id)
            .ok_or(InferenceError::SequenceNotFound(seq_id))?;

        if pages.len() >= self.config.max_pages_per_seq {
            return Err(InferenceError::MaxPagesExceeded(seq_id));
        }

        let new_page = self.pool.alloc_page()?;
        pages.push(new_page);
        Ok(new_page)
    }

    /// Free all pages belonging to `seq_id` and remove it from tracking.
    ///
    /// # Errors
    ///
    /// - [`InferenceError::SequenceNotFound`] — `seq_id` is not registered.
    pub fn free_sequence(&mut self, seq_id: SeqId) -> InferenceResult<()> {
        let pages = self
            .seq_pages
            .remove(&seq_id)
            .ok_or(InferenceError::SequenceNotFound(seq_id))?;

        for pid in pages {
            self.pool.free_page(pid)?;
        }
        Ok(())
    }

    /// Return the ordered page chain for a sequence (immutable).
    ///
    /// # Errors
    ///
    /// - [`InferenceError::SequenceNotFound`] — `seq_id` is not registered.
    pub fn sequence_page_chain(&self, seq_id: SeqId) -> InferenceResult<&[PageId]> {
        self.seq_pages
            .get(&seq_id)
            .map(|v| v.as_slice())
            .ok_or(InferenceError::SequenceNotFound(seq_id))
    }

    /// Write a token's key/value tensors at the given absolute token position.
    ///
    /// The method resolves `token_pos` to the correct page and slot within that
    /// page.  If the resolved page does not yet exist (i.e. we need to extend)
    /// the sequence is extended automatically.
    ///
    /// `key` and `value` must each have shape `[num_heads, head_dim]`.
    ///
    /// # Errors
    ///
    /// - [`InferenceError::SequenceNotFound`] if `seq_id` is unknown.
    /// - [`InferenceError::Oom`] if a new page cannot be allocated.
    /// - [`InferenceError::KvShapeMismatch`] if tensor shapes are wrong.
    pub fn write_token_kv(
        &mut self,
        seq_id: SeqId,
        token_pos: usize,
        key: ArrayView2<F>,
        value: ArrayView2<F>,
    ) -> InferenceResult<()> {
        let block_size = self.pool.config().block_size;
        let page_idx = token_pos / block_size;
        let slot = token_pos % block_size;

        // Extend the chain until we have enough pages.
        loop {
            let chain_len = self
                .seq_pages
                .get(&seq_id)
                .ok_or(InferenceError::SequenceNotFound(seq_id))?
                .len();
            if chain_len > page_idx {
                break;
            }
            self.extend_sequence(seq_id)?;
        }

        // Resolve the page ID for this token position.
        let page_id = *self
            .seq_pages
            .get(&seq_id)
            .ok_or(InferenceError::SequenceNotFound(seq_id))?
            .get(page_idx)
            .ok_or(InferenceError::SequenceNotFound(seq_id))?;

        let page = self.pool.get_page_mut(page_id)?;
        page.write_kv(slot, key, value)
    }

    /// Number of currently active (registered) sequences.
    pub fn active_sequences(&self) -> usize {
        self.seq_pages.len()
    }

    /// Read-only access to the underlying page pool.
    pub fn pool(&self) -> &KvPagePool<F> {
        &self.pool
    }
}

// ─────────────────────────────────────────────
// SharedPrefixCache
// ─────────────────────────────────────────────

/// Cache that maps token-prefix hashes to pre-populated page chains.
///
/// Multiple sequences that share a common prefix (e.g. a system prompt) can
/// read from the same set of pages without re-computing or re-storing them.
///
/// ## Usage pattern
///
/// 1. Before scheduling a new request, call
///    [`compute_prefix_hash`](SharedPrefixCache::compute_prefix_hash) on its
///    token sequence.
/// 2. Call [`lookup`](SharedPrefixCache::lookup) to check for a cached chain.
/// 3. If found, copy those page IDs into the new sequence's chain as
///    read-only prefix pages.
/// 4. After a sequence completes, call [`insert`](SharedPrefixCache::insert)
///    to share its prefix pages for future requests.
#[derive(Debug, Default)]
pub struct SharedPrefixCache {
    /// Maps FNV-1a hash of token prefix → page chain.
    prefix_pages: HashMap<u64, Vec<PageId>>,
}

impl SharedPrefixCache {
    /// Create an empty prefix cache.
    pub fn new() -> Self {
        Self {
            prefix_pages: HashMap::new(),
        }
    }

    /// Look up a page chain by prefix hash.
    ///
    /// Returns `Some(&[PageId])` if the prefix is cached, `None` otherwise.
    pub fn lookup(&self, prefix_hash: u64) -> Option<&[PageId]> {
        self.prefix_pages.get(&prefix_hash).map(|v| v.as_slice())
    }

    /// Insert or overwrite a cached page chain for the given prefix hash.
    pub fn insert(&mut self, prefix_hash: u64, pages: Vec<PageId>) {
        self.prefix_pages.insert(prefix_hash, pages);
    }

    /// Remove a cached entry.  Returns the page chain if it existed.
    pub fn evict(&mut self, prefix_hash: u64) -> Option<Vec<PageId>> {
        self.prefix_pages.remove(&prefix_hash)
    }

    /// Number of cached prefixes.
    pub fn len(&self) -> usize {
        self.prefix_pages.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.prefix_pages.is_empty()
    }

    /// Compute the FNV-1a 64-bit hash of a token sequence.
    ///
    /// This deterministic hash allows prefix deduplication across requests.
    pub fn compute_prefix_hash(tokens: &[u32]) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;

        let mut hash = FNV_OFFSET;
        for &token in tokens {
            let bytes = token.to_le_bytes();
            for byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_page::KvPageConfig;
    use scirs2_core::ndarray::Array2;

    fn make_manager(num_pages: usize, block_size: usize) -> BlockManager<f32> {
        let cfg = KvPageConfig {
            block_size,
            num_heads: 2,
            head_dim: 4,
            dtype_bytes: 4,
        };
        let pool = KvPagePool::<f32>::new(num_pages, cfg);
        BlockManager::new(pool, BlockManagerConfig::default())
    }

    // ── Allocate / free sequence ──────────────

    #[test]
    fn test_allocate_and_free_sequence() {
        let mut mgr = make_manager(8, 4);

        mgr.allocate_sequence(1).expect("allocate");
        assert_eq!(mgr.active_sequences(), 1);
        assert_eq!(mgr.pool().free_count(), 7);

        mgr.free_sequence(1).expect("free");
        assert_eq!(mgr.active_sequences(), 0);
        assert_eq!(mgr.pool().free_count(), 8);
    }

    #[test]
    fn test_free_unknown_sequence_errors() {
        let mut mgr = make_manager(4, 4);
        let err = mgr.free_sequence(42).expect_err("should error");
        assert!(matches!(err, InferenceError::SequenceNotFound(42)));
    }

    // ── Extend sequence ───────────────────────

    #[test]
    fn test_extend_sequence_adds_page() {
        let mut mgr = make_manager(8, 4);
        mgr.allocate_sequence(1).expect("alloc");
        assert_eq!(mgr.sequence_page_chain(1).expect("chain").len(), 1);

        mgr.extend_sequence(1).expect("extend");
        assert_eq!(mgr.sequence_page_chain(1).expect("chain").len(), 2);
        assert_eq!(mgr.pool().free_count(), 6);
    }

    #[test]
    fn test_extend_unknown_sequence_errors() {
        let mut mgr = make_manager(4, 4);
        let err = mgr.extend_sequence(99).expect_err("should error");
        assert!(matches!(err, InferenceError::SequenceNotFound(99)));
    }

    // ── write_token_kv ────────────────────────

    #[test]
    fn test_write_token_kv_in_first_page() {
        let mut mgr = make_manager(8, 4);
        mgr.allocate_sequence(1).expect("alloc");

        let k = Array2::<f32>::from_elem((2, 4), 1.0);
        let v = Array2::<f32>::from_elem((2, 4), 2.0);
        mgr.write_token_kv(1, 0, k.view(), v.view())
            .expect("write token 0");

        let k2 = Array2::<f32>::from_elem((2, 4), 3.0);
        let v2 = Array2::<f32>::from_elem((2, 4), 4.0);
        mgr.write_token_kv(1, 3, k2.view(), v2.view())
            .expect("write token 3");

        // Verify via page chain
        let chain = mgr.sequence_page_chain(1).expect("chain");
        assert_eq!(chain.len(), 1, "should fit in single page");
    }

    #[test]
    fn test_write_token_kv_auto_extends_page() {
        // block_size=2, so tokens 0-1 in page 0, tokens 2-3 in page 1
        let mut mgr = make_manager(8, 2);
        mgr.allocate_sequence(1).expect("alloc");

        for pos in 0..4_usize {
            let k = Array2::<f32>::from_elem((2, 4), pos as f32);
            let v = Array2::<f32>::from_elem((2, 4), pos as f32 + 0.1);
            mgr.write_token_kv(1, pos, k.view(), v.view())
                .expect("write");
        }

        let chain = mgr.sequence_page_chain(1).expect("chain");
        assert_eq!(
            chain.len(),
            2,
            "should have 2 pages for 4 tokens at block_size=2"
        );
    }

    #[test]
    fn test_write_token_kv_values_correct() {
        let mut mgr = make_manager(8, 4);
        mgr.allocate_sequence(5).expect("alloc");

        let k = Array2::<f32>::from_elem((2, 4), 9.9);
        let v = Array2::<f32>::from_elem((2, 4), 8.8);
        mgr.write_token_kv(5, 2, k.view(), v.view()).expect("write");

        let chain = mgr.sequence_page_chain(5).expect("chain");
        let page_id = chain[0];
        let page = mgr.pool().get_page(page_id).expect("page");
        let (k_out, v_out) = page.read_kv(2).expect("read slot 2");
        assert!(k_out.iter().all(|&x| (x - 9.9_f32).abs() < 1e-5));
        assert!(v_out.iter().all(|&x| (x - 8.8_f32).abs() < 1e-5));
    }

    // ── SharedPrefixCache ─────────────────────

    #[test]
    fn test_prefix_cache_insert_and_lookup() {
        let mut cache = SharedPrefixCache::new();
        let tokens: &[u32] = &[1, 2, 3, 4, 5];
        let hash = SharedPrefixCache::compute_prefix_hash(tokens);

        assert!(cache.lookup(hash).is_none());
        cache.insert(hash, vec![0, 1, 2]);
        let chain = cache.lookup(hash).expect("should find");
        assert_eq!(chain, &[0u32, 1, 2]);
    }

    #[test]
    fn test_prefix_cache_deterministic_hash() {
        let tokens: &[u32] = &[100, 200, 300];
        let h1 = SharedPrefixCache::compute_prefix_hash(tokens);
        let h2 = SharedPrefixCache::compute_prefix_hash(tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_prefix_cache_different_tokens_different_hash() {
        let t1: &[u32] = &[1, 2, 3];
        let t2: &[u32] = &[1, 2, 4];
        assert_ne!(
            SharedPrefixCache::compute_prefix_hash(t1),
            SharedPrefixCache::compute_prefix_hash(t2)
        );
    }

    #[test]
    fn test_prefix_cache_evict() {
        let mut cache = SharedPrefixCache::new();
        let hash = SharedPrefixCache::compute_prefix_hash(&[7, 8, 9]);
        cache.insert(hash, vec![10, 11]);
        assert_eq!(cache.len(), 1);
        let evicted = cache.evict(hash).expect("should evict");
        assert_eq!(evicted, vec![10u32, 11]);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_sequence_page_chain_not_found() {
        let mgr = make_manager(4, 4);
        let err = mgr.sequence_page_chain(0).expect_err("should error");
        assert!(matches!(err, InferenceError::SequenceNotFound(0)));
    }

    #[test]
    fn test_multiple_sequences_independent() {
        let mut mgr = make_manager(16, 4);
        mgr.allocate_sequence(1).expect("alloc 1");
        mgr.allocate_sequence(2).expect("alloc 2");
        mgr.allocate_sequence(3).expect("alloc 3");

        assert_eq!(mgr.active_sequences(), 3);
        assert_eq!(mgr.pool().free_count(), 13);

        mgr.free_sequence(2).expect("free 2");
        assert_eq!(mgr.active_sequences(), 2);
        assert_eq!(mgr.pool().free_count(), 14);
    }
}
