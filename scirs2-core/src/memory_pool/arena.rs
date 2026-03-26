//! High-performance arena allocator with size-class free lists.
//!
//! The allocator simulates GPU-like heap semantics entirely in safe Rust:
//!
//! * Ten size classes (64 B … 1 MiB) with per-class free lists.
//! * A bump-pointer for carving new blocks out of the simulated heap.
//! * Batch alloc / batch free helpers.
//! * Simulated `copy` and `zero` operations on the backing byte vector.

use std::collections::HashMap;

use super::types::{
    AllocError, AllocationId, AllocationStats, BlockState, MemoryBlock, PoolConfig,
};

/// Size boundaries for the ten built-in size classes (bytes).
pub const SIZE_CLASS_BOUNDARIES: [usize; 10] = [
    64, 128, 256, 512, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576,
];

/// One free-list bucket for a particular size class.
#[derive(Debug, Clone)]
pub struct SizeClass {
    /// Upper bound (inclusive) for allocations that belong to this class.
    pub class_size: usize,
    /// Ids of currently-free blocks in this size class.
    pub free_blocks: Vec<AllocationId>,
}

impl SizeClass {
    fn new(class_size: usize) -> Self {
        Self {
            class_size,
            free_blocks: Vec::new(),
        }
    }
}

/// Determines which size class (index) should handle an allocation of `size` bytes.
///
/// Returns `None` if `size` exceeds all size classes (buddy allocator territory).
fn size_class_index(size: usize) -> Option<usize> {
    SIZE_CLASS_BOUNDARIES
        .iter()
        .position(|&boundary| size <= boundary)
}

/// Round `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a non-zero power of two; the function panics in debug
/// mode if that precondition is violated.
#[inline]
fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two() && alignment > 0);
    (value + alignment - 1) & !(alignment - 1)
}

/// Validate that `alignment` is a non-zero power of two.
fn check_alignment(alignment: usize) -> Result<(), AllocError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        Err(AllocError::AlignmentError { alignment })
    } else {
        Ok(())
    }
}

/// Free-list arena allocator with GPU-style heap semantics.
///
/// The internal `heap` field is a `Vec<u8>` that simulates a flat GPU memory
/// region.  All offsets returned by `alloc` are valid indices into this vector.
pub struct ArenaAllocator {
    /// All blocks that have ever been allocated (live or freed).
    pub blocks: Vec<MemoryBlock>,
    /// Per-size-class free lists.
    pub size_classes: Vec<SizeClass>,
    /// Simulated GPU memory region.
    pub heap: Vec<u8>,
    /// Bump pointer: next byte available for carving.
    heap_bump: usize,
    /// Monotonically increasing id counter.
    next_id: usize,
    /// Fast lookup: AllocationId → index in `blocks`.
    id_to_index: HashMap<usize, usize>,
    /// Pool configuration used at construction time.
    pub config: PoolConfig,
}

impl ArenaAllocator {
    /// Create a new allocator with the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        let heap = vec![0u8; config.total_size];
        let size_classes = SIZE_CLASS_BOUNDARIES
            .iter()
            .map(|&sz| SizeClass::new(sz))
            .collect();

        Self {
            blocks: Vec::new(),
            size_classes,
            heap,
            heap_bump: 0,
            next_id: 0,
            id_to_index: HashMap::new(),
            config,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn next_id(&mut self) -> AllocationId {
        let id = AllocationId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Look up the block for `id`, returning an error if not found.
    fn get_block(&self, id: AllocationId) -> Result<&MemoryBlock, AllocError> {
        let idx = self
            .id_to_index
            .get(&id.0)
            .copied()
            .ok_or(AllocError::InvalidId(id))?;
        Ok(&self.blocks[idx])
    }

    /// Look up the block for `id` mutably, returning an error if not found.
    fn get_block_mut(&mut self, id: AllocationId) -> Result<&mut MemoryBlock, AllocError> {
        let idx = self
            .id_to_index
            .get(&id.0)
            .copied()
            .ok_or(AllocError::InvalidId(id))?;
        Ok(&mut self.blocks[idx])
    }

    /// Carve a fresh block from the bump heap.
    fn bump_alloc(
        &mut self,
        class_size: usize,
        alignment: usize,
        requested_size: usize,
        class_index: usize,
    ) -> Result<AllocationId, AllocError> {
        // Align the bump pointer.
        let offset = align_up(self.heap_bump, alignment);
        let end = offset + class_size;

        if end > self.heap.len() {
            return Err(AllocError::OutOfMemory {
                requested: class_size,
                available: self.heap.len().saturating_sub(self.heap_bump),
            });
        }

        self.heap_bump = end;

        let id = self.next_id();
        let block = MemoryBlock::new(
            id,
            offset,
            class_size,
            alignment,
            requested_size,
            class_index,
        );
        let idx = self.blocks.len();
        self.id_to_index.insert(id.0, idx);
        self.blocks.push(block);
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Public allocation interface
    // -----------------------------------------------------------------------

    /// Allocate `size` bytes with the given `alignment`.
    ///
    /// The smallest size class ≥ `size` is used.  If the free list for that
    /// class is non-empty, a recycled block is returned; otherwise a new block
    /// is carved from the bump heap.
    pub fn alloc(&mut self, size: usize, alignment: usize) -> Result<AllocationId, AllocError> {
        check_alignment(alignment)?;

        if size == 0 {
            return Err(AllocError::OutOfMemory {
                requested: 0,
                available: self.heap.len(),
            });
        }

        match size_class_index(size) {
            Some(ci) => {
                let class_size = self.size_classes[ci].class_size;

                // Try to recycle from the free list.
                if let Some(recycled_id) = self.size_classes[ci].free_blocks.pop() {
                    let block = self.get_block_mut(recycled_id)?;
                    block.state = BlockState::Allocated;
                    block.requested_size = size;
                    block.alignment = alignment;
                    return Ok(recycled_id);
                }

                // Carve a fresh block.
                self.bump_alloc(class_size, alignment, size, ci)
            }
            None => {
                // Exceeds all size classes — allocate exactly `size` bytes
                // (rounded up to alignment) with no size-class overhead.
                let class_size = align_up(size, alignment);
                self.bump_alloc(class_size, alignment, size, usize::MAX)
            }
        }
    }

    /// Allocate multiple blocks in a single call.
    ///
    /// All allocations use the default alignment from `self.config`.
    pub fn alloc_batch(&mut self, sizes: &[usize]) -> Result<Vec<AllocationId>, AllocError> {
        let alignment = self.config.alignment;
        let mut ids = Vec::with_capacity(sizes.len());
        for &sz in sizes {
            ids.push(self.alloc(sz, alignment)?);
        }
        Ok(ids)
    }

    // -----------------------------------------------------------------------
    // Deallocation
    // -----------------------------------------------------------------------

    /// Free the block identified by `id`, returning it to its size-class free list.
    pub fn free(&mut self, id: AllocationId) -> Result<(), AllocError> {
        let (class_index, _) = {
            let block = self.get_block_mut(id)?;

            if block.state != BlockState::Allocated {
                return Err(AllocError::InvalidId(id));
            }
            block.state = BlockState::Free;
            (block.size_class_index, block.offset)
        };

        if class_index < self.size_classes.len() {
            self.size_classes[class_index].free_blocks.push(id);
        }
        // Blocks beyond size classes are simply marked Free; they are
        // reclaimed during compaction.
        Ok(())
    }

    /// Free multiple blocks in one call.  Returns the number of blocks successfully freed.
    pub fn free_batch(&mut self, ids: &[AllocationId]) -> usize {
        let mut count = 0usize;
        for &id in ids {
            if self.free(id).is_ok() {
                count += 1;
            }
        }
        count
    }

    // -----------------------------------------------------------------------
    // Memory operations (simulated)
    // -----------------------------------------------------------------------

    /// Simulate a `memcpy` from `src` to `dst`.
    ///
    /// Copies `size` bytes; both blocks must be at least `size` bytes large.
    pub fn copy(
        &mut self,
        src: AllocationId,
        dst: AllocationId,
        size: usize,
    ) -> Result<(), AllocError> {
        let (src_offset, src_size) = {
            let b = self.get_block(src)?;
            (b.offset, b.size)
        };
        let (dst_offset, dst_size) = {
            let b = self.get_block(dst)?;
            (b.offset, b.size)
        };

        if size > src_size || size > dst_size {
            return Err(AllocError::OutOfMemory {
                requested: size,
                available: dst_size.min(src_size),
            });
        }

        // Use a temporary buffer to avoid borrow-checker issues with overlapping slices.
        let tmp: Vec<u8> = self.heap[src_offset..src_offset + size].to_vec();
        self.heap[dst_offset..dst_offset + size].copy_from_slice(&tmp);
        Ok(())
    }

    /// Zero all bytes in the block identified by `id`.
    pub fn zero(&mut self, id: AllocationId) -> Result<(), AllocError> {
        let (offset, size) = {
            let block = self.get_block(id)?;
            (block.offset, block.size)
        };
        for byte in &mut self.heap[offset..offset + size] {
            *byte = 0;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of the current allocation statistics.
    pub fn stats(&self) -> AllocationStats {
        let used: usize = self
            .blocks
            .iter()
            .filter(|b| b.state == BlockState::Allocated)
            .map(|b| b.size)
            .sum();

        let n_blocks = self
            .blocks
            .iter()
            .filter(|b| b.state == BlockState::Allocated)
            .count();
        let total = self.heap.len();
        let free = total.saturating_sub(used);

        let fragmentation = compute_fragmentation(&self.blocks);

        AllocationStats {
            total,
            used,
            free,
            n_blocks,
            fragmentation,
        }
    }

    // -----------------------------------------------------------------------
    // Internal accessors used by the defragmenter
    // -----------------------------------------------------------------------

    /// Return an iterator over all blocks (live and free).
    pub fn all_blocks(&self) -> &[MemoryBlock] {
        &self.blocks
    }

    /// Return a mutable reference to all blocks.
    pub fn all_blocks_mut(&mut self) -> &mut Vec<MemoryBlock> {
        &mut self.blocks
    }

    /// Return the current bump pointer.
    pub fn heap_bump(&self) -> usize {
        self.heap_bump
    }

    /// Set the bump pointer (used by compaction to reclaim tail space).
    pub(super) fn set_heap_bump(&mut self, bump: usize) {
        self.heap_bump = bump;
    }

    /// Move block data in the heap and update its offset record.
    pub(super) fn relocate_block(
        &mut self,
        id: AllocationId,
        new_offset: usize,
    ) -> Result<(), AllocError> {
        let (old_offset, size) = {
            let b = self.get_block(id)?;
            (b.offset, b.size)
        };

        if old_offset != new_offset {
            // Copy bytes within the heap.
            self.heap
                .copy_within(old_offset..old_offset + size, new_offset);
            let block = self.get_block_mut(id)?;
            block.offset = new_offset;
        }
        Ok(())
    }

    /// Capacity of the underlying heap.
    pub fn capacity(&self) -> usize {
        self.heap.len()
    }

    /// Expose a read slice of the heap for inspection.
    pub fn heap_slice(&self, offset: usize, len: usize) -> &[u8] {
        &self.heap[offset..offset + len]
    }

    /// Expose a mutable slice of the heap.
    pub fn heap_slice_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        &mut self.heap[offset..offset + len]
    }
}

// ---------------------------------------------------------------------------
// Fragmentation helper (shared with defragmenter)
// ---------------------------------------------------------------------------

/// Compute the combined external + internal fragmentation score in [0.0, 1.0].
///
/// Formula:
/// * **External** = 1 − largest_contiguous_free_region / total_free_bytes
/// * **Internal** = sum of wasted bytes (class_size − requested) / total_allocated
/// * **Score** = 0.5 * external + 0.5 * internal, clamped to [0.0, 1.0]
pub fn compute_fragmentation(blocks: &[MemoryBlock]) -> f64 {
    // Gather all free intervals as (offset, size).
    let mut free_intervals: Vec<(usize, usize)> = blocks
        .iter()
        .filter(|b| b.state == BlockState::Free)
        .map(|b| (b.offset, b.size))
        .collect();

    free_intervals.sort_by_key(|&(off, _)| off);

    // Merge contiguous or overlapping intervals.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (off, sz) in &free_intervals {
        if let Some(last) = merged.last_mut() {
            let (l_off, l_sz) = *last;
            if *off <= l_off + l_sz {
                *last = (l_off, (*off + *sz).max(l_off + l_sz) - l_off);
                continue;
            }
        }
        merged.push((*off, *sz));
    }

    let total_free: usize = free_intervals.iter().map(|&(_, s)| s).sum();
    let largest_free = merged.iter().map(|&(_, s)| s).max().unwrap_or(0);

    let external = if total_free == 0 {
        0.0
    } else {
        1.0 - largest_free as f64 / total_free as f64
    };

    // Internal fragmentation.
    let total_allocated: usize = blocks
        .iter()
        .filter(|b| b.state == BlockState::Allocated)
        .map(|b| b.size)
        .sum();

    let total_requested: usize = blocks
        .iter()
        .filter(|b| b.state == BlockState::Allocated)
        .map(|b| b.requested_size)
        .sum();

    let internal = if total_allocated == 0 {
        0.0
    } else {
        let wasted = total_allocated.saturating_sub(total_requested);
        wasted as f64 / total_allocated as f64
    };

    (0.5 * external + 0.5 * internal).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_arena() -> ArenaAllocator {
        ArenaAllocator::new(PoolConfig::default())
    }

    #[test]
    fn test_alloc_basic() {
        let mut arena = default_arena();
        let id = arena.alloc(128, 64).expect("alloc failed");
        let block = arena.get_block(id).expect("block lookup failed");
        assert_eq!(block.state, BlockState::Allocated);
        assert_eq!(block.requested_size, 128);
    }

    #[test]
    fn test_alloc_alignment() {
        let mut arena = default_arena();
        // Request 200 bytes with 512-byte alignment.
        let id = arena.alloc(200, 512).expect("alloc failed");
        let block = arena.get_block(id).expect("block lookup");
        assert_eq!(block.offset % 512, 0, "offset not aligned to 512");
    }

    #[test]
    fn test_alloc_size_class() {
        let mut arena = default_arena();
        // 64 bytes → size class 0 (boundary 64).
        let id = arena.alloc(64, 64).expect("alloc");
        let block = arena.get_block(id).expect("lookup");
        assert_eq!(block.size_class_index, 0);
        assert_eq!(block.size, 64);
    }

    #[test]
    fn test_free_list_reuse() {
        let mut arena = default_arena();
        let id1 = arena.alloc(64, 64).expect("alloc 1");
        let offset1 = arena.get_block(id1).expect("block").offset;
        arena.free(id1).expect("free");

        // Next alloc of the same class should reuse id1's block.
        let id2 = arena.alloc(64, 64).expect("alloc 2");
        let block2 = arena.get_block(id2).expect("block 2");
        // The recycled block has the same offset.
        assert_eq!(block2.offset, offset1);
    }

    #[test]
    fn test_alloc_batch() {
        let mut arena = default_arena();
        let ids = arena.alloc_batch(&[64, 128, 256]).expect("batch alloc");
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_free_batch() {
        let mut arena = default_arena();
        let ids = arena.alloc_batch(&[64, 128, 256]).expect("batch alloc");
        let freed = arena.free_batch(&ids);
        assert_eq!(freed, 3);
    }

    #[test]
    fn test_out_of_memory() {
        let config = PoolConfig {
            total_size: 256,
            ..Default::default()
        };
        let mut arena = ArenaAllocator::new(config);
        // First allocation: 64 bytes — should succeed.
        arena.alloc(64, 64).expect("first alloc");
        // Try to allocate more than the remaining space.
        let result = arena.alloc(512, 64);
        assert!(matches!(result, Err(AllocError::OutOfMemory { .. })));
    }

    #[test]
    fn test_stats_accounting() {
        let mut arena = default_arena();
        let id = arena.alloc(64, 64).expect("alloc");
        let stats = arena.stats();
        assert!(stats.used >= 64, "used should be ≥ 64");
        assert_eq!(stats.total, arena.capacity());
        assert_eq!(stats.used + stats.free, stats.total);

        arena.free(id).expect("free");
        let stats2 = arena.stats();
        assert_eq!(stats2.used, 0);
    }
}
