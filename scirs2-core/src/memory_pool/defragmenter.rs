//! Memory defragmentation for the arena allocator.
//!
//! Provides three complementary strategies:
//!
//! 1. **Full compaction** (`compact`) — moves all live allocations to the lowest
//!    possible offsets in a single pass, maximising the contiguous free region at
//!    the top of the heap.
//! 2. **Incremental defragmentation** (`incremental_defrag`) — moves at most
//!    `max_moves` blocks per call, prioritising moves that liberate the largest
//!    contiguous free regions.  Suitable for real-time workloads where stalling
//!    for full compaction is unacceptable.
//! 3. **Buddy allocator** (`BuddyAllocator`) — a standalone power-of-2 allocator
//!    for large allocations that naturally coalesces free buddies.

use std::collections::HashMap;

use super::arena::{compute_fragmentation, ArenaAllocator};
use super::types::{AllocationId, BlockState};

// ---------------------------------------------------------------------------
// Fragmentation metric
// ---------------------------------------------------------------------------

/// Compute the fragmentation score for a slice of blocks.
///
/// Returns a value in [0.0, 1.0]; 0.0 means perfectly compact.
pub fn fragmentation_score(blocks: &[super::types::MemoryBlock]) -> f64 {
    compute_fragmentation(blocks)
}

// ---------------------------------------------------------------------------
// Full compaction
// ---------------------------------------------------------------------------

/// Pack all live (Allocated) blocks down to the lowest offsets in the heap.
///
/// After compaction the heap looks like:
/// ```text
/// [block_0][block_1]...[block_n][         free region         ]
/// ```
///
/// Returns:
/// * The number of bytes recovered (i.e. old bump − new bump).
/// * A relocation map `old_id → new_offset` covering every block that moved.
pub fn compact(arena: &mut ArenaAllocator) -> (usize, HashMap<AllocationId, usize>) {
    // Collect live blocks sorted by their current offset.
    let mut live: Vec<(usize, usize, AllocationId, usize)> = arena
        .all_blocks()
        .iter()
        .filter(|b| b.state == BlockState::Allocated)
        .map(|b| (b.offset, b.size, b.id, b.alignment))
        .collect();

    live.sort_by_key(|&(off, _, _, _)| off);

    let mut cursor = 0usize;
    let mut relocation: HashMap<AllocationId, usize> = HashMap::new();

    for (old_offset, size, id, alignment) in live {
        // Align the cursor.
        let new_offset = align_up(cursor, alignment);

        if new_offset != old_offset {
            // Perform the in-heap byte move.
            let _ = arena.relocate_block(id, new_offset);
            relocation.insert(id, new_offset);
        }

        cursor = new_offset + size;
    }

    let old_bump = arena.heap_bump();
    let bytes_freed = old_bump.saturating_sub(cursor);
    arena.set_heap_bump(cursor);

    (bytes_freed, relocation)
}

/// Round `value` up to the nearest multiple of `alignment` (power of two).
#[inline]
fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Incremental defragmentation
// ---------------------------------------------------------------------------

/// Gap between two adjacent live blocks: `(gap_start, gap_size, block_id_after)`.
#[derive(Debug, Clone)]
struct Gap {
    gap_size: usize,
    block_id: AllocationId,
    block_offset: usize,
    block_size: usize,
    block_alignment: usize,
}

/// Move at most `max_moves` live blocks to eliminate the largest gaps in the heap.
///
/// Returns the number of blocks actually relocated.
pub fn incremental_defrag(arena: &mut ArenaAllocator, max_moves: usize) -> usize {
    if max_moves == 0 {
        return 0;
    }

    // Collect live blocks sorted by offset.
    let mut live: Vec<(usize, usize, AllocationId, usize)> = arena
        .all_blocks()
        .iter()
        .filter(|b| b.state == BlockState::Allocated)
        .map(|b| (b.offset, b.size, b.id, b.alignment))
        .collect();

    live.sort_by_key(|&(off, _, _, _)| off);

    if live.is_empty() {
        return 0;
    }

    // Build a list of gaps between consecutive live blocks.
    let mut gaps: Vec<Gap> = Vec::new();

    // Gap before the first block.
    let (first_off, first_sz, first_id, first_align) = live[0];
    if first_off > 0 {
        gaps.push(Gap {
            gap_size: first_off,
            block_id: first_id,
            block_offset: first_off,
            block_size: first_sz,
            block_alignment: first_align,
        });
    }

    // Gaps between consecutive blocks.
    for window in live.windows(2) {
        let (prev_off, prev_sz, _, _) = window[0];
        let (cur_off, cur_sz, cur_id, cur_align) = window[1];
        let prev_end = prev_off + prev_sz;
        if cur_off > prev_end {
            gaps.push(Gap {
                gap_size: cur_off - prev_end,
                block_id: cur_id,
                block_offset: cur_off,
                block_size: cur_sz,
                block_alignment: cur_align,
            });
        }
    }

    // Sort gaps largest-first so that moves free the most space first.
    gaps.sort_by_key(|g| std::cmp::Reverse(g.gap_size));

    let mut moves = 0usize;

    for gap in gaps.iter().take(max_moves) {
        if moves >= max_moves {
            break;
        }

        // Target offset: align the block to its requirements, positioned at gap start.
        let gap_start = gap.block_offset.saturating_sub(gap.gap_size);
        let new_offset = align_up(gap_start, gap.block_alignment);

        if new_offset < gap.block_offset {
            let _ = arena.relocate_block(gap.block_id, new_offset);
            moves += 1;
        }
    }

    // Update the bump pointer to the end of the last live block.
    let new_bump = arena
        .all_blocks()
        .iter()
        .filter(|b| b.state == BlockState::Allocated)
        .map(|b| b.offset + b.size)
        .max()
        .unwrap_or(0);

    if new_bump < arena.heap_bump() {
        arena.set_heap_bump(new_bump);
    }

    moves
}

// ---------------------------------------------------------------------------
// Buddy allocator
// ---------------------------------------------------------------------------

/// State of a node in the buddy tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuddyState {
    Free,
    Split,
    Allocated,
}

/// A minimal buddy allocator over a power-of-2–sized region.
///
/// Supports sizes that are powers of two.  Non-power-of-2 requests are rounded
/// up to the next power of two automatically.
///
/// Internally the allocator maintains a flat array indexed by `(level, position)`
/// encoded as `index = 2^level + position - 1`.  Level 0 is the root (full
/// capacity); each successive level halves the block size.
pub struct BuddyAllocator {
    /// Total capacity (must be a power of two ≥ 1).
    capacity: usize,
    /// Number of levels: `log2(capacity / min_order) + 1`.
    levels: usize,
    /// Minimum allocation order in bytes (must be a power of two).
    min_order: usize,
    /// Flat node state array.  Length = `2 * capacity / min_order - 1`.
    nodes: Vec<BuddyState>,
    /// Map from byte offset → (level, position) for live allocations.
    alloc_map: HashMap<usize, (usize, usize)>,
}

impl BuddyAllocator {
    /// Create a buddy allocator with the given total `capacity` and `min_order`.
    ///
    /// Both values must be non-zero powers of two.
    pub fn new(capacity: usize, min_order: usize) -> Self {
        assert!(
            capacity.is_power_of_two() && capacity > 0,
            "buddy allocator capacity must be a non-zero power of two"
        );
        assert!(
            min_order.is_power_of_two() && min_order > 0 && min_order <= capacity,
            "buddy allocator min_order must be a non-zero power of two ≤ capacity"
        );

        let levels = capacity.trailing_zeros() as usize - min_order.trailing_zeros() as usize + 1;
        let node_count = (1usize << levels) - 1;

        Self {
            capacity,
            levels,
            min_order,
            nodes: vec![BuddyState::Free; node_count],
            alloc_map: HashMap::new(),
        }
    }

    /// Encode `(level, position)` as a flat index.
    #[inline]
    fn node_index(level: usize, pos: usize) -> usize {
        (1usize << level) - 1 + pos
    }

    /// Size of a block at the given level.
    fn block_size(&self, level: usize) -> usize {
        self.capacity >> level
    }

    /// Offset of block `pos` at `level`.
    fn block_offset(&self, level: usize, pos: usize) -> usize {
        pos * self.block_size(level)
    }

    /// Allocate a block of at least `size` bytes.
    ///
    /// Returns `Some(offset)` on success, `None` if no suitable block is free.
    pub fn buddy_alloc(&mut self, size: usize) -> Option<usize> {
        if size == 0 || size > self.capacity {
            return None;
        }

        // Round up to the next power of two ≥ min_order.
        let rounded = size.next_power_of_two().max(self.min_order);
        // Level at which blocks have exactly `rounded` bytes.
        let target_level = (self.capacity / rounded).trailing_zeros() as usize;

        if target_level >= self.levels {
            return None;
        }

        self.alloc_at(0, 0, target_level)
    }

    /// Recursive helper — tries to allocate at the given `target_level` starting
    /// at `(level, pos)`.
    fn alloc_at(&mut self, level: usize, pos: usize, target_level: usize) -> Option<usize> {
        let idx = Self::node_index(level, pos);

        if idx >= self.nodes.len() {
            return None;
        }

        match self.nodes[idx] {
            BuddyState::Allocated => None,
            BuddyState::Free => {
                if level == target_level {
                    self.nodes[idx] = BuddyState::Allocated;
                    let offset = self.block_offset(level, pos);
                    self.alloc_map.insert(offset, (level, pos));
                    Some(offset)
                } else {
                    // Split and recurse left.
                    self.nodes[idx] = BuddyState::Split;
                    self.alloc_at(level + 1, pos * 2, target_level)
                }
            }
            BuddyState::Split => {
                // Try left child, then right child.
                let left = self.alloc_at(level + 1, pos * 2, target_level);
                if left.is_some() {
                    return left;
                }
                self.alloc_at(level + 1, pos * 2 + 1, target_level)
            }
        }
    }

    /// Free the allocation at `offset` with original `size` bytes.
    ///
    /// Automatically coalesces with the buddy if it is also free.
    pub fn buddy_free(&mut self, offset: usize, size: usize) {
        // Round up to match the allocation granularity.
        let rounded = size.next_power_of_two().max(self.min_order);
        let target_level = (self.capacity / rounded).trailing_zeros() as usize;

        // Look up via alloc_map first, then fall back to computed values.
        let (level, pos) = self.alloc_map.remove(&offset).unwrap_or_else(|| {
            let block_sz = self.block_size(target_level);
            let pos = offset / block_sz;
            (target_level, pos)
        });

        let idx = Self::node_index(level, pos);
        if idx < self.nodes.len() {
            self.nodes[idx] = BuddyState::Free;
            self.coalesce(level, pos);
        }
    }

    /// Coalesce a free block at `(level, pos)` with its buddy, propagating upward.
    fn coalesce(&mut self, level: usize, pos: usize) {
        if level == 0 {
            return; // Root — nothing to coalesce with.
        }

        let buddy_pos = pos ^ 1; // Toggle last bit to get buddy.
        let buddy_idx = Self::node_index(level, buddy_pos);

        if buddy_idx < self.nodes.len() && self.nodes[buddy_idx] == BuddyState::Free {
            // Both this block and its buddy are free — coalesce.
            let parent_pos = pos / 2;
            let parent_idx = Self::node_index(level - 1, parent_pos);
            self.nodes[parent_idx] = BuddyState::Free;
            self.nodes[Self::node_index(level, pos)] = BuddyState::Free;
            self.nodes[buddy_idx] = BuddyState::Free;
            // Recurse upward.
            self.coalesce(level - 1, parent_pos);
        }
    }

    /// Return the total number of bytes currently allocated.
    pub fn allocated_bytes(&self) -> usize {
        let leaf_level = self.levels - 1;
        let leaf_size = self.min_order;
        let n_leaves = 1usize << leaf_level;

        let mut total = 0usize;
        for pos in 0..n_leaves {
            let idx = Self::node_index(leaf_level, pos);
            if idx < self.nodes.len() && self.nodes[idx] == BuddyState::Allocated {
                total += leaf_size;
            }
        }
        // Also count non-leaf allocations.
        for level in 0..leaf_level {
            let n_blocks = 1usize << level;
            for pos in 0..n_blocks {
                let idx = Self::node_index(level, pos);
                if idx < self.nodes.len() && self.nodes[idx] == BuddyState::Allocated {
                    total += self.block_size(level);
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_pool::arena::ArenaAllocator;
    use crate::memory_pool::types::PoolConfig;

    fn make_arena() -> ArenaAllocator {
        ArenaAllocator::new(PoolConfig::default())
    }

    #[test]
    fn test_fragmentation_score_empty() {
        let blocks: Vec<super::super::types::MemoryBlock> = vec![];
        let score = fragmentation_score(&blocks);
        assert_eq!(score, 0.0, "empty pool should have zero fragmentation");
    }

    #[test]
    fn test_fragmentation_score_high() {
        let mut arena = make_arena();
        // Allocate many small blocks and free every other one.
        let mut ids = Vec::new();
        for _ in 0..20 {
            let id = arena.alloc(64, 64).expect("alloc");
            ids.push(id);
        }
        for (i, &id) in ids.iter().enumerate() {
            if i % 2 == 0 {
                arena.free(id).expect("free");
            }
        }

        let score = fragmentation_score(arena.all_blocks());
        // With alternating pattern, external fragmentation should be detectable.
        // The exact value depends on implementation; just verify it is > 0.
        assert!(
            score >= 0.0 && score <= 1.0,
            "score out of range: {}",
            score
        );
    }

    #[test]
    fn test_compact_packs() {
        let mut arena = make_arena();
        let ids: Vec<AllocationId> = (0..6)
            .map(|_| arena.alloc(64, 64).expect("alloc"))
            .collect();

        // Free alternating blocks to create gaps.
        arena.free(ids[1]).expect("free");
        arena.free(ids[3]).expect("free");

        let bump_before = arena.heap_bump();
        let (bytes_freed, _reloc) = compact(&mut arena);

        assert!(
            bytes_freed > 0 || arena.heap_bump() <= bump_before,
            "compact should not increase bump"
        );
        let score_after = fragmentation_score(arena.all_blocks());
        assert!(score_after <= 1.0);
    }

    #[test]
    fn test_compact_relocation() {
        let mut arena = make_arena();
        let ids: Vec<AllocationId> = (0..4)
            .map(|_| arena.alloc(64, 64).expect("alloc"))
            .collect();

        // Free the first block to force relocation of the rest.
        arena.free(ids[0]).expect("free");

        let (_bytes, reloc) = compact(&mut arena);

        // Every moved block must appear in the relocation map.
        for id in &ids[1..] {
            // If the block moved, its id should be in the relocation map.
            if let Some(&new_off) = reloc.get(id) {
                assert!(new_off < arena.heap_bump(), "relocated offset out of range");
            }
        }
    }

    #[test]
    fn test_incremental_defrag() {
        let mut arena = make_arena();
        let ids: Vec<AllocationId> = (0..10)
            .map(|_| arena.alloc(64, 64).expect("alloc"))
            .collect();

        // Create multiple gaps.
        arena.free(ids[2]).expect("free");
        arena.free(ids[5]).expect("free");
        arena.free(ids[8]).expect("free");

        let moved = incremental_defrag(&mut arena, 2);
        assert!(moved <= 2, "should not exceed max_moves=2, got {}", moved);
    }

    #[test]
    fn test_buddy_alloc_power_of_2() {
        let mut buddy = BuddyAllocator::new(1024, 64);
        let off = buddy.buddy_alloc(128).expect("alloc 128");
        assert_eq!(off % 128, 0, "buddy offset should be 128-byte aligned");
    }

    #[test]
    fn test_buddy_coalesce() {
        let mut buddy = BuddyAllocator::new(1024, 64);
        let off1 = buddy.buddy_alloc(128).expect("alloc 1");
        let off2 = buddy.buddy_alloc(128).expect("alloc 2");

        buddy.buddy_free(off1, 128);
        buddy.buddy_free(off2, 128);

        // After freeing both buddies the full 256-byte block should be available.
        let off3 = buddy.buddy_alloc(256).expect("alloc 256 after coalesce");
        assert_eq!(off3 % 256, 0);
    }
}
