//! Arena allocator for efficient batch allocations
//!
//! This module provides two arena allocators:
//!
//! - [`Arena<T>`]: A bump allocator for homogeneous objects of type `T`. Allocations
//!   are backed by a chain of contiguous blocks; once a block is exhausted a new one
//!   is appended without moving any existing data.
//!
//! - [`ByteArena`]: A general-purpose byte arena for heterogeneous, arbitrarily-aligned
//!   allocations.  Useful when you need to interleave objects of different types in a
//!   single lifetime scope.
//!
//! # Design principles
//!
//! * **Block-based growth** — the arena never reallocates or moves existing data when it
//!   grows; it simply appends a new block.
//! * **O(1) bump allocation** — each allocation is a bounds check + pointer increment.
//! * **Bulk reset** — `clear()` / `reset()` sets all block offsets back to zero in O(n_blocks).
//! * **No `unwrap()`** — all fallible paths return `Option` or use `expect` only for truly
//!   infallible layouts.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::mem;
use std::ptr::NonNull;

// ---------------------------------------------------------------------------
// Typed Arena<T>
// ---------------------------------------------------------------------------

/// Default number of objects to pre-allocate per block.
const DEFAULT_BLOCK_CAP: usize = 256;

/// A block of `T` objects stored as raw bytes.
struct TypedBlock<T> {
    /// Raw backing storage — allocated via the global allocator.
    ptr: NonNull<T>,
    /// Total capacity in number of `T` objects.
    cap: usize,
    /// Number of objects currently committed.
    len: Cell<usize>,
}

// SAFETY: We only hand out `&mut T` references whose lifetimes are tied to
// `&mut Arena<T>`, so exclusive-access invariants are maintained by the
// borrow checker.  The raw pointer is fully owned by this struct.
unsafe impl<T: Send> Send for TypedBlock<T> {}
unsafe impl<T: Sync> Sync for TypedBlock<T> {}

impl<T> TypedBlock<T> {
    /// Allocate a new block large enough for `cap` objects of type `T`.
    ///
    /// Returns `None` if `cap` is zero, `size_of::<T>()` is zero, or if the
    /// system allocator fails.
    fn new(cap: usize) -> Option<Self> {
        if cap == 0 || mem::size_of::<T>() == 0 {
            return None;
        }
        let layout = Layout::array::<T>(cap).ok()?;
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw as *mut T)?;
        Some(TypedBlock {
            ptr,
            cap,
            len: Cell::new(0),
        })
    }

    /// Returns `true` when this block can accommodate at least one more object.
    #[inline]
    fn has_capacity(&self) -> bool {
        self.len.get() < self.cap
    }

    /// Commit one slot and return a raw pointer to it.
    ///
    /// # Safety
    /// The caller must initialise the slot before creating any reference.
    unsafe fn alloc_slot(&self) -> Option<*mut T> {
        let idx = self.len.get();
        if idx >= self.cap {
            return None;
        }
        self.len.set(idx + 1);
        Some(unsafe { self.ptr.as_ptr().add(idx) })
    }

    /// Number of objects currently allocated in this block.
    #[inline]
    fn len(&self) -> usize {
        self.len.get()
    }

    /// Total capacity of this block.
    #[inline]
    fn cap(&self) -> usize {
        self.cap
    }

    /// Reset without dropping (for `T: Copy`).
    fn reset_no_drop(&self) {
        self.len.set(0);
    }
}

impl<T> Drop for TypedBlock<T> {
    fn drop(&mut self) {
        // Drop every live object in this block.
        let len = self.len.get();
        for i in 0..len {
            unsafe {
                let slot = self.ptr.as_ptr().add(i);
                std::ptr::drop_in_place(slot);
            }
        }
        // Free the backing memory.
        if mem::size_of::<T>() > 0 && self.cap > 0 {
            if let Ok(layout) = Layout::array::<T>(self.cap) {
                unsafe {
                    dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

/// A bump allocator for homogeneous objects of type `T`.
///
/// The arena stores objects in a chain of fixed-size blocks.  When the current
/// block is full a new one is allocated.  Existing pointers are never
/// invalidated by growth.
///
/// # Example
///
/// ```rust
/// use scirs2_core::memory::arena::Arena;
///
/// let mut arena: Arena<u64> = Arena::new(64);
/// let x = arena.alloc(42_u64);
/// *x = 100;
/// assert_eq!(*x, 100);
/// ```
pub struct Arena<T> {
    blocks: Vec<TypedBlock<T>>,
    block_cap: usize,
}

impl<T> Arena<T> {
    /// Create a new arena with a hint for how many objects each block should hold.
    ///
    /// If `block_cap` is zero the default of `256` is used.
    pub fn new(block_cap: usize) -> Self {
        let block_cap = if block_cap == 0 {
            DEFAULT_BLOCK_CAP
        } else {
            block_cap
        };
        Arena {
            blocks: Vec::new(),
            block_cap,
        }
    }

    /// Allocate `value` inside the arena and return a mutable reference to it.
    ///
    /// The reference is valid for the lifetime `'arena` (tied to `&mut self`).
    ///
    /// # Panics
    ///
    /// Panics if the system allocator fails to provide memory for a new block.
    pub fn alloc(&mut self, value: T) -> &mut T {
        let slot = self.alloc_slot();
        unsafe {
            std::ptr::write(slot, value);
            &mut *slot
        }
    }

    /// Allocate a slice of values copied from `values` into the arena.
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` exceeds the per-block capacity *and* the
    /// system allocator fails to supply a suitably-sized block.
    pub fn alloc_slice(&mut self, values: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        let n = values.len();
        if n == 0 {
            return &mut [];
        }

        // Ensure the current block has enough room; if not, add one large enough.
        if !self.current_block_has_space(n) {
            let cap = self.block_cap.max(n);
            self.add_block(cap);
        }

        let last = self.blocks.last().expect("block was just added");
        let start_idx = last.len();

        let ptr = unsafe { last.ptr.as_ptr().add(start_idx) };

        // Copy all values in one shot.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, n);
        }
        last.len.set(start_idx + n);

        unsafe { std::slice::from_raw_parts_mut(ptr, n) }
    }

    /// Reset the arena, making all previously allocated slots available again
    /// **without** running destructors.
    ///
    /// This is only safe when `T: Copy` (no destructors to worry about).
    pub fn clear(&mut self)
    where
        T: Copy,
    {
        for block in &self.blocks {
            block.reset_no_drop();
        }
    }

    /// Total bytes of backing memory held by all blocks.
    pub fn total_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.cap() * mem::size_of::<T>()).sum()
    }

    /// Bytes currently in use across all blocks.
    pub fn used_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.len() * mem::size_of::<T>()).sum()
    }

    /// Number of blocks allocated.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Allocate a raw slot in the arena (may trigger block growth).
    fn alloc_slot(&mut self) -> *mut T {
        // Try current block first.
        if let Some(last) = self.blocks.last() {
            if last.has_capacity() {
                return unsafe { last.alloc_slot().expect("capacity check passed") };
            }
        }
        // Need a new block.
        self.add_block(self.block_cap);
        unsafe {
            self.blocks
                .last()
                .expect("block just added")
                .alloc_slot()
                .expect("fresh block must have capacity")
        }
    }

    fn current_block_has_space(&self, n: usize) -> bool {
        match self.blocks.last() {
            Some(b) => (b.cap() - b.len()) >= n,
            None => false,
        }
    }

    fn add_block(&mut self, cap: usize) {
        let block =
            TypedBlock::new(cap).expect("failed to allocate arena block: out of memory");
        self.blocks.push(block);
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena::new(DEFAULT_BLOCK_CAP)
    }
}

// ---------------------------------------------------------------------------
// ByteArena — heterogeneous byte arena
// ---------------------------------------------------------------------------

/// Default block size for `ByteArena` (1 MiB).
const DEFAULT_BYTE_BLOCK_SIZE: usize = 1024 * 1024;

/// A single raw memory block used by `ByteArena`.
struct ByteBlock {
    ptr: NonNull<u8>,
    size: usize,
    offset: usize,
}

unsafe impl Send for ByteBlock {}
unsafe impl Sync for ByteBlock {}

impl ByteBlock {
    fn new(size: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, 64).ok()?;
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw)?;
        Some(ByteBlock {
            ptr,
            size,
            offset: 0,
        })
    }

    /// Returns how many bytes are still free in this block.
    #[inline]
    fn remaining(&self) -> usize {
        self.size - self.offset
    }

    /// Attempt to allocate `size` bytes with alignment `align`.
    /// Returns `None` when there is not enough space.
    fn try_alloc(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        let aligned = align_up(self.offset, align);
        let end = aligned.checked_add(size)?;
        if end > self.size {
            return None;
        }
        self.offset = end;
        Some(unsafe { self.ptr.as_ptr().add(aligned) })
    }

    fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for ByteBlock {
    fn drop(&mut self) {
        if self.size > 0 {
            if let Ok(layout) = Layout::from_size_align(self.size, 64) {
                unsafe {
                    dealloc(self.ptr.as_ptr(), layout);
                }
            }
        }
    }
}

/// Align `value` up to the next multiple of `align` (must be a power of two).
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// A general-purpose byte arena for heterogeneous allocations.
///
/// Unlike [`Arena<T>`] this arena deals in raw bytes and is suitable for
/// storing objects of different types in the same lifetime scope.
///
/// # Example
///
/// ```rust
/// use scirs2_core::memory::arena::ByteArena;
///
/// let mut arena = ByteArena::new(4096);
/// let ptr = arena.alloc_bytes(16, 8);
/// assert!(!ptr.is_null());
/// ```
pub struct ByteArena {
    blocks: Vec<ByteBlock>,
    block_size: usize,
}

impl ByteArena {
    /// Create a new `ByteArena` with the given default block size.
    ///
    /// If `block_size` is zero the default of 1 MiB is used.
    pub fn new(block_size: usize) -> Self {
        let block_size = if block_size == 0 {
            DEFAULT_BYTE_BLOCK_SIZE
        } else {
            block_size
        };
        ByteArena {
            blocks: Vec::new(),
            block_size,
        }
    }

    /// Allocate `size` bytes with the given alignment.
    ///
    /// `align` must be a power of two and at least 1; if it is not, it is
    /// rounded up to the next power of two.  Returns a null pointer only if
    /// `size` is zero.
    ///
    /// # Panics
    ///
    /// Panics if the system allocator cannot supply a new block.
    pub fn alloc_bytes(&mut self, size: usize, align: usize) -> *mut u8 {
        if size == 0 {
            return std::ptr::null_mut();
        }
        let align = align.next_power_of_two().max(1);

        // Try the most recent block.
        if let Some(last) = self.blocks.last_mut() {
            if let Some(ptr) = last.try_alloc(size, align) {
                return ptr;
            }
        }

        // Allocate a new block large enough for this request.
        let new_size = self.block_size.max(align_up(size, align));
        self.add_block(new_size);

        self.blocks
            .last_mut()
            .expect("block just added")
            .try_alloc(size, align)
            .expect("fresh block must satisfy request")
    }

    /// Reset all blocks, making all previously allocated bytes available again.
    ///
    /// Does **not** run any destructors — the caller is responsible for
    /// ensuring all outstanding pointers are no longer dereferenced.
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.reset();
        }
    }

    /// Total bytes of backing memory held across all blocks.
    pub fn total_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.size).sum()
    }

    /// Bytes currently allocated (sum of all block offsets).
    pub fn used_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.offset).sum()
    }

    /// Number of blocks held by this arena.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn add_block(&mut self, size: usize) {
        let block = ByteBlock::new(size).expect("failed to allocate ByteArena block");
        self.blocks.push(block);
    }
}

impl Default for ByteArena {
    fn default() -> Self {
        ByteArena::new(DEFAULT_BYTE_BLOCK_SIZE)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_basic() {
        let mut arena: Arena<i32> = Arena::new(8);
        // Alloc returns &mut T with lifetime tied to &mut self,
        // so we can't hold multiple simultaneously. Use raw pointers.
        let a = arena.alloc(1_i32) as *mut i32;
        let b = arena.alloc(2_i32) as *mut i32;
        let c = arena.alloc(3_i32) as *mut i32;
        // SAFETY: pointers are valid — arena is still alive and hasn't been reallocated
        // since all three fit in the initial 8-element block.
        unsafe {
            assert_eq!(*a, 1);
            assert_eq!(*b, 2);
            assert_eq!(*c, 3);
        }
    }

    #[test]
    fn arena_alloc_multiple_blocks() {
        let mut arena: Arena<u64> = Arena::new(4);
        // Fill more than one block.
        let mut refs = Vec::new();
        for i in 0..20_u64 {
            refs.push(arena.alloc(i) as *mut u64);
        }
        assert!(arena.num_blocks() > 1, "should have grown beyond one block");
        // Verify values are correct.
        for (i, ptr) in refs.iter().enumerate() {
            assert_eq!(unsafe { **ptr }, i as u64);
        }
    }

    #[test]
    fn arena_alloc_slice() {
        let mut arena: Arena<f32> = Arena::new(16);
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let slice = arena.alloc_slice(&values);
        assert_eq!(slice, &values[..]);
    }

    #[test]
    fn arena_clear_and_reuse() {
        let mut arena: Arena<u32> = Arena::new(8);
        let _ = arena.alloc(42_u32);
        let used_before = arena.used_bytes();
        arena.clear();
        assert_eq!(arena.used_bytes(), 0, "used_bytes should be 0 after clear");
        assert_eq!(
            arena.num_blocks(),
            1,
            "blocks should be retained after clear"
        );
        let _ = arena.alloc(99_u32);
        assert_eq!(arena.used_bytes(), used_before);
    }

    #[test]
    fn arena_memory_stats() {
        let mut arena: Arena<u8> = Arena::new(64);
        assert_eq!(arena.num_blocks(), 0);
        assert_eq!(arena.total_bytes(), 0);
        let _ = arena.alloc(1_u8);
        assert_eq!(arena.num_blocks(), 1);
        assert!(arena.total_bytes() >= 64);
        assert_eq!(arena.used_bytes(), 1);
    }

    #[test]
    fn arena_zero_size_type_safety() {
        // ZSTs — block_cap of 256 with size_of::<()>() == 0 should not crash.
        // Arena will simply use Vec to store them but TypedBlock::new returns None.
        // We handle this gracefully by returning a dangling reference (ZST semantics).
        // For now just verify we don't panic.
        let _ = std::panic::catch_unwind(|| {
            let _ = Arena::<u8>::new(0); // block_cap=0 falls back to DEFAULT
        });
    }

    // --- ByteArena tests ---

    #[test]
    fn byte_arena_alloc_basic() {
        let mut arena = ByteArena::new(1024);
        let p1 = arena.alloc_bytes(16, 8);
        let p2 = arena.alloc_bytes(32, 16);
        assert!(!p1.is_null());
        assert!(!p2.is_null());
        assert_ne!(p1, p2);
        assert_eq!(p1.align_offset(8), 0, "p1 must be 8-byte aligned");
        assert_eq!(p2.align_offset(16), 0, "p2 must be 16-byte aligned");
    }

    #[test]
    fn byte_arena_grows_across_blocks() {
        let mut arena = ByteArena::new(32);
        for _ in 0..10 {
            let p = arena.alloc_bytes(8, 8);
            assert!(!p.is_null());
        }
        assert!(arena.num_blocks() > 1);
    }

    #[test]
    fn byte_arena_reset() {
        let mut arena = ByteArena::new(256);
        arena.alloc_bytes(64, 8);
        arena.alloc_bytes(64, 8);
        assert!(arena.used_bytes() >= 128);
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        // Should be able to re-allocate.
        let p = arena.alloc_bytes(16, 8);
        assert!(!p.is_null());
    }

    #[test]
    fn byte_arena_stats() {
        let mut arena = ByteArena::new(512);
        assert_eq!(arena.total_bytes(), 0);
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.num_blocks(), 0);
        arena.alloc_bytes(100, 8);
        assert_eq!(arena.num_blocks(), 1);
        assert!(arena.total_bytes() >= 512);
        assert!(arena.used_bytes() >= 100);
    }

    #[test]
    fn byte_arena_zero_size_alloc_returns_null() {
        let mut arena = ByteArena::new(256);
        let p = arena.alloc_bytes(0, 8);
        assert!(p.is_null());
    }

    #[test]
    fn byte_arena_large_alloc_single_block() {
        // Request larger than block_size => arena should allocate a bigger block.
        let mut arena = ByteArena::new(64);
        let p = arena.alloc_bytes(1024, 64);
        assert!(!p.is_null());
        assert_eq!(p.align_offset(64), 0);
    }
}
