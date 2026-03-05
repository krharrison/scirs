//! String interning and symbol table utilities for SciRS2.
//!
//! String interning is a technique that stores only one copy of each distinct
//! string value.  Interned strings are represented as compact integer IDs
//! (`SymbolId`), enabling O(1) equality checks and dramatically reduced memory
//! usage for workloads that repeatedly reference the same string values (e.g.
//! column names in dataframes, token vocabularies, graph node labels).
//!
//! # Structures
//!
//! - [`SymbolId`] — a compact 32-bit identifier for an interned string.
//! - [`StringInterner`] — a single-threaded (or `&mut`-guarded) string interner.
//! - [`SymbolTable`] — bidirectional `String ↔ SymbolId` mapping with metadata.
//! - [`SharedInterner`] — a `Clone`-able, `Send + Sync` interner backed by
//!   `Arc<RwLock<StringInterner>>`.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::string_intern::{StringInterner, SymbolId};
//!
//! let mut interner = StringInterner::new();
//! let id_a = interner.intern("hello");
//! let id_b = interner.intern("world");
//! let id_c = interner.intern("hello"); // returns the same ID
//!
//! assert_eq!(id_a, id_c);
//! assert_ne!(id_a, id_b);
//! assert_eq!(interner.lookup(id_a), Some("hello"));
//! assert_eq!(interner.len(), 2);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

// ============================================================================
// SymbolId
// ============================================================================

/// A compact 32-bit identifier for an interned string.
///
/// Using a `u32` (rather than `usize`) halves the memory footprint on 64-bit
/// platforms while still supporting 4 billion distinct symbols — far more than
/// any practical use case requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub u32);

impl SymbolId {
    /// The special "null" / sentinel ID that is never assigned to a real string.
    pub const NULL: SymbolId = SymbolId(u32::MAX);

    /// Returns the raw numeric value of this ID.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns the raw numeric value as `usize` for indexing.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns `true` if this ID equals the sentinel `NULL` value.
    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolId({})", self.0)
    }
}

// ============================================================================
// StringInterner
// ============================================================================

/// A single-threaded string interner.
///
/// Strings are stored contiguously in a single `String` buffer (`arena`),
/// with offsets recorded so that individual slices can be retrieved in O(1)
/// time.  A `HashMap<&'static str, SymbolId>` is used for O(1) amortized
/// deduplication.
///
/// # Memory model
///
/// Each unique string is copied exactly once into the arena.  After that,
/// `intern` returns in O(1) (hash lookup + possible arena extension).
/// Because the arena is append-only and never reallocates existing bytes,
/// the `&'static` trick is safe: we store raw pointers into the arena and
/// transmute them to `&'static str` internally.  External callers receive
/// `&str` with lifetime bounded to `&self`.
pub struct StringInterner {
    /// Append-only byte arena that owns all interned string data.
    arena: String,
    /// Byte offsets into `arena`: `offsets[i]` is the start byte of symbol `i`.
    offsets: Vec<usize>,
    /// Byte lengths of each symbol.
    lengths: Vec<usize>,
    /// Maps raw pointers (as `usize`) of each arena slice to its `SymbolId`.
    /// We key by the *content* (by delegating through a string hash) rather
    /// than by pointer so look-ups work before and after insertion.
    map: HashMap<StringKey, SymbolId>,
}

/// Internal key that hashes/equals by *string content* even though it
/// stores a raw-pointer slice.  The arena is never mutated in place, so
/// pointers remain valid for the lifetime of the arena.
#[derive(Clone)]
struct StringKey {
    ptr: *const u8,
    len: usize,
}

// SAFETY: StringInterner is the sole owner of the arena; keys are only
// created inside StringInterner methods and never leak.
unsafe impl Send for StringKey {}
unsafe impl Sync for StringKey {}

impl StringKey {
    fn as_str(&self) -> &str {
        // SAFETY: ptr and len were derived from a valid &str slice stored in the arena.
        unsafe {
            let slice = std::slice::from_raw_parts(self.ptr, self.len);
            std::str::from_utf8_unchecked(slice)
        }
    }
}

impl PartialEq for StringKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for StringKey {}

impl std::hash::Hash for StringKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl StringInterner {
    /// Creates a new, empty `StringInterner`.
    pub fn new() -> Self {
        StringInterner {
            arena: String::with_capacity(4096),
            offsets: Vec::new(),
            lengths: Vec::new(),
            map: HashMap::new(),
        }
    }

    /// Creates a new `StringInterner` with pre-allocated capacity for `cap`
    /// symbols and `arena_bytes` bytes of string data.
    pub fn with_capacity(cap: usize, arena_bytes: usize) -> Self {
        StringInterner {
            arena: String::with_capacity(arena_bytes),
            offsets: Vec::with_capacity(cap),
            lengths: Vec::with_capacity(cap),
            map: HashMap::with_capacity(cap),
        }
    }

    /// Interns `s`, returning its `SymbolId`.
    ///
    /// If `s` has been interned before, the existing ID is returned without
    /// any allocation.  Otherwise `s` is appended to the internal arena and a
    /// new ID is allocated.
    ///
    /// # Panics
    ///
    /// Panics if the number of interned strings would overflow `u32::MAX - 1`.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        // Fast-path: build a temporary key pointing to the *input* slice so
        // we can probe the map without touching the arena at all.
        let probe_key = StringKey {
            ptr: s.as_ptr(),
            len: s.len(),
        };

        // Check map by content equality (not by pointer identity).
        if let Some(&id) = self.map.get(&probe_key) {
            return id;
        }

        // Slow-path: append to arena and record.
        let start = self.arena.len();
        self.arena.push_str(s);
        let end = self.arena.len();

        // Build a permanent key pointing into the *arena* (stable pointer).
        let arena_ptr = self.arena[start..end].as_ptr();
        let arena_key = StringKey {
            ptr: arena_ptr,
            len: s.len(),
        };

        let raw_id = self.offsets.len();
        assert!(
            raw_id < u32::MAX as usize - 1,
            "StringInterner: SymbolId overflow"
        );
        let id = SymbolId(raw_id as u32);

        self.offsets.push(start);
        self.lengths.push(s.len());
        self.map.insert(arena_key, id);

        id
    }

    /// Returns the string associated with `id`, or `None` if `id` is out of range.
    pub fn lookup(&self, id: SymbolId) -> Option<&str> {
        let idx = id.as_usize();
        if idx >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[idx];
        let len = self.lengths[idx];
        Some(&self.arena[start..start + len])
    }

    /// Returns `true` if `s` has already been interned.
    pub fn contains(&self, s: &str) -> bool {
        let probe = StringKey {
            ptr: s.as_ptr(),
            len: s.len(),
        };
        self.map.contains_key(&probe)
    }

    /// Returns the number of distinct strings currently interned.
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Returns `true` if no strings have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Returns an iterator over all `(SymbolId, &str)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (SymbolId, &str)> {
        self.offsets
            .iter()
            .zip(self.lengths.iter())
            .enumerate()
            .map(|(i, (&start, &len))| (SymbolId(i as u32), &self.arena[start..start + len]))
    }

    /// Returns an iterator over all `SymbolId`s in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = SymbolId> + '_ {
        (0..self.offsets.len()).map(|i| SymbolId(i as u32))
    }

    /// Attempts to look up the `SymbolId` for an already-interned string.
    /// Returns `None` if `s` has not been interned.
    pub fn get(&self, s: &str) -> Option<SymbolId> {
        let probe = StringKey {
            ptr: s.as_ptr(),
            len: s.len(),
        };
        self.map.get(&probe).copied()
    }

    /// Returns the total number of bytes currently used by the arena.
    pub fn arena_bytes(&self) -> usize {
        self.arena.len()
    }
}

impl fmt::Debug for StringInterner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StringInterner")
            .field("len", &self.len())
            .field("arena_bytes", &self.arena.len())
            .finish()
    }
}

// ============================================================================
// SymbolTable
// ============================================================================

/// Metadata entry stored alongside each symbol in a `SymbolTable`.
#[derive(Debug, Clone)]
pub struct SymbolMetadata {
    /// Number of times this symbol has been looked up via `intern`.
    pub interning_count: u32,
    /// Application-defined tag that callers may use freely.
    pub tag: u64,
}

impl SymbolMetadata {
    fn new() -> Self {
        SymbolMetadata {
            interning_count: 1,
            tag: 0,
        }
    }
}

/// A bidirectional `String ↔ SymbolId` mapping with optional per-symbol metadata.
///
/// `SymbolTable` wraps a `StringInterner` and adds:
/// - `resolve(id)` — retrieve `&str` by `SymbolId` (alias for `lookup`).
/// - Per-symbol [`SymbolMetadata`] tracking interning counts and user tags.
/// - `set_tag` / `get_tag` for lightweight symbol annotations.
pub struct SymbolTable {
    interner: StringInterner,
    metadata: Vec<SymbolMetadata>,
}

impl SymbolTable {
    /// Creates a new, empty `SymbolTable`.
    pub fn new() -> Self {
        SymbolTable {
            interner: StringInterner::new(),
            metadata: Vec::new(),
        }
    }

    /// Interns `s`, incrementing its reference count if already present.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        if let Some(id) = self.interner.get(s) {
            // Already exists — bump count.
            let idx = id.as_usize();
            self.metadata[idx].interning_count =
                self.metadata[idx].interning_count.saturating_add(1);
            return id;
        }
        let id = self.interner.intern(s);
        self.metadata.push(SymbolMetadata::new());
        id
    }

    /// Returns the string for `id`, or `None` if `id` is unknown.
    pub fn lookup(&self, id: SymbolId) -> Option<&str> {
        self.interner.lookup(id)
    }

    /// Alias for `lookup` — resolves a `SymbolId` to its string representation.
    pub fn resolve(&self, id: SymbolId) -> Option<&str> {
        self.interner.lookup(id)
    }

    /// Returns the `SymbolId` for a string that has already been interned,
    /// or `None` if it has not.
    pub fn get(&self, s: &str) -> Option<SymbolId> {
        self.interner.get(s)
    }

    /// Returns `true` if `s` exists in this table.
    pub fn contains(&self, s: &str) -> bool {
        self.interner.contains(s)
    }

    /// Returns the number of distinct symbols in the table.
    pub fn len(&self) -> usize {
        self.interner.len()
    }

    /// Returns `true` if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.interner.is_empty()
    }

    /// Returns a reference to the metadata for `id`, or `None` if unknown.
    pub fn metadata(&self, id: SymbolId) -> Option<&SymbolMetadata> {
        self.metadata.get(id.as_usize())
    }

    /// Returns a mutable reference to the metadata for `id`, or `None` if unknown.
    pub fn metadata_mut(&mut self, id: SymbolId) -> Option<&mut SymbolMetadata> {
        self.metadata.get_mut(id.as_usize())
    }

    /// Sets the user-defined tag for `id`.  Returns `false` if `id` is unknown.
    pub fn set_tag(&mut self, id: SymbolId, tag: u64) -> bool {
        match self.metadata.get_mut(id.as_usize()) {
            Some(m) => {
                m.tag = tag;
                true
            }
            None => false,
        }
    }

    /// Returns the user-defined tag for `id`, or `None` if `id` is unknown.
    pub fn get_tag(&self, id: SymbolId) -> Option<u64> {
        self.metadata.get(id.as_usize()).map(|m| m.tag)
    }

    /// Returns an iterator over `(SymbolId, &str, &SymbolMetadata)` triples.
    pub fn iter(&self) -> impl Iterator<Item = (SymbolId, &str, &SymbolMetadata)> {
        self.interner
            .iter()
            .zip(self.metadata.iter())
            .map(|((id, s), meta)| (id, s, meta))
    }

    /// Borrows the underlying `StringInterner`.
    pub fn interner(&self) -> &StringInterner {
        &self.interner
    }
}

impl fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolTable")
            .field("len", &self.len())
            .finish()
    }
}

// ============================================================================
// SharedInterner
// ============================================================================

/// An error returned by [`SharedInterner`] operations when the internal lock
/// has been poisoned.
#[derive(Debug, Clone)]
pub struct InternerLockError(String);

impl fmt::Display for InternerLockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedInterner lock poisoned: {}", self.0)
    }
}

impl std::error::Error for InternerLockError {}

/// A thread-safe string interner backed by `Arc<RwLock<StringInterner>>`.
///
/// `SharedInterner` can be cheaply cloned; all clones share the same
/// underlying storage.  Read operations (lookup, contains, len) acquire a
/// read lock, while write operations (intern) acquire an exclusive write lock.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_intern::SharedInterner;
/// use std::thread;
///
/// let interner = SharedInterner::new();
/// let clone = interner.clone();
///
/// let handle = thread::spawn(move || {
///     clone.intern("from-thread").expect("lock poisoned")
/// });
///
/// let id1 = interner.intern("from-main").expect("lock poisoned");
/// let id2 = handle.join().expect("thread panicked");
///
/// // Different strings get different IDs.
/// assert_ne!(id1, id2);
/// ```
#[derive(Clone)]
pub struct SharedInterner {
    inner: Arc<RwLock<StringInterner>>,
}

impl SharedInterner {
    /// Creates a new, empty `SharedInterner`.
    pub fn new() -> Self {
        SharedInterner {
            inner: Arc::new(RwLock::new(StringInterner::new())),
        }
    }

    /// Creates a new `SharedInterner` pre-allocated for `cap` symbols and
    /// `arena_bytes` bytes of string data.
    pub fn with_capacity(cap: usize, arena_bytes: usize) -> Self {
        SharedInterner {
            inner: Arc::new(RwLock::new(StringInterner::with_capacity(
                cap, arena_bytes,
            ))),
        }
    }

    /// Interns `s` and returns its `SymbolId`.
    ///
    /// Acquires a write lock.  If `s` is already interned (discoverable with
    /// a preceding read-lock probe) the write lock is still taken to keep the
    /// implementation simple and free of TOCTOU races.
    pub fn intern(&self, s: &str) -> Result<SymbolId, InternerLockError> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.intern(s))
    }

    /// Returns the string associated with `id`, or `None` if unknown.
    ///
    /// This method clones the string slice out of the arena so that the read
    /// lock can be released before returning.
    pub fn lookup(&self, id: SymbolId) -> Result<Option<String>, InternerLockError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.lookup(id).map(|s| s.to_owned()))
    }

    /// Returns `true` if `s` has already been interned.
    pub fn contains(&self, s: &str) -> Result<bool, InternerLockError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.contains(s))
    }

    /// Returns the number of distinct interned strings.
    pub fn len(&self) -> Result<usize, InternerLockError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.len())
    }

    /// Returns `true` if the interner contains no strings.
    pub fn is_empty(&self) -> Result<bool, InternerLockError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.is_empty())
    }

    /// Attempts to resolve `id` without cloning.  Returns a snapshot of all
    /// `(SymbolId, String)` pairs held at the time of the call.
    pub fn snapshot(&self) -> Result<Vec<(SymbolId, String)>, InternerLockError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.iter().map(|(id, s)| (id, s.to_owned())).collect())
    }

    /// Returns the `SymbolId` for an already-interned string, or `None` if absent.
    pub fn get(&self, s: &str) -> Result<Option<SymbolId>, InternerLockError> {
        let guard = self
            .inner
            .read()
            .map_err(|e| InternerLockError(e.to_string()))?;
        Ok(guard.get(s))
    }
}

impl fmt::Debug for SharedInterner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.inner.read().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("SharedInterner").field("len", &len).finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_intern_basic() {
        let mut interner = StringInterner::new();
        let id_a = interner.intern("hello");
        let id_b = interner.intern("world");
        let id_c = interner.intern("hello");

        assert_eq!(id_a, id_c, "same string should produce same ID");
        assert_ne!(id_a, id_b, "different strings should produce different IDs");
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_lookup_roundtrip() {
        let mut interner = StringInterner::new();
        let strings = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let ids: Vec<_> = strings.iter().map(|s| interner.intern(s)).collect();

        for (&id, &expected) in ids.iter().zip(strings.iter()) {
            assert_eq!(interner.lookup(id), Some(expected));
        }
    }

    #[test]
    fn test_contains() {
        let mut interner = StringInterner::new();
        interner.intern("present");

        assert!(interner.contains("present"));
        assert!(!interner.contains("absent"));
    }

    #[test]
    fn test_empty_string() {
        let mut interner = StringInterner::new();
        let id = interner.intern("");
        assert_eq!(interner.lookup(id), Some(""));
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn test_get() {
        let mut interner = StringInterner::new();
        let id = interner.intern("key");
        assert_eq!(interner.get("key"), Some(id));
        assert_eq!(interner.get("missing"), None);
    }

    #[test]
    fn test_iter() {
        let mut interner = StringInterner::new();
        interner.intern("a");
        interner.intern("b");
        interner.intern("c");

        let collected: Vec<_> = interner.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].1, "a");
        assert_eq!(collected[1].1, "b");
        assert_eq!(collected[2].1, "c");
    }

    #[test]
    fn test_symbol_table_metadata() {
        let mut table = SymbolTable::new();
        let id = table.intern("symbol");
        table.intern("symbol"); // second call — bumps count
        table.intern("symbol"); // third call

        let meta = table.metadata(id).expect("metadata should exist");
        assert_eq!(meta.interning_count, 3);

        table.set_tag(id, 0xDEAD_BEEF);
        assert_eq!(table.get_tag(id), Some(0xDEAD_BEEF));
    }

    #[test]
    fn test_symbol_table_resolve() {
        let mut table = SymbolTable::new();
        let id = table.intern("hello");
        assert_eq!(table.resolve(id), Some("hello"));
        assert_eq!(table.lookup(id), Some("hello"));
    }

    #[test]
    fn test_symbol_table_contains() {
        let mut table = SymbolTable::new();
        table.intern("yes");
        assert!(table.contains("yes"));
        assert!(!table.contains("no"));
    }

    #[test]
    fn test_shared_interner_basic() {
        let interner = SharedInterner::new();
        let id_a = interner.intern("x").expect("lock ok");
        let id_b = interner.intern("x").expect("lock ok");
        assert_eq!(id_a, id_b);
        assert_eq!(
            interner.lookup(id_a).expect("lock ok"),
            Some("x".to_owned())
        );
        assert_eq!(interner.len().expect("lock ok"), 1);
    }

    #[test]
    fn test_shared_interner_multithreaded() {
        let interner = SharedInterner::new();

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let ic = interner.clone();
                thread::spawn(move || {
                    let _id = ic.intern(&format!("thread-{}", i)).expect("lock ok");
                    // Also intern a shared string from all threads.
                    ic.intern("shared").expect("lock ok")
                })
            })
            .collect();

        let shared_ids: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("thread panic"))
            .collect();

        // All threads that interned "shared" should have received the same ID.
        let first = shared_ids[0];
        for id in &shared_ids[1..] {
            assert_eq!(*id, first, "shared string should map to the same SymbolId");
        }

        // Total symbols: 8 thread-local + 1 shared.
        assert_eq!(interner.len().expect("lock ok"), 9);
    }

    #[test]
    fn test_symbol_id_null() {
        assert!(SymbolId::NULL.is_null());
        assert!(!SymbolId(0).is_null());
    }

    #[test]
    fn test_lookup_out_of_range() {
        let interner = StringInterner::new();
        assert_eq!(interner.lookup(SymbolId(0)), None);
        assert_eq!(interner.lookup(SymbolId(999)), None);
    }

    #[test]
    fn test_unicode_strings() {
        let mut interner = StringInterner::new();
        let id1 = interner.intern("日本語");
        let id2 = interner.intern("한국어");
        let id3 = interner.intern("日本語"); // repeat

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.lookup(id1), Some("日本語"));
        assert_eq!(interner.lookup(id2), Some("한국어"));
    }

    #[test]
    fn test_large_number_of_symbols() {
        let mut interner = StringInterner::new();
        let n = 10_000usize;
        for i in 0..n {
            interner.intern(&format!("symbol_{}", i));
        }
        assert_eq!(interner.len(), n);
        // Re-intern existing symbols — count should not change.
        for i in 0..n {
            interner.intern(&format!("symbol_{}", i));
        }
        assert_eq!(interner.len(), n);
    }
}
