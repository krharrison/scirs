//! Probabilistic concurrent skip list providing O(log n) expected operations.
//!
//! The skip list is an ordered key-value map.  This implementation uses a
//! mutex per node for thread safety and a fixed 32-level tower scheme.
//!
//! # Algorithm
//!
//! A skip list augments a sorted linked list with multiple "express lanes"
//! that allow searches to skip large portions of the list.  Each node is
//! assigned a random height when it is inserted.  The expected height of a
//! node is 1/(1-p); with p = 0.5 the expected number of comparisons per
//! operation is O(log n).
//!
//! # Thread Safety
//!
//! Rather than a global lock the implementation uses a hierarchical locking
//! strategy: each node carries its own `Mutex`.  Insertions and removals
//! acquire a sequence of per-node locks following the standard lock-ordering
//! protocol (always from head → tail) to avoid deadlocks.

use std::sync::{Arc, Mutex};

const MAX_LEVEL: usize = 32;

// ---------------------------------------------------------------------------
// Internal node
// ---------------------------------------------------------------------------

struct SkipNode<K, V> {
    key: Option<K>,
    value: Option<V>,
    /// Forward pointers for each level.  `forward[0]` is the bottom-level next.
    forward: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>>,
}

impl<K, V> SkipNode<K, V> {
    fn new_head(height: usize) -> Self {
        SkipNode {
            key: None,
            value: None,
            forward: vec![None; height],
        }
    }

    fn new(key: K, value: V, height: usize) -> Self {
        SkipNode {
            key: Some(key),
            value: Some(value),
            forward: vec![None; height],
        }
    }
}

// ---------------------------------------------------------------------------
// Random level generator
// ---------------------------------------------------------------------------

/// Fast xorshift PRNG for level generation.
struct LevelGen {
    state: u64,
}

impl LevelGen {
    fn new() -> Self {
        // Seed with current time and stack address.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345678);
        LevelGen {
            state: seed ^ 0xdeadbeef_cafebabe,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn random_level(&mut self, max: usize) -> usize {
        let mut level = 1usize;
        while level < max && (self.next_u64() & 1) == 0 {
            level += 1;
        }
        level
    }
}

// ---------------------------------------------------------------------------
// SkipList
// ---------------------------------------------------------------------------

/// A concurrent ordered map backed by a probabilistic skip list.
///
/// Keys must implement `Ord + Clone`; values must implement `Clone`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::SkipList;
///
/// let mut sl: SkipList<u32, String> = SkipList::new();
/// sl.insert(3, "three".to_string());
/// sl.insert(1, "one".to_string());
/// sl.insert(2, "two".to_string());
///
/// assert_eq!(sl.get(&1), Some("one".to_string()));
/// assert_eq!(sl.get(&2), Some("two".to_string()));
///
/// sl.remove(&2);
/// assert_eq!(sl.get(&2), None);
/// ```
pub struct SkipList<K, V> {
    head: Arc<Mutex<SkipNode<K, V>>>,
    level: usize,
    len: usize,
    rng: LevelGen,
}

impl<K: Ord + Clone, V: Clone> SkipList<K, V> {
    /// Create an empty skip list.
    pub fn new() -> Self {
        SkipList {
            head: Arc::new(Mutex::new(SkipNode::new_head(MAX_LEVEL))),
            level: 1,
            len: 0,
            rng: LevelGen::new(),
        }
    }

    /// Return the number of key-value pairs in the list.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the list contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up the value associated with `key`.
    ///
    /// Returns `None` if no matching key exists.
    #[allow(clippy::while_let_loop)]
    pub fn get(&self, key: &K) -> Option<V> {
        // `current_node` is `Some(arc)` for a data node, or `None` meaning
        // "the head sentinel".  We keep a per-level forward-pointer vector
        // that we read from the current node.
        let head_guard = self.head.lock().ok()?;
        // `forwards[lvl]` is the forward pointer at level `lvl` of the
        // current predecessor node.  Initialised from the head.
        let mut forwards: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>> = head_guard.forward.clone();
        drop(head_guard);

        for lvl in (0..self.level).rev() {
            loop {
                let next = match forwards.get(lvl).and_then(|f| f.as_ref()) {
                    Some(n) => Arc::clone(n),
                    None => break,
                };
                let guard = match next.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                match guard.key.as_ref() {
                    Some(k) if k < key => {
                        // Advance: update only the levels that this node
                        // covers (i.e. 0..node.forward.len()), keeping higher
                        // levels from the previous position intact.
                        let node_fwd = guard.forward.clone();
                        drop(guard);
                        let node_height = node_fwd.len();
                        let copy_len = node_height.min(forwards.len());
                        forwards[..copy_len].clone_from_slice(&node_fwd[..copy_len]);
                    }
                    Some(k) if k == key => {
                        return guard.value.clone();
                    }
                    _ => break,
                }
            }
        }
        None
    }

    /// Insert or replace the value for `key`.
    #[allow(clippy::while_let_loop)]
    pub fn insert(&mut self, key: K, value: V) {
        let new_level = self.rng.random_level(MAX_LEVEL);
        if new_level > self.level {
            self.level = new_level;
        }

        // Collect update pointers: for each level the last node whose key < key.
        let mut update: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>> = vec![None; self.level];

        let head_guard = match self.head.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let mut forwards: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>> = head_guard.forward.clone();
        drop(head_guard);

        // `current_node_arc` tracks which data node we are sitting at (None = head).
        let mut current_node_arc: Option<Arc<Mutex<SkipNode<K, V>>>> = None;

        for lvl in (0..self.level).rev() {
            loop {
                let next = match forwards.get(lvl).and_then(|f| f.as_ref()) {
                    Some(n) => Arc::clone(n),
                    None => break,
                };
                let guard = match next.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                match guard.key.as_ref() {
                    Some(k) if k < &key => {
                        let node_fwd = guard.forward.clone();
                        drop(guard);
                        let node_height = node_fwd.len();
                        let copy_len = node_height.min(forwards.len());
                        forwards[..copy_len].clone_from_slice(&node_fwd[..copy_len]);
                        update[lvl] = Some(Arc::clone(&next));
                        current_node_arc = Some(next);
                    }
                    _ => break,
                }
            }
            // If we didn't advance at this level, the predecessor is
            // whatever we were sitting at from higher levels.
            if update[lvl].is_none() {
                update[lvl] = current_node_arc.clone(); // None means head
            }
        }

        // Check whether we need to update an existing node.
        if let Some(next_arc) = forwards.first().and_then(|f| f.as_ref()) {
            let mut guard = match next_arc.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            if guard.key.as_ref() == Some(&key) {
                guard.value = Some(value);
                return;
            }
        }

        // Allocate a new node.
        let new_node = Arc::new(Mutex::new(SkipNode::new(key, value, new_level)));

        // Splice in at every level.
        for lvl in 0..new_level {
            // Determine predecessor at this level.
            let pred = update.get(lvl).and_then(|u| u.as_ref());

            if let Some(pred_arc) = pred {
                let mut pred_guard = match pred_arc.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                let old_next = pred_guard.forward.get(lvl).and_then(|f| f.clone());
                if let Ok(mut new_guard) = new_node.lock() {
                    new_guard.forward[lvl] = old_next;
                }
                pred_guard.forward[lvl] = Some(Arc::clone(&new_node));
            } else {
                // Predecessor is head.
                let mut head_guard = match self.head.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                let old_next = head_guard.forward.get(lvl).and_then(|f| f.clone());
                if let Ok(mut new_guard) = new_node.lock() {
                    new_guard.forward[lvl] = old_next;
                }
                head_guard.forward[lvl] = Some(Arc::clone(&new_node));
            }
        }

        self.len += 1;
    }

    /// Remove the entry with the given key.  Returns `true` if a key was removed.
    #[allow(clippy::while_let_loop)]
    pub fn remove(&mut self, key: &K) -> bool {
        let mut update: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>> = vec![None; self.level];

        let head_guard = match self.head.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let mut forwards: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>> = head_guard.forward.clone();
        drop(head_guard);

        let mut current_node_arc: Option<Arc<Mutex<SkipNode<K, V>>>> = None;

        for lvl in (0..self.level).rev() {
            loop {
                let next = match forwards.get(lvl).and_then(|f| f.as_ref()) {
                    Some(n) => Arc::clone(n),
                    None => break,
                };
                let guard = match next.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                match guard.key.as_ref() {
                    Some(k) if k < key => {
                        let node_fwd = guard.forward.clone();
                        drop(guard);
                        let node_height = node_fwd.len();
                        let copy_len = node_height.min(forwards.len());
                        forwards[..copy_len].clone_from_slice(&node_fwd[..copy_len]);
                        update[lvl] = Some(Arc::clone(&next));
                        current_node_arc = Some(next);
                    }
                    _ => break,
                }
            }
            if update[lvl].is_none() {
                update[lvl] = current_node_arc.clone();
            }
        }

        // Target node should be forwards[0].
        let target_arc = match forwards.first().and_then(|f| f.as_ref()) {
            Some(a) => Arc::clone(a),
            None => return false,
        };
        let target_guard = match target_arc.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        if target_guard.key.as_ref() != Some(key) {
            return false;
        }
        let target_forward = target_guard.forward.clone();
        drop(target_guard);

        // Unlink from each level.
        for lvl in 0..self.level {
            let target_next = target_forward.get(lvl).and_then(|f| f.clone());

            let pred = update.get(lvl).and_then(|u| u.as_ref());
            if let Some(pred_arc) = pred {
                let mut pred_guard = match pred_arc.lock() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                // Only unlink if pred.forward[lvl] == target.
                let is_target = pred_guard
                    .forward
                    .get(lvl)
                    .and_then(|f| f.as_ref())
                    .map(|a| Arc::ptr_eq(a, &target_arc))
                    .unwrap_or(false);
                if is_target {
                    pred_guard.forward[lvl] = target_next;
                }
            } else {
                let mut head_guard = match self.head.lock() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                let is_target = head_guard
                    .forward
                    .get(lvl)
                    .and_then(|f| f.as_ref())
                    .map(|a| Arc::ptr_eq(a, &target_arc))
                    .unwrap_or(false);
                if is_target {
                    head_guard.forward[lvl] = target_next;
                }
            }
        }

        self.len -= 1;
        true
    }

    /// Return all key-value pairs whose keys fall within `[lo, hi)` in order.
    #[allow(clippy::while_let_loop)]
    pub fn range(&self, lo: &K, hi: &K) -> Vec<(K, V)> {
        let mut result = Vec::new();

        let head_guard = match self.head.lock() {
            Ok(g) => g,
            Err(_) => return result,
        };
        let mut forwards: Vec<Option<Arc<Mutex<SkipNode<K, V>>>>> = head_guard.forward.clone();
        drop(head_guard);

        // Fast-forward to first key >= lo using higher levels.
        'outer: for lvl in (1..self.level).rev() {
            loop {
                let next = match forwards.get(lvl).and_then(|f| f.as_ref()) {
                    Some(n) => Arc::clone(n),
                    None => break,
                };
                let guard = match next.lock() {
                    Ok(g) => g,
                    Err(_) => break 'outer,
                };
                match guard.key.as_ref() {
                    Some(k) if k < lo => {
                        let node_fwd = guard.forward.clone();
                        drop(guard);
                        let node_height = node_fwd.len();
                        let copy_len = node_height.min(forwards.len());
                        forwards[..copy_len].clone_from_slice(&node_fwd[..copy_len]);
                    }
                    _ => break,
                }
            }
        }

        // Scan at level 0.
        loop {
            let next = match forwards.first().and_then(|f| f.as_ref()) {
                Some(n) => Arc::clone(n),
                None => break,
            };
            let guard = match next.lock() {
                Ok(g) => g,
                Err(_) => break,
            };
            match guard.key.as_ref() {
                Some(k) if k >= lo && k < hi => {
                    if let (Some(k2), Some(v2)) = (guard.key.clone(), guard.value.clone()) {
                        result.push((k2, v2));
                    }
                    let fwd0 = guard.forward.first().cloned().flatten();
                    drop(guard);
                    forwards[0] = fwd0;
                }
                Some(k) if k >= hi => break,
                None => break,
                _ => {
                    // k < lo: advance at level 0
                    let fwd0 = guard.forward.first().cloned().flatten();
                    drop(guard);
                    forwards[0] = fwd0;
                }
            }
        }

        result
    }

    /// Check whether the skip list contains the given key.
    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Collect all key-value pairs in sorted order.
    ///
    /// This traverses the bottom level of the skip list, so it is O(n).
    pub fn iter(&self) -> Vec<(K, V)> {
        let mut result = Vec::with_capacity(self.len);

        let head_guard = match self.head.lock() {
            Ok(g) => g,
            Err(_) => return result,
        };
        let mut current = head_guard.forward.first().cloned().flatten();
        drop(head_guard);

        while let Some(node_arc) = current {
            let guard = match node_arc.lock() {
                Ok(g) => g,
                Err(_) => break,
            };
            if let (Some(k), Some(v)) = (guard.key.clone(), guard.value.clone()) {
                result.push((k, v));
            }
            current = guard.forward.first().cloned().flatten();
        }

        result
    }

    /// Remove the entry with the given key, returning the value if it existed.
    pub fn remove_entry(&mut self, key: &K) -> Option<V> {
        let value = self.get(key);
        if value.is_some() && self.remove(key) {
            value
        } else {
            None
        }
    }
}

impl<K: Ord + Clone, V: Clone> Default for SkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_get() {
        let mut sl: SkipList<u32, &str> = SkipList::new();
        sl.insert(5, "five");
        sl.insert(2, "two");
        sl.insert(8, "eight");

        assert_eq!(sl.get(&2), Some("two"));
        assert_eq!(sl.get(&5), Some("five"));
        assert_eq!(sl.get(&8), Some("eight"));
        assert_eq!(sl.get(&1), None);
        assert_eq!(sl.len(), 3);
    }

    #[test]
    fn test_remove() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        for i in 0..10u32 {
            sl.insert(i, i * 10);
        }
        assert_eq!(sl.len(), 10);

        assert!(sl.remove(&5));
        assert_eq!(sl.get(&5), None);
        assert_eq!(sl.len(), 9);

        // Removing non-existent key returns false.
        assert!(!sl.remove(&99));
    }

    #[test]
    fn test_range() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        for i in 0..20u32 {
            sl.insert(i, i);
        }
        let r = sl.range(&5, &10);
        assert_eq!(r.len(), 5);
        let keys: Vec<u32> = r.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        sl.insert(1, 100);
        sl.insert(1, 200);
        assert_eq!(sl.get(&1), Some(200));
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut sl: SkipList<i32, i32> = SkipList::new();
        assert!(sl.is_empty());
        sl.insert(42, 0);
        assert!(!sl.is_empty());
        assert_eq!(sl.len(), 1);
        sl.remove(&42);
        assert!(sl.is_empty());
    }

    #[test]
    fn test_large_insert_ordered() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        // Insert in reverse order.
        for i in (0..100u32).rev() {
            sl.insert(i, i);
        }
        let r = sl.range(&0, &100);
        assert_eq!(r.len(), 100);
        for (i, (k, v)) in r.iter().enumerate() {
            assert_eq!(*k, i as u32);
            assert_eq!(*v, i as u32);
        }
    }

    #[test]
    fn test_contains() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        sl.insert(10, 100);
        sl.insert(20, 200);
        assert!(sl.contains(&10));
        assert!(sl.contains(&20));
        assert!(!sl.contains(&30));
    }

    #[test]
    fn test_iter_sorted_order() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        sl.insert(5, 50);
        sl.insert(1, 10);
        sl.insert(9, 90);
        sl.insert(3, 30);
        sl.insert(7, 70);

        let items = sl.iter();
        assert_eq!(items.len(), 5);
        let keys: Vec<u32> = items.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn test_remove_entry() {
        let mut sl: SkipList<u32, String> = SkipList::new();
        sl.insert(1, "one".to_string());
        sl.insert(2, "two".to_string());

        let removed = sl.remove_entry(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert_eq!(sl.len(), 1);

        let not_found = sl.remove_entry(&99);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_iter_empty() {
        let sl: SkipList<u32, u32> = SkipList::new();
        assert!(sl.iter().is_empty());
    }

    #[test]
    fn test_range_empty_result() {
        let mut sl: SkipList<u32, u32> = SkipList::new();
        for i in 0..10u32 {
            sl.insert(i, i);
        }
        let r = sl.range(&10, &5);
        assert!(r.is_empty());
        let r2 = sl.range(&100, &200);
        assert!(r2.is_empty());
    }

    #[test]
    fn test_concurrent_read_access() {
        use std::sync::Arc;
        use std::thread;

        let mut sl = SkipList::new();
        for i in 0..100u32 {
            sl.insert(i, i * 10);
        }
        let shared = Arc::new(sl);

        let mut handles = Vec::new();
        for t in 0..4 {
            let sl_ref = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                for i in 0..100u32 {
                    let val = sl_ref.get(&i);
                    assert_eq!(val, Some(i * 10), "thread {t} failed for key {i}");
                }
            }));
        }

        for h in handles {
            if let Err(e) = h.join() {
                panic!("Thread panicked: {e:?}");
            }
        }
    }
}
