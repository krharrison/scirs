//! Compressed trie (Patricia / radix tree) for string keys.
//!
//! Path compression collapses single-child chains into one node whose key
//! fragment spans multiple characters, yielding both memory savings and
//! faster traversal.  Shared prefixes are stored only once.
//!
//! # Operations
//!
//! | Operation          | Complexity       |
//! |--------------------|-----------------|
//! | `insert`           | O(k)            |
//! | `get`              | O(k)            |
//! | `remove`           | O(k)            |
//! | `prefix_search`    | O(k + m)        |
//! | `longest_prefix`   | O(k)            |
//!
//! where *k* is the key length and *m* is the number of results.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal node
// ---------------------------------------------------------------------------

/// A node in the compressed trie.
///
/// Each node stores a *fragment* of the key (possibly multiple characters)
/// and an optional value.  Children are indexed by the first byte of the
/// remaining suffix.
struct TrieNode<V> {
    /// The fragment of the key stored at this node.
    fragment: String,
    /// Value stored at this node (present only if this node represents a
    /// complete key insertion).
    value: Option<V>,
    /// Children keyed by the first byte of their fragment.
    children: HashMap<u8, Box<TrieNode<V>>>,
}

impl<V> TrieNode<V> {
    fn new(fragment: String, value: Option<V>) -> Self {
        TrieNode {
            fragment,
            value,
            children: HashMap::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CompressedTrie
// ---------------------------------------------------------------------------

/// A compressed trie (radix / Patricia tree) for string keys.
///
/// Supports insertion, lookup, removal, prefix search, and longest-prefix
/// matching.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::CompressedTrie;
///
/// let mut trie = CompressedTrie::new();
/// trie.insert("hello", 1);
/// trie.insert("help", 2);
/// trie.insert("world", 3);
///
/// assert_eq!(trie.get("hello"), Some(&1));
/// assert_eq!(trie.get("help"), Some(&2));
/// assert_eq!(trie.len(), 3);
///
/// let results = trie.prefix_search("hel");
/// assert_eq!(results.len(), 2);
/// ```
pub struct CompressedTrie<V> {
    root: TrieNode<V>,
    len: usize,
}

impl<V> CompressedTrie<V> {
    /// Create an empty compressed trie.
    pub fn new() -> Self {
        CompressedTrie {
            root: TrieNode::new(String::new(), None),
            len: 0,
        }
    }

    /// Return the number of entries in the trie.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the trie contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a key-value pair.
    ///
    /// If the key already exists, the value is replaced and the old value
    /// is returned.
    pub fn insert(&mut self, key: &str, value: V) -> Option<V> {
        let old = Self::insert_recursive(&mut self.root, key, value);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    fn insert_recursive(node: &mut TrieNode<V>, key: &str, value: V) -> Option<V> {
        if key.is_empty() {
            // This node is the target.
            return node.value.replace(value);
        }

        let first_byte = key.as_bytes()[0];

        if let Some(child) = node.children.get_mut(&first_byte) {
            let common = common_prefix_len(&child.fragment, key);

            if common == child.fragment.len() {
                // The child fragment is a full prefix of the remaining key.
                return Self::insert_recursive(child, &key[common..], value);
            }

            // Need to split the child at the common prefix point.
            let child_remaining = child.fragment[common..].to_string();
            let key_remaining = &key[common..];

            // Create a new internal node for the common prefix.
            let mut split_node = TrieNode::new(child.fragment[..common].to_string(), None);

            // Move the existing child under the split node with its remaining fragment.
            let mut old_child = node.children.remove(&first_byte).expect("child must exist");
            old_child.fragment = child_remaining.clone();
            let old_first_byte = if child_remaining.is_empty() {
                // Edge case: common == child.fragment.len() handled above,
                // so child_remaining is non-empty here. This branch is
                // unreachable, but we handle it safely.
                0u8
            } else {
                child_remaining.as_bytes()[0]
            };
            split_node.children.insert(old_first_byte, old_child);

            if key_remaining.is_empty() {
                split_node.value = Some(value);
            } else {
                let new_child = TrieNode::new(key_remaining.to_string(), Some(value));
                split_node
                    .children
                    .insert(key_remaining.as_bytes()[0], Box::new(new_child));
            }

            node.children.insert(first_byte, Box::new(split_node));
            None
        } else {
            // No child starts with this byte — create a leaf.
            let new_node = TrieNode::new(key.to_string(), Some(value));
            node.children.insert(first_byte, Box::new(new_node));
            None
        }
    }

    /// Look up the value associated with `key`.
    pub fn get(&self, key: &str) -> Option<&V> {
        Self::get_recursive(&self.root, key)
    }

    fn get_recursive<'a>(node: &'a TrieNode<V>, key: &str) -> Option<&'a V> {
        if key.is_empty() {
            return node.value.as_ref();
        }

        let first_byte = key.as_bytes()[0];
        let child = node.children.get(&first_byte)?;
        let common = common_prefix_len(&child.fragment, key);

        if common < child.fragment.len() {
            // The key diverges before the end of this child's fragment.
            return None;
        }

        Self::get_recursive(child, &key[common..])
    }

    /// Remove the entry with the given key, returning its value.
    pub fn remove(&mut self, key: &str) -> Option<V> {
        let removed = Self::remove_recursive(&mut self.root, key);
        if removed.is_some() {
            self.len -= 1;
        }
        removed
    }

    fn remove_recursive(node: &mut TrieNode<V>, key: &str) -> Option<V> {
        if key.is_empty() {
            return node.value.take();
        }

        let first_byte = key.as_bytes()[0];

        // We need to check the child, potentially remove from it, and then
        // potentially merge the child if it has only one child remaining.
        let child = node.children.get_mut(&first_byte)?;
        let common = common_prefix_len(&child.fragment, key);
        if common < child.fragment.len() {
            return None;
        }

        let removed = Self::remove_recursive(child, &key[common..]);
        removed.as_ref()?;

        // After removal, check if the child should be cleaned up.
        let child = match node.children.get(&first_byte) {
            Some(c) => c,
            None => return removed,
        };

        if child.value.is_none() && child.children.is_empty() {
            // Child is now empty — remove it entirely.
            node.children.remove(&first_byte);
        } else if child.value.is_none() && child.children.len() == 1 {
            // Child has no value and exactly one grandchild — merge them.
            let mut child = node.children.remove(&first_byte).expect("child exists");
            let (_, grandchild) = child.children.drain().next().expect("one grandchild");
            let merged_fragment = format!("{}{}", child.fragment, grandchild.fragment);
            let mut merged = grandchild;
            merged.fragment = merged_fragment;
            let new_first = if merged.fragment.is_empty() {
                first_byte
            } else {
                merged.fragment.as_bytes()[0]
            };
            node.children.insert(new_first, merged);
        }

        removed
    }

    /// Return all entries whose keys start with the given prefix.
    ///
    /// Results are returned as `(key, &value)` pairs in arbitrary order.
    pub fn prefix_search(&self, prefix: &str) -> Vec<(String, &V)> {
        let mut results = Vec::new();
        self.prefix_search_inner(&self.root, prefix, String::new(), &mut results);
        results
    }

    fn prefix_search_inner<'a>(
        &'a self,
        node: &'a TrieNode<V>,
        remaining_prefix: &str,
        accumulated_key: String,
        results: &mut Vec<(String, &'a V)>,
    ) {
        if remaining_prefix.is_empty() {
            // We've consumed the entire prefix — collect all entries below.
            self.collect_all(node, &accumulated_key, results);
            return;
        }

        let first_byte = remaining_prefix.as_bytes()[0];
        let child = match node.children.get(&first_byte) {
            Some(c) => c,
            None => return,
        };

        let common = common_prefix_len(&child.fragment, remaining_prefix);

        if common < child.fragment.len() && common < remaining_prefix.len() {
            // The prefix diverges before the fragment ends — no match.
            return;
        }

        if common >= remaining_prefix.len() {
            // The prefix is consumed within or at the end of this fragment.
            // Collect all entries below this child.
            let new_key = format!("{}{}", accumulated_key, child.fragment);
            self.collect_all(child, &new_key, results);
        } else {
            // common == child.fragment.len() and there's more prefix to match.
            let new_key = format!("{}{}", accumulated_key, child.fragment);
            self.prefix_search_inner(child, &remaining_prefix[common..], new_key, results);
        }
    }

    fn collect_all<'a>(
        &'a self,
        node: &'a TrieNode<V>,
        current_key: &str,
        results: &mut Vec<(String, &'a V)>,
    ) {
        if let Some(ref v) = node.value {
            results.push((current_key.to_string(), v));
        }
        for child in node.children.values() {
            let child_key = format!("{}{}", current_key, child.fragment);
            self.collect_all(child, &child_key, results);
        }
    }

    /// Find the longest prefix of `key` that exists in the trie.
    ///
    /// Returns `(matched_prefix, &value)` or `None` if no prefix matches.
    pub fn longest_prefix(&self, key: &str) -> Option<(String, &V)> {
        let mut best: Option<(String, &V)> = None;
        self.longest_prefix_inner(&self.root, key, String::new(), &mut best);
        best
    }

    fn longest_prefix_inner<'a>(
        &'a self,
        node: &'a TrieNode<V>,
        remaining: &str,
        accumulated: String,
        best: &mut Option<(String, &'a V)>,
    ) {
        // If this node has a value, it's a candidate for longest prefix.
        if let Some(ref v) = node.value {
            *best = Some((accumulated.clone(), v));
        }

        if remaining.is_empty() {
            return;
        }

        let first_byte = remaining.as_bytes()[0];
        let child = match node.children.get(&first_byte) {
            Some(c) => c,
            None => return,
        };

        let common = common_prefix_len(&child.fragment, remaining);
        if common < child.fragment.len() {
            // The key diverges before the end of this fragment —
            // cannot descend further.
            return;
        }

        let new_acc = format!("{}{}", accumulated, child.fragment);
        self.longest_prefix_inner(child, &remaining[common..], new_acc, best);
    }

    /// Collect all keys in the trie.
    pub fn keys(&self) -> Vec<String> {
        let mut result = Vec::with_capacity(self.len);
        self.collect_keys(&self.root, "", &mut result);
        result
    }

    fn collect_keys(&self, node: &TrieNode<V>, prefix: &str, result: &mut Vec<String>) {
        if node.value.is_some() {
            result.push(prefix.to_string());
        }
        for child in node.children.values() {
            let child_key = format!("{}{}", prefix, child.fragment);
            self.collect_keys(child, &child_key, result);
        }
    }

    /// Check whether the trie contains the given key.
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}

impl<V> Default for CompressedTrie<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Return the length (in bytes) of the longest common prefix of `a` and `b`.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut trie = CompressedTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);

        assert_eq!(trie.get("hello"), Some(&1));
        assert_eq!(trie.get("help"), Some(&2));
        assert_eq!(trie.get("world"), Some(&3));
        assert_eq!(trie.get("hel"), None);
        assert_eq!(trie.get("xyz"), None);
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_prefix_search() {
        let mut trie = CompressedTrie::new();
        trie.insert("apple", 1);
        trie.insert("application", 2);
        trie.insert("apply", 3);
        trie.insert("banana", 4);

        let results = trie.prefix_search("app");
        assert_eq!(results.len(), 3);
        let mut keys: Vec<String> = results.iter().map(|(k, _)| k.clone()).collect();
        keys.sort();
        assert_eq!(keys, vec!["apple", "application", "apply"]);

        let results2 = trie.prefix_search("ban");
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].0, "banana");

        let results3 = trie.prefix_search("xyz");
        assert!(results3.is_empty());
    }

    #[test]
    fn test_longest_prefix() {
        let mut trie = CompressedTrie::new();
        trie.insert("/", "root");
        trie.insert("/api", "api");
        trie.insert("/api/v1", "api_v1");
        trie.insert("/api/v1/users", "users");

        let result = trie.longest_prefix("/api/v1/users/123");
        assert!(result.is_some());
        let (prefix, val) = result.expect("should find a match");
        assert_eq!(prefix, "/api/v1/users");
        assert_eq!(*val, "users");

        let result2 = trie.longest_prefix("/api/v2/data");
        assert!(result2.is_some());
        let (prefix2, val2) = result2.expect("should find /api");
        assert_eq!(prefix2, "/api");
        assert_eq!(*val2, "api");

        let result3 = trie.longest_prefix("/other");
        assert!(result3.is_some());
        let (prefix3, _) = result3.expect("should find /");
        assert_eq!(prefix3, "/");

        // No match at all
        let mut trie2: CompressedTrie<i32> = CompressedTrie::new();
        trie2.insert("abc", 1);
        assert!(trie2.longest_prefix("xyz").is_none());
    }

    #[test]
    fn test_remove() {
        let mut trie = CompressedTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);

        assert_eq!(trie.remove("hello"), Some(1));
        assert_eq!(trie.get("hello"), None);
        assert_eq!(trie.len(), 2);
        // "help" should still work
        assert_eq!(trie.get("help"), Some(&2));

        assert_eq!(trie.remove("nonexistent"), None);
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_path_compression() {
        let mut trie = CompressedTrie::new();
        // Insert a long key — the fragment should be the full string.
        trie.insert("abcdefghij", 1);
        assert_eq!(trie.get("abcdefghij"), Some(&1));
        assert_eq!(trie.len(), 1);

        // Insert a key that shares a prefix — should split.
        trie.insert("abcde12345", 2);
        assert_eq!(trie.get("abcdefghij"), Some(&1));
        assert_eq!(trie.get("abcde12345"), Some(&2));
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_empty_trie() {
        let trie: CompressedTrie<i32> = CompressedTrie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
        assert_eq!(trie.get("anything"), None);
        assert!(trie.keys().is_empty());
        assert!(trie.prefix_search("").is_empty());
        assert!(trie.longest_prefix("test").is_none());
    }

    #[test]
    fn test_keys() {
        let mut trie = CompressedTrie::new();
        trie.insert("cat", 1);
        trie.insert("car", 2);
        trie.insert("card", 3);
        trie.insert("dog", 4);

        let mut keys = trie.keys();
        keys.sort();
        assert_eq!(keys, vec!["car", "card", "cat", "dog"]);
    }

    #[test]
    fn test_overwrite_value() {
        let mut trie = CompressedTrie::new();
        let old = trie.insert("key", 1);
        assert!(old.is_none());

        let old2 = trie.insert("key", 2);
        assert_eq!(old2, Some(1));
        assert_eq!(trie.get("key"), Some(&2));
        assert_eq!(trie.len(), 1); // length unchanged
    }

    #[test]
    fn test_single_char_keys() {
        let mut trie = CompressedTrie::new();
        trie.insert("a", 1);
        trie.insert("b", 2);
        trie.insert("c", 3);

        assert_eq!(trie.get("a"), Some(&1));
        assert_eq!(trie.get("b"), Some(&2));
        assert_eq!(trie.get("c"), Some(&3));
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_contains() {
        let mut trie = CompressedTrie::new();
        trie.insert("foo", 1);
        assert!(trie.contains("foo"));
        assert!(!trie.contains("bar"));
    }

    #[test]
    fn test_prefix_search_with_exact_prefix_match() {
        let mut trie = CompressedTrie::new();
        trie.insert("test", 1);
        trie.insert("testing", 2);
        trie.insert("tester", 3);

        // Prefix that is itself a key
        let results = trie.prefix_search("test");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_remove_with_merge() {
        let mut trie = CompressedTrie::new();
        trie.insert("abc", 1);
        trie.insert("abcdef", 2);
        trie.insert("abcxyz", 3);

        // Remove "abc", which should merge "abcdef" and "abcxyz"
        // if the internal node has no value and one child.
        trie.remove("abc");
        assert_eq!(trie.get("abcdef"), Some(&2));
        assert_eq!(trie.get("abcxyz"), Some(&3));
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_empty_string_key() {
        let mut trie = CompressedTrie::new();
        trie.insert("", 42);
        assert_eq!(trie.get(""), Some(&42));
        assert_eq!(trie.len(), 1);

        trie.insert("a", 1);
        assert_eq!(trie.len(), 2);
        assert_eq!(trie.get(""), Some(&42));

        // Prefix search with empty prefix should return everything
        let results = trie.prefix_search("");
        assert_eq!(results.len(), 2);
    }
}
