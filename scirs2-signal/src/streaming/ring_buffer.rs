//! Generic ring buffer (circular buffer) for efficient overlap management.
//!
//! [`RingBuffer`] provides O(1) push / pop and contiguous slice access that is
//! essential for the overlap book-keeping in streaming STFT, FIR filters, and
//! envelope followers.

use crate::error::{SignalError, SignalResult};

/// A fixed-capacity circular buffer backed by a contiguous `Vec<T>`.
///
/// Internally we keep the data in a plain `Vec` and maintain a virtual write
/// cursor so that the oldest sample is always at index 0 when viewed through
/// the public API.  This avoids the wrap-around complexity of a traditional
/// ring buffer while still offering O(1) amortised push/pop.
#[derive(Debug, Clone)]
pub struct RingBuffer<T: Clone + Default> {
    /// Internal storage (always exactly `capacity` elements once full).
    data: Vec<T>,
    /// Maximum number of elements.
    capacity: usize,
    /// Number of valid elements currently stored.
    len: usize,
    /// Write position (points to the *next* slot to write).
    write_pos: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if `capacity` is 0.
    pub fn new(capacity: usize) -> SignalResult<Self> {
        if capacity == 0 {
            return Err(SignalError::ValueError(
                "Ring buffer capacity must be > 0".to_string(),
            ));
        }
        Ok(Self {
            data: vec![T::default(); capacity],
            capacity,
            len: 0,
            write_pos: 0,
        })
    }

    /// Current number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Maximum capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Push a single element.  If the buffer is already full the oldest element
    /// is silently overwritten.
    pub fn push(&mut self, value: T) {
        self.data[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Push a slice of elements.  Overwrites oldest data when full.
    pub fn push_slice(&mut self, values: &[T]) {
        for v in values {
            self.push(v.clone());
        }
    }

    /// Read the element at logical index `idx` (0 = oldest).
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if `idx >= len`.
    pub fn get(&self, idx: usize) -> SignalResult<&T> {
        if idx >= self.len {
            return Err(SignalError::ValueError(format!(
                "Index {idx} out of range (len = {})",
                self.len
            )));
        }
        let physical = self.logical_to_physical(idx);
        Ok(&self.data[physical])
    }

    /// Return a snapshot of the buffer contents in chronological order
    /// (oldest first).
    pub fn as_ordered_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let phys = self.logical_to_physical(i);
            out.push(self.data[phys].clone());
        }
        out
    }

    /// Return a contiguous slice of the most recent `n` elements (oldest of
    /// those `n` first).
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if `n > len`.
    pub fn last_n(&self, n: usize) -> SignalResult<Vec<T>> {
        if n > self.len {
            return Err(SignalError::ValueError(format!(
                "Requested {n} elements but only {} available",
                self.len
            )));
        }
        let start = self.len - n;
        let mut out = Vec::with_capacity(n);
        for i in start..self.len {
            let phys = self.logical_to_physical(i);
            out.push(self.data[phys].clone());
        }
        Ok(out)
    }

    /// Peek at the most recent element without removing it.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is empty.
    pub fn peek_last(&self) -> SignalResult<&T> {
        if self.len == 0 {
            return Err(SignalError::ValueError(
                "Cannot peek into empty ring buffer".to_string(),
            ));
        }
        let phys = if self.write_pos == 0 {
            self.capacity - 1
        } else {
            self.write_pos - 1
        };
        Ok(&self.data[phys])
    }

    /// Peek at the oldest element without removing it.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is empty.
    pub fn peek_first(&self) -> SignalResult<&T> {
        if self.len == 0 {
            return Err(SignalError::ValueError(
                "Cannot peek into empty ring buffer".to_string(),
            ));
        }
        let phys = self.logical_to_physical(0);
        Ok(&self.data[phys])
    }

    /// Remove and return the oldest element.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is empty.
    pub fn pop_front(&mut self) -> SignalResult<T> {
        if self.len == 0 {
            return Err(SignalError::ValueError(
                "Cannot pop from empty ring buffer".to_string(),
            ));
        }
        let phys = self.logical_to_physical(0);
        let val = self.data[phys].clone();
        self.len -= 1;
        // No need to adjust write_pos; logical_to_physical handles it.
        Ok(val)
    }

    /// Discard the oldest `n` elements.
    ///
    /// If `n >= len`, the buffer is simply cleared.
    pub fn advance(&mut self, n: usize) {
        if n >= self.len {
            self.len = 0;
        } else {
            self.len -= n;
        }
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.len = 0;
        self.write_pos = 0;
    }

    // ---- internal helpers ----

    /// Map logical index (0 = oldest) to physical storage index.
    #[inline]
    fn logical_to_physical(&self, logical: usize) -> usize {
        // The oldest element sits `len` positions behind `write_pos`.
        let start = if self.write_pos >= self.len {
            self.write_pos - self.len
        } else {
            self.capacity - (self.len - self.write_pos)
        };
        (start + logical) % self.capacity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_and_get() {
        let mut rb = RingBuffer::<f64>::new(4).expect("Failed to create ring buffer");
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);

        assert_eq!(rb.len(), 3);
        assert!(!rb.is_full());

        let v0 = rb.get(0).expect("get(0) failed");
        let v1 = rb.get(1).expect("get(1) failed");
        let v2 = rb.get(2).expect("get(2) failed");
        assert!((v0 - 1.0).abs() < 1e-15);
        assert!((v1 - 2.0).abs() < 1e-15);
        assert!((v2 - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_overwrite_oldest() {
        let mut rb = RingBuffer::<f64>::new(3).expect("Failed to create ring buffer");
        rb.push(10.0);
        rb.push(20.0);
        rb.push(30.0);
        assert!(rb.is_full());

        // Overwrite oldest
        rb.push(40.0);
        assert_eq!(rb.len(), 3);

        let ordered = rb.as_ordered_vec();
        assert!((ordered[0] - 20.0).abs() < 1e-15);
        assert!((ordered[1] - 30.0).abs() < 1e-15);
        assert!((ordered[2] - 40.0).abs() < 1e-15);
    }

    #[test]
    fn test_push_slice() {
        let mut rb = RingBuffer::<f64>::new(5).expect("Failed to create ring buffer");
        rb.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        assert_eq!(rb.len(), 5);
        let ordered = rb.as_ordered_vec();
        assert!((ordered[0] - 3.0).abs() < 1e-15);
        assert!((ordered[4] - 7.0).abs() < 1e-15);
    }

    #[test]
    fn test_last_n() {
        let mut rb = RingBuffer::<f64>::new(6).expect("Failed to create ring buffer");
        for i in 1..=6 {
            rb.push(i as f64);
        }
        let last3 = rb.last_n(3).expect("last_n failed");
        assert!((last3[0] - 4.0).abs() < 1e-15);
        assert!((last3[1] - 5.0).abs() < 1e-15);
        assert!((last3[2] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_pop_front() {
        let mut rb = RingBuffer::<f64>::new(4).expect("Failed to create ring buffer");
        rb.push_slice(&[10.0, 20.0, 30.0]);

        let oldest = rb.pop_front().expect("pop_front failed");
        assert!((oldest - 10.0).abs() < 1e-15);
        assert_eq!(rb.len(), 2);

        let next = rb.pop_front().expect("pop_front failed");
        assert!((next - 20.0).abs() < 1e-15);
    }

    #[test]
    fn test_peek() {
        let mut rb = RingBuffer::<f64>::new(4).expect("Failed to create ring buffer");
        rb.push_slice(&[5.0, 6.0, 7.0]);

        let first = rb.peek_first().expect("peek_first failed");
        assert!((first - 5.0).abs() < 1e-15);

        let last = rb.peek_last().expect("peek_last failed");
        assert!((last - 7.0).abs() < 1e-15);
    }

    #[test]
    fn test_advance_and_clear() {
        let mut rb = RingBuffer::<f64>::new(5).expect("Failed to create ring buffer");
        rb.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        rb.advance(2);
        assert_eq!(rb.len(), 3);
        let first = rb.peek_first().expect("peek_first failed");
        assert!((first - 3.0).abs() < 1e-15);

        rb.clear();
        assert!(rb.is_empty());
    }

    #[test]
    fn test_zero_capacity_error() {
        let result = RingBuffer::<f64>::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_out_of_range() {
        let rb = RingBuffer::<f64>::new(3).expect("Failed to create ring buffer");
        assert!(rb.get(0).is_err());
    }

    #[test]
    fn test_pop_empty() {
        let mut rb = RingBuffer::<f64>::new(3).expect("Failed to create ring buffer");
        assert!(rb.pop_front().is_err());
    }
}
