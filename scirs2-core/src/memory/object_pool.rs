//! Generic object pool with RAII guard and optional thread-safe variant.
//!
//! An [`ObjectPool<T>`] keeps a stock of reusable objects, avoiding repeated
//! allocation/initialisation overhead.  Objects are obtained via
//! [`ObjectPool::acquire`] which returns a [`PoolGuard<T>`]; when the guard is
//! dropped the object is automatically returned to the pool (after a
//! user-supplied *reset* function clears any mutation).
//!
//! Interior mutability is used so that multiple [`PoolGuard`]s can coexist
//! without borrow-checker conflicts.  [`ObjectPool<T>`] is cheaply cloneable
//! — every clone shares the same backing pool.
//!
//! The thread-safe variant [`SyncObjectPool<T>`] is an alias for the same
//! type, since the interior mutex already makes both types safe across threads
//! when `T: Send`.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::memory::object_pool::ObjectPool;
//!
//! let pool: ObjectPool<Vec<u8>> = ObjectPool::builder()
//!     .factory(|| Vec::with_capacity(256))
//!     .reset(|v| v.clear())
//!     .max_size(8)
//!     .build();
//!
//! {
//!     let mut guard = pool.acquire();
//!     guard.extend_from_slice(b"hello");
//!     assert_eq!(&*guard, b"hello");
//! } // returned to pool here
//!
//! assert_eq!(pool.pool_size(), 1);
//! ```

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// PoolState — the shared mutable core
// ---------------------------------------------------------------------------

struct PoolState<T> {
    idle: Vec<T>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    reset: Box<dyn Fn(&mut T) + Send + Sync>,
    max_size: usize,
    total_acquired: usize,
    in_use: usize,
}

impl<T> PoolState<T> {
    fn new<F, R>(factory: F, reset: R, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
        R: Fn(&mut T) + Send + Sync + 'static,
    {
        PoolState {
            idle: Vec::new(),
            factory: Box::new(factory),
            reset: Box::new(reset),
            max_size,
            total_acquired: 0,
            in_use: 0,
        }
    }

    fn acquire_obj(&mut self) -> T {
        let obj = self.idle.pop().unwrap_or_else(|| (self.factory)());
        self.total_acquired += 1;
        self.in_use += 1;
        obj
    }

    fn return_obj(&mut self, mut obj: T) {
        self.in_use = self.in_use.saturating_sub(1);
        if self.idle.len() < self.max_size {
            (self.reset)(&mut obj);
            self.idle.push(obj);
        }
        // Otherwise obj is dropped here.
    }
}

// ---------------------------------------------------------------------------
// ObjectPool<T>
// ---------------------------------------------------------------------------

/// A generic object pool for reusing pre-initialised objects.
///
/// Uses interior mutability (`Arc<Mutex<...>>`) so that multiple [`PoolGuard`]
/// values can coexist without borrow conflicts.  Cloning the pool gives another
/// handle to the **same** backing pool.
///
/// For single-threaded use `T` need not be `Send`.  For multi-threaded use
/// the standard `Arc<Mutex<...>>` requirements apply (i.e. `T: Send`).
#[derive(Clone)]
pub struct ObjectPool<T: 'static> {
    state: Arc<Mutex<PoolState<T>>>,
}

impl<T: 'static> ObjectPool<T> {
    /// Start building an `ObjectPool` with a fluent builder.
    pub fn builder() -> ObjectPoolBuilder<T> {
        ObjectPoolBuilder::new()
    }

    /// Create a pool from explicit factory and reset closures.
    ///
    /// * `factory` — called to create a new object when the pool is empty.
    /// * `reset`   — called on an object before it is returned to the pool.
    /// * `max_size` — maximum number of idle objects kept; excess objects are
    ///   dropped instead of returned.
    pub fn new<F, R>(factory: F, reset: R, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
        R: Fn(&mut T) + Send + Sync + 'static,
    {
        ObjectPool {
            state: Arc::new(Mutex::new(PoolState::new(factory, reset, max_size))),
        }
    }

    /// Lock the pool state, returning an error if the mutex is poisoned.
    fn lock_state(&self) -> std::sync::MutexGuard<'_, PoolState<T>> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take an object from the pool (or create one) and return a RAII guard.
    ///
    /// The guard implements `Deref<Target = T>` and `DerefMut<Target = T>`.
    /// When the guard is dropped the object is returned to the pool (if the
    /// pool still has room) and the reset function is called.
    ///
    /// Multiple guards can be held simultaneously from the same pool without
    /// any borrow-checker issues.
    pub fn acquire(&self) -> PoolGuard<T> {
        let obj = self.lock_state().acquire_obj();
        PoolGuard {
            obj: Some(obj),
            pool: Arc::clone(&self.state),
        }
    }

    /// Pre-populate the pool with `n` freshly-created objects.
    ///
    /// Useful when you want to avoid any latency on the first `acquire` calls.
    /// If `n` is larger than `max_size` only `max_size` objects are added.
    pub fn warm_up(&self, n: usize) {
        let mut state = self.lock_state();
        let to_add = n.min(state.max_size.saturating_sub(state.idle.len()));
        for _ in 0..to_add {
            let obj = (state.factory)();
            state.idle.push(obj);
        }
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    /// Number of idle objects currently sitting in the pool.
    #[inline]
    pub fn pool_size(&self) -> usize {
        self.lock_state().idle.len()
    }

    /// Number of objects currently checked out (acquired but not yet dropped).
    #[inline]
    pub fn in_use_count(&self) -> usize {
        self.lock_state().in_use
    }

    /// Total number of `acquire` calls since the pool was created.
    #[inline]
    pub fn total_acquired(&self) -> usize {
        self.lock_state().total_acquired
    }

    /// Maximum number of idle objects the pool will retain.
    #[inline]
    pub fn max_size(&self) -> usize {
        self.lock_state().max_size
    }
}

// ---------------------------------------------------------------------------
// PoolGuard<T>
// ---------------------------------------------------------------------------

/// RAII wrapper returned by [`ObjectPool::acquire`].
///
/// Implements `Deref` and `DerefMut` to `T`.  On drop the inner object is
/// returned to the pool (via the reset closure).
pub struct PoolGuard<T: 'static> {
    /// `None` only transiently during `drop`.
    obj: Option<T>,
    pool: Arc<Mutex<PoolState<T>>>,
}

impl<T: 'static> Deref for PoolGuard<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.obj.as_ref().expect("PoolGuard already dropped")
    }
}

impl<T: 'static> DerefMut for PoolGuard<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.obj.as_mut().expect("PoolGuard already dropped")
    }
}

impl<T: 'static> Drop for PoolGuard<T> {
    fn drop(&mut self) {
        if let Some(obj) = self.obj.take() {
            if let Ok(mut state) = self.pool.lock() {
                state.return_obj(obj);
            }
            // If poisoned, obj is simply dropped.
        }
    }
}

// ---------------------------------------------------------------------------
// ObjectPoolBuilder<T>
// ---------------------------------------------------------------------------

/// Fluent builder for [`ObjectPool<T>`].
pub struct ObjectPoolBuilder<T> {
    factory: Option<Box<dyn Fn() -> T + Send + Sync>>,
    reset: Option<Box<dyn Fn(&mut T) + Send + Sync>>,
    max_size: usize,
}

impl<T: Default + 'static> ObjectPoolBuilder<T> {
    /// Use `T::default()` as the factory.
    pub fn default_factory(self) -> Self {
        self.factory(T::default)
    }
}

impl<T: 'static> ObjectPoolBuilder<T> {
    fn new() -> Self {
        ObjectPoolBuilder {
            factory: None,
            reset: None,
            max_size: 32,
        }
    }

    /// Set the factory closure.
    pub fn factory<F: Fn() -> T + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.factory = Some(Box::new(f));
        self
    }

    /// Set the reset closure (called before an object is returned to the pool).
    pub fn reset<R: Fn(&mut T) + Send + Sync + 'static>(mut self, r: R) -> Self {
        self.reset = Some(Box::new(r));
        self
    }

    /// Set the maximum number of idle objects kept in the pool.
    pub fn max_size(mut self, n: usize) -> Self {
        self.max_size = n;
        self
    }

    /// Consume the builder and construct the pool.
    ///
    /// # Panics
    ///
    /// Panics if no `factory` was provided.
    pub fn build(self) -> ObjectPool<T> {
        let factory = self
            .factory
            .expect("ObjectPoolBuilder: factory is required");
        let reset = self.reset.unwrap_or_else(|| Box::new(|_| {}));
        ObjectPool::new(factory, reset, self.max_size)
    }
}

// ---------------------------------------------------------------------------
// SyncObjectPool<T>  — thread-safe variant (type alias for ObjectPool<T>)
// ---------------------------------------------------------------------------

/// Thread-safe object pool.
///
/// This is a type alias for [`ObjectPool<T>`], which already uses interior
/// mutability (`Arc<Mutex<...>>`).  It is provided for backward-compatibility
/// and API clarity.  Multiple handles to the same pool can be obtained by
/// cloning.
///
/// # Example
///
/// ```rust
/// use scirs2_core::memory::object_pool::SyncObjectPool;
///
/// let pool = SyncObjectPool::<Vec<u8>>::new(
///     || Vec::with_capacity(64),
///     |v| v.clear(),
///     16,
/// );
///
/// let pool2 = pool.clone();
/// std::thread::spawn(move || {
///     let mut guard = pool2.acquire();
///     guard.push(42);
/// });
/// ```
pub type SyncObjectPool<T> = ObjectPool<T>;

/// RAII guard returned by [`SyncObjectPool::acquire`].
///
/// This is a type alias for [`PoolGuard<T>`].
pub type SyncPoolGuard<T> = PoolGuard<T>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Helper: pool of Vec<u8> with capacity-reserving factory.
    fn vec_pool(max: usize) -> ObjectPool<Vec<u8>> {
        ObjectPool::builder()
            .factory(|| Vec::with_capacity(64))
            .reset(|v| v.clear())
            .max_size(max)
            .build()
    }

    #[test]
    fn pool_basic_acquire_and_return() {
        let pool = vec_pool(4);
        assert_eq!(pool.pool_size(), 0);
        {
            let mut guard = pool.acquire();
            guard.extend_from_slice(b"hello");
            assert_eq!(&*guard, b"hello");
            assert_eq!(pool.in_use_count(), 1);
        }
        assert_eq!(pool.pool_size(), 1);
        assert_eq!(pool.in_use_count(), 0);
        assert_eq!(pool.total_acquired(), 1);
    }

    #[test]
    fn pool_reset_is_called() {
        let pool = vec_pool(4);
        {
            let mut guard = pool.acquire();
            guard.push(1);
            guard.push(2);
        }
        // Acquire again; reset should have cleared it.
        let guard = pool.acquire();
        assert!(guard.is_empty(), "reset should have cleared the vec");
    }

    #[test]
    fn pool_respects_max_size() {
        let pool = vec_pool(2);
        let g1 = pool.acquire();
        let g2 = pool.acquire();
        let g3 = pool.acquire();
        drop(g1);
        drop(g2);
        drop(g3); // third object should be dropped, not returned
        assert_eq!(pool.pool_size(), 2);
    }

    #[test]
    fn pool_warm_up() {
        let pool = vec_pool(8);
        pool.warm_up(5);
        assert_eq!(pool.pool_size(), 5);
    }

    #[test]
    fn pool_statistics() {
        let pool = vec_pool(8);
        let _g1 = pool.acquire();
        let _g2 = pool.acquire();
        assert_eq!(pool.total_acquired(), 2);
        assert_eq!(pool.in_use_count(), 2);
    }

    #[test]
    fn pool_builder_default_factory() {
        let pool: ObjectPool<Vec<i32>> = ObjectPool::builder()
            .default_factory()
            .reset(|v: &mut Vec<i32>| v.clear())
            .max_size(4)
            .build();
        let guard = pool.acquire();
        assert!(guard.is_empty());
    }

    #[test]
    fn pool_multiple_guards_coexist() {
        // This test would previously fail to compile with E0499/E0502.
        let pool = vec_pool(4);
        let g1 = pool.acquire();
        let g2 = pool.acquire();
        let g3 = pool.acquire();
        assert_eq!(pool.in_use_count(), 3);
        assert_eq!(g1.len(), 0);
        assert_eq!(g2.len(), 0);
        assert_eq!(g3.len(), 0);
        drop(g1);
        drop(g2);
        drop(g3);
        assert_eq!(pool.pool_size(), 3);
    }

    // --- SyncObjectPool tests ---

    #[test]
    fn sync_pool_basic() {
        let pool = SyncObjectPool::new(|| Vec::<u8>::with_capacity(32), |v| v.clear(), 8);
        {
            let mut guard = pool.acquire();
            guard.push(99);
            assert_eq!(*guard, vec![99]);
        }
        assert_eq!(pool.pool_size(), 1);
        assert_eq!(pool.in_use_count(), 0);
        assert_eq!(pool.total_acquired(), 1);
    }

    #[test]
    fn sync_pool_multithreaded() {
        let create_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&create_count);
        let pool = SyncObjectPool::new(
            move || {
                cc.fetch_add(1, Ordering::Relaxed);
                Vec::<u8>::new()
            },
            |v| v.clear(),
            4,
        );

        let mut handles = Vec::new();
        for _ in 0..8 {
            let p = pool.clone();
            handles.push(std::thread::spawn(move || {
                let mut guard = p.acquire();
                guard.push(1);
                std::thread::sleep(std::time::Duration::from_millis(5));
                // guard dropped here
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }

        // All 8 acquires completed.
        assert_eq!(pool.total_acquired(), 8);
        // At most max_size (4) objects retained in the pool.
        assert!(pool.pool_size() <= 4);
    }

    #[test]
    fn sync_pool_warm_up() {
        let pool = SyncObjectPool::new(|| 0_u32, |_| {}, 8);
        pool.warm_up(5);
        assert_eq!(pool.pool_size(), 5);
    }

    #[test]
    fn pool_guard_deref() {
        let pool: ObjectPool<String> = ObjectPool::builder()
            .factory(String::new)
            .reset(|s| s.clear())
            .max_size(4)
            .build();
        let mut guard = pool.acquire();
        guard.push_str("hello");
        assert_eq!(guard.as_str(), "hello");
        assert_eq!(guard.len(), 5);
    }
}
