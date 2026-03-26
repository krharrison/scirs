//! Barrier and latch synchronisation primitives.
//!
//! # Primitives
//!
//! | Type | Description |
//! |------|-------------|
//! | [`CyclicBarrier`] | Reusable barrier — all threads wait until every expected thread arrives, then all proceed; can be reused for the next phase automatically. |
//! | [`PhaseBarrier`] | Phased barrier that tracks a monotonically-increasing phase counter; useful for iterative algorithms. |
//! | [`CountDownLatch`] | Single-use latch that opens after `N` count-down calls. |
//! | [`SpinBarrier`] | Spin-wait barrier with optional yield — zero OS involvement; best for very short waits across a fixed number of threads. |
//!
//! All types are `Send + Sync` and avoid `unwrap()`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ── helpers ──────────────────────────────────────────────────────────────────

fn lock_err(context: &'static str, e: impl std::fmt::Display) -> CoreError {
    CoreError::MutexError(
        ErrorContext::new(format!("{context}: mutex poisoned: {e}"))
            .with_location(ErrorLocation::new(file!(), line!())),
    )
}

fn wait_err(context: &'static str, e: impl std::fmt::Display) -> CoreError {
    CoreError::MutexError(
        ErrorContext::new(format!("{context}: condvar wait poisoned: {e}"))
            .with_location(ErrorLocation::new(file!(), line!())),
    )
}

// ── CyclicBarrier ─────────────────────────────────────────────────────────────

/// State shared between all parties to the barrier.
struct CyclicBarrierInner {
    /// Number of threads that still need to arrive this cycle.
    waiting: usize,
    /// Total parties expected each cycle.
    parties: usize,
    /// Monotonically-increasing generation; incremented each time all parties arrive.
    generation: u64,
    /// Set to `true` after [`CyclicBarrier::break_barrier`] is called.
    broken: bool,
}

/// A reusable barrier synchronisation aid.
///
/// `n` threads call [`wait`](CyclicBarrier::wait).  The call blocks until all
/// `n` threads have arrived, at which point all calls return and the barrier
/// automatically resets for the next cycle.  This is safe to use across
/// multiple "rounds" without re-creating the struct.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::barrier::CyclicBarrier;
/// use std::sync::Arc;
///
/// let barrier = Arc::new(CyclicBarrier::new(3));
/// let mut handles = Vec::new();
/// for _ in 0..3 {
///     let b = Arc::clone(&barrier);
///     handles.push(std::thread::spawn(move || b.wait().expect("barrier wait")));
/// }
/// for h in handles { h.join().expect("thread"); }
/// ```
pub struct CyclicBarrier {
    inner: Mutex<CyclicBarrierInner>,
    condvar: Condvar,
}

impl CyclicBarrier {
    /// Create a barrier for `parties` threads.
    ///
    /// # Panics
    ///
    /// Does not panic — returns a valid barrier even for `parties == 0` (which
    /// passes immediately).
    pub fn new(parties: usize) -> Self {
        Self {
            inner: Mutex::new(CyclicBarrierInner {
                waiting: parties,
                parties,
                generation: 0,
                broken: false,
            }),
            condvar: Condvar::new(),
        }
    }

    /// Wait until all parties have arrived.
    ///
    /// Returns `Ok(true)` for exactly one thread per cycle (the "trip" thread
    /// that was the last to arrive).  All other threads get `Ok(false)`.
    pub fn wait(&self) -> CoreResult<bool> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("CyclicBarrier::wait", e))?;

        if g.broken {
            return Err(CoreError::MutexError(ErrorContext::new(
                "CyclicBarrier: barrier is broken",
            )));
        }

        let gen = g.generation;
        g.waiting -= 1;

        if g.waiting == 0 {
            // Last thread: reset barrier and wake everyone.
            g.waiting = g.parties;
            g.generation = gen.wrapping_add(1);
            self.condvar.notify_all();
            return Ok(true);
        }

        // Not the last — wait for the generation to change.
        loop {
            g = self
                .condvar
                .wait(g)
                .map_err(|e| wait_err("CyclicBarrier::wait", e))?;
            if g.broken {
                return Err(CoreError::MutexError(ErrorContext::new(
                    "CyclicBarrier: barrier broken while waiting",
                )));
            }
            if g.generation != gen {
                return Ok(false);
            }
        }
    }

    /// Wait with a timeout.  Returns `Err` on timeout or if the barrier is broken.
    pub fn wait_timeout(&self, timeout: Duration) -> CoreResult<bool> {
        let deadline = Instant::now() + timeout;
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("CyclicBarrier::wait_timeout", e))?;

        if g.broken {
            return Err(CoreError::MutexError(ErrorContext::new(
                "CyclicBarrier: barrier is broken",
            )));
        }

        let gen = g.generation;
        g.waiting -= 1;

        if g.waiting == 0 {
            g.waiting = g.parties;
            g.generation = gen.wrapping_add(1);
            self.condvar.notify_all();
            return Ok(true);
        }

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Timed out — mark barrier as broken to unblock others.
                g.broken = true;
                self.condvar.notify_all();
                return Err(CoreError::TimeoutError(ErrorContext::new(
                    "CyclicBarrier: timed out waiting for all parties",
                )));
            }
            let (next_g, _timeout_result) = self
                .condvar
                .wait_timeout(g, remaining)
                .map_err(|e| wait_err("CyclicBarrier::wait_timeout", e))?;
            g = next_g;
            if g.broken {
                return Err(CoreError::MutexError(ErrorContext::new(
                    "CyclicBarrier: barrier broken while waiting",
                )));
            }
            if g.generation != gen {
                return Ok(false);
            }
        }
    }

    /// Forcibly break the barrier.  All waiting threads receive an error.
    pub fn break_barrier(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.broken = true;
            self.condvar.notify_all();
        }
    }

    /// Reset the barrier to its initial state.
    pub fn reset(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.waiting = g.parties;
            g.broken = false;
            g.generation = g.generation.wrapping_add(1);
            self.condvar.notify_all();
        }
    }

    /// Returns `true` if the barrier is broken.
    pub fn is_broken(&self) -> bool {
        self.inner.lock().map(|g| g.broken).unwrap_or(true)
    }

    /// Number of parties that have not yet arrived in the current cycle.
    pub fn waiting(&self) -> usize {
        self.inner.lock().map(|g| g.waiting).unwrap_or(0)
    }

    /// The total number of parties.
    pub fn parties(&self) -> usize {
        self.inner.lock().map(|g| g.parties).unwrap_or(0)
    }
}

// ── PhaseBarrier ─────────────────────────────────────────────────────────────

/// Internal state for [`PhaseBarrier`].
struct PhaseBarrierInner {
    phase: u64,
    waiting: usize,
    parties: usize,
}

/// A phased barrier with a monotonically-advancing phase counter.
///
/// Conceptually identical to [`CyclicBarrier`] but exposes the *phase number*
/// so that callers can check which phase they are currently in.  Useful for
/// iterative parallel algorithms where each phase represents one iteration.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::barrier::PhaseBarrier;
/// use std::sync::Arc;
///
/// let pb = Arc::new(PhaseBarrier::new(2));
/// let pb2 = Arc::clone(&pb);
/// let t = std::thread::spawn(move || pb2.arrive_and_wait().expect("wait"));
/// pb.arrive_and_wait().expect("wait");
/// t.join().expect("thread");
/// assert!(pb.phase() >= 1);
/// ```
pub struct PhaseBarrier {
    inner: Mutex<PhaseBarrierInner>,
    condvar: Condvar,
}

impl PhaseBarrier {
    /// Create a phased barrier for `parties` threads.
    pub fn new(parties: usize) -> Self {
        Self {
            inner: Mutex::new(PhaseBarrierInner {
                phase: 0,
                waiting: parties,
                parties,
            }),
            condvar: Condvar::new(),
        }
    }

    /// Arrive at the barrier and wait for all other parties.
    ///
    /// Returns the phase number that just completed.
    pub fn arrive_and_wait(&self) -> CoreResult<u64> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("PhaseBarrier::arrive_and_wait", e))?;

        let current_phase = g.phase;
        g.waiting -= 1;

        if g.waiting == 0 {
            // Advance phase and reset counter.
            g.phase = current_phase.wrapping_add(1);
            g.waiting = g.parties;
            self.condvar.notify_all();
            return Ok(current_phase);
        }

        loop {
            g = self
                .condvar
                .wait(g)
                .map_err(|e| wait_err("PhaseBarrier::arrive_and_wait", e))?;
            if g.phase != current_phase {
                return Ok(current_phase);
            }
        }
    }

    /// Arrive but do NOT wait (signal and return immediately).
    ///
    /// Returns the phase number advanced to, or the current phase if other
    /// parties are still pending.
    pub fn arrive(&self) -> CoreResult<u64> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("PhaseBarrier::arrive", e))?;

        g.waiting -= 1;
        if g.waiting == 0 {
            let completed = g.phase;
            g.phase = completed.wrapping_add(1);
            g.waiting = g.parties;
            self.condvar.notify_all();
            Ok(completed)
        } else {
            Ok(g.phase)
        }
    }

    /// Current phase number.
    pub fn phase(&self) -> u64 {
        self.inner.lock().map(|g| g.phase).unwrap_or(0)
    }

    /// Number of parties that have not yet arrived this phase.
    pub fn waiting(&self) -> usize {
        self.inner.lock().map(|g| g.waiting).unwrap_or(0)
    }
}

// ── CountDownLatch ────────────────────────────────────────────────────────────

/// A single-use latch that opens when its counter reaches zero.
///
/// Any number of threads may call [`wait`](CountDownLatch::wait) which blocks
/// until the internal counter reaches 0.  The counter is decremented by calling
/// [`count_down`](CountDownLatch::count_down).
///
/// Once opened the latch *stays open*; subsequent calls to `wait` return
/// immediately.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::barrier::CountDownLatch;
/// use std::sync::Arc;
///
/// let latch = Arc::new(CountDownLatch::new(3));
/// let mut handles = Vec::new();
/// for _ in 0..3 {
///     let l = Arc::clone(&latch);
///     handles.push(std::thread::spawn(move || l.count_down()));
/// }
/// latch.wait().expect("latch wait");
/// for h in handles { h.join().expect("thread"); }
/// ```
pub struct CountDownLatch {
    inner: Mutex<usize>,
    condvar: Condvar,
}

impl CountDownLatch {
    /// Create a latch with an initial count of `n`.
    pub fn new(n: usize) -> Self {
        Self {
            inner: Mutex::new(n),
            condvar: Condvar::new(),
        }
    }

    /// Decrement the count by 1.  When it reaches 0 all waiting threads wake.
    pub fn count_down(&self) {
        if let Ok(mut g) = self.inner.lock() {
            if *g > 0 {
                *g -= 1;
                if *g == 0 {
                    self.condvar.notify_all();
                }
            }
        }
    }

    /// Block until the count reaches 0.
    pub fn wait(&self) -> CoreResult<()> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("CountDownLatch::wait", e))?;
        loop {
            if *g == 0 {
                return Ok(());
            }
            g = self
                .condvar
                .wait(g)
                .map_err(|e| wait_err("CountDownLatch::wait", e))?;
        }
    }

    /// Block until the count reaches 0 or `timeout` elapses.
    ///
    /// Returns `true` if the latch opened within the timeout.
    pub fn wait_timeout(&self, timeout: Duration) -> CoreResult<bool> {
        let deadline = Instant::now() + timeout;
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("CountDownLatch::wait_timeout", e))?;
        loop {
            if *g == 0 {
                return Ok(true);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            let (next_g, _) = self
                .condvar
                .wait_timeout(g, remaining)
                .map_err(|e| wait_err("CountDownLatch::wait_timeout", e))?;
            g = next_g;
        }
    }

    /// Current count (informational, not synchronisation-safe on its own).
    pub fn count(&self) -> usize {
        self.inner.lock().map(|g| *g).unwrap_or(0)
    }

    /// Returns `true` if the latch has already opened.
    pub fn is_open(&self) -> bool {
        self.count() == 0
    }
}

// ── SpinBarrier ──────────────────────────────────────────────────────────────

/// A spin-wait barrier backed by a single atomic counter.
///
/// No OS involvement — threads busy-spin (with optional `yield_now`) until all
/// arrive.  Appropriate only for very short synchronisation gaps (e.g., < 1 µs
/// expected wait).  For longer waits prefer [`CyclicBarrier`].
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::barrier::SpinBarrier;
/// use std::sync::Arc;
///
/// let b = Arc::new(SpinBarrier::new(2));
/// let b2 = Arc::clone(&b);
/// let t = std::thread::spawn(move || b2.wait());
/// b.wait();
/// t.join().expect("thread");
/// ```
pub struct SpinBarrier {
    /// Number of parties still to arrive.
    arrived: AtomicUsize,
    /// Monotonically-increasing epoch; each full round increments it.
    epoch: AtomicUsize,
    parties: usize,
}

impl SpinBarrier {
    /// Create a spin barrier for `parties` threads.
    pub fn new(parties: usize) -> Self {
        let parties = parties.max(1);
        Self {
            arrived: AtomicUsize::new(0),
            epoch: AtomicUsize::new(0),
            parties,
        }
    }

    /// Spin-wait until all parties have called `wait`.
    ///
    /// Uses `std::hint::spin_loop` for low-latency busy waiting.  On each
    /// unsuccessful poll the thread yields via [`std::thread::yield_now`].
    pub fn wait(&self) {
        let current_epoch = self.epoch.load(Ordering::Acquire);
        let prev = self.arrived.fetch_add(1, Ordering::AcqRel);
        let new_count = prev + 1;

        if new_count == self.parties {
            // Last to arrive — reset counter and advance epoch.
            self.arrived.store(0, Ordering::Release);
            self.epoch.fetch_add(1, Ordering::Release);
        } else {
            // Spin until epoch advances.
            let mut spins = 0usize;
            loop {
                let e = self.epoch.load(Ordering::Acquire);
                if e != current_epoch {
                    break;
                }
                if spins < 32 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                spins = spins.saturating_add(1);
            }
        }
    }

    /// Number of parties.
    pub fn parties(&self) -> usize {
        self.parties
    }

    /// Current epoch (completes per round-trip).
    pub fn epoch(&self) -> usize {
        self.epoch.load(Ordering::Relaxed)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AO};
    use std::thread;

    // ── CyclicBarrier ──

    #[test]
    fn cyclic_barrier_all_proceed() {
        const N: usize = 5;
        let barrier = Arc::new(CyclicBarrier::new(N));
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for _ in 0..N {
            let b = Arc::clone(&barrier);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                c.fetch_add(1, AO::Relaxed);
                b.wait().expect("barrier wait");
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(counter.load(AO::Relaxed), N as u64);
    }

    #[test]
    fn cyclic_barrier_trip_thread_count() {
        const N: usize = 4;
        let barrier = Arc::new(CyclicBarrier::new(N));
        let trips = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for _ in 0..N {
            let b = Arc::clone(&barrier);
            let t = Arc::clone(&trips);
            handles.push(thread::spawn(move || {
                let trip = b.wait().expect("wait");
                if trip {
                    t.fetch_add(1, AO::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(trips.load(AO::Relaxed), 1, "exactly one trip thread");
    }

    #[test]
    fn cyclic_barrier_two_cycles() {
        const N: usize = 3;
        let barrier = Arc::new(CyclicBarrier::new(N));
        let phase_counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for _ in 0..N {
            let b = Arc::clone(&barrier);
            let p = Arc::clone(&phase_counter);
            handles.push(thread::spawn(move || {
                // Phase 1
                b.wait().expect("phase 1 wait");
                p.fetch_add(1, AO::Relaxed);
                // Phase 2
                b.wait().expect("phase 2 wait");
                p.fetch_add(1, AO::Relaxed);
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(phase_counter.load(AO::Relaxed), (N * 2) as u64);
    }

    // ── PhaseBarrier ──

    #[test]
    fn phase_barrier_advances_phase() {
        const N: usize = 4;
        let pb = Arc::new(PhaseBarrier::new(N));
        let mut handles = Vec::new();
        for _ in 0..N {
            let p = Arc::clone(&pb);
            handles.push(thread::spawn(move || {
                p.arrive_and_wait().expect("arrive phase 1");
                p.arrive_and_wait().expect("arrive phase 2");
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(pb.phase(), 2);
    }

    // ── CountDownLatch ──

    #[test]
    fn countdown_latch_basic() {
        const N: usize = 5;
        let latch = Arc::new(CountDownLatch::new(N));
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for _ in 0..N {
            let l = Arc::clone(&latch);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                c.fetch_add(1, AO::Relaxed);
                l.count_down();
            }));
        }

        latch.wait().expect("latch wait");
        assert!(latch.is_open());
        assert_eq!(counter.load(AO::Relaxed), N as u64);

        for h in handles {
            h.join().expect("thread");
        }
    }

    #[test]
    fn countdown_latch_already_open() {
        let latch = CountDownLatch::new(0);
        assert!(latch.is_open());
        latch.wait().expect("already open wait");
    }

    #[test]
    fn countdown_latch_timeout_opens() {
        let latch = Arc::new(CountDownLatch::new(1));
        let l2 = Arc::clone(&latch);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            l2.count_down();
        });
        let opened = latch
            .wait_timeout(Duration::from_secs(5))
            .expect("wait_timeout");
        assert!(opened);
    }

    #[test]
    fn countdown_latch_timeout_expires() {
        let latch = CountDownLatch::new(1); // never count down
        let opened = latch
            .wait_timeout(Duration::from_millis(10))
            .expect("wait_timeout");
        assert!(!opened);
    }

    // ── SpinBarrier ──

    #[test]
    fn spin_barrier_basic() {
        const N: usize = 4;
        let b = Arc::new(SpinBarrier::new(N));
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        for _ in 0..N {
            let bar = Arc::clone(&b);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                bar.wait();
                c.fetch_add(1, AO::Relaxed);
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(counter.load(AO::Relaxed), N as u64);
        assert_eq!(b.epoch(), 1);
    }

    #[test]
    fn spin_barrier_multiple_epochs() {
        const N: usize = 3;
        let b = Arc::new(SpinBarrier::new(N));
        let mut handles = Vec::new();

        for _ in 0..N {
            let bar = Arc::clone(&b);
            handles.push(thread::spawn(move || {
                bar.wait(); // epoch 0 → 1
                bar.wait(); // epoch 1 → 2
                bar.wait(); // epoch 2 → 3
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(b.epoch(), 3);
    }
}
