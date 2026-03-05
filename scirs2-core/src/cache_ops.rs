//! Cache-aware matrix operations for high-performance computing.
//!
//! This module provides cache-optimized implementations of common matrix
//! operations, including tiled matrix multiplication and cache-oblivious
//! transpose. The [`CacheAwareConfig`] struct exposes cache topology information
//! and derives optimal blocking parameters so that working sets fit in the
//! appropriate cache level.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::cache_ops::{CacheAwareConfig, tiled_matmul};
//! use ndarray::Array2;
//!
//! let config = CacheAwareConfig::detect();
//! let a = Array2::<f64>::eye(4);
//! let b = Array2::<f64>::eye(4);
//! let c = tiled_matmul(&a, &b);
//! assert_eq!(c, a);
//! ```

use ndarray::Array2;

// ──────────────────────────────────────────────────────────────────────────────
// CacheAwareConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Cache topology description used to derive blocking parameters.
///
/// All sizes are in bytes.  The defaults (L1 = 32 KiB, L2 = 256 KiB,
/// L3 = 8 MiB) are representative of modern x86-64 server CPUs and are
/// used whenever hardware detection is not available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheAwareConfig {
    /// L1 data-cache size in bytes (default 32 KiB)
    pub l1_cache_size: usize,
    /// L2 unified-cache size in bytes (default 256 KiB)
    pub l2_cache_size: usize,
    /// L3 shared-cache size in bytes (default 8 MiB)
    pub l3_cache_size: usize,
    /// Size of a single element in bytes (default 8 for `f64`)
    pub element_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAwareConfig {
    /// Construct with well-known default cache sizes and `element_size = 8`.
    pub fn new() -> Self {
        Self {
            l1_cache_size: 32 * 1024,       // 32 KiB
            l2_cache_size: 256 * 1024,      // 256 KiB
            l3_cache_size: 8 * 1024 * 1024, // 8 MiB
            element_size: 8,                // f64
        }
    }

    /// Attempt to detect L1/L2/L3 sizes from the host hardware.
    ///
    /// On Linux the kernel exposes per-cache information under
    /// `/sys/devices/system/cpu/cpu0/cache/index*/size`.  On macOS
    /// the same data is available through `sysctl`.  If detection
    /// fails for any reason the function silently returns the same
    /// defaults as [`CacheAwareConfig::new`].
    pub fn detect() -> Self {
        let defaults = Self::new();

        #[cfg(target_os = "linux")]
        {
            if let Some(cfg) = detect_linux() {
                return cfg;
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Some(cfg) = detect_macos() {
                return cfg;
            }
        }

        defaults
    }

    /// Compute the optimal square tile edge length for a matrix-multiply
    /// blocking scheme so that **three** tiles fit simultaneously in the L2
    /// cache.
    ///
    /// `n` is the largest matrix dimension; the returned tile size is
    /// clamped to `[4, n]`.
    pub fn tile_size_for_matmul(&self, n: usize) -> usize {
        // We want: 3 * tile^2 * element_size <= l2_cache_size
        // => tile <= sqrt(l2_cache_size / (3 * element_size))
        let max_elements = self.l2_cache_size / (3 * self.element_size.max(1));
        let tile = (max_elements as f64).sqrt() as usize;
        tile.clamp(4, n.max(4))
    }

    /// Compute the block size for a sequential scan so that one block fits
    /// comfortably in the L1 data cache.
    pub fn block_size_for_scan(&self) -> usize {
        (self.l1_cache_size / self.element_size.max(1)).max(1)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Hardware detection helpers (platform-specific)
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a Linux sysfs cache size string like "32K" or "8192K" into bytes.
fn parse_sysfs_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix('K') {
        stripped.trim().parse::<usize>().ok().map(|v| v * 1024)
    } else if let Some(stripped) = s.strip_suffix('M') {
        stripped
            .trim()
            .parse::<usize>()
            .ok()
            .map(|v| v * 1024 * 1024)
    } else {
        s.parse::<usize>().ok()
    }
}

#[cfg(target_os = "linux")]
fn detect_linux() -> Option<CacheAwareConfig> {
    use std::fs;

    // Iterate over sysfs cache index directories.
    let base = "/sys/devices/system/cpu/cpu0/cache";
    let mut l1: Option<usize> = None;
    let mut l2: Option<usize> = None;
    let mut l3: Option<usize> = None;

    for idx in 0..8usize {
        let level_path = format!("{base}/index{idx}/level");
        let size_path = format!("{base}/index{idx}/size");
        let type_path = format!("{base}/index{idx}/type");

        let level_str = match fs::read_to_string(&level_path) {
            Ok(s) => s,
            Err(_) => break,
        };
        let level: usize = match level_str.trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let size_str = match fs::read_to_string(&size_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let size = match parse_sysfs_size(&size_str) {
            Some(s) => s,
            None => continue,
        };
        // Skip instruction caches for L1.
        let cache_type = fs::read_to_string(&type_path).unwrap_or_default();
        let cache_type = cache_type.trim();
        if level == 1 && cache_type == "Instruction" {
            continue;
        }

        match level {
            1 => l1 = Some(size),
            2 => l2 = Some(size),
            3 => l3 = Some(size),
            _ => {}
        }
    }

    if l1.is_none() && l2.is_none() && l3.is_none() {
        return None;
    }

    let defaults = CacheAwareConfig::new();
    Some(CacheAwareConfig {
        l1_cache_size: l1.unwrap_or(defaults.l1_cache_size),
        l2_cache_size: l2.unwrap_or(defaults.l2_cache_size),
        l3_cache_size: l3.unwrap_or(defaults.l3_cache_size),
        element_size: defaults.element_size,
    })
}

#[cfg(target_os = "macos")]
fn detect_macos() -> Option<CacheAwareConfig> {
    fn sysctl_usize(name: &str) -> Option<usize> {
        let out = std::process::Command::new("sysctl")
            .arg("-n")
            .arg(name)
            .output()
            .ok()?;
        let s = std::str::from_utf8(&out.stdout).ok()?.trim();
        s.parse::<usize>().ok()
    }

    let l1 = sysctl_usize("hw.l1dcachesize");
    let l2 = sysctl_usize("hw.l2cachesize");
    let l3 = sysctl_usize("hw.l3cachesize");

    if l1.is_none() && l2.is_none() && l3.is_none() {
        return None;
    }

    let defaults = CacheAwareConfig::new();
    Some(CacheAwareConfig {
        l1_cache_size: l1.unwrap_or(defaults.l1_cache_size),
        l2_cache_size: l2.unwrap_or(defaults.l2_cache_size),
        l3_cache_size: l3.unwrap_or(defaults.l3_cache_size),
        element_size: defaults.element_size,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Cache-oblivious transpose
// ──────────────────────────────────────────────────────────────────────────────

/// In-place cache-oblivious transpose of a **square** `Array2<f64>`.
///
/// For non-square matrices the function falls back to `a.t().to_owned()`.
///
/// The recursive divide-and-conquer decomposition achieves optimal cache
/// performance without knowing the actual cache size at compile time.
pub fn cache_oblivious_transpose(a: &mut Array2<f64>) {
    let (rows, cols) = a.dim();
    if rows != cols {
        // Non-square: replace with transpose clone.
        let transposed = a.t().to_owned();
        *a = transposed;
        return;
    }
    let n = rows;
    // Work on a raw slice; safe because we have exclusive access via &mut.
    let ptr = a.as_mut_ptr();
    // SAFETY: Array2 with standard layout gives a contiguous row-major buffer.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, n * n) };
    recursive_transpose(slice, 0, n, 0, n, n);
}

/// Recursive helper: transpose the submatrix `[row_start..row_end) × [col_start..col_end)`
/// within the flat row-major buffer `buf` of stride `stride` (= total columns = n).
fn recursive_transpose(
    buf: &mut [f64],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    stride: usize,
) {
    const BASE: usize = 32;
    let rows = row_end - row_start;
    let cols = col_end - col_start;

    if rows <= BASE && cols <= BASE {
        // Base case: swap elements across the diagonal for the subblock.
        for i in row_start..row_end {
            // Only process the upper triangle relative to the diagonal.
            let j_min = if col_start > i { col_start } else { i + 1 };
            for j in j_min..col_end {
                buf.swap(i * stride + j, j * stride + i);
            }
        }
        return;
    }

    if rows >= cols {
        let mid = row_start + rows / 2;
        recursive_transpose(buf, row_start, mid, col_start, col_end, stride);
        recursive_transpose(buf, mid, row_end, col_start, col_end, stride);
    } else {
        let mid = col_start + cols / 2;
        recursive_transpose(buf, row_start, row_end, col_start, mid, stride);
        recursive_transpose(buf, row_start, row_end, mid, col_end, stride);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tiled matrix multiply
// ──────────────────────────────────────────────────────────────────────────────

/// Cache-efficient tiled matrix multiplication `C = A × B`.
///
/// Tile size is derived from [`CacheAwareConfig::detect`] so that three
/// tiles fit in the L2 cache simultaneously, maximising reuse.
///
/// # Panics
///
/// Panics if the inner dimensions do not match (`a.ncols() != b.nrows()`).
pub fn tiled_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(
        k, kb,
        "tiled_matmul: inner dimensions must match ({k} vs {kb})"
    );

    let config = CacheAwareConfig::detect();
    let tile = config.tile_size_for_matmul(m.max(n).max(k));

    let mut c = Array2::<f64>::zeros((m, n));

    // Blocked i-k-j loop for cache reuse of B tiles.
    let mut ii = 0;
    while ii < m {
        let i_end = (ii + tile).min(m);
        let mut kk = 0;
        while kk < k {
            let k_end = (kk + tile).min(k);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + tile).min(n);
                // Micro-kernel: accumulate into the C tile.
                for i in ii..i_end {
                    for kp in kk..k_end {
                        let a_ik = a[[i, kp]];
                        for j in jj..j_end {
                            c[[i, j]] += a_ik * b[[kp, j]];
                        }
                    }
                }
                jj += tile;
            }
            kk += tile;
        }
        ii += tile;
    }

    c
}

// ──────────────────────────────────────────────────────────────────────────────
// Prefetch-hinted matrix multiply
// ──────────────────────────────────────────────────────────────────────────────

/// Matrix multiplication with software prefetch hints for pipelined execution.
///
/// On `x86_64` the implementation inserts `_mm_prefetch` intrinsics to pull
/// the next tile of `B` into L2 cache before it is needed.  On other
/// architectures this falls back to the same tiled algorithm as
/// [`tiled_matmul`].
///
/// # Panics
///
/// Panics if the inner dimensions do not match.
pub fn prefetch_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    assert_eq!(
        k, kb,
        "prefetch_matmul: inner dimensions must match ({k} vs {kb})"
    );

    let config = CacheAwareConfig::detect();
    let tile = config.tile_size_for_matmul(m.max(n).max(k));

    let mut c = Array2::<f64>::zeros((m, n));

    let mut ii = 0;
    while ii < m {
        let i_end = (ii + tile).min(m);
        let mut kk = 0;
        while kk < k {
            let k_end = (kk + tile).min(k);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + tile).min(n);

                // Issue prefetch for the *next* B tile.
                let next_jj = jj + tile;
                if next_jj < n {
                    let next_j_end = (next_jj + tile).min(n);
                    prefetch_b_tile(b, kk, k_end, next_jj, next_j_end);
                }

                for i in ii..i_end {
                    for kp in kk..k_end {
                        let a_ik = a[[i, kp]];
                        for j in jj..j_end {
                            c[[i, j]] += a_ik * b[[kp, j]];
                        }
                    }
                }
                jj += tile;
            }
            kk += tile;
        }
        ii += tile;
    }

    c
}

/// Issue cache prefetch hints for a tile of `b`.
#[inline]
fn prefetch_b_tile(b: &Array2<f64>, k_start: usize, k_end: usize, j_start: usize, j_end: usize) {
    // Stride between contiguous prefetch hints (one cache line = 64 bytes = 8 f64s).
    const STRIDE: usize = 8;

    for kp in k_start..k_end {
        let mut j = j_start;
        while j < j_end {
            // Obtain a raw pointer to b[[kp, j]] and issue the prefetch.
            let ptr: *const f64 = &b[[kp, j]];
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: _mm_prefetch only reads the cache line; it never
                // dereferences the pointer beyond a speculative load.
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        ptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T1, // L2 cache
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // On other architectures use a harmless identity hint.
                let _ = std::hint::black_box(ptr);
            }
            j += STRIDE;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ── CacheAwareConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults_are_reasonable() {
        let cfg = CacheAwareConfig::new();
        assert!(cfg.l1_cache_size >= 8 * 1024, "L1 should be at least 8 KiB");
        assert!(cfg.l2_cache_size > cfg.l1_cache_size, "L2 > L1");
        assert!(cfg.l3_cache_size > cfg.l2_cache_size, "L3 > L2");
        assert_eq!(cfg.element_size, 8);
    }

    #[test]
    fn test_config_detect_returns_nonzero_sizes() {
        let cfg = CacheAwareConfig::detect();
        assert!(cfg.l1_cache_size > 0);
        assert!(cfg.l2_cache_size > 0);
        assert!(cfg.l3_cache_size > 0);
        assert!(cfg.element_size > 0);
    }

    #[test]
    fn test_tile_size_within_bounds_small() {
        let cfg = CacheAwareConfig::new();
        let n = 16;
        let tile = cfg.tile_size_for_matmul(n);
        assert!(tile >= 4, "tile_size >= 4");
        assert!(tile <= n, "tile_size <= n");
    }

    #[test]
    fn test_tile_size_within_bounds_large() {
        let cfg = CacheAwareConfig::new();
        for n in [64, 128, 512, 1024] {
            let tile = cfg.tile_size_for_matmul(n);
            assert!(tile >= 4);
            assert!(tile <= n);
        }
    }

    #[test]
    fn test_block_size_for_scan_is_positive() {
        let cfg = CacheAwareConfig::new();
        assert!(cfg.block_size_for_scan() > 0);
    }

    #[test]
    fn test_block_size_for_scan_fits_in_l1() {
        let cfg = CacheAwareConfig::new();
        let block = cfg.block_size_for_scan();
        // block * element_size should be <= l1_cache_size
        assert!(block * cfg.element_size <= cfg.l1_cache_size);
    }

    // ── cache_oblivious_transpose ─────────────────────────────────────────────

    #[test]
    fn test_cache_oblivious_transpose_4x4() {
        let mut a = Array2::<f64>::from_shape_vec((4, 4), (0..16).map(|x| x as f64).collect())
            .expect("valid shape");
        let expected = a.t().to_owned();
        cache_oblivious_transpose(&mut a);
        assert_eq!(a, expected);
    }

    #[test]
    fn test_cache_oblivious_transpose_8x8() {
        let data: Vec<f64> = (0..64).map(|x| x as f64).collect();
        let mut a = Array2::<f64>::from_shape_vec((8, 8), data).expect("valid shape");
        let expected = a.t().to_owned();
        cache_oblivious_transpose(&mut a);
        assert_eq!(a, expected);
    }

    #[test]
    fn test_cache_oblivious_transpose_involutory() {
        // Applying transpose twice should return the original matrix.
        let data: Vec<f64> = (0..64).map(|x| x as f64 * 0.5).collect();
        let mut a = Array2::<f64>::from_shape_vec((8, 8), data.clone()).expect("valid shape");
        let original = a.clone();
        cache_oblivious_transpose(&mut a);
        cache_oblivious_transpose(&mut a);
        assert_eq!(a, original);
    }

    #[test]
    fn test_cache_oblivious_transpose_large() {
        let n = 64;
        let data: Vec<f64> = (0..(n * n)).map(|x| x as f64).collect();
        let mut a = Array2::<f64>::from_shape_vec((n, n), data).expect("valid shape");
        let expected = a.t().to_owned();
        cache_oblivious_transpose(&mut a);
        assert_eq!(a, expected);
    }

    #[test]
    fn test_cache_oblivious_transpose_non_square_fallback() {
        let mut a = Array2::<f64>::from_shape_vec((3, 5), (0..15).map(|x| x as f64).collect())
            .expect("valid shape");
        let expected = a.t().to_owned();
        cache_oblivious_transpose(&mut a);
        assert_eq!(a, expected);
    }

    // ── tiled_matmul ──────────────────────────────────────────────────────────

    #[test]
    fn test_tiled_matmul_identity_4x4() {
        let a = Array2::<f64>::eye(4);
        let b = Array2::<f64>::eye(4);
        let c = tiled_matmul(&a, &b);
        assert_eq!(c, Array2::<f64>::eye(4));
    }

    #[test]
    fn test_tiled_matmul_known_result_2x2() {
        // [1 2] × [5 6]  =  [19 22]
        // [3 4]   [7 8]     [43 50]
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("ok");
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("ok");
        let c = tiled_matmul(&a, &b);
        let expected = Array2::from_shape_vec((2, 2), vec![19.0, 22.0, 43.0, 50.0]).expect("ok");
        for ((i, j), v) in c.indexed_iter() {
            assert!(
                (v - expected[[i, j]]).abs() < 1e-12,
                "mismatch at [{i},{j}]: {v} != {}",
                expected[[i, j]]
            );
        }
    }

    #[test]
    fn test_tiled_matmul_matches_naive_16x16() {
        use ndarray::Array2;
        let n = 16;
        let a = Array2::from_shape_fn((n, n), |(i, j)| (i * n + j) as f64 * 0.01);
        let b = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f64 * 0.01);
        let tiled = tiled_matmul(&a, &b);
        let naive = a.dot(&b);
        for ((i, j), v) in tiled.indexed_iter() {
            assert!(
                (v - naive[[i, j]]).abs() < 1e-9,
                "tiled vs naive mismatch at [{i},{j}]"
            );
        }
    }

    // ── prefetch_matmul ───────────────────────────────────────────────────────

    #[test]
    fn test_prefetch_matmul_matches_tiled_8x8() {
        let n = 8;
        let a = Array2::from_shape_fn((n, n), |(i, j)| (i * n + j) as f64);
        let b = Array2::from_shape_fn((n, n), |(i, j)| (i + j + 1) as f64);
        let tiled = tiled_matmul(&a, &b);
        let prefetched = prefetch_matmul(&a, &b);
        for ((i, j), v) in prefetched.indexed_iter() {
            assert!(
                (v - tiled[[i, j]]).abs() < 1e-9,
                "prefetch vs tiled mismatch at [{i},{j}]"
            );
        }
    }

    #[test]
    fn test_prefetch_matmul_correctness_64x64() {
        let n = 64;
        let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + 1) * (j + 1)) as f64 * 0.001);
        let b = Array2::from_shape_fn((n, n), |(i, j)| (i as f64 - j as f64).abs() * 0.001);
        let reference = a.dot(&b);
        let result = prefetch_matmul(&a, &b);
        for ((i, j), v) in result.indexed_iter() {
            assert!(
                (v - reference[[i, j]]).abs() < 1e-8,
                "prefetch_matmul wrong at [{i},{j}]"
            );
        }
    }

    #[test]
    fn test_prefetch_matmul_identity_8x8() {
        let eye = Array2::<f64>::eye(8);
        let a = Array2::from_shape_fn((8, 8), |(i, j)| (i * j) as f64 + 1.0);
        let result = prefetch_matmul(&a, &eye);
        for ((i, j), v) in result.indexed_iter() {
            assert!(
                (v - a[[i, j]]).abs() < 1e-12,
                "A×I should equal A at [{i},{j}]"
            );
        }
    }

    #[test]
    fn test_tiled_matmul_rect_2x3_times_3x4() {
        // Verify non-square multiplication shapes.
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("ok");
        let b = Array2::from_shape_vec(
            (3, 4),
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
        )
        .expect("ok");
        let tiled = tiled_matmul(&a, &b);
        let naive = a.dot(&b);
        for ((i, j), v) in tiled.indexed_iter() {
            assert!(
                (v - naive[[i, j]]).abs() < 1e-9,
                "rect mismatch at [{i},{j}]"
            );
        }
    }
}
