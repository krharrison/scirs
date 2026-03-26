//! SciPy -> SciRS2 equivalence reference
//!
//! This module provides a searchable mapping from SciPy function names
//! to their SciRS2 equivalents, including usage examples and migration notes.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::scipy_migration::scipy_equiv::{search_scipy, by_category, MigrationCategory};
//!
//! // Search by function name (case-insensitive partial match)
//! let hits = search_scipy("svd");
//! assert!(hits.iter().any(|e| e.scipy_path.contains("svd")));
//!
//! // Filter by category
//! let stats = by_category(MigrationCategory::Statistics);
//! assert!(!stats.is_empty());
//! ```

/// A single migration entry mapping a SciPy function to its SciRS2 equivalent.
#[derive(Debug, Clone)]
pub struct MigrationEntry {
    /// SciPy module and function path (e.g., `"scipy.linalg.det"`)
    pub scipy_path: &'static str,
    /// SciRS2 crate and function path (e.g., `"scirs2_linalg::prelude::det"`)
    pub scirs2_path: &'static str,
    /// Functional category
    pub category: MigrationCategory,
    /// Notes about differences in API or behavior
    pub notes: &'static str,
    /// SciPy usage example (Python code)
    pub scipy_example: &'static str,
    /// SciRS2 usage example (Rust code)
    pub scirs2_example: &'static str,
}

/// Functional categories for migration entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MigrationCategory {
    /// `scipy.linalg` and `numpy.linalg`
    LinearAlgebra,
    /// `scipy.stats`
    Statistics,
    /// `scipy.signal`
    SignalProcessing,
    /// `scipy.optimize`
    Optimization,
    /// `scipy.integrate`
    Integration,
    /// `scipy.interpolate`
    Interpolation,
    /// `scipy.fft` and `numpy.fft`
    FFT,
    /// `scipy.special`
    SpecialFunctions,
    /// `scipy.sparse`
    Sparse,
    /// `scipy.ndimage`
    ImageProcessing,
}

/// Returns the full migration table (all entries).
pub fn migration_table() -> &'static [MigrationEntry] {
    &MIGRATION_TABLE
}

/// Search for a SciPy function by name (case-insensitive partial match).
///
/// Returns entries whose `scipy_path` contains the query substring.
pub fn search_scipy(query: &str) -> Vec<&'static MigrationEntry> {
    let lower = query.to_lowercase();
    MIGRATION_TABLE
        .iter()
        .filter(|e| e.scipy_path.to_lowercase().contains(&lower))
        .collect()
}

/// Filter entries by category.
pub fn by_category(cat: MigrationCategory) -> Vec<&'static MigrationEntry> {
    MIGRATION_TABLE
        .iter()
        .filter(|e| e.category == cat)
        .collect()
}

// ---------------------------------------------------------------------------
// Migration Table
// ---------------------------------------------------------------------------

static MIGRATION_TABLE: [MigrationEntry; 85] = [
    // ===== Linear Algebra (15) =====
    MigrationEntry {
        scipy_path: "scipy.linalg.det",
        scirs2_path: "scirs2_linalg::prelude::det",
        category: MigrationCategory::LinearAlgebra,
        notes: "Returns LinalgResult<f64>. Use .view() on ndarray matrices.",
        scipy_example: r#"from scipy.linalg import det
d = det([[1, 2], [3, 4]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::det;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let d = det(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.inv",
        scirs2_path: "scirs2_linalg::prelude::inv",
        category: MigrationCategory::LinearAlgebra,
        notes: "Returns LinalgResult<Array2<f64>>. Singular matrices return an error instead of raising.",
        scipy_example: r#"from scipy.linalg import inv
a_inv = inv([[1, 2], [3, 4]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::inv;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let a_inv = inv(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.solve",
        scirs2_path: "scirs2_linalg::prelude::solve",
        category: MigrationCategory::LinearAlgebra,
        notes: "Solves Ax = b. Returns LinalgResult<Array1<f64>>.",
        scipy_example: r#"from scipy.linalg import solve
x = solve([[1, 2], [3, 4]], [5, 6])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::solve;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let b = ndarray::array![5.0, 6.0];
let x = solve(&a.view(), &b.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.eig",
        scirs2_path: "scirs2_linalg::prelude::eig",
        category: MigrationCategory::LinearAlgebra,
        notes: "Returns eigenvalues and eigenvectors. For symmetric matrices prefer eigh().",
        scipy_example: r#"from scipy.linalg import eig
vals, vecs = eig([[1, 2], [2, 3]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::eig;
let a = ndarray::array![[1.0, 2.0], [2.0, 3.0]];
let (vals, vecs) = eig(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.eigh",
        scirs2_path: "scirs2_linalg::prelude::eigh",
        category: MigrationCategory::LinearAlgebra,
        notes: "For symmetric/Hermitian matrices. Returns real eigenvalues.",
        scipy_example: r#"from scipy.linalg import eigh
vals, vecs = eigh([[2, 1], [1, 3]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::eigh;
let a = ndarray::array![[2.0, 1.0], [1.0, 3.0]];
let (vals, vecs) = eigh(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.svd",
        scirs2_path: "scirs2_linalg::prelude::svd",
        category: MigrationCategory::LinearAlgebra,
        notes: "Returns (U, S, Vt). full_matrices defaults to true like SciPy.",
        scipy_example: r#"from scipy.linalg import svd
U, s, Vt = svd([[1, 2], [3, 4]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::svd;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let (u, s, vt) = svd(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.qr",
        scirs2_path: "scirs2_linalg::prelude::qr",
        category: MigrationCategory::LinearAlgebra,
        notes: "Returns (Q, R) decomposition.",
        scipy_example: r#"from scipy.linalg import qr
Q, R = qr([[1, 2], [3, 4]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::qr;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let (q, r) = qr(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.lu",
        scirs2_path: "scirs2_linalg::prelude::lu",
        category: MigrationCategory::LinearAlgebra,
        notes: "Returns (P, L, U) decomposition.",
        scipy_example: r#"from scipy.linalg import lu
P, L, U = lu([[1, 2], [3, 4]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::lu;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let (p, l, u) = lu(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.cholesky",
        scirs2_path: "scirs2_linalg::prelude::cholesky",
        category: MigrationCategory::LinearAlgebra,
        notes: "Input must be symmetric positive definite. Returns lower triangular by default.",
        scipy_example: r#"from scipy.linalg import cholesky
L = cholesky([[4, 2], [2, 3]], lower=True)"#,
        scirs2_example: r#"use scirs2_linalg::prelude::cholesky;
let a = ndarray::array![[4.0, 2.0], [2.0, 3.0]];
let l = cholesky(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.expm",
        scirs2_path: "scirs2_linalg::prelude::expm",
        category: MigrationCategory::LinearAlgebra,
        notes: "Matrix exponential via Pade approximation.",
        scipy_example: r#"from scipy.linalg import expm
import numpy as np
result = expm(np.eye(2))"#,
        scirs2_example: r#"use scirs2_linalg::prelude::expm;
let a = ndarray::Array2::<f64>::eye(2);
let result = expm(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.norm",
        scirs2_path: "scirs2_linalg::prelude::matrix_norm",
        category: MigrationCategory::LinearAlgebra,
        notes: "Use matrix_norm for matrices, vector_norm for vectors. Supports Frobenius, 1-norm, inf-norm.",
        scipy_example: r#"from scipy.linalg import norm
n = norm([[1, 2], [3, 4]], 'fro')"#,
        scirs2_example: r#"use scirs2_linalg::prelude::matrix_norm;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let n = matrix_norm(&a.view(), "fro")?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.lstsq",
        scirs2_path: "scirs2_linalg::prelude::lstsq",
        category: MigrationCategory::LinearAlgebra,
        notes: "Least squares solution. Returns (x, residues, rank, singular_values).",
        scipy_example: r#"from scipy.linalg import lstsq
x, res, rank, sv = lstsq([[1, 1], [1, 2], [1, 3]], [1, 2, 3])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::lstsq;
let a = ndarray::array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
let b = ndarray::array![1.0, 2.0, 3.0];
let result = lstsq(&a.view(), &b.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.schur",
        scirs2_path: "scirs2_linalg::prelude::schur",
        category: MigrationCategory::LinearAlgebra,
        notes: "Schur decomposition A = Z T Z^H.",
        scipy_example: r#"from scipy.linalg import schur
T, Z = schur([[1, 2], [3, 4]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::schur;
let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
let (t, z) = schur(&a.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "numpy.dot",
        scirs2_path: "scirs2_linalg::prelude::blocked_matmul",
        category: MigrationCategory::LinearAlgebra,
        notes: "For matrix-matrix multiply use blocked_matmul or ndarray .dot() method.",
        scipy_example: r#"import numpy as np
c = np.dot(a, b)"#,
        scirs2_example: r#"// ndarray native dot:
let c = a.dot(&b);
// Or use scirs2_linalg for optimized blocked multiply:
use scirs2_linalg::prelude::blocked_matmul;
let c = blocked_matmul(&a.view(), &b.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.linalg.sqrtm",
        scirs2_path: "scirs2_linalg::prelude::sqrtm",
        category: MigrationCategory::LinearAlgebra,
        notes: "Matrix square root.",
        scipy_example: r#"from scipy.linalg import sqrtm
S = sqrtm([[4, 0], [0, 9]])"#,
        scirs2_example: r#"use scirs2_linalg::prelude::sqrtm;
let a = ndarray::array![[4.0, 0.0], [0.0, 9.0]];
let s = sqrtm(&a.view())?;"#,
    },
    // ===== Statistics (15) =====
    MigrationEntry {
        scipy_path: "scipy.stats.norm",
        scirs2_path: "scirs2_stats::distributions::norm",
        category: MigrationCategory::Statistics,
        notes: "Returns a Normal distribution object. Use .pdf(), .cdf(), .ppf() methods.",
        scipy_example: r#"from scipy.stats import norm
p = norm.cdf(1.96)
x = norm.ppf(0.975)"#,
        scirs2_example: r#"use scirs2_stats::distributions::norm;
use scirs2_stats::traits::Distribution;
let n = norm(0.0, 1.0)?;
let p = n.cdf(1.96);
let x = n.ppf(0.975)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.beta",
        scirs2_path: "scirs2_stats::distributions::beta",
        category: MigrationCategory::Statistics,
        notes: "Beta(alpha, beta, loc, scale). loc and scale shift/scale the support.",
        scipy_example: r#"from scipy.stats import beta
p = beta.cdf(0.5, 2, 5)"#,
        scirs2_example: r#"use scirs2_stats::distributions::beta;
use scirs2_stats::traits::Distribution;
let b = beta(2.0, 5.0, 0.0, 1.0)?;
let p = b.cdf(0.5);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.gamma",
        scirs2_path: "scirs2_stats::distributions::gamma",
        category: MigrationCategory::Statistics,
        notes: "Gamma(shape, scale, loc). SciPy uses (a, loc, scale).",
        scipy_example: r#"from scipy.stats import gamma
p = gamma.cdf(2.0, a=2, scale=1)"#,
        scirs2_example: r#"use scirs2_stats::distributions::gamma;
use scirs2_stats::traits::Distribution;
let g = gamma(2.0, 1.0, 0.0)?;
let p = g.cdf(2.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.t",
        scirs2_path: "scirs2_stats::distributions::t",
        category: MigrationCategory::Statistics,
        notes: "Student's t-distribution. t(df, loc, scale).",
        scipy_example: r#"from scipy.stats import t
p = t.cdf(2.0, df=10)"#,
        scirs2_example: r#"use scirs2_stats::distributions::t;
use scirs2_stats::traits::Distribution;
let td = t(10.0, 0.0, 1.0)?;
let p = td.cdf(2.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.f",
        scirs2_path: "scirs2_stats::distributions::f",
        category: MigrationCategory::Statistics,
        notes: "F-distribution. f(dfn, dfd, loc, scale).",
        scipy_example: r#"from scipy.stats import f
p = f.cdf(3.0, dfn=5, dfd=10)"#,
        scirs2_example: r#"use scirs2_stats::distributions::f;
use scirs2_stats::traits::Distribution;
let fd = f(5.0, 10.0, 0.0, 1.0)?;
let p = fd.cdf(3.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.chi2",
        scirs2_path: "scirs2_stats::distributions::chi2",
        category: MigrationCategory::Statistics,
        notes: "Chi-squared distribution. chi2(df, loc, scale).",
        scipy_example: r#"from scipy.stats import chi2
p = chi2.cdf(5.0, df=3)"#,
        scirs2_example: r#"use scirs2_stats::distributions::chi2;
use scirs2_stats::traits::Distribution;
let c = chi2(3.0, 0.0, 1.0)?;
let p = c.cdf(5.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.expon",
        scirs2_path: "scirs2_stats::distributions::expon",
        category: MigrationCategory::Statistics,
        notes: "Exponential distribution. expon(rate, loc).",
        scipy_example: r#"from scipy.stats import expon
p = expon.cdf(1.0, scale=0.5)"#,
        scirs2_example: r#"use scirs2_stats::distributions::expon;
use scirs2_stats::traits::Distribution;
let e = expon(2.0, 0.0)?;  // rate=2.0
let p = e.cdf(1.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.binom",
        scirs2_path: "scirs2_stats::distributions::binom",
        category: MigrationCategory::Statistics,
        notes: "Binomial distribution. binom(n, p).",
        scipy_example: r#"from scipy.stats import binom
p = binom.pmf(3, n=10, p=0.5)"#,
        scirs2_example: r#"use scirs2_stats::distributions::binom;
let b = binom(10, 0.5)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.poisson",
        scirs2_path: "scirs2_stats::distributions::poisson",
        category: MigrationCategory::Statistics,
        notes: "Poisson distribution. poisson(mu, loc).",
        scipy_example: r#"from scipy.stats import poisson
p = poisson.pmf(3, mu=5)"#,
        scirs2_example: r#"use scirs2_stats::distributions::poisson;
let po = poisson(5.0, 0.0)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.ttest_ind",
        scirs2_path: "scirs2_stats::ttest_ind",
        category: MigrationCategory::Statistics,
        notes: "Independent two-sample t-test. Returns TTestResult with statistic and p-value.",
        scipy_example: r#"from scipy.stats import ttest_ind
stat, pval = ttest_ind([1,2,3], [4,5,6])"#,
        scirs2_example: r#"use scirs2_stats::ttest_ind;
let a = ndarray::array![1.0, 2.0, 3.0];
let b = ndarray::array![4.0, 5.0, 6.0];
let result = ttest_ind(&a.view(), &b.view(), false)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.pearsonr",
        scirs2_path: "scirs2_stats::pearsonr",
        category: MigrationCategory::Statistics,
        notes: "Returns (correlation, p-value) tuple.",
        scipy_example: r#"from scipy.stats import pearsonr
r, p = pearsonr([1,2,3], [1,2,3])"#,
        scirs2_example: r#"use scirs2_stats::pearsonr;
let x = ndarray::array![1.0, 2.0, 3.0];
let y = ndarray::array![1.0, 2.0, 3.0];
let (r, p) = pearsonr(&x.view(), &y.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.spearmanr",
        scirs2_path: "scirs2_stats::spearmanr",
        category: MigrationCategory::Statistics,
        notes: "Spearman rank-order correlation.",
        scipy_example: r#"from scipy.stats import spearmanr
r, p = spearmanr([1,2,3], [1,2,3])"#,
        scirs2_example: r#"use scirs2_stats::spearmanr;
let x = ndarray::array![1.0, 2.0, 3.0];
let y = ndarray::array![1.0, 2.0, 3.0];
let (r, p) = spearmanr(&x.view(), &y.view())?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.describe",
        scirs2_path: "scirs2_stats::{mean, var, std, median}",
        category: MigrationCategory::Statistics,
        notes: "No single describe() function; use individual functions mean(), var(), std(), median().",
        scipy_example: r#"from scipy.stats import describe
result = describe([1, 2, 3, 4, 5])"#,
        scirs2_example: r#"use scirs2_stats::{mean, var, std, median};
let x = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0];
let m = mean(&x.view())?;
let v = var(&x.view(), 1, None)?;
let s = std(&x.view(), 1, None)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.kstest",
        scirs2_path: "scirs2_stats::tests::kstest",
        category: MigrationCategory::Statistics,
        notes: "Kolmogorov-Smirnov one-sample test. Takes a CDF closure.",
        scipy_example: r#"from scipy.stats import kstest
stat, pval = kstest([0.1, 0.5, 0.9], 'uniform')"#,
        scirs2_example: r#"use scirs2_stats::tests::kstest;
let data = ndarray::array![0.1, 0.5, 0.9];
let (stat, pval) = kstest(&data.view(), |x| x.min(1.0).max(0.0))?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.stats.mannwhitneyu",
        scirs2_path: "scirs2_stats::mann_whitney",
        category: MigrationCategory::Statistics,
        notes: "Mann-Whitney U rank test for two independent samples.",
        scipy_example: r#"from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu([1,2,3], [4,5,6])"#,
        scirs2_example: r#"use scirs2_stats::mann_whitney;
let a = ndarray::array![1.0, 2.0, 3.0];
let b = ndarray::array![4.0, 5.0, 6.0];
let result = mann_whitney(&a.view(), &b.view())?;"#,
    },
    // ===== Signal Processing (10) =====
    MigrationEntry {
        scipy_path: "scipy.signal.butter",
        scirs2_path: "scirs2_signal::butter",
        category: MigrationCategory::SignalProcessing,
        notes: "Butterworth filter design. Returns (b, a) coefficients.",
        scipy_example: r#"from scipy.signal import butter
b, a = butter(4, 0.1, btype='low')"#,
        scirs2_example: r#"use scirs2_signal::butter;
use scirs2_signal::FilterType;
let (b, a) = butter(4, 0.1, FilterType::LowPass)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.firwin",
        scirs2_path: "scirs2_signal::firwin",
        category: MigrationCategory::SignalProcessing,
        notes: "FIR filter design using the window method.",
        scipy_example: r#"from scipy.signal import firwin
h = firwin(51, 0.3)"#,
        scirs2_example: r#"use scirs2_signal::firwin;
use scirs2_signal::FilterType;
let h = firwin(51, 0.3, FilterType::LowPass)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.filtfilt",
        scirs2_path: "scirs2_signal::filtfilt",
        category: MigrationCategory::SignalProcessing,
        notes: "Zero-phase digital filtering (forward-backward).",
        scipy_example: r#"from scipy.signal import filtfilt
y = filtfilt(b, a, x)"#,
        scirs2_example: r#"use scirs2_signal::filtfilt;
let y = filtfilt(&b, &a, &x)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.stft",
        scirs2_path: "scirs2_signal::stft",
        category: MigrationCategory::SignalProcessing,
        notes: "Short-time Fourier transform. Returns (frequencies, times, Zxx).",
        scipy_example: r#"from scipy.signal import stft
f, t, Zxx = stft(x, fs=1.0, nperseg=256)"#,
        scirs2_example: r#"use scirs2_signal::stft;
let (f, t, zxx) = stft(&x, 1.0, 256, None)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.welch",
        scirs2_path: "scirs2_signal::welch",
        category: MigrationCategory::SignalProcessing,
        notes: "Welch's method for power spectral density estimation.",
        scipy_example: r#"from scipy.signal import welch
f, Pxx = welch(x, fs=1.0, nperseg=256)"#,
        scirs2_example: r#"use scirs2_signal::welch;
let (f, pxx) = welch(&x, 1.0, 256)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.periodogram",
        scirs2_path: "scirs2_signal::periodogram",
        category: MigrationCategory::SignalProcessing,
        notes: "Periodogram PSD estimate.",
        scipy_example: r#"from scipy.signal import periodogram
f, Pxx = periodogram(x, fs=1.0)"#,
        scirs2_example: r#"use scirs2_signal::periodogram;
let (f, pxx) = periodogram(&x, 1.0)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.spectrogram",
        scirs2_path: "scirs2_signal::spectrogram",
        category: MigrationCategory::SignalProcessing,
        notes: "Compute a spectrogram with consecutive STFTs.",
        scipy_example: r#"from scipy.signal import spectrogram
f, t, Sxx = spectrogram(x, fs=1.0)"#,
        scirs2_example: r#"use scirs2_signal::spectrogram;
let (f, t, sxx) = spectrogram(&x, 1.0, None)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.convolve",
        scirs2_path: "scirs2_signal::convolve::convolve",
        category: MigrationCategory::SignalProcessing,
        notes: "1-D convolution of two arrays. Supports 'full', 'same', 'valid' modes.",
        scipy_example: r#"from scipy.signal import convolve
y = convolve([1,2,3], [0,1,0.5], mode='full')"#,
        scirs2_example: r#"use scirs2_signal::convolve::convolve;
let y = convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5], "full")?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.hilbert",
        scirs2_path: "scirs2_signal::hilbert",
        category: MigrationCategory::SignalProcessing,
        notes: "Hilbert transform to compute analytic signal.",
        scipy_example: r#"from scipy.signal import hilbert
analytic = hilbert(x)"#,
        scirs2_example: r#"use scirs2_signal::hilbert;
// Module provides Hilbert transform functionality"#,
    },
    MigrationEntry {
        scipy_path: "scipy.signal.find_peaks",
        scirs2_path: "scirs2_signal::measurements",
        category: MigrationCategory::SignalProcessing,
        notes: "Peak detection. Use the measurements module for signal analysis.",
        scipy_example: r#"from scipy.signal import find_peaks
peaks, props = find_peaks(x, height=0)"#,
        scirs2_example: r#"use scirs2_signal::measurements;
// Use measurements module for peak finding and signal analysis"#,
    },
    // ===== FFT (8) =====
    MigrationEntry {
        scipy_path: "scipy.fft.fft",
        scirs2_path: "scirs2_fft::fft",
        category: MigrationCategory::FFT,
        notes: "1-D discrete Fourier Transform. Uses OxiFFT backend (pure Rust).",
        scipy_example: r#"from scipy.fft import fft
X = fft([1.0, 2.0, 3.0, 4.0])"#,
        scirs2_example: r#"use scirs2_fft::fft;
use num_complex::Complex64;
let x: Vec<Complex64> = vec![1.0, 2.0, 3.0, 4.0]
    .into_iter().map(|v| Complex64::new(v, 0.0)).collect();
let big_x = fft(&x)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.ifft",
        scirs2_path: "scirs2_fft::ifft",
        category: MigrationCategory::FFT,
        notes: "Inverse 1-D FFT.",
        scipy_example: r#"from scipy.fft import ifft
x = ifft(X)"#,
        scirs2_example: r#"use scirs2_fft::ifft;
let x = ifft(&big_x)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.rfft",
        scirs2_path: "scirs2_fft::rfft",
        category: MigrationCategory::FFT,
        notes: "FFT of real-valued input (returns only positive frequencies).",
        scipy_example: r#"from scipy.fft import rfft
X = rfft([1.0, 2.0, 3.0, 4.0])"#,
        scirs2_example: r#"use scirs2_fft::rfft;
let big_x = rfft(&[1.0, 2.0, 3.0, 4.0])?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.irfft",
        scirs2_path: "scirs2_fft::irfft",
        category: MigrationCategory::FFT,
        notes: "Inverse of rfft.",
        scipy_example: r#"from scipy.fft import irfft
x = irfft(X)"#,
        scirs2_example: r#"use scirs2_fft::irfft;
let x = irfft(&big_x, None)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.fft2",
        scirs2_path: "scirs2_fft::fft2",
        category: MigrationCategory::FFT,
        notes: "2-D FFT.",
        scipy_example: r#"from scipy.fft import fft2
X = fft2(image)"#,
        scirs2_example: r#"use scirs2_fft::fft2;
let big_x = fft2(&image)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.fftfreq",
        scirs2_path: "scirs2_fft::fftfreq",
        category: MigrationCategory::FFT,
        notes: "DFT sample frequencies.",
        scipy_example: r#"from scipy.fft import fftfreq
freq = fftfreq(256, d=1/1000)"#,
        scirs2_example: r#"use scirs2_fft::fftfreq;
let freq = fftfreq(256, 1.0 / 1000.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.dct",
        scirs2_path: "scirs2_fft::dct",
        category: MigrationCategory::FFT,
        notes: "Discrete Cosine Transform.",
        scipy_example: r#"from scipy.fft import dct
X = dct([1.0, 2.0, 3.0, 4.0], type=2)"#,
        scirs2_example: r#"use scirs2_fft::{dct, DCTType};
let big_x = dct(&[1.0, 2.0, 3.0, 4.0], DCTType::Type2)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.fft.dst",
        scirs2_path: "scirs2_fft::dst",
        category: MigrationCategory::FFT,
        notes: "Discrete Sine Transform.",
        scipy_example: r#"from scipy.fft import dst
X = dst([1.0, 2.0, 3.0, 4.0], type=1)"#,
        scirs2_example: r#"use scirs2_fft::{dst, DSTType};
let big_x = dst(&[1.0, 2.0, 3.0, 4.0], DSTType::Type1)?;"#,
    },
    // ===== Optimization (8) =====
    MigrationEntry {
        scipy_path: "scipy.optimize.minimize",
        scirs2_path: "scirs2_optimize::minimize",
        category: MigrationCategory::Optimization,
        notes: "Unconstrained minimization. Supports BFGS, L-BFGS, Nelder-Mead, etc.",
        scipy_example: r#"from scipy.optimize import minimize
result = minimize(f, x0, method='BFGS')"#,
        scirs2_example: r#"use scirs2_optimize::{minimize, Bounds};
let result = minimize(
    |x| x[0]*x[0] + x[1]*x[1],
    &[1.0, 1.0],
    None, // gradient
    None, // method defaults to BFGS
)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.minimize (constrained)",
        scirs2_path: "scirs2_optimize::minimize_constrained",
        category: MigrationCategory::Optimization,
        notes: "Constrained minimization (SLSQP-like).",
        scipy_example: r#"from scipy.optimize import minimize
result = minimize(f, x0, constraints=cons, bounds=bnds)"#,
        scirs2_example: r#"use scirs2_optimize::minimize_constrained;
// Supports equality and inequality constraints"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.minimize_scalar",
        scirs2_path: "scirs2_optimize::minimize_scalar",
        category: MigrationCategory::Optimization,
        notes: "Scalar (1-D) minimization.",
        scipy_example: r#"from scipy.optimize import minimize_scalar
result = minimize_scalar(lambda x: (x-2)**2, bounds=(0, 5))"#,
        scirs2_example: r#"use scirs2_optimize::minimize_scalar;
let result = minimize_scalar(|x| (x - 2.0).powi(2), Some((0.0, 5.0)))?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.root",
        scirs2_path: "scirs2_optimize::root",
        category: MigrationCategory::Optimization,
        notes: "Find roots of a vector function.",
        scipy_example: r#"from scipy.optimize import root
sol = root(lambda x: [x[0]**2 - 1], [0.5])"#,
        scirs2_example: r#"use scirs2_optimize::root;
// Supports Broyden, Anderson, Krylov methods"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.least_squares",
        scirs2_path: "scirs2_optimize::least_squares",
        category: MigrationCategory::Optimization,
        notes: "Nonlinear least squares with bounds. Supports Levenberg-Marquardt.",
        scipy_example: r#"from scipy.optimize import least_squares
result = least_squares(residuals, x0)"#,
        scirs2_example: r#"use scirs2_optimize::least_squares;
let result = least_squares(residual_fn, &x0, None)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.differential_evolution",
        scirs2_path: "scirs2_optimize::differential_evolution",
        category: MigrationCategory::Optimization,
        notes: "Global optimization via differential evolution.",
        scipy_example: r#"from scipy.optimize import differential_evolution
result = differential_evolution(f, bounds=[(0,5), (0,5)])"#,
        scirs2_example: r#"use scirs2_optimize::differential_evolution;
// Global optimization with population-based search"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.dual_annealing",
        scirs2_path: "scirs2_optimize::dual_annealing",
        category: MigrationCategory::Optimization,
        notes: "Global optimization via dual annealing.",
        scipy_example: r#"from scipy.optimize import dual_annealing
result = dual_annealing(f, bounds=[(0,5), (0,5)])"#,
        scirs2_example: r#"use scirs2_optimize::dual_annealing;
// Dual annealing with local search refinement"#,
    },
    MigrationEntry {
        scipy_path: "scipy.optimize.basinhopping",
        scirs2_path: "scirs2_optimize::basinhopping",
        category: MigrationCategory::Optimization,
        notes: "Basin-hopping global optimization.",
        scipy_example: r#"from scipy.optimize import basinhopping
result = basinhopping(f, x0)"#,
        scirs2_example: r#"use scirs2_optimize::basinhopping;
// Basin-hopping with random perturbation + local minimization"#,
    },
    // ===== Integration (7) =====
    MigrationEntry {
        scipy_path: "scipy.integrate.quad",
        scirs2_path: "scirs2_integrate::quad",
        category: MigrationCategory::Integration,
        notes: "Adaptive numerical integration. Returns (result, error_estimate).",
        scipy_example: r#"from scipy.integrate import quad
result, err = quad(lambda x: x**2, 0, 1)"#,
        scirs2_example: r#"use scirs2_integrate::quad;
let (result, err) = quad(|x| x * x, 0.0, 1.0)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.integrate.trapezoid",
        scirs2_path: "scirs2_integrate::trapezoid",
        category: MigrationCategory::Integration,
        notes: "Trapezoidal rule integration.",
        scipy_example: r#"from scipy.integrate import trapezoid
result = trapezoid([1, 2, 3], [0, 1, 2])"#,
        scirs2_example: r#"use scirs2_integrate::trapezoid;
let result = trapezoid(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0])?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.integrate.simpson",
        scirs2_path: "scirs2_integrate::simpson",
        category: MigrationCategory::Integration,
        notes: "Simpson's rule integration.",
        scipy_example: r#"from scipy.integrate import simpson
result = simpson([1, 4, 1], dx=0.5)"#,
        scirs2_example: r#"use scirs2_integrate::simpson;
let result = simpson(&[1.0, 4.0, 1.0], 0.5)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.integrate.solve_ivp",
        scirs2_path: "scirs2_integrate::solve_ivp",
        category: MigrationCategory::Integration,
        notes: "Solve initial value problems for ODEs. Supports RK45, RK23, DOP853, Radau, BDF.",
        scipy_example: r#"from scipy.integrate import solve_ivp
sol = solve_ivp(fun, [0, 10], y0, method='RK45')"#,
        scirs2_example: r#"use scirs2_integrate::{solve_ivp, ODEMethod, ODEOptions};
let opts = ODEOptions { method: ODEMethod::RK45, ..Default::default() };
let result = solve_ivp(|t, y| Ok(vec![-y[0]]), (0.0, 10.0), &[1.0], &opts)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.integrate.solve_bvp",
        scirs2_path: "scirs2_integrate::solve_bvp",
        category: MigrationCategory::Integration,
        notes: "Solve boundary value problems.",
        scipy_example: r#"from scipy.integrate import solve_bvp
sol = solve_bvp(fun, bc, x, y_init)"#,
        scirs2_example: r#"use scirs2_integrate::solve_bvp;
// Supports collocation-based BVP solving"#,
    },
    MigrationEntry {
        scipy_path: "scipy.integrate.nquad",
        scirs2_path: "scirs2_integrate::cubature",
        category: MigrationCategory::Integration,
        notes: "Multi-dimensional integration. Use cubature() for n-dimensional integrals.",
        scipy_example: r#"from scipy.integrate import nquad
result, err = nquad(f, [[0,1], [0,1]])"#,
        scirs2_example: r#"use scirs2_integrate::cubature;
// Multi-dimensional adaptive cubature"#,
    },
    MigrationEntry {
        scipy_path: "scipy.integrate.romberg",
        scirs2_path: "scirs2_integrate::romberg",
        category: MigrationCategory::Integration,
        notes: "Romberg integration for high accuracy.",
        scipy_example: r#"from scipy.integrate import romberg
result = romberg(lambda x: x**2, 0, 1)"#,
        scirs2_example: r#"use scirs2_integrate::romberg;
// Romberg extrapolation of the trapezoidal rule"#,
    },
    // ===== Interpolation (6) =====
    MigrationEntry {
        scipy_path: "scipy.interpolate.interp1d",
        scirs2_path: "scirs2_interpolate::Interp1d",
        category: MigrationCategory::Interpolation,
        notes: "1-D interpolation. Supports linear, nearest, cubic, etc.",
        scipy_example: r#"from scipy.interpolate import interp1d
f = interp1d([0, 1, 2], [0, 1, 4], kind='linear')
y = f(0.5)"#,
        scirs2_example: r#"use scirs2_interpolate::{Interp1d, InterpolationMethod};
let interp = Interp1d::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0],
    InterpolationMethod::Linear)?;
let y = interp.evaluate(0.5)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.interpolate.CubicSpline",
        scirs2_path: "scirs2_interpolate::CubicSpline",
        category: MigrationCategory::Interpolation,
        notes: "Cubic spline interpolation with various boundary conditions.",
        scipy_example: r#"from scipy.interpolate import CubicSpline
cs = CubicSpline([0, 1, 2], [0, 1, 4])
y = cs(0.5)"#,
        scirs2_example: r#"use scirs2_interpolate::CubicSpline;
let cs = CubicSpline::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0])?;
let y = cs.evaluate(0.5)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.interpolate.PchipInterpolator",
        scirs2_path: "scirs2_interpolate::PchipInterpolator",
        category: MigrationCategory::Interpolation,
        notes: "Piecewise Cubic Hermite Interpolating Polynomial (monotone).",
        scipy_example: r#"from scipy.interpolate import PchipInterpolator
p = PchipInterpolator([0, 1, 2], [0, 1, 4])"#,
        scirs2_example: r#"use scirs2_interpolate::PchipInterpolator;
let p = PchipInterpolator::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0])?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.interpolate.RegularGridInterpolator",
        scirs2_path: "scirs2_interpolate::RegularGridInterpolator",
        category: MigrationCategory::Interpolation,
        notes: "Interpolation on a regular N-D grid.",
        scipy_example: r#"from scipy.interpolate import RegularGridInterpolator
interp = RegularGridInterpolator((x, y), data)"#,
        scirs2_example: r#"use scirs2_interpolate::RegularGridInterpolator;
// Multi-dimensional regular grid interpolation"#,
    },
    MigrationEntry {
        scipy_path: "scipy.interpolate.BSpline",
        scirs2_path: "scirs2_interpolate::bspline",
        category: MigrationCategory::Interpolation,
        notes: "B-spline representation. Use bspline module for construction and evaluation.",
        scipy_example: r#"from scipy.interpolate import BSpline
spl = BSpline(knots, coeffs, degree)"#,
        scirs2_example: r#"use scirs2_interpolate::bspline;
// B-spline construction and evaluation"#,
    },
    MigrationEntry {
        scipy_path: "scipy.interpolate.griddata",
        scirs2_path: "scirs2_interpolate::griddata",
        category: MigrationCategory::Interpolation,
        notes: "Interpolate unstructured data to a grid.",
        scipy_example: r#"from scipy.interpolate import griddata
grid_z = griddata(points, values, (grid_x, grid_y))"#,
        scirs2_example: r#"use scirs2_interpolate::griddata;
// Scattered data interpolation to grid"#,
    },
    // ===== Special Functions (7) =====
    MigrationEntry {
        scipy_path: "scipy.special.gamma",
        scirs2_path: "scirs2_special::gamma",
        category: MigrationCategory::SpecialFunctions,
        notes: "Gamma function. Use gamma_safe() for error-handling version.",
        scipy_example: r#"from scipy.special import gamma
y = gamma(5)  # = 24.0"#,
        scirs2_example: r#"use scirs2_special::gamma;
let y = gamma(5.0);  // = 24.0"#,
    },
    MigrationEntry {
        scipy_path: "scipy.special.gammaln",
        scirs2_path: "scirs2_special::gammaln",
        category: MigrationCategory::SpecialFunctions,
        notes: "Log of the absolute value of the gamma function.",
        scipy_example: r#"from scipy.special import gammaln
y = gammaln(100)"#,
        scirs2_example: r#"use scirs2_special::gammaln;
let y = gammaln(100.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.special.beta",
        scirs2_path: "scirs2_special::beta",
        category: MigrationCategory::SpecialFunctions,
        notes: "Beta function B(a, b).",
        scipy_example: r#"from scipy.special import beta
y = beta(2, 3)"#,
        scirs2_example: r#"use scirs2_special::beta;
let y = beta(2.0, 3.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.special.erf",
        scirs2_path: "scirs2_special::erf",
        category: MigrationCategory::SpecialFunctions,
        notes: "Error function.",
        scipy_example: r#"from scipy.special import erf
y = erf(1.0)"#,
        scirs2_example: r#"use scirs2_special::erf;
let y = erf(1.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.special.erfc",
        scirs2_path: "scirs2_special::erfc",
        category: MigrationCategory::SpecialFunctions,
        notes: "Complementary error function (1 - erf(x)).",
        scipy_example: r#"from scipy.special import erfc
y = erfc(1.0)"#,
        scirs2_example: r#"use scirs2_special::erfc;
let y = erfc(1.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.special.jv",
        scirs2_path: "scirs2_special::bessel::jv",
        category: MigrationCategory::SpecialFunctions,
        notes: "Bessel function of the first kind of real order.",
        scipy_example: r#"from scipy.special import jv
y = jv(0, 1.0)"#,
        scirs2_example: r#"use scirs2_special::jv;
let y = jv(0.0, 1.0);"#,
    },
    MigrationEntry {
        scipy_path: "scipy.special.hyp2f1",
        scirs2_path: "scirs2_special::hyp2f1",
        category: MigrationCategory::SpecialFunctions,
        notes: "Gauss hypergeometric function 2F1(a, b; c; z).",
        scipy_example: r#"from scipy.special import hyp2f1
y = hyp2f1(1, 2, 3, 0.5)"#,
        scirs2_example: r#"use scirs2_special::hyp2f1;
let y = hyp2f1(1.0, 2.0, 3.0, 0.5);"#,
    },
    // ===== Sparse (6) =====
    MigrationEntry {
        scipy_path: "scipy.sparse.csr_matrix",
        scirs2_path: "scirs2_sparse::CsrMatrix",
        category: MigrationCategory::Sparse,
        notes: "Compressed Sparse Row matrix. Also available as CsrArray.",
        scipy_example: r#"from scipy.sparse import csr_matrix
A = csr_matrix([[1, 0], [0, 2]])"#,
        scirs2_example: r#"use scirs2_sparse::CsrMatrix;
// Construct from triplets or dense array"#,
    },
    MigrationEntry {
        scipy_path: "scipy.sparse.csc_matrix",
        scirs2_path: "scirs2_sparse::CscMatrix",
        category: MigrationCategory::Sparse,
        notes: "Compressed Sparse Column matrix.",
        scipy_example: r#"from scipy.sparse import csc_matrix
A = csc_matrix([[1, 0], [0, 2]])"#,
        scirs2_example: r#"use scirs2_sparse::CscMatrix;
// Compressed Sparse Column format"#,
    },
    MigrationEntry {
        scipy_path: "scipy.sparse.coo_matrix",
        scirs2_path: "scirs2_sparse::CooMatrix",
        category: MigrationCategory::Sparse,
        notes: "Coordinate list (COO) sparse matrix.",
        scipy_example: r#"from scipy.sparse import coo_matrix
A = coo_matrix(([1,2], ([0,1], [0,1])), shape=(2,2))"#,
        scirs2_example: r#"use scirs2_sparse::CooMatrix;
// COO format: store (row, col, value) triplets"#,
    },
    MigrationEntry {
        scipy_path: "scipy.sparse.linalg.spsolve",
        scirs2_path: "scirs2_sparse::spsolve",
        category: MigrationCategory::Sparse,
        notes: "Direct solver for sparse linear systems Ax = b.",
        scipy_example: r#"from scipy.sparse.linalg import spsolve
x = spsolve(A, b)"#,
        scirs2_example: r#"use scirs2_sparse::spsolve;
let x = spsolve(&a, &b)?;"#,
    },
    MigrationEntry {
        scipy_path: "scipy.sparse.linalg.eigsh",
        scirs2_path: "scirs2_sparse::eigsh",
        category: MigrationCategory::Sparse,
        notes: "Find k eigenvalues of a symmetric sparse matrix (ARPACK-like).",
        scipy_example: r#"from scipy.sparse.linalg import eigsh
vals, vecs = eigsh(A, k=3)"#,
        scirs2_example: r#"use scirs2_sparse::eigsh;
// Iterative eigenvalue solver for sparse symmetric matrices"#,
    },
    MigrationEntry {
        scipy_path: "scipy.sparse.linalg.cg",
        scirs2_path: "scirs2_sparse::cg",
        category: MigrationCategory::Sparse,
        notes: "Conjugate gradient solver for sparse symmetric positive-definite systems.",
        scipy_example: r#"from scipy.sparse.linalg import cg
x, info = cg(A, b)"#,
        scirs2_example: r#"use scirs2_sparse::cg;
// Conjugate gradient iterative solver"#,
    },
    // ===== Image Processing (ndimage) (3) =====
    MigrationEntry {
        scipy_path: "scipy.ndimage.gaussian_filter",
        scirs2_path: "scirs2_ndimage",
        category: MigrationCategory::ImageProcessing,
        notes: "Gaussian smoothing filter. Use scirs2_ndimage filtering module.",
        scipy_example: r#"from scipy.ndimage import gaussian_filter
smoothed = gaussian_filter(image, sigma=1.0)"#,
        scirs2_example: r#"use scirs2_ndimage;
// Gaussian filtering and other image processing operations"#,
    },
    MigrationEntry {
        scipy_path: "scipy.ndimage.median_filter",
        scirs2_path: "scirs2_ndimage",
        category: MigrationCategory::ImageProcessing,
        notes: "Median filter for noise removal.",
        scipy_example: r#"from scipy.ndimage import median_filter
filtered = median_filter(image, size=3)"#,
        scirs2_example: r#"use scirs2_ndimage;
// Median filtering"#,
    },
    MigrationEntry {
        scipy_path: "scipy.ndimage.label",
        scirs2_path: "scirs2_ndimage",
        category: MigrationCategory::ImageProcessing,
        notes: "Connected component labeling.",
        scipy_example: r#"from scipy.ndimage import label
labeled, num_features = label(binary_image)"#,
        scirs2_example: r#"use scirs2_ndimage;
// Connected component labeling and morphological operations"#,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_table_not_empty() {
        assert!(!migration_table().is_empty());
        assert_eq!(migration_table().len(), 85);
    }

    #[test]
    fn test_search_scipy_det() {
        let results = search_scipy("linalg.det");
        assert!(!results.is_empty());
        assert!(results.iter().any(|e| e.scipy_path == "scipy.linalg.det"));
    }

    #[test]
    fn test_search_scipy_partial_match() {
        let results = search_scipy("norm");
        // Should find scipy.linalg.norm and scipy.stats.norm and scipy.stats.lognorm
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_by_category_linalg() {
        let results = by_category(MigrationCategory::LinearAlgebra);
        assert!(results.len() >= 10);
        assert!(results
            .iter()
            .all(|e| e.category == MigrationCategory::LinearAlgebra));
    }

    #[test]
    fn test_all_categories_have_entries() {
        let categories = [
            MigrationCategory::LinearAlgebra,
            MigrationCategory::Statistics,
            MigrationCategory::SignalProcessing,
            MigrationCategory::Optimization,
            MigrationCategory::Integration,
            MigrationCategory::Interpolation,
            MigrationCategory::FFT,
            MigrationCategory::SpecialFunctions,
            MigrationCategory::Sparse,
            MigrationCategory::ImageProcessing,
        ];
        for cat in &categories {
            let entries = by_category(*cat);
            assert!(!entries.is_empty(), "Category {:?} has no entries", cat);
        }
    }

    #[test]
    fn test_no_duplicate_scipy_paths() {
        let table = migration_table();
        let mut seen = std::collections::HashSet::new();
        for entry in table {
            assert!(
                seen.insert(entry.scipy_path),
                "Duplicate scipy_path: {}",
                entry.scipy_path
            );
        }
    }

    #[test]
    fn test_scirs2_paths_look_valid() {
        for entry in migration_table() {
            assert!(
                entry.scirs2_path.starts_with("scirs2_"),
                "scirs2_path '{}' does not start with 'scirs2_'",
                entry.scirs2_path
            );
        }
    }

    #[test]
    fn test_search_case_insensitive() {
        let upper = search_scipy("LINALG.DET");
        let lower = search_scipy("linalg.det");
        assert_eq!(upper.len(), lower.len());
        assert!(!upper.is_empty());
    }
}
