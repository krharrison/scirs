//! Programmatic API catalog for the SciRS2 ecosystem.
//!
//! Each [`ApiEntry`] documents a public API with its signature, mathematical background,
//! usage example, and cross-references to related functions.

/// Category of a public API entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ApiCategory {
    /// Linear algebra operations (det, inv, solve, norms)
    LinearAlgebra,
    /// Matrix decompositions (LU, QR, SVD, Cholesky, eigendecomposition)
    Decomposition,
    /// Descriptive and inferential statistics
    Statistics,
    /// Probability distributions (Normal, Beta, Gamma, etc.)
    Distribution,
    /// Hypothesis testing (t-test, chi-square, etc.)
    HypothesisTest,
    /// Signal processing (filters, spectral analysis)
    SignalProcessing,
    /// Fast Fourier Transform and related
    FFT,
    /// Numerical optimization (minimization, root-finding)
    Optimization,
    /// Numerical integration and ODE solvers
    Integration,
    /// Interpolation and curve fitting
    Interpolation,
    /// Special mathematical functions (gamma, erf, Bessel)
    SpecialFunction,
    /// Sparse matrix operations
    Sparse,
    /// Image processing and computer vision
    ImageProcessing,
    /// Machine learning and neural networks
    MachineLearning,
    /// Time series analysis and forecasting
    TimeSeries,
}

impl core::fmt::Display for ApiCategory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ApiCategory::LinearAlgebra => write!(f, "Linear Algebra"),
            ApiCategory::Decomposition => write!(f, "Decomposition"),
            ApiCategory::Statistics => write!(f, "Statistics"),
            ApiCategory::Distribution => write!(f, "Distribution"),
            ApiCategory::HypothesisTest => write!(f, "Hypothesis Test"),
            ApiCategory::SignalProcessing => write!(f, "Signal Processing"),
            ApiCategory::FFT => write!(f, "FFT"),
            ApiCategory::Optimization => write!(f, "Optimization"),
            ApiCategory::Integration => write!(f, "Integration"),
            ApiCategory::Interpolation => write!(f, "Interpolation"),
            ApiCategory::SpecialFunction => write!(f, "Special Function"),
            ApiCategory::Sparse => write!(f, "Sparse"),
            ApiCategory::ImageProcessing => write!(f, "Image Processing"),
            ApiCategory::MachineLearning => write!(f, "Machine Learning"),
            ApiCategory::TimeSeries => write!(f, "Time Series"),
            // Future categories added via #[non_exhaustive]
            #[allow(unreachable_patterns)]
            _ => write!(f, "Other"),
        }
    }
}

/// A documented API entry with mathematical background and usage example.
#[derive(Debug, Clone)]
pub struct ApiEntry {
    /// Crate that provides this API (e.g. "scirs2-linalg")
    pub crate_name: &'static str,
    /// Module path within the crate (e.g. "decomposition")
    pub module_path: &'static str,
    /// Function or type name (e.g. "svd")
    pub function_name: &'static str,
    /// Rust signature (e.g. `pub fn svd<F>(...) -> LinalgResult<...>`)
    pub signature: &'static str,
    /// Human-readable description
    pub description: &'static str,
    /// Mathematical formula or reference in plain text
    pub math_reference: &'static str,
    /// Complete Rust code example (syntactically correct)
    pub example: &'static str,
    /// Related API names for cross-reference
    pub see_also: &'static [&'static str],
    /// API category
    pub category: ApiCategory,
}

/// Returns the full static API catalog.
pub fn api_catalog() -> &'static [ApiEntry] {
    &API_CATALOG
}

/// Search the catalog by query string (case-insensitive substring match
/// against function name, description, and module path).
pub fn search_api(query: &str) -> Vec<&'static ApiEntry> {
    let q = query.to_ascii_lowercase();
    API_CATALOG
        .iter()
        .filter(|e| {
            e.function_name.to_ascii_lowercase().contains(&q)
                || e.description.to_ascii_lowercase().contains(&q)
                || e.module_path.to_ascii_lowercase().contains(&q)
                || e.crate_name.to_ascii_lowercase().contains(&q)
        })
        .collect()
}

/// Filter catalog entries by crate name (case-insensitive).
pub fn by_crate(crate_name: &str) -> Vec<&'static ApiEntry> {
    let cn = crate_name.to_ascii_lowercase();
    API_CATALOG
        .iter()
        .filter(|e| e.crate_name.to_ascii_lowercase() == cn)
        .collect()
}

/// Filter catalog entries by category.
pub fn by_category(cat: ApiCategory) -> Vec<&'static ApiEntry> {
    API_CATALOG.iter().filter(|e| e.category == cat).collect()
}

// ---------------------------------------------------------------------------
// Static catalog
// ---------------------------------------------------------------------------

static API_CATALOG: [ApiEntry; 60] = [
    // ======================================================================
    // LINEAR ALGEBRA (indices 0-4)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "basic",
        function_name: "det",
        signature: "pub fn det<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>",
        description: "Compute the determinant of a square matrix via LU factorization.",
        math_reference: "det(A) = product of diagonal entries of U after PA = LU factorization. \
            For an n x n matrix, det(A) = (-1)^s * prod(U_ii) where s is the number of row swaps.",
        example: r#"use scirs2_linalg::basic::det;
use scirs2_core::ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let d = det(&a.view(), None).expect("det failed");
// d is approximately -2.0
assert!((d - (-2.0)).abs() < 1e-10);"#,
        see_also: &["inv", "lu", "solve"],
        category: ApiCategory::LinearAlgebra,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "basic",
        function_name: "inv",
        signature: "pub fn inv<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>",
        description: "Compute the inverse of a square matrix using LU decomposition with partial pivoting.",
        math_reference: "A^{-1} such that A * A^{-1} = I. \
            Computed via PA = LU, then solving LU * X = P for each column of identity.",
        example: r#"use scirs2_linalg::basic::inv;
use scirs2_core::ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let a_inv = inv(&a.view(), None).expect("inv failed");
// a_inv is approximately [[-2.0, 1.0], [1.5, -0.5]]"#,
        see_also: &["det", "solve", "lu"],
        category: ApiCategory::LinearAlgebra,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "matrix_equations",
        function_name: "solve_sylvester",
        signature: "pub fn solve_sylvester<A: MatEqFloat>(a: &ArrayView2<A>, b: &ArrayView2<A>, c: &ArrayView2<A>) -> LinalgResult<Array2<A>>",
        description: "Solve the Sylvester equation AX + XB = C using the Bartels-Stewart algorithm.",
        math_reference: "Find X such that AX + XB = C. \
            Uses Schur decomposition of A and B, then solves the triangular Sylvester equation.",
        example: r#"use scirs2_linalg::matrix_equations::solve_sylvester;
use scirs2_core::ndarray::array;

let a = array![[1.0, 0.0], [0.0, 2.0]];
let b = array![[3.0, 0.0], [0.0, 4.0]];
let c = array![[4.0, 0.0], [0.0, 6.0]];
let x = solve_sylvester(&a.view(), &b.view(), &c.view())
    .expect("solve_sylvester failed");
// x[0][0] should be 1.0 (since 1*1 + 1*3 = 4)"#,
        see_also: &["solve_continuous_lyapunov", "solve_discrete_lyapunov"],
        category: ApiCategory::LinearAlgebra,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "special",
        function_name: "expm",
        signature: "pub fn expm<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>",
        description: "Compute the matrix exponential exp(A) using scaling and squaring with Pade approximation.",
        math_reference: "exp(A) = sum_{k=0}^{inf} A^k / k!. \
            Implemented via scaling-and-squaring: exp(A) = (exp(A/2^s))^{2^s} \
            where exp(A/2^s) is approximated by a Pade rational function.",
        example: r#"use scirs2_linalg::special::expm;
use scirs2_core::ndarray::array;

let a = array![[0.0, 1.0], [-1.0, 0.0]];
let e = expm(&a.view(), None).expect("expm failed");
// exp([[0,1],[-1,0]]) = [[cos(1), sin(1)], [-sin(1), cos(1)]]"#,
        see_also: &["det", "inv"],
        category: ApiCategory::LinearAlgebra,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "matrix_equations",
        function_name: "solve_continuous_riccati",
        signature: "pub fn solve_continuous_riccati<A: MatEqFloat>(a: &ArrayView2<A>, b: &ArrayView2<A>, q: &ArrayView2<A>, r: &ArrayView2<A>) -> LinalgResult<Array2<A>>",
        description: "Solve the continuous-time algebraic Riccati equation (CARE) for optimal control.",
        math_reference: "Find X such that A^T X + X A - X B R^{-1} B^T X + Q = 0. \
            Uses the Schur method on the 2n x 2n Hamiltonian matrix.",
        example: r#"use scirs2_linalg::matrix_equations::solve_continuous_riccati;
use scirs2_core::ndarray::array;

let a = array![[0.0, 1.0], [0.0, 0.0]];
let b = array![[0.0], [1.0]];
let q = array![[1.0, 0.0], [0.0, 1.0]];
let r = array![[1.0]];
let x = solve_continuous_riccati(&a.view(), &b.view(), &q.view(), &r.view())
    .expect("CARE solver failed");"#,
        see_also: &["solve_discrete_riccati", "solve_continuous_lyapunov"],
        category: ApiCategory::LinearAlgebra,
    },
    // ======================================================================
    // DECOMPOSITION (indices 5-14)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "lu",
        signature: "pub fn lu<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)>",
        description: "Compute the LU decomposition with partial pivoting, returning (P, L, U).",
        math_reference: "PA = LU where P is a permutation matrix, L is unit lower triangular, \
            and U is upper triangular. Uses Gaussian elimination with partial pivoting. \
            Complexity: O(2n^3/3) for an n x n matrix.",
        example: r#"use scirs2_linalg::decomposition::lu;
use scirs2_core::ndarray::array;

let a = array![[2.0, 1.0], [4.0, 3.0]];
let (p, l, u) = lu(&a.view(), None).expect("LU failed");
// P * A = L * U"#,
        see_also: &["det", "inv", "qr"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "qr",
        signature: "pub fn qr<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<(Array2<F>, Array2<F>)>",
        description: "Compute the QR decomposition using Householder reflections, returning (Q, R).",
        math_reference: "A = QR where Q is orthogonal (Q^T Q = I) and R is upper triangular. \
            Uses Householder reflections: each step zeroes out below-diagonal entries in one column. \
            Complexity: O(2mn^2 - 2n^3/3) for m x n matrix.",
        example: r#"use scirs2_linalg::decomposition::qr;
use scirs2_core::ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let (q, r) = qr(&a.view(), None).expect("QR failed");
// q is 3x3 orthogonal, r is 3x2 upper triangular"#,
        see_also: &["svd", "lu", "cholesky"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "svd",
        signature: "pub fn svd<F>(a: &ArrayView2<F>, workers: Option<usize>, full_matrices: bool) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>",
        description: "Compute the Singular Value Decomposition, returning (U, S, Vt).",
        math_reference: "A = U * diag(S) * V^T where U (m x m) and V (n x n) are orthogonal, \
            S contains non-negative singular values in decreasing order. \
            Uses bidiagonalization followed by QR iteration (Golub-Kahan). \
            Complexity: O(min(mn^2, m^2 n)).",
        example: r#"use scirs2_linalg::decomposition::svd;
use scirs2_core::ndarray::array;

let a = array![[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
let (u, s, vt) = svd(&a.view(), None, true).expect("SVD failed");
// s contains singular values [2.0, 1.0]"#,
        see_also: &["qr", "eig", "eigh"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "cholesky",
        signature: "pub fn cholesky<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>",
        description: "Compute the Cholesky decomposition of a symmetric positive-definite matrix, returning lower-triangular L.",
        math_reference: "A = L L^T where L is lower triangular with positive diagonal entries. \
            Only valid for symmetric positive-definite matrices. \
            L_jj = sqrt(A_jj - sum_{k<j} L_jk^2), \
            L_ij = (A_ij - sum_{k<j} L_ik L_jk) / L_jj for i > j. \
            Complexity: O(n^3/3).",
        example: r#"use scirs2_linalg::decomposition::cholesky;
use scirs2_core::ndarray::array;

let a = array![[4.0, 2.0], [2.0, 3.0]];
let l = cholesky(&a.view(), None).expect("Cholesky failed");
// L * L^T = A"#,
        see_also: &["lu", "qr", "eigh"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "eig_f64_lapack",
        signature: "pub fn eig_f64_lapack(a: &ArrayView2<f64>) -> EigResult",
        description: "Compute eigenvalues and eigenvectors of a general square matrix (may be complex).",
        math_reference: "Find lambda and v such that A v = lambda v. \
            For a general real matrix, eigenvalues may be complex conjugate pairs. \
            Uses the QR algorithm with implicit shifts after Hessenberg reduction. \
            Complexity: O(10n^3) in practice.",
        example: r#"use scirs2_linalg::decomposition::eig_f64_lapack;
use scirs2_core::ndarray::array;

let a = array![[0.0, -1.0], [1.0, 0.0]];
let result = eig_f64_lapack(&a.view());
// Eigenvalues are +/- i (complex)"#,
        see_also: &["eigh_f64_lapack", "svd", "qr"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "eigh_f64_lapack",
        signature: "pub fn eigh_f64_lapack(a: &ArrayView2<f64>) -> LinalgResult<(Array1<f64>, Array2<f64>)>",
        description: "Compute eigenvalues and eigenvectors of a real symmetric matrix (all eigenvalues are real).",
        math_reference: "For symmetric A, all eigenvalues are real and eigenvectors are orthogonal. \
            A = V diag(lambda) V^T where V is orthogonal. \
            Uses tridiagonal reduction + divide-and-conquer or QR iteration. \
            Complexity: O(4n^3/3) for tridiagonal + O(n^2) per eigenvalue.",
        example: r#"use scirs2_linalg::decomposition::eigh_f64_lapack;
use scirs2_core::ndarray::array;

let a = array![[2.0, 1.0], [1.0, 3.0]];
let (eigenvalues, eigenvectors) = eigh_f64_lapack(&a.view())
    .expect("eigh failed");
// eigenvalues are real, eigenvectors are orthogonal"#,
        see_also: &["eig_f64_lapack", "svd"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "lu_default",
        signature: "pub fn lu_default<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)>",
        description: "Convenience wrapper for LU decomposition with default worker count.",
        math_reference: "Same as lu: PA = LU decomposition with partial pivoting.",
        example: r#"use scirs2_linalg::decomposition::lu_default;
use scirs2_core::ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let (p, l, u) = lu_default(&a.view()).expect("LU failed");"#,
        see_also: &["lu", "qr_default", "svd_default"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "qr_default",
        signature: "pub fn qr_default<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>",
        description: "Convenience wrapper for QR decomposition with default worker count.",
        math_reference: "Same as qr: A = QR via Householder reflections.",
        example: r#"use scirs2_linalg::decomposition::qr_default;
use scirs2_core::ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let (q, r) = qr_default(&a.view()).expect("QR failed");"#,
        see_also: &["qr", "lu_default", "svd_default"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "svd_default",
        signature: "pub fn svd_default<F>(a: &ArrayView2<F>, full_matrices: bool) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>",
        description: "Convenience wrapper for SVD with default worker count.",
        math_reference: "Same as svd: A = U diag(S) V^T decomposition.",
        example: r#"use scirs2_linalg::decomposition::svd_default;
use scirs2_core::ndarray::array;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let (u, s, vt) = svd_default(&a.view(), false).expect("SVD failed");"#,
        see_also: &["svd", "qr_default", "lu_default"],
        category: ApiCategory::Decomposition,
    },
    ApiEntry {
        crate_name: "scirs2-linalg",
        module_path: "decomposition",
        function_name: "cholesky_default",
        signature: "pub fn cholesky_default<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>",
        description: "Convenience wrapper for Cholesky decomposition with default worker count.",
        math_reference: "Same as cholesky: A = L L^T for SPD matrices.",
        example: r#"use scirs2_linalg::decomposition::cholesky_default;
use scirs2_core::ndarray::array;

let a = array![[4.0, 2.0], [2.0, 3.0]];
let l = cholesky_default(&a.view()).expect("Cholesky failed");"#,
        see_also: &["cholesky", "lu_default"],
        category: ApiCategory::Decomposition,
    },
    // ======================================================================
    // STATISTICS (indices 15-19)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "simd_enhanced_core",
        function_name: "mean_enhanced",
        signature: "pub fn mean_enhanced<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<F>",
        description: "Compute the arithmetic mean of a 1-D array using SIMD-enhanced accumulation.",
        math_reference: "mean(x) = (1/n) * sum_{i=1}^{n} x_i. \
            Uses compensated (Kahan) summation for numerical stability.",
        example: r#"use scirs2_stats::simd_enhanced_core::mean_enhanced;
use scirs2_core::ndarray::array;

let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
let m: f64 = mean_enhanced(&data).expect("mean failed");
assert!((m - 3.0).abs() < 1e-10);"#,
        see_also: &["variance_enhanced", "median_abs_deviation_simd"],
        category: ApiCategory::Statistics,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "simd_enhanced_core",
        function_name: "variance_enhanced",
        signature: "pub fn variance_enhanced<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>",
        description: "Compute the sample variance of a 1-D array with configurable degrees-of-freedom correction.",
        math_reference: "var(x, ddof) = (1/(n - ddof)) * sum_{i=1}^{n} (x_i - mean(x))^2. \
            ddof=0 gives population variance; ddof=1 gives unbiased sample variance (Bessel's correction).",
        example: r#"use scirs2_stats::simd_enhanced_core::variance_enhanced;
use scirs2_core::ndarray::array;

let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
let v: f64 = variance_enhanced(&data, 1).expect("variance failed");
// Unbiased sample variance"#,
        see_also: &["mean_enhanced", "median_abs_deviation_simd"],
        category: ApiCategory::Statistics,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "correlation_simd",
        function_name: "pearson_r_simd",
        signature: "pub fn pearson_r_simd<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>",
        description: "Compute the Pearson product-moment correlation coefficient using SIMD-accelerated dot products.",
        math_reference: "r = cov(x, y) / (std(x) * std(y)) \
            = [n sum(xy) - sum(x)sum(y)] / sqrt([n sum(x^2) - (sum x)^2][n sum(y^2) - (sum y)^2]). \
            Range: [-1, 1]. Measures linear relationship strength.",
        example: r#"use scirs2_stats::correlation_simd::pearson_r_simd;
use scirs2_core::ndarray::array;

let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
let r: f64 = pearson_r_simd(&x, &y).expect("pearson_r failed");
assert!((r - 1.0).abs() < 1e-10); // Perfect positive correlation"#,
        see_also: &["spearman_r_simd", "ttest_ind"],
        category: ApiCategory::Statistics,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "correlation_simd_enhanced",
        function_name: "spearman_r_simd",
        signature: "pub fn spearman_r_simd<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>",
        description: "Compute Spearman's rank correlation coefficient using SIMD-accelerated ranking.",
        math_reference: "rho = 1 - 6 sum(d_i^2) / (n(n^2 - 1)) where d_i = rank(x_i) - rank(y_i). \
            Measures monotonic (not necessarily linear) relationship. Range: [-1, 1].",
        example: r#"use scirs2_stats::correlation_simd_enhanced::spearman_r_simd;
use scirs2_core::ndarray::array;

let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![5.0, 6.0, 7.0, 8.0, 7.0];
let rho: f64 = spearman_r_simd(&x, &y).expect("spearman_r failed");"#,
        see_also: &["pearson_r_simd", "ttest_ind"],
        category: ApiCategory::Statistics,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "dispersion_simd",
        function_name: "median_abs_deviation_simd",
        signature: "pub fn median_abs_deviation_simd<F, D>(data: &ArrayBase<D, Ix1>, scale: F) -> StatsResult<F>",
        description: "Compute the Median Absolute Deviation (MAD), a robust measure of spread.",
        math_reference: "MAD = median(|x_i - median(x)|). \
            Scaled MAD = scale * MAD; with scale = 1.4826, it estimates std for normal data. \
            Breakdown point: 50% (robust to up to half the data being outliers).",
        example: r#"use scirs2_stats::dispersion_simd::median_abs_deviation_simd;
use scirs2_core::ndarray::array;

let data = array![1.0, 2.0, 3.0, 4.0, 100.0]; // outlier at 100
let mad: f64 = median_abs_deviation_simd(&data, 1.0)
    .expect("MAD failed");
// MAD is robust to the outlier at 100"#,
        see_also: &["variance_enhanced", "mean_enhanced"],
        category: ApiCategory::Statistics,
    },
    // ======================================================================
    // DISTRIBUTIONS (indices 20-24)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "distributions::normal",
        function_name: "Normal::new",
        signature: "pub fn new(loc: F, scale: F) -> StatsResult<Normal<F>>",
        description: "Create a Normal (Gaussian) distribution with given mean and standard deviation.",
        math_reference: "PDF: f(x) = (1 / (sigma * sqrt(2 pi))) * exp(-(x - mu)^2 / (2 sigma^2)). \
            CDF: Phi(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2)))). \
            Mean = mu, Variance = sigma^2.",
        example: r#"use scirs2_stats::distributions::normal::Normal;
use scirs2_stats::traits::Distribution;

let n = Normal::new(0.0_f64, 1.0).expect("Normal::new failed");
let p = n.pdf(0.0); // ~0.3989 (peak of standard normal)
let c = n.cdf(0.0); // 0.5 (symmetric about mean)"#,
        see_also: &["Beta::new", "Gamma::new", "Uniform::new"],
        category: ApiCategory::Distribution,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "distributions::beta",
        function_name: "Beta::new",
        signature: "pub fn new(alpha: F, beta: F) -> StatsResult<Beta<F>>",
        description: "Create a Beta distribution with shape parameters alpha and beta on [0, 1].",
        math_reference: "PDF: f(x; a, b) = x^{a-1} (1-x)^{b-1} / B(a, b) for x in [0, 1]. \
            B(a, b) = Gamma(a) Gamma(b) / Gamma(a+b) is the beta function. \
            Mean = a/(a+b), Variance = ab / ((a+b)^2 (a+b+1)).",
        example: r#"use scirs2_stats::distributions::beta::Beta;
use scirs2_stats::traits::Distribution;

let b = Beta::new(2.0_f64, 5.0).expect("Beta::new failed");
let p = b.pdf(0.3);
let c = b.cdf(0.5);"#,
        see_also: &["Normal::new", "Gamma::new", "Uniform::new"],
        category: ApiCategory::Distribution,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "distributions::gamma",
        function_name: "Gamma::new",
        signature: "pub fn new(shape: F, scale: F) -> StatsResult<Gamma<F>>",
        description: "Create a Gamma distribution with given shape (a) and scale (theta) parameters.",
        math_reference: "PDF: f(x; a, theta) = x^{a-1} exp(-x/theta) / (theta^a Gamma(a)) for x > 0. \
            Mean = a * theta, Variance = a * theta^2. \
            Special cases: a=1 gives Exponential; a=n/2, theta=2 gives Chi-squared(n).",
        example: r#"use scirs2_stats::distributions::gamma::Gamma;
use scirs2_stats::traits::Distribution;

let g = Gamma::new(2.0_f64, 1.0).expect("Gamma::new failed");
let p = g.pdf(1.0);
let c = g.cdf(2.0);"#,
        see_also: &["Beta::new", "Normal::new", "Poisson::new"],
        category: ApiCategory::Distribution,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "distributions::poisson",
        function_name: "Poisson::new",
        signature: "pub fn new(mu: F) -> StatsResult<Poisson<F>>",
        description: "Create a Poisson distribution with rate parameter mu.",
        math_reference: "PMF: P(X = k) = mu^k exp(-mu) / k! for k = 0, 1, 2, ... \
            Mean = mu, Variance = mu. \
            Models the number of events in a fixed interval given constant average rate.",
        example: r#"use scirs2_stats::distributions::poisson::Poisson;
use scirs2_stats::traits::Distribution;

let p = Poisson::new(3.0_f64).expect("Poisson::new failed");
let prob = p.pmf(2.0); // P(X = 2) for Poisson(3)"#,
        see_also: &["Binomial::new", "Normal::new", "Gamma::new"],
        category: ApiCategory::Distribution,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "distributions::uniform",
        function_name: "Uniform::new",
        signature: "pub fn new(low: F, high: F) -> StatsResult<Uniform<F>>",
        description: "Create a continuous Uniform distribution on the interval [low, high].",
        math_reference: "PDF: f(x) = 1 / (high - low) for x in [low, high], 0 otherwise. \
            CDF: F(x) = (x - low) / (high - low). \
            Mean = (low + high) / 2, Variance = (high - low)^2 / 12.",
        example: r#"use scirs2_stats::distributions::uniform::Uniform;
use scirs2_stats::traits::Distribution;

let u = Uniform::new(0.0_f64, 1.0).expect("Uniform::new failed");
let p = u.pdf(0.5); // 1.0
let c = u.cdf(0.75); // 0.75"#,
        see_also: &["Normal::new", "Beta::new"],
        category: ApiCategory::Distribution,
    },
    // ======================================================================
    // HYPOTHESIS TESTS (indices 25-28)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "tests::ttest",
        function_name: "ttest_1samp",
        signature: "pub fn ttest_1samp<F>(x: &ArrayView1<F>, popmean: F) -> StatsResult<(F, F)>",
        description: "One-sample t-test: test whether the mean of a sample differs from a hypothesized population mean.",
        math_reference: "t = (mean(x) - mu_0) / (s / sqrt(n)) where s is sample std dev. \
            Under H0: mean = mu_0, t follows Student's t-distribution with n-1 degrees of freedom. \
            Returns (t_statistic, two-sided p-value).",
        example: r#"use scirs2_stats::tests::ttest::ttest_1samp;
use scirs2_core::ndarray::array;

let x = array![5.1, 4.9, 5.2, 5.0, 4.8, 5.1];
let (t_stat, p_value) = ttest_1samp(&x.view(), 5.0_f64)
    .expect("ttest_1samp failed");
// Test whether sample mean differs from 5.0"#,
        see_also: &["ttest_ind", "ttest_rel"],
        category: ApiCategory::HypothesisTest,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "tests::ttest",
        function_name: "ttest_ind",
        signature: "pub fn ttest_ind<F>(x: &ArrayView1<F>, y: &ArrayView1<F>, equal_var: bool) -> StatsResult<(F, F)>",
        description: "Independent two-sample t-test: test whether the means of two independent samples differ.",
        math_reference: "If equal_var=true (Student): t = (mean_x - mean_y) / (s_p * sqrt(1/n_x + 1/n_y)) \
            where s_p^2 = ((n_x-1)s_x^2 + (n_y-1)s_y^2) / (n_x + n_y - 2). \
            If equal_var=false (Welch): uses Welch-Satterthwaite df approximation. \
            Returns (t_statistic, two-sided p-value).",
        example: r#"use scirs2_stats::tests::ttest::ttest_ind;
use scirs2_core::ndarray::array;

let group_a = array![5.1, 4.9, 5.2, 5.0, 4.8];
let group_b = array![5.5, 5.3, 5.6, 5.4, 5.7];
let (t_stat, p_value) = ttest_ind(&group_a.view(), &group_b.view(), false)
    .expect("ttest_ind failed");
// Welch's t-test for unequal variances"#,
        see_also: &["ttest_1samp", "ttest_rel", "pearson_r_simd"],
        category: ApiCategory::HypothesisTest,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "tests::ttest",
        function_name: "ttest_rel",
        signature: "pub fn ttest_rel<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> StatsResult<(F, F)>",
        description: "Paired t-test: test whether the mean difference of paired observations is zero.",
        math_reference: "t = mean(d) / (s_d / sqrt(n)) where d_i = x_i - y_i, \
            s_d is the std dev of differences. df = n - 1. \
            More powerful than independent test when observations are naturally paired.",
        example: r#"use scirs2_stats::tests::ttest::ttest_rel;
use scirs2_core::ndarray::array;

let before = array![200.0, 210.0, 190.0, 220.0, 205.0];
let after = array![195.0, 205.0, 185.0, 212.0, 198.0];
let (t_stat, p_value) = ttest_rel(&before.view(), &after.view())
    .expect("ttest_rel failed");"#,
        see_also: &["ttest_1samp", "ttest_ind"],
        category: ApiCategory::HypothesisTest,
    },
    ApiEntry {
        crate_name: "scirs2-stats",
        module_path: "tests::ttest",
        function_name: "ttest_ind_from_stats",
        signature: "pub fn ttest_ind_from_stats<F>(mean1: F, std1: F, nobs1: F, mean2: F, std2: F, nobs2: F, equal_var: bool) -> StatsResult<(F, F)>",
        description: "Two-sample t-test from summary statistics (means, standard deviations, sample sizes).",
        math_reference: "Same formulas as ttest_ind, but accepts pre-computed statistics. \
            Useful when raw data is not available (e.g., from published papers).",
        example: r#"use scirs2_stats::tests::ttest::ttest_ind_from_stats;

let (t_stat, p_value) = ttest_ind_from_stats(
    5.0_f64, 1.0, 30.0,   // group 1: mean=5, std=1, n=30
    5.5_f64, 1.2, 35.0,   // group 2: mean=5.5, std=1.2, n=35
    false,                  // Welch's test
).expect("ttest_ind_from_stats failed");"#,
        see_also: &["ttest_ind", "ttest_1samp"],
        category: ApiCategory::HypothesisTest,
    },
    // ======================================================================
    // SIGNAL PROCESSING (indices 29-36)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "filter::iir",
        function_name: "butter",
        signature: "pub fn butter<T>(order: usize, cutoff: T, btype: FilterType, fs: Option<T>) -> SignalResult<(Vec<f64>, Vec<f64>)>",
        description: "Design a Butterworth IIR filter and return (b, a) coefficients.",
        math_reference: "|H(j omega)|^2 = 1 / (1 + (omega / omega_c)^{2N}) where N is the filter order. \
            Maximally flat magnitude response in the passband. \
            Poles are equally spaced on a circle in the s-plane.",
        example: r#"use scirs2_signal::filter::iir::butter;

// 4th-order lowpass Butterworth at 10 Hz, sampled at 100 Hz
let (b, a) = butter(4, 10.0, scirs2_signal::FilterType::Lowpass, Some(100.0))
    .expect("butter failed");"#,
        see_also: &["firwin", "lfilter", "filtfilt"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "filter::fir",
        function_name: "firwin",
        signature: "pub fn firwin<T>(numtaps: usize, cutoff: T, pass_zero: bool, fs: Option<T>) -> SignalResult<Vec<f64>>",
        description: "Design a FIR filter using the window method (Hamming window by default).",
        math_reference: "h[n] = w[n] * h_ideal[n] where h_ideal is the ideal impulse response \
            and w[n] is the window function. Linear phase guaranteed for symmetric coefficients. \
            numtaps controls filter length (higher = sharper transition).",
        example: r#"use scirs2_signal::filter::fir::firwin;

// 65-tap lowpass FIR at 10 Hz, sampled at 100 Hz
let h = firwin(65, 10.0, true, Some(100.0))
    .expect("firwin failed");
assert_eq!(h.len(), 65);"#,
        see_also: &["butter", "lfilter", "filtfilt"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "filter::application",
        function_name: "lfilter",
        signature: "pub fn lfilter<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>",
        description: "Apply a digital filter (IIR or FIR) to a signal using the direct-form II transposed structure.",
        math_reference: "y[n] = (1/a[0]) * (sum_{k=0}^{M} b[k] x[n-k] - sum_{k=1}^{N} a[k] y[n-k]). \
            For FIR: a = [1.0] (no feedback). Causal (forward-only) filtering introduces phase distortion.",
        example: r#"use scirs2_signal::filter::application::lfilter;

let b = vec![0.1, 0.2, 0.3, 0.2, 0.1]; // FIR coefficients
let a = vec![1.0]; // FIR (no feedback)
let signal = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
let filtered = lfilter(&b, &a, &signal).expect("lfilter failed");"#,
        see_also: &["filtfilt", "butter", "firwin"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "filter::application",
        function_name: "filtfilt",
        signature: "pub fn filtfilt<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>",
        description: "Zero-phase digital filtering: apply filter forward and backward to eliminate phase distortion.",
        math_reference: "y = lfilter(b, a, lfilter(b, a, x, forward), backward). \
            Effective transfer function: |H(omega)|^2 (magnitude squared, zero phase). \
            Result has zero phase distortion but squared magnitude response.",
        example: r#"use scirs2_signal::filter::application::filtfilt;

let b = vec![0.1, 0.2, 0.4, 0.2, 0.1];
let a = vec![1.0];
let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
let filtered = filtfilt(&b, &a, &signal).expect("filtfilt failed");"#,
        see_also: &["lfilter", "butter", "firwin"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "spectral::legacy",
        function_name: "welch",
        signature: "pub fn welch<T>(x: &[T], fs: f64, nperseg: usize, noverlap: Option<usize>) -> SignalResult<(Vec<f64>, Vec<f64>)>",
        description: "Estimate power spectral density using Welch's method (averaged modified periodograms).",
        math_reference: "PSD = (1/K) sum_{k=1}^{K} |FFT(w * x_k)|^2 / (fs * sum(w^2)) \
            where x_k are overlapping segments and w is a window function. \
            Reduces variance at the cost of frequency resolution compared to a single periodogram.",
        example: r#"use scirs2_signal::spectral::legacy::welch;

let fs = 1000.0;
let signal: Vec<f64> = (0..4096)
    .map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs).sin())
    .collect();
let (freqs, psd) = welch(&signal, fs, 256, None)
    .expect("welch failed");
// Peak in PSD near 50 Hz"#,
        see_also: &["stft", "spectrogram", "fft"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "spectral::legacy",
        function_name: "stft",
        signature: "pub fn stft<T>(x: &[T], fs: f64, nperseg: usize, noverlap: Option<usize>) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<Complex64>)>",
        description: "Compute the Short-Time Fourier Transform for time-frequency analysis.",
        math_reference: "STFT(t, f) = sum_{n} x[n] w[n - t] exp(-j 2 pi f n) \
            where w is a window function centered at time t. \
            Returns (frequencies, times, complex STFT matrix).",
        example: r#"use scirs2_signal::spectral::legacy::stft;

let fs = 1000.0;
let signal: Vec<f64> = (0..4096)
    .map(|i| (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin())
    .collect();
let (freqs, times, stft_matrix) = stft(&signal, fs, 256, None)
    .expect("stft failed");"#,
        see_also: &["welch", "spectrogram", "fft"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "spectral::legacy",
        function_name: "spectrogram",
        signature: "pub fn spectrogram<T>(x: &[T], fs: f64, nperseg: usize, noverlap: Option<usize>) -> SignalResult<(Vec<f64>, Vec<f64>, Array2<f64>)>",
        description: "Compute the spectrogram (magnitude-squared STFT) for time-frequency visualization.",
        math_reference: "S(t, f) = |STFT(t, f)|^2. \
            Returns (frequencies, times, power matrix). \
            Useful for visualizing how the spectral content of a signal changes over time.",
        example: r#"use scirs2_signal::spectral::legacy::spectrogram;

let fs = 1000.0;
let signal: Vec<f64> = (0..8192)
    .map(|i| {
        let t = i as f64 / fs;
        (2.0 * std::f64::consts::PI * (50.0 + 200.0 * t) * t).sin()
    })
    .collect();
let (freqs, times, power) = spectrogram(&signal, fs, 256, None)
    .expect("spectrogram failed");"#,
        see_also: &["stft", "welch", "fft"],
        category: ApiCategory::SignalProcessing,
    },
    ApiEntry {
        crate_name: "scirs2-signal",
        module_path: "filter::iir",
        function_name: "butter_bandpass_bandstop",
        signature: "pub fn butter_bandpass_bandstop(order: usize, low: f64, high: f64, btype: FilterType, fs: Option<f64>) -> SignalResult<(Vec<f64>, Vec<f64>)>",
        description: "Design a Butterworth bandpass or bandstop filter with low and high cutoff frequencies.",
        math_reference: "Bandpass: passes frequencies between low and high, attenuates outside. \
            Bandstop (notch): attenuates frequencies between low and high, passes outside. \
            Achieved via analog prototype lowpass-to-bandpass frequency transformation.",
        example: r#"use scirs2_signal::filter::iir::butter_bandpass_bandstop;

// 4th-order bandpass filter, 20-80 Hz, sampled at 500 Hz
let (b, a) = butter_bandpass_bandstop(
    4, 20.0, 80.0,
    scirs2_signal::FilterType::Bandpass,
    Some(500.0),
).expect("bandpass filter design failed");"#,
        see_also: &["butter", "firwin", "filtfilt"],
        category: ApiCategory::SignalProcessing,
    },
    // ======================================================================
    // FFT (indices 37-41)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-fft",
        module_path: "memory_efficient_v2",
        function_name: "fft_optimized",
        signature: "pub fn fft_optimized<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>",
        description: "Compute the 1-D discrete Fourier transform using an optimized Cooley-Tukey FFT algorithm.",
        math_reference: "X[k] = sum_{n=0}^{N-1} x[n] exp(-j 2 pi k n / N) for k = 0, ..., N-1. \
            Uses mixed-radix Cooley-Tukey decomposition (radix-2/4). \
            Complexity: O(N log N) vs O(N^2) for naive DFT.",
        example: r#"use scirs2_fft::memory_efficient_v2::fft_optimized;

let signal = vec![1.0, 0.0, -1.0, 0.0]; // simple test signal
let spectrum = fft_optimized(&signal, None)
    .expect("fft failed");
assert_eq!(spectrum.len(), 4);"#,
        see_also: &["ifft_optimized", "rfft_optimized", "fft_frequencies"],
        category: ApiCategory::FFT,
    },
    ApiEntry {
        crate_name: "scirs2-fft",
        module_path: "memory_efficient_v2",
        function_name: "ifft_optimized",
        signature: "pub fn ifft_optimized<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>",
        description: "Compute the 1-D inverse discrete Fourier transform.",
        math_reference: "x[n] = (1/N) sum_{k=0}^{N-1} X[k] exp(j 2 pi k n / N) for n = 0, ..., N-1. \
            IFFT(X) = conj(FFT(conj(X))) / N. Complexity: O(N log N).",
        example: r#"use scirs2_fft::memory_efficient_v2::{fft_optimized, ifft_optimized};
use scirs2_fft::num_complex::Complex64;

let signal = vec![1.0, 2.0, 3.0, 4.0];
let spectrum = fft_optimized(&signal, None).expect("fft failed");
let recovered = ifft_optimized(&spectrum, None).expect("ifft failed");
// recovered[i].re should be approximately signal[i]"#,
        see_also: &["fft_optimized", "rfft_optimized"],
        category: ApiCategory::FFT,
    },
    ApiEntry {
        crate_name: "scirs2-fft",
        module_path: "real_fft",
        function_name: "rfft_optimized",
        signature: "pub fn rfft_optimized<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>",
        description: "Compute the 1-D FFT of a real-valued signal, returning only the non-redundant half of the spectrum.",
        math_reference: "For real input of length N, the FFT satisfies X[k] = conj(X[N-k]), \
            so only N/2+1 complex values are needed. \
            Approximately 2x faster than complex FFT for real data.",
        example: r#"use scirs2_fft::real_fft::rfft_optimized;

let signal: Vec<f64> = (0..1024)
    .map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / 1000.0).sin())
    .collect();
let spectrum = rfft_optimized(&signal, None).expect("rfft failed");
assert_eq!(spectrum.len(), 513); // N/2 + 1"#,
        see_also: &["fft_optimized", "ifft_optimized"],
        category: ApiCategory::FFT,
    },
    ApiEntry {
        crate_name: "scirs2-fft",
        module_path: "multidimensional",
        function_name: "fft_frequencies",
        signature: "pub fn fft_frequencies(n: usize, sample_rate: f64) -> Vec<f64>",
        description: "Return the DFT sample frequencies for a signal of length n at the given sample rate.",
        math_reference: "f[k] = k * sample_rate / n for k = 0, ..., n/2, \
            then -(n/2-1)*sr/n, ..., -sr/n for the negative frequencies. \
            Matches the ordering of FFT output bins.",
        example: r#"use scirs2_fft::multidimensional::fft_frequencies;

let freqs = fft_frequencies(1024, 1000.0);
assert_eq!(freqs.len(), 1024);
// freqs[0] = 0.0 (DC), freqs[1] ~ 0.977 Hz, ..."#,
        see_also: &["fft_optimized", "rfft_optimized"],
        category: ApiCategory::FFT,
    },
    ApiEntry {
        crate_name: "scirs2-fft",
        module_path: "memory_efficient_v2",
        function_name: "fft2_optimized",
        signature: "pub fn fft2_optimized<T>(x: &[Vec<T>], shape: Option<(usize, usize)>) -> FFTResult<Vec<Vec<Complex64>>>",
        description: "Compute the 2-D discrete Fourier transform of a matrix.",
        math_reference: "X[k1, k2] = sum_{n1} sum_{n2} x[n1, n2] exp(-j 2 pi (k1 n1/N1 + k2 n2/N2)). \
            Computed as 1-D FFTs along rows, then 1-D FFTs along columns. \
            Complexity: O(N1 N2 log(N1 N2)).",
        example: r#"use scirs2_fft::memory_efficient_v2::fft2_optimized;

let matrix = vec![
    vec![1.0, 0.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0, 0.0],
    vec![0.0, 0.0, 1.0, 0.0],
    vec![0.0, 0.0, 0.0, 1.0],
];
let spectrum_2d = fft2_optimized(&matrix, None)
    .expect("fft2 failed");"#,
        see_also: &["fft_optimized", "ifft2_optimized"],
        category: ApiCategory::FFT,
    },
    // ======================================================================
    // OPTIMIZATION (indices 42-47)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-optimize",
        module_path: "unconstrained::lbfgs",
        function_name: "minimize_lbfgs",
        signature: "pub fn minimize_lbfgs<F, G, S>(f: F, grad: G, x0: &[f64], options: S) -> OptimizeResult<OptResult>",
        description: "Minimize a function using the L-BFGS quasi-Newton method (limited-memory BFGS).",
        math_reference: "Approximates inverse Hessian using m most recent (s_k, y_k) pairs \
            where s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k. \
            Two-loop recursion computes H_k g_k in O(mn) time and O(mn) memory. \
            Superlinear convergence for smooth, unconstrained problems.",
        example: r#"use scirs2_optimize::unconstrained::lbfgs::minimize_lbfgs;

// Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
let grad = |x: &[f64]| vec![
    -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
    200.0 * (x[1] - x[0].powi(2)),
];
let x0 = vec![-1.0, 1.0];
let result = minimize_lbfgs(&f, &grad, &x0, ())
    .expect("L-BFGS failed");
// result.x should be near [1.0, 1.0]"#,
        see_also: &["minimize_lbfgsb", "minimize_conjugate_gradient", "minimize_trust_ncg"],
        category: ApiCategory::Optimization,
    },
    ApiEntry {
        crate_name: "scirs2-optimize",
        module_path: "unconstrained::lbfgs",
        function_name: "minimize_lbfgsb",
        signature: "pub fn minimize_lbfgsb<F, G, S>(f: F, grad: G, x0: &[f64], bounds: Option<Vec<(Option<f64>, Option<f64>)>>, options: S) -> OptimizeResult<OptResult>",
        description: "Minimize a function with optional box constraints using L-BFGS-B (bounded L-BFGS).",
        math_reference: "Extension of L-BFGS that handles bound constraints l_i <= x_i <= u_i. \
            Uses the gradient projection method and Cauchy point computation \
            to identify the active set, then applies L-BFGS within the free variables.",
        example: r#"use scirs2_optimize::unconstrained::lbfgs::minimize_lbfgsb;

let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
let bounds = Some(vec![
    (Some(0.5), None),    // x[0] >= 0.5
    (Some(-1.0), Some(1.0)), // -1 <= x[1] <= 1
]);
let result = minimize_lbfgsb(&f, &grad, &[2.0, 0.5], bounds, ())
    .expect("L-BFGS-B failed");
// x[0] should be 0.5 (active bound), x[1] should be 0.0"#,
        see_also: &["minimize_lbfgs", "minimize_trust_ncg"],
        category: ApiCategory::Optimization,
    },
    ApiEntry {
        crate_name: "scirs2-optimize",
        module_path: "scalar",
        function_name: "minimize_scalar",
        signature: "pub fn minimize_scalar<F>(f: F, bracket: Option<(f64, f64, f64)>, bounds: Option<(f64, f64)>, method: ScalarMethod) -> OptimizeResult<ScalarResult>",
        description: "Minimize a scalar function of one variable using Brent's method or golden section search.",
        math_reference: "Brent's method: combines golden section search with parabolic interpolation. \
            Guaranteed convergence with superlinear rate in smooth regions. \
            Tolerance: finds x* such that |x - x*| < tol, where tol ~ sqrt(machine epsilon).",
        example: r#"use scirs2_optimize::scalar::minimize_scalar;

let f = |x: &f64| (x - 3.0).powi(2) + 1.0;
// Minimize (x-3)^2 + 1 near x=3"#,
        see_also: &["minimize_lbfgs", "root"],
        category: ApiCategory::Optimization,
    },
    ApiEntry {
        crate_name: "scirs2-optimize",
        module_path: "roots",
        function_name: "root",
        signature: "pub fn root<F, J, S>(f: F, x0: &[f64], method: RootMethod, jac: Option<J>, options: S) -> OptimizeResult<RootResult>",
        description: "Find a root of a vector-valued function F(x) = 0 using Newton, Broyden, or other methods.",
        math_reference: "Newton: x_{k+1} = x_k - J(x_k)^{-1} F(x_k) where J is the Jacobian. \
            Quadratic convergence near the root. \
            Broyden: approximates Jacobian via rank-1 updates (no Jacobian computation needed). \
            Krylov: uses GMRES to solve the Newton system inexactly.",
        example: r#"use scirs2_optimize::roots::root;

// Find root of f(x) = [x[0]^2 + x[1] - 1, x[0] - x[1]^2 + 1]
let f = |x: &[f64]| vec![
    x[0] * x[0] + x[1] - 1.0,
    x[0] - x[1] * x[1] + 1.0,
];"#,
        see_also: &["minimize_lbfgs", "minimize_scalar"],
        category: ApiCategory::Optimization,
    },
    ApiEntry {
        crate_name: "scirs2-optimize",
        module_path: "stochastic::adam",
        function_name: "minimize_adam",
        signature: "pub fn minimize_adam<F>(f: F, x0: &[f64], learning_rate: f64, max_iter: usize) -> OptimizeResult<OptResult>",
        description: "Minimize a function using the Adam optimizer (Adaptive Moment Estimation).",
        math_reference: "m_t = beta1 * m_{t-1} + (1-beta1) * g_t (1st moment). \
            v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2 (2nd moment). \
            x_{t+1} = x_t - lr * m_hat_t / (sqrt(v_hat_t) + eps) \
            where hat denotes bias-corrected estimates. Default: beta1=0.9, beta2=0.999.",
        example: r#"use scirs2_optimize::stochastic::adam::minimize_adam;

let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
let result = minimize_adam(&f, &[5.0, 5.0], 0.01, 1000)
    .expect("Adam failed");"#,
        see_also: &["minimize_sgd", "minimize_lbfgs"],
        category: ApiCategory::Optimization,
    },
    ApiEntry {
        crate_name: "scirs2-optimize",
        module_path: "unconstrained::conjugate_gradient",
        function_name: "minimize_conjugate_gradient",
        signature: "pub fn minimize_conjugate_gradient<F, G, S>(f: F, grad: G, x0: &[f64], options: S) -> OptimizeResult<OptResult>",
        description: "Minimize a function using the nonlinear conjugate gradient method (Fletcher-Reeves or Polak-Ribiere).",
        math_reference: "d_k = -g_k + beta_k d_{k-1} where beta_k controls the conjugate direction. \
            Fletcher-Reeves: beta_k = ||g_k||^2 / ||g_{k-1}||^2. \
            Polak-Ribiere: beta_k = g_k^T (g_k - g_{k-1}) / ||g_{k-1}||^2. \
            Convergence: n-step quadratic termination for quadratic objectives.",
        example: r#"use scirs2_optimize::unconstrained::conjugate_gradient::minimize_conjugate_gradient;

let f = |x: &[f64]| x[0].powi(2) + 2.0 * x[1].powi(2);
let grad = |x: &[f64]| vec![2.0 * x[0], 4.0 * x[1]];
let result = minimize_conjugate_gradient(&f, &grad, &[5.0, 3.0], ())
    .expect("CG failed");"#,
        see_also: &["minimize_lbfgs", "minimize_trust_ncg"],
        category: ApiCategory::Optimization,
    },
    // ======================================================================
    // INTEGRATION (indices 48-52)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-integrate",
        module_path: "quad",
        function_name: "quad",
        signature: "pub fn quad<F, Func>(f: Func, a: F, b: F, opts: QuadOpts<F>) -> IntegrateResult<(F, F)>",
        description: "Adaptive numerical integration using Gauss-Kronrod quadrature, returning (result, error_estimate).",
        math_reference: "Uses Gauss-Kronrod rule: G_n = sum w_i f(x_i) (n-point Gauss), \
            K_{2n+1} = sum w'_i f(x'_i) (2n+1 point Kronrod extension). \
            Error estimate: |G_n - K_{2n+1}|. Recursively bisects intervals with large errors. \
            Default: G7-K15 (7-point Gauss, 15-point Kronrod).",
        example: r#"use scirs2_integrate::quad::quad;

// Integrate sin(x) from 0 to pi => result should be 2.0
let f = |x: f64| x.sin();
let (result, error) = quad(f, 0.0, std::f64::consts::PI, Default::default())
    .expect("quad failed");
assert!((result - 2.0).abs() < 1e-10);"#,
        see_also: &["trapezoid", "simpson", "solve_ivp"],
        category: ApiCategory::Integration,
    },
    ApiEntry {
        crate_name: "scirs2-integrate",
        module_path: "quad",
        function_name: "trapezoid",
        signature: "pub fn trapezoid<F, Func>(f: Func, a: F, b: F, n: usize) -> F",
        description: "Compute a definite integral using the composite trapezoidal rule with n subintervals.",
        math_reference: "integral ~ h * [f(a)/2 + sum_{i=1}^{n-1} f(a + ih) + f(b)/2] where h = (b-a)/n. \
            Error: O(h^2) for smooth functions (second-order). \
            Exact for piecewise linear functions.",
        example: r#"use scirs2_integrate::quad::trapezoid;

let f = |x: f64| x * x; // x^2
let result = trapezoid(f, 0.0, 1.0, 1000);
// integral of x^2 from 0 to 1 = 1/3
assert!((result - 1.0 / 3.0).abs() < 1e-5);"#,
        see_also: &["simpson", "quad"],
        category: ApiCategory::Integration,
    },
    ApiEntry {
        crate_name: "scirs2-integrate",
        module_path: "quad",
        function_name: "simpson",
        signature: "pub fn simpson<F, Func>(f: Func, a: F, b: F, n: usize) -> IntegrateResult<F>",
        description: "Compute a definite integral using composite Simpson's 1/3 rule (n must be even).",
        math_reference: "integral ~ (h/3) * [f(a) + 4 sum_{odd} f(x_i) + 2 sum_{even} f(x_i) + f(b)] \
            where h = (b-a)/n. Error: O(h^4) (fourth-order). \
            Exact for polynomials up to degree 3.",
        example: r#"use scirs2_integrate::quad::simpson;

let f = |x: f64| x.powi(3); // x^3
let result = simpson(f, 0.0, 1.0, 100).expect("simpson failed");
// integral of x^3 from 0 to 1 = 0.25 (exact for cubic)
assert!((result - 0.25).abs() < 1e-12);"#,
        see_also: &["trapezoid", "quad"],
        category: ApiCategory::Integration,
    },
    ApiEntry {
        crate_name: "scirs2-integrate",
        module_path: "ode::solver",
        function_name: "solve_ivp",
        signature: "pub fn solve_ivp<F, Func>(f: Func, t_span: (F, F), y0: &[F], method: OdeMethod, options: IvpOptions<F>) -> IntegrateResult<OdeSolution<F>>",
        description: "Solve an initial value problem for a system of ordinary differential equations.",
        math_reference: "dy/dt = f(t, y), y(t0) = y0. \
            Methods: RK45 (Dormand-Prince, 4th/5th order adaptive), \
            RK23 (Bogacki-Shampine, 2nd/3rd order), \
            DOP853 (8th order Runge-Kutta). \
            Step size controlled by local error estimate: err < atol + rtol * |y|.",
        example: r#"use scirs2_integrate::ode::solver::solve_ivp;

// dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
let f = |_t: f64, y: &[f64]| vec![-y[0]];
let result = solve_ivp(
    f, (0.0, 5.0), &[1.0],
    Default::default(), Default::default(),
).expect("solve_ivp failed");"#,
        see_also: &["quad", "trapezoid"],
        category: ApiCategory::Integration,
    },
    ApiEntry {
        crate_name: "scirs2-integrate",
        module_path: "quadrature::gaussian",
        function_name: "quad_gauss_legendre",
        signature: "pub fn quad_gauss_legendre<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> IntegrateResult<f64>",
        description: "Numerical integration using Gauss-Legendre quadrature with n nodes.",
        math_reference: "integral ~ sum_{i=1}^{n} w_i f(x_i) where x_i are roots of P_n(x) (Legendre polynomial) \
            and w_i = 2 / ((1-x_i^2) [P'_n(x_i)]^2). \
            Exact for polynomials up to degree 2n-1. \
            Transformed to [a, b] via x = (b-a)/2 * t + (a+b)/2.",
        example: r#"use scirs2_integrate::quadrature::gaussian::quad_gauss_legendre;

let f = |x: f64| x.exp(); // e^x
let result = quad_gauss_legendre(f, 0.0, 1.0, 10)
    .expect("gauss-legendre failed");
// integral of e^x from 0 to 1 = e - 1
assert!((result - (std::f64::consts::E - 1.0)).abs() < 1e-12);"#,
        see_also: &["quad", "quad_gauss_hermite"],
        category: ApiCategory::Integration,
    },
    // ======================================================================
    // INTERPOLATION (indices 53-55)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-interpolate",
        module_path: "spline_modules::api",
        function_name: "interp1d_scipy",
        signature: "pub fn interp1d_scipy<F: InterpolationFloat>(x: &[F], y: &[F], kind: InterpKind) -> InterpolateResult<Box<dyn Fn(F) -> InterpolateResult<F>>>",
        description: "Create a 1-D interpolation function from data points (SciPy-compatible API).",
        math_reference: "Linear: y = y_i + (y_{i+1} - y_i) * (x - x_i) / (x_{i+1} - x_i). \
            Cubic spline: piecewise cubic polynomials S_i(x) with C2 continuity at knots. \
            Solves tridiagonal system for second derivatives.",
        example: r#"use scirs2_interpolate::spline_modules::api::interp1d_scipy;

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
// Create a cubic interpolant"#,
        see_also: &["griddata", "rbf_1d"],
        category: ApiCategory::Interpolation,
    },
    ApiEntry {
        crate_name: "scirs2-interpolate",
        module_path: "griddata",
        function_name: "griddata",
        signature: "pub fn griddata<F>(points: &[(F, F)], values: &[F], xi: &[(F, F)], method: GridMethod) -> InterpolateResult<Vec<F>>",
        description: "Interpolate unstructured 2-D data onto a grid using nearest, linear, or cubic methods.",
        math_reference: "Nearest: value at closest data point (Voronoi cell assignment). \
            Linear: Delaunay triangulation + barycentric interpolation within triangles. \
            Cubic: Clough-Tocher C1 interpolation on Delaunay triangles.",
        example: r#"use scirs2_interpolate::griddata::griddata;

let points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
let values = vec![0.0, 1.0, 1.0, 2.0];
let xi = vec![(0.5, 0.5)]; // query point"#,
        see_also: &["interp1d_scipy", "rbf_1d"],
        category: ApiCategory::Interpolation,
    },
    ApiEntry {
        crate_name: "scirs2-interpolate",
        module_path: "rbf",
        function_name: "rbf_1d",
        signature: "pub fn rbf_1d(x: &[f64], y: &[f64], kernel: RbfKernel, epsilon: Option<f64>) -> InterpolateResult<RbfInterpolator>",
        description: "Create a radial basis function interpolator for scattered 1-D data.",
        math_reference: "f(x) = sum_{i=1}^{n} w_i phi(||x - x_i||) where phi is the RBF kernel. \
            Kernels: Gaussian phi(r) = exp(-r^2/eps^2), Multiquadric phi(r) = sqrt(1 + (r/eps)^2), \
            Thin-plate spline phi(r) = r^2 log(r). Weights w solved via interpolation matrix.",
        example: r#"use scirs2_interpolate::rbf::rbf_1d;

let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![0.0, 0.8, 1.0, 0.8, 0.0]; // bell-shaped data"#,
        see_also: &["interp1d_scipy", "griddata"],
        category: ApiCategory::Interpolation,
    },
    // ======================================================================
    // SPECIAL FUNCTIONS (indices 56-59)
    // ======================================================================
    ApiEntry {
        crate_name: "scirs2-special",
        module_path: "gamma::enhanced",
        function_name: "gamma_enhanced",
        signature: "pub fn gamma_enhanced<F>(x: F) -> SpecialResult<F>",
        description: "Compute the Gamma function with enhanced precision using Lanczos approximation.",
        math_reference: "Gamma(x) = integral_0^inf t^{x-1} e^{-t} dt for x > 0. \
            Gamma(n) = (n-1)! for positive integers. \
            Reflection formula: Gamma(x) Gamma(1-x) = pi / sin(pi x). \
            Lanczos approx: Gamma(x+1) ~ sqrt(2 pi) (x + g + 0.5)^{x+0.5} e^{-(x+g+0.5)} sum_k c_k/(x+k).",
        example: r#"use scirs2_special::gamma::enhanced::gamma_enhanced;

let g5: f64 = gamma_enhanced(5.0).expect("gamma failed");
// Gamma(5) = 4! = 24.0
assert!((g5 - 24.0).abs() < 1e-10);

let g_half: f64 = gamma_enhanced(0.5).expect("gamma failed");
// Gamma(0.5) = sqrt(pi)
assert!((g_half - std::f64::consts::PI.sqrt()).abs() < 1e-10);"#,
        see_also: &["gammaln_enhanced", "gammainc", "erf"],
        category: ApiCategory::SpecialFunction,
    },
    ApiEntry {
        crate_name: "scirs2-special",
        module_path: "gamma::enhanced",
        function_name: "gammaln_enhanced",
        signature: "pub fn gammaln_enhanced<F>(x: F) -> SpecialResult<F>",
        description: "Compute the natural logarithm of the Gamma function (avoids overflow for large arguments).",
        math_reference: "lgamma(x) = ln(|Gamma(x)|). \
            For large x: lgamma(x) ~ x ln(x) - x - 0.5 ln(x) + 0.5 ln(2 pi) + Stirling corrections. \
            Essential for computing log-likelihoods with large factorials.",
        example: r#"use scirs2_special::gamma::enhanced::gammaln_enhanced;

let lg: f64 = gammaln_enhanced(100.0).expect("gammaln failed");
// ln(99!) ~ 363.739
assert!((lg - 363.73937555556347).abs() < 1e-6);"#,
        see_also: &["gamma_enhanced", "gammainc"],
        category: ApiCategory::SpecialFunction,
    },
    ApiEntry {
        crate_name: "scirs2-special",
        module_path: "incomplete_gamma",
        function_name: "gammainc",
        signature: "pub fn gammainc<T>(a: T, x: T) -> SpecialResult<T>",
        description: "Compute the regularized lower incomplete gamma function P(a, x).",
        math_reference: "P(a, x) = gamma(a, x) / Gamma(a) = (1/Gamma(a)) integral_0^x t^{a-1} e^{-t} dt. \
            Range: [0, 1]. P(a, x) + Q(a, x) = 1 where Q is the upper incomplete gamma. \
            Computed via series expansion for x < a+1, continued fraction otherwise.",
        example: r#"use scirs2_special::incomplete_gamma::gammainc;

let p: f64 = gammainc(1.0, 1.0).expect("gammainc failed");
// P(1, 1) = 1 - exp(-1) ~ 0.6321
assert!((p - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);"#,
        see_also: &["gammaincc", "gamma_enhanced"],
        category: ApiCategory::SpecialFunction,
    },
    ApiEntry {
        crate_name: "scirs2-special",
        module_path: "optimizations",
        function_name: "bessel_j0_fast",
        signature: "pub fn bessel_j0_fast(x: f64) -> f64",
        description: "Compute the Bessel function of the first kind J_0(x) using fast polynomial approximation.",
        math_reference: "J_0(x) = (1/pi) integral_0^pi cos(x sin(t)) dt. \
            For small x: J_0(x) = sum_{k=0}^inf (-1)^k (x/2)^{2k} / (k!)^2. \
            For large x: J_0(x) ~ sqrt(2/(pi x)) cos(x - pi/4). \
            J_0(0) = 1, J_0 has infinitely many zeros.",
        example: r#"use scirs2_special::optimizations::bessel_j0_fast;

let j0 = bessel_j0_fast(0.0);
assert!((j0 - 1.0).abs() < 1e-10); // J_0(0) = 1

let j0_at_1 = bessel_j0_fast(1.0);
// J_0(1) ~ 0.7652
assert!((j0_at_1 - 0.7651976865579666).abs() < 1e-4);"#,
        see_also: &["gamma_enhanced", "gammainc"],
        category: ApiCategory::SpecialFunction,
    },
];
