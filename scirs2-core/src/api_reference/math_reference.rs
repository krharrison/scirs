//! Mathematical reference for key algorithms used throughout SciRS2.
//!
//! Each [`MathReference`] describes an algorithm's mathematical foundation,
//! formula, computational complexity, and academic references.

/// A mathematical reference entry for a numerical algorithm.
#[derive(Debug, Clone)]
pub struct MathReference {
    /// Algorithm name (e.g. "LU Decomposition")
    pub algorithm: &'static str,
    /// Plain-text description of the algorithm
    pub description: &'static str,
    /// Mathematical formula or key equation in plain text
    pub formula: &'static str,
    /// Computational complexity (time and space)
    pub complexity: &'static str,
    /// Academic / textbook references
    pub references: &'static [&'static str],
}

/// Returns the full static math reference table.
pub fn math_references() -> &'static [MathReference] {
    &MATH_REFERENCES
}

/// Search math references by keyword (case-insensitive substring match).
pub fn search_math(query: &str) -> Vec<&'static MathReference> {
    let q = query.to_ascii_lowercase();
    MATH_REFERENCES
        .iter()
        .filter(|r| {
            r.algorithm.to_ascii_lowercase().contains(&q)
                || r.description.to_ascii_lowercase().contains(&q)
        })
        .collect()
}

static MATH_REFERENCES: [MathReference; 20] = [
    // 1. LU Decomposition
    MathReference {
        algorithm: "LU Decomposition",
        description: "Factor a square matrix A into a product of a lower triangular matrix L \
            and an upper triangular matrix U, with partial pivoting (PA = LU). \
            Foundation for solving linear systems, computing determinants, and matrix inversion.",
        formula: "PA = LU where P is permutation, L is unit lower triangular (L_ii = 1), U is upper triangular. \
            Forward substitution: Ly = Pb, O(n^2). \
            Back substitution: Ux = y, O(n^2). \
            Total solve: O(2n^3/3) factorization + O(n^2) per right-hand side.",
        complexity: "Time: O(2n^3/3) for factorization. Space: O(n^2) in-place.",
        references: &[
            "Golub & Van Loan, 'Matrix Computations', 4th ed., Ch. 3",
            "Trefethen & Bau, 'Numerical Linear Algebra', Lecture 20-21",
        ],
    },
    // 2. QR Decomposition (Householder)
    MathReference {
        algorithm: "QR Decomposition (Householder Reflections)",
        description: "Factor an m x n matrix A into Q (orthogonal, m x m) and R (upper triangular, m x n) \
            using Householder reflections. Each reflection zeroes out sub-diagonal entries of one column. \
            Numerically stable and the standard method for least-squares and eigenvalue algorithms.",
        formula: "A = QR where Q^T Q = I. Each Householder matrix: H_k = I - 2 v_k v_k^T / (v_k^T v_k). \
            v_k chosen so H_k zeroes entries below diagonal in column k. \
            For least squares: Rx = Q^T b (overdetermined systems).",
        complexity: "Time: O(2mn^2 - 2n^3/3) for m x n matrix. Space: O(mn).",
        references: &[
            "Golub & Van Loan, 'Matrix Computations', 4th ed., Ch. 5",
            "Trefethen & Bau, 'Numerical Linear Algebra', Lectures 7-11",
        ],
    },
    // 3. SVD
    MathReference {
        algorithm: "Singular Value Decomposition (SVD)",
        description: "Decompose any m x n matrix A = U Sigma V^T where U (m x m) and V (n x n) are orthogonal, \
            Sigma is diagonal with non-negative singular values sigma_1 >= sigma_2 >= ... >= 0. \
            The gold standard for rank determination, pseudoinverse, PCA, and low-rank approximation.",
        formula: "A = U diag(sigma_1, ..., sigma_r, 0, ..., 0) V^T. \
            Eckart-Young theorem: best rank-k approx is A_k = sum_{i=1}^k sigma_i u_i v_i^T. \
            ||A - A_k||_2 = sigma_{k+1}, ||A - A_k||_F = sqrt(sum_{i>k} sigma_i^2).",
        complexity: "Time: O(min(m n^2, m^2 n)) via Golub-Kahan bidiagonalization + QR iteration. Space: O(mn).",
        references: &[
            "Golub & Van Loan, 'Matrix Computations', 4th ed., Ch. 8",
            "Demmel, 'Applied Numerical Linear Algebra', Ch. 3",
        ],
    },
    // 4. Cholesky
    MathReference {
        algorithm: "Cholesky Decomposition",
        description: "Factor a symmetric positive-definite (SPD) matrix A = L L^T where L is lower triangular \
            with positive diagonal entries. About 2x faster than LU for SPD matrices. \
            Used in Gaussian processes, Kalman filters, and constrained optimization.",
        formula: "L_jj = sqrt(A_jj - sum_{k=0}^{j-1} L_jk^2). \
            L_ij = (A_ij - sum_{k=0}^{j-1} L_ik L_jk) / L_jj for i > j. \
            Existence iff A is SPD. Failure of algorithm indicates non-positive-definiteness.",
        complexity: "Time: O(n^3/3). Space: O(n^2) (can overwrite lower triangle of A).",
        references: &[
            "Golub & Van Loan, 'Matrix Computations', 4th ed., Ch. 4.2",
            "Press et al., 'Numerical Recipes', Ch. 2.9",
        ],
    },
    // 5. Conjugate Gradient
    MathReference {
        algorithm: "Conjugate Gradient Method (CG)",
        description: "Iterative solver for Ax = b where A is SPD. Generates conjugate directions that are \
            A-orthogonal (d_i^T A d_j = 0 for i != j). Minimizes ||x - x*||_A in a Krylov subspace. \
            Converges in at most n iterations (exact arithmetic); often much faster with preconditioning.",
        formula: "r_0 = b - A x_0, d_0 = r_0. \
            alpha_k = r_k^T r_k / (d_k^T A d_k). \
            x_{k+1} = x_k + alpha_k d_k. \
            r_{k+1} = r_k - alpha_k A d_k. \
            beta_k = r_{k+1}^T r_{k+1} / (r_k^T r_k). \
            d_{k+1} = r_{k+1} + beta_k d_k.",
        complexity: "Time: O(n * nnz(A)) per iteration (one SpMV). Total: O(sqrt(kappa) * n * nnz(A)) \
            where kappa = cond(A). Space: O(n).",
        references: &[
            "Hestenes & Stiefel, 1952, 'Methods of conjugate gradients for solving linear systems'",
            "Shewchuk, 'An Introduction to the Conjugate Gradient Method Without the Agonizing Pain', 1994",
        ],
    },
    // 6. GMRES
    MathReference {
        algorithm: "GMRES (Generalized Minimal Residual)",
        description: "Iterative solver for general (non-symmetric) linear systems Ax = b. \
            Minimizes ||b - A x_k||_2 over the Krylov subspace span{r_0, A r_0, ..., A^{k-1} r_0}. \
            Uses Arnoldi iteration to build an orthonormal basis, then solves a small least-squares problem.",
        formula: "Build orthonormal basis V_k via Arnoldi: A V_k = V_{k+1} H_k (upper Hessenberg). \
            Minimize ||beta e_1 - H_k y||_2, then x_k = V_k y. \
            Restarted GMRES(m): restart after m iterations to limit memory.",
        complexity: "Time: O(k * nnz(A) + k^2 n) for k iterations. Space: O(kn) for storing Krylov basis. \
            GMRES(m): O(m * nnz(A)) per cycle, O(mn) storage.",
        references: &[
            "Saad & Schultz, 'GMRES: A Generalized Minimal Residual Algorithm', 1986",
            "Saad, 'Iterative Methods for Sparse Linear Systems', 2nd ed., Ch. 6",
        ],
    },
    // 7. FFT (Cooley-Tukey)
    MathReference {
        algorithm: "Fast Fourier Transform (Cooley-Tukey)",
        description: "Compute the DFT of length N in O(N log N) operations by recursively decomposing into \
            smaller DFTs. The radix-2 variant requires N = 2^m; mixed-radix handles arbitrary N. \
            Foundation for spectral analysis, convolution, correlation, and signal filtering.",
        formula: "X[k] = sum_{n=0}^{N-1} x[n] W_N^{nk} where W_N = exp(-j 2 pi / N). \
            Radix-2: X[k] = E[k] + W_N^k O[k] where E, O are DFTs of even/odd indexed samples. \
            Inverse: x[n] = (1/N) sum_{k=0}^{N-1} X[k] W_N^{-nk}.",
        complexity: "Time: O(N log N) (vs O(N^2) for naive DFT). Space: O(N) in-place.",
        references: &[
            "Cooley & Tukey, 'An Algorithm for the Machine Calculation of Complex Fourier Series', 1965",
            "Van Loan, 'Computational Frameworks for the Fast Fourier Transform', SIAM, 1992",
        ],
    },
    // 8. Newton's Method (root-finding)
    MathReference {
        algorithm: "Newton's Method (Newton-Raphson)",
        description: "Iterative root-finding for f(x) = 0. Uses first-order Taylor expansion to linearize \
            the function at each step. Quadratic convergence near simple roots. \
            For systems: solves J(x_k) delta = -F(x_k) at each step.",
        formula: "Scalar: x_{k+1} = x_k - f(x_k) / f'(x_k). \
            Vector: x_{k+1} = x_k - J(x_k)^{-1} F(x_k) where J is the Jacobian. \
            Convergence: |e_{k+1}| ~ C |e_k|^2 (quadratic) for simple roots.",
        complexity: "Time per iteration: O(1) scalar, O(n^3) vector (Jacobian solve). \
            Iterations to convergence: O(log log(1/eps)) near root.",
        references: &[
            "Burden & Faires, 'Numerical Analysis', Ch. 2",
            "Kelley, 'Iterative Methods for Linear and Nonlinear Equations', SIAM, 1995",
        ],
    },
    // 9. Gauss-Kronrod Quadrature
    MathReference {
        algorithm: "Gauss-Kronrod Adaptive Quadrature",
        description: "Adaptive numerical integration that embeds a Gauss n-point rule within a Kronrod \
            (2n+1)-point rule. The difference provides an error estimate for automatic step refinement. \
            The standard adaptive quadrature method in most numerical libraries.",
        formula: "G_n = sum_{i=1}^n w_i f(x_i) (Gauss rule, exact for polynomials up to deg 2n-1). \
            K_{2n+1} = sum_{i=1}^{2n+1} w'_i f(x'_i) (Kronrod extension). \
            Error estimate: |G_n - K_{2n+1}|. Bisect intervals with large errors.",
        complexity: "Time: O(N f_evals) where N depends on integrand smoothness. \
            G7-K15: 15 function evaluations per panel.",
        references: &[
            "Piessens et al., 'QUADPACK', Springer, 1983",
            "Kronrod, 'Nodes and Weights of Quadrature Formulas', 1965 (English transl.)",
        ],
    },
    // 10. Runge-Kutta Methods
    MathReference {
        algorithm: "Runge-Kutta Methods (RK4, RK45 Dormand-Prince)",
        description: "Family of explicit ODE solvers for dy/dt = f(t, y). RK4 is the classical 4th-order method. \
            RK45 (Dormand-Prince) uses embedded 4th and 5th order formulas for adaptive step-size control. \
            The default ODE solver in most scientific computing libraries.",
        formula: "RK4: k1 = h f(t, y), k2 = h f(t+h/2, y+k1/2), k3 = h f(t+h/2, y+k2/2), k4 = h f(t+h, y+k3). \
            y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6. \
            Dormand-Prince: 7-stage, embedded pair gives error = |y5 - y4| for step control.",
        complexity: "Time: O(s * n) per step where s is number of stages (4 for RK4, 7 for DP). \
            Total steps depend on required accuracy and problem stiffness.",
        references: &[
            "Dormand & Prince, 'A family of embedded Runge-Kutta formulae', J. Comp. Appl. Math., 1980",
            "Hairer, Norsett & Wanner, 'Solving Ordinary Differential Equations I', Springer",
        ],
    },
    // 11. L-BFGS
    MathReference {
        algorithm: "L-BFGS (Limited-memory BFGS)",
        description: "Quasi-Newton optimization method that approximates the inverse Hessian using \
            the m most recent gradient differences. Ideal for large-scale unconstrained optimization \
            where storing the full Hessian (n x n) is impractical.",
        formula: "Store pairs (s_k, y_k) where s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k. \
            Two-loop recursion computes H_k g_k using only the m stored pairs. \
            Search direction: d_k = -H_k g_k. Line search: strong Wolfe conditions.",
        complexity: "Time: O(mn) per iteration (vs O(n^2) for full BFGS). \
            Space: O(mn) (vs O(n^2) for full BFGS). Typical m = 5-20.",
        references: &[
            "Liu & Nocedal, 'On the limited memory BFGS method for large scale optimization', Math. Programming, 1989",
            "Nocedal & Wright, 'Numerical Optimization', 2nd ed., Ch. 7",
        ],
    },
    // 12. Butterworth Filter Design
    MathReference {
        algorithm: "Butterworth Filter Design",
        description: "Design maximally-flat magnitude IIR filters. The Butterworth filter has no ripple \
            in either passband or stopband (monotonic response). Poles are equally spaced on a circle \
            in the s-plane. Bilinear transform maps analog prototype to digital filter.",
        formula: "|H(j omega)|^2 = 1 / (1 + (omega/omega_c)^{2N}) where N is filter order. \
            Analog poles: s_k = omega_c exp(j pi (2k + N - 1) / (2N)) for k = 1, ..., N. \
            Bilinear transform: s = 2/T * (z-1)/(z+1) with frequency pre-warping.",
        complexity: "Time: O(N) for coefficient computation. O(N) per sample for filtering.",
        references: &[
            "Oppenheim & Schafer, 'Discrete-Time Signal Processing', 3rd ed., Ch. 7",
            "Parks & Burrus, 'Digital Filter Design', Wiley, 1987",
        ],
    },
    // 13. Welch's PSD Estimation
    MathReference {
        algorithm: "Welch's Method for Power Spectral Density",
        description: "Estimate the power spectral density by averaging modified periodograms of \
            overlapping windowed segments. Reduces variance of the periodogram estimate at the cost \
            of reduced frequency resolution. Standard non-parametric spectral estimation method.",
        formula: "PSD(f) = (1/K) sum_{k=1}^K (1/(f_s L U)) |sum_{n=0}^{L-1} w[n] x_k[n] e^{-j 2 pi f n / f_s}|^2 \
            where U = (1/L) sum w[n]^2, K segments, L samples per segment. \
            Variance reduction: var ~ var(periodogram) / K.",
        complexity: "Time: O(K * L log L) for K segments of length L. Space: O(L).",
        references: &[
            "Welch, 'The use of FFT for the estimation of power spectra', IEEE Trans., 1967",
            "Stoica & Moses, 'Spectral Analysis of Signals', Prentice Hall, 2005",
        ],
    },
    // 14. Lanczos Approximation (Gamma function)
    MathReference {
        algorithm: "Lanczos Approximation for the Gamma Function",
        description: "Compute Gamma(z) for complex z using a series approximation based on the \
            Stirling series with improved convergence. Achieves 15+ digit accuracy with ~7 terms. \
            The standard method for gamma function evaluation in most numerical libraries.",
        formula: "Gamma(z+1) ~ sqrt(2 pi) (z + g + 0.5)^{z+0.5} e^{-(z+g+0.5)} A_g(z) \
            where A_g(z) = c_0 + sum_{k=1}^N c_k / (z + k). \
            g ~ 7 (Lanczos parameter), coefficients c_k are precomputed. \
            Reflection: Gamma(z) = pi / (sin(pi z) Gamma(1-z)) for z < 0.5.",
        complexity: "Time: O(N) ~ O(7) per evaluation. Space: O(1) (precomputed coefficients).",
        references: &[
            "Lanczos, 'A Precision Approximation of the Gamma Function', SIAM J. Numer. Anal., 1964",
            "Pugh, 'An Analysis of the Lanczos Gamma Approximation', PhD thesis, 2004",
        ],
    },
    // 15. Gauss-Legendre Quadrature
    MathReference {
        algorithm: "Gauss-Legendre Quadrature",
        description: "Numerical integration using optimally chosen nodes and weights based on Legendre \
            polynomials. An n-point rule integrates polynomials of degree up to 2n-1 exactly. \
            The gold standard for smooth integrands on finite intervals.",
        formula: "integral_{-1}^{1} f(x) dx ~ sum_{i=1}^n w_i f(x_i). \
            Nodes x_i are roots of P_n(x) (Legendre polynomial of degree n). \
            Weights: w_i = 2 / ((1 - x_i^2) [P'_n(x_i)]^2). \
            Transform to [a,b]: integral_a^b f(x) dx = (b-a)/2 * sum w_i f((b-a)x_i/2 + (a+b)/2).",
        complexity: "Time: O(n^2) to compute nodes/weights (Golub-Welsch), O(n) per integral evaluation.",
        references: &[
            "Golub & Welsch, 'Calculation of Gauss Quadrature Rules', Math. Comp., 1969",
            "Abramowitz & Stegun, 'Handbook of Mathematical Functions', Ch. 25",
        ],
    },
    // 16. Cubic Spline Interpolation
    MathReference {
        algorithm: "Cubic Spline Interpolation",
        description: "Construct a piecewise cubic polynomial S(x) that passes through all data points \
            with continuous first and second derivatives (C2 smoothness). Natural splines set \
            S''(x_0) = S''(x_n) = 0. The most widely used interpolation method.",
        formula: "On each interval [x_i, x_{i+1}]: \
            S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3. \
            Continuity of S, S', S'' at interior knots gives a tridiagonal system for the c_i \
            (second derivatives). Natural BC: c_0 = c_n = 0.",
        complexity: "Time: O(n) for setup (tridiagonal solve). O(log n) per evaluation (binary search).",
        references: &[
            "de Boor, 'A Practical Guide to Splines', Springer, 2001",
            "Burden & Faires, 'Numerical Analysis', Ch. 3.5",
        ],
    },
    // 17. Student's t-test
    MathReference {
        algorithm: "Student's t-Test",
        description: "Hypothesis test for comparing means. One-sample: test if population mean equals mu_0. \
            Two-sample: test if two population means differ. Paired: test if mean of differences is zero. \
            Assumes approximately normal distributions; robust for large samples.",
        formula: "One-sample: t = (x_bar - mu_0) / (s / sqrt(n)), df = n - 1. \
            Two-sample (equal var): t = (x1_bar - x2_bar) / (s_p sqrt(1/n1 + 1/n2)), df = n1 + n2 - 2. \
            Welch: df = (s1^2/n1 + s2^2/n2)^2 / (s1^4/(n1^2(n1-1)) + s2^4/(n2^2(n2-1))). \
            p-value: 2 * P(T > |t|) where T ~ t(df).",
        complexity: "Time: O(n) for test statistic, O(1) for p-value (CDF evaluation).",
        references: &[
            "Student (W.S. Gosset), 'The probable error of a mean', Biometrika, 1908",
            "Welch, 'The generalization of Student's problem', Biometrika, 1947",
        ],
    },
    // 18. Pearson Correlation
    MathReference {
        algorithm: "Pearson Product-Moment Correlation",
        description: "Measure the linear relationship between two continuous variables. \
            Values range from -1 (perfect negative) through 0 (no linear relationship) to +1 (perfect positive). \
            Assumes bivariate normality for inference; robust for large samples.",
        formula: "r = sum((x_i - x_bar)(y_i - y_bar)) / sqrt(sum(x_i - x_bar)^2 * sum(y_i - y_bar)^2). \
            Equivalently: r = cov(x, y) / (std(x) * std(y)). \
            Significance test: t = r sqrt(n-2) / sqrt(1-r^2), df = n - 2.",
        complexity: "Time: O(n) single-pass with SIMD acceleration. Space: O(1) streaming.",
        references: &[
            "Pearson, 'Notes on regression and inheritance', Proc. R. Soc. London, 1895",
            "Fisher, 'Frequency distribution of the values of the correlation coefficient', Biometrika, 1915",
        ],
    },
    // 19. Adaptive Simpson's Rule
    MathReference {
        algorithm: "Adaptive Simpson's Quadrature",
        description: "Adaptive version of Simpson's 1/3 rule that recursively bisects intervals \
            where the error estimate exceeds a tolerance. Achieves prescribed accuracy with fewer \
            function evaluations than fixed-step methods for integrands with varying smoothness.",
        formula: "Simpson's rule: S(a,b) = (h/3)[f(a) + 4f(m) + f(b)] where m = (a+b)/2. \
            Error estimate: |S(a,b) - S(a,m) - S(m,b)| / 15 (Richardson extrapolation). \
            If error > tol * (b-a)/(b0-a0), bisect and recurse.",
        complexity: "Time: O(N * f_evals) where N adapts to integrand difficulty. \
            Best case: O(1) for smooth functions; worst case: O(2^d) for d levels of refinement.",
        references: &[
            "Kuncir, 'Algorithm 103: Simpson's rule integrator', CACM, 1962",
            "Lyness, 'Notes on the adaptive Simpson quadrature routine', JACM, 1969",
        ],
    },
    // 20. Normal Distribution
    MathReference {
        algorithm: "Normal (Gaussian) Distribution",
        description: "The most important probability distribution in statistics, arising from the \
            Central Limit Theorem. Characterized by mean (mu) and standard deviation (sigma). \
            The standard normal (mu=0, sigma=1) CDF is denoted Phi(x).",
        formula: "PDF: f(x) = (1 / (sigma sqrt(2 pi))) exp(-(x - mu)^2 / (2 sigma^2)). \
            CDF: Phi(x) = 0.5 (1 + erf((x - mu) / (sigma sqrt(2)))). \
            PPF (quantile): uses Acklam's rational approximation (9-15 digit precision). \
            MGF: M(t) = exp(mu t + sigma^2 t^2 / 2). \
            Entropy: H = 0.5 ln(2 pi e sigma^2).",
        complexity: "PDF: O(1). CDF: O(1) via rational approximation of erf. Random sampling: O(1) Box-Muller.",
        references: &[
            "Abramowitz & Stegun, 'Handbook of Mathematical Functions', Ch. 26",
            "Acklam, 'An algorithm for computing the inverse normal CDF', 2010",
        ],
    },
];
