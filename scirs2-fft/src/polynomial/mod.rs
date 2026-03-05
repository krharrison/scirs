//! # Polynomial Arithmetic and Number Theoretic Transforms
//!
//! This module provides a comprehensive suite of polynomial algorithms:
//!
//! ## Submodules
//!
//! | Submodule | Contents |
//! |-----------|----------|
//! | [`arithmetic`] | Dense polynomial struct with FFT-accelerated multiplication, GCD, composition, Jenkins-Traub root finding |
//! | [`ntt`] | Number Theoretic Transform over Z/p: in-place NTT, polynomial multiplication mod p, prime search |
//! | [`multipoint`] | O(n log²n) multipoint evaluation, interpolation, and partial fraction decomposition |
//! | [`legacy`] | Classic polynomial API (Array1-based) for backward compatibility |
//!
//! ## Quick Reference
//!
//! ```rust
//! use scirs2_fft::polynomial::{
//!     Polynomial,
//!     ntt::{ntt_mul, MOD998244353},
//!     multipoint::{multipoint_eval, interpolate},
//! };
//!
//! // Fast polynomial multiplication via FFT
//! let p = Polynomial::new(vec![1.0, 1.0]); // 1 + x
//! let q = Polynomial::new(vec![1.0, 1.0]); // 1 + x
//! let r = p.mul_fft(&q).expect("fft mul");  // (1+x)² = 1 + 2x + x²
//! assert!((r.coeffs[1] - 2.0).abs() < 1e-10);
//!
//! // NTT multiplication over Z/998244353
//! let c = ntt_mul(&[1u64, 2], &[3u64, 4], MOD998244353).expect("ntt");
//! assert_eq!(c, vec![3, 10, 8]);
//!
//! // Multipoint evaluation
//! let poly = Polynomial::new(vec![0.0, 0.0, 1.0]); // x²
//! let ys = multipoint_eval(&poly, &[1.0, 2.0, 3.0]).expect("eval");
//! assert!((ys[0] - 1.0).abs() < 1e-10);
//! assert!((ys[1] - 4.0).abs() < 1e-10);
//! assert!((ys[2] - 9.0).abs() < 1e-10);
//! ```
//!
//! ## Feature Overview
//!
//! ### Polynomial Arithmetic (`arithmetic`)
//!
//! - Horner evaluation and complex evaluation
//! - Addition, subtraction, naive O(n²) and FFT O(n log n) multiplication
//! - Polynomial long division with remainder
//! - GCD via Euclidean algorithm
//! - Composition f(g(x)) via Horner's scheme over polynomials
//! - Formal derivative and integral
//! - Root finding via Jenkins-Traub three-stage algorithm (with companion
//!   matrix fallback)
//!
//! ### Number Theoretic Transform (`ntt`)
//!
//! - In-place iterative Cooley-Tukey NTT over arbitrary NTT-friendly primes
//! - Fast path for `MOD998244353` (the most common competitive-programming NTT prime)
//! - Polynomial multiplication over Z/p
//! - Polynomial inversion mod x^n (Newton's method)
//! - Polynomial GCD and derivative/integral over Z/p
//! - Exact integer convolution via Garner's CRT (two-prime method)
//! - `find_ntt_prime`: search for a suitable prime+generator pair
//!
//! ### Multipoint Evaluation and Interpolation (`multipoint`)
//!
//! - `multipoint_eval`: O(n log²n) evaluation via subproduct tree
//! - `interpolate`: O(n log²n) polynomial interpolation (barycentric + subproduct tree)
//! - `partial_fraction_decomp`: residues of a rational function at simple poles
//! - `build_product_poly`: build ∏(x - rᵢ) efficiently
//! - Chebyshev node generators (first and second kind)

pub mod arithmetic;
pub mod legacy;
pub mod multipoint;
pub mod ntt;

// ─────────────────────────────────────────────────────────────────────────────
//  Primary re-exports (new API)
// ─────────────────────────────────────────────────────────────────────────────

/// The primary polynomial type with Vec<f64> backing and Jenkins-Traub roots.
///
/// For the legacy Array1-backed type, use `polynomial::legacy::Polynomial`.
pub use arithmetic::Polynomial;

// NTT constants and functions
pub use ntt::{
    convolve_exact, find_ntt_prime, modinv, mulmod, ntt as ntt_transform, ntt_mul,
    ntt_mul_998244353, ntt_998244353, poly_deriv_mod, poly_gcd_mod, poly_integral_mod,
    poly_inv_mod_xn, powmod, KNOWN_NTT_PRIMES, MOD1000000007, MOD469762049, MOD998244353,
};

// Multipoint evaluation and interpolation
pub use multipoint::{
    build_product_poly, chebyshev_nodes_first, chebyshev_nodes_second, interpolate,
    multipoint_eval, partial_fraction_decomp,
};

// ─────────────────────────────────────────────────────────────────────────────
//  Legacy re-exports (backward compatibility)
// ─────────────────────────────────────────────────────────────────────────────

// Legacy polynomial type (Array1-backed)
pub use legacy::Polynomial as LegacyPolynomial;

// Legacy free functions
pub use legacy::{
    chebyshev_eval, chebyshev_expansion, chebyshev_nodes, chebyshev_t, complex_poly_multiply,
    hermite_prob, legendre_p, poly_add, poly_compose, poly_divmod, poly_gcd, poly_interpolate,
    poly_multiply, poly_multipoint_eval, poly_pow, poly_powmod, poly_sub, ComplexPolynomial,
};
