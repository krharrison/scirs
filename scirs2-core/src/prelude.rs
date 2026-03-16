//! # SciRS2 Core Prelude
//!
//! The prelude module provides convenient access to the most commonly used items
//! in the SciRS2 ecosystem. Import this module to get started quickly without
//! needing to know the exact paths of all core functionality.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::prelude::*;
//!
//! // Now you have access to all common functionality:
//! let data = array![[1.0, 2.0], [3.0, 4.0]];  // Array creation
//! let mean = data.mean().expect("Operation failed");             // Array operations
//! let counter = Counter::new("requests".into()); // Metrics
//! ```
//!
//! ## What's Included
//!
//! ### Array Types and Operations
//! - `Array`, `Array1`, `Array2`, `Array3`, `Array4`, `ArrayD` - N-dimensional array types
//! - `ArrayView`, `ArrayView1`, `ArrayView2`, `ArrayView3`, `ArrayView4`, `ArrayViewD` - Array views
//! - `ArrayViewMut`, `ArrayViewMut1`, `ArrayViewMut2` - Mutable array views
//! - `Axis`, `Ix1`, `Ix2`, `Ix3`, `Ix4`, `IxDyn` - Shape and axis types
//! - `Zip` - Parallel element-wise iteration across arrays
//! - `array!`, `s!` - Convenient macros for array creation and slicing
//!
//! ### Numeric Traits
//! - `Float` - Floating-point operations
//! - `FromPrimitive`, `ToPrimitive` - Convert from/to primitive types
//! - `Num`, `NumCast`, `NumAssign` - Numeric operations and in-place ops
//! - `Zero`, `One` - Additive and multiplicative identities
//!
//! ### Random Number Generation
//! - `random()` - Convenient random value generation
//! - `Rng` - Random number generator trait
//! - `SeedableRng` - Seedable RNG trait for reproducibility
//! - `ChaCha8Rng`, `ChaCha12Rng`, `ChaCha20Rng` - Secure random number generators
//! - Common distributions: `Normal`, `Uniform`, `Exponential`, `Gamma`, `Bernoulli`
//!
//! ### Validation Utilities
//! - `check_positive()` - Validate positive values
//! - `check_shape()` - Validate array shapes
//! - `check_finite()` - Validate finite values
//! - `check_in_bounds()` - Validate value bounds
//!
//! ### Metrics and Observability
//! - `Counter` - Monotonically increasing metric
//! - `Gauge` - Arbitrary up/down metric
//! - `Histogram` - Distribution of values
//! - `Timer` - Duration measurements
//! - `global_metrics_registry()` - Global metrics collection
//!
//! ### Error Handling
//! - `CoreError` - Main error type
//! - `CoreResult<T>` - Result type alias
//!
//! ### Complex Numbers
//! - `Complex`, `Complex32`, `Complex64` - Complex number types
//!
//! ### Builder Patterns (IDE-friendly ergonomic construction)
//! - `MatrixBuilder` - Fluent builder for 2D matrices
//! - `VectorBuilder` - Fluent builder for 1D vectors
//! - `ArrayBuilder` - Generic fluent array builder
//!
//! ### Ergonomic Matrix Operations
//! - `mat_dot(a, b)` - Matrix multiply (2D)
//! - `outer(a, b)` - Outer product of two 1D arrays
//! - `kron_product(a, b)` - Kronecker product
//! - `vstack(arrays)` - Vertical stack of 2D arrays
//! - `hstack(arrays)` - Horizontal stack of 2D arrays
//! - `block_diag_stack(blocks)` - Block diagonal matrix
//!
//! ### Domain Modules
//! - `finance` - Option pricing, fixed income, risk, portfolio analytics
//! - `bioinformatics` - Sequence analysis, alignment, phylogenetics
//! - `physics` - Classical, thermodynamics, electrodynamics, quantum
//! - `ml_pipeline` (feature-gated) - ML pipeline framework
//!
//! ## Examples
//!
//! ### Basic Array Operations
//!
//! ```rust
//! use scirs2_core::prelude::*;
//!
//! // Create arrays
//! let a = array![1.0, 2.0, 3.0, 4.0];
//! let b = array![[1.0, 2.0], [3.0, 4.0]];
//!
//! // Array slicing
//! let slice = b.slice(s![.., 0]);
//!
//! // Array operations
//! let sum = a.sum();
//! let mean = a.mean().expect("Operation failed");
//! ```
//!
//! ### Builder Patterns
//!
//! ```rust
//! use scirs2_core::prelude::*;
//!
//! // Fluent matrix construction
//! let identity = MatrixBuilder::<f64>::eye(3);
//! let zeros = MatrixBuilder::<f64>::zeros(4, 4);
//! let from_data = MatrixBuilder::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2)
//!     .expect("valid shape");
//!
//! // Vector construction
//! let v = VectorBuilder::<f64>::linspace(0.0, 1.0, 11);
//! let range = VectorBuilder::<f64>::arange(0.0, 5.0, 1.0);
//! ```
//!
//! ### Ergonomic Matrix Operations
//!
//! ```rust
//! use scirs2_core::prelude::*;
//!
//! let a = MatrixBuilder::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2)
//!     .expect("valid shape");
//! let b = MatrixBuilder::<f64>::eye(2);
//! let c = mat_dot(&a.view(), &b.view());  // Matrix multiply
//!
//! let u = array![1.0, 2.0, 3.0];
//! let v = array![4.0, 5.0, 6.0];
//! let outer_prod = outer(&u.view(), &v.view());  // Outer product
//! ```
//!
//! ### Random Number Generation
//!
//! ```rust,ignore
//! use scirs2_core::prelude::*;
//!
//! // Quick random values
//! let x: f64 = random();
//! let y: bool = random();
//!
//! // Reproducible random generation
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//! let sample = rng.random::<f64>();
//!
//! // Sample from distributions
//! let normal = Normal::new(0.0, 1.0).expect("Operation failed");
//! let value = normal.sample(&mut rng);
//! ```
//!
//! ### Parameter Validation
//!
//! ```rust,ignore
//! use scirs2_core::prelude::*;
//!
//! pub fn my_function(data: &Array2<f64>, k: usize) -> CoreResult<Array1<f64>> {
//!     // Validate inputs
//!     check_positive(k, "k")?;
//!     check_array_finite(data, "data")?;
//!
//!     // Your implementation here
//!     Ok(Array1::zeros(k))
//! }
//! ```
//!
//! ### Metrics Collection
//!
//! ```rust
//! use scirs2_core::prelude::*;
//!
//! // Create metrics
//! let counter = Counter::new("requests_total".into());
//! counter.inc();
//!
//! let gauge = Gauge::new("active_connections".into());
//! gauge.set(42.0);
//!
//! let histogram = Histogram::new("response_time".into());
//! histogram.observe(0.123);
//!
//! let timer = Timer::new("operation_duration".into());
//! let _guard = timer.start(); // Auto-records on drop
//! ```

// ================================
// Array Types and Operations
// ================================

/// Re-export core array types for 1D, 2D, 3D, 4D, and dynamic dimensions.
pub use crate::{
    Array,  // Generic N-dimensional array
    Array1, // 1-dimensional array
    Array2, // 2-dimensional array
    ArrayD, // Dynamic-dimensional array
    ArrayView,
    ArrayView1,
    ArrayView2,   // Immutable array views
    ArrayViewMut, // Mutable array view
    Axis,         // Array axis type
    Ix1,          // 1-dimensional index
    Ix2,          // 2-dimensional index
    IxDyn,        // Dynamic index
};

// Re-export 3D and 4D array types (frequently used in deep learning / physics)
pub use ::ndarray::{
    Array3,        // 3-dimensional array (images, volumes, conv layers)
    Array4,        // 4-dimensional array (batched images: NCHW)
    ArrayView3,    // Immutable 3D view
    ArrayView4,    // Immutable 4D view
    ArrayViewD,    // Immutable dynamic-dimensional view
    ArrayViewMut1, // Mutable 1D view
    ArrayViewMut2, // Mutable 2D view
    ArrayViewMut3, // Mutable 3D view
    ArrayViewMutD, // Mutable dynamic-dimensional view
    Ix3,           // 3-dimensional index type
    Ix4,           // 4-dimensional index type
};

// Re-export Zip for parallel element-wise iteration (critical for IDE ergonomics)
pub use ::ndarray::Zip;

/// Re-export array creation and manipulation macros
pub use crate::{
    array, // Create arrays: array![[1, 2], [3, 4]]
    s,     // Slice arrays: arr.slice(s![.., 0])
};

// ================================
// Random Number Generation
// ================================

#[cfg(feature = "random")]
pub use crate::random::{
    random,       // Convenient random value generation: let x: f64 = random();
    thread_rng,   // Thread-local RNG
    Distribution, // Distribution trait
    Rng,          // Random number generator trait (base trait in rand 0.10)
    RngExt,       // Extension trait for random(), random_range(), sample(), etc.
    SeedableRng,  // Seedable RNG trait for reproducibility
};

#[cfg(feature = "random")]
pub use crate::random::{
    ChaCha12Rng, // Balanced cryptographic RNG
    ChaCha20Rng, // Secure cryptographic RNG
    ChaCha8Rng,  // Fast cryptographic RNG
};

/// Common distributions for convenience
#[cfg(feature = "random")]
pub use crate::random::{
    Bernoulli,   // Bernoulli distribution (coin flip)
    Exponential, // Exponential distribution
    Gamma,       // Gamma distribution
    Normal,      // Normal/Gaussian distribution
    Uniform,     // Uniform distribution
};

// ================================
// Validation Utilities
// ================================

pub use crate::validation::{
    check_finite,    // Validate finite values (no NaN/Inf)
    check_in_bounds, // Validate value is within bounds
    check_positive,  // Validate positive values
};

// For backwards compatibility, also provide the array validation functions
pub use crate::validation::{
    checkarray_finite as check_array_finite, // Validate all array values are finite
    checkshape as check_shape,               // Validate array shape
};

// ================================
// Metrics and Observability
// ================================

pub use crate::metrics::{
    global_metrics_registry, // Access global metrics registry
    Counter,                 // Monotonically increasing counter
    Gauge,                   // Arbitrary up/down value
    Histogram,               // Distribution of values
    Timer,                   // Duration measurements
};

// ================================
// Error Handling
// ================================

pub use crate::error::{
    CoreError,  // Main error type
    CoreResult, // Result<T, CoreError> alias
};

// ================================
// Complex Numbers
// ================================

pub use num_complex::{
    Complex,   // Generic complex number
    Complex32, // 32-bit complex (f32 real/imag)
    Complex64, // 64-bit complex (f64 real/imag)
};

// ================================
// Common Numeric Traits
// ================================

/// Re-export commonly used numerical traits
pub use num_traits::{
    Float,         // Floating-point operations (sin, cos, sqrt, exp, ln, ...)
    FromPrimitive, // Convert from primitive types (usize -> T, i32 -> T, ...)
    Num,           // Basic numeric operations
    NumAssign,     // In-place numeric operations (+=, -=, *=, /=)
    NumCast,       // Numeric type conversions
    One,           // Multiplicative identity (1)
    ToPrimitive,   // Convert to primitive types (T -> f64, T -> usize, ...)
    Zero,          // Additive identity (0)
};

// ================================
// Configuration
// ================================

pub use crate::config::{
    get_config,        // Get current configuration
    set_global_config, // Set global configuration
    Config,            // Configuration management
};

// ================================
// Constants
// ================================

/// Mathematical constants (π, e, φ, etc.)
pub use crate::constants::math;

/// Physical constants (c, h, G, etc.)
pub use crate::constants::physical;

// ================================
// Builder Patterns for IDE Ergonomics
// ================================

/// Fluent builder for 2D matrices — IDE-friendly ergonomic matrix construction.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::MatrixBuilder;
///
/// let identity = MatrixBuilder::<f64>::eye(3);
/// let zeros = MatrixBuilder::<f64>::zeros(4, 4);
/// let from_data = MatrixBuilder::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2)
///     .expect("valid shape");
/// ```
pub use crate::builders::MatrixBuilder;

/// Fluent builder for 1D vectors — IDE-friendly ergonomic vector construction.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::VectorBuilder;
///
/// let linspace = VectorBuilder::<f64>::linspace(0.0, 1.0, 11);
/// let arange = VectorBuilder::<f64>::arange(0.0, 5.0, 1.0);
/// let ones = VectorBuilder::<f64>::ones(5);
/// ```
pub use crate::builders::VectorBuilder;

/// Generic fluent array builder.
pub use crate::builders::ArrayBuilder;

// ================================
// Ergonomic Matrix Operations
// ================================

/// Matrix multiplication (dot product of two 2D arrays).
///
/// Shorthand for the common pattern of multiplying two matrices.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let a = MatrixBuilder::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2).expect("ok");
/// let b = MatrixBuilder::from_vec(vec![3.0, 4.0, 5.0, 6.0], 2, 2).expect("ok");
/// let c = mat_dot(&a.view(), &b.view());
/// ```
pub use crate::ops::dot as mat_dot;

/// Outer product of two 1D arrays.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let u = array![1.0, 2.0, 3.0];
/// let v = array![4.0, 5.0, 6.0];
/// let m = outer(&u.view(), &v.view());
/// assert_eq!(m.shape(), &[3, 3]);
/// ```
pub use crate::ops::outer;

/// Kronecker product of two 2D arrays.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let a = MatrixBuilder::<f64>::eye(2);
/// let b = MatrixBuilder::<f64>::eye(2);
/// let k = kron_product(&a.view(), &b.view());
/// assert_eq!(k.shape(), &[4, 4]);
/// ```
pub use crate::ops::kron as kron_product;

/// Vertical stack of 2D arrays (rows concatenation).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let a = MatrixBuilder::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("ok");
/// let b = MatrixBuilder::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2).expect("ok");
/// let stacked = vstack(&[a.view(), b.view()]).expect("same cols");
/// assert_eq!(stacked.shape(), &[4, 2]);
/// ```
pub use crate::ops::vstack;

/// Horizontal stack of 2D arrays (column concatenation).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let a = MatrixBuilder::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("ok");
/// let b = MatrixBuilder::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2).expect("ok");
/// let stacked = hstack(&[a.view(), b.view()]).expect("same rows");
/// assert_eq!(stacked.shape(), &[2, 4]);
/// ```
pub use crate::ops::hstack;

/// Block diagonal matrix from a sequence of 2D arrays.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let a = MatrixBuilder::<f64>::eye(2);
/// let b = MatrixBuilder::<f64>::eye(3);
/// let bd = block_diag_stack(&[a.view(), b.view()]);
/// assert_eq!(bd.shape(), &[5, 5]);
/// ```
pub use crate::ops::block_diag as block_diag_stack;

// ================================
// Domain Module Re-exports
// ================================

// Financial computing (option pricing, fixed income, risk, portfolio)
// Always available — no feature gate required
pub use crate::finance;

// Bioinformatics (sequence analysis, alignment, phylogenetics, statistics)
// Always available — no feature gate required
pub use crate::bioinformatics;

// Computational physics (classical, thermodynamics, electrodynamics, quantum)
// Always available — no feature gate required
pub use crate::physics;

// ML pipeline (composable pipelines, transformers, predictors)
// Feature-gated — only available when `ml_pipeline` feature is enabled
#[cfg(feature = "ml_pipeline")]
pub use crate::ml_pipeline;
