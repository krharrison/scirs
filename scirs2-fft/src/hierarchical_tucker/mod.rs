//! Hierarchical Tucker (HT) tensor decomposition.
//!
//! Provides an efficient representation of high-dimensional tensors that
//! avoids the "curse of dimensionality":  for a d-dimensional tensor of
//! size n along each mode, the full tensor requires n^d values, while the
//! HT format stores only O(d · n · k + d · k³) values, where k is the
//! HT-rank.
//!
//! ## Modules
//!
//! * [`ht_tree`]  – Binary dimension-partition tree structure.
//! * [`ht_ops`]   – Low-level linear algebra (SVD, n-mode product, Kronecker).
//! * [`ht_decomp`] – High-level `HierarchicalTucker` decomposition type.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_fft::hierarchical_tucker::HierarchicalTucker;
//!
//! // Build a small 3D tensor.
//! let shape = vec![4, 4, 4];
//! let n: usize = shape.iter().product();
//! let tensor: Vec<f64> = (0..n).map(|i| i as f64).collect();
//!
//! // Decompose with maximum rank 4.
//! let ht = HierarchicalTucker::decompose(&tensor, &shape, 4).expect("valid input");
//!
//! println!("Compression ratio: {:.2}x", ht.compression_ratio());
//!
//! // Reconstruct and measure error.
//! let rec = ht.reconstruct().expect("valid input");
//! let err: f64 = tensor.iter().zip(rec.iter())
//!     .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
//! println!("Reconstruction error: {err:.4}");
//! ```

pub mod ht_decomp;
pub mod ht_ops;
pub mod ht_tree;

pub use ht_decomp::HierarchicalTucker;
pub use ht_tree::HTNode;
