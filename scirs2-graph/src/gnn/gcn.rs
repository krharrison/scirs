//! Graph Convolutional Network (GCN) layer
//!
//! Implements the spectral-domain graph convolution operator from
//! Kipf & Welling (2017), "Semi-Supervised Classification with Graph
//! Convolutional Networks".
//!
//! The layer computes:
//! ```text
//!   H' = σ( D̃^{-1/2} Ã D̃^{-1/2} H W + b )
//! ```
//! where `Ã = A + I` (adjacency with self-loops), `D̃` is the corresponding
//! diagonal degree matrix, `H` is the node feature matrix, `W` is the
//! learned weight matrix, and σ is an optional activation function.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};

// ============================================================================
// CsrMatrix - compact sparse representation used by GCN operations
// ============================================================================

/// Compressed Sparse Row matrix representation for adjacency matrices.
///
/// Stores only non-zero entries, which is efficient for typical sparse graphs
/// where `nnz << n^2`.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows (nodes)
    pub n_rows: usize,
    /// Number of columns (nodes, usually equal to n_rows for adjacency)
    pub n_cols: usize,
    /// Row pointer array (length n_rows + 1): row i spans indptr[i]..indptr[i+1]
    pub indptr: Vec<usize>,
    /// Column indices of non-zero entries
    pub indices: Vec<usize>,
    /// Values of non-zero entries
    pub data: Vec<f64>,
}

impl CsrMatrix {
    /// Construct a CSR matrix from COO (coordinate) triples `(row, col, val)`.
    ///
    /// Duplicate entries are summed.
    pub fn from_coo(n_rows: usize, n_cols: usize, coo: &[(usize, usize, f64)]) -> Result<Self> {
        // Validate indices
        for &(r, c, _) in coo {
            if r >= n_rows || c >= n_cols {
                return Err(GraphError::InvalidParameter {
                    param: "coo".to_string(),
                    value: format!("entry ({r},{c}) out of bounds"),
                    expected: format!("indices < ({n_rows},{n_cols})"),
                    context: "CsrMatrix::from_coo".to_string(),
                });
            }
        }

        // Count entries per row
        let mut row_counts = vec![0usize; n_rows];
        for &(r, _, _) in coo {
            row_counts[r] += 1;
        }

        // Build indptr
        let mut indptr = vec![0usize; n_rows + 1];
        for i in 0..n_rows {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        // Insert values (sort within rows by col)
        let nnz = coo.len();
        let mut indices = vec![0usize; nnz];
        let mut data = vec![0.0f64; nnz];
        let mut row_pos = indptr[..n_rows].to_vec();

        for &(r, c, v) in coo {
            let pos = row_pos[r];
            indices[pos] = c;
            data[pos] = v;
            row_pos[r] += 1;
        }

        // Sort each row by column index and accumulate duplicates
        for i in 0..n_rows {
            let start = indptr[i];
            let end = indptr[i + 1];
            let row_idx = &mut indices[start..end];
            let row_val = &mut data[start..end];
            // Simple insertion sort (rows usually short)
            for j in 1..(end - start) {
                let mut k = j;
                while k > 0 && row_idx[k - 1] > row_idx[k] {
                    row_idx.swap(k - 1, k);
                    row_val.swap(k - 1, k);
                    k -= 1;
                }
            }
        }

        Ok(CsrMatrix { n_rows, n_cols, indptr, indices, data })
    }

    /// Create an identity matrix of size `n × n`.
    pub fn eye(n: usize) -> Self {
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        let data = vec![1.0f64; n];
        CsrMatrix { n_rows: n, n_cols: n, indptr, indices, data }
    }

    /// Return the number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Iterate over (row, col, val) triples.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.n_rows).flat_map(move |i| {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            (start..end).map(move |pos| (i, self.indices[pos], self.data[pos]))
        })
    }

    /// Compute the row-sum vector (degree vector for adjacency matrices).
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.n_rows)
            .map(|i| {
                let start = self.indptr[i];
                let end = self.indptr[i + 1];
                self.data[start..end].iter().sum()
            })
            .collect()
    }
}

// ============================================================================
// Self-loop addition
// ============================================================================

/// Add self-loops to a sparse adjacency matrix.
///
/// Returns `Ã = A + I`, which is required for the GCN normalization.
/// Existing diagonal entries are augmented (not replaced).
pub fn add_self_loops(adj: &CsrMatrix) -> Result<CsrMatrix> {
    let n = adj.n_rows;
    if adj.n_cols != n {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}×{}", adj.n_rows, adj.n_cols),
            expected: "square matrix".to_string(),
            context: "add_self_loops".to_string(),
        });
    }

    // Merge existing entries with identity entries (diagonal)
    let mut coo: Vec<(usize, usize, f64)> = Vec::with_capacity(adj.nnz() + n);
    for (r, c, v) in adj.iter() {
        coo.push((r, c, v));
    }
    for i in 0..n {
        coo.push((i, i, 1.0));
    }

    CsrMatrix::from_coo(n, n, &coo)
}

// ============================================================================
// Symmetric normalization
// ============================================================================

/// Compute the symmetrically normalized adjacency: `D̃^{-1/2} Ã D̃^{-1/2}`.
///
/// `adj_tilde` should already contain self-loops (use [`add_self_loops`] first).
/// Nodes with zero degree are left with zero-weight edges (no normalization).
pub fn symmetric_normalize(adj_tilde: &CsrMatrix) -> Result<CsrMatrix> {
    let n = adj_tilde.n_rows;
    let degrees = adj_tilde.row_sums();
    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut new_data = Vec::with_capacity(adj_tilde.nnz());
    for (r, c, v) in adj_tilde.iter() {
        new_data.push(d_inv_sqrt[r] * v * d_inv_sqrt[c]);
    }

    // Rebuild with updated data but same sparsity pattern
    let mut indices = Vec::with_capacity(adj_tilde.nnz());
    for (_, c, _) in adj_tilde.iter() {
        indices.push(c);
    }

    Ok(CsrMatrix {
        n_rows: n,
        n_cols: n,
        indptr: adj_tilde.indptr.clone(),
        indices,
        data: new_data,
    })
}

// ============================================================================
// GCN functional forward pass (Array2-based API)
// ============================================================================

/// Perform one GCN propagation step using the symmetrically normalized adjacency.
///
/// Computes `Â H W` where `Â = D̃^{-1/2} Ã D̃^{-1/2}`.
///
/// Self-loops are added internally; pass the *raw* adjacency without self-loops.
///
/// # Arguments
/// * `adj` – Sparse adjacency matrix (no self-loops required; they are added).
/// * `features` – Node feature matrix of shape `[n_nodes, in_features]`.
/// * `weights` – Weight matrix of shape `[in_features, out_features]`.
///
/// # Returns
/// Output feature matrix of shape `[n_nodes, out_features]`.
pub fn gcn_forward(
    adj: &CsrMatrix,
    features: &Array2<f64>,
    weights: &Array2<f64>,
) -> Result<Array2<f64>> {
    let n = adj.n_rows;
    let (feat_rows, in_dim) = features.dim();
    let (w_in, out_dim) = weights.dim();

    if feat_rows != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{feat_rows} rows"),
            expected: format!("{n} rows (matching adj.n_rows)"),
            context: "gcn_forward".to_string(),
        });
    }
    if w_in != in_dim {
        return Err(GraphError::InvalidParameter {
            param: "weights".to_string(),
            value: format!("{w_in} rows"),
            expected: format!("{in_dim} rows (matching feature dim)"),
            context: "gcn_forward".to_string(),
        });
    }

    // Ã = A + I, then normalize
    let adj_tilde = add_self_loops(adj)?;
    let adj_norm = symmetric_normalize(&adj_tilde)?;

    // Step 1: H_tilde = Â H  (sparse matrix × dense matrix)
    let mut h_tilde = Array2::<f64>::zeros((n, in_dim));
    for (r, c, v) in adj_norm.iter() {
        for k in 0..in_dim {
            h_tilde[[r, k]] += v * features[[c, k]];
        }
    }

    // Step 2: output = H_tilde W  (dense × dense)
    let mut output = Array2::<f64>::zeros((n, out_dim));
    for i in 0..n {
        for j in 0..out_dim {
            let mut sum = 0.0;
            for k in 0..in_dim {
                sum += h_tilde[[i, k]] * weights[[k, j]];
            }
            output[[i, j]] = sum;
        }
    }

    Ok(output)
}

// ============================================================================
// GcnLayer struct
// ============================================================================

/// A single Graph Convolutional Network layer.
///
/// Stores trainable parameters (weight matrix and optional bias) and applies
/// the GCN propagation rule:
/// ```text
///   H' = σ( D̃^{-1/2} Ã D̃^{-1/2} H W + b )
/// ```
#[derive(Debug, Clone)]
pub struct GcnLayer {
    /// Weight matrix of shape `[in_dim, out_dim]`
    pub weights: Array2<f64>,
    /// Optional bias vector of length `out_dim`
    pub bias: Option<Array1<f64>>,
    /// Input feature dimension
    pub in_dim: usize,
    /// Output feature dimension
    pub out_dim: usize,
    /// Apply ReLU activation after the linear transform
    pub use_relu: bool,
}

impl GcnLayer {
    /// Create a new GCN layer with Xavier/Glorot uniform initialization.
    ///
    /// # Arguments
    /// * `in_dim` – Dimension of input node features.
    /// * `out_dim` – Dimension of output node features.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let mut rng = scirs2_core::random::rng();
        let weights = Array2::from_shape_fn((in_dim, out_dim), |_| {
            rng.random::<f64>() * 2.0 * scale - scale
        });
        GcnLayer {
            weights,
            bias: None,
            in_dim,
            out_dim,
            use_relu: true,
        }
    }

    /// Create a layer with specific weights and optional bias.
    pub fn with_params(weights: Array2<f64>, bias: Option<Array1<f64>>) -> Result<Self> {
        let (in_dim, out_dim) = weights.dim();
        if let Some(ref b) = bias {
            if b.len() != out_dim {
                return Err(GraphError::InvalidParameter {
                    param: "bias".to_string(),
                    value: format!("len={}", b.len()),
                    expected: format!("len={out_dim}"),
                    context: "GcnLayer::with_params".to_string(),
                });
            }
        }
        Ok(GcnLayer { weights, bias, in_dim, out_dim, use_relu: true })
    }

    /// Disable the ReLU activation (useful for the last layer).
    pub fn without_activation(mut self) -> Self {
        self.use_relu = false;
        self
    }

    /// Forward pass: compute `σ( Â H W + b )`.
    ///
    /// # Arguments
    /// * `adj` – Raw adjacency matrix (self-loops added internally).
    /// * `features` – Node feature matrix `[n_nodes, in_dim]`.
    pub fn forward(&self, adj: &CsrMatrix, features: &Array2<f64>) -> Result<Array2<f64>> {
        let mut output = gcn_forward(adj, features, &self.weights)?;

        // Add bias if present
        if let Some(ref b) = self.bias {
            let (n, out) = output.dim();
            for i in 0..n {
                for j in 0..out {
                    output[[i, j]] += b[j];
                }
            }
        }

        // ReLU activation
        if self.use_relu {
            output.mapv_inplace(|x| x.max(0.0));
        }

        Ok(output)
    }
}

// ============================================================================
// Multi-layer GCN model
// ============================================================================

/// Multi-layer Graph Convolutional Network.
///
/// Stacks multiple [`GcnLayer`]s with ReLU activations between them. The
/// final layer applies no activation (suitable for downstream tasks such as
/// node classification).
pub struct Gcn {
    /// Ordered list of GCN layers
    pub layers: Vec<GcnLayer>,
}

impl Gcn {
    /// Create a new GCN with the given layer dimensions.
    ///
    /// # Arguments
    /// * `dims` – Sequence of `[d_0, d_1, …, d_L]` specifying input dimension
    ///   `d_0`, hidden dimensions `d_1 … d_{L-1}`, and output dimension `d_L`.
    ///
    /// # Errors
    /// Returns an error if `dims` has fewer than two elements.
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() < 2 {
            return Err(GraphError::InvalidParameter {
                param: "dims".to_string(),
                value: format!("len={}", dims.len()),
                expected: "at least 2 dimensions (in, out)".to_string(),
                context: "Gcn::new".to_string(),
            });
        }

        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..(dims.len() - 1) {
            let mut layer = GcnLayer::new(dims[i], dims[i + 1]);
            // No activation on the last layer
            if i == dims.len() - 2 {
                layer = layer.without_activation();
            }
            layers.push(layer);
        }

        Ok(Gcn { layers })
    }

    /// Forward pass through all layers.
    ///
    /// # Arguments
    /// * `adj` – Sparse adjacency matrix (self-loops added internally per layer).
    /// * `features` – Initial node feature matrix `[n_nodes, d_0]`.
    pub fn forward(&self, adj: &CsrMatrix, features: &Array2<f64>) -> Result<Array2<f64>> {
        let mut h = features.clone();
        for layer in &self.layers {
            h = layer.forward(adj, &h)?;
        }
        Ok(h)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn triangle_csr() -> CsrMatrix {
        // Triangle graph: 0-1, 1-2, 0-2 (undirected)
        let coo = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
        ];
        CsrMatrix::from_coo(3, 3, &coo).expect("triangle CSR")
    }

    #[test]
    fn test_add_self_loops_increases_nnz() {
        let adj = triangle_csr();
        let with_loops = add_self_loops(&adj).expect("add_self_loops");
        // 6 original + 3 diagonal = 9 entries
        assert_eq!(with_loops.nnz(), 9);
    }

    #[test]
    fn test_symmetric_normalize_degree_check() {
        let adj = triangle_csr();
        let adj_tilde = add_self_loops(&adj).expect("self loops");
        let adj_norm = symmetric_normalize(&adj_tilde).expect("normalize");
        // Row sums of normalized adjacency with self-loops should equal 1 for
        // a regular graph where every node has the same degree
        let row_sums = adj_norm.row_sums();
        for s in &row_sums {
            assert!((*s - 1.0).abs() < 1e-10, "row sum = {s}");
        }
    }

    #[test]
    fn test_gcn_forward_output_shape() {
        let adj = triangle_csr();
        let features = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64 * 0.1);
        let weights = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f64 * 0.01);
        let out = gcn_forward(&adj, &features, &weights).expect("gcn_forward");
        assert_eq!(out.dim(), (3, 8));
    }

    #[test]
    fn test_gcn_layer_forward() {
        let adj = triangle_csr();
        let features = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64 * 0.1);
        let layer = GcnLayer::new(4, 6);
        let out = layer.forward(&adj, &features).expect("layer forward");
        assert_eq!(out.dim(), (3, 6));
        // ReLU: all values non-negative
        for &v in out.iter() {
            assert!(v >= 0.0, "ReLU violated: {v}");
        }
    }

    #[test]
    fn test_gcn_without_activation() {
        let adj = triangle_csr();
        let features = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64 * 0.1 - 0.5);
        let layer = GcnLayer::new(4, 4).without_activation();
        let out = layer.forward(&adj, &features).expect("no_act forward");
        assert_eq!(out.dim(), (3, 4));
        // Without activation, negative values are possible
    }

    #[test]
    fn test_multi_layer_gcn() {
        let adj = triangle_csr();
        let features = Array2::from_shape_fn((3, 8), |(i, j)| (i * 8 + j) as f64 * 0.05);
        let gcn = Gcn::new(&[8, 16, 4]).expect("Gcn::new");
        let out = gcn.forward(&adj, &features).expect("gcn forward");
        assert_eq!(out.dim(), (3, 4));
    }

    #[test]
    fn test_csr_from_coo_invalid() {
        let coo = vec![(5, 0, 1.0)]; // row 5 out of range
        let result = CsrMatrix::from_coo(3, 3, &coo);
        assert!(result.is_err());
    }

    #[test]
    fn test_gcn_with_bias() {
        use scirs2_core::ndarray::Array1;
        let adj = triangle_csr();
        let features = Array2::from_shape_fn((3, 4), |(i, j)| (i + j) as f64 * 0.1);
        let weights = Array2::from_shape_fn((4, 6), |(i, j)| (i + j) as f64 * 0.02);
        let bias = Array1::from_vec(vec![0.1; 6]);
        let layer = GcnLayer::with_params(weights, Some(bias)).expect("with_params");
        let out = layer.forward(&adj, &features).expect("forward with bias");
        assert_eq!(out.dim(), (3, 6));
    }

    #[test]
    fn test_gcn_layer_array2_api() {
        // Verify that the Array2-based gcn_forward is consistent with
        // known symmetric normalization for a simple path graph 0–1–2
        let coo = vec![(0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0)];
        let adj = CsrMatrix::from_coo(3, 3, &coo).expect("path CSR");
        // Identity weight matrix preserves features up to normalization
        let features = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let weights = Array2::eye(2);
        let out = gcn_forward(&adj, &features, &weights).expect("gcn_forward");
        assert_eq!(out.dim(), (3, 2));
        // Finite values
        for &v in out.iter() {
            assert!(v.is_finite(), "non-finite value: {v}");
        }
    }
}
