//! Tutorial: Sparse Matrices with SciRS2
//!
//! This tutorial covers creating sparse matrices in various formats,
//! sparse matrix operations, iterative solvers, and preconditioners.
//!
//! Run with: cargo run -p scirs2-sparse --example tutorial_sparse

use scirs2_sparse::{
    coo::CooMatrix,
    csr::CsrMatrix,
    csr_array::CsrArray,
    error::SparseResult,
    linalg::{cg, AsLinearOperator, CGOptions},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Sparse Matrices Tutorial ===\n");

    section_creating_sparse()?;
    section_operations()?;
    section_solvers()?;
    section_graph_algorithms()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Creating sparse matrices in different formats
fn section_creating_sparse() -> SparseResult<()> {
    println!("--- 1. Creating Sparse Matrices ---\n");

    // CSR (Compressed Sparse Row): most common format for arithmetic
    // Stores data by rows. Fast row slicing and matrix-vector products.
    //
    // Matrix:  [1 0 2]
    //          [0 3 0]
    //          [4 0 5]
    let rows = vec![0, 0, 1, 2, 2];
    let cols = vec![0, 2, 1, 0, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let csr = CsrMatrix::new(data.clone(), rows.clone(), cols.clone(), (3, 3))?;
    println!("CSR Matrix (3x3, 5 non-zeros):");
    println!("  Shape: {:?}", csr.shape());
    println!("  NNZ:   {}", csr.nnz());
    // Access elements
    println!("  A[0,0] = {:.1}", csr.get(0, 0));
    println!("  A[0,1] = {:.1} (zero)", csr.get(0, 1));
    println!("  A[2,2] = {:.1}\n", csr.get(2, 2));

    // COO (Coordinate format): easiest to construct
    // Stores (row, col, value) triples. Good for incremental construction.
    let coo = CooMatrix::new(data.clone(), rows.clone(), cols.clone(), (3, 3))?;
    println!("COO Matrix (same data):");
    println!("  Shape: {:?}", coo.shape());
    println!("  NNZ:   {}", coo.nnz());
    println!();

    // Converting between formats
    let csr_from_coo = CsrMatrix::from_triplets(3, 3, rows.clone(), cols.clone(), data.clone())?;
    println!("CSR from triplets:");
    println!("  A[0,0] = {:.1}", csr_from_coo.get(0, 0));
    println!("  A[1,1] = {:.1}", csr_from_coo.get(1, 1));
    println!();

    // Sparse identity matrix
    let n = 5;
    let eye_rows: Vec<usize> = (0..n).collect();
    let eye_cols: Vec<usize> = (0..n).collect();
    let eye_data: Vec<f64> = vec![1.0; n];
    let eye = CsrMatrix::new(eye_data, eye_rows, eye_cols, (n, n))?;
    println!("Sparse identity (5x5):");
    println!("  NNZ: {} (only diagonal elements stored)", eye.nnz());
    println!(
        "  Density: {:.0}%\n",
        100.0 * eye.nnz() as f64 / (n * n) as f64
    );

    Ok(())
}

/// Section 2: Sparse matrix operations
fn section_operations() -> SparseResult<()> {
    println!("--- 2. Sparse Matrix Operations ---\n");

    // Create a sparse matrix for operations
    let a = CsrMatrix::new(
        vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        vec![0, 0, 1, 1, 1, 2, 2],
        vec![0, 1, 0, 1, 2, 1, 2],
        (3, 3),
    )?;

    // Sparse matrix-vector multiply (SpMV)
    // This is the most common sparse operation
    let x = vec![1.0, 2.0, 3.0];
    let op = a.as_linear_operator();
    let y = op.matvec(&x)?;
    println!("SpMV: A * x where x = [1, 2, 3]");
    println!("  Result: {:?}", y);
    // A = [2 -1 0; -1 2 -1; 0 -1 2], x = [1,2,3]
    // y = [2*1-1*2, -1*1+2*2-1*3, -1*2+2*3] = [0, 0, 4]
    println!();

    // Sparse matrix transpose
    let at = a.transpose();
    println!("Transpose (symmetric matrix, so A = A^T):");
    println!(
        "  A[0,1] = {:.1}, A^T[0,1] = {:.1}",
        a.get(0, 1),
        at.get(0, 1)
    );
    println!(
        "  A[1,0] = {:.1}, A^T[1,0] = {:.1}\n",
        a.get(1, 0),
        at.get(1, 0)
    );

    Ok(())
}

/// Section 3: Iterative solvers for sparse systems
fn section_solvers() -> SparseResult<()> {
    println!("--- 3. Iterative Solvers ---\n");

    // Create a symmetric positive-definite sparse matrix (tridiagonal)
    // This is the 1D Laplacian discretization matrix
    let n = 100;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..n {
        // Diagonal: 2
        rows.push(i);
        cols.push(i);
        data.push(2.0);
        // Off-diagonal: -1
        if i > 0 {
            rows.push(i);
            cols.push(i - 1);
            data.push(-1.0);
        }
        if i < n - 1 {
            rows.push(i);
            cols.push(i + 1);
            data.push(-1.0);
        }
    }

    let a = CsrMatrix::new(data, rows, cols, (n, n))?;
    println!("1D Laplacian matrix ({}x{}, tridiagonal)", n, n);
    println!("  NNZ: {}", a.nnz());
    println!(
        "  Density: {:.2}%\n",
        100.0 * a.nnz() as f64 / (n * n) as f64
    );

    // Right-hand side: b = [1, 1, ..., 1]
    let b: Vec<f64> = vec![1.0; n];

    // Solve using Conjugate Gradient (CG)
    // CG is optimal for symmetric positive-definite systems
    let op = a.as_linear_operator();
    let cg_result = cg(
        op.as_ref(),
        &b,
        CGOptions {
            max_iter: 200,
            rtol: 1e-10,
            atol: 1e-12,
            x0: None,
            preconditioner: None,
        },
    )?;

    println!("Conjugate Gradient solver:");
    println!("  Converged:  {}", cg_result.converged);
    println!("  Iterations: {}", cg_result.iterations);
    println!("  Residual:   {:.2e}", cg_result.residual_norm);
    println!(
        "  Solution (first 5): {:?}",
        &cg_result.x[..5]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "  Solution (last 5):  {:?}\n",
        &cg_result.x[n - 5..]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // GMRES: works for non-symmetric systems too
    let gmres_result = scirs2_sparse::linalg::gmres(
        op.as_ref(),
        &b,
        scirs2_sparse::linalg::GMRESOptions {
            max_iter: 200,
            rtol: 1e-10,
            atol: 1e-12,
            restart: 30, // Restart every 30 iterations
            x0: None,
            preconditioner: None,
        },
    )?;

    println!("GMRES solver:");
    println!("  Converged:  {}", gmres_result.converged);
    println!("  Iterations: {}", gmres_result.iterations);
    println!("  Residual:   {:.2e}\n", gmres_result.residual_norm);

    Ok(())
}

/// Section 4: Sparse graph algorithms
fn section_graph_algorithms() -> SparseResult<()> {
    println!("--- 4. Graph Algorithms on Sparse Matrices ---\n");

    // Adjacency matrix of a small graph (symmetric = undirected):
    //   0 -- 1 -- 2
    //   |         |
    //   3 -- 4 -- 5
    // Symmetric adjacency: each undirected edge is stored as two directed entries
    // Edges: 0-1, 0-3, 1-2, 1-4, 2-5, 3-4, 4-5
    let rows = vec![0, 1, 0, 3, 1, 2, 1, 4, 2, 5, 3, 4, 4, 5];
    let cols = vec![1, 0, 3, 0, 2, 1, 4, 1, 5, 2, 4, 3, 5, 4];
    let data = vec![1.0_f64; 14];

    // Use CsrArray for graph algorithms (implements SparseArray trait)
    let adj = CsrArray::from_triplets(&rows, &cols, &data, (6, 6), false)?;

    println!("Graph (6 nodes, grid-like):");
    println!("  0 -- 1 -- 2");
    println!("  |         |");
    println!("  3 -- 4 -- 5\n");

    // Shortest path using Dijkstra (single source from node 0)
    let (dist_matrix, _preds) = scirs2_sparse::csgraph::shortest_path(
        &adj,
        Some(0),    // source vertex
        None,       // all destinations
        "dijkstra", // Dijkstra method
        false,      // undirected
        false,      // do not return predecessors
    )?;

    println!("Shortest distances from node 0 (Dijkstra):");
    for j in 0..6 {
        println!("  0 -> {}: distance = {:.0}", j, dist_matrix[[0, j]]);
    }

    // Connected components
    let (n_components, labels) = scirs2_sparse::csgraph::connected_components(
        &adj, false,  // undirected
        "weak", // connection type
        true,   // return labels
    )?;
    println!("\n  Connected components: {}", n_components);
    if let Some(ref lab) = labels {
        println!("  Labels: {:?}", lab.to_vec());
    }
    println!();

    Ok(())
}
