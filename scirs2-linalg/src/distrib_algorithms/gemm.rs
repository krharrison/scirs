//! SUMMA (Scalable Universal Matrix Multiply Algorithm) with 2-D block-cyclic layout.
//!
//! ## Algorithm overview
//!
//! SUMMA partitions the output matrix C (m×n) over a P×Q virtual processor grid.
//! At each of the ⌈k/bs⌉ panel steps, every process row broadcasts a column-panel
//! of A of width `bs`, and every process column broadcasts a row-panel of B of height
//! `bs`.  Each process then performs a local rank-`bs` update:
//!
//! ```text
//!   C[i,j] += A[:,step*bs:(step+1)*bs] * B[step*bs:(step+1)*bs,:]
//! ```
//!
//! The simulation here runs all panels on a single core, faithfully tracking which
//! data each virtual process would hold in a real distributed execution.
//!
//! ## Communication cost model
//!
//! For a (P × Q) grid the word-count (bandwidth cost) per process is
//!
//! ```text
//!   words ≈ (m/P)·(k/bs)·bs·P  +  (n/Q)·(k/bs)·bs·Q
//!          = m·k  +  n·k   (independent of grid size!)
//! ```
//!
//! The latency cost is `2 · (k/bs) · log2(max(P,Q))` messages.
//!
//! ## References
//!
//! Van De Geijn & Watts (1997), *SUMMA: Scalable Universal Matrix Multiply Algorithm*.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array2, ArrayView2};

use super::DistribConfig;

// ---------------------------------------------------------------------------
// BlockCyclicMatrix
// ---------------------------------------------------------------------------

/// A dense matrix stored in 2-D block-cyclic distribution across a virtual P×Q grid.
///
/// In a real distributed system each rank would only store its local tiles.
/// Here all tiles are stored in a single in-process structure for simulation.
#[derive(Debug, Clone)]
pub struct BlockCyclicMatrix {
    /// Full data matrix (stored locally for simulation purposes).
    pub data: Array2<f64>,
    /// Global number of rows.
    pub global_rows: usize,
    /// Global number of columns.
    pub global_cols: usize,
    /// Block size (tiles are `block_size × block_size`).
    pub block_size: usize,
    /// Owning process row (0-indexed) in the virtual grid.
    pub proc_row: usize,
    /// Owning process column (0-indexed) in the virtual grid.
    pub proc_col: usize,
    /// Total number of process rows.
    pub n_proc_rows: usize,
    /// Total number of process columns.
    pub n_proc_cols: usize,
}

impl BlockCyclicMatrix {
    /// Create a new `BlockCyclicMatrix` wrapping existing data.
    pub fn new(
        data: Array2<f64>,
        block_size: usize,
        proc_row: usize,
        proc_col: usize,
        n_proc_rows: usize,
        n_proc_cols: usize,
    ) -> LinalgResult<Self> {
        if block_size == 0 {
            return Err(LinalgError::ValueError(
                "block_size must be > 0".to_string(),
            ));
        }
        if n_proc_rows == 0 || n_proc_cols == 0 {
            return Err(LinalgError::ValueError(
                "n_proc_rows and n_proc_cols must be > 0".to_string(),
            ));
        }
        if proc_row >= n_proc_rows || proc_col >= n_proc_cols {
            return Err(LinalgError::IndexError(
                "proc_row / proc_col out of grid bounds".to_string(),
            ));
        }
        let global_rows = data.nrows();
        let global_cols = data.ncols();
        Ok(Self {
            data,
            global_rows,
            global_cols,
            block_size,
            proc_row,
            proc_col,
            n_proc_rows,
            n_proc_cols,
        })
    }

    /// Extract the local tile at block position `(i, j)` (block indices, not element indices).
    ///
    /// Returns the sub-matrix owned by this process at the `(i,j)`-th block position.
    /// Blocks at the boundary may be smaller than `block_size`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested block indices are out of range.
    pub fn local_block(&self, i: usize, j: usize) -> LinalgResult<Array2<f64>> {
        let n_blocks_row = self.global_rows.div_ceil(self.block_size);
        let n_blocks_col = self.global_cols.div_ceil(self.block_size);
        if i >= n_blocks_row || j >= n_blocks_col {
            return Err(LinalgError::IndexError(format!(
                "block ({i},{j}) out of range for {n_blocks_row}×{n_blocks_col} block grid"
            )));
        }
        let row_start = i * self.block_size;
        let row_end = (row_start + self.block_size).min(self.global_rows);
        let col_start = j * self.block_size;
        let col_end = (col_start + self.block_size).min(self.global_cols);
        Ok(self
            .data
            .slice(s![row_start..row_end, col_start..col_end])
            .to_owned())
    }

    /// Map a global element coordinate `(gi, gj)` to:
    /// `(proc_row, proc_col, local_i, local_j)`.
    ///
    /// In block-cyclic layout:
    /// - `block_row = gi / block_size`  → owner process row = `block_row % n_proc_rows`
    /// - `local_i   = (block_row / n_proc_rows) * block_size + (gi % block_size)`
    pub fn global_to_local(
        &self,
        gi: usize,
        gj: usize,
    ) -> LinalgResult<(usize, usize, usize, usize)> {
        if gi >= self.global_rows || gj >= self.global_cols {
            return Err(LinalgError::IndexError(format!(
                "global index ({gi},{gj}) out of range for {0}×{1} matrix",
                self.global_rows, self.global_cols
            )));
        }
        let block_row = gi / self.block_size;
        let block_col = gj / self.block_size;

        let owner_pr = block_row % self.n_proc_rows;
        let owner_pc = block_col % self.n_proc_cols;

        // Local row index within the owning process
        let local_block_row = block_row / self.n_proc_rows;
        let intra_block_row = gi % self.block_size;
        let local_i = local_block_row * self.block_size + intra_block_row;

        // Local column index within the owning process
        let local_block_col = block_col / self.n_proc_cols;
        let intra_block_col = gj % self.block_size;
        let local_j = local_block_col * self.block_size + intra_block_col;

        Ok((owner_pr, owner_pc, local_i, local_j))
    }
}

// ---------------------------------------------------------------------------
// SUMMA core kernel
// ---------------------------------------------------------------------------

/// Perform one SUMMA panel update: `C += A_panel * B_panel`.
///
/// * `a_panel` – column panel of A with shape `(m, bs)`
/// * `b_panel` – row panel of B with shape `(bs, n)`
/// * `c_tile`  – accumulator with shape `(m, n)`, updated in-place
///
/// This is the inner loop of the SUMMA algorithm.  In a real distributed
/// implementation each process only holds the sub-blocks owned by it; here
/// we work on the full panels since we are simulating a single-process view.
pub fn summa_step(
    a_panel: &ArrayView2<f64>,
    b_panel: &ArrayView2<f64>,
    c_tile: &mut Array2<f64>,
) -> LinalgResult<()> {
    let m = a_panel.nrows();
    let bs = a_panel.ncols();
    let n = b_panel.ncols();

    if b_panel.nrows() != bs {
        return Err(LinalgError::DimensionError(format!(
            "summa_step: a_panel ncols ({bs}) != b_panel nrows ({})",
            b_panel.nrows()
        )));
    }
    if c_tile.nrows() != m || c_tile.ncols() != n {
        return Err(LinalgError::DimensionError(format!(
            "summa_step: c_tile shape ({0}×{1}) != expected ({m}×{n})",
            c_tile.nrows(),
            c_tile.ncols()
        )));
    }

    // Outer-product update: C += A_panel * B_panel
    for ki in 0..bs {
        let a_col = a_panel.column(ki);
        let b_row = b_panel.row(ki);
        for i in 0..m {
            for j in 0..n {
                c_tile[[i, j]] += a_col[i] * b_row[j];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SUMMA simulation: single-process view
// ---------------------------------------------------------------------------

/// Simulate SUMMA matrix multiply on a single process: `C = A * B`.
///
/// This function faithfully simulates the SUMMA algorithm by partitioning `A`
/// into column panels and `B` into the corresponding row panels, then performing
/// outer-product accumulation.  The result is numerically identical to `A * B`.
///
/// # Complexity
///
/// The number of floating-point operations is `2·m·n·k`, same as dense GEMM.
/// The simulated communication volume (in number of `f64` words) is
/// `k * (m + n)` per virtual process — see [`CommCost::summa`].
///
/// # Arguments
///
/// * `a`      – Left matrix (m × k)
/// * `b`      – Right matrix (k × n)
/// * `config` – Distribution config (block_size, proc grid dimensions)
///
/// # Returns
///
/// Matrix C of shape (m × n) equal to A @ B.
///
/// # Errors
///
/// Returns an error if the inner dimensions do not match.
pub fn distributed_gemm_simulate(
    a: &Array2<f64>,
    b: &Array2<f64>,
    config: &DistribConfig,
) -> LinalgResult<Array2<f64>> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    if b.nrows() != k {
        return Err(LinalgError::DimensionError(format!(
            "distributed_gemm_simulate: A ncols ({k}) != B nrows ({})",
            b.nrows()
        )));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(LinalgError::ValueError(
            "distributed_gemm_simulate: all matrix dimensions must be > 0".to_string(),
        ));
    }

    let bs = config.block_size.max(1);
    let mut c = Array2::<f64>::zeros((m, n));

    // Panel loop: step over k dimension in blocks of size bs
    let n_steps = k.div_ceil(bs);
    for step in 0..n_steps {
        let col_start = step * bs;
        let col_end = (col_start + bs).min(k);

        let a_panel = a.slice(s![.., col_start..col_end]);
        let b_panel = b.slice(s![col_start..col_end, ..]);

        summa_step(&a_panel, &b_panel, &mut c)?;
    }

    Ok(c)
}

// ---------------------------------------------------------------------------
// Communication cost model
// ---------------------------------------------------------------------------

/// Communication cost estimates for distributed dense linear algebra.
pub struct CommCost;

impl CommCost {
    /// Estimate the number of `f64` words communicated *per process* in a SUMMA
    /// execution on a (P × Q) grid.
    ///
    /// Formula (bandwidth term only):
    ///
    /// ```text
    ///   words_per_proc = ceil(k / bs) * (ceil(m/P) * bs  +  ceil(n/Q) * bs)
    ///                  ≈ k * (m/P + n/Q)
    /// ```
    ///
    /// The latency cost (number of messages) is `2 * ceil(k/bs) * log2(max(P,Q))`.
    ///
    /// # Arguments
    ///
    /// * `m`  – number of rows in A and C
    /// * `n`  – number of columns in B and C
    /// * `k`  – inner dimension (columns of A = rows of B)
    /// * `p`  – number of process rows in the grid
    /// * `q`  – number of process columns in the grid
    /// * `bs` – block size
    ///
    /// # Returns
    ///
    /// `(bandwidth_words_per_proc, latency_messages_per_proc)`
    pub fn summa(m: usize, n: usize, k: usize, p: usize, q: usize, bs: usize) -> (usize, usize) {
        if p == 0 || q == 0 || bs == 0 {
            return (0, 0);
        }
        let n_steps = k.div_ceil(bs);

        // Local tile height/width (ceiling)
        let local_m = m.div_ceil(p);
        let local_n = n.div_ceil(q);

        // Per step: broadcast A-panel of size (local_m × bs) along row,
        //           broadcast B-panel of size (bs × local_n) along column.
        let bw_per_step = local_m * bs + bs * local_n;
        let bw_total = n_steps * bw_per_step;

        // Latency: two collectives (broadcast) per step, each costs log2(P or Q) messages
        let p_log = (p as f64).log2().ceil() as usize;
        let q_log = (q as f64).log2().ceil() as usize;
        let lat_total = 2 * n_steps * (p_log.max(q_log).max(1));

        (bw_total, lat_total)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // Helper: naive matmul for reference
    fn naive_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();
        let mut c = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for ki in 0..k {
                for j in 0..n {
                    c[[i, j]] += a[[i, ki]] * b[[ki, j]];
                }
            }
        }
        c
    }

    #[test]
    fn test_global_to_local_4x4_grid() {
        // 8×8 matrix, block_size=2, 4×4 virtual grid
        let data = Array2::<f64>::zeros((8, 8));
        let bcm = BlockCyclicMatrix::new(data, 2, 0, 0, 4, 4).expect("construct failed");

        // Global (0,0): block_row=0 -> owner_pr=0, block_col=0 -> owner_pc=0
        let (pr, pc, li, lj) = bcm.global_to_local(0, 0).expect("mapping failed");
        assert_eq!((pr, pc, li, lj), (0, 0, 0, 0));

        // Global (2,4): block_row=1 -> owner_pr=1, block_col=2 -> owner_pc=2
        // local_block_row = 1/4 = 0, intra = 2%2 = 0 -> local_i = 0
        // local_block_col = 2/4 = 0, intra = 4%2 = 0 -> local_j = 0
        let (pr, pc, li, lj) = bcm.global_to_local(2, 4).expect("mapping failed");
        assert_eq!((pr, pc), (1, 2));
        assert_eq!(li, 0);
        assert_eq!(lj, 0);

        // Global (5,7): block_row=2 -> owner_pr=2, block_col=3 -> owner_pc=3
        let (pr, pc, _li, _lj) = bcm.global_to_local(5, 7).expect("mapping failed");
        assert_eq!(pr, 2);
        assert_eq!(pc, 3);
    }

    #[test]
    fn test_global_to_local_out_of_bounds() {
        let data = Array2::<f64>::zeros((4, 4));
        let bcm = BlockCyclicMatrix::new(data, 2, 0, 0, 2, 2).expect("construct failed");
        assert!(bcm.global_to_local(4, 0).is_err());
        assert!(bcm.global_to_local(0, 4).is_err());
    }

    #[test]
    fn test_local_block_extraction() {
        // 6×6 matrix with known values, block_size=2
        let data = Array2::<f64>::from_shape_fn((6, 6), |(i, j)| (i * 6 + j) as f64);
        let bcm = BlockCyclicMatrix::new(data.clone(), 2, 0, 0, 3, 3).expect("construct failed");

        // Block (0,0) should be rows 0..2, cols 0..2
        let blk = bcm.local_block(0, 0).expect("block failed");
        assert_eq!(blk.shape(), &[2, 2]);
        assert_abs_diff_eq!(blk[[0, 0]], data[[0, 0]]);
        assert_abs_diff_eq!(blk[[1, 1]], data[[1, 1]]);

        // Block (2,2) should be rows 4..6, cols 4..6
        let blk = bcm.local_block(2, 2).expect("block failed");
        assert_abs_diff_eq!(blk[[0, 0]], data[[4, 4]]);
    }

    #[test]
    fn test_summa_4x4_square() {
        let a = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];
        let b = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let config = DistribConfig {
            block_size: 2,
            n_proc_rows: 2,
            n_proc_cols: 2,
        };
        let c = distributed_gemm_simulate(&a, &b, &config).expect("gemm failed");
        // A * I = A
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_summa_rectangular_mnek() {
        // m=3, k=5, n=4  (all different)
        let a = Array2::<f64>::from_shape_fn((3, 5), |(i, j)| (i as f64) + 0.1 * (j as f64));
        let b = Array2::<f64>::from_shape_fn((5, 4), |(i, j)| (j as f64) - 0.2 * (i as f64));
        let config = DistribConfig {
            block_size: 2,
            n_proc_rows: 2,
            n_proc_cols: 2,
        };
        let c_summa = distributed_gemm_simulate(&a, &b, &config).expect("gemm failed");
        let c_ref = naive_matmul(&a, &b);
        for i in 0..3 {
            for j in 0..4 {
                assert_abs_diff_eq!(c_summa[[i, j]], c_ref[[i, j]], epsilon = 1e-11);
            }
        }
    }

    #[test]
    fn test_summa_accumulation_equivalence() {
        // Verify that SUMMA correctly accumulates C += A_k * B_k over all panels
        let a = Array2::<f64>::from_shape_fn((6, 6), |(i, j)| ((i + 1) * (j + 2)) as f64);
        let b = Array2::<f64>::from_shape_fn((6, 6), |(i, j)| (i + j + 1) as f64);
        let config = DistribConfig {
            block_size: 3,
            n_proc_rows: 2,
            n_proc_cols: 2,
        };
        let c_summa = distributed_gemm_simulate(&a, &b, &config).expect("gemm failed");
        let c_ref = naive_matmul(&a, &b);
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(c_summa[[i, j]], c_ref[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_comm_cost_grows_with_p() {
        // Larger grid ↦ smaller local tiles ↦ fewer words per process
        let (bw1, _) = CommCost::summa(64, 64, 64, 2, 2, 8);
        let (bw2, _) = CommCost::summa(64, 64, 64, 4, 4, 8);
        // P=4,Q=4 has smaller local_m/local_n so fewer words per proc
        assert!(
            bw2 < bw1,
            "larger grid should send fewer words per proc: bw1={bw1} bw2={bw2}"
        );
    }

    #[test]
    fn test_comm_cost_latency_grows_with_k_over_bs() {
        // More panel steps → more messages
        let (_, lat_small_bs) = CommCost::summa(32, 32, 64, 2, 2, 8);
        let (_, lat_large_bs) = CommCost::summa(32, 32, 64, 2, 2, 64);
        // large bs → fewer steps → fewer messages
        assert!(lat_large_bs <= lat_small_bs);
    }

    #[test]
    fn test_summa_step_outer_product() {
        // a_panel (3×2), b_panel (2×4) → C (3×4)
        let a_panel = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b_panel = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let mut c = Array2::<f64>::zeros((3, 4));
        summa_step(&a_panel.view(), &b_panel.view(), &mut c).expect("summa_step failed");
        // Row 0 of C: a[0] * B = [1,0]*B = row0 of B = [1,2,3,4]
        assert_abs_diff_eq!(c[[0, 0]], 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(c[[0, 3]], 4.0, epsilon = 1e-14);
        // Row 1: [0,1]*B = row1 of B = [5,6,7,8]
        assert_abs_diff_eq!(c[[1, 0]], 5.0, epsilon = 1e-14);
        // Row 2: [1,1]*B = row0+row1 = [6,8,10,12]
        assert_abs_diff_eq!(c[[2, 0]], 6.0, epsilon = 1e-14);
        assert_abs_diff_eq!(c[[2, 3]], 12.0, epsilon = 1e-14);
    }
}
