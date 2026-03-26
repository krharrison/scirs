//! Types and configuration structures for N-dimensional FFT.

/// Normalization mode for FFT operations.
///
/// Controls how the forward and inverse transforms are scaled.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormMode {
    /// Forward: unnormalized (sum), Inverse: 1/N scaling.
    /// This is the default convention (matches numpy/scipy).
    #[default]
    None,
    /// Both forward and inverse scaled by 1/sqrt(N) — preserves energy.
    Ortho,
    /// Forward: 1/N scaling, Inverse: unnormalized.
    Forward,
}

/// Configuration for N-dimensional FFT operations.
#[derive(Debug, Clone)]
pub struct NdimFftConfig {
    /// Normalization mode applied after transform.
    pub norm: NormMode,
    /// Number of threads to use (0 = auto-detect from available cores).
    pub threads: usize,
    /// Cache tile size for tiled 2D FFT (in elements per dimension).
    /// Tuned for L1 cache; typical value 64 gives 64×64×16B = 64 KB.
    pub tile_size: usize,
}

impl Default for NdimFftConfig {
    fn default() -> Self {
        NdimFftConfig {
            norm: NormMode::default(),
            threads: 0,
            tile_size: 64,
        }
    }
}

/// Result type for N-dimensional FFT operations.
#[derive(Debug, Clone)]
pub struct NdimFftResult {
    /// Flat complex data in row-major (C) order.
    pub data: Vec<(f64, f64)>,
    /// Shape of the N-dimensional array.
    pub shape: Vec<usize>,
}

impl NdimFftResult {
    /// Creates a new result from flat data and shape.
    ///
    /// Returns `None` if `data.len()` does not match the product of `shape`.
    pub fn new(data: Vec<(f64, f64)>, shape: Vec<usize>) -> Option<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return None;
        }
        Some(NdimFftResult { data, shape })
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if there are no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Precomputed FFT plan for a given shape.
///
/// Stores twiddle factor tables for each axis so they can be reused
/// across multiple transforms of the same shape.
#[derive(Debug, Clone)]
pub struct FftPlan {
    /// Shape of the N-dimensional array this plan was built for.
    pub shape: Vec<usize>,
    /// Row-major strides (in elements) for each dimension.
    pub strides: Vec<usize>,
    /// Per-axis twiddle factor tables: `twiddle_tables[axis][k]` = e^{-2πi k/N_axis}.
    pub twiddle_tables: Vec<Vec<(f64, f64)>>,
}

impl FftPlan {
    /// Builds an FFT plan for the given shape, precomputing twiddle tables.
    pub fn build(shape: &[usize]) -> Self {
        let ndim = shape.len();

        // Compute row-major strides
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Precompute twiddle factor tables for each axis
        let twiddle_tables: Vec<Vec<(f64, f64)>> = shape
            .iter()
            .map(|&n| crate::ndim_fft::mixed_radix::compute_twiddles(n))
            .collect();

        FftPlan {
            shape: shape.to_vec(),
            strides,
            twiddle_tables,
        }
    }
}

/// Specifies which axes to transform in a multi-dimensional FFT.
#[derive(Debug, Clone, Default)]
pub enum FftAxis {
    /// Transform all axes (default N-D FFT behaviour).
    #[default]
    All,
    /// Transform only the listed axes (in the given order).
    Selected(Vec<usize>),
}
