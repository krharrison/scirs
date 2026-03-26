//! Configuration types for S4 and Mamba state-space sequence models.

/// Configuration for an S4 (Structured State Space Sequence) layer.
///
/// S4 parameterizes the state transition matrix A using the HiPPO-LegS basis,
/// enabling efficient long-range dependency modeling via convolution kernels.
#[derive(Debug, Clone)]
pub struct S4Config {
    /// Model (input/output) dimension.
    pub d_model: usize,
    /// State dimension (N). Controls the capacity of the SSM memory.
    /// Larger values allow remembering more distant history.
    pub d_state: usize,
    /// Sequence length for precomputed convolution kernels.
    /// Sequences longer than this are processed in chunks.
    pub seq_len: usize,
    /// Minimum log-spacing for the discretized timestep Δ.
    pub dt_min: f64,
    /// Maximum log-spacing for the discretized timestep Δ.
    pub dt_max: f64,
    /// Whether to apply the layer bidirectionally (process forward and backward, sum outputs).
    pub bidirectional: bool,
    /// Dropout probability applied after the SSM (0.0 = no dropout).
    pub dropout: f64,
}

impl Default for S4Config {
    fn default() -> Self {
        Self {
            d_model: 64,
            d_state: 16,
            seq_len: 128,
            dt_min: 0.001,
            dt_max: 0.1,
            bidirectional: false,
            dropout: 0.0,
        }
    }
}

/// Configuration for a Mamba (selective SSM) model.
///
/// Mamba extends S4 by making the state-space parameters (Δ, B, C)
/// input-dependent, allowing the model to selectively integrate or
/// ignore information at each timestep.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Model (input/output) dimension.
    pub d_model: usize,
    /// State dimension (N). Determines SSM memory capacity.
    pub d_state: usize,
    /// Local convolution width applied to the input projection before the SSM.
    pub d_conv: usize,
    /// Expansion factor: inner dimension = `expand * d_model`.
    pub expand: usize,
    /// Rank of the Δ (timestep) projection. Defaults to `ceil(d_model / 16)`.
    pub dt_rank: usize,
    /// Maximum sequence length (used for internal buffer sizing).
    pub seq_len: usize,
    /// Number of Mamba blocks stacked in the full model.
    pub n_layers: usize,
    /// Dropout probability applied between layers (0.0 = no dropout).
    pub dropout: f64,
}

impl Default for MambaConfig {
    fn default() -> Self {
        let d_model = 64_usize;
        // dt_rank = ceil(d_model / 16)
        let dt_rank = (d_model + 15) / 16;
        Self {
            d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank,
            seq_len: 512,
            n_layers: 4,
            dropout: 0.0,
        }
    }
}

impl MambaConfig {
    /// Return the inner (expanded) dimension: `expand * d_model`.
    #[inline]
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// Create a new config with `dt_rank` automatically set to `ceil(d_model / 16)`.
    pub fn with_auto_dt_rank(d_model: usize, d_state: usize, n_layers: usize) -> Self {
        let dt_rank = (d_model + 15) / 16;
        Self {
            d_model,
            d_state,
            d_conv: 4,
            expand: 2,
            dt_rank,
            seq_len: 512,
            n_layers,
            dropout: 0.0,
        }
    }
}
