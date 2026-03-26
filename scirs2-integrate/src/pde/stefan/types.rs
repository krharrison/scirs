//! Types for the Stefan problem solver.

use crate::error::IntegrateError;

/// Configuration for the 1D one-phase Stefan (melting) problem.
///
/// The PDE is:
///
/// ```text
/// ∂u/∂t = α ∂²u/∂x²   for 0 < x < s(t)
/// u(0,t) = T_wall  (Dirichlet at fixed wall)
/// u(s(t),t) = T_m  (temperature at interface = melting point)
/// ds/dt = -(α/St) ∂u/∂x|_{x=s(t)⁻}   (Stefan condition)
/// s(0) = s_init  (initial interface position, ≥ 0)
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StefanConfig {
    /// Number of spatial grid points in the fixed domain `[0, L_max]`.
    /// Default: 100.
    pub nx: usize,
    /// Time step Δt (explicit Euler in the enthalpy method). Default: 1e-4.
    pub dt: f64,
    /// Stefan number St = c_p (T_wall − T_m) / L.  Default: 1.0.
    pub stefan_number: f64,
    /// Thermal diffusivity α.  Default: 1.0.
    pub diffusivity: f64,
    /// Melting temperature T_m.  Default: 0.0.
    pub melting_temp: f64,
    /// Wall temperature T_wall (must be > melting_temp).  Default: 1.0.
    pub wall_temp: f64,
    /// Maximum physical domain length L_max.  Default: 5.0.
    pub l_max: f64,
    /// Final time.  Default: 1.0.
    pub max_time: f64,
    /// Output every `output_every` steps.  Default: 10.
    pub output_every: usize,
}

impl Default for StefanConfig {
    fn default() -> Self {
        Self {
            nx: 100,
            dt: 1e-4,
            stefan_number: 1.0,
            diffusivity: 1.0,
            melting_temp: 0.0,
            wall_temp: 1.0,
            l_max: 5.0,
            max_time: 1.0,
            output_every: 10,
        }
    }
}

impl StefanConfig {
    /// Validate the configuration and return a descriptive error if invalid.
    pub fn validate(&self) -> Result<(), IntegrateError> {
        if self.nx < 4 {
            return Err(IntegrateError::InvalidInput(
                "nx must be at least 4".to_string(),
            ));
        }
        if self.dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "dt must be positive".to_string(),
            ));
        }
        if self.stefan_number <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "stefan_number must be positive".to_string(),
            ));
        }
        if self.diffusivity <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "diffusivity must be positive".to_string(),
            ));
        }
        if self.wall_temp <= self.melting_temp {
            return Err(IntegrateError::InvalidInput(
                "wall_temp must be strictly greater than melting_temp".to_string(),
            ));
        }
        if self.l_max <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "l_max must be positive".to_string(),
            ));
        }
        if self.max_time <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "max_time must be positive".to_string(),
            ));
        }
        if self.output_every == 0 {
            return Err(IntegrateError::InvalidInput(
                "output_every must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Result of a Stefan problem simulation.
#[derive(Debug, Clone)]
pub struct StefanResult {
    /// Output times.
    pub times: Vec<f64>,
    /// Interface position `s(t)` at each output time.
    pub interface_positions: Vec<f64>,
    /// Temperature field `u[i][j]` = temperature at output time `i`, grid node `j`.
    pub temperature_fields: Vec<Vec<f64>>,
    /// Spatial grid `x_j` (fixed, uniform on `[0, L_max]`).
    pub grid: Vec<f64>,
}
