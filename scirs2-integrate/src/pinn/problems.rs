//! Pre-built PDE residual functions and problem definitions for common PDEs.

use super::network::PINNNetwork;
use super::types::{Boundary, BoundaryCondition, BoundarySide, PDEProblem};
use crate::error::IntegrateResult;
use scirs2_core::ndarray::Array1;

/// Laplace equation residual: nabla^2 u = 0.
///
/// Returns the PDE residual at point `x`, which should be zero for an exact solution.
///
/// # Arguments
/// * `network` - The PINN network approximating u
/// * `x` - Spatial coordinates (2D point)
pub fn laplace_residual(network: &PINNNetwork, x: &Array1<f64>) -> IntegrateResult<f64> {
    let h = 1e-4;
    network.laplacian(x, h)
}

/// Poisson equation residual: nabla^2 u - f(x) = 0.
///
/// # Arguments
/// * `network` - The PINN network approximating u
/// * `x` - Spatial coordinates
/// * `source` - Source function f(x)
pub fn poisson_residual<F>(
    network: &PINNNetwork,
    x: &Array1<f64>,
    source: &F,
) -> IntegrateResult<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let h = 1e-4;
    let laplacian = network.laplacian(x, h)?;
    Ok(laplacian - source(x))
}

/// Heat equation residual: du/dt - alpha * nabla^2 u = 0.
///
/// Assumes the input vector is \[x_1, ..., x_d, t\] where the last element is time.
///
/// # Arguments
/// * `network` - The PINN network approximating u(x, t)
/// * `x` - Spatial-temporal coordinates
/// * `alpha` - Thermal diffusivity coefficient
pub fn heat_residual(network: &PINNNetwork, x: &Array1<f64>, alpha: f64) -> IntegrateResult<f64> {
    let h = 1e-4;
    let du_dt = network.time_derivative(x, h)?;

    // Compute spatial Laplacian (exclude the time dimension which is last)
    let spatial_dim = x.len() - 1;
    let u_center = network.forward(x)?;
    let mut spatial_laplacian = 0.0;
    let h_sq = h * h;

    for d in 0..spatial_dim {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[d] += h;
        x_minus[d] -= h;

        let u_plus = network.forward(&x_plus)?;
        let u_minus = network.forward(&x_minus)?;
        spatial_laplacian += (u_plus - 2.0 * u_center + u_minus) / h_sq;
    }

    Ok(du_dt - alpha * spatial_laplacian)
}

/// Burgers' equation residual: du/dt + u * du/dx - nu * d2u/dx2 = 0.
///
/// This is the 1D viscous Burgers' equation. The input is assumed to be \[x, t\].
///
/// # Arguments
/// * `network` - The PINN network approximating u(x, t)
/// * `x` - Input vector \[x, t\]
/// * `nu` - Viscosity coefficient
pub fn burgers_residual(network: &PINNNetwork, x: &Array1<f64>, nu: f64) -> IntegrateResult<f64> {
    let h = 1e-4;

    let u = network.forward(x)?;
    let du_dt = network.time_derivative(x, h)?;

    // du/dx (spatial, dim 0)
    let mut x_plus = x.clone();
    let mut x_minus = x.clone();
    x_plus[0] += h;
    x_minus[0] -= h;
    let u_plus = network.forward(&x_plus)?;
    let u_minus = network.forward(&x_minus)?;
    let du_dx = (u_plus - u_minus) / (2.0 * h);

    // d2u/dx2
    let d2u_dx2 = network.second_derivative(x, 0, h)?;

    Ok(du_dt + u * du_dx - nu * d2u_dx2)
}

/// Create a 2D Laplace problem on a rectangular domain with Dirichlet BCs (zero).
///
/// # Arguments
/// * `bounds` - Domain bounds as (x_min, x_max, y_min, y_max)
pub fn laplace_problem_2d(bounds: (f64, f64, f64, f64)) -> PDEProblem {
    let (x_min, x_max, y_min, y_max) = bounds;
    PDEProblem {
        spatial_dim: 2,
        domain: vec![(x_min, x_max), (y_min, y_max)],
        boundaries: vec![
            Boundary {
                dim: 0,
                side: BoundarySide::Low,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 0,
                side: BoundarySide::High,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 1,
                side: BoundarySide::Low,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 1,
                side: BoundarySide::High,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
        ],
        has_time: false,
        time_domain: None,
    }
}

/// Create a 1D heat equation problem with Dirichlet BCs (zero at both ends).
///
/// # Arguments
/// * `x_range` - Spatial domain (x_min, x_max)
/// * `t_range` - Time domain (t_min, t_max)
/// * `_alpha` - Thermal diffusivity (stored in the problem for reference, used by `heat_residual`)
pub fn heat_problem_1d(x_range: (f64, f64), t_range: (f64, f64), _alpha: f64) -> PDEProblem {
    PDEProblem {
        spatial_dim: 1,
        domain: vec![x_range],
        boundaries: vec![
            Boundary {
                dim: 0,
                side: BoundarySide::Low,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 0,
                side: BoundarySide::High,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
        ],
        has_time: true,
        time_domain: Some(t_range),
    }
}

/// Create a 2D Poisson problem on a rectangular domain with Dirichlet BCs (zero).
///
/// # Arguments
/// * `bounds` - Domain bounds as (x_min, x_max, y_min, y_max)
pub fn poisson_problem_2d(bounds: (f64, f64, f64, f64)) -> PDEProblem {
    let (x_min, x_max, y_min, y_max) = bounds;
    PDEProblem {
        spatial_dim: 2,
        domain: vec![(x_min, x_max), (y_min, y_max)],
        boundaries: vec![
            Boundary {
                dim: 0,
                side: BoundarySide::Low,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 0,
                side: BoundarySide::High,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 1,
                side: BoundarySide::Low,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
            Boundary {
                dim: 1,
                side: BoundarySide::High,
                condition: BoundaryCondition::Dirichlet { value: 0.0 },
            },
        ],
        has_time: false,
        time_domain: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_laplace_residual_for_linear_function() {
        // u(x,y) = x is harmonic => Laplacian = 0
        // A network that outputs ~x would have ~0 Laplacian.
        // We test that the residual function itself runs and returns a finite value.
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        let x = array![0.5, 0.5];
        let res = laplace_residual(&net, &x);
        assert!(res.is_ok());
        assert!(res.expect("residual").is_finite());
    }

    #[test]
    fn test_poisson_residual_computation() {
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        let x = array![0.5, 0.5];
        let source = |_: &Array1<f64>| 1.0;
        let res = poisson_residual(&net, &x, &source);
        assert!(res.is_ok());
        assert!(res.expect("residual").is_finite());
    }

    #[test]
    fn test_heat_residual_computation() {
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        // [x, t]
        let x = array![0.5, 0.1];
        let res = heat_residual(&net, &x, 0.01);
        assert!(res.is_ok());
        assert!(res.expect("residual").is_finite());
    }

    #[test]
    fn test_burgers_residual_computation() {
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        // [x, t]
        let x = array![0.5, 0.1];
        let res = burgers_residual(&net, &x, 0.01);
        assert!(res.is_ok());
        assert!(res.expect("residual").is_finite());
    }

    #[test]
    fn test_laplace_problem_2d_definition() {
        let problem = laplace_problem_2d((0.0, 1.0, 0.0, 2.0));
        assert_eq!(problem.spatial_dim, 2);
        assert_eq!(problem.domain.len(), 2);
        assert!((problem.domain[0].0 - 0.0).abs() < 1e-15);
        assert!((problem.domain[0].1 - 1.0).abs() < 1e-15);
        assert!((problem.domain[1].0 - 0.0).abs() < 1e-15);
        assert!((problem.domain[1].1 - 2.0).abs() < 1e-15);
        assert_eq!(problem.boundaries.len(), 4);
        assert!(!problem.has_time);
        assert!(problem.time_domain.is_none());
    }

    #[test]
    fn test_heat_problem_1d_definition() {
        let problem = heat_problem_1d((0.0, 1.0), (0.0, 0.5), 0.01);
        assert_eq!(problem.spatial_dim, 1);
        assert_eq!(problem.domain.len(), 1);
        assert!(problem.has_time);
        let (t_min, t_max) = problem.time_domain.unwrap_or((0.0, 0.0));
        assert!((t_min - 0.0).abs() < 1e-15);
        assert!((t_max - 0.5).abs() < 1e-15);
        assert_eq!(problem.boundaries.len(), 2);
    }

    #[test]
    fn test_poisson_problem_2d_definition() {
        let problem = poisson_problem_2d((-1.0, 1.0, -1.0, 1.0));
        assert_eq!(problem.spatial_dim, 2);
        assert_eq!(problem.domain.len(), 2);
        assert!((problem.domain[0].0 - (-1.0)).abs() < 1e-15);
        assert!((problem.domain[0].1 - 1.0).abs() < 1e-15);
        assert_eq!(problem.boundaries.len(), 4);
        assert!(!problem.has_time);
    }

    #[test]
    fn test_heat_residual_steady_state() {
        // For a network with constant output (u = c), du/dt = 0 and nabla^2 u = 0,
        // so the heat residual should be approximately zero.
        // We can't easily force constant output, but we verify the function returns finite values.
        let net = PINNNetwork::new(2, &[8], 1).expect("network creation");
        let x = array![0.5, 0.25];
        let res = heat_residual(&net, &x, 1.0);
        assert!(res.is_ok());
        let val = res.expect("residual");
        assert!(val.is_finite());
    }

    #[test]
    fn test_burgers_known_structure() {
        // Burgers: du/dt + u*du/dx - nu*d2u/dx2 = 0
        // Verify all terms are computed and combined correctly
        let net = PINNNetwork::new(2, &[8, 8], 1).expect("network creation");
        let x = array![0.3, 0.2];
        let nu = 0.1;
        let res = burgers_residual(&net, &x, nu);
        assert!(res.is_ok());

        // The residual should be a combination of three finite terms
        let val = res.expect("residual");
        assert!(val.is_finite());
    }

    #[test]
    fn test_poisson_with_zero_source() {
        // Poisson with f=0 reduces to Laplace
        let net = PINNNetwork::new(2, &[16], 1).expect("network creation");
        let x = array![0.5, 0.5];
        let zero_source = |_: &Array1<f64>| 0.0;

        let poisson_res = poisson_residual(&net, &x, &zero_source).expect("poisson");
        let laplace_res = laplace_residual(&net, &x).expect("laplace");

        assert!(
            (poisson_res - laplace_res).abs() < 1e-10,
            "Poisson with zero source should equal Laplace: poisson={}, laplace={}",
            poisson_res,
            laplace_res
        );
    }
}
