//! Types and configuration for Physics-Informed Neural Networks.

/// Boundary condition types for PDE problems.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BoundaryCondition {
    /// Dirichlet boundary condition: u = value
    Dirichlet {
        /// The prescribed value at the boundary
        value: f64,
    },
    /// Neumann boundary condition: du/dn = flux
    Neumann {
        /// The prescribed normal flux at the boundary
        flux: f64,
    },
    /// Robin boundary condition: alpha*u + beta*du/dn = value
    Robin {
        /// Coefficient of u
        alpha: f64,
        /// Coefficient of du/dn
        beta: f64,
        /// Right-hand side value
        value: f64,
    },
    /// Periodic boundary condition
    Periodic,
}

/// Strategy for placing collocation points in the domain interior.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub enum CollocationStrategy {
    /// Uniformly random placement
    #[default]
    Random,
    /// Latin Hypercube Sampling for better space-filling
    LatinHypercube,
    /// Uniform grid placement
    UniformGrid,
    /// Adaptive placement with more points where the PDE residual is high
    AdaptiveResidual,
}

/// Configuration for PINN training.
#[derive(Debug, Clone)]
pub struct PINNConfig {
    /// Number of neurons per hidden layer (default: \[64, 64, 64\])
    pub hidden_layers: Vec<usize>,
    /// Adam optimizer learning rate (default: 1e-3)
    pub learning_rate: f64,
    /// Maximum training epochs (default: 10000)
    pub max_epochs: usize,
    /// Number of interior collocation points (default: 1000)
    pub n_collocation: usize,
    /// Number of boundary points per boundary segment (default: 100)
    pub n_boundary: usize,
    /// Weight for PDE residual loss (default: 1.0)
    pub physics_weight: f64,
    /// Weight for boundary condition loss (default: 10.0)
    pub boundary_weight: f64,
    /// Weight for observational data loss (default: 1.0)
    pub data_weight: f64,
    /// Collocation point placement strategy
    pub collocation: CollocationStrategy,
    /// Stop training when total loss falls below this threshold (default: 1e-6)
    pub convergence_tol: f64,
}

impl Default for PINNConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![64, 64, 64],
            learning_rate: 1e-3,
            max_epochs: 10000,
            n_collocation: 1000,
            n_boundary: 100,
            physics_weight: 1.0,
            boundary_weight: 10.0,
            data_weight: 1.0,
            collocation: CollocationStrategy::default(),
            convergence_tol: 1e-6,
        }
    }
}

/// Result of PINN training.
#[derive(Debug, Clone)]
pub struct PINNResult {
    /// Total loss at the end of training
    pub final_loss: f64,
    /// Physics (PDE residual) loss component
    pub physics_loss: f64,
    /// Boundary condition loss component
    pub boundary_loss: f64,
    /// Observational data loss component
    pub data_loss: f64,
    /// Number of epochs actually trained
    pub epochs_trained: usize,
    /// Whether training converged (loss < convergence_tol)
    pub converged: bool,
    /// Loss value at each epoch
    pub loss_history: Vec<f64>,
}

/// Defines a PDE problem for PINN solving.
#[derive(Debug, Clone)]
pub struct PDEProblem {
    /// Number of spatial dimensions (1, 2, or 3)
    pub spatial_dim: usize,
    /// Domain bounds per spatial dimension: \[(x_min, x_max), (y_min, y_max), ...\]
    pub domain: Vec<(f64, f64)>,
    /// Boundary specifications
    pub boundaries: Vec<Boundary>,
    /// Whether the problem includes a time variable
    pub has_time: bool,
    /// Time domain bounds (t_min, t_max), required when `has_time` is true
    pub time_domain: Option<(f64, f64)>,
}

/// A single boundary specification.
#[derive(Debug, Clone)]
pub struct Boundary {
    /// Which spatial dimension this boundary constrains
    pub dim: usize,
    /// Whether this is the low or high end of the dimension
    pub side: BoundarySide,
    /// The boundary condition to apply
    pub condition: BoundaryCondition,
}

/// Which side of a dimension a boundary lies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundarySide {
    /// Lower bound of the dimension
    Low,
    /// Upper bound of the dimension
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PINNConfig::default();
        assert_eq!(config.hidden_layers, vec![64, 64, 64]);
        assert!((config.learning_rate - 1e-3).abs() < 1e-15);
        assert_eq!(config.max_epochs, 10000);
        assert_eq!(config.n_collocation, 1000);
        assert_eq!(config.n_boundary, 100);
        assert!((config.physics_weight - 1.0).abs() < 1e-15);
        assert!((config.boundary_weight - 10.0).abs() < 1e-15);
        assert!((config.data_weight - 1.0).abs() < 1e-15);
        assert!((config.convergence_tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_boundary_condition_variants() {
        let dirichlet = BoundaryCondition::Dirichlet { value: 1.0 };
        let neumann = BoundaryCondition::Neumann { flux: 0.5 };
        let robin = BoundaryCondition::Robin {
            alpha: 1.0,
            beta: 2.0,
            value: 3.0,
        };
        let periodic = BoundaryCondition::Periodic;

        // Verify Debug trait works
        let _ = format!("{:?}", dirichlet);
        let _ = format!("{:?}", neumann);
        let _ = format!("{:?}", robin);
        let _ = format!("{:?}", periodic);
    }

    #[test]
    fn test_pde_problem_construction() {
        let problem = PDEProblem {
            spatial_dim: 2,
            domain: vec![(0.0, 1.0), (0.0, 1.0)],
            boundaries: vec![
                Boundary {
                    dim: 0,
                    side: BoundarySide::Low,
                    condition: BoundaryCondition::Dirichlet { value: 0.0 },
                },
                Boundary {
                    dim: 0,
                    side: BoundarySide::High,
                    condition: BoundaryCondition::Dirichlet { value: 1.0 },
                },
            ],
            has_time: false,
            time_domain: None,
        };

        assert_eq!(problem.spatial_dim, 2);
        assert_eq!(problem.domain.len(), 2);
        assert_eq!(problem.boundaries.len(), 2);
        assert!(!problem.has_time);
    }

    #[test]
    fn test_boundary_side_equality() {
        assert_eq!(BoundarySide::Low, BoundarySide::Low);
        assert_eq!(BoundarySide::High, BoundarySide::High);
        assert_ne!(BoundarySide::Low, BoundarySide::High);
    }

    #[test]
    fn test_pinn_result_fields() {
        let result = PINNResult {
            final_loss: 0.001,
            physics_loss: 0.0005,
            boundary_loss: 0.0003,
            data_loss: 0.0002,
            epochs_trained: 5000,
            converged: true,
            loss_history: vec![1.0, 0.5, 0.1, 0.01, 0.001],
        };

        assert!(result.converged);
        assert_eq!(result.epochs_trained, 5000);
        assert_eq!(result.loss_history.len(), 5);
        assert!(result.final_loss < 0.01);
    }

    #[test]
    fn test_collocation_strategy_default() {
        let strategy = CollocationStrategy::default();
        assert!(matches!(strategy, CollocationStrategy::Random));
    }

    #[test]
    fn test_time_dependent_problem() {
        let problem = PDEProblem {
            spatial_dim: 1,
            domain: vec![(0.0, 1.0)],
            boundaries: vec![],
            has_time: true,
            time_domain: Some((0.0, 1.0)),
        };

        assert!(problem.has_time);
        assert!(problem.time_domain.is_some());
        let (t_min, t_max) = problem.time_domain.unwrap_or((0.0, 0.0));
        assert!((t_min - 0.0).abs() < 1e-15);
        assert!((t_max - 1.0).abs() < 1e-15);
    }
}
