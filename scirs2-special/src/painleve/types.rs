//! Type definitions for Painleve transcendents
//!
//! This module defines the core types used to configure and represent solutions
//! of the six Painleve equations P-I through P-VI.

use std::fmt;

/// Specifies which Painleve equation to solve.
///
/// The six Painleve equations are a family of second-order nonlinear ODEs whose
/// solutions define new transcendental functions (the Painleve transcendents).
/// They were classified by Painleve and Gambier (1900-1910) as the only
/// second-order ODEs of the form y'' = R(t, y, y') (with R rational in y and y')
/// whose movable singularities are at most poles.
///
/// # References
///
/// - DLMF Chapter 32: <https://dlmf.nist.gov/32>
/// - Clarkson, P.A. (2006), "Painleve Equations -- Nonlinear Special Functions"
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PainleveEquation {
    /// Painleve I: y'' = 6y^2 + t
    PI,
    /// Painleve II: y'' = 2y^3 + ty + alpha
    PII {
        /// Parameter alpha
        alpha: f64,
    },
    /// Painleve III: y'' = (y')^2/y - y'/t + (alpha*y^2 + beta)/t + gamma*y^3 + delta/y
    PIII {
        /// Parameter alpha
        alpha: f64,
        /// Parameter beta
        beta: f64,
        /// Parameter gamma
        gamma: f64,
        /// Parameter delta
        delta: f64,
    },
    /// Painleve IV: y'' = (y')^2/(2y) + 3y^3/2 + 4ty^2 + 2(t^2-alpha)y + beta/y
    PIV {
        /// Parameter alpha
        alpha: f64,
        /// Parameter beta
        beta: f64,
    },
    /// Painleve V: y'' = ((3y-1)/(2y(y-1)))*(y')^2 - y'/t
    ///   + ((y-1)^2*(alpha*y + beta/y))/t^2 + gamma*y/t + delta*y(y+1)/(y-1)
    PV {
        /// Parameter alpha
        alpha: f64,
        /// Parameter beta
        beta: f64,
        /// Parameter gamma
        gamma: f64,
        /// Parameter delta
        delta: f64,
    },
    /// Painleve VI: y'' = (1/2)(1/y + 1/(y-1) + 1/(y-t))*(y')^2
    ///   - (1/t + 1/(t-1) + 1/(y-t))*y'
    ///   + y(y-1)(y-t)/(t^2(t-1)^2) * (alpha + beta*t/y^2 + gamma*(t-1)/(y-1)^2 + delta*t(t-1)/(y-t)^2)
    PVI {
        /// Parameter alpha
        alpha: f64,
        /// Parameter beta
        beta: f64,
        /// Parameter gamma
        gamma: f64,
        /// Parameter delta
        delta: f64,
    },
}

impl fmt::Display for PainleveEquation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PainleveEquation::PI => write!(f, "Painleve I"),
            PainleveEquation::PII { alpha } => write!(f, "Painleve II (alpha={alpha})"),
            PainleveEquation::PIII {
                alpha,
                beta,
                gamma,
                delta,
            } => write!(
                f,
                "Painleve III (alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})"
            ),
            PainleveEquation::PIV { alpha, beta } => {
                write!(f, "Painleve IV (alpha={alpha}, beta={beta})")
            }
            PainleveEquation::PV {
                alpha,
                beta,
                gamma,
                delta,
            } => write!(
                f,
                "Painleve V (alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})"
            ),
            PainleveEquation::PVI {
                alpha,
                beta,
                gamma,
                delta,
            } => write!(
                f,
                "Painleve VI (alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})"
            ),
        }
    }
}

/// Configuration for the Painleve ODE solver.
///
/// Controls the equation type, integration interval, initial conditions,
/// tolerances, and maximum number of steps.
#[derive(Debug, Clone)]
pub struct PainleveConfig {
    /// Which Painleve equation to solve
    pub equation: PainleveEquation,
    /// Start of the integration interval
    pub t_start: f64,
    /// End of the integration interval
    pub t_end: f64,
    /// Initial value y(t_start)
    pub y0: f64,
    /// Initial derivative y'(t_start)
    pub dy0: f64,
    /// Relative and absolute tolerance for the adaptive solver
    pub tolerance: f64,
    /// Maximum number of integration steps
    pub max_steps: usize,
    /// Pole detection threshold: when |y| exceeds this value, a pole is recorded
    pub pole_threshold: f64,
}

impl Default for PainleveConfig {
    fn default() -> Self {
        Self {
            equation: PainleveEquation::PI,
            t_start: 0.0,
            t_end: 1.0,
            y0: 0.0,
            dy0: 0.0,
            tolerance: 1e-10,
            max_steps: 100_000,
            pole_threshold: 1e10,
        }
    }
}

/// Solution of a Painleve ODE initial-value problem.
///
/// Contains the discrete trajectory (t, y, y') together with detected pole
/// locations and convergence information.
#[derive(Debug, Clone)]
pub struct PainleveSolution {
    /// Time (independent variable) values
    pub t_values: Vec<f64>,
    /// Solution values y(t)
    pub y_values: Vec<f64>,
    /// Derivative values y'(t)
    pub dy_values: Vec<f64>,
    /// Approximate locations of detected poles (where |y| exceeded the threshold)
    pub poles: Vec<f64>,
    /// Whether the integration completed without hitting a pole or divergence
    pub converged: bool,
    /// Total number of accepted integration steps
    pub steps_taken: usize,
}

impl PainleveSolution {
    /// Interpolate the solution at a given t value using linear interpolation.
    ///
    /// Returns `None` if `t` is outside the computed range.
    pub fn interpolate(&self, t: f64) -> Option<f64> {
        if self.t_values.is_empty() {
            return None;
        }
        let t0 = self.t_values[0];
        let tn = self.t_values[self.t_values.len() - 1];
        let (t_lo, t_hi) = if t0 <= tn { (t0, tn) } else { (tn, t0) };
        if t < t_lo - 1e-14 || t > t_hi + 1e-14 {
            return None;
        }
        // Binary search for the interval
        let ascending = t0 <= tn;
        let idx = if ascending {
            match self
                .t_values
                .binary_search_by(|v| v.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => return Some(self.y_values[i]),
                Err(i) => i,
            }
        } else {
            // Descending order: reverse search
            match self
                .t_values
                .binary_search_by(|v| t.partial_cmp(v).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => return Some(self.y_values[i]),
                Err(i) => i,
            }
        };
        if idx == 0 {
            return Some(self.y_values[0]);
        }
        if idx >= self.t_values.len() {
            return Some(self.y_values[self.t_values.len() - 1]);
        }
        let t0_local = self.t_values[idx - 1];
        let t1_local = self.t_values[idx];
        let y0_local = self.y_values[idx - 1];
        let y1_local = self.y_values[idx];
        let frac = (t - t0_local) / (t1_local - t0_local);
        Some(y0_local + frac * (y1_local - y0_local))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_painleve_equation_display() {
        let eq = PainleveEquation::PI;
        assert_eq!(format!("{eq}"), "Painleve I");

        let eq2 = PainleveEquation::PII { alpha: 0.5 };
        assert!(format!("{eq2}").contains("alpha=0.5"));
    }

    #[test]
    fn test_default_config() {
        let cfg = PainleveConfig::default();
        assert_eq!(cfg.t_start, 0.0);
        assert_eq!(cfg.t_end, 1.0);
        assert_eq!(cfg.tolerance, 1e-10);
        assert_eq!(cfg.max_steps, 100_000);
    }

    #[test]
    fn test_solution_interpolate_empty() {
        let sol = PainleveSolution {
            t_values: vec![],
            y_values: vec![],
            dy_values: vec![],
            poles: vec![],
            converged: true,
            steps_taken: 0,
        };
        assert!(sol.interpolate(0.5).is_none());
    }

    #[test]
    fn test_solution_interpolate_basic() {
        let sol = PainleveSolution {
            t_values: vec![0.0, 1.0, 2.0],
            y_values: vec![0.0, 1.0, 4.0],
            dy_values: vec![0.0, 1.0, 4.0],
            poles: vec![],
            converged: true,
            steps_taken: 2,
        };
        // Exact node
        let v = sol.interpolate(1.0);
        assert!((v.unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
        // Midpoint interpolation
        let v2 = sol.interpolate(0.5);
        assert!((v2.unwrap_or(f64::NAN) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_solution_interpolate_out_of_range() {
        let sol = PainleveSolution {
            t_values: vec![0.0, 1.0],
            y_values: vec![0.0, 1.0],
            dy_values: vec![0.0, 1.0],
            poles: vec![],
            converged: true,
            steps_taken: 1,
        };
        assert!(sol.interpolate(-1.0).is_none());
        assert!(sol.interpolate(2.0).is_none());
    }

    #[test]
    fn test_painleve_equation_clone() {
        let eq = PainleveEquation::PIII {
            alpha: 1.0,
            beta: -1.0,
            gamma: 1.0,
            delta: -1.0,
        };
        let eq2 = eq.clone();
        assert_eq!(eq, eq2);
    }
}
