//! Total-variation regularised depth completion.
//!
//! Uses the Chambolle-Pock primal-dual algorithm to solve:
//!
//! ```text
//! min  sum_{observed} (D_i - d_i)^2  +  lambda * sum_{adjacent} |D_i - D_j| * w_ij
//! ```
//!
//! where `w_ij = exp(-||RGB_i - RGB_j||^2 / (2 sigma^2))` when an RGB guide
//! is provided (anisotropic TV), otherwise `w_ij = 1` (isotropic TV).

use scirs2_core::ndarray::Array2;

use crate::error::Result;

use super::types::{CompletionMethod, CompletionResult, DepthCompletionConfig, SparseDepthMap};

// ─────────────────────────────────────────────────────────────────────────────
// Gradient / divergence helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Forward-difference gradient with Neumann boundary conditions.
///
/// Returns `(grad_x, grad_y)` where `grad_x` is the horizontal gradient
/// (column differences) and `grad_y` is the vertical gradient (row differences).
fn compute_gradient(d: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (h, w) = d.dim();
    let mut gx = Array2::zeros((h, w));
    let mut gy = Array2::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            // Horizontal gradient (forward difference, zero at right boundary).
            if col + 1 < w {
                gx[[row, col]] = d[[row, col + 1]] - d[[row, col]];
            }
            // Vertical gradient (forward difference, zero at bottom boundary).
            if row + 1 < h {
                gy[[row, col]] = d[[row + 1, col]] - d[[row, col]];
            }
        }
    }
    (gx, gy)
}

/// Negative adjoint of the gradient (discrete divergence).
///
/// `div(p) = -(px_{i,j} - px_{i,j-1}) - (py_{i,j} - py_{i-1,j})`
/// with zero-boundary conditions outside the domain.
fn compute_divergence(px: &Array2<f64>, py: &Array2<f64>) -> Array2<f64> {
    let (h, w) = px.dim();
    let mut div = Array2::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            // Horizontal divergence.
            let dx = if col == 0 {
                px[[row, 0]]
            } else if col == w - 1 {
                -px[[row, col - 1]]
            } else {
                px[[row, col]] - px[[row, col - 1]]
            };

            // Vertical divergence.
            let dy = if row == 0 {
                py[[0, col]]
            } else if row == h - 1 {
                -py[[row - 1, col]]
            } else {
                py[[row, col]] - py[[row - 1, col]]
            };

            div[[row, col]] = dx + dy;
        }
    }
    div
}

/// Compute edge-aware weights from an RGB guide image.
///
/// `w_x[i,j]` is the weight on the horizontal edge between `(i,j)` and `(i,j+1)`.
/// `w_y[i,j]` is the weight on the vertical edge between `(i,j)` and `(i+1,j)`.
fn compute_edge_weights(guide: &Array2<f64>, sigma_intensity: f64) -> (Array2<f64>, Array2<f64>) {
    let (h, w) = guide.dim();
    let mut wx = Array2::ones((h, w));
    let mut wy = Array2::ones((h, w));
    let two_sigma_sq = 2.0 * sigma_intensity * sigma_intensity;

    for row in 0..h {
        for col in 0..w {
            if col + 1 < w {
                let diff = guide[[row, col + 1]] - guide[[row, col]];
                wx[[row, col]] = (-diff * diff / two_sigma_sq).exp();
            }
            if row + 1 < h {
                let diff = guide[[row + 1, col]] - guide[[row, col]];
                wy[[row, col]] = (-diff * diff / two_sigma_sq).exp();
            }
        }
    }
    (wx, wy)
}

// ─────────────────────────────────────────────────────────────────────────────
// TV completion (Chambolle-Pock)
// ─────────────────────────────────────────────────────────────────────────────

/// Total-variation regularised depth completion using the Chambolle-Pock
/// primal-dual algorithm.
///
/// The optimisation problem is:
///
/// ```text
/// min  sum_{observed} (D_i - d_i)^2  +  lambda * TV(D)
/// ```
///
/// where the TV term is optionally weighted by an RGB guide (anisotropic).
///
/// # Arguments
/// * `sparse`    - sparse depth measurements
/// * `rgb_guide` - optional H x W intensity guide for anisotropic TV
/// * `config`    - algorithm parameters (`tv_lambda`, `max_iterations`,
///   `convergence_tol`, `sigma_intensity`)
///
/// # Errors
/// Returns an error if `sparse` has no measurements.
pub fn tv_completion(
    sparse: &SparseDepthMap,
    rgb_guide: Option<&Array2<f64>>,
    config: &DepthCompletionConfig,
) -> Result<CompletionResult> {
    sparse.validate_non_empty()?;

    let h = sparse.height;
    let w = sparse.width;
    let lambda = config.tv_lambda;

    // Edge weights (anisotropic if guide provided).
    let (wx, wy) = if let Some(guide) = rgb_guide {
        compute_edge_weights(guide, config.sigma_intensity)
    } else {
        (Array2::ones((h, w)), Array2::ones((h, w)))
    };

    // Observation data.
    let mask = sparse.observation_mask();
    let observed = sparse.to_dense();

    // Step sizes for Chambolle-Pock.
    // For the gradient operator with norm <= sqrt(8), choose tau*sigma < 1/8.
    let tau = 0.25;
    let sigma = 0.25;

    // Primal variable D (initialised from sparse data + interpolation).
    let mut d_curr = observed.clone();
    let mut d_bar = d_curr.clone();

    // Dual variables (px, py).
    let mut px = Array2::zeros((h, w));
    let mut py = Array2::zeros((h, w));

    let mut iterations = 0;
    let mut prev_energy = f64::MAX;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // ── Dual update ─────────────────────────────────────────────────
        // P^{k+1} = proj_{||.|| <= 1}( P^k + sigma * grad(D_bar) * W )
        let (gx, gy) = compute_gradient(&d_bar);

        for row in 0..h {
            for col in 0..w {
                let new_px = px[[row, col]] + sigma * gx[[row, col]] * wx[[row, col]];
                let new_py = py[[row, col]] + sigma * gy[[row, col]] * wy[[row, col]];

                // Project onto unit ball: max(1, ||(px, py)||)
                let sq_sum: f64 = new_px * new_px + new_py * new_py;
                let norm = sq_sum.sqrt().max(1.0);
                px[[row, col]] = new_px / norm;
                py[[row, col]] = new_py / norm;
            }
        }

        // ── Primal update ───────────────────────────────────────────────
        // D^{k+1} = prox_f( D^k - tau * div(P^{k+1}) )
        //
        // prox_f for data fidelity: for observed pixels, solve
        //   D_new = (D_old + 2*tau*d_obs) / (1 + 2*tau)
        // For unobserved pixels: D_new = D_old (identity).
        let div = compute_divergence(&px, &py);
        let d_prev = d_curr.clone();

        for row in 0..h {
            for col in 0..w {
                let tmp = d_prev[[row, col]] + tau * lambda * div[[row, col]];

                d_curr[[row, col]] = if mask[[row, col]] {
                    (tmp + 2.0 * tau * observed[[row, col]]) / (1.0 + 2.0 * tau)
                } else {
                    tmp
                };

                // Over-relaxation: d_bar = 2*d_new - d_old
                d_bar[[row, col]] = 2.0 * d_curr[[row, col]] - d_prev[[row, col]];
            }
        }

        // ── Convergence check ───────────────────────────────────────────
        // Monitor the primal energy as a proxy for convergence.
        let energy = compute_primal_energy(&d_curr, &observed, &mask, lambda, &wx, &wy);
        let rel_change = if prev_energy.abs() > 1e-12 {
            (prev_energy - energy).abs() / prev_energy.abs()
        } else {
            (prev_energy - energy).abs()
        };
        prev_energy = energy;

        if rel_change < config.convergence_tol && iter > 0 {
            break;
        }
    }

    // Build confidence: 1.0 at observed pixels, decaying elsewhere.
    let mut conf = Array2::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            conf[[row, col]] = if mask[[row, col]] { 1.0 } else { 0.5 };
        }
    }

    Ok(CompletionResult {
        dense_depth: d_curr,
        confidence_map: conf,
        method_used: CompletionMethod::TotalVariation,
        iterations,
    })
}

/// Compute the primal energy: data fidelity + weighted TV.
fn compute_primal_energy(
    d: &Array2<f64>,
    observed: &Array2<f64>,
    mask: &Array2<bool>,
    lambda: f64,
    wx: &Array2<f64>,
    wy: &Array2<f64>,
) -> f64 {
    let (h, w) = d.dim();
    let mut data_term = 0.0;
    let mut tv_term = 0.0;

    for row in 0..h {
        for col in 0..w {
            if mask[[row, col]] {
                let diff = d[[row, col]] - observed[[row, col]];
                data_term += diff * diff;
            }
            if col + 1 < w {
                tv_term += wx[[row, col]] * (d[[row, col + 1]] - d[[row, col]]).abs();
            }
            if row + 1 < h {
                tv_term += wy[[row, col]] * (d[[row + 1, col]] - d[[row, col]]).abs();
            }
        }
    }
    data_term + lambda * tv_term
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth_completion::types::SparseMeasurement;

    #[test]
    fn tv_constant_input_stays_constant() {
        // All observed pixels have depth 5.0 -> result should be 5.0 everywhere.
        let h = 6;
        let w = 6;
        let measurements: Vec<SparseMeasurement> = (0..h)
            .flat_map(|r| {
                (0..w).map(move |c| SparseMeasurement {
                    row: r,
                    col: c,
                    depth: 5.0,
                    confidence: 1.0,
                })
            })
            .collect();
        let sparse = SparseDepthMap::new(h, w, measurements).expect("valid");
        let config = DepthCompletionConfig {
            tv_lambda: 0.1,
            max_iterations: 50,
            convergence_tol: 1e-6,
            ..Default::default()
        };
        let result = tv_completion(&sparse, None, &config).expect("ok");
        for row in 0..h {
            for col in 0..w {
                assert!(
                    (result.dense_depth[[row, col]] - 5.0).abs() < 0.1,
                    "pixel ({row},{col}) = {} should be ~5.0",
                    result.dense_depth[[row, col]]
                );
            }
        }
    }

    #[test]
    fn tv_smooths_noise_in_flat_region() {
        let h = 8;
        let w = 8;
        // A few noisy measurements in a mostly-flat region.
        let measurements = vec![
            SparseMeasurement {
                row: 0,
                col: 0,
                depth: 5.0,
                confidence: 1.0,
            },
            SparseMeasurement {
                row: 0,
                col: 7,
                depth: 5.0,
                confidence: 1.0,
            },
            SparseMeasurement {
                row: 7,
                col: 0,
                depth: 5.0,
                confidence: 1.0,
            },
            SparseMeasurement {
                row: 7,
                col: 7,
                depth: 5.0,
                confidence: 1.0,
            },
            // Noisy outlier
            SparseMeasurement {
                row: 4,
                col: 4,
                depth: 5.5,
                confidence: 1.0,
            },
        ];

        let sparse = SparseDepthMap::new(h, w, measurements).expect("valid");
        let config = DepthCompletionConfig {
            tv_lambda: 1.0,
            max_iterations: 200,
            convergence_tol: 1e-6,
            ..Default::default()
        };
        let result = tv_completion(&sparse, None, &config).expect("ok");
        // The noisy pixel should be pulled closer to 5.0 by the TV term.
        let noisy_val = result.dense_depth[[4, 4]];
        assert!(
            (noisy_val - 5.0).abs() < 0.4,
            "noisy pixel should be smoothed toward 5.0, got {noisy_val}"
        );
    }

    #[test]
    fn tv_preserves_discontinuity_with_rgb_edge() {
        let h = 10;
        let w = 10;
        // Guide: left half=0, right half=1 (sharp edge at col=5).
        let mut guide = Array2::zeros((h, w));
        for row in 0..h {
            for col in 5..w {
                guide[[row, col]] = 1.0;
            }
        }

        // Place multiple measurements on each side for strong data fidelity.
        let mut measurements = Vec::new();
        for r in [2, 5, 8] {
            for c in [1, 2, 3] {
                measurements.push(SparseMeasurement {
                    row: r,
                    col: c,
                    depth: 2.0,
                    confidence: 1.0,
                });
            }
            for c in [6, 7, 8] {
                measurements.push(SparseMeasurement {
                    row: r,
                    col: c,
                    depth: 8.0,
                    confidence: 1.0,
                });
            }
        }

        let sparse = SparseDepthMap::new(h, w, measurements).expect("valid");
        let config = DepthCompletionConfig {
            tv_lambda: 0.5,
            max_iterations: 300,
            convergence_tol: 1e-7,
            sigma_intensity: 0.05,
            ..Default::default()
        };
        let result = tv_completion(&sparse, Some(&guide), &config).expect("ok");

        // Left of edge should be closer to 2.0, right should be closer to 8.0.
        let left_val = result.dense_depth[[5, 3]];
        let right_val = result.dense_depth[[5, 6]];
        assert!(
            left_val < 5.0,
            "left of edge should be < 5.0, got {left_val}"
        );
        assert!(
            right_val > 5.0,
            "right of edge should be > 5.0, got {right_val}"
        );
    }

    #[test]
    fn chambolle_pock_energy_decreases() {
        // Use a denser set of observations to give the algorithm a clear
        // starting point so that energy monotonically decreases.
        let h = 8;
        let w = 8;
        let mut measurements = Vec::new();
        // Border pixels all at 5.0, centre pixel at 8.0.
        for r in 0..h {
            for c in 0..w {
                if r == 0 || r == h - 1 || c == 0 || c == w - 1 {
                    measurements.push(SparseMeasurement {
                        row: r,
                        col: c,
                        depth: 5.0,
                        confidence: 1.0,
                    });
                }
            }
        }
        measurements.push(SparseMeasurement {
            row: 4,
            col: 4,
            depth: 8.0,
            confidence: 1.0,
        });
        let sparse = SparseDepthMap::new(h, w, measurements).expect("valid");

        let config_few = DepthCompletionConfig {
            tv_lambda: 0.1,
            max_iterations: 10,
            convergence_tol: 0.0,
            ..Default::default()
        };
        let config_many = DepthCompletionConfig {
            tv_lambda: 0.1,
            max_iterations: 200,
            convergence_tol: 0.0,
            ..Default::default()
        };

        let result_few = tv_completion(&sparse, None, &config_few).expect("ok");
        let result_many = tv_completion(&sparse, None, &config_many).expect("ok");

        let ones: Array2<f64> = Array2::ones((h, w));
        let mask = sparse.observation_mask();
        let obs = sparse.to_dense();

        let e_few = compute_primal_energy(&result_few.dense_depth, &obs, &mask, 0.1, &ones, &ones);
        let e_many =
            compute_primal_energy(&result_many.dense_depth, &obs, &mask, 0.1, &ones, &ones);

        assert!(
            e_many <= e_few + 1e-3,
            "more iterations should not increase energy: {e_many} > {e_few}"
        );
    }

    #[test]
    fn tv_lambda_zero_is_data_fidelity_only() {
        let h = 6;
        let w = 6;
        let measurements = vec![
            SparseMeasurement {
                row: 2,
                col: 2,
                depth: 3.0,
                confidence: 1.0,
            },
            SparseMeasurement {
                row: 4,
                col: 4,
                depth: 7.0,
                confidence: 1.0,
            },
        ];
        let sparse = SparseDepthMap::new(h, w, measurements).expect("valid");
        let config = DepthCompletionConfig {
            tv_lambda: 0.0,
            max_iterations: 100,
            convergence_tol: 1e-6,
            ..Default::default()
        };
        let result = tv_completion(&sparse, None, &config).expect("ok");
        // Observed pixels should remain at their original values.
        assert!(
            (result.dense_depth[[2, 2]] - 3.0).abs() < 0.01,
            "observed pixel should stay at 3.0"
        );
        assert!(
            (result.dense_depth[[4, 4]] - 7.0).abs() < 0.01,
            "observed pixel should stay at 7.0"
        );
    }

    #[test]
    fn tv_empty_sparse_errors() {
        let sparse = SparseDepthMap::new(5, 5, vec![]).expect("valid empty");
        let config = DepthCompletionConfig::default();
        assert!(tv_completion(&sparse, None, &config).is_err());
    }
}
