//! Kulldorff's spatial scan statistic for cluster detection
//!
//! Evaluates circular scan windows centred at each data point to find
//! the most likely spatial cluster. Supports Bernoulli (case/control)
//! and Poisson (counts with population at risk) models.
//! Monte Carlo significance testing provides p-values.

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::random::{seeded_rng, Rng, RngExt};

use crate::error::{SpatialError, SpatialResult};

/// Statistical model for the scan statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanModel {
    /// Bernoulli model: binary case/control data.
    Bernoulli,
    /// Poisson model: counts with known population (expected counts).
    Poisson,
}

/// Configuration for Kulldorff's scan statistic.
#[derive(Debug, Clone)]
pub struct ScanStatisticConfig {
    /// Statistical model.
    pub model: ScanModel,
    /// Maximum fraction of total population inside a scan window (0, 1].
    /// Default 0.5.
    pub max_population_fraction: f64,
    /// Number of Monte Carlo replications for p-value. Default 999.
    pub n_monte_carlo: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Maximum number of secondary (non-overlapping) clusters to report.
    /// Default 5.
    pub max_secondary_clusters: usize,
}

impl Default for ScanStatisticConfig {
    fn default() -> Self {
        Self {
            model: ScanModel::Poisson,
            max_population_fraction: 0.5,
            n_monte_carlo: 999,
            seed: 12345,
            max_secondary_clusters: 5,
        }
    }
}

/// A detected spatial cluster.
#[derive(Debug, Clone)]
pub struct ScanCluster {
    /// Index of the centre point in the input data.
    pub center_index: usize,
    /// Radius of the cluster window.
    pub radius: f64,
    /// Log-likelihood ratio of the cluster.
    pub llr: f64,
    /// Monte Carlo p-value.
    pub p_value: f64,
    /// Indices of all points inside the cluster.
    pub member_indices: Vec<usize>,
    /// Observed count/cases inside the cluster.
    pub observed_inside: f64,
    /// Expected count/cases inside the cluster.
    pub expected_inside: f64,
}

/// Result of a spatial scan analysis.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Most likely cluster (highest LLR).
    pub primary_cluster: ScanCluster,
    /// Secondary non-overlapping clusters, ordered by LLR descending.
    pub secondary_clusters: Vec<ScanCluster>,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run Kulldorff's spatial scan statistic.
///
/// # Arguments
///
/// * `coordinates` - (n x 2) array of spatial coordinates.
/// * `cases` - Observed counts (or 0/1 for Bernoulli) per location, length n.
/// * `population` - Population at risk (or total trials for Bernoulli) per location, length n.
///   For Bernoulli this is the number of trials at each location. For Poisson
///   this is used to compute expected counts.
/// * `config` - Scan configuration.
///
/// # Returns
///
/// `ScanResult` with primary and secondary clusters.
pub fn kulldorff_scan(
    coordinates: &ArrayView2<f64>,
    cases: &[f64],
    population: &[f64],
    config: &ScanStatisticConfig,
) -> SpatialResult<ScanResult> {
    let n = coordinates.nrows();
    if n < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 locations".to_string(),
        ));
    }
    if coordinates.ncols() < 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D".to_string(),
        ));
    }
    if cases.len() != n || population.len() != n {
        return Err(SpatialError::DimensionError(
            "cases and population must have length n".to_string(),
        ));
    }
    if config.max_population_fraction <= 0.0 || config.max_population_fraction > 1.0 {
        return Err(SpatialError::ValueError(
            "max_population_fraction must be in (0, 1]".to_string(),
        ));
    }

    let total_cases: f64 = cases.iter().sum();
    let total_population: f64 = population.iter().sum();

    if total_cases <= 0.0 || total_population <= 0.0 {
        return Err(SpatialError::ValueError(
            "Total cases and population must be positive".to_string(),
        ));
    }

    let max_pop = config.max_population_fraction * total_population;

    // Precompute pairwise distances
    let distances = precompute_distances(coordinates, n);

    // For each centroid, compute sorted neighbour list by distance
    let sorted_neighbours = build_sorted_neighbours(&distances, n);

    // Evaluate all candidate windows and find the maximum LLR
    let (best_llr, best_center, best_radius, best_members, best_obs, best_exp) = find_best_window(
        &sorted_neighbours,
        &distances,
        cases,
        population,
        total_cases,
        total_population,
        max_pop,
        config.model,
        n,
    );

    // Monte Carlo significance test
    let mc_p = monte_carlo_p_value(
        &sorted_neighbours,
        &distances,
        population,
        total_cases,
        total_population,
        max_pop,
        config.model,
        best_llr,
        config.n_monte_carlo,
        config.seed,
        n,
    );

    let primary = ScanCluster {
        center_index: best_center,
        radius: best_radius,
        llr: best_llr,
        p_value: mc_p,
        member_indices: best_members.clone(),
        observed_inside: best_obs,
        expected_inside: best_exp,
    };

    // Find secondary clusters (non-overlapping with primary)
    let secondary = find_secondary_clusters(
        &sorted_neighbours,
        &distances,
        cases,
        population,
        total_cases,
        total_population,
        max_pop,
        config.model,
        &best_members,
        config.max_secondary_clusters,
        n,
    );

    Ok(ScanResult {
        primary_cluster: primary,
        secondary_clusters: secondary,
    })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn precompute_distances(coordinates: &ArrayView2<f64>, n: usize) -> Vec<Vec<f64>> {
    let mut dists = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
            let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
            let d = (dx * dx + dy * dy).sqrt();
            dists[i][j] = d;
            dists[j][i] = d;
        }
    }
    dists
}

fn build_sorted_neighbours(distances: &[Vec<f64>], n: usize) -> Vec<Vec<usize>> {
    let mut sorted = Vec::with_capacity(n);
    for i in 0..n {
        let mut neighbours: Vec<usize> = (0..n).collect();
        neighbours.sort_by(|&a, &b| {
            distances[i][a]
                .partial_cmp(&distances[i][b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.push(neighbours);
    }
    sorted
}

/// Compute LLR for a candidate window.
fn compute_llr(
    obs_in: f64,
    exp_in: f64,
    total_cases: f64,
    total_expected: f64,
    model: ScanModel,
) -> f64 {
    let obs_out = total_cases - obs_in;
    let exp_out = total_expected - exp_in;

    match model {
        ScanModel::Poisson => {
            // LLR = obs_in * ln(obs_in/exp_in) + obs_out * ln(obs_out/exp_out)
            //       when obs_in/exp_in > obs_out/exp_out (excess risk inside)
            if exp_in <= 0.0 || exp_out <= 0.0 || obs_in <= 0.0 {
                return 0.0;
            }
            let rate_in = obs_in / exp_in;
            let rate_out = if obs_out > 0.0 && exp_out > 0.0 {
                obs_out / exp_out
            } else {
                0.0
            };

            // Only consider clusters with higher rate inside
            if rate_in <= rate_out {
                return 0.0;
            }

            let mut llr = obs_in * (obs_in / exp_in).ln();
            if obs_out > 0.0 && exp_out > 0.0 {
                llr += obs_out * (obs_out / exp_out).ln();
            }
            // Subtract baseline: total_cases * ln(total_cases / total_expected)
            if total_cases > 0.0 && total_expected > 0.0 {
                llr -= total_cases * (total_cases / total_expected).ln();
            }
            llr.max(0.0)
        }
        ScanModel::Bernoulli => {
            // Bernoulli LLR using case/population proportions
            if exp_in <= 0.0 || exp_out <= 0.0 {
                return 0.0;
            }

            let p_in = obs_in / exp_in;
            let p_out = if exp_out > 0.0 {
                obs_out / exp_out
            } else {
                0.0
            };

            if p_in <= p_out || p_in <= 0.0 || p_in >= 1.0 {
                return 0.0;
            }

            // Capped proportions for log safety
            let p_in_c = p_in.clamp(1e-15, 1.0 - 1e-15);
            let p_out_c = p_out.clamp(1e-15, 1.0 - 1e-15);
            let p_total = total_cases / total_expected;
            let p_total_c = p_total.clamp(1e-15, 1.0 - 1e-15);

            let llr = obs_in * (p_in_c / p_total_c).ln()
                + (exp_in - obs_in) * ((1.0 - p_in_c) / (1.0 - p_total_c)).ln()
                + obs_out * (p_out_c / p_total_c).ln()
                + (exp_out - obs_out) * ((1.0 - p_out_c) / (1.0 - p_total_c)).ln();

            llr.max(0.0)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn find_best_window(
    sorted_neighbours: &[Vec<usize>],
    distances: &[Vec<f64>],
    cases: &[f64],
    population: &[f64],
    total_cases: f64,
    total_population: f64,
    max_pop: f64,
    model: ScanModel,
    n: usize,
) -> (f64, usize, f64, Vec<usize>, f64, f64) {
    let mut best_llr = 0.0;
    let mut best_center = 0;
    let mut best_radius = 0.0;
    let mut best_members: Vec<usize> = Vec::new();
    let mut best_obs = 0.0;
    let mut best_exp = 0.0;

    for i in 0..n {
        let mut cum_cases = 0.0;
        let mut cum_pop = 0.0;
        let mut members: Vec<usize> = Vec::new();

        for &j in &sorted_neighbours[i] {
            cum_cases += cases[j];
            cum_pop += population[j];
            members.push(j);

            if cum_pop > max_pop {
                break;
            }

            let exp_in = match model {
                ScanModel::Poisson => cum_pop * total_cases / total_population,
                ScanModel::Bernoulli => cum_pop,
            };

            let total_expected = match model {
                ScanModel::Poisson => total_population * total_cases / total_population,
                ScanModel::Bernoulli => total_population,
            };

            let llr = compute_llr(cum_cases, exp_in, total_cases, total_expected, model);

            if llr > best_llr {
                best_llr = llr;
                best_center = i;
                best_radius = distances[i][j];
                best_members = members.clone();
                best_obs = cum_cases;
                best_exp = exp_in;
            }
        }
    }

    (
        best_llr,
        best_center,
        best_radius,
        best_members,
        best_obs,
        best_exp,
    )
}

#[allow(clippy::too_many_arguments)]
fn monte_carlo_p_value(
    sorted_neighbours: &[Vec<usize>],
    distances: &[Vec<f64>],
    population: &[f64],
    total_cases: f64,
    total_population: f64,
    max_pop: f64,
    model: ScanModel,
    observed_llr: f64,
    n_simulations: usize,
    seed: u64,
    n: usize,
) -> f64 {
    let mut rng = seeded_rng(seed);
    let mut count_ge = 0usize;

    for _sim in 0..n_simulations {
        // Generate random case allocation under H0
        let sim_cases = generate_null_cases(&mut rng, population, total_cases, model, n);

        // Find best LLR in the simulation
        let (sim_best_llr, _, _, _, _, _) = find_best_window(
            sorted_neighbours,
            distances,
            &sim_cases,
            population,
            total_cases,
            total_population,
            max_pop,
            model,
            n,
        );

        if sim_best_llr >= observed_llr {
            count_ge += 1;
        }
    }

    (count_ge as f64 + 1.0) / (n_simulations as f64 + 1.0)
}

fn generate_null_cases<R: Rng + ?Sized>(
    rng: &mut R,
    population: &[f64],
    total_cases: f64,
    model: ScanModel,
    n: usize,
) -> Vec<f64> {
    match model {
        ScanModel::Poisson => {
            // Under H0: cases are distributed proportional to population
            let total_pop: f64 = population.iter().sum();
            let mut sim_cases = vec![0.0; n];
            let mut remaining = total_cases as usize;

            // Multinomial allocation
            for i in 0..n {
                if remaining == 0 {
                    break;
                }
                let prob = population[i] / total_pop;
                // Simple binomial approximation for each location
                let expected = remaining as f64 * prob;
                // Clip to remaining
                let allocated = poisson_sample(rng, expected).min(remaining as f64);
                sim_cases[i] = allocated;
                remaining = remaining.saturating_sub(allocated as usize);
            }
            // Allocate any remaining to random locations
            while remaining > 0 {
                let idx = rng.random_range(0..n);
                sim_cases[idx] += 1.0;
                remaining -= 1;
            }
            sim_cases
        }
        ScanModel::Bernoulli => {
            // Under H0: constant probability p = total_cases / total_population
            let total_pop: f64 = population.iter().sum();
            let p = total_cases / total_pop;
            let mut sim_cases = vec![0.0; n];
            for i in 0..n {
                // Binomial(population[i], p) approximation
                let trials = population[i] as usize;
                let mut count = 0.0;
                for _ in 0..trials {
                    if rng.random::<f64>() < p {
                        count += 1.0;
                    }
                }
                sim_cases[i] = count;
            }
            sim_cases
        }
    }
}

/// Simple Poisson sample using the inverse transform method.
fn poisson_sample<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    let l = (-lambda).exp();
    let mut k: f64 = 0.0;
    let mut p: f64 = 1.0;
    loop {
        k += 1.0;
        let u: f64 = rng.random::<f64>();
        p *= u;
        if p < l {
            break;
        }
    }
    if k - 1.0 > 0.0 {
        k - 1.0
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn find_secondary_clusters(
    sorted_neighbours: &[Vec<usize>],
    distances: &[Vec<f64>],
    cases: &[f64],
    population: &[f64],
    total_cases: f64,
    total_population: f64,
    max_pop: f64,
    model: ScanModel,
    primary_members: &[usize],
    max_secondary: usize,
    n: usize,
) -> Vec<ScanCluster> {
    // Collect all candidate windows, excluding those overlapping with primary
    let mut candidates: Vec<(f64, usize, f64, Vec<usize>, f64, f64)> = Vec::new();

    for i in 0..n {
        // Skip if this centre is in the primary cluster
        if primary_members.contains(&i) {
            continue;
        }

        let mut cum_cases = 0.0;
        let mut cum_pop = 0.0;
        let mut members: Vec<usize> = Vec::new();

        for &j in &sorted_neighbours[i] {
            // Skip members of primary cluster
            if primary_members.contains(&j) {
                continue;
            }

            cum_cases += cases[j];
            cum_pop += population[j];
            members.push(j);

            if cum_pop > max_pop {
                break;
            }

            let exp_in = match model {
                ScanModel::Poisson => cum_pop * total_cases / total_population,
                ScanModel::Bernoulli => cum_pop,
            };

            let total_expected = match model {
                ScanModel::Poisson => total_cases,
                ScanModel::Bernoulli => total_population,
            };

            let llr = compute_llr(cum_cases, exp_in, total_cases, total_expected, model);

            if llr > 0.0 {
                candidates.push((llr, i, distances[i][j], members.clone(), cum_cases, exp_in));
            }
        }
    }

    // Sort by LLR descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top non-overlapping
    let mut result = Vec::new();
    let mut used_indices: Vec<usize> = primary_members.to_vec();

    for (llr, center, radius, members, obs, exp) in candidates {
        if result.len() >= max_secondary {
            break;
        }

        // Check overlap with already-used indices
        let overlaps = members.iter().any(|m| used_indices.contains(m));
        if overlaps {
            continue;
        }

        used_indices.extend_from_slice(&members);
        result.push(ScanCluster {
            center_index: center,
            radius,
            llr,
            p_value: 1.0, // secondary p-values not computed here
            member_indices: members,
            observed_inside: obs,
            expected_inside: exp,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_planted_poisson_cluster() {
        // 10 points on a grid; a planted cluster at top-left with elevated cases
        let coords = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0], // cluster area
            [5.0, 0.0],
            [6.0, 0.0],
            [5.0, 1.0],
            [6.0, 1.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ];

        // High case rate in first 4 locations
        let cases = [20.0, 18.0, 22.0, 19.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0];
        let population = [100.0; 10];

        let config = ScanStatisticConfig {
            model: ScanModel::Poisson,
            max_population_fraction: 0.5,
            n_monte_carlo: 99,
            seed: 42,
            max_secondary_clusters: 3,
        };

        let result =
            kulldorff_scan(&coords.view(), &cases, &population, &config).expect("scan failed");

        // Primary cluster should be centred in the high-rate area (indices 0-3)
        assert!(
            result.primary_cluster.llr > 0.0,
            "Primary cluster LLR should be positive"
        );
        assert!(
            result.primary_cluster.center_index < 4,
            "Primary cluster should be centred in the high-rate area, got index {}",
            result.primary_cluster.center_index
        );

        // P-value should be low for a strong cluster
        assert!(
            result.primary_cluster.p_value < 0.5,
            "p-value should be < 0.5, got {}",
            result.primary_cluster.p_value
        );
    }

    #[test]
    fn test_planted_bernoulli_cluster() {
        let coords = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [5.0, 5.0],
            [5.5, 5.0],
            [5.0, 5.5],
        ];

        // Cases / trials: high rate at first 3 locations
        let cases = [9.0, 8.0, 10.0, 1.0, 2.0, 1.0];
        let population = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0];

        let config = ScanStatisticConfig {
            model: ScanModel::Bernoulli,
            max_population_fraction: 0.5,
            n_monte_carlo: 99,
            seed: 123,
            max_secondary_clusters: 2,
        };

        let result =
            kulldorff_scan(&coords.view(), &cases, &population, &config).expect("bernoulli scan");

        assert!(result.primary_cluster.llr > 0.0);
        // Should detect cluster in first 3 locations
        let centre = result.primary_cluster.center_index;
        assert!(
            centre < 3,
            "Expected cluster centre in [0,3), got {}",
            centre
        );
    }

    #[test]
    fn test_no_cluster_uniform() {
        // Uniform case rates => LLR should be small and p-value high
        let coords = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ];

        let cases = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let population = [100.0; 6];

        let config = ScanStatisticConfig {
            model: ScanModel::Poisson,
            n_monte_carlo: 99,
            seed: 77,
            ..Default::default()
        };

        let result =
            kulldorff_scan(&coords.view(), &cases, &population, &config).expect("uniform scan");

        // p-value should be high (no real cluster)
        assert!(
            result.primary_cluster.p_value > 0.05,
            "p-value should be > 0.05 for uniform data, got {}",
            result.primary_cluster.p_value
        );
    }

    #[test]
    fn test_scan_errors() {
        let coords = array![[0.0, 0.0], [1.0, 0.0]]; // too few
        let cases = [1.0, 1.0];
        let population = [10.0, 10.0];
        let config = ScanStatisticConfig::default();
        assert!(kulldorff_scan(&coords.view(), &cases, &population, &config).is_err());
    }

    #[test]
    fn test_scan_secondary_clusters() {
        // Two distinct clusters far apart
        let coords = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [5.0, 5.0], // low-rate background
            [5.1, 5.0],
            [5.0, 5.1],
        ];

        let cases = [15.0, 14.0, 16.0, 12.0, 13.0, 14.0, 1.0, 1.0, 1.0];
        let population = [20.0; 9];

        let config = ScanStatisticConfig {
            model: ScanModel::Poisson,
            n_monte_carlo: 49,
            seed: 999,
            max_secondary_clusters: 3,
            ..Default::default()
        };

        let result =
            kulldorff_scan(&coords.view(), &cases, &population, &config).expect("secondary scan");

        // Should find primary + potentially a secondary cluster
        assert!(result.primary_cluster.llr > 0.0);
        // Secondary clusters may or may not be found depending on
        // overlap constraints, but the API should work without errors
    }
}
