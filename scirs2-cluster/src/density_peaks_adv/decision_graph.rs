//! Decision graph analysis for density peaks clustering.
//!
//! The decision graph is a scatter plot of (rho, delta) values. Points that
//! appear as outliers with high both rho and delta are natural cluster center
//! candidates (Rodriguez & Laio 2014, Figure 1).

/// Decision graph for selecting density peaks cluster centers.
///
/// Provides utilities to compute gamma = rho * delta and identify the
/// optimal number of cluster centers via the elbow/knee method on the
/// sorted gamma profile.
#[derive(Debug, Clone)]
pub struct DecisionGraph {
    /// Local density for each data point.
    pub rho: Vec<f64>,
    /// Distance to nearest higher-density neighbor for each data point.
    pub delta: Vec<f64>,
}

impl DecisionGraph {
    /// Create a new `DecisionGraph` from precomputed rho and delta vectors.
    ///
    /// # Panics
    ///
    /// Does not panic; mismatched lengths are silently accepted (gamma() will
    /// zip the shorter of the two).
    pub fn new(rho: Vec<f64>, delta: Vec<f64>) -> Self {
        Self { rho, delta }
    }

    /// Compute gamma = rho * delta for each point.
    ///
    /// Points with high gamma are strong cluster center candidates.
    pub fn gamma(&self) -> Vec<f64> {
        self.rho
            .iter()
            .zip(self.delta.iter())
            .map(|(r, d)| r * d)
            .collect()
    }

    /// Estimate the optimal number of cluster centers from the gamma profile.
    ///
    /// Sorts gamma values in descending order and identifies the position of the
    /// largest drop (elbow). This position + 1 is returned as the recommended
    /// center count.
    ///
    /// Returns 1 if there are fewer than 3 data points.
    pub fn optimal_n_centers(&self) -> usize {
        let mut gamma = self.gamma();
        if gamma.len() < 3 {
            return 1;
        }

        // Sort descending.
        gamma.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Consecutive drops in sorted gamma.
        let drops: Vec<f64> = gamma.windows(2).map(|w| w[0] - w[1]).collect();

        // Position of the largest drop.
        drops
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i + 1)
            .unwrap_or(2)
    }

    /// Return the indices of the top-k points by gamma value.
    pub fn top_k_centers(&self, k: usize) -> Vec<usize> {
        let gamma = self.gamma();
        let mut indexed: Vec<(f64, usize)> = gamma.iter().copied().enumerate().map(|(i, g)| (g, i)).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(_, i)| *i).collect()
    }

    /// Compute normalized gamma scores (0..1) for visualization.
    pub fn normalized_gamma(&self) -> Vec<f64> {
        let gamma = self.gamma();
        let max_g = gamma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_g <= 0.0 {
            return vec![0.0; gamma.len()];
        }
        gamma.iter().map(|&g| g / max_g).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_computation() {
        let rho = vec![1.0, 2.0, 3.0];
        let delta = vec![4.0, 5.0, 6.0];
        let dg = DecisionGraph::new(rho, delta);
        let gamma = dg.gamma();
        assert_eq!(gamma, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_optimal_n_centers_two_clusters() {
        // Simulate a dataset with 2 clear cluster centers: two points with high
        // gamma, rest with low gamma.
        let rho = vec![10.0, 9.0, 1.0, 1.0, 1.0, 1.0];
        let delta = vec![10.0, 9.0, 0.1, 0.1, 0.1, 0.1];
        let dg = DecisionGraph::new(rho, delta);
        let n = dg.optimal_n_centers();
        // Expect 2 centers to be identified (big drop after rank 2).
        assert_eq!(n, 2, "should identify 2 cluster centers");
    }

    #[test]
    fn test_optimal_n_centers_small() {
        let rho = vec![1.0, 2.0];
        let delta = vec![1.0, 2.0];
        let dg = DecisionGraph::new(rho, delta);
        assert_eq!(dg.optimal_n_centers(), 1);
    }

    #[test]
    fn test_top_k_centers() {
        let rho = vec![1.0, 5.0, 3.0, 2.0];
        let delta = vec![1.0, 5.0, 3.0, 2.0];
        let dg = DecisionGraph::new(rho, delta);
        let top2 = dg.top_k_centers(2);
        // gamma = [1, 25, 9, 4], so top-2 are index 1 (25) and index 2 (9).
        assert_eq!(top2[0], 1);
        assert_eq!(top2[1], 2);
    }

    #[test]
    fn test_normalized_gamma() {
        let rho = vec![1.0, 2.0, 0.0];
        let delta = vec![2.0, 3.0, 0.0];
        let dg = DecisionGraph::new(rho, delta);
        let ng = dg.normalized_gamma();
        // max gamma = 2*3 = 6
        assert!((ng[0] - 2.0 / 6.0).abs() < 1e-10);
        assert!((ng[1] - 1.0).abs() < 1e-10);
        assert!((ng[2] - 0.0).abs() < 1e-10);
    }
}
