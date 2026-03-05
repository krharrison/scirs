// Test fixtures and sample data for integration tests

use scirs2_core::ndarray::{Array1, Array2};

/// Standard test datasets
pub struct TestDatasets;

impl TestDatasets {
    /// Small XOR dataset for quick testing
    pub fn xor_dataset() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ]).expect("Failed to create XOR input");

        let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        (x, y)
    }

    /// Simple linear dataset for regression testing
    pub fn linear_dataset(n_samples: usize) -> (Array2<f64>, Array1<f64>) {
        let x: Vec<f64> = (0..n_samples)
            .map(|i| i as f64 / n_samples as f64)
            .collect();

        let y: Vec<f64> = x.iter()
            .map(|&xi| 2.0 * xi + 1.0)
            .collect();

        let x_2d = Array2::from_shape_vec((n_samples, 1), x)
            .expect("Failed to create linear input");
        let y_1d = Array1::from_vec(y);

        (x_2d, y_1d)
    }

    /// Sinusoidal signal for FFT/signal processing tests
    pub fn sinusoid_signal(n_samples: usize, frequency: f64, sampling_rate: f64) -> Array1<f64> {
        use std::f64::consts::PI;

        let dt = 1.0 / sampling_rate;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * frequency * t).sin()
            })
            .collect();

        Array1::from_vec(signal)
    }

    /// Random sparse matrix for sparse algebra tests
    pub fn sparse_test_matrix(rows: usize, cols: usize, density: f64) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::new();
        let n_nonzero = ((rows * cols) as f64 * density) as usize;

        for k in 0..n_nonzero {
            let i = (k * 7) % rows;  // Deterministic "random"
            let j = (k * 11) % cols;
            let val = (k as f64 + 1.0) / n_nonzero as f64;
            triplets.push((i, j, val));
        }

        triplets
    }

    /// Test image for ndimage/vision tests
    pub fn test_image_gradient(size: usize) -> Array2<f64> {
        let mut img = Array2::zeros((size, size));

        for i in 0..size {
            for j in 0..size {
                img[[i, j]] = (i + j) as f64 / (2.0 * size as f64);
            }
        }

        img
    }

    /// Statistical test data - normal distribution samples
    pub fn normal_samples(n_samples: usize, mean: f64, std: f64) -> Array1<f64> {
        // Using Box-Muller transform for deterministic "random" normal samples
        let mut samples = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let u1 = (i as f64 + 1.0) / (n_samples as f64 + 1.0);
            let u2 = ((i * 7) as f64 + 1.0) / (n_samples as f64 + 1.0);

            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            samples.push(mean + std * z);
        }

        Array1::from_vec(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_dataset() {
        let (x, y) = TestDatasets::xor_dataset();
        assert_eq!(x.shape(), &[4, 2]);
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn test_linear_dataset() {
        let (x, y) = TestDatasets::linear_dataset(100);
        assert_eq!(x.shape(), &[100, 1]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_sinusoid_signal() {
        let signal = TestDatasets::sinusoid_signal(1000, 10.0, 1000.0);
        assert_eq!(signal.len(), 1000);
    }

    #[test]
    fn test_sparse_matrix() {
        let triplets = TestDatasets::sparse_test_matrix(100, 100, 0.1);
        assert!(triplets.len() > 0);
        assert!(triplets.len() <= 1000); // ~10% of 10000
    }

    #[test]
    fn test_image_gradient() {
        let img = TestDatasets::test_image_gradient(64);
        assert_eq!(img.shape(), &[64, 64]);
    }

    #[test]
    fn test_normal_samples() {
        let samples = TestDatasets::normal_samples(1000, 0.0, 1.0);
        assert_eq!(samples.len(), 1000);
    }
}
