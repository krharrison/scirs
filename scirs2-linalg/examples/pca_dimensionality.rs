//! PCA Dimensionality Reduction Example — SciRS2 Linalg
//!
//! Demonstrates principal component analysis on synthetic high-dimensional data:
//!   1. Generate 200 × 20 data with known low-rank structure (3 latent factors)
//!   2. Run `randomized_pca` to reduce to k components
//!   3. Print explained variance ratio and cumulative curve
//!   4. Project data and reconstruct; compute reconstruction error
//!
//! Run with: cargo run -p scirs2-linalg --example pca_dimensionality

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::{pca_inverse_transform, pca_transform, randomized_pca, RandomizedConfig};

// ------------------------------------------------------------------ //
//  Synthetic dataset with low-rank structure                          //
// ------------------------------------------------------------------ //

/// Generate `n_samples × n_features` data:
///   X = Z · L^T + noise
/// where Z is n_samples × n_factors (latent scores) and L is n_features × n_factors
/// (loadings).  The result has approximately n_factors dominant directions.
fn generate_low_rank_data(
    n_samples: usize,
    n_features: usize,
    n_factors: usize,
    noise_std: f64,
) -> Array2<f64> {
    // Deterministic data using simple arithmetic patterns
    // Latent scores Z: n_samples × n_factors
    let z: Array2<f64> = Array2::from_shape_fn((n_samples, n_factors), |(i, j)| {
        let t = i as f64 / n_samples as f64;
        match j {
            0 => (2.0 * std::f64::consts::PI * t).sin() * 3.0,
            1 => (4.0 * std::f64::consts::PI * t).cos() * 2.0,
            _ => t * 1.5 - 0.75,
        }
    });

    // Loadings L: n_features × n_factors
    let loadings: Array2<f64> = Array2::from_shape_fn((n_features, n_factors), |(i, j)| {
        let fi = i as f64 / n_features as f64;
        match j {
            0 => fi.sin() + 0.5,
            1 => (fi * 2.0).cos() - 0.3,
            _ => fi * fi - 0.5,
        }
    });

    // X_signal = Z @ L^T
    let x_signal = z.dot(&loadings.t());

    // Add noise (LCG-based, deterministic)
    let mut lcg: u64 = 0xA1B2_C3D4_E5F6_7890;
    let noise: Array2<f64> = Array2::from_shape_fn((n_samples, n_features), |_| {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = (lcg >> 33) as f64 / u32::MAX as f64 + 1e-15;
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = (lcg >> 33) as f64 / u32::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * noise_std
    });

    x_signal + noise
}

// ------------------------------------------------------------------ //
//  Reconstruction error                                               //
// ------------------------------------------------------------------ //

fn frobenius_error(original: &Array2<f64>, reconstructed: &Array2<f64>) -> f64 {
    let diff = original - reconstructed;
    let sq_sum: f64 = diff.iter().map(|&v| v * v).sum();
    (sq_sum / original.len() as f64).sqrt() // RMSE per element
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    let sq_sum: f64 = a.iter().map(|&v| v * v).sum();
    sq_sum.sqrt()
}

// ------------------------------------------------------------------ //
//  main                                                               //
// ------------------------------------------------------------------ //

fn main() {
    const N_SAMPLES: usize = 200;
    const N_FEATURES: usize = 20;
    const N_FACTORS: usize = 3; // ground-truth latent dimensionality
    const NOISE_STD: f64 = 0.3;
    const N_COMPONENTS: usize = 10; // components to compute

    println!("=== SciRS2 PCA Dimensionality Reduction ===\n");
    println!("Data: {N_SAMPLES} samples × {N_FEATURES} features");
    println!("Latent factors: {N_FACTORS} (with noise std = {NOISE_STD})\n");

    let data = generate_low_rank_data(N_SAMPLES, N_FEATURES, N_FACTORS, NOISE_STD);

    // ------------------------------------------------------------------ //
    //  PCA fit                                                            //
    // ------------------------------------------------------------------ //
    let pca_result =
        randomized_pca(&data.view(), N_COMPONENTS, false, Some(4)).expect("Randomized PCA failed");

    // ------------------------------------------------------------------ //
    //  Explained variance table                                           //
    // ------------------------------------------------------------------ //
    println!("--- Explained Variance ---");
    println!(
        "{:<8} {:>14} {:>14} {:>14}",
        "PC", "Variance", "Ratio", "Cumulative"
    );
    println!("{}", "-".repeat(54));

    let mut cumulative = 0.0_f64;
    for k in 0..N_COMPONENTS {
        let ratio = pca_result.explained_variance_ratio[k];
        let var = pca_result.explained_variance[k];
        cumulative += ratio;
        let bar_len = (ratio * 40.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        println!(
            "PC{:<5}  {:>10.4}  {:>10.2}%  {:>10.2}%  {}",
            k + 1,
            var,
            ratio * 100.0,
            cumulative * 100.0,
            bar
        );
    }

    // Find the number of components needed to explain 90% variance
    let mut cum = 0.0_f64;
    let n_90pct = pca_result
        .explained_variance_ratio
        .iter()
        .position(|&r| {
            cum += r;
            cum >= 0.90
        })
        .map(|p| p + 1)
        .unwrap_or(N_COMPONENTS);

    println!("\nComponents to explain 90% variance: {n_90pct}");
    println!(
        "First 3 components: {:.2}% of total variance",
        pca_result
            .explained_variance_ratio
            .slice(scirs2_core::ndarray::s![..3])
            .sum()
            * 100.0
    );

    // ------------------------------------------------------------------ //
    //  Singular values                                                     //
    // ------------------------------------------------------------------ //
    println!("\n--- Singular Values (top {N_COMPONENTS}) ---");
    print!("[");
    for (i, sv) in pca_result.singular_values.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.3}", sv);
    }
    println!("]");

    // ------------------------------------------------------------------ //
    //  Projection and reconstruction error per k                         //
    // ------------------------------------------------------------------ //
    println!("\n--- Reconstruction Error vs. Components ---");
    println!(
        "{:<12} {:>16} {:>16}",
        "Components", "RMSE", "Relative Error"
    );
    println!("{}", "-".repeat(48));

    let data_norm = frobenius_norm(&data);
    for k in [1usize, 2, 3, 5, 7, 10] {
        if k > N_COMPONENTS {
            continue;
        }
        // Fit PCA with k components
        let pca_k = randomized_pca(&data.view(), k, false, Some(4)).expect("PCA failed");
        // Project to k-dim
        let projected = pca_transform(&data.view(), &pca_k).expect("PCA transform failed");
        // Reconstruct back to n_features
        let reconstructed =
            pca_inverse_transform(&projected.view(), &pca_k).expect("PCA inverse transform failed");

        let rmse = frobenius_error(&data, &reconstructed);
        let rel_err = frobenius_norm(&(&data - &reconstructed)) / data_norm;
        let cum_var: f64 = pca_k.explained_variance_ratio.iter().sum();
        println!(
            "{:<12} {:>16.6} {:>14.2}%  (cum. var {:.1}%)",
            k,
            rmse,
            rel_err * 100.0,
            cum_var * 100.0
        );
    }

    // ------------------------------------------------------------------ //
    //  Low-dimensional representation (first 3 PCs for first 5 samples)  //
    // ------------------------------------------------------------------ //
    let pca_3 = randomized_pca(&data.view(), 3, false, Some(4)).expect("PCA failed");
    let projected_3 = pca_transform(&data.view(), &pca_3).expect("PCA transform failed");

    println!("\n--- First 5 Samples in 3-D PCA Space ---");
    println!("{:<8} {:>10} {:>10} {:>10}", "Sample", "PC1", "PC2", "PC3");
    println!("{}", "-".repeat(42));
    for i in 0..5usize {
        println!(
            "{:<8} {:>10.4} {:>10.4} {:>10.4}",
            i + 1,
            projected_3[[i, 0]],
            projected_3[[i, 1]],
            projected_3[[i, 2]]
        );
    }

    // ------------------------------------------------------------------ //
    //  Mean vector                                                        //
    // ------------------------------------------------------------------ //
    println!("\n--- Feature Means (first 5 features) ---");
    print!("[");
    for i in 0..5usize {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", pca_3.mean[i]);
    }
    println!(", ...]");

    println!("\nDone.");
}
