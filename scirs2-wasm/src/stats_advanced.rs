//! Advanced statistical WASM API (v0.3.0)
//!
//! This module provides high-level statistical functions that accept raw `&[f64]`
//! slices (compatible with JS `Float64Array` passing via wasm-bindgen) and return
//! primitive values or `JsValue` JSON objects.  All functions follow the
//! `wasm_`-prefixed naming convention introduced in v0.3.0.
//!
//! ## Implemented functions
//!
//! - [`wasm_polynomial_fit`] — polynomial regression with R²
//! - [`wasm_spearman_correlation`] — Spearman rank correlation
//! - [`wasm_t_test_one_sample`] — one-sample t-test (wasm_ prefix alias)
//! - [`wasm_t_test_two_sample`] — two-sample Welch t-test (explicit name)
//! - [`wasm_anova_one_way`] — one-way ANOVA (F-statistic + p-value)
//! - [`wasm_pca`] — principal component analysis (scores, loadings, variance explained)
//! - [`wasm_kmeans`] — k-means clustering (labels, centroids, inertia)

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Polynomial regression
// ---------------------------------------------------------------------------

/// Fit a polynomial of degree `degree` to `(x, y)` data and return
/// coefficients together with R².
///
/// Solves the normal equations  `Vᵀ V c = Vᵀ y`  where `V` is the
/// Vandermonde design matrix.
///
/// # Returns
///
/// `JsValue` JSON:
/// ```json
/// { "coefficients": [c0, c1, ..., c_d], "r2": 0.999, "degree": 2 }
/// ```
/// `coefficients` are in ascending power order: `c0 + c1·x + c2·x² + …`
///
/// Returns `JsValue::NULL` on serialisation failure or a `JsValue` error
/// string for invalid inputs.
#[wasm_bindgen]
pub fn wasm_polynomial_fit(x: &[f64], y: &[f64], degree: usize) -> JsValue {
    let n = x.len();
    if n == 0 {
        return JsValue::from_str("Error: input arrays must not be empty");
    }
    if n != y.len() {
        return JsValue::from_str("Error: x and y must have the same length");
    }
    if degree == 0 {
        return JsValue::from_str("Error: degree must be at least 1");
    }
    if n <= degree {
        return JsValue::from_str(
            "Error: number of data points must exceed the polynomial degree",
        );
    }

    let d = degree + 1; // number of coefficients

    // Build Vandermonde matrix V (n × d), row-major
    let mut vt = vec![0.0_f64; d * n]; // V transposed: d × n
    for j in 0..n {
        let mut xpow = 1.0_f64;
        for i in 0..d {
            vt[i * n + j] = xpow;
            xpow *= x[j];
        }
    }

    // Normal equations: A = VᵀV (d×d), b = Vᵀy (d)
    let mut a = vec![0.0_f64; d * d];
    let mut rhs = vec![0.0_f64; d];

    for i in 0..d {
        for k in 0..n {
            rhs[i] += vt[i * n + k] * y[k];
        }
        for j in 0..d {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += vt[i * n + k] * vt[j * n + k];
            }
            a[i * d + j] = s;
        }
    }

    // Solve A·c = rhs via Gaussian elimination with partial pivoting
    let coeffs = match solve_system_f64(&a, &rhs, d) {
        Ok(c) => c,
        Err(msg) => return JsValue::from_str(&format!("Error: {}", msg)),
    };

    // Compute R²
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = (0..n)
        .map(|j| {
            let y_hat: f64 = (0..d)
                .map(|i| {
                    let xpow = x[j].powi(i as i32);
                    coeffs[i] * xpow
                })
                .sum();
            (y[j] - y_hat).powi(2)
        })
        .sum();
    let r2 = if ss_tot == 0.0 {
        1.0_f64
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    let result = serde_json::json!({
        "coefficients": coeffs,
        "r2": r2,
        "degree": degree,
    });

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// Spearman rank correlation
// ---------------------------------------------------------------------------

/// Compute the Spearman rank correlation coefficient between `x` and `y`.
///
/// Uses the exact rank-correlation formula:
/// ```text
/// rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
/// ```
/// where `d_i` is the difference in ranks for observation `i`.
/// Ties are handled by assigning the average rank.
///
/// # Returns
///
/// Spearman's rho in `[-1, 1]`.  Returns `NaN` for empty or constant arrays,
/// or when lengths differ.
#[wasm_bindgen]
pub fn wasm_spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || n != y.len() {
        return f64::NAN;
    }
    if n == 1 {
        return 1.0;
    }

    let rx = average_ranks(x);
    let ry = average_ranks(y);

    // Pearson correlation of ranks
    let rx_mean = rx.iter().sum::<f64>() / n as f64;
    let ry_mean = ry.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0_f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for i in 0..n {
        let dx = rx[i] - rx_mean;
        let dy = ry[i] - ry_mean;
        cov += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }

    if sx == 0.0 || sy == 0.0 {
        return f64::NAN;
    }
    cov / (sx * sy).sqrt()
}

// ---------------------------------------------------------------------------
// One-sample t-test (wasm_ prefix)
// ---------------------------------------------------------------------------

/// Perform a one-sample t-test of `data` against hypothesised mean `mu0`.
///
/// # Returns
///
/// `JsValue` JSON: `{ "statistic": f64, "p_value": f64, "degrees_of_freedom": f64 }`
///
/// Returns a `JsValue` error string if `data.len() < 2`.
#[wasm_bindgen]
pub fn wasm_t_test_one_sample(data: &[f64], mu0: f64) -> JsValue {
    // Delegate to the existing implementation in stats.rs
    crate::stats::t_test_one_sample(data, mu0)
}

// ---------------------------------------------------------------------------
// Two-sample t-test (explicit name)
// ---------------------------------------------------------------------------

/// Perform a two-sample Welch t-test between samples `a` and `b`.
///
/// # Returns
///
/// `JsValue` JSON: `{ "statistic": f64, "p_value": f64, "degrees_of_freedom": f64 }`
///
/// Returns a `JsValue` error string if either sample has fewer than 2 elements.
#[wasm_bindgen]
pub fn wasm_t_test_two_sample(a: &[f64], b: &[f64]) -> JsValue {
    // Delegate to the existing wasm_t_test implementation
    crate::stats::wasm_t_test(a, b)
}

// ---------------------------------------------------------------------------
// One-way ANOVA
// ---------------------------------------------------------------------------

/// Perform a one-way ANOVA across multiple groups.
///
/// # Arguments
///
/// * `groups_js` — a JS array of number arrays, e.g.
///   `[[1,2,3], [4,5,6], [7,8,9]]`.  Each inner array is one group.
///
/// # Returns
///
/// `JsValue` JSON:
/// ```json
/// { "f_statistic": f64, "p_value": f64, "df_between": usize, "df_within": usize }
/// ```
/// Returns `JsValue::NULL` on parse/serialisation error, or a `JsValue`
/// error string for insufficient data (fewer than 2 groups or any group
/// with fewer than 1 observation).
#[wasm_bindgen]
pub fn wasm_anova_one_way(groups_js: JsValue) -> JsValue {
    // Parse groups from JS value
    let raw: Vec<Vec<f64>> = match serde_wasm_bindgen::from_value(groups_js) {
        Ok(v) => v,
        Err(e) => {
            return JsValue::from_str(&format!(
                "Error: failed to parse groups: {}",
                e
            ));
        }
    };

    let k = raw.len();
    if k < 2 {
        return JsValue::from_str("Error: at least 2 groups are required for ANOVA");
    }

    // Validate each group
    for (i, g) in raw.iter().enumerate() {
        if g.is_empty() {
            return JsValue::from_str(&format!("Error: group {} is empty", i));
        }
    }

    // Grand total and grand mean
    let n_total: usize = raw.iter().map(|g| g.len()).sum();
    let grand_sum: f64 = raw.iter().flat_map(|g| g.iter()).sum();
    let grand_mean = grand_sum / n_total as f64;

    // SS between (SS_B) and SS within (SS_W)
    let mut ss_between = 0.0_f64;
    let mut ss_within = 0.0_f64;

    for g in &raw {
        let n_g = g.len() as f64;
        let mean_g: f64 = g.iter().sum::<f64>() / n_g;
        ss_between += n_g * (mean_g - grand_mean).powi(2);
        ss_within += g.iter().map(|&v| (v - mean_g).powi(2)).sum::<f64>();
    }

    let df_between = k - 1;
    let df_within = n_total - k;

    if df_within == 0 {
        return JsValue::from_str("Error: not enough observations for within-group variance");
    }

    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;

    let f_stat = if ms_within == 0.0 {
        f64::INFINITY
    } else {
        ms_between / ms_within
    };

    // F-distribution p-value using regularised incomplete beta
    // F(d1, d2) -> p = I_{d2/(d2+d1*F)}(d2/2, d1/2)
    let p_value = f_distribution_pvalue(f_stat, df_between as f64, df_within as f64);

    let result = serde_json::json!({
        "f_statistic": f_stat,
        "p_value": p_value,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "ss_between": ss_between,
        "ss_within": ss_within,
    });

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/// Compute principal component analysis (PCA) on a 2-D data matrix.
///
/// # Arguments
///
/// * `data_js` — 2-D JS array of shape `[n_samples, n_features]`.
/// * `n_components` — number of principal components to return.
///
/// # Returns
///
/// `JsValue` JSON:
/// ```json
/// {
///   "scores":           [[...], ...],  // n_samples × n_components
///   "loadings":         [[...], ...],  // n_features × n_components (eigenvectors)
///   "variance_explained": [f64, ...],  // per-component explained variance ratio
///   "eigenvalues":      [f64, ...],    // raw eigenvalues
///   "n_components":     usize,
///   "n_samples":        usize,
///   "n_features":       usize
/// }
/// ```
///
/// Returns a `JsValue` error string for invalid inputs.
#[wasm_bindgen]
pub fn wasm_pca(data_js: JsValue, n_components: usize) -> JsValue {
    // Parse 2-D array
    let data: Vec<Vec<f64>> = match serde_wasm_bindgen::from_value(data_js) {
        Ok(v) => v,
        Err(e) => {
            return JsValue::from_str(&format!("Error: failed to parse data: {}", e));
        }
    };

    let n_samples = data.len();
    if n_samples == 0 {
        return JsValue::from_str("Error: data must not be empty");
    }
    let n_features = data[0].len();
    if n_features == 0 {
        return JsValue::from_str("Error: each sample must have at least one feature");
    }
    // Validate all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != n_features {
            return JsValue::from_str(&format!(
                "Error: row {} has {} features, expected {}",
                i,
                row.len(),
                n_features
            ));
        }
    }
    let k = n_components.min(n_features).min(n_samples);
    if k == 0 {
        return JsValue::from_str("Error: n_components must be at least 1");
    }

    // Flatten to row-major Vec<f64>
    let mut flat = vec![0.0_f64; n_samples * n_features];
    for (i, row) in data.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            flat[i * n_features + j] = v;
        }
    }

    // Centre (subtract column means)
    let mut col_means = vec![0.0_f64; n_features];
    for j in 0..n_features {
        col_means[j] = (0..n_samples).map(|i| flat[i * n_features + j]).sum::<f64>()
            / n_samples as f64;
    }
    for i in 0..n_samples {
        for j in 0..n_features {
            flat[i * n_features + j] -= col_means[j];
        }
    }

    // Covariance matrix C = Xᵀ X / (n_samples - 1)   (n_features × n_features)
    let denom = (n_samples - 1).max(1) as f64;
    let mut cov = vec![0.0_f64; n_features * n_features];
    for a in 0..n_features {
        for b in 0..n_features {
            let mut s = 0.0_f64;
            for i in 0..n_samples {
                s += flat[i * n_features + a] * flat[i * n_features + b];
            }
            cov[a * n_features + b] = s / denom;
        }
    }

    // Eigendecomposition via power-iteration / Gram-Schmidt deflation
    // Returns k eigenvectors (columns) and eigenvalues in descending order
    let (eigenvalues, eigenvectors) = power_iteration_eig(&cov, n_features, k);

    // Scores: X · V   (n_samples × k)
    let mut scores = vec![0.0_f64; n_samples * k];
    for i in 0..n_samples {
        for c in 0..k {
            let mut s = 0.0_f64;
            for j in 0..n_features {
                s += flat[i * n_features + j] * eigenvectors[j * k + c];
            }
            scores[i * k + c] = s;
        }
    }

    // Variance explained ratios
    let total_var: f64 = eigenvalues.iter().sum::<f64>().max(1e-300);
    let variance_explained: Vec<f64> = eigenvalues.iter().map(|&ev| ev / total_var).collect();

    // Reshape scores to 2-D Vec
    let scores_2d: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| scores[i * k..(i + 1) * k].to_vec())
        .collect();

    // Loadings: eigenvectors reshaped to n_features × k
    let loadings_2d: Vec<Vec<f64>> = (0..n_features)
        .map(|j| (0..k).map(|c| eigenvectors[j * k + c]).collect())
        .collect();

    let result = serde_json::json!({
        "scores":             scores_2d,
        "loadings":           loadings_2d,
        "variance_explained": variance_explained,
        "eigenvalues":        eigenvalues,
        "n_components":       k,
        "n_samples":          n_samples,
        "n_features":         n_features,
    });

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

/// Perform k-means clustering on a 2-D data matrix.
///
/// Uses the k-means++ initialisation heuristic for reliable convergence.
///
/// # Arguments
///
/// * `data_js`  — 2-D JS array of shape `[n_samples, n_features]`.
/// * `k`        — number of clusters.
/// * `max_iter` — maximum number of Lloyd iterations.
///
/// # Returns
///
/// `JsValue` JSON:
/// ```json
/// {
///   "labels":    [0, 2, 1, ...],       // cluster index per sample
///   "centroids": [[...], ...],         // k × n_features centroid coordinates
///   "inertia":   f64,                  // sum of squared distances to centroids
///   "n_iter":    usize,                // iterations actually performed
///   "converged": bool
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_kmeans(data_js: JsValue, k: usize, max_iter: usize) -> JsValue {
    let data: Vec<Vec<f64>> = match serde_wasm_bindgen::from_value(data_js) {
        Ok(v) => v,
        Err(e) => {
            return JsValue::from_str(&format!("Error: failed to parse data: {}", e));
        }
    };

    let n_samples = data.len();
    if n_samples == 0 {
        return JsValue::from_str("Error: data must not be empty");
    }
    if k == 0 {
        return JsValue::from_str("Error: k must be at least 1");
    }
    if k > n_samples {
        return JsValue::from_str("Error: k must not exceed the number of samples");
    }
    let n_features = data[0].len();
    if n_features == 0 {
        return JsValue::from_str("Error: samples must have at least one feature");
    }
    for (i, row) in data.iter().enumerate() {
        if row.len() != n_features {
            return JsValue::from_str(&format!(
                "Error: row {} has {} features, expected {}",
                i,
                row.len(),
                n_features
            ));
        }
    }
    let max_iter = max_iter.max(1);

    // Flatten
    let mut flat = vec![0.0_f64; n_samples * n_features];
    for (i, row) in data.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            flat[i * n_features + j] = v;
        }
    }

    // k-means++ initialisation (deterministic seed based on data hash)
    let mut centroids = kmeans_plusplus_init(&flat, n_samples, n_features, k);

    let mut labels = vec![0usize; n_samples];
    let mut n_iter = 0usize;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Assignment step
        let mut changed = false;
        for i in 0..n_samples {
            let mut best_c = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let d = squared_dist(
                    &flat[i * n_features..(i + 1) * n_features],
                    &centroids[c * n_features..(c + 1) * n_features],
                );
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            converged = true;
            break;
        }

        // Update step
        let mut new_centroids = vec![0.0_f64; k * n_features];
        let mut counts = vec![0usize; k];
        for i in 0..n_samples {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..n_features {
                new_centroids[c * n_features + j] += flat[i * n_features + j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f64;
                for j in 0..n_features {
                    new_centroids[c * n_features + j] /= cnt;
                }
            } else {
                // Empty cluster: keep old centroid
                for j in 0..n_features {
                    new_centroids[c * n_features + j] = centroids[c * n_features + j];
                }
            }
        }
        centroids = new_centroids;
    }

    // Compute inertia
    let inertia: f64 = (0..n_samples)
        .map(|i| {
            let c = labels[i];
            squared_dist(
                &flat[i * n_features..(i + 1) * n_features],
                &centroids[c * n_features..(c + 1) * n_features],
            )
        })
        .sum();

    // Reshape centroids to 2-D
    let centroids_2d: Vec<Vec<f64>> = (0..k)
        .map(|c| centroids[c * n_features..(c + 1) * n_features].to_vec())
        .collect();

    let result = serde_json::json!({
        "labels":    labels,
        "centroids": centroids_2d,
        "inertia":   inertia,
        "n_iter":    n_iter,
        "converged": converged,
    });

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

// ============================================================================
// Internal pure-Rust helpers (not exported to WASM)
// ============================================================================

/// Compute average ranks for a slice, handling ties by assigning the mean rank.
fn average_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    // Create (value, original_index) pairs and sort by value
    let mut indexed: Vec<(f64, usize)> = data.iter().copied().zip(0..n).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0usize;
    while i < n {
        // Find the run of equal values
        let mut j = i + 1;
        while j < n && indexed[j].0 == indexed[i].0 {
            j += 1;
        }
        // Average rank for this tie group (1-based ranks)
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for item in indexed[i..j].iter() {
            ranks[item.1] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Solve an `n × n` system `A · x = b` via Gaussian elimination with partial
/// pivoting.  `a_flat` is row-major with `n * n` elements.
pub(crate) fn solve_system_f64(a_flat: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
    // Build augmented matrix [A | b]
    let mut aug = vec![0.0_f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_flat[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[i * (n + 1) + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            let v = aug[k * (n + 1) + i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-14 {
            return Err("Matrix is singular or ill-conditioned".to_string());
        }
        // Swap rows i and max_row
        if max_row != i {
            for j in 0..=n {
                aug.swap(i * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        // Eliminate below
        for k in (i + 1)..n {
            let factor = aug[k * (n + 1) + i] / aug[i * (n + 1) + i];
            for j in i..=n {
                let val = aug[i * (n + 1) + j] * factor;
                aug[k * (n + 1) + j] -= val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / aug[i * (n + 1) + i];
    }
    Ok(x)
}

/// Power-iteration eigen-decomposition of a symmetric `n×n` matrix stored
/// row-major in `cov`.  Returns `(eigenvalues, eigenvectors_col_major)`:
/// - `eigenvalues[c]` is the c-th eigenvalue (descending order)
/// - `eigenvectors` is `n * k` column-major: column `c` is eigenvector `c`
fn power_iteration_eig(cov: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let max_iter = 1000usize;
    let tol = 1e-10_f64;

    // We will deflate: after finding eigenvector v_c, subtract λ_c v_c v_cᵀ
    let mut deflated = cov.to_vec();
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = vec![0.0_f64; n * k]; // column-major n × k

    for c in 0..k {
        // Initialise with a deterministic non-zero vector
        let mut v: Vec<f64> = (0..n).map(|i| if i == c % n { 1.0 } else { 0.0 }).collect();
        // Orthogonalise against previously found eigenvectors
        gram_schmidt_ortho(&mut v, &eigenvectors, c, n);
        normalise(&mut v);

        let mut eigenvalue = 0.0_f64;
        for _iter in 0..max_iter {
            // w = deflated · v
            let mut w = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += deflated[i * n + j] * v[j];
                }
            }
            // Rayleigh quotient
            let rq: f64 = dot_product(&v, &w);
            let w_norm = dot_product(&w, &w).sqrt();
            if w_norm < 1e-300 {
                break;
            }
            let v_new: Vec<f64> = w.iter().map(|&x| x / w_norm).collect();
            let diff = v_new
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            v = v_new;
            eigenvalue = rq;
            if diff < tol {
                break;
            }
        }

        eigenvalues.push(eigenvalue.max(0.0));

        // Store eigenvector as column c (column-major)
        for i in 0..n {
            eigenvectors[i * k + c] = v[i];
        }

        // Deflate: A ← A - λ v vᵀ
        for i in 0..n {
            for j in 0..n {
                deflated[i * n + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

/// Gram-Schmidt orthogonalisation of `v` against the first `c` columns
/// stored in `evecs` (column-major `n × k`).
fn gram_schmidt_ortho(v: &mut Vec<f64>, evecs: &[f64], c: usize, n: usize) {
    if c == 0 {
        return;
    }
    // Number of columns available = c
    let k = if evecs.is_empty() {
        0
    } else {
        evecs.len() / n
    };
    for col in 0..c.min(k) {
        let proj: f64 = (0..n).map(|i| v[i] * evecs[i * k + col]).sum();
        for i in 0..n {
            v[i] -= proj * evecs[i * k + col];
        }
    }
}

/// Normalise `v` in-place to unit length.
fn normalise(v: &mut Vec<f64>) {
    let norm = dot_product(v, v).sqrt();
    if norm > 1e-300 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Dot product of two equal-length slices.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Squared Euclidean distance between two equal-length slices.
fn squared_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

/// k-means++ initialisation.  Picks the first centroid randomly (deterministic
/// LCG), then picks subsequent centroids with probability proportional to the
/// squared distance to the nearest existing centroid.
fn kmeans_plusplus_init(flat: &[f64], n_samples: usize, n_features: usize, k: usize) -> Vec<f64> {
    // Deterministic LCG seeded from data checksum
    let seed: u64 = flat
        .iter()
        .fold(0u64, |acc, &v| acc.wrapping_add(v.to_bits()));
    let mut rng = LcgRng::new(seed ^ 6364136223846793005);

    let mut centroids = vec![0.0_f64; k * n_features];
    let mut chosen = vec![false; n_samples];

    // First centroid: pick sample with index proportional to first LCG draw
    let first = (rng.next() as usize) % n_samples;
    chosen[first] = true;
    centroids[..n_features].copy_from_slice(&flat[first * n_features..(first + 1) * n_features]);

    for c in 1..k {
        // Compute squared distances to nearest chosen centroid
        let mut d2 = vec![f64::INFINITY; n_samples];
        let mut total = 0.0_f64;
        for i in 0..n_samples {
            if chosen[i] {
                d2[i] = 0.0;
                continue;
            }
            for prev in 0..c {
                let dist = squared_dist(
                    &flat[i * n_features..(i + 1) * n_features],
                    &centroids[prev * n_features..(prev + 1) * n_features],
                );
                if dist < d2[i] {
                    d2[i] = dist;
                }
            }
            total += d2[i];
        }

        // Sample proportional to d2
        let threshold = (rng.next_f64()) * total;
        let mut cumsum = 0.0_f64;
        let mut chosen_idx = 0usize;
        for i in 0..n_samples {
            if !chosen[i] {
                cumsum += d2[i];
                if cumsum >= threshold {
                    chosen_idx = i;
                    break;
                }
            }
            chosen_idx = i;
        }

        chosen[chosen_idx] = true;
        let dest = c * n_features;
        centroids[dest..dest + n_features]
            .copy_from_slice(&flat[chosen_idx * n_features..(chosen_idx + 1) * n_features]);
    }

    centroids
}

// ---------------------------------------------------------------------------
// F-distribution p-value
// ---------------------------------------------------------------------------

/// Upper-tail p-value for an F(d1, d2) random variable: P(F > f_stat).
fn f_distribution_pvalue(f_stat: f64, d1: f64, d2: f64) -> f64 {
    if f_stat <= 0.0 {
        return 1.0;
    }
    if f_stat.is_infinite() {
        return 0.0;
    }
    // P(F > x) = I_{d2/(d2+d1*x)}(d2/2, d1/2)
    let x = d2 / (d2 + d1 * f_stat);
    regularized_incomplete_beta(x, d2 / 2.0, d1 / 2.0)
}

/// Regularised incomplete beta function via continued-fraction (Lentz's method).
/// Taken from the same implementation as stats.rs for consistency.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    let max_iter = 200;
    let eps = 1e-14;
    let mut c = 1.0_f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    for m in 1..=max_iter {
        let m_f = m as f64;
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_even / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;
        let num_odd =
            -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_odd / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    front * f
}

/// Natural log of the gamma function via Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    use std::f64::consts::PI;
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    if x < 0.5 {
        let v = PI / ((PI * x).sin() * ln_gamma(1.0 - x).exp());
        return v.ln();
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C[1..].iter().enumerate() {
        a += c / (x + i as f64 + 1.0);
    }
    (2.0 * PI).sqrt().ln() + a.ln() + (x + 0.5) * t.ln() - t
}

// ---------------------------------------------------------------------------
// Deterministic LCG random number generator for WASM (no std::random)
// ---------------------------------------------------------------------------

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        // LCG parameters from Knuth
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_ranks_no_ties() {
        let x = [3.0, 1.0, 2.0];
        let r = average_ranks(&x);
        // sorted: 1.0→rank1, 2.0→rank2, 3.0→rank3
        assert!((r[0] - 3.0).abs() < 1e-10); // 3.0 is rank 3
        assert!((r[1] - 1.0).abs() < 1e-10); // 1.0 is rank 1
        assert!((r[2] - 2.0).abs() < 1e-10); // 2.0 is rank 2
    }

    #[test]
    fn test_average_ranks_with_ties() {
        let x = [1.0, 1.0, 3.0];
        let r = average_ranks(&x);
        // Both 1.0 values share ranks 1 and 2 → average = 1.5
        assert!((r[0] - 1.5).abs() < 1e-10);
        assert!((r[1] - 1.5).abs() < 1e-10);
        assert!((r[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = wasm_spearman_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Spearman = {}", r);
    }

    #[test]
    fn test_spearman_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r = wasm_spearman_correlation(&x, &y);
        assert!((r + 1.0).abs() < 1e-10, "Spearman = {}", r);
    }

    #[test]
    fn test_spearman_length_mismatch() {
        let r = wasm_spearman_correlation(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
        assert!(r.is_nan());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_polynomial_fit_linear() {
        // y = 2x + 1 should be fitted exactly
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 5.0, 7.0, 9.0];
        let result = wasm_polynomial_fit(&x, &y, 1);
        // Should not be NULL
        assert!(!result.is_null(), "polynomial_fit returned NULL");
        assert!(!result.is_string(), "polynomial_fit returned error: {:?}", result);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_polynomial_fit_empty_input() {
        let result = wasm_polynomial_fit(&[], &[], 1);
        assert!(result.is_string());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_polynomial_fit_zero_degree() {
        let result = wasm_polynomial_fit(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 0);
        assert!(result.is_string());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_anova_one_way_basic() {
        // Groups with very different means → should yield a large F-stat
        // We just check it doesn't panic and returns a non-null value.
        // JsValue parsing in unit tests is tricky without wasm-bindgen-test;
        // we verify the result is not an error string by inspecting it.
        use wasm_bindgen::JsValue;
        let groups_js = serde_wasm_bindgen::to_value(&vec![
            vec![1.0_f64, 2.0, 3.0],
            vec![10.0_f64, 11.0, 12.0],
            vec![20.0_f64, 21.0, 22.0],
        ])
        .unwrap_or(JsValue::NULL);
        let result = wasm_anova_one_way(groups_js);
        assert!(!result.is_null(), "ANOVA returned NULL");
        assert!(!result.is_string(), "ANOVA returned error: {:?}", result);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_anova_one_way_too_few_groups() {
        let groups_js =
            serde_wasm_bindgen::to_value(&vec![vec![1.0_f64, 2.0]]).unwrap_or(JsValue::NULL);
        let result = wasm_anova_one_way(groups_js);
        assert!(result.is_string(), "Should return error for 1 group");
    }

    #[test]
    fn test_solve_system() {
        // 2x + y = 5, x + 3y = 10 → x = 1, y = 3
        let a = [2.0_f64, 1.0, 1.0, 3.0];
        let b = [5.0_f64, 10.0];
        let x = solve_system_f64(&a, &b, 2).expect("solve_system_f64 failed");
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_f_pvalue_large_f() {
        // Very large F-stat should give near-zero p-value
        let p = f_distribution_pvalue(1000.0, 2.0, 10.0);
        assert!(p < 0.001, "p-value for large F should be tiny, got {}", p);
    }

    #[test]
    fn test_f_pvalue_zero_f() {
        // F = 0 should give p = 1
        let p = f_distribution_pvalue(0.0, 2.0, 10.0);
        assert!((p - 1.0).abs() < 1e-6, "p-value for F=0 should be 1, got {}", p);
    }

    #[test]
    fn test_lcg_rng_reproducible() {
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        for _ in 0..10 {
            assert_eq!(r1.next(), r2.next());
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_pca_basic() {
        // 4 samples, 2 features; first PC should explain most variance
        let data_js = serde_wasm_bindgen::to_value(&vec![
            vec![1.0_f64, 2.0],
            vec![2.0_f64, 4.0],
            vec![3.0_f64, 6.0],
            vec![4.0_f64, 8.0],
        ])
        .unwrap_or(JsValue::NULL);
        let result = wasm_pca(data_js, 2);
        assert!(!result.is_null(), "PCA returned NULL");
        assert!(!result.is_string(), "PCA returned error: {:?}", result);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_pca_empty_data() {
        let data_js =
            serde_wasm_bindgen::to_value(&Vec::<Vec<f64>>::new()).unwrap_or(JsValue::NULL);
        let result = wasm_pca(data_js, 1);
        assert!(result.is_string());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_kmeans_basic() {
        // Two obvious clusters
        let data_js = serde_wasm_bindgen::to_value(&vec![
            vec![0.0_f64, 0.0],
            vec![0.1_f64, 0.1],
            vec![10.0_f64, 10.0],
            vec![10.1_f64, 10.1],
        ])
        .unwrap_or(JsValue::NULL);
        let result = wasm_kmeans(data_js, 2, 100);
        assert!(!result.is_null(), "kmeans returned NULL");
        assert!(!result.is_string(), "kmeans returned error: {:?}", result);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_kmeans_k_too_large() {
        let data_js = serde_wasm_bindgen::to_value(&vec![vec![1.0_f64, 2.0]])
            .unwrap_or(JsValue::NULL);
        let result = wasm_kmeans(data_js, 5, 10);
        assert!(result.is_string());
    }
}
