//! Medical and healthcare synthetic dataset generators.
//!
//! Provides realistic-looking but entirely synthetic datasets modelled on
//! well-known medical benchmarks (heart disease, diabetes, breast cancer,
//! survival analysis, longitudinal studies).
//!
//! All generators are deterministic (Park-Miller LCG, no external crates).

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Internal LCG RNG
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
    /// N(0,1) via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Exponential(rate) sample.
    fn next_exponential(&mut self, rate: f64) -> f64 {
        let u = self.next_f64().max(1e-15);
        -u.ln() / rate
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Heart Disease Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic heart-disease risk dataset.
///
/// Features (13 columns, matching Cleveland Heart Disease schema):
///
/// | Index | Name               | Range / Type                 |
/// |-------|--------------------|------------------------------|
/// | 0     | age                | 29–77 years                  |
/// | 1     | sex                | 0 = female, 1 = male         |
/// | 2     | chest_pain_type    | 0–3 (typical to asymptomatic)|
/// | 3     | resting_bp         | 94–200 mmHg                  |
/// | 4     | cholesterol        | 126–564 mg/dL                |
/// | 5     | fasting_bs         | 0 = ≤120, 1 = >120 mg/dL    |
/// | 6     | rest_ecg           | 0–2                          |
/// | 7     | max_hr             | 71–202 bpm                   |
/// | 8     | exercise_angina    | 0 = no, 1 = yes              |
/// | 9     | oldpeak            | 0.0–6.2 (ST depression)      |
/// | 10    | slope              | 0–2                          |
/// | 11    | n_vessels          | 0–3 fluoroscopy vessels      |
/// | 12    | thal_defect        | 0–2 (normal/fixed/reversible)|
///
/// Label: `0` = no disease, `1` = disease.
///
/// # Arguments
///
/// * `n_samples` – Number of patients to generate.
/// * `seed`      – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_samples == 0`.
pub fn make_heart_disease_dataset(
    n_samples: usize,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_heart_disease_dataset: n_samples must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut labels: Vec<usize> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let age = 29.0 + rng.next_f64() * 48.0; // 29–77
        let sex = if rng.next_f64() < 0.68 { 1.0 } else { 0.0 }; // ~68% male in original
        let chest_pain_type = rng.next_usize(4) as f64;
        let resting_bp = 94.0 + rng.next_f64() * 106.0;
        let cholesterol = 126.0 + rng.next_f64() * 438.0;
        let fasting_bs = if rng.next_f64() < 0.15 { 1.0 } else { 0.0 };
        let rest_ecg = rng.next_usize(3) as f64;
        let max_hr = 71.0 + rng.next_f64() * 131.0;
        let exercise_angina = if rng.next_f64() < 0.32 { 1.0 } else { 0.0 };
        let oldpeak = rng.next_f64() * 6.2;
        let slope = rng.next_usize(3) as f64;
        let n_vessels = rng.next_usize(4) as f64;
        let thal_defect = rng.next_usize(3) as f64;

        // Disease risk heuristic (logistic-like score).
        let risk_score = 0.04 * (age - 54.0)
            + 0.4 * sex
            + 0.3 * (3.0 - chest_pain_type) // asymptomatic CP → higher risk
            + 0.003 * (resting_bp - 130.0)
            + 0.001 * (cholesterol - 240.0)
            + 0.3 * fasting_bs
            + 0.2 * exercise_angina
            + 0.3 * oldpeak
            + 0.2 * n_vessels
            + 0.2 * (thal_defect - 1.0)
            + rng.next_normal() * 0.3;

        let label = if risk_score > 0.5 { 1 } else { 0 };

        features.push(vec![
            age,
            sex,
            chest_pain_type,
            resting_bp,
            cholesterol,
            fasting_bs,
            rest_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            slope,
            n_vessels,
            thal_defect,
        ]);
        labels.push(label);
    }

    Ok((features, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// Diabetes Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic diabetes progression dataset (regression).
///
/// Inspired by the Efron et al. diabetes dataset: 10 physiological features
/// predict a quantitative measure of disease progression one year later.
///
/// Feature columns (all standardised to mean≈0, std≈1):
/// age, sex, bmi, bp, s1 (total cholesterol), s2 (LDL), s3 (HDL),
/// s4 (TCH), s5 (log lamotrigine), s6 (blood sugar).
///
/// Target: disease progression score (continuous, roughly 25–346).
///
/// # Arguments
///
/// * `n_samples` – Number of samples.
/// * `seed`      – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_samples == 0`.
pub fn make_diabetes_dataset(n_samples: usize, seed: u64) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_diabetes_dataset: n_samples must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut targets: Vec<f64> = Vec::with_capacity(n_samples);

    // Regression coefficients (hand-tuned to mimic original dataset).
    let coeffs = [0.0, -10.0, 58.0, 30.0, -18.0, 5.0, -23.0, 0.0, 44.0, 5.0];
    let intercept = 152.0;

    for _ in 0..n_samples {
        let x: Vec<f64> = (0..10).map(|_| rng.next_normal()).collect();
        let mut y = intercept;
        for (xi, &ci) in x.iter().zip(coeffs.iter()) {
            y += ci * xi;
        }
        y += rng.next_normal() * 25.0; // residual noise
        let y = y.clamp(25.0, 346.0);
        features.push(x);
        targets.push(y);
    }

    Ok((features, targets))
}

// ─────────────────────────────────────────────────────────────────────────────
// Breast Cancer Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic breast cancer classification dataset.
///
/// Features (30 columns) correspond to nucleus measurements derived from
/// Fine Needle Aspirate (FNA) biopsy images, matching the Wisconsin dataset:
/// mean, SE, and worst of radius, texture, perimeter, area, smoothness,
/// compactness, concavity, concave_points, symmetry, fractal_dimension.
///
/// Label: `0` = malignant, `1` = benign.
///
/// # Arguments
///
/// * `n_samples` – Number of samples.
/// * `seed`      – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_samples == 0`.
pub fn make_breast_cancer_dataset(
    n_samples: usize,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_breast_cancer_dataset: n_samples must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut labels: Vec<usize> = Vec::with_capacity(n_samples);

    // Mean values and standard deviations for malignant (0) and benign (1) classes.
    // Format: (benign_mean, benign_std, malignant_mean, malignant_std) for each of 10 base features.
    let feature_params: [(f64, f64, f64, f64); 10] = [
        (12.15, 1.78, 17.46, 3.20),       // radius_mean
        (17.91, 4.01, 21.60, 3.78),       // texture_mean
        (78.08, 11.84, 115.4, 21.90),     // perimeter_mean
        (462.8, 134.3, 978.4, 367.3),     // area_mean
        (0.0924, 0.0134, 0.1028, 0.0138), // smoothness_mean
        (0.0800, 0.0338, 0.1447, 0.0526), // compactness_mean
        (0.0461, 0.0793, 0.1600, 0.1076), // concavity_mean
        (0.0257, 0.0390, 0.0880, 0.0487), // concave_points_mean
        (0.1741, 0.0274, 0.1926, 0.0277), // symmetry_mean
        (0.0628, 0.0062, 0.0627, 0.0066), // fractal_dimension_mean
    ];

    for _ in 0..n_samples {
        // ~63% benign to match original dataset distribution.
        let is_benign = rng.next_f64() < 0.627;
        let label = if is_benign { 1 } else { 0 };

        let mut row: Vec<f64> = Vec::with_capacity(30);
        for &(bm, bs, mm, ms) in &feature_params {
            let (mean, std) = if is_benign { (bm, bs) } else { (mm, ms) };
            // mean, SE, worst
            let base = mean + rng.next_normal() * std;
            let se = (std * 0.15).abs() * rng.next_f64().max(0.01);
            let worst = base + rng.next_normal().abs() * std * 0.5;
            row.push(base.max(0.0));
            row.push(se);
            row.push(worst.max(0.0));
        }

        features.push(row);
        labels.push(label);
    }

    Ok((features, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// SurvivalDataset
// ─────────────────────────────────────────────────────────────────────────────

/// Survival analysis dataset with right-censoring.
#[derive(Debug, Clone)]
pub struct SurvivalDataset {
    /// Covariate matrix: one row per patient, one column per feature.
    pub features: Vec<Vec<f64>>,
    /// Observed (possibly censored) survival time for each patient.
    pub times: Vec<f64>,
    /// `true` = event (death/failure) observed; `false` = right-censored.
    pub events: Vec<bool>,
    /// Feature column names.
    pub feature_names: Vec<String>,
}

/// Generate a synthetic survival analysis dataset.
///
/// Survival times follow an Accelerated Failure Time (AFT) model:
/// `T = exp(X β + ε)` where `ε ~ N(0, σ²)`.  Censoring times are drawn
/// independently from `Exp(rate)` and censoring occurs when `C < T`.
///
/// # Arguments
///
/// * `n_samples`      – Number of patients.
/// * `n_features`     – Number of covariates.
/// * `censoring_rate` – Desired fraction of censored observations ≈ (0, 1).
/// * `seed`           – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_samples == 0`, `n_features == 0`, or
/// `censoring_rate` is not in `[0, 1)`.
pub fn make_survival_dataset(
    n_samples: usize,
    n_features: usize,
    censoring_rate: f64,
    seed: u64,
) -> Result<SurvivalDataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_survival_dataset: n_samples must be >= 1".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_survival_dataset: n_features must be >= 1".to_string(),
        ));
    }
    if !(0.0 <= censoring_rate && censoring_rate < 1.0) {
        return Err(DatasetsError::InvalidFormat(
            "make_survival_dataset: censoring_rate must be in [0, 1)".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);

    // Random AFT coefficients.
    let betas: Vec<f64> = (0..n_features).map(|_| rng.next_normal() * 0.5).collect();

    // Censoring time rate: calibrate so that the expected censoring fraction
    // matches `censoring_rate`.  If censoring_rate = 0 use a very large bound.
    let censor_rate = if censoring_rate < 1e-6 {
        1e-9
    } else {
        // Approximate: rate ≈ -log(1 - censoring_rate) / E[T].
        // E[T] ≈ exp(0.5) ≈ 1.65 (for N(0,1) AFT with σ=0.5).
        -((1.0 - censoring_rate).ln()) / 1.65
    };

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("x{i}")).collect();
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut times: Vec<f64> = Vec::with_capacity(n_samples);
    let mut events: Vec<bool> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        let linear_pred: f64 = x.iter().zip(betas.iter()).map(|(xi, bi)| xi * bi).sum();
        let log_time = linear_pred + rng.next_normal() * 0.5;
        let true_time = log_time.exp();

        let censor_time = rng.next_exponential(censor_rate);
        let observed_time = true_time.min(censor_time);
        let event_observed = true_time <= censor_time;

        features.push(x);
        times.push(observed_time.max(0.001)); // ensure positive
        events.push(event_observed);
    }

    Ok(SurvivalDataset {
        features,
        times,
        events,
        feature_names,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// LongitudinalDataset
// ─────────────────────────────────────────────────────────────────────────────

/// Longitudinal (panel) dataset with repeated measurements per patient.
#[derive(Debug, Clone)]
pub struct LongitudinalDataset {
    /// Patient identifier for each row (enables grouping by patient).
    pub patient_ids: Vec<usize>,
    /// Measurement time (in arbitrary units, e.g. months post-baseline).
    pub times: Vec<f64>,
    /// Covariate matrix aligned with `patient_ids` and `times`.
    pub features: Vec<Vec<f64>>,
    /// Scalar outcome for each measurement.
    pub outcomes: Vec<f64>,
}

/// Generate a synthetic longitudinal medical dataset.
///
/// Each patient has `n_timepoints` measurements at times
/// `{0, Δt, 2Δt, …}` where `Δt ~ U(0.5, 1.5)` per patient.
/// Feature values evolve via a random-walk, and the outcome is a
/// noisy linear function of the features.
///
/// # Arguments
///
/// * `n_patients`   – Number of distinct patients.
/// * `n_timepoints` – Measurements per patient.
/// * `n_features`   – Number of covariates.
/// * `seed`         – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if any argument is zero.
pub fn make_longitudinal_dataset(
    n_patients: usize,
    n_timepoints: usize,
    n_features: usize,
    seed: u64,
) -> Result<LongitudinalDataset> {
    if n_patients == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_longitudinal_dataset: n_patients must be >= 1".to_string(),
        ));
    }
    if n_timepoints == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_longitudinal_dataset: n_timepoints must be >= 1".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_longitudinal_dataset: n_features must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);

    let total_rows = n_patients * n_timepoints;
    let mut patient_ids: Vec<usize> = Vec::with_capacity(total_rows);
    let mut times_vec: Vec<f64> = Vec::with_capacity(total_rows);
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(total_rows);
    let mut outcomes: Vec<f64> = Vec::with_capacity(total_rows);

    // Random outcome coefficients.
    let betas: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
    let intercept = rng.next_normal() * 5.0;

    for pid in 0..n_patients {
        // Baseline feature values for this patient.
        let mut current_x: Vec<f64> = (0..n_features).map(|_| rng.next_normal()).collect();
        // Time increment ~ U(0.5, 1.5) months.
        let dt = 0.5 + rng.next_f64();
        let mut t = 0.0f64;

        for _ in 0..n_timepoints {
            patient_ids.push(pid);
            times_vec.push(t);
            features.push(current_x.clone());

            let y = intercept
                + current_x
                    .iter()
                    .zip(betas.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum::<f64>()
                + rng.next_normal() * 0.5; // measurement noise

            outcomes.push(y);

            // Random-walk evolution of features.
            for xval in current_x.iter_mut() {
                *xval += rng.next_normal() * 0.1;
            }
            t += dt;
        }
    }

    Ok(LongitudinalDataset {
        patient_ids,
        times: times_vec,
        features,
        outcomes,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heart_disease_basic() {
        let (feats, labels) = make_heart_disease_dataset(100, 42).expect("heart disease failed");
        assert_eq!(feats.len(), 100);
        assert_eq!(labels.len(), 100);
        for row in &feats {
            assert_eq!(row.len(), 13);
            // age in [29, 77]
            assert!(
                row[0] >= 29.0 && row[0] <= 77.0,
                "age out of range: {}",
                row[0]
            );
            // sex in {0, 1}
            assert!(row[1] == 0.0 || row[1] == 1.0);
            // cholesterol positive
            assert!(row[4] > 0.0);
        }
        for &l in &labels {
            assert!(l < 2, "label out of range: {l}");
        }
        // Expect both classes present in 100 samples.
        let n_pos = labels.iter().filter(|&&l| l == 1).count();
        assert!(n_pos > 0 && n_pos < 100, "degenerate class distribution");
    }

    #[test]
    fn test_heart_disease_zero_error() {
        assert!(make_heart_disease_dataset(0, 1).is_err());
    }

    #[test]
    fn test_diabetes_basic() {
        let (feats, targets) = make_diabetes_dataset(80, 7).expect("diabetes failed");
        assert_eq!(feats.len(), 80);
        assert_eq!(targets.len(), 80);
        for row in &feats {
            assert_eq!(row.len(), 10);
        }
        for &y in &targets {
            assert!(y >= 25.0 && y <= 346.0, "target out of range: {y}");
        }
    }

    #[test]
    fn test_breast_cancer_basic() {
        let (feats, labels) = make_breast_cancer_dataset(60, 13).expect("breast cancer failed");
        assert_eq!(feats.len(), 60);
        assert_eq!(labels.len(), 60);
        for row in &feats {
            assert_eq!(row.len(), 30);
            for &v in row {
                assert!(v >= 0.0, "negative feature value: {v}");
            }
        }
        for &l in &labels {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_survival_basic() {
        let ds = make_survival_dataset(50, 5, 0.3, 42).expect("survival failed");
        assert_eq!(ds.features.len(), 50);
        assert_eq!(ds.times.len(), 50);
        assert_eq!(ds.events.len(), 50);
        assert_eq!(ds.feature_names.len(), 5);
        for &t in &ds.times {
            assert!(t > 0.0, "non-positive survival time");
        }
        // Check approximate censoring rate.
        let n_censored = ds.events.iter().filter(|&&e| !e).count();
        let frac = n_censored as f64 / 50.0;
        assert!(frac < 0.8, "censoring rate suspiciously high: {frac:.2}");
    }

    #[test]
    fn test_survival_invalid() {
        assert!(make_survival_dataset(0, 5, 0.3, 1).is_err());
        assert!(make_survival_dataset(10, 0, 0.3, 1).is_err());
        assert!(make_survival_dataset(10, 5, 1.0, 1).is_err());
        assert!(make_survival_dataset(10, 5, -0.1, 1).is_err());
    }

    #[test]
    fn test_longitudinal_basic() {
        let ds = make_longitudinal_dataset(20, 5, 4, 77).expect("longitudinal failed");
        assert_eq!(ds.patient_ids.len(), 100);
        assert_eq!(ds.times.len(), 100);
        assert_eq!(ds.features.len(), 100);
        assert_eq!(ds.outcomes.len(), 100);

        // Each patient should appear exactly n_timepoints times.
        let mut counts = vec![0usize; 20];
        for &pid in &ds.patient_ids {
            counts[pid] += 1;
        }
        for c in counts {
            assert_eq!(c, 5);
        }

        // Times for each patient should be non-decreasing.
        for pid in 0..20usize {
            let patient_times: Vec<f64> = ds
                .patient_ids
                .iter()
                .zip(ds.times.iter())
                .filter(|(&p, _)| p == pid)
                .map(|(_, &t)| t)
                .collect();
            for w in patient_times.windows(2) {
                assert!(w[1] >= w[0], "non-monotone times for patient {pid}");
            }
        }
    }

    #[test]
    fn test_longitudinal_invalid() {
        assert!(make_longitudinal_dataset(0, 5, 3, 1).is_err());
        assert!(make_longitudinal_dataset(5, 0, 3, 1).is_err());
        assert!(make_longitudinal_dataset(5, 5, 0, 1).is_err());
    }

    #[test]
    fn test_reproducibility() {
        let (f1, l1) = make_heart_disease_dataset(20, 99).expect("a");
        let (f2, l2) = make_heart_disease_dataset(20, 99).expect("b");
        assert_eq!(l1, l2);
        for (r1, r2) in f1.iter().zip(f2.iter()) {
            for (v1, v2) in r1.iter().zip(r2.iter()) {
                assert!((v1 - v2).abs() < 1e-12);
            }
        }
    }
}
