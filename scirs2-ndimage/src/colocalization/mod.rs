//! Image colocalization analysis for fluorescence microscopy.
//!
//! Provides Pearson correlation, Manders overlap coefficients, Li's ICQ, and
//! the Costes automatic threshold method.

use crate::error::{NdimageError, NdimageResult};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn validate_equal_shape(ch1: &[Vec<f64>], ch2: &[Vec<f64>]) -> NdimageResult<(usize, usize)> {
    let rows = ch1.len();
    if rows == 0 {
        return Err(NdimageError::InvalidInput("Empty channel".into()));
    }
    let cols = ch1[0].len();
    if ch2.len() != rows || ch2.iter().any(|r| r.len() != cols)
        || ch1.iter().any(|r| r.len() != cols)
    {
        return Err(NdimageError::DimensionError(
            "channels must have equal shape".into(),
        ));
    }
    Ok((rows, cols))
}

/// Flatten two 2-D channel images into two `Vec<f64>` pixel vectors.
fn flatten_pair(ch1: &[Vec<f64>], ch2: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = ch1.iter().flat_map(|r| r.iter().copied()).collect();
    let b: Vec<f64> = ch2.iter().flat_map(|r| r.iter().copied()).collect();
    (a, b)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Pearson correlation coefficient of pixel intensities.
///
/// Returns a value in `[-1, 1]`.  Returns `0.0` when either channel has zero
/// variance (to avoid division by zero).
///
/// # Errors
/// Returns `DimensionError` if the channels differ in shape.
pub fn pearson_colocalization(
    channel1: &[Vec<f64>],
    channel2: &[Vec<f64>],
) -> NdimageResult<f64> {
    validate_equal_shape(channel1, channel2)?;
    let (a, b) = flatten_pair(channel1, channel2);
    Ok(pearson_vectors(&a, &b))
}

/// Internal: Pearson correlation on flat vectors.
fn pearson_vectors(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 { return 0.0; }
    let ma = mean(a);
    let mb = mean(b);
    let mut cov = 0.0f64;
    let mut va = 0.0f64;
    let mut vb = 0.0f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va  += da * da;
        vb  += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-300 { return 0.0; }
    (cov / denom).clamp(-1.0, 1.0)
}

/// Manders overlap coefficients M1 and M2.
///
/// * `M1 = Σ(ch1[i] where ch2[i] > threshold2) / Σ ch1`
/// * `M2 = Σ(ch2[i] where ch1[i] > threshold1) / Σ ch2`
///
/// Returns `(M1, M2)`.
///
/// # Errors
/// Returns `DimensionError` if the channels differ in shape.
pub fn manders_coefficients(
    channel1: &[Vec<f64>],
    channel2: &[Vec<f64>],
    threshold1: f64,
    threshold2: f64,
) -> NdimageResult<(f64, f64)> {
    validate_equal_shape(channel1, channel2)?;

    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut m1_num = 0.0f64;
    let mut m2_num = 0.0f64;

    for (r1, r2) in channel1.iter().zip(channel2.iter()) {
        for (&v1, &v2) in r1.iter().zip(r2.iter()) {
            sum1 += v1;
            sum2 += v2;
            if v2 > threshold2 { m1_num += v1; }
            if v1 > threshold1 { m2_num += v2; }
        }
    }

    let m1 = if sum1 < 1e-300 { 0.0 } else { (m1_num / sum1).clamp(0.0, 1.0) };
    let m2 = if sum2 < 1e-300 { 0.0 } else { (m2_num / sum2).clamp(0.0, 1.0) };

    Ok((m1, m2))
}

/// Li's Intensity Correlation Quotient (ICQ).
///
/// ICQ = (# pixels where (a-ā)(b-b̄) > 0 - # pixels where (a-ā)(b-b̄) < 0)
///       divided by total pixel count.
///
/// Returns a value in `(-0.5, 0.5)`. Positive values indicate colocalization.
///
/// # Errors
/// Returns `DimensionError` if the channels differ in shape.
pub fn li_intensity_correlation(
    channel1: &[Vec<f64>],
    channel2: &[Vec<f64>],
) -> NdimageResult<f64> {
    validate_equal_shape(channel1, channel2)?;
    let (a, b) = flatten_pair(channel1, channel2);
    let n = a.len();
    if n == 0 { return Ok(0.0); }
    let ma = mean(&a);
    let mb = mean(&b);
    let mut pos = 0i64;
    let mut neg = 0i64;
    for i in 0..n {
        let prod = (a[i] - ma) * (b[i] - mb);
        if prod > 0.0 { pos += 1; }
        else if prod < 0.0 { neg += 1; }
    }
    Ok((pos - neg) as f64 / n as f64)
}

/// Costes automatic threshold.
///
/// Iteratively lower a joint threshold `(t1, t2)` (linear relationship
/// `t1 = slope * t2 + intercept`, derived from linear regression) until the
/// Pearson coefficient of pixels *below* both thresholds becomes ≤ 0.
///
/// Returns `(t1, t2)` — the thresholds for channel 1 and channel 2.
///
/// # Errors
/// Returns `DimensionError` if channels differ in shape.
/// Returns `ComputationError` if the image is too uniform for regression.
pub fn costes_threshold(
    channel1: &[Vec<f64>],
    channel2: &[Vec<f64>],
) -> NdimageResult<(f64, f64)> {
    validate_equal_shape(channel1, channel2)?;
    let (a, b) = flatten_pair(channel1, channel2);
    let n = a.len();
    if n == 0 {
        return Err(NdimageError::InvalidInput("Empty image".into()));
    }

    // Compute linear regression: b = slope * a + intercept.
    let ma = mean(&a);
    let mb = mean(&b);
    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    for i in 0..n {
        cov   += (a[i] - ma) * (b[i] - mb);
        var_a += (a[i] - ma).powi(2);
    }
    if var_a < 1e-300 {
        return Err(NdimageError::ComputationError(
            "channel1 has zero variance — cannot compute Costes threshold".into(),
        ));
    }
    let slope     = cov / var_a;
    let intercept = mb - slope * ma;

    // Find max of each channel.
    let max_a = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_b = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Binary search on t2 in [0, max_b].
    let n_steps = 256usize;
    let mut best_t1 = max_a;
    let mut best_t2 = max_b;

    for step in 0..=n_steps {
        let t2 = max_b * (1.0 - step as f64 / n_steps as f64);
        let t1 = slope * t2 + intercept;

        // Collect pixels below both thresholds.
        let below_a: Vec<f64> = (0..n).filter(|&i| a[i] <= t1 && b[i] <= t2).map(|i| a[i]).collect();
        let below_b: Vec<f64> = (0..n).filter(|&i| a[i] <= t1 && b[i] <= t2).map(|i| b[i]).collect();

        if below_a.len() < 2 { break; }

        let r = pearson_vectors(&below_a, &below_b);
        if r <= 0.0 {
            best_t1 = t1;
            best_t2 = t2;
            break;
        }
    }

    Ok((best_t1, best_t2))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn const_channel(val: f64, rows: usize, cols: usize) -> Vec<Vec<f64>> {
        vec![vec![val; cols]; rows]
    }

    fn make_channel(data: &[&[f64]]) -> Vec<Vec<f64>> {
        data.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn test_pearson_identical() {
        let ch = make_channel(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let r = pearson_colocalization(&ch, &ch).expect("pearson failed");
        assert!((r - 1.0).abs() < 1e-10, "identical channels → r=1, got {}", r);
    }

    #[test]
    fn test_pearson_anticorrelated() {
        let ch1 = make_channel(&[&[1.0, 2.0, 3.0]]);
        let ch2 = make_channel(&[&[3.0, 2.0, 1.0]]);
        let r = pearson_colocalization(&ch1, &ch2).expect("pearson failed");
        assert!((r + 1.0).abs() < 1e-10, "anti-correlated → r=-1, got {}", r);
    }

    #[test]
    fn test_pearson_zero_variance() {
        let ch1 = const_channel(1.0, 3, 3);
        let ch2 = const_channel(2.0, 3, 3);
        let r = pearson_colocalization(&ch1, &ch2).expect("pearson failed");
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_manders_perfect() {
        // ch1 bright exactly where ch2 is bright.
        let ch1 = make_channel(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let ch2 = make_channel(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let (m1, m2) = manders_coefficients(&ch1, &ch2, 0.5, 0.5).expect("manders failed");
        assert!((m1 - 1.0).abs() < 1e-10);
        assert!((m2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_manders_no_overlap() {
        let ch1 = make_channel(&[&[1.0, 0.0]]);
        let ch2 = make_channel(&[&[0.0, 1.0]]);
        let (m1, m2) = manders_coefficients(&ch1, &ch2, 0.5, 0.5).expect("manders failed");
        assert!((m1 - 0.0).abs() < 1e-10);
        assert!((m2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_li_icq_colocalized() {
        // Channels where high values coincide.
        let ch1 = make_channel(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let ch2 = make_channel(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let icq = li_intensity_correlation(&ch1, &ch2).expect("li icq failed");
        // Perfect colocalization → positive ICQ.
        assert!(icq > 0.0, "expected positive ICQ, got {}", icq);
    }

    #[test]
    fn test_li_icq_range() {
        let ch1 = make_channel(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let ch2 = make_channel(&[&[6.0, 5.0, 4.0], &[3.0, 2.0, 1.0]]);
        let icq = li_intensity_correlation(&ch1, &ch2).expect("li icq failed");
        assert!(icq >= -0.5 && icq <= 0.5, "ICQ out of range: {}", icq);
    }

    #[test]
    fn test_costes_threshold_returns_valid_range() {
        // Build a simple synthetic colocalized image.
        let mut ch1 = vec![vec![0.0f64; 20]; 20];
        let mut ch2 = vec![vec![0.0f64; 20]; 20];
        for r in 0..20 {
            for c in 0..20 {
                ch1[r][c] = (r * 20 + c) as f64;
                ch2[r][c] = (r * 20 + c) as f64 + 1.0; // perfectly correlated
            }
        }
        let (t1, t2) = costes_threshold(&ch1, &ch2).expect("costes failed");
        assert!(t1 >= 0.0 && t2 >= 0.0, "thresholds must be non-negative: t1={}, t2={}", t1, t2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let ch1 = vec![vec![1.0, 2.0]; 3];
        let ch2 = vec![vec![1.0, 2.0]; 4];
        assert!(pearson_colocalization(&ch1, &ch2).is_err());
    }
}
