//! Correlated Topic Model — core model utilities.
//!
//! Provides the softmax link function, log-likelihood, correlation-matrix
//! extraction, and top-word retrieval used by CTM inference.

/// Numerically-stable softmax: subtracts the max before exponentiating.
///
/// # Arguments
/// * `eta` – raw logistic-normal coordinates (length K-1 or K)
///
/// # Returns
/// Probability simplex vector of the same length.
pub fn softmax(eta: &[f64]) -> Vec<f64> {
    if eta.is_empty() {
        return Vec::new();
    }
    let max_val = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = eta.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        // Degenerate: uniform fallback
        let k = eta.len() as f64;
        return vec![1.0 / k; eta.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-likelihood of a document under a topic-word model.
///
/// `doc[w]` is the count of word `w`; `theta[k]` is the document-topic
/// probability; `beta[k][w]` is the topic-word probability.
///
/// Returns `Σ_w count_w · log(Σ_k theta_k · beta_kw)`.
pub fn log_likelihood(doc: &[f64], theta: &[f64], beta: &[Vec<f64>]) -> f64 {
    let vocab_size = doc.len();
    let k = theta.len();
    let mut ll = 0.0_f64;
    for w in 0..vocab_size {
        if doc[w] <= 0.0 {
            continue;
        }
        let mut mix = 0.0_f64;
        for t in 0..k {
            if t < beta.len() && w < beta[t].len() {
                mix += theta[t] * beta[t][w];
            }
        }
        if mix > 0.0 {
            ll += doc[w] * mix.ln();
        }
    }
    ll
}

/// Compute the correlation matrix from a covariance matrix.
///
/// `corr[i][j] = sigma[i][j] / sqrt(sigma[i][i] * sigma[j][j])`
pub fn topic_correlation_matrix(sigma: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = sigma.len();
    let mut corr = vec![vec![0.0_f64; k]; k];
    let diag: Vec<f64> = (0..k).map(|i| sigma[i][i].max(1e-15).sqrt()).collect();
    for i in 0..k {
        for j in 0..k {
            corr[i][j] = sigma[i][j] / (diag[i] * diag[j]);
            // Clamp to [-1, 1] for numerical safety
            corr[i][j] = corr[i][j].clamp(-1.0, 1.0);
        }
    }
    corr
}

/// Return the top-`n` words for each topic, ordered by decreasing probability.
///
/// # Arguments
/// * `beta`  – topic-word matrix `K × V`
/// * `vocab` – vocabulary (length V)
/// * `n`     – number of words per topic
pub fn top_words(beta: &[Vec<f64>], vocab: &[String], n: usize) -> Vec<Vec<String>> {
    beta.iter()
        .map(|bk| {
            let vocab_size = bk.len().min(vocab.len());
            let mut idx_prob: Vec<(usize, f64)> = (0..vocab_size).map(|w| (w, bk[w])).collect();
            idx_prob.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx_prob
                .iter()
                .take(n)
                .map(|&(w, _)| vocab[w].clone())
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let eta = vec![1.0_f64, 2.0, 0.5];
        let p = softmax(&eta);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum={sum}");
        assert!(p.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn softmax_numerically_stable_large_inputs() {
        let eta = vec![1000.0_f64, 1001.0, 999.0];
        let p = softmax(&eta);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(p.iter().all(|&x| x.is_finite() && x >= 0.0));
    }

    #[test]
    fn softmax_empty() {
        let p = softmax(&[]);
        assert!(p.is_empty());
    }

    #[test]
    fn correlation_diagonal_is_one() {
        let sigma = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 9.0, 2.0],
            vec![0.5, 2.0, 1.0],
        ];
        let corr = topic_correlation_matrix(&sigma);
        for i in 0..3 {
            assert!(
                (corr[i][i] - 1.0).abs() < 1e-10,
                "diagonal[{i}]={}",
                corr[i][i]
            );
        }
    }

    #[test]
    fn top_words_returns_n_words() {
        let beta = vec![vec![0.5, 0.3, 0.2], vec![0.1, 0.7, 0.2]];
        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tw = top_words(&beta, &vocab, 2);
        assert_eq!(tw.len(), 2);
        assert_eq!(tw[0].len(), 2);
        assert_eq!(tw[0][0], "a"); // highest prob word in topic 0
        assert_eq!(tw[1][0], "b"); // highest prob word in topic 1
    }

    #[test]
    fn log_likelihood_trivial() {
        // Single document, single topic: theta=[1], beta=[[0.5,0.5]], doc=[2,2]
        let doc = vec![2.0_f64, 2.0];
        let theta = vec![1.0_f64];
        let beta = vec![vec![0.5_f64, 0.5]];
        let ll = log_likelihood(&doc, &theta, &beta);
        let expected = 4.0 * 0.5_f64.ln();
        assert!((ll - expected).abs() < 1e-10, "ll={ll} expected={expected}");
    }
}
