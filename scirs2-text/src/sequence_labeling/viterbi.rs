//! Generic Viterbi and beam search decoding algorithms.
//!
//! Provides efficient implementations of Viterbi decoding for sequence labeling
//! tasks, along with beam search as an approximate alternative.

use crate::error::{Result, TextError};

/// Perform exact Viterbi decoding over a sequence.
///
/// # Arguments
/// * `n_steps`   – number of time steps (sequence length)
/// * `n_states`  – number of label states
/// * `log_start` – log initial probabilities `log_start[s]`
/// * `log_trans`  – `log_trans[s1][s2]` = log P(s2 | s1)
/// * `log_emit`   – `log_emit[t][s]`   = log P(observation_t | state s)
///
/// # Returns
/// `(best_score, best_path)` where `best_path[t]` is the state index at step `t`.
pub fn viterbi_decode(
    n_steps: usize,
    n_states: usize,
    log_start: &[f64],
    log_trans: &[Vec<f64>],
    log_emit: &[Vec<f64>],
) -> Result<(f64, Vec<usize>)> {
    if n_steps == 0 {
        return Ok((0.0, vec![]));
    }
    if log_start.len() != n_states {
        return Err(TextError::InvalidInput(format!(
            "log_start length {} != n_states {}",
            log_start.len(),
            n_states
        )));
    }
    if log_trans.len() != n_states {
        return Err(TextError::InvalidInput(format!(
            "log_trans rows {} != n_states {}",
            log_trans.len(),
            n_states
        )));
    }
    if log_emit.len() != n_steps {
        return Err(TextError::InvalidInput(format!(
            "log_emit rows {} != n_steps {}",
            log_emit.len(),
            n_steps
        )));
    }

    // dp[t][s] = best log-prob path ending at state s at time t
    let mut dp = vec![vec![f64::NEG_INFINITY; n_states]; n_steps];
    // backpointer[t][s] = predecessor state
    let mut bp = vec![vec![0usize; n_states]; n_steps];

    // Initialise
    for s in 0..n_states {
        let emit = log_emit[0].get(s).copied().unwrap_or(f64::NEG_INFINITY);
        dp[0][s] = log_start[s] + emit;
    }

    // Recursion
    for t in 1..n_steps {
        for s in 0..n_states {
            let emit = log_emit[t].get(s).copied().unwrap_or(f64::NEG_INFINITY);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_prev = 0usize;
            for prev in 0..n_states {
                let trans = log_trans[prev].get(s).copied().unwrap_or(f64::NEG_INFINITY);
                let score = dp[t - 1][prev] + trans;
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }
            dp[t][s] = best_score + emit;
            bp[t][s] = best_prev;
        }
    }

    // Find best final state
    let mut best_final_score = f64::NEG_INFINITY;
    let mut best_final_state = 0usize;
    for s in 0..n_states {
        if dp[n_steps - 1][s] > best_final_score {
            best_final_score = dp[n_steps - 1][s];
            best_final_state = s;
        }
    }

    // Back-trace
    let mut path = vec![0usize; n_steps];
    path[n_steps - 1] = best_final_state;
    for t in (1..n_steps).rev() {
        path[t - 1] = bp[t][path[t]];
    }

    Ok((best_final_score, path))
}

/// Hypothesis used internally by beam search.
#[derive(Clone, Debug)]
struct Hypothesis {
    score: f64,
    path: Vec<usize>,
}

/// Approximate decoding via beam search.
///
/// # Arguments
/// * `n_steps`    – sequence length
/// * `n_states`   – number of label states
/// * `score_fn`   – closure `(t, prev_state, curr_state) -> score` (log-domain additive)
/// * `beam_width` – number of hypotheses to retain at each step
///
/// # Returns
/// The best path found, as a `Vec<usize>` of state indices.
pub fn beam_search<F>(
    n_steps: usize,
    n_states: usize,
    score_fn: F,
    beam_width: usize,
) -> Result<Vec<usize>>
where
    F: Fn(usize, usize, usize) -> f64,
{
    if n_steps == 0 {
        return Ok(vec![]);
    }
    if beam_width == 0 {
        return Err(TextError::InvalidInput(
            "beam_width must be > 0".to_string(),
        ));
    }

    // Initialise beam at t=0
    let mut beam: Vec<Hypothesis> = (0..n_states)
        .map(|s| Hypothesis {
            score: score_fn(0, 0, s),
            path: vec![s],
        })
        .collect();

    // Sort descending and truncate to beam_width
    beam.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    beam.truncate(beam_width);

    for t in 1..n_steps {
        let mut candidates: Vec<Hypothesis> = Vec::with_capacity(beam.len() * n_states);
        for hyp in &beam {
            let prev_state = *hyp.path.last().unwrap_or(&0);
            for s in 0..n_states {
                let delta = score_fn(t, prev_state, s);
                let mut new_path = hyp.path.clone();
                new_path.push(s);
                candidates.push(Hypothesis {
                    score: hyp.score + delta,
                    path: new_path,
                });
            }
        }
        candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(beam_width);
        beam = candidates;
    }

    let best = beam
        .into_iter()
        .next()
        .ok_or_else(|| TextError::ProcessingError("Beam is empty".to_string()))?;
    Ok(best.path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_viterbi_trivial() {
        // 1 step, 2 states
        let log_start = [0.0_f64.ln(), 1.0_f64.ln()]; // impossible, but use -inf / 0
        let log_start = [f64::NEG_INFINITY, 0.0];
        let log_trans: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let log_emit: Vec<Vec<f64>> = vec![vec![0.0, f64::NEG_INFINITY]];

        let (score, path) = viterbi_decode(1, 2, &log_start, &log_trans, &log_emit)
            .expect("viterbi failed");
        // Best start is state 1 (log_start=0), but emit forces state 0 to win?
        // log_start[0]=-inf → state 0 path: -inf+0 = -inf
        // log_start[1]=0    → state 1 path: 0+(-inf) = -inf
        // Both are -inf; algorithm picks whichever comes first when equal
        assert_eq!(path.len(), 1);
        let _ = score;
    }

    #[test]
    fn test_viterbi_two_steps() {
        // Simple 2-state, 2-step HMM
        // State 0 = "H" (heads), State 1 = "T" (tails)
        let log_start = [(-2.0_f64).exp().ln(), (-2.0_f64).exp().ln()];
        // Equal transition
        let log_trans = vec![
            vec![-2.0_f64.ln(), -2.0_f64.ln()],
            vec![-2.0_f64.ln(), -2.0_f64.ln()],
        ];
        // Emission: step 0 strongly favours state 0, step 1 strongly favours state 1
        let log_emit = vec![
            vec![0.0, -10.0],   // t=0: state 0
            vec![-10.0, 0.0],   // t=1: state 1
        ];
        let (score, path) = viterbi_decode(2, 2, &log_start, &log_trans, &log_emit)
            .expect("viterbi failed");
        assert_eq!(path, vec![0, 1]);
        assert!(score.is_finite());
    }

    #[test]
    fn test_beam_search_matches_viterbi() {
        // Build a scoring function equivalent to a small HMM
        let log_start = [0.0, f64::NEG_INFINITY];
        let log_trans = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let log_emit = vec![vec![0.0, -5.0], vec![-5.0, 0.0], vec![0.0, -5.0]];

        let score_fn = |t: usize, prev: usize, curr: usize| -> f64 {
            let trans = if t == 0 {
                log_start[curr]
            } else {
                log_trans[prev][curr]
            };
            trans + log_emit[t][curr]
        };

        let path_beam = beam_search(3, 2, score_fn, 4).expect("beam failed");
        let (_, path_viterbi) =
            viterbi_decode(3, 2, &log_start, &log_trans, &log_emit).expect("viterbi failed");

        assert_eq!(path_beam, path_viterbi);
    }

    #[test]
    fn test_empty_sequence() {
        let (score, path) =
            viterbi_decode(0, 2, &[], &[], &[]).expect("viterbi on empty");
        assert_eq!(path.len(), 0);
        approx_eq(score, 0.0, 1e-9);
    }

    #[test]
    fn test_beam_search_empty() {
        let path = beam_search(0, 2, |_, _, _| 0.0, 3).expect("beam on empty");
        assert!(path.is_empty());
    }
}
