//! Dynamic Topic Model — model utilities.
//!
//! Provides helpers for inspecting topic-word trajectories over time.

/// Return the top-`n` words for each topic at time slice `t`.
///
/// # Arguments
/// * `trajectories` – `K × T × V` tensor (topic × time × word prob)
/// * `t`            – time index
/// * `vocab`        – vocabulary (length V)
/// * `n`            – words per topic
///
/// # Returns
/// `K` lists of up to `n` word strings.
pub fn top_words_at_time(
    trajectories: &[Vec<Vec<f64>>],
    t: usize,
    vocab: &[String],
    n: usize,
) -> Vec<Vec<String>> {
    trajectories
        .iter()
        .map(|topic_traj| {
            if t >= topic_traj.len() {
                return Vec::new();
            }
            let bkw = &topic_traj[t];
            let vocab_size = bkw.len().min(vocab.len());
            let mut idx_prob: Vec<(usize, f64)> = (0..vocab_size).map(|w| (w, bkw[w])).collect();
            idx_prob.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx_prob
                .iter()
                .take(n)
                .map(|&(w, _)| vocab[w].clone())
                .collect()
        })
        .collect()
}

/// Return the probability of `word_id` in `topic_id` across all time slices.
///
/// # Arguments
/// * `trajectories` – `K × T × V` tensor
/// * `topic_id`     – topic index (0-based)
/// * `word_id`      – word index (0-based)
///
/// # Returns
/// Vector of length T; empty if indices are out of range.
pub fn topic_evolution(
    trajectories: &[Vec<Vec<f64>>],
    topic_id: usize,
    word_id: usize,
) -> Vec<f64> {
    if topic_id >= trajectories.len() {
        return Vec::new();
    }
    let topic_traj = &trajectories[topic_id];
    topic_traj
        .iter()
        .map(|bkw| {
            if word_id < bkw.len() {
                bkw[word_id]
            } else {
                0.0
            }
        })
        .collect()
}

/// Normalise a slice in place to a probability simplex (L1 projection).
pub(crate) fn normalise_to_simplex(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 1e-15 {
        for x in v.iter_mut() {
            *x = (*x / s).max(1e-15);
        }
        // Re-normalise after clamping
        let s2: f64 = v.iter().sum();
        for x in v.iter_mut() {
            *x /= s2;
        }
    } else {
        let u = 1.0 / v.len() as f64;
        for x in v.iter_mut() {
            *x = u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectories(k: usize, t: usize, v: usize) -> Vec<Vec<Vec<f64>>> {
        (0..k)
            .map(|ki| {
                (0..t)
                    .map(|ti| {
                        let mut row: Vec<f64> =
                            (0..v).map(|wi| ((ki + ti + wi) % 5) as f64 + 0.1).collect();
                        let s: f64 = row.iter().sum();
                        row.iter_mut().for_each(|x| *x /= s);
                        row
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn top_words_at_time_returns_n_words() {
        let traj = make_trajectories(3, 4, 6);
        let vocab: Vec<String> = (0..6).map(|i| format!("w{i}")).collect();
        let tw = top_words_at_time(&traj, 2, &vocab, 3);
        assert_eq!(tw.len(), 3);
        for t in &tw {
            assert_eq!(t.len(), 3);
        }
    }

    #[test]
    fn topic_evolution_length_equals_t() {
        let traj = make_trajectories(2, 5, 4);
        let ev = topic_evolution(&traj, 0, 1);
        assert_eq!(ev.len(), 5);
    }

    #[test]
    fn topic_evolution_out_of_range() {
        let traj = make_trajectories(2, 5, 4);
        assert!(topic_evolution(&traj, 99, 0).is_empty());
    }

    #[test]
    fn normalise_to_simplex_works() {
        let mut v = vec![1.0_f64, 2.0, 3.0];
        normalise_to_simplex(&mut v);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }
}
