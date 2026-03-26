//! MCMC move types for Path Integral Monte Carlo.
//!
//! Provides three move strategies:
//!
//! * [`SingleBeadMove`]    — displace one bead by a uniform random vector.
//! * [`CenterOfMassMove`]  — displace an entire ring polymer (all slices of one particle).
//! * [`BisectionMove`]     — multi-level Lévy bisection of a randomly chosen sub-path.
//!
//! All moves implement the [`PimcMove`] trait and use Metropolis acceptance.

use crate::pimc::paths::{squared_distance, RingPolymer};
use scirs2_core::random::{Distribution, Normal, Rng, RngExt, Uniform};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Trait implemented by every PIMC move type.
pub trait PimcMove {
    /// Propose a move, evaluate acceptance via Metropolis, and apply it if accepted.
    ///
    /// Returns `true` when the move was accepted, `false` otherwise.
    fn propose_and_accept(
        &self,
        polymer: &mut RingPolymer,
        potential: &dyn Fn(&[f64]) -> f64,
        mass: f64,
        tau: f64,
        rng: &mut dyn RngProxy,
    ) -> bool;
}

/// Object-safe wrapper so that move implementations can accept `&mut dyn RngProxy`
/// rather than a generic `R: Rng` (which is not object-safe).
pub trait RngProxy {
    fn next_f64(&mut self) -> f64;
    fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64;
    fn next_usize(&mut self, n: usize) -> usize;
    fn next_normal(&mut self, mean: f64, std: f64) -> f64;
}

/// Blanket impl: any `R: Rng` can be used as a `dyn RngProxy`.
impl<R: Rng> RngProxy for R {
    fn next_f64(&mut self) -> f64 {
        self.random::<f64>()
    }
    fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        Uniform::new(lo, hi)
            .expect("valid uniform range")
            .sample(self)
    }
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        Uniform::new(0usize, n)
            .expect("valid uniform usize range")
            .sample(self)
    }
    fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        if std <= 0.0 {
            return mean;
        }
        Normal::new(mean, std)
            .expect("valid normal params")
            .sample(self)
    }
}

// ── Helper ────────────────────────────────────────────────────────────────────

/// Metropolis acceptance probability for a proposed change in action `Δ S`.
///
/// Returns `true` with probability `min(1, exp(-Δ S))`.
#[inline]
fn metropolis_accept(delta_action: f64, rng: &mut dyn RngProxy) -> bool {
    if delta_action <= 0.0 {
        return true; // always accept downhill moves
    }
    rng.next_f64() < (-delta_action).exp()
}

/// Local action of a single bead at particle `p`, slice `s`:
///
/// ```text
/// s_local(p, s) = (m / 2τ) · (|r_{p,s} − r_{p,s−1}|² + |r_{p,s+1} − r_{p,s}|²) + τ · V(r_{p,s})
/// ```
fn local_action(
    polymer: &RingPolymer,
    p: usize,
    s: usize,
    potential: &dyn Fn(&[f64]) -> f64,
    mass: f64,
    tau: f64,
) -> f64 {
    let m = polymer.n_slices;
    let s_prev = (s + m - 1) % m;
    let s_next = (s + 1) % m;
    let spring = (mass / (2.0 * tau))
        * (squared_distance(&polymer.beads[p][s], &polymer.beads[p][s_prev])
            + squared_distance(&polymer.beads[p][s], &polymer.beads[p][s_next]));
    spring + tau * potential(&polymer.beads[p][s])
}

// ── Single-bead move ──────────────────────────────────────────────────────────

/// Displace a single randomly chosen bead by a uniform random vector in
/// `[-max_displacement, max_displacement]^d` and accept/reject via Metropolis.
#[derive(Debug, Clone)]
pub struct SingleBeadMove {
    /// Maximum per-dimension displacement magnitude.
    pub max_displacement: f64,
}

impl PimcMove for SingleBeadMove {
    fn propose_and_accept(
        &self,
        polymer: &mut RingPolymer,
        potential: &dyn Fn(&[f64]) -> f64,
        mass: f64,
        tau: f64,
        rng: &mut dyn RngProxy,
    ) -> bool {
        let p = rng.next_usize(polymer.n_particles);
        let s = rng.next_usize(polymer.n_slices);

        let old_action = local_action(polymer, p, s, potential, mass, tau);

        // Build proposed position
        let old_pos = polymer.beads[p][s].clone();
        let proposed: Vec<f64> = old_pos
            .iter()
            .map(|&x| x + rng.next_f64_range(-self.max_displacement, self.max_displacement))
            .collect();

        polymer.beads[p][s] = proposed;
        let new_action = local_action(polymer, p, s, potential, mass, tau);

        if metropolis_accept(new_action - old_action, rng) {
            true
        } else {
            // Reject: restore original position
            polymer.beads[p][s] = old_pos;
            false
        }
    }
}

// ── Centre-of-mass move ───────────────────────────────────────────────────────

/// Shift the entire ring polymer of one particle (all `M` slices) by the same
/// random displacement vector.  The kinetic (spring) action is invariant under
/// such a global shift; only the potential action changes.
#[derive(Debug, Clone)]
pub struct CenterOfMassMove {
    /// Maximum per-dimension displacement magnitude.
    pub max_displacement: f64,
}

impl PimcMove for CenterOfMassMove {
    fn propose_and_accept(
        &self,
        polymer: &mut RingPolymer,
        potential: &dyn Fn(&[f64]) -> f64,
        mass: f64,
        tau: f64,
        rng: &mut dyn RngProxy,
    ) -> bool {
        let p = rng.next_usize(polymer.n_particles);

        // Old potential action for this particle
        let old_pot: f64 = (0..polymer.n_slices)
            .map(|s| potential(&polymer.beads[p][s]))
            .sum::<f64>()
            * tau;

        // Draw displacement vector (same for every slice)
        let delta: Vec<f64> = (0..polymer.dimension)
            .map(|_| rng.next_f64_range(-self.max_displacement, self.max_displacement))
            .collect();

        // Apply shift to all slices
        let old_beads: Vec<Vec<f64>> = polymer.beads[p].clone();
        for s in 0..polymer.n_slices {
            for d in 0..polymer.dimension {
                polymer.beads[p][s][d] += delta[d];
            }
        }

        // New potential action for this particle
        let new_pot: f64 = (0..polymer.n_slices)
            .map(|s| potential(&polymer.beads[p][s]))
            .sum::<f64>()
            * tau;

        // Spring action is unchanged → Δ S = Δ S_V only
        let delta_action = new_pot - old_pot;
        let _ = mass; // not needed for this move

        if metropolis_accept(delta_action, rng) {
            true
        } else {
            polymer.beads[p] = old_beads;
            false
        }
    }
}

// ── Multi-level bisection move ────────────────────────────────────────────────

/// Multi-level Lévy bisection move.
///
/// Selects a contiguous sub-path of length `2^max_level` slices for one
/// particle and resamples the internal beads using the Lévy bridge construction,
/// then accepts/rejects with Metropolis.
#[derive(Debug, Clone)]
pub struct BisectionMove {
    /// Number of bisection levels.  The sub-path spans `2^max_level` steps.
    /// Default: 3.
    pub max_level: usize,
}

impl Default for BisectionMove {
    fn default() -> Self {
        Self { max_level: 3 }
    }
}

impl BisectionMove {
    /// Compute the total (kinetic + potential) action for a contiguous slice
    /// range `[s_start, s_end)` (wrapping modulo `M`) for particle `p`.
    fn sub_path_action(
        polymer: &RingPolymer,
        p: usize,
        s_start: usize,
        n_sub: usize,
        potential: &dyn Fn(&[f64]) -> f64,
        mass: f64,
        tau: f64,
    ) -> f64 {
        let m = polymer.n_slices;
        let mut action = 0.0_f64;
        for k in 0..n_sub {
            let s = (s_start + k) % m;
            let s_next = (s_start + k + 1) % m;
            action += (mass / (2.0 * tau))
                * squared_distance(&polymer.beads[p][s], &polymer.beads[p][s_next]);
            action += tau * potential(&polymer.beads[p][s]);
        }
        action
    }
}

impl PimcMove for BisectionMove {
    fn propose_and_accept(
        &self,
        polymer: &mut RingPolymer,
        potential: &dyn Fn(&[f64]) -> f64,
        mass: f64,
        tau: f64,
        rng: &mut dyn RngProxy,
    ) -> bool {
        let m = polymer.n_slices;
        let n_sub = 1usize << self.max_level; // 2^max_level steps
        if n_sub >= m {
            // Sub-path longer than the entire ring → fall back gracefully
            return false;
        }

        let p = rng.next_usize(polymer.n_particles);
        let s_start = rng.next_usize(m);
        let s_end = (s_start + n_sub) % m;

        // Old action for the affected sub-path
        let old_action = Self::sub_path_action(polymer, p, s_start, n_sub, potential, mass, tau);

        // Save the old intermediate beads (not the fixed endpoints)
        let old_beads: Vec<Vec<f64>> = (0..n_sub - 1)
            .map(|k| polymer.beads[p][(s_start + 1 + k) % m].clone())
            .collect();

        // Resample via Lévy bridge between fixed endpoints s_start and s_end
        let start = polymer.beads[p][s_start].clone();
        let end = polymer.beads[p][s_end].clone();
        let bridge = levy_bridge_with_rng_proxy(&start, &end, n_sub - 1, mass, tau, rng);

        for (k, pos) in bridge.into_iter().enumerate() {
            let s = (s_start + 1 + k) % m;
            polymer.beads[p][s] = pos;
        }

        let new_action = Self::sub_path_action(polymer, p, s_start, n_sub, potential, mass, tau);

        if metropolis_accept(new_action - old_action, rng) {
            true
        } else {
            // Restore
            for (k, old) in old_beads.into_iter().enumerate() {
                let s = (s_start + 1 + k) % m;
                polymer.beads[p][s] = old;
            }
            false
        }
    }
}

/// Lévy bridge using `dyn RngProxy` (mirrors `RingPolymer::levy_bridge` but
/// works with object-safe RNG).
fn levy_bridge_with_rng_proxy(
    start: &[f64],
    end: &[f64],
    n_beads: usize,
    mass: f64,
    tau: f64,
    rng: &mut dyn RngProxy,
) -> Vec<Vec<f64>> {
    let dim = start.len();
    let n_total = n_beads + 1;

    (1..=n_beads)
        .map(|k| {
            let k_f = k as f64;
            let n_f = n_total as f64;
            let alpha = k_f / n_f;
            let variance = (tau / mass) * k_f * (n_f - k_f) / n_f;
            let std_dev = variance.max(0.0).sqrt();

            (0..dim)
                .map(|d| {
                    let mean = start[d] + alpha * (end[d] - start[d]);
                    rng.next_normal(mean, std_dev)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pimc::paths::{make_rng, PimcRng};

    fn make_flat_polymer(n_particles: usize, n_slices: usize, dim: usize, val: f64) -> RingPolymer {
        RingPolymer {
            beads: vec![vec![vec![val; dim]; n_slices]; n_particles],
            n_particles,
            n_slices,
            dimension: dim,
        }
    }

    #[test]
    fn test_single_bead_move_returns_bool() {
        let mut poly = make_flat_polymer(1, 4, 1, 0.0);
        let mv = SingleBeadMove {
            max_displacement: 0.5,
        };
        let mut rng: PimcRng = make_rng(1);
        let result = mv.propose_and_accept(&mut poly, &|_| 0.0, 1.0, 0.25, &mut rng);
        // At V=0, kinetic action is the only difference; result is a valid bool.
        let _ = result;
    }

    #[test]
    fn test_com_move_constant_potential_always_accepted() {
        // With V=0 the COM move changes no action → always accepted
        let mut poly = make_flat_polymer(1, 8, 1, 0.0);
        let mv = CenterOfMassMove {
            max_displacement: 0.5,
        };
        let mut rng: PimcRng = make_rng(2);
        for _ in 0..20 {
            let accepted = mv.propose_and_accept(&mut poly, &|_| 0.0, 1.0, 0.125, &mut rng);
            assert!(
                accepted,
                "COM move on V=0 potential should always be accepted"
            );
        }
    }

    #[test]
    fn test_com_move_all_beads_shifted() {
        // For zero potential every COM move is accepted; verify all beads shift equally
        let mut poly = make_flat_polymer(1, 4, 1, 0.0);
        // Override: set a specific starting position so we can detect the shift
        for s in 0..4 {
            poly.beads[0][s][0] = 1.0;
        }
        let mv = CenterOfMassMove {
            max_displacement: 0.3,
        };
        let mut rng: PimcRng = make_rng(99);
        mv.propose_and_accept(&mut poly, &|_| 0.0, 1.0, 0.25, &mut rng);
        // All beads should share the same value (since we started uniform and δ is the same)
        let first = poly.beads[0][0][0];
        for s in 1..4 {
            assert!(
                (poly.beads[0][s][0] - first).abs() < 1e-12,
                "COM shift inconsistent at slice {s}"
            );
        }
    }
}
