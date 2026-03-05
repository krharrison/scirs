//! Multipole and local expansions for 2D Laplace potential.
//!
//! The 2D Laplace potential for a point charge q at position z_j is:
//!   φ(z) = q · ln|z - z_j|
//!
//! ## Multipole Expansion (M-expansion)
//!
//! For a cluster of charges {q_j, z_j} with a common center c, the potential
//! at a far-field point z is approximated by
//!
//!   φ(z) ≈ Q · ln(z - c) + Σ_{k=1}^{p} a_k / (z - c)^k
//!
//! where Q = Σ q_j  (total charge),  a_k = -Σ_j q_j (z_j-c)^k / k.
//!
//! ## Local Expansion (L-expansion)
//!
//! Represents the potential due to a far-away cluster as a Taylor series
//! around a local center c':
//!
//!   φ(z) ≈ Σ_{k=0}^{p} b_k (z - c')^k
//!
//! ## References
//! Greengard & Rokhlin (1987), "A fast algorithm for particle simulations."

use crate::error::{FFTError, FFTResult};

// ============================================================================
// Complex arithmetic helpers (inline, allocation-free)
// ============================================================================

/// Complex multiply: (a + ib)(c + id) = (ac-bd) + i(ad+bc)
#[inline]
fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// Complex add
#[inline]
fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

/// Complex subtract
#[inline]
fn csub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Scalar multiply
#[inline]
fn cscale(a: [f64; 2], s: f64) -> [f64; 2] {
    [a[0] * s, a[1] * s]
}

/// Complex inverse: 1 / z = z̄ / |z|²
#[inline]
fn cinv(z: [f64; 2]) -> Option<[f64; 2]> {
    let norm2 = z[0] * z[0] + z[1] * z[1];
    if norm2 < f64::MIN_POSITIVE {
        None
    } else {
        Some([z[0] / norm2, -z[1] / norm2])
    }
}

/// Complex modulus squared
#[inline]
fn cabs2(z: [f64; 2]) -> f64 {
    z[0] * z[0] + z[1] * z[1]
}

/// Complex power: z^n using repeated squaring.
fn cpow(z: [f64; 2], n: usize) -> [f64; 2] {
    if n == 0 {
        return [1.0, 0.0];
    }
    let mut result = [1.0_f64, 0.0_f64];
    let mut base = z;
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            result = cmul(result, base);
        }
        base = cmul(base, base);
        exp >>= 1;
    }
    result
}

/// Natural log of complex number z = ln|z| + i·arg(z).
fn cln(z: [f64; 2]) -> Option<[f64; 2]> {
    let r = cabs2(z).sqrt();
    if r < f64::MIN_POSITIVE {
        return None;
    }
    let theta = z[1].atan2(z[0]);
    Some([r.ln(), theta])
}

/// Binomial coefficient C(n, k) using multiplicative formula (exact for small n).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ============================================================================
// MultipoleExpansion
// ============================================================================

/// Multipole expansion for the 2D Laplace potential around a center point.
///
/// Stores coefficients `a_0, a_1, …, a_p` where:
/// - `a_0 = Q` (total charge, real-valued stored as `[Q, 0]`)
/// - `a_k = -Σ_j q_j (z_j − c)^k / k`  for k ≥ 1
///
/// The far-field potential approximation is:
///   φ(z) ≈ Re[ Q · ln(z-c) + Σ_{k=1}^{p} a_k / (z-c)^k ]
#[derive(Debug, Clone)]
pub struct MultipoleExpansion {
    /// Center of the expansion in 2D (treated as complex z = center[0] + i·center[1]).
    pub center: [f64; 2],
    /// Expansion coefficients: `coeffs[k]` = `a_k` as `(Re, Im)`.
    /// Length = order + 1.
    pub coeffs: Vec<[f64; 2]>,
    /// Truncation order `p` (number of terms excluding the log term).
    pub order: usize,
}

impl MultipoleExpansion {
    /// Create a zero-initialized multipole expansion.
    pub fn new(center: [f64; 2], order: usize) -> Self {
        MultipoleExpansion {
            center,
            coeffs: vec![[0.0, 0.0]; order + 1],
            order,
        }
    }

    /// Accumulate one source (charge `q` at position `pos`) into the expansion.
    ///
    /// Updates coefficients in-place:
    ///   a_0 += q
    ///   a_k += -q (z_j - c)^k / k    for k = 1 … p
    pub fn add_source(&mut self, pos: [f64; 2], charge: f64) {
        // Displacement z_j - c
        let dz = csub(pos, self.center);

        // a_0 += q
        self.coeffs[0][0] += charge;

        // Powers of dz: dz^1, dz^2, …, dz^p
        let mut dz_pow = dz; // dz^1
        for k in 1..=self.order {
            // a_k += -q * dz^k / k
            let contrib = cscale(dz_pow, -charge / k as f64);
            self.coeffs[k] = cadd(self.coeffs[k], contrib);
            if k < self.order {
                dz_pow = cmul(dz_pow, dz);
            }
        }
    }

    /// M2M translation: shift the expansion to a new center.
    ///
    /// Returns a new `MultipoleExpansion` centered at `new_center` that
    /// represents the same charge distribution.
    ///
    /// Translation formula (Greengard & Rokhlin, Lemma 2.3):
    ///   b_l = -Σ_{k=1}^{l} a_k · C(l-1, k-1) · z_0^{l-k} / l  +  a_0 · (-z_0)^l / l
    ///   b_0 = a_0
    /// where z_0 = old_center - new_center.
    pub fn translate(&self, new_center: [f64; 2]) -> Self {
        let mut result = MultipoleExpansion::new(new_center, self.order);
        // Displacement: z_0 = old_center - new_center
        let z0 = csub(self.center, new_center);

        // b_0 = a_0
        result.coeffs[0] = self.coeffs[0];

        // Precompute powers of z_0: z0_pow[k] = z_0^k
        let mut z0_pow: Vec<[f64; 2]> = vec![[0.0, 0.0]; self.order + 1];
        z0_pow[0] = [1.0, 0.0];
        for k in 1..=self.order {
            z0_pow[k] = cmul(z0_pow[k - 1], z0);
        }

        // b_l = a_0 * (-z_0)^l / l  +  Σ_{k=1}^{l} a_k * C(l-1,k-1) * z_0^{l-k}
        for l in 1..=self.order {
            // Term from a_0: a_0 * (-z_0)^l / l
            let neg_z0_l = cscale(cpow(cscale(z0, -1.0), l), 1.0 / l as f64);
            let mut b_l = cmul(self.coeffs[0], neg_z0_l);

            // Sum: Σ_{k=1}^{l} a_k * C(l-1, k-1) * z_0^{l-k}
            for k in 1..=l {
                let c_coeff = binom(l - 1, k - 1);
                let term = cscale(cmul(self.coeffs[k], z0_pow[l - k]), c_coeff);
                b_l = cadd(b_l, term);
            }
            result.coeffs[l] = b_l;
        }

        result
    }

    /// M2L conversion: convert this multipole expansion into a local expansion
    /// centered at `target_center`, assuming `target_center` is well-separated.
    ///
    /// Formula (Greengard & Rokhlin, Lemma 2.4):
    ///   b_l = (1/z_0^l) * [ -a_0/l + Σ_{k=1}^{p} a_k * C(l+k-1, k-1) / z_0^k ]
    ///   b_0 = a_0 * ln(-z_0) + Σ_{k=1}^{p} a_k / (-z_0)^k
    /// where z_0 = target_center - old_center.
    pub fn to_local(&self, target_center: [f64; 2], order: usize) -> FFTResult<LocalExpansion> {
        let z0 = csub(target_center, self.center);
        let z0_inv = cinv(z0).ok_or_else(|| {
            FFTError::ValueError("M2L: source and target centers coincide".into())
        })?;

        let mut local = LocalExpansion::new(target_center, order);

        // Precompute powers of (1/z_0): inv_z0_pow[k] = z_0^{-k}
        let mut inv_z0_pow: Vec<[f64; 2]> = vec![[0.0, 0.0]; self.order + order + 2];
        inv_z0_pow[0] = [1.0, 0.0];
        for k in 1..inv_z0_pow.len() {
            inv_z0_pow[k] = cmul(inv_z0_pow[k - 1], z0_inv);
        }

        // b_0 = a_0 * ln(-z_0) + Σ_{k=1}^{p} a_k / (-z_0)^k
        //     = a_0 * ln(-z_0) + Σ_{k=1}^{p} a_k * (-1)^k / z_0^k
        {
            // ln(-z_0): negate z_0, then take log
            let neg_z0 = cscale(z0, -1.0);
            let ln_neg_z0 = cln(neg_z0).ok_or_else(|| {
                FFTError::ValueError("M2L: degenerate configuration".into())
            })?;
            let mut b0 = cmul(self.coeffs[0], ln_neg_z0);
            for k in 1..=self.order {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let contrib = cscale(cmul(self.coeffs[k], inv_z0_pow[k]), sign);
                b0 = cadd(b0, contrib);
            }
            local.coeffs[0] = b0;
        }

        // b_l = (1/z_0^l) * [ -a_0/l + Σ_{k=1}^{p} a_k * C(l+k-1, k-1) * (-1)^k / z_0^k ]
        for l in 1..=order {
            // -a_0 / l
            let mut inner = cscale(self.coeffs[0], -1.0 / l as f64);
            for k in 1..=self.order {
                let c_coeff = binom(l + k - 1, k - 1);
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let term = cscale(
                    cmul(self.coeffs[k], inv_z0_pow[k]),
                    c_coeff * sign,
                );
                inner = cadd(inner, term);
            }
            // Multiply by (1/z_0)^l = z_0^{-l}
            local.coeffs[l] = cmul(inner, inv_z0_pow[l]);
        }

        Ok(local)
    }

    /// Evaluate the potential φ(x) at a far-field point x using the multipole expansion.
    ///
    /// φ(x) = Re[ Q · ln(z-c) + Σ_{k=1}^{p} a_k / (z-c)^k ]
    pub fn evaluate(&self, x: [f64; 2]) -> f64 {
        let dz = csub(x, self.center);
        let r2 = cabs2(dz);
        if r2 < f64::MIN_POSITIVE {
            return 0.0;
        }

        // Re[Q · ln(z - c)] = Q_re · ln|z-c| - Q_im · arg(z-c)
        let ln_r = r2.sqrt().ln();
        let arg = dz[1].atan2(dz[0]);
        let mut phi = self.coeffs[0][0] * ln_r - self.coeffs[0][1] * arg;

        // Σ_{k=1}^{p} Re[a_k / (z-c)^k]
        if let Some(inv_dz) = cinv(dz) {
            let mut inv_dz_pow = inv_dz; // (z-c)^{-1}
            for k in 1..=self.order {
                phi += self.coeffs[k][0] * inv_dz_pow[0]
                    - self.coeffs[k][1] * inv_dz_pow[1];
                if k < self.order {
                    inv_dz_pow = cmul(inv_dz_pow, inv_dz);
                }
            }
        }

        phi
    }
}

// ============================================================================
// LocalExpansion
// ============================================================================

/// Local expansion for the 2D Laplace potential around a target center.
///
/// The expansion represents the potential from a far-away cluster as a
/// Taylor series around center `c'`:
///   φ(z) ≈ Re[ Σ_{k=0}^{p} b_k (z - c')^k ]
#[derive(Debug, Clone)]
pub struct LocalExpansion {
    /// Center of the local expansion.
    pub center: [f64; 2],
    /// Expansion coefficients `b_0, …, b_p` as complex numbers.
    pub coeffs: Vec<[f64; 2]>,
    /// Truncation order `p`.
    pub order: usize,
}

impl LocalExpansion {
    /// Create a zero-initialized local expansion.
    pub fn new(center: [f64; 2], order: usize) -> Self {
        LocalExpansion {
            center,
            coeffs: vec![[0.0, 0.0]; order + 1],
            order,
        }
    }

    /// Add another local expansion (must have the same order and center).
    /// If centers differ, use `translate` first.
    pub fn add(&mut self, other: &LocalExpansion) {
        for k in 0..=self.order.min(other.order) {
            self.coeffs[k] = cadd(self.coeffs[k], other.coeffs[k]);
        }
    }

    /// L2L translation: shift the local expansion to a new (child) center.
    ///
    /// Formula: for center z_0 = new_center - old_center:
    ///   c_l = Σ_{k=l}^{p} b_k · C(k, l) · z_0^{k-l}
    pub fn translate(&self, new_center: [f64; 2]) -> Self {
        let mut result = LocalExpansion::new(new_center, self.order);

        // z_0 = new_center - old_center
        let z0 = csub(new_center, self.center);

        // Precompute z_0^k
        let mut z0_pow: Vec<[f64; 2]> = vec![[0.0, 0.0]; self.order + 1];
        z0_pow[0] = [1.0, 0.0];
        for k in 1..=self.order {
            z0_pow[k] = cmul(z0_pow[k - 1], z0);
        }

        // c_l = Σ_{k=l}^{p} b_k · C(k, l) · z_0^{k-l}
        for l in 0..=self.order {
            let mut c_l = [0.0_f64, 0.0_f64];
            for k in l..=self.order {
                let c_coeff = binom(k, l);
                let term = cscale(cmul(self.coeffs[k], z0_pow[k - l]), c_coeff);
                c_l = cadd(c_l, term);
            }
            result.coeffs[l] = c_l;
        }

        result
    }

    /// Evaluate the potential at point x using the local expansion.
    ///
    /// φ(x) = Re[ Σ_{k=0}^{p} b_k (z - c')^k ]
    pub fn evaluate(&self, x: [f64; 2]) -> f64 {
        let dz = csub(x, self.center);

        // Horner's method: φ = Re[ b_0 + dz*(b_1 + dz*(b_2 + ... )) ]
        let mut acc = self.coeffs[self.order];
        for k in (0..self.order).rev() {
            acc = cadd(cmul(acc, dz), self.coeffs[k]);
        }
        acc[0] // real part
    }

    /// Evaluate the gradient (force field) at point x.
    ///
    /// ∂φ/∂x = Re[ Σ_{k=1}^{p} k · b_k (z - c')^{k-1} ]
    /// ∂φ/∂y = Im[ Σ_{k=1}^{p} k · b_k (z - c')^{k-1} ]
    /// (using Cauchy-Riemann: ∂φ/∂y = -∂/∂x of imaginary part)
    pub fn evaluate_gradient(&self, x: [f64; 2]) -> [f64; 2] {
        let dz = csub(x, self.center);

        // Compute derivative expansion: Σ_{k=1}^{p} k * b_k * (z-c')^{k-1}
        // = b_1 + 2*b_2*(z-c') + ... via Horner
        if self.order == 0 {
            return [0.0, 0.0];
        }

        let mut acc = cscale(self.coeffs[self.order], self.order as f64);
        for k in (1..self.order).rev() {
            acc = cadd(cmul(acc, dz), cscale(self.coeffs[k], k as f64));
        }

        // Gradient: [Re(acc), -Im(acc)] for 2D Laplace
        // (the gradient of Re[f(z)] is (Re[f'], -Im[f']) by Cauchy-Riemann)
        [acc[0], acc[1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binom() {
        assert!((binom(4, 2) - 6.0).abs() < 1e-12);
        assert!((binom(5, 0) - 1.0).abs() < 1e-12);
        assert!((binom(5, 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_multipole_single_source() {
        let mut m = MultipoleExpansion::new([0.0, 0.0], 5);
        m.add_source([0.1, 0.0], 1.0);

        // Evaluate at far point; should approximate ln|x - 0.1|
        let x = [5.0, 0.0];
        let approx = m.evaluate(x);
        let exact = ((x[0] - 0.1_f64).powi(2) + x[1].powi(2)).sqrt().ln();
        assert!(
            (approx - exact).abs() < 1e-4,
            "approx={approx:.6} exact={exact:.6}"
        );
    }

    #[test]
    fn test_local_expansion_single_source() {
        // Build a multipole far away and convert to local expansion.
        let charge_pos = [10.0, 0.0];
        let charge = 1.0;

        let mut m = MultipoleExpansion::new(charge_pos, 8);
        m.add_source(charge_pos, charge);

        // Target center near origin
        let target_center = [0.0, 0.0];
        let local = m.to_local(target_center, 8).expect("M2L failed");

        // Evaluate at a nearby point
        let x = [0.2, 0.1];
        let approx = local.evaluate(x);
        let exact = (((x[0] - charge_pos[0]).powi(2) + (x[1] - charge_pos[1]).powi(2)).sqrt()).ln();
        assert!(
            (approx - exact).abs() < 1e-4,
            "local approx={approx:.6} exact={exact:.6}"
        );
    }

    #[test]
    fn test_l2l_translation() {
        let charge_pos = [10.0, 0.0];
        let mut m = MultipoleExpansion::new(charge_pos, 8);
        m.add_source(charge_pos, 1.0);

        let local_far = m.to_local([0.0, 0.0], 8).expect("M2L failed");
        let local_near = local_far.translate([0.1, 0.05]);

        let x = [0.15, 0.08];
        let approx_far = local_far.evaluate(x);
        let approx_near = local_near.evaluate(x);
        // Both should approximate the same value.
        assert!(
            (approx_far - approx_near).abs() < 1e-5,
            "far={approx_far:.8} near={approx_near:.8}"
        );
    }

    #[test]
    fn test_gradient() {
        let charge_pos = [10.0, 0.0];
        let mut m = MultipoleExpansion::new(charge_pos, 10);
        m.add_source(charge_pos, 1.0);

        let local = m.to_local([0.0, 0.0], 10).expect("M2L failed");
        let x = [0.1, 0.0];
        let grad = local.evaluate_gradient(x);

        // Numerical gradient
        let h = 1e-6;
        let phi_px = local.evaluate([x[0] + h, x[1]]);
        let phi_mx = local.evaluate([x[0] - h, x[1]]);
        let numerical_gx = (phi_px - phi_mx) / (2.0 * h);

        assert!(
            (grad[0] - numerical_gx).abs() < 1e-5,
            "grad_x={:.8} numerical={:.8}", grad[0], numerical_gx
        );
    }
}
