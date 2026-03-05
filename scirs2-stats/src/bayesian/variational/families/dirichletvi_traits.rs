//! # DirichletVI - Trait Implementations
//!
//! This module contains trait implementations for `DirichletVI`.
//!
//! ## Implemented Traits
//!
//! - `VariationalFamily`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;

use super::super::{digamma, lgamma};
use super::functions::VariationalFamily;
use super::types::DirichletVI;

impl VariationalFamily for DirichletVI {
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_params(&self) -> usize {
        self.dim
    }
    fn get_params(&self) -> Array1<f64> {
        self.log_alpha.clone()
    }
    fn set_params(&mut self, params: &Array1<f64>) -> Result<()> {
        if params.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "expected {} parameters, got {}",
                self.dim,
                params.len()
            )));
        }
        self.log_alpha = params.to_owned();
        Ok(())
    }
    fn sample_reparam(&self, epsilon: &Array1<f64>) -> Result<(Array1<f64>, f64)> {
        if epsilon.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "epsilon has length {}, expected {}",
                epsilon.len(),
                self.dim
            )));
        }
        let alphas = self.alphas();
        let mut gammas = Array1::zeros(self.dim);
        for i in 0..self.dim {
            gammas[i] = Self::gamma_reparam(alphas[i], epsilon[i]).max(1e-10);
        }
        let sum: f64 = gammas.sum();
        let z = &gammas / sum;
        let log_q = self.log_prob(&z)?;
        Ok((z, log_q))
    }
    fn entropy(&self) -> f64 {
        self.entropy_dirichlet()
    }
    fn log_prob(&self, z: &Array1<f64>) -> Result<f64> {
        if z.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "z has length {}, expected {}",
                z.len(),
                self.dim
            )));
        }
        let alphas = self.alphas();
        let alpha_0: f64 = alphas.sum();
        let log_beta: f64 = alphas.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_0);
        let mut log_q = -log_beta;
        for i in 0..self.dim {
            if z[i] <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            log_q += (alphas[i] - 1.0) * z[i].ln();
        }
        Ok(log_q)
    }
    fn reparam_gradient(
        &self,
        dlog_joint_dz: &Array1<f64>,
        epsilon: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        if dlog_joint_dz.len() != self.dim || epsilon.len() != self.dim {
            return Err(StatsError::DimensionMismatch(
                "dlog_joint_dz and epsilon must match dim".to_string(),
            ));
        }
        let alphas = self.alphas();
        let eps_h = 1e-5;
        let mut grad = Array1::zeros(self.dim);
        for i in 0..self.dim {
            let mut alphas_hi = alphas.clone();
            let mut alphas_lo = alphas.clone();
            alphas_hi[i] *= 1.0 + eps_h;
            alphas_lo[i] *= 1.0 - eps_h;
            let compute_z = |a: &Array1<f64>| -> Array1<f64> {
                let mut gammas = Array1::zeros(self.dim);
                for j in 0..self.dim {
                    gammas[j] = Self::gamma_reparam(a[j], epsilon[j]).max(1e-10);
                }
                let sum: f64 = gammas.sum();
                &gammas / sum
            };
            let z_hi = compute_z(&alphas_hi);
            let z_lo = compute_z(&alphas_lo);
            let dz_d_la: Array1<f64> = (&z_hi - &z_lo) / (2.0 * alphas[i] * eps_h);
            let joint_term: f64 = dlog_joint_dz
                .iter()
                .zip(dz_d_la.iter())
                .map(|(&g, &d)| g * d)
                .sum();
            let alpha_0: f64 = alphas.sum();
            let entropy_grad = (digamma(alpha_0) - digamma(alphas[i])) * alphas[i];
            grad[i] = joint_term + entropy_grad;
        }
        Ok(grad)
    }
    fn kl_from_prior(&self) -> Option<f64> {
        Some(self.kl_to_uniform_dirichlet())
    }
}
