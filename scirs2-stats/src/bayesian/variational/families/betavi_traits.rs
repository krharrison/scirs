//! # BetaVI - Trait Implementations
//!
//! This module contains trait implementations for `BetaVI`.
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
use super::types::BetaVI;

impl VariationalFamily for BetaVI {
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_params(&self) -> usize {
        2 * self.dim
    }
    fn get_params(&self) -> Array1<f64> {
        let mut params = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            params[i] = self.log_alpha[i];
            params[self.dim + i] = self.log_beta[i];
        }
        params
    }
    fn set_params(&mut self, params: &Array1<f64>) -> Result<()> {
        if params.len() != 2 * self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "expected {} parameters, got {}",
                2 * self.dim,
                params.len()
            )));
        }
        for i in 0..self.dim {
            self.log_alpha[i] = params[i];
            self.log_beta[i] = params[self.dim + i];
        }
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
        let betas = self.betas();
        let mut z = Array1::zeros(self.dim);
        for i in 0..self.dim {
            let u = 1.0 / (1.0 + (-epsilon[i]).exp());
            z[i] = Self::approx_beta_quantile(alphas[i], betas[i], u);
        }
        let log_q = self.log_prob(&z)?;
        Ok((z, log_q))
    }
    fn entropy(&self) -> f64 {
        let alphas = self.alphas();
        let betas = self.betas();
        (0..self.dim)
            .map(|i| {
                let a = alphas[i];
                let b = betas[i];
                let log_b = lgamma(a) + lgamma(b) - lgamma(a + b);
                log_b - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
                    + (a + b - 2.0) * digamma(a + b)
            })
            .sum()
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
        let betas = self.betas();
        let mut log_q = 0.0;
        for i in 0..self.dim {
            if z[i] <= 0.0 || z[i] >= 1.0 {
                return Ok(f64::NEG_INFINITY);
            }
            let a = alphas[i];
            let b = betas[i];
            let log_b = lgamma(a) + lgamma(b) - lgamma(a + b);
            log_q += (a - 1.0) * z[i].ln() + (b - 1.0) * (1.0 - z[i]).ln() - log_b;
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
        let betas = self.betas();
        let eps_h = 1e-5;
        let mut grad = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            let u = 1.0 / (1.0 + (-epsilon[i]).exp());
            let z_hi = Self::approx_beta_quantile(alphas[i] * (1.0 + eps_h), betas[i], u);
            let z_lo = Self::approx_beta_quantile(alphas[i] * (1.0 - eps_h), betas[i], u);
            let dz_d_la = (z_hi - z_lo) / (2.0 * alphas[i] * eps_h);
            grad[i] = dlog_joint_dz[i] * dz_d_la;
            let h_hi = {
                let a2 = alphas[i] * (1.0 + eps_h);
                let b = betas[i];
                let log_b = lgamma(a2) + lgamma(b) - lgamma(a2 + b);
                log_b - (a2 - 1.0) * digamma(a2) - (b - 1.0) * digamma(b)
                    + (a2 + b - 2.0) * digamma(a2 + b)
            };
            let h_lo = {
                let a2 = alphas[i] * (1.0 - eps_h);
                let b = betas[i];
                let log_b = lgamma(a2) + lgamma(b) - lgamma(a2 + b);
                log_b - (a2 - 1.0) * digamma(a2) - (b - 1.0) * digamma(b)
                    + (a2 + b - 2.0) * digamma(a2 + b)
            };
            let dh_d_la = (h_hi - h_lo) / (2.0 * alphas[i] * eps_h);
            grad[i] += dh_d_la;
            let z_hi = Self::approx_beta_quantile(alphas[i], betas[i] * (1.0 + eps_h), u);
            let z_lo = Self::approx_beta_quantile(alphas[i], betas[i] * (1.0 - eps_h), u);
            let dz_d_lb = (z_hi - z_lo) / (2.0 * betas[i] * eps_h);
            grad[self.dim + i] = dlog_joint_dz[i] * dz_d_lb;
            let h_hi = {
                let a = alphas[i];
                let b2 = betas[i] * (1.0 + eps_h);
                let log_b = lgamma(a) + lgamma(b2) - lgamma(a + b2);
                log_b - (a - 1.0) * digamma(a) - (b2 - 1.0) * digamma(b2)
                    + (a + b2 - 2.0) * digamma(a + b2)
            };
            let h_lo = {
                let a = alphas[i];
                let b2 = betas[i] * (1.0 - eps_h);
                let log_b = lgamma(a) + lgamma(b2) - lgamma(a + b2);
                log_b - (a - 1.0) * digamma(a) - (b2 - 1.0) * digamma(b2)
                    + (a + b2 - 2.0) * digamma(a + b2)
            };
            let dh_d_lb = (h_hi - h_lo) / (2.0 * betas[i] * eps_h);
            grad[self.dim + i] += dh_d_lb;
        }
        Ok(grad)
    }
    fn kl_from_prior(&self) -> Option<f64> {
        Some(self.kl_to_uniform())
    }
}
