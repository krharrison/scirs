//! # GammaVI - Trait Implementations
//!
//! This module contains trait implementations for `GammaVI`.
//!
//! ## Implemented Traits
//!
//! - `VariationalFamily`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;

use super::super::{digamma, lgamma, trigamma};
use super::functions::VariationalFamily;
use super::types::GammaVI;

impl VariationalFamily for GammaVI {
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_params(&self) -> usize {
        2 * self.dim
    }
    fn get_params(&self) -> Array1<f64> {
        let mut params = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            params[i] = self.log_shape[i];
            params[self.dim + i] = self.log_rate[i];
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
            self.log_shape[i] = params[i];
            self.log_rate[i] = params[self.dim + i];
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
        let shapes = self.shapes();
        let rates = self.rates();
        let mut z = Array1::zeros(self.dim);
        for i in 0..self.dim {
            z[i] = Self::gamma_reparam(shapes[i], epsilon[i]) / rates[i];
        }
        let log_q = self.log_prob(&z)?;
        Ok((z, log_q))
    }
    fn entropy(&self) -> f64 {
        let shapes = self.shapes();
        let rates = self.rates();
        (0..self.dim)
            .map(|i| {
                let a = shapes[i];
                let b = rates[i];
                a - b.ln() + lgamma(a) + (1.0 - a) * digamma(a)
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
        let shapes = self.shapes();
        let rates = self.rates();
        let mut log_q = 0.0;
        for i in 0..self.dim {
            if z[i] <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            let a = shapes[i];
            let b = rates[i];
            log_q += a * b.ln() - lgamma(a) + (a - 1.0) * z[i].ln() - b * z[i];
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
        let shapes = self.shapes();
        let rates = self.rates();
        let eps_h = 1e-5;
        let mut grad = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            let z_hi = Self::gamma_reparam(shapes[i] * (1.0 + eps_h), epsilon[i]) / rates[i];
            let z_lo = Self::gamma_reparam(shapes[i] * (1.0 - eps_h), epsilon[i]) / rates[i];
            let dz_d_ls = (z_hi - z_lo) / (2.0 * shapes[i] * eps_h);
            let dh_d_ls = (1.0 + (1.0 - shapes[i]) * trigamma(shapes[i])) * shapes[i];
            grad[i] = dlog_joint_dz[i] * dz_d_ls + dh_d_ls;
            let z_i = Self::gamma_reparam(shapes[i], epsilon[i]) / rates[i];
            grad[self.dim + i] = dlog_joint_dz[i] * (-z_i) + (-1.0);
        }
        Ok(grad)
    }
    fn kl_from_prior(&self) -> Option<f64> {
        let shape0 = Array1::ones(self.dim);
        let rate0 = Array1::ones(self.dim);
        self.kl_to_gamma_prior(&shape0, &rate0).ok()
    }
}
