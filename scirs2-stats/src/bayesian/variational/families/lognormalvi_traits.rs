//! # LogNormalVI - Trait Implementations
//!
//! This module contains trait implementations for `LogNormalVI`.
//!
//! ## Implemented Traits
//!
//! - `VariationalFamily`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;
use std::f64::consts::PI;

use super::functions::VariationalFamily;
use super::types::LogNormalVI;

impl VariationalFamily for LogNormalVI {
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_params(&self) -> usize {
        2 * self.dim
    }
    fn get_params(&self) -> Array1<f64> {
        let mut params = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            params[i] = self.mu[i];
            params[self.dim + i] = self.log_sigma[i];
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
            self.mu[i] = params[i];
            self.log_sigma[i] = params[self.dim + i];
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
        let sigmas = self.sigmas();
        let mut z = Array1::zeros(self.dim);
        for i in 0..self.dim {
            z[i] = (self.mu[i] + sigmas[i] * epsilon[i]).exp();
        }
        let log_q = self.log_prob(&z)?;
        Ok((z, log_q))
    }
    fn entropy(&self) -> f64 {
        let log_2pi = (2.0 * PI).ln();
        self.log_sigma
            .iter()
            .map(|&ls| ls + 0.5 * (1.0 + log_2pi))
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
        let sigmas = self.sigmas();
        let log_2pi = (2.0 * PI).ln();
        let mut log_q = 0.0;
        for i in 0..self.dim {
            if z[i] <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            let log_z = z[i].ln();
            let normalized = (log_z - self.mu[i]) / sigmas[i];
            log_q += -0.5 * log_2pi - self.log_sigma[i] - 0.5 * normalized * normalized - log_z;
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
        let sigmas = self.sigmas();
        let mut grad = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            let z_i = (self.mu[i] + sigmas[i] * epsilon[i]).exp();
            grad[i] = dlog_joint_dz[i] * z_i;
            grad[self.dim + i] = dlog_joint_dz[i] * z_i * sigmas[i] * epsilon[i] + 1.0;
        }
        Ok(grad)
    }
    fn kl_from_prior(&self) -> Option<f64> {
        let kl: f64 = (0..self.dim)
            .map(|i| {
                let s2 = (2.0 * self.log_sigma[i]).exp();
                0.5 * (s2 + self.mu[i] * self.mu[i] - 1.0 - 2.0 * self.log_sigma[i])
            })
            .sum();
        Some(kl)
    }
}
