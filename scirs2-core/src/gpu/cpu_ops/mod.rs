//! CPU fallback implementations for GPU operations.
//!
//! This module provides correct (not just stub) CPU implementations for all
//! GPU operations. These are used when no GPU backend is available or when
//! running on the CPU fallback backend.
//!
//! Operations are implemented generically where possible, with type-specific
//! dispatch for numeric operations that require floating-point math.

use std::any::TypeId;

use super::{GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuError};

mod arithmetic;
mod matrix;
mod reduction;
mod traits;

use arithmetic::{
    gelu_backward_cpu, gelu_cpu, relu_backward_cpu, relu_cpu, sigmoid_backward_cpu, sigmoid_cpu,
    tanh_backward_cpu, tanh_cpu,
};
use matrix::{gemm_cpu, gemm_transpose_a_cpu, gemm_transpose_b_cpu};
use reduction::{
    broadcast_cpu, max_all_cpu, max_axis_cpu, mean_all_cpu, mean_axis_cpu, min_all_cpu,
    min_axis_cpu, scale_cpu, sum_all_cpu, sum_axis_cpu,
};
use traits::reinterpret_vec;
use traits::{dispatch_numeric, erf_f64, FloatOps, NumericOps};

// =============================================================================
// Public API (GpuContext methods)
// =============================================================================

impl GpuContext {
    /// General matrix multiplication (CPU fallback): C = A * B
    /// A is m x k, B is k x n, C is m x n
    pub(crate) fn gemm_cpu_fallback<T: GpuDataType>(
        &self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(GpuError::InvalidParameter(format!(
                "Matrix A size {} does not match m*k = {}*{} = {}",
                a.len(),
                m,
                k,
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(GpuError::InvalidParameter(format!(
                "Matrix B size {} does not match k*n = {}*{} = {}",
                b.len(),
                k,
                n,
                k * n
            )));
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();

        let result_data = dispatch_numeric::<T, _, _, _>(
            || {
                let a_f32: Vec<f32> = unsafe { reinterpret_vec(a_data.clone()) };
                let b_f32: Vec<f32> = unsafe { reinterpret_vec(b_data.clone()) };
                gemm_cpu(&a_f32, &b_f32, m, k, n)
            },
            || {
                let a_f64: Vec<f64> = unsafe { reinterpret_vec(a_data.clone()) };
                let b_f64: Vec<f64> = unsafe { reinterpret_vec(b_data.clone()) };
                gemm_cpu(&a_f64, &b_f64, m, k, n)
            },
            || {
                // For integer types, use f64 intermediary for GEMM
                let a_f64: Vec<f64> = a_data
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        // Interpret as the native int type - we need type dispatch
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let b_f64: Vec<f64> = b_data
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let c_f64 = gemm_cpu(&a_f64, &b_f64, m, k, n);
                let result: Vec<T> = c_f64
                    .iter()
                    .map(|&v| {
                        let mut val: T = unsafe { std::mem::zeroed() };
                        write_from_f64::<T>(&mut val, v);
                        val
                    })
                    .collect();
                Ok(result)
            },
        )?;

        let result = self.create_buffer::<T>(m * n);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// GEMM with transposed B (CPU fallback): C = A * B^T
    pub(crate) fn gemm_transpose_b_cpu_fallback<T: GpuDataType>(
        &self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        if a.len() != m * k {
            return Err(GpuError::InvalidParameter(format!(
                "Matrix A size {} does not match m*k = {}*{} = {}",
                a.len(),
                m,
                k,
                m * k
            )));
        }
        if b.len() != n * k {
            return Err(GpuError::InvalidParameter(format!(
                "Matrix B size {} does not match n*k = {}*{} = {}",
                b.len(),
                n,
                k,
                n * k
            )));
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();

        let result_data = dispatch_numeric::<T, _, _, _>(
            || {
                let a_f32: Vec<f32> = unsafe { reinterpret_vec(a_data.clone()) };
                let b_f32: Vec<f32> = unsafe { reinterpret_vec(b_data.clone()) };
                gemm_transpose_b_cpu(&a_f32, &b_f32, m, k, n)
            },
            || {
                let a_f64: Vec<f64> = unsafe { reinterpret_vec(a_data.clone()) };
                let b_f64: Vec<f64> = unsafe { reinterpret_vec(b_data.clone()) };
                gemm_transpose_b_cpu(&a_f64, &b_f64, m, k, n)
            },
            || {
                let a_f64: Vec<f64> = a_data
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let b_f64: Vec<f64> = b_data
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let c_f64 = gemm_transpose_b_cpu(&a_f64, &b_f64, m, k, n);
                let result: Vec<T> = c_f64
                    .iter()
                    .map(|&v| {
                        let mut val: T = unsafe { std::mem::zeroed() };
                        write_from_f64::<T>(&mut val, v);
                        val
                    })
                    .collect();
                Ok(result)
            },
        )?;

        let result = self.create_buffer::<T>(m * n);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// GEMM with transposed A (CPU fallback): C = A^T * B
    pub(crate) fn gemm_transpose_a_cpu_fallback<T: GpuDataType>(
        &self,
        a: &GpuBuffer<T>,
        b: &GpuBuffer<T>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        if a.len() != k * m {
            return Err(GpuError::InvalidParameter(format!(
                "Matrix A size {} does not match k*m = {}*{} = {}",
                a.len(),
                k,
                m,
                k * m
            )));
        }
        if b.len() != k * n {
            return Err(GpuError::InvalidParameter(format!(
                "Matrix B size {} does not match k*n = {}*{} = {}",
                b.len(),
                k,
                n,
                k * n
            )));
        }

        let a_data = a.to_vec();
        let b_data = b.to_vec();

        let result_data = dispatch_numeric::<T, _, _, _>(
            || {
                let a_f32: Vec<f32> = unsafe { reinterpret_vec(a_data.clone()) };
                let b_f32: Vec<f32> = unsafe { reinterpret_vec(b_data.clone()) };
                gemm_transpose_a_cpu(&a_f32, &b_f32, m, k, n)
            },
            || {
                let a_f64: Vec<f64> = unsafe { reinterpret_vec(a_data.clone()) };
                let b_f64: Vec<f64> = unsafe { reinterpret_vec(b_data.clone()) };
                gemm_transpose_a_cpu(&a_f64, &b_f64, m, k, n)
            },
            || {
                let a_f64: Vec<f64> = a_data
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let b_f64: Vec<f64> = b_data
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let c_f64 = gemm_transpose_a_cpu(&a_f64, &b_f64, m, k, n);
                let result: Vec<T> = c_f64
                    .iter()
                    .map(|&v| {
                        let mut val: T = unsafe { std::mem::zeroed() };
                        write_from_f64::<T>(&mut val, v);
                        val
                    })
                    .collect();
                Ok(result)
            },
        )?;

        let result = self.create_buffer::<T>(m * n);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// ReLU forward pass (CPU fallback)
    pub(crate) fn relu_cpu_fallback<T: GpuDataType>(
        &self,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = input.to_vec();

        let result_data = dispatch_float_op::<T>(&data, |d| relu_cpu(d), |d| relu_cpu(d))?;

        let result = self.create_buffer::<T>(input.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// ReLU backward pass (CPU fallback)
    pub(crate) fn relu_backward_cpu_fallback<T: GpuDataType>(
        &self,
        grad_output: &GpuBuffer<T>,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();

        let result_data = dispatch_float_op2::<T>(
            &grad_data,
            &input_data,
            |g, i| relu_backward_cpu(g, i),
            |g, i| relu_backward_cpu(g, i),
        )?;

        let result = self.create_buffer::<T>(grad_output.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Sigmoid forward pass (CPU fallback)
    pub(crate) fn sigmoid_cpu_fallback<T: GpuDataType>(
        &self,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = input.to_vec();

        let result_data = dispatch_float_op::<T>(&data, |d| sigmoid_cpu(d), |d| sigmoid_cpu(d))?;

        let result = self.create_buffer::<T>(input.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Sigmoid backward pass (CPU fallback)
    pub(crate) fn sigmoid_backward_cpu_fallback<T: GpuDataType>(
        &self,
        grad_output: &GpuBuffer<T>,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();

        let result_data = dispatch_float_op2::<T>(
            &grad_data,
            &input_data,
            |g, i| sigmoid_backward_cpu(g, i),
            |g, i| sigmoid_backward_cpu(g, i),
        )?;

        let result = self.create_buffer::<T>(grad_output.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Tanh forward pass (CPU fallback)
    pub(crate) fn tanh_cpu_fallback<T: GpuDataType>(
        &self,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = input.to_vec();

        let result_data = dispatch_float_op::<T>(&data, |d| tanh_cpu(d), |d| tanh_cpu(d))?;

        let result = self.create_buffer::<T>(input.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Tanh backward pass (CPU fallback)
    pub(crate) fn tanh_backward_cpu_fallback<T: GpuDataType>(
        &self,
        grad_output: &GpuBuffer<T>,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();

        let result_data = dispatch_float_op2::<T>(
            &grad_data,
            &input_data,
            |g, i| tanh_backward_cpu(g, i),
            |g, i| tanh_backward_cpu(g, i),
        )?;

        let result = self.create_buffer::<T>(grad_output.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// GELU forward pass (CPU fallback)
    pub(crate) fn gelu_cpu_fallback<T: GpuDataType>(
        &self,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = input.to_vec();

        let result_data = dispatch_float_op::<T>(&data, |d| gelu_cpu(d), |d| gelu_cpu(d))?;

        let result = self.create_buffer::<T>(input.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// GELU backward pass (CPU fallback)
    pub(crate) fn gelu_backward_cpu_fallback<T: GpuDataType>(
        &self,
        grad_output: &GpuBuffer<T>,
        input: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();

        let result_data = dispatch_float_op2::<T>(
            &grad_data,
            &input_data,
            |g, i| gelu_backward_cpu(g, i),
            |g, i| gelu_backward_cpu(g, i),
        )?;

        let result = self.create_buffer::<T>(grad_output.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Sum reduction (CPU fallback)
    pub(crate) fn sum_all_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = buffer.to_vec();

        let result_data = dispatch_reduction::<T, _, _, _>(
            &data,
            |d| {
                let typed: &[f32] = d;
                vec![sum_all_cpu(typed)]
            },
            |d| {
                let typed: &[f64] = d;
                vec![sum_all_cpu(typed)]
            },
            |d| {
                // For integer types, convert through f64
                let f64_data: Vec<f64> = d
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let sum = sum_all_cpu(&f64_data);
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, sum);
                Ok(vec![val])
            },
        )?;

        let result = self.create_buffer::<T>(1);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Mean reduction (CPU fallback)
    pub(crate) fn mean_all_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = buffer.to_vec();

        let result_data = dispatch_reduction::<T, _, _, _>(
            &data,
            |d| {
                let typed: &[f32] = d;
                vec![mean_all_cpu(typed)]
            },
            |d| {
                let typed: &[f64] = d;
                vec![mean_all_cpu(typed)]
            },
            |d| {
                let f64_data: Vec<f64> = d
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let mean = mean_all_cpu(&f64_data);
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, mean);
                Ok(vec![val])
            },
        )?;

        let result = self.create_buffer::<T>(1);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Max reduction (CPU fallback)
    pub(crate) fn max_all_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = buffer.to_vec();

        let result_data = dispatch_reduction::<T, _, _, _>(
            &data,
            |d| {
                let typed: &[f32] = d;
                vec![max_all_cpu(typed)]
            },
            |d| {
                let typed: &[f64] = d;
                vec![max_all_cpu(typed)]
            },
            |d| {
                let f64_data: Vec<f64> = d
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let max_val = max_all_cpu(&f64_data);
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, max_val);
                Ok(vec![val])
            },
        )?;

        let result = self.create_buffer::<T>(1);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Min reduction (CPU fallback)
    pub(crate) fn min_all_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = buffer.to_vec();

        let result_data = dispatch_reduction::<T, _, _, _>(
            &data,
            |d| {
                let typed: &[f32] = d;
                vec![min_all_cpu(typed)]
            },
            |d| {
                let typed: &[f64] = d;
                vec![min_all_cpu(typed)]
            },
            |d| {
                let f64_data: Vec<f64> = d
                    .iter()
                    .map(|v| {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                v as *const T as *const u8,
                                std::mem::size_of::<T>(),
                            )
                        };
                        interpret_as_f64::<T>(bytes)
                    })
                    .collect();
                let min_val = min_all_cpu(&f64_data);
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, min_val);
                Ok(vec![val])
            },
        )?;

        let result = self.create_buffer::<T>(1);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Sum axis reduction (CPU fallback)
    pub(crate) fn sum_axis_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
        shape: &[usize],
        axis: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        if axis >= shape.len() {
            return Err(GpuError::InvalidParameter(format!(
                "Axis {} out of bounds for shape {:?}",
                axis, shape
            )));
        }

        let data = buffer.to_vec();

        let result_data = dispatch_axis_reduction::<T>(
            &data,
            shape,
            axis,
            |d, s, a| sum_axis_cpu(d, s, a),
            |d, s, a| sum_axis_cpu(d, s, a),
        )?;

        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        let output_size: usize = output_shape.iter().product();
        let result = self.create_buffer::<T>(output_size);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Mean axis reduction (CPU fallback)
    pub(crate) fn mean_axis_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
        shape: &[usize],
        axis: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        if axis >= shape.len() {
            return Err(GpuError::InvalidParameter(format!(
                "Axis {} out of bounds for shape {:?}",
                axis, shape
            )));
        }

        let data = buffer.to_vec();

        let result_data = dispatch_axis_reduction::<T>(
            &data,
            shape,
            axis,
            |d, s, a| mean_axis_cpu(d, s, a),
            |d, s, a| mean_axis_cpu(d, s, a),
        )?;

        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        let output_size: usize = output_shape.iter().product();
        let result = self.create_buffer::<T>(output_size);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Max axis reduction (CPU fallback)
    pub(crate) fn max_axis_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
        shape: &[usize],
        axis: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        if axis >= shape.len() {
            return Err(GpuError::InvalidParameter(format!(
                "Axis {} out of bounds for shape {:?}",
                axis, shape
            )));
        }

        let data = buffer.to_vec();

        let result_data = dispatch_axis_reduction::<T>(
            &data,
            shape,
            axis,
            |d, s, a| max_axis_cpu(d, s, a),
            |d, s, a| max_axis_cpu(d, s, a),
        )?;

        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        let output_size: usize = output_shape.iter().product();
        let result = self.create_buffer::<T>(output_size);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Min axis reduction (CPU fallback)
    pub(crate) fn min_axis_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
        shape: &[usize],
        axis: usize,
    ) -> Result<GpuBuffer<T>, GpuError> {
        if axis >= shape.len() {
            return Err(GpuError::InvalidParameter(format!(
                "Axis {} out of bounds for shape {:?}",
                axis, shape
            )));
        }

        let data = buffer.to_vec();

        let result_data = dispatch_axis_reduction::<T>(
            &data,
            shape,
            axis,
            |d, s, a| min_axis_cpu(d, s, a),
            |d, s, a| min_axis_cpu(d, s, a),
        )?;

        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        let output_size: usize = output_shape.iter().product();
        let result = self.create_buffer::<T>(output_size);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Broadcast (CPU fallback)
    pub(crate) fn broadcast_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
        from_shape: &[usize],
        to_shape: &[usize],
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = buffer.to_vec();

        let result_data = dispatch_broadcast::<T>(
            &data,
            from_shape,
            to_shape,
            |d, f, t| broadcast_cpu(d, f, t),
            |d, f, t| broadcast_cpu(d, f, t),
        )?;

        let output_size: usize = to_shape.iter().product();
        let result = self.create_buffer::<T>(output_size);
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }

    /// Scale (CPU fallback)
    pub(crate) fn scale_cpu_fallback<T: GpuDataType>(
        &self,
        buffer: &GpuBuffer<T>,
        scalar: T,
    ) -> Result<GpuBuffer<T>, GpuError> {
        let data = buffer.to_vec();

        let result_data = dispatch_scale::<T>(&data, scalar)?;

        let result = self.create_buffer::<T>(buffer.len());
        let _ = result.copy_from_host(&result_data);
        Ok(result)
    }
}

// =============================================================================
// Dispatch helpers for activation functions (float-only operations)
// =============================================================================

/// Dispatch a unary float operation based on the runtime type of T.
/// For non-float types, returns an error since activation functions
/// are only meaningful for floating-point data.
fn dispatch_float_op<T: GpuDataType>(
    data: &[T],
    f32_op: impl FnOnce(&[f32]) -> Vec<f32>,
    f64_op: impl FnOnce(&[f64]) -> Vec<f64>,
) -> Result<Vec<T>, GpuError> {
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let typed: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let result = f32_op(typed);
        Ok(unsafe { reinterpret_vec(result) })
    } else if type_id == TypeId::of::<f64>() {
        let typed: &[f64] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let result = f64_op(typed);
        Ok(unsafe { reinterpret_vec(result) })
    } else {
        Err(GpuError::InvalidParameter(
            "Activation functions are only supported for f32 and f64 types".to_string(),
        ))
    }
}

/// Dispatch a binary float operation (e.g., backward pass) based on runtime type.
fn dispatch_float_op2<T: GpuDataType>(
    data1: &[T],
    data2: &[T],
    f32_op: impl FnOnce(&[f32], &[f32]) -> Vec<f32>,
    f64_op: impl FnOnce(&[f64], &[f64]) -> Vec<f64>,
) -> Result<Vec<T>, GpuError> {
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let typed1: &[f32] =
            unsafe { std::slice::from_raw_parts(data1.as_ptr() as *const f32, data1.len()) };
        let typed2: &[f32] =
            unsafe { std::slice::from_raw_parts(data2.as_ptr() as *const f32, data2.len()) };
        let result = f32_op(typed1, typed2);
        Ok(unsafe { reinterpret_vec(result) })
    } else if type_id == TypeId::of::<f64>() {
        let typed1: &[f64] =
            unsafe { std::slice::from_raw_parts(data1.as_ptr() as *const f64, data1.len()) };
        let typed2: &[f64] =
            unsafe { std::slice::from_raw_parts(data2.as_ptr() as *const f64, data2.len()) };
        let result = f64_op(typed1, typed2);
        Ok(unsafe { reinterpret_vec(result) })
    } else {
        Err(GpuError::InvalidParameter(
            "Activation backward pass is only supported for f32 and f64 types".to_string(),
        ))
    }
}

/// Dispatch a reduction operation.
fn dispatch_reduction<T: GpuDataType, F32Fn, F64Fn, IntFn>(
    data: &[T],
    f32_op: F32Fn,
    f64_op: F64Fn,
    int_op: IntFn,
) -> Result<Vec<T>, GpuError>
where
    F32Fn: FnOnce(&[f32]) -> Vec<f32>,
    F64Fn: FnOnce(&[f64]) -> Vec<f64>,
    IntFn: FnOnce(&[T]) -> Result<Vec<T>, GpuError>,
{
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let typed: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let result = f32_op(typed);
        Ok(unsafe { reinterpret_vec(result) })
    } else if type_id == TypeId::of::<f64>() {
        let typed: &[f64] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let result = f64_op(typed);
        Ok(unsafe { reinterpret_vec(result) })
    } else {
        int_op(data)
    }
}

/// Dispatch an axis reduction operation.
fn dispatch_axis_reduction<T: GpuDataType>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    f32_op: impl FnOnce(&[f32], &[usize], usize) -> Vec<f32>,
    f64_op: impl FnOnce(&[f64], &[usize], usize) -> Vec<f64>,
) -> Result<Vec<T>, GpuError> {
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let typed: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let result = f32_op(typed, shape, axis);
        Ok(unsafe { reinterpret_vec(result) })
    } else if type_id == TypeId::of::<f64>() {
        let typed: &[f64] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let result = f64_op(typed, shape, axis);
        Ok(unsafe { reinterpret_vec(result) })
    } else {
        // For integer types, convert via f64
        let f64_data: Vec<f64> = data
            .iter()
            .map(|v| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>())
                };
                interpret_as_f64::<T>(bytes)
            })
            .collect();
        let f64_result = f64_op(&f64_data, shape, axis);
        let result: Vec<T> = f64_result
            .iter()
            .map(|&v| {
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, v);
                val
            })
            .collect();
        Ok(result)
    }
}

/// Dispatch a broadcast operation.
fn dispatch_broadcast<T: GpuDataType>(
    data: &[T],
    from_shape: &[usize],
    to_shape: &[usize],
    f32_op: impl FnOnce(&[f32], &[usize], &[usize]) -> Result<Vec<f32>, GpuError>,
    f64_op: impl FnOnce(&[f64], &[usize], &[usize]) -> Result<Vec<f64>, GpuError>,
) -> Result<Vec<T>, GpuError> {
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let typed: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let result = f32_op(typed, from_shape, to_shape)?;
        Ok(unsafe { reinterpret_vec(result) })
    } else if type_id == TypeId::of::<f64>() {
        let typed: &[f64] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let result = f64_op(typed, from_shape, to_shape)?;
        Ok(unsafe { reinterpret_vec(result) })
    } else {
        // For non-float types, broadcast is still valid (just copying values)
        // Use a generic approach: treat as bytes
        let f64_data: Vec<f64> = data
            .iter()
            .map(|v| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>())
                };
                interpret_as_f64::<T>(bytes)
            })
            .collect();
        let f64_result = f64_op(&f64_data, from_shape, to_shape)?;
        let result: Vec<T> = f64_result
            .iter()
            .map(|&v| {
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, v);
                val
            })
            .collect();
        Ok(result)
    }
}

/// Dispatch a scale operation.
fn dispatch_scale<T: GpuDataType>(data: &[T], scalar: T) -> Result<Vec<T>, GpuError> {
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<f32>() {
        let typed: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
        let scalar_f32: f32 = unsafe { *(&scalar as *const T as *const f32) };
        let result = scale_cpu(typed, scalar_f32);
        Ok(unsafe { reinterpret_vec(result) })
    } else if type_id == TypeId::of::<f64>() {
        let typed: &[f64] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let scalar_f64: f64 = unsafe { *(&scalar as *const T as *const f64) };
        let result = scale_cpu(typed, scalar_f64);
        Ok(unsafe { reinterpret_vec(result) })
    } else {
        // For integer types, convert through f64
        let scalar_f64 = {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &scalar as *const T as *const u8,
                    std::mem::size_of::<T>(),
                )
            };
            interpret_as_f64::<T>(bytes)
        };
        let f64_data: Vec<f64> = data
            .iter()
            .map(|v| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>())
                };
                interpret_as_f64::<T>(bytes)
            })
            .collect();
        let f64_result = scale_cpu(&f64_data, scalar_f64);
        let result: Vec<T> = f64_result
            .iter()
            .map(|&v| {
                let mut val: T = unsafe { std::mem::zeroed() };
                write_from_f64::<T>(&mut val, v);
                val
            })
            .collect();
        Ok(result)
    }
}

// =============================================================================
// Integer type conversion helpers
// =============================================================================

/// Interpret raw bytes of a GpuDataType value as f64.
/// Uses TypeId dispatch to handle each concrete type.
fn interpret_as_f64<T: GpuDataType>(bytes: &[u8]) -> f64 {
    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<i8>() {
        let val = i8::from_ne_bytes(bytes[..1].try_into().unwrap_or([0]));
        val as f64
    } else if type_id == TypeId::of::<u8>() {
        let val = u8::from_ne_bytes(bytes[..1].try_into().unwrap_or([0]));
        val as f64
    } else if type_id == TypeId::of::<i16>() {
        let val = i16::from_ne_bytes(bytes[..2].try_into().unwrap_or([0; 2]));
        val as f64
    } else if type_id == TypeId::of::<u16>() {
        let val = u16::from_ne_bytes(bytes[..2].try_into().unwrap_or([0; 2]));
        val as f64
    } else if type_id == TypeId::of::<i32>() {
        let val = i32::from_ne_bytes(bytes[..4].try_into().unwrap_or([0; 4]));
        val as f64
    } else if type_id == TypeId::of::<u32>() {
        let val = u32::from_ne_bytes(bytes[..4].try_into().unwrap_or([0; 4]));
        val as f64
    } else if type_id == TypeId::of::<i64>() {
        let val = i64::from_ne_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
        val as f64
    } else if type_id == TypeId::of::<u64>() {
        let val = u64::from_ne_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
        val as f64
    } else if type_id == TypeId::of::<usize>() {
        let size = std::mem::size_of::<usize>();
        if size == 8 {
            let val = u64::from_ne_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
            val as f64
        } else {
            let val = u32::from_ne_bytes(bytes[..4].try_into().unwrap_or([0; 4]));
            val as f64
        }
    } else if type_id == TypeId::of::<isize>() {
        let size = std::mem::size_of::<isize>();
        if size == 8 {
            let val = i64::from_ne_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
            val as f64
        } else {
            let val = i32::from_ne_bytes(bytes[..4].try_into().unwrap_or([0; 4]));
            val as f64
        }
    } else {
        0.0 // Unknown type
    }
}

/// Write an f64 value into a GpuDataType value, converting to the appropriate type.
fn write_from_f64<T: GpuDataType>(dest: &mut T, val: f64) {
    let type_id = TypeId::of::<T>();
    let dest_ptr = dest as *mut T as *mut u8;

    if type_id == TypeId::of::<i8>() {
        let v = val as i8;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 1) };
    } else if type_id == TypeId::of::<u8>() {
        let v = val as u8;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 1) };
    } else if type_id == TypeId::of::<i16>() {
        let v = val as i16;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 2) };
    } else if type_id == TypeId::of::<u16>() {
        let v = val as u16;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 2) };
    } else if type_id == TypeId::of::<i32>() {
        let v = val as i32;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 4) };
    } else if type_id == TypeId::of::<u32>() {
        let v = val as u32;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 4) };
    } else if type_id == TypeId::of::<i64>() {
        let v = val as i64;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 8) };
    } else if type_id == TypeId::of::<u64>() {
        let v = val as u64;
        unsafe { std::ptr::copy_nonoverlapping(v.to_ne_bytes().as_ptr(), dest_ptr, 8) };
    } else if type_id == TypeId::of::<usize>() {
        let v = val as usize;
        let bytes = v.to_ne_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest_ptr, std::mem::size_of::<usize>())
        };
    } else if type_id == TypeId::of::<isize>() {
        let v = val as isize;
        let bytes = v.to_ne_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest_ptr, std::mem::size_of::<isize>())
        };
    }
    // For unknown types, dest remains zeroed
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- GEMM tests ----

    #[test]
    fn test_gemm_cpu_identity() {
        // 2x2 identity * [1,2;3,4] = [1,2;3,4]
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let c = gemm_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gemm_cpu_basic() {
        // [1,2;3,4] * [5,6;7,8] = [19,22;43,50]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let c = gemm_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_cpu_non_square() {
        // A = [1,2,3] (1x3), B = [4;5;6] (3x1) => C = [32] (1x1)
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];
        let c = gemm_cpu(&a, &b, 1, 3, 1);
        assert_eq!(c, vec![32.0]);
    }

    #[test]
    fn test_gemm_transpose_b_cpu() {
        // A = [1,2;3,4] (2x2), B = [5,6;7,8] (2x2)
        // C = A * B^T = [1,2;3,4] * [5,7;6,8] = [17,22;43,58]...wait
        // B^T = [5,7;6,8], so A*B^T:
        // [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]...
        // Wait, B stored as n x k = 2x2: [[5,6],[7,8]]
        // B^T means we use B[j][p] instead of B[p][j]
        // So C[i][j] = sum_p A[i][p] * B[j][p]
        // C[0][0] = 1*5 + 2*6 = 17
        // C[0][1] = 1*7 + 2*8 = 23
        // C[1][0] = 3*5 + 4*6 = 39
        // C[1][1] = 3*7 + 4*8 = 53
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let c = gemm_transpose_b_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![17.0, 23.0, 39.0, 53.0]);
    }

    #[test]
    fn test_gemm_transpose_a_cpu() {
        // A stored as k x m = 2x2: [[1,2],[3,4]] meaning A = [[1,2],[3,4]]
        // A^T = [[1,3],[2,4]]
        // B = [[5,6],[7,8]]
        // C = A^T * B:
        // C[0][0] = 1*5 + 3*7 = 26
        // C[0][1] = 1*6 + 3*8 = 30
        // C[1][0] = 2*5 + 4*7 = 38
        // C[1][1] = 2*6 + 4*8 = 44
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let c = gemm_transpose_a_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![26.0, 30.0, 38.0, 44.0]);
    }

    #[test]
    fn test_gemm_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
        let c = gemm_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // ---- Activation function tests ----

    #[test]
    fn test_relu_cpu() {
        let data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = relu_cpu(&data);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward_cpu() {
        let grad: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = relu_backward_cpu(&grad, &input);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_cpu() {
        let data: Vec<f32> = vec![0.0];
        let result = sigmoid_cpu(&data);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_cpu_large_positive() {
        let data: Vec<f32> = vec![10.0];
        let result = sigmoid_cpu(&data);
        assert!(result[0] > 0.999);
    }

    #[test]
    fn test_sigmoid_cpu_large_negative() {
        let data: Vec<f32> = vec![-10.0];
        let result = sigmoid_cpu(&data);
        assert!(result[0] < 0.001);
    }

    #[test]
    fn test_sigmoid_backward_cpu() {
        // At x=0, sigmoid=0.5, derivative = 0.5*0.5 = 0.25
        let grad: Vec<f32> = vec![1.0];
        let input: Vec<f32> = vec![0.0];
        let result = sigmoid_backward_cpu(&grad, &input);
        assert!((result[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_cpu() {
        let data: Vec<f32> = vec![0.0];
        let result = tanh_cpu(&data);
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_cpu_positive() {
        let data: Vec<f32> = vec![1.0];
        let result = tanh_cpu(&data);
        let expected = 1.0_f32.tanh();
        assert!((result[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_backward_cpu() {
        // At x=0, tanh(0)=0, derivative = 1 - 0^2 = 1
        let grad: Vec<f32> = vec![1.0];
        let input: Vec<f32> = vec![0.0];
        let result = tanh_backward_cpu(&grad, &input);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_cpu_at_zero() {
        let data: Vec<f32> = vec![0.0];
        let result = gelu_cpu(&data);
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_cpu_positive() {
        // GELU(1.0) ~= 0.8413
        let data: Vec<f32> = vec![1.0];
        let result = gelu_cpu(&data);
        let expected = 1.0 * 0.5 * (1.0 + erf_f64(1.0 / std::f64::consts::SQRT_2));
        assert!((result[0] as f64 - expected).abs() < 1e-4);
    }

    #[test]
    fn test_gelu_cpu_negative() {
        // GELU(-1.0) ~= -0.1587
        let data: Vec<f32> = vec![-1.0];
        let result = gelu_cpu(&data);
        let expected = -1.0 * 0.5 * (1.0 + erf_f64(-1.0 / std::f64::consts::SQRT_2));
        assert!((result[0] as f64 - expected).abs() < 1e-4);
    }

    // ---- Reduction tests ----

    #[test]
    fn test_sum_all_cpu() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum_all_cpu(&data);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_mean_all_cpu() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mean_all_cpu(&data);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_max_all_cpu() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let result = max_all_cpu(&data);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_min_all_cpu() {
        let data: Vec<f32> = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let result = min_all_cpu(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_sum_all_empty() {
        let data: Vec<f32> = vec![];
        let result = sum_all_cpu(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mean_all_empty() {
        let data: Vec<f32> = vec![];
        let result = mean_all_cpu(&data);
        assert_eq!(result, 0.0);
    }

    // ---- Axis reduction tests ----

    #[test]
    fn test_sum_axis_2d_axis0() {
        // Shape [2, 3], data = [[1,2,3],[4,5,6]]
        // Sum along axis 0 => [5, 7, 9] (shape [1, 3])
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = sum_axis_cpu(&data, &[2, 3], 0);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_axis_2d_axis1() {
        // Shape [2, 3], data = [[1,2,3],[4,5,6]]
        // Sum along axis 1 => [[6], [15]] (shape [2, 1])
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = sum_axis_cpu(&data, &[2, 3], 1);
        assert_eq!(result, vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean_axis_2d() {
        // Shape [2, 3], mean along axis 0 => [2.5, 3.5, 4.5]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = mean_axis_cpu(&data, &[2, 3], 0);
        assert_eq!(result, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_max_axis_2d() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let result = max_axis_cpu(&data, &[2, 3], 0);
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_min_axis_2d() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let result = min_axis_cpu(&data, &[2, 3], 0);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    // ---- Broadcast tests ----

    #[test]
    fn test_broadcast_scalar_to_vector() {
        let data: Vec<f32> = vec![5.0];
        let result = broadcast_cpu(&data, &[1], &[4]).expect("broadcast should succeed");
        assert_eq!(result, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_broadcast_row_to_matrix() {
        // [1, 2, 3] -> [[1,2,3],[1,2,3]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = broadcast_cpu(&data, &[1, 3], &[2, 3]).expect("broadcast should succeed");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_col_to_matrix() {
        // [[1],[2]] -> [[1,1,1],[2,2,2]]
        let data: Vec<f32> = vec![1.0, 2.0];
        let result = broadcast_cpu(&data, &[2, 1], &[2, 3]).expect("broadcast should succeed");
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        let data: Vec<f32> = vec![1.0, 2.0];
        let result = broadcast_cpu(&data, &[2], &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_add_leading_dims() {
        // Shape [3] -> [2, 3] (leading dimension added)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = broadcast_cpu(&data, &[3], &[2, 3]).expect("broadcast should succeed");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    // ---- Scale tests ----

    #[test]
    fn test_scale_cpu() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let result = scale_cpu(&data, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scale_cpu_zero() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = scale_cpu(&data, 0.0);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scale_cpu_negative() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0];
        let result = scale_cpu(&data, -1.0);
        assert_eq!(result, vec![-1.0, 2.0, -3.0]);
    }

    // ---- erf tests ----

    #[test]
    fn test_erf_at_zero() {
        assert!((erf_f64(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_erf_at_one() {
        // erf(1) ~= 0.8427
        let result = erf_f64(1.0);
        assert!((result - 0.8427).abs() < 0.001);
    }

    #[test]
    fn test_erf_symmetry() {
        let x = 0.5;
        assert!((erf_f64(x) + erf_f64(-x)).abs() < 1e-10);
    }

    #[test]
    fn test_erf_large_value() {
        // erf(3) ~= 0.9999779
        let result = erf_f64(3.0);
        assert!((result - 1.0).abs() < 0.001);
    }

    // ---- Integration tests with GpuContext ----

    #[test]
    fn test_gpu_context_gemm_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let a = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let b = ctx.create_buffer_from_slice(&[5.0f32, 6.0, 7.0, 8.0]);
        let c = ctx.gemm(&a, &b, 2, 2, 2).expect("GEMM failed");
        let result = c.to_vec();
        assert_eq!(result, vec![19.0f32, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_context_relu_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let result = ctx.relu(&input).expect("ReLU failed");
        assert_eq!(result.to_vec(), vec![0.0f32, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gpu_context_sigmoid_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[0.0f32]);
        let result = ctx.sigmoid(&input).expect("Sigmoid failed");
        assert!((result.to_vec()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_context_tanh_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[0.0f32]);
        let result = ctx.tanh(&input).expect("Tanh failed");
        assert!((result.to_vec()[0]).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_context_gelu_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[0.0f32, 1.0, -1.0]);
        let result = ctx.gelu(&input).expect("GELU failed");
        let r = result.to_vec();
        assert!(r[0].abs() < 1e-6); // GELU(0) = 0
        assert!((r[1] - 0.8413).abs() < 0.01); // GELU(1) ~= 0.8413
        assert!((r[2] - (-0.1587)).abs() < 0.01); // GELU(-1) ~= -0.1587
    }

    #[test]
    fn test_gpu_context_sum_all_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let result = ctx.sum_all(&input).expect("Sum failed");
        assert_eq!(result.to_vec(), vec![15.0f32]);
    }

    #[test]
    fn test_gpu_context_mean_all_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let result = ctx.mean_all(&input).expect("Mean failed");
        assert_eq!(result.to_vec(), vec![3.0f32]);
    }

    #[test]
    fn test_gpu_context_max_all_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[1.0f32, 5.0, 3.0, 2.0, 4.0]);
        let result = ctx.max_all(&input).expect("Max failed");
        assert_eq!(result.to_vec(), vec![5.0f32]);
    }

    #[test]
    fn test_gpu_context_min_all_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[3.0f32, 1.0, 5.0, 2.0, 4.0]);
        let result = ctx.min_all(&input).expect("Min failed");
        assert_eq!(result.to_vec(), vec![1.0f32]);
    }

    #[test]
    fn test_gpu_context_broadcast_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0]);
        let result = ctx
            .broadcast(&input, &[1, 3], &[2, 3])
            .expect("Broadcast failed");
        assert_eq!(result.to_vec(), vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gpu_context_scale_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let result = ctx.scale(&input, 2.0f32).expect("Scale failed");
        assert_eq!(result.to_vec(), vec![2.0f32, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_gpu_context_sum_axis_fallback() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = ctx.sum_axis(&input, &[2, 3], 0).expect("Sum axis failed");
        assert_eq!(result.to_vec(), vec![5.0f32, 7.0, 9.0]);
    }

    #[test]
    fn test_gpu_context_f64_operations() {
        let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");

        // Test GEMM with f64
        let a = ctx.create_buffer_from_slice(&[1.0f64, 2.0, 3.0, 4.0]);
        let b = ctx.create_buffer_from_slice(&[5.0f64, 6.0, 7.0, 8.0]);
        let c = ctx.gemm(&a, &b, 2, 2, 2).expect("GEMM f64 failed");
        assert_eq!(c.to_vec(), vec![19.0f64, 22.0, 43.0, 50.0]);
    }
}
