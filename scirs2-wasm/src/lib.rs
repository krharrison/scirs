//! # SciRS2-WASM: WebAssembly Bindings for SciRS2
//!
//! High-performance scientific computing in the browser and Node.js.
//!
//! ## Features
//!
//! - **Pure Rust**: 100% safe Rust code compiled to WASM
//! - **SIMD Support**: Optional WASM SIMD (wasm32-simd128) acceleration
//! - **Async Operations**: Non-blocking computations with async/await
//! - **TypeScript Support**: Full TypeScript type definitions
//! - **Memory Efficient**: Optimized memory management for browser environments
//! - **Zero-copy**: Direct array buffer access when possible
//!
//! ## Modules
//!
//! - `array`: N-dimensional array operations
//! - `linalg`: Linear algebra (matrix operations, decompositions)
//! - `stats`: Statistical functions (distributions, tests, descriptive stats)
//! - `fft`: Fast Fourier Transform operations
//! - `signal`: Signal processing (filtering, convolution, wavelets)
//! - `integrate`: Numerical integration and ODE solvers
//! - `interpolate`: Interpolation (linear, spline, Lagrange, PCHIP, Akima)
//! - `optimize`: Optimization algorithms (minimize, curve fitting)
//! - `random`: Random number generation and distributions
//!
//! ## Example Usage (JavaScript)
//!
//! ```javascript
//! import * as scirs2 from 'scirs2-wasm';
//!
//! // Initialize the library
//! await scirs2.default();
//!
//! // Create arrays
//! const a = scirs2.array([1, 2, 3, 4]);
//! const b = scirs2.array([5, 6, 7, 8]);
//!
//! // Perform operations
//! const sum = scirs2.add(a, b);
//! const dot = scirs2.dot(a, b);
//!
//! // Statistical operations
//! const mean = scirs2.mean(a);
//! const std = scirs2.std(a);
//!
//! // Linear algebra
//! const matrix = scirs2.array2d([[1, 2], [3, 4]]);
//! const inv = scirs2.inv(matrix);
//! const det = scirs2.det(matrix);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use wasm_bindgen::prelude::*;
use web_sys::console;

pub mod array;
pub mod error;
pub mod fft;
pub mod integrate;
pub mod interpolate;
pub mod linalg;
pub mod optimize;
pub mod random;
pub mod signal;
pub mod stats;
pub mod utils;

/// Initialize the WASM module with panic hooks and logging
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error messages in the console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Log initialization message
    log("SciRS2-WASM initialized successfully");
}

/// Get the version of SciRS2-WASM
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn log(message: &str) {
    console::log_1(&JsValue::from_str(message));
}

/// Check if WASM SIMD is supported in the current environment
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

/// Get system capabilities and features available in this build
#[wasm_bindgen]
pub fn capabilities() -> JsValue {
    let caps = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "simd": has_simd_support(),
        "features": {
            "array": true,
            "linalg": cfg!(feature = "linalg"),
            "stats": cfg!(feature = "stats"),
            "fft": cfg!(feature = "fft"),
            "signal": cfg!(feature = "signal"),
            "integrate": cfg!(feature = "integrate"),
            "optimize": cfg!(feature = "optimize"),
            "interpolate": cfg!(feature = "interpolate"),
        },
        "target": {
            "arch": std::env::consts::ARCH,
            "os": "wasm32",
            "family": std::env::consts::FAMILY,
        }
    });

    serde_wasm_bindgen::to_value(&caps).unwrap_or(JsValue::NULL)
}

/// Performance timing utilities for benchmarking WASM operations
#[wasm_bindgen]
pub struct PerformanceTimer {
    start: f64,
    label: String,
}

#[wasm_bindgen]
impl PerformanceTimer {
    /// Create a new performance timer with a label
    #[wasm_bindgen(constructor)]
    pub fn new(label: String) -> Result<PerformanceTimer, JsValue> {
        let start = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object available"))?
            .performance()
            .ok_or_else(|| JsValue::from_str("No performance object available"))?
            .now();

        Ok(PerformanceTimer { start, label })
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed(&self) -> Result<f64, JsValue> {
        let now = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object available"))?
            .performance()
            .ok_or_else(|| JsValue::from_str("No performance object available"))?
            .now();

        Ok(now - self.start)
    }

    /// Log the elapsed time to console
    pub fn log_elapsed(&self) -> Result<(), JsValue> {
        let elapsed = self.elapsed()?;
        let message = format!("{}: {:.3}ms", self.label, elapsed);
        console::log_1(&JsValue::from_str(&message));
        Ok(())
    }
}

/// Memory usage information for the WASM module
#[wasm_bindgen]
pub fn memory_usage() -> JsValue {
    let info = serde_json::json!({
        "note": "WASM memory usage should be checked via JavaScript Memory API"
    });

    serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_has_simd_support() {
        // Should not panic
        let _simd = has_simd_support();
    }
}
