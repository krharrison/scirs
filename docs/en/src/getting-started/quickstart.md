# Quick Start

This guide walks you through the basics of SciRS2 with working code examples.

## Create a Project

```bash
cargo new scirs2-example
cd scirs2-example
```

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-core = "0.4.0"
scirs2-linalg = "0.4.0"
scirs2-stats = "0.4.0"
scirs2-optimize = "0.4.0"
```

## Linear Algebra Basics

Matrix operations are the most fundamental part of SciRS2.

```rust
use scirs2_core::ndarray::{array, Array2};
use scirs2_linalg::{det, inv, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Solve the linear system Ax = b
    let a = array![[3.0, 1.0], [1.0, 2.0]];
    let b = array![9.0, 8.0];

    let x = solve(&a.view(), &b.view())?;
    println!("Solution: {}", x);  // [2.0, 3.0]

    // Eigenvalue decomposition
    // Equivalent to scipy.linalg.eigh
    let matrix = array![[2.0, 1.0], [1.0, 3.0]];
    let eigenvalues = scirs2_linalg::eigh_vals(&matrix.view())?;
    println!("Eigenvalues: {}", eigenvalues);

    Ok(())
}
```

## Statistical Analysis

Work with probability distributions and statistical tests.

```rust
use scirs2_stats::distributions::{Normal, Distribution, ContinuousDistribution};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Standard normal distribution N(0, 1)
    let normal = Normal::new(0.0, 1.0)?;

    // Probability density function (PDF)
    let pdf_val = normal.pdf(0.0);
    println!("PDF(0) = {:.4}", pdf_val);  // 0.3989

    // Cumulative distribution function (CDF)
    let cdf_val = normal.cdf(1.96)?;
    println!("CDF(1.96) = {:.4}", cdf_val);  // 0.9750

    // Percent point function (inverse CDF)
    let ppf_val = normal.ppf(0.975)?;
    println!("PPF(0.975) = {:.4}", ppf_val);  // 1.9600

    Ok(())
}
```

## Optimization

Minimize functions with gradient-based and derivative-free methods.

```rust
use scirs2_optimize::minimize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize the Rosenbrock function
    // f(x, y) = (1-x)^2 + 100*(y-x^2)^2
    let rosenbrock = |x: &[f64]| -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    };

    // Gradient
    let grad = |x: &[f64]| -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dy = 200.0 * (x[1] - x[0] * x[0]);
        vec![dx, dy]
    };

    let x0 = vec![0.0, 0.0];
    let result = minimize(rosenbrock, &x0, Some(grad), None)?;
    println!("Minimum at: {:?}", result.x);     // Close to [1.0, 1.0]
    println!("Minimum value: {:.6}", result.fun); // Close to 0.0

    Ok(())
}
```

## Signal Processing

Design filters and process signals.

```rust
use scirs2_core::ndarray::Array1;
use scirs2_signal::filter_design::butter;
use scirs2_signal::filtering::filtfilt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a signal sampled at 1000 Hz
    let fs = 1000.0;
    let t: Array1<f64> = Array1::linspace(0.0, 1.0, 1000);

    // Sum of 50 Hz and 200 Hz sine waves
    let signal: Array1<f64> = t.mapv(|ti| {
        (2.0 * std::f64::consts::PI * 50.0 * ti).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 200.0 * ti).sin()
    });

    // 4th-order Butterworth low-pass filter (cutoff 100 Hz)
    let (b, a) = butter(4, &[100.0 / (fs / 2.0)], None, None, None)?;

    // Zero-phase filtering
    let filtered = filtfilt(&b, &a, &signal)?;
    println!("Filtered: {} samples", filtered.len());

    Ok(())
}
```

## FFT (Fast Fourier Transform)

Analyze signals in the frequency domain.

```rust
use scirs2_core::ndarray::Array1;
use scirs2_fft::{fft, fftfreq};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;
    let fs = 1000.0;
    let dt = 1.0 / fs;

    // Test signal: 50 Hz + 120 Hz
    let t: Array1<f64> = Array1::linspace(0.0, (n as f64) * dt, n);
    let signal: Array1<f64> = t.mapv(|ti| {
        (2.0 * std::f64::consts::PI * 50.0 * ti).sin()
            + 0.8 * (2.0 * std::f64::consts::PI * 120.0 * ti).sin()
    });

    // Compute FFT
    let spectrum = fft(&signal, None)?;
    let freqs = fftfreq(n, dt)?;

    // Power spectrum
    let power: Array1<f64> = spectrum.mapv(|c| c.norm_sqr() / (n as f64));

    println!("FFT complete: {} frequency bins", freqs.len());

    Ok(())
}
```

## Next Steps

- Review the [Project Structure](./structure.md) for a full module overview
- Dive into [module-specific documentation](../modules/linalg.md) for detailed APIs
- Check the [SciPy Migration Guide](../migration/scipy.md) if you are coming from Python
