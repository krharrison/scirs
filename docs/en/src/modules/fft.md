# FFT (scirs2-fft)

`scirs2-fft` provides Fast Fourier Transform implementations modeled after `scipy.fft`.
It uses OxiFFT as its backend (pure Rust, no FFTW dependency) and supports 1D/2D/ND
transforms, DCT/DST, STFT, NUFFT, and fractional FFT.

## Basic FFT

```rust
use scirs2_fft::{fft, ifft};

fn fft_demo() -> Result<(), Box<dyn std::error::Error>> {
    let signal = vec![1.0, 2.0, 3.0, 4.0];

    // Forward FFT: time domain to frequency domain
    let spectrum = fft(&signal, None)?;

    // Inverse FFT: frequency domain to time domain
    let recovered = ifft(&spectrum, None)?;

    // Verify round-trip
    for (orig, rec) in signal.iter().zip(recovered.iter()) {
        assert!((orig - rec.re).abs() < 1e-10);
    }

    Ok(())
}
```

## Real FFT

For real-valued inputs, `rfft` is more efficient because it exploits conjugate symmetry
and returns only the positive-frequency half of the spectrum:

```rust
use scirs2_fft::{rfft, irfft};

fn rfft_demo() -> Result<(), Box<dyn std::error::Error>> {
    let signal = vec![1.0, 0.5, -0.5, -1.0, 0.0, 0.5, 1.0, -0.5];

    // RFFT returns n/2 + 1 complex bins
    let spectrum = rfft(&signal, None)?;
    assert_eq!(spectrum.len(), signal.len() / 2 + 1);

    // Inverse RFFT recovers the original real signal
    let recovered = irfft(&spectrum, Some(signal.len()))?;

    Ok(())
}
```

## 2D and N-Dimensional FFT

```rust,ignore
use scirs2_core::ndarray::Array2;
use scirs2_fft::{fft2, ifft2, fftn};

// 2D FFT (e.g., for images)
let image = Array2::<f64>::zeros((256, 256));
let spectrum = fft2(&image, None, None, None)?;
let recovered = ifft2(&spectrum, None, None, None)?;

// N-dimensional FFT
let volume = Array3::<f64>::zeros((64, 64, 64));
let spectrum = fftn(&volume, None, None, None)?;
```

## DCT and DST

Discrete Cosine Transform and Discrete Sine Transform:

```rust,ignore
use scirs2_fft::{dct, idct, dst};

// DCT-II (the most common type, used in JPEG)
let coeffs = dct(&signal, 2, None)?;

// Inverse DCT
let recovered = idct(&coeffs, 2, None)?;

// DST
let coeffs = dst(&signal, 1, None)?;
```

## Short-Time Fourier Transform (STFT)

For time-frequency analysis of non-stationary signals:

```rust,ignore
use scirs2_fft::{stft, Window};

// STFT with Hann window
let result = stft(
    &signal,
    Window::Hann,
    256,        // window length
    128,        // hop size (overlap = window_len - hop)
    256,        // FFT size
    1.0,        // sampling rate
)?;
// result contains time, frequency, and complex spectrogram arrays
```

## Frequency Utilities

```rust,ignore
use scirs2_fft::{fftfreq, rfftfreq, fftshift};

// Frequency bins for a length-n FFT at sample rate fs
let freqs = fftfreq(1024, 1.0 / 44100.0);

// Positive-frequency bins for RFFT
let freqs = rfftfreq(1024, 1.0 / 44100.0);

// Shift zero-frequency to center
let shifted = fftshift(&spectrum);
```

## Advanced Transforms

### Non-Uniform FFT (NUFFT)

For irregularly-sampled data:

```rust,ignore
use scirs2_fft::nufft::{nufft_type1, nufft_type2};

// Type 1: non-uniform points to uniform grid
let spectrum = nufft_type1(&non_uniform_points, &values, grid_size, tolerance)?;

// Type 2: uniform grid to non-uniform points
let values = nufft_type2(&spectrum, &query_points, tolerance)?;
```

### Fractional Fourier Transform

```rust,ignore
use scirs2_fft::fractional::{frfft, DFrFT};

// Fractional FFT with order alpha (alpha=1 is standard FFT)
let result = frfft(&signal, 0.5)?;  // half-order transform

// Discrete fractional FT
let dfrft = DFrFT::new(signal.len())?;
let result = dfrft.transform(&signal, 0.75)?;
```

### Quantum Fourier Transform

```rust,ignore
use scirs2_fft::quantum::{qft, qpe};

// Quantum Fourier Transform simulation
let state = qft(&input_state, num_qubits)?;

// Quantum Phase Estimation
let phase = qpe(&unitary, &eigenstate, precision_qubits)?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.fft.fft` | `fft` |
| `scipy.fft.ifft` | `ifft` |
| `scipy.fft.rfft` | `rfft` |
| `scipy.fft.fft2` | `fft2` |
| `scipy.fft.dct` | `dct` |
| `scipy.fft.fftfreq` | `fftfreq` |
| `scipy.fft.fftshift` | `fftshift` |
