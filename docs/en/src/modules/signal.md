# Signal Processing (scirs2-signal)

`scirs2-signal` provides digital signal processing capabilities modeled after `scipy.signal`,
covering filter design, spectral analysis, wavelet transforms, convolution, and LTI systems.

## Convolution

```rust
use scirs2_signal::convolve;

fn convolution_demo() -> Result<(), Box<dyn std::error::Error>> {
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = vec![0.25, 0.5, 0.25];

    // "same" keeps the output length equal to the input
    let filtered = convolve(&signal, &kernel, "same")?;

    // "full" returns the complete convolution (length m+n-1)
    let full = convolve(&signal, &kernel, "full")?;

    Ok(())
}
```

## Filter Design

### Butterworth Filters

```rust,ignore
use scirs2_signal::filter::{butter, sosfilt, FilterType};

// Design a 4th-order lowpass Butterworth filter
// Cutoff at 0.1 * Nyquist frequency
let sos = butter(4, &[0.1], FilterType::Lowpass, false, None)?;

// Apply the filter to a signal
let filtered = sosfilt(&sos, &signal)?;
```

### Other IIR Filters

```rust,ignore
use scirs2_signal::filter::{cheby1, cheby2, ellip, bessel, FilterType};

// Chebyshev Type I (ripple in passband)
let sos = cheby1(4, 0.5, &[0.1], FilterType::Lowpass, false, None)?;

// Chebyshev Type II (ripple in stopband)
let sos = cheby2(4, 40.0, &[0.1], FilterType::Lowpass, false, None)?;

// Elliptic filter (ripple in both bands, steepest rolloff)
let sos = ellip(4, 0.5, 40.0, &[0.1], FilterType::Lowpass, false, None)?;

// Bessel filter (maximally flat group delay)
let sos = bessel(4, &[0.1], FilterType::Lowpass, "phase", None)?;
```

### FIR Filters

```rust,ignore
use scirs2_signal::filter::{firwin, lfilter};

// Design a 64-tap lowpass FIR filter
let taps = firwin(64, &[0.1], None, None, true, None)?;

// Apply FIR filter
let filtered = lfilter(&taps, &[1.0], &signal)?;
```

## Spectral Analysis

### Power Spectral Density

```rust,ignore
use scirs2_signal::spectral::{periodogram, welch};

// Periodogram
let (freqs, psd) = periodogram(&signal, fs, None, None)?;

// Welch's method (averaged periodograms, lower variance)
let (freqs, psd) = welch(&signal, fs, None, None, None, None)?;
```

### Spectrogram

```rust,ignore
use scirs2_signal::spectral::spectrogram;

let (times, freqs, sxx) = spectrogram(&signal, fs, window, nperseg, noverlap)?;
// sxx is a 2D array: frequency x time
```

## Wavelet Transforms

### Discrete Wavelet Transform (DWT)

```rust,ignore
use scirs2_signal::dwt::{dwt, idwt, wavedec, waverec, Wavelet};

// Single-level DWT
let (approx, detail) = dwt(&signal, Wavelet::Db4, None)?;

// Inverse DWT
let reconstructed = idwt(&approx, &detail, Wavelet::Db4, None)?;

// Multi-level decomposition
let coeffs = wavedec(&signal, Wavelet::Db4, None, Some(4))?;

// Reconstruction from coefficients
let reconstructed = waverec(&coeffs, Wavelet::Db4, None)?;
```

### 2D Wavelet Transform

```rust,ignore
use scirs2_signal::dwt2d_enhanced::{dwt2d, idwt2d, Wavelet2D};

// 2D DWT for image processing
let (ll, lh, hl, hh) = dwt2d(&image, Wavelet2D::Db4)?;

// Inverse 2D DWT
let reconstructed = idwt2d(&ll, &lh, &hl, &hh, Wavelet2D::Db4)?;
```

## Window Functions

```rust,ignore
use scirs2_signal::window::{hann, hamming, blackman, kaiser};

let w = hann(256)?;
let w = hamming(256)?;
let w = blackman(256)?;
let w = kaiser(256, 8.6)?;  // beta = 8.6
```

## LTI Systems

```rust,ignore
use scirs2_signal::lti::{TransferFunction, StateSpace};

// Transfer function representation: H(s) = (s + 1) / (s^2 + 2s + 1)
let tf = TransferFunction::new(vec![1.0, 1.0], vec![1.0, 2.0, 1.0])?;

// Frequency response
let (w, h) = tf.freqresp(1000)?;

// Step response
let (t, y) = tf.step(None, None)?;

// Convert to state-space
let ss = tf.to_statespace()?;
```

## Advanced Features

### Echo Cancellation

Multi-delay acoustic echo cancellation with per-band NLMS:

```rust,ignore
use scirs2_signal::echo_cancellation::{MultiDelayAEC, AECConfig};

let config = AECConfig::default()
    .with_filter_length(1024)
    .with_step_size(0.1);
let mut aec = MultiDelayAEC::new(config)?;

// Process frame by frame
let clean = aec.process_frame(&mic_signal, &reference)?;
```

### Modal Analysis

Operational modal analysis for structural health monitoring:

```rust,ignore
use scirs2_signal::modal_analysis::{esprit, music};

// ESPRIT algorithm for frequency estimation
let frequencies = esprit(&signal, model_order, fs)?;

// MUSIC pseudo-spectrum
let (freqs, spectrum) = music(&signal, model_order, fs, nfft)?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.signal.convolve` | `convolve` |
| `scipy.signal.butter` | `filter::butter` |
| `scipy.signal.sosfilt` | `filter::sosfilt` |
| `scipy.signal.cheby1` | `filter::cheby1` |
| `scipy.signal.welch` | `spectral::welch` |
| `scipy.signal.spectrogram` | `spectral::spectrogram` |
| `scipy.signal.get_window` | `window::*` |
