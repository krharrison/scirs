# scirs2-fft Development TODO

## v0.3.3 — COMPLETED

### Sparse FFT Algorithms
- Sublinear sparse FFT (O(k log n) for k-sparse signals, randomized hashing)
- Compressed sensing-based sparse FFT (LASSO-style recovery)
- Iterative sparse FFT (robust to noise via ISTA/FISTA)
- Frequency pruning variant
- Spectral flatness-based sparse FFT
- Prony method for damped sinusoid recovery
- MUSIC algorithm (Multiple Signal Classification) for super-resolution
- Batch sparse FFT with parallel CPU execution

### Chirp-Z Transform (CZT)
- CZT on arbitrary contours in the z-plane
- Zoom FFT as a special case of CZT
- Bluestein's algorithm for prime-length FFT

### Fractional Fourier Transform (FrFT)
- Ozaktas-Arikan algorithm (decomposition into chirp multiply-FFT steps)
- Candan sampling-based algorithm
- Complex signal FrFT (`frft_complex`)
- Batch FrFT for multiple rotation angles

### Number-Theoretic Transform (NTT)
- NTT over arbitrary NTT-friendly primes
- Inverse NTT
- Negacyclic NTT (used in lattice cryptography)
- Polynomial multiplication via NTT

### Lomb-Scargle Periodogram
- Fast Lomb-Scargle (extirpolation + FFT, O(n log n + N log N))
- Generalized Lomb-Scargle with floating mean
- FAP (false alarm probability) estimation

### Mixed-Radix FFT
- Arbitrary composite-length FFT (Cooley-Tukey + Rader + Bluestein)
- Mixed-radix 2D and N-dimensional FFT
- Split-radix FFT for 2^n lengths

### DCT/DST Variants (complete set)
- DCT types I–VIII
- DST types I–VIII
- Inverse transforms for all types
- MDCT and MDST (Modified DCT/DST)
- N-dimensional DCT/DST via separable application

### Wavelet Packet Transform
- Full wavelet packet decomposition tree
- Best-basis selection via Shannon entropy criterion
- Reconstruction from any subtree
- Wavelet families: Daubechies (up to 20), Symlets, Coiflets, Biorthogonal
- Continuous wavelet transform (CWT) via FFT convolution

### Polyphase Filter Bank
- Analysis and synthesis polyphase decomposition
- DFT-modulated (cosine-modulated) filter bank
- Critically sampled and oversampled modes
- Perfect reconstruction condition check

### Hilbert-Huang Transform (HHT)
- EMD (Empirical Mode Decomposition) via cubic spline envelope
- EEMD (Ensemble EMD) with white noise injection
- CEEMDAN (Complete EEMD with Adaptive Noise)
- Hilbert spectrum from IMFs
- Instantaneous frequency via Teager energy operator

### Spectral Analysis Enhancements
- Burg AR model spectral estimation
- Welch's method with configurable averaging
- Multitaper spectral estimation (DPSS)
- ESPRIT frequency estimator
- Capon beamformer spectral estimator

### Multidimensional FFT Utilities
- `multidim.rs` / `multidim_utils.rs` — separable N-dimensional plans
- In-place tiled 2D FFT for large arrays
- Row-column FFT with configurable tile size

### Convolution and Correlation
- Overlap-save (OLS) convolution
- Overlap-add (OLA) convolution
- FFT-based cross-correlation
- FFT-based polynomial multiplication
- Correlation-based delay estimation

### Window Functions Library
- 100+ windows including Kaiser-Bessel derived (KBD), DPSS, Parzen, Bohman
- `window_functions.rs` module with parameterized window generation
- Window coherent gain and noise bandwidth computation

### Spectrogram Enhancements
- Enhanced normalized spectrogram with configurable dynamic range
- Waterfall 3D data generation (mesh, line stacks)
- Reassigned spectrogram (improved time-frequency localization)
- Synchrosqueezed STFT

---

## v0.4.0 — Planned

### GPU FFT via OxiFFT GPU Backend
- [ ] GPU-accelerated 1D/2D/ND FFT using OxiFFT GPU backend when available
- [ ] Automatic CPU/GPU dispatch based on input size and available hardware
- [ ] GPU batch FFT for many same-size transforms in parallel
- [ ] GPU overlap-save convolution for real-time filtering

### Streaming FFT for Large Data
- [ ] Streaming FFT processor with configurable buffer and overlap
- [ ] Out-of-core 2D FFT for images too large for RAM
- [ ] Streaming spectrogram with rolling window output
- [ ] Ring-buffer STFT for online/real-time applications

### Quantum FFT
- [ ] Quantum Fourier Transform circuit simulation (via `scirs2-core` quantum primitives)
- [ ] Phase estimation circuit using QFT
- [ ] Shor's algorithm building blocks

### Additional Algorithms
- [ ] Short-time fractional Fourier transform (STFRFT)
- [ ] Wigner-Ville distribution (full, smoothed)
- [ ] Ambiguity function computation
- [ ] Cyclostationary spectral analysis
- [ ] Ramanujan periodic transform

### Performance
- [ ] AVX-512 butterfly kernels for radix-4 and radix-8 FFT stages
- [ ] NEON/SVE butterfly kernels for ARM
- [ ] Cache-oblivious recursive FFT (Frigo-Johnson style)
- [ ] FFT plan serialization for ahead-of-time compilation

---

## Known Issues / Technical Debt

- `spectral.rs` was deleted and replaced by the `spectral/` submodule; verify no broken re-exports remain
- `nufft_legacy.rs` is retained for backward compatibility; deprecate and remove in v0.4.0
- EMD cubic spline envelope may not converge for highly non-stationary signals; add iteration cap with warning
- NTT works only for inputs whose length divides `p - 1`; document this constraint clearly
- Lomb-Scargle FAP estimation is approximate (chi-squared); implement bootstrap alternative
- Several spectral analysis files exceed 2000 lines; use `rslines 50` to identify split targets
- GPU sparse FFT feature flags (`cuda`, `hip`, `sycl`) depend on external hardware; CI uses mock backend
- STFT `istft` reconstruction requires correct `noverlap`; add assertion for perfect reconstruction condition
- Wavelet packet tree reconstruction is not yet invertible for all wavelet families; test suite should cover round-trip error
