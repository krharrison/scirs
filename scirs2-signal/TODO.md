# scirs2-signal TODO

## Status: v0.3.1 Released (March 9, 2026)

19,685 workspace tests pass (100% pass rate). All v0.3.1 features are complete and production-ready.

---

## v0.3.1 Completed

### Core Filtering
- [x] IIR filter design: Butterworth, Chebyshev I/II, Elliptic, Bessel (analog prototypes + bilinear/impulse-invariance transformation)
- [x] FIR filter design: window method (Hamming, Hanning, Blackman, Kaiser, flat-top), Parks-McClellan / Remez exchange
- [x] Zero-phase filtering: `filtfilt` with edge-padding strategies
- [x] Specialized filters: notch, comb, allpass, peaking EQ, shelving EQ
- [x] Savitzky-Golay filter with arbitrary polynomial order and derivative
- [x] Second-order sections (SOS) cascaded representation for numerical stability
- [x] Filter analysis: `freqz`, `freqs`, group delay, pole-zero maps, stability check
- [x] Filter transformations: LP-to-BP, LP-to-BS, analog-to-digital (bilinear, impulse invariance, matched-z)

### Spectral Analysis
- [x] Periodogram and Bartlett's method
- [x] Welch's method for PSD estimation with overlapping segments
- [x] Short-time Fourier transform (STFT) and inverse STFT
- [x] Spectrogram with configurable window, overlap, FFT size
- [x] Lomb-Scargle periodogram for non-uniform sampling
- [x] Coherence and cross-power spectral density
- [x] Signal detrending (constant, linear, polynomial)

### Multitaper Spectral Estimation
- [x] DPSS (Slepian) window sequences for arbitrary bandwidth-time product
- [x] Adaptive multitaper PSD (eigenspectrum weighting by expected bias/variance trade-off)
- [x] Jackknife confidence intervals for multitaper estimates
- [x] Multitaper coherence estimation

### Parametric Spectral Estimation
- [x] AR model via Yule-Walker equations
- [x] AR model via Burg's method (recursive lattice, exact maximum entropy)
- [x] AR model via covariance and modified covariance methods
- [x] ARMA spectral estimation
- [x] MUSIC (MUltiple SIgnal Classification) pseudo-spectrum
- [x] ESPRIT for superresolution frequency estimation

### Time-Frequency Representations
- [x] Synchrosqueezing transform (SST) with phase-based reassignment
- [x] Ridge extraction from SST and reassigned spectrogram
- [x] Reassigned spectrogram (partial derivatives of phase)
- [x] Wigner-Ville distribution (WVD) and Pseudo-WVD
- [x] Cohen's class: Choi-Williams, Born-Jordan distributions
- [x] Zoom FFT (chirp-z transform) for high-resolution sub-band analysis
- [x] Hilbert transform, analytic signal, instantaneous frequency/amplitude

### Wavelet Transforms
- [x] DWT: Haar, Daubechies (db2-db20), Symlets (sym2-sym20), Coiflets (coif1-coif5), Biorthogonal
- [x] CWT: Morlet, Paul, DOG, Mexican Hat wavelets
- [x] Stationary / undecimated DWT (SWT)
- [x] Dual-tree complex wavelet transform (DTCWT)
- [x] Wavelet packets (full binary tree decomposition with best-basis selection)
- [x] Wavelet denoising: VisuShrink, BayesShrink, SUREshrink; hard and soft thresholding

### EMD / HHT
- [x] EMD: sifting algorithm with Cauchy and S-number stopping criteria; cubic spline envelopes
- [x] EEMD: ensemble EMD with configurable noise amplitude and ensemble size
- [x] CEEMDAN (Complete EEMD with Adaptive Noise)
- [x] HHT: Hilbert transform of each IMF for instantaneous frequency and amplitude
- [x] Hilbert spectrum (time-frequency-energy representation) and marginal spectrum

### Adaptive Filters
- [x] LMS: standard, normalized (NLMS), leaky LMS, sign-error LMS
- [x] RLS: standard RLS with exponential forgetting, QR-based RLS (lattice form)
- [x] Adaptive Kalman filter for time-varying gain

### State Estimation
- [x] Kalman filter with Rauch-Tung-Striebel (RTS) smoother
- [x] Extended Kalman Filter (EKF) with analytical and numerical Jacobians
- [x] Unscented Kalman Filter (UKF) with Van der Merwe sigma-point parametrisation
- [x] Square-root EKF and UKF for improved numerical stability

### Compressed Sensing & Sparse Recovery
- [x] OMP (Orthogonal Matching Pursuit): sparsity and residual tolerance stopping
- [x] CoSaMP (Compressive Sampling Matching Pursuit)
- [x] ISTA and FISTA (Iterative Soft Thresholding Algorithm): convergence-guaranteed L1 minimisation
- [x] Basis Pursuit via ADMM
- [x] Measurement matrix construction: Gaussian, Bernoulli, subsampled DFT
- [x] Recovery quality metrics: relative error, support recovery rate

### Blind Source Separation (BSS) & ICA
- [x] FastICA: fixed-point algorithm with logcosh and kurtosis contrast
- [x] JADE: fourth-order cumulant tensor diagonalisation
- [x] SOBI: second-order blind identification using temporal structure
- [x] Convolutive BSS: frequency-domain approach with permutation alignment
- [x] NMF audio source separation with Itakura-Saito divergence and beta divergence

### Cepstral Analysis & MFCCs
- [x] Complex cepstrum, real cepstrum, inverse cepstrum
- [x] Liftering (quefrency-domain smoothing)
- [x] MFCC: mel filterbank design (HTK and Slaney), log mel spectrogram, DCT-II, delta and delta-delta coefficients
- [x] Pitch (F0) estimation: autocorrelation, YIN algorithm
- [x] Spectral features: centroid, bandwidth, roll-off, flatness, contrast

### System Identification
- [x] ARX model: least-squares estimation, order selection via AIC/MDL
- [x] ARMAX model: iterative least-squares for MA noise component
- [x] N4SID: subspace-based state-space system identification (PI-MOESP, CVA)
- [x] ERA (Eigensystem Realisation Algorithm): Hankel-matrix-based impulse response realisation
- [x] Validation: one-step-ahead prediction, residual whiteness test, fit percentage

### Matched Filter & Radar Detection
- [x] Matched filter: template correlation with SNR-optimal detection
- [x] CA-CFAR (Cell-Averaging CFAR)
- [x] OS-CFAR (Order Statistics CFAR)
- [x] GO-CFAR / SO-CFAR (Greatest Of / Smallest Of)
- [x] Linear FM (LFM/chirp) pulse compression
- [x] Range-Doppler processing: 2D FFT with Doppler windowing
- [x] Ambiguity function computation for waveform analysis

### Music Information Retrieval (MIR)
- [x] Onset detection: spectral flux, high-frequency content (HFC), complex domain
- [x] Beat tracking and tempo estimation via onset strength envelope
- [x] Chroma features: short-time Fourier chroma, CQT-based chroma
- [x] Key detection via chroma profiles
- [x] Tonal centroid (Harmonic Network features)
- [x] Structural segmentation via self-similarity matrices

### Resampling
- [x] Upsampling and downsampling with anti-aliasing filters
- [x] Arbitrary rational resampling (polyphase filterbank)
- [x] Polyphase decomposition for efficient multi-rate processing

### Waveform Generation
- [x] Sine, cosine, square (configurable duty cycle), sawtooth, triangle
- [x] Chirp: linear, quadratic, logarithmic, hyperbolic FM sweep
- [x] Gaussian pulse and Gaussian-modulated sinusoid
- [x] Unit impulse, step, ramp
- [x] Noise: white Gaussian, pink (1/f), brown/red (1/f²)

### Linear System Analysis
- [x] Transfer function and state-space representations (continuous and discrete)
- [x] Bode plot (magnitude and phase), Nyquist diagram, root locus
- [x] Step, impulse, and initial condition responses
- [x] Stability: Routh-Hurwitz (continuous), Jury (discrete), Lyapunov
- [x] Gain margin, phase margin, delay margin
- [x] System interconnection: series, parallel, feedback
- [x] Continuous-to-discrete: ZOH, Tustin / bilinear, matched pole-zero

### Peak Detection & Signal Measurements
- [x] Peak finding with distance, prominence, width, height thresholds
- [x] Peak width at fractional height (FWHM), peak area, peak asymmetry
- [x] RMS, peak, peak-to-peak, crest factor, PAR
- [x] SNR, THD (with harmonic order), SFDR

### Super-Advanced Denoising
- [x] Empirical Wiener filter via multi-estimate combination
- [x] Learnable soft-thresholding with data-driven threshold selection
- [x] Non-local means 1-D denoising

---

## v0.4.0 Roadmap

### Real-Time Streaming Processing
- [ ] Block-based filter processing with state preservation between blocks
- [ ] Ring-buffer abstraction for streaming convolution and correlation
- [ ] Online STFT with overlap-save/overlap-add block updating
- [ ] Streaming OMP for adaptive sparse coding

### GPU-Accelerated FFT Pipeline
- [ ] OxiFFT GPU backend integration for large-batch spectrograms
- [ ] GPU-accelerated matched filter bank (multiple templates simultaneously)
- [ ] Batched Welch PSD for parallel channel processing
- [ ] GPU wavelet transform for high-throughput applications

### Deep Learning-Based Denoising
- [ ] Learned speech enhancement model (Conv-TasNet architecture) in pure Rust
- [ ] Deep filtering via scirs2-neural integration
- [ ] Denoising diffusion probabilistic model for audio restoration
- [ ] Pre-trained model weight loading from oxicode format

### Modal Analysis (Structural Dynamics)
- [ ] Frequency Domain Decomposition (FDD) for operational modal analysis
- [ ] Enhanced FDD (EFDD) with damping estimation
- [ ] Stochastic Subspace Identification (SSI-COV, SSI-DATA)
- [ ] Modal Assurance Criterion (MAC) for mode shape comparison

### Advanced Array Processing
- [ ] Delay-and-sum beamforming for microphone / sensor arrays
- [ ] MVDR (Capon) beamformer
- [ ] MUSIC / ESPRIT for direction-of-arrival (DOA) estimation
- [ ] Adaptive beamforming with interference cancellation

---

## Known Issues

- CEEMDAN with very short signals (<256 samples) may produce spurious IMFs; EEMD is more stable in this regime
- N4SID identification with high model orders (>20) requires well-conditioned data; use regularized variant
- NMF audio separation is sensitive to initialization; multiple random restarts recommended for reliable separation
