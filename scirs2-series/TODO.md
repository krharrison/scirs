# scirs2-series TODO

## Status: v0.3.4 Released (March 18, 2026)

19,644 workspace tests pass (100% pass rate). All v0.3.4 features are complete and production-ready.

---

## v0.3.3 Completed

### Neural Architecture Forecasters
- [x] Temporal Fusion Transformer (TFT): variable selection networks, gating (GLU), static covariate encoding, multi-horizon attention decoder
- [x] N-BEATS: neural basis expansion with trend and seasonality stacks; generic and interpretable variants
- [x] N-HiTS: hierarchical interpolation with multi-rate signal sampling and multi-resolution blocks
- [x] DeepAR: autoregressive LSTM with probabilistic output distributions (Gaussian, negative binomial, student-t)
- [x] Simple neural forecast API: common interface across all neural models with configurable training loops

### State-Space Models & Kalman Filtering
- [x] Kalman filter and Rauch-Tung-Striebel smoother
- [x] Extended Kalman filter (EKF) with analytical or numerical Jacobians
- [x] Unscented Kalman filter (UKF) with sigma-point propagation (Merwe parametrization)
- [x] Structural time series: local level, local linear trend, seasonal, cycle components
- [x] Dynamic linear models (time-varying system matrices)
- [x] Innovations state space representation (ETS models)

### Volatility Models
- [x] GARCH(p,q) — QMLE estimation, forecasting, simulation
- [x] EGARCH — exponential GARCH with asymmetric leverage
- [x] FIGARCH — fractionally integrated GARCH for long-memory volatility
- [x] GJR-GARCH — asymmetric response to positive/negative shocks
- [x] ARCH-LM test, Ljung-Box test on squared residuals

### Long-Memory Processes
- [x] ARFIMA estimation (Whittle, CSS, and exact ML)
- [x] Hurst exponent: R/S analysis, DFA, Whittle spectral estimator, variogram method
- [x] Fractional differencing operator (exact and fast approximate via convolution)
- [x] FARIMA simulation with specified memory parameter

### Granger Causality & Cointegration
- [x] Granger causality: Wald test (F-statistic), bootstrap p-values, multivariate block-exogeneity
- [x] Transfer entropy with bootstrap significance testing and bias correction
- [x] Convergent cross mapping (CCM) for nonlinear causal detection
- [x] Engle-Granger two-step cointegration test
- [x] Johansen trace and maximum-eigenvalue cointegration tests with critical values
- [x] VECM estimation, impulse response functions, and forecast error variance decomposition

### Conformal Prediction
- [x] Split conformal prediction: exchangeable and time-series-adapted (EnbPI) variants
- [x] Adaptive conformal inference (ACI) for online coverage guarantees
- [x] Mondrian conformal for conditional coverage by covariate stratum
- [x] Calibration diagnostics: empirical coverage plots, Winkler score, interval sharpness

### Intermittent Demand Forecasting
- [x] Croston's method (separated demand size and interval models)
- [x] Syntetos-Boylan Approximation (SBA) with bias correction
- [x] Teunter-Syntetos-Babai (TSB) model with demand probability update
- [x] Intermittency classification (smooth, erratic, lumpy, intermittent)

### Hierarchical Forecasting & Reconciliation
- [x] Aggregation strategies: bottom-up, top-down (AHP, PHA, TDA), middle-out
- [x] MinT (trace minimisation) with sample, shrinkage, and structural covariance estimates
- [x] WLS (weighted least squares) reconciliation
- [x] OLS reconciliation (equal weight)
- [x] Cross-temporal reconciliation

### Streaming / Online Algorithms
- [x] ADWIN (Adaptive Windowing) concept drift detector with statistical guarantees
- [x] Online ARIMA with recursive least squares coefficient tracking
- [x] Streaming statistics: mean, variance (Welford), quantiles (P² and KLL sketch)
- [x] Online anomaly: CUSUM, EWMA control charts, streaming isolation forest
- [x] Reservoir sampling and sliding window aggregation

### Functional Data Analysis (FDA)
- [x] Functional PCA (FPCA) with PACE algorithm for sparse and irregular observations
- [x] B-spline and Fourier basis expansions, smoothing spline roughness penalties
- [x] Scalar-on-function regression (functional linear model)
- [x] Functional clustering (k-centres functional, hierarchical functional)
- [x] Dynamic time warping barycenter averaging (DBA)

### Regime-Switching Models
- [x] Markov-switching autoregression (MS-AR) with Hamilton filter and EM estimation
- [x] Threshold autoregressive (TAR) and self-exciting TAR (SETAR) models
- [x] Smooth transition autoregressive (STAR) models (logistic and exponential)
- [x] Bai-Perron multiple structural break test

### Probabilistic Forecasting & Evaluation
- [x] CRPS (Continuous Ranked Probability Score) and log score
- [x] Reliability diagrams and PIT histograms
- [x] Diebold-Mariano test for forecast comparison
- [x] MASE, SMAPE, WAPE, hierarchical MASE

### Classical Models (Enhanced)
- [x] Auto-ARIMA: stepwise AIC/BIC search and grid search with parallel evaluation
- [x] TBATS with automatic period selection
- [x] Theta method and Theta-F (optimized theta)
- [x] Prophet-style seasonality decomposition with Fourier-series seasonal components and holiday effects

### Change Detection & Anomaly Detection
- [x] PELT with multiple cost functions (L1, L2, RBF, AR)
- [x] Binary segmentation (greedy and exact)
- [x] Bayesian online change point detection (BOCPD) with hazard function
- [x] Kernel change detection via MMD statistics
- [x] SPC charts: Shewhart, CUSUM, EWMA with control limits
- [x] Matrix profile and motif/discord discovery

### Feature Engineering (60+ features)
- [x] Statistical: 20+ moment and distributional features
- [x] Frequency domain: spectral entropy, centroid, bandwidth, dominant frequency ratio
- [x] Complexity: ApEn, SampEn, permutation entropy, Lempel-Ziv, fractal dimension
- [x] Lag-based: ACF/PACF at multiple lags, partial correlations
- [x] Automated selection: filter (MI, F-test, variance), wrapper (forward/backward/RFE), embedded (LASSO, RF importance)

### Domain-Specific Extensions
- [x] Financial: GARCH volatility, 15+ technical indicators (RSI, MACD, Bollinger Bands, CCI, MFI, OBV, ATR, Parabolic SAR)
- [x] Environmental: heat wave detection, SPI drought index, growing degree days, SOI/NAO climate indices, atmospheric storm detection
- [x] Biomedical: ECG R-peak detection (Pan-Tompkins), HRV analysis, EEG frequency bands, EMG onset detection
- [x] IoT sensors: environmental sensor fusion, GPS activity recognition, predictive maintenance scoring, data quality assessment

---

## v0.4.0 Roadmap

### Foundation Model Interface
- [ ] Fine-tuning interface for pre-trained time series foundation models (TimeGPT-style)
- [ ] Zero-shot forecasting adapter layer
- [ ] Prompt-based time series conditioning API

### Neural ODE for Time Series
- [ ] Latent ODE / ODE-RNN for irregular time series
- [ ] Continuous normalizing flow models for density estimation
- [ ] Physics-informed neural time series models

### Ultra-Long Context Handling
- [ ] FlashAttention integration for TFT with very long lookback windows (10k+)
- [ ] State-space sequence models (Mamba / S4) for linear-time long-range dependencies
- [ ] Hierarchical attention with sparse patterns for ultra-long sequences

### Advanced Causality
- [ ] PC algorithm for causal structure learning from time series
- [ ] PCMCI algorithm (Peter and Clark Momentary Conditional Independence)
- [ ] Causal discovery with latent confounders (FCI for time series)

### Bayesian Nonparametric Time Series
- [ ] GP-state-space models (GP-SSM) with particle MCMC fitting
- [ ] Infinite hidden Markov model (iHMM) via stick-breaking construction
- [ ] Nonparametric GARCH via GP volatility functions

### Streaming Enhancements
- [ ] RIVER integration bridge for additional online learners
- [ ] Incremental cointegration testing in streaming VAR
- [ ] Online hierarchical reconciliation with incremental MinT

---

## Known Issues

- DeepAR with negative-binomial output can exhibit numerical instability when series contain long runs of zeros; use TSB for highly intermittent demand instead
- FIGARCH estimation is slow for series longer than 10,000 points without the `parallel` feature enabled
- FPCA with very sparse observations (fewer than 5 observations per subject) may produce poorly estimated eigenfunctions
