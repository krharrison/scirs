# Time Series (scirs2-series)

`scirs2-series` provides time series analysis, forecasting, causal discovery, and
state-space models. It covers classical methods (ARIMA, ETS), neural forecasters
(N-BEATS, N-HiTS, PatchTST), and advanced topics like conformal prediction and
hierarchical reconciliation.

## Classical Forecasting

### ARIMA

```rust,ignore
use scirs2_series::arima::{ARIMA, ARIMAOptions};

// ARIMA(p=2, d=1, q=1)
let model = ARIMA::new(2, 1, 1, ARIMAOptions::default())?;
let fitted = model.fit(&training_data)?;

// Forecast 10 steps ahead
let forecast = fitted.predict(10)?;
println!("Forecast: {:?}", forecast.values);
println!("Confidence intervals: {:?}", forecast.intervals);
```

### Exponential Smoothing

```rust,ignore
use scirs2_series::ets::{ExponentialSmoothing, ETSConfig};

let config = ETSConfig::new()
    .with_trend("additive")
    .with_seasonal("multiplicative", 12);

let model = ExponentialSmoothing::new(config)?;
let fitted = model.fit(&monthly_data)?;
let forecast = fitted.predict(24)?;  // 24-month forecast
```

## Neural Forecasting

### N-BEATS

```rust,ignore
use scirs2_series::neural_forecast::nbeats::{NBeats, NBeatsConfig};

let config = NBeatsConfig {
    input_length: 168,
    output_length: 24,
    num_stacks: 30,
    num_blocks_per_stack: 1,
    hidden_dim: 256,
};
let model = NBeats::new(config)?;
let forecast = model.predict(&history)?;
```

### PatchTST

Transformer-based forecasting with patched input:

```rust,ignore
use scirs2_series::neural_forecast::patchtst::{PatchTST, PatchTSTConfig};

let config = PatchTSTConfig {
    input_length: 512,
    output_length: 96,
    patch_length: 16,
    stride: 8,
    d_model: 128,
    num_heads: 8,
    num_layers: 3,
};
let model = PatchTST::new(config)?;
```

### N-HiTS (Hierarchical Interpolation)

```rust,ignore
use scirs2_series::neural_forecast::nhits::{NHiTS, NHiTSConfig};

let model = NHiTS::new(NHiTSConfig {
    input_length: 168,
    output_length: 24,
    num_stacks: 3,
    pooling_sizes: vec![4, 2, 1],
})?;
```

## State-Space Models

### S4 / S4D

Structured state-space models for long-range dependencies:

```rust,ignore
use scirs2_series::state_space::{S4, S4D, S4Config};

let config = S4Config {
    d_model: 64,
    d_state: 64,
    num_layers: 4,
};
let model = S4::new(config)?;
let output = model.forward(&sequence)?;

// S4D: diagonal variant (faster)
let model = S4D::new(config)?;
```

### Mamba (Selective State Spaces)

```rust,ignore
use scirs2_series::state_space::mamba::{Mamba, MambaConfig};

let model = Mamba::new(MambaConfig {
    d_model: 128,
    d_state: 16,
    expand: 2,
    dt_rank: "auto",
})?;
let output = model.forward(&input)?;
```

## Causal Discovery

### PC Algorithm

```rust,ignore
use scirs2_series::causality::{pc_algorithm, PCOptions};

let options = PCOptions::default()
    .with_alpha(0.05)
    .with_max_cond_set(3);

let dag = pc_algorithm(&data, options)?;
println!("Edges: {:?}", dag.edges());
```

### PCMCI

Causal discovery for time series with lagged effects:

```rust,ignore
use scirs2_series::causality::pcmci::{PCMCI, PCMCIOptions};

let pcmci = PCMCI::new(PCMCIOptions {
    max_lag: 5,
    alpha: 0.05,
    independence_test: "partial_correlation",
})?;
let causal_graph = pcmci.run(&multivariate_ts)?;
```

## Hierarchical Reconciliation

Ensure forecasts across hierarchical levels are coherent:

```rust,ignore
use scirs2_series::hierarchical::{BottomUp, MinTrace, ERMReconciler};

// Bottom-up reconciliation
let reconciled = BottomUp::reconcile(&base_forecasts, &summing_matrix)?;

// MinTrace (optimal combination)
let reconciled = MinTrace::reconcile(&base_forecasts, &summing_matrix, &residuals)?;
```

## Streaming and Online

### Online Learning

```rust,ignore
use scirs2_series::streaming::{OnlineLearner, StreamingJohansen};

// Online model that updates with each new observation
let mut learner = OnlineLearner::new(model_config)?;
for observation in stream {
    let prediction = learner.predict(1)?;
    learner.update(&observation)?;
}

// Streaming cointegration testing
let mut johansen = StreamingJohansen::new(num_vars, max_lag)?;
johansen.update(&new_data)?;
let test_result = johansen.test()?;
```

## Volatility Models

```rust,ignore
use scirs2_series::volatility::{GARCH, GPGarch};

// Standard GARCH(1,1)
let model = GARCH::new(1, 1)?;
let fitted = model.fit(&returns)?;
let volatility_forecast = fitted.forecast_variance(10)?;

// GP-GARCH: Gaussian process enhanced GARCH
let model = GPGarch::new(1, 1, kernel)?;
```
