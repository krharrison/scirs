# 統計 (scirs2-stats)

`scirs2-stats` はSciRS2の統計モジュールです。
SciPyの `scipy.stats` に相当する機能を提供し、確率分布、統計検定、
ベイズ推論、因果推論などの幅広い統計手法を実装しています。

## 主な機能

- 連続・離散確率分布（正規、ガンマ、ベータ、ポアソンなど）
- 統計検定（t検定、カイ二乗検定、KS検定など）
- ベイズ推論（MCMC、変分推論、ADVI）
- 因果推論（PC, FCI, ID アルゴリズム）
- コンフォーマル予測（分割、正規化、CQR, RAPS）
- 回帰分析（線形、一般化線形モデル）

## 確率分布

### 連続分布

```rust
use scirs2_stats::distributions::{
    Normal, Gamma, Beta, Uniform,
    Distribution, ContinuousDistribution,
};

// 正規分布
let normal = Normal::new(0.0, 1.0)?;  // N(mu=0, sigma=1)
let pdf = normal.pdf(0.0);             // 確率密度関数
let cdf = normal.cdf(1.96)?;           // 累積分布関数
let ppf = normal.ppf(0.975)?;          // パーセント点関数（逆CDF）

// ガンマ分布
let gamma = Gamma::new(2.0, 1.0)?;    // shape=2, scale=1
let mean = gamma.mean()?;              // 期待値
let var = gamma.variance()?;           // 分散

// ベータ分布
let beta = Beta::new(2.0, 5.0)?;      // alpha=2, beta=5
let mode = beta.mode()?;               // 最頻値
```

### 離散分布

```rust
use scirs2_stats::distributions::{Poisson, Binomial, DiscreteDistribution};

// ポアソン分布
let poisson = Poisson::new(3.0)?;     // lambda=3
let pmf = poisson.pmf(2)?;            // 確率質量関数 P(X=2)

// 二項分布
let binom = Binomial::new(10, 0.3)?;  // n=10, p=0.3
let cdf = binom.cdf(5)?;              // P(X <= 5)
```

## 統計検定

### t検定

```rust
use scirs2_stats::hypothesis::{ttest_1samp, ttest_ind};
use scirs2_core::ndarray::array;

// 一標本t検定: 母平均が0かどうか
let data = array![1.2, 2.3, 1.8, 2.1, 1.5];
let result = ttest_1samp(&data.view(), 0.0)?;
println!("t統計量: {:.4}", result.statistic);
println!("p値: {:.4}", result.pvalue);

// 二標本t検定（独立）
let group_a = array![5.1, 5.5, 4.8, 5.2, 5.0];
let group_b = array![4.2, 4.8, 4.5, 4.3, 4.1];
let result = ttest_ind(&group_a.view(), &group_b.view(), false)?;
```

## ベイズ推論

### MCMC

```rust
use scirs2_stats::mcmc::{MetropolisHastings, MCMCConfig};

// Metropolis-Hastingsサンプリング
let log_posterior = |theta: &[f64]| -> f64 {
    // 対数事後分布を定義
    let mu = theta[0];
    let sigma = theta[1];
    if sigma <= 0.0 { return f64::NEG_INFINITY; }
    // 尤度 + 事前分布
    -0.5 * mu * mu - sigma.ln()
};

let config = MCMCConfig {
    n_samples: 10000,
    n_burnin: 1000,
    ..Default::default()
};

let sampler = MetropolisHastings::new(log_posterior, config)?;
let samples = sampler.sample(&[0.0, 1.0])?;
```

## コンフォーマル予測

```rust
use scirs2_stats::conformal::{SplitConformal, ConformalPredictor};

// 分割コンフォーマル予測
// 校正データのスコア（非適合度スコア）から予測区間を構築
let calibration_scores = vec![0.1, 0.3, 0.2, 0.5, 0.15, 0.4, 0.25];
let alpha = 0.1; // 90% 被覆率

let predictor = SplitConformal::new(&calibration_scores, alpha)?;
let (lower, upper) = predictor.predict_interval(3.5)?;
println!("90%予測区間: [{:.2}, {:.2}]", lower, upper);
```

## SciPy対応表

| SciPy | SciRS2 | 説明 |
|-------|--------|------|
| `scipy.stats.norm` | `Normal::new` | 正規分布 |
| `scipy.stats.gamma` | `Gamma::new` | ガンマ分布 |
| `scipy.stats.beta` | `Beta::new` | ベータ分布 |
| `scipy.stats.ttest_1samp` | `ttest_1samp` | 一標本t検定 |
| `scipy.stats.ttest_ind` | `ttest_ind` | 二標本t検定 |
| `scipy.stats.kstest` | `kstest` | Kolmogorov-Smirnov検定 |
| `scipy.stats.linregress` | `regression::linregress` | 線形回帰 |
