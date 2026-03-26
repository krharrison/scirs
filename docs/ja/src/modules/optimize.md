# 最適化 (scirs2-optimize)

`scirs2-optimize` はSciRS2の最適化モジュールです。
SciPyの `scipy.optimize` に相当する機能を提供します。

## 主な機能

- 無制約最適化（L-BFGS, BFGS, 共役勾配法, Nelder-Mead）
- 制約付き最適化（L-BFGS-B, SLSQP, 信頼領域法）
- 大域最適化（差分進化、シミュレーテッドアニーリング）
- 整数計画法（分枝限定法, CDCL, 列生成法）
- 微分可能最適化（OptNet）
- 量子古典ハイブリッド（QAOA, VQE）
- 求根アルゴリズム（Brent法, Newton法）
- 高次元最適化（確率的座標降下、ランダム化SVD）

## 関数の最小化

### 準ニュートン法 (L-BFGS)

```rust
use scirs2_optimize::minimize;

// 目的関数
let f = |x: &[f64]| -> f64 {
    (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
};

// 勾配
let grad = |x: &[f64]| -> Vec<f64> {
    vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)]
};

let x0 = vec![0.0, 0.0];
let result = minimize(f, &x0, Some(grad), None)?;
println!("最小点: {:?}", result.x);  // [1.0, 2.5]
```

### 制約付き最適化 (L-BFGS-B)

```rust
use scirs2_optimize::lbfgsb::{LBFGSB, LBFGSBConfig, Bound};

// 箱制約付き最適化
let config = LBFGSBConfig {
    bounds: vec![
        Bound::Both(0.0, 10.0),  // 0 <= x[0] <= 10
        Bound::Lower(0.0),       // 0 <= x[1]
    ],
    ..Default::default()
};

let optimizer = LBFGSB::new(f, grad, config)?;
let result = optimizer.minimize(&[5.0, 5.0])?;
```

## 求根

```rust
use scirs2_optimize::roots::{brentq, newton};

// Brent法: f(x) = 0 の根を区間 [a, b] で求める
let f = |x: f64| -> f64 { x * x - 2.0 };
let root = brentq(f, 0.0, 2.0, None)?;
// root ≈ 1.4142 (√2)

// Newton法: 導関数も利用
let df = |x: f64| -> f64 { 2.0 * x };
let root = newton(f, df, 1.0, None)?;
```

## 大域最適化

```rust
use scirs2_optimize::global::{differential_evolution, DEConfig};

// 差分進化法（微分不要の大域最適化）
let rastrigin = |x: &[f64]| -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
};

let config = DEConfig {
    bounds: vec![(-5.12, 5.12); 5],  // 5次元
    max_iter: 1000,
    ..Default::default()
};

let result = differential_evolution(rastrigin, config)?;
```

## SciPy対応表

| SciPy | SciRS2 | 説明 |
|-------|--------|------|
| `scipy.optimize.minimize` | `minimize` | 関数の最小化 |
| `scipy.optimize.minimize(method='L-BFGS-B')` | `lbfgsb::LBFGSB` | 箱制約付きL-BFGS |
| `scipy.optimize.brentq` | `roots::brentq` | Brent法求根 |
| `scipy.optimize.newton` | `roots::newton` | Newton法求根 |
| `scipy.optimize.differential_evolution` | `global::differential_evolution` | 差分進化 |
| `scipy.optimize.linprog` | `linprog` | 線形計画法 |
