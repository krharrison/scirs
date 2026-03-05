//! Financial computing extensions for SciRS2
//!
//! This module provides a comprehensive set of financial analytics tools covering:
//!
//! ## Option Pricing
//!
//! Analytical Black-Scholes formulas, CRR binomial trees, and Monte Carlo simulation
//! for European and American options:
//!
//! ```rust
//! use scirs2_core::finance::options::{black_scholes_call, black_scholes_put, OptionType, ExerciseType};
//!
//! // Black-Scholes European call: S=100, K=100, T=1yr, r=5%, σ=20%
//! let call_price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2).expect("should succeed");
//! assert!((call_price - 10.45).abs() < 0.1);
//!
//! // CRR binomial tree (500 steps, American put)
//! use scirs2_core::finance::options::binomial_option;
//! let am_put = binomial_option(100.0, 100.0, 1.0, 0.05, 0.2, 500,
//!     OptionType::Put, ExerciseType::American).expect("should succeed");
//! assert!(am_put > 0.0);
//! ```
//!
//! ## Fixed Income
//!
//! Plain-vanilla bond pricing, duration, convexity, and YTM solving:
//!
//! ```rust
//! use scirs2_core::finance::fixed_income::{bond_price, bond_duration, yield_to_maturity};
//!
//! // Par bond: coupon_rate == ytm => price == face
//! let price = bond_price(1000.0, 0.05, 0.05, 10).expect("should succeed");
//! assert!((price - 1000.0).abs() < 1e-6);
//!
//! // Solve for YTM given market price
//! let ytm = yield_to_maturity(950.0, 1000.0, 0.05, 10).expect("should succeed");
//! assert!(ytm > 0.05);  // discount bond -> ytm > coupon
//! ```
//!
//! ## Risk Metrics
//!
//! Historical VaR, CVaR/Expected Shortfall, Sharpe & Sortino ratios, max drawdown:
//!
//! ```rust
//! use scirs2_core::finance::risk::{value_at_risk, sharpe_ratio, max_drawdown};
//!
//! let returns = vec![0.01, -0.02, 0.015, -0.005, 0.008, -0.012, 0.02,
//!                    -0.018, 0.005, 0.003, -0.025, 0.011];
//!
//! let var_95 = value_at_risk(&returns, 0.95).expect("should succeed");
//! assert!(var_95 > 0.0);
//!
//! let prices: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
//! let dd = max_drawdown(&prices).expect("should succeed");
//! assert_eq!(dd, 0.0); // monotone rising
//! ```
//!
//! ## Portfolio Analytics
//!
//! Mean-variance portfolio theory (Markowitz 1952):
//!
//! ```rust
//! use scirs2_core::finance::portfolio::{portfolio_return, portfolio_variance, efficient_frontier};
//! use scirs2_core::ndarray::array;
//!
//! let weights = [0.6, 0.4];
//! let returns = [0.08, 0.12];
//! let rp = portfolio_return(&weights, &returns).expect("should succeed");
//! assert!((rp - 0.096).abs() < 1e-10);
//!
//! let cov = array![[0.04, 0.01], [0.01, 0.09]];
//! let frontier = efficient_frontier(&returns, &cov, 20).expect("should succeed");
//! assert_eq!(frontier.len(), 20);
//! ```

pub mod fixed_income;
pub mod options;
pub mod portfolio;
pub mod risk;

// Convenience re-exports of the most commonly used items
pub use fixed_income::{bond_convexity, bond_duration, bond_price, yield_to_maturity};
pub use options::{
    binomial_option, black_scholes_call, black_scholes_greeks, black_scholes_put,
    monte_carlo_option, ExerciseType, Greeks, OptionType,
};
pub use portfolio::{
    efficient_frontier, min_variance_weights, portfolio_return, portfolio_variance,
};
pub use risk::{
    conditional_var, max_drawdown, rolling_sharpe, sharpe_ratio, sortino_ratio, value_at_risk,
};
