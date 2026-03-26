# Special Functions (scirs2-special)

`scirs2-special` provides mathematical special functions modeled after `scipy.special`,
including gamma functions, Bessel functions, elliptic integrals, hypergeometric functions,
and advanced number-theoretic functions.

## Gamma and Related Functions

```rust,ignore
use scirs2_special::{gamma, gammaln, digamma, beta as beta_fn, betainc};

// Gamma function: Gamma(5) = 4! = 24
let g = gamma(5.0)?;

// Log-gamma (numerically stable for large arguments)
let lg = gammaln(1000.0)?;

// Digamma (psi) function: d/dx ln(Gamma(x))
let psi = digamma(2.0)?;

// Beta function: B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
let b = beta_fn(2.0, 3.0)?;

// Regularized incomplete beta function
let ib = betainc(0.5, 2.0, 3.0)?;
```

## Bessel Functions

```rust,ignore
use scirs2_special::{j0, j1, jn, y0, y1, yn, iv, kv};

// Bessel J (first kind)
let j0_val = j0(1.0)?;     // J_0(1)
let j1_val = j1(1.0)?;     // J_1(1)
let jn_val = jn(3, 2.0)?;  // J_3(2)

// Bessel Y (second kind)
let y0_val = y0(1.0)?;
let y1_val = y1(1.0)?;

// Modified Bessel functions
let iv_val = iv(0.0, 1.0)?;  // I_0(1)
let kv_val = kv(0.0, 1.0)?;  // K_0(1)
```

## Elliptic Integrals and Functions

```rust,ignore
use scirs2_special::{ellipk, ellipe, ellipj};

// Complete elliptic integrals
let k = ellipk(0.5)?;  // K(m) with m = k^2
let e = ellipe(0.5)?;  // E(m)

// Jacobi elliptic functions: sn, cn, dn
let (sn, cn, dn, _) = ellipj(1.0, 0.5)?;
```

## Error Function

```rust,ignore
use scirs2_special::{erf, erfc, erfinv};

let e = erf(1.0)?;       // erf(1) ~ 0.8427
let ec = erfc(1.0)?;     // erfc(1) = 1 - erf(1) ~ 0.1573
let ei = erfinv(0.5)?;   // inverse error function
```

## Hypergeometric Functions

```rust,ignore
use scirs2_special::{hyp1f1, hyp2f1};

// Confluent hypergeometric: 1F1(a; b; z)
let val = hyp1f1(1.0, 2.0, 0.5)?;

// Gauss hypergeometric: 2F1(a, b; c; z)
let val = hyp2f1(1.0, 2.0, 3.0, 0.5)?;
```

## Orthogonal Polynomials

```rust,ignore
use scirs2_special::{legendre, hermite, laguerre, chebyshev};

// Legendre polynomial P_n(x)
let p5 = legendre(5, 0.5)?;

// Hermite polynomial H_n(x)
let h3 = hermite(3, 1.0)?;

// Laguerre polynomial L_n(x)
let l4 = laguerre(4, 2.0)?;

// Chebyshev polynomial T_n(x)
let t6 = chebyshev(6, 0.5)?;
```

## Advanced Functions

### Symbolic Differentiation

```rust,ignore
use scirs2_special::symbolic::{Expr, Variable};

// Build symbolic expressions and differentiate
let x = Variable::new("x");
let expr = Expr::sin(x.clone()) * Expr::exp(x.clone());
let derivative = expr.differentiate(&x)?;
let value = derivative.evaluate(&[("x", 1.0)])?;
```

### Painleve Transcendents

```rust,ignore
use scirs2_special::painleve::{painleve_i, painleve_ii};

// Numerical solution of Painleve I: y'' = 6y^2 + t
let (t, y) = painleve_i(y0, yp0, t_span)?;
```

### Number-Theoretic Functions

```rust,ignore
use scirs2_special::{elliptic_modular, dedekind_zeta, selberg_zeta};

// Klein j-invariant (elliptic modular function)
let j = elliptic_modular::klein_j(tau)?;

// Dedekind zeta function for number fields
let z = dedekind_zeta::evaluate(s, discriminant)?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.special.gamma` | `gamma` |
| `scipy.special.gammaln` | `gammaln` |
| `scipy.special.beta` | `beta` |
| `scipy.special.betainc` | `betainc` |
| `scipy.special.j0` | `j0` |
| `scipy.special.jn` | `jn` |
| `scipy.special.erf` | `erf` |
| `scipy.special.erfc` | `erfc` |
| `scipy.special.ellipk` | `ellipk` |
| `scipy.special.ellipe` | `ellipe` |
| `scipy.special.hyp2f1` | `hyp2f1` |
