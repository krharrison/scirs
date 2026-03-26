# Integration (scirs2-integrate)

`scirs2-integrate` provides numerical integration methods and differential equation solvers
modeled after `scipy.integrate`, with extensions for PDEs, DAEs, symplectic systems,
and specialized domain solvers.

## Quadrature (1D Integration)

### Adaptive Quadrature

```rust
use scirs2_integrate::quad::quad;

fn quad_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Integrate x^2 from 0 to 1 (exact: 1/3)
    let result = quad(|x: f64| x * x, 0.0, 1.0, None)?;
    assert!((result.value - 1.0 / 3.0).abs() < 1e-8);
    println!("Integral: {}, Error estimate: {}", result.value, result.error);
    Ok(())
}
```

### Gaussian Quadrature

```rust
use scirs2_integrate::gaussian::gauss_legendre;

fn gauss_demo() -> Result<(), Box<dyn std::error::Error>> {
    // 5-point Gauss-Legendre quadrature
    let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5)?;
    assert!((result - 1.0 / 3.0).abs() < 1e-10);
    Ok(())
}
```

### Romberg Integration

```rust,ignore
use scirs2_integrate::romberg::romberg;

let result = romberg(|x: f64| x.sin(), 0.0, std::f64::consts::PI, None)?;
// result ~ 2.0
```

### Monte Carlo Integration

For high-dimensional integrals where grid-based methods are impractical:

```rust,ignore
use scirs2_integrate::monte_carlo::monte_carlo;

// Integrate over a 10-dimensional unit cube
let result = monte_carlo(
    |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>(),
    &[(0.0, 1.0); 10],
    100_000,  // number of samples
    None,
)?;
```

## ODE Solvers

### Initial Value Problems

```rust,ignore
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

// dy/dt = -2y, y(0) = 1 (exact solution: e^{-2t})
fn rhs(t: f64, y: &[f64]) -> Vec<f64> {
    vec![-2.0 * y[0]]
}

let options = ODEOptions::default()
    .with_method(ODEMethod::RK45)
    .with_rtol(1e-8)
    .with_atol(1e-10);

let result = solve_ivp(rhs, (0.0, 5.0), &[1.0], &options)?;
println!("t: {:?}", result.t);
println!("y: {:?}", result.y);
```

### Stiff Systems

For stiff ODEs, use implicit methods:

```rust,ignore
use scirs2_integrate::ode::{solve_ivp, ODEMethod};

// BDF (Backward Differentiation Formula) for stiff systems
let options = ODEOptions::default().with_method(ODEMethod::BDF);
let result = solve_ivp(stiff_rhs, t_span, &y0, &options)?;

// Radau (implicit Runge-Kutta, L-stable)
let options = ODEOptions::default().with_method(ODEMethod::Radau);
```

### Symplectic Integrators

For Hamiltonian systems that need energy conservation:

```rust,ignore
use scirs2_integrate::symplectic::{leapfrog, yoshida4};

// Leapfrog / Stormer-Verlet (2nd order, symplectic)
let result = leapfrog(force_fn, &q0, &p0, dt, num_steps)?;

// Yoshida 4th-order symplectic integrator
let result = yoshida4(force_fn, &q0, &p0, dt, num_steps)?;
```

## PDE Solvers

### Finite Difference

```rust,ignore
use scirs2_integrate::pde::{heat_equation_1d, wave_equation_1d};

// 1D heat equation: u_t = alpha * u_xx
let solution = heat_equation_1d(alpha, &initial_condition, &boundary, dx, dt, t_final)?;
```

### Discontinuous Galerkin

```rust,ignore
use scirs2_integrate::pde::dg::{DGSolver, DGOptions};

let options = DGOptions::default()
    .with_polynomial_order(3)
    .with_num_elements(100);

let solver = DGSolver::new(flux_fn, &options)?;
let solution = solver.solve(&initial_condition, t_final)?;
```

## Specialized Solvers

### Quantum Mechanics

```rust,ignore
use scirs2_integrate::specialized::quantum::{
    schrodinger_1d, lindblad_master_equation
};

// Time-independent Schrodinger equation
let (energies, wavefunctions) = schrodinger_1d(potential, x_range, num_states)?;

// Lindblad master equation for open quantum systems
let rho_t = lindblad_master_equation(&hamiltonian, &collapse_ops, &rho0, t_span)?;
```

### Stochastic Differential Equations

```rust,ignore
use scirs2_integrate::sde::{euler_maruyama, milstein};

// Euler-Maruyama for dX = a(X)dt + b(X)dW
let paths = euler_maruyama(drift, diffusion, x0, dt, num_steps, num_paths)?;

// Milstein (higher-order, requires diffusion derivative)
let paths = milstein(drift, diffusion, diffusion_deriv, x0, dt, num_steps, num_paths)?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.integrate.quad` | `quad::quad` |
| `scipy.integrate.romberg` | `romberg::romberg` |
| `scipy.integrate.solve_ivp` | `ode::solve_ivp` |
| `scipy.integrate.odeint` | `ode::odeint` |
| `scipy.integrate.solve_bvp` | `bvp::solve_bvp` |
