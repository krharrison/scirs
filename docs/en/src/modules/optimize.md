# Optimization (scirs2-optimize)

`scirs2-optimize` provides mathematical optimization algorithms modeled after `scipy.optimize`,
covering unconstrained and constrained minimization, global optimization, least squares,
root finding, linear/integer programming, and advanced methods like L-BFGS-B and QAOA.

## Unconstrained Minimization

### BFGS

The default choice for smooth, unconstrained problems:

```rust
use scirs2_optimize::unconstrained::{minimize, Method};
use scirs2_core::ndarray::ArrayView1;

// Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (1.0 - x0).powi(2) + 100.0 * (x1 - x0.powi(2)).powi(2)
}

fn bfgs_demo() -> Result<(), Box<dyn std::error::Error>> {
    let x0 = [0.0, 0.0];
    let result = minimize(rosenbrock, &x0, Method::BFGS, None)?;
    println!("Minimum at: {:?}", result.x);       // close to [1.0, 1.0]
    println!("Value: {:.6}", result.fun);           // close to 0.0
    println!("Converged: {}", result.success);
    Ok(())
}
```

### Other Methods

```rust,ignore
use scirs2_optimize::unconstrained::{minimize, Method};

// Conjugate Gradient
let result = minimize(f, &x0, Method::CG, None)?;

// Nelder-Mead (derivative-free)
let result = minimize(f, &x0, Method::NelderMead, None)?;

// Powell (derivative-free, direction-set method)
let result = minimize(f, &x0, Method::Powell, None)?;

// L-BFGS-B (bounded L-BFGS with box constraints)
let result = minimize(f, &x0, Method::LBFGSB, Some(&options))?;
```

## Optimization with Bounds

```rust,ignore
use scirs2_optimize::{Bounds, unconstrained::{minimize, Method, Options}};
use scirs2_core::ndarray::ArrayView1;

let f = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

let bounds = Bounds::new(vec![(-5.0, 5.0), (-5.0, 5.0)]);
let options = Options::default().with_bounds(bounds);
let result = minimize(f, &[3.0, 4.0], Method::LBFGSB, Some(&options))?;
```

## Global Optimization

### Differential Evolution

```rust,ignore
use scirs2_optimize::global::{differential_evolution, DEOptions};

let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
let options = DEOptions::default()
    .with_max_generations(1000)
    .with_population_size(50);

let result = differential_evolution(rosenbrock, &bounds, Some(options))?;
println!("Global minimum: {:?}", result.x);
```

### Basin-Hopping and Simulated Annealing

```rust,ignore
use scirs2_optimize::global::{basin_hopping, simulated_annealing};

// Basin-hopping: random perturbation + local minimization
let result = basin_hopping(f, &x0, 100, 1.0, None)?;

// Simulated annealing
let result = simulated_annealing(f, &x0, &bounds, None)?;
```

## Least Squares

```rust,ignore
use scirs2_optimize::least_squares::{least_squares, Method};

// Nonlinear least squares: minimize sum(residuals^2)
let result = least_squares(residuals, &x0, Method::LevenbergMarquardt, None)?;
```

## Root Finding

```rust,ignore
use scirs2_optimize::roots::{brentq, newton};

// Brent's method (bracketed, guaranteed convergence)
let root = brentq(|x| x.powi(3) - 2.0 * x - 5.0, 2.0, 3.0, None)?;

// Newton's method (requires derivative)
let root = newton(
    |x| x.powi(3) - 2.0 * x - 5.0,
    |x| 3.0 * x.powi(2) - 2.0,
    2.0,
    None
)?;
```

## Linear and Integer Programming

```rust,ignore
use scirs2_optimize::integer::{branch_and_bound, KnapsackSolver};

// 0-1 Knapsack problem
let solver = KnapsackSolver::new(values, weights, capacity);
let (total_value, selected) = solver.solve()?;

// Branch and bound for MIP
let result = branch_and_bound(&c, &a_ub, &b_ub, &integer_vars)?;
```

## Advanced Methods

### Distributed Optimization

```rust,ignore
use scirs2_optimize::distributed::{ADMM, PDMM, EXTRA};

// Consensus ADMM for distributed convex optimization
let admm = ADMM::new(local_objectives, rho)?;
let result = admm.solve(num_iterations)?;

// PDMM (per-edge dual variables)
let pdmm = PDMM::new(graph, local_objectives)?;
let result = pdmm.solve(num_iterations)?;
```

### Coordinate Descent

```rust,ignore
use scirs2_optimize::coordinate_descent::{CoordinateDescent, CDOptions};

let options = CDOptions::default().with_tolerance(1e-8);
let cd = CoordinateDescent::new(f, grad_i, n_vars, options);
let result = cd.solve(&x0)?;
```

### Differentiable Optimization (OptNet)

```rust,ignore
use scirs2_optimize::differentiable_optimization::{OptNet, QPLayer};

// Differentiable QP layer for neural network integration
let qp = QPLayer::new(q, p, a, b)?;
let (solution, backward_fn) = qp.forward(&params)?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.optimize.minimize(method='BFGS')` | `minimize(f, &x0, Method::BFGS, None)` |
| `scipy.optimize.minimize(method='L-BFGS-B')` | `minimize(f, &x0, Method::LBFGSB, opts)` |
| `scipy.optimize.minimize(method='Nelder-Mead')` | `minimize(f, &x0, Method::NelderMead, None)` |
| `scipy.optimize.differential_evolution` | `global::differential_evolution` |
| `scipy.optimize.least_squares` | `least_squares::least_squares` |
| `scipy.optimize.brentq` | `roots::brentq` |
| `scipy.optimize.newton` | `roots::newton` |
| `scipy.optimize.linprog` | `linprog` |
