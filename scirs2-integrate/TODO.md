# scirs2-integrate TODO

## v0.3.1 Completed

### Quadrature
- [x] Adaptive 1D quadrature: `quad`, `dblquad`, `tplquad`, `nquad`
- [x] Gaussian quadrature: Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre, Gauss-Chebyshev
- [x] Romberg integration with Richardson extrapolation
- [x] Tanh-sinh quadrature for endpoint singularities
- [x] Lebedev spherical quadrature rules
- [x] Newton-Cotes coefficient generation
- [x] `quad_vec`: vectorized quadrature for array-valued integrands
- [x] Adaptive cubature for multidimensional integrals

### Monte Carlo and Quasi-Monte Carlo
- [x] Standard Monte Carlo with importance sampling, stratified sampling
- [x] Quasi-Monte Carlo: Sobol sequences, Halton sequences, lattice rules
- [x] `qmc_quad`: high-dimensional integration with low-discrepancy sequences
- [x] Parallel Monte Carlo with work-stealing task distribution

### ODE Solvers (IVP)
- [x] Euler, RK4 (fixed step), RK23 (Bogacki-Shampine), RK45 (Dormand-Prince)
- [x] DOP853: Dormand-Prince 8(5,3) high-precision adaptive
- [x] BDF (orders 1-5): enhanced Jacobian reuse and Broyden updates
- [x] Radau IIA: L-stable implicit Runge-Kutta
- [x] LSODA: automatic stiffness detection and method switching
- [x] `solve_ivp`: unified interface for all methods
- [x] Event detection: zero-crossing, terminal events, dense output
- [x] Mass matrix support: constant, time-dependent, state-dependent
- [x] IMEX splitting for additive stiff + non-stiff systems

### Boundary Value Problems
- [x] Collocation `solve_bvp` with adaptive mesh refinement
- [x] Single shooting method
- [x] Multiple shooting method
- [x] Arc-length continuation for parameter-dependent BVPs

### Differential-Algebraic Equations (DAE)
- [x] BDF-based index-1 DAE solver
- [x] Pantelides algorithm for automatic index reduction
- [x] Block preconditioners for large DAE systems

### Partial Differential Equations (PDE)
- [x] Finite Difference: 1D/2D/3D central, upwind, WENO schemes
- [x] Finite Element: linear/quadratic triangular elements
- [x] Spectral methods: Fourier, Chebyshev, Legendre, spectral element
- [x] Finite Volume: upwind flux, Godunov, Roe
- [x] Time-stepping FEM (space-time Galerkin)
- [x] Adaptive mesh refinement and coarsening

### Stochastic Differential Equations (SDE)
- [x] Euler-Maruyama: first-order SDE solver
- [x] Milstein scheme: strong order 1.0 SDE solver
- [x] Strong order 1.5 iterated stochastic integral methods
- [x] Multi-dimensional SDEs with correlated noise
- [x] Stochastic PDE (SPDE) solvers: space-time white and colored noise

### Lattice Boltzmann Method (LBM)
- [x] D2Q9 and D3Q19 lattice geometries
- [x] BGK single-relaxation-time collision operator
- [x] MRT multi-relaxation-time collision operator
- [x] Bounce-back, Zou-He, and periodic boundary conditions
- [x] Smagorinsky subgrid-scale turbulence model
- [x] Shan-Chen multiphase interaction

### Discontinuous Galerkin (DG)
- [x] Modal DG on reference elements
- [x] Nodal DG interpolation-based formulation
- [x] Numerical fluxes: upwind, Lax-Friedrichs, Roe, HLLC
- [x] hp-adaptivity: simultaneous mesh and degree refinement

### Phase Field Models
- [x] Cahn-Hilliard equation with semi-implicit time stepping
- [x] Allen-Cahn interface dynamics
- [x] Phase field crystal periodic density model
- [x] Chemo-mechanical coupling for electrode models

### Boundary Element Method (BEM)
- [x] Laplace BEM for potential flow and heat conduction
- [x] Helmholtz BEM for acoustic scattering
- [x] Fast multipole BEM O(N log N)
- [x] Galerkin and collocation formulations

### Isogeometric Analysis (IGA)
- [x] B-spline and NURBS basis functions
- [x] k-refinement for NURBS patches
- [x] Structural IGA: shells, beams, solid mechanics

### Port-Hamiltonian Discretization
- [x] Discrete Dirac structures on staggered grids
- [x] Energy-routing interconnection between subsystems
- [x] Passivity-guaranteed dissipation bounds

### Symplectic Integrators
- [x] Stormer-Verlet (2nd order)
- [x] Ruth 4th-order symplectic RK
- [x] Leapfrog / velocity Verlet
- [x] Gauss-Legendre implicit symplectic collocation

### Specialized Domain Solvers
- [x] Schrödinger equation: split-operator and Crank-Nicolson
- [x] Navier-Stokes: projection method for incompressible flow
- [x] Financial PDEs: Black-Scholes, Heston, Monte Carlo exotic derivatives
- [x] Integral equations: Fredholm and Volterra 1st and 2nd kind

### Performance Infrastructure
- [x] Anderson acceleration for iterative solvers
- [x] Hardware auto-tuning (CPU core/cache detection)
- [x] Work-stealing parallel scheduler
- [x] SIMD-accelerated vector operations (feature-gated)
- [x] Memory pool and cache-friendly matrix layouts

## v0.4.0 Roadmap

### GPU Acceleration
- [ ] GPU-accelerated LBM: target millions of cells at interactive frame rates
- [ ] GPU ODE ensemble integration: batched RK45 across thousands of parameter sets
- [ ] GPU FEM assembly: shared-memory atomic scatter for sparse stiffness matrix
- [ ] CUDA graph capture for repeated ODE solve patterns (neural ODE training)

### Adaptive Mesh Refinement
- [ ] Full AMR framework: quad-tree / oct-tree dynamic refinement
- [ ] Conservative prolongation and restriction operators
- [ ] Load-balanced AMR for parallel distributed grids
- [ ] Interface tracking with level-set AMR

### Quantum Chemistry and Physics
- [ ] Hartree-Fock and DFT integrals over Gaussian basis sets
- [ ] Density matrix evolution (Lindblad master equation)
- [ ] Time-dependent Hartree-Fock (TDHF)
- [ ] Path integral Monte Carlo for quantum statistical mechanics

### Advanced SDE and SPDE
- [ ] Weak order 2.0 SDE schemes (Platen-Wagner)
- [ ] Rough SDE driven by fractional Brownian motion
- [ ] Galerkin SPDE solvers with polynomial chaos expansion
- [ ] Real-time particle filter for state estimation

### PDE Solvers
- [ ] Hybridizable DG (HDG) for diffusion-dominated problems
- [ ] Virtual Element Method (VEM) for polygonal meshes
- [ ] Peridynamics for fracture mechanics
- [ ] Free boundary / Stefan problem solvers

### Integration and Quadrature
- [ ] Filon quadrature for highly oscillatory integrands
- [ ] Sparse grid quadrature for high-dimensional smooth functions
- [ ] Clenshaw-Curtis adaptive quadrature with contour deformation

## Known Issues

- BDF order-5 may exhibit slow convergence near singular Jacobians; automatic order reduction to 3 recommended as workaround.
- LBM Shan-Chen multiphase currently only supports D2Q9; D3Q19 multiphase is planned for v0.4.0.
- IGA structural solver does not yet implement trimmed NURBS or multi-patch coupling.
- Port-Hamiltonian BEM coupling is implemented but not yet verified against the BEM module's matrix assembly.
- SPDE colored noise requires manual specification of the correlation kernel; automatic estimation is planned.
