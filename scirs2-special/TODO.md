# scirs2-special TODO

## v0.3.1 Completed

### Classical Special Functions
- [x] Gamma: `gamma`, `log_gamma`, `digamma`, `trigamma`, `polygamma`, `beta`, `log_beta`
- [x] Incomplete gamma: lower `gamma(a,x)`, upper `Gamma(a,x)`, regularized P and Q
- [x] Incomplete beta `I_x(a,b)` and its inverse; `beta` function
- [x] Factorial `n!`, log-factorial, binomial `C(n,k)`, Pochhammer symbol
- [x] Error function `erf`, complementary `erfc`, scaled `erfcx`, imaginary `erfi`
- [x] Dawson integral, inverse `erfinv`, inverse complementary `erfcinv`
- [x] Bessel J_n (integer and real order), Y_n, I_n, K_n; spherical j_n, y_n; Hankel H_n^(1/2)
- [x] Bessel function zeros (first n zeros of J_n, Y_n)
- [x] Complete elliptic K(k), E(k), Pi(n,k); incomplete F, E, Pi; Carlson R_F/R_D/R_J/R_C
- [x] Jacobi elliptic functions sn, cn, dn (12 variants)
- [x] Orthogonal polynomials: Legendre P_n, associated P_n^m; Chebyshev T_n, U_n; Hermite H_n, He_n; Laguerre L_n, L_n^alpha; Gegenbauer C_n^lambda; Jacobi P_n^(alpha,beta); Zernike radial
- [x] Airy Ai, Bi and derivatives; exponentially scaled; complex argument
- [x] Hypergeometric: _0F_1, _1F_1 (Kummer), U (Tricomi), _2F_1 (Gauss) with analytic continuation; generalized _pF_q
- [x] Riemann zeta, Hurwitz zeta, Dirichlet eta, Lerch transcendent, Lambert W (W_0 and W_{-1})
- [x] Struve H_n and L_n with asymptotic expansions
- [x] Kelvin functions ber, bei, ker, kei and derivatives
- [x] Fresnel integrals S(x) and C(x), modulus and phase
- [x] Parabolic cylinder D_n, U(a,x), V(a,x)
- [x] Spheroidal wave functions: prolate and oblate, angular and radial
- [x] Wright omega and Wright Bessel functions
- [x] Coulomb wave functions: regular F_l, irregular G_l, Hankel H_l^+/-
- [x] Logarithmic integral li(x), offset Li(x), exponential integrals Ei, E_n, E_1

### Advanced Functions (v0.3.1 Additions)
- [x] Mathieu functions: characteristic values a_r(q), b_r(q); even ce_r, odd se_r with Fourier coefficients; radial Mc_r, Ms_r; asymptotic expansions
- [x] Real and complex spherical harmonics Y_l^m for arbitrary l, m
- [x] Gaunt coefficients: triple-Y integrals
- [x] Wigner 3-j symbols (Racah formula)
- [x] Wigner 6-j symbols (Racah W-coefficients)
- [x] Wigner 9-j symbols for compound coupling
- [x] Clebsch-Gordan coefficients
- [x] Jacobi theta functions theta_1 through theta_4; logarithmic derivatives
- [x] Weierstrass P-function, zeta, sigma; elliptic invariants g2, g3; discriminant; j-invariant
- [x] Parabolic cylinder extensions: non-integer n via Whittaker; asymptotic expansions for large |x|, |a|
- [x] Fox H-function: general H_{p,q}^{m,n}; series and integral representations
- [x] Appell F_1, F_2, F_3, F_4 hypergeometric functions
- [x] Meixner-Pollaczek polynomials P_n^lambda(x; phi)
- [x] Heun functions: general, confluent, double-confluent, biconfluent, triconfluent
- [x] Polylogarithm Li_s(z) for complex s, z; Fermi-Dirac integrals; Bose-Einstein integrals; Clausen Cl_2
- [x] Q-Gamma function Gamma_q; q-Pochhammer (a;q)_n and (a;q)_inf; q-binomial (Gaussian binomial); q-exponential e_q, E_q
- [x] Q-Bessel functions of first and second kind
- [x] Q-orthogonal polynomials: big/little q-Jacobi, q-Laguerre, q-Hermite, Askey-Wilson
- [x] Number theory: Ramanujan tau, Euler totient phi, Jordan totient, Liouville lambda, von Mangoldt Lambda, Mobius mu, Mertens M, d(n), sigma_k(n), partition function p(n)
- [x] Bell polynomials (complete and partial), Bernoulli/Euler numbers and polynomials
- [x] Stirling numbers first and second kind; Lah numbers
- [x] Information-theoretic: KL divergence, JS divergence, Shannon entropy, Renyi entropy, mutual information, cross-entropy, logistic, softmax, logsumexp
- [x] Combinatorics extensions: Catalan, Narayana, Motzkin numbers; derangements; subfactorial
- [x] Orthogonal polynomial extensions: Wilson, Racah, Askey-Wilson, dual Hahn, Krawtchouk, Meixner, Charlier

### Performance
- [x] SIMD-accelerated array evaluation for gamma, erf, Bessel (via scirs2-core)
- [x] Parallel Rayon-based batch evaluation for arrays > 1000 elements
- [x] Lookup tables and rational approximations for critical hot paths
- [x] Chunked processing for memory-efficient large array evaluation

## v0.4.0 Roadmap

### GPU-Accelerated Batch Evaluation
- [ ] CUDA/ROCm kernels for batch gamma, erf, Bessel evaluation on GPU
- [ ] WebGPU compute shaders for browser-based WASM deployment
- [ ] Auto-dispatch: evaluate on GPU when array size exceeds configurable threshold
- [ ] Mixed-precision: f16 accumulation with f32 correction for throughput-critical paths

### Symbolic Computation Interface
- [ ] Symbolic representation of special functions as expression trees
- [ ] Automatic differentiation of special functions: symbolic derivative rules
- [ ] Series expansion engine: formal power series around regular and irregular points
- [ ] Asymptotic expansion engine: automated derivation of leading-order terms
- [ ] Connection formula generator: transformations between solution bases

### Extended Precision
- [ ] Arbitrary-precision gamma, erf, Bessel via the `rug` MPFR backend (feature-gated)
- [ ] Ball arithmetic for certified enclosure of function values
- [ ] Validated numerics interface: output intervals guaranteed to contain the true value
- [ ] Double-double (quad-double) precision for 30-60 decimal digits without MPFR overhead

### New Function Families
- [ ] Lame functions: solutions to Lame's equation on an ellipsoidal coordinate system
- [ ] Spheroidal wave functions with full asymptotic transitions
- [ ] Nield-Kuznetsov functions for gravity wave theory
- [ ] Mathieu-Hill functions: generalized periodic Hill's equation solutions
- [ ] Painleve transcendents: numerical solution with connection formulas
- [ ] Elliptic modular functions: j-invariant, Dedekind eta, modular lambda

### Number Theory Extensions
- [ ] L-functions: Dirichlet L(s, chi) for primitive characters
- [ ] Hecke L-functions and Maass forms
- [ ] Elliptic curve L-functions (BSD conjecture numerics)
- [ ] Dedekind zeta functions for number fields
- [ ] Selberg zeta function for hyperbolic surfaces

### Combinatorics and Algebra
- [ ] Chromatic polynomial of graphs
- [ ] Tutte polynomial of matroids
- [ ] Schur polynomials and symmetric function bases (power-sum, monomial, elementary)
- [ ] Clebsch-Gordan series for arbitrary Lie groups (SU(3), SO(5), etc.)
- [ ] Hall polynomials for p-group extensions

## Known Issues

- Appell F_2 convergence is slow near the boundary of its natural domain (|x| + |y| = 1); extrapolation via analytic continuation is planned.
- Heun functions (general) use local power series and may fail to converge for large |z| or near Stokes lines; connection formula-based global evaluation is planned.
- Fox H-function series representation is conditional on absolute convergence; the integral representation needed for the divergent-series regime is not yet implemented.
- Q-Bessel functions for |q| close to 1 may exhibit numerical instability due to cancellation in the q-Pochhammer product; regularized representations are planned.
- Wigner 9-j symbols for j > 30 may accumulate rounding errors; arbitrary-precision evaluation via the `rug` feature is recommended for high-j coupling.
- Ramanujan tau function is computed via convolution of Fourier coefficients and is O(n log n); values up to n ~ 10^6 are practical on current hardware.
