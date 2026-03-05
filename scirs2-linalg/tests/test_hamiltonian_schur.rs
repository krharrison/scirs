use scirs2_core::ndarray::{array, Array2};

#[test]
fn test_lyapunov_for_newton_step() {
    // A_0 = [[0,1],[-0.5,-1]], A_0^T = [[0,-0.5],[1,-1]]
    // RHS = [[1.25,0.5],[0.5,2.0]]
    // Solve A_0^T X + X A_0 + RHS = 0
    let a0_t = array![[0.0_f64, -0.5], [1.0, -1.0]];
    let rhs = array![[1.25_f64, 0.5], [0.5, 2.0]];

    let x = scirs2_linalg::matrix_functions::sylvester::solve_continuous_lyapunov(
        &a0_t.view(),
        &rhs.view(),
    );
    match &x {
        Ok(x_val) => {
            eprintln!(
                "X = [[{:.6}, {:.6}], [{:.6}, {:.6}]]",
                x_val[[0, 0]],
                x_val[[0, 1]],
                x_val[[1, 0]],
                x_val[[1, 1]]
            );
            // Verify: A^T X + X A^T^T + RHS = 0 => A_0^T X + X A_0 + RHS = 0
            let resid = a0_t.dot(x_val) + x_val.dot(&a0_t.t()) + &rhs;
            let max_err: f64 = resid.iter().map(|v: &f64| v.abs()).fold(0.0_f64, f64::max);
            eprintln!("Residual: {max_err:.2e}");
            eprintln!("X[0,0]={}, X[1,1]={}", x_val[[0, 0]], x_val[[1, 1]]);
            assert!(max_err < 1e-6, "Lyapunov residual: {max_err}");
            // X should be PD
            assert!(
                x_val[[0, 0]] > 0.0,
                "X[0,0] should be > 0: {}",
                x_val[[0, 0]]
            );
        }
        Err(e) => panic!("Lyapunov failed: {e}"),
    }
}

#[test]
fn test_care_manual_newton() {
    use scirs2_linalg::matrix_functions::sylvester::solve_continuous_lyapunov;

    let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
    let s = array![[0.0_f64, 0.0], [0.0, 1.0]]; // B R^{-1} B^T
    let q = array![[1.0_f64, 0.0], [0.0, 1.0]];

    // Initial X_0 = [[1, 0.5], [0.5, 1]]
    let mut x = array![[1.0_f64, 0.5], [0.5, 1.0]];

    for iter in 0..20 {
        // A_k = A - S * X_k
        let a_k = &a - &s.dot(&x);
        eprintln!(
            "iter {iter}: A_k = [[{:.4},{:.4}],[{:.4},{:.4}]]",
            a_k[[0, 0]],
            a_k[[0, 1]],
            a_k[[1, 0]],
            a_k[[1, 1]]
        );

        // RHS = Q + X_k S X_k
        let rhs = &q + &x.dot(&s).dot(&x);

        // Solve A_k^T X_{k+1} + X_{k+1} A_k + RHS = 0
        let a_k_t = a_k.t().to_owned();
        match solve_continuous_lyapunov(&a_k_t.view(), &rhs.view()) {
            Ok(x_new) => {
                let diff: f64 = (&x_new - &x).iter().map(|v| v * v).sum::<f64>().sqrt();
                eprintln!(
                    "  X = [[{:.6},{:.6}],[{:.6},{:.6}]], diff={:.2e}",
                    x_new[[0, 0]],
                    x_new[[0, 1]],
                    x_new[[1, 0]],
                    x_new[[1, 1]],
                    diff
                );

                // Verify the CARE residual
                let x_ref: &Array2<f64> = &x_new;
                let care_res = a.t().dot(x_ref) + x_ref.dot(&a) - x_ref.dot(&s).dot(x_ref) + &q;
                let care_err: f64 = care_res
                    .iter()
                    .map(|v: &f64| v.abs())
                    .fold(0.0_f64, f64::max);
                eprintln!("  CARE residual: {care_err:.2e}");

                x = x_new;
                if diff < 1e-12 {
                    break;
                }
            }
            Err(e) => {
                eprintln!("  Lyapunov failed: {e}");
                break;
            }
        }
    }

    eprintln!(
        "Final X = [[{:.6},{:.6}],[{:.6},{:.6}]]",
        x[[0, 0]],
        x[[0, 1]],
        x[[1, 0]],
        x[[1, 1]]
    );
    assert!(x[[0, 0]] > 0.0 && x[[1, 1]] > 0.0, "Solution should be PD");
}
