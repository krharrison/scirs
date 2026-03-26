//! VEM basis function projections and geometric utilities
//!
//! In VEM, basis functions are **virtual**: they exist mathematically on each
//! element but are never explicitly computed. Only their polynomial projections
//! onto P_k(E) are needed for assembly.
//!
//! For degree k=1, we work with the following projections:
//!
//! * **Pi^∇_1** (energy projection): minimizes ‖∇(v - Pi^∇v)‖₀,E over P₁(E)
//!   subject to ∫_E (v - Pi^∇v) = 0 (zero-mean correction for well-posedness)
//!
//! * **Pi^0_1** (L²-projection): minimizes ‖v - Pi^0v‖₀,E over P₁(E)
//!
//! Both are computed purely from boundary DOF values (vertex values for k=1).

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array2;

/// Compute centroid and diameter of a polygon
///
/// Centroid via the shoelace formula for polygons.
/// Diameter = maximum distance between any two vertices.
///
/// # Arguments
///
/// * `vertices` - Polygon vertices in CCW order as `[x, y]` slices
pub fn polygon_centroid_and_diameter(vertices: &[[f64; 2]]) -> ([f64; 2], f64) {
    let n = vertices.len();
    debug_assert!(n >= 3, "Polygon must have at least 3 vertices");

    // Signed area via shoelace
    let mut area = 0.0_f64;
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
        area += cross;
        cx += (vertices[i][0] + vertices[j][0]) * cross;
        cy += (vertices[i][1] + vertices[j][1]) * cross;
    }
    area *= 0.5;

    // For degenerate polygons, fall back to vertex average
    if area.abs() < 1e-14 {
        let cx_avg = vertices.iter().map(|v| v[0]).sum::<f64>() / n as f64;
        let cy_avg = vertices.iter().map(|v| v[1]).sum::<f64>() / n as f64;
        let diam = max_vertex_distance(vertices);
        return ([cx_avg, cy_avg], diam.max(1e-14));
    }

    cx /= 6.0 * area;
    cy /= 6.0 * area;

    let diam = max_vertex_distance(vertices);

    ([cx, cy], diam.max(1e-14))
}

/// Maximum distance between any two vertices (diameter of polygon)
pub fn max_vertex_distance(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    let mut diam = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = vertices[j][0] - vertices[i][0];
            let dy = vertices[j][1] - vertices[i][1];
            let d = (dx * dx + dy * dy).sqrt();
            if d > diam {
                diam = d;
            }
        }
    }
    diam
}

/// Evaluate scaled monomials at a point (x, y) for degree k=1
///
/// The scaled monomials are:
///   m_1(x,y) = 1
///   m_2(x,y) = (x - x_E) / h_E
///   m_3(x,y) = (y - y_E) / h_E
///
/// where (x_E, y_E) is the element centroid and h_E is the element diameter.
pub fn scaled_monomials_values(x: f64, y: f64, centroid: [f64; 2], diameter: f64) -> [f64; 3] {
    [
        1.0,
        (x - centroid[0]) / diameter,
        (y - centroid[1]) / diameter,
    ]
}

/// Gradients of scaled monomials (for degree k=1)
///
/// Returns [grad_m1, grad_m2, grad_m3] where grad_mi = [dmi/dx, dmi/dy]
pub fn scaled_monomial_gradients(diameter: f64) -> [[f64; 2]; 3] {
    [
        [0.0, 0.0],            // grad(1) = 0
        [1.0 / diameter, 0.0], // grad((x-xE)/hE)
        [0.0, 1.0 / diameter], // grad((y-yE)/hE)
    ]
}

/// 2-point Gauss-Legendre quadrature on [0,1]
fn gauss_2pt_01() -> ([f64; 2], [f64; 2]) {
    let pts = [0.5 - 0.5 / 3.0_f64.sqrt(), 0.5 + 0.5 / 3.0_f64.sqrt()];
    let weights = [0.5, 0.5];
    (pts, weights)
}

/// Compute the Gram matrix G of scaled monomial gradients
///
/// G[α][β] = ∫_E ∇m_α · ∇m_β dx  for degree k=1
///
/// For degree 1 with monomials (1, (x-xE)/hE, (y-yE)/hE):
///   G[0][0] = 0 (gradient of constant is 0)
///   G[1][1] = |E| / hE^2
///   G[2][2] = |E| / hE^2
///   G[1][2] = G[2][1] = 0 (orthogonal gradients)
///
/// This is computed via boundary integration using Green's theorem.
fn gram_matrix_gradients(vertices: &[[f64; 2]], centroid: [f64; 2], diameter: f64) -> Array2<f64> {
    // For P1 monomials on a polygon, the Gram matrix of gradients can be computed analytically:
    // ∫_E ∇m_i · ∇m_j dx = area(E) * (grad_m_i · grad_m_j)
    // since the gradients of degree-1 monomials are constant.
    let area = polygon_area(vertices);
    let grads = scaled_monomial_gradients(diameter);
    let n_mono = 3;
    let mut g = Array2::<f64>::zeros((n_mono, n_mono));
    for i in 0..n_mono {
        for j in 0..n_mono {
            g[[i, j]] = area * (grads[i][0] * grads[j][0] + grads[i][1] * grads[j][1]);
        }
    }
    let _ = centroid; // used via diameter and area
    g
}

/// Compute polygon area using shoelace formula (absolute value)
pub fn polygon_area(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    let mut area = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[j][0] * vertices[i][1];
    }
    (area * 0.5).abs()
}

/// Compute the energy projection matrix Pi^∇ for degree k=1
///
/// Pi^∇ maps the n_dof virtual basis function values (at vertices) to
/// the coefficients of their P₁ projection.
///
/// For degree 1, Pi^∇ is a (3 × n_v) matrix where n_v is the number of element vertices.
///
/// ## Algorithm
///
/// The energy projection is determined by:
///   ∫_E ∇m_α · ∇(Pi^∇ φ_i) = ∫_E ∇m_α · ∇φ_i   for all m_α ∈ P₁(E), α = 2,...,n_mono
///
/// Since we don't know ∇φ_i explicitly, we use integration by parts:
///   ∫_E ∇m_α · ∇φ_i = -∫_E Δm_α φ_i + ∫_∂E (∇m_α · n) φ_i ds
///
/// For linear monomials, Δm_α = 0, so only boundary integral remains:
///   ∫_E ∇m_α · ∇φ_i = ∫_∂E (∇m_α · n) φ_i ds
///
/// The DOFs for k=1 are vertex values, so φ_i = 1 at vertex i, 0 at other vertices,
/// and piecewise linear on each edge.
///
/// The zero-mean condition: ∫_∂E φ_i ds / |∂E| (boundary average) fixes the constant term.
pub fn compute_pi_nabla(
    vertices: &[[f64; 2]],
    centroid: [f64; 2],
    diameter: f64,
) -> IntegrateResult<Array2<f64>> {
    let n_v = vertices.len();
    let n_mono = 3; // P1: 1, (x-xE)/hE, (y-yE)/hE

    if n_v < 3 {
        return Err(IntegrateError::InvalidInput(
            "Polygon must have at least 3 vertices".to_string(),
        ));
    }

    let grads = scaled_monomial_gradients(diameter);

    // Build the B matrix: B[α][i] = ∫_∂E (∇m_α · n) φ_i ds for α=1,2 (the non-constant monomials)
    // Note: for α=0 (constant), we use the mean condition separately.
    //
    // For each edge from vertex a to vertex b:
    //   outward normal = (dy, -dx) / |edge|  (CCW polygon -> outward normal)
    //   n = (v_b[1] - v_a[1], -(v_b[0] - v_a[0])) / edge_len  (scaled by edge_len already)
    //   So:  ∫_edge (∇m_α · n) φ_i ds = (∇m_α · n_edge) * ∫_edge φ_i ds
    //   where n_edge is the unnormalized normal (dy, -dx) (has magnitude = edge length)
    //   ∫_edge φ_i ds on edge (a,b):
    //     = edge_len/2  if i == a or i == b (hat function)
    //     = 0 otherwise

    // Matrix B: rows = monomials 1..2 (non-constant), cols = vertex DOFs
    // We'll build full 3×n_v and handle α=0 separately
    let mut b_mat = Array2::<f64>::zeros((n_mono, n_v));

    // Row α=0: handled by mean condition (see below)
    // Rows α=1,2: boundary integration
    for edge_start in 0..n_v {
        let edge_end = (edge_start + 1) % n_v;
        let va = vertices[edge_start];
        let vb = vertices[edge_end];
        // Unnormalized outward normal (magnitude = edge length for CCW polygon)
        let n_unnorm = [vb[1] - va[1], -(vb[0] - va[0])];

        for alpha in 1..n_mono {
            // ∇m_α · n_unnorm (n_unnorm already has magnitude = edge length)
            let grad_dot_n = grads[alpha][0] * n_unnorm[0] + grads[alpha][1] * n_unnorm[1];

            // φ_{edge_start} on this edge: integral = edge_len/2  (hat at va)
            // φ_{edge_end} on this edge: integral = edge_len/2  (hat at vb)
            // (edge_len/2 is the 1D integral of a hat function over [0,1] * edge_len)
            b_mat[[alpha, edge_start]] += grad_dot_n * 0.5;
            b_mat[[alpha, edge_end]] += grad_dot_n * 0.5;
        }
    }

    // For α=0 (constant): use the boundary average condition
    // Pi^∇ u = average of boundary DOF values times area/|∂E|...
    // Actually: the condition is that (1/|E|) ∫_E Pi^∇ φ_i = (1/|∂E|) ∫_∂E φ_i ds
    // Equivalently: the mean of Pi^∇ φ_i over boundary edges equals the mean vertex value
    // For k=1: ∫_∂E φ_i ds = (sum of edge-lengths adjacent to vertex i) / 2
    // But simpler: use the arithmetic mean of vertex values for the constant coefficient.
    // Row 0 of Pi^∇ is the normalized mean: (1/n_v) * [1, 1, ..., 1]
    for i in 0..n_v {
        b_mat[[0, i]] = 1.0 / (n_v as f64);
    }

    // Now compute G matrix (of gradients): G[α][β] = ∫_E ∇m_α · ∇m_β dx
    // G is 3×3; G[0][*] = G[*][0] = 0 (constant monomial has zero gradient)
    // G[1][1] = area/hE^2, G[2][2] = area/hE^2, G[1][2] = G[2][1] = 0
    let g_mat = gram_matrix_gradients(vertices, centroid, diameter);

    // Pi^∇ satisfies: G_{22} * Pi^∇_{2:,i} = B_{2:,i}   for rows α=1,2
    // (where G_{22} is the 2×2 lower-right block of G)
    // Row 0 is determined by mean condition.
    //
    // Since G[1][1] and G[2][2] are scalar (diagonal block), invert directly.
    let mut pi_nabla = Array2::<f64>::zeros((n_mono, n_v));

    // Row 0: mean condition (already in b_mat[0, :])
    for i in 0..n_v {
        pi_nabla[[0, i]] = b_mat[[0, i]];
    }

    // Rows 1, 2: solve G_{22} pi = B_{2:, i}
    // G is 2×2 block (rows/cols 1..2):
    let g11 = g_mat[[1, 1]];
    let g12 = g_mat[[1, 2]];
    let g21 = g_mat[[2, 1]];
    let g22 = g_mat[[2, 2]];
    let g_det = g11 * g22 - g12 * g21;

    if g_det.abs() < 1e-20 {
        return Err(IntegrateError::LinearSolveError(format!(
            "Degenerate Gram matrix in Pi_nabla computation, det={g_det}"
        )));
    }

    let g_inv_11 = g22 / g_det;
    let g_inv_12 = -g12 / g_det;
    let g_inv_21 = -g21 / g_det;
    let g_inv_22 = g11 / g_det;

    for i in 0..n_v {
        let b1 = b_mat[[1, i]];
        let b2 = b_mat[[2, i]];
        pi_nabla[[1, i]] = g_inv_11 * b1 + g_inv_12 * b2;
        pi_nabla[[2, i]] = g_inv_21 * b1 + g_inv_22 * b2;
    }

    Ok(pi_nabla)
}

/// Compute the L2 projection Pi^0 for degree k=1
///
/// Pi^0 : V_k(E) → P_k(E) satisfies:
///   ∫_E (Pi^0 v - v) m_α dx = 0  for all m_α ∈ P_k(E)
///
/// For degree 1 and vertex DOFs:
/// The L2 projection uses: `(M Pi^0)[α][i]` = ∫_E m_α φ_i dx
/// where `M[α][β]` = ∫_E m_α m_β dx
///
/// Since we can't integrate φ_i explicitly (they're virtual), we use the
/// consistency condition: Pi^0 on polynomials is exact, and the boundary
/// integral formula gives ∫_E m_α φ_i dx.
///
/// For simplicity and for degree 1, we approximate Pi^0 using the Pi^∇ result
/// and the known boundary values (vertex values).
///
/// Returns Pi^0 as a (3 × n_v) matrix mapping vertex DOFs to monomial coefficients.
pub fn compute_pi0(
    vertices: &[[f64; 2]],
    centroid: [f64; 2],
    diameter: f64,
    pi_nabla: &Array2<f64>,
) -> Array2<f64> {
    let n_v = vertices.len();
    let n_mono = 3;

    // For degree 1 VEM: Pi^0 ≈ Pi^∇ is the standard choice in the literature
    // (the two projections agree for k=1 when computing stiffness matrices)
    // A more refined approach uses the mass matrix, but for k=1 this suffices.
    let _ = (centroid, diameter); // used through pi_nabla
    let mut pi0 = Array2::<f64>::zeros((n_mono, n_v));
    for i in 0..n_mono {
        for j in 0..n_v {
            pi0[[i, j]] = pi_nabla[[i, j]];
        }
    }
    pi0
}

/// Evaluate Pi^∇ v at a point (x, y) given vertex DOF values
///
/// The projected polynomial is: Pi^∇ v (x,y) = sum_α c_α m_α(x,y)
/// where c_α = sum_i (Pi^∇)[α, i] * v_i
pub fn eval_pi_nabla_at(
    x: f64,
    y: f64,
    centroid: [f64; 2],
    diameter: f64,
    pi_nabla: &Array2<f64>,
    vertex_values: &[f64],
) -> f64 {
    let mono_vals = scaled_monomials_values(x, y, centroid, diameter);
    let n_mono = pi_nabla.nrows();
    let n_v = vertex_values.len();

    let mut result = 0.0_f64;
    for alpha in 0..n_mono {
        let mut c_alpha = 0.0_f64;
        for i in 0..n_v {
            c_alpha += pi_nabla[[alpha, i]] * vertex_values[i];
        }
        result += c_alpha * mono_vals[alpha];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_centroid_square() {
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (centroid, diameter) = polygon_centroid_and_diameter(&verts);
        assert!((centroid[0] - 0.5).abs() < 1e-12);
        assert!((centroid[1] - 0.5).abs() < 1e-12);
        assert!((diameter - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_polygon_centroid_triangle() {
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let (centroid, _diameter) = polygon_centroid_and_diameter(&verts);
        // Centroid of triangle = average of vertices
        assert!((centroid[0] - (0.0 + 1.0 + 0.5) / 3.0).abs() < 1e-12);
        assert!((centroid[1] - (0.0 + 0.0 + 1.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_polygon_diameter_square() {
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (_, diameter) = polygon_centroid_and_diameter(&verts);
        // Diameter of unit square = sqrt(2)
        assert!(
            (diameter - 2.0_f64.sqrt()).abs() < 1e-12,
            "diameter={diameter}"
        );
    }

    #[test]
    fn test_pi_nabla_reproduces_linear_square() {
        // For a unit square element with linear DOF values u_i = x_i + y_i,
        // Pi^∇ should reproduce u exactly since u ∈ P₁
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (centroid, diameter) = polygon_centroid_and_diameter(&verts);
        let pi_nabla = compute_pi_nabla(&verts, centroid, diameter).unwrap();

        // Vertex values of u(x,y) = x + y
        let vertex_values: Vec<f64> = verts.iter().map(|v| v[0] + v[1]).collect();

        // Check at interior point
        let test_x = 0.5_f64;
        let test_y = 0.3_f64;
        let u_proj = eval_pi_nabla_at(
            test_x,
            test_y,
            centroid,
            diameter,
            &pi_nabla,
            &vertex_values,
        );
        let u_exact = test_x + test_y;

        assert!(
            (u_proj - u_exact).abs() < 1e-10,
            "Pi^∇ should reproduce u=x+y exactly, got {u_proj} vs {u_exact}"
        );
    }

    #[test]
    fn test_pi_nabla_reproduces_constant() {
        // Pi^∇ should reproduce constant functions
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let (centroid, diameter) = polygon_centroid_and_diameter(&verts);
        let pi_nabla = compute_pi_nabla(&verts, centroid, diameter).unwrap();

        let c = 3.7_f64;
        let vertex_values = vec![c, c, c];

        let u_proj = eval_pi_nabla_at(0.3, 0.2, centroid, diameter, &pi_nabla, &vertex_values);
        assert!(
            (u_proj - c).abs() < 1e-10,
            "Pi^∇ should reproduce const {c}, got {u_proj}"
        );
    }

    #[test]
    fn test_polygon_area_unit_square() {
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let area = polygon_area(&verts);
        assert!((area - 1.0).abs() < 1e-12, "area={area}");
    }

    #[test]
    fn test_polygon_area_triangle() {
        let verts = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let area = polygon_area(&verts);
        assert!((area - 0.5).abs() < 1e-12, "area={area}");
    }
}
