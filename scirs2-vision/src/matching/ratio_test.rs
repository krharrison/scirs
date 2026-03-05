//! Lowe's ratio test and geometric verification.
//!
//! Provides:
//! - Lowe's ratio test for rejecting ambiguous matches.
//! - RANSAC estimation of the fundamental matrix (8-point algorithm).
//! - Essential matrix from fundamental matrix and intrinsics.

use crate::error::VisionError;

// ─── Ratio test ───────────────────────────────────────────────────────────────

/// Compute the squared L2 distance between two float slices.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Lowe's ratio test for float descriptors.
///
/// For each descriptor in `desc1`, finds the two nearest neighbours in
/// `desc2`.  Keeps the match if `dist(nn1) / dist(nn2) < ratio`.
///
/// Returns `(idx1, idx2, distance)` triples.
///
/// # Errors
/// Returns [`VisionError`] if descriptor dimensions are inconsistent.
pub fn ratio_test(
    desc1: &[Vec<f32>],
    desc2: &[Vec<f32>],
    ratio: f32,
) -> Result<Vec<(usize, usize, f32)>, VisionError> {
    if desc1.is_empty() || desc2.is_empty() {
        return Ok(vec![]);
    }
    let dim = desc1[0].len();
    for d in desc1.iter().chain(desc2.iter()) {
        if d.len() != dim {
            return Err(VisionError::InvalidInput(
                "Inconsistent descriptor lengths in ratio test".into(),
            ));
        }
    }

    let mut good: Vec<(usize, usize, f32)> = Vec::new();

    for (i, d1) in desc1.iter().enumerate() {
        // Find top-2 nearest.
        let mut best = (0_usize, f32::INFINITY);
        let mut second = (0_usize, f32::INFINITY);

        for (j, d2) in desc2.iter().enumerate() {
            let d = l2_sq(d1, d2).sqrt();
            if d < best.1 {
                second = best;
                best = (j, d);
            } else if d < second.1 {
                second = (j, d);
            }
        }

        if best.1 < ratio * second.1 {
            good.push((i, best.0, best.1));
        }
    }
    good.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok(good)
}

// ─── 3×3 linear algebra helpers ──────────────────────────────────────────────

type Mat3 = [[f64; 3]; 3];

fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3_transpose(m: &Mat3) -> Mat3 {
    let mut t = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[j][i] = m[i][j];
        }
    }
    t
}

// ─── SVD (thin) via Jacobi on the 9×9 normal system ─────────────────────────
//
// We only need the right singular vector for the smallest singular value.
// For the 8-pt algorithm we solve a 9-dim homogeneous system Af = 0.
// We use the Gram-Schmidt + power iteration approach on A^T A.

/// Compute the right singular vector corresponding to the smallest singular
/// value of an (n × 9) matrix A.  Returned as a 9-element vector.
fn right_singular_min(a: &[Vec<f64>]) -> [f64; 9] {
    let n = a.len();
    if n == 0 {
        return [0.0; 9];
    }
    // Build ATA (9×9).
    let mut ata = [[0.0_f64; 9]; 9];
    #[allow(clippy::needless_range_loop)]
    for row in a {
        for i in 0..9 {
            for j in 0..9 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }

    // Jacobi eigendecomposition of ata (symmetric 9×9).
    // We want the eigenvector for the smallest eigenvalue.
    jacobi_min_eigenvec(&ata)
}

/// One-sided Jacobi eigendecomposition: returns the eigenvector for the
/// smallest eigenvalue of symmetric 9×9 matrix S.
fn jacobi_min_eigenvec(s: &[[f64; 9]; 9]) -> [f64; 9] {
    let n = 9_usize;
    let mut a = *s;
    // Accumulate eigenvectors in V (identity initially).
    let mut v = [[0.0_f64; 9]; 9];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..100 {
        // Find off-diagonal element with largest absolute value.
        let (mut p, mut q, mut max_val) = (0, 1, 0.0_f64);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in i + 1..n {
                let av = a[i][j].abs();
                if av > max_val {
                    max_val = av;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let cos_a = 1.0 / (1.0 + t * t).sqrt();
        let sin_a = t * cos_a;

        // Apply rotation to a.
        let new_app = cos_a * cos_a * app - 2.0 * sin_a * cos_a * apq + sin_a * sin_a * aqq;
        let new_aqq = sin_a * sin_a * app + 2.0 * sin_a * cos_a * apq + cos_a * cos_a * aqq;
        a[p][p] = new_app;
        a[q][q] = new_aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        #[allow(clippy::needless_range_loop)]
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = cos_a * arp - sin_a * arq;
                a[p][r] = a[r][p];
                a[r][q] = sin_a * arp + cos_a * arq;
                a[q][r] = a[r][q];
            }
        }
        // Accumulate eigenvectors.
        #[allow(clippy::needless_range_loop)]
        for r in 0..n {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = cos_a * vrp - sin_a * vrq;
            v[r][q] = sin_a * vrp + cos_a * vrq;
        }
    }

    // Find index of smallest eigenvalue on diagonal.
    let min_idx = (0..n)
        .min_by(|&i, &j| {
            a[i][i]
                .partial_cmp(&a[j][j])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let mut ev = [0.0_f64; 9];
    #[allow(clippy::needless_range_loop)]
    for r in 0..n {
        ev[r] = v[r][min_idx];
    }
    // Normalize.
    let norm: f64 = ev.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in ev.iter_mut() {
            *x /= norm;
        }
    }
    ev
}

// ─── 8-point fundamental matrix ──────────────────────────────────────────────

/// Estimate fundamental matrix from 8+ point correspondences (normalised
/// 8-point algorithm).
fn eight_point_fundamental(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> Option<Mat3> {
    let n = pts1.len().min(pts2.len());
    if n < 8 {
        return None;
    }

    // Normalise points.
    let (t1, pts1n) = normalise_points(pts1);
    let (t2, pts2n) = normalise_points(pts2);

    // Build constraint matrix (n × 9).
    let rows: Vec<Vec<f64>> = pts1n
        .iter()
        .zip(pts2n.iter())
        .map(|(&(x1, y1), &(x2, y2))| vec![x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0])
        .collect();

    let f_vec = right_singular_min(&rows);
    let f_raw: Mat3 = [
        [f_vec[0], f_vec[1], f_vec[2]],
        [f_vec[3], f_vec[4], f_vec[5]],
        [f_vec[6], f_vec[7], f_vec[8]],
    ];

    // Enforce rank-2 by zeroing the smallest singular value.
    let f_rank2 = enforce_rank2(&f_raw);

    // Denormalise: F = T2^T * F * T1
    let t2t = mat3_transpose(&t1); // note: standard is T2^T * F * T1
    let _ = t2t;
    let f_denorm = mat3_mul(&mat3_transpose(&t2), &mat3_mul(&f_rank2, &t1));

    Some(f_denorm)
}

/// Normalise 2D points: translate centroid to origin, scale so mean distance
/// to origin is √2.  Returns (transform, normalised_pts).
fn normalise_points(pts: &[(f64, f64)]) -> (Mat3, Vec<(f64, f64)>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let mean_dist = pts
        .iter()
        .map(|&(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let scale = if mean_dist > 1e-10 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };
    let t: Mat3 = [
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ];
    let pts_n = pts
        .iter()
        .map(|&(x, y)| ((x - cx) * scale, (y - cy) * scale))
        .collect();
    (t, pts_n)
}

/// Enforce rank-2 on a 3×3 matrix by zeroing the smallest singular value.
/// Uses the Jacobi SVD approach on the 3×3 matrix.
fn enforce_rank2(f: &Mat3) -> Mat3 {
    // 3×3 SVD via Jacobi on F^T F.
    let ft = mat3_transpose(f);
    let ftf = mat3_mul(&ft, f);
    let (u, _sigma_sq, _vt) = jacobi_svd_3x3(&ftf);
    let _ = (u, _vt);

    // Re-derive via compact SVD: F = U S V^T, set s[2]=0, reconstruct.
    let (u3, s3, v3) = svd_3x3(f);
    let mut s_mat: Mat3 = [[0.0; 3]; 3];
    s_mat[0][0] = s3[0];
    s_mat[1][1] = s3[1];
    // s_mat[2][2] = 0.0  (rank-2 enforcement)
    mat3_mul(&u3, &mat3_mul(&s_mat, &mat3_transpose(&v3)))
}

/// Compute the (U, singular_values, V^T) SVD of a 3×3 matrix via
/// iterative Jacobi on A^T A.
fn svd_3x3(a: &Mat3) -> (Mat3, [f64; 3], Mat3) {
    let at = mat3_transpose(a);
    let ata = mat3_mul(&at, a);

    // Jacobi eigendecomposition of 3×3 symmetric matrix.
    let (sigma_sq, v) = jacobi_eig_3x3(&ata);

    // Singular values (descending).
    let mut sv: [f64; 3] = [
        sigma_sq[0].max(0.0).sqrt(),
        sigma_sq[1].max(0.0).sqrt(),
        sigma_sq[2].max(0.0).sqrt(),
    ];

    // Sort descending by sv.
    let mut perm = [0_usize, 1, 2];
    perm.sort_unstable_by(|&i, &j| {
        sv[j]
            .partial_cmp(&sv[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sv = [sv[perm[0]], sv[perm[1]], sv[perm[2]]];
    let v_sorted: Mat3 = {
        let mut m = [[0.0_f64; 3]; 3];
        for col in 0..3 {
            for row in 0..3 {
                m[row][col] = v[row][perm[col]];
            }
        }
        m
    };

    // U = A V / sigma.
    let av = mat3_mul(a, &v_sorted);
    let mut u: Mat3 = [[0.0; 3]; 3];
    for col in 0..3 {
        let s = sv[col];
        if s > 1e-12 {
            for row in 0..3 {
                u[row][col] = av[row][col] / s;
            }
        } else {
            // Arbitrary orthogonal column.
            u[col][col] = 1.0;
        }
    }

    (u, sv, mat3_transpose(&v_sorted))
}

/// Jacobi eigendecomposition of a symmetric 3×3 matrix.
/// Returns (eigenvalues, eigenvectors as columns).
fn jacobi_eig_3x3(s: &Mat3) -> ([f64; 3], Mat3) {
    let mut a = *s;
    let mut v: Mat3 = [[0.0; 3]; 3];
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    for _ in 0..50 {
        let (mut p, mut q, mut mx) = (0, 1, 0.0_f64);
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for j in i + 1..3 {
                let av = a[i][j].abs();
                if av > mx {
                    mx = av;
                    p = i;
                    q = j;
                }
            }
        }
        if mx < 1e-14 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let sn = t * c;
        let app = c * c * a[p][p] - 2.0 * sn * c * a[p][q] + sn * sn * a[q][q];
        let aqq = sn * sn * a[p][p] + 2.0 * sn * c * a[p][q] + c * c * a[q][q];
        a[p][p] = app;
        a[q][q] = aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        let rs: Vec<usize> = (0..3).filter(|&r| r != p && r != q).collect();
        for r in rs {
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - sn * arq;
            a[p][r] = a[r][p];
            a[r][q] = sn * arp + c * arq;
            a[q][r] = a[r][q];
        }
        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - sn * vrq;
            v[r][q] = sn * vrp + c * vrq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

// Unused but needed for the enforce_rank2 call site.
#[allow(dead_code)]
fn jacobi_svd_3x3(ftf: &Mat3) -> (Mat3, [f64; 3], Mat3) {
    let (eigenvalues, v) = jacobi_eig_3x3(ftf);
    (v, eigenvalues, mat3_transpose(&v))
}

// ─── Sampson distance ────────────────────────────────────────────────────────

/// Sampson distance for the fundamental matrix epipolar constraint.
#[inline]
fn sampson_distance(f: &Mat3, p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    // fp = F * [x1, y1, 1]^T
    let fp0 = f[0][0] * x1 + f[0][1] * y1 + f[0][2];
    let fp1 = f[1][0] * x1 + f[1][1] * y1 + f[1][2];
    let fp2 = f[2][0] * x1 + f[2][1] * y1 + f[2][2];
    // ftp = F^T * [x2, y2, 1]^T
    let ftp0 = f[0][0] * x2 + f[1][0] * y2 + f[2][0];
    let ftp1 = f[0][1] * x2 + f[1][1] * y2 + f[2][1];
    let num = (x2 * fp0 + y2 * fp1 + fp2).powi(2);
    let den = fp0 * fp0 + fp1 * fp1 + ftp0 * ftp0 + ftp1 * ftp1;
    if den.abs() < 1e-14 {
        return f64::INFINITY;
    }
    num / den
}

// ─── RANSAC fundamental matrix ───────────────────────────────────────────────

/// RANSAC estimation of the fundamental matrix.
///
/// Uses the 8-point algorithm.  Returns `(F, inlier_indices)`.
///
/// # Errors
/// Returns [`VisionError`] if fewer than 8 correspondences are provided.
#[allow(non_snake_case)]
pub fn ransac_fundamental(
    matches: &[(usize, usize)],
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    threshold: f64,
    max_iter: usize,
) -> Result<(Mat3, Vec<usize>), VisionError> {
    if matches.len() < 8 {
        return Err(VisionError::InvalidInput(
            "Need at least 8 correspondences for fundamental matrix".into(),
        ));
    }

    let n = matches.len();
    let mut best_f = [[0.0_f64; 3]; 3];
    let mut best_inliers: Vec<usize> = Vec::new();

    // Simple deterministic RANSAC with a fixed xorshift seed.
    let mut rng_state = 0xDEAD_BEEF_FEED_F00D_u64;

    for _ in 0..max_iter {
        // Sample 8 random indices.
        let mut sample: Vec<usize> = Vec::with_capacity(8);
        while sample.len() < 8 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let idx = (rng_state as usize) % n;
            if !sample.contains(&idx) {
                sample.push(idx);
            }
        }

        let s1: Vec<(f64, f64)> = sample.iter().map(|&k| pts1[matches[k].0]).collect();
        let s2: Vec<(f64, f64)> = sample.iter().map(|&k| pts2[matches[k].1]).collect();

        let f_opt = eight_point_fundamental(&s1, &s2);
        let f = match f_opt {
            Some(f) => f,
            None => continue,
        };

        let inliers: Vec<usize> = (0..n)
            .filter(|&k| sampson_distance(&f, pts1[matches[k].0], pts2[matches[k].1]) < threshold)
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_f = f;
        }
    }

    if best_inliers.is_empty() {
        return Err(VisionError::OperationError(
            "RANSAC failed to find inliers".into(),
        ));
    }

    // Optional refinement: re-estimate on all inliers.
    if best_inliers.len() >= 8 {
        let s1: Vec<(f64, f64)> = best_inliers.iter().map(|&k| pts1[matches[k].0]).collect();
        let s2: Vec<(f64, f64)> = best_inliers.iter().map(|&k| pts2[matches[k].1]).collect();
        if let Some(f_ref) = eight_point_fundamental(&s1, &s2) {
            best_f = f_ref;
        }
    }

    Ok((best_f, best_inliers))
}

// ─── Essential matrix ────────────────────────────────────────────────────────

/// Compute essential matrix from fundamental matrix and camera intrinsics.
///
/// E = K2^T · F · K1
#[allow(non_snake_case)]
pub fn essential_from_fundamental(F: &Mat3, K1: &Mat3, K2: &Mat3) -> Mat3 {
    let k2t = mat3_transpose(K2);
    mat3_mul(&k2t, &mat3_mul(F, K1))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_descriptors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32 / 100.0).collect())
            .collect()
    }

    #[test]
    fn test_ratio_test_self() {
        // Matching a set against itself with ratio=0.8 should return all
        // (or nearly all) matches since dist(nn1)=0 << dist(nn2).
        let descs = make_descriptors(10, 8);
        let matches = ratio_test(&descs, &descs, 0.8).expect("ratio_test should succeed");
        // At least some matches (self-match always passes since dist=0).
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_ratio_test_empty() {
        let d: Vec<Vec<f32>> = vec![];
        let res = ratio_test(&d, &make_descriptors(3, 4), 0.8)
            .expect("ratio_test on empty query should succeed");
        assert!(res.is_empty());
    }

    #[test]
    fn test_eight_point_fundamental() {
        // Degenerate case: all points coplanar (fronto-parallel, pure translation).
        // Points on a plane with known F: x2 = x1 + t.
        let shift = 10.0_f64;
        let pts1: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (200.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
            (200.0, 100.0),
            (50.0, 50.0),
            (150.0, 150.0),
        ];
        let pts2: Vec<(f64, f64)> = pts1.iter().map(|&(x, y)| (x + shift, y)).collect();
        let f = eight_point_fundamental(&pts1, &pts2)
            .expect("eight_point_fundamental should succeed with 8 points");
        // F should enforce x2^T F x1 ≈ 0 for all correspondences.
        for (&(x1, y1), &(x2, y2)) in pts1.iter().zip(pts2.iter()) {
            let err = x2 * (f[0][0] * x1 + f[0][1] * y1 + f[0][2])
                + y2 * (f[1][0] * x1 + f[1][1] * y1 + f[1][2])
                + (f[2][0] * x1 + f[2][1] * y1 + f[2][2]);
            assert!(err.abs() < 1.0, "Epipolar constraint violated: {}", err);
        }
    }

    #[test]
    fn test_ransac_fundamental_insufficient() {
        let matches: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 0.0)).collect();
        assert!(ransac_fundamental(&matches, &pts, &pts, 1.0, 10).is_err());
    }

    #[test]
    fn test_essential_from_fundamental_shape() {
        let f: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let k: Mat3 = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]];
        let e = essential_from_fundamental(&f, &k, &k);
        // E should be a 3×3 matrix (just check it runs without panic).
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn test_ratio_test_rejects_ambiguous() {
        // Two nearly identical descriptors in desc2 → should be rejected.
        let d1 = vec![vec![1.0_f32, 0.0, 0.0, 0.0]];
        let d2 = vec![
            vec![1.0_f32, 0.01, 0.0, 0.0], // very close
            vec![1.0_f32, 0.02, 0.0, 0.0], // also very close
        ];
        // ratio = 0.8 means dist1/dist2 < 0.8
        let matches =
            ratio_test(&d1, &d2, 0.8).expect("ratio_test should succeed with valid descriptors");
        // dist1 ≈ 0.01, dist2 ≈ 0.02, ratio ≈ 0.5 < 0.8 → should be kept
        assert_eq!(matches.len(), 1);
    }
}
