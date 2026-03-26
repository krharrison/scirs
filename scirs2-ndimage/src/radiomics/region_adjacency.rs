//! Region adjacency graph (RAG) construction and merging.
//!
//! Builds a weighted graph from a label image where nodes correspond to
//! unique region labels and edges connect spatially adjacent regions.
//! Edge weights capture the mean absolute intensity difference across the
//! shared boundary.

use std::collections::HashMap;

/// Configuration for RAG construction and merging.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Pixel connectivity used when scanning for adjacent pairs.
    /// For 2-D: 4 (N/S/E/W) or 8 (also diagonals).
    /// For 3-D: 6 (faces) or 26 (faces+edges+corners).
    pub connectivity: u8,
    /// Intensity-difference threshold used when deciding whether to merge.
    pub merge_threshold: f64,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            connectivity: 4,
            merge_threshold: 10.0,
        }
    }
}

/// A single edge in the region adjacency graph.
#[derive(Debug, Clone, PartialEq)]
pub struct RagEdge {
    /// Label of the first region (always ≤ `region2`)
    pub region1: usize,
    /// Label of the second region
    pub region2: usize,
    /// Mean absolute intensity difference across the shared boundary
    pub weight: f64,
}

/// A region adjacency graph.
#[derive(Debug, Clone)]
pub struct RegionAdjacencyGraph {
    /// Sorted list of unique region labels present in the label image.
    pub regions: Vec<usize>,
    /// Edges between adjacent regions (one entry per adjacent pair).
    pub edges: Vec<RagEdge>,
}

// ---------------------------------------------------------------------------
// 2-D RAG
// ---------------------------------------------------------------------------

/// Build a 2-D region adjacency graph.
///
/// * `labels` – 2-D label image (rows × cols); 0 is conventionally background
/// * `image`  – 2-D intensity image of the same shape
/// * `config` – connectivity and merge threshold
pub fn build_rag_2d(
    labels: &[Vec<usize>],
    image: &[Vec<f64>],
    config: &RagConfig,
) -> RegionAdjacencyGraph {
    let rows = labels.len();
    if rows == 0 {
        return RegionAdjacencyGraph {
            regions: vec![],
            edges: vec![],
        };
    }
    let cols = labels[0].len();

    // Accumulate boundary intensity differences per (min,max) label pair
    // key = (label_a, label_b) with label_a < label_b
    // value = (sum_of_abs_diff, count)
    let mut edge_acc: HashMap<(usize, usize), (f64, usize)> = HashMap::new();
    let mut region_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    let use_8conn = config.connectivity == 8;

    for r in 0..rows {
        for c in 0..cols {
            let la = labels[r][c];
            let ia = image[r][c];
            region_set.insert(la);

            // Neighbour offsets for 4-connectivity (always used)
            let neighbours_4 = [(0isize, 1isize), (1, 0)];
            // Additional neighbours for 8-connectivity
            let neighbours_diag = [(1isize, 1isize), (1, -1isize)];

            let check_neighbour =
                |nr: isize, nc: isize, acc: &mut HashMap<(usize, usize), (f64, usize)>| {
                    if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                        return;
                    }
                    let lb = labels[nr as usize][nc as usize];
                    if la == lb {
                        return;
                    }
                    let ib = image[nr as usize][nc as usize];
                    let key = if la < lb { (la, lb) } else { (lb, la) };
                    let diff = (ia - ib).abs();
                    let entry = acc.entry(key).or_insert((0.0, 0));
                    entry.0 += diff;
                    entry.1 += 1;
                };

            for (dr, dc) in &neighbours_4 {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                check_neighbour(nr, nc, &mut edge_acc);
            }
            if use_8conn {
                for (dr, dc) in &neighbours_diag {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    check_neighbour(nr, nc, &mut edge_acc);
                }
            }
        }
    }

    let regions: Vec<usize> = region_set.into_iter().collect();
    let edges = edge_acc
        .into_iter()
        .map(|((r1, r2), (sum, cnt))| RagEdge {
            region1: r1,
            region2: r2,
            weight: if cnt > 0 { sum / cnt as f64 } else { 0.0 },
        })
        .collect();

    RegionAdjacencyGraph { regions, edges }
}

// ---------------------------------------------------------------------------
// 3-D RAG
// ---------------------------------------------------------------------------

/// Build a 3-D region adjacency graph.
///
/// * `labels` – 3-D label volume (z × y × x)
/// * `image`  – 3-D intensity volume of the same shape
/// * `config` – connectivity (6 or 26) and merge threshold
pub fn build_rag_3d(
    labels: &[Vec<Vec<usize>>],
    image: &[Vec<Vec<f64>>],
    config: &RagConfig,
) -> RegionAdjacencyGraph {
    let nz = labels.len();
    if nz == 0 {
        return RegionAdjacencyGraph {
            regions: vec![],
            edges: vec![],
        };
    }
    let ny = labels[0].len();
    let nx = if ny > 0 { labels[0][0].len() } else { 0 };

    let mut edge_acc: HashMap<(usize, usize), (f64, usize)> = HashMap::new();
    let mut region_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    let use_26conn = config.connectivity == 26;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let la = labels[z][y][x];
                let ia = image[z][y][x];
                region_set.insert(la);

                // Generate neighbours according to connectivity
                // 6-connectivity: face neighbours (+z, +y, +x directions only to avoid duplicates)
                let face_offsets: &[(isize, isize, isize)] = &[(1, 0, 0), (0, 1, 0), (0, 0, 1)];
                // 26-connectivity adds edge (12) and corner (8) neighbours
                let extra_offsets: &[(isize, isize, isize)] = &[
                    (1, 1, 0),
                    (1, -1, 0),
                    (1, 0, 1),
                    (1, 0, -1),
                    (0, 1, 1),
                    (0, 1, -1),
                    (1, 1, 1),
                    (1, 1, -1),
                    (1, -1, 1),
                    (1, -1, -1),
                ];

                let check =
                    |nz2: isize,
                     ny2: isize,
                     nx2: isize,
                     acc: &mut HashMap<(usize, usize), (f64, usize)>| {
                        if nz2 < 0
                            || nz2 >= nz as isize
                            || ny2 < 0
                            || ny2 >= ny as isize
                            || nx2 < 0
                            || nx2 >= nx as isize
                        {
                            return;
                        }
                        let lb = labels[nz2 as usize][ny2 as usize][nx2 as usize];
                        if la == lb {
                            return;
                        }
                        let ib = image[nz2 as usize][ny2 as usize][nx2 as usize];
                        let key = if la < lb { (la, lb) } else { (lb, la) };
                        let diff = (ia - ib).abs();
                        let entry = acc.entry(key).or_insert((0.0, 0));
                        entry.0 += diff;
                        entry.1 += 1;
                    };

                for (dz, dy, dx) in face_offsets {
                    check(
                        z as isize + dz,
                        y as isize + dy,
                        x as isize + dx,
                        &mut edge_acc,
                    );
                }
                if use_26conn {
                    for (dz, dy, dx) in extra_offsets {
                        check(
                            z as isize + dz,
                            y as isize + dy,
                            x as isize + dx,
                            &mut edge_acc,
                        );
                    }
                }
            }
        }
    }

    let regions: Vec<usize> = region_set.into_iter().collect();
    let edges = edge_acc
        .into_iter()
        .map(|((r1, r2), (sum, cnt))| RagEdge {
            region1: r1,
            region2: r2,
            weight: if cnt > 0 { sum / cnt as f64 } else { 0.0 },
        })
        .collect();

    RegionAdjacencyGraph { regions, edges }
}

// ---------------------------------------------------------------------------
// Merge small regions
// ---------------------------------------------------------------------------

/// Merge regions in a 2-D label image that are smaller than `min_region_size`.
///
/// Each small region is merged into the adjacent region that shares the
/// lowest-weight boundary (i.e., most similar mean intensity).
///
/// Returns the number of merges performed.
///
/// # Arguments
/// * `rag`    – the region adjacency graph (used to look up adjacency weights)
/// * `labels` – mutable 2-D label image that is updated in place
/// * `min_region_size` – regions with fewer pixels are merged
pub fn merge_small_regions(
    rag: &RegionAdjacencyGraph,
    labels: &mut [Vec<usize>],
    min_region_size: usize,
) -> usize {
    let rows = labels.len();
    if rows == 0 {
        return 0;
    }
    let cols = labels[0].len();

    // Count region sizes
    let mut size_map: HashMap<usize, usize> = HashMap::new();
    for row in labels.iter() {
        for &lbl in row.iter() {
            *size_map.entry(lbl).or_insert(0) += 1;
        }
    }

    // Build adjacency: for each region, which neighbours exist and what are
    // their edge weights?
    let mut adj: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for edge in &rag.edges {
        adj.entry(edge.region1)
            .or_default()
            .push((edge.region2, edge.weight));
        adj.entry(edge.region2)
            .or_default()
            .push((edge.region1, edge.weight));
    }

    // Relabelling map: label → canonical label after merges
    let mut relabel: HashMap<usize, usize> = rag.regions.iter().map(|&r| (r, r)).collect();

    let resolve = |map: &HashMap<usize, usize>, mut lbl: usize| -> usize {
        // Path follows until stable (simple union-find without path compression)
        for _ in 0..1000 {
            let next = *map.get(&lbl).unwrap_or(&lbl);
            if next == lbl {
                break;
            }
            lbl = next;
        }
        lbl
    };

    let mut n_merges = 0usize;

    // Iteratively merge the smallest region that is below the threshold
    loop {
        // Find the smallest region below min_region_size
        let candidate = size_map
            .iter()
            .filter(|(&lbl, &sz)| sz < min_region_size && resolve(&relabel, lbl) == lbl)
            .min_by_key(|&(_, &sz)| sz)
            .map(|(&lbl, _)| lbl);

        let small_lbl = match candidate {
            Some(l) => l,
            None => break,
        };

        // Find the neighbour with the smallest edge weight
        let neighbours = match adj.get(&small_lbl) {
            Some(v) => v.clone(),
            None => break,
        };

        let best_neighbour = neighbours
            .iter()
            .filter_map(|(nb, w)| {
                let canonical = resolve(&relabel, *nb);
                if canonical != small_lbl {
                    Some((canonical, *w))
                } else {
                    None
                }
            })
            .min_by(|(_, wa), (_, wb)| wa.partial_cmp(wb).unwrap_or(std::cmp::Ordering::Equal));

        let target = match best_neighbour {
            Some((nb, _)) => nb,
            None => break, // isolated region — cannot merge
        };

        // Merge small_lbl → target
        relabel.insert(small_lbl, target);
        let small_size = *size_map.get(&small_lbl).unwrap_or(&0);
        *size_map.entry(target).or_insert(0) += small_size;
        size_map.remove(&small_lbl);
        n_merges += 1;
    }

    // Apply relabelling to the label image
    for row in labels.iter_mut() {
        for lbl in row.iter_mut() {
            *lbl = resolve(&relabel, *lbl);
        }
    }

    n_merges
}

// ---------------------------------------------------------------------------
// RAG → adjacency matrix
// ---------------------------------------------------------------------------

/// Convert a `RegionAdjacencyGraph` to a dense adjacency matrix.
///
/// Returns `(sorted_labels, weight_matrix)` where `weight_matrix[i][j]` is the
/// edge weight between `sorted_labels[i]` and `sorted_labels[j]`, or 0.0 if
/// the two regions are not adjacent.
pub fn rag_to_adjacency_matrix(rag: &RegionAdjacencyGraph) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = rag.regions.len();
    let mut labels = rag.regions.clone();
    labels.sort_unstable();

    // Map label → index
    let label_to_idx: HashMap<usize, usize> =
        labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    let mut matrix = vec![vec![0.0f64; n]; n];

    for edge in &rag.edges {
        if let (Some(&i), Some(&j)) = (
            label_to_idx.get(&edge.region1),
            label_to_idx.get(&edge.region2),
        ) {
            matrix[i][j] = edge.weight;
            matrix[j][i] = edge.weight;
        }
    }

    (labels, matrix)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 4×4 label image with exactly 2 regions (region 1 and region 2 separated
    /// at column 2).
    fn two_region_labels() -> Vec<Vec<usize>> {
        vec![
            vec![1, 1, 2, 2],
            vec![1, 1, 2, 2],
            vec![1, 1, 2, 2],
            vec![1, 1, 2, 2],
        ]
    }

    fn two_region_image() -> Vec<Vec<f64>> {
        vec![
            vec![10.0, 10.0, 50.0, 50.0],
            vec![10.0, 10.0, 50.0, 50.0],
            vec![10.0, 10.0, 50.0, 50.0],
            vec![10.0, 10.0, 50.0, 50.0],
        ]
    }

    #[test]
    fn test_two_regions_one_edge() {
        let labels = two_region_labels();
        let image = two_region_image();
        let config = RagConfig::default();
        let rag = build_rag_2d(&labels, &image, &config);
        assert_eq!(
            rag.edges.len(),
            1,
            "exactly 1 edge between region 1 and region 2"
        );
        assert_eq!(rag.regions.len(), 2);
    }

    #[test]
    fn test_edge_weight_correct() {
        let labels = two_region_labels();
        let image = two_region_image();
        let config = RagConfig::default();
        let rag = build_rag_2d(&labels, &image, &config);
        let w = rag.edges[0].weight;
        // All boundary pixels differ by |10 − 50| = 40
        assert!((w - 40.0).abs() < 1e-10, "expected weight 40.0, got {}", w);
    }

    #[test]
    fn test_merge_small_regions() {
        // Create a 4×4 image with a tiny region (1 pixel) in the corner
        //   Labels:  [1,1,1,1]
        //            [1,1,1,1]
        //            [1,1,1,1]
        //            [1,1,1,3]   ← region 3 is 1 pixel
        let mut labels = vec![
            vec![1usize, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 3],
        ];
        let image = vec![
            vec![10.0f64; 4],
            vec![10.0f64; 4],
            vec![10.0f64; 4],
            vec![10.0, 10.0, 10.0, 50.0],
        ];
        let config = RagConfig {
            connectivity: 4,
            merge_threshold: 5.0,
        };
        let rag = build_rag_2d(&labels, &image, &config);
        let n = merge_small_regions(&rag, &mut labels, 2);
        assert_eq!(n, 1, "one merge should have happened");
        // Region 3 (single pixel) should now be merged into region 1
        assert_eq!(labels[3][3], 1);
    }

    #[test]
    fn test_adjacency_matrix_symmetry() {
        let labels = two_region_labels();
        let image = two_region_image();
        let config = RagConfig::default();
        let rag = build_rag_2d(&labels, &image, &config);
        let (_sorted_labels, matrix) = rag_to_adjacency_matrix(&rag);
        // Should be symmetric
        for i in 0..matrix.len() {
            for j in 0..matrix.len() {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-12,
                    "matrix not symmetric at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn test_3d_rag_basic() {
        // Two 1×2×1 regions side by side in y dimension
        let labels = vec![vec![vec![1usize, 2]], vec![vec![1usize, 2]]];
        let image = vec![vec![vec![5.0f64, 15.0f64]], vec![vec![5.0f64, 15.0f64]]];
        let config = RagConfig {
            connectivity: 6,
            ..Default::default()
        };
        let rag = build_rag_3d(&labels, &image, &config);
        assert!(!rag.edges.is_empty(), "should have at least one edge");
    }
}
