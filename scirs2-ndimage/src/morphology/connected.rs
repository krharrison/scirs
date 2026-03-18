//! Connected component operations for binary and labeled arrays

use scirs2_core::ndarray::{Array, Dimension, IxDyn};
use std::collections::HashMap;

use super::Connectivity;
use crate::error::{NdimageError, NdimageResult};

/// Union-Find data structure for connected component labeling
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        UnionFind {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            // Union by rank
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    fn get_component_mapping(&mut self) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();
        let mut next_label = 1;

        for i in 0..self.parent.len() {
            let root = self.find(i);
            if !mapping.contains_key(&root) {
                mapping.insert(root, next_label);
                next_label += 1;
            }
        }

        mapping
    }
}

/// Get neighbors for a given position based on connectivity
#[allow(dead_code)]
fn get_neighbors(
    position: &[usize],
    shape: &[usize],
    connectivity: Connectivity,
) -> Vec<Vec<usize>> {
    let ndim = position.len();
    let mut neighbors = Vec::new();

    match connectivity {
        Connectivity::Face => {
            // Face connectivity: only share a face (4-connectivity in 2D, 6-connectivity in 3D)
            for dim in 0..ndim {
                // Check negative direction
                if position[dim] > 0 {
                    let mut neighbor = position.to_vec();
                    neighbor[dim] -= 1;
                    neighbors.push(neighbor);
                }
                // Check positive direction
                if position[dim] + 1 < shape[dim] {
                    let mut neighbor = position.to_vec();
                    neighbor[dim] += 1;
                    neighbors.push(neighbor);
                }
            }
        }
        Connectivity::FaceEdge => {
            // Face and edge connectivity: 8-connectivity in 2D, 18-connectivity in 3D
            let offsets = generate_face_edge_offsets(ndim);
            for offset in offsets {
                let mut neighbor = Vec::with_capacity(ndim);
                let mut valid = true;

                for (i, &pos) in position.iter().enumerate() {
                    let new_pos = (pos as isize) + offset[i];
                    if new_pos < 0 || new_pos >= shape[i] as isize {
                        valid = false;
                        break;
                    }
                    neighbor.push(new_pos as usize);
                }

                if valid && neighbor != position {
                    neighbors.push(neighbor);
                }
            }
        }
        Connectivity::Full => {
            // Corner connectivity: all possible neighbors (8-connectivity in 2D, 26-connectivity in 3D)
            let offsets = generate_all_offsets(ndim);
            for offset in offsets {
                let mut neighbor = Vec::with_capacity(ndim);
                let mut valid = true;

                for (i, &pos) in position.iter().enumerate() {
                    let new_pos = (pos as isize) + offset[i];
                    if new_pos < 0 || new_pos >= shape[i] as isize {
                        valid = false;
                        break;
                    }
                    neighbor.push(new_pos as usize);
                }

                if valid && neighbor != position {
                    neighbors.push(neighbor);
                }
            }
        }
    }

    neighbors
}

/// Generate all possible offsets for corner connectivity
#[allow(dead_code)]
fn generate_all_offsets(ndim: usize) -> Vec<Vec<isize>> {
    let mut offsets = Vec::new();
    let total_combinations = 3_usize.pow(ndim as u32);

    for i in 0..total_combinations {
        let mut offset = Vec::with_capacity(ndim);
        let mut temp = i;

        for _ in 0..ndim {
            let val = (temp % 3) as isize - 1; // -1, 0, or 1
            offset.push(val);
            temp /= 3;
        }

        // Skip the center point (all zeros)
        if !offset.iter().all(|&x| x == 0) {
            offsets.push(offset);
        }
    }

    offsets
}

/// Generate face and edge neighbor offsets (excludes vertex neighbors in 3D+)
#[allow(dead_code)]
fn generate_face_edge_offsets(ndim: usize) -> Vec<Vec<isize>> {
    let mut offsets = Vec::new();
    let total_combinations = 3_usize.pow(ndim as u32);

    for i in 0..total_combinations {
        let mut offset = Vec::with_capacity(ndim);
        let mut temp = i;
        for _ in 0..ndim {
            let val = (temp % 3) as isize - 1; // -1, 0, or 1
            offset.push(val);
            temp /= 3;
        }

        // Skip the center point (all zeros)
        if offset.iter().all(|&x| x == 0) {
            continue;
        }

        // For face and edge connectivity, exclude vertex neighbors
        // Vertex neighbors have all non-zero components
        let non_zero_count = offset.iter().filter(|&&x| x != 0).count();

        // Include face neighbors (1 non-zero) and edge neighbors (2 non-zero in 3D+)
        // In 2D, this gives 8-connectivity (same as full)
        // In 3D, this gives 18-connectivity (excludes 8 vertex neighbors)
        if non_zero_count <= 2 {
            offsets.push(offset);
        }
    }

    offsets
}

/// Convert multi-dimensional index to flat index
#[allow(dead_code)]
fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat_index = 0;
    let mut stride = 1;

    for i in (0..indices.len()).rev() {
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }

    flat_index
}

/// Convert flat index to multi-dimensional index
#[allow(dead_code)]
fn unravel_index(_flatindex: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = _flatindex;

    // Process dimensions in forward order for row-major (C-order) layout
    for i in 0..shape.len() {
        let stride: usize = shape[(i + 1)..].iter().product();
        indices[i] = remaining / stride;
        remaining %= stride;
    }

    indices
}

/// Label connected components in a binary array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `structure` - Structuring element (if None, uses a box with connectivity 1)
/// * `connectivity` - Connectivity type (default: Face)
/// * `background` - Whether to consider background as a feature (default: false)
///
/// # Returns
///
/// * `Result<(Array<usize, D>, usize)>` - Labeled array and number of labels
#[allow(dead_code)]
pub fn label<D>(
    input: &Array<bool, D>,
    structure: Option<&Array<bool, D>>,
    connectivity: Option<Connectivity>,
    background: Option<bool>,
) -> NdimageResult<(Array<usize, D>, usize)>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);
    let bg = background.unwrap_or(false);

    // Structure must have same rank as input
    if let Some(struct_elem) = structure {
        if struct_elem.ndim() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Structure must have same rank as input (got {} expected {})",
                struct_elem.ndim(),
                input.ndim()
            )));
        }
    }

    let shape = input.shape();
    let total_elements: usize = shape.iter().product();

    if total_elements == 0 {
        let output = Array::<usize, D>::zeros(input.raw_dim());
        return Ok((output, 0));
    }

    // Initialize Union-Find data structure
    let mut uf = UnionFind::new(total_elements);

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    // First pass: scan all pixels and union adjacent foreground pixels
    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let current_pixel = input_dyn[IxDyn(&indices)];

        // Only process foreground pixels (or background if bg=true)
        if current_pixel == !bg {
            // Get neighbors based on connectivity
            let neighbors = get_neighbors(&indices, shape, conn);

            for neighbor_indices in neighbors {
                let neighbor_pixel = input_dyn[IxDyn(&neighbor_indices)];

                // If neighbor is also foreground, union them
                if neighbor_pixel == current_pixel {
                    let neighbor_flat_idx = ravel_index(&neighbor_indices, shape);
                    uf.union(flat_idx, neighbor_flat_idx);
                }
            }
        }
    }

    // Get component mapping (root -> label)
    let component_mapping = uf.get_component_mapping();

    // Create output array
    let mut output = Array::<usize, D>::zeros(input.raw_dim());
    let mut num_labels = 0;

    // Second pass: assign labels
    let mut output_dyn = output.clone().into_dyn();
    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let pixel = input_dyn[IxDyn(&indices)];

        if pixel == !bg {
            let root = uf.find(flat_idx);
            if let Some(&label) = component_mapping.get(&root) {
                output_dyn[IxDyn(&indices)] = label;
                num_labels = num_labels.max(label);
            }
        }
    }

    // Convert back to original dimension type
    output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimension type".into())
    })?;

    Ok((output, num_labels))
}

/// Find the boundaries of objects in a labeled array
///
/// # Arguments
///
/// * `input` - Input labeled array
/// * `connectivity` - Connectivity type (default: Face)
/// * `mode` - Mode for boundary detection: "inner", "outer", or "thick" (default: "outer")
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary array with boundaries
#[allow(dead_code)]
pub fn find_boundaries<D>(
    input: &Array<usize, D>,
    connectivity: Option<Connectivity>,
    mode: Option<&str>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);
    let mode_str = mode.unwrap_or("outer");

    // Validate mode
    if mode_str != "inner" && mode_str != "outer" && mode_str != "thick" {
        return Err(NdimageError::InvalidInput(format!(
            "Mode must be 'inner', 'outer', or 'thick', got '{}'",
            mode_str
        )));
    }

    let shape = input.shape();
    let total_elements: usize = shape.iter().product();
    let mut output = Array::<bool, D>::from_elem(input.raw_dim(), false);

    if total_elements == 0 {
        return Ok(output);
    }

    // Convert to dynamic arrays for easier indexing
    let input_dyn = input.clone().into_dyn();
    let mut output_dyn = output.clone().into_dyn();

    // Scan all pixels to find boundaries
    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let current_label = input_dyn[IxDyn(&indices)];

        // Skip background pixels for inner mode
        if mode_str == "inner" && current_label == 0 {
            continue;
        }

        // Get neighbors based on connectivity
        let neighbors = get_neighbors(&indices, shape, conn);
        let mut is_boundary = false;

        for neighbor_indices in neighbors {
            let neighbor_label = input_dyn[IxDyn(&neighbor_indices)];

            match mode_str {
                "inner"
                    // Inner boundary: foreground pixels adjacent to background or different labels
                    if current_label != 0
                        && (neighbor_label == 0 || neighbor_label != current_label)
                    => {
                        is_boundary = true;
                        break;
                    }
                "outer"
                    // Outer boundary: background pixels adjacent to foreground
                    if current_label == 0 && neighbor_label != 0 => {
                        is_boundary = true;
                        break;
                    }
                "thick"
                    // Thick boundary: both inner and outer
                    if current_label != neighbor_label => {
                        is_boundary = true;
                        break;
                    }
                _ => {} // Already validated above
            }
        }

        if is_boundary {
            output_dyn[IxDyn(&indices)] = true;
        }
    }

    // Convert back to original dimension type
    output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimension type".into())
    })?;

    Ok(output)
}

/// Remove small objects from a labeled array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `min_size` - Minimum size of objects to keep
/// * `connectivity` - Connectivity type (default: Face)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary array with small objects removed
#[allow(dead_code)]
pub fn remove_small_objects<D>(
    input: &Array<bool, D>,
    min_size: usize,
    connectivity: Option<Connectivity>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if min_size == 0 {
        return Err(NdimageError::InvalidInput(
            "min_size must be greater than 0".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);

    // Label connected components
    let (labeled, num_labels) = label(input, None, Some(conn), None)?;

    if num_labels == 0 {
        return Ok(Array::<bool, D>::from_elem(input.raw_dim(), false));
    }

    // Count the _size of each component
    let mut component_sizes = vec![0; num_labels + 1];
    for &label_val in labeled.iter() {
        if label_val > 0 {
            component_sizes[label_val] += 1;
        }
    }

    // Create output array, keeping only large enough components
    let mut output = Array::<bool, D>::from_elem(input.raw_dim(), false);
    let shape = input.shape();
    let total_elements: usize = shape.iter().product();

    // Convert to dynamic arrays for easier indexing
    let labeled_dyn = labeled.clone().into_dyn();
    let mut output_dyn = output.clone().into_dyn();

    for flat_idx in 0..total_elements {
        let indices = unravel_index(flat_idx, shape);
        let label_val = labeled_dyn[IxDyn(&indices)];

        if label_val > 0 && component_sizes[label_val] >= min_size {
            output_dyn[IxDyn(&indices)] = true;
        }
    }

    // Convert back to original dimension type
    output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimension type".into())
    })?;

    Ok(output)
}

/// Remove small holes from a labeled array
///
/// # Arguments
///
/// * `input` - Input binary array
/// * `min_size` - Minimum size of holes to keep
/// * `connectivity` - Connectivity type (default: Face)
///
/// # Returns
///
/// * `Result<Array<bool, D>>` - Binary array with small holes removed
#[allow(dead_code)]
pub fn remove_small_holes<D>(
    input: &Array<bool, D>,
    min_size: usize,
    connectivity: Option<Connectivity>,
) -> NdimageResult<Array<bool, D>>
where
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if min_size == 0 {
        return Err(NdimageError::InvalidInput(
            "min_size must be greater than 0".into(),
        ));
    }

    let conn = connectivity.unwrap_or(Connectivity::Face);

    // To remove small holes, we:
    // 1. Invert the binary image (holes become objects)
    // 2. Remove small objects from the inverted image
    // 3. Invert back

    // Create inverted image
    let mut inverted = input.clone();
    for pixel in inverted.iter_mut() {
        *pixel = !*pixel;
    }

    // Remove small objects from inverted image
    let filtered_inverted = remove_small_objects(&inverted, min_size, Some(conn))?;

    // Invert back to get result
    let mut output = filtered_inverted;
    for pixel in output.iter_mut() {
        *pixel = !*pixel;
    }

    Ok(output)
}

/// 2D bounding box for a labeled region
///
/// Represents the bounding rectangle of a labeled object in a 2D image.
/// All coordinates use the convention: `min` is inclusive, `max` is exclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundingBox2D {
    /// Label value of this object
    pub label: usize,
    /// Minimum row (inclusive)
    pub min_row: usize,
    /// Maximum row (exclusive)
    pub max_row: usize,
    /// Minimum column (inclusive)
    pub min_col: usize,
    /// Maximum column (exclusive)
    pub max_col: usize,
}

impl BoundingBox2D {
    /// Width of the bounding box in pixels
    pub fn width(&self) -> usize {
        self.max_col - self.min_col
    }

    /// Height of the bounding box in pixels
    pub fn height(&self) -> usize {
        self.max_row - self.min_row
    }

    /// Area of the bounding box in pixels
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }
}

/// Optimized 2D connected component labeling using two-pass algorithm with union-find
///
/// This is a specialized 2D version of `label()` that avoids dynamic dimension
/// conversions and runs significantly faster on 2D images.
///
/// # Arguments
///
/// * `input` - Input 2D binary array
/// * `connectivity` - Connectivity type (default: Face = 4-connectivity)
///
/// # Returns
///
/// * `Result<(Array2<usize>, usize)>` - Labeled array and number of features
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::morphology::label_2d;
///
/// let input = array![
///     [true, true, false, false],
///     [true, true, false, false],
///     [false, false, true, true],
///     [false, false, true, true],
/// ];
///
/// let (labeled, num_features) = label_2d(&input, None).expect("label_2d should succeed");
/// assert_eq!(num_features, 2);
/// ```
pub fn label_2d(
    input: &scirs2_core::ndarray::Array2<bool>,
    connectivity: Option<Connectivity>,
) -> NdimageResult<(scirs2_core::ndarray::Array2<usize>, usize)> {
    use scirs2_core::ndarray::Array2;

    let conn = connectivity.unwrap_or(Connectivity::Face);
    let rows = input.nrows();
    let cols = input.ncols();

    if rows == 0 || cols == 0 {
        return Ok((Array2::zeros((rows, cols)), 0));
    }

    let total = rows * cols;
    let mut uf = UnionFind::new(total);

    // Determine offsets: for 2D, Face = 4-connectivity, Full/FaceEdge = 8-connectivity
    let use_diag = matches!(conn, Connectivity::Full | Connectivity::FaceEdge);

    // First pass: scan left-to-right, top-to-bottom, checking already-visited neighbors
    for r in 0..rows {
        for c in 0..cols {
            if !input[[r, c]] {
                continue;
            }

            let idx = r * cols + c;

            // Check neighbor above
            if r > 0 && input[[r - 1, c]] {
                uf.union(idx, (r - 1) * cols + c);
            }

            // Check neighbor to the left
            if c > 0 && input[[r, c - 1]] {
                uf.union(idx, r * cols + (c - 1));
            }

            if use_diag {
                // Check upper-left diagonal
                if r > 0 && c > 0 && input[[r - 1, c - 1]] {
                    uf.union(idx, (r - 1) * cols + (c - 1));
                }
                // Check upper-right diagonal
                if r > 0 && c + 1 < cols && input[[r - 1, c + 1]] {
                    uf.union(idx, (r - 1) * cols + (c + 1));
                }
            }
        }
    }

    // Second pass: build label mapping (only for foreground pixels)
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 1usize;
    let mut output = Array2::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            if !input[[r, c]] {
                continue;
            }
            let idx = r * cols + c;
            let root = uf.find(idx);
            let lbl = match root_to_label.get(&root) {
                Some(&l) => l,
                None => {
                    let l = next_label;
                    root_to_label.insert(root, l);
                    next_label += 1;
                    l
                }
            };
            output[[r, c]] = lbl;
        }
    }

    let num_labels = next_label - 1;
    Ok((output, num_labels))
}

/// Find objects (bounding boxes) in a 2D labeled image
///
/// Returns a `BoundingBox2D` for each unique non-zero label in the input.
/// The bounding boxes are sorted by label value.
///
/// # Arguments
///
/// * `labeled` - 2D labeled array where 0 = background
///
/// # Returns
///
/// * `Result<Vec<BoundingBox2D>>` - Bounding boxes sorted by label
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::morphology::{label_2d, find_objects_2d};
///
/// let input = array![
///     [true, true, false, false],
///     [true, true, false, false],
///     [false, false, true, true],
///     [false, false, true, true],
/// ];
///
/// let (labeled, _) = label_2d(&input, None).expect("label_2d should succeed");
/// let objects = find_objects_2d(&labeled).expect("find_objects_2d should succeed");
/// assert_eq!(objects.len(), 2);
/// ```
pub fn find_objects_2d(
    labeled: &scirs2_core::ndarray::Array2<usize>,
) -> NdimageResult<Vec<BoundingBox2D>> {
    let rows = labeled.nrows();
    let cols = labeled.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    // Collect bounding boxes in a single pass
    let mut bbox_map: HashMap<usize, (usize, usize, usize, usize)> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let lbl = labeled[[r, c]];
            if lbl == 0 {
                continue;
            }
            let entry = bbox_map.entry(lbl).or_insert((r, r, c, c));
            if r < entry.0 {
                entry.0 = r;
            }
            if r > entry.1 {
                entry.1 = r;
            }
            if c < entry.2 {
                entry.2 = c;
            }
            if c > entry.3 {
                entry.3 = c;
            }
        }
    }

    let mut result: Vec<BoundingBox2D> = bbox_map
        .into_iter()
        .map(|(lbl, (min_r, max_r, min_c, max_c))| BoundingBox2D {
            label: lbl,
            min_row: min_r,
            max_row: max_r + 1, // exclusive
            min_col: min_c,
            max_col: max_c + 1, // exclusive
        })
        .collect();

    result.sort_by_key(|b| b.label);
    Ok(result)
}

/// Count pixels per label in a labeled image
///
/// # Arguments
///
/// * `labeled` - Labeled array
///
/// # Returns
///
/// * HashMap mapping label -> pixel count (excludes background label 0)
pub fn count_labels_2d(labeled: &scirs2_core::ndarray::Array2<usize>) -> HashMap<usize, usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &lbl in labeled.iter() {
        if lbl > 0 {
            *counts.entry(lbl).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_label() {
        let input = Array2::from_elem((3, 3), true);
        let (result, _num_labels) = label(&input, None, None, None).expect("Operation failed");
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_find_boundaries() {
        let input = Array2::from_elem((3, 3), 1);
        let result = find_boundaries(&input, None, None).expect("Operation failed");
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_remove_small_objects() {
        let input = Array2::from_elem((3, 3), true);
        let result = remove_small_objects(&input, 1, None).expect("Operation failed");
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_label_2d_two_components() {
        let input = array![
            [true, true, false, false],
            [true, true, false, false],
            [false, false, true, true],
            [false, false, true, true],
        ];
        let (labeled, num) = label_2d(&input, None).expect("label_2d should succeed");
        assert_eq!(num, 2);
        // Top-left block should have one label, bottom-right another
        let l1 = labeled[[0, 0]];
        let l2 = labeled[[2, 2]];
        assert_ne!(l1, 0);
        assert_ne!(l2, 0);
        assert_ne!(l1, l2);
        // All pixels in block 1 should share a label
        assert_eq!(labeled[[0, 0]], labeled[[0, 1]]);
        assert_eq!(labeled[[0, 0]], labeled[[1, 0]]);
        assert_eq!(labeled[[0, 0]], labeled[[1, 1]]);
    }

    #[test]
    fn test_label_2d_single_component_8conn() {
        // With 8-connectivity, diagonal neighbors merge components
        let input = array![
            [true, false, false],
            [false, true, false],
            [false, false, true],
        ];
        let (labeled, num) =
            label_2d(&input, Some(Connectivity::Full)).expect("label_2d 8-conn should succeed");
        assert_eq!(num, 1);
        assert_eq!(labeled[[0, 0]], labeled[[1, 1]]);
        assert_eq!(labeled[[1, 1]], labeled[[2, 2]]);
    }

    #[test]
    fn test_label_2d_multiple_components_4conn() {
        // With 4-connectivity, diagonal pixels are separate
        let input = array![
            [true, false, false],
            [false, true, false],
            [false, false, true],
        ];
        let (labeled, num) =
            label_2d(&input, Some(Connectivity::Face)).expect("label_2d 4-conn should succeed");
        assert_eq!(num, 3);
        // Each diagonal pixel should be a different component
        let l0 = labeled[[0, 0]];
        let l1 = labeled[[1, 1]];
        let l2 = labeled[[2, 2]];
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        assert_ne!(l0, l2);
    }

    #[test]
    fn test_label_2d_empty() {
        let input = Array2::from_elem((3, 3), false);
        let (labeled, num) = label_2d(&input, None).expect("empty should succeed");
        assert_eq!(num, 0);
        for &v in labeled.iter() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_label_2d_all_foreground() {
        let input = Array2::from_elem((4, 4), true);
        let (labeled, num) = label_2d(&input, None).expect("all-foreground should succeed");
        assert_eq!(num, 1);
        let expected_label = labeled[[0, 0]];
        for &v in labeled.iter() {
            assert_eq!(v, expected_label);
        }
    }

    #[test]
    fn test_find_objects_2d_basic() {
        let input = array![
            [true, true, false, false],
            [true, true, false, false],
            [false, false, true, true],
            [false, false, true, true],
        ];
        let (labeled, _) = label_2d(&input, None).expect("label_2d should succeed");
        let objects = find_objects_2d(&labeled).expect("find_objects_2d should succeed");
        assert_eq!(objects.len(), 2);

        // First object (label 1) should be in top-left
        let obj1 = &objects[0];
        assert_eq!(obj1.min_row, 0);
        assert_eq!(obj1.max_row, 2);
        assert_eq!(obj1.min_col, 0);
        assert_eq!(obj1.max_col, 2);
        assert_eq!(obj1.width(), 2);
        assert_eq!(obj1.height(), 2);

        // Second object (label 2) should be in bottom-right
        let obj2 = &objects[1];
        assert_eq!(obj2.min_row, 2);
        assert_eq!(obj2.max_row, 4);
        assert_eq!(obj2.min_col, 2);
        assert_eq!(obj2.max_col, 4);
    }

    #[test]
    fn test_find_objects_2d_no_objects() {
        let labeled = Array2::<usize>::zeros((5, 5));
        let objects = find_objects_2d(&labeled).expect("no objects should succeed");
        assert!(objects.is_empty());
    }

    #[test]
    fn test_find_objects_2d_single_pixel() {
        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[2, 3]] = 1;
        let objects = find_objects_2d(&labeled).expect("single pixel should succeed");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].min_row, 2);
        assert_eq!(objects[0].max_row, 3);
        assert_eq!(objects[0].min_col, 3);
        assert_eq!(objects[0].max_col, 4);
        assert_eq!(objects[0].area(), 1);
    }

    #[test]
    fn test_count_labels_2d() {
        let labeled = array![[0, 1, 1, 0], [0, 1, 0, 2], [3, 0, 0, 2], [3, 3, 0, 0],];
        let counts = count_labels_2d(&labeled);
        assert_eq!(counts.get(&1), Some(&3));
        assert_eq!(counts.get(&2), Some(&2));
        assert_eq!(counts.get(&3), Some(&3));
        assert_eq!(counts.get(&0), None); // background not counted
    }

    #[test]
    fn test_label_2d_l_shape() {
        // Test an L-shaped region that requires union-find path compression
        let input = array![
            [true, false, false],
            [true, false, false],
            [true, true, true],
        ];
        let (labeled, num) = label_2d(&input, None).expect("L-shape should succeed");
        assert_eq!(num, 1);
        let expected = labeled[[0, 0]];
        assert_eq!(labeled[[1, 0]], expected);
        assert_eq!(labeled[[2, 0]], expected);
        assert_eq!(labeled[[2, 1]], expected);
        assert_eq!(labeled[[2, 2]], expected);
    }
}
