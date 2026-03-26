//! 4D spatiotemporal array support for scirs2-ndimage.
//!
//! Provides `Array4D<T>` with shape [T×D×H×W] and operations including
//! Gaussian filtering, temporal differencing, MIP, connected components 4D,
//! and region tracking.

use crate::error::NdimageError;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Array4D
// ---------------------------------------------------------------------------

/// A 4-dimensional array with layout [time, depth, height, width].
#[derive(Debug, Clone)]
pub struct Array4D<T> {
    data: Vec<T>,
    shape: [usize; 4],
}

impl<T: Clone + Default> Array4D<T> {
    /// Create a new array filled with `fill`.
    pub fn new(shape: [usize; 4], fill: T) -> Self {
        let n = shape[0] * shape[1] * shape[2] * shape[3];
        Array4D {
            data: vec![fill; n],
            shape,
        }
    }

    /// Create from flat data vector. Returns error if length mismatches.
    pub fn from_data(data: Vec<T>, shape: [usize; 4]) -> Result<Self, NdimageError> {
        let expected = shape[0] * shape[1] * shape[2] * shape[3];
        if data.len() != expected {
            return Err(NdimageError::DimensionError(format!(
                "from_data: data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            )));
        }
        Ok(Array4D { data, shape })
    }

    /// Return the shape [T, D, H, W].
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// Total number of elements.
    pub fn n_elements(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]
    }

    /// Flat index for (t, d, h, w).
    pub fn index(&self, t: usize, d: usize, h: usize, w: usize) -> usize {
        t * (self.shape[1] * self.shape[2] * self.shape[3])
            + d * (self.shape[2] * self.shape[3])
            + h * self.shape[3]
            + w
    }

    /// Get immutable reference to element, returns None if out of bounds.
    pub fn get(&self, t: usize, d: usize, h: usize, w: usize) -> Option<&T> {
        if t < self.shape[0] && d < self.shape[1] && h < self.shape[2] && w < self.shape[3] {
            let idx = self.index(t, d, h, w);
            self.data.get(idx)
        } else {
            None
        }
    }

    /// Get mutable reference to element, returns None if out of bounds.
    pub fn get_mut(&mut self, t: usize, d: usize, h: usize, w: usize) -> Option<&mut T> {
        if t < self.shape[0] && d < self.shape[1] && h < self.shape[2] && w < self.shape[3] {
            let idx = self.index(t, d, h, w);
            self.data.get_mut(idx)
        } else {
            None
        }
    }

    /// Set element. Returns error if out of bounds.
    pub fn set(
        &mut self,
        t: usize,
        d: usize,
        h: usize,
        w: usize,
        value: T,
    ) -> Result<(), NdimageError> {
        if t >= self.shape[0] || d >= self.shape[1] || h >= self.shape[2] || w >= self.shape[3] {
            return Err(NdimageError::InvalidInput(format!(
                "set: index ({},{},{},{}) out of bounds for shape {:?}",
                t, d, h, w, self.shape
            )));
        }
        let idx = self.index(t, d, h, w);
        self.data[idx] = value;
        Ok(())
    }

    /// Extract the 3D volume at time t as nested Vec `[D][H][W]`.
    pub fn slice_time(&self, t: usize) -> Result<Vec<Vec<Vec<T>>>, NdimageError> {
        if t >= self.shape[0] {
            return Err(NdimageError::InvalidInput(format!(
                "slice_time: t={} out of bounds (shape[0]={})",
                t, self.shape[0]
            )));
        }
        let (nt, nd, nh, nw) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let _ = nt; // suppress unused
        let mut volume = Vec::with_capacity(nd);
        for d in 0..nd {
            let mut plane = Vec::with_capacity(nh);
            for h in 0..nh {
                let mut row = Vec::with_capacity(nw);
                for w in 0..nw {
                    let idx = self.index(t, d, h, w);
                    row.push(self.data[idx].clone());
                }
                plane.push(row);
            }
            volume.push(plane);
        }
        Ok(volume)
    }

    /// Extract the temporal slice at voxel (d, h, w) as Vec length T.
    pub fn slice_spatial(&self, d: usize, h: usize, w: usize) -> Result<Vec<T>, NdimageError> {
        if d >= self.shape[1] || h >= self.shape[2] || w >= self.shape[3] {
            return Err(NdimageError::InvalidInput(format!(
                "slice_spatial: ({},{},{}) out of bounds for shape {:?}",
                d, h, w, self.shape
            )));
        }
        let nt = self.shape[0];
        let mut result = Vec::with_capacity(nt);
        for t in 0..nt {
            let idx = self.index(t, d, h, w);
            result.push(self.data[idx].clone());
        }
        Ok(result)
    }

    /// Immutable access to flat data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to flat data.
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }
}

// ---------------------------------------------------------------------------
// 4D label alias
// ---------------------------------------------------------------------------

/// Label array for 4D connected components (0 = background, ≥1 = labels).
pub type Label4D = Array4D<usize>;

// ---------------------------------------------------------------------------
// Gaussian helpers
// ---------------------------------------------------------------------------

/// Build a 1D Gaussian kernel of given sigma (truncated at 3*sigma, odd length).
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let half = (3.0 * sigma).ceil() as usize;
    let len = 2 * half + 1;
    let mut kernel = Vec::with_capacity(len);
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..len {
        let x = i as f64 - half as f64;
        let v = (-x * x / s2).exp();
        kernel.push(v);
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Convolve a slice (1D) with kernel using reflect padding.
fn convolve_1d(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = data.len();
    let k = kernel.len();
    let half = k / 2;
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut val = 0.0;
        for j in 0..k {
            let src = i as isize + j as isize - half as isize;
            // reflect padding
            let idx = reflect_index(src, n);
            val += data[idx] * kernel[j];
        }
        out[i] = val;
    }
    out
}

fn reflect_index(i: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let n = n as isize;
    let mut i = i;
    if i < 0 {
        i = -i - 1;
    }
    if i >= n {
        i = 2 * n - i - 1;
    }
    i.max(0).min(n - 1) as usize
}

// ---------------------------------------------------------------------------
// Gaussian filter 4D
// ---------------------------------------------------------------------------

/// Apply separable Gaussian filtering to a 4D array.
///
/// * `sigma_t` — temporal smoothing (along axis 0)
/// * `sigma_s` — spatial smoothing (along axes 1, 2, 3)
pub fn gaussian_filter_4d(arr: &Array4D<f64>, sigma_t: f64, sigma_s: f64) -> Array4D<f64> {
    let [nt, nd, nh, nw] = arr.shape();
    let kernel_t = gaussian_kernel_1d(sigma_t);
    let kernel_s = gaussian_kernel_1d(sigma_s);

    // Work on flat data copy
    let mut buf: Vec<f64> = arr.data().to_vec();

    // Helper: get element
    let idx = |t: usize, d: usize, h: usize, w: usize| -> usize {
        t * (nd * nh * nw) + d * (nh * nw) + h * nw + w
    };

    // Smooth along time axis (axis 0)
    if kernel_t.len() > 1 {
        let src = buf.clone();
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let slice: Vec<f64> = (0..nt).map(|t| src[idx(t, d, h, w)]).collect();
                    let smoothed = convolve_1d(&slice, &kernel_t);
                    for t in 0..nt {
                        buf[idx(t, d, h, w)] = smoothed[t];
                    }
                }
            }
        }
    }

    // Smooth along depth axis (axis 1)
    if kernel_s.len() > 1 {
        let src = buf.clone();
        for t in 0..nt {
            for h in 0..nh {
                for w in 0..nw {
                    let slice: Vec<f64> = (0..nd).map(|d| src[idx(t, d, h, w)]).collect();
                    let smoothed = convolve_1d(&slice, &kernel_s);
                    for d in 0..nd {
                        buf[idx(t, d, h, w)] = smoothed[d];
                    }
                }
            }
        }
    }

    // Smooth along height axis (axis 2)
    if kernel_s.len() > 1 {
        let src = buf.clone();
        for t in 0..nt {
            for d in 0..nd {
                for w in 0..nw {
                    let slice: Vec<f64> = (0..nh).map(|h| src[idx(t, d, h, w)]).collect();
                    let smoothed = convolve_1d(&slice, &kernel_s);
                    for h in 0..nh {
                        buf[idx(t, d, h, w)] = smoothed[h];
                    }
                }
            }
        }
    }

    // Smooth along width axis (axis 3)
    if kernel_s.len() > 1 {
        let src = buf.clone();
        for t in 0..nt {
            for d in 0..nd {
                for h in 0..nh {
                    let slice: Vec<f64> = (0..nw).map(|w| src[idx(t, d, h, w)]).collect();
                    let smoothed = convolve_1d(&slice, &kernel_s);
                    for w in 0..nw {
                        buf[idx(t, d, h, w)] = smoothed[w];
                    }
                }
            }
        }
    }

    Array4D {
        data: buf,
        shape: [nt, nd, nh, nw],
    }
}

// ---------------------------------------------------------------------------
// Temporal finite differences
// ---------------------------------------------------------------------------

/// Compute temporal finite differences: `output[t] = input[t+1] - input[t]`.
/// Output shape is `[T-1, D, H, W]`.
pub fn diff_4d_temporal(arr: &Array4D<f64>) -> Array4D<f64> {
    let [nt, nd, nh, nw] = arr.shape();
    if nt < 2 {
        return Array4D::new([0, nd, nh, nw], 0.0);
    }
    let out_t = nt - 1;
    let mut out = Array4D::new([out_t, nd, nh, nw], 0.0);
    for t in 0..out_t {
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let v1 = arr.get(t + 1, d, h, w).copied().unwrap_or(0.0);
                    let v0 = arr.get(t, d, h, w).copied().unwrap_or(0.0);
                    let _ = out.set(t, d, h, w, v1 - v0);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Maximum Intensity Projection
// ---------------------------------------------------------------------------

/// Maximum intensity projection along the specified axis.
///
/// * axis 0 → T dimension collapsed → output shape [1, D, H, W]
/// * axis 1 → D dimension collapsed → output shape [T, 1, H, W]
/// * axis 2 → H dimension collapsed → output shape [T, D, 1, W]
/// * axis 3 → W dimension collapsed → output shape [T, D, H, 1]
pub fn max_intensity_projection_4d(
    arr: &Array4D<f64>,
    axis: usize,
) -> Result<Array4D<f64>, NdimageError> {
    let [nt, nd, nh, nw] = arr.shape();
    if axis > 3 {
        return Err(NdimageError::InvalidInput(format!(
            "max_intensity_projection_4d: axis {} invalid for 4D array",
            axis
        )));
    }

    let out_shape = match axis {
        0 => [1, nd, nh, nw],
        1 => [nt, 1, nh, nw],
        2 => [nt, nd, 1, nw],
        3 => [nt, nd, nh, 1],
        _ => unreachable!(),
    };

    let mut out = Array4D::new(out_shape, f64::NEG_INFINITY);

    for t in 0..nt {
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let v = arr.get(t, d, h, w).copied().unwrap_or(f64::NEG_INFINITY);
                    let (ot, od, oh, ow) = match axis {
                        0 => (0, d, h, w),
                        1 => (t, 0, h, w),
                        2 => (t, d, 0, w),
                        3 => (t, d, h, 0),
                        _ => unreachable!(),
                    };
                    if let Some(cur) = out.get_mut(ot, od, oh, ow) {
                        if v > *cur {
                            *cur = v;
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Connected components 4D
// ---------------------------------------------------------------------------

/// 6-connected neighbors in 4D (one step along exactly one axis).
fn neighbors_6_4d(
    t: usize,
    d: usize,
    h: usize,
    w: usize,
    shape: [usize; 4],
) -> Vec<(usize, usize, usize, usize)> {
    let [nt, nd, nh, nw] = shape;
    let mut result = Vec::with_capacity(8);
    if t > 0 {
        result.push((t - 1, d, h, w));
    }
    if t + 1 < nt {
        result.push((t + 1, d, h, w));
    }
    if d > 0 {
        result.push((t, d - 1, h, w));
    }
    if d + 1 < nd {
        result.push((t, d + 1, h, w));
    }
    if h > 0 {
        result.push((t, d, h - 1, w));
    }
    if h + 1 < nh {
        result.push((t, d, h + 1, w));
    }
    if w > 0 {
        result.push((t, d, h, w - 1));
    }
    if w + 1 < nw {
        result.push((t, d, h, w + 1));
    }
    result
}

/// 26-connected neighbors in 4D spatial part (±1 in D,H,W) and ±1 in T — up to 80 neighbors.
fn neighbors_26_4d(
    t: usize,
    d: usize,
    h: usize,
    w: usize,
    shape: [usize; 4],
) -> Vec<(usize, usize, usize, usize)> {
    let [nt, nd, nh, nw] = shape;
    let mut result = Vec::new();
    let ti_min = if t == 0 { 0isize } else { -1isize };
    let ti_max = if t + 1 < nt { 1isize } else { 0isize };
    let di_min = if d == 0 { 0isize } else { -1isize };
    let di_max = if d + 1 < nd { 1isize } else { 0isize };
    let hi_min = if h == 0 { 0isize } else { -1isize };
    let hi_max = if h + 1 < nh { 1isize } else { 0isize };
    let wi_min = if w == 0 { 0isize } else { -1isize };
    let wi_max = if w + 1 < nw { 1isize } else { 0isize };

    for dt in ti_min..=ti_max {
        for dd in di_min..=di_max {
            for dh in hi_min..=hi_max {
                for dw in wi_min..=wi_max {
                    if dt == 0 && dd == 0 && dh == 0 && dw == 0 {
                        continue;
                    }
                    result.push((
                        (t as isize + dt) as usize,
                        (d as isize + dd) as usize,
                        (h as isize + dh) as usize,
                        (w as isize + dw) as usize,
                    ));
                }
            }
        }
    }
    result
}

/// Compute connected components of a binary 4D array via BFS.
///
/// * `connectivity_26` — if true, use 26-connected spatial + temporal; else 6-connected.
///
/// Returns a `Label4D` where 0 = background, ≥1 = component labels.
pub fn connected_components_4d(binary: &Array4D<bool>, connectivity_26: bool) -> Label4D {
    let shape = binary.shape();
    let [nt, nd, nh, nw] = shape;
    let mut labels = Label4D::new(shape, 0usize);
    let mut current_label = 0usize;

    let flat_idx = |t: usize, d: usize, h: usize, w: usize| -> usize {
        t * (nd * nh * nw) + d * (nh * nw) + h * nw + w
    };

    let n_total = nt * nd * nh * nw;
    // Track visited via label array (0 = unvisited bg, or just check label != 0 for fg visited)
    let mut visited = vec![false; n_total];

    for t in 0..nt {
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let fi = flat_idx(t, d, h, w);
                    let is_fg = binary.get(t, d, h, w).copied().unwrap_or(false);
                    if !is_fg || visited[fi] {
                        continue;
                    }
                    // BFS from (t, d, h, w)
                    current_label += 1;
                    let lbl = current_label;
                    let mut queue = VecDeque::new();
                    queue.push_back((t, d, h, w));
                    visited[fi] = true;
                    let _ = labels.set(t, d, h, w, lbl);

                    while let Some((ct, cd, ch, cw)) = queue.pop_front() {
                        let neighbors = if connectivity_26 {
                            neighbors_26_4d(ct, cd, ch, cw, shape)
                        } else {
                            neighbors_6_4d(ct, cd, ch, cw, shape)
                        };
                        for (nt2, nd2, nh2, nw2) in neighbors {
                            let nfi = flat_idx(nt2, nd2, nh2, nw2);
                            if visited[nfi] {
                                continue;
                            }
                            let nfg = binary.get(nt2, nd2, nh2, nw2).copied().unwrap_or(false);
                            if nfg {
                                visited[nfi] = true;
                                let _ = labels.set(nt2, nd2, nh2, nw2, lbl);
                                queue.push_back((nt2, nd2, nh2, nw2));
                            }
                        }
                    }
                }
            }
        }
    }

    labels
}

// ---------------------------------------------------------------------------
// Region tracking
// ---------------------------------------------------------------------------

/// Result of tracking a labeled region across time frames.
#[derive(Debug, Clone)]
pub struct TrackletResult {
    /// Unique tracklet identifier.
    pub id: usize,
    /// Frame at which this tracklet begins.
    pub start_time: usize,
    /// List of frame indices (time steps) this tracklet spans.
    pub frames: Vec<usize>,
    /// Centroid [d, h, w] per frame in `frames`.
    pub centroid_per_frame: Vec<[f64; 3]>,
}

/// Compute the centroid [d, h, w] of a labeled region in a single time frame.
fn region_centroid(labeled: &Label4D, t: usize, label: usize) -> Option<[f64; 3]> {
    let [_nt, nd, nh, nw] = labeled.shape();
    let mut sum_d = 0.0f64;
    let mut sum_h = 0.0f64;
    let mut sum_w = 0.0f64;
    let mut count = 0usize;
    for d in 0..nd {
        for h in 0..nh {
            for w in 0..nw {
                if labeled.get(t, d, h, w).copied().unwrap_or(0) == label {
                    sum_d += d as f64;
                    sum_h += h as f64;
                    sum_w += w as f64;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        None
    } else {
        Some([
            sum_d / count as f64,
            sum_h / count as f64,
            sum_w / count as f64,
        ])
    }
}

/// Track labeled regions across time frames by majority overlap.
///
/// For each distinct label in frame t, find the label in frame t+1 with maximum
/// voxel overlap, creating/extending tracklets accordingly.
pub fn track_regions_4d(labeled: &Label4D) -> Vec<TrackletResult> {
    let [nt, nd, nh, nw] = labeled.shape();

    // Collect labels present in each frame
    let mut frame_labels: Vec<std::collections::HashSet<usize>> = Vec::with_capacity(nt);
    for t in 0..nt {
        let mut set = std::collections::HashSet::new();
        for d in 0..nd {
            for h in 0..nh {
                for w in 0..nw {
                    let lbl = labeled.get(t, d, h, w).copied().unwrap_or(0);
                    if lbl > 0 {
                        set.insert(lbl);
                    }
                }
            }
        }
        frame_labels.push(set);
    }

    // Map (frame, label) -> tracklet_id
    let mut assignment: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut tracklets: Vec<TrackletResult> = Vec::new();
    let mut next_id = 1usize;

    // Initialize frame 0
    for &lbl in &frame_labels[0] {
        let id = next_id;
        next_id += 1;
        assignment.insert((0, lbl), id);
        let centroid = region_centroid(labeled, 0, lbl).unwrap_or([0.0; 3]);
        tracklets.push(TrackletResult {
            id,
            start_time: 0,
            frames: vec![0],
            centroid_per_frame: vec![centroid],
        });
    }

    // Link across frames
    for t in 1..nt {
        // For each label in frame t, compute overlap with labels in frame t-1
        for &lbl_t in &frame_labels[t] {
            // Count overlapping voxels with each label in t-1
            let mut overlap: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for d in 0..nd {
                for h in 0..nh {
                    for w in 0..nw {
                        let cur = labeled.get(t, d, h, w).copied().unwrap_or(0);
                        let prev = labeled.get(t - 1, d, h, w).copied().unwrap_or(0);
                        if cur == lbl_t && prev > 0 {
                            *overlap.entry(prev).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Find best matching previous label
            let best_prev = overlap.iter().max_by_key(|(_, &cnt)| cnt).map(|(&k, _)| k);

            let centroid = region_centroid(labeled, t, lbl_t).unwrap_or([0.0; 3]);

            if let Some(prev_lbl) = best_prev {
                if let Some(&tid) = assignment.get(&(t - 1, prev_lbl)) {
                    // Extend existing tracklet
                    assignment.insert((t, lbl_t), tid);
                    if let Some(tk) = tracklets.iter_mut().find(|tk| tk.id == tid) {
                        tk.frames.push(t);
                        tk.centroid_per_frame.push(centroid);
                    }
                    continue;
                }
            }

            // No match found — new tracklet
            let id = next_id;
            next_id += 1;
            assignment.insert((t, lbl_t), id);
            tracklets.push(TrackletResult {
                id,
                start_time: t,
                frames: vec![t],
                centroid_per_frame: vec![centroid],
            });
        }
    }

    tracklets
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array4d_create_and_get_set() {
        let shape = [2, 3, 4, 5];
        let mut arr: Array4D<f64> = Array4D::new(shape, 0.0);
        assert_eq!(arr.shape(), shape);
        assert_eq!(arr.n_elements(), 2 * 3 * 4 * 5);

        // Set and get roundtrip
        arr.set(1, 2, 3, 4, 42.0).expect("set failed");
        assert_eq!(arr.get(1, 2, 3, 4).copied(), Some(42.0));
        // Out-of-bounds get returns None
        assert!(arr.get(2, 0, 0, 0).is_none());
    }

    #[test]
    fn test_from_data_shape_mismatch() {
        let data = vec![1.0f64; 10];
        let result = Array4D::from_data(data, [2, 3, 4, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_time() {
        let shape = [3, 2, 4, 5];
        let mut arr: Array4D<f64> = Array4D::new(shape, 0.0);
        arr.set(1, 1, 2, 3, 7.0).expect("set failed");
        let vol = arr.slice_time(1).expect("slice_time failed");
        assert_eq!(vol.len(), 2);
        assert_eq!(vol[1].len(), 4);
        assert_eq!(vol[1][2].len(), 5);
        assert!((vol[1][2][3] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_connected_components_4d_two_cubes() {
        // Two separate 2×2×2 cubes in a 4×4×4 spatial volume over 2 time steps
        // Cube 1: t=0, d=0..1, h=0..1, w=0..1
        // Cube 2: t=0, d=2..3, h=2..3, w=2..3
        // They are spatially separated by a gap so they should get separate labels.
        let shape = [1, 4, 4, 4];
        let mut binary: Array4D<bool> = Array4D::new(shape, false);
        // Cube 1
        for d in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    binary.set(0, d, h, w, true).expect("set failed");
                }
            }
        }
        // Cube 2
        for d in 2..4 {
            for h in 2..4 {
                for w in 2..4 {
                    binary.set(0, d, h, w, true).expect("set failed");
                }
            }
        }
        let labels = connected_components_4d(&binary, false);
        // Collect unique non-zero labels
        let mut unique: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for v in labels.data().iter() {
            if *v > 0 {
                unique.insert(*v);
            }
        }
        assert_eq!(unique.len(), 2, "Expected exactly 2 connected components");
    }

    #[test]
    fn test_diff_4d_temporal() {
        let shape = [3, 2, 2, 2];
        let mut arr: Array4D<f64> = Array4D::new(shape, 0.0);
        // Set time 0 to 1.0, time 1 to 3.0, time 2 to 6.0
        for d in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    arr.set(0, d, h, w, 1.0).unwrap();
                    arr.set(1, d, h, w, 3.0).unwrap();
                    arr.set(2, d, h, w, 6.0).unwrap();
                }
            }
        }
        let diff = diff_4d_temporal(&arr);
        assert_eq!(diff.shape()[0], 2);
        assert!((diff.get(0, 0, 0, 0).copied().unwrap() - 2.0).abs() < 1e-12);
        assert!((diff.get(1, 0, 0, 0).copied().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mip_4d() {
        let shape = [2, 2, 2, 2];
        let mut arr: Array4D<f64> = Array4D::new(shape, 0.0);
        arr.set(0, 0, 0, 0, 5.0).unwrap();
        arr.set(1, 0, 0, 0, 10.0).unwrap();
        // MIP along axis 0 (collapse time)
        let mip = max_intensity_projection_4d(&arr, 0).expect("mip failed");
        assert_eq!(mip.shape()[0], 1);
        assert!((mip.get(0, 0, 0, 0).copied().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_filter_4d_identity_sigma_zero() {
        let shape = [2, 2, 2, 2];
        let mut arr: Array4D<f64> = Array4D::new(shape, 1.0);
        arr.set(0, 0, 0, 0, 5.0).unwrap();
        // sigma=0 → kernel is [1.0], no change
        let out = gaussian_filter_4d(&arr, 0.0, 0.0);
        assert!((out.get(0, 0, 0, 0).copied().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_track_regions_4d() {
        // Simple 1-region tracklet across 2 time frames
        let shape = [2, 3, 3, 3];
        let mut binary: Array4D<bool> = Array4D::new(shape, false);
        // Same cube at t=0 and t=1
        for d in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    binary.set(0, d, h, w, true).unwrap();
                    binary.set(1, d, h, w, true).unwrap();
                }
            }
        }
        let labeled = connected_components_4d(&binary, false);
        let tracklets = track_regions_4d(&labeled);
        // Should have at least 1 tracklet spanning both frames
        assert!(!tracklets.is_empty());
        let multi_frame = tracklets.iter().any(|tk| tk.frames.len() >= 2);
        assert!(
            multi_frame,
            "Expected at least one tracklet spanning 2 frames"
        );
    }
}
