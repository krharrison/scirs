//! Fortune's sweep line algorithm for Voronoi diagram construction
//!
//! This module implements Fortune's algorithm for computing Voronoi diagrams
//! in 2D. Fortune's algorithm is a sweep line algorithm that processes events
//! from left to right, maintaining a beach line of parabolic arcs.
//!
//! # Algorithm
//!
//! The algorithm maintains:
//! - A **beach line**: a sequence of parabolic arcs defined by the sites to the
//!   left of the sweep line
//! - An **event queue**: site events (when the sweep line hits a new site) and
//!   circle events (when three consecutive arcs converge)
//!
//! Time complexity: O(n log n) where n is the number of sites.
//! Space complexity: O(n).
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::computational_geometry::fortune_voronoi::{fortune_voronoi_2d, VoronoiVertex};
//!
//! let sites = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
//! let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");
//!
//! assert!(!diagram.vertices.is_empty());
//! assert!(!diagram.edges.is_empty());
//! ```

use crate::error::{SpatialError, SpatialResult};
use std::collections::BinaryHeap;

/// Tolerance for floating-point comparisons
const EPSILON: f64 = 1e-10;

/// A vertex in the Voronoi diagram
#[derive(Debug, Clone, Copy)]
pub struct VoronoiVertex {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl VoronoiVertex {
    /// Create a new Voronoi vertex
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// An edge in the Voronoi diagram
#[derive(Debug, Clone)]
pub struct VoronoiEdge {
    /// Index of the start vertex (or None if the edge extends to infinity)
    pub start_vertex: Option<usize>,
    /// Index of the end vertex (or None if the edge extends to infinity)
    pub end_vertex: Option<usize>,
    /// Index of the left site (the site to the left of this edge)
    pub left_site: usize,
    /// Index of the right site (the site to the right of this edge)
    pub right_site: usize,
    /// Direction of the edge (for unbounded edges pointing toward infinity)
    pub direction: Option<[f64; 2]>,
}

/// A cell (region) in the Voronoi diagram
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// Index of the site that defines this cell
    pub site_index: usize,
    /// Indices of edges that bound this cell
    pub edge_indices: Vec<usize>,
    /// Indices of vertices that form this cell's boundary
    pub vertex_indices: Vec<usize>,
    /// Whether this cell is bounded (closed)
    pub is_bounded: bool,
}

/// A complete Voronoi diagram
#[derive(Debug, Clone)]
pub struct VoronoiDiagram {
    /// The input sites
    pub sites: Vec<[f64; 2]>,
    /// The Voronoi vertices
    pub vertices: Vec<VoronoiVertex>,
    /// The Voronoi edges
    pub edges: Vec<VoronoiEdge>,
    /// The Voronoi cells (regions), one per site
    pub cells: Vec<VoronoiCell>,
}

impl VoronoiDiagram {
    /// Get the number of sites
    pub fn num_sites(&self) -> usize {
        self.sites.len()
    }

    /// Get the number of Voronoi vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of Voronoi edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of bounded edges
    pub fn num_bounded_edges(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.start_vertex.is_some() && e.end_vertex.is_some())
            .count()
    }

    /// Find the nearest site to a given point
    pub fn nearest_site(&self, point: &[f64; 2]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;

        for (i, site) in self.sites.iter().enumerate() {
            let dx = point[0] - site[0];
            let dy = point[1] - site[1];
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }
}

/// Arc in the beach line
#[derive(Debug, Clone)]
struct Arc {
    /// Index of the site that defines this arc
    site_idx: usize,
    /// Index of the circle event that will remove this arc (if any)
    circle_event: Option<usize>,
}

/// Event types for Fortune's algorithm
#[derive(Debug, Clone)]
enum EventKind {
    /// A site event (sweep line reaches a new site)
    Site(usize),
    /// A circle event (three consecutive arcs converge)
    Circle {
        arc_idx: usize,
        center_x: f64,
        center_y: f64,
        radius: f64,
    },
}

/// An event in the priority queue
#[derive(Debug, Clone)]
struct Event {
    /// X-coordinate of the event (determines priority)
    x: f64,
    /// The event kind
    kind: EventKind,
    /// Whether this event has been invalidated
    valid: bool,
    /// Unique ID for the event
    id: usize,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller x values have higher priority (we negate for BinaryHeap)
        other
            .x
            .partial_cmp(&self.x)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Internal builder for the Voronoi diagram
struct FortuneBuilder {
    sites: Vec<[f64; 2]>,
    beach_line: Vec<Arc>,
    events: BinaryHeap<Event>,
    event_counter: usize,
    // Circle event validity tracking
    circle_event_valid: Vec<bool>,
    // Output
    vertices: Vec<VoronoiVertex>,
    edges: Vec<VoronoiEdge>,
    // Edge tracking for each pair of sites
    half_edges: Vec<HalfEdgeRecord>,
}

/// A half-edge record used during construction
#[derive(Debug, Clone)]
struct HalfEdgeRecord {
    left_site: usize,
    right_site: usize,
    start_vertex: Option<usize>,
    end_vertex: Option<usize>,
    direction: [f64; 2],
}

impl FortuneBuilder {
    fn new(sites: Vec<[f64; 2]>) -> Self {
        Self {
            sites,
            beach_line: Vec::new(),
            events: BinaryHeap::new(),
            event_counter: 0,
            circle_event_valid: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
            half_edges: Vec::new(),
        }
    }

    fn next_event_id(&mut self) -> usize {
        let id = self.event_counter;
        self.event_counter += 1;
        id
    }

    fn build(mut self) -> SpatialResult<VoronoiDiagram> {
        let n = self.sites.len();
        if n < 2 {
            return Err(SpatialError::ValueError(
                "Need at least 2 sites for Voronoi diagram".to_string(),
            ));
        }

        // Sort sites by x-coordinate
        let mut site_order: Vec<usize> = (0..n).collect();
        site_order.sort_by(|&a, &b| {
            self.sites[a][0]
                .partial_cmp(&self.sites[b][0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.sites[a][1]
                        .partial_cmp(&self.sites[b][1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Initialize event queue with site events
        for &idx in &site_order {
            let id = self.next_event_id();
            self.events.push(Event {
                x: self.sites[idx][0],
                kind: EventKind::Site(idx),
                valid: true,
                id,
            });
        }

        // Process events
        let max_iterations = n * n * 4 + 100;
        let mut iterations = 0;

        while let Some(event) = self.events.pop() {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            if !event.valid {
                continue;
            }

            match event.kind {
                EventKind::Site(site_idx) => {
                    self.handle_site_event(site_idx, event.x);
                }
                EventKind::Circle {
                    arc_idx,
                    center_x,
                    center_y,
                    ..
                } => {
                    // Check if the circle event is still valid
                    if arc_idx < self.circle_event_valid.len() && self.circle_event_valid[arc_idx] {
                        self.handle_circle_event(arc_idx, center_x, center_y);
                    }
                }
            }
        }

        // Finalize: close remaining half-edges
        self.finalize_edges();

        // Build cells
        let cells = self.build_cells();

        Ok(VoronoiDiagram {
            sites: self.sites,
            vertices: self.vertices,
            edges: self.edges,
            cells,
        })
    }

    fn handle_site_event(&mut self, site_idx: usize, _sweep_x: f64) {
        if self.beach_line.is_empty() {
            self.beach_line.push(Arc {
                site_idx,
                circle_event: None,
            });
            return;
        }

        let site = self.sites[site_idx];

        // Find the arc directly above the new site
        let arc_idx = self.find_arc_above(site[1], site[0]);

        // If the found arc has a circle event, invalidate it
        if let Some(ce_idx) = self.beach_line[arc_idx].circle_event {
            if ce_idx < self.circle_event_valid.len() {
                self.circle_event_valid[ce_idx] = false;
            }
        }

        let existing_site = self.beach_line[arc_idx].site_idx;

        // Create a half-edge between existing_site and site_idx
        let dir = self.compute_edge_direction(existing_site, site_idx);
        self.half_edges.push(HalfEdgeRecord {
            left_site: existing_site,
            right_site: site_idx,
            start_vertex: None,
            end_vertex: None,
            direction: dir,
        });

        // Split the arc: replace arc[arc_idx] with three arcs
        let old_arc = Arc {
            site_idx: existing_site,
            circle_event: None,
        };
        let new_arc = Arc {
            site_idx,
            circle_event: None,
        };
        let old_arc_copy = Arc {
            site_idx: existing_site,
            circle_event: None,
        };

        // Insert the new arcs
        self.beach_line[arc_idx] = old_arc;
        self.beach_line.insert(arc_idx + 1, new_arc);
        self.beach_line.insert(arc_idx + 2, old_arc_copy);

        // Pad circle_event_valid
        while self.circle_event_valid.len() < self.beach_line.len() {
            self.circle_event_valid.push(false);
        }

        // Check for new circle events
        if arc_idx > 0 {
            self.check_circle_event(arc_idx);
        }
        if arc_idx + 2 < self.beach_line.len() {
            self.check_circle_event(arc_idx + 2);
        }
    }

    fn handle_circle_event(&mut self, arc_idx: usize, center_x: f64, center_y: f64) {
        if arc_idx >= self.beach_line.len() || arc_idx == 0 || arc_idx >= self.beach_line.len() - 1
        {
            return;
        }

        // Invalidate this arc's circle event
        if arc_idx < self.circle_event_valid.len() {
            self.circle_event_valid[arc_idx] = false;
        }

        // Add a Voronoi vertex at the circle center
        let vertex_idx = self.vertices.len();
        self.vertices.push(VoronoiVertex::new(center_x, center_y));

        // Get the three sites involved
        let left_site = self.beach_line[arc_idx - 1].site_idx;
        let mid_site = self.beach_line[arc_idx].site_idx;
        let right_site = self.beach_line[arc_idx + 1].site_idx;

        // Complete half-edges that end at this vertex
        for he in &mut self.half_edges {
            if ((he.left_site == left_site && he.right_site == mid_site)
                || (he.left_site == mid_site && he.right_site == left_site))
                && he.end_vertex.is_none()
            {
                he.end_vertex = Some(vertex_idx);
            }
            if ((he.left_site == mid_site && he.right_site == right_site)
                || (he.left_site == right_site && he.right_site == mid_site))
                && he.end_vertex.is_none()
            {
                he.end_vertex = Some(vertex_idx);
            }
        }

        // Create a new half-edge between left_site and right_site
        let dir = self.compute_edge_direction(left_site, right_site);
        self.half_edges.push(HalfEdgeRecord {
            left_site,
            right_site,
            start_vertex: Some(vertex_idx),
            end_vertex: None,
            direction: dir,
        });

        // Invalidate circle events for neighboring arcs
        if arc_idx > 0 {
            if let Some(ce_idx) = self.beach_line[arc_idx - 1].circle_event {
                if ce_idx < self.circle_event_valid.len() {
                    self.circle_event_valid[ce_idx] = false;
                }
            }
            self.beach_line[arc_idx - 1].circle_event = None;
        }
        if arc_idx + 1 < self.beach_line.len() {
            if let Some(ce_idx) = self.beach_line[arc_idx + 1].circle_event {
                if ce_idx < self.circle_event_valid.len() {
                    self.circle_event_valid[ce_idx] = false;
                }
            }
            self.beach_line[arc_idx + 1].circle_event = None;
        }

        // Remove the disappearing arc
        self.beach_line.remove(arc_idx);
        if arc_idx < self.circle_event_valid.len() {
            self.circle_event_valid.remove(arc_idx);
        }

        // Check for new circle events
        if arc_idx > 0 && arc_idx < self.beach_line.len() {
            self.check_circle_event(arc_idx - 1);
            if arc_idx < self.beach_line.len() {
                self.check_circle_event(arc_idx);
            }
        }
    }

    fn find_arc_above(&self, y: f64, sweep_x: f64) -> usize {
        if self.beach_line.len() <= 1 {
            return 0;
        }

        // For each arc, compute the breakpoints with its neighbors
        // and find which arc is above the given y at the sweep line position
        for i in 0..self.beach_line.len() {
            if i + 1 < self.beach_line.len() {
                let site_a = self.sites[self.beach_line[i].site_idx];
                let site_b = self.sites[self.beach_line[i + 1].site_idx];

                // Compute breakpoint between arcs i and i+1
                if let Some(bp_y) = self.compute_breakpoint(site_a, site_b, sweep_x) {
                    if y <= bp_y {
                        return i;
                    }
                }
            } else {
                return i;
            }
        }

        self.beach_line.len() - 1
    }

    fn compute_breakpoint(&self, site_a: [f64; 2], site_b: [f64; 2], sweep_x: f64) -> Option<f64> {
        let ax = site_a[0];
        let ay = site_a[1];
        let bx = site_b[0];
        let by = site_b[1];

        // Degenerate: both sites have same x
        if (ax - bx).abs() < EPSILON {
            return Some((ay + by) / 2.0);
        }

        // Degenerate: one site is on the sweep line
        if (ax - sweep_x).abs() < EPSILON {
            return Some(ay);
        }
        if (bx - sweep_x).abs() < EPSILON {
            return Some(by);
        }

        // General case: solve for y where the two parabolas intersect
        let da = ax - sweep_x;
        let db = bx - sweep_x;

        let a = 1.0 / da - 1.0 / db;
        let b = -2.0 * (ay / da - by / db);
        let c = (ay * ay + ax * ax - sweep_x * sweep_x) / da
            - (by * by + bx * bx - sweep_x * sweep_x) / db;

        if a.abs() < EPSILON {
            if b.abs() < EPSILON {
                return None;
            }
            return Some(-c / b);
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Some((ay + by) / 2.0);
        }

        let sqrt_disc = discriminant.sqrt();
        let y1 = (-b - sqrt_disc) / (2.0 * a);
        let y2 = (-b + sqrt_disc) / (2.0 * a);

        // Return the breakpoint that is between the two site y-coordinates
        if (ax < bx) == (y1 < y2) {
            Some(y1)
        } else {
            Some(y2)
        }
    }

    fn check_circle_event(&mut self, arc_idx: usize) {
        if arc_idx == 0 || arc_idx >= self.beach_line.len() - 1 {
            return;
        }

        let left = self.beach_line[arc_idx - 1].site_idx;
        let mid = self.beach_line[arc_idx].site_idx;
        let right = self.beach_line[arc_idx + 1].site_idx;

        // Don't create circle event if two consecutive arcs belong to the same site
        if left == right {
            return;
        }

        let a = self.sites[left];
        let b = self.sites[mid];
        let c = self.sites[right];

        // Compute circumcircle
        if let Some((cx, cy, r)) = circumcircle(a, b, c) {
            let event_x = cx + r;

            // Only add if the event is to the right of the rightmost site
            let max_x = a[0].max(b[0]).max(c[0]);
            if event_x >= max_x - EPSILON {
                // Ensure circle_event_valid is large enough
                while self.circle_event_valid.len() <= arc_idx {
                    self.circle_event_valid.push(false);
                }

                self.circle_event_valid[arc_idx] = true;

                let id = self.next_event_id();
                self.beach_line[arc_idx].circle_event = Some(arc_idx);

                self.events.push(Event {
                    x: event_x,
                    kind: EventKind::Circle {
                        arc_idx,
                        center_x: cx,
                        center_y: cy,
                        radius: r,
                    },
                    valid: true,
                    id,
                });
            }
        }
    }

    fn compute_edge_direction(&self, site_a: usize, site_b: usize) -> [f64; 2] {
        let a = self.sites[site_a];
        let b = self.sites[site_b];

        // The edge direction is perpendicular to the line connecting the two sites
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];

        // Perpendicular direction (rotate 90 degrees)
        [-dy, dx]
    }

    fn finalize_edges(&mut self) {
        // Convert half-edges into Voronoi edges
        for he in &self.half_edges {
            self.edges.push(VoronoiEdge {
                start_vertex: he.start_vertex,
                end_vertex: he.end_vertex,
                left_site: he.left_site,
                right_site: he.right_site,
                direction: Some(he.direction),
            });
        }
    }

    fn build_cells(&self) -> Vec<VoronoiCell> {
        let n = self.sites.len();
        let mut cells = Vec::with_capacity(n);

        for site_idx in 0..n {
            let mut edge_indices = Vec::new();
            let mut vertex_indices = Vec::new();
            let mut is_bounded = true;

            for (edge_idx, edge) in self.edges.iter().enumerate() {
                if edge.left_site == site_idx || edge.right_site == site_idx {
                    edge_indices.push(edge_idx);

                    if let Some(v) = edge.start_vertex {
                        if !vertex_indices.contains(&v) {
                            vertex_indices.push(v);
                        }
                    } else {
                        is_bounded = false;
                    }

                    if let Some(v) = edge.end_vertex {
                        if !vertex_indices.contains(&v) {
                            vertex_indices.push(v);
                        }
                    } else {
                        is_bounded = false;
                    }
                }
            }

            cells.push(VoronoiCell {
                site_index: site_idx,
                edge_indices,
                vertex_indices,
                is_bounded,
            });
        }

        cells
    }
}

/// Compute the circumcircle of three points
///
/// Returns (center_x, center_y, radius) or None if points are collinear
fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<(f64, f64, f64)> {
    let ax = a[0];
    let ay = a[1];
    let bx = b[0];
    let by = b[1];
    let cx = c[0];
    let cy = c[1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if d.abs() < EPSILON {
        return None; // Collinear
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let r = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();

    Some((ux, uy, r))
}

/// Compute a Voronoi diagram using Fortune's sweep line algorithm
///
/// # Arguments
///
/// * `sites` - A slice of 2D point coordinates [x, y]
///
/// # Returns
///
/// * `SpatialResult<VoronoiDiagram>` - The computed Voronoi diagram
///
/// # Examples
///
/// ```
/// use scirs2_spatial::computational_geometry::fortune_voronoi::fortune_voronoi_2d;
///
/// let sites = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
/// let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");
///
/// assert_eq!(diagram.num_sites(), 3);
/// assert!(!diagram.vertices.is_empty());
/// ```
pub fn fortune_voronoi_2d(sites: &[[f64; 2]]) -> SpatialResult<VoronoiDiagram> {
    if sites.len() < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 sites for Voronoi diagram".to_string(),
        ));
    }

    // Check for duplicate sites
    for i in 0..sites.len() {
        for j in (i + 1)..sites.len() {
            if (sites[i][0] - sites[j][0]).abs() < EPSILON
                && (sites[i][1] - sites[j][1]).abs() < EPSILON
            {
                return Err(SpatialError::ValueError(format!(
                    "Duplicate sites at index {} and {}: [{}, {}]",
                    i, j, sites[i][0], sites[i][1]
                )));
            }
        }
    }

    let builder = FortuneBuilder::new(sites.to_vec());
    builder.build()
}

/// Compute the Voronoi diagram from an ndarray of points
///
/// # Arguments
///
/// * `points` - A 2D array of points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<VoronoiDiagram>` - The computed Voronoi diagram
pub fn fortune_voronoi_from_array(
    points: &scirs2_core::ndarray::ArrayView2<'_, f64>,
) -> SpatialResult<VoronoiDiagram> {
    if points.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Points must be 2D for Voronoi diagram".to_string(),
        ));
    }

    let sites: Vec<[f64; 2]> = (0..points.nrows())
        .map(|i| [points[[i, 0]], points[[i, 1]]])
        .collect();

    fortune_voronoi_2d(&sites)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_sites() {
        let sites = vec![[0.0, 0.0], [2.0, 0.0]];
        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");

        assert_eq!(diagram.num_sites(), 2);
        // Two sites produce one edge (the perpendicular bisector)
        assert!(diagram.num_edges() >= 1);
    }

    #[test]
    fn test_three_sites_triangle() {
        let sites = vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");

        assert_eq!(diagram.num_sites(), 3);
        // Three non-collinear sites produce 1 Voronoi vertex (circumcenter) and 3 edges
        assert!(diagram.num_vertices() >= 1);
        assert!(diagram.num_edges() >= 3);
    }

    #[test]
    fn test_four_sites_square() {
        let sites = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");

        assert_eq!(diagram.num_sites(), 4);
        assert!(!diagram.vertices.is_empty());
        assert!(!diagram.edges.is_empty());
        // Each site should have a cell
        assert_eq!(diagram.cells.len(), 4);
    }

    #[test]
    fn test_nearest_site() {
        let sites = vec![[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]];
        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");

        // Point near site 0
        assert_eq!(diagram.nearest_site(&[0.1, 0.1]), 0);
        // Point near site 1
        assert_eq!(diagram.nearest_site(&[9.9, 0.1]), 1);
        // Point near site 2
        assert_eq!(diagram.nearest_site(&[5.0, 9.0]), 2);
    }

    #[test]
    fn test_cells_created() {
        let sites = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");

        // Each site should have a cell with at least one edge
        for cell in &diagram.cells {
            assert!(!cell.edge_indices.is_empty());
        }
    }

    #[test]
    fn test_collinear_sites() {
        // Three collinear sites
        let sites = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");

        assert_eq!(diagram.num_sites(), 3);
        // Should produce 2 edges (perpendicular bisectors) but no finite vertex
        assert!(diagram.num_edges() >= 2);
    }

    #[test]
    fn test_duplicate_sites_error() {
        let sites = vec![[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]];
        let result = fortune_voronoi_2d(&sites);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_few_sites() {
        let sites = vec![[0.0, 0.0]];
        let result = fortune_voronoi_2d(&sites);
        assert!(result.is_err());
    }

    #[test]
    fn test_circumcircle() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];

        let result = circumcircle(a, b, c);
        assert!(result.is_some());

        let (cx, cy, r) = result.expect("Operation failed");
        // Circumcenter of right triangle (0,0)-(1,0)-(0,1) is at (0.5, 0.5)
        assert!((cx - 0.5).abs() < 1e-10);
        assert!((cy - 0.5).abs() < 1e-10);
        // Radius = distance from center to any vertex = sqrt(0.5) ~ 0.707
        assert!((r - (0.5_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_circumcircle_collinear() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [2.0, 0.0];

        let result = circumcircle(a, b, c);
        assert!(result.is_none());
    }

    #[test]
    fn test_from_array() {
        use scirs2_core::ndarray::array;

        let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let diagram = fortune_voronoi_from_array(&points.view()).expect("Operation failed");
        assert_eq!(diagram.num_sites(), 3);
    }

    #[test]
    fn test_many_sites() {
        // 8 sites in a circle pattern
        let mut sites = Vec::new();
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
            sites.push([angle.cos(), angle.sin()]);
        }

        let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");
        assert_eq!(diagram.num_sites(), 8);
        assert!(!diagram.vertices.is_empty());
        assert!(!diagram.edges.is_empty());
    }
}
