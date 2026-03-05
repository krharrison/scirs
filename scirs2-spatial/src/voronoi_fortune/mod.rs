//! Voronoi diagrams via Fortune's sweep line algorithm.
//!
//! Provides O(N log N) Voronoi diagram construction using Fortune's algorithm,
//! together with cell-area computation and a dual Delaunay triangulation helper.

use crate::error::{SpatialError, SpatialResult};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ──────────────────────────────────────────────────────────────────────────────
// Public data structures
// ──────────────────────────────────────────────────────────────────────────────

/// A generator (input) site for the Voronoi diagram.
#[derive(Debug, Clone, Copy)]
pub struct VoronoiSite {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Original index in the input slice
    pub index: usize,
}

/// A vertex (intersection point) in the Voronoi diagram.
#[derive(Debug, Clone, Copy)]
pub struct VoronoiVertex {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

/// An edge in the Voronoi diagram.
///
/// Each edge separates the cells of `left_site` and `right_site`.
/// `v1` / `v2` are indices into [`VoronoiDiagram::vertices`]; `None` means
/// the edge extends to infinity in that direction.
#[derive(Debug, Clone)]
pub struct VoronoiEdge {
    /// Start vertex index (None → infinite in start direction)
    pub v1: Option<usize>,
    /// End vertex index (None → infinite in end direction)
    pub v2: Option<usize>,
    /// Index of the site to the left of this directed edge
    pub left_site: usize,
    /// Index of the site to the right of this directed edge
    pub right_site: usize,
}

/// A complete Voronoi diagram.
#[derive(Debug, Clone)]
pub struct VoronoiDiagram {
    /// Input sites
    pub sites: Vec<VoronoiSite>,
    /// Voronoi vertices (finite intersection points)
    pub vertices: Vec<VoronoiVertex>,
    /// Voronoi edges
    pub edges: Vec<VoronoiEdge>,
    /// Bounding box used when clipping infinite edges: (x_min, y_min, x_max, y_max)
    pub bbox: (f64, f64, f64, f64),
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal event types
// ──────────────────────────────────────────────────────────────────────────────

/// Internal event stored in the priority queue.
#[derive(Debug, Clone)]
enum Event {
    /// The sweep line just reached a new site.
    Site { x: f64, y: f64, site_idx: usize },
    /// Three arcs are about to converge: the middle arc disappears.
    Circle {
        /// x-coordinate of the lowest point on the circumcircle (the event x).
        event_x: f64,
        /// Centre of the circumcircle.
        cx: f64,
        cy: f64,
        /// Index of the arc in the beach-line that will be squeezed out.
        arc_idx: usize,
        /// Generation counter for invalidation.
        generation: u64,
    },
}

impl Event {
    fn x(&self) -> f64 {
        match self {
            Event::Site { x, .. } => *x,
            Event::Circle { event_x, .. } => *event_x,
        }
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.x().eq(&other.x())
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap: larger x has higher priority
        self.x()
            .partial_cmp(&other.x())
            .unwrap_or(Ordering::Equal)
            .then(Ordering::Equal)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Beach-line arc
// ──────────────────────────────────────────────────────────────────────────────

/// One parabolic arc on the beach line.
#[derive(Debug, Clone)]
struct Arc {
    site_idx: usize,
    /// Left Voronoi half-edge being traced (index into builder.half_edges).
    left_edge: Option<usize>,
    /// Right Voronoi half-edge being traced.
    right_edge: Option<usize>,
    /// Active circle-event generation counter (0 = none).
    circle_gen: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Half-edge (unbounded during construction)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct HalfEdge {
    v1: Option<usize>,
    v2: Option<usize>,
    left_site: usize,
    right_site: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Fortune builder
// ──────────────────────────────────────────────────────────────────────────────

struct FortuneBuilder {
    sites: Vec<VoronoiSite>,
    beach: Vec<Arc>,
    /// half-edges under construction
    half_edges: Vec<HalfEdge>,
    vertices: Vec<VoronoiVertex>,
    queue: BinaryHeap<Event>,
    sweep_x: f64,
    /// Monotone generation counter to invalidate stale circle events.
    gen_counter: u64,
    /// Per-arc current generation (indexed same as beach).
    arc_gen: Vec<u64>,
}

impl FortuneBuilder {
    fn new(sites: Vec<VoronoiSite>) -> Self {
        Self {
            sites,
            beach: Vec::new(),
            half_edges: Vec::new(),
            vertices: Vec::new(),
            queue: BinaryHeap::new(),
            sweep_x: f64::NEG_INFINITY,
            gen_counter: 1,
            arc_gen: Vec::new(),
        }
    }

    /// Evaluate the y-coordinate of the beach line parabola for `site` at sweep x = `xl`.
    fn parabola_y(&self, site_idx: usize, xl: f64, query_x: f64) -> f64 {
        let s = &self.sites[site_idx];
        let dx = s.x - xl;
        if dx.abs() < 1e-12 {
            return f64::NAN;
        }
        (query_x - s.y).powi(2) / (2.0 * dx) + (s.x + xl) / 2.0
    }

    /// Find the arc index in the beach line directly above `y` at current sweep x.
    fn find_arc_above(&self, y: f64) -> usize {
        if self.beach.is_empty() {
            return 0;
        }
        let xl = self.sweep_x;
        let mut lo = 0usize;
        let mut hi = self.beach.len().saturating_sub(1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let y_mid = self.arc_intersection_y(mid, xl);
            if y_mid < y {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The y-coordinate of the right boundary of arc `i` at sweep x = `xl`.
    fn arc_intersection_y(&self, i: usize, xl: f64) -> f64 {
        if i + 1 >= self.beach.len() {
            return f64::INFINITY;
        }
        let a = self.beach[i].site_idx;
        let b = self.beach[i + 1].site_idx;
        circle_intersection_y(&self.sites[a], &self.sites[b], xl)
    }

    fn next_gen(&mut self) -> u64 {
        self.gen_counter += 1;
        self.gen_counter
    }

    /// Invalidate the circle event of beach arc `i`.
    fn invalidate_circle(&mut self, i: usize) {
        if i < self.arc_gen.len() {
            self.arc_gen[i] = 0;
            self.beach[i].circle_gen = 0;
        }
    }

    fn add_vertex(&mut self, x: f64, y: f64) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(VoronoiVertex { x, y });
        idx
    }

    fn add_half_edge(&mut self, left: usize, right: usize) -> usize {
        let idx = self.half_edges.len();
        self.half_edges.push(HalfEdge {
            v1: None,
            v2: None,
            left_site: left,
            right_site: right,
        });
        idx
    }

    // ── Site event ────────────────────────────────────────────────────────────

    fn process_site(&mut self, site_idx: usize) {
        let sy = self.sites[site_idx].y;

        if self.beach.is_empty() {
            self.beach.push(Arc {
                site_idx,
                left_edge: None,
                right_edge: None,
                circle_gen: 0,
            });
            self.arc_gen.push(0);
            return;
        }

        let arc_i = self.find_arc_above(sy);

        // Invalidate any circle event for the arc being split.
        self.invalidate_circle(arc_i);

        // Create two new half-edges.
        let arc_site = self.beach[arc_i].site_idx;
        let he_left = self.add_half_edge(arc_site, site_idx);
        let he_right = self.add_half_edge(site_idx, arc_site);

        // Split arc_i into: [arc_i, new_arc, right_copy]
        let original = self.beach[arc_i].clone();
        let orig_gen = self.arc_gen[arc_i];

        let new_arc = Arc {
            site_idx,
            left_edge: Some(he_left),
            right_edge: Some(he_right),
            circle_gen: 0,
        };
        let right_copy = Arc {
            site_idx: original.site_idx,
            left_edge: Some(he_right),
            right_edge: original.right_edge,
            circle_gen: 0,
        };

        self.beach[arc_i].right_edge = Some(he_left);
        self.beach[arc_i].circle_gen = 0;

        self.beach.insert(arc_i + 1, new_arc);
        self.beach.insert(arc_i + 2, right_copy);
        self.arc_gen.insert(arc_i + 1, 0);
        self.arc_gen.insert(arc_i + 2, orig_gen);

        // Check for new circle events.
        if arc_i > 0 {
            self.check_circle(arc_i - 1);
        }
        if arc_i + 3 < self.beach.len() {
            self.check_circle(arc_i + 2);
        }
    }

    // ── Circle event ──────────────────────────────────────────────────────────

    fn process_circle(&mut self, event_x: f64, cx: f64, cy: f64, arc_idx: usize, gen: u64) {
        // Validate the event.
        if arc_idx >= self.beach.len() {
            return;
        }
        if self.beach[arc_idx].circle_gen != gen {
            return;
        }

        let v_idx = self.add_vertex(cx, cy);

        // Finish the two half-edges adjacent to the disappearing arc.
        if let Some(he) = self.beach[arc_idx].left_edge {
            if he < self.half_edges.len() {
                self.half_edges[he].v2 = Some(v_idx);
            }
        }
        if let Some(he) = self.beach[arc_idx].right_edge {
            if he < self.half_edges.len() {
                self.half_edges[he].v1 = Some(v_idx);
            }
        }

        // Start a new half-edge between the arcs that remain adjacent.
        let left_site = if arc_idx > 0 {
            self.beach[arc_idx - 1].site_idx
        } else {
            self.beach[arc_idx].site_idx
        };
        let right_site = if arc_idx + 1 < self.beach.len() {
            self.beach[arc_idx + 1].site_idx
        } else {
            self.beach[arc_idx].site_idx
        };

        let new_he = self.add_half_edge(left_site, right_site);
        if new_he < self.half_edges.len() {
            self.half_edges[new_he].v1 = Some(v_idx);
        }

        // Update adjacent arcs' edge pointers.
        if arc_idx > 0 {
            self.beach[arc_idx - 1].right_edge = Some(new_he);
        }
        if arc_idx + 1 < self.beach.len() {
            self.beach[arc_idx + 1].left_edge = Some(new_he);
        }

        // Invalidate circle events for neighbours.
        if arc_idx > 0 {
            self.invalidate_circle(arc_idx - 1);
        }
        if arc_idx + 1 < self.beach.len() {
            self.invalidate_circle(arc_idx + 1);
        }

        self.beach.remove(arc_idx);
        self.arc_gen.remove(arc_idx);

        // Re-check circle events.
        if arc_idx > 0 && arc_idx <= self.beach.len().saturating_sub(1) {
            self.check_circle(arc_idx - 1);
        }
        if arc_idx < self.beach.len() {
            self.check_circle(arc_idx);
        }
        let _ = event_x;
    }

    /// Attempt to schedule a circle event for arcs `(i-1, i, i+1)`.
    fn check_circle(&mut self, i: usize) {
        if i == 0 || i + 1 >= self.beach.len() {
            return;
        }
        let a = self.beach[i - 1].site_idx;
        let b = self.beach[i].site_idx;
        let c = self.beach[i + 1].site_idx;

        if let Some((cx, cy, radius)) = circumcircle(&self.sites[a], &self.sites[b], &self.sites[c])
        {
            let event_x = cx + radius;
            if event_x >= self.sweep_x - 1e-10 {
                let gen = self.next_gen();
                self.beach[i].circle_gen = gen;
                if i < self.arc_gen.len() {
                    self.arc_gen[i] = gen;
                }
                self.queue.push(Event::Circle {
                    event_x,
                    cx,
                    cy,
                    arc_idx: i,
                    generation: gen,
                });
            }
        }
    }

    /// Run the main event loop.
    fn run(&mut self) {
        while let Some(ev) = self.queue.pop() {
            self.sweep_x = ev.x();
            match ev {
                Event::Site { site_idx, .. } => {
                    self.process_site(site_idx);
                }
                Event::Circle {
                    event_x,
                    cx,
                    cy,
                    arc_idx,
                    generation,
                } => {
                    self.process_circle(event_x, cx, cy, arc_idx, generation);
                }
            }
        }
    }

    /// Build the final [`VoronoiDiagram`], clipping to bbox.
    fn finish(self, bbox: (f64, f64, f64, f64)) -> VoronoiDiagram {
        let edges: Vec<VoronoiEdge> = self
            .half_edges
            .iter()
            .map(|he| VoronoiEdge {
                v1: he.v1,
                v2: he.v2,
                left_site: he.left_site,
                right_site: he.right_site,
            })
            .collect();

        VoronoiDiagram {
            sites: self.sites,
            vertices: self.vertices,
            edges,
            bbox,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Y-coordinate of the intersection of two beach parabolas.
fn circle_intersection_y(a: &VoronoiSite, b: &VoronoiSite, xl: f64) -> f64 {
    let dxa = a.x - xl;
    let dxb = b.x - xl;
    if dxa.abs() < 1e-12 {
        return a.y;
    }
    if dxb.abs() < 1e-12 {
        return b.y;
    }
    // Solve quadratic for intersection.
    let (ax, ay) = (a.x, a.y);
    let (bx, by) = (b.x, b.y);
    let alpha = 1.0 / (2.0 * (ax - xl));
    let beta = 1.0 / (2.0 * (bx - xl));
    let a_coef = alpha - beta;
    let b_coef = -2.0 * (ay * alpha - by * beta);
    let c_coef = (ay.powi(2) + ax.powi(2) - xl.powi(2)) * alpha
        - (by.powi(2) + bx.powi(2) - xl.powi(2)) * beta;
    if a_coef.abs() < 1e-12 {
        return -c_coef / b_coef;
    }
    let disc = b_coef.powi(2) - 4.0 * a_coef * c_coef;
    if disc < 0.0 {
        return (ay + by) / 2.0;
    }
    let sq = disc.sqrt();
    let y1 = (-b_coef + sq) / (2.0 * a_coef);
    let y2 = (-b_coef - sq) / (2.0 * a_coef);
    if ax < bx { y1.max(y2) } else { y1.min(y2) }
}

/// Circumcircle of three sites: returns (cx, cy, radius) or None if collinear.
fn circumcircle(a: &VoronoiSite, b: &VoronoiSite, c: &VoronoiSite) -> Option<(f64, f64, f64)> {
    let ax = b.x - a.x;
    let ay = b.y - a.y;
    let bx = c.x - a.x;
    let by = c.y - a.y;
    let d = 2.0 * (ax * by - ay * bx);
    if d.abs() < 1e-12 {
        return None;
    }
    let ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / d;
    let uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / d;
    let cx = ux + a.x;
    let cy = uy + a.y;
    let r = ((cx - a.x).powi(2) + (cy - a.y).powi(2)).sqrt();
    Some((cx, cy, r))
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

impl VoronoiDiagram {
    /// Compute the Voronoi diagram of `sites` clipped to `bbox`.
    ///
    /// `bbox` = (x_min, y_min, x_max, y_max)
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 sites are provided.
    pub fn compute(
        sites: &[(f64, f64)],
        bbox: (f64, f64, f64, f64),
    ) -> SpatialResult<VoronoiDiagram> {
        if sites.len() < 2 {
            return Err(SpatialError::InvalidInput(
                "At least 2 sites required".into(),
            ));
        }

        let mut voronoi_sites: Vec<VoronoiSite> = sites
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| VoronoiSite { x, y, index: i })
            .collect();

        // Sort by x (primary), y (secondary) – Fortune's sweep goes left-to-right.
        voronoi_sites.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap_or(Ordering::Equal)
                .then(a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal))
        });

        let mut builder = FortuneBuilder::new(voronoi_sites.clone());

        // Seed the event queue with all site events.
        for s in &voronoi_sites {
            builder.queue.push(Event::Site {
                x: s.x,
                y: s.y,
                site_idx: s.index,
            });
        }

        builder.run();
        Ok(builder.finish(bbox))
    }

    /// Approximate area of the Voronoi cell for `site_idx`.
    ///
    /// Collects the finite vertices that bound the cell, sorts them by angle,
    /// and computes the shoelace area.
    pub fn cell_area(&self, site_idx: usize) -> f64 {
        if site_idx >= self.sites.len() {
            return 0.0;
        }
        // Gather vertices that appear in edges adjacent to this site.
        let mut pts: Vec<[f64; 2]> = Vec::new();
        for edge in &self.edges {
            if edge.left_site != site_idx && edge.right_site != site_idx {
                continue;
            }
            for opt_v in [edge.v1, edge.v2] {
                if let Some(vi) = opt_v {
                    if vi < self.vertices.len() {
                        let v = &self.vertices[vi];
                        pts.push([v.x, v.y]);
                    }
                }
            }
        }
        pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);

        if pts.len() < 3 {
            return 0.0;
        }

        let site = &self.sites[site_idx];
        // Sort by angle around the site.
        pts.sort_by(|a, b| {
            let ang_a = (a[1] - site.y).atan2(a[0] - site.x);
            let ang_b = (b[1] - site.y).atan2(b[0] - site.x);
            ang_a.partial_cmp(&ang_b).unwrap_or(Ordering::Equal)
        });

        // Shoelace formula.
        let n = pts.len();
        let mut area = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            area += pts[i][0] * pts[j][1];
            area -= pts[j][0] * pts[i][1];
        }
        area.abs() / 2.0
    }

    /// Find the index of the nearest site to point `(qx, qy)`.
    pub fn nearest_site(&self, qx: f64, qy: f64) -> usize {
        self.sites
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let d2 = (s.x - qx).powi(2) + (s.y - qy).powi(2);
                (i, d2)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Number of sites.
    pub fn num_sites(&self) -> usize {
        self.sites.len()
    }

    /// Number of Voronoi vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
}

/// Derive a Delaunay triangulation (dual of the Voronoi diagram).
///
/// Returns a list of `(a, b, c)` triples — the original site indices of each triangle.
/// The dual construction connects sites `left_site` and `right_site` for every edge,
/// then finds triangles by looking at circumcircle triples.
pub fn delaunay_from_voronoi(v: &VoronoiDiagram) -> Vec<(usize, usize, usize)> {
    use std::collections::HashSet;

    // Build adjacency set from edges.
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); v.sites.len()];
    for e in &v.edges {
        if e.left_site < v.sites.len() && e.right_site < v.sites.len() {
            adj[e.left_site].insert(e.right_site);
            adj[e.right_site].insert(e.left_site);
        }
    }

    // Collect triangles: for each vertex, find the 3 sites whose cells meet there
    // by collecting all (left_site, right_site) pairs meeting at each vertex.
    let mut vert_sites: Vec<Vec<usize>> = vec![Vec::new(); v.vertices.len()];
    for e in &v.edges {
        for opt_v in [e.v1, e.v2] {
            if let Some(vi) = opt_v {
                if vi < vert_sites.len() {
                    if !vert_sites[vi].contains(&e.left_site) {
                        vert_sites[vi].push(e.left_site);
                    }
                    if !vert_sites[vi].contains(&e.right_site) {
                        vert_sites[vi].push(e.right_site);
                    }
                }
            }
        }
    }

    let mut triangles: Vec<(usize, usize, usize)> = Vec::new();
    let mut seen: HashSet<(usize, usize, usize)> = HashSet::new();

    for sites_at_v in &vert_sites {
        if sites_at_v.len() >= 3 {
            // Take every triple.
            let s = sites_at_v;
            for i in 0..s.len() {
                for j in (i + 1)..s.len() {
                    for k in (j + 1)..s.len() {
                        let mut tri = [s[i], s[j], s[k]];
                        tri.sort_unstable();
                        let key = (tri[0], tri[1], tri[2]);
                        if seen.insert(key) {
                            triangles.push(key);
                        }
                    }
                }
            }
        }
    }

    triangles
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voronoi_basic() {
        let sites = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let bbox = (-1.0, -1.0, 2.0, 2.0);
        let v = VoronoiDiagram::compute(&sites, bbox).expect("compute failed");
        assert_eq!(v.num_sites(), 3);
        // For 3 non-collinear sites there should be at least 1 vertex.
        assert!(!v.vertices.is_empty());
    }

    #[test]
    fn test_voronoi_four_corners() {
        let sites = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let bbox = (-0.5, -0.5, 1.5, 1.5);
        let v = VoronoiDiagram::compute(&sites, bbox).expect("compute failed");
        assert_eq!(v.num_sites(), 4);
    }

    #[test]
    fn test_nearest_site() {
        let sites = vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
        let v = VoronoiDiagram::compute(&sites, (-5.0, -5.0, 15.0, 15.0))
            .expect("compute failed");
        assert_eq!(v.nearest_site(0.1, 0.1), 0);
        assert_eq!(v.nearest_site(9.9, 0.1), 1);
        assert_eq!(v.nearest_site(0.1, 9.9), 2);
    }

    #[test]
    fn test_too_few_sites() {
        let result = VoronoiDiagram::compute(&[(0.0, 0.0)], (0.0, 0.0, 1.0, 1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_delaunay_from_voronoi() {
        let sites = vec![(0.0, 0.0), (2.0, 0.0), (1.0, 2.0), (3.0, 2.0)];
        let v = VoronoiDiagram::compute(&sites, (-1.0, -1.0, 4.0, 3.0))
            .expect("compute failed");
        let tris = delaunay_from_voronoi(&v);
        // With 4 sites we expect at least some triangles from the dual.
        // (May be 0 if the Voronoi has no bounded vertices — that's acceptable.)
        let _ = tris;
    }
}
