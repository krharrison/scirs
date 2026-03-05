//! Minimum Bounding Rectangle (MBR) for 2D spatial operations.
//!
//! Provides an efficient 2D axis-aligned bounding rectangle with all standard
//! spatial predicates: intersection, containment, union, enlargement, etc.

/// 2D minimum bounding rectangle.
///
/// Coordinates are stored as `(min_x, min_y, max_x, max_y)` where `min_*` ≤ `max_*`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MBR {
    /// Minimum x-coordinate.
    pub min_x: f64,
    /// Minimum y-coordinate.
    pub min_y: f64,
    /// Maximum x-coordinate.
    pub max_x: f64,
    /// Maximum y-coordinate.
    pub max_y: f64,
}

impl MBR {
    /// Create a new MBR with explicit bounds.
    ///
    /// # Panics
    ///
    /// Panics (in debug mode) if `min_x > max_x` or `min_y > max_y`.
    #[inline]
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        debug_assert!(
            min_x <= max_x && min_y <= max_y,
            "MBR: min must be <= max"
        );
        Self { min_x, min_y, max_x, max_y }
    }

    /// Create an MBR that covers a single point (zero area).
    #[inline]
    pub fn from_point(x: f64, y: f64) -> Self {
        Self::new(x, y, x, y)
    }

    /// Compute the tight bounding rectangle of a slice of 2D points.
    ///
    /// Returns `None` if `points` is empty.
    pub fn from_points(points: &[[f64; 2]]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for p in points {
            if p[0] < min_x { min_x = p[0]; }
            if p[0] > max_x { max_x = p[0]; }
            if p[1] < min_y { min_y = p[1]; }
            if p[1] > max_y { max_y = p[1]; }
        }
        Some(Self::new(min_x, min_y, max_x, max_y))
    }

    /// Area of the rectangle (`width * height`).
    #[inline]
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// Perimeter of the rectangle (`2 * (width + height)`).
    #[inline]
    pub fn perimeter(&self) -> f64 {
        2.0 * ((self.max_x - self.min_x) + (self.max_y - self.min_y))
    }

    /// Return `true` if the point `(x, y)` lies inside or on the boundary.
    #[inline]
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Return `true` if this MBR and `other` share any area (including boundary touch).
    #[inline]
    pub fn intersects(&self, other: &MBR) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Return `true` if `other` is fully contained within this MBR.
    #[inline]
    pub fn contains(&self, other: &MBR) -> bool {
        other.min_x >= self.min_x
            && other.max_x <= self.max_x
            && other.min_y >= self.min_y
            && other.max_y <= self.max_y
    }

    /// Smallest MBR that encloses both `self` and `other`.
    #[inline]
    pub fn union(&self, other: &MBR) -> MBR {
        MBR::new(
            self.min_x.min(other.min_x),
            self.min_y.min(other.min_y),
            self.max_x.max(other.max_x),
            self.max_y.max(other.max_y),
        )
    }

    /// Overlap rectangle of `self` and `other`, or `None` if they do not intersect.
    pub fn intersection(&self, other: &MBR) -> Option<MBR> {
        let min_x = self.min_x.max(other.min_x);
        let min_y = self.min_y.max(other.min_y);
        let max_x = self.max_x.min(other.max_x);
        let max_y = self.max_y.min(other.max_y);
        if min_x <= max_x && min_y <= max_y {
            Some(MBR::new(min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    /// Increase in area required to accommodate `other`.
    ///
    /// Used by the R-tree insertion algorithm to choose the least-cost subtree.
    #[inline]
    pub fn enlargement_needed(&self, other: &MBR) -> f64 {
        self.union(other).area() - self.area()
    }

    /// Geometric centre of the MBR.
    #[inline]
    pub fn center(&self) -> [f64; 2] {
        [
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        ]
    }

    /// Minimum Euclidean distance from the MBR boundary/interior to a point.
    ///
    /// Returns 0.0 if the point is inside or on the boundary.
    #[inline]
    pub fn distance_to_point(&self, x: f64, y: f64) -> f64 {
        let dx = f64::max(self.min_x - x, f64::max(0.0, x - self.max_x));
        let dy = f64::max(self.min_y - y, f64::max(0.0, y - self.max_y));
        (dx * dx + dy * dy).sqrt()
    }

    /// Minimum Euclidean distance between the boundaries of two MBRs.
    ///
    /// Returns 0.0 if they intersect.
    #[inline]
    pub fn distance_to_mbr(&self, other: &MBR) -> f64 {
        let dx = f64::max(0.0, f64::max(self.min_x - other.max_x, other.min_x - self.max_x));
        let dy = f64::max(0.0, f64::max(self.min_y - other.max_y, other.min_y - self.max_y));
        (dx * dx + dy * dy).sqrt()
    }

    /// Width of the MBR.
    #[inline]
    pub fn width(&self) -> f64 { self.max_x - self.min_x }

    /// Height of the MBR.
    #[inline]
    pub fn height(&self) -> f64 { self.max_y - self.min_y }

    /// Expand the MBR uniformly by `delta` in all directions.
    #[inline]
    pub fn expand(&self, delta: f64) -> MBR {
        MBR::new(
            self.min_x - delta,
            self.min_y - delta,
            self.max_x + delta,
            self.max_y + delta,
        )
    }
}

impl std::fmt::Display for MBR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MBR([{}, {}] – [{}, {}])",
            self.min_x, self.min_y, self.max_x, self.max_y
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbr_basic() {
        let r = MBR::new(0.0, 0.0, 4.0, 3.0);
        assert!((r.area() - 12.0).abs() < 1e-12);
        assert!((r.perimeter() - 14.0).abs() < 1e-12);
        assert_eq!(r.center(), [2.0, 1.5]);
    }

    #[test]
    fn test_mbr_from_point() {
        let r = MBR::from_point(5.0, 7.0);
        assert_eq!(r.area(), 0.0);
        assert!(r.contains_point(5.0, 7.0));
    }

    #[test]
    fn test_mbr_from_points() {
        let pts = [[1.0, 2.0], [3.0, 0.0], [2.0, 5.0]];
        let r = MBR::from_points(&pts).expect("Should return Some");
        assert!((r.min_x - 1.0).abs() < 1e-12);
        assert!((r.min_y - 0.0).abs() < 1e-12);
        assert!((r.max_x - 3.0).abs() < 1e-12);
        assert!((r.max_y - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_mbr_from_points_empty() {
        assert!(MBR::from_points(&[]).is_none());
    }

    #[test]
    fn test_mbr_intersects() {
        let a = MBR::new(0.0, 0.0, 2.0, 2.0);
        let b = MBR::new(1.0, 1.0, 3.0, 3.0);
        let c = MBR::new(5.0, 5.0, 6.0, 6.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_mbr_contains() {
        let outer = MBR::new(0.0, 0.0, 10.0, 10.0);
        let inner = MBR::new(2.0, 2.0, 5.0, 5.0);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_mbr_union() {
        let a = MBR::new(0.0, 0.0, 2.0, 2.0);
        let b = MBR::new(1.0, 1.0, 4.0, 3.0);
        let u = a.union(&b);
        assert_eq!((u.min_x, u.min_y, u.max_x, u.max_y), (0.0, 0.0, 4.0, 3.0));
    }

    #[test]
    fn test_mbr_intersection() {
        let a = MBR::new(0.0, 0.0, 3.0, 3.0);
        let b = MBR::new(2.0, 2.0, 5.0, 5.0);
        let i = a.intersection(&b).expect("Should intersect");
        assert!((i.area() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mbr_distance_to_point() {
        let r = MBR::new(1.0, 1.0, 3.0, 3.0);
        // Inside: 0
        assert_eq!(r.distance_to_point(2.0, 2.0), 0.0);
        // To the left
        let d = r.distance_to_point(0.0, 2.0);
        assert!((d - 1.0).abs() < 1e-12);
        // Diagonal
        let d2 = r.distance_to_point(0.0, 0.0);
        assert!((d2 - 2_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mbr_enlargement_needed() {
        let a = MBR::new(0.0, 0.0, 1.0, 1.0);
        let b = MBR::new(0.0, 0.0, 2.0, 2.0);
        let enl = a.enlargement_needed(&b);
        assert!((enl - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mbr_expand() {
        let r = MBR::new(1.0, 1.0, 3.0, 3.0);
        let e = r.expand(0.5);
        assert!((e.min_x - 0.5).abs() < 1e-12);
        assert!((e.max_x - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_mbr_display() {
        let r = MBR::new(0.0, 0.0, 1.0, 1.0);
        let s = format!("{}", r);
        assert!(s.contains("MBR"));
    }
}
