//! Spatial tree data structures for Barnes-Hut approximation in t-SNE
//!
//! This module provides QuadTree (2D) and OctTree (3D) implementations
//! used by the Barnes-Hut variant of t-SNE for O(N log N) complexity.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TransformError};

// Constants for numerical stability
const MACHINE_EPSILON: f64 = 1e-14;

/// Spatial tree data structure for Barnes-Hut approximation
#[derive(Debug, Clone)]
pub(crate) enum SpatialTree {
    /// QuadTree for 2D embeddings
    QuadTree(QuadTreeNode),
    /// OctTree for 3D embeddings
    OctTree(OctTreeNode),
}

/// Node in a quadtree (for 2D embeddings)
#[derive(Debug, Clone)]
pub(crate) struct QuadTreeNode {
    /// Bounding box of this node
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    /// Center of mass
    center_of_mass: Option<Array1<f64>>,
    /// Total mass (number of points)
    total_mass: f64,
    /// Point indices in this node (for leaf nodes)
    point_indices: Vec<usize>,
    /// Children nodes (NW, NE, SW, SE)
    children: Option<[Box<QuadTreeNode>; 4]>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

/// Node in an octree (for 3D embeddings)
#[derive(Debug, Clone)]
pub(crate) struct OctTreeNode {
    /// Bounding box of this node
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    /// Center of mass
    center_of_mass: Option<Array1<f64>>,
    /// Total mass (number of points)
    total_mass: f64,
    /// Point indices in this node (for leaf nodes)
    point_indices: Vec<usize>,
    /// Children nodes (8 octants)
    children: Option<[Box<OctTreeNode>; 8]>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl SpatialTree {
    /// Create a new quadtree for 2D embeddings
    pub(crate) fn new_quadtree(embedding: &Array2<f64>) -> Result<Self> {
        let n_samples = embedding.shape()[0];

        if embedding.shape()[1] != 2 {
            return Err(TransformError::InvalidInput(
                "QuadTree requires 2D embedding".to_string(),
            ));
        }

        // Find bounding box
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for i in 0..n_samples {
            let x = embedding[[i, 0]];
            let y = embedding[[i, 1]];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }

        // Add small margin to avoid edge cases
        let margin = 0.01 * ((x_max - x_min) + (y_max - y_min));
        x_min -= margin;
        x_max += margin;
        y_min -= margin;
        y_max += margin;

        let point_indices: Vec<usize> = (0..n_samples).collect();

        let mut root = QuadTreeNode {
            x_min,
            x_max,
            y_min,
            y_max,
            center_of_mass: None,
            total_mass: 0.0,
            point_indices,
            children: None,
            is_leaf: true,
        };

        root.build_tree(embedding)?;

        Ok(SpatialTree::QuadTree(root))
    }

    /// Create a new octree for 3D embeddings
    pub(crate) fn new_octree(embedding: &Array2<f64>) -> Result<Self> {
        let n_samples = embedding.shape()[0];

        if embedding.shape()[1] != 3 {
            return Err(TransformError::InvalidInput(
                "OctTree requires 3D embedding".to_string(),
            ));
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for i in 0..n_samples {
            let x = embedding[[i, 0]];
            let y = embedding[[i, 1]];
            let z = embedding[[i, 2]];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }

        let margin = 0.01 * ((x_max - x_min) + (y_max - y_min) + (z_max - z_min));
        x_min -= margin;
        x_max += margin;
        y_min -= margin;
        y_max += margin;
        z_min -= margin;
        z_max += margin;

        let point_indices: Vec<usize> = (0..n_samples).collect();

        let mut root = OctTreeNode {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            center_of_mass: None,
            total_mass: 0.0,
            point_indices,
            children: None,
            is_leaf: true,
        };

        root.build_tree(embedding)?;

        Ok(SpatialTree::OctTree(root))
    }

    /// Compute forces on a point using Barnes-Hut approximation
    pub(crate) fn compute_forces(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
    ) -> Result<(Array1<f64>, f64)> {
        match self {
            SpatialTree::QuadTree(root) => {
                root.compute_forces_quad(point, point_idx, angle, degrees_of_freedom)
            }
            SpatialTree::OctTree(root) => {
                root.compute_forces_oct(point, point_idx, angle, degrees_of_freedom)
            }
        }
    }
}

impl QuadTreeNode {
    /// Build the quadtree recursively
    fn build_tree(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.point_indices.len() <= 1 {
            self.update_center_of_mass(embedding)?;
            return Ok(());
        }

        let x_mid = (self.x_min + self.x_max) / 2.0;
        let y_mid = (self.y_min + self.y_max) / 2.0;

        let mut quadrants: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for &idx in &self.point_indices {
            let x = embedding[[idx, 0]];
            let y = embedding[[idx, 1]];

            let quadrant = match (x >= x_mid, y >= y_mid) {
                (false, false) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (true, true) => 3,
            };

            quadrants[quadrant].push(idx);
        }

        let mut children = [
            Box::new(QuadTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: self.y_min,
                y_max: y_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[0].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(QuadTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: y_mid,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[1].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(QuadTreeNode {
                x_min: self.x_min,
                x_max: x_mid,
                y_min: y_mid,
                y_max: self.y_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[2].clone(),
                children: None,
                is_leaf: true,
            }),
            Box::new(QuadTreeNode {
                x_min: x_mid,
                x_max: self.x_max,
                y_min: y_mid,
                y_max: self.y_max,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: quadrants[3].clone(),
                children: None,
                is_leaf: true,
            }),
        ];

        for child in &mut children {
            child.build_tree(embedding)?;
        }

        self.children = Some(children);
        self.is_leaf = false;
        self.point_indices.clear();
        self.update_center_of_mass(embedding)?;

        Ok(())
    }

    /// Update center of mass for this node
    fn update_center_of_mass(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.is_leaf {
            if self.point_indices.is_empty() {
                self.total_mass = 0.0;
                self.center_of_mass = None;
                return Ok(());
            }

            let mut com = Array1::zeros(2);
            for &idx in &self.point_indices {
                com[0] += embedding[[idx, 0]];
                com[1] += embedding[[idx, 1]];
            }

            self.total_mass = self.point_indices.len() as f64;
            com.mapv_inplace(|x| x / self.total_mass);
            self.center_of_mass = Some(com);
        } else if let Some(ref children) = self.children {
            let mut com = Array1::zeros(2);
            let mut total_mass = 0.0;

            for child in children.iter() {
                if let Some(ref child_com) = child.center_of_mass {
                    total_mass += child.total_mass;
                    for i in 0..2 {
                        com[i] += child_com[i] * child.total_mass;
                    }
                }
            }

            if total_mass > 0.0 {
                com.mapv_inplace(|x| x / total_mass);
                self.center_of_mass = Some(com);
                self.total_mass = total_mass;
            } else {
                self.center_of_mass = None;
                self.total_mass = 0.0;
            }
        }

        Ok(())
    }

    /// Compute forces using Barnes-Hut approximation for quadtree
    fn compute_forces_quad(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
    ) -> Result<(Array1<f64>, f64)> {
        let mut force = Array1::zeros(2);
        let mut sum_q = 0.0;

        self.compute_forces_recursive_quad(
            point,
            point_idx,
            angle,
            degrees_of_freedom,
            &mut force,
            &mut sum_q,
        )?;

        Ok((force, sum_q))
    }

    /// Recursive force computation for quadtree
    #[allow(clippy::too_many_arguments)]
    fn compute_forces_recursive_quad(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
        force: &mut Array1<f64>,
        sum_q: &mut f64,
    ) -> Result<()> {
        if let Some(ref com) = self.center_of_mass {
            if self.total_mass == 0.0 {
                return Ok(());
            }

            let dx = point[0] - com[0];
            let dy = point[1] - com[1];
            let dist_squared = dx * dx + dy * dy;

            if dist_squared < MACHINE_EPSILON {
                return Ok(());
            }

            let node_size = (self.x_max - self.x_min).max(self.y_max - self.y_min);
            let distance = dist_squared.sqrt();

            if self.is_leaf || (node_size / distance) < angle {
                let q_factor = (1.0 + dist_squared / degrees_of_freedom)
                    .powf(-(degrees_of_freedom + 1.0) / 2.0);

                *sum_q += self.total_mass * q_factor;

                let force_factor =
                    (degrees_of_freedom + 1.0) * self.total_mass * q_factor / degrees_of_freedom;
                force[0] += force_factor * dx;
                force[1] += force_factor * dy;
            } else if let Some(ref children) = self.children {
                for child in children.iter() {
                    child.compute_forces_recursive_quad(
                        point,
                        point_idx,
                        angle,
                        degrees_of_freedom,
                        force,
                        sum_q,
                    )?;
                }
            }
        }

        Ok(())
    }
}

impl OctTreeNode {
    /// Build the octree recursively
    fn build_tree(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.point_indices.len() <= 1 {
            self.update_center_of_mass(embedding)?;
            return Ok(());
        }

        let x_mid = (self.x_min + self.x_max) / 2.0;
        let y_mid = (self.y_min + self.y_max) / 2.0;
        let z_mid = (self.z_min + self.z_max) / 2.0;

        let mut octants: [Vec<usize>; 8] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];

        for &idx in &self.point_indices {
            let x = embedding[[idx, 0]];
            let y = embedding[[idx, 1]];
            let z = embedding[[idx, 2]];

            let octant = match (x >= x_mid, y >= y_mid, z >= z_mid) {
                (false, false, false) => 0,
                (true, false, false) => 1,
                (false, true, false) => 2,
                (true, true, false) => 3,
                (false, false, true) => 4,
                (true, false, true) => 5,
                (false, true, true) => 6,
                (true, true, true) => 7,
            };

            octants[octant].push(idx);
        }

        let bounds = [
            (self.x_min, x_mid, self.y_min, y_mid, self.z_min, z_mid),
            (x_mid, self.x_max, self.y_min, y_mid, self.z_min, z_mid),
            (self.x_min, x_mid, y_mid, self.y_max, self.z_min, z_mid),
            (x_mid, self.x_max, y_mid, self.y_max, self.z_min, z_mid),
            (self.x_min, x_mid, self.y_min, y_mid, z_mid, self.z_max),
            (x_mid, self.x_max, self.y_min, y_mid, z_mid, self.z_max),
            (self.x_min, x_mid, y_mid, self.y_max, z_mid, self.z_max),
            (x_mid, self.x_max, y_mid, self.y_max, z_mid, self.z_max),
        ];

        let mut children: Vec<Box<OctTreeNode>> = Vec::with_capacity(8);
        for (i, &(xlo, xhi, ylo, yhi, zlo, zhi)) in bounds.iter().enumerate() {
            children.push(Box::new(OctTreeNode {
                x_min: xlo,
                x_max: xhi,
                y_min: ylo,
                y_max: yhi,
                z_min: zlo,
                z_max: zhi,
                center_of_mass: None,
                total_mass: 0.0,
                point_indices: octants[i].clone(),
                children: None,
                is_leaf: true,
            }));
        }

        for child in &mut children {
            child.build_tree(embedding)?;
        }

        // Convert Vec to fixed-size array
        let children_array: [Box<OctTreeNode>; 8] = match children.try_into() {
            Ok(arr) => arr,
            Err(_) => {
                return Err(TransformError::ComputationError(
                    "Failed to build octree children array".to_string(),
                ))
            }
        };

        self.children = Some(children_array);
        self.is_leaf = false;
        self.point_indices.clear();
        self.update_center_of_mass(embedding)?;

        Ok(())
    }

    /// Update center of mass for this octree node
    fn update_center_of_mass(&mut self, embedding: &Array2<f64>) -> Result<()> {
        if self.is_leaf {
            if self.point_indices.is_empty() {
                self.total_mass = 0.0;
                self.center_of_mass = None;
                return Ok(());
            }

            let mut com = Array1::zeros(3);
            for &idx in &self.point_indices {
                com[0] += embedding[[idx, 0]];
                com[1] += embedding[[idx, 1]];
                com[2] += embedding[[idx, 2]];
            }

            self.total_mass = self.point_indices.len() as f64;
            com.mapv_inplace(|x| x / self.total_mass);
            self.center_of_mass = Some(com);
        } else if let Some(ref children) = self.children {
            let mut com = Array1::zeros(3);
            let mut total_mass = 0.0;

            for child in children.iter() {
                if let Some(ref child_com) = child.center_of_mass {
                    total_mass += child.total_mass;
                    for i in 0..3 {
                        com[i] += child_com[i] * child.total_mass;
                    }
                }
            }

            if total_mass > 0.0 {
                com.mapv_inplace(|x| x / total_mass);
                self.center_of_mass = Some(com);
                self.total_mass = total_mass;
            } else {
                self.center_of_mass = None;
                self.total_mass = 0.0;
            }
        }

        Ok(())
    }

    /// Compute forces using Barnes-Hut approximation for octree
    fn compute_forces_oct(
        &self,
        point: &Array1<f64>,
        point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
    ) -> Result<(Array1<f64>, f64)> {
        let mut force = Array1::zeros(3);
        let mut sum_q = 0.0;

        self.compute_forces_recursive_oct(
            point,
            point_idx,
            angle,
            degrees_of_freedom,
            &mut force,
            &mut sum_q,
        )?;

        Ok((force, sum_q))
    }

    /// Recursive force computation for octree
    #[allow(clippy::too_many_arguments)]
    fn compute_forces_recursive_oct(
        &self,
        point: &Array1<f64>,
        _point_idx: usize,
        angle: f64,
        degrees_of_freedom: f64,
        force: &mut Array1<f64>,
        sum_q: &mut f64,
    ) -> Result<()> {
        if let Some(ref com) = self.center_of_mass {
            if self.total_mass == 0.0 {
                return Ok(());
            }

            let dx = point[0] - com[0];
            let dy = point[1] - com[1];
            let dz = point[2] - com[2];
            let dist_squared = dx * dx + dy * dy + dz * dz;

            if dist_squared < MACHINE_EPSILON {
                return Ok(());
            }

            let node_size = (self.x_max - self.x_min)
                .max(self.y_max - self.y_min)
                .max(self.z_max - self.z_min);
            let distance = dist_squared.sqrt();

            if self.is_leaf || (node_size / distance) < angle {
                let q_factor = (1.0 + dist_squared / degrees_of_freedom)
                    .powf(-(degrees_of_freedom + 1.0) / 2.0);

                *sum_q += self.total_mass * q_factor;

                let force_factor =
                    (degrees_of_freedom + 1.0) * self.total_mass * q_factor / degrees_of_freedom;
                force[0] += force_factor * dx;
                force[1] += force_factor * dy;
                force[2] += force_factor * dz;
            } else if let Some(ref children) = self.children {
                for child in children.iter() {
                    child.compute_forces_recursive_oct(
                        point,
                        _point_idx,
                        angle,
                        degrees_of_freedom,
                        force,
                        sum_q,
                    )?;
                }
            }
        }

        Ok(())
    }
}
