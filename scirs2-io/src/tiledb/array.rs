//! TileDB array storage engine.
//!
//! Implements a tile-based multidimensional array storage system supporting
//! both dense and sparse arrays with configurable tile sizes, compression,
//! and range queries.

use std::collections::BTreeMap;

use super::types::{
    ArraySchema, Compression, DataType, Dimension, TileDBConfig, TileDBError, TileOrder,
};

/// A TileDB array supporting dense and sparse storage.
///
/// Data is organized into tiles for efficient I/O. Dense arrays store all
/// cells within dimension bounds; sparse arrays store only occupied cells
/// with explicit coordinates.
#[derive(Debug)]
pub struct TileDBArray {
    /// The array schema.
    schema: ArraySchema,
    /// Configuration.
    config: TileDBConfig,
    /// Dense storage: flat array in tile order.
    /// Indexed by linearized cell position.
    dense_data: Vec<f64>,
    /// Sparse storage: coordinate -> value mapping.
    /// Key is a sorted tuple of dimension coordinates serialized as a `Vec<i64>`.
    sparse_data: BTreeMap<Vec<i64>, f64>,
    /// Whether this array has been initialized with a write.
    initialized: bool,
}

impl TileDBArray {
    /// Create a new TileDB array from a schema.
    pub fn create(schema: ArraySchema, config: TileDBConfig) -> Result<Self, TileDBError> {
        match &schema {
            ArraySchema::Dense {
                dimensions,
                attributes,
            } => {
                if dimensions.is_empty() {
                    return Err(TileDBError::SchemaError(
                        "dense array must have at least one dimension".to_string(),
                    ));
                }
                if attributes.is_empty() {
                    return Err(TileDBError::SchemaError(
                        "array must have at least one attribute".to_string(),
                    ));
                }
                // Pre-allocate dense storage
                let total_cells: usize = dimensions.iter().map(|d| d.num_cells()).product();
                let dense_data = vec![f64::NAN; total_cells];
                Ok(Self {
                    schema,
                    config,
                    dense_data,
                    sparse_data: BTreeMap::new(),
                    initialized: false,
                })
            }
            ArraySchema::Sparse {
                dimensions,
                attributes,
            } => {
                if dimensions.is_empty() {
                    return Err(TileDBError::SchemaError(
                        "sparse array must have at least one dimension".to_string(),
                    ));
                }
                if attributes.is_empty() {
                    return Err(TileDBError::SchemaError(
                        "array must have at least one attribute".to_string(),
                    ));
                }
                Ok(Self {
                    schema,
                    config,
                    dense_data: Vec::new(),
                    sparse_data: BTreeMap::new(),
                    initialized: false,
                })
            }
        }
    }

    /// Return the schema of this array.
    pub fn schema(&self) -> &ArraySchema {
        &self.schema
    }

    /// Return the config of this array.
    pub fn config(&self) -> &TileDBConfig {
        &self.config
    }

    /// Write data to a dense subarray.
    ///
    /// `data` contains the values to write in row-major order.
    /// `subarray` defines the range for each dimension as (start, end) inclusive.
    pub fn write_dense(
        &mut self,
        data: &[f64],
        subarray: &[(usize, usize)],
    ) -> Result<(), TileDBError> {
        if !self.schema.is_dense() {
            return Err(TileDBError::SchemaError(
                "write_dense called on sparse array".to_string(),
            ));
        }

        let dims = self.schema.dimensions();
        if subarray.len() != dims.len() {
            return Err(TileDBError::InvalidQuery(format!(
                "subarray has {} ranges but array has {} dimensions",
                subarray.len(),
                dims.len()
            )));
        }

        // Validate bounds
        for (i, (start, end)) in subarray.iter().enumerate() {
            let dim = &dims[i];
            let dim_start = dim.domain.0 as usize;
            let dim_end = dim.domain.1 as usize;
            if *start < dim_start || *end > dim_end || *start > *end {
                return Err(TileDBError::OutOfBounds(format!(
                    "subarray range ({start}, {end}) out of bounds for dimension '{}' [{dim_start}, {dim_end}]",
                    dim.name
                )));
            }
        }

        // Calculate expected number of elements
        let expected_len: usize = subarray.iter().map(|(s, e)| e - s + 1).product();
        if data.len() != expected_len {
            return Err(TileDBError::InvalidQuery(format!(
                "data length {} does not match subarray size {expected_len}",
                data.len()
            )));
        }

        // Write data into the flat dense array
        let dim_sizes: Vec<usize> = dims.iter().map(|d| d.num_cells()).collect();
        let dim_offsets: Vec<usize> = dims.iter().map(|d| d.domain.0 as usize).collect();

        let sub_sizes: Vec<usize> = subarray.iter().map(|(s, e)| e - s + 1).collect();

        for flat_idx in 0..expected_len {
            // Convert flat index to sub-coordinates
            let sub_coords = flat_to_nd(flat_idx, &sub_sizes);

            // Convert to global coordinates
            let global_coords: Vec<usize> = sub_coords
                .iter()
                .enumerate()
                .map(|(i, &c)| subarray[i].0 + c)
                .collect();

            // Convert global coordinates to storage index
            let rel_coords: Vec<usize> = global_coords
                .iter()
                .enumerate()
                .map(|(i, &c)| c - dim_offsets[i])
                .collect();

            let storage_idx = nd_to_flat(&rel_coords, &dim_sizes, &self.config.cell_order);

            if storage_idx < self.dense_data.len() {
                self.dense_data[storage_idx] = data[flat_idx];
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Read data from a dense subarray.
    ///
    /// `subarray` defines the range for each dimension as (start, end) inclusive.
    /// Returns data in row-major order.
    pub fn read_dense(&self, subarray: &[(usize, usize)]) -> Result<Vec<f64>, TileDBError> {
        if !self.schema.is_dense() {
            return Err(TileDBError::SchemaError(
                "read_dense called on sparse array".to_string(),
            ));
        }

        let dims = self.schema.dimensions();
        if subarray.len() != dims.len() {
            return Err(TileDBError::InvalidQuery(format!(
                "subarray has {} ranges but array has {} dimensions",
                subarray.len(),
                dims.len()
            )));
        }

        // Validate bounds
        for (i, (start, end)) in subarray.iter().enumerate() {
            let dim = &dims[i];
            let dim_start = dim.domain.0 as usize;
            let dim_end = dim.domain.1 as usize;
            if *start < dim_start || *end > dim_end || *start > *end {
                return Err(TileDBError::OutOfBounds(format!(
                    "subarray range ({start}, {end}) out of bounds for dim '{}' [{dim_start}, {dim_end}]",
                    dim.name
                )));
            }
        }

        let dim_sizes: Vec<usize> = dims.iter().map(|d| d.num_cells()).collect();
        let dim_offsets: Vec<usize> = dims.iter().map(|d| d.domain.0 as usize).collect();
        let sub_sizes: Vec<usize> = subarray.iter().map(|(s, e)| e - s + 1).collect();
        let total: usize = sub_sizes.iter().product();

        let mut result = Vec::with_capacity(total);

        for flat_idx in 0..total {
            let sub_coords = flat_to_nd(flat_idx, &sub_sizes);
            let global_coords: Vec<usize> = sub_coords
                .iter()
                .enumerate()
                .map(|(i, &c)| subarray[i].0 + c)
                .collect();
            let rel_coords: Vec<usize> = global_coords
                .iter()
                .enumerate()
                .map(|(i, &c)| c - dim_offsets[i])
                .collect();

            let storage_idx = nd_to_flat(&rel_coords, &dim_sizes, &self.config.cell_order);

            if storage_idx < self.dense_data.len() {
                result.push(self.dense_data[storage_idx]);
            } else {
                result.push(f64::NAN);
            }
        }

        Ok(result)
    }

    /// Write sparse data to the array.
    ///
    /// `coords` is a slice of coordinate vectors (one per dimension).
    /// `values` contains the attribute values for each coordinate tuple.
    pub fn write_sparse(&mut self, coords: &[Vec<f64>], values: &[f64]) -> Result<(), TileDBError> {
        if self.schema.is_dense() {
            return Err(TileDBError::SchemaError(
                "write_sparse called on dense array".to_string(),
            ));
        }

        let dims = self.schema.dimensions();
        if coords.len() != dims.len() {
            return Err(TileDBError::InvalidQuery(format!(
                "coords has {} dimension vectors but array has {} dimensions",
                coords.len(),
                dims.len()
            )));
        }

        let num_points = values.len();
        for (i, coord_vec) in coords.iter().enumerate() {
            if coord_vec.len() != num_points {
                return Err(TileDBError::InvalidQuery(format!(
                    "dimension {} has {} coords but {} values provided",
                    i,
                    coord_vec.len(),
                    num_points
                )));
            }
        }

        // Validate bounds
        for point_idx in 0..num_points {
            for (dim_idx, dim) in dims.iter().enumerate() {
                let c = coords[dim_idx][point_idx];
                if c < dim.domain.0 || c > dim.domain.1 {
                    return Err(TileDBError::OutOfBounds(format!(
                        "coordinate {c} out of bounds for dimension '{}' [{}, {}]",
                        dim.name, dim.domain.0, dim.domain.1
                    )));
                }
            }
        }

        // Insert into sparse storage
        for point_idx in 0..num_points {
            let key: Vec<i64> = coords.iter().map(|cv| cv[point_idx] as i64).collect();
            self.sparse_data.insert(key, values[point_idx]);
        }

        self.initialized = true;
        Ok(())
    }

    /// Read sparse data within a range query.
    ///
    /// `query_range` defines the (min, max) range for each dimension.
    /// Returns (coordinates, values) for all cells within the range.
    pub fn read_sparse(
        &self,
        query_range: &[(f64, f64)],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), TileDBError> {
        if self.schema.is_dense() {
            return Err(TileDBError::SchemaError(
                "read_sparse called on dense array".to_string(),
            ));
        }

        let dims = self.schema.dimensions();
        if query_range.len() != dims.len() {
            return Err(TileDBError::InvalidQuery(format!(
                "query has {} ranges but array has {} dimensions",
                query_range.len(),
                dims.len()
            )));
        }

        let num_dims = dims.len();
        let mut result_coords: Vec<Vec<f64>> = (0..num_dims).map(|_| Vec::new()).collect();
        let mut result_values: Vec<f64> = Vec::new();

        for (key, &value) in &self.sparse_data {
            let in_range = key.iter().enumerate().all(|(i, &c)| {
                let cf = c as f64;
                cf >= query_range[i].0 && cf <= query_range[i].1
            });

            if in_range {
                for (i, &c) in key.iter().enumerate() {
                    result_coords[i].push(c as f64);
                }
                result_values.push(value);
            }
        }

        Ok((result_coords, result_values))
    }

    /// Return the number of non-empty cells in the array.
    pub fn num_cells(&self) -> usize {
        if self.schema.is_dense() {
            self.dense_data.iter().filter(|v| !v.is_nan()).count()
        } else {
            self.sparse_data.len()
        }
    }

    /// Return the total capacity of the dense array.
    pub fn total_capacity(&self) -> usize {
        self.dense_data.len()
    }

    /// Return the tile information for each dimension.
    pub fn tile_info(&self) -> Vec<(String, usize, usize)> {
        self.schema
            .dimensions()
            .iter()
            .map(|d| (d.name.clone(), d.num_cells(), d.num_tiles()))
            .collect()
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// Convert a flat index to N-dimensional coordinates (row-major).
fn flat_to_nd(flat_idx: usize, sizes: &[usize]) -> Vec<usize> {
    let ndim = sizes.len();
    let mut coords = vec![0usize; ndim];
    let mut remaining = flat_idx;

    for i in (0..ndim).rev() {
        if sizes[i] > 0 {
            coords[i] = remaining % sizes[i];
            remaining /= sizes[i];
        }
    }

    coords
}

/// Convert N-dimensional coordinates to a flat index.
fn nd_to_flat(coords: &[usize], sizes: &[usize], order: &TileOrder) -> usize {
    let ndim = coords.len();
    if ndim == 0 {
        return 0;
    }

    match order {
        TileOrder::RowMajor | TileOrder::Hilbert => {
            // Row-major: last index varies fastest
            let mut idx = 0usize;
            let mut stride = 1usize;
            for i in (0..ndim).rev() {
                idx += coords[i] * stride;
                stride *= sizes[i];
            }
            idx
        }
        TileOrder::ColMajor => {
            // Column-major: first index varies fastest
            let mut idx = 0usize;
            let mut stride = 1usize;
            for i in 0..ndim {
                idx += coords[i] * stride;
                stride *= sizes[i];
            }
            idx
        }
    }
}
