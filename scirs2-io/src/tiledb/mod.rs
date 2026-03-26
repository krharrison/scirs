//! TileDB-inspired multidimensional array storage.
//!
//! TileDB is an array data management system for scientific data. This module
//! provides a pure-Rust implementation of TileDB-style array storage with:
//!
//! - Dense and sparse multidimensional arrays
//! - Tile-based I/O with configurable tile sizes
//! - Row-major, column-major, and Hilbert tile ordering
//! - Range queries on sparse arrays
//! - Subarray read/write operations on dense arrays

pub mod array;
pub mod types;

pub use array::TileDBArray;
pub use types::{
    ArraySchema, Attribute, Compression, DataType, Dimension, TileDBConfig, TileDBError, TileOrder,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_2d_schema() -> ArraySchema {
        ArraySchema::Dense {
            dimensions: vec![
                Dimension::new("rows", (0.0, 3.0), 2.0),
                Dimension::new("cols", (0.0, 3.0), 2.0),
            ],
            attributes: vec![Attribute::new("val", DataType::Float64)],
        }
    }

    fn sparse_2d_schema() -> ArraySchema {
        ArraySchema::Sparse {
            dimensions: vec![
                Dimension::new("x", (0.0, 99.0), 10.0),
                Dimension::new("y", (0.0, 99.0), 10.0),
            ],
            attributes: vec![Attribute::new("value", DataType::Float64)],
        }
    }

    // ─── Dense array tests ───────────────────────────────────────────────

    #[test]
    fn test_dense_write_read_roundtrip() {
        let schema = dense_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Write a 4x4 array
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        arr.write_dense(&data, &[(0, 3), (0, 3)]).expect("write");

        // Read it back
        let result = arr.read_dense(&[(0, 3), (0, 3)]).expect("read");
        assert_eq!(result, data);
    }

    #[test]
    fn test_dense_subarray_query() {
        let schema = dense_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Write full 4x4 array: values = row*4 + col
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        arr.write_dense(&data, &[(0, 3), (0, 3)]).expect("write");

        // Read subarray [1..2, 1..2] (2x2 = 4 elements)
        let sub = arr.read_dense(&[(1, 2), (1, 2)]).expect("read sub");
        // Row 1: cols 1,2 => 5, 6
        // Row 2: cols 1,2 => 9, 10
        assert_eq!(sub, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_dense_subarray_write() {
        let schema = dense_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Write only a 2x2 subarray at [1..2, 1..2]
        arr.write_dense(&[100.0, 200.0, 300.0, 400.0], &[(1, 2), (1, 2)])
            .expect("write sub");

        let sub = arr.read_dense(&[(1, 2), (1, 2)]).expect("read sub");
        assert_eq!(sub, vec![100.0, 200.0, 300.0, 400.0]);

        // Unwritten cells should be NaN
        let corner = arr.read_dense(&[(0, 0), (0, 0)]).expect("read corner");
        assert!(corner[0].is_nan());
    }

    #[test]
    fn test_dense_tile_boundary_alignment() {
        // Create array with tile_extent=2 and 4 cells per dim
        // Tiles: [0,1] and [2,3] along each dimension
        let schema = ArraySchema::Dense {
            dimensions: vec![
                Dimension::new("rows", (0.0, 3.0), 2.0),
                Dimension::new("cols", (0.0, 3.0), 2.0),
            ],
            attributes: vec![Attribute::new("val", DataType::Float64)],
        };
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Write data that spans tile boundaries
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        arr.write_dense(&data, &[(0, 3), (0, 3)]).expect("write");

        // Read across tile boundary: rows [1,2], all cols
        let cross_tile = arr.read_dense(&[(1, 2), (0, 3)]).expect("read cross");
        // Row 1: 4,5,6,7 ; Row 2: 8,9,10,11
        assert_eq!(cross_tile, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

        // Verify tile info
        let info = arr.tile_info();
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].1, 4); // 4 cells in rows
        assert_eq!(info[0].2, 2); // 2 tiles in rows
    }

    #[test]
    fn test_dense_out_of_bounds() {
        let schema = dense_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Out of bounds write
        let result = arr.write_dense(&[1.0], &[(5, 5), (0, 0)]);
        assert!(result.is_err());

        // Out of bounds read
        let result = arr.read_dense(&[(0, 10), (0, 0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dense_column_major() {
        let schema = ArraySchema::Dense {
            dimensions: vec![
                Dimension::new("rows", (0.0, 1.0), 2.0),
                Dimension::new("cols", (0.0, 1.0), 2.0),
            ],
            attributes: vec![Attribute::new("val", DataType::Float64)],
        };
        let config = TileDBConfig {
            cell_order: TileOrder::ColMajor,
            ..Default::default()
        };
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Write 2x2 in row-major input order
        arr.write_dense(&[1.0, 2.0, 3.0, 4.0], &[(0, 1), (0, 1)])
            .expect("write");

        // Read back — should get same values
        let result = arr.read_dense(&[(0, 1), (0, 1)]).expect("read");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ─── Sparse array tests ─────────────────────────────────────────────

    #[test]
    fn test_sparse_write_read_roundtrip() {
        let schema = sparse_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        let x_coords = vec![10.0, 20.0, 30.0];
        let y_coords = vec![15.0, 25.0, 35.0];
        let values = vec![1.1, 2.2, 3.3];

        arr.write_sparse(&[x_coords.clone(), y_coords.clone()], &values)
            .expect("write");

        // Read everything
        let (coords, vals) = arr
            .read_sparse(&[(0.0, 99.0), (0.0, 99.0)])
            .expect("read all");

        assert_eq!(vals.len(), 3);
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0].len(), 3);
    }

    #[test]
    fn test_sparse_range_query() {
        let schema = sparse_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        // Write points at (10,10), (20,20), (50,50), (80,80)
        arr.write_sparse(
            &[vec![10.0, 20.0, 50.0, 80.0], vec![10.0, 20.0, 50.0, 80.0]],
            &[1.0, 2.0, 3.0, 4.0],
        )
        .expect("write");

        // Query only the range [0, 30] x [0, 30]
        let (coords, vals) = arr
            .read_sparse(&[(0.0, 30.0), (0.0, 30.0)])
            .expect("range query");

        assert_eq!(vals.len(), 2);
        assert_eq!(vals, vec![1.0, 2.0]);
        assert_eq!(coords[0], vec![10.0, 20.0]);
    }

    #[test]
    fn test_sparse_overwrite() {
        let schema = sparse_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        arr.write_sparse(&[vec![10.0], vec![10.0]], &[1.0])
            .expect("write1");
        arr.write_sparse(&[vec![10.0], vec![10.0]], &[99.0])
            .expect("write2");

        let (_, vals) = arr
            .read_sparse(&[(10.0, 10.0), (10.0, 10.0)])
            .expect("read");
        assert_eq!(vals, vec![99.0]);
    }

    #[test]
    fn test_sparse_out_of_bounds() {
        let schema = sparse_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        let result = arr.write_sparse(&[vec![200.0], vec![10.0]], &[1.0]);
        assert!(result.is_err());
    }

    // ─── Config and type tests ───────────────────────────────────────────

    #[test]
    fn test_tiledb_config_default() {
        let config = TileDBConfig::default();
        assert_eq!(config.tile_capacity, 1024);
        assert_eq!(config.compression, Compression::None);
        assert_eq!(config.tile_order, TileOrder::RowMajor);
        assert_eq!(config.cell_order, TileOrder::RowMajor);
    }

    #[test]
    fn test_dimension_properties() {
        let dim = Dimension::new("x", (0.0, 9.0), 5.0);
        assert_eq!(dim.num_cells(), 10);
        assert_eq!(dim.num_tiles(), 2);

        let dim2 = Dimension::new("y", (0.0, 9.0), 3.0);
        assert_eq!(dim2.num_cells(), 10);
        assert_eq!(dim2.num_tiles(), 4); // ceil(10/3) = 4
    }

    #[test]
    fn test_schema_accessors() {
        let schema = dense_2d_schema();
        assert!(schema.is_dense());
        assert_eq!(schema.dimensions().len(), 2);
        assert_eq!(schema.attributes().len(), 1);

        let sparse = sparse_2d_schema();
        assert!(!sparse.is_dense());
    }

    #[test]
    fn test_num_cells() {
        let schema = dense_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        assert_eq!(arr.num_cells(), 0); // all NaN initially

        arr.write_dense(&[1.0, 2.0, 3.0, 4.0], &[(0, 1), (0, 1)])
            .expect("write");
        assert_eq!(arr.num_cells(), 4);
        assert_eq!(arr.total_capacity(), 16);
    }

    #[test]
    fn test_dense_write_on_sparse_errors() {
        let schema = sparse_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        let result = arr.write_dense(&[1.0], &[(0, 0), (0, 0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_write_on_dense_errors() {
        let schema = dense_2d_schema();
        let config = TileDBConfig::default();
        let mut arr = TileDBArray::create(schema, config).expect("create");

        let result = arr.write_sparse(&[vec![0.0], vec![0.0]], &[1.0]);
        assert!(result.is_err());
    }
}
