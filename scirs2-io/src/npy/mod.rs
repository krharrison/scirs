//! NumPy NPY/NPZ binary file format support
//!
//! Provides reading and writing of NumPy's binary file formats:
//! - `.npy` files: single arrays with header metadata
//! - `.npz` files: ZIP archives of multiple named `.npy` arrays
//!
//! # Supported dtypes
//! - `f32` (float32), `f64` (float64)
//! - `i32` (int32), `i64` (int64)
//! - Both little-endian and big-endian byte orders
//!
//! # Examples
//!
//! ## Writing and reading a single array
//!
//! ```rust,no_run
//! use scirs2_io::npy::{write_npy_f64, read_npy, NpyArray};
//!
//! // Write f64 data
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! write_npy_f64("array.npy", &data).expect("Write failed");
//!
//! // Read it back
//! let array = read_npy("array.npy").expect("Read failed");
//! let values = array.as_f64().expect("Type mismatch");
//! assert_eq!(values, &[1.0, 2.0, 3.0, 4.0, 5.0]);
//! ```
//!
//! ## Working with NPZ archives
//!
//! ```rust,no_run
//! use scirs2_io::npy::{write_npz, read_npz, NpzArchive, NpyArray};
//!
//! let mut archive = NpzArchive::new();
//! archive.add("weights", NpyArray::Float64 {
//!     data: vec![0.1, 0.2, 0.3],
//!     shape: vec![3],
//! });
//! archive.add("bias", NpyArray::Float32 {
//!     data: vec![1.0, 2.0],
//!     shape: vec![2],
//! });
//!
//! write_npz("model.npz", &archive).expect("Write failed");
//! let loaded = read_npz("model.npz").expect("Read failed");
//! assert_eq!(loaded.len(), 2);
//! ```

pub mod npy_reader;
pub mod npy_writer;
pub mod npz;
pub mod types;

pub use npy_reader::read_npy;
pub use npy_writer::{
    write_npy, write_npy_f32, write_npy_f64, write_npy_f64_2d, write_npy_i32, write_npy_i64,
};
pub use npz::{read_npz, write_npz, NpzArchive};
pub use types::{ByteOrder, NpyArray, NpyDtype, NpyHeader};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npy_roundtrip_f64() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_f64");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_f64.npy");

        let data = vec![1.0, 2.5, std::f64::consts::PI, -0.5, 100.0];
        write_npy_f64(&path, &data).expect("Failed to write");

        let array = read_npy(&path).expect("Failed to read");
        assert_eq!(array.shape(), &[5]);
        let values = array.as_f64().expect("Type mismatch");
        for (a, b) in data.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-10);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_roundtrip_f32() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_f32");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_f32.npy");

        let data = vec![1.0f32, 2.5, std::f32::consts::PI, -0.5];
        write_npy_f32(&path, &data).expect("Failed to write");

        let array = read_npy(&path).expect("Failed to read");
        assert_eq!(array.shape(), &[4]);
        let values = array.as_f32().expect("Type mismatch");
        for (a, b) in data.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_roundtrip_i32() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_i32");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_i32.npy");

        let data = vec![10i32, -20, 30, 0, 42];
        write_npy_i32(&path, &data).expect("Failed to write");

        let array = read_npy(&path).expect("Failed to read");
        let values = array.as_i32().expect("Type mismatch");
        assert_eq!(values, &data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_roundtrip_i64() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_i64");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_i64.npy");

        let data = vec![100i64, -200, 300, 0, i64::MAX, i64::MIN];
        write_npy_i64(&path, &data).expect("Failed to write");

        let array = read_npy(&path).expect("Failed to read");
        let values = array.as_i64().expect("Type mismatch");
        assert_eq!(values, &data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_2d_array() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_2d");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_2d.npy");

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        write_npy_f64_2d(&path, &data, 2, 3).expect("Failed to write");

        let array = read_npy(&path).expect("Failed to read");
        assert_eq!(array.shape(), &[2, 3]);
        let values = array.as_f64().expect("Type mismatch");
        assert_eq!(values, &data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_type_mismatch() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_mismatch");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_mismatch.npy");

        write_npy_f64(&path, &[1.0, 2.0]).expect("Failed to write");
        let array = read_npy(&path).expect("Failed to read");

        assert!(array.as_f64().is_ok());
        assert!(array.as_f32().is_err());
        assert!(array.as_i32().is_err());
        assert!(array.as_i64().is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_2d_shape_mismatch() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_shape_mismatch");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad_shape.npy");

        let result = write_npy_f64_2d(&path, &[1.0, 2.0, 3.0], 2, 3);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npz_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_npz");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.npz");

        let mut archive = NpzArchive::new();
        archive.add(
            "weights",
            NpyArray::Float64 {
                data: vec![0.1, 0.2, 0.3, 0.4],
                shape: vec![2, 2],
            },
        );
        archive.add(
            "bias",
            NpyArray::Float32 {
                data: vec![1.0, 2.0],
                shape: vec![2],
            },
        );
        archive.add(
            "indices",
            NpyArray::Int32 {
                data: vec![0, 1, 2, 3],
                shape: vec![4],
            },
        );

        write_npz(&path, &archive).expect("Failed to write");

        let loaded = read_npz(&path).expect("Failed to read");
        assert_eq!(loaded.len(), 3);

        let weights = loaded.get("weights").expect("Missing weights");
        assert_eq!(weights.shape(), &[2, 2]);
        let w_data = weights.as_f64().expect("Type error");
        assert!((w_data[0] - 0.1).abs() < 1e-10);

        let bias = loaded.get("bias").expect("Missing bias");
        assert_eq!(bias.shape(), &[2]);

        let indices = loaded.get("indices").expect("Missing indices");
        assert_eq!(indices.as_i32().expect("Type error"), &[0, 1, 2, 3]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npz_empty() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_npz_empty");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.npz");

        let archive = NpzArchive::new();
        write_npz(&path, &archive).expect("Failed to write");

        let loaded = read_npz(&path).expect("Failed to read");
        assert!(loaded.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npz_missing_array() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_npz_missing");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("missing.npz");

        let mut archive = NpzArchive::new();
        archive.add(
            "x",
            NpyArray::Float64 {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        write_npz(&path, &archive).expect("Failed to write");

        let loaded = read_npz(&path).expect("Failed to read");
        assert!(loaded.get("x").is_ok());
        assert!(loaded.get("nonexistent").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_npy_large_array() {
        let dir = std::env::temp_dir().join("scirs2_npy_test_large");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("large.npy");

        let data: Vec<f64> = (0..10000).map(|x| x as f64 * 0.01).collect();
        write_npy_f64(&path, &data).expect("Failed to write");

        let array = read_npy(&path).expect("Failed to read");
        let values = array.as_f64().expect("Type error");
        assert_eq!(values.len(), 10000);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[9999] - 99.99).abs() < 1e-10);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_header_parsing() {
        let header_str = "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }";
        let header = types::parse_header_dict(header_str).expect("Failed to parse");
        assert_eq!(header.dtype, NpyDtype::Float64);
        assert!(!header.fortran_order);
        assert_eq!(header.shape, vec![3, 4]);
    }

    #[test]
    fn test_header_1d_shape() {
        let header_str = "{'descr': '<i4', 'fortran_order': False, 'shape': (10,), }";
        let header = types::parse_header_dict(header_str).expect("Failed to parse");
        assert_eq!(header.dtype, NpyDtype::Int32);
        assert_eq!(header.shape, vec![10]);
    }

    #[test]
    fn test_dtype_parsing() {
        let (dtype, order) = NpyDtype::from_descr("<f4").expect("Parse failed");
        assert_eq!(dtype, NpyDtype::Float32);
        assert_eq!(order, ByteOrder::LittleEndian);

        let (dtype, order) = NpyDtype::from_descr(">f8").expect("Parse failed");
        assert_eq!(dtype, NpyDtype::Float64);
        assert_eq!(order, ByteOrder::BigEndian);

        let (dtype, _) = NpyDtype::from_descr("<i4").expect("Parse failed");
        assert_eq!(dtype, NpyDtype::Int32);

        let (dtype, _) = NpyDtype::from_descr("<i8").expect("Parse failed");
        assert_eq!(dtype, NpyDtype::Int64);
    }

    #[test]
    fn test_dtype_invalid() {
        assert!(NpyDtype::from_descr("abc").is_err());
        assert!(NpyDtype::from_descr("<u4").is_err()); // unsigned not supported
    }
}
