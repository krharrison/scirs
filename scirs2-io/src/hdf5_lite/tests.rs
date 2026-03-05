//! Tests for HDF5-lite pure Rust reader
//!
//! These tests create minimal valid HDF5 binary structures in memory and verify
//! the parser handles them correctly. We test superblock parsing, data type
//! identification, dataspace parsing, and error handling.

use super::*;
use crate::error::IoError;

// Re-define message type constants for testing (these are private in reader.rs)
const MSG_LINK: u16 = 0x0006;
const MSG_DATASPACE: u16 = 0x0001;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_DATA_LAYOUT: u16 = 0x0008;

/// Build a minimal valid HDF5 v0 superblock followed by an empty v1 object header
fn build_minimal_hdf5_v0() -> Vec<u8> {
    let mut buf = Vec::new();

    // HDF5 signature (8 bytes)
    buf.extend_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);

    // Superblock version: 0
    buf.push(0); // sb version
    buf.push(0); // free-space version
    buf.push(0); // root group version
    buf.push(0); // reserved
    buf.push(0); // shared header msg version
    buf.push(8); // offset size
    buf.push(8); // length size
    buf.push(0); // reserved

    // Group leaf node K (2 bytes)
    buf.extend_from_slice(&4u16.to_le_bytes());
    // Group internal node K (2 bytes)
    buf.extend_from_slice(&16u16.to_le_bytes());

    // File consistency flags (4 bytes)
    buf.extend_from_slice(&0u32.to_le_bytes());

    // Base address (8 bytes) = 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Free-space info address (8 bytes) = undefined
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    // End of file address (8 bytes)
    buf.extend_from_slice(&512u64.to_le_bytes());
    // Driver info block address (8 bytes) = undefined
    buf.extend_from_slice(&u64::MAX.to_le_bytes());

    // Root group symbol table entry:
    // Link name offset (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Object header address (8 bytes) - point to address 128
    let oh_address = 128u64;
    buf.extend_from_slice(&oh_address.to_le_bytes());
    // Cache type (4 bytes)
    buf.extend_from_slice(&0u32.to_le_bytes());
    // Reserved (4 bytes)
    buf.extend_from_slice(&0u32.to_le_bytes());
    // Scratch pad (16 bytes)
    buf.extend_from_slice(&[0u8; 16]);

    // Pad to object header address
    while buf.len() < oh_address as usize {
        buf.push(0);
    }

    // V1 Object header at address 128
    buf.push(1); // version
    buf.push(0); // reserved
    buf.extend_from_slice(&0u16.to_le_bytes()); // num messages = 0
    buf.extend_from_slice(&1u32.to_le_bytes()); // ref count
    buf.extend_from_slice(&0u32.to_le_bytes()); // header data size = 0

    // Pad to 512 bytes
    while buf.len() < 512 {
        buf.push(0);
    }

    buf
}

/// Build a minimal HDF5 v2 superblock
fn build_minimal_hdf5_v2() -> Vec<u8> {
    let mut buf = Vec::new();

    // HDF5 signature
    buf.extend_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);

    // Superblock version: 2
    buf.push(2);
    // Offset size: 8
    buf.push(8);
    // Length size: 8
    buf.push(8);
    // Consistency flags: 0
    buf.push(0);

    // Base address (8 bytes) = 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Superblock extension address (8 bytes) = undefined
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    // End of file address (8 bytes) = 512
    buf.extend_from_slice(&512u64.to_le_bytes());
    // Root group object header address (8 bytes) = 64
    let oh_address = 64u64;
    buf.extend_from_slice(&oh_address.to_le_bytes());
    // Superblock checksum (4 bytes)
    buf.extend_from_slice(&0u32.to_le_bytes());

    // Pad to object header address
    while buf.len() < oh_address as usize {
        buf.push(0);
    }

    // V2 Object header "OHDR" at address 64
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0); // flags (chunk size = 1 byte)
    buf.push(0); // chunk#0 size = 0 (no messages)

    // Pad to 512
    while buf.len() < 512 {
        buf.push(0);
    }

    buf
}

#[test]
fn test_invalid_signature() {
    let data = vec![0u8; 64];
    let result = Hdf5Reader::from_bytes(data);
    assert!(result.is_err());
    if let Err(IoError::FormatError(msg)) = result {
        assert!(
            msg.contains("signature"),
            "Expected signature error, got: {msg}"
        );
    }
}

#[test]
fn test_too_short_file() {
    let data = vec![0x89, 0x48, 0x44, 0x46]; // partial signature
    let result = Hdf5Reader::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_parse_v0_superblock() {
    let data = build_minimal_hdf5_v0();
    let reader = Hdf5Reader::from_bytes(data);
    assert!(reader.is_ok(), "Failed to parse v0: {:?}", reader.err());
    let reader = reader.expect("already checked");
    let sb = reader.superblock();
    assert_eq!(sb.version, 0);
    assert_eq!(sb.offset_size, 8);
    assert_eq!(sb.length_size, 8);
    assert_eq!(sb.base_address, 0);
}

#[test]
fn test_parse_v2_superblock() {
    let data = build_minimal_hdf5_v2();
    let reader = Hdf5Reader::from_bytes(data);
    assert!(reader.is_ok(), "Failed to parse v2: {:?}", reader.err());
    let reader = reader.expect("already checked");
    let sb = reader.superblock();
    assert_eq!(sb.version, 2);
    assert_eq!(sb.offset_size, 8);
    assert_eq!(sb.length_size, 8);
}

#[test]
fn test_unsupported_superblock_version() {
    let mut data = vec![0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
    data.push(99); // unsupported version
    data.extend_from_slice(&[0u8; 128]);
    let result = Hdf5Reader::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_v0_root_group_empty() {
    let data = build_minimal_hdf5_v0();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");
    let root = reader.root_group();
    assert!(root.is_ok(), "Failed root group: {:?}", root.err());
    let root = root.expect("already checked");
    assert_eq!(root.path, "/");
    assert!(root.children.is_empty());
}

#[test]
fn test_v2_root_group_empty() {
    let data = build_minimal_hdf5_v2();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");
    let root = reader.root_group();
    assert!(root.is_ok(), "Failed root group: {:?}", root.err());
    let root = root.expect("already checked");
    assert_eq!(root.path, "/");
}

#[test]
fn test_file_not_found() {
    let result = Hdf5Reader::open("/nonexistent/path/to/file.h5");
    assert!(result.is_err());
    if let Err(IoError::FileNotFound(_)) = result {
        // Expected
    } else {
        panic!("Expected FileNotFound error");
    }
}

#[test]
fn test_list_all_on_empty() {
    let data = build_minimal_hdf5_v0();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");
    let nodes = reader.list_all();
    assert!(nodes.is_ok());
    let nodes = nodes.expect("already checked");
    assert!(nodes.is_empty());
}

#[test]
fn test_datatype_element_sizes() {
    assert_eq!(Hdf5DataType::Int8.element_size(), 1);
    assert_eq!(Hdf5DataType::Int16.element_size(), 2);
    assert_eq!(Hdf5DataType::Int32.element_size(), 4);
    assert_eq!(Hdf5DataType::Int64.element_size(), 8);
    assert_eq!(Hdf5DataType::UInt8.element_size(), 1);
    assert_eq!(Hdf5DataType::UInt16.element_size(), 2);
    assert_eq!(Hdf5DataType::UInt32.element_size(), 4);
    assert_eq!(Hdf5DataType::UInt64.element_size(), 8);
    assert_eq!(Hdf5DataType::Float32.element_size(), 4);
    assert_eq!(Hdf5DataType::Float64.element_size(), 8);
    assert_eq!(Hdf5DataType::FixedString(10).element_size(), 10);
    assert_eq!(Hdf5DataType::VarString.element_size(), 16);
    assert_eq!(Hdf5DataType::Opaque(32).element_size(), 32);
}

#[test]
fn test_value_len_and_empty() {
    let v = Hdf5Value::Float64(vec![1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());

    let v_empty = Hdf5Value::Int32(Vec::new());
    assert_eq!(v_empty.len(), 0);
    assert!(v_empty.is_empty());
}

#[test]
fn test_value_as_f64_conversions() {
    let v = Hdf5Value::Int32(vec![1, 2, 3]);
    let f = v.as_f64();
    assert!(f.is_some());
    let f = f.expect("already checked");
    assert_eq!(f, vec![1.0, 2.0, 3.0]);

    let v = Hdf5Value::UInt8(vec![10, 20]);
    let f = v.as_f64();
    assert!(f.is_some());
    let f = f.expect("already checked");
    assert_eq!(f, vec![10.0, 20.0]);

    let v = Hdf5Value::Float32(vec![1.5, 2.5]);
    let f = v.as_f64();
    assert!(f.is_some());
    let f = f.expect("already checked");
    assert!((f[0] - 1.5).abs() < 1e-5);
    assert!((f[1] - 2.5).abs() < 1e-5);

    let v = Hdf5Value::Strings(vec!["test".to_string()]);
    assert!(v.as_f64().is_none());
}

#[test]
fn test_node_type_equality() {
    assert_eq!(Hdf5NodeType::Group, Hdf5NodeType::Group);
    assert_eq!(Hdf5NodeType::Dataset, Hdf5NodeType::Dataset);
    assert_ne!(Hdf5NodeType::Group, Hdf5NodeType::Dataset);
}

#[test]
fn test_group_attributes_empty() {
    let data = build_minimal_hdf5_v0();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");
    let root = reader.root_group().expect("root");
    assert!(root.attributes.is_empty());
}

#[test]
fn test_read_dataset_at_root_path_error() {
    let data = build_minimal_hdf5_v0();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");
    let result = reader.read_dataset("/");
    assert!(result.is_err());
}

#[test]
fn test_read_nonexistent_dataset() {
    let data = build_minimal_hdf5_v0();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");
    let result = reader.read_dataset("/nonexistent");
    assert!(result.is_err());
}

/// Build an HDF5 v2 file with a dataset containing f64 contiguous data
/// using link messages
fn build_hdf5_v2_with_dataset() -> Vec<u8> {
    let mut buf = Vec::new();

    // HDF5 signature
    buf.extend_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);

    // Superblock version 2
    buf.push(2);
    buf.push(8); // offset size
    buf.push(8); // length size
    buf.push(0); // flags

    // Base address
    buf.extend_from_slice(&0u64.to_le_bytes());
    // Extension address (undefined)
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    // EOF address
    buf.extend_from_slice(&1024u64.to_le_bytes());
    // Root group object header at 64
    buf.extend_from_slice(&64u64.to_le_bytes());
    // Checksum
    buf.extend_from_slice(&0u32.to_le_bytes());

    // Pad to 64
    while buf.len() < 64 {
        buf.push(0);
    }

    // Now build a v2 object header for root group at address 64
    // with a link message pointing to a dataset at address 256
    let dataset_oh_addr = 256u64;

    // Build link message data:
    // version=1, flags=0 (hard link, 1-byte name length)
    // name length = 4 ("data")
    // name = "data"
    // address = 256
    let mut link_data = Vec::new();
    link_data.push(1); // version
    link_data.push(0); // flags (name size = 1 byte, hard link)
    link_data.push(4); // name length (1 byte)
    link_data.extend_from_slice(b"data"); // name
    link_data.extend_from_slice(&dataset_oh_addr.to_le_bytes()); // hard link address

    // Build the OHDR for root group
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0); // flags (1-byte chunk size)

    // We need to encode the link message
    // msg_type=6 (LINK), msg_size, msg_flags=0
    let msg_size = link_data.len() as u16;
    let chunk_size = 1 + 2 + 1 + link_data.len(); // type(1) + size(2) + flags(1) + data
    buf.push(chunk_size as u8); // chunk#0 size

    buf.push(MSG_LINK as u8); // msg type
    buf.extend_from_slice(&msg_size.to_le_bytes()); // msg size
    buf.push(0); // msg flags
    buf.extend_from_slice(&link_data); // msg data

    // Pad to 256
    while buf.len() < dataset_oh_addr as usize {
        buf.push(0);
    }

    // Build dataset object header at 256 with:
    // - dataspace message (shape=[3])
    // - datatype message (float64)
    // - data layout message (contiguous at address 512, size=24)
    let data_addr = 512u64;
    let data_size = 24u64; // 3 * 8 bytes

    // Dataspace message: version=2, ndims=1, flags=0, dim=3
    let mut ds_msg = Vec::new();
    ds_msg.push(2); // version
    ds_msg.push(1); // ndims
    ds_msg.push(0); // flags
    ds_msg.extend_from_slice(&3u64.to_le_bytes()); // dim[0] = 3

    // Datatype message: class=1 (float), version=1, size=8
    let mut dt_msg = Vec::new();
    dt_msg.push(0x11); // class=1, version=1 => (1 << 4) | 1
    dt_msg.push(0x20); // bit field byte 1 (byte order=LE, padding)
    dt_msg.push(0); // bit field byte 2
    dt_msg.push(0); // bit field byte 3
    dt_msg.extend_from_slice(&8u32.to_le_bytes()); // size=8
                                                   // Float properties: bit offset, bit precision, exponent location/size, mantissa location/size
    dt_msg.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    dt_msg.extend_from_slice(&64u16.to_le_bytes()); // bit precision
    dt_msg.push(52); // exponent location
    dt_msg.push(11); // exponent size
    dt_msg.push(0); // mantissa location
    dt_msg.push(52); // mantissa size
    dt_msg.extend_from_slice(&1023u32.to_le_bytes()); // exponent bias

    // Layout message: version=3, class=1 (contiguous), address, size
    let mut layout_msg = Vec::new();
    layout_msg.push(3); // version
    layout_msg.push(1); // class = contiguous
    layout_msg.extend_from_slice(&data_addr.to_le_bytes()); // address
    layout_msg.extend_from_slice(&data_size.to_le_bytes()); // size

    // Build OHDR for dataset
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0); // flags

    // Calculate chunk size
    let total_msg_size = (1 + 2 + 1) * 3 + ds_msg.len() + dt_msg.len() + layout_msg.len();
    buf.push(total_msg_size as u8); // chunk size

    // Dataspace message (type=1)
    buf.push(MSG_DATASPACE as u8);
    buf.extend_from_slice(&(ds_msg.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(&ds_msg);

    // Datatype message (type=3)
    buf.push(MSG_DATATYPE as u8);
    buf.extend_from_slice(&(dt_msg.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(&dt_msg);

    // Layout message (type=8)
    buf.push(MSG_DATA_LAYOUT as u8);
    buf.extend_from_slice(&(layout_msg.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(&layout_msg);

    // Pad to data address
    while buf.len() < data_addr as usize {
        buf.push(0);
    }

    // Write 3 float64 values: 1.0, 2.0, 3.0
    buf.extend_from_slice(&1.0f64.to_le_bytes());
    buf.extend_from_slice(&2.0f64.to_le_bytes());
    buf.extend_from_slice(&3.0f64.to_le_bytes());

    // Pad to 1024
    while buf.len() < 1024 {
        buf.push(0);
    }

    buf
}

#[test]
fn test_v2_with_dataset_root_group() {
    let data = build_hdf5_v2_with_dataset();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");

    let root = reader.root_group().expect("root group");
    assert_eq!(root.path, "/");
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0], "data");
}

#[test]
fn test_v2_with_dataset_list_all() {
    let data = build_hdf5_v2_with_dataset();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");

    let nodes = reader.list_all().expect("list");
    assert!(!nodes.is_empty());
    assert_eq!(nodes[0].name, "data");
    assert_eq!(nodes[0].path, "/data");
}

#[test]
fn test_v2_read_dataset() {
    let data = build_hdf5_v2_with_dataset();
    let reader = Hdf5Reader::from_bytes(data).expect("valid file");

    let dataset = reader.read_dataset("/data").expect("dataset");
    assert_eq!(dataset.name, "data");
    assert_eq!(dataset.shape, vec![3]);
    assert_eq!(dataset.dtype, Hdf5DataType::Float64);

    if let Hdf5Value::Float64(values) = &dataset.data {
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
    } else {
        panic!("Expected Float64 data, got {:?}", dataset.data);
    }
}

#[test]
fn test_hdf5_value_raw() {
    let v = Hdf5Value::Raw(vec![1, 2, 3]);
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
    assert!(v.as_f64().is_none());
}

#[test]
fn test_datatype_compound_element_size() {
    let dt = Hdf5DataType::Compound(vec![
        ("x".to_string(), Hdf5DataType::Float64, 0),
        ("y".to_string(), Hdf5DataType::Float64, 8),
    ]);
    assert_eq!(dt.element_size(), 16);
}

#[test]
fn test_datatype_array_element_size() {
    let dt = Hdf5DataType::Array(Box::new(Hdf5DataType::Int32), vec![3, 4]);
    assert_eq!(dt.element_size(), 48); // 3*4*4
}

#[test]
fn test_datatype_unknown_element_size() {
    let dt = Hdf5DataType::Unknown(99, 12);
    assert_eq!(dt.element_size(), 12);
}
