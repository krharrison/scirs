//! # ONNX Ecosystem Interoperability
//!
//! Pure Rust implementation of ONNX model format support for scirs2-autograd.
//! Enables import and export of computation graphs in ONNX-compatible JSON format
//! without requiring protobuf or external ONNX library dependencies.
//!
//! ## Features
//!
//! - **Export**: Convert scirs2-autograd models to ONNX-compatible JSON
//! - **Import**: Load ONNX JSON models into scirs2-autograd data structures
//! - **Builder API**: Fluent graph builder for constructing ONNX graphs programmatically
//! - **Tensor Conversion**: Bidirectional conversion between OnnxTensor and ndarray types
//! - **Validation**: Graph structure validation on build
//! - **Weight I/O**: Export and import model weights as ONNX tensors
//!
//! ## Quick Start
//!
//! ### Building an ONNX graph
//!
//! ```rust
//! use scirs2_autograd::onnx::*;
//!
//! // Build a simple MLP graph
//! let graph = OnnxGraphBuilder::new("mlp")
//!     .add_input("x", &[-1, 784], OnnxDataType::Float32)
//!     .add_input("w1", &[784, 256], OnnxDataType::Float32)
//!     .add_input("b1", &[256], OnnxDataType::Float32)
//!     .add_matmul("mm1", "x", "w1", "h_raw")
//!     .add_add("add1", "h_raw", "b1", "h_biased")
//!     .add_relu("relu1", "h_biased", "h")
//!     .add_output("h", &[-1, 256], OnnxDataType::Float32)
//!     .build()
//!     .expect("Failed to build graph");
//!
//! // Export to JSON
//! let json = export_to_onnx_json(&graph).expect("export failed");
//!
//! // Import back
//! let restored = import_from_onnx_json(&json).expect("import failed");
//! assert_eq!(restored.nodes.len(), 3);
//! ```
//!
//! ### Converting tensors
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_autograd::onnx::*;
//!
//! // ndarray -> OnnxTensor
//! let weights = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//!     .expect("shape error");
//! let tensor = array2_to_onnx_tensor("layer.weight", &weights);
//!
//! // OnnxTensor -> ndarray
//! let restored = onnx_tensor_to_array2(&tensor).expect("conversion failed");
//! assert_eq!(weights, restored);
//! ```

pub mod builder;
pub mod convert;
pub mod error;
pub mod types;

// Re-export core types
pub use builder::OnnxGraphBuilder;
pub use convert::{
    array1_to_onnx_tensor, array1_to_onnx_tensor_f32, array2_to_onnx_tensor,
    array2_to_onnx_tensor_f32, arrayd_to_onnx_tensor, export_weights, import_weights,
    onnx_tensor_to_array1, onnx_tensor_to_array1_f32, onnx_tensor_to_array2,
    onnx_tensor_to_array2_f32, onnx_tensor_to_arrayd,
};
pub use error::{OnnxError, OnnxResult};
pub use io::{
    export_model_to_json, export_to_onnx_json, export_to_onnx_json_compact,
    export_to_onnx_json_file, get_onnx_json_summary, import_from_onnx_json,
    import_from_onnx_json_file, import_model_from_json, import_model_from_json_file,
    validate_onnx_json, OnnxModelSummary,
};
pub use types::{
    ConvAttributes, GemmAttributes, OnnxAttribute, OnnxDataType, OnnxGraph, OnnxModel, OnnxNode,
    OnnxOpsetImport, OnnxTensor, PoolAttributes, ONNX_IR_VERSION, ONNX_OPSET_VERSION,
    ONNX_PRODUCER_NAME, ONNX_PRODUCER_VERSION,
};

pub mod io;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_full_mlp_export_import() {
        // Build a 2-layer MLP with weights
        let w1_data: Vec<f64> = (0..6).map(|i| (i as f64) * 0.1).collect();
        let w2_data: Vec<f64> = (0..6).map(|i| (i as f64) * 0.2 - 0.5).collect();
        let b1_data: Vec<f64> = vec![0.01, 0.02, 0.03];
        let b2_data: Vec<f64> = vec![-0.1, 0.1];

        let graph = OnnxGraphBuilder::new("mlp_2layer")
            .add_input("input", &[-1, 2], OnnxDataType::Float64)
            .add_initializer(OnnxTensor::from_f64("w1", &[2, 3], w1_data.clone()))
            .add_initializer(OnnxTensor::from_f64("b1", &[3], b1_data.clone()))
            .add_initializer(OnnxTensor::from_f64("w2", &[3, 2], w2_data.clone()))
            .add_initializer(OnnxTensor::from_f64("b2", &[2], b2_data.clone()))
            .add_matmul("mm1", "input", "w1", "h1_raw")
            .add_add("add_b1", "h1_raw", "b1", "h1_biased")
            .add_relu("relu1", "h1_biased", "h1")
            .add_matmul("mm2", "h1", "w2", "h2_raw")
            .add_add("add_b2", "h2_raw", "b2", "logits")
            .add_sigmoid("output_act", "logits", "output")
            .add_output("output", &[-1, 2], OnnxDataType::Float64)
            .build()
            .expect("build failed");

        // Export to JSON
        let json = export_to_onnx_json(&graph).expect("export failed");

        // Validate
        assert!(validate_onnx_json(&json).is_ok());

        // Import back
        let restored = import_from_onnx_json(&json).expect("import failed");

        // Verify structure
        assert_eq!(restored.name, "mlp_2layer");
        assert_eq!(restored.nodes.len(), 6);
        assert_eq!(restored.inputs.len(), 1);
        assert_eq!(restored.outputs.len(), 1);
        assert_eq!(restored.initializers.len(), 4);

        // Verify weights survived roundtrip
        let w1_tensor = restored.get_initializer("w1").expect("w1 not found");
        assert_eq!(w1_tensor.double_data, w1_data);

        // Verify node ordering
        assert_eq!(restored.nodes[0].op_type, "MatMul");
        assert_eq!(restored.nodes[2].op_type, "Relu");
        assert_eq!(restored.nodes[5].op_type, "Sigmoid");
    }

    #[test]
    fn test_conv_network_graph() {
        let graph = OnnxGraphBuilder::new("conv_net")
            .add_input("images", &[-1, 1, 28, 28], OnnxDataType::Float32)
            .add_input("conv1_w", &[16, 1, 3, 3], OnnxDataType::Float32)
            .add_input("conv1_b", &[16], OnnxDataType::Float32)
            .add_conv_with_bias(
                "conv1",
                "images",
                "conv1_w",
                "conv1_b",
                "conv1_out",
                ConvAttributes::new(vec![3, 3]).with_pads(vec![1, 1, 1, 1]),
            )
            .add_relu("relu1", "conv1_out", "relu1_out")
            .add_max_pool(
                "pool1",
                "relu1_out",
                "pool1_out",
                PoolAttributes::new(vec![2, 2]).with_strides(vec![2, 2]),
            )
            .add_flatten("flatten", "pool1_out", "flat", 1)
            .add_output("flat", &[-1, 3136], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.nodes[0].op_type, "Conv");

        // Verify conv attributes
        let conv_node = &graph.nodes[0];
        let ks = conv_node
            .get_attribute("kernel_shape")
            .expect("no kernel_shape");
        assert_eq!(ks.as_ints(), Some(&[3i64, 3][..]));
    }

    #[test]
    fn test_onnx_model_metadata() {
        let graph = OnnxGraphBuilder::new("meta_test")
            .add_input("x", &[1], OnnxDataType::Float32)
            .add_relu("r", "x", "y")
            .add_output("y", &[1], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let model = OnnxModel::new(graph)
            .with_metadata("framework", "scirs2")
            .with_metadata("task", "classification")
            .with_doc_string("Test model for metadata");

        let json = export_model_to_json(&model).expect("export failed");
        let restored = import_model_from_json(&json).expect("import failed");

        assert_eq!(restored.ir_version, ONNX_IR_VERSION);
        assert_eq!(restored.producer_name, ONNX_PRODUCER_NAME);
        assert_eq!(
            restored.metadata.get("framework"),
            Some(&"scirs2".to_string())
        );
        assert_eq!(
            restored.doc_string,
            Some("Test model for metadata".to_string())
        );
    }

    #[test]
    fn test_weight_tensor_conversion_in_graph() {
        // Create weights as ndarray
        let w = Array2::from_shape_vec((4, 3), (0..12).map(|i| i as f64 * 0.5).collect())
            .expect("shape");
        let b = scirs2_core::ndarray::Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Convert to ONNX tensors
        let w_tensor = array2_to_onnx_tensor("dense.weight", &w);
        let b_tensor = array1_to_onnx_tensor("dense.bias", &b);

        // Build graph with initializers
        let graph = OnnxGraphBuilder::new("dense_layer")
            .add_input("x", &[-1, 4], OnnxDataType::Float64)
            .add_initializer(w_tensor)
            .add_initializer(b_tensor)
            .add_matmul("mm", "x", "dense.weight", "mm_out")
            .add_add("bias_add", "mm_out", "dense.bias", "y")
            .add_output("y", &[-1, 3], OnnxDataType::Float64)
            .build()
            .expect("build failed");

        assert_eq!(graph.total_parameters(), 12 + 3);

        // Roundtrip weights
        let restored_w =
            onnx_tensor_to_array2(graph.get_initializer("dense.weight").expect("no w"))
                .expect("conv w");
        let restored_b = onnx_tensor_to_array1(graph.get_initializer("dense.bias").expect("no b"))
            .expect("conv b");

        assert_eq!(w, restored_w);
        assert_eq!(b, restored_b);
    }

    #[test]
    fn test_batch_norm_graph() {
        let graph = OnnxGraphBuilder::new("bn_test")
            .add_input("x", &[-1, 64, 32, 32], OnnxDataType::Float32)
            .add_input("bn_scale", &[64], OnnxDataType::Float32)
            .add_input("bn_bias", &[64], OnnxDataType::Float32)
            .add_input("bn_mean", &[64], OnnxDataType::Float32)
            .add_input("bn_var", &[64], OnnxDataType::Float32)
            .add_batch_norm(
                "bn1", "x", "bn_scale", "bn_bias", "bn_mean", "bn_var", "bn_out", 1e-5,
            )
            .add_output("bn_out", &[-1, 64, 32, 32], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "BatchNormalization");
        assert_eq!(graph.nodes[0].inputs.len(), 5);

        let eps = graph.nodes[0].get_attribute("epsilon").expect("no epsilon");
        assert_eq!(eps.as_float(), Some(1e-5));
    }

    #[test]
    fn test_summary_display() {
        let graph = OnnxGraphBuilder::new("summary_model")
            .add_input("x", &[-1, 100], OnnxDataType::Float32)
            .add_initializer(OnnxTensor::from_f32("w", &[100, 50], vec![0.0f32; 5000]))
            .add_matmul("mm", "x", "w", "h")
            .add_relu("r", "h", "y")
            .add_output("y", &[-1, 50], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let json = export_to_onnx_json(&graph).expect("export");
        let summary = get_onnx_json_summary(&json).expect("summary");

        assert_eq!(summary.num_nodes, 2);
        assert_eq!(summary.num_initializers, 1);
        assert_eq!(summary.total_parameters, 5000);
        assert!(summary.op_types.contains(&"MatMul".to_string()));
        assert!(summary.op_types.contains(&"Relu".to_string()));

        // Verify Display impl doesn't panic
        let display = format!("{}", summary);
        assert!(display.contains("ONNX Model Summary"));
        assert!(display.contains("5000"));
    }

    #[test]
    fn test_data_type_properties() {
        assert_eq!(OnnxDataType::Float32.code(), 1);
        assert_eq!(OnnxDataType::Float64.code(), 11);
        assert_eq!(OnnxDataType::Int64.code(), 7);
        assert_eq!(OnnxDataType::Bool.code(), 9);

        assert_eq!(OnnxDataType::from_code(1), Some(OnnxDataType::Float32));
        assert_eq!(OnnxDataType::from_code(99), None);

        assert_eq!(OnnxDataType::Float32.element_size(), 4);
        assert_eq!(OnnxDataType::Float64.element_size(), 8);
        assert_eq!(OnnxDataType::Bool.element_size(), 1);
        assert_eq!(OnnxDataType::Int16.element_size(), 2);

        assert_eq!(OnnxDataType::Float32.name(), "float32");
        assert_eq!(format!("{}", OnnxDataType::Float64), "float64");
    }

    #[test]
    fn test_onnx_tensor_num_elements() {
        let t1 = OnnxTensor::spec("a", &[2, 3, 4], OnnxDataType::Float32);
        assert_eq!(t1.num_elements(), Some(24));

        let t2 = OnnxTensor::spec("b", &[-1, 10], OnnxDataType::Float32);
        assert_eq!(t2.num_elements(), None); // dynamic dim

        let t3 = OnnxTensor::spec("c", &[], OnnxDataType::Float32);
        assert_eq!(t3.num_elements(), Some(1)); // scalar
    }

    #[test]
    fn test_attribute_accessors() {
        let attr_int = OnnxAttribute::Int(42);
        assert_eq!(attr_int.as_int(), Some(42));
        assert_eq!(attr_int.as_float(), None);

        let attr_float = OnnxAttribute::Float(3.14);
        assert_eq!(attr_float.as_float(), Some(3.14));

        let attr_str = OnnxAttribute::String("hello".to_string());
        assert_eq!(attr_str.as_string(), Some("hello"));

        let attr_ints = OnnxAttribute::Ints(vec![1, 2, 3]);
        assert_eq!(attr_ints.as_ints(), Some(&[1i64, 2, 3][..]));

        let attr_floats = OnnxAttribute::Floats(vec![1.0, 2.0]);
        assert_eq!(attr_floats.as_floats(), Some(&[1.0f64, 2.0][..]));

        let attr_strings = OnnxAttribute::Strings(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(attr_strings.as_strings().map(|s| s.len()), Some(2));
    }

    #[test]
    fn test_conv_attributes_roundtrip() {
        let attrs = ConvAttributes::new(vec![3, 3])
            .with_strides(vec![2, 2])
            .with_pads(vec![1, 1, 1, 1])
            .with_dilations(vec![1, 1])
            .with_group(4);

        let map = attrs.to_attributes();
        let restored = ConvAttributes::from_attributes(&map).expect("parse failed");

        assert_eq!(restored.kernel_shape, vec![3, 3]);
        assert_eq!(restored.strides, vec![2, 2]);
        assert_eq!(restored.pads, vec![1, 1, 1, 1]);
        assert_eq!(restored.group, 4);
    }

    #[test]
    fn test_gemm_attributes() {
        let attrs = GemmAttributes {
            alpha: 2.0,
            beta: 0.5,
            trans_a: 1,
            trans_b: 0,
        };

        let map = attrs.to_attributes();
        assert_eq!(map.get("alpha").and_then(|a| a.as_float()), Some(2.0));
        assert_eq!(map.get("beta").and_then(|a| a.as_float()), Some(0.5));
        assert_eq!(map.get("transA").and_then(|a| a.as_int()), Some(1));
        // transB should not be in map since it's 0 (default)
        assert!(map.get("transB").is_none());
    }

    #[test]
    fn test_node_with_attributes() {
        let node = OnnxNode::new(
            "Softmax",
            "softmax_1",
            vec!["logits".to_string()],
            vec!["probs".to_string()],
        )
        .with_attribute("axis", OnnxAttribute::Int(-1));

        assert_eq!(node.op_type, "Softmax");
        assert_eq!(
            node.get_attribute("axis").and_then(|a| a.as_int()),
            Some(-1)
        );
        assert!(node.get_attribute("missing").is_none());
    }

    #[test]
    fn test_graph_parameter_count() {
        let graph = OnnxGraphBuilder::new("param_count")
            .add_input("x", &[-1, 10], OnnxDataType::Float32)
            .add_initializer(OnnxTensor::from_f32("w1", &[10, 20], vec![0.0f32; 200]))
            .add_initializer(OnnxTensor::from_f32("b1", &[20], vec![0.0f32; 20]))
            .add_initializer(OnnxTensor::from_f32("w2", &[20, 5], vec![0.0f32; 100]))
            .add_initializer(OnnxTensor::from_f32("b2", &[5], vec![0.0f32; 5]))
            .add_matmul("mm1", "x", "w1", "h1")
            .add_add("add1", "h1", "b1", "h1b")
            .add_relu("r1", "h1b", "h1a")
            .add_matmul("mm2", "h1a", "w2", "h2")
            .add_add("add2", "h2", "b2", "out")
            .add_output("out", &[-1, 5], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        // 200 + 20 + 100 + 5 = 325
        assert_eq!(graph.total_parameters(), 325);
    }

    #[test]
    fn test_error_display() {
        let err = OnnxError::UnsupportedOp {
            op_type: "CustomOp".to_string(),
        };
        assert!(format!("{}", err).contains("CustomOp"));

        let err2 = OnnxError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        assert!(format!("{}", err2).contains("[2, 3]"));

        let err3 = OnnxError::InvalidAttribute {
            node: "conv1".to_string(),
            name: "kernel_shape".to_string(),
            reason: "must be non-empty".to_string(),
        };
        assert!(format!("{}", err3).contains("conv1"));
        assert!(format!("{}", err3).contains("kernel_shape"));
    }

    #[test]
    fn test_initializer_validation_mismatch() {
        // Create tensor with wrong data count for shape
        let bad_tensor = OnnxTensor::from_f32("bad", &[2, 3], vec![1.0f32, 2.0]); // 2 elements, should be 6
        let result = OnnxGraphBuilder::new("bad_init")
            .add_input("x", &[1, 2], OnnxDataType::Float32)
            .add_initializer(bad_tensor)
            .add_matmul("mm", "x", "bad", "y")
            .add_output("y", &[1, 3], OnnxDataType::Float32)
            .build();

        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should fail"));
        assert!(err_msg.contains("expects 6 elements but got 2"));
    }

    #[test]
    fn test_pool_attributes() {
        let attrs = PoolAttributes::new(vec![3, 3])
            .with_strides(vec![2, 2])
            .with_pads(vec![1, 1, 1, 1]);

        let map = attrs.to_attributes();
        assert_eq!(
            map.get("kernel_shape").and_then(|a| a.as_ints()),
            Some(&[3i64, 3][..])
        );
        assert_eq!(
            map.get("strides").and_then(|a| a.as_ints()),
            Some(&[2i64, 2][..])
        );
    }
}
