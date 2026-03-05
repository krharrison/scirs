//! ONNX JSON import/export functions.
//!
//! Provides serialization of ONNX models and graphs to/from JSON format
//! for interoperability with other frameworks.

use std::fs;
use std::path::Path;

use super::error::{OnnxError, OnnxResult};
use super::types::{
    OnnxGraph, OnnxModel, OnnxOpsetImport, ONNX_IR_VERSION, ONNX_OPSET_VERSION, ONNX_PRODUCER_NAME,
    ONNX_PRODUCER_VERSION,
};

/// Export an OnnxGraph to ONNX-compatible JSON string.
///
/// This wraps the graph in an OnnxModel with standard metadata
/// (IR version, opset, producer info).
///
/// # Example
/// ```
/// use scirs2_autograd::onnx::{OnnxGraphBuilder, OnnxDataType, export_to_onnx_json};
///
/// let graph = OnnxGraphBuilder::new("example")
///     .add_input("x", &[1, 10], OnnxDataType::Float32)
///     .add_relu("relu", "x", "y")
///     .add_output("y", &[1, 10], OnnxDataType::Float32)
///     .build()
///     .expect("build failed");
///
/// let json = export_to_onnx_json(&graph).expect("export failed");
/// assert!(json.contains("scirs2-autograd"));
/// ```
pub fn export_to_onnx_json(graph: &OnnxGraph) -> OnnxResult<String> {
    let model = OnnxModel::new(graph.clone());
    serde_json::to_string_pretty(&model).map_err(OnnxError::from)
}

/// Export an OnnxGraph to compact (non-pretty) JSON string.
pub fn export_to_onnx_json_compact(graph: &OnnxGraph) -> OnnxResult<String> {
    let model = OnnxModel::new(graph.clone());
    serde_json::to_string(&model).map_err(OnnxError::from)
}

/// Export an OnnxModel directly to JSON string.
pub fn export_model_to_json(model: &OnnxModel) -> OnnxResult<String> {
    serde_json::to_string_pretty(model).map_err(OnnxError::from)
}

/// Export an OnnxGraph to a JSON file on disk.
///
/// Creates parent directories if they do not exist.
pub fn export_to_onnx_json_file(graph: &OnnxGraph, path: &Path) -> OnnxResult<()> {
    let json = export_to_onnx_json(graph)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, json)?;
    Ok(())
}

/// Import an OnnxGraph from an ONNX-compatible JSON string.
///
/// Expects the JSON to contain an OnnxModel wrapper with `graph` field.
///
/// # Example
/// ```
/// use scirs2_autograd::onnx::{OnnxGraphBuilder, OnnxDataType, export_to_onnx_json, import_from_onnx_json};
///
/// let graph = OnnxGraphBuilder::new("roundtrip")
///     .add_input("x", &[1, 5], OnnxDataType::Float64)
///     .add_sigmoid("sig", "x", "y")
///     .add_output("y", &[1, 5], OnnxDataType::Float64)
///     .build()
///     .expect("build failed");
///
/// let json = export_to_onnx_json(&graph).expect("export failed");
/// let imported = import_from_onnx_json(&json).expect("import failed");
/// assert_eq!(imported.name, "roundtrip");
/// assert_eq!(imported.nodes.len(), 1);
/// ```
pub fn import_from_onnx_json(json: &str) -> OnnxResult<OnnxGraph> {
    let model: OnnxModel = serde_json::from_str(json).map_err(OnnxError::from)?;
    Ok(model.graph)
}

/// Import a full OnnxModel from JSON string.
pub fn import_model_from_json(json: &str) -> OnnxResult<OnnxModel> {
    serde_json::from_str(json).map_err(OnnxError::from)
}

/// Import an OnnxGraph from a JSON file on disk.
pub fn import_from_onnx_json_file(path: &Path) -> OnnxResult<OnnxGraph> {
    let json = fs::read_to_string(path)?;
    import_from_onnx_json(&json)
}

/// Import a full OnnxModel from a JSON file on disk.
pub fn import_model_from_json_file(path: &Path) -> OnnxResult<OnnxModel> {
    let json = fs::read_to_string(path)?;
    import_model_from_json(&json)
}

/// Validate that a JSON string can be parsed as a valid ONNX model.
///
/// Returns Ok(()) if valid, or an error with details.
pub fn validate_onnx_json(json: &str) -> OnnxResult<()> {
    let model: OnnxModel = serde_json::from_str(json).map_err(OnnxError::from)?;

    // Validate IR version
    if model.ir_version < 1 {
        return Err(OnnxError::ValidationError(format!(
            "Invalid IR version: {}",
            model.ir_version
        )));
    }

    // Validate opset
    if model.opset_imports.is_empty() {
        return Err(OnnxError::ValidationError(
            "Model has no opset imports".to_string(),
        ));
    }

    // Validate graph has inputs and outputs
    if model.graph.inputs.is_empty() {
        return Err(OnnxError::ValidationError(
            "Graph has no inputs".to_string(),
        ));
    }
    if model.graph.outputs.is_empty() {
        return Err(OnnxError::ValidationError(
            "Graph has no outputs".to_string(),
        ));
    }

    Ok(())
}

/// Get metadata summary from an ONNX JSON string without fully deserializing.
///
/// Returns (producer_name, producer_version, graph_name, num_nodes, num_params).
pub fn get_onnx_json_summary(json: &str) -> OnnxResult<OnnxModelSummary> {
    let model: OnnxModel = serde_json::from_str(json).map_err(OnnxError::from)?;

    let num_nodes = model.graph.nodes.len();
    let num_initializers = model.graph.initializers.len();
    let total_params = model.graph.total_parameters();

    let op_types: Vec<String> = model
        .graph
        .nodes
        .iter()
        .map(|n| n.op_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    Ok(OnnxModelSummary {
        producer_name: model.producer_name,
        producer_version: model.producer_version,
        ir_version: model.ir_version,
        opset_version: model.opset_imports.first().map(|o| o.version).unwrap_or(0),
        graph_name: model.graph.name,
        num_nodes,
        num_initializers,
        total_parameters: total_params,
        op_types,
    })
}

/// Summary information about an ONNX model
#[derive(Debug, Clone)]
pub struct OnnxModelSummary {
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// IR version
    pub ir_version: i64,
    /// Opset version
    pub opset_version: i64,
    /// Graph name
    pub graph_name: String,
    /// Number of computation nodes
    pub num_nodes: usize,
    /// Number of initializer tensors
    pub num_initializers: usize,
    /// Total parameter count
    pub total_parameters: usize,
    /// Unique op types used
    pub op_types: Vec<String>,
}

impl std::fmt::Display for OnnxModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ONNX Model Summary")?;
        writeln!(
            f,
            "  Producer: {} v{}",
            self.producer_name, self.producer_version
        )?;
        writeln!(f, "  IR Version: {}", self.ir_version)?;
        writeln!(f, "  Opset Version: {}", self.opset_version)?;
        writeln!(f, "  Graph: {}", self.graph_name)?;
        writeln!(f, "  Nodes: {}", self.num_nodes)?;
        writeln!(f, "  Initializers: {}", self.num_initializers)?;
        writeln!(f, "  Total Parameters: {}", self.total_parameters)?;
        write!(f, "  Op Types: {:?}", self.op_types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{OnnxDataType, OnnxGraphBuilder, OnnxTensor};

    #[test]
    fn test_roundtrip_json() {
        let graph = OnnxGraphBuilder::new("rt_test")
            .add_input("x", &[1, 10], OnnxDataType::Float32)
            .add_relu("relu", "x", "y")
            .add_output("y", &[1, 10], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let json = export_to_onnx_json(&graph).expect("export failed");
        let imported = import_from_onnx_json(&json).expect("import failed");

        assert_eq!(imported.name, "rt_test");
        assert_eq!(imported.nodes.len(), 1);
        assert_eq!(imported.nodes[0].op_type, "Relu");
        assert_eq!(imported.inputs.len(), 1);
        assert_eq!(imported.outputs.len(), 1);
    }

    #[test]
    fn test_roundtrip_with_initializers() {
        let weight_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = OnnxTensor::from_f32("w", &[2, 3], weight_data.clone());

        let graph = OnnxGraphBuilder::new("init_test")
            .add_input("x", &[-1, 2], OnnxDataType::Float32)
            .add_initializer(weight)
            .add_matmul("mm", "x", "w", "y")
            .add_output("y", &[-1, 3], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let json = export_to_onnx_json(&graph).expect("export failed");
        let imported = import_from_onnx_json(&json).expect("import failed");

        assert_eq!(imported.initializers.len(), 1);
        assert_eq!(imported.initializers[0].float_data, weight_data);
    }

    #[test]
    fn test_validate_json() {
        let graph = OnnxGraphBuilder::new("valid")
            .add_input("x", &[1], OnnxDataType::Float32)
            .add_relu("r", "x", "y")
            .add_output("y", &[1], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let json = export_to_onnx_json(&graph).expect("export failed");
        assert!(validate_onnx_json(&json).is_ok());
    }

    #[test]
    fn test_model_summary() {
        let graph = OnnxGraphBuilder::new("summary_test")
            .add_input("x", &[-1, 784], OnnxDataType::Float32)
            .add_input("w", &[784, 10], OnnxDataType::Float32)
            .add_matmul("mm", "x", "w", "logits")
            .add_softmax("sm", "logits", "probs", -1)
            .add_output("probs", &[-1, 10], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let json = export_to_onnx_json(&graph).expect("export failed");
        let summary = get_onnx_json_summary(&json).expect("summary failed");

        assert_eq!(summary.graph_name, "summary_test");
        assert_eq!(summary.num_nodes, 2);
        assert!(summary.op_types.contains(&"MatMul".to_string()));
        assert!(summary.op_types.contains(&"Softmax".to_string()));
    }

    #[test]
    fn test_file_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_onnx_test");
        let _ = fs::create_dir_all(&dir);
        let file_path = dir.join("test_model.onnx.json");

        let graph = OnnxGraphBuilder::new("file_test")
            .add_input("x", &[1, 5], OnnxDataType::Float64)
            .add_sigmoid("sig", "x", "y")
            .add_output("y", &[1, 5], OnnxDataType::Float64)
            .build()
            .expect("build failed");

        export_to_onnx_json_file(&graph, &file_path).expect("file export failed");
        let imported = import_from_onnx_json_file(&file_path).expect("file import failed");

        assert_eq!(imported.name, "file_test");
        assert_eq!(imported.nodes.len(), 1);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compact_json() {
        let graph = OnnxGraphBuilder::new("compact")
            .add_input("x", &[1], OnnxDataType::Float32)
            .add_relu("r", "x", "y")
            .add_output("y", &[1], OnnxDataType::Float32)
            .build()
            .expect("build failed");

        let compact = export_to_onnx_json_compact(&graph).expect("compact export failed");
        assert!(!compact.contains('\n'));

        let pretty = export_to_onnx_json(&graph).expect("pretty export failed");
        assert!(pretty.contains('\n'));

        // Both should parse to same result
        let g1 = import_from_onnx_json(&compact).expect("import compact");
        let g2 = import_from_onnx_json(&pretty).expect("import pretty");
        assert_eq!(g1, g2);
    }
}
