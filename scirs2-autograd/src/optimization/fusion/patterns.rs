//! Fusion pattern definitions and matching logic
//!
//! This module defines the graph node types, operation kinds, and the
//! `can_fuse` predicates that determine whether two adjacent nodes
//! in a computation graph can be merged into a single fused operation.

use std::fmt;

// ---------------------------------------------------------------------------
// Operation kinds
// ---------------------------------------------------------------------------

/// The kind of operation a graph node represents.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpKind {
    // -- Linear algebra --
    /// Matrix multiplication: C = A @ B
    MatMul,
    /// Bias addition (element-wise add of a bias vector to each row)
    BiasAdd,

    // -- Activations --
    /// Rectified Linear Unit
    Relu,
    /// Gaussian Error Linear Unit
    Gelu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Swish / SiLU activation
    Swish,

    // -- Convolution / Normalization --
    /// 2-D convolution
    Conv2d,
    /// Batch normalization
    BatchNorm,

    // -- Element-wise --
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Element-wise negation
    Neg,
    /// Element-wise square (x^2)
    Square,
    /// Element-wise exponentiation (e^x)
    Exp,
    /// Element-wise logarithm
    Log,
    /// Element-wise square root
    Sqrt,

    // -- Reductions --
    /// Sum reduction along axes
    Sum,
    /// Mean reduction along axes
    Mean,
    /// Max reduction along axes
    Max,
    /// Min reduction along axes
    Min,

    // -- Misc --
    /// Placeholder / input node
    Input,
    /// Constant node
    Constant,
    /// A custom / unknown operation
    Custom(String),
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpKind::MatMul => write!(f, "matmul"),
            OpKind::BiasAdd => write!(f, "bias_add"),
            OpKind::Relu => write!(f, "relu"),
            OpKind::Gelu => write!(f, "gelu"),
            OpKind::Sigmoid => write!(f, "sigmoid"),
            OpKind::Tanh => write!(f, "tanh"),
            OpKind::Swish => write!(f, "swish"),
            OpKind::Conv2d => write!(f, "conv2d"),
            OpKind::BatchNorm => write!(f, "batch_norm"),
            OpKind::Add => write!(f, "add"),
            OpKind::Sub => write!(f, "sub"),
            OpKind::Mul => write!(f, "mul"),
            OpKind::Div => write!(f, "div"),
            OpKind::Neg => write!(f, "neg"),
            OpKind::Square => write!(f, "square"),
            OpKind::Exp => write!(f, "exp"),
            OpKind::Log => write!(f, "log"),
            OpKind::Sqrt => write!(f, "sqrt"),
            OpKind::Sum => write!(f, "sum"),
            OpKind::Mean => write!(f, "mean"),
            OpKind::Max => write!(f, "max"),
            OpKind::Min => write!(f, "min"),
            OpKind::Input => write!(f, "input"),
            OpKind::Constant => write!(f, "constant"),
            OpKind::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

impl OpKind {
    /// Returns `true` if this operation is an element-wise unary or binary op.
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            OpKind::Add
                | OpKind::Sub
                | OpKind::Mul
                | OpKind::Div
                | OpKind::Neg
                | OpKind::Square
                | OpKind::Exp
                | OpKind::Log
                | OpKind::Sqrt
                | OpKind::Relu
                | OpKind::Gelu
                | OpKind::Sigmoid
                | OpKind::Tanh
                | OpKind::Swish
        )
    }

    /// Returns `true` if this operation is a pure activation function.
    pub fn is_activation(&self) -> bool {
        matches!(
            self,
            OpKind::Relu | OpKind::Gelu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Swish
        )
    }

    /// Returns `true` if this operation is a reduction.
    pub fn is_reduction(&self) -> bool {
        matches!(self, OpKind::Sum | OpKind::Mean | OpKind::Max | OpKind::Min)
    }
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// A lightweight representation of a node in the computation graph
/// that is used by the fusion analysis passes.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier within the graph
    pub id: usize,
    /// The operation this node performs
    pub op: OpKind,
    /// Identifiers of the input nodes
    pub inputs: Vec<usize>,
    /// Shape of the output tensor (empty if unknown)
    pub output_shape: Vec<usize>,
    /// Number of consumers (other nodes that read this output)
    pub num_consumers: usize,
}

impl GraphNode {
    /// Create a new graph node.
    pub fn new(id: usize, op: OpKind, inputs: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            id,
            op,
            inputs,
            output_shape,
            num_consumers: 0,
        }
    }

    /// Returns the number of elements in the output tensor, or `None`
    /// if the shape is unknown (empty).
    pub fn output_numel(&self) -> Option<usize> {
        if self.output_shape.is_empty() {
            None
        } else {
            Some(self.output_shape.iter().product())
        }
    }
}

// ---------------------------------------------------------------------------
// Fusible operation pattern (kept compatible with original API)
// ---------------------------------------------------------------------------

/// High-level fusible operation pattern categories.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Element-wise operations (add, mul, relu, etc.)
    ElementWise,
    /// Matrix multiplication + bias add
    MatMulBias,
    /// Matrix multiplication + activation
    MatMulActivation,
    /// Matrix multiplication + bias + activation (e.g. fused_linear_relu)
    MatMulBiasActivation,
    /// Convolution + batch norm
    ConvBN,
    /// Convolution + batch norm + activation
    ConvBNActivation,
    /// Reduction + element-wise
    ReductionElementWise,
    /// Element-wise affine: x * scale + shift
    Affine,
    /// Reduction fusion: sum + div = mean
    SumDivToMean,
    /// Reduction fusion: square + mean = variance
    SquareMeanToVariance,
    /// Softmax fusion: exp + sum + div
    Softmax,
    /// Custom fusion pattern
    Custom(String),
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionPattern::ElementWise => write!(f, "elementwise_chain"),
            FusionPattern::MatMulBias => write!(f, "matmul_bias"),
            FusionPattern::MatMulActivation => write!(f, "matmul_activation"),
            FusionPattern::MatMulBiasActivation => write!(f, "matmul_bias_activation"),
            FusionPattern::ConvBN => write!(f, "conv_bn"),
            FusionPattern::ConvBNActivation => write!(f, "conv_bn_activation"),
            FusionPattern::ReductionElementWise => write!(f, "reduction_elementwise"),
            FusionPattern::Affine => write!(f, "affine"),
            FusionPattern::SumDivToMean => write!(f, "sum_div_to_mean"),
            FusionPattern::SquareMeanToVariance => write!(f, "square_mean_to_variance"),
            FusionPattern::Softmax => write!(f, "softmax"),
            FusionPattern::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern matching predicates
// ---------------------------------------------------------------------------

/// Determine whether two adjacent nodes can be fused under the
/// **MatMul + Bias** pattern.  Returns `true` when `node_a` is a
/// `MatMul` and `node_b` is a `BiasAdd` (or plain `Add`) that
/// consumes `node_a`'s output and `node_a` has only one consumer.
pub fn can_fuse_matmul_bias(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if node_a.op != OpKind::MatMul {
        return false;
    }
    if !matches!(node_b.op, OpKind::BiasAdd | OpKind::Add) {
        return false;
    }
    // node_b must consume node_a
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    // node_a must have exactly one consumer (node_b) for safe fusion
    node_a.num_consumers == 1
}

/// Determine whether two adjacent nodes can be fused under the
/// **MatMul + Activation** pattern.
pub fn can_fuse_matmul_activation(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if node_a.op != OpKind::MatMul {
        return false;
    }
    if !node_b.op.is_activation() {
        return false;
    }
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    node_a.num_consumers == 1
}

/// Determine whether three adjacent nodes form a **MatMul + Bias + Activation**
/// chain.  `node_a` must be MatMul, `node_b` bias-add, `node_c` activation.
pub fn can_fuse_matmul_bias_activation(
    node_a: &GraphNode,
    node_b: &GraphNode,
    node_c: &GraphNode,
) -> bool {
    can_fuse_matmul_bias(node_a, node_b)
        && node_b.op.is_elementwise()
        && node_c.op.is_activation()
        && node_c.inputs.contains(&node_b.id)
        && node_b.num_consumers == 1
}

/// Determine whether two nodes can be fused under the **Conv + BN** pattern.
pub fn can_fuse_conv_bn(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if node_a.op != OpKind::Conv2d {
        return false;
    }
    if node_b.op != OpKind::BatchNorm {
        return false;
    }
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    node_a.num_consumers == 1
}

/// Determine whether three nodes form a **Conv + BN + Activation** chain.
pub fn can_fuse_conv_bn_activation(
    node_a: &GraphNode,
    node_b: &GraphNode,
    node_c: &GraphNode,
) -> bool {
    can_fuse_conv_bn(node_a, node_b)
        && node_c.op.is_activation()
        && node_c.inputs.contains(&node_b.id)
        && node_b.num_consumers == 1
}

/// Determine whether two consecutive element-wise nodes can be fused.
pub fn can_fuse_elementwise(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if !node_a.op.is_elementwise() || !node_b.op.is_elementwise() {
        return false;
    }
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    // Fusing only makes sense if node_a has one consumer
    node_a.num_consumers == 1
}

/// Determine whether a Mul + Add pair can be fused into an affine transform
/// (x * scale + shift).
pub fn can_fuse_affine(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if node_a.op != OpKind::Mul {
        return false;
    }
    if node_b.op != OpKind::Add {
        return false;
    }
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    node_a.num_consumers == 1
}

/// Determine whether a Sum + Div pair can be fused into a Mean.
pub fn can_fuse_sum_div_to_mean(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if node_a.op != OpKind::Sum {
        return false;
    }
    if node_b.op != OpKind::Div {
        return false;
    }
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    node_a.num_consumers == 1
}

/// Determine whether a Square + Mean pair can be fused into a Variance computation.
pub fn can_fuse_square_mean_to_variance(node_a: &GraphNode, node_b: &GraphNode) -> bool {
    if node_a.op != OpKind::Square {
        return false;
    }
    if node_b.op != OpKind::Mean {
        return false;
    }
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    node_a.num_consumers == 1
}

/// Determine whether three consecutive nodes form a Softmax pattern (exp + sum + div).
pub fn can_fuse_softmax(node_a: &GraphNode, node_b: &GraphNode, node_c: &GraphNode) -> bool {
    if node_a.op != OpKind::Exp {
        return false;
    }
    if node_b.op != OpKind::Sum {
        return false;
    }
    if node_c.op != OpKind::Div {
        return false;
    }
    // Sum must consume Exp output
    if !node_b.inputs.contains(&node_a.id) {
        return false;
    }
    // Div must consume both Exp and Sum outputs
    if !node_c.inputs.contains(&node_a.id) || !node_c.inputs.contains(&node_b.id) {
        return false;
    }
    // Exp must have exactly 2 consumers (Sum and Div)
    node_a.num_consumers == 2 && node_b.num_consumers == 1
}

/// Generic predicate: returns the `FusionPattern` if two nodes can be fused
/// under *any* known two-node pattern, or `None` otherwise.
pub fn detect_two_node_pattern(node_a: &GraphNode, node_b: &GraphNode) -> Option<FusionPattern> {
    if can_fuse_matmul_bias(node_a, node_b) {
        return Some(FusionPattern::MatMulBias);
    }
    if can_fuse_matmul_activation(node_a, node_b) {
        return Some(FusionPattern::MatMulActivation);
    }
    if can_fuse_conv_bn(node_a, node_b) {
        return Some(FusionPattern::ConvBN);
    }
    if can_fuse_sum_div_to_mean(node_a, node_b) {
        return Some(FusionPattern::SumDivToMean);
    }
    if can_fuse_square_mean_to_variance(node_a, node_b) {
        return Some(FusionPattern::SquareMeanToVariance);
    }
    if can_fuse_affine(node_a, node_b) {
        return Some(FusionPattern::Affine);
    }
    if can_fuse_elementwise(node_a, node_b) {
        return Some(FusionPattern::ElementWise);
    }
    None
}

/// Generic predicate: returns the `FusionPattern` if three nodes can be fused
/// under a known three-node pattern, or `None` otherwise.
pub fn detect_three_node_pattern(
    node_a: &GraphNode,
    node_b: &GraphNode,
    node_c: &GraphNode,
) -> Option<FusionPattern> {
    if can_fuse_matmul_bias_activation(node_a, node_b, node_c) {
        return Some(FusionPattern::MatMulBiasActivation);
    }
    if can_fuse_conv_bn_activation(node_a, node_b, node_c) {
        return Some(FusionPattern::ConvBNActivation);
    }
    if can_fuse_softmax(node_a, node_b, node_c) {
        return Some(FusionPattern::Softmax);
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: usize, op: OpKind, inputs: Vec<usize>, consumers: usize) -> GraphNode {
        let mut node = GraphNode::new(id, op, inputs, vec![2, 3]);
        node.num_consumers = consumers;
        node
    }

    // -- OpKind tests -------------------------------------------------------

    #[test]
    fn test_op_kind_is_elementwise() {
        assert!(OpKind::Add.is_elementwise());
        assert!(OpKind::Relu.is_elementwise());
        assert!(!OpKind::MatMul.is_elementwise());
        assert!(!OpKind::Conv2d.is_elementwise());
        assert!(!OpKind::Sum.is_elementwise());
    }

    #[test]
    fn test_op_kind_is_activation() {
        assert!(OpKind::Relu.is_activation());
        assert!(OpKind::Gelu.is_activation());
        assert!(OpKind::Sigmoid.is_activation());
        assert!(!OpKind::Add.is_activation());
        assert!(!OpKind::MatMul.is_activation());
    }

    #[test]
    fn test_op_kind_is_reduction() {
        assert!(OpKind::Sum.is_reduction());
        assert!(OpKind::Mean.is_reduction());
        assert!(!OpKind::Add.is_reduction());
    }

    #[test]
    fn test_op_kind_display() {
        assert_eq!(format!("{}", OpKind::MatMul), "matmul");
        assert_eq!(format!("{}", OpKind::Relu), "relu");
        assert_eq!(
            format!("{}", OpKind::Custom("my_op".to_string())),
            "custom(my_op)"
        );
    }

    // -- can_fuse MatMul+Bias -----------------------------------------------

    #[test]
    fn test_can_fuse_matmul_bias() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let bias_add = make_node(1, OpKind::BiasAdd, vec![0], 1);
        assert!(can_fuse_matmul_bias(&matmul, &bias_add));
    }

    #[test]
    fn test_cannot_fuse_matmul_bias_multiple_consumers() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 2);
        let bias_add = make_node(1, OpKind::BiasAdd, vec![0], 1);
        assert!(!can_fuse_matmul_bias(&matmul, &bias_add));
    }

    #[test]
    fn test_cannot_fuse_matmul_bias_no_edge() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let bias_add = make_node(1, OpKind::BiasAdd, vec![5], 1);
        assert!(!can_fuse_matmul_bias(&matmul, &bias_add));
    }

    // -- can_fuse MatMul+Activation -----------------------------------------

    #[test]
    fn test_can_fuse_matmul_activation() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let relu = make_node(1, OpKind::Relu, vec![0], 1);
        assert!(can_fuse_matmul_activation(&matmul, &relu));
    }

    #[test]
    fn test_cannot_fuse_matmul_nonactivation() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let add = make_node(1, OpKind::Add, vec![0], 1);
        assert!(!can_fuse_matmul_activation(&matmul, &add));
    }

    // -- can_fuse MatMul+Bias+Activation ------------------------------------

    #[test]
    fn test_can_fuse_matmul_bias_activation() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let bias_add = make_node(1, OpKind::Add, vec![0], 1);
        let relu = make_node(2, OpKind::Relu, vec![1], 1);
        assert!(can_fuse_matmul_bias_activation(&matmul, &bias_add, &relu));
    }

    #[test]
    fn test_can_fuse_matmul_bias_gelu() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let bias_add = make_node(1, OpKind::Add, vec![0], 1);
        let gelu = make_node(2, OpKind::Gelu, vec![1], 1);
        assert!(can_fuse_matmul_bias_activation(&matmul, &bias_add, &gelu));
    }

    // -- can_fuse Conv+BN ---------------------------------------------------

    #[test]
    fn test_can_fuse_conv_bn() {
        let conv = make_node(0, OpKind::Conv2d, vec![], 1);
        let bn = make_node(1, OpKind::BatchNorm, vec![0], 1);
        assert!(can_fuse_conv_bn(&conv, &bn));
    }

    #[test]
    fn test_cannot_fuse_conv_bn_wrong_order() {
        let bn = make_node(0, OpKind::BatchNorm, vec![], 1);
        let conv = make_node(1, OpKind::Conv2d, vec![0], 1);
        assert!(!can_fuse_conv_bn(&bn, &conv));
    }

    // -- can_fuse Conv+BN+Activation ----------------------------------------

    #[test]
    fn test_can_fuse_conv_bn_relu() {
        let conv = make_node(0, OpKind::Conv2d, vec![], 1);
        let bn = make_node(1, OpKind::BatchNorm, vec![0], 1);
        let relu = make_node(2, OpKind::Relu, vec![1], 1);
        assert!(can_fuse_conv_bn_activation(&conv, &bn, &relu));
    }

    // -- can_fuse elementwise chain -----------------------------------------

    #[test]
    fn test_can_fuse_elementwise() {
        let add = make_node(0, OpKind::Add, vec![], 1);
        let mul = make_node(1, OpKind::Mul, vec![0], 1);
        assert!(can_fuse_elementwise(&add, &mul));
    }

    #[test]
    fn test_cannot_fuse_elementwise_non_elementwise() {
        let add = make_node(0, OpKind::Add, vec![], 1);
        let matmul = make_node(1, OpKind::MatMul, vec![0], 1);
        assert!(!can_fuse_elementwise(&add, &matmul));
    }

    // -- can_fuse affine (Mul+Add) ------------------------------------------

    #[test]
    fn test_can_fuse_affine() {
        let mul = make_node(0, OpKind::Mul, vec![], 1);
        let add = make_node(1, OpKind::Add, vec![0], 1);
        assert!(can_fuse_affine(&mul, &add));
    }

    // -- can_fuse Sum+Div -> Mean -------------------------------------------

    #[test]
    fn test_can_fuse_sum_div_to_mean() {
        let sum = make_node(0, OpKind::Sum, vec![], 1);
        let div = make_node(1, OpKind::Div, vec![0], 1);
        assert!(can_fuse_sum_div_to_mean(&sum, &div));
    }

    // -- can_fuse Square+Mean -> Variance -----------------------------------

    #[test]
    fn test_can_fuse_square_mean_to_variance() {
        let sq = make_node(0, OpKind::Square, vec![], 1);
        let mean = make_node(1, OpKind::Mean, vec![0], 1);
        assert!(can_fuse_square_mean_to_variance(&sq, &mean));
    }

    // -- can_fuse Softmax (exp + sum + div) ---------------------------------

    #[test]
    fn test_can_fuse_softmax() {
        let exp = make_node(0, OpKind::Exp, vec![], 2);
        let sum = make_node(1, OpKind::Sum, vec![0], 1);
        let div = make_node(2, OpKind::Div, vec![0, 1], 1);
        assert!(can_fuse_softmax(&exp, &sum, &div));
    }

    #[test]
    fn test_cannot_fuse_softmax_wrong_consumers() {
        let exp = make_node(0, OpKind::Exp, vec![], 1); // wrong: needs 2
        let sum = make_node(1, OpKind::Sum, vec![0], 1);
        let div = make_node(2, OpKind::Div, vec![0, 1], 1);
        assert!(!can_fuse_softmax(&exp, &sum, &div));
    }

    // -- detect_two_node_pattern --------------------------------------------

    #[test]
    fn test_detect_two_node_matmul_bias() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let bias_add = make_node(1, OpKind::BiasAdd, vec![0], 1);
        assert_eq!(
            detect_two_node_pattern(&matmul, &bias_add),
            Some(FusionPattern::MatMulBias)
        );
    }

    #[test]
    fn test_detect_two_node_none() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let conv = make_node(1, OpKind::Conv2d, vec![0], 1);
        assert_eq!(detect_two_node_pattern(&matmul, &conv), None);
    }

    // -- detect_three_node_pattern ------------------------------------------

    #[test]
    fn test_detect_three_node_matmul_bias_activation() {
        let matmul = make_node(0, OpKind::MatMul, vec![], 1);
        let bias_add = make_node(1, OpKind::Add, vec![0], 1);
        let relu = make_node(2, OpKind::Relu, vec![1], 1);
        assert_eq!(
            detect_three_node_pattern(&matmul, &bias_add, &relu),
            Some(FusionPattern::MatMulBiasActivation)
        );
    }

    // -- GraphNode tests ----------------------------------------------------

    #[test]
    fn test_graph_node_output_numel() {
        let node = GraphNode::new(0, OpKind::Add, vec![], vec![2, 3, 4]);
        assert_eq!(node.output_numel(), Some(24));

        let empty = GraphNode::new(1, OpKind::Add, vec![], vec![]);
        assert_eq!(empty.output_numel(), None);
    }

    // -- FusionPattern display ----------------------------------------------

    #[test]
    fn test_fusion_pattern_display() {
        assert_eq!(format!("{}", FusionPattern::MatMulBias), "matmul_bias");
        assert_eq!(
            format!("{}", FusionPattern::MatMulBiasActivation),
            "matmul_bias_activation"
        );
        assert_eq!(
            format!("{}", FusionPattern::ConvBNActivation),
            "conv_bn_activation"
        );
    }
}
