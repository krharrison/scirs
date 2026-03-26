//! LoRA (Low-Rank Adaptation) and Adapter layers for parameter-efficient fine-tuning.
//!
//! This module provides implementations of several parameter-efficient fine-tuning (PEFT)
//! techniques:
//!
//! - **LoRA**: Decomposes weight updates into low-rank matrices, dramatically reducing
//!   the number of trainable parameters while maintaining model quality.
//! - **DoRA**: Weight-decomposed LoRA that separately learns magnitude and direction.
//! - **AdaLoRA**: Adaptive rank allocation via SVD parameterisation and importance scoring.
//! - **IA³**: Infused Adapter — element-wise scaling vectors, extremely parameter-efficient.
//! - **VeRA**: Vector-based Random Matrix Adaptation — shares frozen random matrices,
//!   only learns tiny per-layer scaling vectors.
//! - **Bottleneck Adapters**: Inserts small trainable bottleneck modules with optional
//!   residual connections.
//!
//! # References
//!
//! - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
//! - Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", 2019
//! - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024
//! - Zhang et al., "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", 2023
//! - Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper…", 2022
//! - Kopiczko et al., "VeRA: Vector-Based Random Matrix Adaptation", 2024

pub mod adalora;
pub mod adapter;
pub mod dora;
pub mod ia3;
pub mod linear;
pub mod merge;
pub mod types;
pub mod vera;

pub use adalora::{AdaLoraConfig, AdaLoraLayer};
pub use adapter::BottleneckAdapter;
pub use dora::{DoraConfig, DoraLinear};
pub use ia3::{Ia3Adapter, Ia3Config};
pub use linear::LoRALinear;
pub use merge::{
    compute_effective_weight, merge_lora_weights, merge_multiple_lora, quantize_merged_weight,
    QuantizedWeight,
};
pub use types::{AdapterActivation, AdapterConfig, LoRAConfig, LoRAStats, LoRATarget};
pub use vera::{SharedRandomMatrices, VeRAConfig, VeRALayer, VeraConfig, VeraLayer};
