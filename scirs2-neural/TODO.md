# scirs2-neural TODO

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Attention Mechanisms
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Linear attention
- Efficient attention
- Sparse attention
- Multi-head latent attention

### Mixture of Experts
- Top-k routing with load balancing
- Expert capacity and auxiliary loss
- MoE transformer block integration

### Capsule Networks
- Dynamic routing between capsules
- Squash activation
- EM routing variant

### Spiking Neural Networks (SNN)
- Leaky Integrate-and-Fire (LIF) neurons
- Spike-Timing Dependent Plasticity (STDP)
- Rate coding and temporal coding

### Graph Neural Networks (GNN)
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE
- Graph Isomorphism Network (GIN)
- Message Passing Neural Networks
- DiffPool and SAGPool graph pooling
- Global add/mean/max pooling

### Vision Architectures
- SWIN Transformer (shifted window self-attention)
- Vision Transformer (ViT) with patch embeddings
- UNet encoder-decoder
- CLIP dual-encoder (vision + text)
- ConvNeXt (Tiny, Small, Base, Large, XLarge)
- PatchEmbedding module

### NLP / Sequence Architectures
- GPT-2 causal language model
- T5 encoder-decoder
- Full transformer (encoder + decoder)
- Positional encodings: sinusoidal, learned, RoPE, relative

### Generative Models
- Generative Adversarial Networks (GAN)
- Variational Autoencoders (VAE)
- Diffusion models (DDPM)
- Normalizing flow models
- Energy-based models

### Training Infrastructure
- Knowledge distillation (response-based and feature-based)
- Continual learning (EWC)
- Meta-learning (MAML-style)
- Contrastive learning (SimCLR, MoCo)
- Multitask learning
- Self-supervised pretraining
- Magnitude-based and structured pruning
- Post-training quantization and QAT
- DPO (Direct Preference Optimization)
- PPO for RLHF
- Reward modeling and preference data
- Gradient checkpointing
- Half-precision (FP16) training utilities

### Serialization
- Model graph serialization format
- Portable weight format (versioned)

### Compression
- Model compression utilities
- On-device optimization

## v0.4.0 Roadmap

### Attention
- Flash Attention v2 (tiled memory-efficient attention)
- Multi-query attention (MQA)

### Quantization
- INT4 weight-only quantization
- INT8 activation quantization
- GPTQ-style post-training quantization

### Export and Interop
- ONNX-like model export
- Weight conversion utilities for interop with other frameworks

### Efficient Fine-Tuning
- LoRA (Low-Rank Adaptation)
- Adapter layers
- Prefix tuning

### Distributed Training
- Gradient compression (TopK sparsification, PowerSGD)
- Pipeline parallelism
- Tensor parallelism primitives

### Architecture Search
- Neural Architecture Search (NAS) integration
- Differentiable NAS (DARTS)

## Known Issues / Technical Debt

- Some doc tests are marked `#[ignore]` pending API stabilization
- WASM target requires additional feature gating for large model weights
- GPU acceleration stubs exist in `hardware/` but require `scirs2-core::gpu` completion
