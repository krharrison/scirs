# Neural Networks (scirs2-neural)

`scirs2-neural` provides neural network building blocks, training utilities, quantization,
model export, and neural architecture search. It is not a full deep learning framework but
offers the core primitives needed for inference and research.

## Layers

### Dense (Fully Connected)

```rust,ignore
use scirs2_neural::layers::{Linear, Activation};

// Linear layer: input_dim=784, output_dim=256
let linear = Linear::new(784, 256)?;

// Forward pass
let output = linear.forward(&input)?;

// With activation
let relu_output = Activation::relu(&output)?;
```

### Convolution

```rust,ignore
use scirs2_neural::layers::conv::{Conv2d, Conv3d};

// 2D convolution: in_channels=3, out_channels=64, kernel=3x3
let conv = Conv2d::new(3, 64, (3, 3), (1, 1), (1, 1))?;
let output = conv.forward(&input)?;

// 3D convolution for volumetric data
let conv3d = Conv3d::new(1, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1))?;
```

### Normalization

```rust,ignore
use scirs2_neural::layers::{BatchNorm, LayerNorm};

let bn = BatchNorm::new(64)?;
let ln = LayerNorm::new(&[64])?;
```

## Attention Mechanisms

### Multi-Head Attention

```rust,ignore
use scirs2_neural::attention::{MultiHeadAttention, FlashAttention};

// Standard multi-head attention
let mha = MultiHeadAttention::new(embed_dim, num_heads)?;
let output = mha.forward(&query, &key, &value, mask)?;

// Flash attention (memory-efficient, O(N) memory instead of O(N^2))
let flash = FlashAttention::new(embed_dim, num_heads)?;
let output = flash.forward(&query, &key, &value, mask)?;
```

### KV Cache for Inference

```rust,ignore
use scirs2_neural::inference::KVCache;

let mut cache = KVCache::new(num_layers, max_seq_len, embed_dim, num_heads)?;

// Autoregressive generation with cached keys/values
for token in tokens {
    let output = model.forward_with_cache(&token, &mut cache)?;
}
```

## Quantization

### Post-Training Quantization

```rust,ignore
use scirs2_neural::quantization::{
    gptq_quantize, awq_quantize, smoothquant_quantize
};

// GPTQ: weight-only quantization to INT4
let quantized = gptq_quantize(&model, &calibration_data, 4)?;

// AWQ: activation-aware weight quantization
let quantized = awq_quantize(&model, &calibration_data)?;

// SmoothQuant: W8A8 (weights and activations both INT8)
let quantized = smoothquant_quantize(&model, &calibration_data, 0.5)?;
```

### Quantization-Aware Training (QAT)

```rust,ignore
use scirs2_neural::quantization::qat::{FakeQuantize, CalibrationCollector};

// Insert fake quantization nodes for training
let fake_quant = FakeQuantize::new(bits, symmetric)?;
let qat_output = fake_quant.forward(&tensor)?;

// Calibration with min-max, percentile, or KL divergence
let collector = CalibrationCollector::min_max();
collector.observe(&activations)?;
let scale = collector.compute_scale()?;
```

## Training

### Optimizers

```rust,ignore
use scirs2_neural::optimizers::{Adam, SGD, AdaFactor, LAMB, LARS, Sophia};

let optimizer = Adam::new(&parameters, 1e-3)?;
let optimizer = SGD::new(&parameters, 0.01, 0.9)?;  // lr, momentum

// Large-batch optimizers
let optimizer = LAMB::new(&parameters, 1e-3, (0.9, 0.999))?;
let optimizer = LARS::new(&parameters, 0.01, 0.001)?;  // lr, weight_decay
```

### Distributed Training

```rust,ignore
use scirs2_neural::training::{
    pipeline_parallel, tensor_parallel, gradient_compression
};

// Pipeline parallelism (GPipe-style micro-batching)
let pp = pipeline_parallel::PipelineSchedule::new(
    model_stages, num_micro_batches
)?;

// Tensor parallelism (Megatron-LM style column/row splitting)
let tp = tensor_parallel::TensorParallel::new(model, world_size)?;

// Gradient compression (PowerSGD, TopK)
let compressor = gradient_compression::PowerSGD::new(rank)?;
```

## Parameter-Efficient Fine-Tuning

### LoRA, DoRA, and AdaLoRA

```rust,ignore
use scirs2_neural::lora::{LoRA, DoRA, AdaLoRA};

// LoRA: low-rank adaptation
let lora = LoRA::new(&linear_layer, rank, alpha)?;

// DoRA: weight-decomposed low-rank adaptation
let dora = DoRA::new(&linear_layer, rank)?;

// AdaLoRA: adaptive rank allocation
let adalora = AdaLoRA::new(&model, initial_rank, target_rank)?;
```

## Neural Architecture Search

```rust,ignore
use scirs2_neural::nas::{DARTS, GDAS, SNAS};

// DARTS: differentiable architecture search
let darts = DARTS::new(search_space, num_epochs)?;
let architecture = darts.search(&train_data, &val_data)?;

// GDAS: Gumbel-softmax based
let gdas = GDAS::new(search_space, temperature)?;
```

## Model Export

```rust,ignore
use scirs2_neural::export::{save_onnx, save_safetensors};

// Export to ONNX format
save_onnx(&model, &dummy_input, "model.onnx")?;

// Export weights to SafeTensors format
save_safetensors(&model, "model.safetensors")?;
```
