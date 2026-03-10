# scirs2-datasets Benchmarks

This directory contains benchmarks for comparing scirs2-datasets performance with PyTorch DataLoader.

## Running Rust Benchmarks

```bash
# Run all benchmarks
cargo bench -p scirs2-datasets --features benchmarks

# Run specific benchmark group
cargo bench -p scirs2-datasets --features benchmarks -- throughput_benches
cargo bench -p scirs2-datasets --features benchmarks -- memory_benches
cargo bench -p scirs2-datasets --features benchmarks -- latency_benches
```

## Benchmark Categories

### 1. Throughput Benchmarks
- **sequential_loading**: Measures dataset loading speed for various sizes
- **batch_iteration**: Tests batch iteration performance with different batch sizes
- **streaming_iteration**: Evaluates streaming performance with various chunk sizes

### 2. Memory Efficiency Benchmarks
- **memory_efficient_processing**: Tests chunked processing of large datasets

### 3. Latency Benchmarks
- **single_sample_access**: Measures random sample access latency
- **feature_extraction**: Tests column/feature extraction speed

### 4. Preprocessing Benchmarks
- **normalization**: Evaluates data normalization performance

### 5. Scaling Benchmarks
- **parallel_loading**: Tests parallel data loading with multiple workers

## PyTorch Comparison (Python Side)

To compare with PyTorch DataLoader, create a Python benchmark script:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

class SyntheticDataset(Dataset):
    def __init__(self, n_samples, n_features):
        self.data = np.random.randn(n_samples, n_features).astype(np.float32)
        self.targets = np.random.randint(0, 2, n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Throughput benchmark
def bench_pytorch_dataloader(n_samples=10000, batch_size=64, num_workers=4):
    dataset = SyntheticDataset(n_samples, 20)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=num_workers, shuffle=False)

    start = time.time()
    for batch_data, batch_targets in dataloader:
        pass
    elapsed = time.time() - start

    throughput = n_samples / elapsed
    print(f"PyTorch Throughput: {throughput:.2f} samples/sec")
    return throughput

if __name__ == "__main__":
    print("=== PyTorch DataLoader Benchmarks ===")

    # Batch iteration
    print("\nBatch Iteration:")
    for batch_size in [32, 64, 128, 256]:
        print(f"  Batch size {batch_size}:")
        bench_pytorch_dataloader(10000, batch_size, 1)

    # Parallel loading
    print("\nParallel Loading:")
    for num_workers in [1, 2, 4]:
        print(f"  Workers {num_workers}:")
        bench_pytorch_dataloader(25000, 64, num_workers)
```

## Expected Performance Characteristics

### scirs2-datasets Advantages
- **Lower Memory Overhead**: Pure Rust implementation with zero-copy views
- **Better Cache Locality**: Contiguous memory layout with ndarray
- **No GIL**: True parallelism without Global Interpreter Lock
- **Memory Mapping**: Efficient handling of datasets larger than RAM

### PyTorch DataLoader Advantages
- **Mature Ecosystem**: Battle-tested with extensive optimizations
- **GPU Integration**: Seamless CUDA tensor conversion
- **Python Flexibility**: Easy integration with Python preprocessing

## Results Interpretation

Compare results using:
1. **Samples/Second**: Throughput metric
2. **Latency**: Time to first batch
3. **Memory Usage**: Peak memory consumption
4. **CPU Utilization**: Parallel efficiency

## Continuous Benchmarking

Track performance over time:
```bash
# Save baseline
cargo bench -p scirs2-datasets --features benchmarks -- --save-baseline v0.3.1

# Compare against baseline
cargo bench -p scirs2-datasets --features benchmarks -- --baseline v0.3.1
```
