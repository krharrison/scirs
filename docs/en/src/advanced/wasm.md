# WebAssembly (scirs2-wasm)

`scirs2-wasm` provides WebAssembly bindings for running SciRS2 in the browser or
Node.js. It includes WebGPU compute shader support, SharedArrayBuffer-based parallelism,
and streaming FFT for real-time audio processing.

## Getting Started

### Building for WASM

```bash
# Install wasm-pack
cargo install wasm-pack

# Build the package
cd scirs2-wasm
wasm-pack build --target web
```

This produces a `pkg/` directory containing `.wasm`, `.js`, and `.d.ts` files ready
for import in a web application.

### Using from JavaScript

```javascript
import init, { WasmFFT, WasmLinAlg } from './pkg/scirs2_wasm.js';

async function main() {
    await init();

    // FFT
    const fft = new WasmFFT();
    const signal = new Float64Array([1.0, 2.0, 3.0, 4.0]);
    const spectrum = fft.fft(signal);

    // Linear algebra
    const linalg = new WasmLinAlg();
    const matrix = new Float64Array([1, 2, 3, 4]);  // 2x2 row-major
    const det = linalg.determinant(matrix, 2);
}
main();
```

## WebGPU Compute

Run GPU-accelerated computations in the browser using WebGPU:

```rust,ignore
use scirs2_wasm::webgpu::{WebGPUContext, GPUBuffer};

// Initialize WebGPU context
let ctx = WebGPUContext::new().await?;

// Create GPU buffers
let a = ctx.create_buffer(&matrix_data, BufferUsage::Storage)?;
let b = ctx.create_buffer(&vector_data, BufferUsage::Storage)?;
let result = ctx.create_buffer_empty(output_size, BufferUsage::Storage)?;

// Run matrix-vector multiplication on GPU
ctx.run_shader("spmv", &[&a, &b, &result], workgroup_size)?;
let output = result.read().await?;
```

### Available GPU Shaders

| Shader | Description |
|--------|-------------|
| `matmul` | Dense matrix multiplication |
| `spmv` | Sparse matrix-vector multiplication |
| `fft` | Radix-2 FFT |
| `reduce` | Parallel reduction (sum, max, min) |
| `sort` | Bitonic sort |

## Parallel Workers

Use SharedArrayBuffer and Web Workers for CPU parallelism:

```rust,ignore
use scirs2_wasm::parallel::{WorkerPool, SharedBuffer};

// Create a pool of Web Workers
let pool = WorkerPool::new(num_workers).await?;

// Shared memory buffer accessible from all workers
let shared = SharedBuffer::new(data_size)?;
shared.write(&input_data)?;

// Dispatch parallel work
pool.map(&shared, |chunk| {
    // Each worker processes a chunk
    chunk.iter().map(|x| x * x).collect()
}).await?;

let results = shared.read()?;
```

## Streaming FFT

Real-time audio processing in the browser:

```rust,ignore
use scirs2_wasm::streaming_fft::{StreamingFFT, StreamConfig};

let config = StreamConfig {
    fft_size: 2048,
    hop_size: 512,
    window: "hann",
    sample_rate: 44100.0,
};
let mut processor = StreamingFFT::new(config)?;

// Process audio frames from Web Audio API
// (called from AudioWorklet)
fn process_audio(input: &[f32], output: &mut [f32]) {
    let spectrum = processor.process_frame(input)?;
    // Modify spectrum...
    processor.synthesize_frame(&spectrum, output)?;
}
```

## Wavelet Transforms in WASM

```rust,ignore
use scirs2_wasm::wavelets::{WasmDWT, WasmCWT, WasmMFCC};

// Discrete Wavelet Transform
let dwt = WasmDWT::new("db4", 4)?;  // Daubechies-4, 4 levels
let coefficients = dwt.decompose(&signal)?;
let reconstructed = dwt.reconstruct(&coefficients)?;

// Continuous Wavelet Transform
let cwt = WasmCWT::new("morlet", scales)?;
let scalogram = cwt.transform(&signal)?;

// MFCC (Mel-Frequency Cepstral Coefficients) for audio features
let mfcc = WasmMFCC::new(sample_rate, num_coefficients)?;
let features = mfcc.compute(&audio_frame)?;
```

## Incremental PCA

For dimensionality reduction on streaming data:

```rust,ignore
use scirs2_wasm::incremental_pca::IncrementalPCA;

let mut pca = IncrementalPCA::new(n_components)?;

// Update with batches of data
for batch in data_stream {
    pca.partial_fit(&batch)?;
}

// Transform new data
let reduced = pca.transform(&new_data)?;
```

## Browser Integration Tips

- Use `wasm-pack build --target web` for ES modules (recommended for modern browsers).
- Use `wasm-pack build --target bundler` when using webpack or similar.
- WebGPU requires Chrome 113+ or Firefox Nightly with `dom.webgpu.enabled`.
- SharedArrayBuffer requires `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` headers.
- For AudioWorklet integration, the WASM module must be loaded inside the worklet scope.
