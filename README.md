# HypEmbed

**A pure-Rust, local-first embedding inference library.**

HypEmbed loads transformer model weights, tokenizes text, runs a complete forward pass, and returns normalized embedding vectors — all in safe Rust with **zero external ML runtime dependencies**.

## Features

- **Pure Rust** — No Python, ONNX, libtorch, or any external ML runtime
- **Local-first** — All computation happens on-device
- **Inference-only** — Optimized for embedding generation, not training
- **Minimal dependencies** — Only `serde`, `serde_json`, `thiserror`
- **Correctness-first** — Numerically stable softmax, layer norm, and normalization
- **BERT-compatible** — Supports BERT-like encoder architectures (BERT, MiniLM, etc.)

## Quick Start

```rust
use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};

// Load a model from a directory containing:
//   config.json, vocab.txt, model.safetensors
let model = Embedder::load("./model").unwrap();

let options = EmbeddingOptions::default()
    .with_normalize(true)
    .with_pooling(PoolingStrategy::Mean);

let embeddings = model.embed(&["hello world", "rust embeddings"], &options).unwrap();

println!("Embedding dim: {}", embeddings[0].len());
println!("First 5 values: {:?}", &embeddings[0][..5]);
```

## Supported Model Format

HypEmbed loads models from a directory with three files:

| File | Description |
|------|-------------|
| `config.json` | HuggingFace-format model configuration |
| `vocab.txt` | BERT-format vocabulary (one token per line) |
| `model.safetensors` | SafeTensors model weights (F32 or F16) |

### Getting a Model

Download a compatible model from HuggingFace:

```bash
# Example: all-MiniLM-L6-v2 (a popular sentence embedding model)
# Download config.json, vocab.txt, and model.safetensors from:
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

## Architecture

```
Input text
  → Pre-tokenize (lowercase, split whitespace/punctuation)
  → WordPiece tokenize
  → Add [CLS]/[SEP], truncate, pad
  → Token + Position + Segment embeddings + LayerNorm
  → N × Encoder layers (Self-Attention → Residual+LN → FFN → Residual+LN)
  → Pooling (Mean or CLS)
  → L2 Normalization (optional)
  → Embedding vector
```

## Pooling Strategies

| Strategy | Description |
|----------|-------------|
| `Mean` | Average of non-padding token hidden states (recommended for sentence embeddings) |
| `Cls` | Hidden state at position 0 ([CLS] token) |

## Numerical Stability

| Operation | Stability Measure |
|-----------|-------------------|
| Softmax | Subtract max before exp (prevents overflow) |
| LayerNorm | Epsilon (1e-12) in denominator |
| L2 Norm | `max(norm, 1e-12)` (handles zero vectors) |
| Mean Pool | `max(token_count, 1e-9)` (handles empty sequences) |
| Attention | Add -10000 to masked positions before softmax |

## Performance

The library is designed for correctness-first with performance in mind:

- **Cache-friendly matmul** — ikj loop ordering for row-major data
- **Contiguous storage** — All tensors use flat `Vec<f32>` with row-major layout
- **Preallocated buffers** — No unnecessary allocations in hot paths
- **Batched inference** — Process multiple texts in a single forward pass

### Future optimizations (not yet implemented):
- SIMD vectorization for matmul and element-wise ops
- Thread-parallel batch processing via `rayon`
- Memory-mapped weight loading via `memmap2`

## Limitations

- **CPU only** — No GPU support in v0.1
- **BERT-like models only** — GPT, T5, and other architectures are not supported
- **No quantization** — F32 inference only (F16 weights auto-convert to F32)
- **No training** — Inference only
- **No HuggingFace Hub** — Models must be downloaded manually

## Project Structure

```
src/
├── tensor/          # Minimal tensor engine (shape, matmul, softmax, GELU, LayerNorm, L2)
├── tokenizer/       # BERT-compatible tokenizer (vocab, WordPiece, encoding)
├── model/           # BERT encoder (config, SafeTensors, attention, FFN, pooling)
├── pipeline/        # High-level Embedder API
└── error/           # Typed error handling
```

## License

MIT OR Apache-2.0
