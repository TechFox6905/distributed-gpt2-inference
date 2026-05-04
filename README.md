# GPT-2 Architecture & Distributed Inference System

GPT-2 architecture reimplementation in PyTorch with KV-cache optimized decoding and a distributed inference system built using FastAPI and Redis.

The project demonstrates:

* Transformer architecture implementation
* KV-cache optimized autoregressive decoding
* Dynamic batching concepts for LLM inference
* Queue-based distributed serving architecture
* Sampling and decoding pipelines
* Evaluation and benchmarking workflows
* Unit and integration testing practices

Core technologies used:

* PyTorch
* FastAPI
* Redis
* HuggingFace Transformers
* Docker
* Pytest

The implementation includes:

* KV-cache optimized autoregressive decoding
* Advanced sampling strategies
* Queue-based distributed inference architecture
* Dynamic request batching prototype
* Evaluation and benchmarking pipeline
* Unit and integration test coverage

The project focuses on understanding both:

1. Transformer internals
2. Real-world LLM inference system design

---

# Features

## Model

* GPT-2 style decoder-only Transformer
* Multi-head causal self-attention
* Pre-LayerNorm architecture
* Weight tying
* GELU activation
* Dropout regularization
* HuggingFace GPT-2 weight compatibility

## Generation

* KV-cache optimized decoding
* Temperature sampling
* Top-K sampling
* Top-P (nucleus) sampling
* EOS-aware batched generation

## Distributed Inference

* FastAPI inference API
* Redis queue-based request handling
* Worker-based inference engine
* Dynamic batching prototype
* Async request/result workflow

## Evaluation

* WikiText-2 perplexity evaluation
* Generation benchmarking
* KV-cache benchmarking
* Sample generation pipeline

## Testing

* Unit tests
* Integration tests
* API tests
* Redis pipeline tests
* KV-cache tests
* Sampling tests
* Performance tests

---

# Project Structure

```text
src/
└── gpt/
    ├── api/
    │   └── main.py
    │
    ├── core/
    │   ├── generate.py
    │   └── model.py
    │
    ├── eval/
    │   ├── benchmark.py
    │   ├── evaluate.py
    │   ├── kv_cache.py
    │   └── sample_generation.py
    │
    ├── results/
    │
    └── worker/
        └── worker.py


tests/
├── integration_tests/
├── performance/
└── unit_tests/
```

---

# Transformer Architecture

## Configuration

| Parameter           | Value  |
| ------------------- | ------ |
| Context Length      | 1024   |
| Embedding Dimension | 768    |
| Transformer Layers  | 12     |
| Attention Heads     | 12     |
| Vocabulary Size     | 50,257 |
| Dropout             | 0.1    |

---

## Forward Pass

```text
Input Tokens
    ↓
Token Embeddings + Positional Embeddings
    ↓
Transformer Blocks
    ↓
Final LayerNorm
    ↓
Linear Projection (LM Head)
    ↓
Vocabulary Logits
```

---

## Transformer Block

Each block contains:

```text
LayerNorm
    ↓
Masked Multi-Head Self-Attention
    ↓
Residual Connection
    ↓
LayerNorm
    ↓
MLP (Feedforward Network)
    ↓
Residual Connection
```

---

## Embeddings

The model uses:

* Token embeddings (`wte`)
* Positional embeddings (`wpe`)

Combined as:

```python
x = token_embedding + positional_embedding
```

---

## Causal Self-Attention

The implementation includes:

* Scaled dot-product attention
* Multi-head attention
* Causal masking
* KV-cache support

Causal masking prevents tokens from attending to future positions during autoregressive decoding.

---

## Feedforward Network (MLP)

```text
Linear
   ↓
GELU
   ↓
Linear
```

The hidden dimension expands by 4× before projection back to embedding size.

---

## Weight Tying

The output projection layer shares weights with token embeddings:

```python
lm_head.weight = wte.weight
```

Benefits:

* Reduced parameter count
* Improved generalization
* Standard GPT-2 optimization

---

# KV Cache Optimization

## Motivation

Naive autoregressive generation recomputes attention over the full sequence at every decoding step:

```text
Token 1 → compute
Token 2 → recompute (1,2)
Token 3 → recompute (1,2,3)
```

This results in quadratic decoding complexity.

---

## Implementation

The model caches:

* Keys (K)
* Values (V)

for every Transformer layer.

During generation:

* only the newest token is processed
* previous K/V tensors are reused

---

## Cache Structure

```text
K, V: (B, n_head, T, head_dim)
```

Full cache:

```text
past_kvs = [
    (K₁, V₁),
    (K₂, V₂),
    ...
]
```

---

## Generation Flow

```text
Step 1:
    Full forward pass → initialize cache

Step t:
    Input only latest token
    Reuse cached K/V
    Predict next token
```

---

## Complexity Improvement

| Metric                    | Without Cache | With Cache |
| ------------------------- | ------------- | ---------- |
| Attention Complexity      | O(n²)         | O(n)       |
| Tokens Processed Per Step | Full Sequence | 1 Token    |
| Decoding Speed            | Slower        | Faster     |

---

# Generation & Sampling

Text generation is performed autoregressively using KV-cache optimized decoding.

At each decoding step:

1. Predict logits
2. Apply sampling strategy
3. Sample next token
4. Append token to sequence

---

## Temperature Sampling

Controls randomness:

```python
logits = logits / temperature
```

| Temperature | Effect             |
| ----------- | ------------------ |
| < 1.0       | More deterministic |
| = 1.0       | Default behavior   |
| > 1.0       | More random        |

---

## Top-K Sampling

Restricts sampling to the top-k most probable tokens.

---

## Top-P (Nucleus) Sampling

Selects the smallest set of tokens whose cumulative probability exceeds `top_p`.

---

## Sampling Pipeline

```text
Temperature
    ↓
Top-K
    ↓
Top-P
    ↓
Softmax
    ↓
Multinomial Sampling
```

---

## EOS Handling

During batched generation:

* completed sequences are marked finished
* future outputs are forced to EOS
* unfinished sequences continue generating

This preserves:

* fixed tensor shapes
* KV-cache alignment
* batch consistency

---

# Distributed Inference Architecture

## System Overview

```text
Client
   ↓
FastAPI API
   ↓
Redis Queue
   ↓
Worker Process
   ↓
GPT-2 Model
   ↓
Redis Result Store
   ↓
Client Polling API
```

---

## API Layer (FastAPI)

### `POST /generate`

Accepts:

* prompt
* generation parameters

Workflow:

1. Generate unique `task_id`
2. Push request into Redis queue
3. Return immediately

---

### `GET /result/{task_id}`

Returns:

* queued
* processing
* completed result
* not_found

---

### `GET /health`

Redis health check endpoint.

---

## Redis Queue Layer

Redis is used as:

* task queue
* message broker
* temporary result store

Queue:

```text
llm_queue
```

Status keys:

```text
status:{task_id}
```

Result keys:

```text
result:{task_id}
```

---

## Worker-Based Inference

Workers continuously:

```python
while True:
    task = queue.pop()
    run inference
    store result
```

Responsibilities:

* tokenization
* batching
* inference
* decoding
* result storage

---

# Dynamic Batching Prototype

The worker implements a timeout-based batching strategy.

## Batching Flow

```text
1. Wait for first request
2. Start timeout window
3. Collect additional queued requests
4. Run batched inference
```

---

## Current Batch Configuration

| Parameter     | Value |
| ------------- | ----- |
| Batch Size    | 8     |
| Batch Timeout | 10ms  |

---

## Current Limitation

The current batching implementation assumes homogeneous sampling parameters within a batch.

For example:

* all requests in the same batch share the same temperature
* all requests share the same top-k/top-p configuration

This is a prototype batching implementation, not a continuous batching runtime.

---

# Evaluation Pipeline

Evaluation scripts:

```text
eval/
├── benchmark.py
├── evaluate.py
├── kv_cache.py
└── sample_generation.py
```

Generated artifacts:

```text
results/
├── generation_benchmark.json
├── generations.md
├── kv_cache_benchmarks.json
└── perplexity.json
```

---

# Perplexity Evaluation

## Dataset

* WikiText-2 validation split

---

## Evaluation Flow

```text
WikiText-2 Validation Set
            ↓
Tokenization
            ↓
Forward Pass
            ↓
Cross-Entropy Loss
            ↓
Perplexity
```

---

## Current Results

| Metric          | Value   |
| --------------- | ------- |
| Validation Loss | 4.1028  |
| Perplexity      | 60.5097 |

These results validate correctness of the custom GPT-2 implementation when loading pretrained GPT-2 weights.

---

# Inference Benchmark

## Throughput Results

| Metric           | Value   |
| ---------------- | ------- |
| Device           | CPU     |
| Latency          | 2.7757s |
| Tokens/sec       | 36.03   |
| Prompt Tokens    | 5       |
| Generated Tokens | 100     |

This benchmark measures end-to-end autoregressive generation throughput.

---

# KV Cache Benchmark

## Results

| Configuration    | Time    |
| ---------------- | ------- |
| Without KV Cache | 7.8321s |
| With KV Cache    | 1.9318s |

Observed speedup:

```text
4.05× faster autoregressive decoding
```

---

## Complexity Comparison

| Metric                    | Without Cache | With Cache |
| ------------------------- | ------------- | ---------- |
| Attention Complexity      | O(n²)         | O(n)       |
| Tokens Processed Per Step | Full Sequence | 1 Token    |
| Decoding Speed            | Slower        | Faster     |

---

# Sample Generation

Example prompt:

```text
The future of artificial intelligence
```

Example configuration:

```text
temperature = 0.8
top_k = 50
top_p = 0.9
```

The project demonstrates:

* short-range coherence
* autoregressive generation
* configurable decoding behavior

Generation quality reflects the limitations of GPT-2 small relative to modern instruction-tuned LLMs.

---

# Testing

The project includes both unit and integration tests.

## Test Categories

### Unit Tests

* attention correctness
* causal masking
* dropout behavior
* embeddings
* generation pipeline
* KV-cache behavior
* sampling
* tensor shape validation

### Integration Tests

* API endpoints
* Redis interaction
* worker batching
* full worker pipeline

### Performance Tests

* generation speed benchmarking

---

## Test Results

```text
20 passed in 23.83s
```

Run tests:

```bash
pytest
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/TechFox6905/distributed-gpt2-inference.git
cd Gpt-from-Scratch
```

---

## Install uv

### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:

```bash
uv --version
```

---

## Create Virtual Environment

```bash
uv venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Project

```bash
uv pip install -e .
```

Install development dependencies:

```bash
uv pip install -e .[dev]
```

---

# Docker Deployment

## Dockerfile

The project includes a CUDA-enabled Docker setup based on:

```text
pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
```

The container:

* installs the project using `pyproject.toml`
* exposes the FastAPI server on port `8000`
* supports HuggingFace cache persistence
* supports GPU inference through Docker Compose

---

## Run With Docker Compose

Start the full stack:

```bash
docker compose up --build
```

Services started:

| Service | Purpose                  |
| ------- | ------------------------ |
| api     | FastAPI inference server |
| worker  | GPT-2 inference worker   |
| redis   | Queue and result store   |

---

## GPU Support

The worker service is configured with:

```yaml
gpus: all
```

This enables GPU inference when NVIDIA Container Toolkit is installed.

---

## Persistent HuggingFace Cache

Model weights are cached using a Docker volume:

```text
hf_cache
```

This avoids repeated model downloads between container restarts.

---

# Running the System

## Start Redis

```bash
docker run -p 6379:6379 redis
```

---

## Start API Server

```bash
python -m gpt.api.main
```

---

## Start Worker

```bash
python -m gpt.worker.worker
```

---

# Example API Usage

## Generate Text

```bash
curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": "The future of AI",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9
}'
```

Response:

```json
{
  "task_id": "<task-id>"
}
```

---

## Fetch Result

```bash
curl http://localhost:8000/result/<task-id>
```

---

# Current Limitations

## Model

* No attention-mask support for padded batches
* No FlashAttention / SDPA optimization
* No distributed training
* No mixed precision training
* No tokenizer training pipeline

---

## Inference System

* Prototype batching implementation
* No request prioritization
* No streaming responses
* No WebSocket support
* No authentication
* No rate limiting
* No continuous batching runtime

---

# Future Improvements

## Training

* Learning rate scheduling
* Gradient clipping
* Mixed precision training
* Checkpointing
* Distributed training
* Larger-scale datasets

---

## Inference

* Attention-mask support for padded batches
* Continuous batching
* Streaming token generation
* Async Redis client
* GPU inference optimization
* Request prioritization
* Better observability and metrics

---

# Comparison with Official GPT-2

| Feature                          | This Project | GPT-2 (OpenAI) |
| -------------------------------- | ------------ | -------------- |
| Decoder-only Transformer         | ✅            | ✅              |
| Weight Tying                     | ✅            | ✅              |
| KV Cache                         | ✅            | ✅              |
| Dropout                          | ✅            | ✅              |
| HuggingFace Weight Compatibility | ✅            | ✅              |
| Distributed Training             | ❌            | ✅              |
| FlashAttention / Fused Kernels   | ❌            | ✅              |
| Production Inference Runtime     | ❌            | ✅              |

---

# Technical Scope

This project demonstrates:

* Transformer architecture implementation
* GPT-2 weight compatibility
* KV-cache optimized decoding
* Sampling pipelines
* Queue-based inference architecture
* Dynamic batching concepts
* Evaluation and benchmarking workflows

The project does not implement:

* large-scale GPT training
* distributed optimization
* production-grade LLM serving
* tokenizer training
* modern optimized inference kernels

---

# Purpose

This project is intended for:

* learning Transformer internals
* understanding GPT-style architectures
* studying KV-cache optimization
* exploring LLM inference system design
* benchmarking autoregressive decoding
* experimenting with distributed inference concepts

---

# Disclaimer

This is a learning-oriented GPT-2 reimplementation and inference system prototype.

Additional engineering would be required for:

* production deployment
* large-scale serving
* optimized GPU inference
* reliability engineering
* observability
* distributed orchestration
