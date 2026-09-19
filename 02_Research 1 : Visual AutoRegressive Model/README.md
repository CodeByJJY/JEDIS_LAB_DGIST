# Research 1: Visual AutoRegressive Model

This directory contains notes, papers, and research ideas related to efficient Visual AutoRegressive (VAR) model inference.

The current research focuses on accelerating VAR inference, especially the computational bottleneck at high-resolution scale steps.

---

## 1. Research Overview

Visual AutoRegressive Modeling (VAR) reformulates conventional autoregressive image generation from **next-token prediction** to **next-scale prediction**.

Instead of generating visual tokens one by one, VAR generates an entire token map at each scale and progressively increases the image resolution.

```text
Conventional AR

Token 1
  ↓
Token 2
  ↓
Token 3
  ↓
...
  ↓
Token N
```

```text
Visual AutoRegressive Modeling

Coarse Scale
    ↓
Higher Resolution
    ↓
Higher Resolution
    ↓
...
    ↓
Fine Scale
```

Within each scale, multiple visual tokens can be generated in parallel.

This significantly reduces the number of sequential decoding steps compared with conventional autoregressive image generation.

---

## 2. Why Visual AutoRegressive Modeling?

Conventional visual autoregressive models suffer from several limitations:

- Slow token-by-token generation
- Loss of spatial structure caused by raster-order generation
- Limited scalability toward high-resolution image generation
- Lower generation quality compared with diffusion models

VAR addresses these problems through **coarse-to-fine next-scale prediction**.

The original VAR model demonstrated that GPT-style autoregressive models can achieve image generation quality competitive with or better than diffusion transformers while requiring substantially fewer sequential generation steps.

---

## 3. VAR Inference

VAR represents an image using multiple token maps of progressively increasing resolution.

```text
Scale 1
  ↓
Scale 2
  ↓
Scale 3
  ↓
...
  ↓
Scale K
  ↓
Decoder
  ↓
Image
```

The early scales mainly establish the global structure of the image, while later scales refine local textures and details.

A key characteristic of VAR inference is therefore:

```text
Early scales
Small token maps
Low computation cost

        ↓

Later scales
Large token maps
High computation cost
High KV-cache cost
```

---

## 4. Remaining Bottlenecks

Although VAR requires fewer sequential steps than conventional autoregressive models and diffusion models, its inference cost is still highly unbalanced across scales.

### Computation Bottleneck

Later scale steps contain substantially more tokens than early scale steps.

As the resolution increases:

- Attention cost increases rapidly
- FFN computation increases
- GPU execution time becomes concentrated in the last few scales

### Memory Bottleneck

VAR inference also keeps K/V states from previous scales.

As more tokens are introduced at higher resolutions:

- KV-cache size increases
- Memory traffic increases
- High-resolution generation becomes difficult
- Large batch inference becomes increasingly memory intensive

Therefore, efficient VAR inference requires optimization of both:

```text
Computation
+
Memory
```

---

# 5. Key Papers

## 00. Visual Autoregressive Modeling

**Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction**

- Introduces the Visual AutoRegressive modeling paradigm.
- Replaces next-token prediction with coarse-to-fine next-scale prediction.
- Establishes the basic architecture and inference mechanism of VAR.

Paper:
https://arxiv.org/abs/2404.02905

Notes:

---

## 01. Infinity

**Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis**

- Extends VAR toward high-resolution text-to-image generation.
- Replaces index-wise token prediction with bitwise prediction.
- Introduces a massive effective vocabulary and improves reconstruction quality.

Paper:
https://arxiv.org/abs/2412.04431

Notes:

---

## 02. FastVAR

**FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning**

- Analyzes the computation bottleneck of later VAR scales.
- Prunes less important tokens at large-scale steps.
- Restores removed tokens using cached information from previous scales.

Paper:
https://arxiv.org/abs/2503.23367

Notes:

---

## 03. ScaleKV

**Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression**

- Analyzes the KV-cache memory bottleneck of VAR.
- Identifies different attention behaviors across layers and scales.
- Allocates KV-cache budgets adaptively using scale-aware cache compression.

Paper:
https://arxiv.org/abs/2505.19602

Notes:

---

# 6. Research Direction 1: Multi-Sample VAR Pipelining

## Motivation

VAR inference has strongly imbalanced computation across scale steps.

```text
Early scales  → relatively cheap
Later scales  → increasingly expensive
```

If multiple images are generated independently, executing each image sequentially may leave opportunities for better GPU utilization.

The current idea is to reorganize multi-sample inference into a pipeline.

---

## Naive Pipeline Idea

Group scale steps with similar computational costs into pipeline stages.

Example:

```text
Stage 1 : Scale 0 - 5
Stage 2 : Scale 6 - 7
Stage 3 : Scale 8 - 9
Stage 4 : Scale 10
Stage 5 : Scale 11
Stage 6 : Scale 12
```

Multiple image generations can then overlap across stages.

```text
Time ------------------------------------------------------>

Stage 1   Img A   Img B   Img C   Img D   Img E   Img F

Stage 2           Img A   Img B   Img C   Img D   Img E

Stage 3                   Img A   Img B   Img C   Img D

Stage 4                           Img A   Img B   Img C

Stage 5                                   Img A   Img B

Stage 6                                           Img A
```

The goal is to improve **batch inference throughput** by exploiting the asymmetric execution cost across VAR scales.

---

## Research Questions

Important questions include:

- How should VAR scales be grouped into pipeline stages?
- How should stage boundaries be determined?
- Can multiple samples actually execute concurrently on the GPU?
- How much GPU utilization is currently lost during single-sample inference?
- How should intermediate activations and KV caches be managed?
- What is the throughput gain as batch size increases?
- What additional memory overhead does pipelining introduce?
- How does pipelining affect end-to-end latency?
- What is the optimal trade-off between latency, throughput, and memory usage?

---

## Evaluation Metrics

### Performance

- End-to-end latency
- Throughput
- Images / second
- GPU utilization
- SM utilization
- Memory bandwidth utilization

### Memory

- Peak GPU memory
- KV-cache memory
- Intermediate activation memory

### Generation Quality

- FID
- GenEval
- DPG
- CLIP Score

---

# 7. Research Direction 2: KV Cache Channel Compression

Existing VAR KV-cache compression methods mainly reduce the cache along the **token dimension**.

```text
Full KV Cache

Tokens × Channels
```

Previous direction:

```text
Reduce Tokens
    ↓
Keep fewer historical K/V states
```

Possible alternative direction:

```text
Reduce Channels
    ↓
Keep fewer K/V feature dimensions
```

Questions to investigate:

- Are all KV channels equally important?
- Are important channels consistent across scales?
- Are important channels consistent across transformer layers?
- Can activation statistics identify redundant KV dimensions?
- Can channel pruning reduce memory bandwidth as well as memory capacity?
- How much reconstruction or generation quality is lost after channel reduction?

---

# 8. Related Efficiency Techniques

Relevant optimization directions include:

- Token pruning
- Token merging
- KV-cache compression
- KV-cache eviction
- Quantization
- Parallel decoding
- Pipelining
- Batch inference
- FlashAttention
- Kernel fusion

---

# 9. Research Materials

## Internal Materials

### VAR Literature Review

`Meeting_20260626_VARliteraturereview.pdf`

Topics:

- Conventional visual autoregressive modeling
- VQ-based visual tokenization
- Parallel autoregressive generation
- VAR
- Infinity
- Efficient visual generation

### VAR Overview

`Meeting_20260824_VAROverview.pdf`

Topics:

- VAR inference mechanism
- Scale-wise computation characteristics
- VAR inference bottlenecks
- FastVAR
- ScaleKV
- Multi-sample VAR pipelining
- KV-cache channel pruning

---

# 10. Study Plan

## Phase 1. Understand VAR

- [ ] Autoregressive modeling
- [ ] Visual tokenization
- [ ] VQVAE / VQGAN
- [ ] Next-token prediction
- [ ] Next-scale prediction
- [ ] VAR inference mechanism
- [ ] KV cache in VAR

## Phase 2. Read Core Papers

- [ ] Visual Autoregressive Modeling
- [ ] Infinity
- [ ] FastVAR
- [ ] ScaleKV

## Phase 3. Profile VAR Inference

- [ ] Reproduce baseline inference
- [ ] Measure per-scale latency
- [ ] Measure per-scale token count
- [ ] Measure GPU utilization
- [ ] Measure KV-cache memory
- [ ] Identify late-scale bottlenecks

## Phase 4. Multi-Sample Pipeline

- [ ] Define pipeline stages
- [ ] Implement baseline multi-sample execution
- [ ] Implement pipelined execution
- [ ] Measure latency
- [ ] Measure throughput
- [ ] Measure GPU utilization
- [ ] Measure memory overhead

## Phase 5. Optimization

- [ ] Optimize stage partitioning
- [ ] Explore asynchronous execution
- [ ] Explore CUDA stream scheduling
- [ ] Compare against normal batching
- [ ] Evaluate quality / performance trade-offs

---

# Research Goal

The ultimate goal is to understand the remaining system bottlenecks of Visual AutoRegressive models and develop efficient inference techniques that improve:

- Throughput
- GPU utilization
- Memory efficiency
- High-resolution scalability

while preserving generation quality.
