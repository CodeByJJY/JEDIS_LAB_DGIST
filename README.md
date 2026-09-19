# DGIST JEDIS Lab

Research notes, paper reviews, and study materials for research preparation and ongoing projects at DGIST JEDIS Lab.

This repository is organized into three main parts:

1. Background studies
2. Fundamental papers
3. Research projects

---

## Table of Contents

1. [Background](#1-background)
2. [Fundamental Papers](#2-fundamental-papers)
3. [Research 1: Visual Autoregressive Modeling](#3-research-1-visual-autoregressive-modeling)
4. [Research 2](#4-research-2)

---

# 1. Background

Fundamental topics required for understanding and conducting research.

| Category | Topic | Notes |
|---|---|---|
| Mathematics | Linear Algebra |  |
| Computer Systems | Computer Architecture |  |
| Mathematics | Probability & Statistics |  |

---

## 1.1 Linear Algebra

Topics to study:

- Vector spaces
- Linear independence
- Basis and dimension
- Matrix operations
- Rank
- Eigenvalues and eigenvectors
- Orthogonality
- Singular Value Decomposition (SVD)
- Positive definite matrices
- Quadratic forms

---

## 1.2 Computer Architecture

Topics to study:

- Instruction Set Architecture (ISA)
- Pipelining
- Data / control / structural hazards
- Memory hierarchy
- Cache
- Virtual memory
- Parallelism
- SIMD / MIMD
- GPU architecture
- Memory bandwidth
- Compute-bound vs. memory-bound workloads

---

## 1.3 Probability & Statistics

Topics to study:

- Random variables
- Probability distributions
- Conditional probability
- Bayes' theorem
- Expectation
- Variance
- Covariance
- Correlation
- Common probability distributions
- Maximum Likelihood Estimation
- Sampling
- Law of Large Numbers
- Central Limit Theorem

---

# 2. Fundamental Papers

This section covers foundational and recent papers on:

- Neural network compression
- Quantization
- Post-Training Quantization (PTQ)
- LLM quantization
- Rotation-based quantization
- Structured pruning

The goal is to understand how model compression methods evolved from classical neural network compression techniques to recent methods for large-scale Transformer and Vision Transformer models.

---

## Paper List

| No. | Year | Paper | Topic | Notes |
|:---:|:---:|---|---|:---:|
| 00 | 2015 | Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding | `compression` `pruning` `quantization` |  |
| 01 | 2018 | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | `quantization` `QAT` |  |
| 02 | 2020 | Up or Down? Adaptive Rounding for Post-Training Quantization | `PTQ` `rounding` |  |
| 03 | 2022 | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | `LLM quantization` `outlier` |  |
| 04 | 2022 | Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning | `PTQ` `pruning` `second-order` |  |
| 05 | 2022 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | `LLM PTQ` `second-order` |  |
| 06 | 2022 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | `W8A8` `activation` |  |
| 07 | 2023 | AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration | `weight-only quantization` |  |
| 08 | 2024 | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | `rotation` `quantization` |  |
| 09 | 2024 | SpinQuant: LLM Quantization with Learned Rotations | `learned rotation` `quantization` |  |
| 10 | 2025 | Variance-Based Pruning for Accelerating and Compressing Trained Networks | `structured pruning` `variance` |  |
| 11 | 2026 | Denoised Variance-Based Pruning with Optimal Brain Bias Compensation | `structured pruning` `covariance` `OBC` |  |

---

## 2.1 Classical Model Compression

### Deep Compression

Main topics:

- Network pruning
- Trained quantization
- Weight sharing
- Huffman coding
- Memory footprint reduction
- Memory access reduction

---

## 2.2 Integer Quantization and Quantization-Aware Training

### Quantization and Training of Neural Networks

Main topics:

- Integer-only inference
- INT8 arithmetic
- Scale
- Zero-point
- Quantization-aware training
- Hardware-efficient inference

---

## 2.3 Post-Training Quantization

### AdaRound

Main topics:

- Post-training quantization
- Adaptive rounding
- Layer-wise reconstruction
- Data-dependent quantization

### Optimal Brain Compression

Main topics:

- Optimal Brain Surgeon
- Second-order information
- Layer-wise reconstruction
- Unified pruning and quantization
- Optimal Brain Quantization

### GPTQ

Main topics:

- LLM post-training quantization
- Second-order information
- Quantization error compensation
- Low-bit weight quantization

---

## 2.4 Large Language Model Quantization

### LLM.int8()

Main topics:

- Activation outliers
- Vector-wise quantization
- Mixed-precision decomposition
- INT8 matrix multiplication

### SmoothQuant

Main topics:

- W8A8 quantization
- Activation smoothing
- Activation outliers
- Migration of quantization difficulty from activations to weights

### AWQ

Main topics:

- Activation-aware weight importance
- Salient weight channels
- Low-bit weight-only quantization
- Per-channel scaling

---

## 2.5 Rotation-Based Quantization

### QuaRot

Main topics:

- Rotation-based quantization
- Hadamard transformation
- Activation outlier removal
- Weight quantization
- Activation quantization
- KV cache quantization

### SpinQuant

Main topics:

- Learned rotation matrices
- Rotation optimization
- Quantization-aware learning
- Outlier reduction

---

## 2.6 Structured Pruning

### Variance-Based Pruning

Main topics:

- Structured pruning
- Activation variance
- Neuron importance
- Mean-shift compensation

### Denoised Variance-Based Pruning

Main topics:

- Activation covariance
- Covariance denoising
- Random matrix theory
- Structured pruning
- Optimal Brain Bias Compensation
- Remaining-weight reconstruction

---

# 3. Research 1: Visual Autoregressive Modeling

## Goal

Research on efficient Visual Autoregressive Model inference, with the goal of developing a research idea suitable for a CVPR submission.

Main research direction:

> Efficient batch inference for Visual Autoregressive Models through pipelining and inference optimization.

---

## 3.1 Background

Visual Autoregressive Modeling replaces conventional raster-scan next-token prediction with coarse-to-fine next-scale prediction.

Main concepts:

- Visual tokenization
- Autoregressive modeling
- Next-token prediction
- Next-scale prediction
- Multi-scale token maps
- Coarse-to-fine generation
- Parallel token prediction within each scale

---

## 3.2 Core Papers

| Year | Paper | Topic | Notes |
|:---:|---|---|:---:|
| 2024 | Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction | `generation` `VAR` |  |
| 2024 | Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis | `generation` `high-resolution` |  |
| 2025 | FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning | `efficiency` `token pruning` |  |
| 2025 | Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression | `memory efficiency` `KV cache` |  |

---

## 3.3 Seminar Materials

Materials for understanding the research background and current research direction.

### VAR Literature Review

Topics:

- Visual autoregressive modeling
- Related work
- Evolution of VAR methods
- Research trends

### VAR Overview

Topics:

- VAR architecture
- Inference process
- Scale-by-scale generation
- Efficiency bottlenecks
- Pipelining-based batch inference idea

---

## 3.4 Research Questions

Main questions to investigate:

1. What operations dominate VAR inference latency?
2. How does inference cost change across scales?
3. Which dependencies prevent parallel execution?
4. Which operations can be overlapped?
5. How does batch size affect GPU utilization?
6. Can different images or scales be pipelined?
7. What is the trade-off between latency and throughput?
8. How does KV cache usage change across scales?
9. What are the major memory and compute bottlenecks?
10. Can GPU utilization be improved without degrading generation quality?

---

## 3.5 Profiling

Metrics to analyze:

- End-to-end latency
- Per-scale latency
- GPU kernel execution time
- GPU utilization
- SM utilization
- Memory bandwidth
- Kernel launch overhead
- KV cache memory
- Batch size scaling
- Throughput
- Peak GPU memory usage

Tools:

- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- PyTorch Profiler
- CUDA Events

---

## 3.6 Experiments

### Baseline

Record:

- Model
- Resolution
- Number of scales
- Batch size
- GPU
- Precision
- Latency
- Throughput
- Peak memory usage

### Profiling

Analyze:

- Per-scale execution
- Kernel breakdown
- Memory behavior
- GPU idle periods
- Synchronization points

### Optimization

Potential directions:

- Batch inference
- Pipelining
- Kernel overlap
- KV cache optimization
- Token pruning
- Scale-aware scheduling

---

## 3.7 Experiment Log

Each experiment should record:

- Date
- Goal
- Environment
- Configuration
- Hypothesis
- Measurement
- Result
- Interpretation
- Next step

---

# 4. Research 2

To be added.

Future research projects can be organized using the same structure:

- Background
- Core Papers
- Research Questions
- Baseline
- Profiling
- Experiments
- Results

---

# Repository Structure

```text
DGIST-JEDIS-Lab/
│
├── README.md
│
├── 01_Background/
│   ├── Linear_Algebra/
│   ├── Computer_Architecture/
│   └── Probability_and_Statistics/
│
├── 02_Fundamental_Papers/
│
├── 03_Research_VAR/
│   ├── Papers/
│   ├── Seminars/
│   └── Experiments/
│
└── 04_Research_02/
