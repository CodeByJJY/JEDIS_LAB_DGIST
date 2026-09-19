# Computer Architecture

This section summarizes the core computer architecture concepts required for research in:

- Efficient AI inference
- GPU systems
- Memory systems
- Model compression
- Hardware-aware optimization
- Performance profiling

---

## Table of Contents

1. [Instruction Set Architecture](#1-instruction-set-architecture)
2. [Processor Execution Models](#2-processor-execution-models)
3. [Pipelining](#3-pipelining)
4. [Pipeline Hazards](#4-pipeline-hazards)
5. [Performance Metrics](#5-performance-metrics)
6. [Memory Hierarchy](#6-memory-hierarchy)
7. [Cache Memory](#7-cache-memory)
8. [Virtual Memory](#8-virtual-memory)
9. [Parallelism](#9-parallelism)
10. [SIMD and MIMD](#10-simd-and-mimd)
11. [GPU Architecture](#11-gpu-architecture)
12. [Memory Bandwidth and Data Movement](#12-memory-bandwidth-and-data-movement)
13. [Compute-Bound vs Memory-Bound Workloads](#13-compute-bound-vs-memory-bound-workloads)
14. [Research Connections](#14-research-connections)

---

# 1. Instruction Set Architecture

Topics:

- Instruction Set Architecture (ISA)
- Instructions
- Registers
- Operands
- Addressing modes
- Load / store architecture
- RISC vs CISC

Key idea:

ISA defines the interface between software and hardware.

---

# 2. Processor Execution Models

Topics:

- Single-cycle processor
- Multi-cycle processor
- Datapath
- Control unit
- Instruction execution stages

Key idea:

Processor performance depends on how instruction execution is divided and overlapped.

---

# 3. Pipelining

Typical pipeline stages:

- IF: Instruction Fetch
- ID: Instruction Decode
- EX: Execute
- MEM: Memory Access
- WB: Write Back

Key idea:

Pipelining overlaps the execution of multiple instructions to improve throughput.

Important distinction:

- Latency: time required for one instruction or task to complete
- Throughput: number of instructions or tasks completed per unit time

---

# 4. Pipeline Hazards

Main types:

- Data hazard
- Control hazard
- Structural hazard

Common solutions:

- Stall
- Forwarding / bypassing
- Branch prediction
- Pipeline flushing
- Resource duplication

Key idea:

Hazards prevent ideal pipeline execution and reduce performance.

---

# 5. Performance Metrics

Important metrics:

- Latency
- Throughput
- Clock frequency
- CPI
- IPC
- Execution time
- Speedup
- Efficiency

Basic relationship:

```math
\mathrm{Execution\ Time}
=
\mathrm{Instruction\ Count}
\times
\mathrm{CPI}
\times
\mathrm{Clock\ Cycle\ Time}
```

Amdahl's Law:

```math
\mathrm{Speedup}
=
\frac{1}
{(1-P)+\frac{P}{S}}
```

where:

- $P$ is the fraction of execution that can be accelerated
- $S$ is the speedup of the accelerated portion

Research relevance:

- Optimization bottleneck analysis
- Parallel acceleration
- Kernel optimization
- End-to-end inference speedup

---

# 6. Memory Hierarchy

Typical hierarchy:

```text
Registers
   ↓
Cache
   ↓
Main Memory
   ↓
Storage
```

Topics:

- Registers
- SRAM
- DRAM
- Storage
- Access latency
- Capacity
- Bandwidth
- Locality

Key idea:

Faster memory is usually smaller and more expensive, while larger memory is slower.

---

# 7. Cache Memory

Topics:

- Temporal locality
- Spatial locality
- Cache line
- Cache hit / miss
- Hit rate
- Miss penalty
- Direct-mapped cache
- Set-associative cache
- Fully associative cache
- Cache replacement policies

Key idea:

Caches reduce average memory access time by exploiting locality.

---

# 8. Virtual Memory

Topics:

- Virtual address
- Physical address
- Page
- Page table
- Translation Lookaside Buffer (TLB)
- Page fault

Key idea:

Virtual memory provides an abstraction between program address spaces and physical memory.

---

# 9. Parallelism

Important forms of parallelism:

- Instruction-Level Parallelism (ILP)
- Data-Level Parallelism (DLP)
- Thread-Level Parallelism (TLP)
- Task-Level Parallelism

Key idea:

Performance can be improved by executing independent work simultaneously.

---

# 10. SIMD and MIMD

Flynn's taxonomy:

- SISD
- SIMD
- MISD
- MIMD

### SIMD

Single Instruction, Multiple Data

The same instruction is applied to multiple data elements in parallel.

Applications:

- Vector processors
- GPU execution
- Matrix operations

### MIMD

Multiple Instruction, Multiple Data

Multiple processors execute different instruction streams on different data.

Applications:

- Multicore CPUs
- Distributed systems

---

# 11. GPU Architecture

Important concepts:

- Streaming Multiprocessor (SM)
- CUDA cores
- Tensor Cores
- Warps
- Threads
- Thread blocks
- Grid
- Shared memory
- Registers
- Global memory

Execution hierarchy:

```text
Grid
 └── Thread Blocks
      └── Warps
           └── Threads
```

Key idea:

GPUs achieve high throughput by executing large numbers of lightweight threads in parallel.

---

# 12. Memory Bandwidth and Data Movement

Topics:

- Memory bandwidth
- Memory latency
- DRAM access
- Cache access
- Data movement
- Arithmetic intensity

Key idea:

Modern AI workloads are often limited not only by computation, but also by the cost of moving data between memory levels.

Research relevance:

- Neural network inference
- Quantization
- Model compression
- GPU optimization
- Memory-bound workloads

---

# 13. Compute-Bound vs Memory-Bound Workloads

## Compute-Bound

A workload is compute-bound when execution time is mainly limited by arithmetic throughput.

Typical indicators:

- High compute utilization
- High arithmetic intensity
- Compute units close to saturation

---

## Memory-Bound

A workload is memory-bound when execution time is mainly limited by data movement or memory bandwidth.

Typical indicators:

- High memory bandwidth utilization
- Low compute utilization
- Frequent memory accesses
- Low arithmetic intensity

---

## Roofline Perspective

Performance can be limited by either:

- Peak compute throughput
- Peak memory bandwidth

Arithmetic intensity is commonly defined as:

```math
\mathrm{Arithmetic\ Intensity}
=
\frac{\mathrm{FLOPs}}
{\mathrm{Bytes\ Transferred}}
```

This helps determine whether a workload is more likely to be compute-bound or memory-bound.

---

# 14. Research Connections

## Neural Network Quantization

Important concepts:

- Integer arithmetic
- Memory bandwidth
- Data movement
- Tensor Core support
- Hardware efficiency

---

## Model Compression

Important concepts:

- Memory footprint
- Cache behavior
- DRAM access
- Arithmetic intensity
- Sparse computation

---

## GPU Profiling

Important concepts:

- Kernel execution
- SM utilization
- Memory throughput
- Occupancy
- Warp scheduling
- Kernel launch overhead

---

## Visual Autoregressive Modeling

Important concepts:

- Sequential dependency
- Parallel execution
- Pipelining
- Batch inference
- GPU utilization
- KV cache
- Memory bandwidth
- Throughput vs latency

---

# Study Goal

The goal is to understand how processor architecture, memory systems, and parallel execution affect the real performance of AI models.

The focus is not only on theoretical computation, but also on how hardware characteristics determine actual inference latency and throughput.
