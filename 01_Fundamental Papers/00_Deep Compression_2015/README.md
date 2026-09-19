# Deep Compression

> **Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding**

- **Authors:** Song Han, Huizi Mao, William J. Dally
- **Venue:** ICLR 2016
- **Topic:** Neural Network Compression
- **Keywords:** Pruning, Quantization, Weight Sharing, Huffman Coding

---

## 1. Problem Background

Deep neural networks contain a large number of parameters, which leads to:

- Large model size
- High memory bandwidth requirements
- High energy consumption
- Difficulty deploying models on mobile and embedded systems

A particularly important problem is **memory access cost**.

The paper points out that under 45nm CMOS technology:

| Operation | Energy Cost |
|---|---:|
| 32-bit floating-point addition | 0.9 pJ |
| 32-bit SRAM access | 5 pJ |
| 32-bit DRAM access | 640 pJ |

Therefore, reducing model size can also reduce expensive off-chip DRAM accesses.

---

## 2. Main Idea

Deep Compression proposes a three-stage compression pipeline:

```text
Original Network
      │
      ▼
1. Pruning
      │
      ▼
2. Trained Quantization
   + Weight Sharing
      │
      ▼
3. Huffman Coding
      │
      ▼
Compressed Network
```

Each stage removes a different type of redundancy.

| Stage | Purpose |
|---|---|
| Pruning | Reduce the number of connections |
| Quantization | Reduce the number of bits required per weight |
| Huffman Coding | Losslessly compress weights and sparse indices |

---

## 3. Network Pruning

### Idea

Remove connections whose weights have small magnitudes.

```text
Train Network
     ↓
Prune Small Weights
     ↓
Retrain Remaining Weights
```

After pruning, the dense weight matrix becomes sparse.

The sparse weights are stored using formats such as:

- CSR (Compressed Sparse Row)
- CSC (Compressed Sparse Column)

The paper also stores **relative indices** instead of absolute indices to reduce storage overhead.

### Result

- AlexNet: about **9× fewer parameters**
- VGG-16: about **13× fewer parameters**

---

## 4. Trained Quantization and Weight Sharing

### Idea

Instead of storing every remaining weight independently, similar weights share the same value.

The paper uses **k-means clustering** for each layer.

```text
Original Weights
      ↓
K-means Clustering
      ↓
Shared Weight Centroids
      ↓
Store Centroid Index for Each Weight
```

For example:

```text
Many FP32 weights
      ↓
32 shared weights
      ↓
5-bit index per weight
```

For AlexNet:

- CONV layers: 256 shared weights → **8-bit indices**
- FC layers: 32 shared weights → **5-bit indices**

The shared centroids are then retrained to recover accuracy.

---

## 5. Huffman Coding

Huffman coding is applied after pruning and quantization.

It assigns:

- Short codes to frequently occurring values
- Long codes to less frequent values

The paper applies Huffman coding to:

- Quantized weights
- Sparse matrix indices

This provides an additional **20–30% storage reduction**.

---

## 6. Overall Compression Pipeline

```text
Original Network
      │
      │ Pruning
      ▼
9×–13× Reduction
      │
      │ Quantization
      ▼
27×–31× Reduction
      │
      │ Huffman Coding
      ▼
35×–49× Reduction
```

The important observation is that **pruning and quantization work well together**.

---

## 7. Experimental Results

### Compression Results

| Model | Original Size | Compressed Size | Compression Ratio |
|---|---:|---:|---:|
| LeNet-300-100 | 1070 KB | 27 KB | 40× |
| LeNet-5 | 1720 KB | 44 KB | 39× |
| AlexNet | 240 MB | 6.9 MB | 35× |
| VGG-16 | 552 MB | 11.3 MB | 49× |

The paper reports no meaningful accuracy degradation after compression.

### AlexNet

| Model | Top-1 Error | Top-5 Error |
|---|---:|---:|
| Original | 42.78% | 19.73% |
| Compressed | 42.78% | 19.70% |

### VGG-16

| Model | Top-1 Error | Top-5 Error |
|---|---:|---:|
| Original | 31.50% | 11.32% |
| Compressed | 31.17% | 10.91% |

---

## 8. Speedup and Energy Efficiency

The paper benchmarks the **pruned sparse network** with batch size = 1.

### Average Speedup

| Hardware | Speedup |
|---|---:|
| CPU | 3× |
| GPU | 3.5× |
| Mobile GPU | 4.2× |

### Energy Efficiency Improvement

| Hardware | Improvement |
|---|---:|
| CPU | 7× |
| GPU | 3.3× |
| Mobile GPU | 4.2× |

An important point is that these runtime measurements correspond mainly to the **pruned sparse layers**, not the entire pruning + quantization + Huffman pipeline.

---

## 9. Important Observations

### Pruning + Quantization

Pruning and quantization are more effective when used together than independently.

```text
Pruning
   +
Quantization
   ↓
Higher Compression
with Little Accuracy Loss
```

### CONV vs. FC Layers

CONV layers are more sensitive to aggressive quantization than FC layers.

- CONV layers require relatively higher precision.
- FC layers tolerate lower precision better.

### Memory Matters

For batch size = 1, inference is dominated more strongly by memory access.

Reducing model size can therefore improve:

- Memory bandwidth requirements
- Cache utilization
- Energy efficiency
- Inference latency

---

## 10. Limitations

The full compressed representation was difficult to execute efficiently using standard CPU/GPU libraries at the time.

In particular:

- Sparse computation requires efficient sparse kernels.
- Weight sharing requires indirect codebook lookup.
- Existing libraries did not efficiently support the complete compressed representation.

The paper therefore suggests:

- Custom GPU kernels
- Specialized hardware accelerators

---

## 11. Key Takeaways

1. Neural networks contain substantial parameter redundancy.
2. Pruning reduces the number of weights.
3. Quantization reduces the number of bits required per weight.
4. Huffman coding removes additional encoding redundancy.
5. Combining the three methods achieves **35×–49× model compression** without accuracy loss.
6. Reducing model size can also reduce expensive DRAM accesses.
7. Compression ratio and actual inference speedup are not necessarily the same.
8. Hardware and software support determine whether compression translates into real acceleration.

---

## 12. Keywords to Review

- Network Pruning
- Sparse Matrix
- CSR / CSC
- Quantization
- Weight Sharing
- K-means Clustering
- Codebook
- Huffman Coding
- SRAM / DRAM
- Memory Bandwidth
- Matrix-Vector Multiplication
- Sparse Computation
