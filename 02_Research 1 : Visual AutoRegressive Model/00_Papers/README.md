# Visual Autoregressive Modeling - Papers

This directory contains the core papers for studying **Visual Autoregressive (VAR) Models** and their efficient inference.

The papers are organized in the following order:

1. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
2. Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis
3. FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning
4. Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression

---

# 00. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

> **Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang, 2024**

- **Problem:** Conventional image autoregressive models flatten 2D visual tokens into a 1D raster-scan sequence and generate them token by token, resulting in poor spatial modeling and very high generation cost as image resolution increases.
- **Main Idea:** VAR replaces conventional **next-token prediction** with coarse-to-fine **next-scale prediction**, where an entire token map at the next resolution is predicted in parallel conditioned on previously generated lower-resolution token maps.
- **Results:** On ImageNet 256×256, VAR achieves **FID 1.73 / IS 350.2**, substantially improves the VQGAN AR baseline, and reports roughly **20× faster inference**, while also demonstrating power-law scaling behavior and zero-shot image editing capabilities.

---

# 01. Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

> **Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu**

- **Problem:** VAR still relies on index-based discrete visual tokens whose limited vocabulary causes quantization error and loss of fine-grained details, while conventional large-vocabulary classifiers become prohibitively expensive and teacher forcing introduces cumulative generation errors.
- **Main Idea:** Infinity replaces index-wise tokens with **bitwise tokens** using a bitwise visual tokenizer, an **Infinite-Vocabulary Classifier (IVC)**, and **Bitwise Self-Correction (BSC)**, enabling extremely large effective vocabularies while retaining VAR's next-scale generation framework.
- **Results:** Infinity-2B achieves **GenEval 0.73**, **ImageReward 0.962**, and generates a **1024×1024 image in about 0.8 s**, outperforming several diffusion models while being approximately **2.6× faster than SD3-Medium**.

---

# 02. FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning

> **Hang Guo, Yawei Li, Taolin Zhang, Jiangshan Wang, Tao Dai, Shu-Tao Xia, Luca Benini**

- **Problem:** Although VAR reduces the number of autoregressive steps, every large-scale step still processes the entire high-resolution token map, causing token count, attention cost, GPU memory consumption, and inference latency to grow rapidly with resolution.
- **Main Idea:** FastVAR observes that low-frequency structure has mostly converged at later scales and introduces **Cached Token Pruning**, consisting of **Pivotal Token Selection (PTS)** to process only high-frequency important tokens and **Cached Token Restoration (CTR)** to reconstruct pruned positions using tokens cached from previous scales.
- **Results:** FastVAR provides up to **2.7× additional speedup over a FlashAttention-accelerated VAR baseline with less than 1% performance degradation**, and enables high-resolution generation such as a **2K image using about 15 GB memory in 1.5 s on a single RTX 3090**.

---

# 03. Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression

> **Kunjun Li, Zigeng Chen, Cheng-Yen Yang, Jenq-Neng Hwang**

- **Problem:** VAR must preserve KV states from all previous scales, causing the KV cache to grow rapidly as resolution and batch size increase; for example, Infinity-8B requires approximately **85 GB of KV cache** for 1024×1024 generation at batch size 8.
- **Main Idea:** **ScaleKV** analyzes layer- and scale-dependent attention patterns, classifies layer-scale pairs into **Drafters** that require broad historical context and **Refiners** that mainly use local/current-scale information, and allocates different KV-cache budgets accordingly.
- **Results:** ScaleKV reduces Infinity-8B KV-cache usage from roughly **85 GB to 8.5 GB (10%)** while keeping **GenEval 0.792 → 0.790** and **DPG 86.61 → 86.49**, and also provides up to approximately **1.25× inference speedup**.

---

# Research Flow

These papers form the following progression:

```text
VAR
│
│  Next-Token Prediction
│          ↓
│  Next-Scale Prediction
│
▼
Infinity
│
│  Index-Wise Tokens
│          ↓
│  Bitwise Tokens
│  + Infinite Vocabulary
│  + Self-Correction
│
▼
FastVAR
│
│  Process Every Token
│          ↓
│  Process Only Pivotal Tokens
│  + Restore from Cached Scales
│
▼
ScaleKV
   │
   │  Uniform KV Cache
   │          ↓
   │  Layer- and Scale-Aware
   │  KV Cache Allocation
   │
   ▼
Efficient High-Resolution
Visual Autoregressive Inference
```

---

# Research Perspective

The four papers can be viewed from two complementary directions:

### Generation Architecture

```text
VAR
→ Next-Scale Prediction

Infinity
→ Better Token Representation
   and High-Resolution Generation
```

### Inference Efficiency

```text
FastVAR
→ Reduce Forwarded Tokens

ScaleKV
→ Reduce Stored KV Tokens
```

Therefore, the main efficiency bottlenecks to investigate are:

```text
Computation
+
Token Processing
+
Attention
+
KV Cache
+
Memory Traffic
```

especially at the later high-resolution scales of Visual Autoregressive Models.
