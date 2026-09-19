# LLM.int8()

> **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**

- **Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Venue:** NeurIPS 2022
- **Topic:** Large Language Model Quantization
- **Keywords:** INT8, LLM Quantization, Activation Outliers, Vector-wise Quantization, Mixed Precision, Memory Efficiency

---

## 1. Problem Background

Large language models require substantial GPU memory during inference.

For large Transformer language models, most parameters and computation are concentrated in:

- Feed-forward layers
- Attention projection layers

The paper states that for models at and beyond 6.7B parameters, these matrix multiplication layers account for approximately:

- **95% of parameters**
- **65–85% of computation**

Reducing these matrices from 16-bit to 8-bit can therefore significantly reduce inference memory.

However, conventional INT8 quantization begins to fail as Transformer models become larger.

The main question of this paper is:

> Why does INT8 quantization fail at large model scales, and how can we quantize very large Transformers without performance degradation?

---

# 2. Main Idea

LLM.int8() combines two techniques:

```text
Vector-wise Quantization
          +
Mixed-Precision Decomposition
          ↓
       LLM.int8()
```

The method separates Transformer features into:

```text
Normal Features
    ↓
INT8 Matrix Multiplication

Outlier Features
    ↓
FP16 Matrix Multiplication
```

More than **99.9% of values** are still processed in INT8.

Only a very small number of important outlier feature dimensions are computed in FP16.

---

# 3. Why Conventional INT8 Quantization Fails

The main difficulty is the appearance of **large-magnitude activation features** in large Transformers.

Consider a hidden-state matrix:

```math
X \in \mathbb{R}^{s \times h}
```

where:

- $s$: sequence / token dimension
- $h$: hidden / feature dimension

At large model scales, some feature dimensions contain values whose magnitude is much larger than the remaining activations.

Conceptually:

```text
Normal Features

-2  1  0  3  -1  2

Outlier Feature

-2  1  0  60  -1  2
          ↑
       Outlier
```

If a single quantization scale is chosen to represent the entire tensor, the large outlier forces the quantization range to become very wide.

As a result:

```text
Large Quantization Range
        ↓
Large Quantization Step
        ↓
Poor Resolution for Normal Values
        ↓
Large Quantization Error
```

Many small values may even be rounded to zero.

---

# 4. Emergent Outlier Features

A major contribution of the paper is the analysis of these outlier features.

The authors observe that as Transformer models scale, large-magnitude activation features become increasingly systematic.

Around the **6.7B parameter scale**, a strong shift occurs:

- Outliers appear in essentially all Transformer layers.
- About 75% of sequence positions are affected.
- The outliers are concentrated in only a few hidden dimensions.
- Their magnitude can be much larger than ordinary features.

For a 6.7B Transformer with sequence length 2048:

```text
~150,000 outlier occurrences
```

are observed across the network, but these occurrences are concentrated in only:

```text
6 hidden dimensions
```

Therefore, the outliers are:

```text
Sparse in feature dimension
+
Systematic across layers and tokens
```

---

# 5. Why the Outliers Matter

The outlier dimensions are not numerical noise.

They are highly important for Transformer behavior.

The paper experimentally removes the outlier dimensions and observes:

- Mean top-1 attention softmax probability drops substantially.
- Validation perplexity increases by approximately 600–1000%.

In contrast, removing the same number of random feature dimensions causes almost no degradation.

Therefore:

> Outlier dimensions are rare, but they are highly important for model performance.

This creates the central quantization challenge:

```text
Outliers need high precision
but
Almost all other values can use INT8
```

---

# 6. Baseline INT8 Quantization

The paper studies several INT8 quantization strategies.

## Absmax Quantization

Absmax quantization maps values into the range:

```text
[-127, 127]
```

using the maximum absolute value.

Conceptually:

```math
X_{\mathrm{int8}}
=
\mathrm{round}
\left(
\frac{127}
{\max |X|}
X
\right)
```

The problem is that a single large outlier increases $\max |X|$ and reduces the effective precision for all other values.

---

## Zeropoint Quantization

Zeropoint quantization uses an affine transformation that exploits the full INT8 range.

It is particularly useful for asymmetric distributions.

The paper finds that it handles some outlier distributions better than symmetric absmax quantization.

However, even zeropoint quantization eventually fails at sufficiently large model scales.

---

# 7. Vector-Wise Quantization

The first component of LLM.int8() is **vector-wise quantization**.

Instead of using one scaling constant for an entire matrix, matrix multiplication is interpreted as a collection of independent inner products.

Given:

```math
X \in \mathbb{R}^{s \times h}
```

and

```math
W \in \mathbb{R}^{h \times o}
```

the method uses:

- One scaling constant for each row of $X$
- One scaling constant for each column of $W$

Conceptually:

```text
Activation Matrix X
↓
Row-wise Scaling

Weight Matrix W
↓
Column-wise Scaling
```

The INT8 matrix multiplication is performed first.

The result is then dequantized using the outer product of the corresponding row and column scaling factors.

---

## Why It Helps

Compared with tensor-wise quantization:

```text
One scale for entire tensor
```

vector-wise quantization provides much finer-grained normalization:

```text
Separate scale
for each inner product
```

This improves quantization precision significantly.

The paper shows that vector-wise quantization can preserve performance reasonably well at smaller billion-parameter scales.

However, it is still insufficient once the large systematic outlier features emerge.

---

# 8. Mixed-Precision Decomposition

The second and most important component is **mixed-precision decomposition**.

The idea is simple:

> Do not force the outlier feature dimensions into INT8.

Let the set of outlier feature dimensions be:

```math
O
=
\{i \mid |x_i| > \alpha\}
```

The paper uses:

```math
\alpha = 6.0
```

as the outlier threshold.

The matrix multiplication is decomposed into two parts:

```text
Normal Dimensions
        ↓
INT8 MatMul

Outlier Dimensions
        ↓
FP16 MatMul
```

The outputs are then added together.

Conceptually:

```math
XW
\approx
X_{\mathrm{outlier}}W_{\mathrm{outlier}}
+
X_{\mathrm{regular}}W_{\mathrm{regular}}
```

where:

```text
Outlier part  → FP16
Regular part  → INT8
```

---

# 9. Why Mixed Precision Is Efficient

The outlier dimensions are extremely sparse.

The paper reports that approximately:

```text
0.1%
```

of feature dimensions need high-precision treatment.

Therefore:

```text
~0.1%  → FP16
~99.9% → INT8
```

This preserves the memory benefit of INT8 while avoiding destructive quantization of the important outlier dimensions.

---

# 10. LLM.int8() Pipeline

The overall method can be summarized as:

```text
FP16 Activations X
FP16 Weights W
       │
       ▼
Detect Outlier Feature Dimensions
       │
       ├───────────────────────┐
       │                       │
       ▼                       ▼
Normal Features          Outlier Features
       │                       │
       ▼                       ▼
Vector-wise INT8          FP16 MatMul
Quantization
       │
       ▼
INT8 MatMul
       │
       ▼
INT32 Accumulation
       │
       ▼
Dequantization
       │                       │
       └───────────┬───────────┘
                   ▼
            FP16 Accumulation
                   │
                   ▼
                 Output
```

This is the core of **LLM.int8()**.

---

# 11. Experimental Setup

The paper evaluates quantization across Transformer models ranging from:

```text
125M
↓
1.3B
↓
2.7B
↓
6.7B
↓
13B
↓
...
↓
175B
```

Evaluation includes:

- C4 language modeling perplexity
- Zero-shot task accuracy
- Memory usage
- Matrix multiplication runtime
- End-to-end inference runtime

Models include:

- Fairseq Transformers
- OPT
- BLOOM
- GPT-J

---

# 12. Quantization Scaling Results

The paper compares:

- Tensor-wise absmax
- Zeropoint
- Row-wise quantization
- Vector-wise quantization
- Mixed-precision decomposition
- LLM.int8()

A representative result from C4 perplexity:

| Method | 6.7B | 13B |
|---|---:|---:|
| FP32 | 13.30 | 12.45 |
| INT8 Absmax | 14.59 | 19.08 |
| INT8 Zeropoint | 13.49 | 13.94 |
| INT8 Vector-wise | 14.13 | 16.48 |
| **LLM.int8()** | **13.24** | **12.45** |

The important observation is:

```text
Conventional INT8
→ performance degrades as model scale increases

LLM.int8()
→ preserves full-precision perplexity
```

---

# 13. Zero-Shot Performance

The paper evaluates OPT models on:

- WinoGrande
- HellaSwag
- PIQA
- LAMBADA

At smaller scales, conventional INT8 methods can remain close to FP16.

However, after systematic outliers emerge around 6.7B parameters, conventional INT8 performance rapidly degrades.

LLM.int8() continues to track the FP16 baseline up to:

```text
175B parameters
```

without measurable predictive performance degradation in the reported experiments.

---

# 14. Memory Reduction

A major goal of LLM.int8() is memory efficiency.

Converting the major Transformer matrix multiplications from 16-bit to 8-bit approximately halves the parameter memory requirement.

For BLOOM-176B, the paper reports approximately:

```text
1.96× memory reduction
```

This makes very large models usable on hardware that otherwise cannot hold the FP16 model.

For example:

| Hardware | Largest Model in 16-bit | Largest Model in 8-bit |
|---|---|---|
| 8× A100 40GB | OPT-66B | OPT-175B / BLOOM |
| 8× RTX 3090 24GB | OPT-66B | OPT-175B / BLOOM |
| 4× RTX 3090 24GB | OPT-30B | OPT-66B |
| 15GB Cloud GPU | GPT-J 6B | OPT-13B |

---

# 15. Runtime Characteristics

LLM.int8() is primarily designed for **memory reduction**, not guaranteed runtime acceleration.

This distinction is important.

INT8 matrix multiplication itself can be faster than FP16, but LLM.int8() introduces overhead from:

- Quantization
- Dequantization
- Outlier detection
- Matrix decomposition
- Separate FP16 computation

Therefore:

```text
INT8 arithmetic is faster
≠
End-to-end inference is automatically faster
```

---

# 16. Matrix Multiplication Speed

The paper shows that INT8 becomes beneficial mainly for sufficiently large matrices.

For smaller models, quantization overhead can make INT8 slower.

For the largest model sizes:

| Model Scale | LLM.int8() MatMul Speedup |
|---|---:|
| 6.7B | 0.86× |
| 13B | 1.22× |
| 175B | 1.81× |

A value below 1 means slowdown relative to the FP16 baseline.

Therefore:

```text
Small Matrices
→ Quantization overhead dominates

Large Matrices
→ INT8 compute becomes beneficial
```

---

# 17. End-to-End Runtime

For BLOOM-176B, the end-to-end latency is similar to the native 16-bit model.

For example, at batch size 1:

```text
BF16 baseline, 8× A100
239 ms / token
```

versus:

```text
LLM.int8(), 8× A100
253 ms / token
```

However, INT8 allows the model to fit onto fewer GPUs.

For example:

```text
LLM.int8(), 4× A100
246 ms / token
```

Therefore, the key benefit is:

```text
Similar Runtime
+
Much Lower Memory Requirement
+
Fewer GPUs Required
```

rather than universally faster inference.

---

# 18. Why Row-Wise / Vector-Wise Quantization Alone Fails

The outlier structure explains why ordinary quantization methods fail.

Outliers occur primarily along:

```text
Feature Dimension
       h
```

while row-wise or activation-side vector-wise scaling operates along:

```text
Sequence Dimension
       s
```

Therefore, one row often contains:

```text
Many Normal Values
       +
One Extreme Feature
```

The scale must accommodate the extreme value, reducing precision for everything else.

Mixed-precision decomposition solves this by separating the problematic **feature dimensions themselves**.

---

# 19. Outlier Distribution

The paper also observes that most outlier features have strongly asymmetric distributions.

For example, an outlier feature may be predominantly:

```text
Large Positive Values
```

or:

```text
Large Negative Values
```

rather than being symmetrically distributed around zero.

This explains why asymmetric zeropoint quantization performs better than symmetric absmax quantization before mixed-precision decomposition is applied.

Once the outliers are separated, this advantage largely disappears.

---

# 20. Main Contributions

## 1. Large-Scale INT8 Quantization

Demonstrates INT8 inference for Transformers up to 175B parameters without reported predictive performance degradation.

---

## 2. Outlier Feature Analysis

Identifies systematic large-magnitude feature dimensions that emerge in large Transformers and cause conventional INT8 quantization to fail.

---

## 3. Vector-Wise Quantization

Uses separate scaling constants for matrix multiplication vectors to improve quantization precision.

---

## 4. Mixed-Precision Decomposition

Processes:

```text
Outlier Features → FP16
Normal Features  → INT8
```

instead of forcing all features into the same precision.

---

## 5. Large Memory Reduction

Approximately halves the memory required for the major Transformer weights.

---

# 21. Limitations

The paper identifies several limitations.

### INT8 Only

The analysis focuses on INT8 and does not investigate FP8.

---

### Models Only Up to 175B

The method is validated up to 175B parameters.

Larger models may exhibit additional emergent properties.

---

### Attention Function Not Quantized

LLM.int8() quantizes:

- Feed-forward matrix multiplication
- Attention projection matrix multiplication

but does **not** quantize the attention function itself to INT8.

---

### Inference Focus

The main method focuses on inference.

INT8 training and fine-tuning require additional techniques.

---

### Runtime Overhead

Mixed-precision decomposition and quantization operations introduce additional overhead.

As a result, memory reduction is more consistent than inference speedup.

---

# 22. Important Observations

## Quantization Difficulty Changes with Model Scale

A quantization technique that works well for smaller Transformers may fail at larger scales.

```text
Small Transformer
→ Ordinary INT8 may work

Large Transformer
→ Emergent outliers appear
→ Ordinary INT8 fails
```

---

## Outliers Are Structured

The important outliers are not random individual values.

They repeatedly occur in a very small number of specific hidden dimensions.

This structure makes it possible to isolate them efficiently.

---

## Mixed Precision Is Selective

LLM.int8() does not simply keep an entire layer in FP16.

Instead:

```text
Important ~0.1%
→ High Precision

Remaining ~99.9%
→ Low Precision
```

This is what allows both accuracy preservation and memory reduction.

---

# 23. Key Takeaways

1. Conventional INT8 quantization becomes difficult as Transformers scale.

2. The main reason is the emergence of large-magnitude activation feature dimensions.

3. These outlier dimensions are sparse but critical for model performance.

4. A single large outlier can reduce quantization precision for many normal values.

5. Vector-wise quantization improves scaling granularity.

6. Vector-wise quantization alone is insufficient for very large models.

7. LLM.int8() isolates outlier feature dimensions and computes them in FP16.

8. More than 99.9% of values can still be processed using INT8.

9. The method preserves reported FP16 predictive performance up to 175B parameters.

10. The most reliable benefit is approximately halving model memory, not guaranteed latency reduction.

11. Quantization performance depends not only on bit width but also on the statistical structure of activations.

---

# 24. Connection to Previous Papers

## Quantization-Aware Training

The earlier integer-only quantization approach uses:

```text
Training
    ↓
Simulate Quantization
    ↓
Network Adapts to INT8
```

LLM.int8() instead targets already pretrained large Transformers:

```text
Pretrained LLM
    ↓
Analyze Activation Outliers
    ↓
Selective Mixed Precision
    ↓
Immediate INT8 Inference
```

No full quantization-aware retraining is required.

---

## AdaRound

AdaRound focuses primarily on:

```text
How should each weight be rounded?
```

LLM.int8() focuses on a different problem:

```text
Why do large Transformer activation distributions
make INT8 quantization fail?
```

Therefore:

```text
AdaRound
→ Optimize weight rounding

LLM.int8()
→ Handle large activation outlier dimensions
```

---

# 25. Connection to Later Papers

The outlier problem identified by LLM.int8() becomes a major theme in later LLM quantization research.

```text
LLM.int8()
    │
    └── Keep outliers in FP16
             ↓
       Mixed Precision
```

Later methods ask whether the outlier problem can instead be transformed or redistributed.

For example:

```text
SmoothQuant
→ Move quantization difficulty
  from activations toward weights

QuaRot / SpinQuant
→ Rotate representations
  to reduce outliers
```

Therefore, LLM.int8() is important not only as an INT8 method, but also because it clearly identifies **activation outliers as a fundamental obstacle in LLM quantization**.

---

# 26. Concepts to Review

- Transformer
- Feed-Forward Network
- Attention Projection
- Hidden State
- Feature Dimension
- INT8
- FP16
- INT32 Accumulation
- Symmetric Quantization
- Asymmetric Quantization
- Absmax Quantization
- Zero-Point Quantization
- Vector-Wise Quantization
- Mixed-Precision Computation
- Activation Outlier
- Perplexity
- Zero-Shot Evaluation
- Matrix Multiplication
- Quantization / Dequantization Overhead
- GPU Memory
