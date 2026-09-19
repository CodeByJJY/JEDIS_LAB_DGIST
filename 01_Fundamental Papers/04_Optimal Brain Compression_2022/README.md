# Optimal Brain Compression

> **Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning**

- **Authors:** Elias Frantar, Sidak Pal Singh, Dan Alistarh
- **Venue:** NeurIPS 2022
- **Topic:** Post-Training Model Compression
- **Keywords:** Pruning, Quantization, Optimal Brain Surgeon, Second-Order Information, Hessian, OBC, OBQ, ExactOBS

---

## 1. Problem Background

Modern neural networks have large parameter counts and high computational costs.

Two major compression techniques are:

```text
Pruning
→ Remove weights

Quantization
→ Reduce numerical precision
```

However, these two approaches are usually treated independently.

Another problem is that recovering accuracy after compression often requires:

- Fine-tuning
- Partial retraining
- Full retraining

This paper instead focuses on the **post-training compression** setting.

```text
Pretrained Model
      +
Small Calibration Dataset
      ↓
Compression
      ↓
No Retraining
```

The goal is to obtain an accurate compressed model in a single post-training stage.

---

# 2. Main Idea

The paper proposes a unified framework for:

```text
Pruning
    +
Quantization
    ↓
Optimal Brain Compression
```

The framework is built on the classical:

> **Optimal Brain Surgeon (OBS)**

idea.

The main components are:

```text
OBC
│
├── ExactOBS
│     └── Post-training pruning
│
└── OBQ
      └── Post-training quantization
```

Both methods use second-order information to estimate the effect of modifying individual weights and compensate for the resulting error by adjusting the remaining weights.

---

# 3. Layer-Wise Compression Problem

Instead of optimizing the entire neural network at once, OBC divides compression into independent layer-wise problems.

Consider a layer with:

```math
Y = WX
```

where:

- $W$: original weight matrix
- $X$: calibration input
- $Y$: original layer output

We want to find a compressed weight matrix $W_c$ such that:

```math
W_c X
```

remains close to:

```math
WX
```

The layer-wise objective is:

```math
\min_{W_c}
\left\|
WX - W_cX
\right\|_2^2
```

subject to a compression constraint.

Depending on the constraint, the same formulation can represent:

```text
Pruning
or
Quantization
```

This layer-wise reconstruction problem is the foundation of OBC.

---

# 4. Optimal Brain Surgeon

Optimal Brain Surgeon is a classical second-order pruning method.

Suppose we want to remove one weight $w_p$.

The naive approach would be:

```text
Set w_p = 0
```

but this causes reconstruction error.

OBS instead asks:

> If $w_p$ is removed, how should all remaining weights change to compensate for that removal?

Using second-order information, OBS determines both:

1. Which weight causes the smallest loss increase when removed
2. How the remaining weights should be adjusted

---

## OBS Pruning Score

The weight to remove is selected using:

```math
p
=
\arg\min_p
\frac{w_p^2}
{[H^{-1}]_{pp}}
```

where:

- $H$: Hessian matrix
- $H^{-1}$: inverse Hessian
- $[H^{-1}]_{pp}$: diagonal element corresponding to weight $p$

This is different from magnitude pruning.

Magnitude pruning considers only:

```math
|w_p|
```

while OBS also considers:

```text
Weight Magnitude
        +
Loss Curvature
```

---

# 5. OBS Weight Compensation

After selecting a weight to prune, OBS updates the remaining weights.

Conceptually:

```text
Remove Weight
      ↓
Estimate Resulting Error
      ↓
Adjust Remaining Weights
      ↓
Compensate for Error
```

The update uses the inverse Hessian.

Therefore, OBS does not simply remove a weight.

It tries to redistribute the effect of that removed weight across the remaining parameters.

---

# 6. Why Classical OBS Is Expensive

Classical OBS is difficult to apply directly to modern neural networks.

If a layer contains a very large number of parameters:

```text
Hessian
→ Extremely Large Matrix

Hessian Inverse
→ Expensive

One-Weight-at-a-Time Updates
→ Very Expensive
```

A naive implementation would have approximately:

```text
O(d^4)
```

total complexity for a layer with $d$ parameters.

This is infeasible for modern DNNs.

The main technical contribution of the paper is making exact OBS practical at modern neural network scale.

---

# 7. ExactOBS

The pruning algorithm proposed by the paper is called:

> **ExactOBS**

The first important observation is that the layer-wise reconstruction error can be decomposed by rows of the weight matrix.

For:

```math
W \in \mathbb{R}^{d_{row} \times d_{col}}
```

the reconstruction objective becomes:

```math
\sum_i
\left\|
W_{i,:}X -
W_{c,i,:}X
\right\|_2^2
```

Therefore, different output rows can be processed independently.

For each row, the Hessian becomes:

```math
H = 2XX^T
```

The same Hessian structure can be reused across weight rows.

This dramatically reduces memory and computation requirements.

---

# 8. Efficient Hessian Inverse Update

A major challenge is that after pruning a weight, the Hessian inverse corresponding to the remaining weights changes.

Recomputing the inverse from scratch after every pruning step would be extremely expensive.

ExactOBS instead updates the inverse directly after removing a row and column.

Conceptually:

```text
Initial Hessian Inverse
        ↓
Prune One Weight
        ↓
Rank / Elimination Update
        ↓
Updated Hessian Inverse
        ↓
Prune Next Weight
```

This allows ExactOBS to perform true one-weight-at-a-time OBS pruning efficiently.

The paper reduces the total layer complexity to approximately:

```text
O(d_row × d_col^3)
```

with memory proportional to:

```text
O(d_col^2)
```

rather than operating on the full flattened parameter Hessian.

---

# 9. ExactOBS Procedure

The pruning process can be summarized as:

```text
Calibration Inputs X
        │
        ▼
Compute Hessian
H = 2XX^T
        │
        ▼
Compute H^-1
        │
        ▼
Evaluate OBS Scores
        │
        ▼
Select Lowest-Loss Weight
        │
        ▼
Prune Weight
        │
        ▼
Update Remaining Weights
        │
        ▼
Update H^-1
        │
        ▼
Repeat
```

The key distinction from many approximate pruning methods is:

> ExactOBS performs compensation after every individual pruning decision.

---

# 10. Global Pruning Across Rows

Rows can be processed independently because there are no Hessian interactions across output rows.

For each row, ExactOBS records:

- Order in which weights would be pruned
- Loss increase caused by each pruning step

Afterwards, these pruning traces are combined to determine the global pruning mask.

Conceptually:

```text
Row 1 → pruning trace
Row 2 → pruning trace
Row 3 → pruning trace
...
        ↓
Compare Loss Increases
        ↓
Global Pruning Mask
```

This avoids storing every row-wise inverse Hessian simultaneously on the GPU.

---

# 11. Structured Sparsity Support

ExactOBS can also support hardware-friendly sparsity patterns.

Examples include:

- N:M sparsity
- Block sparsity
- Unstructured sparsity

---

## N:M Sparsity

In an N:M sparsity pattern:

```text
Every M weights
      ↓
Keep N non-zero weights
```

For example:

```text
2:4 Sparsity

4 weights
↓
2 non-zero
2 zero
```

This pattern is relevant because modern NVIDIA GPUs support 2:4 structured sparsity.

---

## Block Sparsity

Instead of removing individual weights:

```text
Single Weights
```

block sparsity removes groups of consecutive weights:

```text
Block of Weights
```

This can be useful for CPU inference engines that support block-sparse computation.

---

# 12. From Pruning to Quantization

A major contribution of the paper is extending the OBS framework from pruning to quantization.

Pruning can be interpreted as:

```text
Current Weight
     ↓
Target Value = 0
```

Quantization can instead be interpreted as:

```text
Current Weight
     ↓
Target Value = Quantized Grid Point
```

This observation allows the OBS formulation to be generalized.

---

# 13. Optimal Brain Quantizer

The resulting quantization algorithm is called:

> **Optimal Brain Quantizer (OBQ)**

Instead of selecting which weight should be pruned next, OBQ selects which weight should be quantized next.

The selection score becomes:

```math
p
=
\arg\min_p
\frac{
\left(
q(w_p)-w_p
\right)^2
}
{
[H^{-1}]_{pp}
}
```

where:

- $w_p$: current weight
- $q(w_p)$: quantized value
- $H^{-1}$: inverse Hessian

The numerator represents the quantization error.

The denominator accounts for the sensitivity of the reconstruction loss.

---

# 14. OBQ Weight Update

Once a weight is quantized, the remaining unquantized weights are adjusted to compensate for the quantization error.

Conceptually:

```text
Choose Weight
      ↓
Quantize Weight
      ↓
Quantization Error
      ↓
Use H^-1
      ↓
Adjust Remaining Weights
      ↓
Choose Next Weight
```

This process continues until every weight in the layer has been quantized.

---

# 15. Why Quantization Order Matters

At first, quantizing weights one at a time may seem unnecessary because eventually all weights must be quantized.

However, the important point is that:

```text
Quantize Weight 1
        ↓
Adjust Remaining Weights
        ↓
Their Values Change
        ↓
Quantize Weight 2
        ↓
Adjust Again
```

Therefore, later weights are quantized after previous quantization errors have already been compensated.

As a result, the final quantization assignments may differ from simply rounding all original weights independently.

---

# 16. Pruning and Quantization as the Same Framework

OBC provides a unified interpretation.

```text
Pruning

q(w) = 0
```

and:

```text
Quantization

q(w) = nearest quantization value
```

can both be handled using the same OBS-style optimization.

Therefore:

```text
           OBC
            │
      ┌─────┴─────┐
      ▼           ▼
  ExactOBS       OBQ
      │           │
   Pruning    Quantization
```

---

# 17. Calibration Data

OBC is designed for the post-training setting.

The experiments primarily use:

```text
1024 calibration samples
```

without retraining the original network.

The calibration inputs are used to estimate:

```math
XX^T
```

which determines the layer-wise Hessian:

```math
H = 2XX^T
```

This means the compression procedure depends on actual activation statistics rather than only weight values.

---

# 18. Experimental Tasks

The paper evaluates OBC on several tasks and model families.

### Image Classification

- ResNet
- ImageNet

### Object Detection

- YOLOv5
- COCO

### Language Modeling / Question Answering

- BERT variants
- SQuAD

The experiments study:

- Unstructured pruning
- N:M sparsity
- Block sparsity
- Weight quantization
- Joint pruning + quantization

---

# 19. Unstructured Pruning Results

The paper compares:

- Global Magnitude Pruning
- L-OBS
- AdaPrune
- ExactOBS

Example results:

### ResNet-50

| FLOP Reduction | GMP | AdaPrune | ExactOBS |
|---|---:|---:|---:|
| 2× | 74.86 | 75.53 | **75.64** |
| 3× | 71.44 | 74.47 | **75.01** |
| 4× | 64.84 | 72.39 | **74.05** |

Dense accuracy:

```text
76.13%
```

The advantage becomes larger at more aggressive pruning levels.

---

# 20. BERT Pruning Results

BERT is substantially more sensitive to aggressive pruning.

Dense F1:

```text
88.53
```

At a 4× FLOP reduction:

| Method | F1 |
|---|---:|
| GMP | 9.23 |
| L-OBS | 6.63 |
| AdaPrune | 18.75 |
| **ExactOBS** | **82.10** |

This demonstrates the importance of accurate second-order compensation for difficult compression settings.

---

# 21. N:M Sparsity Results

For ResNet models, ExactOBS performs well with hardware-friendly structured sparsity.

Example:

### ResNet-50

```text
Dense
76.13
```

```text
AdaPrune 4:8
74.75
```

```text
ExactOBS 2:4
74.71
```

```text
ExactOBS 4:8
75.20
```

The paper emphasizes that ExactOBS can achieve competitive results even under the stricter 2:4 pattern.

---

# 22. Quantization Results

OBQ is compared with:

- AdaRound
- AdaQuant
- BRECQ

using asymmetric per-channel weight quantization.

### ResNet-18

| Method | 4-bit | 3-bit | 2-bit |
|---|---:|---:|---:|
| AdaRound | 69.34 | 68.37 | 63.37 |
| AdaQuant | 68.12 | 59.21 | 0.10 |
| BRECQ | 69.37 | 68.47 | **64.70** |
| **OBQ** | **69.56** | **68.69** | 64.04 |

Dense accuracy:

```text
69.76%
```

---

## ResNet-50

| Method | 4-bit | 3-bit | 2-bit |
|---|---:|---:|---:|
| AdaRound | **75.84** | 75.14 | 71.58 |
| AdaQuant | 74.68 | 64.98 | 0.10 |
| BRECQ | 75.88 | **75.32** | **72.41** |
| OBQ | 75.72 | 75.24 | 70.71 |

Dense accuracy:

```text
76.13%
```

An important property of OBQ is that layers can be compressed independently.

---

# 23. Independent Layer Compression

Methods such as AdaRound, AdaQuant, and BRECQ often benefit from sequential compression.

Conceptually:

```text
Compress Layer 1
      ↓
Use Quantized Output
      ↓
Compress Layer 2
      ↓
...
```

OBC instead focuses primarily on **independent layer compression**.

```text
Layer 1 ── Compress independently
Layer 2 ── Compress independently
Layer 3 ── Compress independently
...
```

These independently compressed layers can later be:

```text
Stitched Together
```

This is useful when searching over:

- Different bit widths
- Different sparsity levels
- Different hardware constraints

because every possible model does not have to be recompressed from scratch.

---

# 24. Joint Pruning and Quantization

A major advantage of the unified framework is that pruning and quantization can be combined.

For GPU inference, the paper considers:

```text
Quantization
      +
2:4 Sparsity
```

Possible layer configurations include:

```text
8w8a
4w4a
8w8a + 2:4
4w4a + 2:4
```

This allows the system to search for different compression configurations under a compute budget.

---

# 25. GPU Compound Compression Results

For ResNet models, the paper reports approximately:

```text
12–14× BOP Reduction
```

with roughly:

```text
2.5% relative performance degradation
```

For YOLO and BERT:

```text
7–8× BOP Reduction
```

is achieved at similar relative performance degradation.

The paper also reports approximately:

```text
12× theoretical operation reduction
```

with around:

```text
2% accuracy loss
```

for GPU-supported compound compression.

---

# 26. CPU Runtime Results

The paper also evaluates real inference speed rather than only theoretical FLOPs.

For ResNet-50 on a 12-core Intel Xeon CPU using DeepSparse:

```text
Dense INT8
      +
Block Sparsity
```

achieves:

```text
4× actual speedup
→ ~1% accuracy loss
```

and:

```text
5× actual speedup
→ ~2% accuracy loss
```

This is an important distinction:

```text
Theoretical Compression
≠
Actual Hardware Speedup
```

OBC evaluates both.

---

# 27. Runtime of OBC

The algorithms are more computationally expensive than simple PTQ methods, but remain practical.

For ResNet-50 4-bit quantization on one RTX 3090:

| Method | Runtime |
|---|---:|
| BitSplit | 124 min |
| AdaRound | 55 min |
| AdaQuant | 17 min |
| BRECQ | 53 min |
| OBQ | 65 min |

Therefore, OBQ has a runtime comparable to existing high-accuracy PTQ methods.

---

# 28. Important Observations

## Second-Order Information Matters

OBC does not determine importance based only on:

```text
Weight Magnitude
```

It also considers:

```text
Loss Curvature
```

through the inverse Hessian.

This allows it to estimate the actual reconstruction impact of pruning or quantizing a weight.

---

## Error Compensation Is Critical

The method does not simply perform:

```text
Prune
or
Quantize
```

Instead:

```text
Modify One Weight
       ↓
Compensate Using Remaining Weights
       ↓
Modify Next Weight
```

This iterative correction is central to OBC.

---

## Pruning and Quantization Are Closely Related

In the OBC formulation:

```text
Pruning
=
Quantization to Zero
```

This makes it possible to use nearly the same mathematical framework for both compression methods.

---

## Calibration Activations Matter

Since:

```math
H = 2XX^T
```

the compression decision depends on the calibration input distribution.

Therefore, OBC is:

```text
Weight Aware
+
Activation Aware
```

rather than purely based on the weight tensor.

---

# 29. Relation to AdaRound

AdaRound asks:

> Should each weight be rounded up or down?

and uses gradient-based continuous relaxation to minimize layer reconstruction error.

OBC takes a different approach:

```text
AdaRound
│
├── Optimize rounding choices
├── Continuous relaxation
└── Gradient-based optimization

OBC / OBQ
│
├── Use second-order information
├── Quantize one weight at a time
└── Compensate through remaining weights
```

Both methods optimize layer reconstruction, but their optimization strategies are fundamentally different.

---

# 30. Relation to Optimal Brain Surgeon

Classical OBS:

```text
Select Weight
      ↓
Set Weight to Zero
      ↓
Update Remaining Weights
```

OBC makes this practical for modern DNNs.

OBQ then generalizes the same idea:

```text
Select Weight
      ↓
Move Weight to Quantization Grid
      ↓
Update Remaining Weights
```

Therefore:

```text
OBS
 ↓
ExactOBS
 ↓
OBQ
 ↓
OBC
```

---

# 31. Connection to GPTQ

This paper is especially important because OBQ directly leads to GPTQ.

OBQ performs accurate second-order quantization but still has relatively high computational cost for very large models.

The next question becomes:

> Can the OBQ idea be simplified and scaled to Transformers with billions or hundreds of billions of parameters?

This motivates:

```text
Optimal Brain Quantization
          ↓
        GPTQ
```

GPTQ retains the central idea:

```text
Quantize Weights
      +
Compensate Remaining Weights
using Second-Order Information
```

but introduces algorithmic changes that make this practical for extremely large language models.

---

# 32. Limitations

OBC still has several practical limitations.

### Compression Cost

Exact second-order compression is more computationally expensive than simpler PTQ approaches.

---

### Cubic Dependence on Layer Width

The complexity depends strongly on the column dimension:

```text
O(d_row × d_col^3)
```

Therefore, a few very wide layers can dominate compression time.

---

### Calibration Data Required

The Hessian estimate depends on calibration inputs.

Poor calibration data may reduce compression quality.

---

### No Training Adaptation

OBC operates entirely in the post-training setting.

Unlike QAT, the original network is not retrained to adapt globally to compression.

---

# 33. Key Contributions

### 1. Unified Compression Framework

Pruning and quantization are handled under the same mathematical formulation.

### 2. ExactOBS

Makes one-weight-at-a-time OBS pruning practical for modern neural networks.

### 3. Optimal Brain Quantizer

Extends OBS from pruning to weight quantization.

### 4. Error Compensation

Remaining weights are updated after every pruning or quantization decision.

### 5. Compound Compression

Pruning and quantization can be combined for additional hardware acceleration.

### 6. Post-Training Setting

All compression is performed using only a small calibration dataset without retraining.

---

# 34. Key Takeaways

1. Post-training pruning and quantization can be formulated as the same layer-wise reconstruction problem.

2. Optimal Brain Surgeon uses second-order information to estimate which weight can be modified with the smallest loss increase.

3. Removing or quantizing a weight should be followed by an optimal update of the remaining weights.

4. ExactOBS makes exact greedy OBS pruning computationally practical for modern DNNs.

5. OBQ extends the OBS principle from pruning to quantization.

6. In OBC, pruning can essentially be interpreted as quantization whose target value is zero.

7. Calibration activations determine the layer Hessian through $XX^T$.

8. OBC achieves strong post-training pruning and quantization performance without model retraining.

9. Pruning and quantization can be combined to obtain larger practical inference speedups.

10. OBQ provides the direct conceptual foundation for GPTQ.

---

# 35. Concepts to Review

- Post-Training Compression
- Layer-Wise Reconstruction
- Pruning
- Unstructured Sparsity
- N:M Sparsity
- Block Sparsity
- Quantization
- Hessian
- Inverse Hessian
- Second-Order Taylor Approximation
- Quadratic Form
- Least Squares
- Optimal Brain Surgeon
- Error Compensation
- Calibration Data
- FLOPs
- BOPs
- Hardware-Aware Compression
