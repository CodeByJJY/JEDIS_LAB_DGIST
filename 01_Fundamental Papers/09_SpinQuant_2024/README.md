# SpinQuant

> **SpinQuant: LLM Quantization with Learned Rotations**

- **Authors:** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort
- **Venue:** ICLR 2025
- **Initial arXiv:** 2024
- **Topic:** Learned Rotation-Based LLM Quantization
- **Keywords:** SpinQuant, Learned Rotation, Quantization, Outlier Suppression, Stiefel Manifold, Cayley SGD, W4A4KV4

---

## 1. Problem Background

Post-Training Quantization (PTQ) reduces the precision of:

- Weights
- Activations
- KV Cache

and can reduce:

- Model memory
- Memory bandwidth
- Inference latency
- Power consumption

However, aggressive low-bit quantization becomes difficult because LLM weights and activations contain:

> **Outliers**

These outliers stretch the quantization range.

Conceptually:

```text
Most Values
-0.2  0.1  0.3  -0.1

Outlier
8.0
```

The quantizer must cover:

```text
-8.0 ~ 8.0
```

even though most values lie around:

```text
-0.3 ~ 0.3
```

Therefore:

```text
Outlier
   ↓
Large Quantization Range
   ↓
Large Quantization Step
   ↓
Poor Resolution for Normal Values
   ↓
Large Quantization Error
```

---

# 2. Motivation from QuaRot

Previous rotation-based methods such as QuaRot showed that orthogonal rotations can redistribute outliers across dimensions.

Conceptually:

```text
Before Rotation

small
small
HUGE
small
small
```

becomes:

```text
After Rotation

medium
medium
medium
medium
medium
```

This makes both:

- Activations
- Weights

easier to quantize.

QuaRot mainly uses:

```text
Randomized Hadamard Rotations
```

SpinQuant starts from the same fundamental idea.

---

# 3. Main Question

SpinQuant asks:

> **Are all valid rotations equally good for quantization?**

The answer is:

```text
No
```

Even though different orthogonal rotations produce an equivalent full-precision network, they can produce very different quantized models.

The paper finds up to approximately:

```text
13 percentage points
```

of zero-shot accuracy difference between different random rotations.

Therefore:

```text
Random Rotation
→ Sometimes Good
→ Sometimes Bad
```

This motivates:

> **Learn the rotation specifically for quantization.**

---

# 4. Main Idea

SpinQuant replaces:

```text
Random Rotation
```

with:

```text
Learned Rotation
```

The overall idea is:

```text
Pretrained LLM
      ↓
Insert Valid Orthogonal Rotations
      ↓
Keep Original Weights Frozen
      ↓
Simulate Quantization
      ↓
Optimize Rotation Matrices
for Quantized Network Loss
      ↓
Merge Rotations into Weights
      ↓
Quantize Model
```

The full-precision model remains functionally unchanged.

Only the behavior after quantization changes.

---

# 5. Why Rotation Can Preserve the Original Network

Suppose a representation is transformed by an orthogonal matrix:

```math
R
```

where:

```math
R^T R = I
```

Then:

```math
R^{-1} = R^T
```

If one part of the model applies:

```math
R
```

and the corresponding inverse is inserted later:

```math
R^{-1}
```

then:

```math
R R^{-1} = I
```

and the original full-precision computation is preserved.

Therefore:

```text
Different Internal Representation
            +
Same Final FP Output
```

is possible.

---

# 6. Important Insight

The key difference between full precision and quantization is:

```text
Full Precision

R
+
R^-1
→ Exactly Cancel
```

but:

```text
Quantized Network

Q(RX)
and
Q(WR^-1)
```

introduce quantization errors.

Different choices of $R$ produce different quantization errors.

Therefore:

```text
Full Precision Performance
→ Independent of Rotation

Quantized Performance
→ Strongly Dependent on Rotation
```

This is the fundamental reason why learning the rotation makes sense.

---

# 7. Outlier Reduction

SpinQuant analyzes activation distributions using:

> **Kurtosis**

Kurtosis measures how heavy the tails of a distribution are.

Conceptually:

```text
Large Kurtosis
→ Strong Outliers

Kurtosis ≈ 3
→ Gaussian-like Distribution
```

Before rotation, some LLaMA-2 activation distributions have kurtosis:

```text
> 200
```

After rotation:

```text
≈ 3
```

across many layers.

This indicates that rotation transforms highly outlier-heavy distributions into much more regular distributions.

---

# 8. Quantization Error After Rotation

The paper measures quantization error before and after rotation for:

- Activations
- Weights

Rotation reduces the reconstruction error of both.

Conceptually:

```text
Before Rotation

Outlier-Dominated Distribution
        ↓
Large Quantization Error
```

```text
After Rotation

More Uniform Distribution
        ↓
Lower Quantization Error
```

Therefore, rotation improves quantization on both sides of matrix multiplication.

---

# 9. Random Rotation Variance

The authors perform:

```text
100 random rotation trials
```

on LLaMA-2 7B under:

```text
W4A4
```

quantization.

They observe a very large performance spread.

For random orthogonal rotations:

```text
Best - Worst
≈ 13 accuracy points
```

Even random Hadamard rotations show approximately:

```text
6 points
```

of variation.

This means:

> Removing outliers alone is not sufficient to determine the best quantized representation.

---

# 10. Why Hadamard Often Works Better Than Random Rotation

Random Hadamard matrices generally perform better than arbitrary random orthogonal matrices.

A Hadamard transformation efficiently mixes dimensions.

Conceptually:

```text
Large Coordinate
      ↓
Mixed Uniformly
Across Many Coordinates
```

This helps reduce maximum values and outlier concentration.

However:

```text
Hadamard
≠
Optimal Rotation
```

because it does not use the actual quantization loss of the model.

---

# 11. SpinQuant Rotation Parameterization

SpinQuant introduces four rotations:

```text
R1
R2
R3
R4
```

They serve different parts of the Transformer.

Broadly:

```text
R1
→ Residual Stream Rotation

R2
→ Attention Value / Output Rotation

R3
→ KV-Cache Related Online Rotation

R4
→ FFN Intermediate Activation Rotation
```

---

# 12. R1: Residual Stream Rotation

The first rotation:

```math
R_1
```

rotates the Transformer residual stream.

Conceptually:

```text
Original Residual

X
```

becomes:

```text
Rotated Residual

X R1
```

This makes the activations feeding major linear layers easier to quantize.

The corresponding inverse rotations are absorbed into:

- Query projection
- Key projection
- Value projection
- Up projection
- Gate projection

weights.

---

# 13. R1 Can Be Merged into Weights

Because the inverse rotation can be absorbed into the weight matrices:

```text
Runtime

No Explicit R1 Kernel Needed
```

After learning:

```text
Original Weight
      ↓
Multiply Learned Rotation
      ↓
Store Rotated Weight
```

Therefore, the inference graph does not need an explicit residual-stream rotation operation.

---

# 14. R2: Attention Rotation

SpinQuant also introduces:

```math
R_2
```

inside the attention block.

It rotates the value representation and compensates for it before or within the output projection.

Conceptually:

```text
Value
  ↓
R2
  ↓
Attention
  ↓
R2^-1
  ↓
Output Projection
```

The two rotations cancel in the full-precision network.

---

# 15. Why R2 Helps

R2 improves the quantization of:

- Value cache
- Activation entering the output projection

This is particularly useful because outliers can also exist inside the attention block, not only in the residual stream.

---

# 16. SpinQuant_no_had

The first SpinQuant configuration is:

> **SpinQuant_no_had**

It uses only:

```text
R1
+
R2
```

Both can be absorbed into model weights after learning.

Therefore:

```text
Inference-Time Additional Rotation
= None
```

The forward graph does not need to be structurally modified.

---

# 17. When SpinQuant_no_had Is Useful

SpinQuant_no_had is particularly suitable for settings such as:

```text
W4A8
W4A8KV8
```

where activation precision is not extremely low.

Advantages:

- No online Hadamard kernel
- No forward-pass modification
- Easy deployment
- Large accuracy improvement over naive quantization

---

# 18. Remaining Problem at A4

When activation precision is reduced to:

```text
4 bits
```

R1 and R2 alone are not sufficient.

Outliers can still emerge:

- Inside the FFN
- Inside the attention mechanism
- In the KV cache

Therefore, additional online rotations are introduced.

---

# 19. R3: KV-Cache Rotation

SpinQuant uses:

```math
R_3
```

for stronger outlier suppression in attention and KV-cache quantization.

R3 remains an online Hadamard transform.

Why?

Because it cannot be completely absorbed into the surrounding weight matrices while preserving the desired attention computation.

---

# 20. R4: FFN Intermediate Rotation

Inside the feed-forward network:

```text
Up / Gate Projection
       ↓
Nonlinearity
       ↓
Intermediate Activation
       ↓
Down Projection
```

the intermediate activation can develop new outliers.

SpinQuant uses:

```math
R_4
```

before the down projection.

R4 is implemented using an online Hadamard transformation.

---

# 21. SpinQuant_had

The second configuration is:

> **SpinQuant_had**

It uses:

```text
Learned:
R1
R2

Online Hadamard:
R3
R4
```

Therefore:

```text
SpinQuant_had
=
Learned Rotation
+
Online Hadamard Rotation
```

This configuration is designed for extreme quantization such as:

```text
W4A4KV4
```

---

# 22. SpinQuant Variants

The two versions can be summarized as:

| Method | R1 | R2 | R3 | R4 | Runtime Rotation |
|---|---|---|---|---|---|
| SpinQuant_no_had | Learned | Learned | - | - | None |
| SpinQuant_had | Learned | Learned | Hadamard | Hadamard | Yes |

Therefore:

```text
SpinQuant_no_had
→ Simpler / Faster

SpinQuant_had
→ Better for Extreme Low-Bit Activation
```

---

# 23. Learning the Rotation

The core optimization problem is:

```math
\min_{R_1,R_2 \in \mathcal{M}}
L_Q(R_1,R_2 \mid W,X)
```

where:

- $W$: frozen pretrained model weights
- $X$: calibration inputs
- $L_Q$: loss of the quantized network
- $\mathcal{M}$: set of orthonormal matrices

The important point is:

```text
Weights
→ Frozen

Rotations
→ Trainable
```

---

# 24. Stiefel Manifold

Rotation matrices must remain orthonormal.

The set of orthonormal matrices is known as the:

> **Stiefel Manifold**

Conceptually:

```text
All Matrices
   ↓
Only Matrices Satisfying

R^T R = I
```

SpinQuant must optimize within this constrained space.

Ordinary SGD does not automatically preserve this property.

---

# 25. Why Ordinary SGD Is Not Enough

Suppose:

```math
R^T R = I
```

initially.

A normal SGD update:

```math
R'
=
R - \eta G
```

does not guarantee:

```math
R'^T R' = I
```

Therefore, after ordinary gradient updates:

```text
R
→ No Longer Orthogonal
```

and the computational invariance property can break.

SpinQuant therefore uses a specialized optimizer.

---

# 26. Cayley SGD

SpinQuant uses:

> **Cayley SGD**

to optimize rotation matrices while preserving orthogonality.

The update has the form:

```math
R'
=
\left(
I - \frac{\alpha}{2}Y
\right)^{-1}
\left(
I + \frac{\alpha}{2}Y
\right)
R
```

where $Y$ is skew-symmetric:

```math
Y^T = -Y
```

The transformation:

```math
\left(
I - \frac{\alpha}{2}Y
\right)^{-1}
\left(
I + \frac{\alpha}{2}Y
\right)
```

is the:

> **Cayley Transform**

---

# 27. Why Cayley Transform Is Useful

If $R$ is orthogonal before the update:

```math
R^T R = I
```

Cayley SGD guarantees that the updated matrix remains orthogonal:

```math
R'^T R' = I
```

Therefore:

```text
Gradient-Based Optimization
         +
Orthogonality Constraint
```

can be satisfied simultaneously.

---

# 28. Trainable Parameter Overhead

Only the rotation matrices are optimized.

The paper reports that:

```text
R1 + R2
≈ 0.26%
```

of the original model parameter count.

Therefore:

```text
Original LLM Weights
→ Frozen

Tiny Fraction of Rotation Parameters
→ Optimized
```

This makes rotation optimization much cheaper than full fine-tuning.

---

# 29. Calibration Setup

The main rotation optimization uses:

```text
800 WikiText-2 Samples
```

for:

```text
100 Iterations
```

The learning rate:

```text
starts at 1.5
```

and linearly decays to:

```text
0
```

R1 and R2 are initialized using random Hadamard matrices.

---

# 30. Optimization Time

Reported SpinQuant optimization times include:

| Model | SpinQuant Time |
|---|---:|
| LLaMA-3 1B | ~13 min |
| LLaMA-3 3B | ~18 min |
| LLaMA-3 8B | ~30 min |
| LLaMA-2 7B | ~25 min |
| LLaMA-2 13B | ~30 min |
| LLaMA-2 70B | ~3.5 h |
| Mistral-7B | ~16 min |

The optimization cost is comparable in scale to GPTQ preprocessing.

---

# 31. Fewer Calibration Samples Also Work

Although the default experiment uses:

```text
800 samples
```

the paper finds that:

```text
128 samples
```

produce very similar WikiText-2 perplexity.

For LLaMA-2 7B W4A4KV4:

```text
128 samples
PPL ≈ 6.2

800 samples
PPL ≈ 6.2
```

Therefore, SpinQuant is relatively insensitive to the calibration set size in this experiment.

---

# 32. Why SpinQuant Optimizes Activation Quantization

The final method combines SpinQuant with GPTQ.

GPTQ already handles:

```text
Weight Quantization Error
```

using second-order compensation.

Therefore, SpinQuant primarily learns rotations under a network where:

```text
Activations Are Quantized
Weights Remain FP16
```

during the rotation-learning stage.

Conceptually:

```text
SpinQuant
→ Focus on Activation Quantization

GPTQ
→ Focus on Weight Quantization
```

This division of labor produces better performance.

---

# 33. SpinQuant + GPTQ

After rotation learning:

```text
Learn R1 / R2
      ↓
Merge Rotations into Weights
      ↓
Apply GPTQ
      ↓
Quantize Weights
```

Therefore:

```text
Learned Rotation
        +
Second-Order Weight Quantization
```

are complementary.

---

# 34. Experimental Models

The paper evaluates:

### LLaMA-2

- 7B
- 13B
- 70B

### LLaMA-3 / LLaMA-3.2

- 1B
- 3B
- 8B

### Mistral

- 7B

The evaluation covers several bit-width settings.

---

# 35. Evaluated Precision Settings

The main configurations are:

```text
W4A8KV16
```

```text
W4A8KV8
```

```text
W4A4KV16
```

```text
W4A4KV4
```

The last configuration is the most aggressive:

```text
Weight     → 4 bit
Activation → 4 bit
KV Cache   → 4 bit
```

---

# 36. Zero-Shot Evaluation

The paper evaluates eight commonsense reasoning tasks:

- ARC-Easy
- ARC-Challenge
- BoolQ
- PIQA
- SIQA
- HellaSwag
- OpenBookQA
- WinoGrande

It reports:

```text
Average Zero-Shot Accuracy
```

as the main accuracy metric.

WikiText-2 perplexity is also evaluated.

---

# 37. LLaMA-2 7B W4A4KV4

For LLaMA-2 7B:

```text
Full Precision
66.9
```

average zero-shot accuracy.

Under W4A4KV4:

```text
RTN
37.1
```

```text
SmoothQuant
39.0
```

```text
LLM-QAT
44.9
```

```text
GPTQ
36.8
```

```text
SpinQuant_no_had
56.0
```

```text
SpinQuant_had
64.0
```

Therefore:

```text
FP16 Gap
=
66.9 - 64.0
=
2.9 points
```

---

# 38. Extreme Quantization Result

The W4A4KV4 LLaMA-2 7B result is one of the paper's main headline results.

Compared with:

```text
LLM-QAT
44.9
```

SpinQuant achieves:

```text
64.0
```

an improvement of:

```text
+19.1 points
```

while keeping:

```text
Weights
Activations
KV Cache
```

all at 4 bits.

---

# 39. LLaMA-2 13B W4A4KV4

For LLaMA-2 13B:

```text
Full Precision
68.3
```

SpinQuant_had:

```text
66.9
```

Gap:

```text
1.4 points
```

This shows that the learned-rotation approach continues to work well at larger scale.

---

# 40. LLaMA-2 70B W4A4KV4

For LLaMA-2 70B:

```text
Full Precision
72.9
```

SpinQuant_had:

```text
71.2
```

Gap:

```text
1.7 points
```

The corresponding WikiText-2 perplexity is:

```text
FP16
3.3

SpinQuant
3.8
```

---

# 41. Why SpinQuant_no_had Is Enough for W4A8

When activations use 8 bits:

```text
A8
```

the quantization challenge is less severe.

Therefore, learned R1 and R2 already remove enough problematic outliers.

For example, on LLaMA-3 8B W4A8KV16:

```text
Full Precision
69.6

SpinQuant_no_had
68.6
```

The gap is only:

```text
1.0 point
```

without requiring online Hadamard transforms.

---

# 42. Why SpinQuant_had Is Needed for A4

At:

```text
A4
```

the activation quantization range is much more constrained.

Internal activations that are not covered by R1/R2 can still contain severe outliers.

Therefore:

```text
R3 + R4
```

become important.

For LLaMA-2 7B W4A4KV4:

```text
SpinQuant_no_had
56.0
```

versus:

```text
SpinQuant_had
64.0
```

an improvement of:

```text
+8.0 points
```

---

# 43. Learned Rotation vs. Random Hadamard

The paper directly compares:

```text
Random Hadamard
```

with:

```text
Learned SpinQuant Rotation
```

For Mistral-7B W4A4KV4:

```text
Random Hadamard
52.4
```

SpinQuant_had:

```text
68.6
```

Improvement:

```text
+16.2 points
```

This is strong evidence that:

> **Rotation selection matters, not merely the existence of rotation.**

---

# 44. SpinQuant vs. QuaRot

QuaRot uses:

```text
Random Hadamard Rotations
```

SpinQuant instead uses:

```text
Learned Rotations
```

for the mergeable rotations.

Conceptually:

```text
QuaRot
→ Randomly choose a good representation

SpinQuant
→ Optimize the representation
   for quantization loss
```

---

# 45. Why SpinQuant Improves QuaRot

QuaRot already removes many outliers.

However:

```text
Outlier Removal
≠
Minimum Quantization Loss
```

Two rotations may produce similarly smooth-looking distributions while still producing different errors after quantization.

SpinQuant directly optimizes:

```text
Final Quantized Network Loss
```

rather than relying only on statistical outlier suppression.

---

# 46. LLaMA-3 Comparison with QuaRot

The difference becomes especially large on difficult-to-quantize LLaMA-3 models.

For LLaMA-3 8B W4A4KV4 with GPTQ:

```text
QuaRot
63.3
```

SpinQuant_had:

```text
65.5
```

For the much more sensitive LLaMA-3 70B configuration reported in the paper:

```text
QuaRot + GPTQ
65.1
```

versus:

```text
SpinQuant + GPTQ
69.3
```

and the perplexity gap is also substantially reduced.

---

# 47. Rotation Initialization

Before optimization:

```text
Random Hadamard
```

usually performs better than arbitrary random orthogonal matrices.

However, after Cayley optimization, the initialization type matters much less.

For LLaMA-2 7B W4A4KV4:

```text
Optimized FP Rotation Init
Avg ≈ 61.5

Optimized Hadamard Init
Avg ≈ 61.5
```

This suggests that loss-aware optimization can compensate for the initial rotation choice.

---

# 48. Why Learned Rotation Works

The paper provides an intuitive geometric explanation.

Imagine a 2D activation distribution where:

```text
x1 Range
>>
x2 Range
```

A shared quantizer wastes much of the representational range of the second dimension.

Conceptually:

```text
Before Rotation

x1: █████████████████
x2: ███
```

A rotation can redistribute the magnitudes:

```text
After Rotation

x1': ██████████
x2': ██████████
```

This allows both dimensions to use the quantization range more effectively.

---

# 49. Why 45-Degree Rotation Is Not Always Optimal

In a simple 2D symmetric case:

```text
45° Rotation
```

may evenly distribute the magnitude.

However, real Transformer distributions are:

- High dimensional
- Layer dependent
- Unequal across dimensions
- Affected differently by quantization

Therefore:

```text
Fixed Hadamard Rotation
```

cannot always produce the best possible distribution.

A learned rotation can adapt to the actual activation statistics.

---

# 50. End-to-End SNR Analysis

The paper also measures:

> **Signal-to-Quantization-Noise Ratio (SNR)**

for LLaMA-2 7B with W4A4.

Without rotation:

```text
R = I
SNR = -2.9 dB
```

Random rotation:

```text
SNR = 0.9 dB
```

Learned rotation:

```text
SNR = 6.8 dB
```

Therefore:

```text
No Rotation
      ↓
Random Rotation
      ↓
Learned Rotation

-2.9 dB
→ 0.9 dB
→ 6.8 dB
```

The learned representation substantially reduces quantization noise.

---

# 51. Interesting Layer-Wise Observation

Not every layer improves equally after rotation learning.

The paper finds:

```text
A Few Layers
→ Large SNR Improvement

Most Layers
→ Small Change

One Layer
→ Can Even Get Worse
```

This suggests that the optimizer may sacrifice less important layers to significantly improve more sensitive layers.

Therefore, SpinQuant does not simply minimize quantization error uniformly across every layer.

It optimizes:

> **End-to-end quantized network performance.**

---

# 52. 3-Bit Weight Quantization

SpinQuant also evaluates:

```text
W3A8
```

quantization.

Across the evaluated models, previous methods show roughly:

```text
9 ~ 28 point
```

gaps from full precision.

SpinQuant reduces this to roughly:

```text
1.2 ~ 5.3 points
```

depending on the model.

---

# 53. LLaMA-2 70B W3A8

For LLaMA-2 70B:

```text
FP16
72.9
```

GPTQ:

```text
63.9
```

SpinQuant_had:

```text
71.7
```

The gap to full precision becomes:

```text
1.2 points
```

despite using 3-bit weights.

---

# 54. Weight-Only Quantization

SpinQuant also improves purely weight-only quantization.

For LLaMA-2 7B:

```text
FP16
PPL = 5.5
```

AWQ:

```text
6.2
```

SpinQuant:

```text
5.6
```

For LLaMA-2 13B:

```text
FP16
5.0

SpinQuant
5.0
```

Therefore, rotation is also useful even when activations remain high precision.

---

# 55. Calibration Data Robustness

The paper compares rotation learning using:

- WikiText-2
- C4

For LLaMA-2 7B W4A4KV4:

```text
WikiText-2 Calibration
Avg ≈ 64.0

C4 Calibration
Avg ≈ 64.3
```

The results are very similar.

Therefore, SpinQuant is relatively robust to the calibration-data choice in the reported experiments.

---

# 56. Quantization Choice

The paper also compares:

- Symmetric quantization
- Asymmetric quantization
- Clipping
- No clipping

For activation and KV-cache quantization, asymmetric quantization generally performs better.

The final implementation favors:

```text
Min-Max Asymmetric Quantization
```

without clipping for simplicity.

---

# 57. End-to-End CPU Speed

The paper measures LLaMA-3 8B on:

```text
MacBook M1 Pro CPU
```

### FP16

```text
177.15 ms/token
```

### SpinQuant_no_had W4A8

```text
58.88 ms/token
```

### SpinQuant_had W4A8

```text
63.90 ms/token
```

Therefore, low-bit inference provides approximately:

```text
3× speedup
```

over the floating-point baseline in this experiment.

---

# 58. Online Hadamard Overhead

Comparing:

```text
SpinQuant_no_had
58.88 ms/token
```

with:

```text
SpinQuant_had
63.90 ms/token
```

the online Hadamard operations introduce approximately:

```text
8% latency overhead
```

Therefore:

```text
More Accuracy
↕
More Runtime Overhead
```

is an explicit design trade-off.

---

# 59. GPU Hadamard Overhead

The paper also evaluates LLaMA-3 70B on NVIDIA H100.

For batch size 1 and sequence length 4096:

### Without Hadamard

```text
TTFT = 153.58 ms
TTIT = 9.85 ms
```

### With Hadamard

```text
TTFT = 158.25 ms
TTIT = 10.15 ms
```

The latency difference remains relatively small when the Hadamard kernel is efficiently implemented.

---

# 60. SpinQuant vs. SmoothQuant

## SmoothQuant

Core idea:

```text
Activation Outlier
       ↓
Scale Activations Down
       ↓
Scale Weights Up
```

Uses:

```text
Diagonal Scaling
```

Goal:

```text
Move Quantization Difficulty
from Activations to Weights
```

---

## SpinQuant

Core idea:

```text
Activation / Weight Outliers
        ↓
Rotate Representation
        ↓
Spread Magnitude Across Dimensions
```

Uses:

```text
Orthogonal Rotation
```

and goes further by:

```text
Learning the Rotation
```

from quantization loss.

---

# 61. SpinQuant vs. AWQ

## AWQ

Uses:

```text
Activation Magnitude
       ↓
Find Salient Weight Channels
       ↓
Protect Those Channels by Scaling
```

Main target:

```text
Weight-Only Quantization
```

---

## SpinQuant

Uses:

```text
Quantized Network Loss
       ↓
Optimize Whole Rotation Basis
```

Main target includes:

```text
Weights
+
Activations
+
KV Cache
```

up to:

```text
W4A4KV4
```

---

# 62. SpinQuant vs. GPTQ

GPTQ:

```text
Given a Representation
      ↓
Quantize Weights Accurately
using Hessian-Based Compensation
```

SpinQuant:

```text
Change the Representation
      ↓
Make Quantization Easier
```

Therefore:

```text
SpinQuant
+
GPTQ
```

works well because the two methods solve different problems.

---

# 63. SpinQuant vs. QuaRot

The conceptual difference is especially simple:

```text
QuaRot
→ Random Rotation
```

```text
SpinQuant
→ Learned Rotation
```

More precisely:

```text
QuaRot
→ Random Hadamard-based
  Outlier Removal
```

```text
SpinQuant
→ Quantization-Loss-Aware
  Rotation Optimization
```

This is the main conceptual contribution of SpinQuant.

---

# 64. Evolution of Rotation-Based Quantization

A useful progression is:

```text
LLM.int8()
→ Detect outliers
→ Keep them in higher precision
```

```text
SmoothQuant
→ Scale outliers
→ Move difficulty into weights
```

```text
AWQ
→ Use activations to identify
  important weight channels
```

```text
QuaRot
→ Rotate representation
→ Remove outliers
```

```text
SpinQuant
→ Learn which rotation
  minimizes quantization loss
```

This is the key progression from the previous papers.

---

# 65. Main Contributions

### 1. Learned Rotation

SpinQuant replaces random rotations with rotations optimized specifically for quantization.

### 2. Rotation Performance Variance Analysis

The paper shows that different random rotations can differ by up to approximately 13 zero-shot accuracy points.

### 3. Stiefel-Manifold Optimization

Rotation matrices are optimized while preserving orthogonality.

### 4. Cayley SGD

Provides an efficient method for gradient-based optimization of orthogonal rotation matrices.

### 5. Two Deployment Modes

```text
SpinQuant_no_had
→ No additional inference rotation

SpinQuant_had
→ Stronger extreme low-bit accuracy
```

### 6. End-to-End W4A4KV4 Quantization

Achieves strong performance while quantizing:

- Weights
- Activations
- KV cache

to 4 bits.

### 7. GPTQ Compatibility

Learned rotations can be combined with second-order weight quantization.

### 8. Real Hardware Evaluation

The paper evaluates end-to-end latency on:

- Apple M1 Pro
- NVIDIA H100

---

# 66. Limitations

## Rotation Optimization Is Not Free

Unlike QuaRot's random rotations, SpinQuant requires an optimization stage.

For LLaMA-2 70B:

```text
~3.5 hours
```

are required in the reported setup.

---

## Calibration Data Is Required

Rotation learning requires calibration inputs.

Although the method is relatively robust to dataset choice, it is not calibration-free.

---

## Online Rotation May Still Be Needed

For aggressive activation and KV-cache quantization:

```text
R3
R4
```

remain online Hadamard operations.

Therefore:

```text
Extreme Accuracy
↕
Additional Runtime Cost
```

must be considered.

---

## Not Every Model Is Equally Easy to Quantize

The paper shows substantial differences between model families.

For example:

- LLaMA-2
- LLaMA-3
- Mistral

respond differently to the same low-bit settings.

Thus, rotation learning improves robustness but does not make quantization difficulty identical across models.

---

# 67. Key Takeaways

1. SpinQuant is a **learned rotation-based LLM quantization method**.

2. Rotation reduces activation and weight outliers by redistributing magnitude across dimensions.

3. Full-precision Transformer outputs can remain identical under valid paired rotations.

4. Quantized performance, however, strongly depends on which rotation is chosen.

5. Different random rotations can produce up to approximately **13 points** of zero-shot accuracy difference.

6. SpinQuant therefore learns the rotation directly from quantized network loss.

7. R1 rotates the residual stream.

8. R2 rotates internal attention representations.

9. R3 and R4 provide additional online Hadamard rotation for aggressive KV-cache and activation quantization.

10. SpinQuant_no_had uses only learned R1/R2 and requires no additional online rotation.

11. SpinQuant_had adds R3/R4 and is particularly useful for **W4A4KV4**.

12. The rotation matrices are optimized on the **Stiefel manifold**.

13. Cayley SGD preserves orthogonality during gradient-based optimization.

14. Only a small fraction of the original parameter count is optimized.

15. SpinQuant and GPTQ are complementary: SpinQuant optimizes the representation while GPTQ optimizes weight quantization.

16. On LLaMA-2 7B W4A4KV4, SpinQuant_had reaches **64.0** average zero-shot accuracy compared with **66.9** in full precision.

17. This leaves only a **2.9-point gap**, compared with a **22-point gap** for LLM-QAT in the same reported setting.

18. Learned rotations consistently outperform random Hadamard rotations.

19. On Mistral-7B, learned SpinQuant_had improves W4A4KV4 zero-shot accuracy by **16.2 points** over random Hadamard rotation in the reported ablation.

20. SpinQuant demonstrates that the optimal low-bit representation is not necessarily the representation of the pretrained model.

21. Quantization can be improved not only by designing a better quantizer, but also by **learning a better coordinate system in which to quantize**.

---

# 68. Concepts to Review

- Post-Training Quantization
- Weight Quantization
- Activation Quantization
- KV Cache Quantization
- W4A4KV4
- Activation Outlier
- Orthogonal Matrix
- Rotation Matrix
- Hadamard Matrix
- Randomized Hadamard Matrix
- Rotational Invariance
- Computational Invariance
- Kurtosis
- Quantization Error
- Signal-to-Quantization-Noise Ratio
- Stiefel Manifold
- Skew-Symmetric Matrix
- Cayley Transform
- Cayley SGD
- Calibration Dataset
- GPTQ
- RMSNorm
- Residual Stream
- Multi-Head Attention
- KV Cache
- Online Hadamard Transform
- TTFT
- TTIT
