# QuaRot

> **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**

- **Authors:** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman
- **Venue:** NeurIPS 2024
- **Topic:** End-to-End Low-Bit LLM Quantization
- **Keywords:** QuaRot, Rotation, Hadamard Transform, Activation Outliers, A4W4KV4, KV Cache Quantization, Computational Invariance

---

## 1. Problem Background

Large Language Model inference requires substantial:

- Computation
- Memory capacity
- Memory bandwidth
- Energy

Quantization reduces these costs by representing model data using lower precision.

However, most LLM quantization methods primarily focus on:

```text
Weight-Only Quantization
```

such as:

```text
W4A16
```

where:

```text
Weight      → INT4
Activation  → FP16
```

This reduces model memory, but the major matrix multiplications are still performed using higher-precision activations.

QuaRot aims for a more aggressive target:

```text
Weight      → INT4
Activation  → INT4
KV Cache    → INT4
```

or:

```text
A4W4KV4
```

---

# 2. Why Activation Quantization Is Difficult

Weights are generally easier to quantize than activations.

The major obstacle is:

> **Activation Outliers**

Some hidden-state dimensions contain values much larger than the majority of activations.

Conceptually:

```text
Normal activation values

0.03
-0.08
0.04
0.06
...

Outlier channel

0.03
-0.08
0.91
0.06
     ↑
   Outlier
```

A large outlier determines the quantization range.

Therefore:

```text
Large Outlier
     ↓
Large Quantization Range
     ↓
Large Quantization Step
     ↓
Lower Resolution for Normal Values
     ↓
Higher Quantization Error
```

This problem becomes particularly severe at 4-bit precision.

---

# 3. Existing Solutions

Previous approaches generally deal with activation outliers by:

```text
Identify Outlier Channels
         ↓
Keep Them in Higher Precision
```

For example:

```text
Normal Channels
→ INT4

Outlier Channels
→ FP16 / INT8
```

Although accurate, this leads to:

- Mixed-precision execution
- More complicated kernels
- Reordering overhead
- Hardware inefficiency

QuaRot asks a different question:

> Can the outliers be removed instead of treated specially?

---

# 4. Main Idea

The central idea of QuaRot is:

> **Rotate the representation so that outliers disappear while preserving the model's output.**

The method uses:

```text
Orthogonal Transformations
        +
Hadamard Rotations
```

to redistribute large activation values across dimensions.

Conceptually:

```text
Before Rotation

Dimension 1   small
Dimension 2   small
Dimension 3   HUGE
Dimension 4   small
Dimension 5   small
```

After rotation:

```text
After Rotation

Dimension 1   moderate
Dimension 2   moderate
Dimension 3   moderate
Dimension 4   moderate
Dimension 5   moderate
```

The information is preserved, but it is distributed more evenly.

---

# 5. Key Observation: Rotation Does Not Need to Change the Model

QuaRot relies on a property called:

> **Computational Invariance**

If an orthogonal transformation is inserted into one part of a Transformer and its inverse is absorbed into another part, the overall function of the network remains unchanged.

Therefore:

```text
Original Model
      ↓
Rotate Representation
      ↓
Modify Corresponding Weights
      ↓
Same Model Output
```

while the internal representation becomes easier to quantize.

---

# 6. Orthogonal Matrices

An orthogonal matrix $Q$ satisfies:

```math
QQ^T = I
```

and therefore:

```math
Q^{-1} = Q^T
```

Orthogonal transformations preserve vector norms.

For a vector $x$:

```math
\|xQ\|_2 = \|x\|_2
```

This property is important because Transformers commonly use normalization layers such as RMSNorm.

---

# 7. Hadamard Matrices

QuaRot primarily uses Hadamard transformations.

A simple normalized Hadamard matrix is:

```math
H_2
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
```

Larger Walsh-Hadamard matrices can be recursively constructed.

The Hadamard transform has two important properties:

1. It is orthogonal.
2. It can be computed efficiently.

For a vector of dimension $d$:

```text
Dense Matrix Multiplication
→ O(d²)

Fast Walsh-Hadamard Transform
→ O(d log d)
```

This makes Hadamard rotation attractive for inference.

---

# 8. Randomized Hadamard Rotation

QuaRot also uses randomized Hadamard matrices.

Conceptually:

```math
\tilde{H}
=
H \mathrm{diag}(s)
```

where the entries of $s$ are randomly selected from:

```text
+1
or
-1
```

The randomized Hadamard matrix remains orthogonal.

The random signs help spread high-magnitude components across the hidden dimensions.

---

# 9. Incoherence Processing

The paper relates rotation to:

> **Incoherence Processing**

A matrix is difficult to quantize when a few elements are much larger than its typical values.

Conceptually:

```text
Highly Coherent

small small small HUGE small
```

versus:

```text
More Incoherent

medium medium medium medium medium
```

Orthogonal transformations can make the representation more incoherent by spreading large values across dimensions.

This helps quantize both:

- Weights
- Activations

---

# 10. Effect on Activation Distribution

The paper's Figure 1 directly shows the effect of QuaRot.

Before QuaRot:

```text
Activation distribution
→ Large channel-wise spikes
→ Strong outliers
```

After QuaRot:

```text
Activation distribution
→ Much more uniform
→ Outliers essentially removed
```

The important point is:

> QuaRot changes the internal representation, not the model's underlying function.

---

# 11. Computational Invariance with RMSNorm

Transformers such as LLaMA use RMSNorm.

Ignoring its learned scaling factor temporarily:

```math
\mathrm{RMSNorm}(X)
=
\frac{X}{\|X\|}
```

Since orthogonal rotations preserve the norm:

```math
\|XQ^T\|
=
\|X\|
```

QuaRot uses the property:

```math
\mathrm{RMSNorm}(X)
=
\mathrm{RMSNorm}(XQ^T)Q
```

This allows rotations to propagate through Transformer blocks without changing the final output.

---

# 12. QuaRot Overview

QuaRot consists of two broad stages.

```text
Stage 1
Rotate / Modify Model
        ↓
Remove Outliers

Stage 2
Quantize Model
        ↓
Weights      → INT4
Activations  → INT4
KV Cache     → INT4
```

More specifically:

```text
Stage 1a
Weight Modification

Stage 1b
Rotate FFN Activations

Stage 1c
Rotate Attention Values

Stage 1d
Rotate Attention Keys

Stage 2a
Weight Quantization

Stage 2b
Online Activation Quantization

Stage 2c
KV Cache Quantization
```

---

# 13. Stage 1a: Weight Modification

QuaRot first absorbs normalization scaling factors into adjacent weight matrices.

It then introduces a randomized Hadamard rotation $Q$ into the hidden state.

For example, a key projection weight can be modified as:

```math
W_k
\leftarrow
Q^T
\mathrm{diag}(\alpha)
W_k
```

where $\alpha$ corresponds to the RMSNorm scaling parameters.

Similar transformations are applied to other projection matrices.

---

# 14. Why Weight Modification Is Almost Free

Many Hadamard transformations do not need to be explicitly executed during inference.

Instead, they can be:

```text
Precomputed
     ↓
Multiplied into Weight Matrices
     ↓
Stored as Modified Weights
```

Therefore:

```text
Rotation
≠
Always Additional Runtime Kernel
```

A major part of QuaRot's rotations is fused into weights offline.

---

# 15. Rotated Hidden State

After the global transformation, the hidden state effectively becomes:

```math
X
\rightarrow
XQ
```

This hidden state has a much more quantization-friendly distribution.

The next layer contains the corresponding inverse transformation in its modified weights.

Therefore:

```text
Representation Changes
but
Network Function Does Not
```

---

# 16. Stage 1b: FFN Activation Rotation

A LLaMA-style feed-forward network contains approximately:

```text
Input
  ↓
Wup / Wgate
  ↓
Activation + Gating
  ↓
Wdown
```

The intermediate activation before:

```text
Wdown
```

can still contain outliers.

QuaRot therefore performs an online Hadamard transformation before the down-projection.

Conceptually:

```text
FFN Intermediate Activation
          ↓
      Hadamard
          ↓
     Outliers Spread
          ↓
        INT4
          ↓
       Wdown
```

The inverse transformation is fused into the modified down-projection weight.

---

# 17. QuaRot FFN Precision Flow

The modified FFN approximately follows:

```text
Rotated Hidden State
        ↓
       INT4
        ↓
INT4 Wup / Wgate
        ↓
INT4 × INT4 GEMM
        ↓
INT32 Accumulator
        ↓
FP16
        ↓
Activation / Gating
        ↓
Online Hadamard
        ↓
       INT4
        ↓
INT4 Wdown
        ↓
FP16 Rotated Output
```

Not every operation in the network is executed in INT4.

For example:

- RMSNorm remains higher precision.
- Nonlinear operations are performed in floating point.
- INT4 matrix multiplications accumulate in INT32.

The main matrix multiplications, however, can use INT4 inputs and weights.

---

# 18. Stage 1c: Attention Value Rotation

QuaRot also applies rotation to the attention value path.

Inside a single attention head:

```text
V
→ Attention Weight × V
→ Wout
```

The value projection and output projection have a structure that allows an orthogonal transform to be inserted between them without changing the result.

Conceptually:

```text
Wv
      ↓
Hadamard Rotation
      ↓
Attention
      ↓
Inverse Rotation
      ↓
Wout
```

Parts of these transformations can again be fused into the corresponding weights.

---

# 19. Head-Wise Hadamard Rotation

For multi-head attention, QuaRot applies Hadamard transformations inside the head dimensions.

The value projection is modified so that its output becomes rotated.

The output projection is modified correspondingly so that the transformation is canceled.

This improves the distribution of attention activations before quantization.

---

# 20. Stage 1d: Key Rotation

The KV cache introduces another difficulty.

Keys also contain outliers.

However, keys interact directly with queries:

```math
QK^T
```

Therefore, rotating only the keys would change attention scores.

QuaRot instead rotates:

```text
Query
and
Key
```

using the same orthogonal transformation.

Because:

```math
(QH)(KH)^T
=
QHH^TK^T
```

and:

```math
HH^T = I
```

the attention score remains unchanged.

---

# 21. Interaction with RoPE

Modern LLaMA models apply:

> **Rotary Positional Embedding (RoPE)**

to keys and queries.

This prevents QuaRot from simply fusing every key/query rotation into the projection weights.

Therefore, the paper applies online head-wise Hadamard transformations after positional encoding.

Conceptually:

```text
Q Projection
      ↓
RoPE
      ↓
Hadamard

K Projection
      ↓
RoPE
      ↓
Hadamard
```

Because both receive the same rotation, the attention scores remain unchanged.

---

# 22. Why Post-RoPE Caching Is Used

The paper discusses two possible designs:

```text
Pre-RoPE Cache
or
Post-RoPE Cache
```

QuaRot uses:

> **Post-RoPE Caching**

During autoregressive decoding:

```text
1 New Query
+
Many Cached Keys
```

Re-rotating every cached key at each decoding step would be expensive.

With Post-RoPE caching, only the newly generated key/query vector needs the online rotation before it is stored or used.

---

# 23. Stage 2a: Weight Quantization

After the rotations are applied, weights become easier to quantize.

The default method used by QuaRot is:

```text
GPTQ
```

The important point is that QuaRot itself is not tied to GPTQ.

Conceptually:

```text
QuaRot
→ Improve Representation

GPTQ
→ Quantize Modified Weights
```

Other weight quantization methods can also be applied afterward.

---

# 24. QuaRot + GPTQ

GPTQ provides:

```text
Second-Order
Weight Quantization
```

while QuaRot provides:

```text
Rotation-Based
Outlier Removal
```

Therefore:

```text
QuaRot
        +
GPTQ
        ↓
Better Weight Quantization
+
Better Activation Quantization
```

The two techniques solve different parts of the problem.

---

# 25. Stage 2b: Activation Quantization

Activations are quantized online.

QuaRot uses symmetric per-token quantization.

For each token:

```math
\Delta
=
\frac{
\max(|x|)
}{
7
}
```

because signed INT4 can represent values up to approximately:

```text
+7
```

The activation is then quantized using round-to-nearest.

Conceptually:

```text
FP16 Token
      ↓
Find Max Magnitude
      ↓
Calculate Scale
      ↓
Divide by Scale
      ↓
Round
      ↓
INT4 Token
```

---

# 26. INT4 GEMM

The major linear operations can then use:

```text
INT4 Activation
       ×
INT4 Weight
       ↓
INT32 Accumulator
```

The result is then scaled and cast back to FP16 for the rest of the model.

Conceptually:

```text
INT4 × INT4
     ↓
INT32
     ↓
Rescale
     ↓
FP16
```

---

# 27. Stage 2c: KV Cache Quantization

QuaRot also quantizes:

```text
Key Cache
+
Value Cache
```

to low precision.

The default experiments use:

```text
4-bit KV Cache
```

with grouped asymmetric quantization.

This is especially important for:

- Long sequences
- Large batch sizes

where KV cache memory becomes a major bottleneck.

---

# 28. Attention Precision

QuaRot does not perform every part of attention in INT4.

The paper keeps query processing and some attention computation in higher precision.

Conceptually:

```text
KV Cache
→ Stored in INT4

Load Cache
→ Dequantize

Attention Dot Product
→ FP16
```

The memory benefit therefore comes primarily from storing the large KV cache in low precision.

---

# 29. Full QuaRot Target

The main configuration is:

```text
A4W4KV4
```

meaning:

```text
Activation → 4 bit
Weight     → 4 bit
KV Cache   → 4 bit
```

No special outlier channels need to remain in higher precision.

This is one of the central distinctions from previous 4-bit approaches.

---

# 30. Experimental Setup

Main model family:

```text
LLaMA-2
```

including:

- 7B
- 13B
- 70B

For GPTQ calibration:

```text
128 WikiText-2 Samples
```

with:

```text
Sequence Length = 2048
```

For LLaMA2-70B on one A100:

```text
QuaRot Model Modification
≈ 5 minutes

GPTQ Quantization
≈ 2 hours
```

---

# 31. 4-Bit Perplexity Results

WikiText-2 perplexity:

| Model | FP16 | QuaRot A4W4KV4 |
|---|---:|---:|
| LLaMA2-7B | 5.47 | 6.10 |
| LLaMA2-13B | 4.88 | 5.40 |
| LLaMA2-70B | 3.32 | 3.79 |

For LLaMA2-70B:

```text
3.32
→
3.79
```

which corresponds to:

```text
+0.47 PPL
```

while quantizing:

```text
Weights
+
Activations
+
KV Cache
```

to 4 bits.

---

# 32. Comparison with Previous 4-Bit Methods

For LLaMA2-7B:

| Method | PPL |
|---|---:|
| FP16 | 5.47 |
| SmoothQuant 4-bit | 83.12 |
| OmniQuant | 14.26 |
| QUIK-4B | 8.87 |
| QuaRot | **6.10** |

QuaRot does this with:

```text
0 High-Precision Outlier Features
```

while QUIK retains:

```text
256 Outlier Features
```

in higher precision.

---

# 33. Group-Wise QuaRot

Group-wise quantization further improves accuracy.

For LLaMA2-70B:

```text
Baseline
3.32
```

```text
QuaRot
3.79
```

```text
QuaRot-256G
3.63
```

```text
QuaRot-128G
3.61
```

```text
QuaRot-64G
3.58
```

Smaller groups improve quantization quality at the cost of:

- More scaling metadata
- More complicated kernels

---

# 34. Zero-Shot Results

The paper evaluates:

- PIQA
- WinoGrande
- HellaSwag
- ARC-Easy
- ARC-Challenge
- LAMBADA

For LLaMA2-70B:

```text
FP16 Average
77.07
```

```text
QuaRot A4W4KV4
75.98
```

The paper summarizes this as retaining approximately:

```text
99%
```

of zero-shot performance.

---

# 35. 6-Bit and 8-Bit Quantization

QuaRot also evaluates higher precision.

For LLaMA2-70B:

### INT6

```text
FP16 PPL
3.32

QuaRot-RTN
3.36
```

### INT8

```text
QuaRot-RTN
3.33
```

At 6 and 8 bits, the paper reports essentially lossless quantization.

---

# 36. Calibration-Free RTN

An interesting result is that QuaRot works with simple:

```text
Round-to-Nearest
```

weight quantization.

Unlike GPTQ:

```text
RTN
→ No Calibration Set
→ No Hessian
→ No Optimization
```

At INT8, QuaRot-RTN is essentially lossless.

This demonstrates how much easier quantization becomes once the representation is rotated into a more uniform distribution.

---

# 37. KV Cache Quantization Results

When the rest of the model remains high precision, 4-bit KV cache quantization produces almost no degradation.

For LLaMA2-70B:

```text
FP16 KV
3.32
```

```text
K4 V4
3.33
```

Even:

```text
K3 V3
```

gives:

```text
3.39
```

This shows that rotation makes KV-cache quantization substantially easier.

---

# 38. Keys Are More Sensitive Than Values

The KV-cache ablation shows an asymmetry.

For LLaMA2-7B:

```text
K4 V3
PPL = 5.54
```

while:

```text
K3 V4
PPL = 5.65
```

Therefore:

> Keys are more sensitive to quantization than values.

This matches observations in other KV-cache quantization research.

---

# 39. Weight-Only Quantization Benefit

QuaRot also improves weight-only quantization.

For LLaMA2-70B at W4A16:

```text
GPTQ
3.87
```

versus:

```text
QuaRot + GPTQ
3.41
```

Therefore, even without activation quantization, rotation makes the weight distribution easier to quantize.

The benefit becomes even larger at lower bit widths.

---

# 40. Prefill vs. Decoding

QuaRot targets two different bottlenecks.

## Prefill

Prefill processes many tokens simultaneously.

It is relatively:

```text
Compute-Bound
```

Therefore:

```text
INT4 Matrix Multiplication
```

can directly accelerate computation.

---

## Decoding

Autoregressive decoding processes one new token at a time.

It is more strongly affected by:

```text
Memory Traffic
```

Therefore:

```text
4-bit Weights
+
4-bit KV Cache
```

primarily improve memory efficiency.

---

# 41. Prefill Speedup

Performance is measured on NVIDIA RTX 3090 GPUs.

For sequence length:

```text
2048
```

LLaMA2-7B achieves approximately:

```text
1.97× ~ 2.16×
```

prefill speedup depending on batch size.

LLaMA2-70B achieves:

```text
3.16× ~ 3.33×
```

speedup.

---

# 42. Larger Batch Size Helps

For LLaMA2-70B:

| Batch Size | Prefill Speedup |
|---:|---:|
| 1 | 3.16× |
| 4 | 3.27× |
| 16 | 3.32× |
| 32 | 3.33× |

As batch size increases:

```text
Compute Utilization ↑
```

and the benefit from INT4 Tensor Core computation becomes clearer.

---

# 43. INT4 Linear Layer Speedup

The paper separately benchmarks the 4-bit linear layer.

For LLaMA2 FFN-sized matrices:

```text
LLaMA2-7B
≈ 3.2×
```

and:

```text
LLaMA2-70B
≈ 4.3×
```

speedup over FP16 for the tested kernel configuration.

---

# 44. Hadamard Runtime Overhead

A possible concern is:

> Does the online Hadamard transformation eliminate the INT4 speedup?

The paper measures the overhead and finds it relatively small.

For the tested linear layers:

```text
Hadamard Overhead
≤ ~7%
```

Therefore:

```text
INT4 Benefit
>>
Hadamard Cost
```

for the main linear operations.

---

# 45. Decoding Memory Saving

The decoding stage benefits strongly from 4-bit storage.

For LLaMA2-7B, peak memory saving reaches approximately:

```text
3.75×
```

For LLaMA2-70B:

```text
≈ 3.89×
```

for the tested batch-size-16 configurations.

This comes from compressing both:

- Model data
- KV cache

---

# 46. Why 4-Bit KV Cache Is Not Always Faster

Low-bit KV cache reduces memory I/O.

However, quantization and dequantization also introduce overhead.

For small batch sizes:

```text
Quantization Overhead
>
Memory Saving Benefit
```

so INT4 cache access can actually be slower than FP16.

For larger batch sizes or longer sequences:

```text
Memory I/O Cost ↑
```

and the 4-bit representation becomes beneficial.

This is an important hardware observation:

```text
Smaller Precision
≠
Automatic Speedup
```

---

# 47. QuaRot vs. LLM.int8()

## LLM.int8()

Problem:

```text
Activation Outliers
```

Solution:

```text
Normal Features
→ INT8

Outlier Features
→ FP16
```

Therefore:

```text
Mixed Precision
```

is used.

---

## QuaRot

Problem:

```text
Activation Outliers
```

Solution:

```text
Rotate Representation
       ↓
Remove Outliers
       ↓
Quantize Everything Uniformly
```

Therefore:

```text
No Special High-Precision
Outlier Channels
```

are required.

---

# 48. QuaRot vs. SmoothQuant

## SmoothQuant

Uses:

```text
Per-Channel Scaling
```

to move quantization difficulty:

```text
Activation
→
Weight
```

Goal:

```text
W8A8
```

---

## QuaRot

Uses:

```text
Orthogonal Rotation
```

to redistribute activation magnitude across dimensions.

Goal:

```text
A4W4KV4
```

Rather than shifting outlier magnitude into another tensor, QuaRot spreads it across dimensions.

---

# 49. QuaRot vs. AWQ

## AWQ

Uses activation statistics to determine:

```text
Which Weight Channels
Are Important?
```

and scales those weights to reduce their relative quantization error.

Primary target:

```text
W4A16
```

---

## QuaRot

Uses rotations to make the entire representation more quantization-friendly.

Primary target:

```text
A4W4KV4
```

Therefore:

```text
AWQ
→ Protect Important Weights

QuaRot
→ Remove Outliers from Representation
```

---

# 50. QuaRot vs. GPTQ

GPTQ and QuaRot are largely complementary.

## GPTQ

```text
Weight Quantization Error
       ↓
Second-Order Compensation
```

## QuaRot

```text
Poor Representation Distribution
       ↓
Orthogonal Rotation
       ↓
Remove Outliers
```

QuaRot uses GPTQ as its default weight quantizer.

Therefore:

```text
QuaRot
→ Make Quantization Easier

GPTQ
→ Perform Accurate Weight Quantization
```

---

# 51. Evolution of LLM Quantization

A useful progression is:

```text
LLM.int8()
     ↓
Outliers exist
→ Keep them in FP16
```

```text
SmoothQuant
     ↓
Move activation difficulty
into weights using scaling
```

```text
AWQ
     ↓
Use activation information
to protect important weights
```

```text
QuaRot
     ↓
Rotate the representation
so the outliers disappear
```

This represents an important shift:

> Instead of designing more complicated quantizers around bad distributions, transform the distributions into something easier to quantize.

---

# 52. Main Contributions

### 1. Rotation-Based End-to-End Quantization

Uses orthogonal rotations to enable low-bit quantization of weights, activations, and KV cache.

### 2. Outlier Removal

Randomized Hadamard transforms remove activation outlier features.

### 3. Computational Invariance

Many rotations are absorbed into model weights without changing the original model function.

### 4. Efficient Online Hadamard Operations

Only a small number of runtime Hadamard transforms are required.

### 5. 4-Bit KV Cache

Keys and values can be stored using low precision without retaining high-precision outlier channels.

### 6. Uniform INT4 MatMul

Major linear layers use INT4 activations and INT4 weights.

### 7. No High-Precision Outlier Channels

Unlike previous mixed-precision approaches, QuaRot does not need to explicitly preserve special outlier dimensions.

### 8. Real Kernel Evaluation

The paper implements INT4 kernels and evaluates actual speed and memory behavior on RTX 3090 GPUs.

---

# 53. Limitations

## Runtime Hadamard Operations

Not every rotation can be completely fused offline.

Some Hadamard transforms must still be executed during inference.

---

## Some Operations Remain Higher Precision

QuaRot does not literally perform every Transformer operation in INT4.

For example:

- RMSNorm
- Nonlinear functions
- Softmax
- Some attention computation

remain in FP16 or FP32.

---

## 4-Bit Accuracy Is Not Lossless

For LLaMA2-70B:

```text
FP16
PPL = 3.32
```

versus:

```text
A4W4KV4
PPL = 3.79
```

There is still measurable degradation at 4 bits.

---

## Newer Models Can Be More Sensitive

The appendix reports that LLaMA-3 is more sensitive to 4-bit QuaRot than LLaMA-2.

For example, the LLaMA3-70B WikiText-2 perplexity changes from:

```text
2.86
```

to:

```text
5.51
```

with 128-group QuaRot.

Therefore, the same quantization scheme does not necessarily transfer equally well to every model family.

---

## KV Cache Speedup Depends on Workload

Low-bit KV storage helps most when:

- Batch size is large
- Sequence length is long
- KV-cache I/O dominates

For smaller workloads, quantization overhead can dominate.

---

# 54. Key Takeaways

1. QuaRot is a rotation-based method for low-bit LLM quantization.

2. Its primary target is **A4W4KV4** inference.

3. The main obstacle to activation quantization is large outlier features.

4. QuaRot removes these outliers by rotating hidden representations using orthogonal Hadamard transformations.

5. Orthogonal rotations preserve vector norms and can be combined with Transformer normalization through computational invariance.

6. Many rotations can be absorbed directly into model weights offline.

7. A small number of online Hadamard operations are used for intermediate FFN and attention activations.

8. Keys and queries are rotated consistently so that attention scores remain unchanged.

9. Rotation allows the KV cache to be quantized to 4 bits without preserving special high-precision outlier channels.

10. QuaRot can be combined with GPTQ because rotation and second-order weight quantization address different problems.

11. On LLaMA2-70B, A4W4KV4 QuaRot changes WikiText-2 perplexity from **3.32 to 3.79**.

12. The same model retains approximately **99% of the reported zero-shot performance**.

13. 6-bit and 8-bit configurations are essentially lossless in the paper's LLaMA-2 experiments.

14. LLaMA2-70B achieves up to approximately **3.33× prefill speedup** in the reported single-block RTX 3090 measurements.

15. Decoding peak memory usage is reduced by approximately **3.89×** in the reported LLaMA2-70B experiments.

16. Lower precision does not automatically guarantee lower latency because quantization, dequantization, and rotation also have runtime costs.

17. QuaRot demonstrates that **changing the internal representation can be as important as designing the quantizer itself**.

---

# 55. Concepts to Review

- Activation Outliers
- Weight Quantization
- Activation Quantization
- KV Cache Quantization
- A4W4KV4
- Orthogonal Matrix
- Rotation Matrix
- Hadamard Matrix
- Walsh-Hadamard Transform
- Randomized Hadamard Transform
- Incoherence
- Computational Invariance
- RMSNorm
- Feed-Forward Network
- Multi-Head Attention
- Query / Key / Value
- RoPE
- Post-RoPE Caching
- Round-to-Nearest
- GPTQ
- Per-Token Quantization
- Group-Wise Quantization
- INT4 GEMM
- INT32 Accumulator
- Prefill
- Decoding
- Compute-Bound Workload
- Memory-Bound Workload
