# GPTQ

> **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**

- **Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Venue:** ICLR 2023
- **Topic:** Post-Training Quantization for Large Language Models
- **Keywords:** GPTQ, PTQ, Weight Quantization, Second-Order Information, Hessian, Error Compensation, LLM

---

## 1. Problem Background

Large language models such as GPT, OPT, and BLOOM contain billions or hundreds of billions of parameters.

This creates very large inference costs in terms of:

- GPU memory
- Memory bandwidth
- Number of GPUs
- Inference latency

For example, the paper states that OPT-175B stored in FP16 requires approximately:

```text
326 GB
```

of parameter memory.

This means that even inference requires multiple high-end GPUs.

---

## Why Not Retrain?

Quantization-Aware Training or fine-tuning can improve quantization accuracy, but retraining models with hundreds of billions of parameters is extremely expensive.

Therefore, the paper focuses on:

> **Post-Training Quantization (PTQ)**

```text
Pretrained LLM
      +
Small Calibration Dataset
      ↓
Quantization
      ↓
No Retraining
```

---

## Existing Problem

Simple Round-to-Nearest (RTN) quantization scales easily to very large models.

However:

```text
8-bit
→ Usually acceptable

3-bit / 4-bit
→ Significant accuracy degradation
```

More accurate PTQ methods such as:

- AdaRound
- BRECQ
- Optimal Brain Quantization

are too computationally expensive to directly apply to 100B-scale models.

The central question is therefore:

> Can an accurate second-order PTQ method be made efficient enough for models with hundreds of billions of parameters?

---

# 2. Main Idea

GPTQ is a **one-shot weight quantization method based on approximate second-order information**.

Its main goal is:

```text
High Accuracy
     +
Low Bit Width
     +
Large-Scale LLM Support
     +
Reasonable Quantization Time
```

GPTQ builds directly on **Optimal Brain Quantization (OBQ)**.

The key principle remains:

```text
Quantize Weight
      ↓
Quantization Error
      ↓
Use Second-Order Information
      ↓
Update Remaining Weights
      ↓
Compensate Error
```

However, GPTQ changes the algorithm so that it becomes practical for extremely large Transformer layers.

---

# 3. Layer-Wise Quantization Objective

For a linear layer:

```math
Y = WX
```

where:

- $W$: original weight matrix
- $X$: calibration input
- $Y$: original layer output

GPTQ searches for quantized weights $W_c$ that minimize:

```math
\left\|
WX - W_cX
\right\|_2^2
```

The goal is therefore not simply:

```text
Make each quantized weight close to the FP16 weight
```

but rather:

```text
Preserve the output of the original layer
```

---

# 4. Connection to Optimal Brain Quantization

GPTQ directly builds on OBQ.

OBQ quantizes one weight at a time and compensates for the resulting error using the remaining unquantized weights.

For a weight $w_q$, the quantization score is based on:

```math
\frac{
\left(
q(w_q)-w_q
\right)^2
}
{
[H^{-1}]_{qq}
}
```

where:

- $q(w_q)$ is the quantized value
- $H$ is the Hessian of the layer-wise reconstruction objective

The corresponding weight update distributes the quantization error to the remaining weights using $H^{-1}$.

---

# 5. Hessian in Layer-Wise Quantization

For the reconstruction objective:

```math
\left\|
WX-W_cX
\right\|_2^2
```

the Hessian depends on the layer input $X$.

Conceptually:

```math
H = 2XX^T
```

Therefore, the calibration activations contain information about:

> Which weight perturbations matter for the actual layer output.

This makes GPTQ different from methods that only examine weight magnitude.

---

# 6. Why OBQ Does Not Scale

OBQ is accurate, but computationally expensive.

For a weight matrix:

```math
W \in
\mathbb{R}^{d_{row}\times d_{col}}
```

the original OBQ algorithm has complexity approximately:

```text
O(d_row × d_col^3)
```

because:

1. Rows are treated independently.
2. Weights are greedily selected one by one.
3. Hessian inverse information must repeatedly be updated.

This is practical for models such as ResNet-50.

It is not practical for:

```text
OPT-175B
BLOOM-176B
```

GPTQ therefore modifies OBQ in three major steps.

---

# 7. GPTQ Step 1: Arbitrary Quantization Order

OBQ selects the next weight greedily:

```text
Find Weight with Lowest Quantization Cost
        ↓
Quantize It
        ↓
Update Remaining Weights
```

GPTQ makes an important empirical observation:

> Greedy ordering provides only a small benefit over using a fixed arbitrary order, especially for large layers.

Therefore, GPTQ uses the **same column order across all rows**.

Instead of:

```text
Row 1 → Different Quantization Order
Row 2 → Different Quantization Order
Row 3 → Different Quantization Order
```

GPTQ uses:

```text
Column 1
   ↓
Column 2
   ↓
Column 3
   ↓
...
```

for all rows.

---

## Why This Matters

The Hessian depends on:

```text
Layer Inputs X
```

not on the individual weight row.

Therefore, if all rows follow the same column order:

```text
All Rows
   ↓
Same Remaining Columns
   ↓
Same Hessian State
```

The expensive Hessian inverse update only needs to be performed once per column rather than once per weight.

The complexity is reduced from approximately:

```text
O(d_row × d_col^3)
```

to:

```text
O(max(d_row × d_col^2, d_col^3))
```

This provides orders-of-magnitude speedup for large layers.

---

# 8. GPTQ Step 2: Lazy Batch Updates

Even after removing greedy ordering, a direct implementation is still inefficient on GPUs.

The reason is that repeated small weight updates have:

```text
Low Compute
+
High Memory Traffic
```

and therefore do not utilize GPU compute efficiently.

GPTQ addresses this using **lazy batch updates**.

---

## Block Processing

Instead of immediately updating the entire remaining weight matrix after every column:

```text
Quantize Column
      ↓
Update Entire Matrix
      ↓
Quantize Next Column
      ↓
Update Entire Matrix
```

GPTQ groups multiple columns into a block.

The paper commonly uses:

```text
B = 128 columns
```

Conceptually:

```text
Block 1
│
├── Quantize Column 1
├── Quantize Column 2
├── ...
└── Quantize Column 128
        ↓
Apply Global Update

Block 2
        ↓
...
```

---

## Why Lazy Update Works

The rounding decision for the current column depends only on updates affecting that column.

Updates to future columns do not need to be immediately materialized.

Therefore, GPTQ can accumulate errors locally and apply a larger matrix update at the end of the block.

This transforms many small memory-bound operations into larger GPU-friendly operations.

The paper reports that this optimization provides approximately an **order-of-magnitude practical speedup** for very large models.

---

# 9. GPTQ Step 3: Cholesky Reformulation

Repeated direct updates of the inverse Hessian can accumulate numerical errors.

For very large models, this can make the inverse Hessian numerically unstable or indefinite.

This may produce incorrect and very large weight updates.

GPTQ therefore reformulates the required inverse-Hessian information using:

> **Cholesky decomposition**

Conceptually:

```text
Hessian
   ↓
Inverse Hessian
   ↓
Cholesky Factorization
   ↓
Stable Second-Order Information
```

A small diagonal damping term is also applied.

The paper uses approximately:

```text
1% of the average diagonal value
```

as damping.

---

# 10. GPTQ Algorithm Overview

The complete procedure can be summarized as:

```text
Pretrained Model
       │
       ▼
Calibration Inputs
       │
       ▼
Compute Layer Hessian
H = 2XX^T
       │
       ▼
Add Damping
       │
       ▼
Compute Cholesky Representation
of H^-1
       │
       ▼
Process Columns in Blocks
       │
       ├── Quantize Column
       │
       ├── Compute Quantization Error
       │
       └── Update Remaining Columns
       │
       ▼
Apply Lazy Global Update
       │
       ▼
Next Block
       │
       ▼
Quantized Weight Matrix
```

---

# 11. Quantization Error Compensation

The defining feature of GPTQ is not simply low-bit rounding.

Suppose:

```text
Original Weight
     ↓
Quantized Weight
```

creates error:

```math
e
=
w-q(w)
```

GPTQ uses inverse-Hessian information to redistribute this error to weights that have not yet been quantized.

Therefore:

```text
Quantize
   ↓
Measure Error
   ↓
Compensate Remaining Weights
   ↓
Continue Quantization
```

This is why GPTQ performs much better than naive RTN at aggressive bit widths.

---

# 12. GPTQ vs. Round-to-Nearest

Round-to-Nearest:

```text
Weight
   ↓
Nearest Grid Point
   ↓
Done
```

GPTQ:

```text
Weight
   ↓
Quantization
   ↓
Error Estimation
   ↓
Second-Order Compensation
   ↓
Update Remaining Weights
   ↓
Next Weight
```

The difference becomes especially significant at:

```text
3-bit
and
4-bit
```

quantization.

---

# 13. Calibration Setup

The paper uses only:

```text
128 random segments
```

from the C4 dataset.

Each segment contains:

```text
2048 tokens
```

No task-specific training data is required.

GPTQ therefore remains a one-shot PTQ method.

---

# 14. Memory-Efficient Quantization Procedure

Even quantizing OPT-175B poses a problem because the full FP16 model cannot fit onto one GPU.

GPTQ solves this by processing one Transformer block at a time.

Conceptually:

```text
Load Transformer Block
        ↓
Collect Hessian Statistics
        ↓
Quantize Block
        ↓
Run Quantized Block
        ↓
Produce Inputs for Next Block
        ↓
Remove Current Block
        ↓
Load Next Block
```

The paper processes Transformer blocks containing six layers at a time.

This allows even 175B models to be quantized using:

```text
1 × NVIDIA A100 80GB
```

---

# 15. Sequential Block Quantization

The input to each later Transformer block is not taken from the original FP16 model.

Instead:

```text
Block 1 Quantized
      ↓
Quantized Block 1 Output
      ↓
Input to Block 2 Quantization
```

This allows later blocks to observe errors already introduced by earlier quantization.

The paper reports that this improves quantization accuracy at negligible additional cost.

---

# 16. Experimental Models

The main large-scale experiments use two model families:

### OPT

```text
125M
350M
1.3B
2.7B
6.7B
13B
30B
66B
175B
```

### BLOOM

```text
560M
1.1B
1.7B
3B
7.1B
176B
```

The paper evaluates:

- 4-bit quantization
- 3-bit quantization
- 2-bit quantization
- Ternary quantization

---

# 17. Quantization Runtime

GPTQ scales to extremely large models.

### OPT

| Model | Quantization Time |
|---|---:|
| 13B | 20.9 min |
| 30B | 44.9 min |
| 66B | 1.6 h |
| 175B | 4.2 h |

### BLOOM

| Model | Quantization Time |
|---|---:|
| 1.7B | 2.9 min |
| 3B | 5.2 min |
| 7.1B | 10.0 min |
| 176B | 3.8 h |

All models were quantized using:

```text
1 × NVIDIA A100 80GB
```

---

# 18. 4-Bit Quantization Results

GPTQ shows very small degradation at 4-bit.

For OPT-175B on WikiText2:

```text
FP16
8.34 PPL
```

```text
RTN 4-bit
10.54 PPL
```

```text
GPTQ 4-bit
8.37 PPL
```

Therefore:

```text
FP16 → GPTQ 4-bit

8.34 → 8.37
```

corresponds to only:

```text
+0.03 perplexity
```

---

# 19. 3-Bit Quantization Results

The difference between RTN and GPTQ becomes much larger at 3 bits.

For OPT-175B:

```text
FP16
8.34
```

```text
RTN 3-bit
~7300
```

```text
GPTQ 3-bit
8.68
```

RTN effectively collapses.

GPTQ remains close to the FP16 model.

---

## BLOOM-176B

On WikiText2:

```text
FP16
8.11
```

```text
RTN 3-bit
571
```

```text
GPTQ 3-bit
8.64
```

Again, second-order error compensation becomes critical at very low bit widths.

---

# 20. Group-Wise Quantization

GPTQ can also be combined with smaller quantization groups.

Instead of using one quantization grid for an entire row:

```text
Whole Row
→ One Quantization Grid
```

the row can be divided into smaller groups:

```text
Group 1
Group 2
Group 3
...
```

Each group receives its own quantization parameters.

Examples include:

```text
Group Size = 1024
Group Size = 128
Group Size = 32
```

Smaller group sizes usually improve accuracy but require additional metadata for scale and zero-point values.

---

# 21. 175B Results with Grouping

For OPT-175B at 3-bit:

| Method | WikiText2 PPL |
|---|---:|
| FP16 | 8.34 |
| GPTQ 3-bit | 8.68 |
| GPTQ 3-bit / g1024 | 8.45 |
| GPTQ 3-bit / g128 | 8.45 |

For BLOOM-176B:

| Method | WikiText2 PPL |
|---|---:|
| FP16 | 8.11 |
| GPTQ 3-bit | 8.64 |
| GPTQ 3-bit / g1024 | 8.35 |
| GPTQ 3-bit / g128 | 8.26 |

Grouping can therefore recover additional accuracy.

---

# 22. Extreme Quantization

GPTQ also explores approximately 2-bit quantization.

For example, with group size 128:

```text
≈ 2.2 bits / weight
```

and with group size 32:

```text
≈ 2.6 bits / weight
```

the paper reports usable perplexity even for 175B models.

---

## Ternary Quantization

The authors also experiment with:

```text
{-1, 0, +1}
```

ternary weights.

With sufficiently small grouping, OPT-175B achieves:

```text
9.20 WikiText2 PPL
```

compared with:

```text
8.34
```

for FP16.

This demonstrates that very large models contain substantial redundancy.

---

# 23. Memory Reduction

The 3-bit GPTQ version of OPT-175B requires approximately:

```text
63 GB
```

for the model.

The paper additionally reports approximately:

```text
9 GB
```

for the full KV-cache history at a maximum sequence length of 2048.

Therefore:

```text
Model       ≈ 63 GB
KV Cache    ≈  9 GB
------------------
Total       ≈ 72 GB
```

This allows OPT-175B to fit into:

```text
1 × A100 80GB
```

---

# 24. GPU Requirement Reduction

For OPT-175B:

```text
FP16
→ 5 × A100 80GB
```

```text
LLM.int8()
→ 3 × A100 80GB
```

```text
GPTQ 3-bit
→ 1 × A100 80GB
```

This is one of the major practical benefits of GPTQ.

---

# 25. Why GPTQ Can Improve Inference Speed

GPTQ reduces weight precision, but current GPUs do not directly support arbitrary:

```text
FP16 Activation × INT3 Weight
```

matrix multiplication in the way required by the method.

Instead, the paper develops custom kernels that:

```text
Load Compressed Weight
        ↓
Dynamically Dequantize
        ↓
Multiply with FP16 Activation
```

At first this may seem slower because dequantization adds computation.

However, autoregressive generation at batch size 1 is primarily:

```text
Memory-Bandwidth Bound
```

rather than compute-bound.

---

# 26. Memory Movement vs. Compute

During token-by-token generation:

```text
Weight Matrix
       ×
Activation Vector
```

is close to a matrix-vector operation.

The weights must repeatedly be loaded from GPU memory.

Therefore:

```text
Less Weight Data
       ↓
Less Memory Traffic
       ↓
Lower Latency
```

Even though GPTQ introduces extra dequantization computation, the reduction in memory transfer can dominate the additional arithmetic cost.

This is why weight compression can produce real inference speedups.

---

# 27. End-to-End Inference Speedup

The paper evaluates OPT-175B generation with:

```text
Batch Size = 1
Sequence Length = 128
```

### NVIDIA A100

```text
FP16
230 ms / token
```

```text
GPTQ 3-bit
71 ms / token
```

Speedup:

```text
3.24×
```

GPU count:

```text
5 GPUs → 1 GPU
```

---

## NVIDIA A6000

```text
FP16
589 ms / token
```

```text
GPTQ 3-bit
130 ms / token
```

Speedup:

```text
4.53×
```

GPU count:

```text
8 GPUs → 2 GPUs
```

The benefit is even larger on the A6000 because its memory bandwidth is lower, making memory traffic a stronger bottleneck.

---

# 28. Important Hardware Observation

GPTQ does **not** primarily reduce the mathematical number of operations.

Its speedup comes from:

> **Reduced memory movement**

The paper explicitly distinguishes:

```text
Compute Reduction
```

from:

```text
Memory Traffic Reduction
```

GPTQ primarily achieves the latter.

Therefore:

```text
Low-Bit Weight
        ↓
Less DRAM Traffic
        ↓
Faster Memory-Bound Inference
```

---

# 29. Zero-Shot Performance

GPTQ is also evaluated on several zero-shot tasks:

- LAMBADA
- PIQA
- ARC-Easy
- ARC-Challenge
- StoryCloze

The same overall pattern appears:

```text
4-bit GPTQ
→ Usually very close to FP16

3-bit GPTQ
→ Some degradation but still usable

3-bit RTN
→ Often severe collapse
```

This shows that GPTQ's benefits are not limited to perplexity measurements.

---

# 30. GPTQ vs. LLM.int8()

LLM.int8() and GPTQ solve different problems.

## LLM.int8()

Focus:

```text
Activation Outliers
```

Approach:

```text
Normal Dimensions
→ INT8

Outlier Dimensions
→ FP16
```

Typical precision:

```text
8-bit
```

Main benefit:

```text
~2× Weight Memory Reduction
```

---

## GPTQ

Focus:

```text
Weight Quantization Error
```

Approach:

```text
Second-Order Error Compensation
```

Typical precision:

```text
3-bit / 4-bit
```

Main benefit:

```text
Much Higher Weight Compression
```

GPTQ does not require activation quantization.

---

# 31. GPTQ vs. OBQ

GPTQ preserves the core idea of OBQ:

```text
Quantization Error
       ↓
Second-Order Compensation
```

but changes the computational structure.

### OBQ

```text
Greedy Weight Ordering
+
Independent Row Processing
+
Frequent Hessian Updates
```

### GPTQ

```text
Fixed Column Ordering
+
Shared Hessian State
+
Block Processing
+
Lazy Updates
+
Cholesky Reformulation
```

This produces more than three orders of magnitude computational improvement for large models.

---

# 32. Evolution from OBC to GPTQ

The conceptual progression is:

```text
Optimal Brain Surgeon
        │
        ▼
Prune One Weight
        │
        ▼
Compensate Remaining Weights
```

then:

```text
Optimal Brain Quantization
        │
        ▼
Quantize One Weight
        │
        ▼
Compensate Remaining Weights
```

and finally:

```text
GPTQ
        │
        ▼
Keep Error Compensation
        │
        ├── Remove Expensive Greedy Ordering
        ├── Quantize Columns Together
        ├── Use Lazy Block Updates
        └── Use Cholesky Representation
        │
        ▼
Scale to 175B Models
```

---

# 33. Relation to AdaRound

AdaRound:

```text
Learn Whether Each Weight
Should Round Up or Down
        ↓
Gradient-Based Optimization
```

GPTQ:

```text
Quantize Weight
        ↓
Compute Quantization Error
        ↓
Use Second-Order Information
        ↓
Modify Remaining Weights
```

Both minimize layer reconstruction error, but use fundamentally different optimization strategies.

---

# 34. Limitations

The paper identifies several important limitations.

## No Activation Quantization

GPTQ focuses on:

```text
Weight Quantization
```

Activations remain in higher precision.

---

## No Direct Compute Reduction

The custom kernels obtain acceleration mainly through:

```text
Reduced Memory Movement
```

not through reducing the number of arithmetic operations.

---

## Generative Inference Focus

The custom kernels are designed primarily for:

```text
Low-Batch
Autoregressive Generation
```

where matrix-vector operations are memory-bound.

Large-batch workloads may instead become compute-bound.

---

## Accuracy Degradation at Extreme Precision

Although GPTQ works remarkably well at 3–4 bits, aggressive:

```text
2-bit
or
ternary
```

quantization still introduces greater degradation and may require fine-grained grouping.

---

# 35. Main Contributions

### 1. Large-Scale Second-Order PTQ

Scales accurate second-order weight quantization to models with hundreds of billions of parameters.

### 2. Efficient OBQ Reformulation

Replaces expensive greedy weight ordering with a common fixed order.

### 3. Lazy Block Updates

Improves GPU efficiency by batching weight updates.

### 4. Cholesky Reformulation

Improves numerical stability and computational efficiency.

### 5. 3–4 Bit LLM Quantization

Quantizes models such as OPT-175B and BLOOM-176B with small accuracy degradation.

### 6. Practical GPU Memory Reduction

Allows GPTQ-compressed OPT-175B to fit on a single A100 80GB GPU.

### 7. Real Inference Speedup

Achieves approximately:

```text
3.24× on A100
4.53× on A6000
```

for OPT-175B autoregressive generation.

---

# 36. Key Takeaways

1. GPTQ is a **weight-only post-training quantization** method.

2. It is directly derived from Optimal Brain Quantization.

3. Its objective is to preserve layer outputs rather than independently minimize individual weight errors.

4. Second-order information from calibration activations determines how quantization errors should be compensated.

5. The key insight is that OBQ's expensive greedy ordering is largely unnecessary for large layers.

6. Using a common column order allows the Hessian state to be shared across all weight rows.

7. Lazy block updates convert many small memory-bound operations into efficient matrix operations.

8. Cholesky decomposition improves numerical stability for extremely large models.

9. GPTQ makes accurate 3-bit and 4-bit quantization practical for 175B-scale models.

10. Its inference acceleration comes mainly from **reducing memory traffic**, not reducing FLOPs.

11. This is especially effective for batch-size-1 autoregressive inference, where weight loading is a major bottleneck.

12. GPTQ demonstrates that model compression, memory traffic, and actual hardware latency must be analyzed together.

---

# 37. Concepts to Review

- Post-Training Quantization
- Weight-Only Quantization
- Round-to-Nearest
- Layer-Wise Reconstruction
- Hessian
- Inverse Hessian
- Second-Order Approximation
- Optimal Brain Surgeon
- Optimal Brain Quantization
- Quantization Error Compensation
- Cholesky Decomposition
- Block Processing
- Lazy Update
- Calibration Dataset
- Per-Row Quantization
- Group-Wise Quantization
- Perplexity
- Matrix-Vector Multiplication
- Matrix-Matrix Multiplication
- Memory Bandwidth
- Memory-Bound Workload
- GPU Kernel
