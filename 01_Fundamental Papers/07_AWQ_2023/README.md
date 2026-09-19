# AWQ

> **AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration**

- **Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
- **Venue:** MLSys 2024, Best Paper Award
- **Initial arXiv:** 2023
- **Topic:** Low-Bit Weight-Only Quantization for Large Language Models
- **Keywords:** AWQ, Weight-Only Quantization, Activation-Aware Quantization, INT4, INT3, Salient Weights, On-Device LLM, TinyChat

---

## 1. Problem Background

Large Language Models require a large amount of memory during inference.

For example:

```text
175B Parameters
×
2 Bytes (FP16)
≈
350 GB
```

This makes deployment particularly difficult on:

- Desktop GPUs
- Laptop GPUs
- Mobile GPUs
- Edge devices
- Embedded systems

For on-device inference, reducing model memory is especially important.

---

## Why Weight-Only Quantization?

There are two major LLM quantization settings:

```text
W8A8
Weight → INT8
Activation → INT8
```

and:

```text
W4A16
Weight → INT4
Activation → FP16
```

AWQ focuses on:

> **Low-bit weight-only quantization**

such as:

```text
W4A16
W3A16
```

because autoregressive token generation at small batch sizes is often **memory-bandwidth bound**.

Reducing weight precision directly reduces the amount of weight data that must be loaded from memory.

---

# 2. Main Idea

AWQ is based on a simple but important observation:

> **Not all weights are equally important.**

Only a very small fraction of weights are particularly important for preserving model performance.

The paper calls them:

> **Salient Weights**

Experiments show that protecting only approximately:

```text
0.1% ~ 1%
```

of important weight channels can dramatically improve low-bit quantization accuracy.

---

# 3. How Should Important Weights Be Identified?

A natural idea is to identify important weights using:

```text
Weight Magnitude
```

or:

```text
Weight L2 Norm
```

However, the paper finds that this does not work particularly well.

Selecting important weights using weight magnitude performs similarly to random selection.

Instead, AWQ finds that weight importance should be determined using:

> **Activation Magnitude**

---

## Activation-Aware Importance

Consider:

```math
Y = WX
```

A weight channel processes a corresponding activation channel.

If a particular activation channel consistently has a large magnitude:

```text
Large Activation
      ↓
Corresponding Weight Channel
      ↓
Large Contribution to Output
```

then quantization error in that weight channel can have a larger effect on the final output.

Therefore:

```text
Weight Importance
       ↓
Determine Using Activation Statistics
```

even though the method itself performs **weight-only quantization**.

This is why the method is called:

> **Activation-Aware Weight Quantization**

---

# 4. Evidence: Protecting 1% Salient Weights

The paper first tests a simple mixed-precision experiment.

```text
Normal Weights
→ INT3

Salient Weight Channels
→ FP16
```

For OPT-6.7B with INT3-g128:

```text
FP16
PPL = 10.86
```

```text
RTN
PPL = 23.54
```

If approximately 1% of activation-selected salient weight channels remain FP16:

```text
PPL = 11.39
```

which is very close to the FP16 baseline.

This demonstrates that a tiny fraction of weights dominates quantization sensitivity.

---

# 5. Why Not Simply Keep Salient Weights in FP16?

The mixed-precision idea works well for accuracy:

```text
99% Low Bit
+
1% FP16
```

but it creates hardware problems.

The runtime system now needs to support:

```text
INT4 / INT3 Path
      +
FP16 Path
```

which introduces:

- Irregular memory access
- Mixed-precision kernels
- Extra control logic
- Hardware inefficiency

Therefore, AWQ asks:

> Can we protect the salient weights while still quantizing all weights to the same low-bit format?

---

# 6. Core AWQ Idea: Scale Salient Weight Channels

AWQ protects important channels by **scaling them before quantization**.

Suppose:

```text
Important Weight
w
```

is multiplied by:

```text
s > 1
```

To preserve the original linear operation, the corresponding activation is inversely scaled:

```text
Weight
w → sw

Activation
x → x/s
```

Then:

```math
(sw)\left(\frac{x}{s}\right)
=
wx
```

Therefore, the original floating-point computation remains mathematically equivalent.

---

# 7. Quantization Function

The paper considers weight quantization:

```math
Q(w)
=
\Delta
\cdot
\mathrm{Round}
\left(
\frac{w}{\Delta}
\right)
```

where:

```math
\Delta
=
\frac{
\max(|w|)
}{
2^{N-1}
}
```

and:

- $N$: number of quantization bits
- $\Delta$: quantization scale

For AWQ, typical values are:

```text
N = 3
or
N = 4
```

---

# 8. Why Scaling Reduces Quantization Error

Without scaling:

```math
Q(w)x
```

After scaling:

```math
Q(ws)\frac{x}{s}
```

The quantization error after scaling approximately contains a factor:

```math
\frac{1}{s}
```

when the quantization step size does not change significantly.

Therefore:

```text
s > 1
     ↓
Relative Quantization Error
     ↓
```

for the protected salient channel.

Conceptually:

```text
Important Weight

Before Scaling
small distance in FP scale
→ large relative quantization error

After Scaling
weight magnitude larger
→ quantization error becomes relatively smaller
```

---

# 9. Scaling Cannot Be Too Large

Increasing $s$ indefinitely is not optimal.

If the salient weight becomes too large:

```text
Large Scaled Weight
       ↓
Quantization Range Expands
       ↓
Quantization Step Δ Increases
       ↓
Errors of Other Weights Increase
```

Therefore:

```text
Too Small s
→ Salient weights insufficiently protected

Too Large s
→ Non-salient weights become harder to quantize
```

AWQ must find a balance.

---

# 10. Scaling Experiment

For OPT-6.7B with INT3-g128:

| Scaling | WikiText PPL |
|---|---:|
| s = 1 | 23.54 |
| s = 1.25 | 12.87 |
| s = 1.5 | 12.48 |
| s = 2 | **11.92** |
| s = 4 | 12.36 |

Scaling important channels significantly improves quantization accuracy.

However:

```text
s = 4
```

starts to hurt non-salient channels.

Therefore, an optimal scaling factor must be searched.

---

# 11. AWQ Optimization Objective

Instead of manually selecting one scaling value, AWQ searches for a per-input-channel scaling factor.

The objective is:

```math
s^*
=
\arg\min_s
L(s)
```

where:

```math
L(s)
=
\left\|
Q
\left(
W \cdot \mathrm{diag}(s)
\right)
\left(
\mathrm{diag}(s)^{-1}X
\right)
-
WX
\right\|
```

where:

- $W$: original FP16 weights
- $X$: calibration activations
- $Q$: low-bit weight quantization
- $s$: per-input-channel scaling factor

The goal is to minimize the difference between:

```text
Original Layer Output
```

and:

```text
Quantized + Scaled Layer Output
```

---

# 12. Activation-Aware Search Space

Directly optimizing $s$ is difficult because quantization contains a non-differentiable rounding function.

AWQ therefore defines a simple search space using activation statistics.

```math
s
=
s_X^\alpha
```

where:

```text
s_X
=
Average Activation Magnitude
per Channel
```

and:

```math
\alpha^*
=
\arg\min_\alpha
L(s_X^\alpha)
```

The paper searches:

```text
0 ≤ α ≤ 1
```

using a small grid search.

Interpretation:

```text
α = 0
→ No scaling

α → 1
→ Stronger protection of
   activation-important channels
```

---

# 13. AWQ Pipeline

The complete AWQ algorithm can be summarized as:

```text
Pretrained FP16 LLM
        │
        ▼
Small Calibration Dataset
        │
        ▼
Collect Activation Statistics
        │
        ▼
Find Salient Channels
        │
        ▼
Search Scaling Factor α
        │
        ▼
Scale Important Weight Channels
        │
        ▼
Inverse-Scale Activations
        │
        ▼
Apply Weight Clipping
        │
        ▼
INT3 / INT4 Group Quantization
        │
        ▼
Quantized LLM
```

---

# 14. No Backpropagation or Reconstruction

An important property of AWQ is that it does not require:

```text
Backpropagation
```

or:

```text
Weight Reconstruction
```

The calibration data is used mainly to measure:

```text
Average Activation Magnitude
```

This differs from methods that directly optimize quantized weights to reconstruct calibration outputs.

---

# 15. Why Avoid Reconstruction?

The paper argues that reconstruction-based methods can overfit the calibration dataset.

LLMs are general-purpose models expected to operate across:

- Different text domains
- Coding
- Mathematics
- Instruction following
- Vision-language inputs

If compression is heavily optimized toward one calibration distribution, performance may degrade when the evaluation distribution changes.

AWQ instead relies primarily on simple activation statistics.

---

# 16. Calibration Data Efficiency

AWQ requires relatively little calibration data.

For OPT-6.7B with INT3-g128, the paper reports that AWQ reaches strong performance with roughly:

```text
16 calibration sequences
```

while GPTQ requires approximately:

```text
192 sequences
```

for comparable behavior in the reported experiment.

That is roughly:

```text
10× less calibration data
```

---

# 17. Robustness to Calibration Distribution

The paper compares calibration and evaluation using:

- PubMed
- Enron Emails

When calibration and evaluation distributions differ:

```text
GPTQ
→ PPL degradation: +2.3 ~ +4.9
```

while:

```text
AWQ
→ PPL degradation: +0.5 ~ +0.6
```

in the reported OPT-6.7B INT3-g128 experiment.

This supports the claim that AWQ relies less strongly on the exact calibration distribution.

---

# 18. Quantization Setting

AWQ primarily studies:

```text
INT4
INT3
```

weight-only quantization.

Activations remain:

```text
FP16
```

Therefore:

```text
INT4 Weight + FP16 Activation
=
W4A16
```

The experiments usually use:

```text
Group Size = 128
```

denoted as:

```text
INT4-g128
INT3-g128
```

---

# 19. Group-Wise Quantization

Instead of using one quantization scale for an entire weight row, weights are divided into smaller groups.

```text
Weight Row
│
├── Group 1 → Scale 1
├── Group 2 → Scale 2
├── Group 3 → Scale 3
└── ...
```

Smaller groups usually improve accuracy because each quantization range covers fewer weights.

AWQ commonly uses:

```text
128 weights per group
```

---

# 20. LLaMA and Llama-2 Results

AWQ is evaluated on models ranging from:

```text
7B
to
70B
```

For Llama-2-70B with INT4-g128:

```text
FP16
PPL = 3.32
```

```text
RTN
PPL = 3.46
```

```text
GPTQ
PPL = 3.42
```

```text
AWQ
PPL = 3.41
```

AWQ consistently improves over RTN and generally outperforms GPTQ in the reported LLaMA / Llama-2 experiments.

---

# 21. INT3 Results

The benefit becomes more noticeable at lower precision.

For LLaMA-7B:

```text
FP16
5.68
```

```text
RTN INT3
7.01
```

```text
GPTQ INT3
8.81
```

```text
GPTQ-Reorder
6.53
```

```text
AWQ INT3
6.35
```

As quantization becomes more aggressive, identifying and protecting salient channels becomes increasingly important.

---

# 22. Mistral and Mixtral

The paper also evaluates:

- Mistral-7B
- Mixtral-8×7B

showing that AWQ can also be applied to newer architectures including:

```text
Grouped Query Attention
```

and:

```text
Mixture-of-Experts
```

models.

---

# 23. Instruction-Tuned Models

AWQ is also evaluated on Vicuna.

The paper uses GPT-4 to compare outputs from:

```text
Quantized Model
vs.
FP16 Model
```

AWQ performs better than:

- RTN
- GPTQ

for both:

```text
Vicuna-7B
Vicuna-13B
```

under INT3-g128 quantization.

This demonstrates that AWQ is not limited to base language modeling perplexity.

---

# 24. Coding and Mathematics

AWQ is evaluated on:

- MBPP for programming
- GSM8K for mathematical reasoning

Under INT4-g128, AWQ achieves performance close to the original FP16 models.

For example, Llama-2-70B on GSM8K:

```text
FP16
56.41
```

```text
GPTQ
56.03
```

```text
AWQ
56.40
```

---

# 25. Visual Language Models

A major contribution of AWQ is demonstrating low-bit quantization beyond text-only LLMs.

The paper evaluates models such as:

- OpenFlamingo
- LLaVA
- VILA

Because AWQ does not strongly reconstruct toward a text calibration dataset, it transfers well to other modalities.

---

## OpenFlamingo

For OpenFlamingo-9B under INT4-g128:

```text
FP16 32-shot CIDEr
81.70
```

```text
RTN
77.13
```

```text
GPTQ
74.98
```

```text
AWQ
80.53
```

Therefore, degradation relative to FP16 is:

```text
RTN
-4.57

GPTQ
-6.72

AWQ
-1.17
```

---

# 26. AWQ and GPTQ Are Orthogonal

AWQ and GPTQ are not fundamentally incompatible.

They address different aspects of quantization.

```text
GPTQ
→ Second-order reconstruction
   and error compensation

AWQ
→ Activation-aware
   channel scaling
```

The paper demonstrates that they can be combined.

For extreme:

```text
INT2-g64
```

quantization:

```text
AWQ + GPTQ
```

outperforms GPTQ alone.

---

# 27. Hardware Motivation

AWQ is specifically designed for **on-device autoregressive inference**.

The paper profiles Llama-2-7B on an RTX 4090.

For an example workload:

```text
Context: 200 tokens
Generation: 20 tokens
```

the measured latency is approximately:

```text
Context
10 ms

Generation
310 ms
```

Therefore, token generation dominates latency in this on-device setup.

---

# 28. Generation Is Memory-Bound

The paper performs a Roofline analysis.

RTX 4090:

```text
Peak Compute
≈ 165 TFLOPS

Memory Bandwidth
≈ 1 TB/s
```

During FP16 token generation:

```text
Arithmetic Intensity
≈ 1 FLOP/Byte
```

This is far below the compute-bound region.

Therefore:

> Token generation is strongly memory-bandwidth bound.

---

# 29. Why Weight Quantization Helps

For batch size 1 autoregressive generation:

```text
Weight Matrix
×
Activation Vector
```

resembles matrix-vector multiplication.

There is little weight reuse.

The model weights therefore need to be repeatedly loaded from memory.

The paper finds that weight traffic dominates activation traffic by a large margin.

Therefore:

```text
FP16 Weight
→ 16 bits

INT4 Weight
→ 4 bits
```

reduces weight traffic by approximately:

```text
4×
```

---

# 30. Arithmetic Intensity

With FP16 weights:

```text
Arithmetic Intensity
≈ 1 FLOP/Byte
```

With INT4 weight storage:

```text
Arithmetic Intensity
≈ 4 FLOPs/Byte
```

Therefore, low-bit weight quantization raises the theoretical performance ceiling for memory-bound generation.

---

# 31. AWQ Algorithm vs. TinyChat

It is important to distinguish:

```text
AWQ
=
Quantization Algorithm
```

from:

```text
TinyChat
=
Inference System
```

AWQ creates a low-bit model.

TinyChat turns the memory reduction into actual hardware speedup.

---

# 32. Why a Custom Inference System Is Needed

Simply storing weights in INT4 does not automatically produce faster inference.

Commodity hardware generally does not provide a direct:

```text
INT4 Weight
×
FP16 Activation
```

instruction matching the required computation.

Therefore:

```text
INT4 Weight
     ↓
Dequantization
     ↓
FP16 Weight
     ↓
Matrix Computation
```

must occur efficiently.

TinyChat is designed to minimize this overhead.

---

# 33. On-the-Fly Weight Dequantization

TinyChat fuses:

```text
Weight Dequantization
       +
Matrix Multiplication
```

into one kernel.

Instead of:

```text
INT4 Weight
      ↓
Dequantize
      ↓
Write FP16 Weight to DRAM
      ↓
Read Weight Again
      ↓
MatMul
```

TinyChat performs:

```text
Load INT4
      ↓
Dequantize
      ↓
Immediately Compute
```

without writing the intermediate FP16 weight back to DRAM.

This reduces memory traffic.

---

# 34. SIMD-Aware Weight Packing

4-bit values must be packed efficiently into byte-oriented hardware.

TinyChat rearranges and packs the weights offline according to the target SIMD architecture.

This allows multiple weights to be unpacked using vectorized:

- Shift
- Bitwise AND

instructions.

The paper reports up to roughly:

```text
1.2×
```

additional speedup from SIMD-aware packing on ARM CPUs.

---

# 35. Kernel Fusion

TinyChat also applies extensive kernel fusion.

Examples include:

```text
LayerNorm Operations
→ One Kernel
```

```text
Q + K + V Projection
→ Fused Kernel
```

and:

```text
KV Cache Update
→ Fused into Attention Kernel
```

This reduces:

- Intermediate DRAM traffic
- Kernel launch overhead

The paper notes that individual FP16 kernels can take around:

```text
0.01 ms
```

on RTX 4090, which is comparable to GPU kernel launch overhead.

Therefore, reducing the number of kernel calls can directly improve performance.

---

# 36. TinyChat Performance

TinyChat converts the theoretical 4× weight compression into substantial real-world acceleration.

Compared with HuggingFace FP16 inference, the paper reports average speedups around:

```text
3.2× ~ 3.3×
```

on desktop and mobile GPUs.

Individual cases reach:

```text
3.9×
```

speedup.

---

# 37. RTX 4090 Results

For several LLM families:

- Llama-2
- MPT
- Falcon

TinyChat achieves approximately:

```text
2.7× ~ 3.9×
```

speedup compared with the HuggingFace FP16 implementation.

For Llama-2-7B, the authors first optimize the FP16 implementation and then obtain an additional:

```text
~3.1×
```

speedup from quantized linear kernels.

---

# 38. Mobile GPU Deployment

TinyChat also targets mobile devices such as:

```text
NVIDIA Jetson Orin
```

The paper reports:

```text
Up to ~3.5×
```

speedup relative to the HuggingFace FP16 implementation.

AWQ also enables deployment of:

```text
Llama-2-70B
```

on a single Jetson Orin with:

```text
64 GB Memory
```

---

# 39. Laptop GPU Deployment

On an RTX 4070 laptop GPU with:

```text
8 GB Memory
```

TinyChat can run:

```text
Llama-2-13B
```

at approximately:

```text
33 tokens / second
```

while the FP16 implementation cannot fit even some smaller models in the available memory.

---

# 40. Raspberry Pi Deployment

TinyChat also supports:

```text
Raspberry Pi 4B
```

The system can run models up to approximately:

```text
7B parameters
```

with reported performance around:

```text
0.7 token / second
```

demonstrating deployment on highly resource-constrained devices.

---

# 41. AWQ vs. GPTQ

Both are:

```text
Low-Bit
Weight-Only
Post-Training Quantization
```

but their approaches differ.

## GPTQ

Core idea:

```text
Quantize Weight
      ↓
Estimate Error
      ↓
Use Second-Order Information
      ↓
Modify Remaining Weights
```

Uses:

- Hessian information
- Reconstruction
- Error compensation

---

## AWQ

Core idea:

```text
Measure Activation Magnitude
       ↓
Identify Important Channels
       ↓
Scale Weight Channels
       ↓
Quantize
```

Uses:

- Activation statistics
- Equivalent scaling
- Small grid search

No backpropagation or second-order reconstruction is required.

---

# 42. AWQ vs. SmoothQuant

Both methods use equivalent per-channel scaling, but their objectives are different.

## SmoothQuant

Problem:

```text
Activation Outliers
make W8A8 difficult
```

Transformation:

```text
Activation Magnitude ↓
Weight Magnitude ↑
```

Goal:

```text
Make both
Weight and Activation
INT8-friendly
```

---

## AWQ

Problem:

```text
Some Weight Channels
are much more important
```

Transformation:

```text
Important Weight Magnitude ↑
Corresponding Activation ↓
```

Goal:

```text
Reduce relative quantization
error of important weights
```

Therefore:

```text
SmoothQuant
→ Smooth activations

AWQ
→ Protect weights
```

even though both use mathematically equivalent scaling transformations.

---

# 43. Evolution of Scaling-Based Quantization

A useful conceptual progression is:

```text
LLM.int8()
     ↓
Identify Activation Outliers

SmoothQuant
     ↓
Move Activation Difficulty
into Weights

AWQ
     ↓
Use Activation Statistics
to Determine
Which Weights Need Protection
```

SmoothQuant and AWQ both demonstrate that:

> Equivalent transformations can change quantization difficulty without changing the original model function.

---

# 44. Main Contributions

### 1. Salient Weight Observation

Only a small fraction of LLM weight channels are highly sensitive to quantization.

### 2. Activation-Aware Importance

Important weights are better identified using activation statistics than weight magnitude.

### 3. Salient Weight Protection

AWQ protects important weights using per-channel scaling instead of mixed precision.

### 4. No Backpropagation

No QAT, gradient optimization, or full reconstruction is required.

### 5. Better Generalization

The method is less dependent on the exact calibration distribution.

### 6. Weight-Only Low-Bit Quantization

Supports practical INT3 / INT4 LLM compression.

### 7. TinyChat

Introduces an inference framework for translating weight compression into real edge-device speedup.

### 8. Multimodal Quantization

Demonstrates strong quantization performance on visual-language models.

---

# 45. Limitations

## Activations Remain FP16

AWQ mainly targets:

```text
W4A16
or
W3A16
```

Therefore, activation memory and KV-cache precision are not directly reduced by the basic AWQ algorithm.

---

## Specialized Kernels Are Needed for Speedup

INT4 model size reduction alone does not guarantee runtime acceleration.

Efficient inference requires:

- Fused dequantization
- Custom kernels
- Weight packing
- Kernel fusion

---

## Best for Memory-Bound Workloads

AWQ is particularly suitable for:

```text
Batch Size = 1
Autoregressive Generation
```

where weight loading dominates memory traffic.

The relative advantage may differ for highly compute-bound large-batch workloads.

---

## Calibration Is Still Required

Although AWQ uses only simple activation statistics, it still requires a small calibration dataset.

---

# 46. Key Takeaways

1. AWQ is a **low-bit weight-only PTQ method** for LLMs.

2. Not all LLM weights are equally important.

3. Only approximately **0.1–1% of salient weight channels** can dominate quantization sensitivity.

4. Weight magnitude alone is not a good indicator of this importance.

5. Activation magnitude provides a much stronger importance signal.

6. Keeping salient weights in FP16 works well but is hardware-inefficient.

7. AWQ instead scales important weight channels before quantization.

8. The corresponding activations are inversely scaled, so the original linear operation remains mathematically equivalent.

9. Scaling reduces the relative quantization error of salient weights.

10. AWQ searches for the scaling strength using simple activation statistics rather than backpropagation.

11. AWQ does not rely on Hessian reconstruction like GPTQ.

12. Its limited dependence on calibration data improves robustness across domains and modalities.

13. AWQ works well for LLMs, instruction-tuned models, coding/math workloads, and visual-language models.

14. Weight-only quantization is particularly effective for batch-size-1 autoregressive generation because inference is strongly memory-bandwidth bound.

15. TinyChat converts AWQ's theoretical weight-memory reduction into approximately **3× real inference speedup** on multiple edge platforms.

---

# 47. Concepts to Review

- Post-Training Quantization
- Weight-Only Quantization
- W4A16
- W3A16
- Group-Wise Quantization
- Salient Weights
- Activation Statistics
- Activation Magnitude
- Quantization Error
- Quantization Scale
- Per-Channel Scaling
- Equivalent Transformation
- Calibration Dataset
- Round-to-Nearest
- Weight Clipping
- Grid Search
- Arithmetic Intensity
- Roofline Model
- Memory-Bound Workload
- Matrix-Vector Multiplication
- Memory Bandwidth
- On-the-Fly Dequantization
- SIMD
- Weight Packing
- Kernel Fusion
- Kernel Launch Overhead
- Edge AI
