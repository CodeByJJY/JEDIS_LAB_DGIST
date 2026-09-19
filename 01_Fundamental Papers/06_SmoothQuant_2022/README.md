# SmoothQuant

> **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**

- **Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **Venue:** ICML 2023
- **Topic:** Post-Training Quantization for Large Language Models
- **Keywords:** SmoothQuant, W8A8, Activation Quantization, Activation Outliers, INT8 GEMM, PTQ

---

## 1. Problem Background

Large Language Models require enormous amounts of memory and computation during inference.

For example, a 175B-parameter model stored in FP16 requires roughly:

```text
175B Parameters
×
2 Bytes
≈
350 GB
```

Therefore, large models require multiple high-end GPUs even for inference.

Quantization can reduce:

- Model size
- Memory bandwidth
- GPU memory usage
- Inference latency

However, quantizing LLMs is significantly more difficult than quantizing CNNs or smaller Transformers.

The major reason is:

> **Activation outliers**

---

# 2. Why W8A8 Quantization?

Quantizing only weights from FP16 to INT8 reduces weight storage.

```text
FP16 Weights
    ↓
INT8 Weights
    ↓
~2× Smaller
```

However, to efficiently use hardware INT8 matrix multiplication kernels, both operands should be INT8.

Therefore, the target of SmoothQuant is:

```text
Weight      → INT8
Activation  → INT8
```

or:

```text
W8A8
```

This allows compute-intensive operations such as:

- GEMM in linear layers
- BMM in attention

to use hardware-efficient INT8 arithmetic.

---

# 3. Quantization Basics

For symmetric uniform quantization:

```math
\overline{X}^{INT8}
=
\mathrm{round}
\left(
\frac{X^{FP16}}{\Delta}
\right)
```

where the quantization step size is:

```math
\Delta
=
\frac{
\max(|X|)
}{
2^{N-1}-1
}
```

For INT8:

```text
N = 8
```

The largest absolute value determines the quantization range.

This creates a major problem when activation outliers exist.

---

# 4. Activation Outlier Problem

Suppose most activation values are relatively small:

```text
1
-2
3
1
-1
```

but one value is extremely large:

```text
100
```

The quantization range must include the outlier.

Therefore:

```text
Large Outlier
     ↓
Large Quantization Range
     ↓
Large Quantization Step Size
     ↓
Few Effective Quantization Levels
for Normal Values
     ↓
Large Quantization Error
```

This is why naive W8A8 quantization fails badly for large LLMs.

---

# 5. Key Observations

SmoothQuant makes three important observations about LLM quantization.

## 5.1 Activations Are Harder to Quantize Than Weights

LLM weights usually have relatively smooth and uniform distributions.

```text
Weights
→ Easy to Quantize
```

Activations contain large outliers.

```text
Activations
→ Hard to Quantize
```

The paper notes that weights can often tolerate INT8 or even INT4 quantization without large accuracy degradation.

---

## 5.2 Activation Outliers Dominate the Quantization Range

The activation outliers can be roughly:

```text
~100×
```

larger than typical activation values.

As a result, the large outliers determine the quantization scale while most ordinary values use only a very small part of the INT8 range.

---

## 5.3 Outliers Persist in Fixed Channels

The activation outliers are not randomly distributed.

Instead, they tend to repeatedly occur in the same feature channels across different tokens.

Conceptually:

```text
Token 1   [small small OUTLIER small]
Token 2   [small small OUTLIER small]
Token 3   [small small OUTLIER small]
Token 4   [small small OUTLIER small]
                     ↑
             Same Channel
```

Therefore:

```text
Variance across channels
→ Large

Variance of one channel across tokens
→ Relatively small
```

This observation motivates SmoothQuant.

---

# 6. Why Not Per-Channel Activation Quantization?

A natural solution would be:

```text
Each Activation Channel
       ↓
Own Quantization Scale
```

This greatly reduces activation quantization error.

The paper shows that simulated per-channel activation quantization can preserve FP16 accuracy.

However, this is difficult to implement efficiently using standard INT8 GEMM kernels.

For matrix multiplication:

```math
Y = XW
```

where:

```math
X \in \mathbb{R}^{T \times C_i}
```

and:

```math
W \in \mathbb{R}^{C_i \times C_o}
```

the input-channel dimension $C_i$ is the **inner dimension** of the matrix multiplication.

Efficient hardware GEMM kernels cannot easily apply independent scaling along this dimension without adding low-throughput operations inside the matrix multiplication.

Therefore:

```text
Per-Channel Activation Quantization
→ Accurate
→ Hardware-Unfriendly
```

SmoothQuant tries to obtain the same benefit without requiring such a kernel.

---

# 7. Main Idea

The central idea of SmoothQuant is:

> **Migrate quantization difficulty from activations to weights.**

Originally:

```text
Activation
→ Hard to Quantize

Weight
→ Very Easy to Quantize
```

SmoothQuant transforms them into:

```text
Smoothed Activation
→ Easy to Quantize

Adjusted Weight
→ Still Easy to Quantize
```

The key idea is to exploit the fact that weights have much more quantization tolerance than activations.

---

# 8. Mathematically Equivalent Transformation

Consider a linear layer:

```math
Y = XW
```

SmoothQuant introduces a per-channel smoothing factor:

```math
s \in \mathbb{R}^{C_i}
```

and rewrites the linear operation as:

```math
Y
=
\left(
X \mathrm{diag}(s)^{-1}
\right)
\left(
\mathrm{diag}(s)W
\right)
```

Define:

```math
\hat{X}
=
X\mathrm{diag}(s)^{-1}
```

and:

```math
\hat{W}
=
\mathrm{diag}(s)W
```

Then:

```math
Y
=
\hat{X}\hat{W}
=
XW
```

Therefore, the transformation is mathematically equivalent.

The original FP16 output does not change.

---

# 9. What the Transformation Does

For a problematic activation channel:

```text
Large Activation
```

SmoothQuant divides it by a scaling factor:

```text
Activation
÷
s
```

At the same time, the corresponding weight channel is multiplied by the same factor:

```text
Weight
×
s
```

Therefore:

```text
Activation Magnitude ↓
Weight Magnitude     ↑
```

while:

```text
Output
=
Same
```

Conceptually:

```text
Before

Activation  ███████████████████
Weight      ███


After SmoothQuant

Activation  ███████
Weight      ███████
```

The quantization difficulty becomes more balanced.

---

# 10. Smoothing Factor

The smoothing factor for input channel $j$ is defined as:

```math
s_j
=
\frac{
\max(|X_j|)^{\alpha}
}{
\max(|W_j|)^{1-\alpha}
}
```

where:

- $X_j$: activation values of channel $j$
- $W_j$: corresponding weight channel
- $\alpha$: migration strength

The hyperparameter $\alpha$ controls how much quantization difficulty is moved from activations to weights.

---

# 11. Migration Strength

The role of $\alpha$ can be interpreted as:

```text
Small α
→ Keep more difficulty in activations

Large α
→ Move more difficulty into weights
```

If $\alpha$ is too small:

```text
Activation
→ Still Hard to Quantize
```

If $\alpha$ is too large:

```text
Weight
→ Becomes Hard to Quantize
```

Therefore, a balance is required.

---

## Typical Values

For OPT and BLOOM:

```math
\alpha = 0.5
```

works well.

For models with more severe activation outliers such as GLM-130B:

```math
\alpha = 0.75
```

is used.

For the OPT-175B ablation, the paper finds a sweet spot roughly around:

```text
0.4 ~ 0.6
```

---

# 12. Why Smoothing Can Be Offline

A major advantage of SmoothQuant is that the smoothing does not need to be executed during inference.

Suppose the activation $X$ is produced by a previous layer.

The factor:

```math
\mathrm{diag}(s)^{-1}
```

can often be fused into the parameters of the previous operator.

Therefore:

```text
Calibration
     ↓
Compute Smoothing Factors
     ↓
Modify Model Parameters Offline
     ↓
Inference
```

At runtime:

```text
No Additional Smoothing Kernel
```

is required in the common case.

This is crucial for hardware efficiency.

---

# 13. SmoothQuant Workflow

The complete procedure can be summarized as:

```text
Pretrained FP16 LLM
        │
        ▼
Calibration Dataset
        │
        ▼
Collect Per-Channel
Activation Statistics
        │
        ▼
Calculate Smoothing Factor s
        │
        ▼
Activation ÷ s
Weight × s
        │
        ▼
Fuse Scaling Offline
        │
        ▼
Quantize Weight → INT8
Quantize Activation → INT8
        │
        ▼
INT8 GEMM / BMM
```

No retraining is required.

---

# 14. Activation Distribution Before and After Smoothing

Before SmoothQuant:

```text
Activation Channels

small
small
small
HUGE OUTLIER
small
small
```

After SmoothQuant:

```text
Activation Channels

medium
medium
medium
medium
medium
medium
```

The large channel-wise variation is substantially reduced.

At the same time, some difficulty is transferred into weights.

However, because the original weight distribution is easy to quantize, the transformed weights remain manageable.

---

# 15. Transformer Precision Mapping

SmoothQuant does not quantize every operation in a Transformer to INT8.

Compute-intensive operators use INT8:

```text
Linear Layers
→ INT8

Attention BMM
→ INT8
```

Lightweight operations remain FP16:

```text
LayerNorm
Softmax
ReLU
Residual Operations
```

Conceptually:

```text
          FP16
        LayerNorm
           │
           ▼
       INT8 Q/K/V
           │
           ▼
        INT8 BMM
           │
           ▼
       FP16 Softmax
           │
           ▼
        INT8 BMM
           │
           ▼
     INT8 Projection
```

The goal is to quantize the operations that dominate computation while avoiding unnecessary complexity for lightweight operators.

---

# 16. SmoothQuant Quantization Levels

The paper provides three SmoothQuant configurations.

| Method | Weight | Activation |
|---|---|---|
| SmoothQuant-O1 | Per-tensor | Per-token dynamic |
| SmoothQuant-O2 | Per-tensor | Per-tensor dynamic |
| SmoothQuant-O3 | Per-tensor | Per-tensor static |

The efficiency increases from:

```text
O1
↓
O2
↓
O3
```

O3 is the most hardware-efficient configuration.

---

# 17. Why Static Quantization Is Faster

Dynamic quantization requires activation statistics to be computed at runtime.

```text
Runtime Activation
       ↓
Calculate Min / Max
       ↓
Determine Scale
       ↓
Quantize
```

Static quantization performs this calibration beforehand.

```text
Calibration
     ↓
Fixed Scale
     ↓
Runtime Quantization
```

Therefore:

```text
Static Quantization
→ Less Runtime Overhead
→ Lower Latency
```

This is why SmoothQuant-O3 is generally the fastest configuration.

---

# 18. Calibration

SmoothQuant obtains activation statistics using:

```text
512 random sentences
```

from the Pile validation set.

These samples are used to determine:

- Per-channel activation magnitude
- Smoothing factors
- Static quantization scales

The same calibrated model is then evaluated on downstream tasks.

Therefore:

```text
No Task-Specific Fine-Tuning
```

is required.

---

# 19. OPT-175B Results

The paper evaluates several PTQ methods on OPT-175B.

### Average Zero-Shot Accuracy

```text
FP16
66.9%
```

```text
Naive W8A8
35.5%
```

```text
ZeroQuant
35.8%
```

```text
LLM.int8()
66.7%
```

```text
SmoothQuant-O3
66.8%
```

Therefore, SmoothQuant achieves essentially FP16-level accuracy while using fully INT8 compute-intensive operations.

---

# 20. WikiText Perplexity

For OPT-175B:

| Method | WikiText PPL |
|---|---:|
| FP16 | 10.99 |
| W8A8 | 93080 |
| ZeroQuant | 84648 |
| LLM.int8() | 11.10 |
| SmoothQuant-O1 | 11.11 |
| SmoothQuant-O2 | 11.14 |
| SmoothQuant-O3 | 11.17 |

Naive activation quantization completely destroys the model.

SmoothQuant maintains perplexity close to the FP16 baseline.

---

# 21. Results Across LLM Families

SmoothQuant is tested on multiple model families.

### Large Models

- OPT-175B
- BLOOM-176B
- GLM-130B
- MT-NLG 530B

It also works on:

- LLaMA
- Llama-2
- Falcon
- Mistral
- Mixtral

This demonstrates that the smoothing principle is not specific to one Transformer architecture.

---

# 22. LLaMA Results

SmoothQuant also enables W8A8 quantization of LLaMA.

Example WikiText-2 perplexity:

| Model | FP16 | SmoothQuant W8A8 |
|---|---:|---:|
| LLaMA-7B | 11.51 | 11.56 |
| LLaMA-13B | 10.05 | 10.08 |
| LLaMA-30B | 7.53 | 7.56 |
| LLaMA-65B | 6.17 | 6.20 |

The accuracy degradation is negligible.

---

# 23. PyTorch Performance

The paper implements SmoothQuant using INT8 CUTLASS GEMM kernels.

For the context stage, the PyTorch implementation achieves up to:

```text
1.51× Speedup
```

and approximately:

```text
1.96× Memory Saving
```

compared with FP16.

The benefit becomes larger as model size increases.

---

# 24. FasterTransformer Performance

SmoothQuant is also integrated into NVIDIA FasterTransformer.

Despite FasterTransformer already being highly optimized, SmoothQuant achieves up to:

```text
1.56× Speedup
```

for smaller models such as OPT-13B and OPT-30B.

For larger models:

```text
OPT-66B
FP16       → 2 GPUs
SmoothQuant → 1 GPU
```

and:

```text
OPT-175B
FP16       → 8 GPUs
SmoothQuant → 4 GPUs
```

while maintaining similar or better latency.

---

# 25. Decoding-Stage Performance

SmoothQuant also improves autoregressive decoding.

For OPT-30B:

```text
Up to 1.42× speedup
```

is reported.

For example:

```text
Batch Size = 16
Sequence Length = 512
```

FP16:

```text
2488 ms
```

SmoothQuant:

```text
1753 ms
```

Speedup:

```text
1.42×
```

---

# 26. Memory Reduction

Because both weights and major activations use INT8:

```text
FP16
→ 16 bits
```

```text
INT8
→ 8 bits
```

SmoothQuant can almost halve the inference memory footprint.

The experiments report memory reductions close to:

```text
2×
```

for several large models.

---

# 27. Scaling to MT-NLG 530B

SmoothQuant is also applied to the:

```text
MT-NLG 530B
```

model.

Accuracy:

```text
FP16 Average
73.1%
```

```text
INT8 Average
73.1%
```

The W8A8 model therefore maintains essentially the same reported accuracy.

---

## GPU Requirement

FP16 inference requires:

```text
16 × A100 80GB
```

SmoothQuant reduces this to:

```text
8 × A100 80GB
```

allowing the 530B model to be served within a single 8-GPU node.

---

## Example Memory Usage

At sequence length 512:

```text
FP16
1068 GB
```

versus:

```text
INT8
545 GB
```

while latency remains approximately the same.

---

# 28. SmoothQuant vs. LLM.int8()

Both methods address the same core problem:

> Large activation outliers make LLM INT8 quantization difficult.

However, they solve it differently.

---

## LLM.int8()

```text
Normal Features
→ INT8

Outlier Features
→ FP16
```

Advantages:

- High accuracy

Disadvantages:

- Mixed-precision execution
- Outlier extraction
- Separate FP16 computation
- Runtime overhead

---

## SmoothQuant

```text
Activation Outliers
       ↓
Move Difficulty into Weights
       ↓
Smooth Activations
       ↓
All Heavy MatMuls Use INT8
```

Advantages:

- No mixed-precision decomposition
- Standard INT8 GEMM
- Hardware-friendly
- Faster inference

The paper reports that LLM.int8() is often slower than FP16 in their implementation, whereas SmoothQuant consistently reduces latency.

---

# 29. SmoothQuant vs. GPTQ

GPTQ and SmoothQuant target different inference settings.

## GPTQ

Primarily:

```text
Weight-Only Quantization
```

for example:

```text
W4A16
W3A16
```

The weight is stored at low precision and dequantized during inference.

This is especially effective for:

```text
Low-Batch
Autoregressive Decoding
```

where matrix-vector computation is highly memory-bandwidth-bound.

---

## SmoothQuant

Targets:

```text
W8A8
```

Both activations and weights use INT8 for major matrix multiplications.

This allows:

```text
INT8 Tensor-Core GEMM
```

and is especially attractive for:

- Context stage
- Batched inference
- Compute-intensive GEMM workloads

---

# 30. Weight-Only vs. Weight-Activation Quantization

The paper emphasizes an important hardware distinction.

### Weight-Only Quantization

```text
Low-Bit Weight
+
FP16 Activation
```

Main advantage:

```text
Reduced Weight Memory Traffic
```

This can be effective when inference is:

```text
Memory-Bound
```

---

### W8A8 Quantization

```text
INT8 Weight
+
INT8 Activation
```

Main advantage:

```text
Reduced Memory
+
Hardware INT8 Compute
```

This can be more advantageous for:

```text
Large Batch
or
Context Processing
```

where matrix multiplication is more compute-intensive.

---

# 31. Why SmoothQuant Is Hardware-Friendly

SmoothQuant is designed so that the heavy runtime operations can use standard INT8 kernels.

The smoothing transformation occurs:

```text
Offline
```

Therefore the inference path remains approximately:

```text
INT8 Activation
      ×
INT8 Weight
      ↓
INT8 GEMM
```

rather than:

```text
INT8 Path
     +
Special FP16 Outlier Path
```

This is one of the main differences from LLM.int8().

---

# 32. Main Contributions

### 1. Activation Outlier Analysis

Recognizes that LLM activation outliers are persistent across fixed feature channels.

### 2. Quantization Difficulty Migration

Moves part of the activation quantization difficulty into weights.

### 3. Mathematically Equivalent Transformation

Uses per-channel scaling without changing the original linear transformation.

### 4. Training-Free PTQ

Requires calibration but no retraining or fine-tuning.

### 5. Hardware-Efficient W8A8

Allows compute-intensive Transformer operations to use standard INT8 kernels.

### 6. Large-Scale Validation

Demonstrates W8A8 quantization up to:

```text
530B Parameters
```

### 7. Real Inference Acceleration

Achieves up to:

```text
1.56× Speedup
```

and nearly:

```text
2× Memory Reduction
```

---

# 33. Important Observations

## Activation Quantization Is the Main Challenge

For large LLMs:

```text
Weight Quantization
→ Relatively Easy

Activation Quantization
→ Difficult
```

Therefore, solving activation outliers is central to efficient W8A8 inference.

---

## Outliers Are Structured

Outliers persist in particular channels.

This means their structure can be exploited with:

```text
Per-Channel Transformation
```

performed offline.

---

## Do Not Remove the Outliers

SmoothQuant does not:

```text
Clip Outliers
or
Discard Outliers
```

Instead, it preserves their information while changing where the large magnitude appears.

---

## Quantization Difficulty Can Be Redistributed

A key conceptual insight is:

```text
Activation Difficulty
       +
Weight Difficulty
```

does not need to remain fixed in its original allocation.

Using an equivalent transformation, the difficulty can be redistributed to a representation that is easier for hardware to quantize.

---

# 34. Limitations

SmoothQuant still has several limitations.

### INT8 Focus

The main method targets:

```text
W8A8
```

rather than more aggressive 3-bit or 4-bit inference.

---

### Calibration Required

Activation statistics must be estimated using calibration data.

---

### Migration Strength Selection

Different models may require different values of $\alpha$.

For example:

```text
OPT / BLOOM
≈ 0.5
```

while:

```text
GLM-130B
≈ 0.75
```

---

### Static Statistics May Not Perfectly Match Runtime Data

The most efficient O3 configuration uses static activation scales.

If runtime activation distributions differ from calibration statistics, small accuracy degradation can occur.

---

# 35. Connection to Previous Papers

## LLM.int8()

LLM.int8() identifies the key problem:

```text
Activation Outliers
```

and solves it using:

```text
Mixed Precision
```

SmoothQuant takes the next step:

```text
LLM.int8()
     ↓
Activation Outliers Identified
     ↓
SmoothQuant
     ↓
Transform Outliers
Instead of Separating Them
```

---

## GPTQ

GPTQ focuses primarily on:

```text
Aggressive Weight Quantization
```

using second-order error compensation.

SmoothQuant instead focuses on:

```text
Weight + Activation Quantization
```

using activation smoothing.

Therefore:

```text
GPTQ
→ Improve Weight Quantization

SmoothQuant
→ Make Activation Quantization Practical
```

The two directions are largely orthogonal.

---

# 36. Connection to Later Rotation-Based Quantization

SmoothQuant introduces an important general idea:

> Change the representation while preserving the mathematical function of the model.

SmoothQuant uses:

```text
Per-Channel Scaling
```

Later approaches such as QuaRot and SpinQuant use:

```text
Rotation
```

to redistribute activation outliers.

Conceptually:

```text
LLM.int8()
→ Separate Outliers

SmoothQuant
→ Scale Outliers

QuaRot
→ Rotate Outliers

SpinQuant
→ Learn the Rotation
```

This provides a useful progression for understanding later LLM quantization papers.

---

# 37. Key Takeaways

1. SmoothQuant is a **training-free W8A8 post-training quantization method** for LLMs.

2. The main obstacle to W8A8 quantization is activation outliers, not weight quantization.

3. LLM activation outliers tend to persist in a small number of fixed feature channels.

4. Per-channel activation quantization would solve much of the problem but is difficult to execute efficiently using standard INT8 GEMM kernels.

5. SmoothQuant instead performs an offline per-channel transformation.

6. Activations are divided by a smoothing factor while corresponding weights are multiplied by the same factor.

7. The linear layer output remains mathematically unchanged.

8. The hyperparameter $\alpha$ controls how much quantization difficulty is moved from activations to weights.

9. The transformation can usually be fused offline, introducing little or no runtime smoothing overhead.

10. SmoothQuant allows the major matrix multiplications in LLMs to use standard INT8 arithmetic.

11. SmoothQuant preserves accuracy on models with hundreds of billions of parameters.

12. It achieves up to approximately **1.56× inference speedup** and nearly **2× memory reduction** in the reported experiments.

13. SmoothQuant demonstrates that efficient quantization depends not only on reducing bit width, but also on transforming data distributions into a hardware-friendly form.

---

# 38. Concepts to Review

- Post-Training Quantization
- W8A8 Quantization
- Activation Quantization
- Weight Quantization
- Activation Outlier
- Uniform Quantization
- Quantization Range
- Quantization Step Size
- Per-Tensor Quantization
- Per-Token Quantization
- Per-Channel Quantization
- Static Quantization
- Dynamic Quantization
- Calibration
- GEMM
- BMM
- INT8 Tensor Core
- Matrix Multiplication
- Memory Bandwidth
- Compute-Bound Workload
- Memory-Bound Workload
- Mixed Precision
- Equivalent Transformation
