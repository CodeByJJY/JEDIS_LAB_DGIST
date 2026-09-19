# Variance-Based Pruning

> **Variance-Based Pruning for Accelerating and Compressing Trained Networks**

- **Authors:** Uranik Berisha, Jens Mehnert, Alexandru Paul Condurache
- **Initial arXiv:** 2025
- **Current Version:** arXiv:2507.12988v2, April 2026
- **Topic:** Structured Neural Network Pruning
- **Keywords:** Structured Pruning, Activation Variance, Mean-Shift Compensation, MLP Pruning, Vision Transformer, Model Compression

---

## 1. Problem Background

Large pretrained models provide strong performance, but their deployment remains expensive in terms of:

- Memory
- Computation
- Inference latency

One option is to reuse existing pretrained models instead of training new models from scratch.

However:

```text
Reuse Pretrained Model
        ↓
No Training Cost
        ↓
But Still
Large Model Size
+
High Inference Cost
```

Model pruning can reduce these costs.

---

# 2. Unstructured vs. Structured Pruning

## Unstructured Pruning

Unstructured pruning removes individual weights.

```text
Dense Matrix
     ↓
Individual Weights Removed
     ↓
Sparse Matrix
```

Advantages:

- Can preserve accuracy well

Disadvantages:

- Irregular sparsity
- Difficult to exploit efficiently on standard hardware
- Theoretical FLOP savings may not translate into real speedup

Modern hardware is highly optimized for:

```text
Dense Matrix Multiplication
```

rather than irregular sparse computation.

---

## Structured Pruning

Structured pruning removes complete structures such as:

- Neurons
- Channels
- Filters
- Layers

Conceptually:

```text
Large Dense Matrix
        ↓
Remove Entire Neurons
        ↓
Smaller Dense Matrix
```

Advantages:

- Direct parameter reduction
- Direct MAC reduction
- Standard dense kernels remain usable
- Real hardware speedup is easier to obtain

However, aggressive structural changes often cause large accuracy degradation.

---

# 3. Main Problem

Previous structured pruning methods often require significant additional training.

For example, the paper notes that NViT performs iterative structured pruning and requires:

```text
50 Epochs of Pruning
+
300 Epochs of Retraining
```

in its original procedure.

This reduces the benefit of starting from an already trained model.

The goal of this paper is therefore:

> **Perform one-shot structured pruning on a trained network while retaining enough accuracy that only minimal fine-tuning is required.**

---

# 4. Main Idea

The paper proposes:

> **Variance-Based Pruning (VBP)**

VBP consists of three steps:

```text
1. Activation Statistics Computation
             ↓
2. Variance-Based Pruning
             ↓
3. Mean-Shift Compensation
```

The central idea is:

```text
Low Activation Variance
        ↓
Neuron behaves almost like a constant
        ↓
Replace its activation by its mean
        ↓
Move that mean contribution into next-layer bias
        ↓
Remove the neuron structurally
```

---

# 5. Where Does VBP Prune?

VBP only prunes the hidden neurons of:

```text
MLP Blocks
```

The input and output dimensions of the complete MLP are kept unchanged.

For a two-layer MLP:

```text
Input
  ↓
W1
  ↓
Hidden Neurons
  ↓
Activation
  ↓
W2
  ↓
Output
```

VBP reduces:

```text
Hidden Dimension
```

while preserving:

```text
Input Dimension
Output Dimension
```

This makes the pruned block compatible with the rest of the pretrained network.

---

# 6. MLP Formulation

Consider an MLP:

```math
h = \sigma(W_1 x + b_1)
```

```math
y = W_2 h + b_2
```

where:

- $x$: MLP input
- $h$: hidden activation
- $y$: MLP output
- $W_1$: first linear layer
- $W_2$: second linear layer
- $\sigma$: nonlinear activation

The hidden dimension is:

```math
D_{hid}
```

VBP decides which hidden neurons can be removed.

---

# 7. Step 1: Activation Statistics Computation

For each hidden neuron, the method collects:

```text
Mean
+
Variance
```

of its **post-activation** output.

For hidden neuron $i$:

```math
\mu_i = E[h_i]
```

and:

```math
\sigma_i^2 = E[(h_i-\mu_i)^2]
```

These statistics are collected over calibration data.

---

# 8. Welford's Algorithm

VBP computes activation statistics using:

> **Welford's Algorithm**

This is an online and numerically stable method for computing mean and variance.

After receiving sample $j$:

```math
\mu^{(j)}
=
\frac{j-1}{j}\mu^{(j-1)}
+
\frac{1}{j}h^{(j)}
```

The running second moment is updated as:

```math
m_2^{(j)}
=
m_2^{(j-1)}
+
(h^{(j)}-\mu^{(j-1)})
\odot
(h^{(j)}-\mu^{(j)})
```

Finally:

```math
\sigma^2
=
\frac{m_2^{(N)}}{N-1}
```

The use of online statistics allows the method to process as much available data as desired without storing all activations.

---

# 9. Step 2: Variance-Based Pruning

After the activation variance of every hidden neuron has been computed, neurons are ranked by:

```math
\sigma_i^2
```

The neurons with the **smallest activation variance** are pruned first.

Conceptually:

```text
Neuron A
Large Variance
→ Activation changes significantly
→ Keep

Neuron B
Small Variance
→ Activation almost constant
→ Prune
```

---

# 10. Why Variance?

Suppose the activation of neuron $i$ is replaced by its mean:

```math
h_i
\rightarrow
\mu_i
```

The reconstruction error is:

```math
h_i-\mu_i
```

The expected squared error is:

```math
E[(h_i-\mu_i)^2]
=
\sigma_i^2
```

Therefore:

> **If a neuron is approximated by its mean, the neuron with the lowest variance produces the smallest expected reconstruction error.**

This gives the variance-based criterion a direct theoretical interpretation.

---

# 11. Zero-Variance Intuition

Consider a neuron whose output is always:

```text
0.7
0.7
0.7
0.7
0.7
```

Then:

```text
Mean = 0.7
Variance = 0
```

Replacing every output with the mean introduces:

```text
Zero Error
```

Therefore, this neuron does not need to be dynamically computed.

Its constant contribution can instead be incorporated elsewhere.

---

# 12. Global Pruning

VBP does not choose a fixed percentage independently for every MLP.

Instead, all hidden neurons across the network are gathered into a global score set.

Conceptually:

```text
Layer 1 Variances
Layer 2 Variances
Layer 3 Variances
...
        ↓
Global Ranking
        ↓
Remove Lowest-Variance Neurons
```

For pruning ratio $p$:

```text
Bottom p%
of all MLP hidden neurons
```

are removed.

Therefore, different layers can lose different numbers of neurons.

---

# 13. Mean Replacement

The theoretical pruning procedure would be:

```text
Pruned Neuron
     ↓
Do Not Use Original Activation
     ↓
Replace With Mean μ
```

For neuron $j$:

```math
h_j
\rightarrow
\mu_j
```

This preserves the neuron's average contribution.

However, explicitly inserting the mean still requires the second linear layer to retain the original hidden dimension.

That would prevent full computational savings.

---

# 14. Why Mean Replacement Alone Is Not Enough

Suppose the original hidden vector has dimension:

```text
3072
```

and 50% of neurons are pruned.

If the removed neurons are still explicitly replaced with their means:

```text
3072-dimensional vector
```

must still be multiplied by $W_2$.

Therefore:

```text
W1 Becomes Smaller
but
W2 Remains Full Size
```

This wastes half of the potential structured-pruning benefit.

---

# 15. Step 3: Mean-Shift Compensation

VBP solves this through:

> **Mean-Shift Compensation**

For:

```math
y = W_2 h + b_2
```

define a vector:

```math
\Delta \mu
```

where:

```math
(\Delta\mu)_j
=
\mu_j
```

for a pruned neuron, and:

```math
(\Delta\mu)_j
=
0
```

for a retained neuron.

Then:

```math
W_2 h
=
W_2(h-\Delta\mu)
+
W_2\Delta\mu
```

The second term is constant.

Therefore it can be absorbed into the bias.

---

# 16. Bias Update

The new bias is:

```math
b'_2
=
b_2
+
W_2\Delta\mu
```

The output becomes:

```math
y
=
W_2(h-\Delta\mu)
+
b'_2
```

For a pruned neuron:

```math
h_j = \mu_j
```

therefore:

```math
(h-\Delta\mu)_j = 0
```

This means the corresponding hidden dimension no longer needs to exist.

---

# 17. Structural Removal

After Mean-Shift Compensation:

```text
Pruned Hidden Neuron j
```

allows removal of:

```text
Row j of W1
```

and:

```text
Column j of W2
```

Therefore:

```text
Original MLP

Din
 ↓
Dhid
 ↓
Dout
```

becomes:

```text
Pruned MLP

Din
 ↓
Smaller Dhid
 ↓
Dout
```

Both MLP matrix multiplications become smaller.

---

# 18. Why the Method Uses Trained Networks

The method assumes that:

```text
W1
W2
b1
b2
```

are fixed parameters from a trained network.

Because the network is already trained:

```text
Mean Activation Statistics
```

have a meaningful stable interpretation.

The fixed weights also allow the removed mean contribution to be directly folded into the next bias.

---

# 19. Complete VBP Procedure

The full method can be summarized as:

```text
Pretrained Model
       │
       ▼
Run Calibration Data
       │
       ▼
Collect Post-Activation Statistics
       │
       ├── Mean
       └── Variance
       │
       ▼
Globally Rank MLP Neurons
by Variance
       │
       ▼
Select Low-Variance Neurons
       │
       ▼
Compute Mean Contribution
W2 Δμ
       │
       ▼
Add Contribution to Bias
b2' = b2 + W2 Δμ
       │
       ▼
Delete Corresponding
Rows of W1
and
Columns of W2
       │
       ▼
Smaller Dense Model
       │
       ▼
Optional Short Fine-Tuning
```

---

# 20. Why Post-Activation Statistics?

An important design decision is:

> **Measure variance after the nonlinear activation function.**

Not before it.

This matters because nonlinearities can strongly change the effective variance of a neuron.

---

# 21. Pre-Activation vs. Post-Activation

Suppose a neuron has strongly varying negative pre-activation values.

For example:

```text
-5
-4
-3
-2
```

Before GeLU:

```text
Variance
→ Large
```

After GeLU, the negative values can all become values close to zero.

```text
Near Zero
Near Zero
Near Zero
Near Zero
```

Then:

```text
Post-Activation Variance
→ Very Small
```

For an already trained model, the **post-activation output** is what actually enters the next layer.

Therefore, post-activation variance better represents the neuron's effective contribution.

---

# 22. Pre- vs. Post-Activation Experiment

For DeiT-Base at 50% pruning:

| Statistics Location | Retained Accuracy | Final Accuracy |
|---|---:|---:|
| Pre-Activation | 0.43% | 77.92% |
| Post-Activation | **66.40%** | **80.99%** |

Both cases have identical:

```text
MACs = 12.0 G
Parameters = 58.24 M
```

Therefore, the statistic location has a dramatic impact on pruning quality.

---

# 23. Experimental Setup

VBP is evaluated primarily on:

- DeiT
- Swin Transformer
- ConvNeXt

using:

```text
ImageNet-1k
```

Models include:

```text
Tiny
Small
Base
```

Fine-tuning after pruning uses only:

```text
10 epochs
```

with knowledge distillation from the original model.

Experiments are performed using:

```text
NVIDIA H200 GPUs
```

---

# 24. Two Accuracy Measurements

The paper reports accuracy at two stages.

## Retained Accuracy

```text
Immediately After Pruning
Before Fine-Tuning
```

This measures how well the pruning method preserves the pretrained representation.

---

## Final Accuracy

```text
After 10 Epochs
of Fine-Tuning
```

This measures how much of the original accuracy can ultimately be recovered.

This distinction is central to the paper.

---

# 25. DeiT-Base Main Result

For DeiT-Base:

```text
MLP Pruning Rate
55%
```

Original:

```text
MACs
17.58 G

Parameters
86.57 M

Top-1
81.73%
```

After VBP:

```text
MACs
11.44 G
→ -34.93%

Parameters
55.40 M
→ -36.01%
```

Immediately after pruning:

```text
57.58% Top-1
```

which corresponds to:

```text
70.48%
```

of original accuracy.

---

# 26. DeiT-Base After Fine-Tuning

After only:

```text
10 fine-tuning epochs
```

the accuracy becomes:

```text
80.67%
```

compared with the original:

```text
81.73%
```

Therefore VBP retains:

```text
98.74%
```

of the original accuracy while reducing:

```text
MACs by ~35%
Parameters by ~36%
```

---

# 27. DeiT-Base Runtime

On NVIDIA H200:

```text
Original
30.73 ms
```

VBP:

```text
21.36 ms
```

Speedup:

```text
1.44×
```

Therefore, the structured reduction translates into a measurable real-hardware speedup.

---

# 28. Off-the-Shelf DeiT-Base Pruning

The paper also evaluates a lower pruning ratio:

```text
20%
```

For DeiT-Base:

```text
Original Top-1
81.73%
```

Immediately after VBP:

```text
80.87%
```

Accuracy retention:

```text
98.98%
```

without fine-tuning.

After fine-tuning:

```text
81.76%
```

The model slightly exceeds the reported original baseline accuracy.

---

# 29. 20% Pruning Cost Reduction

For DeiT-Base with 20% MLP pruning:

```text
MAC Reduction
12.68%

Parameter Reduction
13.09%
```

H200 runtime:

```text
30.73 ms
→
27.72 ms
```

Speedup:

```text
1.11×
```

This demonstrates that modest VBP pruning can be used almost directly without additional training.

---

# 30. DeiT and Swin Results

The main ImageNet results include:

| Model | MLP Pruning | MAC Reduction | Parameter Reduction | Final Accuracy Retention |
|---|---:|---:|---:|---:|
| DeiT-T | 45% | 25.16% | 27.97% | 97.33% |
| DeiT-S | 50% | 30.37% | 32.15% | 98.64% |
| DeiT-B | 55% | 34.93% | 36.01% | 98.74% |
| Swin-T | 45% | 28.22% | 24.67% | 98.15% |
| Swin-S | 50% | 32.19% | 29.41% | 98.58% |
| Swin-B | 55% | 33.89% | 35.87% | 98.70% |

Larger models tolerate more pruning because they contain more redundancy.

---

# 31. Comparison with Magnitude and SNIP

The paper compares VBP with structured versions of:

- Magnitude Pruning
- SNIP

at the same:

```text
50% MLP pruning rate
```

Therefore all methods have the same:

- MAC count
- Parameter count

The difference comes only from **which neurons are selected**.

---

# 32. DeiT-Base Comparison

At 50% pruning:

| Method | Retained Accuracy | Final Accuracy |
|---|---:|---:|
| Magnitude | 0.37% | 78.88% |
| SNIP | 53.24% | 80.40% |
| VBP | **66.40%** | **80.99%** |

VBP therefore provides a much better starting point immediately after pruning.

---

# 33. Why Immediate Retention Matters

If pruning produces:

```text
Very Low Accuracy
```

the model requires substantial retraining.

If pruning instead preserves much of the pretrained representation:

```text
High Accuracy Retention
        ↓
Short Fine-Tuning
        ↓
Recover Original Accuracy
```

This is one of the paper's central motivations.

---

# 34. Comparison with NViT

The paper also compares against:

> **NViT**

a Hessian-aware structured pruning method.

For a similar model size:

| Method | Pruning Duration | Retained Accuracy | Final Accuracy |
|---|---:|---:|---:|
| NViT | 50 epochs | 81.13% | 82.18% |
| NViT | 1 epoch | 69.10% | 81.92% |
| VBP | 1 epoch | **72.37%** | **82.32%** |

Both methods are followed by:

```text
10 epochs fine-tuning
```

VBP achieves higher final accuracy than the reported NViT configurations in this comparison.

---

# 35. Combining VBP with ToMe

VBP removes:

```text
MLP Neurons
```

while Token Merging (ToMe) reduces:

```text
Number of Tokens
```

Therefore, the two methods target different dimensions of computation.

Conceptually:

```text
ToMe
→ Reduce Sequence Length

VBP
→ Reduce MLP Width
```

This makes the methods largely orthogonal.

---

# 36. Hybrid ToMe + VBP

For DeiT-Base:

```text
ToMe-14 + VBP
```

achieves:

```text
2.05× Speedup
```

with:

```text
80.09% Top-1
```

compared with the original:

```text
81.73%
```

It also reduces the parameter count.

---

# 37. Why Hybrid Pruning Helps

Using ToMe alone for more aggressive token reduction can achieve similar speedup, but causes more accuracy loss.

For example:

```text
ToMe-24
Speedup = 2.15×
Final Accuracy = 75.74%
```

while:

```text
ToMe-14 + VBP
Speedup = 2.05×
Final Accuracy = 80.09%
```

Therefore, distributing compression across:

```text
Token Dimension
+
MLP Width
```

can produce a better accuracy-efficiency trade-off.

---

# 38. Ablation: Variance and Mean Shift

The paper tests both components independently.

For DeiT-Base at 50% pruning:

| Variance Criterion | Mean Shift | Retained Accuracy | Final Accuracy |
|---|---|---:|---:|
| No | Yes | 55.19% | 80.23% |
| Yes | No | 26.04% | 80.62% |
| Yes | Yes | **66.40%** | **80.99%** |

The best result occurs when both are used together.

---

# 39. Importance of Mean-Shift Compensation

Without Mean-Shift Compensation:

```text
Retained Accuracy
26.04%
```

With both VBP and Mean Shift:

```text
66.40%
```

Therefore, Mean-Shift Compensation improves immediate accuracy dramatically.

The key insight is:

> Pruned neurons may vary little, but their average activation can still make a large constant contribution to the output.

Simply deleting them removes this contribution.

---

# 40. Why Low Variance Does Not Mean Low Mean

A neuron can have:

```text
Mean = Large
Variance = Small
```

For example:

```text
4.9
5.0
5.1
5.0
```

Its variance is low.

But deleting the neuron entirely would remove a contribution close to:

```text
5.0
```

from every sample.

VBP avoids this by preserving:

```text
Mean Contribution
```

inside the bias.

---

# 41. Variance Distribution

The paper analyzes the variance of all MLP neurons.

The variance distribution is strongly non-uniform.

It finds that approximately:

```text
60% of the lowest-variance neurons
```

are needed to account for only:

```text
10% of cumulative variance
```

This indicates that many neurons vary very little.

However, once these low-variance neurons have been removed, additional pruning removes increasingly expressive neurons.

---

# 42. Pruning-Rate Sensitivity

For DeiT-Base:

```text
5% ~ 25% pruning
```

retains very high accuracy before fine-tuning.

As the pruning rate increases beyond this region:

```text
Immediate Accuracy
```

begins to decline more quickly.

This corresponds to entering the region of the variance distribution where neurons have substantially higher variance.

---

# 43. DeiT-Base Across Pruning Rates

Examples:

| MLP Pruning | Retained Accuracy | Final Accuracy |
|---:|---:|---:|
| 5% | 81.63% | 81.79% |
| 10% | 81.54% | 81.72% |
| 20% | 80.87% | 81.76% |
| 30% | 78.97% | 81.68% |
| 40% | 75.78% | 81.55% |
| 50% | 66.40% | 80.99% |

Original:

```text
81.73%
```

This shows that even when immediate accuracy drops at higher pruning rates, short fine-tuning can recover much of the original performance.

---

# 44. ConvNeXt Generalization

VBP is also applied to:

- ConvNeXt-T
- ConvNeXt-S
- ConvNeXt-B

at 50% MLP pruning.

The method achieves:

```text
> 50% Parameter Reduction
```

for all three ConvNeXt sizes.

MAC reductions reach:

```text
~34% to ~42%
```

---

# 45. ConvNeXt Results

| Model | MAC Reduction | Parameter Reduction | Final Accuracy Retention |
|---|---:|---:|---:|
| ConvNeXt-T | 33.8% | 55.9% | 98.1% |
| ConvNeXt-S | 41.3% | 53.2% | 97.9% |
| ConvNeXt-B | 42.1% | 53.4% | 97.6% |

This demonstrates that VBP is not restricted to standard Vision Transformers.

---

# 46. ConvNeXt Accuracy Retention

ConvNeXt models show substantially lower immediate accuracy after pruning.

For example:

```text
ConvNeXt-T
Original = 82.90%

Immediately After Pruning = 16.80%
```

yet after 10 epochs:

```text
81.30%
```

is recovered.

The paper attributes the stronger immediate degradation partly to differences in ConvNeXt's architecture and the relative importance of its MLP-like components.

---

# 47. Real Hardware Evaluation

An important strength of the paper is that it measures actual latency on multiple hardware platforms:

- NVIDIA H200 GPU
- NVIDIA T4 GPU
- Intel Xeon E5-2680v4 CPU

This distinguishes:

```text
MAC Reduction
```

from:

```text
Actual Runtime Reduction
```

---

# 48. DeiT-Base Runtime

For DeiT-Base with 55% MLP pruning:

### H200

```text
30.73 ms
→
21.36 ms

1.44×
```

### T4

```text
378.63 ms
→
273.94 ms

1.38×
```

### CPU

```text
5.71 s
→
3.81 s

1.50×
```

The smaller dense matrices translate into speedup across different hardware platforms.

---

# 49. ConvNeXt-Base Runtime

For ConvNeXt-Base:

### H200

```text
32.39 ms
→
21.75 ms

1.49×
```

### T4

```text
384.87 ms
→
249.95 ms

1.54×
```

### CPU

```text
5.64 s
→
3.30 s

1.71×
```

The paper reports up to:

```text
1.71×
```

runtime speedup across its tested hardware configurations.

---

# 50. Why Structured Pruning Produces Real Speedup

Unlike unstructured sparsity:

```text
Original Matrix
→ Sparse Matrix of Same Dimensions
```

VBP changes the actual matrix dimensions:

```text
W1
Dhid × Din
```

becomes:

```text
Smaller Dhid × Din
```

and:

```text
W2
Dout × Dhid
```

becomes:

```text
Dout × Smaller Dhid
```

Therefore, standard dense GEMM operations directly become smaller.

No sparse kernel is required.

---

# 51. Layer-Wise Pruning Distribution

Because VBP performs global pruning, the pruning rate differs across layers.

The supplementary analysis shows that VBP tends to prune more neurons in earlier layers.

SNIP shows a different trend and often prunes more aggressively in deeper layers.

This suggests that different pruning criteria identify very different notions of neuron importance.

---

# 52. VBP vs. Magnitude Pruning

Magnitude pruning asks:

```text
Which weights are numerically small?
```

VBP asks:

```text
Which hidden neurons produce activations
that vary the least across data?
```

Therefore, the two methods operate on different signals:

```text
Magnitude
→ Parameter Statistics

VBP
→ Activation Statistics
```

---

# 53. VBP vs. SNIP

SNIP uses:

```text
Gradient-Based Sensitivity
```

VBP uses:

```text
Activation Variance
```

Advantages of VBP include:

- No backward pass needed for the pruning score
- Simple online statistics
- Direct mean-replacement interpretation
- Strong immediate accuracy retention in the reported experiments

---

# 54. VBP vs. Dynamic Token Pruning

Dynamic token pruning:

```text
Input Dependent
```

and may reduce computation differently for every sample.

VBP:

```text
Static
Structured
```

and permanently modifies the model architecture.

Therefore:

```text
Dynamic Token Pruning
→ Reduce Runtime Computation
→ Model Parameters Unchanged
```

while:

```text
VBP
→ Reduce Runtime Computation
→ Reduce Model Size
```

---

# 55. VBP vs. OBC / OBS

Earlier Optimal Brain methods determine pruning importance using second-order curvature information.

Conceptually:

```text
OBC / OBS
→ Hessian-Based Sensitivity
```

VBP instead uses:

```text
Activation Variance
```

and does not require:

- Hessian computation
- Hessian inversion
- Weight-by-weight compensation

The compensation mechanism is much simpler:

```text
Mean of Removed Activation
        ↓
Next-Layer Bias
```

---

# 56. Key Difference from Optimal Brain Compression

OBC compensation:

```text
Remove Weight
      ↓
Modify Remaining Weights
using H^-1
```

VBP compensation:

```text
Remove Neuron
      ↓
Preserve Mean Activation
      ↓
Modify Only Next-Layer Bias
```

Therefore VBP sacrifices the more elaborate second-order reconstruction machinery for a very low-cost structured pruning scheme.

---

# 57. Computational Perspective

For a Transformer MLP:

```text
Dmodel
   ↓
Dhidden
   ↓
Dmodel
```

often:

```text
Dhidden >> Dmodel
```

Therefore, MLP layers contain substantial:

- Parameters
- MACs

Reducing $D_{hidden}$ directly reduces both large matrix multiplications.

This makes the MLP an attractive target for structured compression.

---

# 58. Why Mean and Variance Work Together

The method can be interpreted as decomposing neuron behavior into:

```text
Activation
=
Mean Component
+
Variable Component
```

That is:

```math
h_i
=
\mu_i
+
(h_i-\mu_i)
```

VBP treats them differently:

```text
Mean Component
→ Preserve in Bias

Variable Component
→ Decide Importance Using Variance
```

For low-variance neurons:

```text
Variable Component
≈ Small
```

so only the mean needs to be preserved.

This is perhaps the simplest way to understand the entire method.

---

# 59. Limitations

## Fine-Tuning Is Still Needed at High Pruning Rates

Although moderate pruning can be used nearly off-the-shelf, aggressive pruning causes a substantial immediate accuracy drop.

For example, DeiT-Base at 55% MLP pruning drops from:

```text
81.73%
```

to:

```text
57.58%
```

before fine-tuning.

---

## Only MLP Neurons Are Pruned

The method intentionally focuses on:

```text
MLP Hidden Neurons
```

It does not directly prune:

- Attention heads
- Attention dimensions
- Complete Transformer blocks
- Tokens

---

## Importance Criterion Is Statistical

Variance provides a simple importance measure, but it does not explicitly model:

- Task loss curvature
- Inter-neuron correlation
- Redundancy between neurons

Each neuron is primarily evaluated through its own activation variance.

---

## Calibration Statistics Matter

The pruning scores depend on observed activation statistics.

Therefore, the representativeness of the data used to gather statistics can affect the pruning decisions.

---

## Aggressive Pruning Eventually Removes Expressive Neurons

Because neuron variances are not uniformly distributed, after the many low-variance neurons have been removed, further pruning rapidly removes more informative neurons.

This explains the nonlinear degradation at high pruning ratios.

---

# 60. Main Contributions

### 1. Variance-Based Structured Pruning

Uses post-activation variance as a simple neuron-importance metric.

### 2. Mean-Replacement Interpretation

Shows that variance directly corresponds to expected squared error when replacing a neuron by its mean.

### 3. Mean-Shift Compensation

Moves the average contribution of removed neurons into the next-layer bias.

### 4. Real Structural Compression

Removes rows and columns from MLP matrices, reducing both model size and MACs.

### 5. Minimal Fine-Tuning

Requires only 10 epochs in the main experiments to recover most original accuracy.

### 6. Broad Architecture Support

Demonstrated on:

- DeiT
- Swin
- ConvNeXt

### 7. Orthogonality with Token Pruning

Can be combined with ToMe to obtain roughly 2× inference speedups.

### 8. Real Hardware Speedup

Demonstrates latency reductions on:

- H200
- T4
- CPU

rather than relying only on theoretical MAC savings.

---

# 61. Key Takeaways

1. VBP is a **one-shot structured pruning method** for already trained networks.

2. It targets hidden neurons in MLP blocks.

3. Neuron importance is measured using **post-activation variance**.

4. A neuron with low variance behaves approximately like a constant across inputs.

5. Replacing a neuron by its mean produces expected squared error equal to its variance.

6. Therefore, pruning the lowest-variance neurons minimizes the mean-replacement error under this criterion.

7. Simply deleting a low-variance neuron is not sufficient because its mean contribution may still be large.

8. Mean-Shift Compensation preserves this contribution by adding $W_2\Delta\mu$ to the next-layer bias.

9. This allows both the corresponding row of $W_1$ and column of $W_2$ to be physically removed.

10. VBP therefore creates smaller **dense** matrices rather than irregular sparse matrices.

11. The resulting structural reductions directly produce real hardware speedups.

12. Post-activation statistics are much more effective than pre-activation statistics for pruning an already trained network.

13. For DeiT-Base at 55% MLP pruning, VBP reduces MACs by about **35%** and parameters by about **36%**.

14. After only 10 epochs of fine-tuning, DeiT-Base retains about **98.7%** of its original Top-1 accuracy.

15. The same DeiT-Base configuration reaches about **1.44× speedup on H200**.

16. At a modest 20% MLP pruning ratio, DeiT-Base immediately retains about **99% of its original accuracy** without fine-tuning.

17. VBP can be combined with token-reduction methods such as ToMe because they compress different dimensions of the network.

18. The paper demonstrates that simple activation statistics can provide a strong structured-pruning signal without Hessian computation or expensive retraining.

---

# 62. Concepts to Review

- Structured Pruning
- Unstructured Pruning
- Neuron Pruning
- MLP
- Vision Transformer
- Activation Statistics
- Mean
- Variance
- Expected Squared Error
- Welford's Algorithm
- Global Pruning
- Post-Activation Statistics
- GeLU
- Mean Replacement
- Bias Folding
- Mean-Shift Compensation
- MACs
- Dense Matrix Multiplication
- Structured Sparsity
- Knowledge Distillation
- Token Pruning
- Token Merging
- SNIP
- Magnitude Pruning
- NViT
