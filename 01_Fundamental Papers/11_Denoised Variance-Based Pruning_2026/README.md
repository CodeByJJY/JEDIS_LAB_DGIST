# Denoised Variance-Based Pruning with Optimal Brain Bias Compensation

> **Denoised Variance-Based Pruning with Optimal Brain Bias Compensation**

- **Authors:** Geon Tack Lee, Jaegul Choo, Kang Eun Jeon
- **Year:** 2026
- **Topic:** Structured Neural Network Pruning
- **Keywords:** DVBP, OB2C, Structured Pruning, Activation Covariance, Random Matrix Theory, Marchenko-Pastur Distribution, Optimal Brain Compression

---

## 1. Problem Background

Structured pruning physically removes complete neurons or channels from a neural network.

Compared with unstructured pruning:

```text
Unstructured Pruning
→ Individual weights become zero
→ Irregular sparsity
→ Specialized kernels may be required
```

structured pruning produces:

```text
Smaller Dense Matrices
→ Standard hardware can directly exploit the reduction
```

However, structured pruning often causes large accuracy degradation because removing an entire neuron also removes all of the information carried by that feature dimension.

---

# 2. Starting Point: Variance-Based Pruning

Variance-Based Pruning (VBP) addresses structured neuron pruning using activation statistics.

For a hidden neuron $i$:

```math
\mu_i = \mathbb{E}[x_i]
```

and:

```math
\sigma_i^2
=
\mathbb{E}
\left[
(x_i-\mu_i)^2
\right]
```

VBP assumes that a neuron with low activation variance behaves approximately like a constant.

Therefore:

```text
Low Variance
     ↓
Activation Changes Little
     ↓
Replace Activation by Mean
     ↓
Remove Dynamic Computation
```

---

# 3. Mean-Shift Compensation in VBP

Suppose the downstream linear layer is:

```math
y = Wx + b
```

For a pruned neuron, VBP replaces its activation by its mean.

The mean contribution can then be absorbed into the bias:

```math
b'
=
b
+
W \Delta\mu
```

where $\Delta\mu$ contains the mean values of pruned neurons and zeros elsewhere.

Therefore:

```text
Pruned Neuron
      ↓
Mean Contribution
      ↓
Move into Bias
```

This allows the neuron to be structurally removed while preserving its average contribution.

---

# 4. Two Limitations of VBP

This paper identifies two major limitations.

## Limitation 1: Noisy Variance Estimates

VBP uses:

```text
Activation Variance
```

as the neuron importance score.

However, variance is estimated from a finite calibration set.

Therefore:

```text
True Activation Statistics
        +
Finite-Sample Noise
        ↓
Observed Variance
```

At aggressive pruning ratios, noisy importance estimates can cause important neurons to be incorrectly removed.

---

## Limitation 2: Bias-Only Compensation

VBP only compensates for pruning by changing the next-layer bias.

```text
Removed Neuron
     ↓
Mean Contribution
     ↓
Bias Update
```

The remaining weight matrix itself is unchanged.

However, removing a complete neuron changes the layer reconstruction more generally.

Therefore:

> Bias correction alone cannot optimally compensate for the structural reconstruction error.

---

# 5. Main Idea

The proposed method is:

> **Denoised Variance-Based Pruning + Optimal Brain Bias Compensation**

or:

```text
DVBP + OB2C
```

The method contains two main improvements:

```text
VBP
│
├── Raw Activation Variance
│
└── Mean-Shift Compensation
```

becomes:

```text
DVBP + OB2C
│
├── Denoised Covariance
│      ↓
│   Better Neuron Selection
│
└── OBC-Based Weight Recovery
       +
    Optimal Bias Compensation
```

---

# 6. Core Insight

The same activation covariance matrix can be used for:

```text
1. Neuron Selection
       +
2. Weight Compensation
```

This creates a unified statistical framework.

Conceptually:

```text
Activation Data
      ↓
Covariance Matrix C
      │
      ├── Denoise
      │      ↓
      │   Pruning Score
      │
      └── OBC / OB2C
             ↓
         Weight Recovery
```

---

# 7. Optimal Brain Compression Background

Optimal Brain Compression (OBC) minimizes layer-wise reconstruction error.

Given:

```math
Y = WX
```

and compressed weights $\hat{W}$:

```math
\min_{\hat{W}}
\left\|
WX-\hat{W}X
\right\|_F^2
```

The goal is to preserve the original layer output after pruning or quantization.

---

# 8. Standard OBC Hessian

For the reconstruction objective:

```math
\left\|
WX-\hat{W}X
\right\|_F^2
```

the layer-wise proxy Hessian is:

```math
H
=
2XX^T
```

This uses the **uncentered second moment** of the activation matrix.

---

# 9. Optimal Brain Bias Compensation

The paper extends the reconstruction objective by explicitly introducing a bias term.

The new objective is:

```math
\min_{\hat{W},b}
\left\|
WX
-
\left(
\hat{W}X+b\mathbf{1}^T
\right)
\right\|_F^2
```

Instead of fixing the bias, the method asks:

> What bias minimizes the reconstruction error for the compressed weight matrix?

---

# 10. Optimal Bias

Taking the derivative with respect to the bias gives the optimal bias:

```math
b^*
=
(W-\hat{W})\mu
```

where:

```math
\mu
=
\frac{1}{N}
X\mathbf{1}
```

is the activation mean.

Therefore:

```text
Weight Perturbation
      ↓
Multiply by Activation Mean
      ↓
Optimal Bias Compensation
```

---

# 11. Mean-Centered Activations

Define:

```math
\widetilde{X}
=
X-\mu\mathbf{1}^T
```

After substituting the optimal bias back into the reconstruction objective, the problem becomes:

```math
\min_{\Delta W}
\left\|
\Delta W \widetilde{X}
\right\|_F^2
```

where:

```math
\Delta W
=
W-\hat{W}
```

The mean component has disappeared from the remaining reconstruction error.

---

# 12. Key Mathematical Result

The Hessian now becomes:

```math
H
=
2\widetilde{X}\widetilde{X}^T
```

The activation covariance matrix is:

```math
C
=
\frac{1}{N}
\widetilde{X}\widetilde{X}^T
```

Therefore:

```math
H = 2NC
```

This is the key result of the paper.

---

# 13. Why This Result Matters

Standard OBC uses:

```math
XX^T
```

while OB2C uses:

```math
\widetilde{X}\widetilde{X}^T
```

after optimally accounting for the mean through the bias.

Therefore:

```text
Optimal Bias Compensation
        ↓
Remove Mean Component
        ↓
Hessian
        =
Centered Activation Covariance
```

The same covariance matrix can now support both:

```text
Neuron Importance
+
Optimal Weight Recovery
```

---

# 14. OB2C Weight Update

Let $P$ denote the indices of neurons selected for pruning.

The weight update is:

```math
\delta_W
=
-
W_{:,P}
\left(
[C^{-1}]_{P,P}
\right)^{-1}
[C^{-1}]_{P,:}
```

This adjusts the remaining weights so that they optimally compensate for the removed dimensions under the layer-wise reconstruction objective.

---

# 15. VBP vs. OB2C Compensation

VBP:

```text
Prune Neuron
      ↓
Preserve Mean
      ↓
Update Bias Only
```

OB2C:

```text
Prune Neuron
      ↓
Update Remaining Weights
using Covariance
      +
Optimal Bias Update
```

Therefore:

```text
VBP
→ Constant Component Compensation

OB2C
→ Constant Component
  +
  Correlated Reconstruction Compensation
```

---

# 16. Why Covariance Matters

Variance only describes each neuron individually:

```math
C_{ii}
=
\mathrm{Var}(x_i)
```

Covariance also describes relationships between neurons:

```math
C_{ij}
=
\mathrm{Cov}(x_i,x_j)
```

Therefore:

```text
Variance
→ How much neuron i changes

Covariance
→ How neuron i changes together
  with other neurons
```

OB2C can exploit these correlations when modifying the remaining weights.

---

# 17. Online Covariance Computation

Computing the covariance matrix naively would require storing all activation vectors.

For a large calibration dataset:

```text
Store Every Activation
        ↓
Large Memory Requirement
```

The paper therefore computes covariance online in batches.

---

# 18. Online Mean Update

Suppose:

```text
P
→ Previously Processed Samples

N
→ New Batch
```

with:

```text
nP samples
nN samples
```

and total:

```text
nC = nP + nN
```

The running activation mean can be updated without storing the full activation history.

---

# 19. Online Covariance Update

Define:

```math
\alpha
=
\frac{n_P}{n_C}
```

and:

```math
\delta
=
\mu_N-\mu_P
```

Then the combined covariance is:

```math
C_C
=
\alpha C_P
+
(1-\alpha)C_N
+
\alpha(1-\alpha)
\delta\delta^T
```

This allows exact covariance accumulation using only:

- Current covariance
- Current mean
- New batch statistics

---

# 20. Why Covariance Denoising?

The observed sample covariance can be written conceptually as:

```text
Observed Covariance
        =
Signal
+
Statistical Noise
```

With finite calibration data, small eigenvalues may largely reflect sampling noise rather than meaningful activation structure.

This is particularly problematic when pruning aggressively.

---

# 21. Random Matrix Theory

DVBP uses:

> **Random Matrix Theory**

to distinguish:

```text
Signal Eigenvalues
```

from:

```text
Noise Eigenvalues
```

The main tool is the:

> **Marchenko-Pastur Distribution**

---

# 22. Marchenko-Pastur Distribution

For a large random matrix with i.i.d. zero-mean entries, the eigenvalues of its sample covariance follow the Marchenko-Pastur distribution.

The theoretical eigenvalue support is:

```math
\lambda_{\pm}
=
\sigma^2
\left(
1\pm\sqrt{\lambda}
\right)^2
```

where:

```math
\lambda
=
\frac{d}{N}
```

with:

- $d$: activation dimension
- $N$: calibration set size
- $\sigma^2$: estimated noise variance

---

# 23. Interpretation of the MP Spectrum

Conceptually:

```text
Small / Medium Eigenvalues
inside MP Bulk
        ↓
Likely Noise

Large Eigenvalues
above lambda+
        ↓
Likely Signal
```

Therefore:

```text
Eigenvalue <= lambda+
→ Remove

Eigenvalue > lambda+
→ Preserve
```

---

# 24. Why Calibration Set Size Uses Images

Vision Transformer activations contain many tokens or patches per image.

However, patches within one image are strongly correlated.

Using:

```text
Number of All Patches
```

as the independent sample count would violate the i.i.d. assumption underlying the MP model.

Therefore, the paper defines:

```math
\lambda
=
\frac{d}{N}
```

where $N$ is:

```text
Number of Calibration Images
```

rather than the total number of patches.

---

# 25. Noise-Level Estimation

The true noise variance is unknown.

The paper estimates it using the median eigenvalue of the observed covariance spectrum.

Conceptually:

```text
Observed Median Eigenvalue
        ↓
Compare with
Theoretical MP Median
        ↓
Estimate Noise Variance
```

The resulting estimate is:

```math
\hat{\sigma}
=
\sqrt{
\frac{
\Lambda_{\mathrm{med}}
}{
\mu_{\lambda}
}
}
```

where:

- $\Lambda_{\mathrm{med}}$: median covariance eigenvalue
- $\mu_{\lambda}$: theoretical median of the MP distribution

---

# 26. Spectral Denoising

Decompose the covariance matrix:

```math
C
=
V\Lambda V^T
```

Then retain only eigenpairs satisfying:

```math
\lambda_i > \lambda_+
```

Conceptually:

```text
Covariance
    ↓
Eigendecomposition
    ↓
 ┌───────────────┐
 │ Noise Bulk    │ → Remove
 └───────────────┘

Large Eigenvalues
→ Preserve
```

---

# 27. Denoised Covariance

The denoised covariance matrix is reconstructed from the retained signal components:

```math
C_{\mathrm{denoised}}
=
V_{\mathrm{signal}}
\Lambda_{\mathrm{signal}}
V_{\mathrm{signal}}^T
```

This removes covariance components considered consistent with finite-sample random noise.

---

# 28. Denoised Variance Score

After reconstructing the denoised covariance:

```math
s_{\mathrm{dvar}}
=
\mathrm{diag}
\left(
C_{\mathrm{denoised}}
\right)
```

Each diagonal entry becomes the denoised variance importance score for one neuron.

---

# 29. Raw Variance vs. Denoised Variance

Raw VBP score:

```math
s_{\mathrm{var}}
=
\mathrm{diag}(C)
```

DVBP score:

```math
s_{\mathrm{dvar}}
=
\mathrm{diag}
\left(
C_{\mathrm{denoised}}
\right)
```

The paper observes:

```text
Low Pruning Ratio
→ Raw Variance Often Better

High Pruning Ratio
→ Denoised Variance Better
```

---

# 30. Why Denoising Can Hurt at Low Pruning Ratios

At low pruning ratios, only the least important neurons need to be removed.

Some small eigenvalue components may contain:

```text
Weak but Real Signal
```

rather than pure noise.

Hard MP thresholding can accidentally remove these components.

Therefore:

```text
Aggressive Denoising
→ Can Lose Small Signal
```

which explains why raw variance can work better at modest pruning ratios.

---

# 31. Hybrid Variance Score

To combine the strengths of both metrics, the paper introduces:

```math
s_{\mathrm{hybrid}}
=
(1-p)s_{\mathrm{var}}
+
p s_{\mathrm{dvar}}
```

where:

```text
p = Target Pruning Ratio
```

Therefore:

```text
Small p
→ More Raw Variance

Large p
→ More Denoised Variance
```

---

# 32. Interpretation of the Hybrid Score

For example:

```text
p = 0.2
```

gives:

```text
80% Raw Variance
+
20% Denoised Variance
```

while:

```text
p = 0.7
```

gives:

```text
30% Raw Variance
+
70% Denoised Variance
```

The score gradually shifts toward the denoised estimate as pruning becomes more aggressive.

---

# 33. Complete DVBP + OB2C Pipeline

The full method is:

```text
Pretrained Model
       │
       ▼
Calibration Data
       │
       ▼
Collect Activation Mean
and Covariance
       │
       ▼
Eigendecompose Covariance
       │
       ▼
Fit Marchenko-Pastur Spectrum
       │
       ▼
Remove Noise Eigencomponents
       │
       ▼
Compute Denoised Variance
       │
       ▼
Hybrid Variance Score
       │
       ▼
Globally Rank MLP Neurons
       │
       ▼
Select Pruned Neurons
       │
       ▼
OB2C Weight Recovery
       │
       ▼
Optimal Mean/Bias Compensation
       │
       ▼
Structurally Remove Neurons
       │
       ▼
Smaller Dense Network
```

---

# 34. Global Structured Pruning

Like VBP, DVBP + OB2C performs global pruning across MLP hidden dimensions.

It does not enforce:

```text
Same Pruning Ratio
for Every Layer
```

Instead:

```text
All Hidden Neurons
        ↓
Global Importance Ranking
        ↓
Bottom p% Removed
```

Different layers therefore retain different amounts of capacity.

---

# 35. Structural Model Reduction

For a two-layer MLP:

```text
Input Dimension
      ↓
Hidden Dimension
      ↓
Output Dimension
```

pruning reduces:

```text
d_hidden
→
d_pruned
```

Therefore both weight matrices shrink.

Conceptually:

```text
First Linear Layer
din × dhidden
       ↓
din × dpruned
```

and:

```text
Second Linear Layer
dhidden × dout
       ↓
dpruned × dout
```

The external input and output dimensions remain unchanged.

---

# 36. Experimental Setup

The method is evaluated on:

- DeiT
- Swin Transformer
- ConvNeXt

with:

```text
Tiny
Small
Base
```

variants.

Dataset:

```text
ImageNet-1K
```

The main calibration set contains:

```text
8,192 images
```

randomly sampled from the ImageNet training set.

No data augmentation is used during calibration.

---

# 37. Main Evaluation Goal

Unlike the original VBP paper, the primary evaluation focuses on:

> **Immediate accuracy after pruning**

without:

```text
Fine-Tuning
or
Retraining
```

Therefore the main result measures:

```text
Zero-Shot Post-Pruning Accuracy
```

This directly tests how much information the pruning and compensation method preserves.

---

# 38. 50% Pruning: DeiT

At 50% MLP pruning:

| Model | Full | Magnitude | SNIP | VBP | DVBP + OB2C |
|---|---:|---:|---:|---:|---:|
| DeiT-T | 72.16 | 3.73 | 39.49 | 42.89 | **56.40** |
| DeiT-S | 79.86 | 4.24 | 50.65 | 66.03 | **70.25** |
| DeiT-B | 81.98 | 0.38 | 62.78 | 68.92 | **75.85** |

No post-pruning fine-tuning is used for these results.

---

# 39. DeiT-B Improvement

For DeiT-B:

```text
Full
81.98
```

At 50% pruning:

```text
VBP
68.92
```

```text
DVBP + OB2C
75.85
```

Improvement over VBP:

```text
+6.93 percentage points
```

without fine-tuning.

---

# 40. 50% Pruning: Swin

At the same 50% pruning ratio:

| Model | Full | Magnitude | SNIP | VBP | DVBP + OB2C |
|---|---:|---:|---:|---:|---:|
| Swin-T | 81.38 | 26.40 | 28.87 | 49.98 | **66.12** |
| Swin-S | 83.32 | 0.22 | 45.36 | 69.91 | **77.24** |
| Swin-B | 85.27 | 6.44 | 38.55 | 70.86 | **76.16** |

The improvement is especially large for Swin-T and Swin-S.

---

# 41. Swin-S Improvement

For Swin-S at 50% pruning:

```text
VBP
69.91
```

```text
DVBP + OB2C
77.24
```

Improvement:

```text
+7.33 percentage points
```

---

# 42. Accuracy Gap Grows with Pruning Ratio

One of the most important experimental observations is:

> **The advantage over VBP grows as pruning becomes more aggressive.**

For Swin-S:

```text
40% Pruning
DVBP+OB2C - VBP
= +2.28 points
```

```text
50% Pruning
= +7.33 points
```

```text
60% Pruning
= +25.49 points
```

This supports the argument that noise-aware selection and stronger compensation become increasingly important at high pruning ratios.

---

# 43. 60% Pruning Example

For Swin-S:

```text
Full
83.32
```

At 60% pruning:

```text
VBP
45.30
```

while:

```text
DVBP + OB2C
70.79
```

The gap is:

```text
25.49 percentage points
```

despite both methods structurally pruning the same fraction of MLP neurons.

---

# 44. Same Compression, Better Accuracy

For DeiT models, the methods use the same target layers and pruning ratios.

Therefore:

```text
Parameters
and
MACs
```

are effectively the same.

For DeiT-B at 50%:

```text
Parameters
86.57 M
→
58.24 M
```

and:

```text
MACs
17.59 G
→
12.01 G
```

The main difference is:

```text
Which neurons are removed
+
How surviving weights are compensated
```

---

# 45. ConvNeXt Results

At 50% pruning:

| Model | Full | VBP | DVBP + OB2C |
|---|---:|---:|---:|
| ConvNeXt-T | 84.17 | 15.44 | **44.90** |
| ConvNeXt-S | 85.17 | 28.13 | **50.01** |
| ConvNeXt-B | 85.81 | 56.12 | **70.76** |

The improvement over VBP is:

```text
ConvNeXt-T
+29.46 points
```

```text
ConvNeXt-S
+21.88 points
```

```text
ConvNeXt-B
+14.64 points
```

---

# 46. Why ConvNeXt Is Important

Magnitude and SNIP pruning collapse almost completely on several ConvNeXt models.

VBP performs better, but still loses substantial accuracy.

DVBP + OB2C achieves much stronger zero-shot retention.

This demonstrates that the framework is not limited to conventional Vision Transformers.

---

# 47. Spectral Evidence for MP Noise

The paper visualizes empirical covariance eigenvalue distributions for:

- DeiT-B
- Swin-S

and fits the Marchenko-Pastur distribution to them.

The spectra show a large bulk of eigenvalues whose shape is consistent with MP-like noise.

Conceptually:

```text
Covariance Spectrum

Small Eigenvalue Bulk
→ MP-Like Noise

Large Outlying Eigenvalues
→ Signal
```

This provides empirical motivation for the spectral denoising step.

---

# 48. Raw vs. Denoised Rank

The raw-variance ranking and denoised-variance ranking remain broadly correlated.

However, the paper observes substantial deviations in the middle parts of the ranking.

This means:

```text
Denoising
does not completely redefine
neuron importance
```

but:

```text
Reorders a meaningful subset
of ambiguous neurons
```

which becomes important during aggressive pruning.

---

# 49. Layer-Wise Pruning Pattern

At 50% global pruning, DVBP and VBP produce similar broad pruning patterns.

However, DVBP + OB2C tends to:

```text
Prune Early Layers
Less Aggressively
```

and:

```text
Prune Some Deeper Layers
More Aggressively
```

This suggests that denoising changes how redundancy is distributed across network depth.

---

# 50. Ablation: Denoising vs. OB2C

The paper separately evaluates:

```text
VBP
```

```text
VBP + Denoising
```

and:

```text
VBP + Denoising + OB2C
```

At 50% pruning:

| Model | VBP | + Denoising | + Denoising + OB2C |
|---|---:|---:|---:|
| DeiT-T | 42.90 | 44.32 | **56.40** |
| DeiT-S | 66.03 | 65.37 | **70.25** |
| DeiT-B | 68.92 | 69.90 | **75.85** |
| Swin-T | 49.98 | 52.48 | **66.12** |
| Swin-S | 69.91 | 71.21 | **77.24** |
| Swin-B | 70.86 | 70.11 | **76.16** |

---

# 51. Main Driver of Accuracy Recovery

The ablation shows:

```text
Denoising
→ Usually +1 to +2 points
```

while:

```text
OB2C
→ Often +4 to +14 points
```

Therefore:

> **OB2C is the main source of performance recovery.**

Denoising improves the pruning decisions, while OB2C performs the more substantial reconstruction compensation.

---

# 52. Example: DeiT-T Ablation

At 50% pruning:

```text
VBP
42.90
```

Denoising only:

```text
44.32
```

With OB2C:

```text
56.40
```

Thus OB2C adds roughly:

```text
+12 points
```

beyond the denoised-selection stage.

---

# 53. Calibration Set Size

The paper studies calibration sizes from approximately:

```text
1K
to
65K
```

samples.

For DeiT-B:

```text
Performance improves
until around 8K samples
```

and then largely converges.

For Swin-S:

```text
Performance continues improving
until roughly 32K samples
```

---

# 54. Why More Calibration Data Helps DVBP

DVBP estimates the full covariance structure.

More samples improve estimates of:

```text
Variance
+
Cross-Neuron Covariance
+
Covariance Eigenvectors
+
Covariance Eigenvalues
```

This provides progressively better information for both:

```text
Denoised Selection
+
OB2C Recovery
```

---

# 55. Interesting Contrast with VBP

For Swin-S, the paper reports that increasing the calibration set size from approximately:

```text
1K
to
65K
```

provides little measurable improvement to VBP.

DVBP + OB2C, however, continues to improve substantially.

This is consistent with the fact that the proposed method uses richer covariance information rather than only marginal variance.

---

# 56. Fine-Tuning Experiment

Although the paper primarily targets training-free pruning, it also tests subsequent fine-tuning.

DeiT-B and Swin-S are pruned by:

```text
70%
```

using both:

- VBP
- DVBP + OB2C

and then fine-tuned for 10 epochs using the VBP training setup.

The proposed method maintains a consistent accuracy lead throughout fine-tuning.

---

# 57. Why Better Post-Pruning Accuracy Matters

The fine-tuning experiment supports:

```text
Better Initial Pruned Model
       ↓
Better Fine-Tuning Starting Point
       ↓
Better Final Model
```

Therefore, stronger zero-shot reconstruction remains useful even when fine-tuning is later allowed.

---

# 58. Denoising Ablation

The paper compares several spectral denoising strategies.

Examples include keeping:

```text
Top 50%
Top 60%
Top 70%
Top 80%
```

of eigenvectors.

The MP-based rule instead determines the threshold adaptively using:

```math
\lambda_+
```

The MP-based signal selection performs best or near-best across the evaluated aggressive pruning settings.

---

# 59. High-Pruning Example

For DeiT-B at 70% pruning:

```text
No Denoising
45.49
```

Using the MP $\lambda_+$ threshold:

```text
57.18
```

This is a:

```text
+11.69 point
```

improvement.

For Swin-S:

```text
No Denoising
37.26
```

versus:

```text
MP Denoising
57.13
```

an improvement of:

```text
+19.87 points
```

---

# 60. Raw vs. Denoised Variance at Low Pruning

At 30% pruning, raw variance often performs better.

For DeiT-B:

```text
Raw Variance
79.41
```

```text
Denoised Variance
77.45
```

For Swin-S:

```text
Raw Variance
81.70
```

```text
Denoised Variance
81.05
```

This confirms that denoising can discard useful small-signal components at low pruning ratios.

---

# 61. Raw vs. Denoised Variance at High Pruning

At 70% pruning, the trend reverses.

For DeiT-B:

```text
Raw Variance
28.04
```

```text
Denoised Variance
42.07
```

For Swin-S:

```text
Raw Variance
9.37
```

```text
Denoised Variance
35.37
```

Therefore:

```text
Aggressive Pruning
→ Noise-Robust Ranking
becomes much more important
```

---

# 62. Eigenvector Pruning by Depth

The MP denoising process removes approximately:

```text
60% ~ 70%
```

of covariance eigenvectors in some early layers.

In deeper layers, this increases to roughly:

```text
80% ~ 90%
```

for the evaluated DeiT-B and Swin-S models.

The paper interprets this as evidence that representations become increasingly low-rank deeper in the network.

---

# 63. Computational Overhead

The proposed method is more expensive than VBP because it computes:

- Full covariance matrices
- Eigendecompositions
- Matrix inverses / OBC updates

However, the reported cost remains modest.

Peak VRAM remains:

```text
< 10 GB
```

for all tested models.

---

# 64. Runtime of the Pruning Procedure

Examples:

| Model | VBP | DVBP + OB2C |
|---|---:|---:|
| DeiT-T | 14.44 s | 19.25 s |
| DeiT-S | 13.16 s | 21.79 s |
| DeiT-B | 26.66 s | 51.15 s |
| Swin-T | 15.03 s | 24.50 s |
| Swin-S | 23.88 s | 44.14 s |
| Swin-B | 34.89 s | 66.92 s |

The proposed method is roughly up to:

```text
~2× slower than VBP
```

during pruning, but still completes in around one minute for most tested architectures.

---

# 65. Memory Usage

Examples of peak VRAM:

```text
DeiT-B
VBP         = 2.01 GB
DVBP+OB2C   = 5.25 GB
```

```text
Swin-B
VBP         = 4.58 GB
DVBP+OB2C   = 8.11 GB
```

The increased cost comes from richer covariance and compensation computations.

---

# 66. DVBP + OB2C vs. VBP

The conceptual progression is:

```text
VBP

Activation Variance
      ↓
Prune Low-Variance Neurons
      ↓
Mean Shift to Bias
```

becomes:

```text
DVBP + OB2C

Full Activation Covariance
      ↓
Spectral Denoising
      ↓
Better Neuron Selection
      ↓
OBC-Based Multi-Weight Recovery
      ↓
Optimal Bias Compensation
```

---

# 67. Connection to OBC

OBC provides:

```text
Second-Order
Layer Reconstruction
```

using:

```text
Hessian-Based
Weight Compensation
```

DVBP + OB2C adapts this idea to structured neuron pruning.

The important new insight is:

```text
Optimal Bias
        ↓
Mean-Centered Activations
        ↓
Hessian = 2N × Covariance
```

This makes activation covariance the natural object connecting VBP and OBC.

---

# 68. Connection to GPTQ

GPTQ also relies on second-order reconstruction information.

However:

```text
GPTQ
→ Quantize weights
→ Error compensation

DVBP + OB2C
→ Remove complete neurons
→ Error compensation
```

Both exploit calibration activation statistics to understand how parameter perturbations affect layer outputs.

---

# 69. Variance vs. Covariance Perspective

VBP uses only:

```text
diag(C)
```

that is:

```text
Individual Neuron Variances
```

DVBP + OB2C uses the full matrix:

```text
C
```

which includes:

```text
Diagonal
→ Variance

Off-Diagonal
→ Cross-Neuron Covariance
```

This is one of the most important conceptual transitions from VBP to this paper.

---

# 70. Statistical Interpretation

A useful way to understand the progression is:

```text
VBP

"How much does this neuron vary?"
```

versus:

```text
DVBP + OB2C

"How much does this neuron vary,
which parts of that variation are signal,
and how is it correlated with
the surviving neurons?"
```

The latter provides richer information for aggressive structured pruning.

---

# 71. Main Contributions

### 1. Denoised Variance-Based Pruning

Uses random matrix theory to remove finite-sample noise from activation covariance.

### 2. Marchenko-Pastur Denoising

Fits the covariance eigenvalue spectrum and preserves signal components above the MP upper edge.

### 3. Hybrid Variance Score

Combines raw and denoised variance according to the pruning ratio.

### 4. Optimal Brain Bias Compensation

Extends the OBC reconstruction problem by explicitly optimizing the bias.

### 5. Covariance-Hessian Equivalence

Shows that after optimal mean compensation:

```math
H = 2NC
```

### 6. Multi-Weight Recovery

Updates surviving weights using the full activation covariance structure.

### 7. Training-Free Structured Compression

Achieves strong zero-shot pruning accuracy without post-pruning fine-tuning.

### 8. Broad Architecture Evaluation

Evaluated on:

- DeiT
- Swin
- ConvNeXt

---

# 72. Limitations

## Hybrid Score Is Heuristic

The score:

```math
s_{\mathrm{hybrid}}
=
(1-p)s_{\mathrm{var}}
+
p s_{\mathrm{dvar}}
```

is empirically motivated.

The paper explicitly notes that a theoretically grounded hybrid score remains future work.

---

## MP Noise Assumption

The denoising method assumes that noise approximately follows conditions compatible with Marchenko-Pastur theory.

Real neural activations are not perfectly i.i.d.

Therefore, the MP fit should be understood as a useful approximation rather than an exact generative model of network activations.

---

## More Expensive Than VBP

DVBP + OB2C requires:

- Full covariance estimation
- Eigendecomposition
- Matrix inversion
- Block-wise OBC updates

It therefore uses more time and memory than VBP.

---

## Calibration Data Is Important

The quality of:

```text
Covariance Estimation
+
Spectral Denoising
+
Weight Recovery
```

depends on the calibration dataset.

---

## Denoising Can Remove Weak Signal

At low pruning ratios, pure denoised variance can perform worse than raw variance.

This motivates the hybrid score.

---

# 73. Key Takeaways

1. DVBP + OB2C is a **training-free structured pruning method** built directly on VBP.

2. VBP has two central weaknesses: noisy finite-sample variance estimates and bias-only compensation.

3. DVBP addresses the first problem using the full activation covariance spectrum.

4. Random Matrix Theory and the Marchenko-Pastur distribution are used to distinguish signal eigenvalues from noise-like eigenvalues.

5. The covariance matrix is reconstructed using only signal eigencomponents.

6. Denoised neuron variance is obtained from the diagonal of this reconstructed covariance.

7. Raw variance works better at modest pruning ratios, while denoised variance becomes much more useful under aggressive pruning.

8. The paper combines the two using a pruning-ratio-dependent hybrid score.

9. OB2C addresses VBP's second weakness by optimizing both surviving weights and bias.

10. The optimal bias is:

```math
b^*
=
(W-\hat{W})\mu
```

11. After substituting this optimal bias, the reconstruction objective depends only on mean-centered activations.

12. The resulting layer-wise Hessian becomes:

```math
H = 2NC
```

where $C$ is the activation covariance matrix.

13. The same covariance therefore supports both neuron selection and OBC-style weight recovery.

14. OB2C is the main contributor to the accuracy improvement in the ablation experiments.

15. Denoising becomes particularly important as the pruning ratio becomes aggressive.

16. At 50% pruning, DeiT-B improves from **68.92% with VBP to 75.85% with DVBP + OB2C** without fine-tuning.

17. Swin-S improves from **69.91% to 77.24%** at the same pruning ratio.

18. At 60% pruning, the Swin-S gap grows to **25.49 percentage points**.

19. On ConvNeXt-T at 50% pruning, the method improves over VBP by **29.46 percentage points**.

20. The method demonstrates that structured pruning can benefit substantially from combining statistical denoising with second-order reconstruction.

---

# 74. Concepts to Review

- Structured Pruning
- Variance-Based Pruning
- Activation Mean
- Activation Variance
- Covariance
- Covariance Matrix
- Mean-Centered Data
- Eigendecomposition
- Eigenvalue Spectrum
- Eigenvectors
- Random Matrix Theory
- Marchenko-Pastur Distribution
- Spectral Density
- Low-Rank Signal
- Noise Eigenvalues
- Signal Eigenvalues
- Covariance Denoising
- Sample Covariance
- Online Covariance Update
- Optimal Brain Surgeon
- Optimal Brain Compression
- Hessian
- Inverse Hessian
- Block-Wise Weight Update
- Bias Compensation
- Calibration Dataset
- Structured MLP Pruning
