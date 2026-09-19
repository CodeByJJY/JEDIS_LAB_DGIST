# AdaRound

> **Up or Down? Adaptive Rounding for Post-Training Quantization**

- **Authors:** Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort
- **Venue:** ICML 2020
- **Topic:** Post-Training Quantization
- **Keywords:** PTQ, Adaptive Rounding, Weight Quantization, Reconstruction Error, Hessian, QUBO

---

## 1. Problem Background

Post-Training Quantization (PTQ) quantizes a pretrained neural network without full retraining.

Compared with Quantization-Aware Training (QAT), PTQ is attractive because it:

- Does not require expensive retraining
- Requires little calibration data
- Can be applied during deployment
- Has relatively low optimization cost

However, aggressive low-bit quantization often causes significant accuracy degradation.

A common weight quantization method is **rounding-to-nearest**:

```text
Floating-Point Weight
        ↓
Nearest Quantization Grid Point
        ↓
Quantized Weight
```

The intuition is simple:

> Choose the quantized value that minimizes the numerical error of each individual weight.

AdaRound questions this assumption.

The main observation is:

> Minimizing each individual weight error does not necessarily minimize the final task loss.

---

# 2. Why Rounding-to-Nearest Can Be Suboptimal

Consider a pretrained network with weights $\mathbf{w}$.

Quantization introduces a perturbation:

```math
\Delta \mathbf{w}
=
\hat{\mathbf{w}} - \mathbf{w}
```

where $\hat{\mathbf{w}}$ is the quantized weight vector.

Using a second-order Taylor approximation, the change in task loss can be approximated as:

```math
\Delta L
\approx
\Delta \mathbf{w}^{T} \mathbf{g}
+
\frac{1}{2}
\Delta \mathbf{w}^{T}
H
\Delta \mathbf{w}
```

where:

- $\mathbf{g}$ is the gradient of the task loss
- $H$ is the Hessian matrix

For a model trained close to convergence:

```math
\mathbf{g} \approx 0
```

so the dominant term becomes:

```math
\Delta L
\approx
\frac{1}{2}
\Delta \mathbf{w}^{T}
H
\Delta \mathbf{w}
```

The important point is that the Hessian contains **off-diagonal terms**.

These terms describe interactions between different weight perturbations.

Therefore:

```text
Rounding Error of Weight 1
            +
Rounding Error of Weight 2
            ↓
Joint Effect on Task Loss
```

Rounding each weight independently to its nearest value ignores these interactions.

---

# 3. Evidence Against Nearest Rounding

The paper quantizes only the first layer of ResNet-18 to 4 bits and compares different rounding strategies.

| Rounding Scheme | Accuracy |
|---|---:|
| Nearest | 52.29% |
| Ceil | 0.10% |
| Floor | 0.10% |
| Stochastic | 52.06 ± 5.52% |
| Best Stochastic Sample | 63.06% |

Among 100 stochastic rounding choices:

- 48 outperform rounding-to-nearest
- The best sampled solution improves accuracy by more than 10 percentage points over nearest rounding

This suggests that:

> There are many rounding configurations that are better than independently rounding every weight to the nearest grid point.

---

# 4. Rounding as an Optimization Problem

For each weight, AdaRound considers two possible quantized values:

```text
Round Down
or
Round Up
```

For a weight $w_i$:

```math
\hat{w}_i
\in
\left\{
w_i^{\mathrm{floor}},
w_i^{\mathrm{ceil}}
\right\}
```

The goal is to choose the rounding direction for all weights so that the increase in task loss is minimized.

Using the second-order approximation, this becomes a:

> **Quadratic Unconstrained Binary Optimization (QUBO)** problem.

Conceptually:

```text
Each Weight
   ↓
Binary Decision

0 → Round Down
1 → Round Up

        ↓
Choose all decisions jointly
        ↓
Minimize task-loss increase
```

---

# 5. Why Direct Hessian Optimization Is Difficult

The theoretically motivated objective depends on the Hessian:

```math
\Delta \mathbf{w}^{T}
H
\Delta \mathbf{w}
```

However, directly optimizing this is impractical because:

1. The Hessian is extremely large.
2. Computing and storing it is expensive.
3. The binary optimization problem is NP-hard.
4. The complexity rapidly increases with layer size.

AdaRound therefore simplifies the objective.

---

# 6. From Task Loss to Local Reconstruction Loss

The paper makes assumptions about the Hessian with respect to layer preactivations.

Under these assumptions, the original second-order task-loss objective can be simplified to a layer-wise reconstruction objective.

For a layer with weight matrix $W$ and input $x$:

```text
Original Layer Output
        =
       Wx

Quantized Layer Output
        =
       W_hat x
```

The local objective becomes approximately:

```math
\left\|
Wx - \hat{W}x
\right\|_F^2
```

In other words:

> Choose rounding directions so that the output of the quantized layer remains as close as possible to the output of the original layer.

This removes the need to explicitly compute the full Hessian or the final task loss.

---

# 7. AdaRound

The local reconstruction objective is still a discrete optimization problem because each weight must eventually be rounded either up or down.

AdaRound solves this using a **continuous relaxation**.

Instead of immediately choosing:

```text
0 → Round Down
1 → Round Up
```

AdaRound introduces a continuous variable:

```text
0 ≤ h(V) ≤ 1
```

The soft-quantized weight is defined conceptually as:

```math
\tilde{W}
=
s
\left(
\left\lfloor
\frac{W}{s}
\right\rfloor
+
h(V)
\right)
```

where:

- $s$ is the quantization scale
- $h(V)$ determines the rounding direction

If:

```text
h(V) → 0
```

the weight is rounded down.

If:

```text
h(V) → 1
```

the weight is rounded up.

---

# 8. Rectified Sigmoid

AdaRound uses a rectified sigmoid function to parameterize $h(V)$.

Conceptually:

```text
Continuous Variable V
        ↓
Rectified Sigmoid
        ↓
h(V) between 0 and 1
        ↓
Soft Rounding Decision
```

During optimization, $h(V)$ can initially move freely.

Later, it is encouraged to converge toward:

```text
0 or 1
```

so that the final solution becomes a valid discrete rounding decision.

---

# 9. Rounding Regularization

The AdaRound objective consists of two parts:

```text
Reconstruction Loss
        +
Rounding Regularization
```

Conceptually:

```math
L_{\mathrm{AdaRound}}
=
\left\|
Wx-\tilde{W}x
\right\|_F^2
+
\lambda L_{\mathrm{reg}}
```

The reconstruction term tries to preserve layer outputs.

The regularization term encourages:

```text
h(V) → 0
or
h(V) → 1
```

During early optimization:

```text
Rounding variables can move freely
```

During later optimization:

```text
Rounding variables are pushed toward binary decisions
```

This allows AdaRound to search the rounding space using gradient-based optimization.

---

# 10. Asymmetric Reconstruction

Optimizing each layer independently introduces another problem.

When quantizing a deep network layer by layer:

```text
Layer 1 quantized
      ↓
Error propagates
      ↓
Layer 2 receives already-quantized input
      ↓
Additional error
      ↓
...
```

To account for accumulated errors from previous layers, AdaRound uses **asymmetric reconstruction**.

The original branch uses the original input:

```text
Original Weight + Original Input
```

while the quantized branch uses the input produced by previously quantized layers:

```text
Quantized Weight + Quantized Input
```

The objective also includes the activation function.

Conceptually:

```text
Original:
f(Wx)

Quantized:
f(W_hat x_hat)

        ↓
Minimize Difference
```

This significantly improves whole-network quantization performance.

---

# 11. AdaRound Procedure

The overall procedure is:

```text
Pretrained FP32 Model
        │
        ▼
Choose Quantization Grid
        │
        ▼
Collect Small Unlabelled Dataset
        │
        ▼
Process Layer by Layer
        │
        ├─ Original Layer Output
        │
        ├─ Quantized Layer Output
        │
        └─ Optimize Rounding Variables
        │
        ▼
Force Rounding Decisions to 0 / 1
        │
        ▼
Quantized Model
```

Important characteristics:

- No end-to-end fine-tuning
- Small amount of unlabeled data
- Layer-wise optimization
- Weight rounding is optimized
- Applicable to convolutional and fully connected layers

---

# 12. Experimental Setup

The main experiments use:

- ImageNet
- ResNet-18
- ResNet-50
- InceptionV3
- MobileNetV2

Default AdaRound configuration:

- 4-bit weight quantization
- FP32 activations unless otherwise specified
- 1024 unlabeled ImageNet images
- Adam optimizer
- 10,000 optimization iterations
- Batch size 32

For ResNet-18, the paper reports that AdaRound optimization takes approximately:

```text
10 minutes
```

on a single NVIDIA GTX 1080 Ti.

---

# 13. Ablation: From Nearest Rounding to AdaRound

For ResNet-18 with 4-bit weight quantization:

| Method | First Layer | All Layers |
|---|---:|---:|
| Nearest | 52.29% | 23.99% |
| Hessian Task-Loss Optimization | 68.62% | N/A |
| Local MSE Loss | 69.39% | 65.83% |
| Continuous Relaxation | 69.58% | 66.56% |

The results show that:

1. Nearest rounding performs poorly at 4 bits.
2. Task-loss-aware rounding dramatically improves accuracy.
3. Local reconstruction loss works well despite the approximations.
4. Continuous relaxation makes optimization practical.

---

# 14. Effect of Asymmetric Reconstruction

For ResNet-18:

| Optimization | Accuracy |
|---|---:|
| Layer-wise | 66.56% |
| Asymmetric Reconstruction | 68.37% |
| Asymmetric + ReLU | 68.60% |

Therefore:

```text
Local Reconstruction
        ↓
Asymmetric Reconstruction
        ↓
Include Activation Function
        ↓
Better Accuracy
```

---

# 15. Comparison with Straight-Through Estimator

The paper also compares AdaRound with optimizing quantized weights using the Straight-Through Estimator (STE).

| Method | Accuracy |
|---|---:|
| Nearest | 23.99% |
| STE | 66.63% |
| AdaRound | 68.60% |

AdaRound performs better despite restricting each weight to the two neighboring quantization grid values.

---

# 16. ImageNet Results

## 4-bit Weights / FP32 Activations

| Model | FP32 | Nearest | AdaRound |
|---|---:|---:|---:|
| ResNet-18 | 69.68% | 23.99% | **68.71%** |
| ResNet-50 | 76.07% | 35.60% | **75.23%** |
| InceptionV3 | 77.40% | 1.67% | **75.76%** |
| MobileNetV2 | 71.72% | 8.09% | **69.78%** |

A key result is:

> ResNet-18 and ResNet-50 can be quantized to 4-bit weights while remaining within approximately 1 percentage point of FP32 accuracy.

---

# 17. Weight + Activation Quantization

The paper also evaluates:

```text
Weights     → 4 bit
Activations → 8 bit
```

Results:

| Model | FP32 | AdaRound W4A8 |
|---|---:|---:|
| ResNet-18 | 69.68% | 68.55% |
| ResNet-50 | 76.07% | 75.01% |
| InceptionV3 | 77.40% | 75.72% |
| MobileNetV2 | 71.72% | 69.25% |

Activation quantization causes relatively little additional degradation in these experiments.

---

# 18. How Much Calibration Data Is Needed?

AdaRound requires only a small amount of unlabeled data.

The paper shows that even:

```text
256 images
```

are sufficient to obtain performance within roughly 2% of the original FP32 ResNet-18 accuracy.

The calibration images do not necessarily have to come from the exact original training dataset.

Experiments using:

- ImageNet
- Pascal VOC
- MS COCO

show similar performance.

This means AdaRound mainly needs data with sufficiently similar input statistics rather than labels from the original training set.

---

# 19. Semantic Segmentation

The paper also evaluates AdaRound on:

- DeeplabV3+
- MobileNetV2 backbone
- Pascal VOC

Results:

| Method | W/A | mIOU |
|---|---:|---:|
| FP32 | 32/32 | 72.94 |
| DFQ | 8/8 | 72.33 |
| Nearest | 4/8 | 6.09 |
| DFQ | 4/8 | 14.45 |
| AdaRound | 4/32 | **70.89** |
| AdaRound | 4/8 | **70.86** |

This demonstrates that AdaRound is not limited to image classification.

---

# 20. Key Contributions

### 1. Questions Rounding-to-Nearest

The closest quantized value is not necessarily the best choice for minimizing network loss.

### 2. Formulates Rounding Using Task Loss

The effect of quantization is analyzed using a second-order Taylor expansion and the Hessian.

### 3. Converts the Problem into Local Reconstruction

A computationally expensive task-loss objective is simplified into a layer-wise output reconstruction problem.

### 4. Introduces Adaptive Rounding

Each weight learns whether it should be rounded:

```text
Up
or
Down
```

### 5. Uses Continuous Relaxation

The binary optimization problem becomes differentiable and practical to solve.

### 6. Requires No End-to-End Fine-Tuning

Only a small amount of unlabeled calibration data is required.

---

# 21. Key Takeaways

1. **Rounding is an optimization problem.**

   Quantization is not only about choosing a scale and bit width.

2. **Nearest rounding minimizes individual weight error, not task loss.**

3. Weight perturbations interact with each other through the curvature of the loss function.

4. The theoretically motivated Hessian objective can be approximated by a layer-wise reconstruction loss.

5. AdaRound optimizes whether each weight should be rounded up or down.

6. Continuous relaxation allows this binary optimization problem to be solved with gradient-based methods.

7. Asymmetric reconstruction helps compensate for quantization error accumulated in previous layers.

8. AdaRound achieves strong 4-bit PTQ accuracy without full network retraining.

---

# 22. Connection to Previous Paper

The previous quantization paper focused on:

```text
Quantization-Aware Training
        ↓
Simulate Quantization During Training
        ↓
Train Network to Adapt
```

AdaRound instead focuses on:

```text
Already-Trained Model
        ↓
No Full Retraining
        ↓
Optimize Rounding Decisions
        ↓
Post-Training Quantization
```

Therefore:

```text
QAT
→ Modify training to tolerate quantization

AdaRound
→ Modify rounding after training
```

---

# 23. Connection to Later Papers

AdaRound introduces an important idea that appears repeatedly in later PTQ research:

> Preserve the behavior of the original layer by minimizing reconstruction error using a small calibration dataset.

This leads naturally to later methods that also use:

- Layer-wise reconstruction
- Calibration data
- Second-order information
- Quantization-error compensation

especially:

```text
AdaRound
    ↓
Optimal Brain Compression
    ↓
GPTQ
```

---

# 24. Concepts to Review

- Post-Training Quantization
- Quantization Grid
- Rounding-to-Nearest
- Floor / Ceil
- Quantization Error
- Taylor Expansion
- Gradient
- Hessian
- Quadratic Form
- QUBO
- Frobenius Norm
- Reconstruction Error
- Continuous Relaxation
- Sigmoid
- Regularization
- Calibration Data
- Asymmetric Reconstruction
