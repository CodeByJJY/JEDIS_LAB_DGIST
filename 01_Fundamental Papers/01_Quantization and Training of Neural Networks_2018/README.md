# Quantization and Training of Neural Networks

> **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**

- **Authors:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko
- **Topic:** Neural Network Quantization
- **Keywords:** INT8, Integer-only Inference, Quantization-Aware Training, Scale, Zero-Point, Mobile Inference

---

## 1. Problem Background

Modern CNNs achieve high accuracy but require substantial computation and memory, making deployment on mobile devices difficult.

Two major approaches existed:

1. Design more efficient neural network architectures
2. Quantize weights and activations to lower precision

However, previous quantization methods had several limitations:

- Some quantized only the weights and mainly reduced model storage.
- Some used extremely low precision such as binary or ternary values, causing accuracy degradation.
- Many approaches did not demonstrate actual latency improvements on commonly available hardware.
- Simple post-training quantization could significantly degrade the accuracy of small and efficient networks such as MobileNet.

The goal of this paper is therefore to achieve:

```text
Low Precision
     +
High Accuracy
     +
Real Hardware Speedup
```

---

## 2. Main Idea

The paper proposes two components that are co-designed together:

```text
1. Integer-Arithmetic-Only Inference
                 +
2. Training with Simulated Quantization
```

The inference system uses mainly:

- 8-bit integer weights
- 8-bit integer activations
- 32-bit integer accumulators
- 32-bit integer biases

During training, the effects of quantization are simulated so that the network can adapt to the errors introduced by low-precision inference.

---

# 3. Quantization Scheme

The paper represents a real value using an affine mapping between an integer value and a real value.

```math
r = S(q - Z)
```

where:

- $r$: real value
- $q$: quantized integer value
- $S$: scale
- $Z$: zero-point

---

## Scale

$S$ determines the distance between adjacent quantized values.

A smaller scale provides finer numerical resolution over a smaller range.

---

## Zero-Point

$Z$ is the integer value that represents the real value zero.

```math
r = 0
\quad \Longleftrightarrow \quad
q = Z
```

This is particularly useful because neural network operators frequently rely on zero values, for example:

- Zero padding
- ReLU
- Sparse regions

The quantization scheme therefore guarantees that real zero can be represented exactly.

---

# 4. Integer-Arithmetic-Only Inference

The main goal is to perform neural network inference without floating-point arithmetic in the main computation path.

For matrix multiplication, the real-valued computation can be transformed into operations on quantized integers.

The core computation becomes:

```text
8-bit Weight
      ×
8-bit Activation
      ↓
32-bit Accumulator
```

In the implementation described in the paper:

```text
uint8 × uint8 → int32 accumulation
```

The wider accumulator prevents overflow while summing many products.

---

## Bias

Bias values are stored with higher precision.

```text
Weights      → 8 bit
Activations  → 8 bit
Bias         → 32 bit
Accumulator  → 32 bit
```

The paper argues that bias errors can systematically affect many output activations, making higher bias precision important for maintaining accuracy.

---

## Requantization

After accumulation, the 32-bit result must be converted back into the 8-bit representation used for the next layer.

Conceptually:

```text
INT8 Input
    ↓
INT8 × INT8
    ↓
INT32 Accumulation
    ↓
INT32 Bias Addition
    ↓
Rescaling
    ↓
Clamp
    ↓
INT8 Output
```

The required scale multiplier is implemented using fixed-point multiplication and bit shifting so that inference can remain integer-only.

---

# 5. Fused Layer Execution

The paper describes a typical quantized convolution layer as:

```text
INT8 Input
      │
      ▼
Convolution with INT8 Weights
      │
      ▼
INT32 Accumulator
      │
      ▼
INT32 Bias Addition
      │
      ▼
Requantization
      │
      ▼
Activation Function
      │
      ▼
INT8 Output
```

Operations such as ReLU and ReLU6 can be implemented as simple clamping operations in the quantized domain.

---

# 6. Training with Simulated Quantization

Simple post-training quantization works reasonably well for large models but can cause significant accuracy degradation for smaller models.

The paper therefore simulates quantization during training.

```text
Floating-Point Parameters
        │
        ▼
Fake Quantization
        │
        ▼
Quantized-like Forward Pass
        │
        ▼
Loss
        │
        ▼
Normal Backpropagation
```

Important distinction:

```text
Forward Pass
→ Simulates quantization

Backward Pass
→ Conventional floating-point training

Stored Parameters
→ Floating point
```

The network can therefore adapt its weights to quantization noise while retaining the ability to make small parameter updates during training.

---

# 7. Weight and Activation Quantization

Fake quantization is inserted at the locations where actual quantization will occur during inference.

## Weights

Weights are quantized before convolution or matrix multiplication.

The quantization range is determined mainly from:

```text
min(weight)
~
max(weight)
```

---

## Activations

Activation ranges depend on the input data and therefore vary during training.

The paper estimates activation ranges using:

```text
Observed Activation Range
        ↓
Exponential Moving Average
        ↓
Quantization Range
```

Activation quantization may also be delayed during the early stages of training so that the network first reaches a more stable state.

---

# 8. Batch Normalization Folding

During training, Batch Normalization may appear as a separate operation.

During inference, however, Batch Normalization is typically folded into the preceding convolution.

Conceptually:

```text
Convolution
     +
Batch Normalization
     ↓
Equivalent Convolution
with Modified Weights and Bias
```

Therefore, the paper performs quantization **after accounting for Batch Normalization folding**.

This is important because the inference-time weights are different from the original training-time convolution weights.

---

# 9. Experimental Results

The paper evaluates the method on:

- ResNet
- Inception v3
- MobileNet
- MobileNet SSD

Tasks include:

- ImageNet classification
- COCO object detection
- Face detection
- Face attribute classification

---

## ResNet on ImageNet

| Model | Floating Point | Integer Quantized |
|---|---:|---:|
| ResNet-50 | 76.4% | 74.9% |
| ResNet-100 | 78.0% | 76.6% |
| ResNet-150 | 78.8% | 76.7% |

The integer-quantized models remain within roughly 2 percentage points of the floating-point models.

---

## ResNet-50 Quantization Comparison

| Method | Weight Bits | Activation Bits | Accuracy |
|---|---:|---:|---:|
| BWN | 1 | FP32 | 68.7% |
| TWN | 2 | FP32 | 72.5% |
| INQ | 5 | FP32 | 74.8% |
| FGQ | 2 | 8 | 70.8% |
| **Proposed Method** | **8** | **8** | **74.9%** |

A major distinction is that the proposed method quantizes both weights and activations while supporting practical integer-only inference.

---

# 10. MobileNet Inference

A central contribution of the paper is evaluating quantization on models that are already optimized for mobile inference.

The experiments use Qualcomm Snapdragon CPUs.

The paper compares models based on:

```text
Accuracy
vs.
Actual Device Latency
```

rather than evaluating only the accuracy drop of a fixed architecture.

Integer-only MobileNets generally achieve better accuracy for the same latency budget than floating-point MobileNets.

---

# 11. COCO Object Detection

The paper evaluates MobileNet SSD using floating-point and 8-bit integer inference.

| Depth Multiplier | Type | mAP | LITTLE Core | Big Core |
|---|---|---:|---:|---:|
| 100% | FP32 | 22.1 | 778 ms | 370 ms |
| 100% | INT8 | 21.7 | 687 ms | 272 ms |
| 50% | FP32 | 16.7 | 270 ms | 121 ms |
| 50% | INT8 | 16.6 | 146 ms | 61 ms |

For the 50% MobileNet SSD:

```text
121 ms → 61 ms
```

on a Snapdragon 835 big core, while mAP changes only from:

```text
16.7 → 16.6
```

The paper reports up to approximately **50% reduction in runtime** with minimal accuracy degradation.

---

# 12. Effect of Bit Width

The paper also studies different weight and activation bit widths.

The main observations are:

1. **Weights are generally more sensitive to lower precision than activations.**
2. 7-bit and 8-bit models perform relatively close to floating-point models.
3. When the total bit budget is similar, keeping weight and activation precision balanced tends to work better.
4. Very aggressive 4-bit quantization causes significant accuracy degradation in the evaluated models.

---

# 13. ReLU vs. ReLU6

The experiments show that quantized models using ReLU6 suffer less accuracy degradation than models using unrestricted ReLU.

Reason:

```text
ReLU
→ activation range can become very large

ReLU6
→ activation range is bounded to [0, 6]
```

A bounded range is easier to represent accurately with a fixed number of quantization levels.

---

# 14. Key Contributions

The paper's main contributions can be summarized as:

### 1. Practical INT8 Quantization

```text
Weights     → INT8
Activations → INT8
Bias        → INT32
```

### 2. Integer-Only Inference

The primary inference path avoids floating-point arithmetic.

### 3. Quantization-Aware Training

Quantization effects are simulated during training to recover accuracy.

### 4. Real Hardware Evaluation

Performance is measured on actual ARM CPUs rather than inferred only from theoretical operation counts.

### 5. Accuracy-Latency Evaluation

The paper emphasizes the practical trade-off between:

```text
Accuracy
↕
Inference Latency
```

---

# 15. Important Observations

## Quantization Is Not Only Model Compression

Reducing numerical precision can improve:

- Model size
- Memory traffic
- Arithmetic efficiency
- Inference latency

provided that the target hardware efficiently supports low-precision arithmetic.

---

## Hardware Support Matters

The benefit of INT8 inference depends strongly on the processor.

The paper observes that the latency improvement differs across different Snapdragon microarchitectures because their relative floating-point and integer performance differs.

Therefore:

```text
Lower Bit Width
≠
Automatic Speedup
```

Actual speedup depends on hardware support.

---

## Training and Inference Must Match

The training graph should simulate the numerical behavior of the eventual inference graph.

This motivates:

- Fake quantization
- BatchNorm folding
- Activation range estimation
- Matching quantization locations

---

# 16. Key Takeaways

1. Both weights and activations can be quantized to 8-bit integers while retaining useful model accuracy.

2. Quantized inference can be implemented primarily using integer arithmetic.

3. The affine quantization equation

```math
r = S(q-Z)
```

provides the basis for mapping between real and integer values.

4. **Scale** determines numerical resolution.

5. **Zero-point** allows real zero to be represented exactly.

6. INT8 multiplication uses wider INT32 accumulation.

7. Simulating quantization during training significantly improves post-quantization accuracy.

8. Quantization should be evaluated using actual hardware latency, not only model size or theoretical operation counts.

9. Hardware support determines whether lower numerical precision translates into real inference acceleration.

---

# 17. Concepts to Review

- Fixed-Point Arithmetic
- INT8 / INT32
- Quantization
- Scale
- Zero-Point
- Quantization Error
- Quantization-Aware Training
- Fake Quantization
- Requantization
- Saturation / Clamping
- Batch Normalization Folding
- GEMM
- SIMD
- ARM NEON
- MobileNet
- Latency vs. Accuracy
