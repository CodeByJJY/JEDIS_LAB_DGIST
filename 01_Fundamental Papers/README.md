# Fundamental Papers

This section summarizes fundamental papers on neural network compression, quantization, and pruning.

Each paper is organized into three points:

- **Problem Background**
- **Main Idea**
- **Experimental Results**

---

## 00. Deep Compression

**Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding**

- **Problem Background:** Large neural networks require substantial storage, memory bandwidth, and energy, making deployment on resource-constrained devices difficult.
- **Main Idea:** Apply a three-stage compression pipeline consisting of network pruning, trained quantization with weight sharing, and Huffman coding.
- **Experimental Results:** AlexNet is compressed by 35× and VGG-16 by 49× without accuracy loss, while compressed models achieve 3–4× layer-wise speedup and 3–7× better energy efficiency.

---

## 01. Quantization and Training of Neural Networks

**Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**

- **Problem Background:** Existing CNNs are difficult to deploy efficiently on mobile hardware, and many quantization methods either lose accuracy or fail to provide measurable hardware speedups.
- **Main Idea:** Quantize both weights and activations to 8-bit integers using scale and zero-point, while simulating quantization during training to preserve model accuracy.
- **Experimental Results:** The proposed integer-only inference scheme improves the latency–accuracy trade-off of MobileNet-based models on ImageNet and COCO using common ARM CPUs.

---

## 02. AdaRound

**Up or Down? Adaptive Rounding for Post-Training Quantization**

- **Problem Background:** Conventional post-training quantization uses round-to-nearest, although minimizing individual weight error does not necessarily minimize the overall task loss.
- **Main Idea:** Formulate weight rounding as a layer-wise optimization problem and learn whether each weight should be rounded up or down using a soft relaxation of a second-order loss approximation.
- **Experimental Results:** AdaRound achieves state-of-the-art PTQ performance and quantizes ResNet-18 and ResNet-50 weights to 4 bits with less than 1% accuracy loss without fine-tuning.

---

## 03. LLM.int8()

**LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**

- **Problem Background:** Standard INT8 quantization fails for large Transformers because a small number of systematic high-magnitude activation outliers strongly affect model performance.
- **Main Idea:** Combine vector-wise INT8 quantization for normal features with mixed-precision decomposition that processes outlier feature dimensions in FP16.
- **Experimental Results:** LLM.int8() enables inference of models up to 175B parameters with no performance degradation while reducing inference memory requirements by approximately half.

---

## 04. Optimal Brain Compression

**Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning**

- **Problem Background:** Post-training pruning and quantization are usually handled independently and often struggle to preserve accuracy without retraining.
- **Main Idea:** Extend the Optimal Brain Surgeon framework into a unified post-training compression method that greedily prunes or quantizes weights while optimally updating the remaining weights.
- **Experimental Results:** OBC improves compression–accuracy trade-offs for both pruning and quantization and demonstrates compound compression with up to 12× theoretical operation reduction at about 2% accuracy loss and 4× CPU runtime speedup at about 1% accuracy loss.

---

## 05. GPTQ

**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**

- **Problem Background:** Existing accurate PTQ methods are too computationally expensive to scale to LLMs containing tens or hundreds of billions of parameters.
- **Main Idea:** Build on Optimal Brain Quantization using approximate second-order information and efficient weight updates to perform scalable one-shot low-bit weight quantization.
- **Experimental Results:** GPTQ quantizes OPT-175B and BLOOM-176B to 3–4 bits in roughly four GPU hours with negligible accuracy degradation and achieves about 3.25× speedup on A100 and 4.5× on A6000 GPUs.

---

## 06. SmoothQuant

**SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**

- **Problem Background:** LLM activation outliers make activation quantization substantially harder than weight quantization, preventing efficient and accurate W8A8 inference.
- **Main Idea:** Apply a mathematically equivalent per-channel scaling transformation that migrates quantization difficulty from activations to weights and smooths activation outliers.
- **Experimental Results:** SmoothQuant enables W8A8 quantization with negligible accuracy loss, achieving up to 1.56× inference speedup and approximately 2× memory reduction.

---

## 07. AWQ

**AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration**

- **Problem Background:** Low-bit weight-only PTQ can substantially reduce LLM memory usage, but aggressive quantization causes accuracy degradation and reconstruction-based methods may overfit calibration data.
- **Main Idea:** Use activation statistics to identify salient weight channels and protect them through per-channel scaling instead of hardware-inefficient mixed-precision quantization.
- **Experimental Results:** AWQ improves low-bit quantization accuracy across LLMs and multimodal models, while the TinyChat inference system achieves approximately 3.2–3.3× speedup over FP16 implementations on desktop, laptop, and mobile GPUs.

---

## 08. QuaRot

**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**

- **Problem Background:** Activation outliers make end-to-end 4-bit quantization of LLM weights, activations, and KV cache difficult without retaining some values in higher precision.
- **Main Idea:** Apply randomized Hadamard rotations using computational invariance to remove outliers while preserving the full-precision model output.
- **Experimental Results:** QuaRot enables 4-bit weight, activation, and KV-cache quantization while retaining 99% of zero-shot performance, with up to 3.33× prefill speedup and 3.89× decoding-stage memory savings on LLaMA2-70B.

---

## 09. SpinQuant

**SpinQuant: LLM Quantization with Learned Rotations**

- **Problem Background:** Random rotations reduce quantization outliers, but different random rotations can produce large variations in the final accuracy of quantized LLMs.
- **Main Idea:** Learn rotation matrices directly for quantization accuracy while preserving the full-precision network output, rather than relying on randomly selected rotations.
- **Experimental Results:** Under W4A4KV4 quantization, SpinQuant reduces the zero-shot accuracy gap from full precision to 2.9 points on LLaMA-2 7B and consistently outperforms random-rotation methods such as QuaRot.

---

## 10. Variance-Based Pruning

**Variance-Based Pruning for Accelerating and Compressing Trained Networks**

- **Problem Background:** Structured pruning can directly reduce model size and computation, but often causes substantial accuracy degradation and requires expensive retraining.
- **Main Idea:** Identify low-importance MLP neurons using activation variance and compensate for removed neurons by shifting their mean contribution into the bias of the following layer.
- **Experimental Results:** On DeiT-Base, VBP reduces MACs by 35% and model size by 36%, achieves 1.44× speedup, and recovers 99% of the original accuracy with only 10 epochs of fine-tuning.

---

## 11. Denoised Variance-Based Pruning with Optimal Brain Bias Compensation

**Denoised Variance-Based Pruning with Optimal Brain Bias Compensation**

- **Problem Background:** VBP suffers from statistical noise in finite-sample covariance estimates and its bias-only compensation cannot fully recover reconstruction errors after structured pruning.
- **Main Idea:** Denoise the activation covariance spectrum using random matrix theory and combine variance-based neuron selection with OBC-style optimal weight recovery through Optimal Brain Bias Compensation.
- **Experimental Results:** At 50% MLP pruning, DVBP + OB2C retains over 90% of the original Top-1 accuracy on Small and Base models and improves over VBP by up to 29.46 percentage points on ConvNeXt-T and 7.33 points on Swin-S.

---
