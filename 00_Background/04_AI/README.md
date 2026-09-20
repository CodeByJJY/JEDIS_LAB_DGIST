# AI

AI 모델을 공부하면서 필요한 기본 개념을 정리하고, 주요 모델을 직접 from-scratch로 구현해보기 위한 디렉토리이다.

논문이나 오픈소스 코드를 단순히 따라가는 것이 아니라,

- 각 모델이 왜 등장했는지
- 이전 방법의 어떤 문제를 해결했는지
- 내부에서 어떤 연산이 수행되는지
- Tensor shape이 어떻게 변화하는지
- 실제 계산이 코드에서 어떻게 구현되는지

를 이해하는 것을 목표로 한다.

특히 CNN, RNN/LSTM, Transformer, VAE/VQ-VAE, Autoregressive Modeling, Diffusion 등의 기초를 순서대로 공부하고, 최종적으로 VAR, Infinity와 같은 Visual Generative Model을 직접 이해하고 구현할 수 있는 기반을 만드는 것이 목적이다.

각 디렉토리에서는 다음 두 가지를 중심으로 정리한다.

1. `README.md`
   - 핵심 개념
   - 연구 배경 및 motivation
   - 주요 수식
   - 모델 구조
   - 이전/이후 방법과의 연결

2. From-scratch Implementation
   - 가능한 한 고수준 모듈에 의존하지 않고 직접 구현
   - 작은 matrix와 tensor를 이용해 중간 계산 확인
   - Tensor shape 및 intermediate output 확인
   - 구현 결과 및 관찰 내용 정리


## Contents

### [00_CNN](./00_CNN/)

Convolutional Neural Network의 기본 원리와 구조를 공부한다.

- Convolution
- Kernel / Filter
- Stride / Padding
- Pooling
- Feature Map
- 대표적인 CNN 구조
- CNN from scratch


### [01_RNN_LSTM](./01_RNN_LSTM/)

Sequence modeling을 위한 RNN과 LSTM의 기본 원리를 공부한다.

- Sequence Modeling
- Hidden State
- RNN
- Vanishing Gradient
- LSTM
- GRU
- RNN/LSTM from scratch


### [02_Attention_Transformer](./02_Attention_Transformer/)

Attention과 Transformer의 내부 연산을 공부한다.

- Query / Key / Value
- Scaled Dot-Product Attention
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- RoPE
- Causal Mask
- Cross Attention
- Transformer Block
- KV Cache
- Transformer from scratch


### [03_AutoEncoder_VAE_VQVAE](./03_AutoEncoder_VAE_VQVAE/)

이미지를 latent representation으로 표현하는 방법을 공부한다.

- AutoEncoder
- Latent Space
- Variational AutoEncoder
- Reparameterization Trick
- Vector Quantization
- Codebook
- VQ-VAE
- Quantization Error
- Visual Tokenizer
- VQ-VAE from scratch


### [04_Autoregressive](./04_Autoregressive/)

Autoregressive Modeling의 기본 원리를 공부한다.

- Autoregressive Factorization
- Next-Token Prediction
- Teacher Forcing
- Causal Modeling
- Training vs Inference
- Error Accumulation
- Autoregressive generation from scratch


### [05_Diffusion](./05_Diffusion/)

Diffusion 기반 generative model의 기본 원리를 공부한다.

- Forward Diffusion
- Reverse Process
- Noise Prediction
- DDPM
- Sampling
- Latent Diffusion
- Diffusion Transformer (DiT)
- Diffusion from scratch


### [06_Visual_Autoregressive](./06_Visual_Autoregressive/)

Autoregressive modeling을 image generation에 적용한 모델들을 공부한다.

- Image Autoregressive Modeling
- Visual Tokenizer
- Next-Token Prediction
- Next-Scale Prediction
- VAR
- Infinity
- Bitwise Tokenization
- Visual Autoregressive model from scratch


## Goal

최종적으로 새로운 AI 논문이나 모델을 접했을 때,

> 이 모델은 왜 필요한가?

> 기존 모델과 무엇이 다른가?

> 입력과 출력 Tensor의 shape은 무엇인가?

> 내부에서 어떤 matrix 연산이 수행되는가?

> 학습과 inference 과정은 어떻게 다른가?

를 스스로 분석할 수 있는 수준의 기초를 갖추는 것을 목표로 한다.
