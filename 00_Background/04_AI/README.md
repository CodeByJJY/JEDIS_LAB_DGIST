# AI Background

이 디렉토리는 최신 AI 모델, 특히 **Transformer 기반 모델과 Visual Generative Model**을 이해하기 위해 필요한 기초 개념을 정리한다.

단순히 모델의 구조를 외우는 것이 아니라,

> **왜 이런 구조가 필요했는가?  
> 입력과 출력은 무엇인가?  
> 내부에서 어떤 계산이 수행되는가?  
> 이전 방법의 어떤 문제를 해결했는가?**

를 이해하는 것을 목표로 한다.

최종적으로 다음과 같은 모델과 논문을 코드 및 수식 수준에서 이해할 수 있는 기반을 만드는 것이 목표이다.

- Transformer
- Vision Transformer (ViT)
- Autoregressive Language / Image Models
- VAE / VQ-VAE / Visual Tokenizer
- Diffusion Models
- Diffusion Transformer (DiT)
- Visual AutoRegressive Modeling (VAR)
- Infinity and related Visual Generative Models

---

# 1. Big Picture

AI 모델의 발전을 단순히

```text
CNN
 ↓
RNN / LSTM
 ↓
Transformer
 ↓
Diffusion
 ↓
Autoregressive
