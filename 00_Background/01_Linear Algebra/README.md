# Linear Algebra

This section summarizes the core linear algebra concepts required for research in:

- Neural network compression
- Quantization
- Pruning
- Second-order optimization
- Covariance analysis
- Transformer models

---

## Table of Contents

1. [Vectors and Vector Spaces](#1-vectors-and-vector-spaces)
2. [Linear Independence, Basis, and Dimension](#2-linear-independence-basis-and-dimension)
3. [Matrices and Linear Transformations](#3-matrices-and-linear-transformations)
4. [Rank and Null Space](#4-rank-and-null-space)
5. [Inner Product and Orthogonality](#5-inner-product-and-orthogonality)
6. [Eigenvalues and Eigenvectors](#6-eigenvalues-and-eigenvectors)
7. [Diagonalization](#7-diagonalization)
8. [Singular Value Decomposition](#8-singular-value-decomposition)
9. [Positive Definite Matrices](#9-positive-definite-matrices)
10. [Quadratic Forms](#10-quadratic-forms)
11. [Matrix Norms](#11-matrix-norms)
12. [Covariance Matrix](#12-covariance-matrix)
13. [Research Connections](#13-research-connections)

---

# 1. Vectors and Vector Spaces

Topics:

- Scalars and vectors
- Vector addition
- Scalar multiplication
- Linear combinations
- Vector spaces
- Subspaces
- Span

Key question:

> What combinations of vectors can represent a given vector space?

---

# 2. Linear Independence, Basis, and Dimension

Topics:

- Linear independence
- Linear dependence
- Basis
- Dimension
- Coordinates with respect to a basis

Key idea:

A basis is a minimal set of linearly independent vectors that spans a vector space.

---

# 3. Matrices and Linear Transformations

Topics:

- Matrix operations
- Matrix multiplication
- Transpose
- Inverse
- Identity matrix
- Linear transformations
- Change of basis

Key idea:

A matrix can be interpreted as a linear transformation between vector spaces.

---

# 4. Rank and Null Space

Topics:

- Column space
- Row space
- Null space
- Rank
- Rank-nullity theorem

Key idea:

The rank represents the number of independent directions preserved by a linear transformation.

---

# 5. Inner Product and Orthogonality

Topics:

- Dot product
- Inner product
- Norm
- Orthogonal vectors
- Orthonormal basis
- Projection
- Gram-Schmidt process

Key idea:

Orthogonality allows vectors to be represented independently from one another.

---

# 6. Eigenvalues and Eigenvectors

Definition:

For a matrix $A$,

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

where

- $\mathbf{v}$ is an eigenvector
- $\lambda$ is the corresponding eigenvalue

Topics:

- Characteristic equation
- Eigenvalues
- Eigenvectors
- Eigenspace
- Spectral interpretation

Key idea:

Eigenvectors represent directions whose orientation is preserved by a linear transformation.

---

# 7. Diagonalization

Topics:

- Diagonal matrices
- Eigen-decomposition
- Diagonalizable matrices

If a matrix $A$ is diagonalizable,

$$
A = P D P^{-1}
$$

where $D$ contains eigenvalues.

Key idea:

Diagonalization simplifies repeated matrix operations and reveals the structure of a transformation.

---

# 8. Singular Value Decomposition

For a matrix $A$,

$$
A = U \Sigma V^T
$$

where

- $U$: left singular vectors
- $\Sigma$: singular values
- $V$: right singular vectors

Topics:

- Singular values
- Left / right singular vectors
- Low-rank approximation
- Relationship with PCA

Research relevance:

- Model compression
- Low-rank approximation
- Dimensionality reduction
- Weight matrix analysis

---

# 9. Positive Definite Matrices

A symmetric matrix $A$ is positive definite if

$$
\mathbf{x}^T A \mathbf{x} > 0
$$

for every non-zero vector $\mathbf{x}$.

Topics:

- Positive definite matrices
- Positive semi-definite matrices
- Eigenvalue conditions
- Hessian matrices
- Covariance matrices

Research relevance:

- Optimization
- Second-order methods
- Hessian-based pruning and quantization

---

# 10. Quadratic Forms

General form:

$$
\mathbf{x}^T A \mathbf{x}
$$

Topics:

- Quadratic forms
- Positive / negative definiteness
- Optimization interpretation

Research relevance:

Second-order approximation of neural network loss often takes the form

$$
\Delta L
\approx
\frac{1}{2}
\Delta \mathbf{w}^T
H
\Delta \mathbf{w}
$$

where $H$ is the Hessian matrix.

This formulation appears in methods such as:

- Optimal Brain Surgeon
- Optimal Brain Compression
- GPTQ

---

# 11. Matrix Norms

A norm measures the magnitude or size of a vector or matrix.

## Vector Norms

### L1 Norm

The L1 norm is the sum of the absolute values of vector elements.

```math
\lVert \mathbf{x} \rVert_1
=
\sum_i |x_i|
```

### L2 Norm

The L2 norm, or Euclidean norm, measures the Euclidean length of a vector.

```math
\lVert \mathbf{x} \rVert_2
=
\sqrt{\sum_i x_i^2}
```

## Matrix Norms

### Frobenius Norm

The Frobenius norm is the square root of the sum of the squared elements of a matrix.

```math
\lVert A \rVert_F
=
\sqrt{\sum_{i,j} A_{ij}^2}
```

### Spectral Norm

The spectral norm of a matrix is its largest singular value.

```math
\lVert A \rVert_2
=
\sigma_{\max}(A)
```

## Research Relevance

Matrix and vector norms are frequently used to measure:

- Reconstruction error
- Quantization error
- Weight perturbation
- Approximation error
- Layer-wise output difference

For example, layer-wise reconstruction-based compression methods often minimize an objective of the form:

```math
\lVert WX - \hat{W}X \rVert_F^2
```

where:

- $W$ is the original weight matrix
- $\hat{W}$ is the compressed weight matrix
- $X$ is the layer input

This type of objective appears in post-training compression methods such as OBC and GPTQ.

---

# 12. Covariance Matrix

A covariance matrix describes how multiple variables vary together.

Consider a random vector:

```math
\mathbf{x}
=
[x_1, x_2, \dots, x_d]^T
```

Its mean vector is:

```math
\boldsymbol{\mu}
=
\mathbb{E}[\mathbf{x}]
```

The covariance matrix is defined as:

```math
C
=
\mathbb{E}
\left[
(\mathbf{x}-\boldsymbol{\mu})
(\mathbf{x}-\boldsymbol{\mu})^T
\right]
```

For a $d$-dimensional random vector,

```math
C \in \mathbb{R}^{d \times d}
```

and each element $C_{ij}$ represents the covariance between $x_i$ and $x_j$.

```math
C_{ij}
=
\operatorname{Cov}(x_i, x_j)
```

Therefore, the covariance matrix has the form:

```math
C =
\begin{bmatrix}
\operatorname{Var}(x_1) & \operatorname{Cov}(x_1,x_2) & \cdots \\
\operatorname{Cov}(x_2,x_1) & \operatorname{Var}(x_2) & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
```

The diagonal elements represent variances:

```math
C_{ii} = \operatorname{Var}(x_i)
```

while the off-diagonal elements represent relationships between different variables.

## Key Properties

- A covariance matrix is symmetric.
- Its diagonal entries are variances.
- It is positive semi-definite.
- Its eigenvectors represent principal directions of variation.
- Its eigenvalues represent the amount of variance along those directions.

## Research Relevance

Covariance matrices are particularly important for:

- Activation statistics
- Principal Component Analysis (PCA)
- Variance-Based Pruning
- Denoised Variance-Based Pruning
- Second-order optimization
- Random matrix analysis

In neural network pruning, activation covariance can reveal how neurons vary individually and how their activations are correlated with one another.

---

# 13. Research Connections

## Quantization

Important concepts:

- Vector norms
- Matrix multiplication
- Projection
- Reconstruction error

---

## Optimal Brain Compression / GPTQ

Important concepts:

- Hessian matrix
- Matrix inverse
- Positive definite matrices
- Quadratic forms
- Second-order approximation

---

## Variance-Based Pruning

Important concepts:

- Mean
- Variance
- Covariance
- Covariance matrix

---

## QuaRot / SpinQuant

Important concepts:

- Orthogonal matrices
- Rotation matrices
- Hadamard matrices
- Invariance under orthogonal transformations

---

## Low-Rank Compression

Important concepts:

- Singular Value Decomposition
- Rank
- Low-rank approximation

---

# Study Goal

The goal is to understand these concepts both mathematically and in the context of modern neural network compression and efficient inference.
---

# 13. Research Connections

## Quantization

Important concepts:

- Vector norms
- Matrix multiplication
- Projection
- Reconstruction error

---

## Optimal Brain Compression / GPTQ

Important concepts:

- Hessian matrix
- Matrix inverse
- Positive definite matrices
- Quadratic forms
- Second-order approximation

---

## Variance-Based Pruning

Important concepts:

- Mean
- Variance
- Covariance
- Covariance matrix

---

## QuaRot / SpinQuant

Important concepts:

- Orthogonal matrices
- Rotation matrices
- Hadamard matrices
- Invariance under orthogonal transformations

---

## Low-Rank Compression

Important concepts:

- Singular Value Decomposition
- Rank
- Low-rank approximation

---

# Study Goal

The goal is to understand these concepts both mathematically and in the context of modern neural network compression and efficient inference.
