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

For a matrix \(A\),

\[
A\mathbf{v} = \lambda \mathbf{v}
\]

where

- \(\mathbf{v}\) is an eigenvector
- \(\lambda\) is the corresponding eigenvalue

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

If a matrix \(A\) is diagonalizable,

\[
A = P D P^{-1}
\]

where \(D\) contains eigenvalues.

Key idea:

Diagonalization simplifies repeated matrix operations and reveals the structure of a transformation.

---

# 8. Singular Value Decomposition

For a matrix \(A\),

\[
A = U \Sigma V^T
\]

where

- \(U\): left singular vectors
- \(\Sigma\): singular values
- \(V\): right singular vectors

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

A symmetric matrix \(A\) is positive definite if

\[
\mathbf{x}^T A \mathbf{x} > 0
\]

for every non-zero vector \(\mathbf{x}\).

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

\[
\mathbf{x}^T A \mathbf{x}
\]

Topics:

- Quadratic forms
- Positive / negative definiteness
- Optimization interpretation

Research relevance:

Second-order approximation of neural network loss often takes the form

\[
\Delta L \approx
\frac{1}{2}
\Delta \mathbf{w}^T
H
\Delta \mathbf{w}
\]

where \(H\) is the Hessian matrix.

This formulation appears in methods such as:

- Optimal Brain Surgeon
- Optimal Brain Compression
- GPTQ

---

# 11. Matrix Norms

Topics:

- L1 norm
- L2 norm
- Frobenius norm
- Spectral norm

Examples:

\[
\|\mathbf{x}\|_2
=
\sqrt{\sum_i x_i^2}
\]

\[
\|A\|_F
=
\sqrt{\sum_{i,j} A_{ij}^2}
\]

Research relevance:

- Reconstruction error
- Quantization error
- Weight perturbation
- Layer-wise optimization

---

# 12. Covariance Matrix

For a random vector \(\mathbf{x}\),

\[
C
=
\mathbb{E}
[
(\mathbf{x}-\boldsymbol{\mu})
(\mathbf{x}-\boldsymbol{\mu})^T
]
\]

where

\[
\boldsymbol{\mu}
=
\mathbb{E}[\mathbf{x}]
\]

Topics:

- Mean vector
- Variance
- Covariance
- Covariance matrix
- Eigenvalue decomposition of covariance

Research relevance:

- Activation statistics
- Variance-Based Pruning
- Denoised Variance-Based Pruning
- PCA
- Random matrix analysis

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
