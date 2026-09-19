# Probability & Statistics

This section summarizes the core concepts in probability and statistics required for research in:

- Neural network compression
- Quantization
- Pruning
- Activation statistics
- Covariance analysis
- Random matrix analysis
- Machine learning

---

## Table of Contents

1. [Probability Basics](#1-probability-basics)
2. [Conditional Probability and Bayes' Theorem](#2-conditional-probability-and-bayes-theorem)
3. [Random Variables](#3-random-variables)
4. [Probability Distributions](#4-probability-distributions)
5. [Expectation](#5-expectation)
6. [Variance and Standard Deviation](#6-variance-and-standard-deviation)
7. [Covariance and Correlation](#7-covariance-and-correlation)
8. [Joint, Marginal, and Conditional Distributions](#8-joint-marginal-and-conditional-distributions)
9. [Law of Large Numbers and Central Limit Theorem](#9-law-of-large-numbers-and-central-limit-theorem)
10. [Sampling and Estimation](#10-sampling-and-estimation)
11. [Maximum Likelihood Estimation](#11-maximum-likelihood-estimation)
12. [Hypothesis Testing](#12-hypothesis-testing)
13. [Research Connections](#13-research-connections)

---

# 1. Probability Basics

Topics:

- Sample space
- Events
- Probability axioms
- Union and intersection
- Complement
- Independence

Key idea:

Probability provides a mathematical framework for reasoning about uncertainty.

---

# 2. Conditional Probability and Bayes' Theorem

Topics:

- Conditional probability
- Independence
- Bayes' theorem
- Law of total probability

Key idea:

Conditional probability describes how uncertainty changes when additional information is given.

---

# 3. Random Variables

Topics:

- Discrete random variables
- Continuous random variables
- Probability mass function (PMF)
- Probability density function (PDF)
- Cumulative distribution function (CDF)

Key idea:

A random variable maps outcomes of a random experiment to numerical values.

---

# 4. Probability Distributions

Important distributions:

- Bernoulli distribution
- Binomial distribution
- Uniform distribution
- Gaussian distribution
- Poisson distribution
- Exponential distribution

Research relevance:

- Noise modeling
- Parameter initialization
- Activation distribution analysis
- Statistical modeling

---

# 5. Expectation

Topics:

- Expected value
- Linearity of expectation
- Expected value of functions

Key idea:

Expectation represents the average value of a random variable over repeated observations.

---

# 6. Variance and Standard Deviation

Topics:

- Variance
- Standard deviation
- Second moment
- Mean-centered data

Key idea:

Variance measures how much a random variable fluctuates around its mean.

Research relevance:

- Activation variance
- Variance-Based Pruning
- Outlier analysis

---

# 7. Covariance and Correlation

Topics:

- Covariance
- Correlation coefficient
- Positive / negative correlation
- Covariance matrix

Key idea:

Covariance measures how two random variables vary together.

Research relevance:

- Activation statistics
- Structured pruning
- Covariance-based neuron analysis
- PCA
- Random matrix methods

---

# 8. Joint, Marginal, and Conditional Distributions

Topics:

- Joint probability distribution
- Marginal distribution
- Conditional distribution
- Independence of random variables

Key idea:

A joint distribution describes the behavior of multiple random variables together.

---

# 9. Law of Large Numbers and Central Limit Theorem

Topics:

- Law of Large Numbers
- Central Limit Theorem
- Sample mean
- Convergence

Key idea:

As the number of samples increases, empirical statistics become more reliable.

Research relevance:

- Calibration data
- Activation statistics
- Estimation of mean / variance / covariance

---

# 10. Sampling and Estimation

Topics:

- Population vs. sample
- Sample mean
- Sample variance
- Bias
- Consistency
- Confidence interval

Key idea:

Statistics uses finite samples to estimate unknown properties of an underlying distribution.

---

# 11. Maximum Likelihood Estimation

Topics:

- Likelihood
- Log-likelihood
- Maximum Likelihood Estimation (MLE)
- Parameter estimation

Key idea:

MLE finds model parameters that maximize the probability of the observed data.

Research relevance:

- Statistical modeling
- Machine learning objectives
- Probabilistic model fitting

---

# 12. Hypothesis Testing

Topics:

- Null hypothesis
- Alternative hypothesis
- Test statistic
- p-value
- Significance level
- Type I / Type II error

Research relevance:

- Experiment evaluation
- Statistical comparison
- Result validation

---

# 13. Research Connections

## Neural Network Compression

Important concepts:

- Mean
- Variance
- Covariance
- Distribution
- Sampling

---

## Variance-Based Pruning

Important concepts:

- Activation mean
- Activation variance
- Sample statistics
- Calibration data

---

## Denoised Variance-Based Pruning

Important concepts:

- Covariance matrix
- Eigenvalue distribution
- Statistical noise
- Finite-sample estimation
- Random matrix theory

---

## Quantization

Important concepts:

- Activation distribution
- Outliers
- Range
- Statistical calibration
- Quantization error

---

## Visual Autoregressive Modeling

Important concepts:

- Probability distribution
- Conditional probability
- Joint distribution
- Autoregressive factorization
- Sampling

---

# Study Goal

The goal is to understand probability and statistics not only as mathematical theory, but also as tools for analyzing model behavior, activation distributions, compression methods, and experimental results.
