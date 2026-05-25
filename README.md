# Fourier-Based Optimization of Neural Networks

## Overview

This project explores a frequency-domain approach to analyzing and modifying the internal representations of deep neural networks. Instead of operating purely in the spatial (activation) domain, the method applies Fourier transforms to output representations to study whether frequency-based structure can be leveraged for improved optimization and generalization.

The core idea is that an overfitted neural network's output function may contain redundant or noisy components that can be better understood—and potentially filtered—when expressed in the frequency domain.

---

## Key Insight

Overfitted neural networks often possess high frequency components that the network uses to memorize training data. This can be filtered out using spectral representations.

---

## Motivation

Traditional deep learning optimization operates directly on activations and gradients without explicitly considering their spectral structure. However, many signals (e.g., images, time-series data) exhibit meaningful patterns in the frequency domain.

This project investigates:

* Whether neural activations exhibit exploitable frequency structure
* Whether transforming activations into the frequency domain enables new forms of regularization or optimization
* Whether selective modification of spectral components can improve model behavior

---

## Approach

### 1. Transformation to Frequency Domain

Intermediate activations from a neural network are transformed using the Fast Fourier Transform (FFT), allowing analysis of their spectral components.

### 2. Spectral Manipulation

In the frequency domain, different strategies are explored, such as:

* Suppressing low-magnitude (potentially noisy) components
* Modifying frequency distributions
* Applying structured filters to emphasize or de-emphasize certain bands

### 3. Reconstruction

The modified representations are transformed back into the spatial domain using the inverse FFT and passed forward through the network.

---

## Implementation

* Language: Python
* Libraries: NumPy, PyTorch (or TensorFlow — adjust this)

The system is designed as a “black box” wrapper that can be applied to existing architectures under controlled conditions.

---

## Experiments

### Setup

* Dataset: Toy dataset with controlled noise components
* Model: Deep Neural Network
* Baseline: Standard training without spectral modification

### Current Findings

* Preliminary experiments suggest that certain spectral modifications can influence training dynamics
* Effects vary significantly depending on model architecture and input dimensionality
* Consistent improvements are observed with varying noise settings.

---

## Limitations

* Results are preliminary and not yet statistically robust
* Computational overhead introduced by FFT operations
* Interaction of the model with real-world data is yet to be seen
* Unclear theoretical grounding for when and why improvements occur

---

## Future Work

* Extend method to higher-dimensional feature representations
* Explore connections to regularization techniques (e.g., dropout, spectral normalization)
* Apply approach to structured signals such as ECG or time-series data
* Develop theoretical framework explaining observed behaviors

---


## Notes

This is an ongoing research project. The goal is exploratory: to test whether frequency-domain representations provide a useful lens for understanding and improving neural networks, rather than to present a finalized optimization method.

---

## Contact

Aditya Bhagwani
University of Washington
bhagwani@uw.edu
