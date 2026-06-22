# Curve Optimizer — Spectral Post-Processing for Overfit DNNs

Training-free method for improving DNN generalization: treat the model's output as a 1-D signal and strip memorization artefacts in the frequency domain via FFT low-pass filtering — zero retraining required.

## Core Hypothesis

Memorization is high-frequency noise; true signal structure is low-frequency. If this spectral separation holds, filtering an overfit model's output curve should improve test performance post-hoc.

## Method

1. **Probe** the trained DNN on a regular grid (uniform sampling needed for DFT) → dense 1-D output curve
2. **FFT** the curve, inspect the magnitude spectrum
3. **Low-pass filter**: zero all spectral coefficients above cutoff $f_c$
4. **IFFT** to reconstruct a denoised curve
5. **Threshold search**: sweep $f_c \in [0,1)$ in 0.001 steps, optimizing R² (scale-free) or MSE
6. **DC correction** (real-world data only): shift the reconstructed curve by $(\bar y_{test} - \bar y_{train})$ to correct for non-stationary distribution shift

## Experiments

| # | Signal | Noise | Key Result |
|---|--------|-------|------------|
| 1 | Lag-1, shared noise | σ=30 | R² +0.033, RMSE −11.3 — validates core hypothesis |
| 2 | Staggered multi-lag, per-component noise | 10% | Blackbox pipeline generalizes; MSE-sweep robust |
| 3 | AR(1), coeff=10 (strong feedback) | 30% | Still improves; margin narrows under strong AR |
| 4 | Multi-lag AR (5 terms, lags 4–10) | 40% | Robust to highest noise + complex AR structure |
| Final | Real electricity demand (2020–24, DNN) | Real | RMSE −20 MW, R² +0.0074 on 2024 holdout |

**Final project detail:** 7-layer MLP (8→512→1028→512→256→128→64→1), 1000 epochs, no regularization, train loss → 0 (memorized). Probed at hourly resolution across 2024 (8,784 pts), LPF cutoff = 0.1 cycles/hr, DC shift +127 MW.

| Metric | DNN | FFT Optimizer |
|---|---|---|
| RMSE | 377.18 | 357.09 |
| R² | 0.9287 | 0.9361 |
| MAE | 293.21 | 277.77 |

## Limitations

- **Effectively 1-D only.** Works because the input varies along a single cyclic axis (time). For $d \geq 2$: sampling cost is $N^d$ (infeasible), regular grids rarely exist in high-D, and "frequency" loses physical meaning.
- Best suited to 1-D/2-D time-indexed signals where a model can be probed along a natural temporal axis.
- **DC offset problem.** LPF preserves the training-set mean; non-stationary train/test distributions require a manual mean-shift correction not tunable via the LPF threshold.

## Repo Structure

```
├── Experiment 1.ipynb
├── Experiment 2-Copy1.ipynb
├── Experiment 3.ipynb
├── Experiment 4.ipynb
├── final project.ipynb
└── electricity-demand-dataset.csv
```

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `openpyxl`

---

## Contact

Aditya Bhagwani
University of Washington
bhagwani@uw.edu
