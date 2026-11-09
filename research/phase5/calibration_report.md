# VocalSense Phase 5: Confidence Calibration, Uncertainty & Abstention Report

## 1. Executive Summary
Deep neural networks and uncalibrated linear classifiers consistently produce **overconfident predictions**: a classifier outputting $p=0.90$ may only be correct 65% of the time. In a voice companion, presenting uncertain guesses as definitive emotional diagnoses violates safety and trust. This research implements post-hoc **Temperature Scaling** and a **Selective Abstention Gate**.

## 2. Temperature Scaling & Calibration Metrics

| Calibration Regime | Expected Calibration Error (ECE) | Brier Score | Relative ECE Improvement |
|---|---|---|---|
| **Raw Probabilities (Uncalibrated)** | 19.40% | 0.5109 | Baseline |
| **Temperature-Scaled ($T=2.55$)** | **1.29%** | **0.4476** | **-93.3% ECE Reduction** |

## 3. Selective Classification & Abstention Policy
Rather than forcing a prediction on every noisy or ambiguous utterance, VocalSense deploys an active abstention policy:
$$\text{Accept if } \max_k p_k(T) \ge \tau_{\text{conf}} \quad \text{and} \quad H_{\text{norm}}(p) \le \tau_{\text{ent}}$$

### Recommended Operational Thresholds (Selected on Validation Data):
- **Confidence Threshold (tau_conf)**: `0.35`
- **Normalized Entropy Threshold (tau_ent)**: `0.85`
- **Empirical Coverage**: **92.71%** of user messages receive an emotion estimate.
- **Abstention Rate**: **7.29%** gracefully decline to guess.
- **Selective Accuracy**: **69.89%** (up from 67.0% unconstrained accuracy).
- **Selective Risk**: Reduced from 33.0% to **30.11%**.

### User-Facing Abstention Messages:
- *"Unable to estimate vocal emotion reliably."*
- *"The recording contains unclear vocal patterns. No reliable estimate is available."*
