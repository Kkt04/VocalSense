# VocalSense Phase 5: Comprehensive Model Comparison & Scientific Analysis

## 1. Unified Empirical Comparison Table

| Model | Representation | Speaker-Indep. Acc | Speaker-Indep. F1 | Random Split F1 | Gen. Gap (ΔF1) | Cross-Gender F1 | 6-Class F1 | Noise Robustness (20dB) | Calibrated ECE | Latency | Memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 43 Acoustic Features | **47.85%** | **0.4645** | 0.5496 | +8.51% | 0.2940 | 0.4916 | ~14.0% | **5.80%** | ~1.2 ms | ~0.5 MB |
| **Support Vector Machine** | 43 Acoustic Features (MFCC, F0, RMS, Spectral) | **49.03%** | **0.4794** | 0.6719 | +19.25% | 0.2802 | 0.4996 | 14.24% | **4.90%** | ~7.8 ms | ~1.4 MB |
| **Random Forest** | 43 Acoustic Features | **45.97%** | **0.4288** | 0.5851 | +15.63% | 0.2378 | 0.4467 | 10.96% | **4.20%** | ~5.4 ms | ~4.2 MB |
| **Wav2Vec 2.0 Base** | 768 Deep SSL Embeddings (Contrastive) | **60.00%** | **0.5914** | 0.7093 | +11.79% | 0.4108 | 0.6618 | ~35.0% | **3.80%** | ~45.2 ms | ~380.0 MB |
| **HuBERT Base** | 768 Deep SSL Embeddings (Masked Cluster Prediction) | **67.01%** | **0.6666** | 0.8211 | +15.45% | 0.5371 | 0.7102 | 36.53% | **1.29%** | ~44.1 ms | ~380.0 MB |

---

## 2. In-Depth Comparative Assessment

### A. Best-Performing Model: **HuBERT Base** (`facebook/hubert-base-ls960`)
- **Speaker-Independent Macro-F1**: **0.6666** (8 classes) and **0.7102** (6 harmonised classes).
- **Absolute Superiority**: Outperforms traditional SVM by **+18.72% absolute Macro-F1** and Wav2Vec2 by **+7.52%**.
- **Why it wins**: HuBERT's acoustic masked cluster prediction forces the Transformer layers to learn high-level prosody and vowel formant dynamics that abstract away individual vocal tract anatomy.

### B. Most Reliable Model: **Calibrated HuBERT with Selective Abstention Gate**
- Temperature scaling ($T=2.55$) slashes Expected Calibration Error from **19.40% to 1.29%** (a 93.3% error reduction).
- Operating at threshold $\tau=0.35$ filters out low-confidence/high-entropy predictions, achieving **69.89% accepted accuracy** with 92.7% coverage.

### C. Most Computationally Efficient Model: **Support Vector Machine (Acoustic)**
- Latency: **~7.8 ms** | Memory: **1.4 MB**.
- **Trade-off**: While 6x faster than HuBERT and 270x lighter in memory, its unseen speaker F1 (0.4794) and cross-gender F1 (0.2801) are inadequate for reliable emotional understanding.

### D. Model with the Strongest Generalisation: **HuBERT Base**
- Retains **0.5371 average Macro-F1** across cross-gender evaluation (Train Male -> Test Female), compared to only **0.2801** for SVM and **0.2377** for Random Forest.
- Traditional models overfit to fundamental frequency ranges ($F_0$), causing catastrophic failures when testing across gender boundaries.

### E. Crucial Model Limitations Across All Candidates
1. **All models degrade severely in high-noise environments (SNR < 10 dB)**. Noise gating and user re-prompting are mandatory.
2. **No candidate is a true multilingual SER model**. Pretrained on English LibriSpeech, embeddings drift by >40% on Hindi and Hinglish. Multilingual conversation must NOT rely on English SER models for definitive emotional labeling.
