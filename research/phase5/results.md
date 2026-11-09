# VocalSense Phase 5: Comprehensive Empirical Results & Findings

## 1. Speaker-Independent Generalisation Results

### 1.1 Random Split (Speaker Leakage) vs Speaker-Independent (`GroupKFold`)

| Model | Representation | Random Accuracy | Random Macro-F1 | Speaker-Indep. Accuracy | Speaker-Indep. Macro-F1 | Leakage Gap (ΔF1) | Cross-Gender F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 43 Acoustic Features | 56.18% | 0.5496 | 47.85% | 0.4645 | **+8.51%** | 0.2940 |
| **Support Vector Machine** | 43 Acoustic Features | 68.68% | 0.6719 | 49.03% | 0.4794 | **+19.25%** | 0.2801 |
| **Random Forest** | 43 Acoustic Features | 60.69% | 0.5851 | 45.97% | 0.4288 | **+15.63%** | 0.2377 |
| **Wav2Vec 2.0 Base** | 768 Deep Embeddings | 71.81% | 0.7093 | 60.00% | 0.5914 | **+11.79%** | 0.4108 |
| **HuBERT Base** | 768 Deep Embeddings | **82.43%** | **0.8211** | **67.01%** | **0.6666** | **+15.45%** | **0.5371** |

> **Key Discovery 1**: Random split evaluation severely overstates real-world reliability. Traditional SVM drops by **-19.25% Macro-F1** and Random Forest by **-15.63%** when tested on unseen speakers. Naive models achieve high random accuracy by memorizing speaker-specific vocal tract pitch ranges rather than emotional dynamics.
>
> **Key Discovery 2**: HuBERT Base achieves the highest speaker-independent generalization (**0.6666 Macro-F1**), outperforming SVM by **+18.72% absolute Macro-F1**. Furthermore, in cross-gender evaluation (Train Male -> Test Female), HuBERT maintains **0.5909 Macro-F1**, whereas SVM collapses to **0.2419**, demonstrating HuBERT's pitch and vocal tract invariance.

---

## 2. Cross-Dataset Harmonisation & 6-Class Benchmark

Harmonised 6-class common taxonomy (*anger, disgust, fear, happy, neutral, sad*):

| Model Architecture | 8-Class Macro-F1 | 6-Class Harmonised Macro-F1 | 6-Class Accuracy |
|---|---|---|---|
| **Logistic Regression** | 0.4645 | 0.4916 | 50.57% |
| **Support Vector Machine** | 0.4794 | 0.4996 | 51.04% |
| **Random Forest** | 0.4288 | 0.4467 | 46.40% |
| **Wav2Vec 2.0 Base** | 0.5914 | 0.6618 | 67.05% |
| **HuBERT Base** | **0.6666** | **0.7102** | **71.21%** |

> **Key Discovery 3**: Eliminating the ambiguous *calm* vs *neutral* boundary improves HuBERT's Macro-F1 from 0.6666 to **0.7102** (71.21% accuracy on unseen actors). *Calm* and *neutral* share overlapping low-arousal pitch contours, causing the majority of within-corpus classification errors.

---

## 3. Multilingual SER & Cross-Lingual Representation Drift

| Language Modality | Apparent SER Accuracy | Mean Confidence | Mean Cosine Sim to EN Centroid | Primary Acoustic Shift Observed |
|---|---|---|---|---|
| **English (Native Baseline)** | 100% (Clean Formants) | 1.000 | **0.864** | Baseline F0 (145–260 Hz) |
| **Hindi (Devanagari)** | 25.0% (Severe Drift) | 1.000 (Overconfident) | **0.465** | Elevated F0 (+15 Hz mean), retroflex consonant bursts |
| **Hinglish (Code-Switched)** | 0.0% (Complete Drift) | 0.976 (Overconfident) | **0.467** | Pitch resets at code-switching junctures, flat loanword prosody |

> **Key Discovery 4 (The Multilingual Conflation Fallacy)**: A model trained on English speech representation does **NOT** work on Hindi or Hinglish. Pretrained embeddings drift by over 46% in cosine distance, and the uncalibrated linear head makes overconfident, incorrect predictions. VocalSense must NEVER claim multilingual SER without native-language labelled corpora.

---

## 4. Acoustic Robustness Under Real-World Degradation

| Acoustic Condition | HuBERT Macro-F1 | HuBERT F1 Retention | HuBERT Stability | SVM Macro-F1 | SVM F1 Retention | SVM Stability |
|---|---|---|---|---|---|---|
| **Clean Audio Baseline** | **0.9142** | 100.0% | 100.0% | **0.9500** | 100.0% | 100.0% |
| **Gain +6 dB (Proximity)** | 0.9227 | 100.9% | 95.0% | 0.9747 | 102.6% | 97.5% |
| **Gain -6 dB (Low Volume)** | 0.8154 | 89.2% | 90.0% | 0.9270 | 97.6% | 97.5% |
| **Gain -12 dB (Very Low)** | 0.6694 | 73.2% | 77.5% | 0.8706 | 91.6% | 92.5% |
| **Reverb Mild (0.2s)** | 0.5545 | 60.7% | 65.0% | 0.7733 | 81.4% | 82.5% |
| **Telephony 8 kHz** | **0.6837** | **74.8%** | **72.5%** | **0.0648** | **6.8%** | **15.0%** |
| **Reverb Cavernous (0.5s)** | 0.3727 | 40.8% | 45.0% | 0.5614 | 59.1% | 60.0% |
| **Duration 2.0s** | 0.3195 | 34.9% | 37.5% | 0.5062 | 53.3% | 47.5% |
| **Noise SNR 20 dB (Mild)** | **0.3339** | **36.5%** | **47.5%** | **0.1352** | **14.2%** | **22.5%** |
| **Noise SNR 10 dB (Moderate)** | 0.2487 | 27.2% | 35.0% | 0.0278 | 2.9% | 12.5% |
| **Duration 1.0s** | 0.1167 | 12.8% | 12.5% | 0.0278 | 2.9% | 12.5% |
| **Noise SNR 0 dB (Severe)** | 0.0604 | 6.6% | 15.0% | 0.0278 | 2.9% | 12.5% |
| **Duration 0.5s (Brief)** | 0.0233 | 2.5% | 10.0% | 0.0278 | 2.9% | 12.5% |

> **Key Discovery 5 (Vulnerability Hierarchy)**:
> 1. Resampling to 8 kHz destroys SVM performance (F1 drops from 0.95 to 0.0648, a 93.2% collapse), whereas HuBERT retains **74.8% F1** (0.6837).
> 2. Additive background noise ($\le 10$ dB SNR) causes catastrophic failure across all models.
> 3. Short speech segments ($<1.0$s) lack sufficient pitch contour frames, causing complete breakdown. VocalSense must enforce duration and SNR gating before emotion inference.

---

## 5. Confidence Calibration & Selective Abstention Results

| Calibration Regime | Expected Calibration Error (ECE) | Brier Score | Relative Error Reduction |
|---|---|---|---|
| **Uncalibrated Raw Head** | 19.40% (Severe Overconfidence) | 0.5109 | Baseline |
| **Temperature Scaled ($T=2.55$)** | **1.29% (Near-Perfect Calibration)** | **0.4476** | **-93.3% Relative ECE Drop** |

### Selective Abstention Performance ($\tau_{\text{conf}} = 0.35$):
- **Coverage**: **92.71%** of user utterances receive predictions.
- **Abstention Rate**: **7.29%** decline to guess due to high ambiguity/entropy.
- **Accepted Selective Accuracy**: **69.89%** (vs 67.01% unconstrained).
- **Selective Risk (Error Rate)**: Drops from 33.0% to **30.11%**.

---

## 6. Speech Recognition & Language Matching Reliability

- **English Transcription**: 2.08% WER | 0.37% CER
- **Hindi Transcription (Devanagari)**: 2.22% WER | 0.55% CER
- **Hinglish Transcription (Latin Script)**: 0.00% WER | 0.00% CER (Zero unwanted translation to English)
- **Language-Matched Routing Success**: **100.0%** (0.0% wrong-language response rate).
- **Intra-Conversation Dynamic Switching**: 100% verified across English -> Hindi -> Hinglish -> English turns.
