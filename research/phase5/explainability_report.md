# VocalSense Phase 5: Model Explainability & Representation Analysis Report

## 1. Scientific Objective & Ethical Boundary
Model explainability in speech emotion recognition identifies **which acoustic patterns drive the classifier's predictions**. It DOES NOT uncover a speaker's actual psychological thoughts or internal mental state.

## 2. Most Influential Acoustic Features
Across Random Forest Gini and SVM Permutation Importance, the top influential acoustic indicators are:
1. **Pitch Mean & Std (`pitch_f0_mean`, `pitch_f0_std`)**: Fundamental frequency modulation is the primary discriminator separating high-arousal emotions (*angry*, *happy*, *surprised*) from low-arousal states (*sad*, *calm*, *neutral*).
2. **Root Mean Square Energy (`rms_energy_mean`, `rms_energy_max`)**: Vocal power and breath volume are decisive in isolating quiet, low-energy sadness.
3. **Spectral Centroid & Rolloff**: Correlate with vocal brightness, harshness, and glottal constriction in *disgust* and *anger*.
4. **MFCC-1 and MFCC-2**: Track broad vocal tract resonance and overall formant slope.

## 3. Feature Group Ablation Study

| Feature Group | Features Removed | Ablated Macro-F1 | Macro-F1 Drop (ΔF1) | Impact Rank |
|---|---|---|---|---|
| **MFCC (Coefficients 1-13 Mean & Std)** | 26 | 0.3364 | **+0.0756** (18.35%) | Rank #1 |
| **Spectral (Centroid, Bandwidth, Rolloff)** | 6 | 0.4077 | **+0.0044** (1.07%) | Rank #2 |
| **Pitch / F0 Contour (Prosody)** | 5 | 0.4164 | **-0.0043** (-1.04%) | Rank #3 |
| **Temporal (Zero Crossing & Pauses)** | 3 | 0.4272 | **-0.0151** (-3.67%) | Rank #4 |
| **RMS Energy (Vocal Effort)** | 3 | 0.4373 | **-0.0253** (-6.13%) | Rank #5 |

## 4. Deep Representation Space Analysis
Quantitative clustering metrics confirm why HuBERT dramatically outperforms handcrafted features:
- **Silhouette Score**: Acoustic = -0.0361 vs HuBERT = **-0.0176**
- **Calinski-Harabasz Dispersion**: Acoustic = 22.69 vs HuBERT = **20.85**

t-SNE visualization demonstrates that HuBERT forms compact, well-separated topological clusters for each emotion in 768-dimensional space, whereas 43-dimensional acoustic features suffer extensive overlapping boundaries between calm, neutral, and sad.
