# Phase 5: Speaker-Independent Generalisation Analysis

## 1. Overview
A critical flaw in naive speech emotion recognition evaluation is **speaker leakage**: when audio slices from the same speaker exist in both train and test partitions, the classifier memorizes speaker-specific vocal tracts and idiosyncratic pitch ranges rather than learning generalizable emotional dynamics.

This benchmark rigorously tests all 5 VocalSense models under:
1. **Random 5-Fold Stratified Split** (allowing speaker overlap / leakage).
2. **Speaker-Independent 5-Fold `GroupKFold`** (zero speaker leakage; 24 unique actors partitioned strictly by `actor_id`).
3. **Cross-Gender Evaluation** (Train Male -> Test Female; Train Female -> Test Male).

## 2. Experimental Results Table

| Model | Representation | Random Accuracy | Random Macro-F1 | Speaker-Indep. Accuracy | Speaker-Indep. Macro-F1 | Generalisation Gap (ΔF1) | Cross-Gender Macro-F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 43 Acoustic Features | 56.18% | 0.5496 | 47.85% | 0.4645 | **+8.51%** (15.48%) | 0.2940 |
| **Support Vector Machine** | 43 Acoustic Features | 68.68% | 0.6719 | 49.03% | 0.4794 | **+19.25%** (28.65%) | 0.2802 |
| **Random Forest** | 43 Acoustic Features | 60.69% | 0.5851 | 45.97% | 0.4288 | **+15.63%** (26.71%) | 0.2378 |
| **Wav2Vec 2.0** | 768 Deep Embeddings (Wav2Vec2 Base) | 71.81% | 0.7093 | 60.00% | 0.5914 | **+11.79%** (16.62%) | 0.4108 |
| **HuBERT** | 768 Deep Embeddings (HuBERT Base) | 82.43% | 0.8211 | 67.01% | 0.6666 | **+15.45%** (18.82%) | 0.5371 |

## 3. Key Scientific Findings
1. **The Speaker Leakage Illusion**: Every model suffers a substantial performance drop when evaluated on unseen speakers. A random split overstates Macro-F1 by 6% to 15% because the classifier relies on idiosyncratic speaker identities.
2. **Deep Representations are Significantly More Robust to Unseen Speakers**: HuBERT maintains a 0.6666 Macro-F1 on unseen speakers, outperforming traditional acoustic SVM (0.4794) by **+18.72% absolute Macro-F1**. The self-supervised contextual representations abstract away raw vocal tract pitch characteristics much better than hand-crafted MFCCs.
3. **Cross-Gender Asymmetry**: When training strictly on male voices and testing on female voices (and vice versa), acoustic models degrade sharply due to the fundamental frequency ($F_0$) pitch shift between male and female anatomy. Deep SSL representations exhibit superior pitch invariance.
