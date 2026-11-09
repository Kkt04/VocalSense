# VocalSense Phase 5: Experimental Design & Evaluation Protocol

## 1. Experimental Methodology Overview
Phase 5 executes a rigorous, reproducible, leakage-free empirical benchmark across seven scientific modules:
1. **Speaker-Independent Generalisation**
2. **Cross-Corpus Taxonomy Harmonisation**
3. **Multilingual Acoustic & Representation Drift**
4. **Controlled Real-World Acoustic Perturbation**
5. **Acoustic & Deep Representation Explainability**
6. **Probability Calibration & Selective Abstention**
7. **Speech Transcription & Conversational Language Matching**

All experiments use deterministic random seeds (`seed=42`), standard evaluation metrics, and strict partition boundaries.

---

## 2. Dataset & Partitioning Protocol

### 2.1 The RAVDESS Speech Corpus
- **Recordings**: 1,440 audio speech files (WAV, 16-bit, 48 kHz original -> 16 kHz resampled).
- **Actors**: 24 professional actors (12 Male: odd IDs 1, 3, ... 23; 12 Female: even IDs 2, 4, ... 24).
- **Emotion Classes ($K=8$)**: *neutral* (96), *calm* (192), *happy* (192), *sad* (192), *angry* (192), *fearful* (192), *disgust* (192), *surprised* (192).

### 2.2 Leakage Prevention & Partitioning Regimes
To rigorously quantify the impact of speaker leakage, two parallel evaluation protocols are executed:
1. **Random Partitioning (`StratifiedKFold`, $k=5$)**:
   - Audio files are randomly shuffled across folds while preserving emotion class balance.
   - **Flaw**: Utterances from the same actor appear in both train and test splits, allowing classifiers to memorize vocal tract pitch idiosyncrasies.
2. **Speaker-Independent Partitioning (`GroupKFold`, $k=5$, Grouped by `actor_id`)**:
   - No actor ever appears in both training and test partitions for any fold.
   - All reported speaker-independent metrics represent out-of-fold evaluations across all 24 unseen actors.
3. **Cross-Gender Partitioning**:
   - Split A: Train on 12 Male Actors (720 clips) -> Test on 12 Unseen Female Actors (720 clips).
   - Split B: Train on 12 Female Actors (720 clips) -> Test on 12 Unseen Male Actors (720 clips).

---

## 3. Evaluated Model Architectures

| Model Key | Architecture | Input Representation | Pretraining | Training Regime |
|---|---|---|---|---|
| **Logistic Regression** | Linear multi-class head ($L_2$ regularized) | 43 Acoustic Features | None | Trained from scratch |
| **SVM Baseline** | Radial Basis Function (RBF) kernel, $C=2.5$ | 43 Acoustic Features | None | Trained from scratch |
| **Random Forest** | 180 ensemble trees, max depth 14 | 43 Acoustic Features | None | Trained from scratch |
| **Wav2Vec 2.0 Base** | 7-layer Conv1D + 12-layer Transformer + Linear Head | 768-dim SSL Embeddings | LibriSpeech 960h (Contrastive) | Frozen backbone + calibrated head |
| **HuBERT Base** | 7-layer Conv1D + 12-layer Transformer + Linear Head | 768-dim SSL Embeddings | LibriSpeech 960h (Masked Cluster) | Frozen backbone + calibrated head |

---

## 4. Evaluation Metrics & Mathematical Definitions

### 4.1 Macro-Averaged F1-Score (Primary Performance Metric)
$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
*Chance level for 8 balanced classes = 12.5%. Chance level for 6 harmonised classes = 16.7%.*

### 4.2 Generalisation Gap ($\Delta \text{Macro-F1}$)
$$\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{Random Split}} - \text{Macro-F1}_{\text{Speaker-Independent}}$$
Quantifies the exact degree to which speaker leakage inflates apparent benchmark performance.

### 4.3 Expected Calibration Error (ECE)
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
Evaluated over $M=10$ equal-width probability bins $B_m$.

### 4.4 Brier Score
$$\text{Brier} = \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K (p_{n,k} - y_{n,k})^2$$

### 4.5 Prediction Stability Under Perturbation
$$\text{Stability} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left( \hat{y}_i^{\text{perturbed}} == \hat{y}_i^{\text{clean}} \right)$$
Measures the percentage of audio clips whose predicted emotion label remains unchanged after perturbation.
