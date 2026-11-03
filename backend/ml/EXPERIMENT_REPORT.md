# VocalSense — Phase 3: Speech Emotion Recognition (SER) ML Experiment Report

## 1. Executive Summary

In Phase 3 of the **VocalSense** project, we designed, implemented, evaluated, and deployed our first baseline Speech Emotion Recognition (SER) pipeline using traditional machine learning algorithms on the **RAVDESS** dataset.

The system extracts **43 interpretable acoustic features** (MFCCs, fundamental frequency $F_0$, RMS energy, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate, and pause fraction) and classifies 8 vocal emotion categories.

A central scientific focus of this phase was validating **Speaker-Independent Generalization** versus **Speaker Leakage**:
- **Speaker-Independent Evaluation (5-fold `GroupKFold` grouped by `actor_id`)**: The Support Vector Machine (SVM with RBF kernel) achieved an accuracy of **49.03%** and macro-F1 of **0.4794** (chance level is $12.5\%$).
- **Random Stratified Evaluation (5-fold `StratifiedKFold`)**: The same SVM pipeline achieved **68.68%** accuracy and **0.6719** macro-F1.
- **Key Takeaway**: Random splits inflate performance by **+19.65%** because the model memorizes speaker-specific vocal traits (speaker identity) rather than emotion-invariant acoustics. GroupKFold on speaker ID is the only scientifically valid benchmark for generalization.

---

## 2. Dataset: RAVDESS Speech Corpus

**Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:
- **Files**: 1,440 speech recordings in uncompressed 16-bit, 48kHz stereo WAV.
- **Actors**: 24 professional actors (12 male, 12 female), balanced distribution.
- **Statements**: 2 standardized neutral statements:
  - Statement 01: *"Kids are talking by the door"*
  - Statement 02: *"Dogs are sitting by the door"*
- **Emotions (8 classes)**:
  - Neutral ($n = 96$)
  - Calm ($n = 192$)
  - Happy ($n = 192$)
  - Sad ($n = 192$)
  - Angry ($n = 192$)
  - Fearful ($n = 192$)
  - Disgust ($n = 192$)
  - Surprised ($n = 192$)
- **Data Integrity**: Verified 1,440 valid WAV files, 0 corrupted files, 0 missing files.

### Dataset Inherent Limitations
1. **Acted Speech**: Recordings are acted by professional actors under studio conditions. Spontaneous natural speech in conversation contains subtler, blended, and context-dependent emotions.
2. **Controlled Lexicon**: Only two fixed sentences were spoken; lexical information cannot aid or confound emotion detection here.
3. **Monolingual (English)**: RAVDESS only contains North American English speech. It cannot be used to establish multilingual emotion recognition across Hindi or Hinglish.

---

## 3. Audio Preprocessing Pipeline

Every audio sample undergoes deterministic preprocessing before feature extraction:
1. **Resampling**: Standardized to **16,000 Hz** (matches Whisper and modern speech models).
2. **Channel Downmixing**: Dual/multichannel audio is averaged to single-channel **mono**.
3. **Leading/Trailing Silence Trimming**: Non-speech boundaries are trimmed using `librosa.effects.trim(top_db=25)` to prevent silent margins from skewing temporal statistics.
4. **Peak Normalization**: Waveforms are scaled such that the maximum absolute amplitude is 0.95:
   $$x_{\text{norm}}(t) = 0.95 \cdot \frac{x(t)}{\max(|x(t)|) + \epsilon}$$
5. **Quality Gate**: Signals with duration $< 0.3$ seconds or root-mean-square amplitude $< 0.001$ are rejected.

---

## 4. Acoustic Feature Engineering (43 Features)

Features were engineered using domain knowledge from acoustic phonetics and psychoacoustics:

| Feature Category | Dim | Description | Emotional Relevance |
|---|---|---|---|
| **MFCCs (1–13 Mean & Std)** | 26 | Mel-Frequency Cepstral Coefficients capturing the spectral envelope and vocal tract filter shape. | Distinguishes vowel color, timbre, and throat constriction changes across emotions. |
| **Pitch ($F_0$) Statistics** | 5 | Fundamental frequency tracked via the YIN algorithm (`fmin=65Hz`, `fmax=500Hz`). Extracted: Mean, Std, Min, Max, and Voiced Frame Fraction. | Primary indicator of physiological arousal; excitement and anger elevate $F_0$ mean and standard deviation. |
| **RMS Energy Statistics** | 3 | Short-time Root-Mean-Square signal energy. Extracted: Mean, Std, Max. | Correlates with vocal effort and loudness dynamics (high in anger, low in sadness/calm). |
| **Spectral Centroid (Mean, Std)** | 2 | Center of mass of the frequency spectrum ("brightness"). | Brightness increases with vocal tension and high-frequency harmonics (e.g. scream, surprise). |
| **Spectral Bandwidth (Mean, Std)** | 2 | Spread of frequencies around the spectral centroid. | Reflects spectral richness and harmonic dispersion. |
| **Spectral Rolloff (Mean, Std)** | 2 | Frequency below which 85% of total spectral energy lies. | Highlights energy distribution between low and high registers. |
| **Zero Crossing Rate (Mean, Std)** | 2 | Rate of sign changes in the temporal waveform. | Detects noisiness, unvoiced consonants, and breathiness. |
| **Pause Fraction** | 1 | Fraction of silent/unvoiced frames below $-35\text{ dB}$ threshold. | Pauses and hesitation increase in sadness, contemplative states, or grief. |
| **Total** | **43** | All features are standardized via `StandardScaler`. | — |

---

## 5. Exploratory Data Analysis (EDA) Insights

Plots generated in `backend/ml/eda_plots/`:
1. **`emotion_distribution.png`**: Confirms balanced classes (192 each, 96 for neutral).
2. **`acoustic_features_boxplot.png`**:
   - Angry and Happy speech exhibit markedly elevated $F_0$ mean (>220 Hz for females, >140 Hz for males) and higher RMS energy.
   - Sad and Calm speech exhibit compressed pitch variance ($F_0$ std < 25 Hz) and elevated pause fractions.
3. **`feature_correlation_heatmap.png`**:
   - Spectral Centroid, Bandwidth, and Rolloff show high positive correlation ($r > 0.85$).
   - RMS Energy and Spectral Centroid moderately correlate ($r \approx 0.62$).
   - Lower MFCCs (1–4) capture energy distribution, while higher MFCCs (9–13) capture fine spectral formant details.
4. **`pca_emotions.png`**:
   - PC1 aligns with **Arousal / Energy** (separating Angry/Happy/Surprised from Sad/Calm/Neutral).
   - Linear separation between **Valence** (Happy vs Angry) is notably weak in 2D space, illustrating the classic limitation of acoustic features alone in separating positive high-arousal states from negative high-arousal states.

---

## 6. Machine Learning Model Comparison

Three algorithms were tuned and benchmarked on identical 43-feature vectors across two validation paradigms:

### A. Speaker-Independent Evaluation (`GroupKFold`, $k=5$, grouped by `actor_id`)
*Primary benchmark: No actor appears in both train and validation folds.*

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| **Support Vector Machine (RBF, $C=5.0$)** | **0.4903** | **0.4794** | **0.4882** | **0.4885** | **0.4779** |
| Logistic Regression (L2, $C=1.0$) | 0.4785 | 0.4645 | 0.4737 | 0.4640 | 0.4674 |
| Random Forest ($n=300$, `max_depth=16`) | 0.4507 | 0.4214 | 0.4368 | 0.4250 | 0.4316 |

### B. Random Stratified Evaluation (`StratifiedKFold`, $k=5$)
*Demonstrating Speaker Leakage: Samples from the same actor appear in both train and test.*

| Model | Accuracy | Macro F1 | Weighted F1 | Inflation vs GroupKFold |
|---|---|---|---|---|
| **Support Vector Machine (RBF)** | **0.6868** | **0.6719** | **0.6832** | **+19.65%** |
| Random Forest | 0.5944 | 0.5721 | 0.5844 | +14.37% |
| Logistic Regression | 0.5618 | 0.5496 | 0.5589 | +8.33% |

### Why Did SVM Outperform Random Forest?
With $N=1,440$ samples and $D=43$ continuous, normalized acoustic features, the RBF kernel constructs smooth non-linear decision boundaries that generalize better across speaker variations than orthogonal axis-aligned decision trees, which are prone to overfitting to specific actor pitch ranges.

---

## 7. Detailed Per-Class Breakdown (Top Model: SVM)

| Emotion | Precision | Recall | F1-Score | Support | Analysis |
|---|---|---|---|---|---|
| **Angry** | 0.607 | 0.651 | **0.628** | 192 | Distinct acoustic signature (high energy, high pitch variance, high centroid). |
| **Surprised** | 0.667 | 0.531 | **0.591** | 192 | Sharp pitch spikes and high voiced fraction; strong precision. |
| **Calm** | 0.588 | 0.594 | **0.591** | 192 | Low energy and gentle contours; distinct from high-arousal classes. |
| **Disgust** | 0.508 | 0.630 | **0.563** | 192 | Characterized by vocal fry, lower centroid, and distinct MFCC profiles. |
| **Happy** | 0.388 | 0.406 | **0.397** | 192 | Often confused with Angry and Surprised due to shared high arousal acoustics. |
| **Fearful** | 0.364 | 0.396 | **0.379** | 192 | Moderate pitch elevation; overlaps with both Sad and Surprised. |
| **Neutral** | 0.431 | 0.292 | **0.348** | 96 | Smaller class size; frequently confused with Calm. |
| **Sad** | 0.356 | 0.323 | **0.339** | 192 | Overlaps with Calm and Neutral in low-energy regions. |
| **Macro Average** | **0.489** | **0.478** | **0.479** | 1440 | 3.9x higher than random chance ($12.5\%$). |

---

## 8. Runtime Integration & Reliability Guardrails

1. **Prediction Abstention**:
   The model produces calibrated posterior probabilities via Platt scaling (`SVC(probability=True)`). If the top predicted class probability is $< 0.35$ (or margin over runner-up is $< 0.08$), the system sets `is_reliable=false` and returns `emotion_label="uncertain"`.
2. **Acoustic Trait Extraction**:
   The response returns physical traits (`pitch_f0_mean_hz`, `rms_energy_mean`, `pause_fraction`, `speech_rate_estimate`) alongside the probability distribution.
3. **Frontend Integration**:
   The conversation interface in `src/pages/Conversation.tsx` displays these acoustic metrics in real time with gentle, qualified language (*"Measured vocal pattern suggests..."*).
4. **Zero Audio Persistence**:
   Uploaded audio is processed in memory and immediately discarded.

---

## 9. Conclusion & Transition to Phase 4

Phase 3 established that:
1. Traditional acoustic features ($F_0$, energy, MFCCs) capture **vocal arousal** effectively, separating calm/sad speech from angry/surprised speech.
2. Acoustic features alone struggle with **vocal valence** (distinguishing Happy from Angry) without lexical context.
3. Traditional models cap at ~49% speaker-independent accuracy due to handcrafted feature limitations.

In **Phase 4**, we will investigate self-supervised speech representations (**Wav2Vec 2.0** and **HuBERT**), fine-tuning on emotional speech to learn rich latent representations that generalize across speakers and languages.
