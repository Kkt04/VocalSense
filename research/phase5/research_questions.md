# VocalSense Phase 5: Research Questions, Scientific Motivations & Hypotheses

## 1. Project Context & Research Motivation
VocalSense is a voice-first, emotionally responsive conversational companion designed under the guiding principle:
> *"Sometimes, you just need someone to listen."*

To be a supportive, trustworthy companion, VocalSense must understand:
1. **What is said**: Accurate speech transcription preserving native syntax and language.
2. **How it is said**: Acoustic dynamics, vocal prosody, pitch, and emotion-correlated cues.
3. **How to respond**: Warm, context-aware, language-matched replies without claiming unearned clinical or psychological knowledge.

In Phase 3 and Phase 4, VocalSense trained baseline traditional machine learning models (SVM, Random Forest) and self-supervised deep learning representations (Wav2Vec 2.0, HuBERT) on the benchmark RAVDESS corpus. While HuBERT achieved 67.01% accuracy under 5-fold cross-validation, benchmark accuracy alone does not guarantee real-world reliability.

**Phase 5 investigates whether VocalSense's speech emotion recognition system is genuinely reliable in human conversations.**

---

## 2. Core Research Questions (RQs)

### RQ1: Generalisation Across Unseen Speakers
- **Question**: Does the model learn invariant emotional speech characteristics, or does it merely memorize the idiosyncratic vocal tracts, pitch ranges, and accents of individual training actors?
- **Hypothesis**: Naive random train-test splitting introduces severe *speaker leakage*, artificially inflating Macro-F1 by 10%–20%. Strict speaker-independent partitioning (`GroupKFold` by actor ID) will expose the true generalization gap.
- **Sub-question (Cross-Gender Generalisation)**: Can a model trained exclusively on male voices generalize to female voices (and vice versa), or do fundamental pitch ($F_0$) disparities cause catastrophic failure?

### RQ2: Cross-Corpus Generalisation & Harmonisation
- **Question**: Can a speech emotion recognition model trained on controlled studio speech (RAVDESS) generalize to different corpora such as CREMA-D (diverse actors), MSP-Podcast (naturalistic conversation), and IITKGP-SEHSC (Hindi)?
- **Hypothesis**: Incompatible emotion taxonomies (e.g. *calm* vs *neutral*) and acoustic domain shifts (microphone color, room acoustics, posed theatricality vs natural conversation) cause severe performance collapse without systematic label harmonization.

### RQ3: Multilingual SER & Cross-Lingual Representation Drift
- **Question**: Does an SER model trained on English speech generalize to Hindi and Hinglish? How does intra-sentential code-switching impact self-supervised speech representations?
- **Hypothesis**: Pretrained speech backbones (Wav2Vec2, HuBERT) trained on English LibriSpeech undergo substantial embedding drift when presented with Hindi/Hinglish phonology. Multilingual ASR (Whisper) does **NOT** imply multilingual SER; language-specific acoustic prosody differs fundamentally.

### RQ4: Acoustic Robustness Under Real-World Degradation
- **Question**: How resilient is VocalSense when audio is captured through low-cost smartphone mics, at variable distances, amidst background environmental noise, room echo, telephony downsampling (8 kHz), or in short (<1.0s) speech bursts?
- **Hypothesis**: Handcrafted acoustic features (MFCC, $F_0$) collapse rapidly under additive noise (SNR $\le 10$ dB) and telephony filtering, while deep self-supervised representations exhibit greater noise retention but still require acoustic quality gating.

### RQ5: Model Explainability & Attribution
- **Question**: Which acoustic and representational features drive emotion classification? Are model explanations consistent across traditional and deep architectures?
- **Hypothesis**: Pitch ($F_0$) and RMS energy dominate arousal discrimination (angry/happy vs sad/calm), while spectral centroid and MFCCs separate valence. In deep models, topological embedding clusters form distinct manifold boundaries corresponding to phonetic and prosodic features.

### RQ6: Confidence Calibration & Selective Abstention
- **Question**: Are classifier probabilities well-calibrated (i.e. does $p=0.85$ imply 85% empirical accuracy)? Can post-hoc temperature scaling and entropy gating enable VocalSense to abstain gracefully when uncertainty is high?
- **Hypothesis**: Raw classifier softmax outputs are severely overconfident (ECE $\sim 15\%-25\%$). Temperature scaling will reduce ECE to $<3\%$, and dual-threshold abstention will lower selective risk by $>5\%$, allowing the system to decline low-confidence predictions gracefully.

### RQ7: Speech & Language Pipeline Integrity
- **Question**: Does VocalSense faithfully preserve original spoken Hindi (in Devanagari) and Hinglish (in Latin script) without automatic translation to English?
- **Hypothesis**: Configuring Whisper with `task="transcribe"` combined with script-aware language validation prevents unwanted translation and guarantees 100% language-matched conversational responses.
