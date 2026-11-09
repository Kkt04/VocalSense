# VocalSense Phase 5: Honest Research Limitations, Boundary Conditions & Constraints

## 1. Dataset & Access Limitations

### 1.1 Lack of Open-Access Naturalistic Emotion Corpora
- **RAVDESS** is an *acted, theatrical* speech dataset recorded in an anechoic studio chamber by 24 professional actors reading scripted pseudo-sentences (*"Kids are talking by the door"*).
- In real conversational companion interactions, emotional expression is subtle, socially modulated, continuous, and intertwined with linguistic meaning rather than exaggerated into discrete, isolated peaks.
- **Access Restrictions on MSP-Podcast**: MSP-Podcast contains naturalistic podcast speech, but its licensing terms require an institutional End User License Agreement (EULA) signed by a recognized academic institution, preventing automated redistribution.
- **Access Restrictions on IITKGP-SEHSC**: IITKGP-SEHSC is an academic Hindi speech dataset restricted under IEEE ICDeCom 2011 copyright and institutional research permissions.

### 1.2 The Hinglish Labelled Speech Void
- No large-scale, openly licensed, peer-reviewed emotional speech corpus exists for **code-switched Hinglish**.
- Consequently, while VocalSense can transcribe Hinglish accurately via Whisper and converse in Hinglish via LLM prompts, its Speech Emotion Recognition classifier cannot be empirically certified on Hinglish without native training data.
- **Scientific Integrity Rule**: VocalSense explicitly documents this limitation and treats all vocal emotion estimates on Hindi and Hinglish as **unvalidated and experimental**.

---

## 2. Model & Representation Limitations

### 2.1 The Pretraining Linguistic Bias
- Pretrained speech backbones (**Wav2Vec 2.0 Base** and **HuBERT Base**) were pretrained on **LibriSpeech (960 hours of English read audiobooks)**.
- The self-supervised representations inherently encode the acoustic and phonological characteristics of English phonemes, syllable timing, and stress patterns.
- When exposed to syllable-timed languages (such as Hindi) or intra-sentential code-switching (Hinglish), the Transformer hidden layers exhibit substantial representation drift.

### 2.2 Discrete Category vs. Dimensional Reality
- Classifying complex human vocalizations into 8 discrete categories (*angry, calm, disgust, fearful, happy, neutral, sad, surprised*) is an oversimplification of human affect.
- In everyday conversation, emotions are dimensional (Valence, Arousal, Dominance) and frequently mixed (e.g. *bittersweet*, *anxious calm*, *frustrated sadness*). Forcing a single label onto blended speech creates artificial classification ambiguity.

---

## 3. Acoustic & Environmental Constraints

### 3.1 The Severe Noise Ceiling
- When environmental background noise exceeds 10 dB SNR, all existing SER models experience severe accuracy collapse.
- Low-cost microphones introduce significant frequency coloration and clipping distortion that corrupts both spectral centroids and deep convolutional feature maps.

### 3.2 Utterance Length Threshold
- Audio recordings shorter than 1.0–1.5 seconds do not contain sufficient vocal tract cycles or prosodic contours for reliable emotion recognition.
- Short affirmations (*"Okay"*, *"Yeah"*, *"Sure"*) cannot be reliably classified and must be rejected by the abstention gate.

---

## 4. Ethical & Product Positioning Boundaries
- **VocalSense is NOT a Therapist, Medical Doctor, or Clinical Diagnostic System.**
- It does not diagnose depression, anxiety, PTSD, or any psychiatric condition from vocal recordings.
- Emotion predictions are treated strictly as **probabilistic, uncertain acoustic observations**, never ground truth psychological facts.
- The user's explicit words and conversational choices always take absolute priority over any acoustic emotion estimate.
