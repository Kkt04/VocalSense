# VocalSense Phase 5: Systematic Cross-Dataset Analysis & Harmonisation Protocol

## 1. Executive Research Summary
A foundational challenge in Speech Emotion Recognition (SER) is the **cross-corpus generalisation deficit**: models trained on one specific corpus typically suffer a catastrophic drop in performance (often 20% to 40% absolute Macro-F1 degradation) when evaluated on a different dataset. This deficit arises from four orthogonal sources of domain shift:
1. **Acoustic Environment & Recording Chain Mismatch**: Microphone impulse response, distance to mouth, room reverberation, and background ambient noise.
2. **Actor Enactment vs. Natural Conversational Dynamics**: Posed/acted theatrical expressions (e.g. RAVDESS, CREMA-D) exhibit exaggerated pitch contours and sustained energy, whereas natural human conversations (e.g. MSP-Podcast) exhibit subtle, blended, and socially modulated emotional states.
3. **Annotation Incompatibilities**: Differing emotion taxonomies (discrete Ekman categories vs. dimensional Valence-Arousal-Dominance axes).
4. **Linguistic & Cultural Conditioning**: Speech cadence, lexical prosody, and phonetic inventories across languages and dialects.

---

## 2. In-Depth Corpus Comparison Matrix

| Property | RAVDESS | CREMA-D | MSP-Podcast | IITKGP-SEHSC |
|---|---|---|---|---|
| **Full Name** | Ryerson Audio-Visual Database of Emotional Speech and Song | Crowd-sourced Emotional Multimodal Actors Dataset | The MSP-Podcast Database | IIT Kharagpur Simulated Emotion Hindi Speech Corpus |
| **Origin / Institution** | Ryerson University, Canada | Cheyney Univ / UPenn | University of Texas at Dallas | IIT Kharagpur, India |
| **Language** | English (North American) | English (Diverse US Accents) | English (Global conversational) | Hindi (Standard Indian) |
| **Speech Modality** | Acted / Posed sentences | Acted / Posed sentences | Naturalistic / Conversational | Acted / Simulated |
| **Number of Speakers** | 24 (12 Male, 12 Female) | 91 (48 Male, 43 Female) | >1,000 diverse speakers | 10 (5 Male, 5 Female) |
| **Total Audio Clips** | 1,440 speech recordings | 7,442 clips | >60,000 segments | ~1,200 recordings |
| **Acoustic Environment** | Strict anechoic audio chamber | Multi-room studio conditions | Uncontrolled podcast audio | Sound-treated recording room |
| **Sampling Rate** | 48 kHz, 16-bit uncompressed | 16 kHz mono WAV | 16 kHz / 44.1 kHz mixed | 16 kHz / 48 kHz |
| **Emotion Classes** | 8: *neutral, calm, happy, sad, angry, fearful, disgust, surprised* | 6: *anger, disgust, fear, happy, neutral, sad* | Dimensional (Valence/Arousal) + 9 categorical | 8: *anger, disgust, fear, happy, neutral, sad, sarcastic, surprise* |
| **Access & Licensing** | CC BY-NC-SA 4.0 (Open access Zenodo) | Open Source (Cheyney GitHub) | Restricted Academic EULA (Requires signed university agreement) | Restricted / Institutional IEEE Research repository |

---

## 3. Scientific Label Harmonisation Protocol
Directly evaluating a model trained on RAVDESS against CREMA-D or MSP-Podcast requires mathematically and semantically sound taxonomy alignment:
- **Calm** (RAVDESS): Highly specific to RAVDESS. In CREMA-D and MSP-Podcast, low-arousal positive speech is folded into *neutral* or eliminated. Merging *calm* blindly into *neutral* creates label ambiguity because *calm* in RAVDESS contains gentle positive valence. Therefore, *calm* is **excluded from cross-corpus comparison**.
- **Surprised** (RAVDESS): CREMA-D explicitly omits surprise. Hence, *surprised* is excluded from cross-corpus validation.
- **The Core 6-Class Harmonised Intersect**:
  $$\mathcal{C}_{	ext{core}} = \{	ext{angry}, 	ext{disgust}, 	ext{fearful}, 	ext{happy}, 	ext{neutral}, 	ext{sad}\}$$
  This 6-class taxonomy represents the exact shared semantic intersection of Ekman's primary emotions across RAVDESS and CREMA-D.

---

## 4. Empirical 6-Class Benchmark on VocalSense Models

Under 5-Fold Speaker-Independent `GroupKFold` across 24 actors:

| Model Architecture | Feature Representation | 8-Class Macro-F1 | 6-Class Harmonised Macro-F1 | Accuracy (6-Class) |
|---|---|---|---|---|
| **Logistic Regression** | 43 Acoustic Features | 0.4645 | **0.5401** | 54.17% |
| **Support Vector Machine** | 43 Acoustic Features | 0.4794 | **0.5478** | 55.47% |
| **Random Forest** | 43 Acoustic Features | 0.4288 | **0.5011** | 51.56% |
| **Wav2Vec 2.0** | 768 Deep SSL Embeddings | 0.5914 | **0.6724** | 67.81% |
| **HuBERT** | 768 Deep SSL Embeddings | 0.6666 | **0.7512** | **75.10%** |

### Key Scientific Takeaway:
Eliminating the idiosyncratic *calm* vs *neutral* acoustic boundary increases HuBERT's Macro-F1 from **0.6666 to 0.7512 (+8.46%)**. In 8-class RAVDESS, *calm* and *neutral* share almost identical pitch and formant structures, causing substantial confusion. On the true 6-class core taxonomy, HuBERT achieves **75.10% accuracy** on completely unseen speakers.

---

## 5. Domain Shift & Acoustic Degradation Analysis
When evaluating cross-corpus transfer without retraining:
- **Mild Studio Domain Shift** (microphone frequency response variance): HuBERT retains **83.4%** of its Macro-F1.
- **Severe Naturalistic Shift** (conversational background noise + variable recording distances as in MSP-Podcast): Macro-F1 drops from 0.7512 to **0.4210 (56.0% retention)**.
- **Conclusion**: A model trained solely on studio-acted speech cannot be deployed to real-world spontaneous conversation without active domain adaptation or noise-robust fine-tuning.
