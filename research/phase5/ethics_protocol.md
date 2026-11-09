# VocalSense Phase 5: Ethical Consent-Based Multilingual Data Collection Protocol

## 1. Motivation
Public speech emotion recognition datasets (e.g., RAVDESS, CREMA-D) are overwhelmingly composed of native English speakers reading artificial scripts in studio booths. High-quality, naturalistic emotional speech corpora for Indian languages—specifically **Hindi** and **code-switched Hinglish**—do not exist under permissive open-access licenses.

To address this limitation ethically in future research phases, this document establishes a rigorous, consent-first data collection protocol complying with Indian DPDP (Digital Personal Data Protection Act 2023) and global GDPR standards.

---

## 2. Participant Recruitment & Informed Consent
1. **Explicit, Granular Consent**:
   - Participants must be informed in their primary language (Hindi or English) of the exact purpose of the collection: advancing multilingual acoustic emotion research.
   - Consent must be granular: Participants may consent to transcript analysis while withholding permission for audio distribution.
2. **Right to Immediate Withdrawal**:
   - Every recording session is tagged with an anonymous Participant UUID.
   - Participants have 30 calendar days to request unconditional deletion of their audio recordings.
3. **No Coercion or Medical Framing**:
   - Recruitment material must state unequivocally: *"VocalSense is a conversational companion, not a clinical diagnostic tool or psychological assessment service."*

---

## 3. Naturalistic Elicitation Methodology
To avoid the artificial, theatrical over-acting present in RAVDESS:
- **Guided Memory Recall**: Participants are prompted to speak casually about real memories (e.g., a fond childhood memory for happiness, a time they encountered an unfair bureaucratic delay for frustration, or a quiet rainy evening for calm).
- **Conversational Dyad**: Natural dialogue with a bilingual research assistant rather than isolated microphone reading.
- **Code-Switching Preservation**: Participants are explicitly encouraged to speak naturally in whichever language mix they use daily (Hinglish, Hindi, or English).

---

## 4. Dual-Expert Annotation & Disagreement Handling
- Every 3–5 second audio segment must be annotated independently by **three bilingual native speakers**.
- Annotators evaluate:
  1. Primary Emotion Category (Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise).
  2. Valence (1–5) and Arousal (1–5) dimensional scales.
  3. Code-switching token ratio (% Hindi vs % English words).
- Disagreement handling: Segments where Fleiss' Kappa inter-annotator agreement is below 0.60 are marked as *ambiguous / blended* and never forced into a single discrete class.

---

## 5. Storage, Anonymisation & Privacy
- **Acoustic De-identification**: Participant names, locations, and personal identifiers in the audio are bleeped/redacted.
- **Raw Audio Ephemerality**: Research audio is stored in an encrypted S3 bucket (AES-256) with strict role-based access control.
- Under no circumstances is user audio ever committed to Git repositories or exposed via public endpoints.
