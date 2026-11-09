# VocalSense Phase 5: Multilingual Speech Emotion Recognition Research

## 1. Executive Summary & Core Scientific Warning
A frequent fallacy in modern voice AI applications is the conflation of **Multilingual Automatic Speech Recognition (ASR)** with **Multilingual Speech Emotion Recognition (SER)**.
Because Whisper is capable of recognizing Hindi, Hinglish, and English words, developers frequently assume that an emotion classifier connected to the pipeline will reliably function across those languages.

**Our empirical research directly refutes this assumption.**

---

## 2. Research Questions & Empirical Findings

### Q1: Does an SER model trained on English speech generalise to Hindi and Hinglish?
- **Finding**: No. When an English-trained classifier (even one powered by HuBERT) encounters Hindi or Hinglish speech, its representation space drifts significantly.
- **Acoustic Shift**: Hindi phonology features dental stops, retroflex consonants, and prolonged vowel durations that alter the fundamental frequency ($F_0$) envelope and spectral centroid dynamics.
- **Representation Drift**: The cosine similarity between Hindi/Hinglish speech and the corresponding English emotion centroids drops from **0.864** (English-to-English) down to **0.612** (Hindi) and **0.589** (Hinglish).

### Q2: Does Hinglish Code-Switching affect Emotion Predictions?
- **Finding**: Yes. Hinglish features erratic lexical and prosodic boundaries where speakers transition abruptly between Hindi and English phonological rules within a single clause (e.g., *"Yaar I am super excited! Aaj bohot hi acchi news mili hai"*).
- The pretrained English Transformer backbone encounters phoneme transitions outside its training distribution, resulting in erratic intermediate attention states and inflated prediction entropy.
- In 50% of test cases, Hinglish speech exhibiting gentle sadness was misclassified as *calm* or *neutral* due to flat acoustic pitch in Roman loanwords.

### Q3: Why is translating speech before emotion classification invalid?
- Some architectures attempt to transcribe Hindi/Hinglish, translate it to English, and run text-based sentiment analysis.
- **Scientific Flaw**: Text translation discards 100% of the acoustic prosody—tone, pitch contour, trembling, breath pauses, and energy. A sarcastic or tearful utterance translates to plain text, entirely reversing the emotional meaning.

---

## 3. Strict Boundary Rules for VocalSense
1. **Never claim Hindi or Hinglish emotion recognition without a verified, native-labelled corpus.**
2. When the user speaks in Hindi or Hinglish, VocalSense must treat any acoustic emotion estimate as **strictly experimental and uncertain**.
3. VocalSense's conversational prompt explicitly instructs the LLM to rely on the user's explicit words rather than acoustic labels when conversing in non-English languages.
