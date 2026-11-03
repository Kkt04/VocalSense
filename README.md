# VocalSense

**More than speech recognition. A new way to listen, understand and respond.**

VocalSense is a multilingual, voice-first, emotion-aware AI conversational companion. It is designed to understand both:
1. **What a person says:** Speech recognition, transcription, language identification, and conversational context.
2. **How a person speaks:** Acoustic characteristics such as pitch variation, speaking rate, pauses, and vocal energy.

> **Phase 1, Phase 2 & Phase 3 Complete: Full-Stack Voice Stage, Speech Recognition & Speech Emotion Recognition (SER).**
> - **Phase 1**: Real microphone recording, audio format validation, and animated React frontend.
> - **Phase 2**: `faster-whisper` multilingual transcription for English, Hindi, and Hinglish with zero-translation guarantee, editable transcripts, and WER/CER evaluation.
> - **Phase 3**: Traditional ML Speech Emotion Recognition pipeline on RAVDESS (1,440 recordings, 24 actors), 43 acoustic features (`librosa`), speaker-independent 5-fold evaluation (`GroupKFold`), serialized SVM pipeline, FastAPI inference endpoints with abstention thresholds, and real-time frontend acoustic insights.

*Product Principle:* VocalSense is a calm, supportive conversational companion. It is **not** a therapist, a medical device, or a diagnostic tool. Model outputs are uncertain predictions, never facts or diagnoses.

---

## Architecture Overview

```
vocalsense/
├── backend/                               # FastAPI Python backend & ML pipeline
│   ├── app/
│   │   ├── main.py                        # FastAPI application, CORS, routers
│   │   ├── config.py                      # Pydantic Settings (sizes, Whisper, ML model path)
│   │   ├── api/
│   │   │   ├── routes_health.py           # GET /api/health
│   │   │   ├── routes_audio.py            # POST /api/audio/upload
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── transcription.py       # POST /api/v1/transcription/transcribe
│   │   │       └── emotion.py             # POST /api/v1/emotion/predict, GET /api/v1/emotion/health
│   │   ├── services/
│   │   │   ├── audio_validator.py         # Format, size, and magic bytes detection
│   │   │   ├── audio_storage.py           # Privacy-first ephemeral storage handler
│   │   │   ├── transcription/             # Phase 2 Whisper pipeline
│   │   │   │   ├── whisper_service.py     # faster-whisper model manager & inference
│   │   │   │   ├── audio_preprocessing.py # PyAV 16kHz mono resampling & decoding
│   │   │   │   └── language_detection.py  # Code-switching & Hinglish classification
│   │   │   └── emotion_recognition/       # Phase 3 SER runtime services
│   │   │       ├── __init__.py
│   │   │       ├── preprocessor.py        # 16kHz mono resampling & silence trimming
│   │   │       ├── feature_extractor.py   # 43 acoustic features (MFCC, F0, RMS, spectral)
│   │   │       └── emotion_service.py     # Scikit-learn SVM inference & abstention logic
│   │   └── schemas/
│   │       ├── audio.py                   # Pydantic upload/health schemas
│   │       ├── transcription.py           # Pydantic transcription response schemas
│   │       └── emotion.py                 # Pydantic emotion prediction & health schemas
│   ├── ml/                                # Phase 3 Machine Learning training & EDA
│   │   ├── dataset.py                     # RAVDESS downloader, filename parser, manifest
│   │   ├── audio_preprocessing.py         # Training-time audio normalization & trimming
│   │   ├── feature_extraction.py          # 43 acoustic features batch extractor
│   │   ├── eda.py                         # EDA visualization scripts & distribution plots
│   │   ├── eda_plots/                     # Generated charts (boxplots, PCA, heatmap, matrices)
│   │   ├── train.py                       # GroupKFold vs Stratified training & benchmarking
│   │   └── EXPERIMENT_REPORT.md           # Comprehensive SER scientific report
│   ├── saved_models/                      # Serialized ML artifacts
│   │   ├── emotion_classifier.joblib      # Production SVM pipeline (StandardScaler + SVC)
│   │   └── model_metadata.json            # Metrics, classes, feature names, threshold
│   ├── evaluation/
│   │   └── wer_cer.py                     # Word Error Rate & Character Error Rate evaluator
│   ├── tests/
│   │   ├── test_health.py                 # Health check unit tests
│   │   ├── test_audio_upload.py           # Audio upload validation unit tests
│   │   ├── test_transcription.py          # Whisper transcription & preprocessing tests
│   │   └── test_emotion.py                # SER runtime & abstention unit tests (7 tests)
│   └── requirements.txt
├── data/
│   └── ravdess/                           # RAVDESS dataset (1,440 WAV files & metadata.csv)
├── src/                                   # React + TypeScript frontend
│   ├── components/
│   │   ├── Orb.tsx                        # Animated responsive voice orb (Framer Motion)
│   │   ├── Waveform.tsx                   # Live frequency waveform
│   │   ├── AudioUploader.tsx              # Audio file uploader with direct Whisper transcription
│   │   └── ...
│   ├── hooks/
│   │   ├── useAudioRecorder.ts            # Real Web Audio API AnalyserNode + MediaRecorder
│   │   └── useSimulatedAudio.ts           # Fallback simulation for no-mic preview
│   ├── services/
│   │   └── api.ts                         # Backend API client (health, transcribe, emotion)
│   └── pages/
│       ├── Landing.tsx                    # Landing hero and overview
│       └── Conversation.tsx               # Voice stage: transcripts + real-time emotion insights
└── .env.example
```

---

## Quick Start Guide

### 1. Backend Setup (FastAPI + Whisper + SER ML Model)

Requirements: **Python 3.10+**

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run all backend tests (23 unit tests across Phases 1, 2, and 3)
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. pytest backend/tests -v

# 4. Optional: Re-train the SER models and generate EDA plots
# (Uses RAVDESS dataset in data/ravdess/)
python3 backend/ml/eda.py
python3 backend/ml/train.py

# 5. Start the backend development server
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at:
- Health check: `http://localhost:8000/api/health`
- Transcription health: `http://localhost:8000/api/v1/transcription/health`
- Emotion recognition health: `http://localhost:8000/api/v1/emotion/health`
- Emotion prediction: `POST http://localhost:8000/api/v1/emotion/predict`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`

---

### 2. Frontend Setup (React + TypeScript)

Requirements: **Node.js 18+** and npm

```bash
# 1. Install dependencies
npm install

# 2. Start Vite development server
npm run dev
```

The frontend will be running at `http://localhost:5173/#/talk`.

Other commands:
```bash
npm run typecheck    # Validate TypeScript types
npm run build        # Production build
npm run preview      # Preview production build locally
```

---

## Phase 3 Benchmark & Speaker Leakage Results

Traditional ML models evaluated on the **RAVDESS** dataset (1,440 recordings, 24 actors, 8 emotions) using 43 acoustic features:

### Primary Evaluation: Speaker-Independent (`GroupKFold`, $k=5$, grouped by `actor_id`)
*No actor appears in both training and test folds. Chance level is 12.5%.*

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| **Support Vector Machine (RBF, $C=5.0$)** | **49.03%** | **0.4794** | **0.4882** | **0.4885** | **0.4779** |
| Logistic Regression (L2, $C=1.0$) | 47.85% | 0.4645 | 0.4737 | 0.4640 | 0.4674 |
| Random Forest ($n=300$, `max_depth=16`) | 45.07% | 0.4214 | 0.4368 | 0.4250 | 0.4316 |

### Speaker Leakage Demonstration: Random Stratified (`StratifiedKFold`, $k=5$)
*Actors appear in both train and test folds.*

| Model | Accuracy | Macro F1 | Weighted F1 | Leakage Inflation |
|---|---|---|---|---|
| **Support Vector Machine (RBF)** | **68.68%** | **0.6719** | **0.6832** | **+19.65%** |
| Random Forest | 59.44% | 0.5721 | 0.5844 | +14.37% |
| Logistic Regression | 56.18% | 0.5496 | 0.5589 | +8.33% |

**Key Finding**: Random train/test splits falsely inflate apparent accuracy by nearly **20%** because models memorize speaker identity rather than generalizable emotional acoustics.

---

## Deliverables Summary

| Phase | Milestone | Key Features | Status |
|---|---|---|---|
| **Phase 1** | Foundation & Product | React/TS interface, Web Audio API recording, FastAPI backend, format validation, privacy-first ephemeral storage. | Completed |
| **Phase 2** | Speech Recognition | `faster-whisper` CTranslate2 engine, English/Hindi/Hinglish transcription, language detection, editable transcripts, WER/CER evaluation. | Completed |
| **Phase 3** | Speech Emotion Recognition | RAVDESS dataset pipeline, 43 acoustic features (`librosa`), EDA visualizations, GroupKFold vs Stratified ML training, SVM serialized pipeline, abstention guardrails, real-time acoustic insights in UI. | Completed |
| **Phase 4** | Deep Learning SER | Wav2Vec2 and HuBERT self-supervised representations, cross-corpus Hindi evaluation. | Next |

---

## Roadmap

- [x] **Phase 1 — Build the Basic Product**: Working frontend and backend foundation, recording controls, audio upload endpoint, health check, format validation, unit tests.
- [x] **Phase 2 — Speech Recognition**: Whisper integration (`faster-whisper`), multilingual STT (English, Hindi, Hinglish), audio preprocessing, editable transcripts, language detection, WER/CER evaluation pipeline.
- [x] **Phase 3 — First Machine Learning Model**: Baseline speech emotion classification on RAVDESS, acoustic feature extraction (MFCC, pitch/F0, RMS energy, pauses), SVM & Random Forest comparison, speaker-independent evaluation, FastAPI integration.
- [ ] **Phase 4 — Advanced Speech Models**: Deep learning representations with Wav2Vec2, HuBERT, and comparative evaluation.
- [ ] **Phase 5 — Research and Reliability**: Multilingual transfer learning, noise robustness, explainability (SHAP / feature importance), calibrated uncertainty estimation.
- [ ] **Phase 6 — Complete VocalSense**: LLM conversational engine, conversation styles, text-to-speech, voice insights panel, and deployment.
