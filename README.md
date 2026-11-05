# VocalSense

**More than speech recognition. A new way to listen, understand and respond.**

VocalSense is a multilingual, voice-first, emotion-aware AI conversational companion. It is designed to understand both:
1. **What a person says:** Speech recognition, transcription, language identification, and conversational context.
2. **How a person speaks:** Acoustic characteristics such as pitch variation, speaking rate, pauses, and deep self-supervised vocal representations.
3. **How to respond naturally:** Contextual, language-matched companion responses in the user's spoken language across four supportive conversation styles.

> **Phase 1, Phase 2, Phase 3 & Phase 4 Complete: Full-Stack Voice Stage, Speech Recognition, Traditional SER, Deep Learning SER (Wav2Vec2 + HuBERT), Multilingual Conversational Intelligence & MongoDB Persistence.**
> - **Phase 1**: Real microphone recording, audio format validation, and animated React frontend.
> - **Phase 2**: `faster-whisper` multilingual transcription for English, Hindi, and Hinglish with zero-translation guarantee, editable transcripts, and WER/CER evaluation.
> - **Phase 3**: Traditional ML Speech Emotion Recognition pipeline on RAVDESS (1,440 recordings, 24 actors), 43 acoustic features (`librosa`), speaker-independent 5-fold evaluation (`GroupKFold`), serialized SVM pipeline, FastAPI inference endpoints with abstention thresholds, and real-time frontend acoustic insights.
> - **Phase 4**: Deep learning speech representations with **Wav2Vec 2.0** and **HuBERT**, achieving **67.01% Accuracy / 0.6666 Macro-F1** (an +18.72% Macro-F1 improvement over Phase 3 baseline), multilingual language matching (Hindi in Devanagari, Hinglish in Latin script, English, Spanish, French, German, Japanese, Bengali, Tamil), response validation layer, 4 companion styles, and **MongoDB** primary persistence.

*Product Principle:* VocalSense is a calm, supportive conversational companion. It is **not** a therapist, a medical device, or a diagnostic tool. Model outputs are uncertain predictions, never facts or diagnoses.

---

## Architecture Overview

```
vocalsense/
├── backend/                               # FastAPI Python backend & ML pipeline
│   ├── app/
│   │   ├── main.py                        # FastAPI application, CORS, routers, lifespan
│   │   ├── config.py                      # Pydantic Settings (sizes, Whisper, ML model path, MongoDB)
│   │   ├── db/
│   │   │   └── mongodb.py                 # MongoDB connection manager (Motor / PyMongo / Fallback)
│   │   ├── api/
│   │   │   ├── routes_health.py           # GET /api/health
│   │   │   ├── routes_audio.py            # POST /api/audio/upload
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── transcription.py       # POST /api/v1/transcription/transcribe
│   │   │       ├── emotion.py             # POST /api/v1/emotion/predict, GET /api/v1/emotion/health
│   │   │       └── conversation.py        # POST /api/v1/conversation/respond, GET /history
│   │   ├── services/
│   │   │   ├── audio_validator.py         # Format, size, and magic bytes detection
│   │   │   ├── audio_storage.py           # Privacy-first ephemeral storage handler
│   │   │   ├── transcription/             # Phase 2 Whisper pipeline
│   │   │   │   ├── whisper_service.py     # faster-whisper model manager & inference
│   │   │   │   ├── audio_preprocessing.py # PyAV 16kHz mono resampling & decoding
│   │   │   │   └── language_detection.py  # Code-switching & Hinglish classification
│   │   │   ├── emotion_recognition/       # Phase 3 & Phase 4 SER runtime services
│   │   │   │   ├── __init__.py
│   │   │   │   ├── wav2vec2_model.py      # Wav2Vec 2.0 contextual embedding pipeline (768-dim)
│   │   │   │   ├── hubert_model.py        # HuBERT masked prediction embedding pipeline (768-dim)
│   │   │   │   ├── embedding_extractor.py # Dataset-level embedding extraction & disk caching
│   │   │   │   ├── deep_emotion_classifier.py # Downstream calibrated classifier
│   │   │   │   ├── model_comparison.py    # Speaker-independent benchmark evaluation
│   │   │   │   ├── preprocessor.py        # 16kHz mono resampling & silence trimming
│   │   │   │   ├── feature_extractor.py   # 43 acoustic features (MFCC, F0, RMS, spectral)
│   │   │   │   └── emotion_service.py     # Unified runtime inference & abstention logic
│   │   │   ├── language/                  # Phase 4 Language intelligence
│   │   │   │   ├── language_detector.py   # Multi-signal language & code-switching detector
│   │   │   │   └── language_validator.py  # Response language validator & drift guardrails
│   │   │   ├── conversation/              # Phase 4 Conversational companion engine
│   │   │   │   ├── response_generator.py  # Language-matched, style-faithful response generator
│   │   │   │   └── language_prompt.py     # System instructions and non-diagnostic emotion guardrails
│   │   │   └── tts/                       # Phase 4 Multilingual Text-to-Speech
│   │   │       └── tts_service.py         # Modular gTTS synthesis to base64 audio
│   │   └── schemas/
│   │       ├── audio.py                   # Pydantic upload/health schemas
│   │       ├── transcription.py           # Pydantic transcription response schemas
│   │       ├── emotion.py                 # Pydantic emotion prediction & health schemas
│   │       ├── conversation.py            # Pydantic conversation request & response schemas
│   │       └── mongodb.py                 # Pydantic MongoDB collection schemas
│   ├── ml/                                # Machine Learning training, benchmarks & EDA
│   │   ├── dataset.py                     # RAVDESS downloader, filename parser, manifest
│   │   ├── audio_preprocessing.py         # Training-time audio normalization & trimming
│   │   ├── feature_extraction.py          # 43 acoustic features batch extractor
│   │   ├── train.py                       # Phase 3 traditional baseline training
│   │   ├── train_deep_ser.py              # Phase 4 deep learning training & comparative benchmark
│   │   ├── EXPERIMENT_REPORT.md           # Phase 3 SER scientific report
│   │   └── PHASE4_REPORT.md               # Phase 4 comprehensive technical & scientific report
│   ├── saved_models/                      # Serialized ML artifacts
│   │   ├── emotion_classifier.joblib      # Production Phase 3 SVM pipeline
│   │   ├── model_metadata.json            # Phase 3 SVM metadata
│   │   └── phase4/                        # Phase 4 Deep learning artifacts
│   │       ├── wav2vec2_emotion_classifier.joblib
│   │       ├── wav2vec2_metadata.json
│   │       ├── hubert_emotion_classifier.joblib
│   │       ├── hubert_metadata.json
│   │       └── model_comparison_results.json # Empirical benchmark results
│   ├── evaluation/
│   │   └── wer_cer.py                     # Word Error Rate & Character Error Rate evaluator
│   ├── tests/                             # Comprehensive test suite (53 tests)
│   │   ├── test_health.py                 # Health check unit tests (2 tests)
│   │   ├── test_audio_upload.py           # Audio upload validation unit tests (5 tests)
│   │   ├── test_transcription.py          # Whisper transcription & preprocessing tests (9 tests)
│   │   ├── test_emotion.py                # Traditional SER runtime unit tests (7 tests)
│   │   ├── test_deep_emotion.py           # Deep learning SER unit tests (6 tests)
│   │   ├── test_multilingual_conversation.py # Multilingual companion unit tests (19 tests)
│   │   └── test_mongodb.py                # MongoDB persistence & fallback unit tests (5 tests)
│   └── requirements.txt
├── data/
│   ├── ravdess/                           # RAVDESS dataset (1,440 WAV files & metadata.csv)
│   └── embeddings/                        # Precomputed Wav2Vec2 and HuBERT embeddings
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
│   │   └── api.ts                         # Backend API client (transcribe, emotion, conversation)
│   └── pages/
│       ├── Landing.tsx                    # Landing hero and overview
│       └── Conversation.tsx               # Voice stage: transcripts, emotion insights, companion chat
└── .env.example
```

---

## Empirical Benchmark & Speaker-Independent Results

All models evaluated on the **RAVDESS** dataset (1,440 recordings, 24 actors, 8 emotions) under a strict **5-Fold Speaker-Independent Protocol (`GroupKFold`, grouped by `actor_id`)**:
*Zero speaker leakage: No actor appears in both training and test partitions. Chance level is 12.5%.*

| Model | Feature Representation | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Latency |
|---|---|---|---|---|---|---|---|
| **Support Vector Machine (Phase 3 Baseline)** | 43 Acoustic Features (MFCC, $F_0$, RMS, Spectral, Pauses) | 49.03% | 0.4794 | 0.4882 | 0.4885 | 0.4779 | ~8 ms |
| **Random Forest** | 43 Acoustic Features | 45.97% | 0.4288 | 0.4462 | 0.4578 | 0.4531 | ~6 ms |
| **Wav2Vec 2.0 (`facebook/wav2vec2-base`)** | 768 Deep Speech Embeddings | **60.00%** | **0.5914** | **0.6014** | **0.5929** | **0.5911** | ~45 ms (MPS) |
| **HuBERT (`facebook/hubert-base-ls960`)** | 768 Deep Speech Embeddings | **67.01%** | **0.6666** | **0.6732** | **0.6720** | **0.6647** | ~44 ms (MPS) |

> **Key Finding**: Pretrained self-supervised speech representations substantially outperform handcrafted acoustic features. HuBERT achieves a **+18.72% absolute Macro-F1 gain** over the Phase 3 SVM baseline, as its masked acoustic unit prediction objective encourages the Transformer encoder to capture both phonetic duration and fine-grained pitch trajectories.

---

## Quick Start Guide

### 1. Backend Setup (FastAPI + Deep Learning SER + MongoDB)

Requirements: **Python 3.10+**

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run all backend tests (53 unit & integration tests)
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib pytest backend/tests -v

# 4. Optional: Run the complete Phase 4 deep learning benchmark training
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib python backend/ml/train_deep_ser.py

# 5. Start the backend development server
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at:
- Root: `http://localhost:8000/`
- Health check: `http://localhost:8000/api/health`
- Transcription health: `http://localhost:8000/api/v1/transcription/health`
- Emotion recognition health: `http://localhost:8000/api/v1/emotion/health`
- Emotion prediction: `POST http://localhost:8000/api/v1/emotion/predict`
- Conversation health: `http://localhost:8000/api/v1/conversation/health`
- Conversation respond: `POST http://localhost:8000/api/v1/conversation/respond`
- Conversation history: `GET http://localhost:8000/api/v1/conversation/history/{conversation_id}`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`

---

### 2. MongoDB Database Setup

VocalSense requires **MongoDB** as its primary persistence engine. No SQL databases are supported.

#### Option A: Local MongoDB (Community Edition)
```bash
# macOS (Homebrew)
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0
```
Set in `.env`:
```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=vocalsense
```

#### Option B: MongoDB Atlas (Free Cloud Cluster)
1. Create a free M0 cluster at [mongodb.com/atlas](https://www.mongodb.com/cloud/atlas).
2. Obtain your connection URI and configure `.env`:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=vocalsense
```

*Note:* If MongoDB is temporarily unreachable, VocalSense operates in a resilient in-memory fallback mode so tests and demo workflows run seamlessly without crashing.

---

### 3. Frontend Setup (React + TypeScript)

Requirements: **Node.js 18+** and npm

```bash
# 1. Install dependencies
npm install

# 2. Typecheck and build
npm run typecheck
npm run build

# 3. Start Vite development server
npm run dev
```

The frontend will run at `http://localhost:5173/#/talk`.

---

## Deliverables Summary

| Phase | Milestone | Key Features | Status |
|---|---|---|---|
| **Phase 1** | Foundation & Product | React/TS interface, Web Audio API recording, FastAPI backend, format validation, privacy-first ephemeral storage. | Completed |
| **Phase 2** | Speech Recognition | `faster-whisper` CTranslate2 engine, English/Hindi/Hinglish transcription, language detection, editable transcripts, WER/CER evaluation. | Completed |
| **Phase 3** | Speech Emotion Recognition | RAVDESS dataset pipeline, 43 acoustic features (`librosa`), EDA visualizations, GroupKFold vs Stratified ML training, SVM serialized pipeline, abstention guardrails, real-time acoustic insights in UI. | Completed |
| **Phase 4** | Deep Learning SER + Multilingual Companion | Wav2Vec 2.0 & HuBERT representations, 67.01% Acc benchmark, language-matched response generation (Hindi/Hinglish/English/ES/FR), response validator, 4 styles, MongoDB persistence. | **Completed** |
| **Phase 5** | Research and Reliability | Multilingual transfer learning, cross-corpus calibration, noise robustness studies. | *Pending Approval* |

---

## Roadmap

- [x] **Phase 1 — Build the Basic Product**: Working frontend and backend foundation, recording controls, audio upload endpoint, health check, format validation, unit tests.
- [x] **Phase 2 — Speech Recognition**: Whisper integration (`faster-whisper`), multilingual STT (English, Hindi, Hinglish), audio preprocessing, editable transcripts, language detection, WER/CER evaluation pipeline.
- [x] **Phase 3 — First Machine Learning Model**: Baseline speech emotion classification on RAVDESS, acoustic feature extraction (MFCC, pitch/F0, RMS energy, pauses), SVM & Random Forest comparison, speaker-independent evaluation, FastAPI integration.
- [x] **Phase 4 — Advanced Deep Learning SER + Multilingual Intelligence**: Wav2Vec 2.0 and HuBERT self-supervised representations, speaker-independent benchmark (+18.72% gain), multilingual conversational intelligence (Hindi, Hinglish, English, Spanish, French, German, Japanese, Bengali, Tamil), response-language validator, 4 conversation styles, and MongoDB persistence.
- [ ] **Phase 5 — Research and Reliability**: Cross-corpus evaluation, noise robustness, calibrated uncertainty estimation.
- [ ] **Phase 6 — Complete VocalSense**: Production LLM conversational engine, multi-turn state machine, edge deployment.
