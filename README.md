# VocalSense

**More than speech recognition. A new way to listen, understand and respond.**

VocalSense is a multilingual, voice-first, emotion-aware AI conversational companion. It is designed to understand both:
1. **What a person says:** Speech recognition, transcription, language identification, and conversational context.
2. **How a person speaks:** Acoustic characteristics such as pitch variation, speaking rate, pauses, and vocal energy.

> **Phase 1 & Phase 2 Complete: Frontend, Backend & Speech Recognition.**
> Integrates real microphone recording, audio format validation, faster-whisper speech recognition for English, Hindi, and Hinglish, editable transcripts, language detection, and a WER/CER evaluation pipeline.

*Product Principle:* VocalSense is a calm, supportive conversational companion. It is **not** a therapist, a medical device, or a diagnostic tool. Model outputs are uncertain predictions, never facts or diagnoses.

---

## Architecture Overview (Phase 1 & 2)

```
vocalsense/
├── backend/                               # FastAPI Python backend
│   ├── app/
│   │   ├── main.py                        # FastAPI application, CORS, routers
│   │   ├── config.py                      # Pydantic Settings (sizes, Whisper config)
│   │   ├── api/
│   │   │   ├── routes_health.py           # GET /api/health
│   │   │   ├── routes_audio.py            # POST /api/audio/upload
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── transcription.py       # POST /api/v1/transcription/transcribe
│   │   ├── services/
│   │   │   ├── audio_validator.py         # Format, size, and magic bytes detection
│   │   │   ├── audio_storage.py           # Privacy-first ephemeral storage handler
│   │   │   └── transcription/
│   │   │       ├── __init__.py
│   │   │       ├── whisper_service.py     # faster-whisper model manager & inference
│   │   │       ├── audio_preprocessing.py # PyAV 16kHz mono resampling & decoding
│   │   │       └── language_detection.py  # Code-switching & Hinglish classification
│   │   └── schemas/
│   │       ├── audio.py                   # Pydantic upload/health schemas
│   │       └── transcription.py           # Pydantic transcription response schemas
│   ├── evaluation/
│   │   └── wer_cer.py                     # Word Error Rate & Character Error Rate evaluator
│   ├── tests/
│   │   ├── test_health.py                 # Health check unit tests
│   │   ├── test_audio_upload.py           # Audio upload validation unit tests
│   │   └── test_transcription.py          # Whisper transcription & preprocessing tests
│   └── requirements.txt
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
│   │   └── api.ts                         # Backend API client (health, upload, transcribe)
│   └── pages/
│       ├── Landing.tsx                    # Landing hero and overview
│       └── Conversation.tsx               # Voice stage with live transcription & editing
└── .env.example
```

---

## Quick Start Guide

### 1. Backend Setup (FastAPI + Whisper)

Requirements: **Python 3.10+**

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run all backend tests (16 unit tests)
PYTHONPATH=. pytest backend/tests -v

# 4. Run the WER / CER evaluation demonstration
PYTHONPATH=. python3 backend/evaluation/wer_cer.py

# 5. Start the backend development server
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at:
- Health check: `http://localhost:8000/api/health`
- Transcription health: `http://localhost:8000/api/v1/transcription/health`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Transcription Endpoint: `POST http://localhost:8000/api/v1/transcription/transcribe`

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

## Deliverables in Phase 2

| Feature | Details |
|---|---|
| **Speech Recognition Engine** | `faster-whisper` (CTranslate2) inference engine with configurable model size (`base` default, int8 CPU quantization) and auto-detection of GPU/CPU. |
| **Multilingual Support** | Automatic language detection and zero-shot preservation for English (`en`), Hindi (`hi`), and Hinglish (code-switching). |
| **No-Translation Guarantee** | Inference enforces `task="transcribe"`, guaranteeing that Hindi/Hinglish speech is never forced into English translation. |
| **Audio Preprocessing** | `AudioPreprocessor` converts incoming WAV, WebM, MP3, M4A, OGG into 16,000 Hz mono `float32` arrays via PyAV, skipping external FFmpeg binary dependencies. |
| **Hinglish Detection** | Heuristic analyzer checks Romanized Hindi markers (`yaar`, `mujhe`, `karna`, `hai`), mixed Latin/Devanagari scripts, and token distributions. |
| **Editable Transcripts** | Frontend conversation interface displays live transcripts with an inline textarea editor. User corrections persist to the conversation context. |
| **Speech Evaluation** | `backend/evaluation/wer_cer.py` implements Word Error Rate (WER) and Character Error Rate (CER) via `jiwer` with Unicode NFC normalization. |
| **Comprehensive Tests** | 16 backend unit tests covering health endpoints, container uploads, audio resampling, silence handling, language detection, and end-to-end Whisper inference. |

---

## Roadmap

- [x] **Phase 1 — Build the Basic Product**: Working frontend and backend foundation, recording controls, audio upload endpoint, health check, format validation, unit tests.
- [x] **Phase 2 — Speech Recognition**: Whisper integration (`faster-whisper`), multilingual STT (English, Hindi, Hinglish), audio preprocessing, editable transcripts, language detection, WER/CER evaluation pipeline.
- [ ] **Phase 3 — First Machine Learning Model**: Baseline speech emotion classification on RAVDESS, acoustic feature extraction (MFCC, pitch/F0, RMS energy, pauses), SVM & Random Forest comparison.
- [ ] **Phase 4 — Advanced Speech Models**: Deep learning representations with Wav2Vec2, HuBERT, and comparative evaluation.
- [ ] **Phase 5 — Research and Reliability**: Speaker-independent evaluation, noise robustness, explainability (SHAP / feature importance), uncertainty estimation & abstention.
- [ ] **Phase 6 — Complete VocalSense**: LLM conversational engine, conversation styles, text-to-speech, voice insights panel, and deployment.
