# VocalSense

**More than speech recognition. A new way to listen, understand and respond.**

VocalSense is a multilingual, voice-first, emotion-aware AI conversational companion. It is designed to understand both:
1. **What a person says:** Speech recognition, transcription, language identification, and conversational context.
2. **How a person speaks:** Acoustic characteristics such as pitch variation, speaking rate, pauses, and vocal energy.

> **Phase 1 Complete: Frontend & Backend Foundation.**
> Includes real microphone recording via Web Audio API & MediaRecorder, audio upload endpoint with header validation, CORS, privacy zero-retention handling, animated voice orb, and comprehensive unit tests.

*Product Principle:* VocalSense is a calm, supportive conversational companion. It is **not** a therapist, a medical device, or a diagnostic tool. Model outputs are uncertain predictions, never facts or diagnoses.

---

## Architecture Overview (Phase 1)

```
vocalsense/
├── backend/                        # FastAPI Python backend
│   ├── app/
│   │   ├── main.py                 # FastAPI application, CORS, error handling
│   │   ├── config.py               # Pydantic Settings (sizes, MIME types, CORS)
│   │   ├── api/
│   │   │   ├── routes_health.py    # GET /api/health
│   │   │   └── routes_audio.py     # POST /api/audio/upload
│   │   ├── services/
│   │   │   ├── audio_validator.py  # Format, size, and magic bytes detection
│   │   │   └── audio_storage.py    # Privacy-first ephemeral storage handler
│   │   └── schemas/
│   │       └── audio.py            # Pydantic response models
│   ├── tests/
│   │   ├── test_health.py          # Health check endpoint unit tests
│   │   └── test_audio_upload.py    # Audio upload validation & size tests
│   └── requirements.txt
├── src/                            # React + TypeScript frontend
│   ├── components/
│   │   ├── Orb.tsx                 # Animated responsive voice orb (Framer Motion)
│   │   ├── Waveform.tsx            # Live frequency waveform
│   │   ├── AudioUploader.tsx       # Drag-and-drop / file selector audio uploader
│   │   └── ...
│   ├── hooks/
│   │   ├── useAudioRecorder.ts     # Real Web Audio API AnalyserNode + MediaRecorder
│   │   └── useSimulatedAudio.ts    # Fallback simulation for no-mic preview
│   ├── services/
│   │   └── api.ts                  # Backend API client
│   └── pages/
│       ├── Landing.tsx             # Landing hero and overview
│       └── Conversation.tsx        # Conversation stage with live mic & upload
└── .env.example
```

---

## Quick Start Guide

### 1. Backend Setup (FastAPI)

Requirements: **Python 3.10+**

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run backend unit tests
PYTHONPATH=. pytest backend/tests -v

# 4. Start the backend development server
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at:
- Health check: `http://localhost:8000/api/health`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Audio Upload Endpoint: `POST http://localhost:8000/api/audio/upload`

---

### 2. Frontend Setup (React + TypeScript)

Requirements: **Node.js 18+** and npm

```bash
# 1. Install dependencies
npm install

# 2. Start Vite development server
npm run dev
```

The frontend will be running at `http://localhost:5173`.

Other commands:
```bash
npm run typecheck    # Validate TypeScript types
npm run build        # Production build
npm run preview      # Preview production build locally
```

---

## Deliverables in Phase 1

| Area | Implementation Details |
| --- | --- |
| **Microphone Recording** | Real browser recording with `navigator.mediaDevices.getUserMedia` and `MediaRecorder`. Supports Chrome, Firefox, and Safari audio containers. |
| **Live Audio Waveform** | Web Audio API `AudioContext` and `AnalyserNode` computing real-time frequency distribution driving the waveform and voice orb. |
| **Audio File Upload** | Direct file upload via `AudioUploader.tsx` (drag-and-drop / file picker) supporting WAV, WebM, OGG, MP3, and M4A up to 25MB. |
| **FastAPI Backend** | Modular structure: `app/main.py`, `app/config.py`, `app/api/`, `app/services/`, `app/schemas/`. |
| **Audio Validation** | `audio_validator.py` checks file size, extension, MIME type, and magic header bytes (`RIFF`, `EBML`, `OggS`, `ID3`, `ftyp`). |
| **Privacy Safeguards** | Zero permanent raw audio retention by default. Ephemeral processing with immediate purge. |
| **Error Handling** | Clear feedback for microphone permission denial, device errors, unsupported formats, and network disconnections. |
| **Backend Unit Tests** | 7 automated unit tests covering health status, root metadata, valid WAV/WebM uploads, empty files, unsupported extensions, and file size limits. |

---

## Roadmap

- [x] **Phase 1 — Build the Basic Product**: Working frontend and backend foundation, recording controls, audio upload endpoint, health check, format validation, unit tests.
- [ ] **Phase 2 — Speech Recognition**: Whisper / faster-whisper integration, multilingual STT (English, Hindi, Hinglish), editable transcripts, language identification.
- [ ] **Phase 3 — First Machine Learning Model**: Baseline speech emotion classification on RAVDESS, acoustic feature extraction (MFCC, pitch/F0, RMS energy, pauses), SVM & Random Forest comparison.
- [ ] **Phase 4 — Advanced Speech Models**: Deep learning representations with Wav2Vec2, HuBERT, and comparative evaluation.
- [ ] **Phase 5 — Research and Reliability**: Speaker-independent evaluation, noise robustness, explainability (SHAP / feature importance), uncertainty estimation & abstention.
- [ ] **Phase 6 — Complete VocalSense**: LLM conversational engine, conversation styles, text-to-speech, voice insights panel, and deployment.
