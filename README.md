# VocalSense

**More than speech recognition. A new way to listen, understand and respond.**

VocalSense is a multilingual, voice-first, emotion-aware AI conversational companion. It is designed to understand:
1. **What a person says:** Speech recognition, transcription, language identification, and conversational context using `faster-whisper`.
2. **How a person speaks:** Acoustic traits such as pitch variation (F0), speaking rate, pauses, and deep self-supervised vocal representations (`Wav2Vec2` & `HuBERT`).
3. **How to respond naturally:** Contextual, language-matched companion responses in the user's spoken language across four supportive conversation styles powered by conversational intelligence.

> **Phase 1 to Phase 6 Complete: Production-Ready Full-Stack Conversational Companion.**
> - **Phase 1**: Voice stage, real microphone recording, audio format validation, and animated pastel React UI.
> - **Phase 2**: `faster-whisper` multilingual transcription for English, Hindi, and Hinglish with zero-translation guarantee, editable transcripts, and WER/CER evaluation.
> - **Phase 3**: Traditional ML Speech Emotion Recognition pipeline on RAVDESS (1,440 recordings, 24 actors), 43 acoustic features (`librosa`), speaker-independent 5-fold evaluation (`GroupKFold`), serialized SVM pipeline, and FastAPI inference endpoints with abstention thresholds.
> - **Phase 4**: Deep learning speech representations with **Wav2Vec 2.0** and **HuBERT**, achieving **67.01% Accuracy / 0.6666 Macro-F1** (+18.72% Macro-F1 improvement over traditional baseline), multilingual language matching (Hindi in Devanagari, Hinglish in Latin script, English, Spanish, French, German, Japanese, Bengali, Tamil), response validation layer, 4 companion styles, and **MongoDB** primary persistence.
> - **Phase 5**: Model reliability, cross-dataset generalization (CREMA-D / RAVDESS harmonized 6-class taxonomy), multilingual SER representation drift analysis, 12 acoustic/noise perturbation robustness suite, model explainability (feature importance, group ablation, 2D t-SNE manifold projections), and temperature-scaled probability calibration ($T=2.549$, reducing ECE from 19.4% to 1.29%) with selective abstention ($\tau=0.35$).
> - **Phase 6**: Production hardening, strict non-fabrication guardrail (zero hardcoded conversational responses masquerading as AI), deep component-level health checks (`/api/health/components`), correlation ID observability (`X-Request-ID`), scrollable conversation thread, conversation history drawer, GDPR / DPDP right-to-erasure data purge, containerization (`backend/Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`), and cloud deployment guides (`DEPLOYMENT.md`).

*Core Principle:* VocalSense is a calm, supportive conversational companion. It is **not** a therapist, a medical device, or a diagnostic tool. Model outputs are treated strictly as uncertain acoustic signals, never clinical diagnoses.

---

## Architecture Overview

```
vocalsense/
├── backend/                               # FastAPI Python backend & ML pipelines
│   ├── app/
│   │   ├── main.py                        # FastAPI app, correlation IDs, timing middleware, routers
│   │   ├── config.py                      # Pydantic Settings (Whisper, ML model path, MongoDB, LLM)
│   │   ├── db/
│   │   │   └── mongodb.py                 # MongoDB connection manager (Motor / PyMongo / In-memory fallback)
│   │   ├── api/
│   │   │   ├── routes_health.py           # GET /api/health, GET /api/health/components
│   │   │   ├── routes_audio.py            # POST /api/audio/upload
│   │   │   └── v1/
│   │   │       ├── transcription.py       # POST /api/v1/transcription/transcribe
│   │   │       ├── emotion.py             # POST /api/v1/emotion/predict, GET /api/v1/emotion/health
│   │   │       └── conversation.py        # POST /respond, GET /list, GET /history, DELETE /{id}, DELETE /user/{id}/data
│   │   ├── services/
│   │   │   ├── audio_validator.py         # Format, size, and magic bytes detection
│   │   │   ├── audio_storage.py           # Privacy-first ephemeral storage handler (zero raw retention)
│   │   │   ├── transcription/             # faster-whisper ASR pipeline & PyAV 16kHz resampler
│   │   │   ├── emotion_recognition/       # Traditional (SVM/RF) & Deep (Wav2Vec2/HuBERT) SER
│   │   │   ├── language/                  # Language identification & output drift guardrails
│   │   │   ├── conversation/              # Conversational response generator & LLM client
│   │   │   └── tts/                       # Modular multilingual Text-to-Speech (gTTS)
│   │   └── schemas/                       # Pydantic data contracts
│   ├── ml/                                # Model training, feature extraction & scientific reports
│   ├── saved_models/                      # Serialized models (.joblib, temperature calibration JSON)
│   ├── tests/                             # Comprehensive automated test suite (127 tests)
│   ├── Dockerfile                         # Production backend container definition
│   └── requirements.txt
├── research/phase5/                       # Phase 5 empirical research papers & findings
├── experiments/phase5/                    # Phase 5 benchmarking & calibration scripts
├── src/                                   # React + TypeScript frontend
│   ├── components/                        # Orb, Waveform, AudioUploader, ThemeToggle, Logo, Icon
│   ├── hooks/                             # useAudioRecorder, useSimulatedAudio
│   ├── pages/                             # Landing page, Conversation view (History drawer, Error cards)
│   └── services/                          # API client (Health, STT, SER, Conversation, MongoDB)
├── docker-compose.yml                     # 1-command local orchestration (MongoDB + Backend + Frontend)
├── Dockerfile.frontend                    # Multi-stage production Nginx frontend container
├── DEPLOYMENT.md                          # Full deployment manual (Vercel, Render, Atlas, Docker)
└── .env.example                           # Standard environment configuration template
```

---

## Key Features

### 1. Multilingual Speech Recognition (Hinglish Preservation)
- Transcribes English, Hindi (Devanagari), and code-switched Hinglish (*e.g., "Yaar, mujhe thoda stress lag raha hai"*).
- Uses `task="transcribe"` with zero-translation guarantee to prevent Hinglish from being translated into English.
- User transcript editing interface allows correcting words before they enter conversational memory.

### 2. Speech Emotion Recognition (SER) with Honest Uncertainty
- Extract 43 acoustic features (F0 mean/std, RMS energy, spectral centroid, pause fraction) alongside deep self-supervised embeddings (`facebook/wav2vec2-base` & `facebook/hubert-base-ls960`).
- Probability calibration via Temperature Scaling ($T=2.549$) reducing Expected Calibration Error (ECE) from 19.4% to 1.29%.
- Selective abstention ($\tau=0.35$): VocalSense transparently abstains when prediction confidence is low.
- Non-diagnostic acoustic signals: always framed as physical voice measurements, never psychological labels.

### 3. Strict Non-Fabrication Rule (No Hardcoded Responses)
- VocalSense never pretends to be an AI by returning pre-written conversational template replies.
- Responses are generated dynamically through an OpenAI-compatible conversational LLM endpoint (Gemini, Groq, OpenAI, Ollama).
- If the conversational AI service is unconfigured or unavailable, the backend returns HTTP 503 Service Unavailable and the UI displays an informative error card with a **"Retry Response"** button.

### 4. Full Conversation Lifecycle & MongoDB Persistence
- Chronological message history with role, transcript, detected language, and optional vocal pattern insights.
- History Drawer allows browsing past conversations, switching active chats, and deleting individual sessions.
- Full GDPR / DPDP Right-to-Erasure (`DELETE /api/v1/conversation/user/{user_id}/data`) to permanently purge all user records in a single operation.

---

## Quickstart

### Option A: Local One-Command Docker Setup
```bash
# 1. Clone repository & configure environment
cp .env.example .env

# 2. Add your LLM API Key to .env (Google Gemini, Groq, or OpenAI)
# LLM_API_KEY=your_key_here

# 3. Start MongoDB, Backend, and Frontend containers
docker compose up --build
```
- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Component Health: [http://localhost:8000/api/health/components](http://localhost:8000/api/health/components)

### Option B: Local Manual Setup

#### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start backend server
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
# In project root
npm install
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, app version, environment |
| `GET` | `/api/health/components` | Deep health check (DB, Whisper, Emotion model, LLM, TTS) |
| `POST` | `/api/audio/upload` | Upload & validate audio format with zero retention |
| `POST` | `/api/v1/transcription/transcribe` | faster-whisper ASR (preserves Hinglish/Hindi) |
| `GET` | `/api/v1/transcription/health` | ASR model readiness and device info |
| `POST` | `/api/v1/emotion/predict` | Vocal acoustic traits & calibrated emotion pattern |
| `GET` | `/api/v1/emotion/health` | SER model readiness and class taxonomy |
| `POST` | `/api/v1/conversation/respond` | Generate language-matched companion response |
| `GET` | `/api/v1/conversation/list` | Paginated list of user conversations |
| `GET` | `/api/v1/conversation/history/{id}` | Chronological message history for a conversation |
| `DELETE` | `/api/v1/conversation/{id}` | Delete conversation and all messages |
| `DELETE` | `/api/v1/conversation/user/{id}/data` | GDPR / DPDP user data purge |
| `POST` | `/api/v1/conversation/user/preference` | Update language preference in MongoDB |

---

## Running the Automated Test Suite

Run all 127 automated tests:
```bash
PYTHONPATH=. ./venv/bin/pytest backend/tests
```

Run frontend production build verification:
```bash
npm run build
```

---

## Deployment Manual

See [DEPLOYMENT.md](file:///Users/kalashkumarithakur/Desktop/vocalsense/DEPLOYMENT.md) for full deployment instructions for **Vercel** (frontend), **Render / Docker** (backend), and **MongoDB Atlas** (database).
