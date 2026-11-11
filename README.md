# VocalSense

> **A multilingual, voice-first, emotion-aware conversational AI companion.**
> Built to understand what you say, how you say it, and how to respond naturally.

---

## What Is VocalSense?

VocalSense is a full-stack AI application that turns your spoken voice into a meaningful, context-aware conversation — in your own language.

It does three things simultaneously:

1. **Listens and transcribes** — using `faster-whisper` (Whisper-based ASR), it converts your speech to text, in English, Hindi, Devanagari, or naturally code-switched Hinglish, without forced translation.
2. **Analyses your vocal patterns** — extracting measurable acoustic traits like pitch variation (F0), speaking pace, pause density, and energy envelope using both handcrafted features (43 MFCC-derived) and deep self-supervised representations (Wav2Vec2, HuBERT).
3. **Responds naturally** — generating contextual, language-matched companion responses via an OpenAI-compatible conversational LLM, in four different conversation styles.

VocalSense is a **supportive conversational companion**. It is not a therapist, a clinical diagnostic tool, or a mental health service. All model outputs are framed as uncertain acoustic measurements — never as psychological labels or medical guidance.

---

## Six-Phase Development Journey

VocalSense was built incrementally across six research and engineering phases:

| Phase | Focus | Highlights |
|-------|-------|------------|
| **Phase 1** | Frontend, recording, and audio upload | Real microphone capture, animated pastel React UI, audio format validation, FastAPI skeleton |
| **Phase 2** | Speech recognition (STT) | `faster-whisper` multilingual ASR, zero-translation guarantee, editable transcripts, WER/CER evaluation |
| **Phase 3** | Traditional ML Speech Emotion Recognition | RAVDESS dataset (1,440 recordings, 24 actors), 43 acoustic features, speaker-independent 5-fold `GroupKFold` evaluation, SVM pipeline serialised to `.joblib` |
| **Phase 4** | Deep learning SER + multilingual conversation | Wav2Vec 2.0 & HuBERT achieving **67.01% accuracy / 0.6666 Macro-F1** (+18.72% improvement), 9-language response matching, MongoDB persistence, 4 companion styles |
| **Phase 5** | Research: reliability, generalisation & explainability | Cross-dataset generalisation (RAVDESS → CREMA-D), 12-condition robustness suite, t-SNE manifold projections, temperature-scaled calibration (ECE: 19.4% → **1.29%**, T=2.549), selective abstention (τ=0.35) |
| **Phase 6** | Production hardening & deployment | Strict non-fabrication guardrail, deep component health checks, correlation ID observability, GDPR/DPDP data purge, Docker + cloud deployment |

---

## Architecture Overview

```
vocalsense/
├── backend/                               # FastAPI Python backend & ML pipelines
│   ├── app/
│   │   ├── main.py                        # FastAPI app — CORS, correlation ID middleware, router mounting
│   │   ├── config.py                      # Pydantic Settings — Whisper, ML model path, MongoDB URI, LLM key/base URL
│   │   ├── db/
│   │   │   └── mongodb.py                 # MongoDB connection manager (Motor async driver) with in-memory fallback
│   │   ├── api/
│   │   │   ├── routes_health.py           # GET /api/health — service status, component health checks
│   │   │   ├── routes_audio.py            # POST /api/audio/upload — audio validation with zero retention
│   │   │   └── v1/
│   │   │       ├── transcription.py       # POST /api/v1/transcription/transcribe — faster-whisper ASR
│   │   │       ├── emotion.py             # POST /api/v1/emotion/predict — SER with calibrated probabilities
│   │   │       └── conversation.py        # POST /respond, GET /list, GET /history, DELETE /{id}
│   │   ├── services/
│   │   │   ├── audio_validator.py         # Magic-bytes format detection, size limits
│   │   │   ├── audio_storage.py           # Ephemeral privacy-first audio handler (no raw retention)
│   │   │   ├── transcription/             # faster-whisper pipeline, PyAV 16kHz resampler, language detection
│   │   │   ├── emotion_recognition/       # Traditional SVM/RF & deep Wav2Vec2/HuBERT SER, temperature calibration
│   │   │   ├── language/                  # Language identification, output drift guardrails
│   │   │   ├── conversation/              # LLM client, response generator, language-to-prompt mapper
│   │   │   └── tts/                       # Modular multilingual TTS (gTTS)
│   │   └── schemas/                       # Pydantic request/response contracts
│   ├── ml/                                # Training scripts, feature extraction, EDA plots, Phase 5 experiments
│   ├── saved_models/                      # Serialised models — SVM.joblib, RF.joblib, temperature_calibration.json
│   ├── tests/                             # 128 automated tests covering API, ML, MongoDB, pipeline integration
│   ├── Dockerfile                         # Production backend container (Python 3.12, uvicorn)
│   └── requirements.txt
├── research/phase5/                       # Phase 5 empirical findings — generalisation, calibration, explainability papers
├── experiments/phase5/                    # Benchmarking scripts — robustness suite, t-SNE, calibration curves
├── src/                                   # React 18 + TypeScript frontend (Vite)
│   ├── components/                        # Orb (audio-reactive), Waveform (live bars), Icon, Logo, ThemeToggle
│   ├── hooks/                             # useAudioRecorder (MediaRecorder), useSimulatedAudio (demo mode)
│   ├── pages/                             # Landing page, Conversation view (drawer, error cards, history)
│   └── services/                          # Typed API client — health, STT, SER, conversation, MongoDB
├── docker-compose.yml                     # 1-command local orchestration (MongoDB + Backend + Frontend + Nginx)
├── Dockerfile.frontend                    # Multi-stage Nginx production build
├── DEPLOYMENT.md                          # Full deployment manual — Vercel, Render, Railway, MongoDB Atlas, Docker
└── .env.example                           # Commented environment variable template
```

---

## Key Features

### Multilingual Speech Recognition with Hinglish Preservation

VocalSense transcribes English, Hindi (Devanagari script), and naturally code-switched Hinglish without forcing translation. For example, `"Yaar, mujhe thoda stress lag raha hai"` stays exactly as spoken — not converted to English. Users can also edit their transcript inline before it enters the conversation, so only the words they intend to share are stored.

### Calibrated Speech Emotion Recognition (SER)

The SER pipeline is built in two layers:

- **Traditional layer** — 43 handcrafted acoustic features (F0 mean/std, RMS energy, spectral centroid, pause fraction) fed into a serialised SVM pipeline trained on RAVDESS with speaker-independent cross-validation.
- **Deep layer** — `facebook/wav2vec2-base` and `facebook/hubert-base-ls960` embeddings through a lightweight classifier head, achieving 67.01% accuracy and 0.6666 Macro-F1 on held-out RAVDESS data.

Both layers go through **temperature scaling** (T=2.549) which reduces Expected Calibration Error from 19.4% to just 1.29%. When prediction confidence falls below the abstention threshold (τ=0.35), VocalSense explicitly says it cannot reliably read the voice — rather than fabricating a confident wrong answer.

### Strict Non-Fabrication Rule

VocalSense never returns pre-written, hardcoded conversational templates. Every response is generated dynamically by an OpenAI-compatible LLM configured via `.env` (Groq, Gemini, OpenAI, or any compatible endpoint). If the LLM is unavailable or the API key is missing, the backend returns `HTTP 503` and the UI shows a clear error card with a retry button — not a fake placeholder reply.

### Four Conversation Styles

Each session, the user picks how they want the AI to respond:

| Style | What it means |
|-------|--------------|
| **Just listen** | Acknowledge and reflect. No advice, no suggestions. |
| **Help me think** | Gentle questions and perspective, not answers. |
| **Practical suggestions** | Concrete, actionable ideas when that's what's needed. |
| **Distract me** | Shift the topic. Lighter energy when that's the right call. |

### Full Conversation Memory with GDPR Controls

Every exchange is stored to MongoDB (with an in-memory fallback when the database is unavailable). The History Drawer shows past conversations with meaningful titles derived from the actual conversation content. Users can delete individual conversations or permanently purge all their data in a single step — no account needed, no waiting period.

---

## Privacy Commitments

- **Recording only starts when you press Start and allow access.** No passive listening. No ambient capture.
- **Raw audio is never retained.** Audio bytes are discarded immediately after transcription and acoustic analysis. Only the text transcript and computed measurements are stored.
- **Conversation memory and voice baselines are opt-in and separate.** You can use voice insights without saving conversations, and vice versa.
- **Deleting your data is always one step away.** A single GDPR/DPDP-compliant purge endpoint (`DELETE /api/v1/conversation/user/{id}/data`) erases all messages, conversations, and acoustic insights.
- **VocalSense is a conversational companion.** It is not a therapist and does not diagnose. All emotion-related outputs are framed as physical voice measurements, not psychological assessments.

---

## Quickstart

### Option A — One-Command Docker Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/vocalsense.git
cd vocalsense

# 2. Copy the environment template and add your LLM key
cp .env.example .env
# Open .env and set:
#   LLM_API_KEY=your_key_here
#   LLM_API_BASE_URL=https://api.groq.com/openai/v1   # or Gemini / OpenAI
#   LLM_MODEL=qwen/qwen3.8-27b                         # or any compatible model

# 3. Start everything
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health |
| Component health | http://localhost:8000/api/health/components |

### Option B — Manual Local Setup

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# From project root:
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
# From project root:
npm install
npm run dev
# → http://localhost:5173
```

**Environment variables** (`.env` in project root):
```bash
# LLM — required for conversational responses
LLM_API_KEY=gsk_...
LLM_API_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=qwen/qwen3.8-27b

# MongoDB — optional, falls back to in-memory if not set
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=vocalsense

# Whisper — defaults shown
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service status, app version, environment |
| `GET` | `/api/health/components` | Deep component health — DB, Whisper, SER model, LLM, TTS |
| `POST` | `/api/audio/upload` | Validate and accept audio (zero retention) |
| `POST` | `/api/v1/transcription/transcribe` | faster-whisper ASR — preserves Hinglish / Hindi |
| `GET` | `/api/v1/transcription/health` | ASR model readiness, device, loaded language |
| `POST` | `/api/v1/emotion/predict` | Acoustic traits + calibrated SER prediction with abstention |
| `GET` | `/api/v1/emotion/health` | SER model readiness, class taxonomy, calibration status |
| `POST` | `/api/v1/conversation/respond` | Generate language-matched LLM companion response |
| `GET` | `/api/v1/conversation/list` | Paginated conversation list with meaningful titles |
| `GET` | `/api/v1/conversation/history/{id}` | Chronological messages for a conversation |
| `DELETE` | `/api/v1/conversation/{id}` | Delete a single conversation and its messages |
| `DELETE` | `/api/v1/conversation/user/{id}/data` | GDPR / DPDP — permanent user data purge |
| `POST` | `/api/v1/conversation/user/preference` | Update user language preference |

---

## Running Tests

```bash
# All 128 backend tests
PYTHONPATH=. ./venv/bin/pytest backend/tests -v

# Frontend production build verification
npm run build
```

The test suite covers audio upload validation, transcription pipeline, traditional and deep SER, MongoDB operations, multilingual conversation, Phase 5 reliability research, Phase 6 production integration, and pipeline correctness.

---

## Phase 5 Research Findings (Summary)

The Phase 5 reliability investigation found that the deep SER model (HuBERT) generalises meaningfully beyond its training speakers, but shows measurable performance degradation under real-world conditions:

- **Speaker generalisation** — HuBERT maintains 60–65% accuracy on unseen speakers vs. 67% on training speakers. SVM degrades more sharply (−12%) under speaker shift.
- **Cross-dataset generalisation** — Accuracy drops 15–20% when moving from RAVDESS to CREMA-D, primarily due to class imbalance and recording condition differences.
- **Robustness** — Performance drops sharply below SNR 10 dB. Reverberation is more damaging than additive noise. Compression artefacts (MP3 128kbps) have minimal impact.
- **Calibration** — Temperature scaling (T=2.549) reduces ECE from 19.4% to 1.29%. The model was significantly overconfident before calibration.
- **Abstention** — At τ=0.35, the model correctly withholds predictions on ~22% of ambiguous inputs rather than producing unreliable confident labels.

Full methodology, confusion matrices, t-SNE plots, calibration curves, and robustness degradation charts are in `research/phase5/` and `experiments/phase5/`.

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions for deploying to:
- **Vercel** — React frontend
- **Render / Railway / Docker** — FastAPI backend
- **MongoDB Atlas** — managed cloud database
