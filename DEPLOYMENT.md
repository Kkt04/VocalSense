# VocalSense — Deployment & Production Guide

This guide covers deployment procedures for **VocalSense**, an AI-powered, multilingual, voice-first companion that listens, transcribes, understands acoustic traits, and responds with emotional intelligence.

---

## 1. System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │          VocalSense Frontend            │
                  │   (React 18 + Vite + Tailwind CSS)      │
                  │  Voice Orb · Live Waveform · Pastels   │
                  └────────────────────┬────────────────────┘
                                       │ HTTP / REST
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │           FastAPI Backend               │
                  │     (Python 3.12 · Uvicorn · ASGI)      │
                  │   Correlation IDs · Safe Temp Cleanup   │
                  └───────┬────────────┬───────────┬────────┘
                          │            │           │
           ┌──────────────┴───┐ ┌──────┴─────┐ ┌───┴────────────────┐
           │ faster-whisper   │ │ Acoustic / │ │ OpenAI-Compatible   │
           │ Multilingual ASR │ │ Wav2Vec2   │ │ LLM Engine          │
           │ (Hinglish/Hi/En) │ │ Emotion    │ │ (Gemini/Groq/OAI)   │
           └──────────────────┘ └────────────┘ └────────────────────┘
                          │
                          ▼
                  ┌─────────────────────────────────────────┐
                  │        MongoDB Primary Storage          │
                  │   (Conversations · Messages · GDPR)     │
                  └─────────────────────────────────────────┘
```

---

## 2. Environment Variables Reference

Create a `.env` file in the project root based on `.env.example`:

| Variable | Description | Default | Example |
|---|---|---|---|
| `ENVIRONMENT` | Application mode | `development` | `production` |
| `PORT` | Backend port | `8000` | `8000` |
| `HOST` | Backend bind host | `0.0.0.0` | `0.0.0.0` |
| `CORS_ORIGINS` | Permitted origins (JSON array) | `["http://localhost:5173"]` | `["https://vocalsense.vercel.app"]` |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` | `mongodb+srv://user:pass@cluster0.net/` |
| `MONGODB_DB_NAME` | Primary database name | `vocalsense` | `vocalsense` |
| `WHISPER_MODEL_SIZE` | faster-whisper model | `base` | `base` / `small` |
| `WHISPER_DEVICE` | ASR inference device | `auto` | `cpu` / `cuda` |
| `EMOTION_MODEL_TYPE` | SER pipeline | `traditional_svm` | `wav2vec2` / `traditional_svm` |
| `LLM_API_KEY` | Provider API key | *(empty)* | `AIzaSy...` |
| `LLM_API_BASE_URL` | OpenAI-compatible endpoint | `.../openai` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `LLM_MODEL` | Conversational model ID | `gemini-2.0-flash` | `gemini-2.0-flash` / `llama-3.3-70b-versatile` |
| `ENABLE_TTS` | Enable modular speech synthesis | `true` | `true` |
| `VITE_API_BASE_URL` | Frontend API target | `http://localhost:8000` | `https://api.vocalsense.app` |

---

## 3. Local One-Command Deployment (Docker Compose)

The fastest way to spin up MongoDB, the FastAPI backend, and the React frontend together:

```bash
# 1. Clone repository & configure environment
cp .env.example .env

# 2. Add your LLM API Key to .env
# LLM_API_KEY=your_actual_gemini_or_groq_api_key

# 3. Start all services
docker compose up --build
```

- **Frontend**: http://localhost:5173
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **Component Subsystem Health**: http://localhost:8000/api/health/components

To stop services:
```bash
docker compose down
```

---

## 4. Cloud Production Deployment

### A. Database: MongoDB Atlas (Managed Cloud)
1. Log in to [MongoDB Atlas](https://cloud.mongodb.com/).
2. Create a free shared cluster (M0 or higher).
3. Under **Database Access**, create a user with `readWriteAnyDatabase` privileges.
4. Under **Network Access**, add `0.0.0.0/0` (or your backend hosting static IP).
5. Copy your connection string:
   ```env
   MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority
   MONGODB_DB_NAME=vocalsense
   ```

### B. Backend: Render / AWS ECS / Fly.io / GCP Cloud Run
Deploying with the provided `backend/Dockerfile`:

1. **Docker build context**: set context to the repository root and dockerfile to `backend/Dockerfile`.
2. **Environment Variables**:
   ```env
   ENVIRONMENT=production
   PORT=8000
   HOST=0.0.0.0
   CORS_ORIGINS=["https://vocalsense.vercel.app"]
   MONGODB_URI=mongodb+srv://...
   LLM_API_KEY=<your-gemini-or-groq-key>
   LLM_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   LLM_MODEL=gemini-2.0-flash
   WHISPER_MODEL_SIZE=base
   WHISPER_DEVICE=cpu
   WHISPER_COMPUTE_TYPE=int8
   ```
3. **Health Check Path**: `/api/health`

### C. Frontend: Vercel
1. Import your GitHub repository to Vercel.
2. Select **Vite** framework preset.
3. Set the Environment Variable:
   - `VITE_API_BASE_URL`: `https://your-backend-api-url.com`
4. Build command: `npm run build`
5. Output directory: `dist`
6. Click **Deploy**.

---

## 5. Security & Privacy Guarantees

1. **Zero Raw Audio Retention**: Uploaded speech is processed in `/tmp/vocalsense/audio` and cleaned up immediately after inference unless explicitly persisted.
2. **Strict Non-Fabrication Rule**: VocalSense never returns fake conversational text disguised as an AI response. If the LLM is unconfigured or unavailable, HTTP 503 is returned with an explicit error card and retry button.
3. **Non-Diagnostic Acoustic Guidance**: Acoustic pattern insights strictly report measurable physical voice traits (pitch variation, vocal energy, speech rate, pause fraction) with honest uncertainty and abstain when confidence is low.
4. **GDPR / DPDP Compliance**: Complete right-to-erasure endpoint (`DELETE /api/v1/conversation/user/{user_id}/data`) permanently purges all profile records, conversation turns, messages, and acoustic insights in a single transaction.
5. **Observability**: Every request is tagged with an `X-Request-ID` correlation ID and `X-Response-Time-Ms` duration header for end-to-end tracing.
