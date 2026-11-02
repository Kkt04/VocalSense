# VocalSense

**More than speech recognition. A new way to listen, understand and respond.**

VocalSense is a voice-first, multilingual, emotion-aware conversational AI companion. It is designed to look at two things: *what* a person says (transcript, language, context) and *how* they say it (measurable acoustic traits such as pace, pitch variation, pauses and energy).

> **Current status: Phase 1 of 8 — frontend design-system prototype.**
> The interface is complete and interactive, but **audio, transcripts, replies and insights are simulated**. No microphone is accessed, no backend exists yet, and no ML model is trained or running. Every mock element is labelled in the UI.

VocalSense is a conversational companion. It is **not** a therapist, a medical device or a diagnostic tool.

---

## What is in Phase 1

| Area | Details |
| --- | --- |
| Landing page | Hero, features, how-it-works, privacy section, footer. The microphone never starts automatically. |
| Conversation screen | Animated voice orb, simulated live waveform, recording timer, transcript, response panel, style and mode selectors, end-session control. |
| Conversation states | Idle, permission, listening, paused, processing, analysing, responding, error — with the exact copy from the product brief. |
| Recording controls | Start, pause, resume, stop, retry, mute. Permission is explained before any (simulated) access. |
| Transcript | Mock transcript, with inline editing before it is used. |
| Responses | Voice mode (simulated playback with pause, replay, mute, stop, progress) and text mode. Switchable at any time. |
| Conversation styles | Just listen, Help me think, Practical suggestions, Distract me. Each returns a different mock reply. |
| Speech insights | Optional panel with neutral wording and the "Unable to estimate reliably" state. Values are samples only. |
| Themes | **Light and dark, both pastel.** Follows the OS setting on first visit, remembers your choice afterwards. |
| Accessibility | Keyboard navigation, visible focus rings, ARIA roles and live regions, `prefers-reduced-motion` respected. |
| Responsive | Desktop, tablet and mobile layouts. |

## Quick start

Requirements: **Node.js 18+** and npm.

```bash
npm install
npm run dev        # http://localhost:5173
```

Other scripts:

```bash
npm run typecheck  # TypeScript check
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build locally
```

Routes (hash-based, no router dependency): `#/` landing page, `#/talk` conversation screen.

## Trying the prototype

1. Open the landing page and press **Start Talking**.
2. On the conversation screen press **Start talking**, then **Allow (demo)**. Nothing real is recorded.
3. Use pause, resume, mute, stop or retry. Stop produces the mock transcript, then the processing, analysing and responding states.
4. Edit the transcript, switch **Voice / Text**, change the **Conversation style**, and turn on **Speech insights**.
5. Use **Preview the error state** (visible when idle) to see the error and retry flow.
6. Toggle **Light / Dark** from the header on any page.

## Design system

Colours are CSS variables defined in `src/styles/index.css` and mapped to Tailwind in `tailwind.config.js`. Components use tokens (`bg-bg`, `bg-lav`, `text-ink`, `bg-accent`), so a theme switch needs no component changes.

| Token | Light | Dark |
| --- | --- | --- |
| `bg` | `#FAF8F4` warm ivory | `#1B1822` dusky plum |
| `lav` | `#E9E0F2` soft lavender | `#3D3450` |
| `pink` | `#F3E5E8` pale pink | `#4F3B47` |
| `sage` | `#DDE7DD` muted sage | `#344239` |
| `peach` | `#F4E1D4` soft peach | `#544238` |
| `deep` / `accent` | `#8E7BAA` / `#70608C` | `#9684B8` / `#CBBCE6` |
| `ink` | `#34303A` | `#ECE7F2` |

Typography: **DM Serif Display** for headings, **DM Sans** for interface text (loaded from Google Fonts in `index.html`).

The orb reacts only to audio level and activity. Its colour never encodes an emotion, by design.

## Project structure

```
src/
  App.tsx                    hash router
  main.tsx                   entry, ThemeProvider
  pages/
    Landing.tsx              landing page
    Conversation.tsx         voice conversation screen and state machine
  components/
    Orb.tsx  Waveform.tsx  Icon.tsx  Logo.tsx  ThemeToggle.tsx
  context/ThemeContext.tsx   light/dark theme, persisted
  hooks/useSimulatedAudio.ts MOCK amplitude source (replaced in Phase 2)
  styles/index.css           theme tokens and shared component classes
```

Where mock data lives: `MOCK_TRANSCRIPT`, `STYLES[].reply` and `INSIGHTS` at the top of `src/pages/Conversation.tsx`, plus `useSimulatedAudio.ts`.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Design system, landing page, conversation UI | **Done (this repo)** |
| 2 | Microphone, MediaRecorder, Whisper transcription, editable transcripts | Next |
| 3 | LLM conversation, styles, text-to-speech | Planned |
| 4 | Acoustic features (MFCC, pitch, energy, pauses), baseline classifiers | Planned |
| 5 | Wav2Vec2 and HuBERT, speaker-independent evaluation | Planned |
| 6 | Explainability, calibration, abstention, robustness | Planned |
| 7 | Hindi/English/Hinglish evaluation, personal voice baseline, memory | Planned |
| 8 | Tests, security checks, Docker, deployment | Planned |

Planned stack: React, TypeScript, Tailwind, Framer Motion, Web Audio API and MediaRecorder on the frontend; Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, PyTorch, librosa, torchaudio, Hugging Face Transformers, faster-whisper and scikit-learn on the backend and ML side.

## Principles

- **Consent first.** Recording starts only after an explicit action and permission.
- **Privacy by design.** No raw audio stored by default. Voice baselines and memory are opt-in and kept separate.
- **Scientific honesty.** No placeholder model will ever be presented as real emotion recognition. The system must be able to abstain.
- **User words first.** Transcripts and stated preferences outrank any model estimate.
- **No diagnosis.** Insights use neutral language ("higher-than-usual vocal energy was observed"), never "you were angry".
- **Safety.** Phase 3 will add a dedicated response path for immediate danger or self-harm. Vocal emotion classification will never be used as a risk detector.

## Known limitations (Phase 1)

- All audio, transcripts, replies, playback and insights are simulated.
- No unsupported-browser detection, network-failure handling or audio-quality checks yet (they need real audio, Phase 2).
- The response language selector, personal voice baseline, memory controls and self-reported check-in are not built yet (Phases 3 and 7).
- No automated tests yet. Only a TypeScript type-check and production build were run.
- Fonts load from Google Fonts, so offline use falls back to Georgia and system sans.
- Not yet checked with a screen reader or on physical devices.

## Environment variables

Phase 1 needs none. `.env.example` lists placeholders for later phases. Never put API keys in frontend code.
# VocalSense
