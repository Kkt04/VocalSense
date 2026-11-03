import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Icon from '../components/Icon';
import Logo from '../components/Logo';
import Orb, { OrbMode } from '../components/Orb';
import ThemeToggle from '../components/ThemeToggle';
import Waveform from '../components/Waveform';
import AudioUploader from '../components/AudioUploader';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { useSimulatedAudio } from '../hooks/useSimulatedAudio';
import {
  checkBackendHealth,
  checkTranscriptionHealth,
  checkEmotionHealth,
  transcribeAudio,
  predictEmotion,
  TranscriptionResponse,
  EmotionPredictionResponse,
} from '../services/api';

type Phase = 'idle' | 'permission' | 'listening' | 'paused' | 'processing' | 'analysing' | 'responding' | 'error';
type StyleId = 'listen' | 'think' | 'practical' | 'distract';
type Msg = {
  id: number;
  role: 'user' | 'ai';
  text: string;
  isEdited?: boolean;
  detectedLanguage?: string;
  languageProb?: number;
};

/* ---------- CONVERSATION STYLES ---------- */
const STYLES: { id: StyleId; name: string; hint: string; reply: string }[] = [
  {
    id: 'listen',
    name: 'Just listen',
    hint: 'Gentle acknowledgement and follow-up questions.',
    reply: "That sounds like a lot to carry, especially when you've already put in the work. What does that thought usually sound like when it shows up?",
  },
  {
    id: 'think',
    name: 'Help me think',
    hint: 'Organise thoughts and explore perspectives.',
    reply: "Let's separate two things: what you've prepared, and what you can't control. Which specific part feels most uncertain right now?",
  },
  {
    id: 'practical',
    name: 'Practical suggestions',
    hint: 'Clear, actionable steps.',
    reply: "Here are three small steps. 1) Do one timed practice round today. 2) Outline a one-minute story for your strongest experience. 3) Plan a calming routine for the evening before.",
  },
  {
    id: 'distract',
    name: 'Distract me',
    hint: 'Light topics and simple activities.',
    reply: "Let's take a short pause from deadlines. Quick question: if you could instantly master any creative skill, what would it be?",
  },
];

const STATUS: Record<Phase, string> = {
  idle: "Whenever you're ready.",
  permission: 'Allow microphone access to begin.',
  listening: "I'm listening. Take your time.",
  paused: 'Paused. Resume when you like.',
  processing: 'Transcribing speech & extracting acoustic traits...',
  analysing: 'Analyzing speech container & vocal patterns...',
  responding: "Here's something to think about.",
  error: 'Something went wrong.',
};

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

export default function Conversation() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [style, setStyle] = useState<StyleId>('listen');
  const [mode, setMode] = useState<'voice' | 'text'>('voice');
  const [languagePreference, setLanguagePreference] = useState<'auto' | 'en' | 'hi'>('auto');
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState('');
  const [progress, setProgress] = useState(0);
  const [paused, setPaused] = useState(false);
  const [spkMuted, setSpkMuted] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  const [hasInsights, setHasInsights] = useState(false);
  const [confirmEnd, setConfirmEnd] = useState(false);
  const [showUploader, setShowUploader] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [transcriptionNotice, setTranscriptionNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [sttModelName, setSttModelName] = useState<string>('');
  const [emotionModelName, setEmotionModelName] = useState<string>('');
  const [customError, setCustomError] = useState<string | null>(null);
  const [lastAudioBlob, setLastAudioBlob] = useState<Blob | null>(null);
  const [emotionData, setEmotionData] = useState<EmotionPredictionResponse | null>(null);

  const timers = useRef<number[]>([]);
  const later = useCallback((fn: () => void, ms: number) => {
    timers.current.push(window.setTimeout(fn, ms));
  }, []);
  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  useEffect(() => clearTimers, []);

  // Real microphone audio recorder hook
  const recorder = useAudioRecorder(36);

  // Fallback simulated audio (used during demo mode or AI speaking simulation)
  const speaking = phase === 'responding' && mode === 'voice' && !paused;
  const isSimulatedActive = (demoMode && phase === 'listening') || speaking;
  const simulatedAudio = useSimulatedAudio(isSimulatedActive, 36);

  // Check backend, Whisper, and Emotion model health
  const checkHealth = useCallback(async () => {
    setBackendStatus('checking');
    const [backRes, sttRes, emoRes] = await Promise.all([
      checkBackendHealth(),
      checkTranscriptionHealth(),
      checkEmotionHealth(),
    ]);

    if (backRes.online) {
      setBackendStatus('online');
      if (sttRes.online && sttRes.data) {
        setSttModelName(`Whisper ${sttRes.data.model_size}`);
      }
      if (emoRes.online && emoRes.data) {
        setEmotionModelName(emoRes.data.model_name || 'SVM (RAVDESS)');
      }
    } else {
      setBackendStatus('offline');
      setSttModelName('');
      setEmotionModelName('');
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Audio visualizer bars and level
  const isListeningReal = !demoMode && (phase === 'listening' || recorder.state === 'listening');
  const bars = isListeningReal
    ? recorder.bars
    : isSimulatedActive
    ? simulatedAudio.bars
    : Array<number>(36).fill(0.06);

  const level = isListeningReal
    ? recorder.level
    : isSimulatedActive
    ? simulatedAudio.level
    : 0;

  const orb: OrbMode =
    isListeningReal || (demoMode && phase === 'listening')
      ? 'listening'
      : speaking
      ? 'speaking'
      : phase === 'processing' || phase === 'analysing'
      ? 'thinking'
      : 'idle';

  // Handle recorder error
  useEffect(() => {
    if (recorder.errorMessage) {
      setCustomError(recorder.errorMessage);
      setPhase('error');
    }
  }, [recorder.errorMessage]);

  // AI playback progression
  useEffect(() => {
    if (phase !== 'responding') return;
    if (mode === 'text') {
      const t = setTimeout(() => setPhase('idle'), 2500);
      return () => clearTimeout(t);
    }
    if (paused) return;
    const id = setInterval(() => setProgress((p) => Math.min(100, p + 2)), 120);
    return () => clearInterval(id);
  }, [phase, mode, paused]);

  useEffect(() => {
    if (phase === 'responding' && mode === 'voice' && progress >= 100) {
      setPhase('idle');
    }
  }, [progress, phase, mode]);

  // Start real recording
  const beginListeningReal = async () => {
    setDemoMode(false);
    setTranscriptionNotice(null);
    setCustomError(null);
    const ok = await recorder.startRecording();
    if (ok) {
      setPhase('listening');
    } else {
      setPhase('error');
    }
  };

  // Start demo simulation mode
  const beginListeningDemo = () => {
    setDemoMode(true);
    setTranscriptionNotice('Running in Demo Mode (simulated speech & insights)');
    setCustomError(null);
    setPhase('listening');
  };

  // Process transcription & emotion response into conversation state
  const handleTranscriptionAndEmotionResult = (
    sttResult: TranscriptionResponse,
    emoResult?: EmotionPredictionResponse | null
  ) => {
    const transcriptText = sttResult.transcript.trim() || "(No speech detected in recording)";
    const langLabel = sttResult.detected_language === 'hinglish' ? 'Hinglish' : sttResult.detected_language.toUpperCase();
    const confidencePct = Math.round(sttResult.language_probability * 100);

    let notice = `✓ Transcribed in ${sttResult.processing_time_sec}s · Language: ${langLabel} (${confidencePct}%)`;
    if (emoResult && emoResult.is_reliable) {
      notice += ` · Vocal pattern: ${emoResult.emotion_label}-like (${Math.round(emoResult.confidence * 100)}%)`;
    }

    setTranscriptionNotice(notice);

    if (emoResult) {
      setEmotionData(emoResult);
      setHasInsights(true);
    }

    // Append user message with transcript
    setMsgs((m) => [
      ...m,
      {
        id: Date.now(),
        role: 'user',
        text: transcriptText,
        detectedLanguage: sttResult.detected_language,
        languageProb: sttResult.language_probability,
      },
    ]);

    setPhase('analysing');

    const reply = STYLES.find((s) => s.id === style)!.reply;
    later(() => {
      setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: reply }]);
      setProgress(0);
      setPaused(false);
      setPhase('responding');
    }, 1800);
  };

  // Stop recording and process with both Whisper STT and Emotion SER
  const stop = async () => {
    setPhase('processing');

    if (demoMode) {
      later(() => {
        setMsgs((m) => [
          ...m,
          {
            id: Date.now(),
            role: 'user',
            text: "Yaar, mujhe kal presentation complete karna hai, lekin thoda stress lag raha hai.",
            detectedLanguage: 'hinglish',
            languageProb: 0.94,
          },
        ]);
        setHasInsights(true);
        setEmotionData({
          success: true,
          emotion_label: 'calm',
          confidence: 0.62,
          probabilities: { calm: 0.62, sad: 0.18, neutral: 0.12, happy: 0.08 },
          vocal_traits: {
            pitch_mean_hz: 185.4,
            pitch_std_hz: 22.8,
            rms_energy: 0.048,
            spectral_centroid_hz: 1740.0,
            pause_fraction: 0.18,
            duration_sec: 4.2,
          },
          is_reliable: true,
          interpretation_notice: "Identifies measurable acoustic patterns in voice.",
          model_used: "Support Vector Machine (RAVDESS)",
          processing_time_sec: 0.08,
        });
        setPhase('analysing');
      }, 1200);

      const reply = STYLES.find((s) => s.id === style)!.reply;
      later(() => {
        setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: reply }]);
        setProgress(0);
        setPaused(false);
        setPhase('responding');
      }, 2500);
      return;
    }

    // Stop real recording
    const blob = await recorder.stopRecording();

    if (blob) {
      setLastAudioBlob(blob);
      try {
        setTranscriptionNotice('Analyzing audio: transcribing words and measuring acoustic patterns...');

        const filename = `recording.${blob.type.includes('wav') ? 'wav' : 'webm'}`;

        // Execute STT and SER in parallel
        const [sttOutcome, emoOutcome] = await Promise.allSettled([
          transcribeAudio(blob, filename, languagePreference === 'auto' ? undefined : languagePreference),
          predictEmotion(blob, filename),
        ]);

        if (sttOutcome.status === 'fulfilled') {
          const emoResult = emoOutcome.status === 'fulfilled' ? emoOutcome.value : null;
          handleTranscriptionAndEmotionResult(sttOutcome.value, emoResult);
        } else {
          const errorMsg = sttOutcome.reason?.message || 'Transcription failed';
          setCustomError(`Speech recognition failed: ${errorMsg}`);
          setPhase('error');
        }
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : 'Processing failed';
        setCustomError(`Processing error: ${errorMsg}`);
        setPhase('error');
      }
    } else {
      setCustomError('No audio data was captured.');
      setPhase('error');
    }
  };

  // Retry transcription & emotion analysis
  const retryTranscription = async () => {
    if (!lastAudioBlob) {
      retry();
      return;
    }
    setPhase('processing');
    setCustomError(null);
    try {
      setTranscriptionNotice('Retrying speech and emotion analysis...');
      const [sttOutcome, emoOutcome] = await Promise.allSettled([
        transcribeAudio(lastAudioBlob, 'recording.webm', languagePreference === 'auto' ? undefined : languagePreference),
        predictEmotion(lastAudioBlob, 'recording.webm'),
      ]);

      if (sttOutcome.status === 'fulfilled') {
        const emoResult = emoOutcome.status === 'fulfilled' ? emoOutcome.value : null;
        handleTranscriptionAndEmotionResult(sttOutcome.value, emoResult);
      } else {
        setCustomError('Retry failed to transcribe audio.');
        setPhase('error');
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Retry failed';
      setCustomError(`Processing error: ${errorMsg}`);
      setPhase('error');
    }
  };

  // Clear all messages
  const clearTranscript = () => {
    setMsgs([]);
    setHasInsights(false);
    setEmotionData(null);
    setTranscriptionNotice('Transcript cleared.');
  };

  const retry = () => {
    clearTimers();
    recorder.clearError();
    setCustomError(null);
    setTranscriptionNotice(null);
    setPhase('idle');
  };

  const replay = () => {
    setProgress(0);
    setPaused(false);
    setPhase('responding');
  };

  const stopPlayback = () => {
    setPaused(false);
    setPhase('idle');
  };

  const endSession = () => {
    clearTimers();
    recorder.cancelRecording();
    window.location.hash = '#/';
  };

  const lastAi = [...msgs].reverse().find((m) => m.role === 'ai')?.id;
  const busy = phase === 'processing' || phase === 'analysing';

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-4">
        <div className="flex items-center gap-4">
          <Logo />
          {/* Backend Status indicator */}
          <div
            className="hidden sm:inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border border-ink/10 cursor-pointer"
            onClick={checkHealth}
            title="Click to re-check backend connection"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendStatus === 'online'
                  ? 'bg-emerald-500'
                  : backendStatus === 'checking'
                  ? 'bg-amber-400 animate-pulse'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-muted text-[11px]">
              {backendStatus === 'online'
                ? `${sttModelName || 'STT Ready'} · ${emotionModelName || 'SER Ready'}`
                : backendStatus === 'checking'
                ? 'Checking backend...'
                : 'Backend Offline'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Language selector */}
          <div className="inline-flex rounded-full bg-ink/5 p-0.5 text-xs">
            {(['auto', 'en', 'hi'] as const).map((l) => (
              <button
                key={l}
                type="button"
                onClick={() => setLanguagePreference(l)}
                className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition ${
                  languagePreference === l
                    ? 'bg-accent text-onaccent shadow-xs'
                    : 'text-muted hover:text-ink'
                }`}
                title={`Language: ${l === 'auto' ? 'Auto-Detect (EN/HI/Hinglish)' : l === 'en' ? 'English' : 'Hindi'}`}
              >
                {l === 'auto' ? 'Auto' : l === 'en' ? 'EN' : 'HI'}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="btn btn-ghost !py-2 text-xs"
            onClick={() => setShowUploader((u) => !u)}
            title="Upload audio file for transcription & emotion analysis"
          >
            <Icon name="mic" size={14} />
            {showUploader ? 'Hide Upload' : 'Upload File'}
          </button>
          <ThemeToggle />
          {confirmEnd ? (
            <>
              <button className="btn btn-primary !py-2 text-xs" onClick={endSession}>
                Confirm end
              </button>
              <button className="btn btn-ghost !py-2 text-xs" onClick={() => setConfirmEnd(false)}>
                Keep going
              </button>
            </>
          ) : (
            <button className="btn btn-ghost !py-2 text-xs" onClick={() => setConfirmEnd(true)}>
              End session
            </button>
          )}
        </div>
      </header>

      {/* Notice Banner */}
      {transcriptionNotice && (
        <div className="mx-auto mb-2 w-full max-w-6xl px-5">
          <span className="chip !text-[11px] !bg-lav border border-accent/20">
            {transcriptionNotice}
          </span>
        </div>
      )}

      {/* File Uploader Drawer */}
      <AnimatePresence>
        {showUploader && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mx-auto w-full max-w-6xl px-5 mb-5 overflow-hidden"
          >
            <AudioUploader
              onTranscriptionSuccess={async (sttRes) => {
                // Also infer emotion
                let emoRes: EmotionPredictionResponse | null = null;
                try {
                  if (lastAudioBlob) {
                    emoRes = await predictEmotion(lastAudioBlob);
                  }
                } catch {
                  // Ignore emotion failure
                }
                handleTranscriptionAndEmotionResult(sttRes, emoRes);
              }}
              onCancel={() => setShowUploader(false)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-5 px-5 pb-8 lg:grid-cols-[1fr_1.05fr]">
        {/* Voice Stage */}
        <section className="card flex flex-col items-center px-6 py-8" aria-label="Voice stage">
          <Orb mode={orb} level={level} className="w-56 sm:w-64" />

          <p className="mt-8 text-center font-serif text-2xl" role="status" aria-live="polite">
            {STATUS[phase]}
          </p>

          <div className="mt-4 w-full max-w-sm">
            <Waveform bars={bars} active={phase === 'listening' || speaking} />
          </div>

          <p className="mt-1 h-5 text-sm tabular-nums text-muted">
            {(phase === 'listening' || phase === 'paused') && (
              <>
                Recording {fmt(demoMode ? 5 : recorder.duration)}
                {recorder.isMuted && ' · microphone muted'}
                {demoMode && ' · simulated'}
              </>
            )}
          </p>

          <div className="mt-5 flex min-h-[3.5rem] w-full flex-col items-center justify-center gap-3">
            {/* Idle State Controls */}
            {phase === 'idle' && (
              <div className="flex flex-col items-center gap-2">
                <button
                  className="btn btn-primary !px-8 !py-3.5 text-base shadow-sm"
                  onClick={() => setPhase('permission')}
                >
                  <Icon name="mic" />
                  Start talking
                </button>
              </div>
            )}

            {/* Permission Stage */}
            {phase === 'permission' && (
              <div className="max-w-sm rounded-2xl bg-lav p-5 text-center">
                <p className="text-sm leading-relaxed">
                  VocalSense listens to your words (Whisper STT) and measurable vocal traits (pitch, energy, pauses) with zero raw retention by default.
                </p>
                <div className="mt-4 flex flex-col sm:flex-row justify-center gap-2">
                  <button className="btn btn-primary text-xs" onClick={beginListeningReal}>
                    Allow Microphone
                  </button>
                  <button className="btn btn-soft text-xs" onClick={beginListeningDemo}>
                    Try Demo (No Mic)
                  </button>
                  <button className="btn btn-ghost text-xs" onClick={() => setPhase('idle')}>
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Listening / Paused Controls */}
            {(phase === 'listening' || phase === 'paused') && (
              <div className="flex flex-wrap items-center justify-center gap-2">
                {!demoMode && (
                  <>
                    {phase === 'listening' ? (
                      <button
                        className="btn btn-soft"
                        onClick={() => {
                          recorder.pauseRecording();
                          setPhase('paused');
                        }}
                      >
                        <Icon name="pause" size={18} />
                        Pause
                      </button>
                    ) : (
                      <button
                        className="btn btn-soft"
                        onClick={() => {
                          recorder.resumeRecording();
                          setPhase('listening');
                        }}
                      >
                        <Icon name="play" size={18} />
                        Resume
                      </button>
                    )}

                    <button
                      className="btn btn-soft"
                      aria-pressed={recorder.isMuted}
                      onClick={recorder.toggleMute}
                    >
                      <Icon name={recorder.isMuted ? 'micoff' : 'mic'} size={18} />
                      {recorder.isMuted ? 'Unmute' : 'Mute'}
                    </button>
                  </>
                )}

                <button className="btn btn-primary" onClick={stop}>
                  <Icon name="stop" size={18} />
                  Stop & Process
                </button>

                <button
                  className="btn btn-ghost"
                  onClick={() => {
                    recorder.cancelRecording();
                    retry();
                  }}
                >
                  <Icon name="replay" size={18} />
                  Cancel
                </button>
              </div>
            )}

            {busy && (
              <div className="flex items-center gap-2 text-sm text-muted">
                <span className="h-2 w-2 rounded-full bg-accent animate-ping" />
                <span>Running speech recognition & acoustic pattern analysis...</span>
              </div>
            )}

            {phase === 'responding' && mode === 'voice' && (
              <button className="btn btn-ghost" onClick={stopPlayback}>
                <Icon name="stop" size={18} />
                Stop reply
              </button>
            )}

            {phase === 'error' && (
              <div className="max-w-sm rounded-2xl bg-pink p-5 text-center">
                <p className="text-sm leading-relaxed text-ink">
                  {customError || "Speech processing encountered an error. Please verify your connection or microphone and try again."}
                </p>
                <div className="mt-4 flex justify-center gap-2">
                  {lastAudioBlob && (
                    <button className="btn btn-primary text-xs" onClick={retryTranscription}>
                      <Icon name="replay" size={16} />
                      Retry Analysis
                    </button>
                  )}
                  <button className="btn btn-soft text-xs" onClick={retry}>
                    New Recording
                  </button>
                </div>
              </div>
            )}
          </div>

          {phase === 'idle' && (
            <div className="mt-4 flex items-center gap-4 text-xs text-muted">
              <button
                className="underline underline-offset-4 hover:text-ink"
                onClick={beginListeningDemo}
              >
                Quick demo (no mic)
              </button>
              <span>·</span>
              <button
                className="underline underline-offset-4 hover:text-ink"
                onClick={() => {
                  setCustomError('Previewing simulated speech error state.');
                  setPhase('error');
                }}
              >
                Preview error state
              </button>
            </div>
          )}
        </section>

        {/* Conversation Transcript & Controls */}
        <div className="flex flex-col gap-5">
          <section className="card flex min-h-[18rem] flex-1 flex-col p-6" aria-label="Conversation">
            <div className="flex items-center justify-between">
              <h2 className="font-serif text-xl">Conversation</h2>
              {msgs.length > 0 && (
                <button
                  type="button"
                  className="text-xs text-muted hover:text-ink underline underline-offset-2"
                  onClick={clearTranscript}
                >
                  Clear transcript
                </button>
              )}
            </div>

            <div className="mt-4 flex flex-1 flex-col gap-3 overflow-y-auto" aria-live="polite">
              {msgs.length === 0 && (
                <div className="grid flex-1 place-items-center rounded-2xl border border-dashed border-ink/15 p-6 text-center text-sm text-muted">
                  Speak in English, Hindi, or Hinglish. Your words will be transcribed by Whisper and displayed here.
                </div>
              )}

              <AnimatePresence initial={false}>
                {msgs.map((m) => (
                  <motion.article
                    key={m.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`max-w-[92%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
                      m.role === 'user' ? 'self-end bg-lav' : 'self-start bg-peach'
                    }`}
                  >
                    <div className="mb-1 flex items-center justify-between gap-2 text-xs font-medium text-muted">
                      <span>{m.role === 'user' ? 'You' : 'VocalSense'}</span>
                      {m.role === 'user' && m.detectedLanguage && (
                        <span className="chip !text-[10px] !py-0.5 !px-2 uppercase tracking-wider">
                          {m.detectedLanguage} {m.languageProb ? `· ${Math.round(m.languageProb * 100)}%` : ''}
                        </span>
                      )}
                    </div>

                    {editing === m.id ? (
                      <div className="mt-1">
                        <textarea
                          className="w-full rounded-xl border border-ink/20 bg-surface p-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                          rows={3}
                          value={draft}
                          onChange={(e) => setDraft(e.target.value)}
                          aria-label="Edit transcript"
                        />
                        <div className="mt-2 flex gap-2">
                          <button
                            className="btn btn-primary !px-4 !py-1 text-xs"
                            onClick={() => {
                              setMsgs((all) =>
                                all.map((x) =>
                                  x.id === m.id ? { ...x, text: draft, isEdited: true } : x
                                )
                              );
                              setEditing(null);
                            }}
                          >
                            Save correction
                          </button>
                          <button
                            className="btn btn-ghost !px-4 !py-1 text-xs"
                            onClick={() => setEditing(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p>{m.text}</p>
                        {m.role === 'user' && (
                          <div className="mt-2 flex items-center gap-3 text-xs text-muted">
                            <button
                              className="inline-flex items-center gap-1 hover:text-ink"
                              onClick={() => {
                                setEditing(m.id);
                                setDraft(m.text);
                              }}
                            >
                              <Icon name="edit" size={13} />
                              Edit transcript
                            </button>
                            {m.isEdited && <span className="text-[11px] italic">(edited)</span>}
                          </div>
                        )}
                        {m.role === 'ai' && m.id === lastAi && mode === 'voice' && (
                          <div className="mt-3 flex items-center gap-2">
                            {phase === 'responding' ? (
                              <button
                                className="btn btn-soft !p-2"
                                aria-label={paused ? 'Resume playback' : 'Pause playback'}
                                onClick={() => setPaused((p) => !p)}
                              >
                                <Icon name={paused ? 'play' : 'pause'} size={16} />
                              </button>
                            ) : (
                              <button
                                className="btn btn-soft !p-2"
                                aria-label="Replay response"
                                onClick={replay}
                              >
                                <Icon name="replay" size={16} />
                              </button>
                            )}
                            <button
                              className="btn btn-soft !p-2"
                              aria-label={spkMuted ? 'Unmute playback' : 'Mute playback'}
                              aria-pressed={spkMuted}
                              onClick={() => setSpkMuted((s) => !s)}
                            >
                              <Icon name={spkMuted ? 'speakeroff' : 'speaker'} size={16} />
                            </button>
                            <div
                              className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink/10"
                              role="progressbar"
                              aria-valuenow={progress}
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-label="Playback progress"
                            >
                              <div
                                className="h-full rounded-full bg-accent transition-all"
                                style={{ width: `${progress}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </motion.article>
                ))}
              </AnimatePresence>
            </div>
          </section>

          {/* Settings & Controls */}
          <section className="card p-6" aria-label="Settings">
            <h2 className="text-sm font-semibold">Conversation style</h2>
            <div
              role="radiogroup"
              aria-label="Conversation style"
              className="mt-3 grid gap-2 sm:grid-cols-2"
            >
              {STYLES.map((s) => (
                <button
                  key={s.id}
                  role="radio"
                  aria-checked={style === s.id}
                  onClick={() => setStyle(s.id)}
                  className={`rounded-2xl border p-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                    style === s.id
                      ? 'border-accent bg-lav'
                      : 'border-ink/10 hover:bg-ink/5'
                  }`}
                >
                  <span className="block text-sm font-semibold">{s.name}</span>
                  <span className="block text-xs text-muted">{s.hint}</span>
                </button>
              ))}
            </div>

            <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold">Response mode</h2>
                <div
                  className="mt-2 inline-flex rounded-full bg-ink/5 p-1"
                  role="group"
                  aria-label="Response mode"
                >
                  {(['voice', 'text'] as const).map((m) => (
                    <button
                      key={m}
                      aria-pressed={mode === m}
                      onClick={() => setMode(m)}
                      className={`rounded-full px-4 py-1.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                        mode === m ? 'bg-accent text-onaccent' : 'text-ink'
                      }`}
                    >
                      {m === 'voice' ? 'Voice' : 'Text'}
                    </button>
                  ))}
                </div>
              </div>

              <label className="flex cursor-pointer items-center gap-3 text-sm">
                <span>Speech insights</span>
                <input
                  type="checkbox"
                  role="switch"
                  className="h-5 w-9 cursor-pointer accent-[rgb(var(--accent))]"
                  checked={showInsights}
                  onChange={(e) => setShowInsights(e.target.checked)}
                />
              </label>
            </div>

            {/* Speech Insights Panel (Actual Acoustic Measurements + Emotion Patterns) */}
            {showInsights && (
              <div className="mt-5 border-t border-ink/10 pt-5">
                {!hasInsights || !emotionData ? (
                  <p className="text-sm text-muted">Record or upload speech to analyze acoustic features and vocal patterns.</p>
                ) : (
                  <>
                    <p className="mb-3 text-xs text-muted">
                      Measurable physical acoustic traits from your recording. Values reflect acoustic measurements, not psychological diagnoses.
                    </p>
                    <dl className="grid gap-2 sm:grid-cols-2">
                      <div className="rounded-2xl p-3 bg-lav">
                        <dt className="text-xs font-medium text-ink/70">Pitch Variation (F0)</dt>
                        <dd className="text-sm font-medium">
                          {emotionData.vocal_traits.pitch_mean_hz > 0
                            ? `${emotionData.vocal_traits.pitch_mean_hz} Hz (±${emotionData.vocal_traits.pitch_std_hz} Hz)`
                            : 'Unvoiced / Whisper'}
                        </dd>
                      </div>

                      <div className="rounded-2xl p-3 bg-pink">
                        <dt className="text-xs font-medium text-ink/70">Vocal Energy (RMS)</dt>
                        <dd className="text-sm font-medium">
                          RMS {emotionData.vocal_traits.rms_energy} · {emotionData.vocal_traits.rms_energy > 0.08 ? 'Higher' : emotionData.vocal_traits.rms_energy < 0.03 ? 'Soft' : 'Moderate'}
                        </dd>
                      </div>

                      <div className="rounded-2xl p-3 bg-sage">
                        <dt className="text-xs font-medium text-ink/70">Pauses & Cadence</dt>
                        <dd className="text-sm font-medium">
                          {emotionData.vocal_traits.duration_sec}s audio · {Math.round(emotionData.vocal_traits.pause_fraction * 100)}% silence
                        </dd>
                      </div>

                      <div className="rounded-2xl p-3 bg-peach">
                        <dt className="text-xs font-medium text-ink/70">Spectral Centroid</dt>
                        <dd className="text-sm font-medium">
                          {Math.round(emotionData.vocal_traits.spectral_centroid_hz)} Hz frequency center
                        </dd>
                      </div>
                    </dl>

                    <div className="mt-3 rounded-2xl border border-ink/10 p-3.5 bg-surface text-sm">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs text-muted">Model: {emotionData.model_used}</span>
                        {emotionData.is_reliable && (
                          <span className="chip !text-[10px] !py-0.5 !px-2 uppercase">
                            {Math.round(emotionData.confidence * 100)}% confidence
                          </span>
                        )}
                      </div>
                      <p className="mt-1">
                        <span className="font-semibold">Vocal Pattern Estimate: </span>
                        {emotionData.is_reliable ? (
                          <span className="capitalize font-medium text-accent">
                            {emotionData.emotion_label}-like acoustic pattern
                          </span>
                        ) : (
                          <span className="italic text-muted">
                            Unable to estimate vocal emotion reliably.
                          </span>
                        )}
                      </p>
                    </div>

                    <p className="mt-2 text-xs text-muted leading-relaxed">
                      {emotionData.interpretation_notice || "These are measurable acoustic signals in your voice, not statements about your actual feelings. Your own words always come first."}
                    </p>
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
