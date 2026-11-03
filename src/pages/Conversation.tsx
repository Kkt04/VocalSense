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
import { checkBackendHealth, uploadAudio, AudioUploadResponse } from '../services/api';

type Phase = 'idle' | 'permission' | 'listening' | 'paused' | 'processing' | 'analysing' | 'responding' | 'error';
type StyleId = 'listen' | 'think' | 'practical' | 'distract';
type Msg = { id: number; role: 'user' | 'ai'; text: string };

/* ---------- MOCK DATA for Phase 1 dialogue and insights preview ---------- */
const MOCK_TRANSCRIPT = "I'm honestly worried about my placement. I have prepared, but I keep thinking what if I mess up?";
const STYLES: { id: StyleId; name: string; hint: string; reply: string }[] = [
  {
    id: 'listen',
    name: 'Just listen',
    hint: 'Gentle acknowledgement and follow-up questions.',
    reply: "That sounds like a lot to carry, especially when you've already put in the work. What does the 'what if I mess up' thought usually sound like when it shows up?",
  },
  {
    id: 'think',
    name: 'Help me think',
    hint: 'Organise thoughts and explore perspectives.',
    reply: "Let's separate two things: what you've prepared, and what you can't control. Which part of the process feels most uncertain right now?",
  },
  {
    id: 'practical',
    name: 'Practical suggestions',
    hint: 'Clear, actionable steps.',
    reply: "Here are three small steps. 1) Do one timed mock interview this week. 2) Write a one-minute story for each of your strongest projects. 3) Plan a short routine for the night before. Want me to turn this into a schedule?",
  },
  {
    id: 'distract',
    name: 'Distract me',
    hint: 'Light topics and simple activities.',
    reply: "Let's take a short break from placements. Quick one: if you could instantly learn any skill that has nothing to do with computers, what would it be?",
  },
];

const INSIGHTS = [
  { label: 'Speaking pace', value: 'Higher than reference range', bg: 'bg-lav' },
  { label: 'Pitch variation', value: 'Within sample reference range', bg: 'bg-pink' },
  { label: 'Pauses', value: '4 pauses, longest 1.2 s', bg: 'bg-sage' },
  { label: 'Vocal energy', value: 'Slightly higher than reference', bg: 'bg-peach' },
];

const STATUS: Record<Phase, string> = {
  idle: "Whenever you're ready.",
  permission: 'Allow microphone access to begin.',
  listening: "I'm listening. Take your time.",
  paused: 'Paused. Resume when you like.',
  processing: 'Sending audio to backend & processing...',
  analysing: 'Analyzing speech container & traits...',
  responding: "Here's something to think about.",
  error: 'Something went wrong.',
};

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

export default function Conversation() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [style, setStyle] = useState<StyleId>('listen');
  const [mode, setMode] = useState<'voice' | 'text'>('voice');
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
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [customError, setCustomError] = useState<string | null>(null);

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

  // Check backend health on mount and periodically
  const pingBackend = useCallback(async () => {
    setBackendStatus('checking');
    const res = await checkBackendHealth();
    setBackendStatus(res.online ? 'online' : 'offline');
  }, []);

  useEffect(() => {
    pingBackend();
  }, [pingBackend]);

  // Derive current audio visualizer bars and level
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

  // Handle errors from the real recorder
  useEffect(() => {
    if (recorder.errorMessage) {
      setCustomError(recorder.errorMessage);
      setPhase('error');
    }
  }, [recorder.errorMessage]);

  // Simulated AI response playback or text display progression
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

  // Start real recording flow
  const beginListeningReal = async () => {
    setDemoMode(false);
    setUploadNotice(null);
    setCustomError(null);
    const ok = await recorder.startRecording();
    if (ok) {
      setPhase('listening');
    } else {
      setPhase('error');
    }
  };

  // Start demo simulation mode (no microphone needed)
  const beginListeningDemo = () => {
    setDemoMode(true);
    setUploadNotice('Running in Simulated Demo Mode (no microphone)');
    setCustomError(null);
    setPhase('listening');
  };

  // Handle Stop Recording & Upload
  const stop = async () => {
    const reply = STYLES.find((s) => s.id === style)!.reply;
    setPhase('processing');

    if (demoMode) {
      // Demo mode simulated progression
      later(() => {
        setMsgs((m) => [...m, { id: Date.now(), role: 'user', text: MOCK_TRANSCRIPT }]);
        setPhase('analysing');
      }, 1200);
      later(() => {
        setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: reply }]);
        setHasInsights(true);
        setProgress(0);
        setPaused(false);
        setPhase('responding');
      }, 2500);
      return;
    }

    // Real audio recording stop
    const blob = await recorder.stopRecording();

    if (blob) {
      try {
        setUploadNotice('Uploading audio to backend...');
        const uploadResult: AudioUploadResponse = await uploadAudio(
          blob,
          `recording.${blob.type.includes('wav') ? 'wav' : 'webm'}`
        );

        setUploadNotice(
          `✓ Audio received (${uploadResult.audio.size_kb} KB, ${uploadResult.audio.format_detected.toUpperCase()}) · Validated by backend`
        );
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : 'Backend upload error';
        setUploadNotice(`Upload warning: ${errorMsg}. (Backend offline, proceeding in local mode)`);
      }
    }

    // Progress conversation
    later(() => {
      setMsgs((m) => [...m, { id: Date.now(), role: 'user', text: MOCK_TRANSCRIPT }]);
      setPhase('analysing');
    }, 1200);

    later(() => {
      setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: reply }]);
      setHasInsights(true);
      setProgress(0);
      setPaused(false);
      setPhase('responding');
    }, 2600);
  };

  // Callback when user uploads an audio file via the uploader
  const handleUploadedAudio = (response: AudioUploadResponse) => {
    setShowUploader(false);
    setUploadNotice(`✓ Uploaded ${response.audio.filename} (${response.audio.size_kb} KB) successfully`);
    setPhase('processing');

    const reply = STYLES.find((s) => s.id === style)!.reply;
    later(() => {
      setMsgs((m) => [
        ...m,
        {
          id: Date.now(),
          role: 'user',
          text: `[Audio File: ${response.audio.filename}] ${MOCK_TRANSCRIPT}`,
        },
      ]);
      setPhase('analysing');
    }, 1000);

    later(() => {
      setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: reply }]);
      setHasInsights(true);
      setProgress(0);
      setPaused(false);
      setPhase('responding');
    }, 2400);
  };

  const retry = () => {
    clearTimers();
    recorder.clearError();
    setCustomError(null);
    setUploadNotice(null);
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
            onClick={pingBackend}
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
              {backendStatus === 'online' ? 'Backend Connected' : backendStatus === 'checking' ? 'Checking...' : 'Backend Offline'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className="btn btn-ghost !py-2 text-xs"
            onClick={() => setShowUploader((u) => !u)}
            title="Upload pre-recorded audio file"
          >
            <Icon name="mic" size={14} />
            {showUploader ? 'Hide Uploader' : 'Upload File'}
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

      {/* Upload Notification Banner */}
      {uploadNotice && (
        <div className="mx-auto mb-2 w-full max-w-6xl px-5">
          <span className="chip !text-[11px] !bg-lav border border-accent/20">
            {uploadNotice}
          </span>
        </div>
      )}

      {/* File Uploader Drawer/Panel */}
      <AnimatePresence>
        {showUploader && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mx-auto w-full max-w-6xl px-5 mb-5 overflow-hidden"
          >
            <AudioUploader
              onUploadSuccess={handleUploadedAudio}
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
                  VocalSense needs access to your microphone to capture your speech and analyze acoustic characteristics. Raw audio is processed with zero-retention by default.
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

            {busy && <p className="text-sm text-muted">This usually takes a few seconds.</p>}

            {phase === 'responding' && mode === 'voice' && (
              <button className="btn btn-ghost" onClick={stopPlayback}>
                <Icon name="stop" size={18} />
                Stop reply
              </button>
            )}

            {phase === 'error' && (
              <div className="max-w-sm rounded-2xl bg-pink p-5 text-center">
                <p className="text-sm leading-relaxed text-ink">
                  {customError || "We couldn't access or process your speech. Please check your microphone permissions and connection, then try again."}
                </p>
                <div className="mt-4 flex justify-center gap-2">
                  <button className="btn btn-primary" onClick={retry}>
                    <Icon name="replay" size={18} />
                    Try again
                  </button>
                  <button className="btn btn-soft" onClick={beginListeningDemo}>
                    Use Demo Mode
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
                  setCustomError('Previewing simulated microphone error state.');
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
            <h2 className="font-serif text-xl">Conversation</h2>

            <div className="mt-4 flex flex-1 flex-col gap-3 overflow-y-auto" aria-live="polite">
              {msgs.length === 0 && (
                <div className="grid flex-1 place-items-center rounded-2xl border border-dashed border-ink/15 p-6 text-center text-sm text-muted">
                  Your words will appear here as text. Press Start talking or Upload File to begin.
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
                    <p className="mb-1 text-xs font-medium text-muted">
                      {m.role === 'user' ? 'You' : 'VocalSense'}
                    </p>

                    {editing === m.id ? (
                      <div>
                        <textarea
                          className="w-full rounded-xl border border-ink/20 bg-surface p-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                          rows={3}
                          value={draft}
                          onChange={(e) => setDraft(e.target.value)}
                          aria-label="Edit transcript"
                        />
                        <div className="mt-2 flex gap-2">
                          <button
                            className="btn btn-primary !px-4 !py-1.5"
                            onClick={() => {
                              setMsgs((all) =>
                                all.map((x) => (x.id === m.id ? { ...x, text: draft } : x))
                              );
                              setEditing(null);
                            }}
                          >
                            Save
                          </button>
                          <button
                            className="btn btn-ghost !px-4 !py-1.5"
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
                          <button
                            className="mt-2 inline-flex items-center gap-1 text-xs text-muted hover:text-ink"
                            onClick={() => {
                              setEditing(m.id);
                              setDraft(m.text);
                            }}
                          >
                            <Icon name="edit" size={14} />
                            Edit transcript
                          </button>
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

            {showInsights && (
              <div className="mt-5 border-t border-ink/10 pt-5">
                {!hasInsights ? (
                  <p className="text-sm text-muted">Record or upload speech to preview acoustic insights.</p>
                ) : (
                  <>
                    <p className="mb-3 text-xs text-muted">
                      Measurable acoustic speech traits (preview mode). Values are strictly non-diagnostic.
                    </p>
                    <dl className="grid gap-2 sm:grid-cols-2">
                      {INSIGHTS.map((i) => (
                        <div key={i.label} className={`rounded-2xl p-3 ${i.bg}`}>
                          <dt className="text-xs font-medium text-ink/70">{i.label}</dt>
                          <dd className="text-sm">{i.value}</dd>
                        </div>
                      ))}
                    </dl>
                    <p className="mt-3 rounded-2xl border border-ink/10 p-3 text-sm">
                      <span className="font-medium">Model estimate: </span>Unable to estimate reliably from this recording.
                    </p>
                    <p className="mt-2 text-xs text-muted">
                      These are measurable acoustic signals, not claims about emotional states. Your stated words always take precedence.
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
