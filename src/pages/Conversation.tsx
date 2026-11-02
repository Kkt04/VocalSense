import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Icon from '../components/Icon';
import Logo from '../components/Logo';
import Orb, { OrbMode } from '../components/Orb';
import ThemeToggle from '../components/ThemeToggle';
import Waveform from '../components/Waveform';
import { useSimulatedAudio } from '../hooks/useSimulatedAudio';

type Phase = 'idle' | 'permission' | 'listening' | 'paused' | 'processing' | 'analysing' | 'responding' | 'error';
type StyleId = 'listen' | 'think' | 'practical' | 'distract';
type Msg = { id: number; role: 'user' | 'ai'; text: string };

/* ---------- MOCK DATA (Phase 1): replaced by Whisper + LLM in later phases ---------- */
const MOCK_TRANSCRIPT = "I'm honestly worried about my placement. I have prepared, but I keep thinking what if I mess up?";
const STYLES: { id: StyleId; name: string; hint: string; reply: string }[] = [
  { id: 'listen', name: 'Just listen', hint: 'Gentle acknowledgement and follow-up questions.',
    reply: "That sounds like a lot to carry, especially when you've already put in the work. What does the 'what if I mess up' thought usually sound like when it shows up?" },
  { id: 'think', name: 'Help me think', hint: 'Organise thoughts and explore perspectives.',
    reply: "Let's separate two things: what you've prepared, and what you can't control. Which part of the process feels most uncertain right now?" },
  { id: 'practical', name: 'Practical suggestions', hint: 'Clear, actionable steps.',
    reply: "Here are three small steps. 1) Do one timed mock interview this week. 2) Write a one-minute story for each of your strongest projects. 3) Plan a short routine for the night before. Want me to turn this into a schedule?" },
  { id: 'distract', name: 'Distract me', hint: 'Light topics and simple activities.',
    reply: "Let's take a short break from placements. Quick one: if you could instantly learn any skill that has nothing to do with computers, what would it be?" },
];
const INSIGHTS = [
  { label: 'Speaking pace', value: 'Higher than the sample reference range', bg: 'bg-lav' },
  { label: 'Pitch variation', value: 'Within the sample reference range', bg: 'bg-pink' },
  { label: 'Pauses', value: '4 pauses, longest 1.2 s', bg: 'bg-sage' },
  { label: 'Vocal energy', value: 'Slightly higher than the sample reference', bg: 'bg-peach' },
];
const STATUS: Record<Phase, string> = {
  idle: "Whenever you're ready.", permission: 'Allow microphone access to begin.', listening: "I'm listening. Take your time.",
  paused: 'Paused. Resume when you like.', processing: 'Putting your words together...', analysing: 'Looking at your speech patterns...',
  responding: "Here's something to think about.", error: 'Something went wrong.',
};
const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

export default function Conversation() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [seconds, setSeconds] = useState(0);
  const [muted, setMuted] = useState(false);
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
  const timers = useRef<number[]>([]);
  const later = useCallback((fn: () => void, ms: number) => { timers.current.push(window.setTimeout(fn, ms)); }, []);
  const clearTimers = () => { timers.current.forEach(clearTimeout); timers.current = []; };
  useEffect(() => clearTimers, []);

  const speaking = phase === 'responding' && mode === 'voice' && !paused;
  const micLive = phase === 'listening' && !muted;
  const { bars, level } = useSimulatedAudio(micLive || speaking);
  const orb: OrbMode = micLive ? 'listening' : speaking ? 'speaking' : phase === 'processing' || phase === 'analysing' ? 'thinking' : 'idle';

  // recording timer
  useEffect(() => {
    if (phase !== 'listening') return;
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [phase]);

  // simulated playback / text display
  useEffect(() => {
    if (phase !== 'responding') return;
    if (mode === 'text') { const t = setTimeout(() => setPhase('idle'), 2500); return () => clearTimeout(t); }
    if (paused) return;
    const id = setInterval(() => setProgress((p) => Math.min(100, p + 2)), 120);
    return () => clearInterval(id);
  }, [phase, mode, paused]);
  useEffect(() => { if (phase === 'responding' && mode === 'voice' && progress >= 100) setPhase('idle'); }, [progress, phase, mode]);

  const beginListening = () => { setSeconds(0); setMuted(false); setPhase('listening'); };
  const stop = () => {
    const reply = STYLES.find((s) => s.id === style)!.reply;
    setPhase('processing');
    later(() => { setMsgs((m) => [...m, { id: Date.now(), role: 'user', text: MOCK_TRANSCRIPT }]); setPhase('analysing'); }, 1400);
    later(() => {
      setMsgs((m) => [...m, { id: Date.now() + 1, role: 'ai', text: reply }]);
      setHasInsights(true); setProgress(0); setPaused(false); setPhase('responding');
    }, 2900);
  };
  const retry = () => { clearTimers(); beginListening(); };
  const replay = () => { setProgress(0); setPaused(false); setPhase('responding'); };
  const stopPlayback = () => { setPaused(false); setPhase('idle'); };
  const endSession = () => { clearTimers(); window.location.hash = '#/'; };
  const lastAi = [...msgs].reverse().find((m) => m.role === 'ai')?.id;
  const busy = phase === 'processing' || phase === 'analysing';

  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-4">
        <Logo />
        <div className="flex items-center gap-2">
          <ThemeToggle />
          {confirmEnd ? (
            <>
              <button className="btn btn-primary !py-2" onClick={endSession}>Confirm end</button>
              <button className="btn btn-ghost !py-2" onClick={() => setConfirmEnd(false)}>Keep going</button>
            </>
          ) : (
            <button className="btn btn-ghost !py-2" onClick={() => setConfirmEnd(true)}>End session</button>
          )}
        </div>
      </header>

      <p className="mx-auto mb-2 w-full max-w-6xl px-5">
        <span className="chip">Prototype: audio, transcript and replies are simulated. No microphone is used.</span>
      </p>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-5 px-5 pb-8 lg:grid-cols-[1fr_1.05fr]">
        {/* Voice stage */}
        <section className="card flex flex-col items-center px-6 py-8" aria-label="Voice stage">
          <Orb mode={orb} level={level} className="w-56 sm:w-64" />
          <p className="mt-8 text-center font-serif text-2xl" role="status" aria-live="polite">{STATUS[phase]}</p>
          <div className="mt-4 w-full max-w-sm"><Waveform bars={bars} active={micLive || speaking} /></div>
          <p className="mt-1 h-5 text-sm tabular-nums text-muted">
            {(phase === 'listening' || phase === 'paused') && <>Recording {fmt(seconds)}{muted && ' · microphone muted'}</>}
          </p>

          <div className="mt-5 flex min-h-[3.5rem] w-full flex-col items-center justify-center gap-3">
            {phase === 'idle' && <button className="btn btn-primary !px-8 !py-3.5 text-base" onClick={() => setPhase('permission')}><Icon name="mic" />Start talking</button>}
            {phase === 'permission' && (
              <div className="max-w-sm rounded-2xl bg-lav p-5 text-center">
                <p className="text-sm leading-relaxed">VocalSense needs your microphone to hear you. Audio is processed to make a transcript and is not stored by default.</p>
                <div className="mt-4 flex justify-center gap-2">
                  <button className="btn btn-primary" onClick={beginListening}>Allow (demo)</button>
                  <button className="btn btn-ghost" onClick={() => setPhase('idle')}>Not now</button>
                </div>
              </div>
            )}
            {(phase === 'listening' || phase === 'paused') && (
              <div className="flex flex-wrap items-center justify-center gap-2">
                {phase === 'listening'
                  ? <button className="btn btn-soft" onClick={() => setPhase('paused')}><Icon name="pause" size={18} />Pause</button>
                  : <button className="btn btn-soft" onClick={() => setPhase('listening')}><Icon name="play" size={18} />Resume</button>}
                <button className="btn btn-soft" aria-pressed={muted} onClick={() => setMuted((m) => !m)}><Icon name={muted ? 'micoff' : 'mic'} size={18} />{muted ? 'Unmute' : 'Mute'}</button>
                <button className="btn btn-primary" onClick={stop}><Icon name="stop" size={18} />Stop</button>
                <button className="btn btn-ghost" onClick={retry}><Icon name="replay" size={18} />Retry</button>
              </div>
            )}
            {busy && <p className="text-sm text-muted">This usually takes a few seconds.</p>}
            {phase === 'responding' && mode === 'voice' && (
              <button className="btn btn-ghost" onClick={stopPlayback}><Icon name="stop" size={18} />Stop reply</button>
            )}
            {phase === 'error' && (
              <div className="max-w-sm rounded-2xl bg-pink p-5 text-center">
                <p className="text-sm leading-relaxed">We couldn't reach the speech service, so your recording wasn't processed. Check your connection and try again.</p>
                <button className="btn btn-primary mt-4" onClick={retry}><Icon name="replay" size={18} />Retry recording</button>
              </div>
            )}
          </div>
          {phase === 'idle' && (
            <button className="mt-4 text-xs text-muted underline underline-offset-4 hover:text-ink" onClick={() => setPhase('error')}>Preview the error state</button>
          )}
        </section>

        {/* Transcript + settings */}
        <div className="flex flex-col gap-5">
          <section className="card flex min-h-[18rem] flex-1 flex-col p-6" aria-label="Conversation">
            <h2 className="font-serif text-xl">Conversation</h2>
            <div className="mt-4 flex flex-1 flex-col gap-3 overflow-y-auto" aria-live="polite">
              {msgs.length === 0 && (
                <div className="grid flex-1 place-items-center rounded-2xl border border-dashed border-ink/15 p-6 text-center text-sm text-muted">
                  Your words will appear here as text. Press Start talking to begin.
                </div>
              )}
              <AnimatePresence initial={false}>
                {msgs.map((m) => (
                  <motion.article key={m.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                    className={`max-w-[92%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${m.role === 'user' ? 'self-end bg-lav' : 'self-start bg-peach'}`}>
                    <p className="mb-1 text-xs font-medium text-muted">{m.role === 'user' ? 'You (mock transcript)' : 'VocalSense'}</p>
                    {editing === m.id ? (
                      <div>
                        <textarea className="w-full rounded-xl border border-ink/20 bg-surface p-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                          rows={3} value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Edit transcript" />
                        <div className="mt-2 flex gap-2">
                          <button className="btn btn-primary !px-4 !py-1.5" onClick={() => { setMsgs((all) => all.map((x) => (x.id === m.id ? { ...x, text: draft } : x))); setEditing(null); }}>Save</button>
                          <button className="btn btn-ghost !px-4 !py-1.5" onClick={() => setEditing(null)}>Cancel</button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p>{m.text}</p>
                        {m.role === 'user' && (
                          <button className="mt-2 inline-flex items-center gap-1 text-xs text-muted hover:text-ink" onClick={() => { setEditing(m.id); setDraft(m.text); }}>
                            <Icon name="edit" size={14} />Edit transcript
                          </button>
                        )}
                        {m.role === 'ai' && m.id === lastAi && mode === 'voice' && (
                          <div className="mt-3 flex items-center gap-2">
                            {phase === 'responding' ? (
                              <button className="btn btn-soft !p-2" aria-label={paused ? 'Resume playback' : 'Pause playback'} onClick={() => setPaused((p) => !p)}><Icon name={paused ? 'play' : 'pause'} size={16} /></button>
                            ) : (
                              <button className="btn btn-soft !p-2" aria-label="Replay response" onClick={replay}><Icon name="replay" size={16} /></button>
                            )}
                            <button className="btn btn-soft !p-2" aria-label={spkMuted ? 'Unmute playback' : 'Mute playback'} aria-pressed={spkMuted} onClick={() => setSpkMuted((s) => !s)}><Icon name={spkMuted ? 'speakeroff' : 'speaker'} size={16} /></button>
                            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink/10" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="Playback progress (simulated)">
                              <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${progress}%` }} />
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

          <section className="card p-6" aria-label="Settings">
            <h2 className="text-sm font-semibold">Conversation style</h2>
            <div role="radiogroup" aria-label="Conversation style" className="mt-3 grid gap-2 sm:grid-cols-2">
              {STYLES.map((s) => (
                <button key={s.id} role="radio" aria-checked={style === s.id} onClick={() => setStyle(s.id)}
                  className={`rounded-2xl border p-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${style === s.id ? 'border-accent bg-lav' : 'border-ink/10 hover:bg-ink/5'}`}>
                  <span className="block text-sm font-semibold">{s.name}</span>
                  <span className="block text-xs text-muted">{s.hint}</span>
                </button>
              ))}
            </div>

            <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold">Response mode</h2>
                <div className="mt-2 inline-flex rounded-full bg-ink/5 p-1" role="group" aria-label="Response mode">
                  {(['voice', 'text'] as const).map((m) => (
                    <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}
                      className={`rounded-full px-4 py-1.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${mode === m ? 'bg-accent text-onaccent' : 'text-ink'}`}>
                      {m === 'voice' ? 'Voice' : 'Text'}
                    </button>
                  ))}
                </div>
              </div>
              <label className="flex cursor-pointer items-center gap-3 text-sm">
                <span>Speech insights</span>
                <input type="checkbox" role="switch" className="h-5 w-9 cursor-pointer accent-[rgb(var(--accent))]" checked={showInsights} onChange={(e) => setShowInsights(e.target.checked)} />
              </label>
            </div>

            {showInsights && (
              <div className="mt-5 border-t border-ink/10 pt-5">
                {!hasInsights ? (
                  <p className="text-sm text-muted">Record something to see sample insights.</p>
                ) : (
                  <>
                    <p className="mb-3 text-xs text-muted">Sample values for design preview only. Nothing here was measured.</p>
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
                    <p className="mt-2 text-xs text-muted">These are measurable speech traits, not statements about how you feel. Your own words always come first.</p>
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
