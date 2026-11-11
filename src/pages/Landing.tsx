import { motion } from 'framer-motion';
import Icon from '../components/Icon';
import Logo from '../components/Logo';
import Orb from '../components/Orb';
import ThemeToggle from '../components/ThemeToggle';
import Waveform from '../components/Waveform';
import { useSimulatedAudio } from '../hooks/useSimulatedAudio';

const features = [
  { t: 'Speak in your own language', d: 'English, Hindi and Hinglish. Your words are never translated unless you ask.', bg: 'bg-lav' },
  { t: 'Four ways to be heard', d: 'Just listen, help me think, practical suggestions, or distract me. You choose, every time.', bg: 'bg-pink' },
  { t: 'Optional speech insights', d: 'See measurable traits like pace and pauses, in neutral language, with honest uncertainty.', bg: 'bg-sage' },
  { t: 'Memory you control', d: 'Save what is useful, edit it, or delete it. Temporary sessions leave nothing behind.', bg: 'bg-peach' },
];

const privacyPoints = [
  { icon: 'mic', t: 'Recording only starts when you press start and allow access.', c: 'bg-lav' },
  { icon: 'shield', t: 'Raw audio is never stored. Deleting your data is always one step away.', c: 'bg-pink' },
  { icon: 'history', t: 'Voice baselines and conversation memory are opt-in, and separate from each other.', c: 'bg-peach' },
  { icon: 'error', t: 'VocalSense is a conversational companion. It is not a therapist and never diagnoses.', c: 'bg-sage' },
];

const steps = [
  {
    t: 'Press start — microphone activates',
    d: "Your browser's native permission prompt appears once. After granting, recording begins immediately. The animated orb and waveform show that VocalSense is actively listening.",
    ui: (
      <div className="mt-3 rounded-2xl bg-white/60 dark:bg-white/5 border border-ink/10 p-4 flex flex-col items-center gap-2">
        <div className="text-xs text-muted font-mono mb-1">— Listening —</div>
        <div className="flex gap-1 items-end h-7">
          {[0.3,0.6,0.9,0.7,0.5,0.8,0.4,0.95,0.6,0.3].map((h,i) => (
            <div key={i} className="w-1.5 rounded-full bg-accent/60 animate-pulse" style={{height:`${h*100}%`, animationDelay:`${i*80}ms`}} />
          ))}
        </div>
        <span className="inline-flex items-center gap-1 text-[11px] text-muted mt-1">
          <span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-pulse inline-block" /> Recording · 0:04
        </span>
      </div>
    ),
  },
  {
    t: 'Your words appear as text — edit freely',
    d: 'Whisper transcribes what you said in your spoken language. Hinglish stays as Hinglish. You can tap any word to correct it before it enters the conversation.',
    ui: (
      <div className="mt-3 rounded-2xl bg-white/60 dark:bg-white/5 border border-ink/10 p-4 space-y-1.5">
        <div className="text-xs text-muted font-mono mb-1">— Transcript —</div>
        <div className="rounded-xl bg-lav/60 p-2.5 text-xs leading-relaxed text-ink">
          "Yaar, aaj kuch acha nahi lag raha, pata nahi kyun…"
        </div>
        <div className="flex gap-1.5 mt-1">
          <span className="chip !text-[10px] !px-1.5 !py-0.5">hinglish · 94%</span>
          <span className="chip !text-[10px] !px-1.5 !py-0.5">✎ Edit transcript</span>
        </div>
      </div>
    ),
  },
  {
    t: 'Pick how you want to be responded to',
    d: 'Choose from four conversation styles — Just listen, Help me think, Practical suggestions, or Distract me — and VocalSense replies in your language, matched to how you speak.',
    ui: (
      <div className="mt-3 rounded-2xl bg-white/60 dark:bg-white/5 border border-ink/10 p-4 space-y-2">
        <div className="text-xs text-muted font-mono mb-1">— Style picker —</div>
        {['Just listen','Help me think','Practical','Distract me'].map((s,i) => (
          <div key={s} className={`rounded-xl px-3 py-1.5 text-xs text-ink ${i === 0 ? 'bg-accent/20 font-semibold ring-1 ring-accent/40' : 'bg-ink/5'}`}>{s}</div>
        ))}
      </div>
    ),
  },
  {
    t: 'See what was measured in your voice',
    d: 'The optional insights panel shows pitch variation, speaking pace, energy, and pauses — described in neutral, measurable language. No labels. No diagnosis. Just honest numbers.',
    ui: (
      <div className="mt-3 rounded-2xl bg-white/60 dark:bg-white/5 border border-ink/10 p-4 space-y-1.5">
        <div className="text-xs text-muted font-mono mb-1">— Vocal traits —</div>
        {[['Pitch mean','185 Hz'],['Pace','Moderate'],['Pause fraction','18%'],['Energy','Low–mid']].map(([k,v]) => (
          <div key={k} className="flex justify-between text-xs">
            <span className="text-muted">{k}</span>
            <span className="font-mono text-ink">{v}</span>
          </div>
        ))}
        <p className="mt-2 text-[10px] text-muted leading-relaxed">These are acoustic measurements. They are not diagnoses.</p>
      </div>
    ),
  },
];

export default function Landing() {
  const { bars } = useSimulatedAudio(true, 44);
  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-5 py-5">
        <Logo />
        <nav className="flex items-center gap-2">
          <a href="#how" className="btn hidden sm:inline-flex">How it works</a>
          <a href="#privacy" className="btn hidden sm:inline-flex">Privacy</a>
          <ThemeToggle />
        </nav>
      </header>

      <main>
        <section className="mx-auto grid max-w-6xl items-center gap-10 px-5 pb-20 pt-8 lg:grid-cols-[1.1fr_.9fr] lg:pt-16">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
            <h1 className="max-w-xl font-serif text-5xl leading-[1.08] sm:text-6xl">Sometimes, you just need someone to listen.</h1>
            <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted">
              A space to speak freely, reflect, and feel heard. Take a breath. Speak naturally. We'll take it one conversation at a time.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a href="#/talk" className="btn btn-primary !px-7 !py-3.5 text-base"><Icon name="mic" size={18} />Start Talking</a>
              <a href="#how" className="btn btn-ghost !px-7 !py-3.5 text-base">Explore How It Works</a>
            </div>
            <p className="mt-6 flex items-center gap-2 text-sm text-muted">
              <Icon name="shield" size={16} />The microphone only turns on when you press start. Audio is not stored by default.
            </p>
          </motion.div>

          <div className="mx-auto w-full max-w-md">
            <Orb mode="listening" level={bars.reduce((a, b) => a + b, 0) / bars.length} className="mx-auto w-4/5" />
            <div className="mt-6"><Waveform bars={bars} active /></div>
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-5 pb-20">
          <h2 className="max-w-lg font-serif text-3xl sm:text-4xl">Built to listen to what you say, and how you say it.</h2>
          <div className="mt-10 grid gap-4 sm:grid-cols-2">
            {features.map((f) => (
              <article key={f.t} className={`rounded-3xl p-7 ${f.bg}`}>
                <h3 className="text-lg font-semibold">{f.t}</h3>
                <p className="mt-2 max-w-sm leading-relaxed text-ink/80">{f.d}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="how" className="mx-auto max-w-6xl scroll-mt-6 px-5 pb-20">
          <h2 className="font-serif text-3xl sm:text-4xl">How it works</h2>
          <p className="mt-3 max-w-xl text-muted leading-relaxed">Four steps, in the order you'll experience them. No account needed. No setup required.</p>
          <ol className="mt-10 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
            {steps.map((s, i) => (
              <li key={s.t} className="card p-6 flex flex-col">
                <span className="grid h-8 w-8 place-items-center rounded-full bg-lav text-sm font-semibold flex-shrink-0">{i + 1}</span>
                <h3 className="mt-4 font-semibold leading-snug">{s.t}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted flex-1">{s.d}</p>
                {s.ui}
              </li>
            ))}
          </ol>
        </section>

        <section id="privacy" className="mx-auto max-w-6xl scroll-mt-6 px-5 pb-24">
          <div className="rounded-[2rem] bg-sage p-8 sm:p-12">
            <h2 className="font-serif text-3xl sm:text-4xl">Your voice, your rules.</h2>
            <ul className="mt-8 grid gap-4 sm:grid-cols-2">
              {privacyPoints.map((pt) => (
                <li key={pt.t} className="flex items-start gap-3">
                  <span className={`mt-0.5 flex-shrink-0 grid h-8 w-8 place-items-center rounded-full ${pt.c}`}>
                    <Icon name={pt.icon} size={15} />
                  </span>
                  <span className="text-sm leading-relaxed text-ink/85">{pt.t}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-ink/10">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-8 text-sm text-muted">
          <p>VocalSense · Multilingual Emotion-Aware Conversational AI Companion.</p>
          <p>If you are in immediate danger, contact local emergency services or someone you trust.</p>
        </div>
      </footer>
    </div>
  );
}
