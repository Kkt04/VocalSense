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
const steps = [
  { t: 'Speak naturally', d: 'Press start and talk. Pause or stop whenever you like.' },
  { t: 'Check the transcript', d: 'Your words appear as text. Fix anything before it is used.' },
  { t: 'Choose how to be answered', d: 'Pick a conversation style and get a spoken or written reply.' },
  { t: 'Look deeper, if you want', d: 'Open the insights panel to see what was measured in your voice.' },
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
            <p className="mt-2 text-center text-xs text-muted">Illustration of the live waveform. No microphone is in use.</p>
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
          <ol className="mt-10 grid gap-4 md:grid-cols-4">
            {steps.map((s, i) => (
              <li key={s.t} className="card p-6">
                <span className="grid h-8 w-8 place-items-center rounded-full bg-lav text-sm font-semibold">{i + 1}</span>
                <h3 className="mt-4 font-semibold">{s.t}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{s.d}</p>
              </li>
            ))}
          </ol>
        </section>

        <section id="privacy" className="mx-auto max-w-6xl scroll-mt-6 px-5 pb-24">
          <div className="rounded-[2rem] bg-sage p-8 sm:p-12">
            <h2 className="font-serif text-3xl sm:text-4xl">Your voice, your rules.</h2>
            <ul className="mt-6 grid gap-3 text-ink/85 sm:grid-cols-2">
              <li>Recording starts only after you press start and allow access.</li>
              <li>Raw audio is not kept by default. Deleting your data is always one step away.</li>
              <li>Voice baselines and conversation memory are opt-in, and separate from each other.</li>
              <li>VocalSense is a conversational companion. It is not a therapist and never diagnoses.</li>
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-ink/10">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-8 text-sm text-muted">
          <p>VocalSense. Phase 1 prototype: interface only, with simulated audio and replies.</p>
          <p>If you are in immediate danger, contact local emergency services or someone you trust.</p>
        </div>
      </footer>
    </div>
  );
}
