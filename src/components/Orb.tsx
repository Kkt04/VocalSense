import { motion, useReducedMotion } from 'framer-motion';

export type OrbMode = 'idle' | 'listening' | 'thinking' | 'speaking';

/** Decorative orb. It reacts to audio level and activity only — its colour never encodes an emotion. */
export default function Orb({ mode, level = 0, className = '' }: { mode: OrbMode; level?: number; className?: string }) {
  const reduce = useReducedMotion();
  const live = mode === 'listening' || mode === 'speaking';
  const core = reduce ? { scale: 1 } : live ? { scale: 1 + level * 0.22 } : { scale: [1, 1.035, 1] };
  const coreT = live ? { duration: 0.12 } : { duration: 5, repeat: Infinity, ease: 'easeInOut' as const };
  return (
    <div className={`relative aspect-square ${className}`} aria-hidden="true">
      <motion.div className="absolute -inset-[14%] rounded-full opacity-80 blur-3xl"
        style={{ background: 'radial-gradient(circle, rgb(var(--pink)) 0%, rgb(var(--lav)) 55%, transparent 72%)' }}
        animate={reduce ? {} : { scale: live ? 1 + level * 0.18 : [1, 1.07, 1] }}
        transition={live ? { duration: 0.12 } : { duration: 7, repeat: Infinity, ease: 'easeInOut' }} />
      {!reduce && [0, 1].map((i) => (
        <motion.div key={i} className="absolute inset-[6%] rounded-full border border-accent/30"
          animate={{ scale: [1, 1.3 + i * 0.12], opacity: [live ? 0.6 : 0.25, 0] }}
          transition={{ duration: 3.4, repeat: Infinity, delay: i * 1.6, ease: 'easeOut' }} />
      ))}
      <motion.div className="absolute inset-[8%] rounded-full shadow-orb"
        style={{ background: 'radial-gradient(circle at 32% 28%, rgb(var(--peach)), rgb(var(--lav)) 52%, rgb(var(--deep)) 125%)' }}
        animate={core} transition={coreT} />
      {mode === 'thinking' && (
        <motion.div className="absolute inset-[3%] rounded-full border-2 border-transparent border-t-accent/70"
          animate={reduce ? {} : { rotate: 360 }} transition={{ duration: 1.6, repeat: Infinity, ease: 'linear' }} />
      )}
    </div>
  );
}
