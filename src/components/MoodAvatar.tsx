import { motion, useReducedMotion } from 'framer-motion';

export type MoodType = 'neutral' | 'happy' | 'sad' | 'angry' | 'calm' | 'fearful' | 'surprised' | 'disgust';

const COLORS: Record<MoodType, { bg: string; cheek: string }> = {
  neutral:   { bg: '#E5DAF2', cheek: '#E8C8D4' },
  happy:     { bg: '#FFF0C8', cheek: '#FFB8C8' },
  sad:       { bg: '#C8D8F0', cheek: '#B8C0D8' },
  angry:     { bg: '#F8C8C0', cheek: '#E89888' },
  calm:      { bg: '#D6E8D4', cheek: '#C0D8B8' },
  fearful:   { bg: '#E0D0F0', cheek: '#D0B8E0' },
  surprised: { bg: '#FFF0D8', cheek: '#FFD0A8' },
  disgust:   { bg: '#D8E0C8', cheek: '#C8D0A0' },
};

/** Tiny SVG companion whose face changes with the detected emotion. */
export default function MoodAvatar({ mood = 'neutral', size = 48 }: { mood?: MoodType; size?: number }) {
  const reduce = useReducedMotion();
  const c = COLORS[mood] || COLORS.neutral;

  // Mouth path by mood
  const mouth = (() => {
    switch (mood) {
      case 'happy':     return 'M 14 22 Q 20 28, 26 22'; // wide smile
      case 'sad':       return 'M 14 25 Q 20 20, 26 25'; // frown
      case 'angry':     return 'M 15 24 L 25 24';         // flat tight line
      case 'calm':      return 'M 15 23 Q 20 25, 25 23'; // gentle smile
      case 'fearful':   return 'M 17 23 Q 20 20, 23 23'; // small frown
      case 'surprised': return '';                         // use circle instead
      case 'disgust':   return 'M 15 24 Q 20 22, 25 24'; // slight squiggle
      default:          return 'M 16 23 Q 20 24.5, 24 23'; // neutral slight curve
    }
  })();

  // Eye shape adjustments
  const eyeR = mood === 'surprised' ? 2.8 : mood === 'sad' ? 1.8 : mood === 'angry' ? 1.6 : 2.2;

  // Eyebrow tilt for angry / sad
  const browL = (() => {
    switch (mood) {
      case 'angry':  return { x1: 13, y1: 13, x2: 17, y2: 11.5 };
      case 'sad':    return { x1: 13, y1: 12, x2: 17, y2: 13.5 };
      case 'surprised': return { x1: 13, y1: 10.5, x2: 17, y2: 10.5 };
      default:       return null;
    }
  })();
  const browR = (() => {
    switch (mood) {
      case 'angry':  return { x1: 23, y1: 11.5, x2: 27, y2: 13 };
      case 'sad':    return { x1: 23, y1: 13.5, x2: 27, y2: 12 };
      case 'surprised': return { x1: 23, y1: 10.5, x2: 27, y2: 10.5 };
      default:       return null;
    }
  })();

  // Gentle bounce on mood change
  const bounce = reduce ? {} : { scale: [1, 1.12, 1] };
  const bounceT = { duration: 0.45, ease: 'easeOut' as const };

  return (
    <motion.svg
      key={mood}
      viewBox="0 0 40 40"
      width={size}
      height={size}
      animate={bounce}
      transition={bounceT}
      aria-label={`Companion mood: ${mood}`}
      role="img"
    >
      {/* Head */}
      <circle cx="20" cy="20" r="17" fill={c.bg} stroke="rgb(var(--ink))" strokeWidth="0.8" strokeOpacity="0.15" />

      {/* Cheeks */}
      <circle cx="12" cy="21" r="3.2" fill={c.cheek} opacity="0.5" />
      <circle cx="28" cy="21" r="3.2" fill={c.cheek} opacity="0.5" />

      {/* Eyes */}
      <motion.circle
        cx="15" cy="16" r={eyeR}
        fill="rgb(var(--ink))"
        animate={reduce ? {} : { r: [eyeR, eyeR * 0.3, eyeR] }}
        transition={{ duration: 3.5, repeat: Infinity, repeatDelay: 2, ease: 'easeInOut' }}
      />
      <motion.circle
        cx="25" cy="16" r={eyeR}
        fill="rgb(var(--ink))"
        animate={reduce ? {} : { r: [eyeR, eyeR * 0.3, eyeR] }}
        transition={{ duration: 3.5, repeat: Infinity, repeatDelay: 2, ease: 'easeInOut', delay: 0.05 }}
      />

      {/* Eye highlights */}
      <circle cx="15.8" cy="15" r="0.7" fill="white" opacity="0.85" />
      <circle cx="25.8" cy="15" r="0.7" fill="white" opacity="0.85" />

      {/* Eyebrows (conditional) */}
      {browL && (
        <line x1={browL.x1} y1={browL.y1} x2={browL.x2} y2={browL.y2}
          stroke="rgb(var(--ink))" strokeWidth="1.2" strokeLinecap="round" strokeOpacity="0.6" />
      )}
      {browR && (
        <line x1={browR.x1} y1={browR.y1} x2={browR.x2} y2={browR.y2}
          stroke="rgb(var(--ink))" strokeWidth="1.2" strokeLinecap="round" strokeOpacity="0.6" />
      )}

      {/* Mouth */}
      {mood === 'surprised' ? (
        <circle cx="20" cy="24" r="2.5" fill="none" stroke="rgb(var(--ink))" strokeWidth="1.2" strokeOpacity="0.75" />
      ) : (
        <path d={mouth} fill="none" stroke="rgb(var(--ink))" strokeWidth="1.3" strokeLinecap="round" strokeOpacity="0.75" />
      )}

      {/* Happy mode: extra sparkle dots */}
      {mood === 'happy' && (
        <>
          <circle cx="8" cy="10" r="0.8" fill="rgb(var(--accent))" opacity="0.5" />
          <circle cx="32" cy="12" r="0.6" fill="rgb(var(--accent))" opacity="0.4" />
          <circle cx="10" cy="30" r="0.5" fill="rgb(var(--accent))" opacity="0.35" />
        </>
      )}

      {/* Sad mode: tear drop */}
      {mood === 'sad' && (
        <motion.ellipse
          cx="26.5" cy="19.5" rx="1" ry="1.8"
          fill="#88B8E0" opacity="0.7"
          animate={reduce ? {} : { cy: [19.5, 22, 19.5], opacity: [0.7, 0, 0.7] }}
          transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}
    </motion.svg>
  );
}
