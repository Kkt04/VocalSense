import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';

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

/**
 * Robust multilingual mood keyword detector.
 * Returns a MoodType if text contains strong emotional indicators, or null if neutral/conversational filler.
 */
export function analyzeMood(text: string): MoodType | null {
  if (!text) return null;
  const t = text.toLowerCase();

  const rules: { mood: MoodType; regex: RegExp }[] = [
    {
      mood: 'happy',
      regex: /\b(happy|happiness|joy|joyful|great|excited|exciting|smile|smiling|yay|haha|hehe|lol|love|loved|loving|awesome|amazing|wonderful|glad|cheerful|delighted|pleased|proud|celebrate|celebrating|blessed|fantastic|fun|sweet|superb|brilliant|khush|khushi|maza|mazedaar|badhiya|badiya|mast|achha|achhi|achhe|shandar|zabardast|rocking|delight)\b|\b(feeling good|feel good|so good|feeling happy|felt great)\b/i,
    },
    {
      mood: 'sad',
      regex: /\b(sad|sadness|unhappy|depressed|depression|crying|cry|cried|tears|tear|heartbroken|sorrow|lonely|down|gloomy|gloom|grief|hurt|hurting|pain|painful|upset|disappointed|hopeless|miserable|weep|weeping|udaas|udas|dukh|dukhi|dard|rona|ro raha|ro rahi|tut gaya|toot gaya|akela|akeli|bura lag|broken)\b|\b(feeling low|feel low|felt sad|feel sad|feeling down|lost someone)\b/i,
    },
    {
      mood: 'angry',
      regex: /\b(angry|anger|mad|furious|rage|irritated|irritating|irritat|annoyed|annoying|hate|hating|frustrated|frustrating|pissed|disgusted|disgust|gussa|gusse|bakwas|bekaar|chidh|nafrat|dimag kharab)\b|\b(so mad|very angry|fed up)\b/i,
    },
    {
      mood: 'fearful',
      regex: /\b(scared|scary|fear|fearful|afraid|terrified|anxious|anxiety|panic|panicking|nervous|worry|worried|worries|worrying|stress|stressed|stressful|tension|overwhelmed|horrified|dread|frightened|dar|darr|khauf|ghabrahat|ghabra|tension ho|scared of)\b/i,
    },
    {
      mood: 'calm',
      regex: /\b(calm|calming|peace|peaceful|relax|relaxed|relaxing|chill|chilling|serene|tranquil|soothing|soothe|rested|content|comfort|zen|sukoon|shanti|araam|aram|thanda|chain|khamosh)\b/i,
    },
    {
      mood: 'surprised',
      regex: /\b(surprised|surprising|shocked|shocking|shock|wow|whoa|omg|unexpected|amazed|astonished|unbelievable|hairan|chauk|chouka)\b/i,
    },
  ];

  for (const { mood, regex } of rules) {
    if (regex.test(t)) {
      return mood;
    }
  }

  return null;
}

/**
 * Animated SVG companion avatar.
 * Changes expression and pastel theme smoothly with seamless color interpolation and soft head tilts.
 * Maintains crisp, high-contrast dark features in both Light and Dark themes so expressions are never washed out.
 */
export default function MoodAvatar({ mood = 'neutral', size = 48 }: { mood?: MoodType; size?: number }) {
  const reduce = useReducedMotion();
  const { theme } = useTheme();
  const isDark = theme === 'dark' || (typeof document !== 'undefined' && document.documentElement.classList.contains('dark'));
  const c = COLORS[mood] || COLORS.neutral;

  // Crisp facial feature color: always dark so eyes, mouth, and eyebrows stand out sharply on the pastel face
  const faceInk = '#211C2B';

  // Mouth path definition
  const mouthPath = (() => {
    switch (mood) {
      case 'happy':     return 'M 14 22 Q 20 28.5, 26 22';
      case 'sad':       return 'M 14 25.5 Q 20 20, 26 25.5';
      case 'angry':     return 'M 15 24 L 25 24';
      case 'calm':      return 'M 15 23 Q 20 25.5, 25 23';
      case 'fearful':   return 'M 17 23.5 Q 20 20.5, 23 23.5';
      case 'disgust':   return 'M 15 24 Q 20 22, 25 24';
      default:          return 'M 16 23 Q 20 24.8, 24 23';
    }
  })();

  const eyeR = mood === 'surprised' ? 2.8 : mood === 'sad' ? 1.8 : mood === 'angry' ? 1.6 : 2.2;

  // Eyebrow geometry
  const browL = (() => {
    switch (mood) {
      case 'angry':     return { x1: 13, y1: 13, x2: 17, y2: 11.5 };
      case 'sad':       return { x1: 13, y1: 12, x2: 17, y2: 13.5 };
      case 'surprised': return { x1: 13, y1: 10.5, x2: 17, y2: 10.5 };
      case 'fearful':   return { x1: 13, y1: 11.5, x2: 17, y2: 12.5 };
      default:          return null;
    }
  })();

  const browR = (() => {
    switch (mood) {
      case 'angry':     return { x1: 23, y1: 11.5, x2: 27, y2: 13 };
      case 'sad':       return { x1: 23, y1: 13.5, x2: 27, y2: 12 };
      case 'surprised': return { x1: 23, y1: 10.5, x2: 27, y2: 10.5 };
      case 'fearful':   return { x1: 23, y1: 12.5, x2: 27, y2: 11.5 };
      default:          return null;
    }
  })();

  return (
    <motion.svg
      viewBox="0 0 40 40"
      width={size}
      height={size}
      animate={reduce ? {} : {
        rotate: mood === 'happy' ? [0, -3, 3, 0] : mood === 'sad' ? [0, 2, -1, 0] : [0, -1, 0],
        scale: [1, 1.05, 1],
      }}
      transition={{ duration: 0.8, ease: 'easeInOut' }}
      style={{
        filter: isDark ? 'drop-shadow(0 2px 8px rgba(0,0,0,0.45))' : 'drop-shadow(0 2px 5px rgba(38,34,46,0.08))',
      }}
      aria-label={`Companion mood: ${mood}`}
      role="img"
    >
      {/* Head — smoothly transitions fill color with distinct outline */}
      <motion.circle
        cx="20"
        cy="20"
        r="17"
        animate={{ fill: c.bg }}
        transition={{ duration: 0.8, ease: 'easeInOut' }}
        stroke={isDark ? 'rgba(255, 255, 255, 0.35)' : 'rgba(38, 34, 46, 0.15)'}
        strokeWidth="1"
      />

      {/* Cheeks — smooth color and scale transition */}
      <motion.circle
        cx="12"
        cy="21"
        r="3.2"
        animate={{ fill: c.cheek, scale: mood === 'happy' ? 1.25 : 1 }}
        transition={{ duration: 0.8, ease: 'easeInOut' }}
        opacity="0.6"
      />
      <motion.circle
        cx="28"
        cy="21"
        r="3.2"
        animate={{ fill: c.cheek, scale: mood === 'happy' ? 1.25 : 1 }}
        transition={{ duration: 0.8, ease: 'easeInOut' }}
        opacity="0.6"
      />

      {/* Left eye — always distinct high-contrast ink */}
      <motion.circle
        cx="15"
        cy="16"
        animate={{ r: eyeR }}
        transition={{ duration: 0.6, ease: 'easeInOut' }}
        fill={faceInk}
      />
      {/* Right eye */}
      <motion.circle
        cx="25"
        cy="16"
        animate={{ r: eyeR }}
        transition={{ duration: 0.6, ease: 'easeInOut' }}
        fill={faceInk}
      />

      {/* Eye glints */}
      <circle cx="15.8" cy="15" r="0.75" fill="#FFFFFF" opacity="0.9" />
      <circle cx="25.8" cy="15" r="0.75" fill="#FFFFFF" opacity="0.9" />

      {/* Eyebrows with smooth crossfade */}
      <AnimatePresence>
        {browL && (
          <motion.line
            key={`bl-${mood}`}
            x1={browL.x1}
            y1={browL.y1}
            x2={browL.x2}
            y2={browL.y2}
            stroke={faceInk}
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeOpacity="0.85"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.85 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {browR && (
          <motion.line
            key={`br-${mood}`}
            x1={browR.x1}
            y1={browR.y1}
            x2={browR.x2}
            y2={browR.y2}
            stroke={faceInk}
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeOpacity="0.85"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.85 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          />
        )}
      </AnimatePresence>

      {/* Mouth with smooth crossfade */}
      <AnimatePresence mode="wait">
        {mood === 'surprised' ? (
          <motion.circle
            key="surprised-mouth"
            cx="20"
            cy="24"
            r="2.5"
            fill="none"
            stroke={faceInk}
            strokeWidth="1.4"
            strokeOpacity="0.85"
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
          />
        ) : (
          <motion.path
            key={`mouth-${mood}`}
            d={mouthPath}
            fill="none"
            stroke={faceInk}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeOpacity="0.85"
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.85 }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
          />
        )}
      </AnimatePresence>

      {/* Happy sparkles */}
      <AnimatePresence>
        {mood === 'happy' && (
          <motion.g
            key="sparkles"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            transition={{ duration: 0.6 }}
          >
            <circle cx="8" cy="10" r="1" fill={isDark ? '#FBBF24' : '#7C3AED'} opacity="0.8" />
            <circle cx="32" cy="12" r="0.8" fill={isDark ? '#FBBF24' : '#7C3AED'} opacity="0.75" />
            <circle cx="10" cy="30" r="0.7" fill={isDark ? '#FBBF24' : '#7C3AED'} opacity="0.65" />
          </motion.g>
        )}
      </AnimatePresence>

      {/* Sad teardrop */}
      <AnimatePresence>
        {mood === 'sad' && (
          <motion.ellipse
            key="teardrop"
            cx="26.5"
            cy="19.5"
            rx="1.1"
            ry="1.9"
            fill="#38BDF8"
            stroke="#0284C7"
            strokeWidth="0.3"
            opacity="0.9"
            initial={{ opacity: 0, scale: 0.3 }}
            animate={reduce ? { opacity: 0.9 } : {
              opacity: [0, 0.9, 0.9, 0],
              cy: [19, 21.5, 23.5, 25],
            }}
            exit={{ opacity: 0 }}
            transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
          />
        )}
      </AnimatePresence>
    </motion.svg>
  );
}
