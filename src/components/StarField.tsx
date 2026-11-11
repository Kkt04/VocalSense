import { useEffect, useRef } from 'react';
import { useTheme } from '../context/ThemeContext';

interface Star {
  x: number;
  y: number;
  r: number;
  alpha: number;
  speed: number;
  drift: number;
  driftDir: number;
  phase: number;
  isSparkle: boolean;
}

const COUNT = 100;

function mkStar(W: number, H: number, idx: number): Star {
  return {
    x: Math.random() * W,
    y: Math.random() * H,
    r: Math.random() * 1.5 + 0.7,
    alpha: Math.random() * 0.4 + 0.4,
    speed: Math.random() * 0.16 + 0.05,
    drift: Math.random() * 0.35 + 0.08,
    driftDir: Math.random() > 0.5 ? 1 : -1,
    phase: Math.random() * Math.PI * 2,
    isSparkle: idx % 7 === 0, // 4-pointed sparkle star every few stars
  };
}

/**
 * Full-page celestial background.
 * Stars twinkle and drift:
 * - Pure starry white in Dark Mode
 * - Vivid sparkling golden yellow in Light Mode
 */
export default function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const { theme } = useTheme();
  const reduced = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    let W = 0, H = 0;
    const stars: Star[] = [];

    const resize = () => {
      W = canvas.width = window.innerWidth;
      H = canvas.height = window.innerHeight;
      stars.length = 0;
      for (let i = 0; i < COUNT; i++) stars.push(mkStar(W, H, i));
    };

    resize();
    window.addEventListener('resize', resize, { passive: true });

    let t = 0;
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      t += 0.014;

      const isDark = document.documentElement.classList.contains('dark') || theme === 'dark';

      for (const s of stars) {
        // Twinkle factor via sine oscillation
        const twinkle = 0.5 + 0.5 * Math.sin(t * s.speed * 8 + s.phase);
        const a = Math.min(1, s.alpha * twinkle * (isDark ? 1 : 1.3));

        // Colors: pure white in dark theme, vivid golden-yellow in light theme
        const color = isDark
          ? `rgba(255, 255, 255, ${a})`
          : `rgba(235, 175, 25, ${a})`;

        ctx.fillStyle = color;

        if (s.isSparkle && s.r > 1.2) {
          // Draw delicate 4-pointed celestial star
          const size = s.r * 2.8 * twinkle;
          ctx.beginPath();
          ctx.moveTo(s.x, s.y - size);
          ctx.quadraticCurveTo(s.x, s.y, s.x + size, s.y);
          ctx.quadraticCurveTo(s.x, s.y, s.x, s.y + size);
          ctx.quadraticCurveTo(s.x, s.y, s.x - size, s.y);
          ctx.quadraticCurveTo(s.x, s.y, s.x, s.y - size);
          ctx.fill();

          // Soft ambient glow
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r * 0.7, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // Standard circular starry dot
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
          ctx.fill();
        }

        if (!reduced) {
          // Gentle celestial float upward and horizontal drift
          s.y -= s.speed;
          s.x += s.drift * s.driftDir * Math.sin(t * 0.8 + s.phase);
          if (s.y < -8) { s.y = H + 8; s.x = Math.random() * W; }
          if (s.x < -8) s.x = W + 8;
          if (s.x > W + 8) s.x = -8;
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [reduced, theme]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0"
      style={{ opacity: 0.95 }}
    />
  );
}
