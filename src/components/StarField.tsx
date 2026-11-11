import { useEffect, useRef } from 'react';

interface Star {
  x: number;
  y: number;
  r: number;
  alpha: number;
  speed: number;
  drift: number;
  driftDir: number;
  phase: number;
}

const COUNT = 90;

function mkStar(W: number, H: number): Star {
  return {
    x: Math.random() * W,
    y: Math.random() * H,
    r: Math.random() * 1.1 + 0.3,
    alpha: Math.random() * 0.5 + 0.15,
    speed: Math.random() * 0.18 + 0.04,
    drift: Math.random() * 0.3 + 0.05,
    driftDir: Math.random() > 0.5 ? 1 : -1,
    phase: Math.random() * Math.PI * 2,
  };
}

/** Renders a full-page canvas of gently twinkling, drifting stars.
 *  Respects prefers-reduced-motion — static dots only. */
export default function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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
      for (let i = 0; i < COUNT; i++) stars.push(mkStar(W, H));
    };

    resize();
    window.addEventListener('resize', resize, { passive: true });

    let t = 0;
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      t += 0.012;

      for (const s of stars) {
        // gentle twinkle via sin wave
        const a = s.alpha * (0.55 + 0.45 * Math.sin(t * s.speed * 6 + s.phase));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);

        // white in dark mode, soft ink in light
        const isDark = document.documentElement.classList.contains('dark');
        ctx.fillStyle = isDark
          ? `rgba(255,255,255,${a})`
          : `rgba(100,83,132,${a * 0.45})`;
        ctx.fill();

        if (!reduced) {
          // gentle float upward + horizontal drift
          s.y -= s.speed;
          s.x += s.drift * s.driftDir * Math.sin(t + s.phase);
          if (s.y < -4) { s.y = H + 4; s.x = Math.random() * W; }
          if (s.x < -4) s.x = W + 4;
          if (s.x > W + 4) s.x = -4;
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [reduced]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0"
      style={{ opacity: 0.85 }}
    />
  );
}
