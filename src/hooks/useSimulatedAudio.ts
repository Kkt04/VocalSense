import { useEffect, useState } from 'react';

/** PHASE 1 MOCK: generates fake amplitude data. Replaced by the Web Audio API AnalyserNode in Phase 2. */
export function useSimulatedAudio(active: boolean, n = 36) {
  const quiet = () => Array<number>(n).fill(0.06);
  const [bars, setBars] = useState<number[]>(quiet);
  useEffect(() => {
    if (!active) { setBars(quiet()); return; }
    let t = 0;
    const id = setInterval(() => {
      t += 0.35;
      const env = 0.5 + 0.5 * Math.sin(t * 0.6);
      setBars(Array.from({ length: n }, (_, i) =>
        Math.min(1, Math.max(0.06, 0.15 + 0.55 * Math.abs(Math.sin(t + i * 0.45)) * env + Math.random() * 0.2))));
    }, 90);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, n]);
  const level = bars.reduce((a, b) => a + b, 0) / n;
  return { bars, level: active ? level : 0 };
}
