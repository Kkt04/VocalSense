export default function Waveform({ bars, active }: { bars: number[]; active: boolean }) {
  return (
    <div className="flex h-14 items-center justify-center gap-[3px]" role="img"
      aria-label={active ? 'Live audio waveform (simulated in this prototype)' : 'Audio waveform, currently quiet'}>
      {bars.map((b, i) => (
        <span key={i} className={`w-[3px] rounded-full transition-all duration-100 ${active ? 'bg-accent' : 'bg-ink/20'}`}
          style={{ height: `${Math.round(b * 100)}%` }} />
      ))}
    </div>
  );
}
