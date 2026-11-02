const P: Record<string, JSX.Element> = {
  mic: <><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" /><path d="M19 11a7 7 0 0 1-14 0M12 18v3" /></>,
  micoff: <><path d="M3 3l18 18" /><path d="M9 9v3a3 3 0 0 0 5 2.2M15 9.3V6a3 3 0 0 0-5.7-1.3M19 11a7 7 0 0 1-1 3.6M5 11a7 7 0 0 0 11 5.7M12 18v3" /></>,
  pause: <path d="M8 5v14M16 5v14" />,
  play: <path d="M7 4.5v15l12-7.5-12-7.5Z" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z" />,
  speaker: <><path d="M4 9v6h4l5 4V5L8 9H4Z" /><path d="M17 9a4 4 0 0 1 0 6" /></>,
  speakeroff: <><path d="M4 9v6h4l5 4V5L8 9H4Z" /><path d="M17 9l5 6M22 9l-5 6" /></>,
  replay: <path d="M4 12a8 8 0 1 0 3-6.2M4 4v5h5" />,
  edit: <path d="M4 20h4L19 9l-4-4L4 16v4Z" />,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  shield: <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3Z" />,
  back: <path d="M19 12H5M11 6l-6 6 6 6" />,
};
export default function Icon({ name, size = 20 }: { name: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{P[name]}</svg>
  );
}
