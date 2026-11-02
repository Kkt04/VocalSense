export default function Logo() {
  return (
    <a href="#/" className="flex items-center gap-2.5 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
      <span className="h-7 w-7 rounded-full shadow-orb"
        style={{ background: 'radial-gradient(circle at 32% 28%, rgb(var(--peach)), rgb(var(--lav)) 55%, rgb(var(--deep)))' }} />
      <span className="font-serif text-xl">VocalSense</span>
    </a>
  );
}
