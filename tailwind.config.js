const c = (v) => `rgb(var(--${v}) / <alpha-value>)`;
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: { bg: c('bg'), surface: c('surface'), lav: c('lav'), deep: c('deep'), accent: c('accent'),
        onaccent: c('onaccent'), pink: c('pink'), sage: c('sage'), peach: c('peach'), ink: c('ink'), muted: c('muted') },
      fontFamily: { serif: ['"DM Serif Display"', 'Georgia', 'serif'], sans: ['"DM Sans"', 'Inter', 'system-ui', 'sans-serif'] },
      boxShadow: { soft: '0 10px 40px -18px rgb(var(--ink) / .22)', orb: '0 30px 80px -24px rgb(var(--deep) / .55)' },
    },
  },
  plugins: [],
};
