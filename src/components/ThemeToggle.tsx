import { useTheme } from '../context/ThemeContext';
import Icon from './Icon';

export default function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const dark = theme === 'dark';
  return (
    <button onClick={toggle} className="btn btn-ghost !px-3 !py-2" aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`} title="Switch theme">
      <Icon name={dark ? 'sun' : 'moon'} size={18} />
      <span className="hidden text-xs sm:inline">{dark ? 'Light' : 'Dark'}</span>
    </button>
  );
}
