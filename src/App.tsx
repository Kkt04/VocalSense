import { useEffect, useState } from 'react';
import Landing from './pages/Landing';
import Conversation from './pages/Conversation';

/** Minimal hash router (#/ and #/talk) — avoids an extra dependency in Phase 1. */
function useRoute() {
  const read = () => (window.location.hash.replace('#', '') || '/');
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => { setRoute(read()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

export default function App() {
  const route = useRoute();
  return route.startsWith('/talk') ? <Conversation /> : <Landing />;
}
