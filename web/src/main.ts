import './styles.css';
import { h } from './ui/dom';
import { session } from './ui/session';
import { aboutView } from './ui/views/about';
import { diagnosticsView } from './ui/views/diagnostics';
import { liveView } from './ui/views/live';
import { methodView } from './ui/views/method';
import { recordView } from './ui/views/record';
import { syntheticView } from './ui/views/synthetic';

type View = { element: HTMLElement; dispose: () => void };

const ROUTES: { id: string; label: string; make: () => View }[] = [
  { id: 'live', label: 'Live', make: liveView },
  { id: 'record', label: 'Record', make: recordView },
  { id: 'synthetic', label: 'Synthetic lab', make: syntheticView },
  { id: 'diagnostics', label: 'Diagnostics', make: diagnosticsView },
  { id: 'method', label: 'Method & status', make: methodView },
  { id: 'about', label: 'About', make: aboutView },
];

const THEME_KEY = 'parallax-lab-theme';

function readTheme(): string | null {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch {
    return null;
  }
}

function applyTheme(theme: string | null): void {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function boot(): void {
  applyTheme(readTheme());
  const nav = h('nav', { class: 'tabs', 'aria-label': 'Sections' });
  const links = ROUTES.map((r) => {
    const a = h('a', { href: `#${r.id}` }, r.label);
    nav.append(a);
    return a;
  });
  const themeBtn = h('button', { class: 'theme-toggle', type: 'button', 'aria-label': 'Toggle colour theme' }, 'Theme');
  themeBtn.addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    applyTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* storage unavailable: theme still applies for this page */
    }
    window.dispatchEvent(new Event('themechange')); // canvases redraw with the new colours
  });
  const header = h(
    'header',
    { class: 'app-header' },
    h('div', { class: 'app-header-inner' }, h('div', { class: 'brand' }, 'Parallax Lab', h('small', {}, 'Planarity test for face presentations · research instrument')), nav, themeBtn),
  );
  const main = h('main', { id: 'main' });
  document.body.append(header, main);

  let current: View | null = null;
  function route(): void {
    const id = location.hash.replace('#', '') || 'live';
    const r = ROUTES.find((x) => x.id === id) ?? ROUTES[0];
    links.forEach((a, i) => (ROUTES[i] === r ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
    current?.dispose();
    current = r.make();
    main.replaceChildren(current.element);
    document.title = `${r.label} · Parallax Lab`;
  }
  window.addEventListener('hashchange', route);
  window.addEventListener('pagehide', () => session.stop());
  route();
}

boot();
