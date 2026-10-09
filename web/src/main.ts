import './styles.css';
import { h } from './ui/dom';
import { getLang, LANGS, setLang, t, type Lang, type MessageKey } from './ui/i18n';
import { session } from './ui/session';
import { aboutView } from './ui/views/about';
import { analyzeView } from './ui/views/analyze';
import { diagnosticsView } from './ui/views/diagnostics';
import { experimentView } from './ui/views/experiment';
import { liveView } from './ui/views/live';
import { methodView } from './ui/views/method';
import { recordView } from './ui/views/record';
import { syntheticView } from './ui/views/synthetic';

type View = { element: HTMLElement; dispose: () => void };

const ROUTES: { id: string; label: MessageKey; make: () => View }[] = [
  { id: 'live', label: 'nav.live', make: liveView },
  { id: 'experiment', label: 'nav.experiment', make: experimentView },
  { id: 'record', label: 'nav.record', make: recordView },
  { id: 'analyze', label: 'nav.analyze', make: analyzeView },
  { id: 'synthetic', label: 'nav.synthetic', make: syntheticView },
  { id: 'diagnostics', label: 'nav.diagnostics', make: diagnosticsView },
  { id: 'method', label: 'nav.method', make: methodView },
  { id: 'about', label: 'nav.about', make: aboutView },
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
  setLang(getLang());
  const header = h('header', { class: 'app-header' });
  const main = h('main', { id: 'main' });
  document.body.append(header, main);

  let current: View | null = null;
  let links: HTMLAnchorElement[] = [];

  function renderHeader(): void {
    const nav = h('nav', { class: 'tabs', 'aria-label': 'Sections' });
    links = ROUTES.map((r) => {
      const a = h('a', { href: `#${r.id}` }, t(r.label)) as HTMLAnchorElement;
      nav.append(a);
      return a;
    });
    const themeBtn = h('button', { class: 'theme-toggle', type: 'button', 'aria-label': t('shell.themeAria') }, t('shell.theme'));
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
    const langSelect = h('select', { class: 'lang-select', 'aria-label': t('shell.language') }) as HTMLSelectElement;
    for (const l of LANGS) langSelect.append(h('option', { value: l.id, selected: l.id === getLang() }, l.label));
    langSelect.addEventListener('change', () => {
      setLang(langSelect.value as Lang);
      renderHeader();
      route();
    });
    header.replaceChildren(
      h('div', { class: 'app-header-inner' }, h('div', { class: 'brand' }, 'Parallax Lab', h('small', {}, t('app.tagline'))), nav, langSelect, themeBtn),
    );
  }

  function route(): void {
    const id = location.hash.replace('#', '') || 'live';
    const r = ROUTES.find((x) => x.id === id) ?? ROUTES[0];
    links.forEach((a, i) => (ROUTES[i] === r ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
    current?.dispose();
    current = r.make();
    main.replaceChildren(current.element);
    document.title = `${t(r.label)} · Parallax Lab`;
  }

  renderHeader();
  window.addEventListener('hashchange', route);
  window.addEventListener('pagehide', () => session.stop());
  route();
}

boot();
