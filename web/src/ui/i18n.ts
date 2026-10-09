import { en, type MessageKey } from './locales/en';
import { kk } from './locales/kk';
import { ru } from './locales/ru';

export type Lang = 'ru' | 'kk' | 'en';

export const LANGS: { id: Lang; label: string }[] = [
  { id: 'ru', label: 'RU' },
  { id: 'kk', label: 'KZ' },
  { id: 'en', label: 'EN' },
];

const DICTS: Record<Lang, Record<MessageKey, string>> = { en, ru, kk };
const STORAGE_KEY = 'parallax-lab-lang';

function detect(): Lang {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'ru' || stored === 'kk' || stored === 'en') return stored;
  } catch {
    /* storage unavailable */
  }
  const nav = (navigator.language || '').toLowerCase();
  if (nav.startsWith('kk')) return 'kk';
  if (nav.startsWith('en')) return 'en';
  return 'ru';
}

let current: Lang = typeof navigator === 'undefined' ? 'ru' : detect();

export function getLang(): Lang {
  return current;
}

export function setLang(lang: Lang): void {
  current = lang;
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    /* storage unavailable: the choice still applies to this page */
  }
  if (typeof document !== 'undefined') document.documentElement.lang = lang;
}

/** Translate a key, replacing {name} placeholders. */
export function t(key: MessageKey, params: Record<string, string | number> = {}): string {
  const template = DICTS[current][key] ?? en[key];
  return template.replace(/\{(\w+)\}/g, (_, name: string) => (name in params ? String(params[name]) : `{${name}}`));
}

export type { MessageKey };
