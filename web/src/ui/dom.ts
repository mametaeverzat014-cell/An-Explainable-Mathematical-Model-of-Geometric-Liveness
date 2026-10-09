import type { Provenance } from '../core/provenance';
import { t, type MessageKey } from './i18n';

type Attrs = Record<string, string | number | boolean | undefined | null | EventListener>;
type Child = Node | string | number | null | undefined | false;

/** Minimal element factory: h('div', { class: 'card' }, 'text', child). */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
    } else if (value === true) {
      el.setAttribute(key, '');
    } else {
      el.setAttribute(key, String(value));
    }
  }
  append(el, children);
  return el;
}

export function append(el: Element, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function badge(p: Provenance): HTMLElement {
  return h('span', { class: `badge badge-${p}`, title: t(`provHelp.${p}` as MessageKey) }, t(`prov.${p}` as MessageKey));
}

/** Format a number for display; em dash for missing values. */
export function fmt(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(Math.max(1, digits - 1));
  return value.toFixed(digits);
}

/** Format a number with a unit, or an em dash (without unit) when missing. */
export function fmtUnit(value: number | null | undefined, unit: string, digits = 2): string {
  const s = fmt(value, digits);
  return s === '—' ? s : `${s} ${unit}`;
}

export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return '—';
  if (p < 1e-4) return '< 0.0001';
  return p.toFixed(4);
}

/** A labelled value row: label · value · provenance badge. */
export function row(label: string, value: string | Node, provenance?: Provenance, hint?: string): HTMLElement {
  return h(
    'div',
    { class: 'kv' },
    h('span', { class: 'kv-label', title: hint }, label),
    h('span', { class: 'kv-value' }, value),
    provenance ? badge(provenance) : h('span'),
  );
}

export function card(title: string, ...children: Child[]): HTMLElement {
  return h('section', { class: 'card' }, h('h2', { class: 'card-title' }, title), ...children);
}

export function note(text: string | Node, kind: 'info' | 'warn' = 'info'): HTMLElement {
  return h('p', { class: `note note-${kind}` }, text);
}

export function select<T extends string>(
  options: readonly { value: T; label: string }[],
  value: T,
  onChange: (v: T) => void,
  attrs: Attrs = {},
): HTMLSelectElement {
  const el = h('select', attrs) as HTMLSelectElement;
  for (const o of options) el.append(h('option', { value: o.value, selected: o.value === value }, o.label));
  el.addEventListener('change', () => onChange(el.value as T));
  return el;
}

export function numberInput(value: number, onChange: (v: number) => void, attrs: Attrs = {}): HTMLInputElement {
  const el = h('input', { type: 'number', value, ...attrs }) as HTMLInputElement;
  el.addEventListener('change', () => {
    const v = Number(el.value);
    if (Number.isFinite(v)) onChange(v);
  });
  return el;
}

export function rangeInput(value: number, min: number, max: number, step: number, onInput: (v: number) => void, label: string): HTMLElement {
  const out = h('output', {}, String(value));
  const input = h('input', { type: 'range', min, max, step, value, 'aria-label': label }) as HTMLInputElement;
  input.addEventListener('input', () => {
    out.textContent = input.value;
    onInput(Number(input.value));
  });
  return h('label', { class: 'field field-range' }, h('span', {}, label), input, out);
}

export function field(label: string, control: Node, hint?: string): HTMLElement {
  return h('label', { class: 'field' }, h('span', {}, label), control, hint ? h('small', {}, hint) : null);
}

export function table(headers: string[], rows: (string | Node)[][], caption?: string): HTMLTableElement {
  return h(
    'table',
    { class: 'data-table' },
    caption ? h('caption', {}, caption) : null,
    h('thead', {}, h('tr', {}, ...headers.map((t) => h('th', { scope: 'col' }, t)))),
    h('tbody', {}, ...rows.map((r) => h('tr', {}, ...r.map((c) => h('td', {}, c))))),
  );
}
