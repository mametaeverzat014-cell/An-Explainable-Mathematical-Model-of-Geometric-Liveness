import type { Outcome } from '@student';
import type { PairAnalysis } from '../core/method';
import { badge, h } from './dom';

// Outcome display. Wording is deliberately about PLANARITY, not liveness:
// rejecting planarity can also be caused by expressions, a curved print or a
// replayed video, and failing to reject it is not proof of an attack.

const ICON: Record<Outcome['kind'], string> = {
  'planar-consistent': '▭',
  'non-planar': '◇',
  inconclusive: '?',
};

const TITLE: Record<Outcome['kind'], string> = {
  'planar-consistent': 'Consistent with a planar surface',
  'non-planar': 'Evidence against a planar surface',
  inconclusive: 'Inconclusive',
};

export function outcomeView(a: PairAnalysis | null, emptyText: string): HTMLElement {
  if (!a) {
    return h('div', { class: 'outcome' }, h('span', { class: 'outcome-icon', 'aria-hidden': 'true' }, '…'), h('div', {}, h('div', { class: 'outcome-title' }, 'No analysis'), h('div', { class: 'outcome-reason' }, emptyText)));
  }
  const decision = a.stages.find((s) => s.stage === 'decision');
  if (!a.outcome) {
    const missing = a.stages.filter((s) => s.state === 'not-implemented').map((s) => s.stage);
    const errors = a.stages.filter((s) => s.state === 'error');
    const text = missing.length
      ? `Method stages not implemented yet: ${missing.join(', ')} (see web/docs/MATH_SPEC.md).`
      : errors.length
        ? `Error in ${errors[0].stage}: ${errors[0].message}`
        : (decision?.message ?? 'No decision.');
    return h(
      'div',
      { class: 'outcome' },
      h('span', { class: 'outcome-icon', 'aria-hidden': 'true' }, '○'),
      h('div', {}, h('div', { class: 'outcome-title' }, 'No decision'), h('div', { class: 'outcome-reason' }, text)),
      badge(missing.length ? 'not-implemented' : 'method-output'),
    );
  }
  return h(
    'div',
    { class: 'outcome', role: 'status' },
    h('span', { class: 'outcome-icon', 'aria-hidden': 'true' }, ICON[a.outcome.kind]),
    h('div', {}, h('div', { class: 'outcome-title' }, TITLE[a.outcome.kind]), h('div', { class: 'outcome-reason' }, a.outcome.reason)),
    badge('method-output'),
  );
}
