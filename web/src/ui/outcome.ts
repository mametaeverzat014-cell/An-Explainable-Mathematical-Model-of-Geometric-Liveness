import type { OutcomeKind } from '@method';
import type { PairAnalysis } from '../core/method';
import { badge, h } from './dom';
import { t, type MessageKey } from './i18n';

// Outcome display. Wording is deliberately about PLANARITY, not liveness:
// rejecting planarity can also be caused by expressions, a curved print or a
// replayed video, and failing to reject it is not proof of an attack.
//
// The method's own `reason` string is English; the text shown here is built
// in the interface language from the same numbers and the same rule order
// as `decide` (statistic missing -> motion gate -> p-value).

const ICON: Record<OutcomeKind, string> = {
  'planar-consistent': '▭',
  'non-planar': '◇',
  inconclusive: '?',
};

const TITLE: Record<OutcomeKind, MessageKey> = {
  'planar-consistent': 'out.planar.title',
  'non-planar': 'out.nonplanar.title',
  inconclusive: 'out.inconclusive.title',
};

export function formatP(p: number): string {
  return p < 1e-4 ? p.toExponential(1) : p.toFixed(4);
}

export function outcomeReason(a: PairAnalysis): string {
  if (!a.stat) return t('out.reason.noStat');
  if (a.motionDeg < a.minMotionDeg) return t('out.reason.motion', { motion: a.motionDeg.toFixed(1), min: a.minMotionDeg });
  const p = formatP(a.stat.pValue);
  return a.stat.pValue < a.alpha ? t('out.reason.nonplanar', { p, alpha: a.alpha }) : t('out.reason.planar', { p, alpha: a.alpha });
}

export function outcomeView(a: PairAnalysis | null, emptyText: string): HTMLElement {
  if (!a) {
    return h(
      'div',
      { class: 'outcome' },
      h('span', { class: 'outcome-icon', 'aria-hidden': 'true' }, '…'),
      h('div', {}, h('div', { class: 'outcome-title' }, t('out.none.title')), h('div', { class: 'outcome-reason' }, emptyText)),
    );
  }
  if (!a.outcome) {
    const missing = a.stages.filter((s) => s.state === 'not-implemented').map((s) => s.stage);
    const errors = a.stages.filter((s) => s.state === 'error');
    const text = missing.length
      ? t('out.reason.missing', { stages: missing.join(', ') })
      : errors.length
        ? t('out.reason.error', { stage: errors[0].stage, message: errors[0].message ?? '' })
        : t('out.reason.noStat');
    return h(
      'div',
      { class: 'outcome' },
      h('span', { class: 'outcome-icon', 'aria-hidden': 'true' }, '○'),
      h('div', {}, h('div', { class: 'outcome-title' }, t('out.nodecision.title')), h('div', { class: 'outcome-reason' }, text)),
      badge(missing.length ? 'not-implemented' : 'method-output'),
    );
  }
  return h(
    'div',
    { class: 'outcome', role: 'status' },
    h('span', { class: 'outcome-icon', 'aria-hidden': 'true' }, ICON[a.outcome.kind]),
    h('div', {}, h('div', { class: 'outcome-title' }, t(TITLE[a.outcome.kind])), h('div', { class: 'outcome-reason' }, outcomeReason(a))),
    badge('method-output'),
  );
}
