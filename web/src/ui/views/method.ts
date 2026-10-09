import { CONFIG_NOTES, DEFAULT_CONFIG } from '../../core/config';
import { probeMethod } from '../../core/method';
import type { Provenance } from '../../core/provenance';
import { badge, card, h, note, table } from '../dom';
import { t, type MessageKey } from '../i18n';

const PROVENANCES: Provenance[] = ['measured', 'mediapipe-estimate', 'method-output', 'synthetic-truth', 'configuration', 'not-implemented'];
const ASSUMPTION_IDS = [1, 2, 3, 4, 5, 6, 7, 8] as const;

export function methodView(): { element: HTMLElement; dispose: () => void } {
  const modules = probeMethod();
  const implemented = modules.filter((m) => m.implemented).length;

  const statusCard = card(
    t('meth.cardStatus'),
    h('p', { class: 'outcome-reason' }, t('meth.statusLine', { done: implemented, total: modules.length })),
    h(
      'div',
      { class: 'table-wrap' },
      table(
        [t('meth.colTask'), t('meth.colFunction'), t('meth.colStatus')],
        modules.map((m) => [
          m.task,
          h('code', {}, m.name),
          h('span', { class: m.implemented ? 'status-ok' : 'status-missing' }, m.implemented ? t('meth.implemented') : t('meth.notImplemented')),
        ]),
      ),
    ),
    note(t('meth.statusNote'), 'info'),
  );

  const findingsCard = card(t('meth.cardFindings'), h('p', {}, t('meth.noneYet')), h('p', { class: 'outcome-reason' }, t('meth.findingsNote')));

  const provenanceCard = card(
    t('meth.cardLabels'),
    ...PROVENANCES.map((p) => h('div', { class: 'label-row' }, badge(p), h('span', { class: 'kv-label' }, t(`provHelp.${p}` as MessageKey)))),
  );

  const outcomesCard = card(
    t('meth.cardOutcomes'),
    table(
      [t('meth.colOutcome'), t('meth.colMeaning'), t('meth.colNot')],
      [
        [t('out.planar.title'), t('meth.o1.meaning'), t('meth.o1.not')],
        [t('out.nonplanar.title'), t('meth.o2.meaning'), t('meth.o2.not')],
        [t('out.inconclusive.title'), t('meth.o3.meaning'), t('meth.o3.not')],
      ],
    ),
  );

  const configCard = card(
    t('meth.cardConfig'),
    h(
      'div',
      { class: 'table-wrap' },
      table(
        [t('meth.colSetting'), t('meth.colDefault'), t('meth.colStatus'), t('meth.colJustification')],
        (Object.keys(CONFIG_NOTES) as (keyof typeof CONFIG_NOTES)[]).map((k) => [
          h('code', {}, k),
          String(DEFAULT_CONFIG[k]),
          t(`cfg.${CONFIG_NOTES[k].status}` as MessageKey),
          t(`cfg.${k}` as MessageKey),
        ]),
      ),
    ),
  );

  const assumptionsCard = card(
    t('meth.cardAssumptions'),
    h('p', { class: 'outcome-reason' }, t('meth.assumptionsIntro')),
    h(
      'div',
      { class: 'table-wrap' },
      table(
        ['', t('meth.colAssumption'), t('meth.colWhy'), t('meth.colStatus')],
        ASSUMPTION_IDS.map((i) => [`A${i}`, t(`meth.a${i}.s` as MessageKey), t(`meth.a${i}.w` as MessageKey), t(`meth.a${i}.t` as MessageKey)]),
      ),
    ),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, t('meth.title')),
    h('p', { class: 'page-intro' }, t('meth.intro')),
    h('div', { class: 'grid-2' }, h('div', { class: 'stack' }, statusCard, findingsCard, outcomesCard), h('div', { class: 'stack' }, provenanceCard, configCard)),
    h('div', { style: 'margin-top:16px' }, assumptionsCard),
  );
  return { element, dispose: () => {} };
}
