import { LANDMARK_SETS, type LandmarkSetId } from '../../core/landmark-sets';
import { parseRecordingsFile, type Recording } from '../../core/recording';
import { analyzeSet, DEFAULT_SET_SETTINGS, rowsToCsv, type SetAnalysis, type SetAnalysisSettings } from '../../core/recording-analysis';
import { card, clear, field, fmt, h, note, numberInput, row, select, table } from '../dom';
import { t, type MessageKey } from '../i18n';
import { Plot, type Series } from '../plot';
import { downloadJson, downloadText, tabRecordings } from '../recorder';

/** Recordings handed over by the Experiment view ("Open in Analyze"). */
export const handoff: { recordings: Recording[] } = { recordings: [] };

const SERIES_STYLE = [
  { colorVar: '--series-1', marker: 'circle' as const },
  { colorVar: '--series-2', marker: 'square' as const },
  { colorVar: '--series-3', marker: 'triangle' as const },
];

export function analyzeView(): { element: HTMLElement; dispose: () => void } {
  const loaded: Recording[] = [];
  const settings: SetAnalysisSettings = { ...DEFAULT_SET_SETTINGS };
  const filesSlot = h('div');
  const errorsSlot = h('div');
  const resultSlot = h('div', { class: 'stack' });
  const plot = new Plot(t('ana.chartTitle'), 240);

  if (handoff.recordings.length) {
    loaded.push(...handoff.recordings);
    handoff.recordings = [];
  }

  // ---------------- loading ----------------
  const fileInput = h('input', { type: 'file', accept: '.json,application/json', multiple: true, class: 'sr-only', id: 'ana-files' }) as HTMLInputElement;
  const chooseBtn = h('label', { class: 'btn btn-secondary', for: 'ana-files', role: 'button', tabindex: 0 }, t('ana.choose'));
  chooseBtn.addEventListener('keydown', (e) => {
    if ((e as KeyboardEvent).key === 'Enter' || (e as KeyboardEvent).key === ' ') fileInput.click();
  });
  fileInput.addEventListener('change', () => void loadFiles(fileInput.files));
  const drop = h('div', { class: 'dropzone' }, h('p', {}, t('ana.drop')), chooseBtn, fileInput);
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('dropzone-active');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('dropzone-active'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dropzone-active');
    void loadFiles((e as DragEvent).dataTransfer?.files ?? null);
  });

  const useTabBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('ana.useTab', { n: tabRecordings.length }));
  useTabBtn.hidden = tabRecordings.length === 0;
  useTabBtn.addEventListener('click', () => {
    for (const r of tabRecordings) if (!loaded.some((x) => x.meta.id === r.meta.id)) loaded.push(r);
    renderFiles();
  });
  const clearBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('ana.clear'));
  clearBtn.addEventListener('click', () => {
    loaded.length = 0;
    clear(errorsSlot);
    clear(resultSlot);
    renderFiles();
  });

  async function loadFiles(files: FileList | null): Promise<void> {
    if (!files) return;
    clear(errorsSlot);
    for (const file of Array.from(files)) {
      try {
        const recs = parseRecordingsFile(await file.text());
        for (const r of recs) if (!loaded.some((x) => x.meta.id === r.meta.id)) loaded.push(r);
      } catch (e) {
        errorsSlot.append(note(t('ana.loadError', { file: file.name, message: e instanceof Error ? e.message : String(e) }), 'warn'));
      }
    }
    fileInput.value = '';
    renderFiles();
  }

  function renderFiles(): void {
    clear(filesSlot);
    if (!loaded.length) {
      filesSlot.append(h('p', { class: 'outcome-reason' }, t('ana.noFiles')));
      runBtn.disabled = true;
      return;
    }
    runBtn.disabled = false;
    filesSlot.append(
      h(
        'div',
        { class: 'table-wrap' },
        table(
          [t('ana.colRole'), t('ana.colCondition'), t('ana.colFrames'), t('ana.colSubject'), t('ana.colId')],
          loaded.map((r) => [
            r.meta.role === 'calibration' ? t('ana.roleCalibration') : t('ana.roleTrial'),
            r.meta.condition,
            String(r.frames.length),
            r.meta.category === 'human-participant' ? (r.meta.pseudonym ?? '') : '—',
            h('code', {}, r.meta.id),
          ]),
        ),
      ),
    );
  }

  // ---------------- settings ----------------
  const runBtn = h('button', { class: 'btn', type: 'button', disabled: true }, t('ana.run')) as HTMLButtonElement;
  runBtn.addEventListener('click', () => render(analyzeSet(loaded, settings)));
  const settingsCard = card(
    t('ana.cardSettings'),
    h(
      'div',
      { class: 'controls' },
      field(
        t('live.landmarkSet'),
        select(
          (Object.keys(LANDMARK_SETS) as LandmarkSetId[]).map((k) => ({ value: k, label: t(`set.${k}.label` as MessageKey) })),
          settings.landmarkSet,
          (v) => (settings.landmarkSet = v),
        ),
      ),
      field(t('live.window'), numberInput(settings.windowMs, (v) => (settings.windowMs = v), { min: 200, step: 100 })),
      field(t('live.minMotion'), numberInput(settings.minMotionDeg, (v) => (settings.minMotionDeg = v), { min: 0, step: 0.5 })),
      field('α', numberInput(settings.alpha, (v) => (settings.alpha = v), { min: 0.001, max: 0.5, step: 0.01 })),
      field(t('ana.gap'), numberInput(settings.calibrationGap, (v) => (settings.calibrationGap = Math.max(1, Math.round(v))), { min: 1, step: 1 })),
      field(t('ana.maxCal'), numberInput(settings.maxCalibrationMotionDeg, (v) => (settings.maxCalibrationMotionDeg = v), { min: 0, step: 0.5 })),
      field(t('ana.fixedSigma'), numberInput(0, (v) => (settings.fixedSigmaPx = v > 0 ? v : null), { min: 0, step: 0.05 })),
    ),
    note(t('ana.settingsNote'), 'warn'),
    h('div', { class: 'btn-row' }, runBtn),
  );

  // ---------------- results ----------------
  function render(a: SetAnalysis): void {
    clear(resultSlot);
    if (!a.ok) {
      resultSlot.append(note(t('ana.error', { error: a.error }), 'warn'));
      return;
    }
    const c = a.calibration;
    resultSlot.append(
      card(
        t('ana.cardCalibration'),
        row('σ', `${fmt(c.sigmaPx, 3)} px`, 'method-output'),
        row(t('live.rowInterval'), c.interval95 ? `${fmt(c.interval95[0], 3)} – ${fmt(c.interval95[1], 3)}` : '—', 'method-output'),
        row('', c.source === 'fixed' ? t('ana.sigmaFixed') : t('ana.sigmaFrom', { n: c.calibrationRecordings })),
        c.rejectedForMotion.length ? note(t('ana.rejectedCal', { ids: c.rejectedForMotion.join(', ') }), 'warn') : null,
      ),
    );

    const ci = (v: [number, number] | null) => (v ? `[${fmt(v[0], 3)}, ${fmt(v[1], 3)}]` : '—');
    resultSlot.append(
      card(
        t('ana.cardConditions'),
        h(
          'div',
          { class: 'table-wrap' },
          table(
            [t('ana.colCondition'), t('ana.colRecordings'), t('ana.colTested'), t('ana.colRejected'), t('ana.colMeanRate'), t('ana.colPooled')],
            a.conditions.map((k) => [
              k.condition,
              String(k.recordings),
              String(k.tested),
              String(k.rejected),
              `${fmt(k.meanRecordingRate, 3)} ${ci(k.meanRateCi95)}`,
              `${fmt(k.pooledRate, 3)} ${ci(k.pooledCi95)}`,
            ]),
          ),
        ),
        note(t('ana.condNote')),
        plot.element,
      ),
    );

    // Strip plot: one point per recording, grouped by condition on the x axis.
    const conds = a.conditions.map((k) => k.condition);
    const series: Series[] = a.conditions.slice(0, 3).map((k, i) => ({
      label: k.condition,
      ...SERIES_STYLE[i],
      line: false,
      points: k.recordingRates.map((r, j) => ({ x: i + (k.recordingRates.length > 1 ? (j / (k.recordingRates.length - 1) - 0.5) * 0.3 : 0), y: r })),
    }));
    const means: Series = {
      label: t('ana.colMeanRate'),
      colorVar: '--text-secondary',
      marker: 'square',
      line: false,
      points: a.conditions.slice(0, 3).flatMap((k, i) => (k.meanRecordingRate === null ? [] : [{ x: i + 0.25, y: k.meanRecordingRate, lo: k.meanRateCi95?.[0], hi: k.meanRateCi95?.[1] }])),
    };
    plot.render({
      xLabel: t('ana.chartX'),
      yLabel: t('ana.chartY'),
      xDomain: [-0.5, Math.min(3, conds.length) - 0.5],
      yDomain: [0, 1],
      formatX: (v) => (Number.isInteger(v) && conds[v] ? conds[v] : ''),
      formatY: (v) => v.toFixed(2),
      series: [...series, means],
      refLines: [{ y: a.settings.alpha, label: `α = ${a.settings.alpha}` }],
    });

    resultSlot.append(
      card(
        t('ana.cardRecordings'),
        h(
          'div',
          { class: 'table-wrap' },
          table(
            [t('ana.colId'), t('ana.colCondition'), t('ana.colFrames'), t('ana.colPairs'), t('ana.colTested'), t('ana.colRejected'), t('ana.colRate')],
            a.summaries.map((s) => [
              h('code', {}, s.recordingId),
              s.condition,
              String(s.frames),
              String(s.pairs),
              String(s.tested),
              String(s.rejected),
              s.blockedStage ? s.blockedStage : fmt(s.rejectionRate, 3),
            ]),
          ),
        ),
        h(
          'div',
          { class: 'btn-row', style: 'margin-top:8px' },
          button(t('ana.downloadCsv'), () => downloadText(rowsToCsv(a.rows), 'pairs.csv', 'text/csv')),
          button(t('ana.downloadSummary'), () =>
            downloadJson({ settings: a.settings, calibration: a.calibration, conditions: a.conditions, recordings: a.summaries }, 'summary.json'),
          ),
        ),
      ),
    );
  }

  const element = h(
    'div',
    {},
    h('h1', {}, t('ana.title')),
    h('p', { class: 'page-intro' }, t('ana.intro')),
    h(
      'div',
      { class: 'grid-live' },
      h('div', { class: 'stack' }, card(t('ana.cardFiles'), drop, h('div', { class: 'btn-row', style: 'margin:8px 0' }, useTabBtn, clearBtn), errorsSlot, filesSlot), resultSlot),
      h('div', { class: 'stack' }, settingsCard),
    ),
  );
  renderFiles();
  return { element, dispose: () => {} };
}

function button(label: string, onClick: () => void): HTMLButtonElement {
  const b = h('button', { class: 'btn btn-secondary', type: 'button' }, label) as HTMLButtonElement;
  b.addEventListener('click', onClick);
  return b;
}
