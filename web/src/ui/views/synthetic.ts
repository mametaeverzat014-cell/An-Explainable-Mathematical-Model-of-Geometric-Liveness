import { relativeRotationDeg, rotationFromAngles } from '../../core/geometry3d';
import { LANDMARK_SETS, type LandmarkSetId } from '../../core/landmark-sets';
import { analyzePair } from '../../core/method';
import { runSweep, type SweepPoint } from '../../core/montecarlo';
import { Rng } from '../../core/rng';
import { DEFAULT_INTRINSICS, makeObject, syntheticPair, type ObjectKind } from '../../core/synthetic';
import { card, clear, field, fmt, fmtP, h, note, numberInput, rangeInput, row, select, table } from '../dom';
import { t, type MessageKey } from '../i18n';
import { outcomeView } from '../outcome';
import { Plot } from '../plot';

const KIND_LABEL: Record<ObjectKind, () => string> = {
  face3d: () => t('syn.kind.face3d'),
  plane: () => t('syn.kind.plane'),
  cylinder: () => t('syn.kind.cylinder'),
};
const KIND_STYLE: Record<ObjectKind, { colorVar: string; marker: 'circle' | 'square' | 'triangle' }> = {
  face3d: { colorVar: '--series-1', marker: 'circle' },
  plane: { colorVar: '--series-2', marker: 'square' },
  cylinder: { colorVar: '--series-3', marker: 'triangle' },
};
const KINDS: ObjectKind[] = ['face3d', 'plane', 'cylinder'];
const VECTOR_GAIN = 10;

export function syntheticView(): { element: HTMLElement; dispose: () => void } {
  // ---------------- single pair ----------------
  const single = { kind: 'face3d' as ObjectKind, yaw: 8, pitch: 0, sigma: 1, distance: 50, set: 'rigid' as LandmarkSetId, seed: 1, alpha: 0.05 };
  const pairCanvas = h('canvas', { role: 'img', 'aria-label': 'Synthetic image plane: landmarks in two frames and residual vectors', style: 'width:100%;height:320px;display:block' }) as HTMLCanvasElement;
  const pairReadout = h('div');

  const pairControls = h(
    'div',
    { class: 'controls' },
    field(t('syn.object'), select(KINDS.map((k) => ({ value: k, label: KIND_LABEL[k]() })), single.kind, (v) => ((single.kind = v), renderPair()))),
    rangeInput(single.yaw, -25, 25, 0.5, (v) => ((single.yaw = v), renderPair()), t('syn.yaw')),
    rangeInput(single.pitch, -25, 25, 0.5, (v) => ((single.pitch = v), renderPair()), t('syn.pitch')),
    rangeInput(single.sigma, 0, 5, 0.1, (v) => ((single.sigma = v), renderPair()), t('syn.noise')),
    rangeInput(single.distance, 25, 200, 5, (v) => ((single.distance = v), renderPair()), t('syn.distance')),
    field(t('syn.landmarks'), select((Object.keys(LANDMARK_SETS) as LandmarkSetId[]).map((k) => ({ value: k, label: t(`set.${k}.label` as MessageKey) })), single.set, (v) => ((single.set = v), renderPair()))),
    field(t('syn.seed'), numberInput(single.seed, (v) => ((single.seed = Math.round(v)), renderPair()), { min: 0, step: 1 })),
  );

  function renderPair(): void {
    const indices = LANDMARK_SETS[single.set].indices;
    const corr = syntheticPair(
      {
        object: makeObject(single.kind),
        pose0: { yawDeg: 0, pitchDeg: 0 },
        pose1: { yawDeg: single.yaw, pitchDeg: single.pitch },
        distanceCm: single.distance,
        sigmaPx: single.sigma,
        indices,
      },
      new Rng(single.seed),
    );
    const motion = relativeRotationDeg(rotationFromAngles(0, 0), rotationFromAngles(single.yaw, single.pitch));
    const sigma = single.sigma > 0 ? single.sigma : null;
    const a = analyzePair(corr, { sigmaPx: sigma, alpha: single.alpha, minMotionDeg: 0, motionDeg: motion });

    // Draw: frame-0 points hollow, frame-1 points filled, residual vectors magnified.
    const dpr = window.devicePixelRatio || 1;
    const w = pairCanvas.clientWidth || 600;
    const hgt = pairCanvas.clientHeight || 320;
    pairCanvas.width = Math.round(w * dpr);
    pairCanvas.height = Math.round(hgt * dpr);
    const ctx = pairCanvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const style = getComputedStyle(element);
      ctx.fillStyle = style.getPropertyValue('--surface-2').trim();
      ctx.fillRect(0, 0, w, hgt);
      const all = [...corr.src, ...corr.dst];
      const xs = all.map((p) => p.x);
      const ys = all.map((p) => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const s = Math.min((w - 40) / (maxX - minX || 1), (hgt - 40) / (maxY - minY || 1));
      const ox = (w - s * (maxX - minX)) / 2 - s * minX;
      const oy = (hgt - s * (maxY - minY)) / 2 - s * minY;
      const P = (p: { x: number; y: number }) => [ox + s * p.x, oy + s * p.y] as const;
      const muted = style.getPropertyValue('--text-muted').trim();
      const c1 = style.getPropertyValue('--series-1').trim();
      const vec = style.getPropertyValue('--series-2').trim();
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1.5;
      corr.src.forEach((p) => {
        const [x, y] = P(p);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.stroke();
      });
      if (a.residualVectors) {
        ctx.strokeStyle = vec;
        ctx.lineWidth = 2;
        corr.dst.forEach((p, i) => {
          const [x, y] = P(p);
          const r = a.residualVectors![i];
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + s * VECTOR_GAIN * r.x, y + s * VECTOR_GAIN * r.y);
          ctx.stroke();
        });
      }
      ctx.fillStyle = c1;
      corr.dst.forEach((p) => {
        const [x, y] = P(p);
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    clear(pairReadout);
    pairReadout.append(
      outcomeView(a, ''),
      row(t('syn.rowRotation'), `${fmt(motion, 1)}°`, 'synthetic-truth'),
      row(t('syn.rowSigma'), `${fmt(single.sigma, 2)} px`, 'synthetic-truth'),
      row(t('syn.rowNdof'), `${a.n} / ${2 * a.n - 8}`, 'configuration'),
      row('T', fmt(a.stat?.T, 2), a.stat ? 'method-output' : 'not-implemented'),
      row(t('live.rowP'), fmtP(a.stat?.pValue), a.stat ? 'method-output' : 'not-implemented'),
    );
    if (single.sigma === 0) pairReadout.append(note(t('syn.sigmaZero'), 'warn'));
  }

  const pairCard = card(
    t('syn.cardPair'),
    h('p', { class: 'outcome-reason' }, t('syn.pairHelp')),
    pairControls,
    h('div', { class: 'grid-2' }, pairCanvas, pairReadout),
  );

  // ---------------- Monte-Carlo sweep ----------------
  const sweep = { axis: 'yaw' as 'yaw' | 'pitch', sigma: 1, distance: 50, set: 'rigid' as LandmarkSetId, trials: 200, alpha: 0.05, seed: 2024, rotations: '0, 1, 2, 3, 5, 8, 12' };
  const runBtn = h('button', { class: 'btn', type: 'button' }, t('syn.run')) as HTMLButtonElement;
  const stopBtn = h('button', { class: 'btn btn-secondary', type: 'button', disabled: true }, t('common.stop')) as HTMLButtonElement;
  const progress = h('progress', { max: 1, value: 0 }) as HTMLProgressElement;
  const status = h('p', { class: 'outcome-reason' }, t('common.notRun'));
  const ratePlot = new Plot(t('syn.cardSweep'), 260);
  const tableSlot = h('div', { class: 'table-wrap' });
  let stopRequested = false;

  const rotationsInput = h('input', { type: 'text', value: sweep.rotations, style: 'width:180px' }) as HTMLInputElement;
  rotationsInput.addEventListener('change', () => (sweep.rotations = rotationsInput.value));

  const sweepControls = h(
    'div',
    { class: 'controls' },
    field(t('syn.axis'), select([{ value: 'yaw', label: t('syn.axisYaw') }, { value: 'pitch', label: t('syn.axisPitch') }], sweep.axis, (v) => (sweep.axis = v))),
    field('σ (px)', numberInput(sweep.sigma, (v) => (sweep.sigma = v), { min: 0.05, step: 0.1 })),
    field(t('syn.distance'), numberInput(sweep.distance, (v) => (sweep.distance = v), { min: 25, max: 300, step: 5 })),
    field(t('syn.landmarks'), select((Object.keys(LANDMARK_SETS) as LandmarkSetId[]).map((k) => ({ value: k, label: t(`set.${k}.label` as MessageKey) })), sweep.set, (v) => (sweep.set = v))),
    field(t('syn.trials'), numberInput(sweep.trials, (v) => (sweep.trials = Math.max(10, Math.round(v))), { min: 10, step: 50 })),
    field('α', numberInput(sweep.alpha, (v) => (sweep.alpha = v), { min: 0.001, max: 0.5, step: 0.01 })),
    field(t('syn.seed'), numberInput(sweep.seed, (v) => (sweep.seed = Math.round(v)), { step: 1 })),
    field(t('syn.rotations'), rotationsInput),
    h('div', { class: 'btn-row' }, runBtn, stopBtn),
  );

  runBtn.addEventListener('click', async () => {
    const rotations = sweep.rotations
      .split(',')
      .map((v) => Number(v.trim()))
      .filter((v) => Number.isFinite(v) && v >= 0);
    if (!rotations.length) {
      status.textContent = t('syn.enterRotation');
      return;
    }
    runBtn.disabled = true;
    stopBtn.disabled = false;
    stopRequested = false;
    status.textContent = t('syn.running');
    const t0 = performance.now();
    const result = await runSweep(
      { kinds: KINDS, rotationsDeg: rotations, axis: sweep.axis, sigmaPx: sweep.sigma, distanceCm: sweep.distance, indices: LANDMARK_SETS[sweep.set].indices, trials: sweep.trials, alpha: sweep.alpha, seed: sweep.seed, intrinsics: DEFAULT_INTRINSICS },
      (p) => (progress.value = p.done / p.total),
      undefined,
      () => stopRequested,
    );
    runBtn.disabled = false;
    stopBtn.disabled = true;
    if (!result.ok) {
      status.textContent = result.reason === 'stopped' ? t('syn.stopped') : t('syn.cannotRun', { reason: result.reason });
      return;
    }
    status.textContent = t('syn.done', { seconds: ((performance.now() - t0) / 1000).toFixed(1), seed: sweep.seed });
    renderSweep(result.points);
  });
  stopBtn.addEventListener('click', () => (stopRequested = true));

  function renderSweep(points: SweepPoint[]): void {
    ratePlot.render({
      xLabel: t('syn.chartX', { axis: sweep.axis === 'yaw' ? t('syn.axisYaw') : t('syn.axisPitch') }),
      yLabel: t('syn.chartY'),
      yDomain: [0, 1],
      formatY: (v) => v.toFixed(2),
      series: KINDS.map((k) => ({
        label: KIND_LABEL[k](),
        ...KIND_STYLE[k],
        points: points.filter((p) => p.kind === k).map((p) => ({ x: p.rotationDeg, y: p.rate, lo: p.ci95[0], hi: p.ci95[1] })),
      })),
      refLines: [{ y: sweep.alpha, label: `α = ${sweep.alpha}` }],
    });
    clear(tableSlot);
    tableSlot.append(
      table(
        [t('syn.colObject'), t('syn.colRotation'), t('syn.colTrials'), t('syn.colRejected'), t('syn.colRate'), t('syn.colCi'), t('syn.colMeanT')],
        points.map((p) => [KIND_LABEL[p.kind](), fmt(p.rotationDeg, 1), String(p.trials), String(p.rejections), fmt(p.rate, 3), `${fmt(p.ci95[0], 3)} – ${fmt(p.ci95[1], 3)}`, fmt(p.meanTOverDof, 3)]),
        t('syn.tableCaption'),
      ),
    );
  }

  ratePlot.render({ xLabel: t('syn.colRotation'), yLabel: t('syn.chartY'), emptyMessage: t('syn.chartEmpty') });

  const sweepCard = card(
    t('syn.cardSweep'),
    h(
      'p',
      { class: 'outcome-reason' },
      t('syn.sweepHelp'),
    ),
    sweepControls,
    progress,
    status,
    ratePlot.element,
    tableSlot,
    note(
      t('syn.idealNote'),
      'warn',
    ),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, t('syn.title')),
    h('p', { class: 'page-intro' }, t('syn.intro')),
    h('div', { class: 'stack' }, pairCard, sweepCard),
  );

  const ro = new ResizeObserver(() => renderPair());
  ro.observe(pairCanvas);
  window.addEventListener('themechange', renderPair);
  return {
    element,
    dispose: () => {
      ro.disconnect();
      window.removeEventListener('themechange', renderPair);
    },
  };
}
