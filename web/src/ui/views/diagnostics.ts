import { sensorReport } from '../../capture/sensors';
import { supportsVideoFrameCallback } from '../../capture/timing';
import { card, clear, fmt, fmtUnit, h, note, row } from '../dom';
import { t } from '../i18n';
import { Plot } from '../plot';
import { session } from '../session';

function histogram(values: number[], bins: number): { edges: number[]; heights: number[] } {
  if (!values.length) return { edges: [0, 1], heights: [0] };
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + (i * span) / bins);
  const heights = new Array<number>(bins).fill(0);
  for (const v of values) heights[Math.min(bins - 1, Math.floor(((v - lo) / span) * bins))]++;
  return { edges, heights };
}

export function diagnosticsView(): { element: HTMLElement; dispose: () => void } {
  const timingSlot = h('div');
  const cameraSlot = h('div');
  const intervalPlot = new Plot(t('diag.cardIntervals'), 180);
  const inferencePlot = new Plot(t('diag.cardInference'), 180);

  const sensors = sensorReport();
  const yn = (b: boolean) => (b ? t('common.yes') : t('common.no'));
  const gl = (() => {
    try {
      return !!document.createElement('canvas').getContext('webgl2');
    } catch {
      return false;
    }
  })();

  const envCard = card(
    t('diag.cardEnv'),
    row(t('diag.secure'), yn(sensors.secureContext), 'measured'),
    row(t('diag.rvfc'), yn(supportsVideoFrameCallback(document.createElement('video'))), 'measured', t('diag.rvfcHint')),
    row(t('diag.webgl'), yn(gl), 'measured'),
    row(t('diag.dme'), yn(sensors.deviceMotionEvent), 'measured'),
    row(t('diag.motionPerm'), yn(sensors.motionPermissionApi), 'measured'),
    row(t('diag.gyro'), yn(sensors.genericSensorGyroscope), 'measured'),
    row(t('diag.touch'), yn(sensors.touchDevice), 'measured'),
    row(t('diag.cores'), String(navigator.hardwareConcurrency ?? '—'), 'measured'),
    note(t('diag.sensorNote')),
    h('details', {}, h('summary', {}, t('diag.userAgent')), h('p', { class: 'mono' }, navigator.userAgent)),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, t('diag.title')),
    h('p', { class: 'page-intro' }, t('diag.intro')),
    h(
      'div',
      { class: 'grid-2' },
      h('div', { class: 'stack' }, card(t('diag.cardTiming'), timingSlot), card(t('diag.cardIntervals'), intervalPlot.element), card(t('diag.cardInference'), inferencePlot.element)),
      h('div', { class: 'stack' }, card(t('diag.cardCamera'), cameraSlot), envCard),
    ),
  );

  function update(): void {
    const tm = session.timing.summary();
    clear(timingSlot);
    timingSlot.append(
      row(t('diag.state'), session.state, 'measured'),
      row(t('diag.framesProcessed'), String(tm.frames), 'measured'),
      row(t('diag.tsSource'), tm.source, 'measured', t('diag.tsSourceHint')),
      row(t('diag.meanInterval'), fmtUnit(tm.meanIntervalMs, 'ms', 1), 'measured'),
      row(t('diag.p95Interval'), fmtUnit(tm.p95IntervalMs, 'ms', 1), 'measured'),
      row(t('diag.maxInterval'), fmtUnit(tm.maxIntervalMs, 'ms', 1), 'measured'),
      row(t('diag.skipped'), tm.skippedFrames === null ? t('diag.unknown') : String(tm.skippedFrames), 'measured'),
      row(t('diag.nonMonotonic'), String(tm.nonMonotonic), 'measured'),
      row(t('diag.inferenceMeanP95'), Number.isFinite(tm.meanInferenceMs) ? `${fmt(tm.meanInferenceMs, 1)} / ${fmt(tm.p95InferenceMs, 1)} ms` : '—', 'measured'),
      row(t('diag.detectionRate'), tm.frames ? `${fmt(100 * tm.detectionRate, 1)} %` : '—', 'mediapipe-estimate'),
    );
    if (session.error) timingSlot.append(note(session.error, 'warn'));

    clear(cameraSlot);
    const cam = session.camera;
    if (!cam) cameraSlot.append(note(t('diag.cameraNotStarted')));
    else {
      const s = cam.settings;
      cameraSlot.append(
        row(t('diag.deviceLabel'), cam.label || t('diag.hidden'), 'measured'),
        row(t('live.rowResolution'), `${s.width ?? '—'} × ${s.height ?? '—'}`, 'measured'),
        row(t('diag.requestedFps'), s.frameRate ? `${fmt(s.frameRate, 1)} fps` : '—', 'measured'),
        row(t('diag.facing'), s.facingMode ?? '—', 'measured'),
        row(t('diag.resize'), (s as MediaTrackSettings & { resizeMode?: string }).resizeMode ?? '—', 'measured', t('diag.resizeHint')),
      );
      cameraSlot.append(h('details', {}, h('summary', {}, t('diag.fullSettings')), h('pre', { class: 'mono' }, JSON.stringify(s, null, 2))));
    }

    const intervals = session.timing.intervalSamples();
    intervalPlot.render({
      xLabel: 'ms',
      yLabel: t('diag.framesAxis'),
      histogram: intervals.length ? { label: 'interval', colorVar: '--series-1', ...histogram(intervals, 24) } : undefined,
      emptyMessage: t('diag.noFrames'),
    });
    const inference = session.timing.inferenceSamples();
    inferencePlot.render({
      xLabel: 'ms',
      yLabel: t('diag.framesAxis'),
      histogram: inference.length ? { label: 'inference', colorVar: '--series-1', ...histogram(inference, 24) } : undefined,
      emptyMessage: t('diag.noFrames'),
    });
  }

  const unsub = session.subscribe(update);
  update();
  return { element, dispose: unsub };
}
