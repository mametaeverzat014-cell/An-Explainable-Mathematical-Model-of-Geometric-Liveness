import { sensorReport } from '../../capture/sensors';
import { supportsVideoFrameCallback } from '../../capture/timing';
import { card, clear, fmt, fmtUnit, h, note, row } from '../dom';
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
  const intervalPlot = new Plot('Histogram of intervals between processed frames', 180);
  const inferencePlot = new Plot('Histogram of landmark inference time per frame', 180);

  const sensors = sensorReport();
  const yn = (b: boolean) => (b ? 'yes' : 'no');
  const gl = (() => {
    try {
      return !!document.createElement('canvas').getContext('webgl2');
    } catch {
      return false;
    }
  })();

  const envCard = card(
    'Browser and sensors',
    row('Secure context (camera allowed)', yn(sensors.secureContext), 'measured'),
    row('requestVideoFrameCallback', yn(supportsVideoFrameCallback(document.createElement('video'))), 'measured', 'Needed for per-frame timestamps from the video pipeline.'),
    row('WebGL 2 (GPU inference)', yn(gl), 'measured'),
    row('DeviceMotionEvent', yn(sensors.deviceMotionEvent), 'measured'),
    row('Motion permission API (iOS)', yn(sensors.motionPermissionApi), 'measured'),
    row('Generic Sensor Gyroscope', yn(sensors.genericSensorGyroscope), 'measured'),
    row('Touch device', yn(sensors.touchDevice), 'measured'),
    row('Logical CPU cores', String(navigator.hardwareConcurrency ?? '—'), 'measured'),
    note('Phase 3 (gyroscope consistency) will use these sensors. A desktop browser may expose DeviceMotionEvent without having any motion hardware; availability here does not mean data will arrive.'),
    h('details', {}, h('summary', {}, 'User agent'), h('p', { class: 'mono' }, navigator.userAgent)),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, 'Diagnostics'),
    h(
      'p',
      { class: 'page-intro' },
      'Camera and timing quality determine whether the noise model can hold. Dropped frames, irregular intervals and slow inference all change what the landmarks measure. Start the camera in the Live view; this page reads the same session.',
    ),
    h(
      'div',
      { class: 'grid-2' },
      h('div', { class: 'stack' }, card('Frame timing', timingSlot), card('Interval between processed frames', intervalPlot.element), card('Inference time', inferencePlot.element)),
      h('div', { class: 'stack' }, card('Camera track', cameraSlot), envCard),
    ),
  );

  function update(): void {
    const t = session.timing.summary();
    clear(timingSlot);
    timingSlot.append(
      row('Session state', session.state, 'measured'),
      row('Frames processed', String(t.frames), 'measured'),
      row('Timestamp source', t.source, 'measured', 'captureTime: camera capture clock; mediaTime: video timeline; animationFrame: page refresh clock (least precise).'),
      row('Mean interval', fmtUnit(t.meanIntervalMs, 'ms', 1), 'measured'),
      row('95th percentile interval', fmtUnit(t.p95IntervalMs, 'ms', 1), 'measured'),
      row('Longest interval', fmtUnit(t.maxIntervalMs, 'ms', 1), 'measured'),
      row('Presented but not processed', t.skippedFrames === null ? 'unknown' : String(t.skippedFrames), 'measured'),
      row('Non-increasing timestamps', String(t.nonMonotonic), 'measured'),
      row('Inference mean / p95', Number.isFinite(t.meanInferenceMs) ? `${fmt(t.meanInferenceMs, 1)} / ${fmt(t.p95InferenceMs, 1)} ms` : '—', 'measured'),
      row('Face detection rate', t.frames ? `${fmt(100 * t.detectionRate, 1)} %` : '—', 'mediapipe-estimate'),
    );
    if (session.error) timingSlot.append(note(session.error, 'warn'));

    clear(cameraSlot);
    const cam = session.camera;
    if (!cam) cameraSlot.append(note('Camera not started.'));
    else {
      const s = cam.settings;
      cameraSlot.append(
        row('Device label', cam.label || '(hidden)', 'measured'),
        row('Resolution', `${s.width ?? '—'} × ${s.height ?? '—'}`, 'measured'),
        row('Requested frame rate', s.frameRate ? `${fmt(s.frameRate, 1)} fps` : '—', 'measured'),
        row('Facing mode', s.facingMode ?? '—', 'measured'),
        row('Resize mode', (s as MediaTrackSettings & { resizeMode?: string }).resizeMode ?? '—', 'measured', 'If the browser crops or scales the camera image, pixel noise is no longer in sensor units.'),
      );
      cameraSlot.append(h('details', {}, h('summary', {}, 'Full track settings'), h('pre', { class: 'mono' }, JSON.stringify(s, null, 2))));
    }

    const intervals = session.timing.intervalSamples();
    intervalPlot.render({
      xLabel: 'ms',
      yLabel: 'frames',
      histogram: intervals.length ? { label: 'interval', colorVar: '--series-1', ...histogram(intervals, 24) } : undefined,
      emptyMessage: 'No frames yet',
    });
    const inference = session.timing.inferenceSamples();
    inferencePlot.render({
      xLabel: 'ms',
      yLabel: 'frames',
      histogram: inference.length ? { label: 'inference', colorVar: '--series-1', ...histogram(inference, 24) } : undefined,
      emptyMessage: 'No frames yet',
    });
  }

  const unsub = session.subscribe(update);
  update();
  return { element, dispose: unsub };
}
