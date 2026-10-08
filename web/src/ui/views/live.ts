import { CONFIG_NOTES } from '../../core/config';
import { LANDMARK_SETS, type LandmarkSetId } from '../../core/landmark-sets';
import { chiSquarePdf, chiSquareQuantile } from '../../core/stats';
import { card, clear, field, fmt, fmtP, fmtUnit, h, note, numberInput, row, select } from '../dom';
import { outcomeView } from '../outcome';
import { Plot } from '../plot';
import { session } from '../session';

const VECTOR_GAIN = 10;

export function liveView(): { element: HTMLElement; dispose: () => void } {
  let delegate: 'GPU' | 'CPU' = 'GPU';
  let facing: 'user' | 'environment' = 'user';
  let mirrored = true;

  const overlay = h('canvas', { class: 'overlay', 'aria-hidden': 'true' }) as HTMLCanvasElement;
  const placeholder = h(
    'div',
    { class: 'stage-placeholder' },
    'The camera is off. Video is processed only inside this browser tab; nothing is uploaded or stored.',
  );
  const stage = h('div', { class: 'stage mirrored' }, session.video, overlay, placeholder);

  const startBtn = h('button', { class: 'btn', type: 'button' }, 'Start camera') as HTMLButtonElement;
  const stopBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, 'Stop') as HTMLButtonElement;
  startBtn.addEventListener('click', () => void session.start(delegate, facing));
  stopBtn.addEventListener('click', () => session.stop());

  const controls = h(
    'div',
    { class: 'controls' },
    h('div', { class: 'btn-row' }, startBtn, stopBtn),
    field(
      'Camera',
      select(
        [
          { value: 'user', label: 'Front' },
          { value: 'environment', label: 'Rear' },
        ],
        facing,
        (v) => (facing = v),
      ),
    ),
    field(
      'Inference',
      select(
        [
          { value: 'GPU', label: 'GPU (falls back to CPU)' },
          { value: 'CPU', label: 'CPU' },
        ],
        delegate,
        (v) => (delegate = v),
      ),
    ),
    field(
      'Display',
      select(
        [
          { value: 'mirror', label: 'Mirrored' },
          { value: 'raw', label: 'Not mirrored' },
        ],
        'mirror',
        (v) => {
          mirrored = v === 'mirror';
          stage.classList.toggle('mirrored', mirrored);
        },
      ),
      'Display only; analysis always uses unmirrored coordinates.',
    ),
  );

  const overlayLegend = h(
    'div',
    { class: 'legend-inline' },
    h('span', {}, h('span', { class: 'dot', style: 'background:var(--overlay-point)' }), 'landmarks used by the test'),
    h('span', {}, h('span', { class: 'dot', style: 'background:var(--overlay-vector)' }), `residual  dst − H·src  (×${VECTOR_GAIN})`),
  );

  const historyPlot = new Plot('Test statistic T divided by its degrees of freedom over time', 180);
  const residualPlot = new Plot('Histogram of per-landmark normalised squared residuals with chi-square reference', 180);

  const outcomeSlot = h('div');
  const testSlot = h('div');
  const noiseSlot = h('div');
  const measureSlot = h('div');

  const calibBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, 'Calibrate noise floor') as HTMLButtonElement;
  calibBtn.addEventListener('click', () => session.startCalibration());

  const settings = card(
    'Test settings',
    h(
      'div',
      { class: 'controls' },
      field(
        'Landmark set',
        select(
          (Object.keys(LANDMARK_SETS) as LandmarkSetId[]).map((k) => ({ value: k, label: LANDMARK_SETS[k].label })),
          session.config.landmarkSet,
          (v) => session.setConfig({ landmarkSet: v }),
        ),
      ),
      field('α', numberInput(session.config.alpha, (v) => session.setConfig({ alpha: v }), { min: 0.001, max: 0.5, step: 0.01 })),
      field(
        'Min. motion (°)',
        numberInput(session.config.minMotionDeg, (v) => session.setConfig({ minMotionDeg: v }), { min: 0, max: 45, step: 0.5 }),
        'placeholder',
      ),
      field('Window (ms)', numberInput(session.config.windowMs, (v) => session.setConfig({ windowMs: v }), { min: 200, max: 5000, step: 100 })),
    ),
    note(`Minimum motion: ${CONFIG_NOTES.minMotionDeg.note}`, 'warn'),
    h('p', { class: 'outcome-reason' }, LANDMARK_SETS[session.config.landmarkSet].description),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, 'Live planarity test'),
    h(
      'p',
      { class: 'page-intro' },
      'Tests whether the landmarks of the presented face move like points on a single plane between two frames. ' +
        'It is a test of planarity, not a liveness verdict: a flat photo should be "consistent with a planar surface"; a real head that turns should produce evidence against planarity — but so can a curved print, a replayed video or a changing facial expression.',
    ),
    h(
      'div',
      { class: 'grid-live' },
      h(
        'div',
        { class: 'stack' },
        h('section', { class: 'card' }, stage, overlayLegend, h('div', { style: 'margin-top:10px' }, controls)),
        card('Statistic over time (T / dof)', historyPlot.element),
        card('Per-landmark residuals vs. the noise model', residualPlot.element),
      ),
      h(
        'div',
        { class: 'stack' },
        card('Outcome', outcomeSlot),
        card('Planarity test', testSlot),
        card('Noise floor (σ)', noiseSlot, h('div', { class: 'btn-row', style: 'margin-top:8px' }, calibBtn)),
        card('Measurements', measureSlot),
        settings,
      ),
    ),
  );

  function update(): void {
    const s = session;
    const running = s.state === 'running';
    startBtn.disabled = running || s.state === 'starting';
    stopBtn.disabled = !running;
    calibBtn.disabled = !running || s.calibration.phase === 'collecting';
    placeholder.hidden = running;
    if (s.state === 'starting') placeholder.textContent = 'Starting camera and loading the face landmark model…';
    if (s.state === 'error') {
      placeholder.hidden = false;
      placeholder.textContent = `Could not start: ${s.error}`;
    }

    const a = s.latestAnalysis;
    clear(outcomeSlot);
    outcomeSlot.append(outcomeView(running ? a : null, running ? s.analysisNote || 'Waiting…' : 'Start the camera to begin.'));

    clear(testSlot);
    const dof = a?.stat?.dof ?? (a ? 2 * a.n - 8 : null);
    const crit = dof && dof > 0 ? chiSquareQuantile(1 - s.config.alpha, dof) : null;
    testSlot.append(
      row('Landmarks n', a ? String(a.n) : '—', 'configuration'),
      row('Rotation between frames', a ? `${fmt(a.motionDeg, 1)}°` : '—', 'mediapipe-estimate', 'From MediaPipe’s facial transformation matrix; the reference frame is the one with the largest rotation in the window.'),
      row('Motion gate', a ? (a.motionDeg >= s.config.minMotionDeg ? 'passed' : `below ${s.config.minMotionDeg}°`) : '—', 'configuration'),
      row('T = Σe²/σ²', fmt(a?.stat?.T, 1), a?.stat ? 'method-output' : stageProvenance(a, 'statistic')),
      row('Degrees of freedom 2n − 8', dof !== null ? String(dof) : '—', 'method-output'),
      row(`Critical value χ²(1 − α)`, fmt(crit, 1), 'method-output'),
      row('p-value', fmtP(a?.stat?.pValue), a?.stat ? 'method-output' : stageProvenance(a, 'statistic')),
    );
    const blocked = a?.stages.filter((st) => st.state !== 'ok') ?? [];
    if (blocked.length) {
      testSlot.append(
        h('details', {}, h('summary', {}, 'Stage status'), ...blocked.map((st) => h('div', { class: 'outcome-reason' }, `${st.stage}: ${st.state}${st.message ? ` — ${st.message}` : ''}`))),
      );
    }

    clear(noiseSlot);
    const c = s.calibration;
    const est = c.result?.estimate;
    noiseSlot.append(
      row('σ (px per coordinate)', c.phase === 'done' ? fmt(est?.sigmaPx, 3) : '—', 'method-output'),
      row('95 % interval', c.result?.interval95 ? `${fmt(c.result.interval95[0], 3)} – ${fmt(c.result.interval95[1], 3)}` : '—', 'method-output'),
      row('Degrees of freedom', est ? String(est.dof) : '—', 'method-output'),
      row('Pose change during hold', c.phase === 'none' ? '—' : `${fmt(c.maxRotationDeg, 2)}°`, 'mediapipe-estimate'),
    );
    if (c.phase === 'collecting') noiseSlot.append(note('Hold the presented object (face or photo) still…'));
    else if (c.phase === 'none') noiseSlot.append(note('Not calibrated. Without σ the test cannot run: the outcome stays inconclusive.', 'warn'));
    else if (c.message) noiseSlot.append(note(c.message, c.phase === 'rejected' ? 'warn' : 'info'));
    noiseSlot.append(
      note(
        'Assumes landmark errors are independent, Gaussian and equal in both axes, and that noise during motion equals noise during the hold. These assumptions are open questions Q1 and Q3 in MATH_SPEC.md.',
      ),
    );

    clear(measureSlot);
    const t = s.timing.summary();
    const settings = s.camera?.settings;
    measureSlot.append(
      row('Frame rate', t.frames > 1 ? `${fmt(t.fps, 1)} fps` : '—', 'measured'),
      row('Resolution', settings?.width ? `${settings.width} × ${settings.height}` : '—', 'measured'),
      row('Inference time (mean)', fmtUnit(t.meanInferenceMs, 'ms', 1), 'measured'),
      row('Face detected', t.frames ? `${fmt(100 * t.detectionRate, 0)} % of frames` : '—', 'mediapipe-estimate'),
      row('Face size (outer eye corners)', fmtUnit(s.faceSizePx(), 'px', 0), 'mediapipe-estimate'),
      row('Inference backend', s.delegate ?? '—', 'measured'),
    );

    historyPlot.render({
      xLabel: 'time (s, relative)',
      yLabel: 'T / dof',
      series: [
        {
          label: 'T / dof',
          colorVar: '--series-1',
          marker: 'circle',
          showMarkers: false,
          points: historyPoints(),
        },
      ],
      refLines: crit && dof ? [{ y: crit / dof, label: `critical (α = ${s.config.alpha})` }, { y: 1, label: 'expected under H₀' }] : [],
      emptyMessage: s.sigmaPx === null ? 'Calibrate the noise floor to see the statistic' : 'No statistic yet',
    });

    const sigma = s.sigmaPx;
    if (a?.errorsSquared && sigma) {
      const z = a.errorsSquared.map((e) => e / (sigma * sigma));
      const maxZ = Math.max(10, ...z.filter(Number.isFinite));
      const bins = 20;
      const edges = Array.from({ length: bins + 1 }, (_, i) => (i * maxZ) / bins);
      const counts = new Array<number>(bins).fill(0);
      for (const v of z) counts[Math.min(bins - 1, Math.floor((v / maxZ) * bins))]++;
      const width = maxZ / bins;
      residualPlot.render({
        xLabel: 'e² / σ² per landmark',
        yLabel: 'density',
        histogram: { label: 'observed', colorVar: '--series-1', edges, heights: counts.map((n) => n / (z.length * width)) },
        curves: [{ label: 'χ²(2) reference (approx.)', colorVar: '--series-2', dashed: true, points: edges.map((x) => ({ x: Math.max(x, 0.02), y: chiSquarePdf(Math.max(x, 0.02), 2) })) }],
      });
    } else {
      residualPlot.render({ xLabel: 'e² / σ²', yLabel: 'density', emptyMessage: sigma === null ? 'Needs σ (calibration) and residuals (tasks M1, M2)' : 'Residuals unavailable (tasks M1, M2)' });
    }
  }

  function historyPoints() {
    const hst = session.history.filter((p) => p.tOverDof !== null);
    if (!hst.length) return [];
    const t0 = hst[hst.length - 1].t;
    return hst.map((p) => ({ x: (p.t - t0) / 1000, y: p.tOverDof as number }));
  }

  function drawOverlay(): void {
    const v = session.video;
    const frame = session.latestFrame;
    if (!v.videoWidth) return;
    if (overlay.width !== v.videoWidth || overlay.height !== v.videoHeight) {
      overlay.width = v.videoWidth;
      overlay.height = v.videoHeight;
      // Match the stage to the camera's aspect ratio (portrait on phones).
      stage.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
    }
    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!frame) return;
    const style = getComputedStyle(element);
    const pointColor = style.getPropertyValue('--overlay-point').trim();
    const vecColor = style.getPropertyValue('--overlay-vector').trim();
    const scale = overlay.width / 1280;
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    for (const p of frame.points) ctx.fillRect(p.x - scale, p.y - scale, 2 * scale, 2 * scale);
    const idx = LANDMARK_SETS[session.config.landmarkSet].indices;
    const a = session.latestAnalysis;
    if (a?.residualVectors && a.residualVectors.length === idx.length) {
      ctx.lineWidth = 2.5 * scale;
      idx.forEach((li, k) => {
        const p = frame.points[li];
        const r = a.residualVectors![k];
        ctx.strokeStyle = 'rgba(0,0,0,0.6)';
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + VECTOR_GAIN * r.x, p.y + VECTOR_GAIN * r.y);
        ctx.stroke();
        ctx.strokeStyle = vecColor;
        ctx.lineWidth = 1.5 * scale;
        ctx.stroke();
        ctx.lineWidth = 2.5 * scale;
      });
    }
    for (const li of idx) {
      const p = frame.points[li];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.2 * scale, 0, 2 * Math.PI);
      ctx.fillStyle = pointColor;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2 * scale;
      ctx.fill();
      ctx.stroke();
    }
  }

  const unsub = session.subscribe(update);
  const unsubFrame = session.onFrame(drawOverlay);
  update();
  return {
    element,
    dispose: () => {
      unsub();
      unsubFrame();
      session.parkVideo();
    },
  };
}

function stageProvenance(a: { stages: { stage: string; state: string }[] } | null, stage: string) {
  const st = a?.stages.find((s) => s.stage === stage);
  return st?.state === 'not-implemented' ? ('not-implemented' as const) : ('method-output' as const);
}

