import { LANDMARK_SETS, type LandmarkSetId } from '../../core/landmark-sets';
import { chiSquarePdf, chiSquareQuantile } from '../../core/stats';
import { card, clear, field, fmt, fmtP, fmtUnit, h, note, numberInput, row, select } from '../dom';
import { t, type MessageKey } from '../i18n';
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
    t('live.cameraOff'),
  );
  const stage = h('div', { class: 'stage mirrored' }, session.video, overlay, placeholder);

  const startBtn = h('button', { class: 'btn', type: 'button' }, t('common.startCamera')) as HTMLButtonElement;
  const stopBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('common.stop')) as HTMLButtonElement;
  startBtn.addEventListener('click', () => void session.start(delegate, facing));
  stopBtn.addEventListener('click', () => session.stop());

  const controls = h(
    'div',
    { class: 'controls' },
    h('div', { class: 'btn-row' }, startBtn, stopBtn),
    field(
      t('live.camera'),
      select(
        [
          { value: 'user', label: t('live.front') },
          { value: 'environment', label: t('live.rear') },
        ],
        facing,
        (v) => (facing = v),
      ),
    ),
    field(
      t('live.inference'),
      select(
        [
          { value: 'GPU', label: t('live.gpu') },
          { value: 'CPU', label: t('live.cpu') },
        ],
        delegate,
        (v) => (delegate = v),
      ),
    ),
    field(
      t('live.display'),
      select(
        [
          { value: 'mirror', label: t('live.mirrored') },
          { value: 'raw', label: t('live.notMirrored') },
        ],
        'mirror',
        (v) => {
          mirrored = v === 'mirror';
          stage.classList.toggle('mirrored', mirrored);
        },
      ),
      t('live.displayHint'),
    ),
  );

  const overlayLegend = h(
    'div',
    { class: 'legend-inline' },
    h('span', {}, h('span', { class: 'dot', style: 'background:var(--overlay-point)' }), t('live.legendPoints')),
    h('span', {}, h('span', { class: 'dot', style: 'background:var(--overlay-vector)' }), t('live.legendVectors', { gain: VECTOR_GAIN })),
  );

  const historyPlot = new Plot(t('live.cardHistory'), 180);
  const residualPlot = new Plot(t('live.cardResiduals'), 180);

  const outcomeSlot = h('div');
  const testSlot = h('div');
  const noiseSlot = h('div');
  const measureSlot = h('div');

  const calibBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('live.calibrate')) as HTMLButtonElement;
  calibBtn.addEventListener('click', () => session.startCalibration());

  const settings = card(
    t('live.cardSettings'),
    h(
      'div',
      { class: 'controls' },
      field(
        t('live.landmarkSet'),
        select(
          (Object.keys(LANDMARK_SETS) as LandmarkSetId[]).map((k) => ({ value: k, label: t(`set.${k}.label` as MessageKey) })),
          session.config.landmarkSet,
          (v) => session.setConfig({ landmarkSet: v }),
        ),
      ),
      field('α', numberInput(session.config.alpha, (v) => session.setConfig({ alpha: v }), { min: 0.001, max: 0.5, step: 0.01 })),
      field(
        t('live.minMotion'),
        numberInput(session.config.minMotionDeg, (v) => session.setConfig({ minMotionDeg: v }), { min: 0, max: 45, step: 0.5 }),
        t('live.placeholder'),
      ),
      field(t('live.window'), numberInput(session.config.windowMs, (v) => session.setConfig({ windowMs: v }), { min: 200, max: 5000, step: 100 })),
    ),
    note(t('live.minMotionNote'), 'warn'),
    h('p', { class: 'outcome-reason' }, t(`set.${session.config.landmarkSet}.desc` as MessageKey)),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, t('live.title')),
    h('p', { class: 'page-intro' }, t('live.intro')),
    h(
      'div',
      { class: 'grid-live' },
      h(
        'div',
        { class: 'stack' },
        h('section', { class: 'card' }, stage, overlayLegend, h('div', { style: 'margin-top:10px' }, controls)),
        card(t('live.cardHistory'), historyPlot.element),
        card(t('live.cardResiduals'), residualPlot.element),
      ),
      h(
        'div',
        { class: 'stack' },
        card(t('live.cardOutcome'), outcomeSlot),
        card(t('live.cardTest'), testSlot),
        card(t('live.cardNoise'), noiseSlot, h('div', { class: 'btn-row', style: 'margin-top:8px' }, calibBtn)),
        card(t('live.cardMeasure'), measureSlot),
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
    if (s.state === 'starting') placeholder.textContent = t('live.starting');
    if (s.state === 'error') {
      placeholder.hidden = false;
      placeholder.textContent = t('live.couldNotStart', { error: s.error });
    }

    const a = s.latestAnalysis;
    clear(outcomeSlot);
    const noteText = s.analysisNote ? t(s.analysisNote.key, s.analysisNote.params) : t('live.waiting');
    outcomeSlot.append(outcomeView(running ? a : null, running ? noteText : t('live.startToBegin')));

    clear(testSlot);
    const dof = a?.stat?.dof ?? (a ? 2 * a.n - 8 : null);
    const crit = dof && dof > 0 ? chiSquareQuantile(1 - s.config.alpha, dof) : null;
    testSlot.append(
      row(t('live.rowN'), a ? String(a.n) : '—', 'configuration'),
      row(t('live.rowRotation'), a ? `${fmt(a.motionDeg, 1)}°` : '—', 'mediapipe-estimate', t('live.rowRotationHint')),
      row(t('live.rowGate'), a ? (a.motionDeg >= s.config.minMotionDeg ? t('live.gatePassed') : t('live.gateBelow', { min: s.config.minMotionDeg })) : '—', 'configuration'),
      row(t('live.rowT'), fmt(a?.stat?.T, 1), a?.stat ? 'method-output' : stageProvenance(a, 'statistic')),
      row(t('live.rowDof'), dof !== null ? String(dof) : '—', 'method-output'),
      row(t('live.rowCrit'), fmt(crit, 1), 'method-output'),
      row(t('live.rowP'), fmtP(a?.stat?.pValue), a?.stat ? 'method-output' : stageProvenance(a, 'statistic')),
    );
    const blocked = a?.stages.filter((st) => st.state !== 'ok') ?? [];
    if (blocked.length) {
      testSlot.append(
        h('details', {}, h('summary', {}, t('live.stageStatus')), ...blocked.map((st) => h('div', { class: 'outcome-reason' }, `${st.stage}: ${st.state}${st.message ? ` — ${st.message}` : ''}`))),
      );
    }

    clear(noiseSlot);
    const c = s.calibration;
    const est = c.result?.estimate;
    noiseSlot.append(
      row(t('live.rowSigma'), c.phase === 'done' ? fmt(est?.sigmaPx, 3) : '—', 'method-output'),
      row(t('live.rowInterval'), c.result?.interval95 ? `${fmt(c.result.interval95[0], 3)} – ${fmt(c.result.interval95[1], 3)}` : '—', 'method-output'),
      row(t('live.rowSigmaDof'), est ? String(est.dof) : '—', 'method-output'),
      row(t('live.rowHoldMotion'), c.phase === 'none' ? '—' : `${fmt(c.maxRotationDeg, 2)}°`, 'mediapipe-estimate'),
    );
    if (c.phase === 'collecting') noiseSlot.append(note(t('live.holdStill')));
    else if (c.phase === 'none') noiseSlot.append(note(c.message ? t(c.message.key, c.message.params) : t('live.notCalibrated'), 'warn'));
    else if (c.message) noiseSlot.append(note(t(c.message.key, c.message.params), c.phase === 'rejected' ? 'warn' : 'info'));
    noiseSlot.append(note(t('live.noiseAssumption')));

    clear(measureSlot);
    const tm = s.timing.summary();
    const settings = s.camera?.settings;
    measureSlot.append(
      row(t('live.rowFps'), tm.frames > 1 ? `${fmt(tm.fps, 1)} fps` : '—', 'measured'),
      row(t('live.rowResolution'), settings?.width ? `${settings.width} × ${settings.height}` : '—', 'measured'),
      row(t('live.rowInferenceMean'), fmtUnit(tm.meanInferenceMs, 'ms', 1), 'measured'),
      row(t('live.rowDetected'), tm.frames ? t('live.detectedValue', { pct: fmt(100 * tm.detectionRate, 0) }) : '—', 'mediapipe-estimate'),
      row(t('live.rowFaceSize'), fmtUnit(s.faceSizePx(), 'px', 0), 'mediapipe-estimate'),
      row(t('live.rowBackend'), s.delegate ?? '—', 'measured'),
    );

    historyPlot.render({
      xLabel: t('live.histX'),
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
      refLines: crit && dof ? [{ y: crit / dof, label: t('live.histCritical', { alpha: s.config.alpha }) }, { y: 1, label: t('live.histExpected') }] : [],
      emptyMessage: s.sigmaPx === null ? t('live.histEmptyNoSigma') : t('live.histEmpty'),
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
        xLabel: t('live.residX'),
        yLabel: t('live.residY'),
        histogram: { label: t('live.residObserved'), colorVar: '--series-1', edges, heights: counts.map((n) => n / (z.length * width)) },
        curves: [{ label: t('live.residReference'), colorVar: '--series-2', dashed: true, points: edges.map((x) => ({ x: Math.max(x, 0.02), y: chiSquarePdf(Math.max(x, 0.02), 2) })) }],
      });
    } else {
      residualPlot.render({ xLabel: t('live.residX'), yLabel: t('live.residY'), emptyMessage: sigma === null ? t('live.residEmptyNoSigma') : t('live.residEmpty') });
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

