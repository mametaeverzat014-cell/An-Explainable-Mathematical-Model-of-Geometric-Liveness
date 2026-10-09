import { LANDMARK_SETS } from '../../core/landmark-sets';
import { bundleFileName, makeBundle, type Recording } from '../../core/recording';
import {
  analyzeRecording,
  analyzeSet,
  calibrationFromRecording,
  DEFAULT_SET_SETTINGS,
  type RecordingSummary,
} from '../../core/recording-analysis';
import { card, clear, field, fmt, h, note, numberInput, row, select, table } from '../dom';
import { t, type MessageKey } from '../i18n';
import { buildRecording, Capture, downloadJson, tabRecordings } from '../recorder';
import { session } from '../session';
import { cameraStage } from '../stage';
import { handoff } from './analyze';

// Guided experiment for open question Q4: is a moving flat print planar to
// MediaPipe? All settings are DEFAULT_SET_SETTINGS, so the wizard, the
// Analyze view and the command-line tool give identical numbers.

const SETTINGS = DEFAULT_SET_SETTINGS;
const INDICES = LANDMARK_SETS[SETTINGS.landmarkSet].indices;
const CALIBRATION_S = 3;
const CONDITIONS = ['flat-print', 'screen-photo', 'curved-print'] as const;

interface Calibration {
  recording: Recording;
  sigmaPx: number;
  interval95: [number, number] | null;
  pairs: number;
}

interface ExperimentState {
  step: 1 | 2 | 3 | 4;
  nTrials: number;
  trialSeconds: number;
  condition: string;
  protocolId: string;
  notes: string;
  confirmed: boolean;
  calibration: Calibration | null;
  calibrationMessage: string;
  trials: { recording: Recording; summary: RecordingSummary }[];
}

function freshState(): ExperimentState {
  return {
    step: 1,
    nTrials: 10,
    trialSeconds: 10,
    condition: 'flat-print',
    protocolId: `Q4-${new Date().toISOString().slice(0, 10)}`,
    notes: '',
    confirmed: false,
    calibration: null,
    calibrationMessage: '',
    trials: [],
  };
}

/** Kept at module level so switching tabs does not lose progress. */
let state = freshState();

export function experimentView(): { element: HTMLElement; dispose: () => void } {
  let capture: Capture | null = null;
  const stage = cameraStage(() => capture?.active ?? false);
  const body = h('div', { class: 'stack' });
  const steps = h('div', { class: 'steps', 'aria-label': t('exp.step', { i: state.step }) });
  const progress = h('progress', { max: 1, value: 0 }) as HTMLProgressElement;

  function render(): void {
    clear(steps);
    (['exp.s1.title', 'exp.s2.title', 'exp.s3.title', 'exp.s4.title'] as const).forEach((k, i) => {
      const n = i + 1;
      steps.append(
        h('span', { class: `step-pill${n < state.step ? ' done' : ''}`, 'aria-current': n === state.step ? 'step' : undefined }, `${n}. ${t(k)}`),
      );
    });
    clear(body);
    if (state.step === 1) renderSetup();
    else if (state.step === 2) renderCalibration();
    else if (state.step === 3) renderTrials();
    else renderResult();
  }

  function cameraControls(): HTMLElement {
    const camBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('common.startCamera')) as HTMLButtonElement;
    camBtn.disabled = session.state === 'running' || session.state === 'starting';
    camBtn.addEventListener('click', () => void session.start('GPU', 'user'));
    return camBtn;
  }

  function info(role: 'calibration' | 'trial') {
    return {
      category: 'non-human-target' as const,
      approvalReference: null,
      pseudonym: null,
      role,
      condition: state.condition,
      protocolId: state.protocolId || null,
      notes: state.notes,
    };
  }

  function record(seconds: number, onDone: (rec: Recording | null, error: string | null) => void): void {
    if (capture?.active || session.state !== 'running') return;
    capture = new Capture(
      seconds * 1000,
      (p) => (progress.value = Math.min(1, p.elapsedMs / p.durationMs)),
      (frames, noFace) => {
        progress.value = 0;
        try {
          onDone(buildRecording(frames, noFace, info(state.step === 2 ? 'calibration' : 'trial')), null);
        } catch (e) {
          onDone(null, frames.length < 2 ? t('rec.nothing') : e instanceof Error ? e.message : String(e));
        }
      },
    );
    capture.start();
    render();
  }

  // ---------- step 1 ----------
  function renderSetup(): void {
    const confirm = h('input', { type: 'checkbox', checked: state.confirmed }) as HTMLInputElement;
    const next = h('button', { class: 'btn', type: 'button' }, t('common.next')) as HTMLButtonElement;
    const update = () => (next.disabled = !(state.confirmed && session.state === 'running'));
    confirm.addEventListener('change', () => {
      state.confirmed = confirm.checked;
      update();
    });
    next.addEventListener('click', () => {
      state.step = 2;
      render();
    });
    const protocolInput = h('input', { type: 'text', value: state.protocolId, style: 'width:180px' }) as HTMLInputElement;
    protocolInput.addEventListener('input', () => (state.protocolId = protocolInput.value.trim()));
    const notesInput = h('textarea', { rows: 2, style: 'width:100%;font:inherit' }, state.notes) as HTMLTextAreaElement;
    notesInput.addEventListener('input', () => (state.notes = notesInput.value));
    body.append(
      card(
        t('exp.s1.title'),
        h('p', {}, t('exp.s1.materials')),
        note(t('exp.s1.photoNote')),
        h(
          'div',
          { class: 'controls' },
          field(t('rec.condition'), select(CONDITIONS.map((c) => ({ value: c, label: t(`cond.${c}` as MessageKey) })), state.condition as (typeof CONDITIONS)[number], (v) => (state.condition = v))),
          field(t('exp.trials'), numberInput(state.nTrials, (v) => (state.nTrials = Math.min(50, Math.max(2, Math.round(v)))), { min: 2, max: 50, step: 1 })),
          field(t('exp.trialSeconds'), numberInput(state.trialSeconds, (v) => (state.trialSeconds = Math.min(60, Math.max(3, v))), { min: 3, max: 60, step: 1 })),
          field(t('rec.protocolId'), protocolInput),
        ),
        field(t('rec.notes'), notesInput, t('rec.notesHint')),
        h('label', { class: 'field', style: 'flex-direction:row;gap:8px;align-items:center;margin-top:8px' }, confirm, h('span', {}, t('exp.confirmNoPerson'))),
        h('div', { class: 'btn-row', style: 'margin-top:10px' }, cameraControls(), next),
      ),
    );
    update();
  }

  // ---------- step 2 ----------
  function renderCalibration(): void {
    const c = state.calibration;
    const recBtn = h('button', { class: 'btn', type: 'button' }, c ? t('exp.recalibrate') : t('exp.recordCalibration')) as HTMLButtonElement;
    recBtn.disabled = capture?.active === true || session.state !== 'running';
    recBtn.addEventListener('click', () =>
      record(CALIBRATION_S, (rec, error) => {
        if (!rec) {
          state.calibration = null;
          state.calibrationMessage = error ?? '';
          render();
          return;
        }
        const r = calibrationFromRecording(rec, INDICES, SETTINGS.calibrationGap, SETTINGS.maxCalibrationMotionDeg);
        if (r.state === 'ok' && r.estimate) {
          state.calibration = { recording: rec, sigmaPx: r.estimate.sigmaPx, interval95: r.interval95, pairs: r.pairs };
          state.calibrationMessage = '';
          tabRecordings.unshift(rec);
        } else {
          state.calibration = null;
          state.calibrationMessage = r.rejectedForMotion
            ? t('live.calibrationMotion', { motion: fmt(r.maxRotationDeg, 1), limit: SETTINGS.maxCalibrationMotionDeg })
            : t('live.calibrationFailed', { message: r.message ?? '' });
        }
        render();
      }),
    );
    const next = h('button', { class: 'btn', type: 'button', disabled: !c }, t('common.next')) as HTMLButtonElement;
    next.addEventListener('click', () => {
      state.step = 3;
      render();
    });
    const back = h('button', { class: 'btn btn-secondary', type: 'button' }, t('common.back'));
    back.addEventListener('click', () => {
      state.step = 1;
      render();
    });
    body.append(
      card(
        t('exp.s2.title'),
        h('p', {}, t('exp.s2.instructions', { seconds: CALIBRATION_S })),
        stage.element,
        progress,
        h('div', { class: 'btn-row', style: 'margin-top:10px' }, cameraControls(), recBtn),
        c
          ? h('p', { class: 'big-result' }, t('exp.sigmaResult', { sigma: fmt(c.sigmaPx, 3), lo: fmt(c.interval95?.[0], 3), hi: fmt(c.interval95?.[1], 3), pairs: c.pairs }))
          : null,
        state.calibrationMessage ? note(state.calibrationMessage, 'warn') : null,
        h('div', { class: 'btn-row', style: 'margin-top:10px' }, back, next),
      ),
    );
  }

  // ---------- step 3 ----------
  function renderTrials(): void {
    const done = state.trials.length;
    const recBtn = h(
      'button',
      { class: 'btn', type: 'button' },
      t('exp.recordTrial', { i: Math.min(done + 1, state.nTrials), n: state.nTrials }),
    ) as HTMLButtonElement;
    recBtn.disabled = capture?.active === true || session.state !== 'running' || done >= state.nTrials;
    recBtn.addEventListener('click', () =>
      record(state.trialSeconds, (rec, error) => {
        if (rec && state.calibration) {
          const r = analyzeRecording(rec, {
            indices: INDICES,
            windowMs: SETTINGS.windowMs,
            minMotionDeg: SETTINGS.minMotionDeg,
            alpha: SETTINGS.alpha,
            sigmaPx: state.calibration.sigmaPx,
          });
          state.trials.push({ recording: rec, summary: r.summary });
          tabRecordings.unshift(rec);
          state.calibrationMessage = '';
        } else {
          state.calibrationMessage = error ?? '';
        }
        render();
      }),
    );
    const redo = h('button', { class: 'btn btn-secondary', type: 'button', disabled: done === 0 }, t('exp.redoLast'));
    redo.addEventListener('click', () => {
      const last = state.trials.pop();
      if (last) {
        const i = tabRecordings.indexOf(last.recording);
        if (i >= 0) tabRecordings.splice(i, 1);
      }
      render();
    });
    const finish = h('button', { class: 'btn', type: 'button', disabled: done === 0 }, t('exp.finish'));
    finish.addEventListener('click', () => {
      state.step = 4;
      render();
    });
    body.append(
      card(
        t('exp.s3.title'),
        h('p', {}, t('exp.s3.instructions')),
        stage.element,
        progress,
        h('div', { class: 'btn-row', style: 'margin-top:10px' }, cameraControls(), recBtn, redo),
        state.calibrationMessage ? note(state.calibrationMessage, 'warn') : null,
        done
          ? h(
              'div',
              { class: 'table-wrap', style: 'margin-top:10px' },
              table(
                [t('exp.colTrial'), t('ana.colPairs'), t('ana.colTested'), t('ana.colRejected'), t('ana.colRate')],
                state.trials.map((tr, i) => [
                  String(i + 1),
                  String(tr.summary.pairs),
                  String(tr.summary.tested),
                  String(tr.summary.rejected),
                  fmt(tr.summary.rejectionRate, 3),
                ]),
              ),
            )
          : null,
        done > 0 && done < state.nTrials ? note(t('exp.finishEarly')) : null,
        h('div', { class: 'btn-row', style: 'margin-top:10px' }, finish),
      ),
    );
  }

  // ---------- step 4 ----------
  function renderResult(): void {
    const recordings = [state.calibration!.recording, ...state.trials.map((x) => x.recording)];
    const a = analyzeSet(recordings, SETTINGS);
    const restart = h('button', { class: 'btn btn-secondary', type: 'button' }, t('exp.restart'));
    restart.addEventListener('click', () => {
      state = freshState();
      render();
    });
    if (!a.ok) {
      body.append(card(t('exp.s4.title'), note(t('ana.error', { error: a.error }), 'warn'), restart));
      return;
    }
    const k = a.conditions[0];
    const alpha = SETTINGS.alpha;
    const ci = k.meanRateCi95;
    const verdict = !ci ? t('exp.tooFew') : ci[0] > alpha ? t('exp.higher', { alpha }) : ci[1] < alpha ? t('exp.lower', { alpha }) : t('exp.consistent', { alpha });
    const bundleBtn = h('button', { class: 'btn', type: 'button' }, t('exp.downloadBundle'));
    bundleBtn.addEventListener('click', () => {
      const b = makeBundle(recordings, state.protocolId || null, state.notes);
      downloadJson(b, bundleFileName(b));
    });
    const openBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('exp.openAnalyze'));
    openBtn.addEventListener('click', () => {
      handoff.recordings = recordings;
      location.hash = '#analyze';
    });
    body.append(
      card(
        t('exp.s4.title'),
        h('p', {}, t('exp.prediction', { alpha })),
        row(t('ana.colMeanRate'), `${fmt(k.meanRecordingRate, 3)} ${ci ? `[${fmt(ci[0], 3)}, ${fmt(ci[1], 3)}]` : ''}`, 'method-output'),
        row(t('ana.colRecordings'), String(k.recordings), 'measured'),
        row(`${t('ana.colRejected')} / ${t('ana.colTested')}`, `${k.rejected} / ${k.tested}`, 'method-output'),
        row('σ', `${fmt(a.calibration.sigmaPx, 3)} px`, 'method-output'),
        h('p', { class: 'big-result' }, verdict),
        note(t('exp.scope', { n: k.recordings }), 'warn'),
        h('div', { class: 'btn-row', style: 'margin-top:10px' }, bundleBtn, openBtn, restart),
      ),
    );
  }

  const element = h('div', {}, h('h1', {}, t('exp.title')), h('p', { class: 'page-intro' }, t('exp.intro')), h('p', { class: 'page-intro' }, t('exp.prediction', { alpha: SETTINGS.alpha })), steps, body);
  // Re-render only when the camera state changes (the session emits ~10 times
  // a second while running, which would otherwise reset inputs being typed).
  let lastState = session.state;
  const unsub = session.subscribe(() => {
    if (session.state !== lastState) {
      lastState = session.state;
      if (!capture?.active) render();
    }
  });
  render();
  return {
    element,
    dispose: () => {
      unsub();
      capture?.cancel();
      stage.dispose();
    },
  };
}
