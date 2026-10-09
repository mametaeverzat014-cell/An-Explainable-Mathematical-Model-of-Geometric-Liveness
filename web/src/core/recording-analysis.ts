import { calibrationPairs, correspondences, frameRotationDeg, type LandmarkFrame } from './frames';
import { rotationFromAngles } from './geometry3d';
import { analyzePair, calibrateNoise, DEFAULT_METHOD, type NoiseCalibration, type PlanarityMethod } from './method';
import { framesOf, frameToRecorded, RECORDING_FORMAT, RECORDING_VERSION, type Recording, type RecordingMeta } from './recording';
import { Rng } from './rng';
import { clusteredRateInterval, wilsonInterval } from './stats';
import { LANDMARK_SETS, type LandmarkSetId } from './landmark-sets';
import type { Correspondences } from './types';
import { addNoise, DEFAULT_INTRINSICS, makeObject, projectObject, type ObjectKind } from './synthetic';

// Offline analysis of recordings. Used by scripts/analyze-recordings.ts and
// by the tests. The pair-selection rule below is a DESIGN CHOICE that the
// protocol must fix before data are analysed; it is parameterised here, not
// decided.

export interface WindowRuleOptions {
  /** Length of each non-overlapping window, ms. */
  windowMs: number;
}

export interface SelectedPair {
  window: number;
  ref: LandmarkFrame;
  cur: LandmarkFrame;
  motionDeg: number;
}

/**
 * Pair rule "non-overlapping windows": split the recording into consecutive
 * windows of `windowMs`. In each window the reference is the FIRST frame and
 * the partner is the frame with the LARGEST MediaPipe rotation from it.
 * Windows share no frames, so no frame is used twice. The rule looks only
 * at motion, never at residuals. Windows with fewer than 2 frames or without
 * pose estimates are skipped.
 */
export function selectWindowPairs(frames: readonly LandmarkFrame[], opts: WindowRuleOptions): SelectedPair[] {
  if (!(opts.windowMs > 0)) throw new Error('windowMs must be positive');
  const pairs: SelectedPair[] = [];
  if (!frames.length) return pairs;
  const t0 = frames[0].timestampMs;
  const byWindow = new Map<number, LandmarkFrame[]>();
  for (const f of frames) {
    const w = Math.floor((f.timestampMs - t0) / opts.windowMs);
    if (!byWindow.has(w)) byWindow.set(w, []);
    byWindow.get(w)!.push(f);
  }
  for (const [window, fs] of [...byWindow.entries()].sort((a, b) => a[0] - b[0])) {
    if (fs.length < 2) continue;
    const ref = fs[0];
    let best: SelectedPair | null = null;
    for (const f of fs.slice(1)) {
      const m = frameRotationDeg(ref, f);
      if (m === null) continue;
      if (!best || m > best.motionDeg) best = { window, ref, cur: f, motionDeg: m };
    }
    if (best) pairs.push(best);
  }
  return pairs;
}

export interface AnalysisOptions extends WindowRuleOptions {
  indices: readonly number[];
  alpha: number;
  minMotionDeg: number;
  sigmaPx: number | null;
}

export interface PairRow {
  recordingId: string;
  condition: string;
  window: number;
  tRefMs: number;
  tCurMs: number;
  motionDeg: number;
  n: number;
  T: number | null;
  dof: number | null;
  pValue: number | null;
  outcome: string;
  reason: string;
}

export interface RecordingSummary {
  recordingId: string;
  condition: string;
  category: string;
  frames: number;
  pairs: number;
  /** Pairs with enough motion and a statistic. */
  tested: number;
  rejected: number;
  rejectionRate: number | null;
  ci95: [number, number] | null;
  blockedStage: string | null;
}

export function analyzeRecording(rec: Recording, opts: AnalysisOptions, method: PlanarityMethod = DEFAULT_METHOD): { rows: PairRow[]; summary: RecordingSummary } {
  const frames = framesOf(rec);
  const pairs = selectWindowPairs(frames, opts);
  const rows: PairRow[] = [];
  let tested = 0;
  let rejected = 0;
  let blockedStage: string | null = null;
  for (const p of pairs) {
    const a = analyzePair(correspondences(p.ref, p.cur, opts.indices), {
      sigmaPx: opts.sigmaPx,
      alpha: opts.alpha,
      minMotionDeg: opts.minMotionDeg,
      motionDeg: p.motionDeg,
    }, method);
    const blocked = a.stages.find((s) => s.state === 'not-implemented' || s.state === 'error');
    if (blocked && !blockedStage) blockedStage = `${blocked.stage}: ${blocked.message ?? blocked.state}`;
    if (a.stat && p.motionDeg >= opts.minMotionDeg) {
      tested++;
      if (a.stat.pValue < opts.alpha) rejected++;
    }
    rows.push({
      recordingId: rec.meta.id,
      condition: rec.meta.condition,
      window: p.window,
      tRefMs: p.ref.timestampMs,
      tCurMs: p.cur.timestampMs,
      motionDeg: p.motionDeg,
      n: a.n,
      T: a.stat?.T ?? null,
      dof: a.stat?.dof ?? null,
      pValue: a.stat?.pValue ?? null,
      outcome: a.outcome?.kind ?? 'none',
      reason: a.outcome?.reason ?? (blocked ? `${blocked.stage} ${blocked.state}` : 'no decision'),
    });
  }
  return {
    rows,
    summary: {
      recordingId: rec.meta.id,
      condition: rec.meta.condition,
      category: rec.meta.category,
      frames: frames.length,
      pairs: pairs.length,
      tested,
      rejected,
      rejectionRate: tested ? rejected / tested : null,
      ci95: tested ? wilsonInterval(rejected, tested) : null,
      blockedStage,
    },
  };
}

/** Estimate sigma from a calibration recording (object held still). */
export function calibrationFromRecording(
  rec: Recording,
  indices: readonly number[],
  gap: number,
  maxRotationDeg: number,
  method: PlanarityMethod = DEFAULT_METHOD,
): NoiseCalibration & { maxRotationDeg: number; rejectedForMotion: boolean } {
  if (rec.meta.role !== 'calibration') throw new Error(`recording ${rec.meta.id} is not a calibration recording`);
  const built = calibrationPairs(framesOf(rec), indices, gap);
  if (built.maxRotationDeg > maxRotationDeg) {
    return { estimate: null, interval95: null, pairs: built.pairs.length, state: 'error', message: `pose changed by ${built.maxRotationDeg.toFixed(2)}° during the hold (limit ${maxRotationDeg}°)`, maxRotationDeg: built.maxRotationDeg, rejectedForMotion: true };
  }
  return { ...calibrateNoise(built.pairs, method), maxRotationDeg: built.maxRotationDeg, rejectedForMotion: false };
}

const CSV_COLUMNS: (keyof PairRow)[] = ['recordingId', 'condition', 'window', 'tRefMs', 'tCurMs', 'motionDeg', 'n', 'T', 'dof', 'pValue', 'outcome', 'reason'];

export function rowsToCsv(rows: readonly PairRow[]): string {
  const esc = (v: unknown) => {
    if (v === null || v === undefined) return '';
    const s = typeof v === 'number' ? String(Number(v.toPrecision(10))) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [CSV_COLUMNS.join(','), ...rows.map((r) => CSV_COLUMNS.map((c) => esc(r[c])).join(','))].join('\n') + '\n';
}

export interface SyntheticRecordingSpec {
  kind: ObjectKind;
  role: 'calibration' | 'trial';
  /** Peak yaw / pitch of a sinusoidal motion, degrees (0 for a still hold). */
  yawAmplitudeDeg: number;
  pitchAmplitudeDeg: number;
  periodMs: number;
  durationMs: number;
  fps: number;
  distanceCm: number;
  sigmaPx: number;
  seed: number;
}

/**
 * A recording generated from the synthetic scene, in the same format as a
 * real one. Category is always "non-human-target"; condition is
 * "synthetic-<kind>". Lets the full analysis path be exercised before any
 * real data exist. The noise is i.i.d. Gaussian: real landmark noise is not
 * known to be (MATH_SPEC.md, Q1).
 */
export function syntheticRecording(spec: SyntheticRecordingSpec): Recording {
  const rng = new Rng(spec.seed);
  const obj = makeObject(spec.kind);
  const cam = DEFAULT_INTRINSICS;
  const frames: LandmarkFrame[] = [];
  const n = Math.max(2, Math.round((spec.durationMs / 1000) * spec.fps));
  for (let k = 0; k < n; k++) {
    const t = (k * 1000) / spec.fps;
    const phase = (2 * Math.PI * t) / spec.periodMs;
    const yaw = spec.yawAmplitudeDeg * Math.sin(phase);
    const pitch = spec.pitchAmplitudeDeg * Math.sin(phase + Math.PI / 3);
    const pts = addNoise(projectObject(obj, { yawDeg: yaw, pitchDeg: pitch }, spec.distanceCm, cam), spec.sigmaPx, rng);
    frames.push({ timestampMs: t, width: cam.width, height: cam.height, points: pts, rotation: rotationFromAngles(yaw, pitch) });
  }
  const meta: RecordingMeta = {
    format: RECORDING_FORMAT,
    version: RECORDING_VERSION,
    id: `synthetic-${spec.kind}-${spec.role}-${spec.seed}`,
    createdAt: '2026-01-01T00:00:00.000Z',
    appVersion: 'synthetic',
    category: 'non-human-target',
    approvalReference: null,
    pseudonym: null,
    role: spec.role,
    condition: `synthetic-${spec.kind}`,
    protocolId: null,
    notes: `synthetic: yaw ±${spec.yawAmplitudeDeg}°, pitch ±${spec.pitchAmplitudeDeg}°, sigma ${spec.sigmaPx}px, ${spec.distanceCm}cm, seed ${spec.seed}`,
    camera: { label: 'synthetic pinhole', width: cam.width, height: cam.height, frameRate: spec.fps, facingMode: null },
    timestampSource: 'synthetic',
    landmarkModel: 'synthetic (canonical face model)',
    framesWithoutFace: 0,
  };
  return { meta, frames: frames.map((f) => frameToRecorded(f, 0)) };
}

export interface SetAnalysisSettings {
  landmarkSet: LandmarkSetId;
  windowMs: number;
  minMotionDeg: number;
  alpha: number;
  calibrationGap: number;
  maxCalibrationMotionDeg: number;
  /** Use this sigma instead of the calibration recordings. */
  fixedSigmaPx?: number | null;
}

export const DEFAULT_SET_SETTINGS: SetAnalysisSettings = {
  landmarkSet: 'rigid',
  windowMs: 1500,
  minMotionDeg: 5,
  alpha: 0.05,
  calibrationGap: 5,
  maxCalibrationMotionDeg: 1,
  fixedSigmaPx: null,
};

export interface ConditionSummary {
  condition: string;
  recordings: number;
  tested: number;
  rejected: number;
  /** Pooled rate: rejected / tested over all recordings of the condition. */
  pooledRate: number | null;
  /** Wilson interval treating all pairs as independent. Too narrow: pairs from one recording are dependent. */
  pooledCi95: [number, number] | null;
  /** One rate per recording that had at least one tested pair. */
  recordingRates: number[];
  /**
   * Interval for the pooled rate that accounts for dependence within
   * recordings (stats.ts, clusteredRateInterval). Use this one for
   * conclusions. Null with fewer than 2 recordings with tested pairs.
   */
  clusterCi95: [number, number] | null;
  /** Estimated design effect behind clusterCi95 (1 = pairs behave as independent). */
  designEffect: number | null;
}

export interface CalibrationResult {
  source: 'recordings' | 'fixed';
  sigmaPx: number;
  dof: number | null;
  interval95: [number, number] | null;
  calibrationRecordings: number;
  rejectedForMotion: string[];
}

export type SetAnalysis =
  | {
      ok: true;
      settings: SetAnalysisSettings;
      calibration: CalibrationResult;
      summaries: RecordingSummary[];
      rows: PairRow[];
      conditions: ConditionSummary[];
    }
  | { ok: false; error: string };

/**
 * Analyse a whole set of recordings: sigma from all calibration recordings
 * (pooled), every trial with the pair rule, then per-condition summaries.
 */
export function analyzeSet(recordings: readonly Recording[], settings: SetAnalysisSettings, method: PlanarityMethod = DEFAULT_METHOD): SetAnalysis {
  const indices = LANDMARK_SETS[settings.landmarkSet].indices;
  const trials = recordings.filter((r) => r.meta.role === 'trial');
  const cals = recordings.filter((r) => r.meta.role === 'calibration');
  if (!trials.length) return { ok: false, error: 'no trial recordings' };

  let calibration: CalibrationResult;
  if (settings.fixedSigmaPx && settings.fixedSigmaPx > 0) {
    calibration = { source: 'fixed', sigmaPx: settings.fixedSigmaPx, dof: null, interval95: null, calibrationRecordings: 0, rejectedForMotion: [] };
  } else {
    if (!cals.length) return { ok: false, error: 'no calibration recording (hold-still) and no fixed sigma' };
    const pairs: Correspondences[] = [];
    const rejectedForMotion: string[] = [];
    for (const cal of cals) {
      const built = calibrationPairs(framesOf(cal), indices, settings.calibrationGap);
      if (built.maxRotationDeg > settings.maxCalibrationMotionDeg) rejectedForMotion.push(cal.meta.id);
      else pairs.push(...built.pairs);
    }
    if (!pairs.length) return { ok: false, error: 'every calibration recording moved too much during the hold' };
    const c = calibrateNoise(pairs, method);
    if (c.state !== 'ok' || !c.estimate) return { ok: false, error: `calibration failed: ${c.message ?? c.state}` };
    calibration = {
      source: 'recordings',
      sigmaPx: c.estimate.sigmaPx,
      dof: c.estimate.dof,
      interval95: c.interval95,
      calibrationRecordings: cals.length - rejectedForMotion.length,
      rejectedForMotion,
    };
  }

  const opts = { indices, windowMs: settings.windowMs, minMotionDeg: settings.minMotionDeg, alpha: settings.alpha, sigmaPx: calibration.sigmaPx };
  const rows: PairRow[] = [];
  const summaries: RecordingSummary[] = [];
  for (const rec of trials) {
    const r = analyzeRecording(rec, opts, method);
    rows.push(...r.rows);
    summaries.push(r.summary);
  }

  const byCondition = new Map<string, RecordingSummary[]>();
  for (const s of summaries) {
    if (!byCondition.has(s.condition)) byCondition.set(s.condition, []);
    byCondition.get(s.condition)!.push(s);
  }
  // Sorted, so that the output does not depend on the order of the input files.
  const conditions: ConditionSummary[] = [...byCondition.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([condition, ss]) => {
      const tested = ss.reduce((a, s) => a + s.tested, 0);
      const rejected = ss.reduce((a, s) => a + s.rejected, 0);
      const cluster = clusteredRateInterval(
        ss.map((s) => s.rejected),
        ss.map((s) => s.tested),
      );
      return {
        condition,
        recordings: ss.length,
        tested,
        rejected,
        pooledRate: tested ? rejected / tested : null,
        pooledCi95: tested ? wilsonInterval(rejected, tested) : null,
        recordingRates: ss.filter((s) => s.tested > 0).map((s) => s.rejected / s.tested),
        clusterCi95: cluster ? cluster.ci : null,
        designEffect: cluster ? cluster.designEffect : null,
      };
    });
  // A copy: the caller's settings object may change after the run (UI inputs).
  return { ok: true, settings: { ...settings }, calibration, summaries, rows, conditions };
}
