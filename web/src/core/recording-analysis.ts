import { calibrationPairs, correspondences, frameRotationDeg, type LandmarkFrame } from './frames';
import { rotationFromAngles } from './geometry3d';
import { analyzePair, calibrateNoise, STUDENT_METHOD, type NoiseCalibration, type PlanarityMethod } from './method';
import { framesOf, frameToRecorded, RECORDING_FORMAT, RECORDING_VERSION, type Recording, type RecordingMeta } from './recording';
import { Rng } from './rng';
import { wilsonInterval } from './stats';
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

export function analyzeRecording(rec: Recording, opts: AnalysisOptions, method: PlanarityMethod = STUDENT_METHOD): { rows: PairRow[]; summary: RecordingSummary } {
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
  method: PlanarityMethod = STUDENT_METHOD,
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
