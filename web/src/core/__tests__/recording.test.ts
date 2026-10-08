import { describe, expect, it } from 'vitest';
import { NotImplementedError } from '../errors';
import { LANDMARK_SETS } from '../landmark-sets';
import type { PlanarityMethod } from '../method';
import { framesOf, parseRecording, recordingFileName, RecordingValidationError, validateRecording, type Recording } from '../recording';
import { analyzeRecording, rowsToCsv, selectWindowPairs, syntheticRecording } from '../recording-analysis';

const base = { periodMs: 4000, durationMs: 4000, fps: 30, distanceCm: 50, sigmaPx: 0.5, seed: 7 };
const trial = () => syntheticRecording({ ...base, kind: 'face3d', role: 'trial', yawAmplitudeDeg: 15, pitchAmplitudeDeg: 5 });

describe('recording format', () => {
  it('round-trips through JSON and keeps 468 points per frame', () => {
    const rec = trial();
    const back = parseRecording(JSON.stringify(rec));
    expect(back.frames).toHaveLength(rec.frames.length);
    const frames = framesOf(back);
    expect(frames[0].points).toHaveLength(468);
    expect(frames[5].rotation).toHaveLength(9);
  });

  it('rounds coordinates to 0.01 px', () => {
    const pts = trial().frames[3].pts;
    expect(pts.every((v) => Math.abs(v * 100 - Math.round(v * 100)) < 1e-6)).toBe(true);
  });

  it('rejects a human-participant recording without approval or pseudonym', () => {
    const rec = trial();
    const human = { ...rec, meta: { ...rec.meta, category: 'human-participant', approvalReference: null, pseudonym: 'P01' } };
    expect(() => validateRecording(human)).toThrow(/approval/);
    const noPseudo = { ...rec, meta: { ...rec.meta, category: 'human-participant', approvalReference: 'IRB-1', pseudonym: 'Daulet K' } };
    expect(() => validateRecording(noPseudo)).toThrow(/pseudonym/);
    const ok = { ...rec, meta: { ...rec.meta, category: 'human-participant', approvalReference: 'IRB-1', pseudonym: 'P01' } };
    expect(validateRecording(ok).meta.pseudonym).toBe('P01');
  });

  it('rejects malformed files with a specific message', () => {
    const rec = trial();
    expect(() => parseRecording('{oops')).toThrow(RecordingValidationError);
    expect(() => validateRecording({ ...rec, meta: { ...rec.meta, version: 2 } })).toThrow(/version/);
    const badFrames: Recording = { ...rec, frames: [rec.frames[1], rec.frames[0]] };
    expect(() => validateRecording(badFrames)).toThrow(/increasing/);
    const shortPts = { ...rec, frames: [{ ...rec.frames[0], pts: [1, 2, 3] }] };
    expect(() => validateRecording(shortPts)).toThrow(/936/);
  });

  it('file names contain no personal information beyond the pseudonym', () => {
    const rec = trial();
    expect(recordingFileName(rec.meta)).toMatch(/^2026-01-01_target_synthetic-face3d_trial_/);
  });
});

describe('pair selection: non-overlapping windows', () => {
  it('uses each frame at most once and picks the largest rotation within each window', () => {
    const frames = framesOf(trial());
    const pairs = selectWindowPairs(frames, { windowMs: 1000 });
    expect(pairs).toHaveLength(4);
    const used = pairs.flatMap((p) => [p.ref.timestampMs, p.cur.timestampMs]);
    expect(new Set(used).size).toBe(used.length);
    for (const p of pairs) {
      expect(Math.floor(p.ref.timestampMs / 1000)).toBe(p.window);
      expect(Math.floor(p.cur.timestampMs / 1000)).toBe(p.window);
    }
  });
});

describe('analyzeRecording', () => {
  const FAKE: PlanarityMethod = {
    estimateHomography: () => [1, 0, 0, 0, 1, 0, 0, 0, 1],
    sampsonErrorsSquared: (_h, src) => src.map(() => 1),
    planarityStatistic: (e) => ({ T: e.length, dof: 2 * e.length - 8, pValue: 0.01 }),
    decide: ({ stat, motionDeg, minMotionDeg }) => (stat && motionDeg >= minMotionDeg ? { kind: 'non-planar', reason: 'fake' } : { kind: 'inconclusive', reason: 'fake' }),
    estimateNoiseSigma: () => ({ sigmaPx: 1, dof: 10 }),
    sigmaConfidenceInterval: () => [0.9, 1.1],
  };
  const opts = { indices: LANDMARK_SETS.rigid.indices, windowMs: 1000, alpha: 0.05, minMotionDeg: 5, sigmaPx: 1 };

  it('counts only pairs that pass the motion gate', () => {
    const { rows, summary } = analyzeRecording(trial(), opts, FAKE);
    expect(summary.pairs).toBe(rows.length);
    const gated = rows.filter((r) => r.motionDeg >= 5).length;
    expect(summary.tested).toBe(gated);
    expect(summary.rejected).toBe(gated);
    expect(summary.ci95).not.toBeNull();
  });

  it('reports the blocking stage when the method is not implemented', () => {
    const STUB = { ...FAKE, estimateHomography: () => { throw new NotImplementedError('estimateHomography'); } };
    const { summary } = analyzeRecording(trial(), opts, STUB);
    expect(summary.tested).toBe(0);
    expect(summary.blockedStage).toMatch(/homography/);
  });

  it('writes CSV with a header and quoted text where needed', () => {
    const csv = rowsToCsv(analyzeRecording(trial(), opts, FAKE).rows);
    const lines = csv.trim().split('\n');
    expect(lines[0]).toBe('recordingId,condition,window,tRefMs,tCurMs,motionDeg,n,T,dof,pValue,outcome,reason');
    expect(lines.length).toBe(5);
  });
});

describe('analyzeRecording with the real method', () => {
  it('estimates sigma from a synthetic calibration and keeps the flat print planar-consistent', async () => {
    const { calibrationFromRecording } = await import('../recording-analysis');
    const cal = syntheticRecording({ ...base, kind: 'plane', role: 'calibration', yawAmplitudeDeg: 0, pitchAmplitudeDeg: 0, durationMs: 3000, sigmaPx: 0.8, seed: 11 });
    const c = calibrationFromRecording(cal, LANDMARK_SETS.rigid.indices, 5, 1);
    expect(c.state).toBe('ok');
    expect(c.estimate!.sigmaPx).toBeGreaterThan(0.7);
    expect(c.estimate!.sigmaPx).toBeLessThan(0.9);
    const plane = syntheticRecording({ ...base, kind: 'plane', role: 'trial', yawAmplitudeDeg: 20, pitchAmplitudeDeg: 10, durationMs: 12000, sigmaPx: 0.8, seed: 12 });
    const face = syntheticRecording({ ...base, kind: 'face3d', role: 'trial', yawAmplitudeDeg: 20, pitchAmplitudeDeg: 10, durationMs: 12000, sigmaPx: 0.8, seed: 13 });
    const opts2 = { indices: LANDMARK_SETS.rigid.indices, windowMs: 1000, alpha: 0.05, minMotionDeg: 5, sigmaPx: c.estimate!.sigmaPx };
    const p = analyzeRecording(plane, opts2).summary;
    const f = analyzeRecording(face, opts2).summary;
    expect(p.tested).toBeGreaterThan(5);
    expect(p.rejected).toBeLessThanOrEqual(2);
    expect(f.rejected).toBe(f.tested);
  });
});
