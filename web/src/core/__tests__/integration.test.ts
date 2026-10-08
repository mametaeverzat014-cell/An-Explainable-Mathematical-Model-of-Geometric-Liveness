// Integration tests of the application layer around the method. They use
// (a) a STUB method whose functions all throw NotImplementedError, to check
// that the app degrades to "not implemented" instead of inventing numbers,
// (b) a FAKE method with deliberately simple, non-scientific behaviour, to
// check the plumbing, and (c) the real method, end to end.
import { describe, expect, it } from 'vitest';
import { NotImplementedError } from '../errors';
import { FrameBuffer, calibrationPairs, correspondences, selectReference, type LandmarkFrame } from '../frames';
import { rotationFromAngles } from '../geometry3d';
import { analyzePair, calibrateNoise, probeMethod, DEFAULT_METHOD, type PlanarityMethod } from '../method';
import { LANDMARK_SETS } from '../landmark-sets';
import { runSweep } from '../montecarlo';

const square = [
  { x: 0, y: 0 },
  { x: 100, y: 0 },
  { x: 100, y: 100 },
  { x: 0, y: 100 },
  { x: 50, y: 50 },
  { x: 20, y: 70 },
];
const corr = { src: square, dst: square.map((p) => ({ x: p.x + 1, y: p.y })) };

const notImplemented = (name: string) => () => {
  throw new NotImplementedError(name);
};
const STUB: PlanarityMethod = {
  estimateHomography: notImplemented('estimateHomography'),
  sampsonErrorsSquared: notImplemented('sampsonErrorsSquared'),
  planarityStatistic: notImplemented('planarityStatistic'),
  decide: notImplemented('decide'),
  estimateNoiseSigma: notImplemented('estimateNoiseSigma'),
  sigmaConfidenceInterval: notImplemented('sigmaConfidenceInterval'),
};

const FAKE: PlanarityMethod = {
  estimateHomography: () => [1, 0, 1, 0, 1, 0, 0, 0, 1],
  sampsonErrorsSquared: (_h, src) => src.map(() => 0.5),
  planarityStatistic: (e, s) => ({ T: e.reduce((a, b) => a + b, 0) / (s * s), dof: 2 * e.length - 8, pValue: 0.01 }),
  decide: ({ stat, motionDeg, minMotionDeg, alpha }) =>
    !stat || motionDeg < minMotionDeg
      ? { kind: 'inconclusive', reason: 'fake' }
      : stat.pValue < alpha
        ? { kind: 'non-planar', reason: 'fake' }
        : { kind: 'planar-consistent', reason: 'fake' },
  estimateNoiseSigma: (pairs) => ({ sigmaPx: 0.7, dof: pairs.length * 4 }),
  sigmaConfidenceInterval: (e) => [e.sigmaPx * 0.9, e.sigmaPx * 1.1],
};

describe('analyzePair with an unimplemented (stub) method', () => {
  it('reports every stage as not implemented / skipped and produces no numbers', () => {
    const a = analyzePair(corr, { sigmaPx: 1, alpha: 0.05, minMotionDeg: 5, motionDeg: 10 }, STUB);
    expect(a.homography).toBeNull();
    expect(a.stat).toBeNull();
    expect(a.outcome).toBeNull();
    expect(a.stages.find((s) => s.stage === 'homography')?.state).toBe('not-implemented');
    expect(a.stages.find((s) => s.stage === 'decision')?.state).toBe('not-implemented');
  });

  it('probeMethod lists all six functions as not implemented', () => {
    const status = probeMethod(STUB);
    expect(status).toHaveLength(6);
    expect(status.every((s) => !s.implemented)).toBe(true);
  });

  it('calibration reports not-implemented', () => {
    expect(calibrateNoise([corr], STUB).state).toBe('not-implemented');
  });

  it('a Monte-Carlo sweep stops with a clear reason', async () => {
    const r = await runSweep({ kinds: ['plane'], rotationsDeg: [5], axis: 'yaw', sigmaPx: 1, distanceCm: 50, indices: [1, 4, 33, 263, 61, 291], trials: 3, alpha: 0.05, seed: 1 }, undefined, STUB);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/homography/);
  });
});

describe('analyzePair plumbing with a fake method', () => {
  it('passes data through all stages and computes display residuals', () => {
    const a = analyzePair(corr, { sigmaPx: 1, alpha: 0.05, minMotionDeg: 5, motionDeg: 10 }, FAKE);
    expect(a.stages.every((s) => s.state === 'ok')).toBe(true);
    expect(a.stat?.dof).toBe(4);
    expect(a.outcome?.kind).toBe('non-planar');
    // fake H translates by +1 in x, dst also shifted by +1: residuals are zero.
    a.residualVectors!.forEach((v) => expect(Math.hypot(v.x, v.y)).toBeCloseTo(0, 12));
  });

  it('skips the statistic (and stays inconclusive) when sigma is not calibrated', () => {
    const a = analyzePair(corr, { sigmaPx: null, alpha: 0.05, minMotionDeg: 5, motionDeg: 10 }, FAKE);
    expect(a.stages.find((s) => s.stage === 'statistic')?.state).toBe('skipped');
    expect(a.outcome?.kind).toBe('inconclusive');
  });

  it('records a genuine error separately from not-implemented', () => {
    const broken: PlanarityMethod = {
      ...FAKE,
      estimateHomography: () => {
        throw new Error('degenerate');
      },
    };
    const a = analyzePair(corr, { sigmaPx: 1, alpha: 0.05, minMotionDeg: 5, motionDeg: 10 }, broken);
    expect(a.stages[0]).toEqual({ stage: 'homography', state: 'error', message: 'degenerate' });
  });

  it('probeMethod distinguishes implemented from unimplemented functions', () => {
    const partial: PlanarityMethod = {
      ...FAKE,
      decide: () => {
        throw new NotImplementedError('decide');
      },
    };
    const s = probeMethod(partial);
    expect(s.find((m) => m.name === 'decide')?.implemented).toBe(false);
    expect(s.find((m) => m.name === 'estimateHomography')?.implemented).toBe(true);
  });

  it('sweep returns rates with Wilson intervals', async () => {
    const r = await runSweep(
      { kinds: ['plane', 'face3d'], rotationsDeg: [0, 5], axis: 'pitch', sigmaPx: 1, distanceCm: 50, indices: [1, 4, 33, 263, 61, 291], trials: 10, alpha: 0.05, seed: 1 },
      undefined,
      FAKE,
    );
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.points).toHaveLength(4);
      for (const p of r.points) {
        expect(p.rate).toBe(1); // fake p-value is always 0.01
        expect(p.ci95[0]).toBeLessThan(1);
      }
    }
  });
});

describe('frame buffer and pair selection', () => {
  const frame = (t: number, yaw: number): LandmarkFrame => ({
    timestampMs: t,
    width: 640,
    height: 480,
    points: Array.from({ length: 468 }, (_, i) => ({ x: i, y: t })),
    rotation: rotationFromAngles(yaw, 0),
  });

  it('keeps a bounded number of frames and filters by time window', () => {
    const buf = new FrameBuffer(5);
    for (let t = 0; t < 10; t++) buf.push(frame(t * 100, 0));
    expect(buf.size).toBe(5);
    expect(buf.within(250).map((f) => f.timestampMs)).toEqual([700, 800, 900]);
  });

  it('selects the reference frame with the largest rotation from the current frame', () => {
    const frames = [frame(0, 2), frame(100, -6), frame(200, 3), frame(300, 4)];
    const current = frames[3];
    const sel = selectReference(frames, current)!;
    expect(sel.ref.timestampMs).toBe(100);
    expect(sel.motionDeg).toBeCloseTo(10, 9);
  });

  it('returns null when no pose is available', () => {
    const old = { ...frame(0, 0), rotation: null };
    const current = { ...frame(100, 0), rotation: null };
    expect(selectReference([old, current], current)).toBeNull();
  });

  it('builds calibration pairs with the requested gap and reports motion during the hold', () => {
    const frames = Array.from({ length: 21 }, (_, i) => frame(i * 33, i === 20 ? 2 : 0));
    const c = calibrationPairs(frames, [1, 2, 3], 5);
    expect(c.pairs).toHaveLength(4);
    expect(c.maxRotationDeg).toBeCloseTo(2, 9);
    expect(correspondences(frames[0], frames[5], [7]).dst[0]).toEqual({ x: 7, y: 165 });
  });
});

describe('the real method, end to end', () => {
  it('probeMethod reports all six functions as implemented', () => {
    expect(probeMethod(DEFAULT_METHOD).every((m) => m.implemented)).toBe(true);
  });

  it('a small sweep separates the flat print from the 3D face', async () => {
    const r = await runSweep({
      kinds: ['plane', 'face3d'],
      rotationsDeg: [8],
      axis: 'yaw',
      sigmaPx: 1,
      distanceCm: 50,
      indices: LANDMARK_SETS.rigid.indices,
      trials: 100,
      alpha: 0.05,
      seed: 5,
    });
    expect(r.ok).toBe(true);
    if (r.ok) {
      const plane = r.points.find((p) => p.kind === 'plane')!;
      const face = r.points.find((p) => p.kind === 'face3d')!;
      expect(plane.rate).toBeLessThan(0.15);
      expect(face.rate).toBeGreaterThan(0.95);
    }
  });
});
