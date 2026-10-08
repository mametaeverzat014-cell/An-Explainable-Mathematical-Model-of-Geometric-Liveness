// End-to-end statistical validation of the method on the IDEAL synthetic
// scene (pinhole camera, i.i.d. Gaussian landmark noise of known sigma).
// Passing these tests shows the implementation is consistent with the
// theory under its assumptions. It says NOTHING about real MediaPipe
// landmarks, whose noise is not i.i.d. Gaussian (MATH_SPEC.md, Q1).
import { describe, expect, it } from 'vitest';
import { estimateHomography, planarityStatistic, sampsonErrorsSquared } from '@student';
import { LANDMARK_SETS } from '../../core/landmark-sets';
import { Rng } from '../../core/rng';
import { makeObject, syntheticPair, type ObjectKind } from '../../core/synthetic';

function rejectionRate(kind: ObjectKind, yawDeg: number, pitchDeg: number, sigmaPx: number, trials: number, seed: number) {
  const rng = new Rng(seed);
  const obj = makeObject(kind);
  let rejected = 0;
  let sumTOverDof = 0;
  for (let i = 0; i < trials; i++) {
    const c = syntheticPair(
      { object: obj, pose0: { yawDeg: 0, pitchDeg: 0 }, pose1: { yawDeg, pitchDeg }, distanceCm: 50, sigmaPx, indices: LANDMARK_SETS.rigid.indices },
      rng,
    );
    const s = planarityStatistic(sampsonErrorsSquared(estimateHomography(c.src, c.dst), c.src, c.dst), sigmaPx);
    if (s.pValue < 0.05) rejected++;
    sumTOverDof += s.T / s.dof;
  }
  return { rate: rejected / trials, meanTOverDof: sumTOverDof / trials };
}

describe('Monte-Carlo validation on the ideal synthetic scene', () => {
  it('size: a moving plane is rejected at close to the nominal 5 % rate', () => {
    const { rate, meanTOverDof } = rejectionRate('plane', 15, 5, 1, 2000, 41);
    // Binomial SE at p = 0.05, n = 2000 is ~0.005; [0.03, 0.07] is ~ +/- 4 SE.
    expect(rate).toBeGreaterThan(0.03);
    expect(rate).toBeLessThan(0.07);
    expect(meanTOverDof).toBeGreaterThan(0.95);
    expect(meanTOverDof).toBeLessThan(1.05);
  });

  it('power: the 3D canonical face turning 5 degrees is rejected in > 95 % of trials (sigma = 1 px, 50 cm)', () => {
    expect(rejectionRate('face3d', 5, 0, 1, 300, 42).rate).toBeGreaterThan(0.95);
    expect(rejectionRate('face3d', 0, 5, 1, 300, 43).rate).toBeGreaterThan(0.95);
  });

  it('no motion: the 3D face without rotation behaves like the null (nothing to detect)', () => {
    const { rate } = rejectionRate('face3d', 0, 0, 1, 1000, 44);
    expect(rate).toBeLessThan(0.08);
  });
});
