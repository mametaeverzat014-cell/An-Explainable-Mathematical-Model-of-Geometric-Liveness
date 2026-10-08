// Validation of tasks M5 (noise estimate) and M6 (confidence interval), MATH_SPEC.md.
import { describe, expect, it } from 'vitest';
import { expectValidationError } from './helpers';
import { estimateNoiseSigma, sigmaConfidenceInterval } from '@method';
import { LANDMARK_SETS } from '../../core/landmark-sets';
import { Rng } from '../../core/rng';
import { makeObject, syntheticPair } from '../../core/synthetic';
import type { Correspondences } from '../../core/types';

function planarPairs(count: number, sigmaPx: number, rng: Rng): Correspondences[] {
  const plane = makeObject('plane');
  return Array.from({ length: count }, () =>
    syntheticPair(
      {
        object: plane,
        pose0: { yawDeg: rng.between(-10, 10), pitchDeg: rng.between(-10, 10) },
        pose1: { yawDeg: rng.between(-10, 10), pitchDeg: rng.between(-10, 10) },
        distanceCm: 50,
        sigmaPx,
        indices: LANDMARK_SETS.rigid.indices,
      },
      rng,
    ),
  );
}

describe('M5 estimateNoiseSigma', () => {
  it('recovers a known sigma within 5 % from >= 2000 degrees of freedom', () => {
    for (const sigma of [0.4, 1.5]) {
      const est = estimateNoiseSigma(planarPairs(60, sigma, new Rng(31)));
      expect(est.dof).toBe(60 * (2 * 25 - 8));
      expect(Math.abs(est.sigmaPx / sigma - 1)).toBeLessThan(0.05);
    }
  });

  it('throws when no pair has enough correspondences', () => {
    const tiny = [{ src: [{ x: 0, y: 0 }], dst: [{ x: 1, y: 1 }] }];
    expectValidationError(() => estimateNoiseSigma(tiny));
  });
});

describe('M6 sigmaConfidenceInterval', () => {
  it('brackets the estimate and narrows with more degrees of freedom', () => {
    const [l1, u1] = sigmaConfidenceInterval({ sigmaPx: 1, dof: 40 }, 0.95);
    const [l2, u2] = sigmaConfidenceInterval({ sigmaPx: 1, dof: 4000 }, 0.95);
    expect(l1).toBeLessThan(1);
    expect(u1).toBeGreaterThan(1);
    expect(u2 - l2).toBeLessThan(u1 - l1);
  });

  it('has close to nominal coverage on synthetic planar data (1000 replications)', () => {
    const rng = new Rng(32);
    const sigma = 0.9;
    let covered = 0;
    const reps = 1000;
    for (let r = 0; r < reps; r++) {
      const est = estimateNoiseSigma(planarPairs(1, sigma, rng));
      const [lo, hi] = sigmaConfidenceInterval(est, 0.95);
      if (lo <= sigma && sigma <= hi) covered++;
    }
    // Nominal 0.95; Monte-Carlo SE ~ 0.007, so [0.93, 0.97] is ~ +/- 3 SE.
    expect(covered / reps).toBeGreaterThan(0.93);
    expect(covered / reps).toBeLessThan(0.97);
  });
});
