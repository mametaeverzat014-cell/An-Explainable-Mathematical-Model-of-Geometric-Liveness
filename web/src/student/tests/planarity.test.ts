// Validation of tasks M3 (statistic) and M4 (decision rule), MATH_SPEC.md.
import { describe, expect, it } from 'vitest';
import { expectValidationError } from './helpers';
import { decide, planarityStatistic, type PlanarityStatistic } from '@student';
import { chiSquareSf } from '../../core/stats';

describe('M3 planarityStatistic', () => {
  it('computes T = sum(e^2) / sigma^2 with dof = 2n - 8', () => {
    const e2 = [1, 2, 0.5, 3, 0.25, 1.25]; // n = 6, sum = 8
    const s = planarityStatistic(e2, 2);
    expect(s.T).toBeCloseTo(2, 12);
    expect(s.dof).toBe(4);
    expect(s.pValue).toBeCloseTo(chiSquareSf(2, 4), 12);
  });

  it('p-value is 1 for a perfect fit and decreases as T grows', () => {
    const e = Array(10).fill(0);
    expect(planarityStatistic(e, 1).pValue).toBeCloseTo(1, 12);
    const p1 = planarityStatistic(Array(10).fill(1), 1).pValue;
    const p2 = planarityStatistic(Array(10).fill(2), 1).pValue;
    expect(p2).toBeLessThan(p1);
  });

  it('rejects invalid input', () => {
    expectValidationError(() => planarityStatistic([1, 1, 1, 1, 1], 0));
    expectValidationError(() => planarityStatistic([1, 1, 1, 1, 1], -1));
    expectValidationError(() => planarityStatistic([1, -1, 1, 1, 1], 1));
    expectValidationError(() => planarityStatistic([1, Number.NaN, 1, 1, 1], 1));
    expectValidationError(() => planarityStatistic([1, 1, 1, 1], 1)); // n = 4 -> dof = 0
  });
});

describe('M4 decide', () => {
  const stat = (pValue: number): PlanarityStatistic => ({ T: 1, dof: 10, pValue });
  const base = { alpha: 0.05, motionDeg: 10, minMotionDeg: 5 };

  it('returns non-planar when p < alpha and motion is sufficient', () => {
    expect(decide({ ...base, stat: stat(0.01) }).kind).toBe('non-planar');
  });

  it('returns planar-consistent when p >= alpha and motion is sufficient', () => {
    expect(decide({ ...base, stat: stat(0.2) }).kind).toBe('planar-consistent');
    expect(decide({ ...base, stat: stat(0.05) }).kind).toBe('planar-consistent');
  });

  it('returns inconclusive when motion is insufficient, whatever the p-value', () => {
    for (const p of [1e-12, 0.01, 0.5, 1]) {
      expect(decide({ ...base, motionDeg: 2, stat: stat(p) }).kind).toBe('inconclusive');
    }
  });

  it('returns inconclusive when no statistic is available', () => {
    expect(decide({ ...base, stat: null }).kind).toBe('inconclusive');
  });

  it('always gives a non-empty reason', () => {
    for (const s of [null, stat(0.01), stat(0.5)]) {
      for (const motionDeg of [1, 10]) {
        expect(decide({ ...base, motionDeg, stat: s }).reason.length).toBeGreaterThan(0);
      }
    }
  });
});
