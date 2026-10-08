// Validation of task M2 (MATH_SPEC.md): Sampson error.
// The expected values below are derived in MATH_SPEC.md, M2, "check cases":
// for constraints that are LINEAR in the point coordinates the Sampson
// approximation is exact, which gives closed-form answers to test against.
import { describe, expect, it } from 'vitest';
import { sampsonErrorsSquared } from '@method';
import { IDENTITY3 } from '../../core/linalg';
import { Rng } from '../../core/rng';
import { H_TRUE, mapAll, randomPoints } from './helpers';

describe('M2 sampsonErrorsSquared', () => {
  it('returns one value per correspondence', () => {
    const src = randomPoints(7, new Rng(21));
    expect(sampsonErrorsSquared(H_TRUE, src, mapAll(H_TRUE, src))).toHaveLength(7);
  });

  it('is zero for exact correspondences', () => {
    const src = randomPoints(20, new Rng(22));
    const e = sampsonErrorsSquared(H_TRUE, src, mapAll(H_TRUE, src));
    e.forEach((v) => expect(Math.abs(v)).toBeLessThan(1e-12));
  });

  it('equals |d|^2 / 2 for H = identity', () => {
    const src = [
      { x: 10, y: 20 },
      { x: 300, y: 40 },
      { x: -5, y: 7 },
    ];
    const d = [
      { x: 3, y: 4 },
      { x: -1, y: 0 },
      { x: 0.25, y: -0.5 },
    ];
    const dst = src.map((p, i) => ({ x: p.x + d[i].x, y: p.y + d[i].y }));
    const e = sampsonErrorsSquared(IDENTITY3, src, dst);
    d.forEach((di, i) => expect(e[i]).toBeCloseTo((di.x * di.x + di.y * di.y) / 2, 12));
  });

  it('equals (s x - x\')^2/(1+s^2) + (s y - y\')^2/(1+s^2) for H = diag(s, s, 1)', () => {
    const s = 1.7;
    const h = [s, 0, 0, 0, s, 0, 0, 0, 1];
    const src = [
      { x: 12, y: -3 },
      { x: 100, y: 250 },
    ];
    const dst = [
      { x: 21, y: -4.6 },
      { x: 168, y: 427 },
    ];
    const e = sampsonErrorsSquared(h, src, dst);
    src.forEach((p, i) => {
      const expected = ((s * p.x - dst[i].x) ** 2 + (s * p.y - dst[i].y) ** 2) / (1 + s * s);
      expect(e[i]).toBeCloseTo(expected, 10);
    });
  });

  it('is invariant to the scale of H', () => {
    const rng = new Rng(23);
    const src = randomPoints(10, rng);
    const dst = mapAll(H_TRUE, src).map((p) => ({ x: p.x + rng.normal(), y: p.y + rng.normal() }));
    const e1 = sampsonErrorsSquared(H_TRUE, src, dst);
    const e2 = sampsonErrorsSquared(H_TRUE.map((v) => -3 * v), src, dst);
    e1.forEach((v, i) => expect(e2[i]).toBeCloseTo(v, 10));
  });

  it('is non-negative and has mean 2 sigma^2 when both points carry N(0, sigma^2) noise', () => {
    // For a fixed, exact H the correspondence (x, y, x', y') must lie on a
    // surface of codimension 2 in R^4. Isotropic noise has 2 components
    // normal to that surface, so E[e^2] ~= 2 sigma^2 to first order
    // (MATH_SPEC.md, M2, check case 3).
    const rng = new Rng(24);
    const sigma = 0.8;
    const n = 20000;
    const src = randomPoints(n, rng);
    const dst = mapAll(H_TRUE, src);
    const noisy = (p: { x: number; y: number }) => ({ x: p.x + sigma * rng.normal(), y: p.y + sigma * rng.normal() });
    const e = sampsonErrorsSquared(H_TRUE, src.map(noisy), dst.map(noisy));
    expect(e.every((v) => v >= 0)).toBe(true);
    const meanE = e.reduce((a, b) => a + b, 0) / n;
    expect(meanE / (2 * sigma * sigma)).toBeGreaterThan(0.97);
    expect(meanE / (2 * sigma * sigma)).toBeLessThan(1.03);
  });
});
