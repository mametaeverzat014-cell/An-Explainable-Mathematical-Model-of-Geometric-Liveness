import { describe, expect, it } from 'vitest';
import {
  chiSquareCdf,
  chiSquarePdf,
  chiSquareQuantile,
  chiSquareSf,
  logGamma,
  median,
  normalQuantile,
  quantile,
  sampleVariance,
  wilsonInterval,
} from '../stats';

// Reference values: standard chi-square and normal tables (any statistics
// textbook appendix), and Gamma function identities.
describe('logGamma', () => {
  it('matches log((n-1)!) for integers and Gamma(1/2) = sqrt(pi)', () => {
    expect(logGamma(1)).toBeCloseTo(0, 12);
    expect(logGamma(5)).toBeCloseTo(Math.log(24), 12);
    expect(logGamma(11)).toBeCloseTo(Math.log(3628800), 10);
    expect(logGamma(0.5)).toBeCloseTo(0.5 * Math.log(Math.PI), 12);
  });
});

describe('chi-square distribution', () => {
  const table: [number, number, number][] = [
    // [dof, upper-tail probability, critical value]
    [1, 0.05, 3.841459],
    [2, 0.05, 5.991465],
    [10, 0.05, 18.307038],
    [10, 0.01, 23.209251],
    [30, 0.05, 43.772972],
    [42, 0.05, 58.124038],
    [100, 0.05, 124.342113],
  ];

  it.each(table)('dof=%i: P(X > crit) = %f', (k, tail, crit) => {
    expect(chiSquareSf(crit, k)).toBeCloseTo(tail, 6);
    expect(chiSquareCdf(crit, k)).toBeCloseTo(1 - tail, 6);
    expect(chiSquareQuantile(1 - tail, k)).toBeCloseTo(crit, 4);
  });

  it('dof = 2 has the closed form CDF 1 - exp(-x/2)', () => {
    for (const x of [0.1, 1, 5, 20]) expect(chiSquareCdf(x, 2)).toBeCloseTo(1 - Math.exp(-x / 2), 12);
  });

  it('survival function stays accurate far in the tail', () => {
    // P(X > 100) for dof = 2 is exp(-50).
    expect(chiSquareSf(100, 2) / Math.exp(-50)).toBeCloseTo(1, 8);
  });

  it('pdf integrates to ~1', () => {
    const k = 12;
    let s = 0;
    const dx = 0.01;
    for (let x = dx / 2; x < 100; x += dx) s += chiSquarePdf(x, k) * dx;
    expect(s).toBeCloseTo(1, 4);
  });
});

describe('normal quantile and Wilson interval', () => {
  it('normal quantiles match the table', () => {
    expect(normalQuantile(0.975)).toBeCloseTo(1.959964, 5);
    expect(normalQuantile(0.5)).toBeCloseTo(0, 9);
    expect(normalQuantile(0.001)).toBeCloseTo(-3.090232, 5);
  });

  it('Wilson interval matches a worked example (k = 5, n = 100)', () => {
    // Hand calculation with z = 1.959964: [0.02154, 0.11175]
    const [lo, hi] = wilsonInterval(5, 100);
    expect(lo).toBeCloseTo(0.02154, 4);
    expect(hi).toBeCloseTo(0.11175, 4);
  });

  it('Wilson interval stays inside [0, 1] for 0 and n successes', () => {
    const [l0, u0] = wilsonInterval(0, 20);
    const [l1, u1] = wilsonInterval(20, 20);
    expect(l0).toBe(0);
    expect(u0).toBeGreaterThan(0);
    expect(l1).toBeLessThan(1);
    expect(u1).toBe(1);
  });
});

describe('descriptive statistics', () => {
  it('median, quantile and variance', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(quantile([1, 2, 3, 4, 5], 0.25)).toBe(2);
    expect(sampleVariance([1, 2, 3, 4])).toBeCloseTo(5 / 3, 12);
  });
});

describe('bootstrapMeanInterval', () => {
  it('is reproducible, brackets the mean, and is null for a single value', async () => {
    const { bootstrapMeanInterval } = await import('../stats');
    const v = [0.1, 0.0, 0.05, 0.2, 0.0, 0.1, 0.05, 0.15];
    const a = bootstrapMeanInterval(v, 1)!;
    expect(bootstrapMeanInterval(v, 1)).toEqual(a);
    const m = v.reduce((x, y) => x + y, 0) / v.length;
    expect(a[0]).toBeLessThan(m);
    expect(a[1]).toBeGreaterThan(m);
    expect(bootstrapMeanInterval([0.3], 1)).toBeNull();
    expect(bootstrapMeanInterval([0, 0, 0], 1)).toEqual([0, 0]);
  });
});
