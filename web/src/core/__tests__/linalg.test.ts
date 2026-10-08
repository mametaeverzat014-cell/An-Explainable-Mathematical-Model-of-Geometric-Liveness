import { describe, expect, it } from 'vitest';
import {
  IDENTITY3,
  applyHomography,
  mat3Det,
  mat3Inv,
  mat3Mul,
  normalizeHomographyScale,
  smallestRightSingularVector,
  symmetricEigen,
} from '../linalg';
import { Rng } from '../rng';

describe('3x3 helpers', () => {
  const m = [2, 1, 0, -1, 3, 2, 0.5, 0, 1];

  it('inverse times matrix is the identity', () => {
    mat3Mul(mat3Inv(m), m).forEach((v, i) => expect(v).toBeCloseTo(IDENTITY3[i], 12));
  });

  it('determinant matches a hand calculation', () => {
    // 2(3*1 - 2*0) - 1(-1*1 - 2*0.5) + 0 = 6 + 2 = 8
    expect(mat3Det(m)).toBeCloseTo(8, 12);
  });

  it('throws on a singular matrix', () => {
    expect(() => mat3Inv([1, 2, 3, 2, 4, 6, 0, 0, 1])).toThrow();
  });

  it('applyHomography handles projective division', () => {
    const p = applyHomography([1, 0, 0, 0, 1, 0, 0.001, 0, 1], { x: 1000, y: 500 });
    expect(p.x).toBeCloseTo(500, 12);
    expect(p.y).toBeCloseTo(250, 12);
  });

  it('normalizeHomographyScale removes scale and sign', () => {
    const a = normalizeHomographyScale(m);
    const b = normalizeHomographyScale(m.map((v) => -7 * v));
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12));
    expect(Math.hypot(...a)).toBeCloseTo(1, 12);
  });
});

describe('symmetricEigen', () => {
  it('diagonalises a random symmetric 9x9 matrix: A v = lambda v, orthonormal vectors', () => {
    const rng = new Rng(3);
    const n = 9;
    const b = Array.from({ length: n }, () => Array.from({ length: n }, () => rng.normal()));
    const a = b.map((_, i) => b.map((_, j) => b.reduce((s, row) => s + row[i] * row[j], 0)));
    const { values, vectors } = symmetricEigen(a);
    for (let k = 0; k < n; k++) {
      const v = vectors[k];
      for (let i = 0; i < n; i++) {
        const av = a[i].reduce((s, aij, j) => s + aij * v[j], 0);
        expect(av).toBeCloseTo(values[k] * v[i], 9);
      }
      for (let l = 0; l < n; l++) {
        const dot = v.reduce((s, vi, i) => s + vi * vectors[l][i], 0);
        expect(dot).toBeCloseTo(k === l ? 1 : 0, 10);
      }
    }
    for (let k = 1; k < n; k++) expect(values[k]).toBeGreaterThanOrEqual(values[k - 1]);
  });

  it('rejects a non-symmetric matrix', () => {
    expect(() => symmetricEigen([[1, 2], [3, 4]])).toThrow();
  });
});

describe('smallestRightSingularVector', () => {
  it('finds the exact null vector of a rank-deficient matrix', () => {
    const x = [1, -2, 0.5];
    const norm = Math.hypot(...x);
    const rows = [
      [2, 1, 0],
      [0, 1, 4],
      [1, 1, 2],
      [3, 2, 2],
    ];
    expect(rows.every((r) => Math.abs(r[0] * x[0] + r[1] * x[1] + r[2] * x[2]) < 1e-12)).toBe(true);
    const { vector, smallestEigenvalue } = smallestRightSingularVector(rows);
    const sign = Math.sign(vector[0]);
    vector.forEach((v, i) => expect(sign * v).toBeCloseTo(x[i] / norm, 10));
    expect(smallestEigenvalue).toBeLessThan(1e-12);
  });
});
