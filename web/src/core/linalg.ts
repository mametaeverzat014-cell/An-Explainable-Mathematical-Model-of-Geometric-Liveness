import type { Mat3, Point2 } from './types';

// General numerical routines. These are infrastructure, not the research
// method: the method (how the homography is estimated, which error is
// minimised, how the test is built) lives in src/method/.

export const IDENTITY3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mat3Mul(a: Mat3, b: Mat3): number[] {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

export function mat3Det(m: Mat3): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

export function mat3Inv(m: Mat3): number[] {
  const det = mat3Det(m);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-300) {
    throw new Error('mat3Inv: matrix is singular');
  }
  const inv = [
    m[4] * m[8] - m[5] * m[7],
    m[2] * m[7] - m[1] * m[8],
    m[1] * m[5] - m[2] * m[4],
    m[5] * m[6] - m[3] * m[8],
    m[0] * m[8] - m[2] * m[6],
    m[2] * m[3] - m[0] * m[5],
    m[3] * m[7] - m[4] * m[6],
    m[1] * m[6] - m[0] * m[7],
    m[0] * m[4] - m[1] * m[3],
  ];
  return inv.map((v) => v / det);
}

export function mat3Transpose(m: Mat3): number[] {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

/** Scale a homography so that its Frobenius norm is 1 and its largest-magnitude entry is positive. */
export function normalizeHomographyScale(h: Mat3): number[] {
  const norm = Math.hypot(...h);
  if (!(norm > 0)) throw new Error('normalizeHomographyScale: zero matrix');
  let maxIdx = 0;
  for (let i = 1; i < 9; i++) if (Math.abs(h[i]) > Math.abs(h[maxIdx])) maxIdx = i;
  const sign = h[maxIdx] < 0 ? -1 : 1;
  return h.map((v) => (sign * v) / norm);
}

/** Map an inhomogeneous point through a homography: x' ~ H x. */
export function applyHomography(h: Mat3, p: Point2): Point2 {
  const w = h[6] * p.x + h[7] * p.y + h[8];
  if (Math.abs(w) < 1e-300) return { x: Number.NaN, y: Number.NaN };
  return {
    x: (h[0] * p.x + h[1] * p.y + h[2]) / w,
    y: (h[3] * p.x + h[4] * p.y + h[5]) / w,
  };
}

/** Euclidean distances |H src_i - dst_i| in pixels (one-image transfer error). */
export function transferDistances(h: Mat3, src: readonly Point2[], dst: readonly Point2[]): number[] {
  if (src.length !== dst.length) throw new Error('transferDistances: length mismatch');
  return src.map((p, i) => {
    const q = applyHomography(h, p);
    return Math.hypot(q.x - dst[i].x, q.y - dst[i].y);
  });
}

/**
 * Eigen-decomposition of a real symmetric matrix by the cyclic Jacobi method.
 * Returns eigenvalues in ascending order with matching unit eigenvectors
 * (vectors[k] is the k-th eigenvector).
 *
 * Accurate to near machine precision for the small (<= 12x12) matrices used
 * here. Input is not modified.
 */
export function symmetricEigen(input: readonly (readonly number[])[]): {
  values: number[];
  vectors: number[][];
} {
  const n = input.length;
  for (const row of input) {
    if (row.length !== n) throw new Error('symmetricEigen: matrix must be square');
  }
  const a = input.map((row) => row.slice());
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const scale = Math.max(1, Math.abs(a[i][j]), Math.abs(a[j][i]));
      if (Math.abs(a[i][j] - a[j][i]) > 1e-9 * scale) {
        throw new Error('symmetricEigen: matrix is not symmetric');
      }
    }
  }
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-300) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => v.map((row) => row[i])),
  };
}

/**
 * Unit vector x minimising |A x| subject to |x| = 1, for a tall matrix A
 * given by its rows. Computed as the eigenvector of A^T A with the smallest
 * eigenvalue. Also returns that eigenvalue (= min |A x|^2) and the second
 * smallest one, whose ratio indicates how well-determined the solution is.
 *
 * Note: forming A^T A squares the condition number of A. With well-scaled
 * (normalised) data this is harmless in double precision; with unnormalised
 * pixel coordinates it is not. See MATH_SPEC.md, task M1.
 */
export function smallestRightSingularVector(rows: readonly (readonly number[])[]): {
  vector: number[];
  smallestEigenvalue: number;
  secondSmallestEigenvalue: number;
} {
  if (rows.length === 0) throw new Error('smallestRightSingularVector: no rows');
  const m = rows[0].length;
  const ata = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  for (const row of rows) {
    if (row.length !== m) throw new Error('smallestRightSingularVector: ragged rows');
    for (let i = 0; i < m; i++) {
      const ri = row[i];
      if (ri === 0) continue;
      for (let j = i; j < m; j++) ata[i][j] += ri * row[j];
    }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) ata[i][j] = ata[j][i];
  const { values, vectors } = symmetricEigen(ata);
  return {
    vector: vectors[0],
    smallestEigenvalue: values[0],
    secondSmallestEigenvalue: values.length > 1 ? values[1] : Number.NaN,
  };
}
