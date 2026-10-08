// Task M1 in web/docs/MATH_SPEC.md: normalised DLT homography estimation.
//
// Written by the AI assistant at the project lead's request (2026-10-08),
// see AI_ASSISTANCE.md. The authors must be able to derive and explain every
// step: web/docs/METHOD_EXPLAINED_RU.md walks through it.

import { mat3Inv, mat3Mul, smallestRightSingularVector } from '../core/linalg';
import type { Mat3, Point2 } from '../core/types';

/**
 * Hartley normalisation: a similarity transform T that moves the centroid of
 * the points to the origin and scales them so the mean distance from the
 * origin is sqrt(2). Without it, the columns of the DLT matrix differ in
 * size by ~10^6 (pixel coordinates ~1000 vs. the constant 1) and the
 * solution becomes numerically fragile (Hartley, 1997).
 */
function normalizingTransform(points: readonly Point2[]): number[] {
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.x;
    cy += p.y;
  }
  cx /= points.length;
  cy /= points.length;
  let meanDist = 0;
  for (const p of points) meanDist += Math.hypot(p.x - cx, p.y - cy);
  meanDist /= points.length;
  if (!(meanDist > 0)) throw new Error('estimateHomography: all points coincide');
  const s = Math.SQRT2 / meanDist;
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
}

function applyAffine(t: readonly number[], p: Point2): Point2 {
  return { x: t[0] * p.x + t[1] * p.y + t[2], y: t[3] * p.x + t[4] * p.y + t[5] };
}

/**
 * Below this ratio between the second-smallest and the largest eigenvalue of
 * AᵀA, the null space is (numerically) more than one-dimensional: infinitely
 * many homographies fit equally well, so the configuration is degenerate
 * (for example, all points on one line). With normalised data the
 * eigenvalues of a healthy problem are O(1)..O(n); 1e-10 is far below any
 * well-posed case and far above double-precision round-off.
 */
const DEGENERACY_RATIO = 1e-10;

/**
 * Estimate H with dst_i ~ H src_i (Hartley & Zisserman, Algorithm 4.2).
 *
 * Each correspondence x = (x, y, 1) -> x' = (x', y', 1) gives, from
 * x' × (H x) = 0, two linear equations in the 9 entries h of H:
 *
 *   [ 0  0  0   -x  -y  -1   y'x  y'y  y' ] · h = 0
 *   [ x  y  1    0   0   0  -x'x -x'y -x' ] · h = 0
 *
 * Stacking them gives A h = 0. With noise there is no exact solution, so h
 * is the unit vector minimising |A h|: the eigenvector of AᵀA with the
 * smallest eigenvalue.
 */
export function estimateHomography(src: readonly Point2[], dst: readonly Point2[]): Mat3 {
  if (src.length !== dst.length) throw new Error('estimateHomography: src and dst differ in length');
  if (src.length < 4) throw new Error('estimateHomography: at least 4 correspondences are needed');

  // 1. Normalise each image separately.
  const T = normalizingTransform(src);
  const Tp = normalizingTransform(dst);

  // 2. Build A (2n x 9) from the normalised points.
  const rows: number[][] = [];
  for (let i = 0; i < src.length; i++) {
    const a = applyAffine(T, src[i]);
    const b = applyAffine(Tp, dst[i]);
    rows.push([0, 0, 0, -a.x, -a.y, -1, b.y * a.x, b.y * a.y, b.y]);
    rows.push([a.x, a.y, 1, 0, 0, 0, -b.x * a.x, -b.x * a.y, -b.x]);
  }

  // 3. Unit h minimising |A h|, with a check that it is unique.
  const { vector, secondSmallestEigenvalue } = smallestRightSingularVector(rows);
  let largest = 0;
  for (const r of rows) for (const v of r) largest = Math.max(largest, v * v);
  if (!(secondSmallestEigenvalue > DEGENERACY_RATIO * largest * rows.length)) {
    throw new Error('estimateHomography: degenerate configuration (e.g. collinear points)');
  }

  // 4. Undo the normalisation: H = T'^-1 · H~ · T.
  const h = mat3Mul(mat3Inv(Tp), mat3Mul(vector, T));
  if (h.some((v) => !Number.isFinite(v))) throw new Error('estimateHomography: non-finite result');
  return h;
}
