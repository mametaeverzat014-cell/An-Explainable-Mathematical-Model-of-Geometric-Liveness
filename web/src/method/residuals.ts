// Task M2 in web/docs/MATH_SPEC.md: squared Sampson error.
//
// Written by the AI assistant at the project lead's request (2026-10-08),
// see AI_ASSISTANCE.md and web/docs/METHOD_EXPLAINED_RU.md.

import type { Mat3, Point2 } from '../core/types';

/**
 * Squared Sampson error of each correspondence, px² (H&Z section 4.2.6).
 *
 * For one correspondence the algebraic error is the 2-vector
 *
 *   e1 = -(h4 x + h5 y + h6) + y'·w,     e2 = (h1 x + h2 y + h3) - x'·w,
 *   with w = h7 x + h8 y + h9,
 *
 * (the two DLT rows times h). J is its derivative with respect to the four
 * measured coordinates (x, y, x', y'):
 *
 *   ∂e1 = [ -h4 + y' h7,  -h5 + y' h8,   0,   w ]
 *   ∂e2 = [  h1 - x' h7,   h2 - x' h8,  -w,   0 ]
 *
 * and the squared Sampson error is eᵀ (J Jᵀ)⁻¹ e: the squared length of the
 * smallest correction of (x, y, x', y') that makes the linearised constraint
 * hold. Multiplying H by a constant multiplies e and J by the same constant,
 * which cancels, so the result does not depend on the scale of H.
 */
export function sampsonErrorsSquared(h: Mat3, src: readonly Point2[], dst: readonly Point2[]): number[] {
  if (src.length !== dst.length) throw new Error('sampsonErrorsSquared: src and dst differ in length');
  return src.map((p, i) => {
    const q = dst[i];
    const w = h[6] * p.x + h[7] * p.y + h[8];
    const e1 = -(h[3] * p.x + h[4] * p.y + h[5]) + q.y * w;
    const e2 = h[0] * p.x + h[1] * p.y + h[2] - q.x * w;
    const j1 = [-h[3] + q.y * h[6], -h[4] + q.y * h[7], 0, w];
    const j2 = [h[0] - q.x * h[6], h[1] - q.x * h[7], -w, 0];
    // J Jᵀ is the symmetric 2x2 matrix [[a, b], [b, c]].
    const a = j1[0] * j1[0] + j1[1] * j1[1] + j1[2] * j1[2] + j1[3] * j1[3];
    const b = j1[0] * j2[0] + j1[1] * j2[1] + j1[2] * j2[2] + j1[3] * j2[3];
    const c = j2[0] * j2[0] + j2[1] * j2[1] + j2[2] * j2[2] + j2[3] * j2[3];
    const det = a * c - b * b;
    if (!(det > 0)) return Number.POSITIVE_INFINITY;
    // eᵀ [[a, b], [b, c]]⁻¹ e with the inverse written out.
    return (c * e1 * e1 - 2 * b * e1 * e2 + a * e2 * e2) / det;
  });
}
