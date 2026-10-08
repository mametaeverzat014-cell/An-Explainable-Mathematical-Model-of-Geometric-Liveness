// STUDENT-OWNED MODULE — task M2 in web/docs/MATH_SPEC.md.

import { NotImplementedError } from '../core/errors';
import type { Mat3, Point2 } from '../core/types';

/**
 * Squared Sampson error of each correspondence with respect to H, in px^2
 * (Hartley & Zisserman, 2nd ed., section 4.2.6). It is the first-order
 * approximation of the squared geometric distance by which the two points
 * (src_i AND dst_i, both noisy) must be moved so that dst_i = H src_i holds
 * exactly.
 *
 * Contract (checked by src/student/tests/residuals.test.ts):
 * - Returns an array with one non-negative number per correspondence.
 * - Invariant to the scale of H (H and 3H give identical results).
 * - For H = identity, the error of (p, p + d) equals |d|^2 / 2 exactly.
 * - Equals 0 (to 1e-12 relative) for correspondences satisfying dst = H src exactly.
 */
export function sampsonErrorsSquared(h: Mat3, src: readonly Point2[], dst: readonly Point2[]): number[] {
  void h;
  void src;
  void dst;
  throw new NotImplementedError('sampsonErrorsSquared');
}
