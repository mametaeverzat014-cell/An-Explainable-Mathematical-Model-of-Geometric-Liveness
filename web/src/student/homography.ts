// STUDENT-OWNED MODULE — task M1 in web/docs/MATH_SPEC.md.
//
// This file must be written by the student authors. Infrastructure you may
// use: smallestRightSingularVector, mat3Inv, mat3Mul from ../core/linalg.
// Do not copy an implementation from elsewhere; derive it, then make
// `npm run test:student` pass.

import { NotImplementedError } from '../core/errors';
import type { Mat3, Point2 } from '../core/types';

/**
 * Estimate the homography H (3x3, row-major, defined up to scale) such that
 * dst_i ~ H src_i, using the NORMALISED Direct Linear Transform
 * (Hartley & Zisserman, Multiple View Geometry, 2nd ed., Algorithm 4.2).
 *
 * Contract (checked by src/student/tests/homography.test.ts):
 * - src.length === dst.length >= 4, otherwise throw an Error.
 * - Exact (noise-free) correspondences of a planar scene are reproduced:
 *   max_i |H src_i - dst_i| < 1e-6 px for image coordinates up to ~2000 px.
 * - The result does not depend on the order of the correspondences.
 * - If the configuration is degenerate (e.g. all points collinear), throw an Error
 *   instead of returning an arbitrary matrix.
 * - Never return NaN entries.
 */
export function estimateHomography(src: readonly Point2[], dst: readonly Point2[]): Mat3 {
  void src;
  void dst;
  throw new NotImplementedError('estimateHomography');
}
