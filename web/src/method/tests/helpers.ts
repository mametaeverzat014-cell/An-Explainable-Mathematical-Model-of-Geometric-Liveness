// Shared fixtures for the method validation suite. Infrastructure only:
// these helpers generate data with KNOWN properties; they do not contain any
// part of the method under test.
import { isNotImplemented } from '../../core/errors';
import { applyHomography } from '../../core/linalg';
import { Rng } from '../../core/rng';
import type { Mat3, Point2 } from '../../core/types';

/** A fixed, moderately projective homography used as ground truth in several tests. */
export const H_TRUE: Mat3 = [1.02, 0.08, 35, -0.05, 0.97, -12, 1.5e-4, -8e-5, 1];

/** n random points spread over a w x h image (not collinear with probability 1). */
export function randomPoints(n: number, rng: Rng, w = 1280, h = 720): Point2[] {
  return Array.from({ length: n }, () => ({ x: rng.between(0.1 * w, 0.9 * w), y: rng.between(0.1 * h, 0.9 * h) }));
}

export function mapAll(h: Mat3, pts: readonly Point2[]): Point2[] {
  return pts.map((p) => applyHomography(h, p));
}

/** Largest |H src_i - dst_i| over all correspondences, in px. */
export function maxTransferError(h: Mat3, src: readonly Point2[], dst: readonly Point2[]): number {
  let m = 0;
  src.forEach((p, i) => {
    const q = applyHomography(h, p);
    m = Math.max(m, Math.hypot(q.x - dst[i].x, q.y - dst[i].y));
  });
  return m;
}

/**
 * Assert that fn throws a genuine input-validation error. A NotImplementedError
 * does not count, so these tests cannot pass against the unimplemented stubs.
 */
export function expectValidationError(fn: () => unknown): void {
  let thrown: unknown = null;
  try {
    fn();
  } catch (error) {
    thrown = error;
  }
  if (thrown === null) throw new Error('expected an error to be thrown, but none was');
  if (isNotImplemented(thrown)) throw new Error(`function is not implemented yet: ${(thrown as Error).message}`);
}
