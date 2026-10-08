// STUDENT-OWNED MODULE — tasks M5 and M6 in web/docs/MATH_SPEC.md.

import { NotImplementedError } from '../core/errors';
import type { Correspondences } from '../core/types';

export interface NoiseEstimate {
  /** Estimated landmark noise standard deviation per coordinate, px. */
  readonly sigmaPx: number;
  /** Total degrees of freedom the estimate is based on (sum of 2n_k - 8). */
  readonly dof: number;
}

/**
 * Task M5. Estimate the landmark noise level sigma from frame pairs recorded
 * while the presented object is held still (calibration).
 *
 * Method to derive (MATH_SPEC.md, M5): pooled estimator
 *   sigma^2 = (sum_k C_k) / (sum_k (2 n_k - 8)),
 * where C_k is the sum of squared Sampson errors of pair k after fitting a
 * homography to that pair.
 *
 * Contract (src/student/tests/noise.test.ts):
 * - Throws an Error if no pair has at least 5 correspondences.
 * - On synthetic planar pairs with known sigma, the estimate is within 5 %
 *   of the truth when based on >= 2000 degrees of freedom.
 */
export function estimateNoiseSigma(pairs: readonly Correspondences[]): NoiseEstimate {
  void pairs;
  throw new NotImplementedError('estimateNoiseSigma');
}

/**
 * Task M6. Two-sided confidence interval [lower, upper] for sigma given an
 * estimate and its degrees of freedom, under the assumptions of M5.
 *
 * Contract: lower < sigmaPx < upper; the interval narrows as dof grows;
 * coverage on synthetic data is close to `confidence` (tested by simulation).
 */
export function sigmaConfidenceInterval(estimate: NoiseEstimate, confidence: number): [number, number] {
  void estimate;
  void confidence;
  throw new NotImplementedError('sigmaConfidenceInterval');
}
