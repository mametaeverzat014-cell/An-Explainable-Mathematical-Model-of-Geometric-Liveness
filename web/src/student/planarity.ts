// STUDENT-OWNED MODULE — tasks M3 and M4 in web/docs/MATH_SPEC.md.
// You may use chiSquareSf / chiSquareQuantile from ../core/stats.

import { NotImplementedError } from '../core/errors';

export interface PlanarityStatistic {
  /** Test statistic T = C / sigma^2, where C is the sum of squared Sampson errors. */
  readonly T: number;
  /** Degrees of freedom of the reference chi-square distribution (2n - 8). */
  readonly dof: number;
  /** P(chi2_dof >= T): probability of a statistic at least this large if H0 (planar) holds. */
  readonly pValue: number;
}

/**
 * Task M3. Build the planarity test statistic from per-point squared errors
 * and the landmark noise level sigma (px).
 *
 * Contract (src/student/tests/planarity.test.ts):
 * - Throw an Error if sigmaPx <= 0, if any error is negative or non-finite,
 *   or if there are too few points for at least one degree of freedom.
 * - dof = 2n - 8 for n correspondences.
 * - pValue = chiSquareSf(T, dof).
 */
export function planarityStatistic(errorsSquared: readonly number[], sigmaPx: number): PlanarityStatistic {
  void errorsSquared;
  void sigmaPx;
  throw new NotImplementedError('planarityStatistic');
}

export type OutcomeKind = 'planar-consistent' | 'non-planar' | 'inconclusive';

export interface Outcome {
  readonly kind: OutcomeKind;
  /** One sentence explaining why, shown to the user. */
  readonly reason: string;
}

export interface DecisionInput {
  /** null when the statistic could not be computed (e.g. noise floor not calibrated). */
  readonly stat: PlanarityStatistic | null;
  /** Significance level of the test, fixed in advance (e.g. 0.05). */
  readonly alpha: number;
  /** Rotation of the presented object between the two frames, degrees. */
  readonly motionDeg: number;
  /** Minimum rotation for which the test is considered informative, degrees. */
  readonly minMotionDeg: number;
}

/**
 * Task M4. Turn the statistic into one of three outcomes. The order of the
 * checks matters scientifically (see MATH_SPEC.md, M4): the motion gate must
 * be applied WITHOUT looking at the p-value.
 *
 * Contract (src/student/tests/planarity.test.ts):
 * - stat === null                  -> 'inconclusive'
 * - motionDeg < minMotionDeg       -> 'inconclusive' (regardless of the p-value)
 * - pValue < alpha                 -> 'non-planar'
 * - otherwise                      -> 'planar-consistent'
 */
export function decide(input: DecisionInput): Outcome {
  void input;
  throw new NotImplementedError('decide');
}
