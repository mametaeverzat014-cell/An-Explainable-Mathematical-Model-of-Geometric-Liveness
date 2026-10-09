// Tasks M3 and M4 in web/docs/MATH_SPEC.md: test statistic and decision rule.
//
// Written by the AI assistant at the project lead's request (2026-10-08),
// see AI_ASSISTANCE.md and web/docs/METHOD_EXPLAINED_RU.md.

import { chiSquareSf } from '../core/stats';

export interface PlanarityStatistic {
  /** Test statistic T = C / sigma^2, where C is the sum of squared Sampson errors. */
  readonly T: number;
  /** Degrees of freedom of the reference chi-square distribution (2n - 8). */
  readonly dof: number;
  /** P(chi2_dof >= T): probability of a statistic at least this large if H0 (planar) holds. */
  readonly pValue: number;
}

/**
 * T = Σ e_i² / σ², dof = 2n − 8, p = P(χ²_dof ≥ T).
 *
 * Why 2n − 8: the n correspondences contain 4n measured coordinates. Under
 * H0 the fit estimates 2n + 8 numbers (the n "true" point positions in the
 * first image, which fix those in the second, plus the 8 degrees of freedom
 * of H). What is left over, 4n − (2n + 8), is the number of independent
 * directions in which noise can show up as misfit.
 */
export function planarityStatistic(errorsSquared: readonly number[], sigmaPx: number): PlanarityStatistic {
  if (!(sigmaPx > 0) || !Number.isFinite(sigmaPx)) throw new Error('planarityStatistic: sigma must be a positive number');
  for (const e of errorsSquared) {
    if (!Number.isFinite(e) || e < 0) throw new Error('planarityStatistic: squared errors must be finite and non-negative');
  }
  const dof = 2 * errorsSquared.length - 8;
  if (dof < 1) throw new Error('planarityStatistic: need at least 5 correspondences');
  let sum = 0;
  for (const e of errorsSquared) sum += e;
  const T = sum / (sigmaPx * sigmaPx);
  return { T, dof, pValue: chiSquareSf(T, dof) };
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
 * Three outcomes, checked in this order. The motion gate comes before the
 * p-value and does not look at it: with too little motion even a 3D face is
 * almost a plane, so "not rejected" would mean nothing.
 */
export function decide(input: DecisionInput): Outcome {
  if (input.stat === null) {
    return { kind: 'inconclusive', reason: 'No test statistic: the noise floor is not calibrated or a step failed.' };
  }
  if (input.motionDeg < input.minMotionDeg) {
    return {
      kind: 'inconclusive',
      reason: `Rotation ${input.motionDeg.toFixed(1)}° is below ${input.minMotionDeg}°: too little motion for parallax to be detectable.`,
    };
  }
  if (input.stat.pValue < input.alpha) {
    return {
      kind: 'non-planar',
      reason: `p = ${formatP(input.stat.pValue)} < α = ${input.alpha}: the residuals are larger than landmark noise explains for a single plane.`,
    };
  }
  return {
    kind: 'planar-consistent',
    reason: `p = ${formatP(input.stat.pValue)} ≥ α = ${input.alpha}: one plane explains the landmark motion within the measured noise.`,
  };
}

function formatP(p: number): string {
  if (p === 0) return '< 1e-300'; // below the smallest representable double
  return p < 1e-4 ? p.toExponential(1) : p.toFixed(4);
}
