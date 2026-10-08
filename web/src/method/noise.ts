// Tasks M5 and M6 in web/docs/MATH_SPEC.md: noise estimate and its interval.
//
// Written by the AI assistant at the project lead's request (2026-10-08),
// see AI_ASSISTANCE.md and web/docs/METHOD_EXPLAINED_RU.md.

import { chiSquareQuantile } from '../core/stats';
import type { Correspondences } from '../core/types';
import { estimateHomography } from './homography';
import { sampsonErrorsSquared } from './residuals';

export interface NoiseEstimate {
  /** Estimated landmark noise standard deviation per coordinate, px. */
  readonly sigmaPx: number;
  /** Total degrees of freedom the estimate is based on (sum of 2n_k - 8). */
  readonly dof: number;
}

/**
 * Pooled estimate from a still hold:  σ̂² = Σ_k C_k / Σ_k (2 n_k − 8).
 *
 * Under H0 each pair's cost satisfies E[C_k] ≈ σ² (2 n_k − 8) (see
 * planarityStatistic), so dividing the summed costs by the summed degrees
 * of freedom gives an unbiased estimate of σ². Pairs with fewer than 5
 * correspondences carry no information about σ and are skipped.
 */
export function estimateNoiseSigma(pairs: readonly Correspondences[]): NoiseEstimate {
  let cost = 0;
  let dof = 0;
  for (const pair of pairs) {
    if (pair.src.length < 5) continue;
    const h = estimateHomography(pair.src, pair.dst);
    for (const e of sampsonErrorsSquared(h, pair.src, pair.dst)) cost += e;
    dof += 2 * pair.src.length - 8;
  }
  if (dof === 0) throw new Error('estimateNoiseSigma: no pair with at least 5 correspondences');
  return { sigmaPx: Math.sqrt(cost / dof), dof };
}

/**
 * If ν σ̂² / σ² follows χ²_ν (ν = dof), then with probability `confidence`
 *   χ²_{a/2} ≤ ν σ̂² / σ² ≤ χ²_{1−a/2},   a = 1 − confidence,
 * and solving for σ gives the interval below.
 */
export function sigmaConfidenceInterval(estimate: NoiseEstimate, confidence: number): [number, number] {
  if (!(confidence > 0 && confidence < 1)) throw new Error('sigmaConfidenceInterval: confidence must be in (0, 1)');
  if (!(estimate.dof > 0)) throw new Error('sigmaConfidenceInterval: dof must be positive');
  const a = 1 - confidence;
  const nu = estimate.dof;
  const lower = estimate.sigmaPx * Math.sqrt(nu / chiSquareQuantile(1 - a / 2, nu));
  const upper = estimate.sigmaPx * Math.sqrt(nu / chiSquareQuantile(a / 2, nu));
  return [lower, upper];
}
