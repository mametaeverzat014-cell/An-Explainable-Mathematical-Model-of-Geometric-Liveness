import * as method from '@method';
import type { NoiseEstimate, Outcome, PlanarityStatistic } from '@method';
import { isNotImplemented } from './errors';
import { applyHomography } from './linalg';
import type { Correspondences, Mat3, Point2 } from './types';

// Runs the method stage by stage and records, for each stage,
// whether it ran, is not implemented yet, or failed. Nothing here computes
// any part of the method itself.

/** The method functions the application depends on (injectable for testing). */
export interface PlanarityMethod {
  estimateHomography: typeof method.estimateHomography;
  sampsonErrorsSquared: typeof method.sampsonErrorsSquared;
  planarityStatistic: typeof method.planarityStatistic;
  decide: typeof method.decide;
  estimateNoiseSigma: typeof method.estimateNoiseSigma;
  sigmaConfidenceInterval: typeof method.sigmaConfidenceInterval;
}

export const DEFAULT_METHOD: PlanarityMethod = method;

export type StageName = 'homography' | 'residuals' | 'statistic' | 'decision';
export type StageState = 'ok' | 'not-implemented' | 'error' | 'skipped';

export interface StageResult {
  stage: StageName;
  state: StageState;
  message?: string;
}

export interface PairAnalysis {
  n: number;
  motionDeg: number;
  alpha: number;
  minMotionDeg: number;
  sigmaPx: number | null;
  homography: Mat3 | null;
  /** Per-landmark squared Sampson error, px^2. */
  errorsSquared: number[] | null;
  /** dst_i - H src_i, px: what the fitted plane cannot explain (for display). */
  residualVectors: Point2[] | null;
  stat: PlanarityStatistic | null;
  outcome: Outcome | null;
  stages: StageResult[];
}

export interface PairOptions {
  sigmaPx: number | null;
  alpha: number;
  minMotionDeg: number;
  motionDeg: number;
}

function run<T>(stage: StageName, stages: StageResult[], fn: () => T): T | null {
  try {
    const value = fn();
    stages.push({ stage, state: 'ok' });
    return value;
  } catch (error) {
    if (isNotImplemented(error)) stages.push({ stage, state: 'not-implemented', message: error.message });
    else stages.push({ stage, state: 'error', message: error instanceof Error ? error.message : String(error) });
    return null;
  }
}

function skip(stage: StageName, stages: StageResult[], message: string): null {
  stages.push({ stage, state: 'skipped', message });
  return null;
}

/** Analyse one frame pair with the planarity test. */
export function analyzePair(corr: Correspondences, options: PairOptions, method: PlanarityMethod = DEFAULT_METHOD): PairAnalysis {
  const stages: StageResult[] = [];
  const homography = run('homography', stages, () => method.estimateHomography(corr.src, corr.dst));

  const errorsSquared = homography
    ? run('residuals', stages, () => method.sampsonErrorsSquared(homography, corr.src, corr.dst))
    : skip('residuals', stages, 'needs a homography');

  let stat: PlanarityStatistic | null = null;
  if (!errorsSquared) skip('statistic', stages, 'needs residuals');
  else if (options.sigmaPx === null) skip('statistic', stages, 'noise floor not calibrated');
  else stat = run('statistic', stages, () => method.planarityStatistic(errorsSquared, options.sigmaPx as number));

  // The decision rule is evaluated even without a statistic: it must report "inconclusive".
  const outcome = run('decision', stages, () =>
    method.decide({ stat, alpha: options.alpha, motionDeg: options.motionDeg, minMotionDeg: options.minMotionDeg }),
  );

  const residualVectors = homography
    ? corr.src.map((p, i) => {
        const q = applyHomography(homography, p);
        return { x: corr.dst[i].x - q.x, y: corr.dst[i].y - q.y };
      })
    : null;

  return {
    n: corr.src.length,
    motionDeg: options.motionDeg,
    alpha: options.alpha,
    minMotionDeg: options.minMotionDeg,
    sigmaPx: options.sigmaPx,
    homography,
    errorsSquared,
    residualVectors,
    stat,
    outcome,
    stages,
  };
}

export interface NoiseCalibration {
  estimate: NoiseEstimate | null;
  interval95: [number, number] | null;
  pairs: number;
  state: StageState;
  message?: string;
}

export function calibrateNoise(pairs: readonly Correspondences[], method: PlanarityMethod = DEFAULT_METHOD): NoiseCalibration {
  const stages: StageResult[] = [];
  const estimate = run('statistic', stages, () => method.estimateNoiseSigma(pairs));
  if (!estimate) return { estimate: null, interval95: null, pairs: pairs.length, state: stages[0].state, message: stages[0].message };
  let interval95: [number, number] | null = null;
  try {
    interval95 = method.sigmaConfidenceInterval(estimate, 0.95);
  } catch {
    interval95 = null; // optional task M6; the estimate is still usable
  }
  return { estimate, interval95, pairs: pairs.length, state: 'ok' };
}

export interface ModuleStatus {
  name: keyof PlanarityMethod;
  task: string;
  implemented: boolean;
}

/** Probe which method functions are implemented, using a tiny valid input. */
export function probeMethod(method: PlanarityMethod = DEFAULT_METHOD): ModuleStatus[] {
  const sq = [
    { x: 0, y: 0 },
    { x: 100, y: 0 },
    { x: 100, y: 100 },
    { x: 0, y: 100 },
    { x: 50, y: 30 },
  ];
  const corr = { src: sq, dst: sq.map((p) => ({ x: p.x + 1, y: p.y + 2 })) };
  const probe = (fn: () => unknown) => {
    try {
      fn();
      return true;
    } catch (error) {
      return !isNotImplemented(error);
    }
  };
  const h: Mat3 = [1, 0, 1, 0, 1, 2, 0, 0, 1];
  return [
    { name: 'estimateHomography', task: 'M1', implemented: probe(() => method.estimateHomography(corr.src, corr.dst)) },
    { name: 'sampsonErrorsSquared', task: 'M2', implemented: probe(() => method.sampsonErrorsSquared(h, corr.src, corr.dst)) },
    { name: 'planarityStatistic', task: 'M3', implemented: probe(() => method.planarityStatistic([1, 1, 1, 1, 1], 1)) },
    { name: 'decide', task: 'M4', implemented: probe(() => method.decide({ stat: null, alpha: 0.05, motionDeg: 0, minMotionDeg: 5 })) },
    { name: 'estimateNoiseSigma', task: 'M5', implemented: probe(() => method.estimateNoiseSigma([corr])) },
    { name: 'sigmaConfidenceInterval', task: 'M6', implemented: probe(() => method.sigmaConfidenceInterval({ sigmaPx: 1, dof: 10 }, 0.95)) },
  ];
}
