// Public surface of the planarity method. The application imports the
// method only through this file (via the "@method" alias in vite.config.ts),
// so an alternative implementation directory can be substituted for
// verification without touching application code.

export { estimateHomography } from './homography';
export { sampsonErrorsSquared } from './residuals';
export { planarityStatistic, decide } from './planarity';
export type { PlanarityStatistic, Outcome, OutcomeKind, DecisionInput } from './planarity';
export { estimateNoiseSigma, sigmaConfidenceInterval } from './noise';
export type { NoiseEstimate } from './noise';
