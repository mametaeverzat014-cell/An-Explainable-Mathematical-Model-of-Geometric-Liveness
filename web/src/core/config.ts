import type { LandmarkSetId } from './landmark-sets';

export interface AnalysisConfig {
  /** Significance level of the planarity test. */
  alpha: number;
  /** Below this rotation (degrees) the outcome is "inconclusive". */
  minMotionDeg: number;
  /** Frames older than this (ms) are not considered as the reference frame. */
  windowMs: number;
  landmarkSet: LandmarkSetId;
  /** Length of the noise-calibration hold, ms. */
  calibrationMs: number;
  /** A calibration is rejected if the head pose changes by more than this during the hold (degrees). */
  maxCalibrationMotionDeg: number;
  /** Frame gap between the two frames of each calibration pair (reduces temporal correlation). */
  calibrationPairGap: number;
}

export interface ConfigNote {
  value: string;
  status: 'convention' | 'placeholder' | 'design-choice';
  note: string;
}

export const DEFAULT_CONFIG: AnalysisConfig = {
  alpha: 0.05,
  minMotionDeg: 5,
  windowMs: 1500,
  landmarkSet: 'rigid',
  calibrationMs: 3000,
  maxCalibrationMotionDeg: 1,
  calibrationPairGap: 5,
};

/** Why each default has its value. Shown in the Method view. */
export const CONFIG_NOTES: Record<keyof AnalysisConfig, ConfigNote> = {
  alpha: { value: '0.05', status: 'convention', note: 'Conventional significance level. Must be fixed in the protocol before data collection.' },
  minMotionDeg: {
    value: '5°',
    status: 'placeholder',
    note: 'Not derived. Simulation suggests power depends strongly on noise level and face size; task M7 (power analysis) must replace this value.',
  },
  windowMs: { value: '1500 ms', status: 'design-choice', note: 'How far back the reference frame may be. Longer windows allow larger rotations but more drift.' },
  landmarkSet: {
    value: 'rigid',
    status: 'design-choice',
    note: 'Expression-free points. Comparing subsets is an experiment (open question Q2), not a tuning knob.',
  },
  calibrationMs: { value: '3000 ms', status: 'design-choice', note: 'Duration of the still hold used to estimate landmark noise.' },
  maxCalibrationMotionDeg: {
    value: '1°',
    status: 'placeholder',
    note: 'A hold with more rotation than this is rejected, because real parallax would inflate the noise estimate.',
  },
  calibrationPairGap: {
    value: '5 frames',
    status: 'placeholder',
    note: 'MediaPipe tracking smooths landmarks over time, so consecutive frames have correlated errors (open question Q3).',
  },
};
