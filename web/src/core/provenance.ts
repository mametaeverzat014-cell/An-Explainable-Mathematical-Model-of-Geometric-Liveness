/**
 * Every number shown by the application carries one of these labels, so a
 * reader can always tell what kind of claim it is.
 */
export type Provenance =
  /** Read directly from the device (timestamps, resolution). No learned model involved. */
  | 'measured'
  /** Output of the MediaPipe neural network (landmark positions, head pose). An estimate with unknown error. */
  | 'mediapipe-estimate'
  /** Output of this project's statistical method. Conditional on its stated assumptions; NOT validated. */
  | 'method-output'
  /** Known by construction in a synthetic scene. */
  | 'synthetic-truth'
  /** A configuration value chosen in advance, not derived from data. */
  | 'configuration'
  /** The responsible method function has not been implemented yet. */
  | 'not-implemented';

export const PROVENANCE_LABEL: Record<Provenance, string> = {
  measured: 'Measured',
  'mediapipe-estimate': 'MediaPipe estimate',
  'method-output': 'Method output · unvalidated',
  'synthetic-truth': 'Synthetic ground truth',
  configuration: 'Configuration',
  'not-implemented': 'Not implemented',
};

export const PROVENANCE_HELP: Record<Provenance, string> = {
  measured: 'Read directly from the browser or camera; no learned model involved.',
  'mediapipe-estimate': 'Produced by the MediaPipe face landmark network. Its accuracy on your camera is not known.',
  'method-output':
    'Computed by the planarity test. Valid only under the assumptions in MATH_SPEC.md; not yet validated on real presentations.',
  'synthetic-truth': 'Known exactly because the scene is simulated.',
  configuration: 'A setting fixed in advance. Defaults marked "placeholder" still need justification.',
  'not-implemented': 'The method function for this step is not implemented in the build you are running.',
};
