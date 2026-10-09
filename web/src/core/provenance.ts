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

// Display labels and help texts are translated: see src/ui/locales/*.ts
// (keys prov.<id> and provHelp.<id>).
