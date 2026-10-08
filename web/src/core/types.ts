/** A point in image coordinates, in pixels. x grows right, y grows down. */
export interface Point2 {
  readonly x: number;
  readonly y: number;
}

/** A point in 3D, in centimetres. */
export interface Point3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/**
 * A 3x3 matrix stored row-major: [h11, h12, h13, h21, h22, h23, h31, h32, h33].
 * Homographies are defined only up to a non-zero scale factor.
 */
export type Mat3 = readonly number[];

/** One point correspondence set between two frames (same landmarks, same order). */
export interface Correspondences {
  readonly src: readonly Point2[];
  readonly dst: readonly Point2[];
}
