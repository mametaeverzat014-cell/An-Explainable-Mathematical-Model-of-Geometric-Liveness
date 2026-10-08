// Landmark subsets of the 468-point MediaPipe face mesh.
//
// The region lists (oval, eyes, brows, lips) are the unions of the connection
// sets exported by @mediapipe/tasks-vision (FaceLandmarker.FACE_LANDMARKS_*);
// a unit test checks that these copies still match the installed package.
// The rigid set was chosen by inspecting the canonical face model
// (src/data/canonical-face.json): forehead/nose midline, nostril wings and
// the four eye corners - points that do not move with facial expressions.
//
// Which subset is used is a design decision that must be fixed in the
// protocol BEFORE collecting data (MATH_SPEC.md, open question Q2).

export const FACE_OVAL = [
  10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323,
  332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454,
] as const;

export const LEFT_EYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466] as const;
export const RIGHT_EYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246] as const;
export const LEFT_EYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336] as const;
export const RIGHT_EYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107] as const;
export const LIPS = [
  0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308,
  310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
] as const;

/** Forehead and nose midline (x = 0 in the canonical model), top to bottom. */
export const MIDLINE_RIGID = [151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2] as const;
/** Nostril wings, symmetric pairs. */
export const NOSE_WINGS = [48, 278, 64, 294, 98, 327, 129, 358] as const;
/** Outer and inner eye corners (canthi). */
export const EYE_CORNERS = [33, 133, 362, 263] as const;

export type LandmarkSetId = 'rigid' | 'features' | 'interior' | 'all';

function uniqueSorted(values: Iterable<number>): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

const OVAL_SET = new Set<number>(FACE_OVAL);
const ALL_MESH = Array.from({ length: 468 }, (_, i) => i);

export const LANDMARK_SETS: Readonly<Record<LandmarkSetId, { label: string; description: string; indices: readonly number[] }>> = {
  rigid: {
    label: 'Rigid (25 points)',
    description:
      'Forehead/nose midline, nostril wings and eye corners. Least affected by expressions and blinks. Default.',
    indices: uniqueSorted([...MIDLINE_RIGID, ...NOSE_WINGS, ...EYE_CORNERS]),
  },
  features: {
    label: 'Facial features',
    description: 'Rigid set plus eyes, eyebrows and lips. More points, but eyelids, brows and lips deform non-rigidly.',
    indices: uniqueSorted([
      ...MIDLINE_RIGID,
      ...NOSE_WINGS,
      ...EYE_CORNERS,
      ...LEFT_EYE,
      ...RIGHT_EYE,
      ...LEFT_EYEBROW,
      ...RIGHT_EYEBROW,
      ...LIPS,
    ]),
  },
  interior: {
    label: 'Interior mesh',
    description:
      'All mesh points except the face outline. Outline points on a real face slide along the silhouette and are not fixed surface points.',
    indices: ALL_MESH.filter((i) => !OVAL_SET.has(i)),
  },
  all: {
    label: 'All 468 points',
    description: 'Includes silhouette points, which violate the fixed-point assumption on real faces. For comparison only.',
    indices: ALL_MESH,
  },
};
