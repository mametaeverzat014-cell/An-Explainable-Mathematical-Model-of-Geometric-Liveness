import canonicalFace from '../data/canonical-face.json';
import { rotApply, rotationFromAngles } from './geometry3d';
import type { Rng } from './rng';
import type { Correspondences, Point2, Point3 } from './types';

// Synthetic scene used for (1) validating the statistical method where the
// ground truth is known and (2) the "Synthetic lab" view. It is a model of
// the world, not a measurement: every number produced from it is conditional
// on the assumptions listed in docs/MATH_SPEC.md, section "Synthetic scene".

/** The 468 vertices of MediaPipe's canonical face model, in cm (x right on image, y up, z towards viewer). */
export const CANONICAL_FACE: readonly Point3[] = (canonicalFace.vertices as number[][]).map(([x, y, z]) => ({ x, y, z }));

export type ObjectKind = 'face3d' | 'plane' | 'cylinder';

export interface SceneObject {
  readonly kind: ObjectKind;
  /** Points in the object frame, cm, indexed like MediaPipe landmarks 0..467. */
  readonly points: readonly Point3[];
  /** Centre of rotation in the object frame, cm. */
  readonly pivot: Point3;
}

export interface ObjectOptions {
  /** Uniform scale of the depicted face (1 = life size). */
  scale?: number;
  /** Cylinder radius in cm for kind 'cylinder' (a print bent around a vertical axis). */
  cylinderRadiusCm?: number;
  /** Override the rotation centre. */
  pivot?: Point3;
}

/**
 * Build a presented object.
 * - face3d: the canonical face; default pivot 4 cm behind the canonical origin (inside the head).
 * - plane: a frontal orthographic image of the face printed on a flat sheet (z = 0); pivot at the sheet centre.
 * - cylinder: the same sheet bent around a vertical axis (a curved print). Edges bend away from the camera.
 */
export function makeObject(kind: ObjectKind, options: ObjectOptions = {}): SceneObject {
  const s = options.scale ?? 1;
  const R = options.cylinderRadiusCm ?? 10;
  let points: Point3[];
  let pivot: Point3;
  switch (kind) {
    case 'face3d':
      points = CANONICAL_FACE.map((p) => ({ x: s * p.x, y: s * p.y, z: s * p.z }));
      pivot = { x: 0, y: 0, z: -4 * s };
      break;
    case 'plane':
      points = CANONICAL_FACE.map((p) => ({ x: s * p.x, y: s * p.y, z: 0 }));
      pivot = { x: 0, y: 0, z: 0 };
      break;
    case 'cylinder':
      points = CANONICAL_FACE.map((p) => ({
        x: R * Math.sin((s * p.x) / R),
        y: s * p.y,
        z: R * Math.cos((s * p.x) / R) - R,
      }));
      pivot = { x: 0, y: 0, z: 0 };
      break;
  }
  return { kind, points, pivot: options.pivot ?? pivot };
}

export interface Pose {
  yawDeg: number;
  pitchDeg: number;
  rollDeg?: number;
  /** Translation in cm, applied after rotation (x right, y up, z towards camera). */
  tx?: number;
  ty?: number;
  tz?: number;
}

export interface Intrinsics {
  /** Focal length in pixels. */
  readonly f: number;
  readonly cx: number;
  readonly cy: number;
  readonly width: number;
  readonly height: number;
}

/** A 1280x720 camera with ~65 degree horizontal field of view, typical of laptop webcams. */
export const DEFAULT_INTRINSICS: Intrinsics = { f: 1000, cx: 640, cy: 360, width: 1280, height: 720 };

/**
 * Ideal pinhole projection of the object in the given pose, with the pivot
 * placed `distanceCm` in front of the camera on the optical axis.
 * Camera frame: X right, Y down, Z forward (OpenCV convention).
 * No lens distortion, no rolling shutter, no noise.
 */
export function projectObject(obj: SceneObject, pose: Pose, distanceCm: number, cam: Intrinsics = DEFAULT_INTRINSICS): Point2[] {
  const r = rotationFromAngles(pose.yawDeg, pose.pitchDeg, pose.rollDeg ?? 0);
  return obj.points.map((p) => {
    const q = rotApply(r, { x: p.x - obj.pivot.x, y: p.y - obj.pivot.y, z: p.z - obj.pivot.z });
    const X = q.x + (pose.tx ?? 0);
    const Y = -(q.y + (pose.ty ?? 0));
    const Z = distanceCm - (q.z + (pose.tz ?? 0));
    if (Z <= 0) throw new Error('projectObject: point behind the camera');
    return { x: (cam.f * X) / Z + cam.cx, y: (cam.f * Y) / Z + cam.cy };
  });
}

/** Add independent isotropic Gaussian noise N(0, sigma^2) to each coordinate. */
export function addNoise(points: readonly Point2[], sigmaPx: number, rng: Rng): Point2[] {
  if (sigmaPx === 0) return points.map((p) => ({ ...p }));
  return points.map((p) => ({ x: p.x + sigmaPx * rng.normal(), y: p.y + sigmaPx * rng.normal() }));
}

export function selectIndices<T>(items: readonly T[], indices: readonly number[]): T[] {
  return indices.map((i) => {
    if (i < 0 || i >= items.length) throw new Error(`selectIndices: index ${i} out of range`);
    return items[i];
  });
}

export interface PairSpec {
  object: SceneObject;
  pose0: Pose;
  pose1: Pose;
  distanceCm: number;
  sigmaPx: number;
  indices: readonly number[];
  intrinsics?: Intrinsics;
}

/** One noisy synthetic frame pair: the same landmarks seen in pose0 and pose1. */
export function syntheticPair(spec: PairSpec, rng: Rng): Correspondences {
  const cam = spec.intrinsics ?? DEFAULT_INTRINSICS;
  const a = selectIndices(projectObject(spec.object, spec.pose0, spec.distanceCm, cam), spec.indices);
  const b = selectIndices(projectObject(spec.object, spec.pose1, spec.distanceCm, cam), spec.indices);
  return { src: addNoise(a, spec.sigmaPx, rng), dst: addNoise(b, spec.sigmaPx, rng) };
}
