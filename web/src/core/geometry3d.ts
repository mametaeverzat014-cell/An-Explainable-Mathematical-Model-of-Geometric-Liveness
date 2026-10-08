import type { Point3 } from './types';

/** 3x3 rotation matrix, row-major. */
export type Rot3 = readonly number[];

const DEG = Math.PI / 180;

export function degToRad(deg: number): number {
  return deg * DEG;
}

export function radToDeg(rad: number): number {
  return rad / DEG;
}

/** Rotation about the vertical (y) axis: positive yaw turns the face towards image-left. */
export function rotY(rad: number): Rot3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}

/** Rotation about the horizontal (x) axis. */
export function rotX(rad: number): Rot3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}

/** Rotation about the viewing (z) axis. */
export function rotZ(rad: number): Rot3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

export function rotMul(a: Rot3, b: Rot3): number[] {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

export function rotApply(r: Rot3, p: Point3): Point3 {
  return {
    x: r[0] * p.x + r[1] * p.y + r[2] * p.z,
    y: r[3] * p.x + r[4] * p.y + r[5] * p.z,
    z: r[6] * p.x + r[7] * p.y + r[8] * p.z,
  };
}

/** Rotation for the given yaw, pitch and roll (degrees), applied as R = Ry * Rx * Rz. */
export function rotationFromAngles(yawDeg: number, pitchDeg: number, rollDeg = 0): number[] {
  return rotMul(rotY(degToRad(yawDeg)), rotMul(rotX(degToRad(pitchDeg)), rotZ(degToRad(rollDeg))));
}

/**
 * Angle (degrees) of the relative rotation between two rotation matrices,
 * from trace(R1^T R2) = 1 + 2 cos(angle).
 *
 * trace(R1^T R2) equals the sum of element-wise products of R1 and R2, so the
 * result is the same whether the matrices are stored row- or column-major.
 */
export function relativeRotationDeg(r1: Rot3, r2: Rot3): number {
  let tr = 0;
  for (let i = 0; i < 9; i++) tr += r1[i] * r2[i];
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  return radToDeg(Math.acos(c));
}

/**
 * Extract the upper-left 3x3 block of a 4x4 matrix stored as 16 numbers
 * (layout-agnostic for the purpose of relativeRotationDeg; see above), and
 * remove any uniform scale so that the result is a rotation.
 */
export function rotationFrom4x4(m: ArrayLike<number>): number[] | null {
  if (m.length !== 16) return null;
  const r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const scale = Math.cbrt(
    r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]),
  );
  if (!Number.isFinite(scale) || Math.abs(scale) < 1e-12) return null;
  return r.map((v) => v / scale);
}
