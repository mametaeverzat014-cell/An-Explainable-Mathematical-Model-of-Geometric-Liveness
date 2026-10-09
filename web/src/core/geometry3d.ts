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
 * Out-of-plane rotation between two poses (degrees): the angle by which the
 * camera's viewing axis, expressed in the face's own coordinates, turns.
 *
 * Why not the total rotation angle: turning the face about the viewing axis
 * (in-plane roll) rotates the image, and an image rotation is exactly a
 * homography even for a 3D face. Such motion produces no parallax, so it must
 * not count towards the motion gate. Yaw and pitch count fully.
 *
 * R maps face coordinates to camera coordinates and is stored row-major, so
 * the viewing axis in face coordinates, R^T z, is the third row of R.
 * Computed with atan2 for accuracy at small angles.
 */
export function outOfPlaneRotationDeg(r1: Rot3, r2: Rot3): number {
  const a = [r1[6], r1[7], r1[8]];
  const b = [r2[6], r2[7], r2[8]];
  const cross = Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return radToDeg(Math.atan2(cross, dot));
}

/**
 * Extract the rotation from a 4x4 pose matrix (MediaPipe's facial
 * transformation matrix) as a row-major 3x3 matrix, removing any uniform
 * scale.
 *
 * MediaPipe's JavaScript API does not state the storage order of the 16
 * numbers. The pose contains a translation (the face is tens of centimetres
 * from the camera), which sits in elements 12-14 when the matrix is stored
 * column-major and in elements 3, 7, 11 when row-major; the order is detected
 * from that. Detection matters for outOfPlaneRotationDeg, not for
 * relativeRotationDeg.
 */
export function rotationFrom4x4(m: ArrayLike<number>): number[] | null {
  if (m.length !== 16) return null;
  const columnMajor = Math.abs(m[12]) + Math.abs(m[13]) + Math.abs(m[14]) > Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]);
  const r = columnMajor ? [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]] : [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const scale = Math.cbrt(
    r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]),
  );
  if (!Number.isFinite(scale) || Math.abs(scale) < 1e-12) return null;
  return r.map((v) => v / scale);
}
