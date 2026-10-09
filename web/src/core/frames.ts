import { outOfPlaneRotationDeg, relativeRotationDeg } from './geometry3d';
import type { Correspondences, Point2 } from './types';

/** One processed camera frame. Landmarks are in pixels of the camera image (not mirrored). */
export interface LandmarkFrame {
  /** Capture time in ms (camera timestamp when available, otherwise presentation time). */
  timestampMs: number;
  width: number;
  height: number;
  /** All 468 mesh landmarks in px (iris points are dropped). */
  points: Point2[];
  /** Head rotation estimated by MediaPipe (3x3), or null if unavailable. */
  rotation: number[] | null;
}

/** Fixed-capacity buffer of recent frames, oldest first. */
export class FrameBuffer {
  private frames: LandmarkFrame[] = [];

  constructor(private readonly capacity = 300) {}

  push(frame: LandmarkFrame): void {
    this.frames.push(frame);
    if (this.frames.length > this.capacity) this.frames.shift();
  }

  clear(): void {
    this.frames = [];
  }

  get size(): number {
    return this.frames.length;
  }

  latest(): LandmarkFrame | null {
    return this.frames.length ? this.frames[this.frames.length - 1] : null;
  }

  /** Frames with timestamp >= now - windowMs. */
  within(windowMs: number, nowMs?: number): LandmarkFrame[] {
    const now = nowMs ?? this.latest()?.timestampMs ?? 0;
    return this.frames.filter((f) => f.timestampMs >= now - windowMs);
  }

  all(): LandmarkFrame[] {
    return [...this.frames];
  }
}

/**
 * Motion between two frames for the motion gate and the pair rules: the
 * OUT-OF-PLANE rotation (yaw/pitch) according to MediaPipe's pose estimate,
 * in degrees, or null. In-plane roll is excluded because it produces no
 * parallax (geometry3d.ts, outOfPlaneRotationDeg).
 */
export function frameRotationDeg(a: LandmarkFrame, b: LandmarkFrame): number | null {
  if (!a.rotation || !b.rotation) return null;
  return outOfPlaneRotationDeg(a.rotation, b.rotation);
}

/** Total rotation angle between two frames (all axes), for checking that a hold is still. */
export function frameTotalRotationDeg(a: LandmarkFrame, b: LandmarkFrame): number | null {
  if (!a.rotation || !b.rotation) return null;
  return relativeRotationDeg(a.rotation, b.rotation);
}

/**
 * Pair-selection rule (fixed in advance; MATH_SPEC.md, section "Pair selection"):
 * among the frames in the window, use as reference the one whose MediaPipe
 * head pose differs MOST from the current frame. The rule looks only at
 * motion, never at residuals, so it does not bias the test towards rejection.
 */
export function selectReference(window: readonly LandmarkFrame[], current: LandmarkFrame): { ref: LandmarkFrame; motionDeg: number } | null {
  let best: { ref: LandmarkFrame; motionDeg: number } | null = null;
  for (const f of window) {
    if (f === current) continue;
    const m = frameRotationDeg(f, current);
    if (m === null) continue;
    if (!best || m > best.motionDeg) best = { ref: f, motionDeg: m };
  }
  return best;
}

export function correspondences(a: LandmarkFrame, b: LandmarkFrame, indices: readonly number[]): Correspondences {
  return { src: indices.map((i) => a.points[i]), dst: indices.map((i) => b.points[i]) };
}

/** Inter-ocular distance (outer eye corners 33-263) in px: a measure of face size in the image. */
export function interocularPx(frame: LandmarkFrame): number {
  const a = frame.points[33];
  const b = frame.points[263];
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export interface CalibrationPairs {
  pairs: Correspondences[];
  /** Largest total rotation between ANY two frames of the hold (degrees). */
  maxRotationDeg: number;
  frames: number;
}

/**
 * Build calibration pairs (frame i, frame i + gap) from a still hold, with
 * DISJOINT pairs: (0, gap), (2 gap, 3 gap), ... No frame is used twice, so the
 * pair costs are independent when the per-frame noise is; the χ² interval for
 * σ assumes that. (Sharing frames between neighbouring pairs left σ̂ unbiased
 * but made the 95 % interval cover only about 88 % of the time in simulation.)
 */
export function calibrationPairs(frames: readonly LandmarkFrame[], indices: readonly number[], gap: number): CalibrationPairs {
  if (!Number.isInteger(gap) || gap < 1) throw new RangeError(`calibration gap must be a positive integer, got ${gap}`);
  const pairs: Correspondences[] = [];
  for (let i = 0; i + gap < frames.length; i += 2 * gap) {
    pairs.push(correspondences(frames[i], frames[i + gap], indices));
  }
  let maxRotationDeg = 0;
  for (let i = 0; i < frames.length; i++) {
    for (let j = i + 1; j < frames.length; j++) {
      const m = frameTotalRotationDeg(frames[i], frames[j]);
      if (m !== null) maxRotationDeg = Math.max(maxRotationDeg, m);
    }
  }
  return { pairs, maxRotationDeg, frames: frames.length };
}
