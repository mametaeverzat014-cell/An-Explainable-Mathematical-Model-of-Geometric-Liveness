import type { LandmarkFrame } from './frames';

// Recording format for research mode (Phase 2). A recording stores LANDMARK
// TRACKS ONLY: 468 image points and MediaPipe's head rotation per frame.
// No video, no images. For a real person these tracks are still derived
// biometric data, which is why human-participant recordings require an
// ethics approval reference (checked by validateRecording).

export const RECORDING_FORMAT = 'parallax-lab/recording';
export const RECORDING_VERSION = 1;

/** Coordinates are rounded to this step (px). Rounding adds ~0.003 px RMS, negligible next to landmark noise. */
export const COORD_STEP_PX = 0.01;

export type SubjectCategory = 'non-human-target' | 'human-participant';
export type RecordingRole = 'calibration' | 'trial';

export interface RecordingMeta {
  format: typeof RECORDING_FORMAT;
  version: typeof RECORDING_VERSION;
  id: string;
  createdAt: string;
  appVersion: string;
  /** non-human-target: a print or screen held by the researchers; human-participant: a person. */
  category: SubjectCategory;
  /** Required for human-participant recordings: the reference of the ethics approval (IRB/SRC/school committee). */
  approvalReference: string | null;
  /** Pseudonymous participant code (e.g. "P07"). Never a name. Null for non-human targets. */
  pseudonym: string | null;
  role: RecordingRole;
  /** What was presented, e.g. "flat-print", "screen-photo", "curved-print", "bona-fide". */
  condition: string;
  protocolId: string | null;
  notes: string;
  camera: { label: string | null; width: number; height: number; frameRate: number | null; facingMode: string | null };
  timestampSource: string;
  landmarkModel: string;
  /** Frames delivered by the camera in which no face was found (not stored). */
  framesWithoutFace: number;
}

export interface RecordedFrame {
  /** ms since the first stored frame. */
  t: number;
  /** MediaPipe head rotation, 3x3 (9 numbers), or null. */
  rot: number[] | null;
  /** 936 numbers: x0, y0, x1, y1, ... for landmarks 0..467, in px. */
  pts: number[];
}

export interface Recording {
  meta: RecordingMeta;
  frames: RecordedFrame[];
}

export class RecordingValidationError extends Error {}

const round = (v: number) => Math.round(v / COORD_STEP_PX) * COORD_STEP_PX;

export function frameToRecorded(frame: LandmarkFrame, t0: number): RecordedFrame {
  const pts = new Array<number>(936);
  frame.points.forEach((p, i) => {
    pts[2 * i] = Number(round(p.x).toFixed(2));
    pts[2 * i + 1] = Number(round(p.y).toFixed(2));
  });
  return {
    t: Number((frame.timestampMs - t0).toFixed(3)),
    rot: frame.rotation ? frame.rotation.map((v) => Number(v.toFixed(6))) : null,
    pts,
  };
}

export function recordedToFrame(f: RecordedFrame, width: number, height: number): LandmarkFrame {
  const points = Array.from({ length: 468 }, (_, i) => ({ x: f.pts[2 * i], y: f.pts[2 * i + 1] }));
  return { timestampMs: f.t, width, height, points, rotation: f.rot };
}

export function framesOf(rec: Recording): LandmarkFrame[] {
  return rec.frames.map((f) => recordedToFrame(f, rec.meta.camera.width, rec.meta.camera.height));
}

function fail(msg: string): never {
  throw new RecordingValidationError(msg);
}

/**
 * Parse and check a recording. Throws RecordingValidationError with a
 * specific message if anything is wrong, including a human-participant
 * recording without an approval reference.
 */
export function validateRecording(input: unknown): Recording {
  if (typeof input !== 'object' || input === null) fail('not a JSON object');
  const obj = input as { meta?: unknown; frames?: unknown };
  const meta = obj.meta as Partial<RecordingMeta> | undefined;
  if (!meta || typeof meta !== 'object') fail('missing "meta"');
  if (meta.format !== RECORDING_FORMAT) fail(`unknown format "${String(meta.format)}"`);
  if (meta.version !== RECORDING_VERSION) fail(`unsupported version ${String(meta.version)} (expected ${RECORDING_VERSION})`);
  if (meta.category !== 'non-human-target' && meta.category !== 'human-participant') fail('meta.category must be "non-human-target" or "human-participant"');
  if (meta.category === 'human-participant') {
    if (typeof meta.approvalReference !== 'string' || meta.approvalReference.trim().length === 0) {
      fail('human-participant recording without an ethics approval reference');
    }
    if (typeof meta.pseudonym !== 'string' || !/^[A-Za-z0-9_-]{1,20}$/.test(meta.pseudonym)) {
      fail('human-participant recording needs a pseudonym of 1-20 letters, digits, "_" or "-"');
    }
  }
  if (meta.role !== 'calibration' && meta.role !== 'trial') fail('meta.role must be "calibration" or "trial"');
  if (typeof meta.condition !== 'string' || !meta.condition) fail('meta.condition is required');
  const cam = meta.camera;
  if (!cam || !(cam.width! > 0) || !(cam.height! > 0)) fail('meta.camera.width/height are required');
  if (!Array.isArray(obj.frames)) fail('missing "frames" array');
  const frames = obj.frames as RecordedFrame[];
  let prevT = -Infinity;
  frames.forEach((f, k) => {
    if (!f || !Array.isArray(f.pts) || f.pts.length !== 936) fail(`frame ${k}: pts must have 936 numbers`);
    if (!f.pts.every(Number.isFinite)) fail(`frame ${k}: non-finite coordinate`);
    if (!Number.isFinite(f.t) || f.t <= prevT) fail(`frame ${k}: timestamps must be finite and strictly increasing`);
    if (f.rot !== null && (!Array.isArray(f.rot) || f.rot.length !== 9 || !f.rot.every(Number.isFinite))) fail(`frame ${k}: rot must be null or 9 numbers`);
    prevT = f.t;
  });
  return { meta: meta as RecordingMeta, frames };
}

export function parseRecording(json: string): Recording {
  let data: unknown;
  try {
    data = JSON.parse(json);
  } catch (e) {
    fail(`invalid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
  return validateRecording(data);
}

export function newRecordingId(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** File name that carries no personal information. */
export function recordingFileName(meta: RecordingMeta): string {
  const date = meta.createdAt.slice(0, 10);
  const who = meta.category === 'human-participant' ? meta.pseudonym : 'target';
  const safe = (s: string) => s.replace(/[^A-Za-z0-9_-]+/g, '-').slice(0, 40);
  return `${date}_${safe(who ?? 'target')}_${safe(meta.condition)}_${meta.role}_${meta.id}.json`;
}
