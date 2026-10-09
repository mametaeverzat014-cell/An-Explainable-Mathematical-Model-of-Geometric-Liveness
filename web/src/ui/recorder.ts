import type { LandmarkFrame } from '../core/frames';
import {
  frameToRecorded,
  newRecordingId,
  RECORDING_FORMAT,
  RECORDING_VERSION,
  validateRecording,
  type Recording,
  type RecordingRole,
  type SubjectCategory,
} from '../core/recording';
import { session } from './session';

// Shared recording logic for the Record view and the Experiment wizard.

/** Recordings made in this tab, newest first. Memory only. */
export const tabRecordings: Recording[] = [];

export interface CaptureProgress {
  elapsedMs: number;
  durationMs: number;
  frames: LandmarkFrame[];
  noFace: number;
  latest: LandmarkFrame | null;
}

/**
 * Collects the frames the live session processes for `durationMs`. Only
 * frames with a face are kept; frames without one are counted.
 */
export class Capture {
  private frames: LandmarkFrame[] = [];
  private noFace = 0;
  private startedAt = 0;
  private unsubscribe: (() => void) | null = null;

  constructor(
    private readonly durationMs: number,
    private readonly onProgress: (p: CaptureProgress) => void,
    private readonly onDone: (frames: LandmarkFrame[], noFace: number) => void,
  ) {}

  get active(): boolean {
    return this.unsubscribe !== null;
  }

  start(): void {
    if (this.active) return;
    this.frames = [];
    this.noFace = 0;
    this.startedAt = performance.now();
    this.unsubscribe = session.onFrame(() => this.tick());
  }

  /** Stop early; the frames collected so far are delivered. */
  stop(): void {
    if (!this.active) return;
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.onDone(this.frames, this.noFace);
  }

  /** Stop without delivering anything. */
  cancel(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  private tick(): void {
    const f = session.latestFrame;
    if (f) {
      const last = this.frames[this.frames.length - 1];
      if (!last || f.timestampMs > last.timestampMs) this.frames.push(f);
    } else {
      this.noFace++;
    }
    const elapsedMs = performance.now() - this.startedAt;
    this.onProgress({ elapsedMs, durationMs: this.durationMs, frames: this.frames, noFace: this.noFace, latest: f });
    if (elapsedMs >= this.durationMs) this.stop();
  }
}

export interface RecordingInfo {
  category: SubjectCategory;
  approvalReference: string | null;
  pseudonym: string | null;
  role: RecordingRole;
  condition: string;
  protocolId: string | null;
  notes: string;
}

/** Build and validate a recording from captured frames. Throws RecordingValidationError. */
export function buildRecording(frames: readonly LandmarkFrame[], noFace: number, info: RecordingInfo): Recording {
  if (frames.length < 2) throw new Error('no face was found in the frames');
  const s = session.camera?.settings;
  const t0 = frames[0].timestampMs;
  const rec: Recording = {
    meta: {
      format: RECORDING_FORMAT,
      version: RECORDING_VERSION,
      id: newRecordingId(),
      createdAt: new Date().toISOString(),
      appVersion: import.meta.env.MODE === 'production' ? 'build' : 'dev',
      category: info.category,
      approvalReference: info.category === 'human-participant' ? info.approvalReference : null,
      pseudonym: info.category === 'human-participant' ? info.pseudonym : null,
      role: info.role,
      condition: info.condition,
      protocolId: info.protocolId,
      notes: info.notes,
      camera: {
        label: session.camera?.label || null,
        width: s?.width ?? frames[0].width,
        height: s?.height ?? frames[0].height,
        frameRate: s?.frameRate ?? null,
        facingMode: s?.facingMode ?? null,
      },
      timestampSource: session.timing.summary().source,
      landmarkModel: 'MediaPipe face_landmarker float16/1 (sha256 64184e22…)',
      framesWithoutFace: noFace,
    },
    frames: frames.map((f) => frameToRecorded(f, t0)),
  };
  return validateRecording(rec);
}

/** Offer a JSON value to the user as a file download. */
export function downloadJson(value: unknown, fileName: string): void {
  downloadText(JSON.stringify(value), fileName, 'application/json');
}

export function downloadText(text: string, fileName: string, type: string): void {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
