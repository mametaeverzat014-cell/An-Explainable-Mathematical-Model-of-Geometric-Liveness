import { startCamera, stopCamera, type CameraInfo } from '../capture/camera';
import { createLandmarkerWithFallback, detect, type Delegate, type LandmarkerHandle } from '../capture/landmarker';
import { TimingStats, frameLoop, type FrameTick } from '../capture/timing';
import { DEFAULT_CONFIG, type AnalysisConfig } from '../core/config';
import { FrameBuffer, calibrationPairs, correspondences, interocularPx, selectReference, type LandmarkFrame } from '../core/frames';
import { LANDMARK_SETS } from '../core/landmark-sets';
import { analyzePair, calibrateNoise, type NoiseCalibration, type PairAnalysis } from '../core/method';
import type { MessageKey } from './i18n';

/** A message stored as a translation key, so it follows language changes. */
export interface Msg {
  key: MessageKey;
  params?: Record<string, string | number>;
}

// The single live capture session shared by the Live and Diagnostics views.
// Everything stays in memory in this tab; nothing is stored or transmitted.

export type SessionState = 'idle' | 'starting' | 'running' | 'error';

export interface CalibrationState {
  phase: 'none' | 'collecting' | 'done' | 'rejected';
  startedAt: number;
  frames: LandmarkFrame[];
  result: NoiseCalibration | null;
  message: Msg | null;
  maxRotationDeg: number;
}

export interface HistoryPoint {
  t: number;
  tOverDof: number | null;
  motionDeg: number;
}

type Listener = () => void;

class LiveSession {
  state: SessionState = 'idle';
  error = '';
  camera: CameraInfo | null = null;
  delegate: Delegate | null = null;
  config: AnalysisConfig = { ...DEFAULT_CONFIG };
  readonly buffer = new FrameBuffer(300);
  readonly timing = new TimingStats();
  latestFrame: LandmarkFrame | null = null;
  latestAnalysis: PairAnalysis | null = null;
  latestReference: LandmarkFrame | null = null;
  analysisNote: Msg | null = null;
  history: HistoryPoint[] = [];
  calibration: CalibrationState = { phase: 'none', startedAt: 0, frames: [], result: null, message: null, maxRotationDeg: 0 };
  readonly video: HTMLVideoElement;

  private stream: MediaStream | null = null;
  private handle: LandmarkerHandle | null = null;
  private stopLoop: (() => void) | null = null;
  private lastMpTs = 0;
  private lastAnalysisAt = 0;
  private listeners = new Set<Listener>();
  private frameListeners = new Set<Listener>();

  private readonly parking: HTMLElement;

  constructor() {
    this.video = document.createElement('video');
    this.video.setAttribute('playsinline', '');
    this.video.muted = true;
    // Browsers stop delivering frames to a <video> that is not in the
    // document. When the Live view is not shown, the element is parked in a
    // tiny, invisible holder so measurement continues in other views.
    this.parking = document.createElement('div');
    this.parking.setAttribute('aria-hidden', 'true');
    this.parking.style.cssText = 'position:fixed;left:0;bottom:0;width:2px;height:2px;overflow:hidden;opacity:0;pointer-events:none';
  }

  /** Move the video out of a view that is being removed, keeping it in the document. */
  parkVideo(): void {
    if (!this.parking.isConnected) document.body.append(this.parking);
    this.parking.append(this.video);
  }

  /** Notified ~5 times a second with new numbers. */
  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** Notified on every processed frame (for drawing the overlay). */
  onFrame(fn: Listener): () => void {
    this.frameListeners.add(fn);
    return () => this.frameListeners.delete(fn);
  }

  private emit(): void {
    this.listeners.forEach((fn) => fn());
  }

  async start(delegate: Delegate, facingMode: 'user' | 'environment'): Promise<void> {
    if (this.state === 'running' || this.state === 'starting') return;
    this.state = 'starting';
    this.error = '';
    this.emit();
    try {
      if (!this.handle || this.handle.delegate !== delegate) {
        this.handle?.landmarker.close();
        this.handle = await createLandmarkerWithFallback(delegate);
      }
      this.delegate = this.handle.delegate;
      const { stream, info } = await startCamera(this.video, { facingMode });
      this.stream = stream;
      this.camera = info;
      this.buffer.clear();
      this.timing.reset();
      this.history = [];
      this.state = 'running';
      this.stopLoop = frameLoop(this.video, (tick) => this.process(tick));
    } catch (e) {
      this.state = 'error';
      this.error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      stopCamera(this.stream, this.video);
      this.stream = null;
    }
    this.emit();
  }

  stop(): void {
    this.stopLoop?.();
    this.stopLoop = null;
    stopCamera(this.stream, this.video);
    this.stream = null;
    this.state = 'idle';
    this.latestFrame = null;
    this.latestAnalysis = null;
    this.emit();
  }

  setConfig(patch: Partial<AnalysisConfig>): void {
    const setChanged = patch.landmarkSet && patch.landmarkSet !== this.config.landmarkSet;
    this.config = { ...this.config, ...patch };
    if (setChanged) {
      // sigma was estimated on a different set of landmarks: it no longer applies.
      this.calibration = { phase: 'none', startedAt: 0, frames: [], result: null, message: { key: 'live.landmarkSetChanged' }, maxRotationDeg: 0 };
    }
    this.history = [];
    this.emit();
  }

  startCalibration(): void {
    this.calibration = { phase: 'collecting', startedAt: performance.now(), frames: [], result: null, message: { key: 'live.holdStill' }, maxRotationDeg: 0 };
    this.emit();
  }

  get sigmaPx(): number | null {
    const c = this.calibration;
    return c.phase === 'done' && c.result?.estimate ? c.result.estimate.sigmaPx : null;
  }

  private process(tick: FrameTick): void {
    if (!this.handle || this.video.readyState < 2) return;
    // MediaPipe requires strictly increasing timestamps; analysis uses the camera timestamp.
    const mpTs = Math.max(performance.now(), this.lastMpTs + 1);
    this.lastMpTs = mpTs;
    let out;
    try {
      out = detect(this.handle, this.video, mpTs);
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      return;
    }
    this.timing.record(tick, out.inferenceMs, out.frame !== null);
    const frame = out.frame ? { ...out.frame, timestampMs: tick.timestampMs } : null;
    this.latestFrame = frame;
    if (frame) {
      this.buffer.push(frame);
      if (this.calibration.phase === 'collecting') this.collectCalibration(frame);
    }
    const now = performance.now();
    if (now - this.lastAnalysisAt > 100) {
      this.lastAnalysisAt = now;
      this.analyze();
      this.emit();
    }
    this.frameListeners.forEach((fn) => fn());
  }

  private collectCalibration(frame: LandmarkFrame): void {
    const c = this.calibration;
    c.frames.push(frame);
    if (performance.now() - c.startedAt < this.config.calibrationMs) return;
    const indices = LANDMARK_SETS[this.config.landmarkSet].indices;
    const built = calibrationPairs(c.frames, indices, this.config.calibrationPairGap);
    c.maxRotationDeg = built.maxRotationDeg;
    if (built.maxRotationDeg > this.config.maxCalibrationMotionDeg) {
      c.phase = 'rejected';
      c.message = { key: 'live.calibrationMotion', params: { motion: built.maxRotationDeg.toFixed(1), limit: this.config.maxCalibrationMotionDeg } };
      return;
    }
    const result = calibrateNoise(built.pairs);
    c.result = result;
    if (result.state === 'ok') {
      c.phase = 'done';
      c.message = { key: 'live.calibrationOk', params: { frames: built.frames, pairs: built.pairs.length } };
    } else {
      c.phase = 'rejected';
      c.message = { key: 'live.calibrationFailed', params: { message: result.message ?? result.state } };
    }
    this.history = [];
  }

  private analyze(): void {
    const current = this.buffer.latest();
    if (!current || this.latestFrame === null) {
      this.latestAnalysis = null;
      this.analysisNote = { key: 'live.noFace' };
      return;
    }
    const window = this.buffer.within(this.config.windowMs);
    const sel = selectReference(window, current);
    if (!sel) {
      this.latestAnalysis = null;
      this.analysisNote = { key: 'live.noPose' };
      return;
    }
    this.latestReference = sel.ref;
    const corr = correspondences(sel.ref, current, LANDMARK_SETS[this.config.landmarkSet].indices);
    const a = analyzePair(corr, {
      sigmaPx: this.sigmaPx,
      alpha: this.config.alpha,
      minMotionDeg: this.config.minMotionDeg,
      motionDeg: sel.motionDeg,
    });
    this.latestAnalysis = a;
    this.analysisNote = null;
    this.history.push({ t: current.timestampMs, tOverDof: a.stat ? a.stat.T / a.stat.dof : null, motionDeg: sel.motionDeg });
    if (this.history.length > 300) this.history.shift();
  }

  faceSizePx(): number | null {
    return this.latestFrame ? interocularPx(this.latestFrame) : null;
  }
}

export const session = new LiveSession();
