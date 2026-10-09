import { mean, quantile } from '../core/stats';

// Frame-timing diagnostics. When the browser supports
// HTMLVideoElement.requestVideoFrameCallback, timestamps come from the
// video pipeline (captureTime if the camera provides it, else mediaTime);
// otherwise from requestAnimationFrame, which is less precise. The source is
// always reported, because Phase 3 (gyroscope synchronisation) depends on it.

export type TimestampSource = 'captureTime' | 'mediaTime' | 'animationFrame';

export interface FrameTick {
  /** Timestamp used for analysis, ms. */
  timestampMs: number;
  source: TimestampSource;
  /** Frames the compositor presented since the previous callback (from rVFC metadata), or null. */
  presentedFramesDelta: number | null;
}

export interface TimingSummary {
  frames: number;
  source: TimestampSource | 'none';
  meanIntervalMs: number;
  p95IntervalMs: number;
  maxIntervalMs: number;
  fps: number;
  /** Frames the video presented but we did not process (rVFC only). */
  skippedFrames: number | null;
  nonMonotonic: number;
  meanInferenceMs: number;
  p95InferenceMs: number;
  detectionRate: number;
}

export class TimingStats {
  private intervals: number[] = [];
  private inference: number[] = [];
  private lastTs: number | null = null;
  private source: TimestampSource | 'none' = 'none';
  private skipped = 0;
  private hasPresented = false;
  private nonMonotonic = 0;
  private detected = 0;
  private total = 0;

  constructor(private readonly maxSamples = 600) {}

  record(tick: FrameTick, inferenceMs: number, faceFound: boolean): void {
    this.source = tick.source;
    this.total++;
    if (faceFound) this.detected++;
    if (this.lastTs !== null) {
      const d = tick.timestampMs - this.lastTs;
      if (d <= 0) this.nonMonotonic++;
      else this.push(this.intervals, d);
    }
    this.lastTs = tick.timestampMs;
    this.push(this.inference, inferenceMs);
    if (tick.presentedFramesDelta !== null) {
      this.hasPresented = true;
      if (tick.presentedFramesDelta > 1) this.skipped += tick.presentedFramesDelta - 1;
    }
  }

  intervalSamples(): number[] {
    return [...this.intervals];
  }

  inferenceSamples(): number[] {
    return [...this.inference];
  }

  summary(): TimingSummary {
    const m = mean(this.intervals);
    return {
      frames: this.total,
      source: this.source,
      meanIntervalMs: m,
      p95IntervalMs: quantile(this.intervals, 0.95),
      maxIntervalMs: this.intervals.length ? Math.max(...this.intervals) : Number.NaN,
      fps: m > 0 ? 1000 / m : Number.NaN,
      skippedFrames: this.hasPresented ? this.skipped : null,
      nonMonotonic: this.nonMonotonic,
      meanInferenceMs: mean(this.inference),
      p95InferenceMs: quantile(this.inference, 0.95),
      detectionRate: this.total ? this.detected / this.total : Number.NaN,
    };
  }

  reset(): void {
    this.intervals = [];
    this.inference = [];
    this.lastTs = null;
    this.skipped = 0;
    this.hasPresented = false;
    this.nonMonotonic = 0;
    this.detected = 0;
    this.total = 0;
  }

  private push(arr: number[], v: number): void {
    arr.push(v);
    if (arr.length > this.maxSamples) arr.shift();
  }
}

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: VideoFrameCallbackMetadata) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

interface VideoFrameCallbackMetadata {
  presentedFrames: number;
  mediaTime: number;
  captureTime?: number;
}

export function supportsVideoFrameCallback(video: HTMLVideoElement): boolean {
  return typeof (video as VideoWithRvfc).requestVideoFrameCallback === 'function';
}

/** Call onFrame once per new video frame until the returned stop function is called. */
export function frameLoop(video: HTMLVideoElement, onFrame: (tick: FrameTick) => void): () => void {
  const v = video as VideoWithRvfc;
  let stopped = false;
  if (v.requestVideoFrameCallback) {
    let lastPresented: number | null = null;
    let handle = 0;
    const cb = (_now: number, meta: VideoFrameCallbackMetadata) => {
      if (stopped) return;
      const source: TimestampSource = typeof meta.captureTime === 'number' ? 'captureTime' : 'mediaTime';
      const ts = source === 'captureTime' ? (meta.captureTime as number) : meta.mediaTime * 1000;
      const delta = lastPresented === null ? null : meta.presentedFrames - lastPresented;
      lastPresented = meta.presentedFrames;
      // Re-register first: an exception in onFrame must not stop the loop.
      handle = v.requestVideoFrameCallback!(cb);
      onFrame({ timestampMs: ts, source, presentedFramesDelta: delta });
    };
    handle = v.requestVideoFrameCallback(cb);
    return () => {
      stopped = true;
      v.cancelVideoFrameCallback?.(handle);
    };
  }
  let lastTime = -1;
  let raf = 0;
  const loop = () => {
    if (stopped) return;
    raf = requestAnimationFrame(loop);
    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      onFrame({ timestampMs: performance.now(), source: 'animationFrame', presentedFramesDelta: null });
    }
  };
  raf = requestAnimationFrame(loop);
  return () => {
    stopped = true;
    cancelAnimationFrame(raf);
  };
}
