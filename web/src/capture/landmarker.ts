import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { rotationFrom4x4 } from '../core/geometry3d';
import type { LandmarkFrame } from '../core/frames';

// Wrapper around the MediaPipe Face Landmarker. All assets are served from
// this origin (public/models, public/wasm; see scripts/fetch-assets.mjs).

export type Delegate = 'CPU' | 'GPU';

export interface LandmarkerHandle {
  landmarker: FaceLandmarker;
  delegate: Delegate;
}

const ASSET_BASE = import.meta.env.BASE_URL;

export async function createLandmarker(delegate: Delegate): Promise<LandmarkerHandle> {
  const fileset = await FilesetResolver.forVisionTasks(`${ASSET_BASE}wasm`);
  const landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${ASSET_BASE}models/face_landmarker.task`, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
  });
  return { landmarker, delegate };
}

/** Create with the GPU delegate if possible, falling back to CPU. */
export async function createLandmarkerWithFallback(preferred: Delegate): Promise<LandmarkerHandle> {
  if (preferred === 'CPU') return createLandmarker('CPU');
  try {
    return await createLandmarker('GPU');
  } catch {
    return createLandmarker('CPU');
  }
}

export interface DetectionOutput {
  frame: LandmarkFrame | null;
  inferenceMs: number;
}

/**
 * Run detection on the current video frame. Landmarks are converted from
 * MediaPipe's normalised coordinates (x / width, y / height) to PIXELS, so
 * that noise is measured in the same unit along both axes. The 10 iris
 * points (468-477) are dropped. MediaPipe's z output is not used.
 */
export function detect(handle: LandmarkerHandle, video: HTMLVideoElement, timestampMs: number): DetectionOutput {
  const t0 = performance.now();
  const result = handle.landmarker.detectForVideo(video, timestampMs);
  const inferenceMs = performance.now() - t0;
  const face = result.faceLandmarks[0];
  if (!face || face.length < 468) return { frame: null, inferenceMs };
  const w = video.videoWidth;
  const h = video.videoHeight;
  const points = face.slice(0, 468).map((p) => ({ x: p.x * w, y: p.y * h }));
  const matrix = result.facialTransformationMatrixes?.[0];
  const rotation = matrix && matrix.data.length === 16 ? rotationFrom4x4(matrix.data) : null;
  return { frame: { timestampMs, width: w, height: h, points, rotation }, inferenceMs };
}
