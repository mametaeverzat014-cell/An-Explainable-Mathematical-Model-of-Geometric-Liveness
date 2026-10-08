// Camera access. The camera is started only by an explicit user action and
// frames never leave the page (see the Content-Security-Policy in vite.config.ts).

export interface CameraInfo {
  label: string;
  settings: MediaTrackSettings;
  capabilities: Partial<MediaTrackCapabilities> | null;
}

export interface CameraRequest {
  width?: number;
  height?: number;
  frameRate?: number;
  facingMode?: 'user' | 'environment';
  deviceId?: string;
}

export class CameraUnavailableError extends Error {}

export async function startCamera(video: HTMLVideoElement, req: CameraRequest = {}): Promise<{ stream: MediaStream; info: CameraInfo }> {
  if (!window.isSecureContext) {
    throw new CameraUnavailableError('Camera access needs a secure context (https:// or http://localhost).');
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new CameraUnavailableError('This browser does not provide camera access (navigator.mediaDevices.getUserMedia).');
  }
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      width: { ideal: req.width ?? 1280 },
      height: { ideal: req.height ?? 720 },
      frameRate: { ideal: req.frameRate ?? 30 },
      ...(req.deviceId ? { deviceId: { exact: req.deviceId } } : { facingMode: req.facingMode ?? 'user' }),
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  const track = stream.getVideoTracks()[0];
  return {
    stream,
    info: {
      label: track.label,
      settings: track.getSettings(),
      capabilities: typeof track.getCapabilities === 'function' ? track.getCapabilities() : null,
    },
  };
}

export function stopCamera(stream: MediaStream | null, video: HTMLVideoElement): void {
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
}
