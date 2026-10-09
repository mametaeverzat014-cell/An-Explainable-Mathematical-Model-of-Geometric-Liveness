import { h } from './dom';
import { session } from './session';

/**
 * Camera preview with all landmarks drawn on top. Used by the Record view and
 * the Experiment wizard. `isRecording` changes the point colour so the user
 * can see when frames are being stored.
 */
export function cameraStage(isRecording: () => boolean): { element: HTMLElement; dispose: () => void } {
  const overlay = h('canvas', { class: 'overlay', 'aria-hidden': 'true' }) as HTMLCanvasElement;
  const element = h('div', { class: 'stage mirrored' }, session.video, overlay);

  function draw(): void {
    const v = session.video;
    if (!v.videoWidth) return;
    if (overlay.width !== v.videoWidth || overlay.height !== v.videoHeight) {
      overlay.width = v.videoWidth;
      overlay.height = v.videoHeight;
      // Match the stage to the camera's aspect ratio (portrait on phones).
      element.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
    }
    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const f = session.latestFrame;
    if (!f) return;
    ctx.fillStyle = isRecording() ? 'rgba(235,104,52,0.9)' : 'rgba(255,255,255,0.55)';
    const s = overlay.width / 1280;
    for (const p of f.points) ctx.fillRect(p.x - 1.2 * s, p.y - 1.2 * s, 2.4 * s, 2.4 * s);
  }

  const unsub = session.onFrame(draw);
  return {
    element,
    dispose: () => {
      unsub();
      session.parkVideo();
    },
  };
}
