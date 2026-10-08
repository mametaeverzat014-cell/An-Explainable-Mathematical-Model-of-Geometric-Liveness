// Sensor availability report (Phase 1: detection only; Phase 3 will record
// gyroscope data). Nothing here requests permission or starts a sensor.

export interface SensorReport {
  secureContext: boolean;
  deviceMotionEvent: boolean;
  deviceOrientationEvent: boolean;
  /** iOS Safari requires an explicit permission request from a user gesture. */
  motionPermissionApi: boolean;
  genericSensorGyroscope: boolean;
  touchDevice: boolean;
}

export function sensorReport(): SensorReport {
  const dme = (window as unknown as { DeviceMotionEvent?: { requestPermission?: unknown } }).DeviceMotionEvent;
  return {
    secureContext: window.isSecureContext,
    deviceMotionEvent: typeof dme !== 'undefined',
    deviceOrientationEvent: typeof (window as unknown as { DeviceOrientationEvent?: unknown }).DeviceOrientationEvent !== 'undefined',
    motionPermissionApi: typeof dme?.requestPermission === 'function',
    genericSensorGyroscope: 'Gyroscope' in window,
    touchDevice: navigator.maxTouchPoints > 0,
  };
}
