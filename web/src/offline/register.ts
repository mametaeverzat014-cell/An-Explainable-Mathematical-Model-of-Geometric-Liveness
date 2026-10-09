// Registers the service worker (src/offline/service-worker.js, built to
// sw.js) in the production build, and reports whether the app is ready to
// run offline.

export type OfflineStatus = 'dev' | 'unsupported' | 'pending' | 'ready' | 'error';

let registrationError = false;

export function registerOffline(onUpdated: () => void): void {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
  // A controller already present means an older version is running; when the
  // new worker takes over, the page should be reloaded to use it.
  const hadController = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (hadController) onUpdated();
  });
  navigator.serviceWorker.register('./sw.js').catch(() => {
    registrationError = true;
  });
}

/** Must match PREFIX in service-worker.js. */
function cachePrefix(scope: string): string {
  return `parallax-lab:${new URL(scope).pathname}:`;
}

/** 'ready' means the active cache holds the page and the face model. */
export async function offlineStatus(): Promise<OfflineStatus> {
  if (!import.meta.env.PROD) return 'dev';
  if (!('serviceWorker' in navigator) || typeof caches === 'undefined') return 'unsupported';
  if (registrationError) return 'error';
  try {
    const reg = await navigator.serviceWorker.getRegistration();
    if (!reg?.active) return 'pending';
    const prefix = cachePrefix(reg.scope);
    for (const key of await caches.keys()) {
      if (!key.startsWith(prefix)) continue;
      const cache = await caches.open(key);
      const page = await cache.match(new URL('index.html', reg.scope).href);
      const model = await cache.match(new URL('models/face_landmarker.task', reg.scope).href);
      if (page && model) return 'ready';
    }
    return 'pending';
  } catch {
    return 'error';
  }
}
