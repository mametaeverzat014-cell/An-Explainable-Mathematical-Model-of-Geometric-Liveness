// Service worker: lets the app open and run without a network connection
// after the first visit (e.g. at a competition venue without Wi-Fi).
//
// This file is a template. The build (vite.config.ts, offlinePlugin) replaces
// __VERSION__ and __PRECACHE__ and writes the result to dist/sw.js.
//
// Privacy: it handles only GET requests to the app's own origin and path, and
// only stores the app's own files (page, scripts, model, WebAssembly) in the
// browser's cache. It never stores camera frames or landmarks and never
// contacts another server.

const VERSION = __VERSION__;
const PRECACHE = __PRECACHE__;

const SCOPE_PATH = new URL(self.registration.scope).pathname;
// Several projects can share one github.io origin, so cache names carry the
// scope path and only our own old caches are deleted. src/offline/register.ts
// uses the same prefix.
const PREFIX = `parallax-lab:${SCOPE_PATH}:`;
const CACHE = PREFIX + VERSION;

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // cache: 'reload' bypasses the HTTP cache, so the stored files belong
      // to this version and not to a stale earlier one.
      await cache.addAll(PRECACHE.map((path) => new Request(path, { cache: 'reload' })));
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) {
        if (key.startsWith(PREFIX) && key !== CACHE) await caches.delete(key);
      }
      await self.clients.claim();
    })(),
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(SCOPE_PATH)) return;
  event.respondWith(respond(req));
});

async function respond(req) {
  const cache = await caches.open(CACHE);
  // The app is a single page: every navigation inside the scope gets the
  // cached index.html of this version. A new deployment installs a new
  // service worker in the background; it takes effect on the next reload.
  if (req.mode === 'navigate') {
    const page = await cache.match('index.html', { ignoreVary: true });
    return page ?? fetch(req);
  }
  // ignoreVary: module scripts are requested with an Origin header that the
  // precache requests did not send; with "Vary: Origin" they would not match.
  // There is only one stored version of each file, so this is safe.
  const hit = await cache.match(req, { ignoreVary: true });
  if (hit) return hit;
  const res = await fetch(req);
  // Files not precached (e.g. the WebAssembly variant for browsers without
  // SIMD) are stored the first time they are used.
  if (res.status === 200 && res.type === 'basic') await cache.put(req, res.clone());
  return res;
}
