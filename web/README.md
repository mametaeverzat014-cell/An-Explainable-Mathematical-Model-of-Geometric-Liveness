# Parallax Lab

An open research instrument that tests whether the landmarks of a presented
face move like points on a single plane. It runs entirely in the browser,
on laptops and phones. No video or landmark data leaves the device.

**Status: Phase 1 (instrument).** The statistical method itself
(`src/student/`) is to be implemented by the student authors. Until then the
app shows measurements and reports every method step as *not implemented*.
It does not invent results. There are no validated findings yet.

- What the test is and is not: [`docs/MATH_SPEC.md`](docs/MATH_SPEC.md),
  sections 2 and 9
- Student tasks: [`docs/MATH_SPEC.md`](docs/MATH_SPEC.md), section 4
- Audit and roadmap: [`docs/AUDIT_AND_PLAN.md`](docs/AUDIT_AND_PLAN.md)
- Who wrote what: [`../AI_ASSISTANCE.md`](../AI_ASSISTANCE.md)

## Requirements

Node.js 22 (or 20.19+) and npm. A browser with camera access: recent Chrome,
Edge, Firefox or Safari.

## Commands

```bash
cd web
npm ci                  # install exact dependency versions (package-lock.json)
npm run dev             # fetch assets, start dev server at http://localhost:5173
npm test                # infrastructure unit + integration tests (must pass)
npm run test:student    # validation suite for the students' method (fails until implemented)
npm run typecheck
npm run build           # production build in dist/ (adds the Content-Security-Policy)
npm run test:e2e        # browser smoke tests against the production build
```

`npm run dev` and `npm run build` first run `scripts/fetch-assets.mjs`. It
downloads the MediaPipe Face Landmarker model (verified against a pinned
SHA-256; the build refuses a mismatching file) and copies the MediaPipe
WebAssembly runtime from `node_modules`. Both are served from the app's own
origin.

For the end-to-end tests, Playwright needs a Chromium. Either run
`npx playwright install chromium`, or point `CHROMIUM_PATH` at an existing
Chromium binary.

## Testing on a phone

Browsers allow camera access only on `https://` or `localhost`. Options:

1. Deploy to GitHub Pages (below) and open the HTTPS URL on the phone.
2. Or run `npm run build && npx vite preview --host`. The phone then needs an
   HTTPS tunnel or a locally trusted certificate, because a plain LAN IP is
   not a secure context.

## Deployment (GitHub Pages)

The app is a static site, with relative paths (`base: './'`).

1. In the repository settings → Pages, set **Source: GitHub Actions**.
2. Run the workflow **“Deploy web app to GitHub Pages”** manually (Actions
   tab). It builds `web/` and publishes `web/dist`.

Any static host works: upload `web/dist/`. The CSP is a `<meta>` tag, so it
applies wherever the files are served.

## Structure

```
src/
  student/        method — OWNED BY THE STUDENT AUTHORS (stubs + validation tests)
  core/           infrastructure: linear algebra, statistics, synthetic scene,
                  landmark sets, frame buffer, method runner, Monte-Carlo
  capture/        camera, MediaPipe wrapper, frame timing, sensor detection
  ui/             views (Live, Synthetic lab, Diagnostics, Method, About), charts
  data/           canonical face geometry (from MediaPipe, Apache-2.0)
scripts/          asset fetching (checksummed), canonical-face conversion
e2e/              Playwright smoke tests
docs/             specification, audit and plan
```

`@student` is an import alias for `src/student/index.ts`. Setting
`STUDENT_IMPL_DIR=/path/to/dir` points it at another implementation, so a
supervisor can verify the tests independently.

## Privacy and data

- The camera starts only when the user presses *Start camera*.
- Frames and landmarks are processed in memory and discarded when the tab is
  closed. Phase 1 has no export and no storage.
- The production build forbids network connections to other origins
  (`connect-src 'self'`).
- Recording other people is research with human participants. Do not use this
  tool on anyone else until the required ethics approval (IRB/SRC for ISEF)
  exists.

## Licences

Code: Apache-2.0. MediaPipe model, runtime and canonical face geometry:
Apache-2.0 (Google).
