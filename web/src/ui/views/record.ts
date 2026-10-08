import { frameRotationDeg, type LandmarkFrame } from '../../core/frames';
import {
  frameToRecorded,
  newRecordingId,
  RECORDING_FORMAT,
  RECORDING_VERSION,
  recordingFileName,
  validateRecording,
  type Recording,
  type RecordingRole,
  type SubjectCategory,
} from '../../core/recording';
import { card, clear, field, fmt, h, note, numberInput, row, select } from '../dom';
import { session } from '../session';

const CONDITIONS = [
  { value: 'flat-print', label: 'Flat print (photo on paper, held rigid)' },
  { value: 'screen-photo', label: 'Still photo on a screen' },
  { value: 'curved-print', label: 'Curved / bent print' },
  { value: 'bona-fide', label: 'Bona fide (a real person)' },
  { value: 'replay', label: 'Replayed video on a screen' },
  { value: 'other', label: 'Other (describe in notes)' },
] as const;

interface FormState {
  category: SubjectCategory;
  approvalReference: string;
  pseudonym: string;
  consentConfirmed: boolean;
  role: RecordingRole;
  condition: string;
  protocolId: string;
  notes: string;
  durationS: number;
}

/** Recordings made in this tab. Memory only: lost on reload unless downloaded. */
const madeThisSession: Recording[] = [];

export function recordView(): { element: HTMLElement; dispose: () => void } {
  const form: FormState = {
    category: 'non-human-target',
    approvalReference: '',
    pseudonym: '',
    consentConfirmed: false,
    role: 'calibration',
    condition: 'flat-print',
    protocolId: '',
    notes: '',
    durationS: 3,
  };

  const overlay = h('canvas', { class: 'overlay', 'aria-hidden': 'true' }) as HTMLCanvasElement;
  const stage = h('div', { class: 'stage mirrored' }, session.video, overlay);
  const camBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, 'Start camera') as HTMLButtonElement;
  camBtn.addEventListener('click', () => void session.start('GPU', 'user'));
  const recBtn = h('button', { class: 'btn', type: 'button' }, 'Record') as HTMLButtonElement;
  const stopBtn = h('button', { class: 'btn btn-secondary', type: 'button', disabled: true }, 'Stop') as HTMLButtonElement;
  const gateNote = h('div');
  const progress = h('progress', { max: 1, value: 0 }) as HTMLProgressElement;
  const liveStats = h('div');
  const listSlot = h('div');

  // ---------------- form ----------------
  const humanFields = h('div', { class: 'stack', style: 'gap:8px' });
  const approvalInput = h('input', { type: 'text', placeholder: 'e.g. school ethics committee decision no. / IRB number', style: 'width:100%' }) as HTMLInputElement;
  approvalInput.addEventListener('input', () => ((form.approvalReference = approvalInput.value), updateGate()));
  const pseudoInput = h('input', { type: 'text', placeholder: 'P01', maxlength: 20, style: 'width:120px' }) as HTMLInputElement;
  pseudoInput.addEventListener('input', () => ((form.pseudonym = pseudoInput.value), updateGate()));
  const consentBox = h('input', { type: 'checkbox' }) as HTMLInputElement;
  consentBox.addEventListener('change', () => ((form.consentConfirmed = consentBox.checked), updateGate()));
  humanFields.append(
    field('Ethics approval reference', approvalInput, 'Required. Without an approval, recording people is not allowed.'),
    field('Pseudonym', pseudoInput, 'Letters, digits, "_" or "-". Never a name.'),
    h('label', { class: 'field', style: 'flex-direction:row;gap:8px;align-items:center' }, consentBox, h('span', {}, 'Informed consent (and parental consent for minors) has been obtained and recorded for this person.')),
  );

  const durationInput = numberInput(form.durationS, (v) => (form.durationS = Math.min(120, Math.max(1, v))), { min: 1, max: 120, step: 1 });
  const notesInput = h('textarea', { rows: 2, style: 'width:100%;font:inherit' }) as HTMLTextAreaElement;
  notesInput.addEventListener('input', () => (form.notes = notesInput.value));
  const protocolInput = h('input', { type: 'text', placeholder: 'optional', style: 'width:160px' }) as HTMLInputElement;
  protocolInput.addEventListener('input', () => (form.protocolId = protocolInput.value.trim()));

  const formCard = card(
    'What is being recorded',
    h(
      'div',
      { class: 'controls' },
      field(
        'Subject',
        select(
          [
            { value: 'non-human-target', label: 'A print or screen held by the researchers (no participant)' },
            { value: 'human-participant', label: 'A person (requires ethics approval)' },
          ],
          form.category,
          (v) => {
            form.category = v;
            updateGate();
          },
        ),
      ),
      field(
        'Role',
        select(
          [
            { value: 'calibration', label: 'Calibration: hold still' },
            { value: 'trial', label: 'Trial: move as the protocol says' },
          ],
          form.role,
          (v) => {
            form.role = v;
            form.durationS = v === 'calibration' ? 3 : 10;
            durationInput.value = String(form.durationS);
          },
        ),
      ),
      field('Condition', select(CONDITIONS.map((c) => ({ value: c.value, label: c.label })), 'flat-print', (v) => (form.condition = v))),
      field('Duration (s)', durationInput),
      field('Protocol ID', protocolInput),
    ),
    humanFields,
    field('Notes', notesInput, 'Describe the setup (lighting, distance, how the object was moved). No personal information.'),
  );

  function gateReason(): string | null {
    if (session.state !== 'running') return 'Start the camera first.';
    if (form.category === 'human-participant') {
      if (!form.approvalReference.trim()) return 'Recording a person requires an ethics approval reference.';
      if (!/^[A-Za-z0-9_-]{1,20}$/.test(form.pseudonym)) return 'Enter a pseudonym (letters, digits, "_" or "-").';
      if (!form.consentConfirmed) return 'Confirm that informed consent has been obtained.';
    }
    return null;
  }

  function updateGate(): void {
    humanFields.hidden = form.category !== 'human-participant';
    const reason = gateReason();
    recBtn.disabled = reason !== null || recording !== null;
    camBtn.disabled = session.state === 'running' || session.state === 'starting';
    clear(gateNote);
    if (reason && recording === null) gateNote.append(note(reason, 'warn'));
  }

  // ---------------- recording ----------------
  let recording: { frames: LandmarkFrame[]; noFace: number; startedAt: number; durationMs: number } | null = null;

  recBtn.addEventListener('click', () => {
    if (gateReason()) return;
    recording = { frames: [], noFace: 0, startedAt: performance.now(), durationMs: form.durationS * 1000 };
    stopBtn.disabled = false;
    updateGate();
  });
  stopBtn.addEventListener('click', () => finish());

  function onFrame(): void {
    drawOverlay();
    if (!recording) return;
    const f = session.latestFrame;
    if (f) {
      const last = recording.frames[recording.frames.length - 1];
      if (!last || f.timestampMs > last.timestampMs) recording.frames.push(f);
    } else recording.noFace++;
    const elapsed = performance.now() - recording.startedAt;
    progress.value = Math.min(1, elapsed / recording.durationMs);
    clear(liveStats);
    const first = recording.frames[0];
    const rot = first && f ? frameRotationDeg(first, f) : null;
    liveStats.append(
      row('Frames stored', String(recording.frames.length), 'measured'),
      row('Frames without a face', String(recording.noFace), 'mediapipe-estimate'),
      row('Rotation from first frame', rot === null ? '—' : `${fmt(rot, 1)}°`, 'mediapipe-estimate'),
    );
    if (elapsed >= recording.durationMs) finish();
  }

  function finish(): void {
    if (!recording) return;
    const r = recording;
    recording = null;
    stopBtn.disabled = true;
    progress.value = 0;
    if (r.frames.length < 2) {
      clear(liveStats);
      liveStats.append(note('Nothing recorded: no face was found in the frames.', 'warn'));
      updateGate();
      return;
    }
    const s = session.camera?.settings;
    const t0 = r.frames[0].timestampMs;
    const rec: Recording = {
      meta: {
        format: RECORDING_FORMAT,
        version: RECORDING_VERSION,
        id: newRecordingId(),
        createdAt: new Date().toISOString(),
        appVersion: import.meta.env.MODE === 'production' ? 'build' : 'dev',
        category: form.category,
        approvalReference: form.category === 'human-participant' ? form.approvalReference.trim() : null,
        pseudonym: form.category === 'human-participant' ? form.pseudonym : null,
        role: form.role,
        condition: form.condition,
        protocolId: form.protocolId || null,
        notes: form.notes,
        camera: {
          label: session.camera?.label || null,
          width: s?.width ?? r.frames[0].width,
          height: s?.height ?? r.frames[0].height,
          frameRate: s?.frameRate ?? null,
          facingMode: s?.facingMode ?? null,
        },
        timestampSource: session.timing.summary().source,
        landmarkModel: 'MediaPipe face_landmarker float16/1 (sha256 64184e22…)',
        framesWithoutFace: r.noFace,
      },
      frames: r.frames.map((f) => frameToRecorded(f, t0)),
    };
    try {
      validateRecording(rec);
      madeThisSession.unshift(rec);
    } catch (e) {
      clear(liveStats);
      liveStats.append(note(`Recording rejected: ${e instanceof Error ? e.message : String(e)}`, 'warn'));
    }
    renderList();
    updateGate();
  }

  function download(rec: Recording): void {
    const blob = new Blob([JSON.stringify(rec)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = h('a', { href: url, download: recordingFileName(rec.meta) });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function renderList(): void {
    clear(listSlot);
    if (!madeThisSession.length) {
      listSlot.append(h('p', { class: 'outcome-reason' }, 'No recordings yet in this tab.'));
      return;
    }
    for (const rec of madeThisSession) {
      const frames = rec.frames.length;
      const dur = frames ? rec.frames[frames - 1].t / 1000 : 0;
      const dl = h('button', { class: 'btn btn-secondary', type: 'button' }, 'Download JSON');
      dl.addEventListener('click', () => download(rec));
      const del = h('button', { class: 'btn btn-secondary', type: 'button' }, 'Discard');
      del.addEventListener('click', () => {
        madeThisSession.splice(madeThisSession.indexOf(rec), 1);
        renderList();
      });
      listSlot.append(
        h(
          'div',
          { class: 'card', style: 'margin-bottom:8px' },
          h('div', { class: 'outcome-title' }, `${rec.meta.role} · ${rec.meta.condition}${rec.meta.pseudonym ? ` · ${rec.meta.pseudonym}` : ''}`),
          h('div', { class: 'outcome-reason' }, `${frames} frames, ${fmt(dur, 1)} s, ${rec.meta.framesWithoutFace} frames without a face · id ${rec.meta.id}`),
          h('div', { class: 'btn-row', style: 'margin-top:6px' }, dl, del),
        ),
      );
    }
  }

  function drawOverlay(): void {
    const v = session.video;
    if (!v.videoWidth) return;
    if (overlay.width !== v.videoWidth) {
      overlay.width = v.videoWidth;
      overlay.height = v.videoHeight;
    }
    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const f = session.latestFrame;
    if (!f) return;
    ctx.fillStyle = recording ? 'rgba(235,104,52,0.9)' : 'rgba(255,255,255,0.5)';
    const s = overlay.width / 1280;
    for (const p of f.points) ctx.fillRect(p.x - 1.2 * s, p.y - 1.2 * s, 2.4 * s, 2.4 * s);
  }

  const element = h(
    'div',
    {},
    h('h1', {}, 'Record'),
    h(
      'p',
      { class: 'page-intro' },
      'Research mode. Stores landmark tracks only (468 points and head pose per frame), never video, and keeps them in this tab until you download them. Analyse downloaded files with `npm run analyze` (see web/docs/RESEARCH_MODE.md).',
    ),
    h(
      'div',
      { class: 'grid-live' },
      h(
        'div',
        { class: 'stack' },
        h('section', { class: 'card' }, stage, h('div', { class: 'btn-row', style: 'margin-top:10px' }, camBtn, recBtn, stopBtn), progress, gateNote, liveStats),
        formCard,
      ),
      h(
        'div',
        { class: 'stack' },
        card('Recordings in this tab', listSlot, note('Not saved anywhere until you press “Download JSON”. Reloading the page discards them.')),
        card(
          'Before recording',
          h('ul', { style: 'margin:0;padding-left:18px' }, ...[
            'Fix the protocol (conditions, motion, duration, landmark set, α, pair rule) before the first recording, and record its ID.',
            'Record a calibration hold for every setup (same camera, distance and light) before its trials.',
            'Prints and screens held by the researchers need no participant. Confirm with your supervisor whether a print of your own face needs approval.',
            'People: only with an approval reference and recorded consent.',
          ].map((t) => h('li', {}, t))),
        ),
      ),
    ),
  );

  const unsub = session.subscribe(updateGate);
  const unsubFrame = session.onFrame(onFrame);
  updateGate();
  renderList();
  return {
    element,
    dispose: () => {
      unsub();
      unsubFrame();
      if (recording) finish();
      session.parkVideo();
    },
  };
}
