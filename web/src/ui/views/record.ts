import { frameRotationDeg } from '../../core/frames';
import { recordingFileName, type Recording, type RecordingRole, type SubjectCategory } from '../../core/recording';
import { card, clear, field, fmt, h, note, numberInput, row, select } from '../dom';
import { t, type MessageKey } from '../i18n';
import { buildRecording, Capture, downloadJson, tabRecordings } from '../recorder';
import { session } from '../session';
import { cameraStage } from '../stage';

export const CONDITIONS = ['flat-print', 'screen-photo', 'curved-print', 'bona-fide', 'replay', 'other'] as const;

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

  let capture: Capture | null = null;
  const stage = cameraStage(() => capture?.active ?? false);
  const camBtn = h('button', { class: 'btn btn-secondary', type: 'button' }, t('common.startCamera')) as HTMLButtonElement;
  camBtn.addEventListener('click', () => void session.start('GPU', 'user'));
  const recBtn = h('button', { class: 'btn', type: 'button' }, t('rec.record')) as HTMLButtonElement;
  const stopBtn = h('button', { class: 'btn btn-secondary', type: 'button', disabled: true }, t('common.stop')) as HTMLButtonElement;
  const gateNote = h('div');
  const progress = h('progress', { max: 1, value: 0 }) as HTMLProgressElement;
  const liveStats = h('div');
  const listSlot = h('div');

  // ---------------- form ----------------
  const humanFields = h('div', { class: 'stack', style: 'gap:8px' });
  const approvalInput = h('input', { type: 'text', placeholder: t('rec.approvalPlaceholder'), style: 'width:100%' }) as HTMLInputElement;
  approvalInput.addEventListener('input', () => ((form.approvalReference = approvalInput.value), updateGate()));
  const pseudoInput = h('input', { type: 'text', placeholder: 'P01', maxlength: 20, style: 'width:120px' }) as HTMLInputElement;
  pseudoInput.addEventListener('input', () => ((form.pseudonym = pseudoInput.value), updateGate()));
  const consentBox = h('input', { type: 'checkbox' }) as HTMLInputElement;
  consentBox.addEventListener('change', () => ((form.consentConfirmed = consentBox.checked), updateGate()));
  humanFields.append(
    field(t('rec.approval'), approvalInput, t('rec.approvalHint')),
    field(t('rec.pseudonym'), pseudoInput, t('rec.pseudonymHint')),
    h('label', { class: 'field', style: 'flex-direction:row;gap:8px;align-items:center' }, consentBox, h('span', {}, t('rec.consent'))),
  );

  const durationInput = numberInput(form.durationS, (v) => (form.durationS = Math.min(120, Math.max(1, v))), { min: 1, max: 120, step: 1 });
  const notesInput = h('textarea', { rows: 2, style: 'width:100%;font:inherit' }) as HTMLTextAreaElement;
  notesInput.addEventListener('input', () => (form.notes = notesInput.value));
  const protocolInput = h('input', { type: 'text', placeholder: t('rec.optional'), style: 'width:160px' }) as HTMLInputElement;
  protocolInput.addEventListener('input', () => (form.protocolId = protocolInput.value.trim()));

  const formCard = card(
    t('rec.cardWhat'),
    h(
      'div',
      { class: 'controls' },
      field(
        t('rec.subject'),
        select(
          [
            { value: 'non-human-target', label: t('rec.subjectTarget') },
            { value: 'human-participant', label: t('rec.subjectHuman') },
          ],
          form.category,
          (v) => {
            form.category = v;
            updateGate();
          },
        ),
      ),
      field(
        t('rec.role'),
        select(
          [
            { value: 'calibration', label: t('rec.roleCalibration') },
            { value: 'trial', label: t('rec.roleTrial') },
          ],
          form.role,
          (v) => {
            form.role = v;
            form.durationS = v === 'calibration' ? 3 : 10;
            durationInput.value = String(form.durationS);
          },
        ),
      ),
      field(t('rec.condition'), select(CONDITIONS.map((c) => ({ value: c, label: t(`cond.${c}` as MessageKey) })), 'flat-print', (v) => (form.condition = v))),
      field(t('rec.duration'), durationInput),
      field(t('rec.protocolId'), protocolInput),
    ),
    humanFields,
    field(t('rec.notes'), notesInput, t('rec.notesHint')),
  );

  function gateReason(): string | null {
    if (session.state !== 'running') return t('rec.gateCamera');
    if (form.category === 'human-participant') {
      if (!form.approvalReference.trim()) return t('rec.gateApproval');
      if (!/^[A-Za-z0-9_-]{1,20}$/.test(form.pseudonym)) return t('rec.gatePseudonym');
      if (!form.consentConfirmed) return t('rec.gateConsent');
    }
    return null;
  }

  function updateGate(): void {
    humanFields.hidden = form.category !== 'human-participant';
    const reason = gateReason();
    const busy = capture?.active ?? false;
    recBtn.disabled = reason !== null || busy;
    stopBtn.disabled = !busy;
    camBtn.disabled = session.state === 'running' || session.state === 'starting';
    clear(gateNote);
    if (reason && !busy) gateNote.append(note(reason, 'warn'));
  }

  recBtn.addEventListener('click', () => {
    if (gateReason()) return;
    const info = {
      category: form.category,
      approvalReference: form.approvalReference.trim(),
      pseudonym: form.pseudonym,
      role: form.role,
      condition: form.condition,
      protocolId: form.protocolId || null,
      notes: form.notes,
    };
    capture = new Capture(
      form.durationS * 1000,
      (p) => {
        progress.value = Math.min(1, p.elapsedMs / p.durationMs);
        clear(liveStats);
        const first = p.frames[0];
        const rot = first && p.latest ? frameRotationDeg(first, p.latest) : null;
        liveStats.append(
          row(t('rec.framesStored'), String(p.frames.length), 'measured'),
          row(t('rec.framesNoFace'), String(p.noFace), 'mediapipe-estimate'),
          row(t('rec.rotationFromFirst'), rot === null ? '—' : `${fmt(rot, 1)}°`, 'mediapipe-estimate'),
        );
      },
      (frames, noFace) => {
        progress.value = 0;
        try {
          tabRecordings.unshift(buildRecording(frames, noFace, info));
        } catch (e) {
          clear(liveStats);
          const msg = e instanceof Error ? e.message : String(e);
          liveStats.append(note(frames.length < 2 ? t('rec.nothing') : t('rec.rejected', { message: msg }), 'warn'));
        }
        renderList();
        updateGate();
      },
    );
    capture.start();
    updateGate();
  });
  stopBtn.addEventListener('click', () => capture?.stop());

  function renderList(): void {
    clear(listSlot);
    if (!tabRecordings.length) {
      listSlot.append(h('p', { class: 'outcome-reason' }, t('rec.none')));
      return;
    }
    for (const rec of tabRecordings) listSlot.append(recordingCard(rec, renderList));
  }

  const element = h(
    'div',
    {},
    h('h1', {}, t('rec.title')),
    h('p', { class: 'page-intro' }, t('rec.intro')),
    h(
      'div',
      { class: 'grid-live' },
      h(
        'div',
        { class: 'stack' },
        h('section', { class: 'card' }, stage.element, h('div', { class: 'btn-row', style: 'margin-top:10px' }, camBtn, recBtn, stopBtn), progress, gateNote, liveStats),
        formCard,
      ),
      h(
        'div',
        { class: 'stack' },
        card(t('rec.cardList'), listSlot, note(t('rec.notSaved'))),
        card(
          t('rec.cardBefore'),
          h('ul', { style: 'margin:0;padding-left:18px' }, ...(['rec.before1', 'rec.before2', 'rec.before3', 'rec.before4'] as const).map((k) => h('li', {}, t(k)))),
        ),
      ),
    ),
  );

  const unsub = session.subscribe(updateGate);
  updateGate();
  renderList();
  return {
    element,
    dispose: () => {
      unsub();
      capture?.stop();
      stage.dispose();
    },
  };
}

/** A recording with download and discard buttons. */
export function recordingCard(rec: Recording, onChange: () => void): HTMLElement {
  const frames = rec.frames.length;
  const dur = frames ? rec.frames[frames - 1].t / 1000 : 0;
  const dl = h('button', { class: 'btn btn-secondary', type: 'button' }, t('rec.downloadJson'));
  dl.addEventListener('click', () => downloadJson(rec, recordingFileName(rec.meta)));
  const del = h('button', { class: 'btn btn-secondary', type: 'button' }, t('common.discard'));
  del.addEventListener('click', () => {
    const i = tabRecordings.indexOf(rec);
    if (i >= 0) tabRecordings.splice(i, 1);
    onChange();
  });
  const role = rec.meta.role === 'calibration' ? t('ana.roleCalibration') : t('ana.roleTrial');
  return h(
    'div',
    { class: 'card', style: 'margin-bottom:8px' },
    h('div', { class: 'outcome-title' }, `${role} · ${rec.meta.condition}${rec.meta.pseudonym ? ` · ${rec.meta.pseudonym}` : ''}`),
    h('div', { class: 'outcome-reason' }, t('rec.listSummary', { frames, seconds: fmt(dur, 1), noFace: rec.meta.framesWithoutFace, id: rec.meta.id })),
    h('div', { class: 'btn-row', style: 'margin-top:6px' }, dl, del),
  );
}
