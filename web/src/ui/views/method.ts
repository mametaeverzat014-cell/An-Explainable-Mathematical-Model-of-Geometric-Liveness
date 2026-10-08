import { CONFIG_NOTES, DEFAULT_CONFIG } from '../../core/config';
import { probeMethod } from '../../core/method';
import { PROVENANCE_HELP, PROVENANCE_LABEL, type Provenance } from '../../core/provenance';
import { badge, card, h, note, table } from '../dom';

interface Assumption {
  id: string;
  statement: string;
  whyItMatters: string;
  status: string;
}

export const ASSUMPTIONS: Assumption[] = [
  { id: 'A1', statement: 'Ideal pinhole camera (no lens distortion).', whyItMatters: 'Distortion bends straight lines, so even a flat print stops mapping by a homography; wide-angle webcams distort near the image edges.', status: 'Assumed. Untested on real cameras.' },
  { id: 'A2', statement: 'Global shutter: every pixel of a frame is captured at the same instant.', whyItMatters: 'Most webcam and phone sensors use a rolling shutter. A moving flat print is then sheared non-projectively and can look non-planar. Prediction: false rejections grow with angular speed.', status: 'Known to be violated by most devices. Effect size unknown.' },
  { id: 'A3', statement: 'Each landmark is the image of a fixed physical point on the object.', whyItMatters: 'False for silhouette points of a real face (they slide along the outline) and for points that move with expressions. The rigid landmark set is chosen to limit this.', status: 'Design mitigation only.' },
  { id: 'A4', statement: 'A presentation attack instrument (PAI) is rigid and flat.', whyItMatters: 'Curved or bent prints are not planar. In the Synthetic lab a print curved with a 10 cm radius is rarely rejected at small rotations and more often at larger ones, so the test gives no guarantee about curved prints in either direction.', status: 'Violated by curved prints.' },
  { id: 'A5', statement: 'Landmark errors are independent, Gaussian, equal in x and y, with known σ.', whyItMatters: 'The χ²(2n − 8) reference distribution, and therefore every p-value, depends on it. MediaPipe predicts all points jointly, so errors are probably correlated.', status: 'Open question Q1. Must be tested on flat prints.' },
  { id: 'A6', statement: 'σ during motion equals σ measured during the still calibration hold.', whyItMatters: 'Motion blur and tracking lag may raise the noise during motion, which would turn ordinary noise into apparent parallax.', status: 'Open question Q3.' },
  { id: 'A7', statement: 'On a flat photo, MediaPipe’s 2D landmark positions follow the image content.', whyItMatters: 'If the network “imagines” a 3D head and moves landmarks accordingly, a tilted photo produces non-planar landmark motion. This could invalidate the whole approach and must be measured first.', status: 'Open question Q4. Highest priority.' },
  { id: 'A8', statement: 'The presented content is static (a still image).', whyItMatters: 'A video replayed on a flat screen contains real head motion recorded earlier, so its landmarks are NOT planar-consistent. This test does not address replay attacks (Phase 3 does).', status: 'Violated by replay attacks, by definition.' },
];

export function methodView(): { element: HTMLElement; dispose: () => void } {
  const modules = probeMethod();
  const implemented = modules.filter((m) => m.implemented).length;

  const statusCard = card(
    'Implementation status of the method',
    h('p', { class: 'outcome-reason' }, `${implemented} of ${modules.length} method functions are implemented. Run \`npm run test:method\` for the validation suite.`),
    h(
      'div',
      { class: 'table-wrap' },
      table(
        ['Task', 'Function', 'Status'],
        modules.map((m) => [m.task, h('code', {}, m.name), h('span', { class: m.implemented ? 'status-ok' : 'status-missing' }, m.implemented ? 'implemented' : 'not implemented')]),
      ),
    ),
    note('“Implemented” only means the function no longer throws NotImplementedError. Correctness is established by the validation tests, not by this table.', 'info'),
  );

  const findingsCard = card(
    'Validated findings',
    h('p', {}, 'None yet.'),
    h(
      'p',
      { class: 'outcome-reason' },
      'A finding is listed here only after a pre-registered experiment with approved participants (Phase 2) has been run, and only with its uncertainty. Results from the Synthetic lab describe the method under idealised assumptions and are not findings about real presentations.',
    ),
  );

  const provenanceCard = card(
    'How to read the labels',
    ...(Object.keys(PROVENANCE_LABEL) as Provenance[]).map((p) => h('div', { class: 'label-row' }, badge(p), h('span', { class: 'kv-label' }, PROVENANCE_HELP[p]))),
  );

  const outcomesCard = card(
    'What the three outcomes mean',
    table(
      ['Outcome', 'Meaning', 'What it does NOT mean'],
      [
        ['Consistent with a planar surface', 'p ≥ α: the landmark motion is explained by one plane within the measured noise.', 'Not “this is an attack”. A real face that barely moved, or a noisy camera, gives the same result.'],
        ['Evidence against a planar surface', 'p < α with enough motion: residuals exceed what noise explains.', 'Not “this is a live person”. Curved prints, masks, replayed video and facial expressions also reject planarity.'],
        ['Inconclusive', 'Not enough rotation, no calibrated noise floor, or a method stage missing.', 'Not a failure of the person or the object.'],
      ],
    ),
  );

  const configCard = card(
    'Configuration and its justification',
    table(
      ['Setting', 'Default', 'Status', 'Justification'],
      (Object.keys(CONFIG_NOTES) as (keyof typeof CONFIG_NOTES)[]).map((k) => [
        h('code', {}, k),
        String(DEFAULT_CONFIG[k]),
        CONFIG_NOTES[k].status,
        CONFIG_NOTES[k].note,
      ]),
    ),
  );

  const assumptionsCard = card(
    'Assumptions behind the test',
    h('p', { class: 'outcome-reason' }, 'The test is exact only when all of these hold. Each one is either a known limitation or a question the research must answer.'),
    h('div', { class: 'table-wrap' }, table(['', 'Assumption', 'Why it matters', 'Status'], ASSUMPTIONS.map((a) => [a.id, a.statement, a.whyItMatters, a.status]))),
  );

  const element = h(
    'div',
    {},
    h('h1', {}, 'Method and status'),
    h(
      'p',
      { class: 'page-intro' },
      'Null hypothesis H₀: the selected landmarks in two frames are images of points on one plane, observed with independent Gaussian noise of standard deviation σ. Under H₀ the minimised sum of squared geometric errors divided by σ² follows approximately a χ² distribution with 2n − 8 degrees of freedom (Hartley & Zisserman, Multiple View Geometry, 2nd ed., ch. 4–5). The full specification is in web/docs/MATH_SPEC.md.',
    ),
    h('div', { class: 'grid-2' }, h('div', { class: 'stack' }, statusCard, findingsCard, outcomesCard), h('div', { class: 'stack' }, provenanceCard, configCard)),
    h('div', { style: 'margin-top:16px' }, assumptionsCard),
  );
  return { element, dispose: () => {} };
}
