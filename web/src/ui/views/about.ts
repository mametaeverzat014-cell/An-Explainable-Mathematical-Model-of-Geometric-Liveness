import { h } from '../dom';

export function aboutView(): { element: HTMLElement; dispose: () => void } {
  const element = h(
    'div',
    { class: 'prose' },
    h('h1', {}, 'About'),
    h('h2', {}, 'What this is'),
    h(
      'p',
      {},
      'An open research instrument for studying one geometric question: does the presented face move like a single plane? It is part of a school research project and is not a security product. It must not be used to grant or deny access to anything.',
    ),
    h('h2', {}, 'Privacy'),
    h('ul', {}, ...[
      'The camera starts only when you press “Start camera”.',
      'Video frames and landmarks are processed in this browser tab and kept in memory only. Nothing is uploaded, stored or sent anywhere.',
      'The production build sets a Content-Security-Policy with connect-src \'self\': the page is technically unable to send data to another server. The model and WebAssembly files are served from the same site.',
      'Closing or reloading the tab discards everything.',
    ].map((t) => h('li', {}, t))),
    h('h2', {}, 'Human participants'),
    h(
      'p',
      {},
      'Recording or analysing other people is research with human participants. No participant data may be collected with this tool until the ethics review required by the competition rules (for ISEF: IRB / SRC approval) has been obtained. Research-mode features (Phase 2) will require a recorded approval reference before they can be used.',
    ),
    h('h2', {}, 'Who wrote what'),
    h(
      'p',
      {},
      'The application, including the implementation of the statistical method (folder src/method/), was written with the AI assistant Claude (Anthropic) at the request of the project team. The student authors are responsible for understanding and defending the method, for the derivations and for the experiments. The repository file AI_ASSISTANCE.md records this in detail.',
    ),
    h('h2', {}, 'Third-party components'),
    h('ul', {}, ...[
      'MediaPipe Face Landmarker model and Tasks Vision runtime (Google, Apache-2.0).',
      'MediaPipe canonical face model geometry, used for the synthetic scene (Apache-2.0).',
    ].map((t) => h('li', {}, t))),
  );
  return { element, dispose: () => {} };
}
