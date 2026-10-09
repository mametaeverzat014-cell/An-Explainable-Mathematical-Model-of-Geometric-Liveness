import { h } from '../dom';
import { getLang, t } from '../i18n';

export function aboutView(): { element: HTMLElement; dispose: () => void } {
  const element = h(
    'div',
    { class: 'prose' },
    h('h1', {}, t('about.title')),
    h('h2', {}, t('about.whatTitle')),
    h('p', {}, t('about.what')),
    h('h2', {}, t('about.privacyTitle')),
    h('ul', {}, ...(['about.privacy1', 'about.privacy2', 'about.privacy3', 'about.privacy4'] as const).map((k) => h('li', {}, t(k)))),
    h('h2', {}, t('about.humansTitle')),
    h('p', {}, t('about.humans')),
    h('h2', {}, t('about.whoTitle')),
    h('p', {}, t('about.who')),
    getLang() === 'kk' ? h('p', { class: 'note note-warn' }, t('about.translationNote')) : null,
    h('h2', {}, t('about.thirdTitle')),
    h('ul', {}, h('li', {}, t('about.third1')), h('li', {}, t('about.third2'))),
  );
  return { element, dispose: () => {} };
}
