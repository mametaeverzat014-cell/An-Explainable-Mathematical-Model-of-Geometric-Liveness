import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));

/**
 * Directory of the method implementation. Defaults to
 * src/method. Setting METHOD_IMPL_DIR lets a supervisor (or the
 * maintainers, when checking that the validation tests are satisfiable) run
 * the same tests and app against another implementation without editing code.
 */
export function methodAlias(): Record<string, string> {
  const dir = process.env.METHOD_IMPL_DIR ?? resolve(here, 'src/method');
  return { '@method': resolve(dir, 'index.ts') };
}
