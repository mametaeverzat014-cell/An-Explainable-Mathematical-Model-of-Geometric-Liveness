import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));

/**
 * Directory of the student-owned method implementation. Defaults to
 * src/student. Setting STUDENT_IMPL_DIR lets a supervisor (or the
 * maintainers, when checking that the validation tests are satisfiable) run
 * the same tests and app against another implementation without editing code.
 */
export function studentAlias(): Record<string, string> {
  const dir = process.env.STUDENT_IMPL_DIR ?? resolve(here, 'src/student');
  return { '@student': resolve(dir, 'index.ts') };
}
