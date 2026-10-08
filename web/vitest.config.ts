import { defineConfig } from 'vitest/config';
import { studentAlias } from './studentAlias.ts';

// Infrastructure tests: everything except the student validation suite.
export default defineConfig({
  resolve: { alias: studentAlias() },
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['src/student/tests/**', 'node_modules/**'],
    environment: 'node',
  },
});
