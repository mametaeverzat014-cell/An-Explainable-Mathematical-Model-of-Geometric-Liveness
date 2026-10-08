import { defineConfig } from 'vitest/config';
import { studentAlias } from './studentAlias.ts';

// Validation suite for the student-owned method (web/docs/MATH_SPEC.md).
// These tests FAIL until the corresponding functions are implemented.
export default defineConfig({
  resolve: { alias: studentAlias() },
  test: {
    include: ['src/student/tests/**/*.test.ts'],
    environment: 'node',
    testTimeout: 120_000,
  },
});
