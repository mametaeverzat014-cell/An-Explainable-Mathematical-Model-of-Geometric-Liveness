import { defineConfig } from 'vitest/config';
import { methodAlias } from './methodAlias.ts';

// Validation suite for the planarity method (web/docs/MATH_SPEC.md).
export default defineConfig({
  resolve: { alias: methodAlias() },
  test: {
    include: ['src/method/tests/**/*.test.ts'],
    environment: 'node',
    testTimeout: 120_000,
  },
});
