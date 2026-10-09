import { describe, expect, it } from 'vitest';
import { en } from '../locales/en';
import { kk } from '../locales/kk';
import { ru } from '../locales/ru';

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe('translations', () => {
  for (const [name, dict] of [['ru', ru], ['kk', kk]] as const) {
    it(`${name}: same keys as English, no empty strings, same placeholders`, () => {
      expect(Object.keys(dict).sort()).toEqual(Object.keys(en).sort());
      for (const [key, value] of Object.entries(dict)) {
        expect(value.trim().length, key).toBeGreaterThan(0);
        expect(placeholders(value), key).toEqual(placeholders(en[key as keyof typeof en]));
      }
    });
  }
});
