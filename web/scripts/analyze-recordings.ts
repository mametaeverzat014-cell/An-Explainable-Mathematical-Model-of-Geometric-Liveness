// Reproducible offline analysis of exported recordings with the SAME code the
// app's Analyze view uses (analyzeSet in src/core/recording-analysis.ts).
//
//   npm run analyze -- [options] <files...>
//
// Files may be single recordings or experiment bundles. Calibration (hold
// still) and trial recordings are recognised by their role.
//
// Options (fix them in the protocol BEFORE looking at results):
//   --set rigid|features|interior|all   landmark set            (default rigid)
//   --window <ms>                        pair-rule window        (default 1500)
//   --min-motion <deg>                   motion gate             (default 5)
//   --alpha <a>                          significance level      (default 0.05)
//   --gap <frames>                       calibration pair gap    (default 5)
//   --max-cal-motion <deg>               calibration motion cap  (default 1)
//   --sigma <px>                         fixed sigma instead of calibration recordings
//   --calibration <file>                 (optional) an extra calibration file
//   --out <dir>                          output directory        (default analysis-out)
//
// Writes <out>/pairs.csv and <out>/summary.json.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { LANDMARK_SETS, type LandmarkSetId } from '../src/core/landmark-sets';
import { parseRecordingsFile, type Recording } from '../src/core/recording';
import { analyzeSet, rowsToCsv, DEFAULT_SET_SETTINGS } from '../src/core/recording-analysis';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    calibration: { type: 'string' },
    sigma: { type: 'string' },
    set: { type: 'string', default: DEFAULT_SET_SETTINGS.landmarkSet },
    window: { type: 'string', default: String(DEFAULT_SET_SETTINGS.windowMs) },
    'min-motion': { type: 'string', default: String(DEFAULT_SET_SETTINGS.minMotionDeg) },
    alpha: { type: 'string', default: String(DEFAULT_SET_SETTINGS.alpha) },
    gap: { type: 'string', default: String(DEFAULT_SET_SETTINGS.calibrationGap) },
    'max-cal-motion': { type: 'string', default: String(DEFAULT_SET_SETTINGS.maxCalibrationMotionDeg) },
    out: { type: 'string', default: 'analysis-out' },
  },
});

function die(msg: string): never {
  console.error(`error: ${msg}`);
  process.exit(1);
}
const num = (s: string | undefined, name: string) => {
  const v = Number(s);
  if (!Number.isFinite(v)) die(`--${name} must be a number`);
  return v;
};

const setId = values.set as LandmarkSetId;
if (!(setId in LANDMARK_SETS)) die(`unknown landmark set "${values.set}"`);
const files = [...(values.calibration ? [values.calibration] : []), ...positionals];
if (!files.length) die('no recording files given');

const recordings: Recording[] = [];
for (const f of files) {
  try {
    recordings.push(...parseRecordingsFile(readFileSync(f, 'utf8')));
  } catch (e) {
    die(`${f}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

const settings = {
  landmarkSet: setId,
  windowMs: num(values.window, 'window'),
  minMotionDeg: num(values['min-motion'], 'min-motion'),
  alpha: num(values.alpha, 'alpha'),
  calibrationGap: num(values.gap, 'gap'),
  maxCalibrationMotionDeg: num(values['max-cal-motion'], 'max-cal-motion'),
  fixedSigmaPx: values.sigma !== undefined ? num(values.sigma, 'sigma') : null,
};
const a = analyzeSet(recordings, settings);
if (!a.ok) die(a.error);

const c = a.calibration;
console.log(
  c.source === 'fixed'
    ? `sigma = ${c.sigmaPx} px (fixed)`
    : `sigma = ${c.sigmaPx.toFixed(3)} px [${c.interval95?.map((v) => v.toFixed(3)).join(', ') ?? '—'}] from ${c.calibrationRecordings} calibration recording(s)` +
        (c.rejectedForMotion.length ? `; rejected for motion: ${c.rejectedForMotion.join(', ')}` : ''),
);
const f3 = (v: number) => v.toFixed(3);
for (const s of a.summaries) {
  const rate = s.rejectionRate === null ? '—' : `${s.rejected}/${s.tested} = ${f3(s.rejectionRate)}`;
  console.log(`  ${s.condition.padEnd(20)} ${s.recordingId.padEnd(28)} ${s.pairs} pairs, rejected ${rate}${s.blockedStage ? `  (${s.blockedStage})` : ''}`);
}
for (const k of a.conditions) {
  const ci = k.clusterCi95 ? `95 % CI [${k.clusterCi95.map(f3).join(', ')}], design effect ${f3(k.designEffect!)}` : '(interval needs ≥ 2 recordings with tested pairs)';
  console.log(`${k.condition}: ${k.recordings} recordings, rejected ${k.rejected}/${k.tested} = ${k.pooledRate === null ? '—' : f3(k.pooledRate)} ${ci}`);
}

mkdirSync(values.out!, { recursive: true });
writeFileSync(join(values.out!, 'pairs.csv'), rowsToCsv(a.rows));
writeFileSync(
  join(values.out!, 'summary.json'),
  JSON.stringify(
    {
      note: 'Pairs within one recording are not independent. Use clusterCi95 (Wilson interval with a design-effect correction and Student t over recordings) for inference; pooledCi95 treats pairs as independent and is too narrow.',
      settings: a.settings,
      calibration: a.calibration,
      conditions: a.conditions,
      recordings: a.summaries,
    },
    null,
    2,
  ) + '\n',
);
console.log(`wrote ${join(values.out!, 'pairs.csv')} and summary.json`);
