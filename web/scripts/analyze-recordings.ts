// Reproducible offline analysis of exported recordings with the SAME method
// code the app uses (src/student/). Usage:
//
//   npm run analyze -- --calibration cal.json [options] trial1.json trial2.json ...
//
// Options (fix them in the protocol BEFORE looking at results):
//   --set rigid|features|interior|all   landmark set            (default rigid)
//   --window <ms>                        pair-rule window        (default 1500)
//   --min-motion <deg>                   motion gate             (default 5)
//   --alpha <a>                          significance level      (default 0.05)
//   --gap <frames>                       calibration pair gap    (default 5)
//   --max-cal-motion <deg>               calibration motion cap  (default 1)
//   --sigma <px>                         use a fixed sigma instead of a calibration file
//   --out <dir>                          output directory        (default analysis-out)
//
// Writes <out>/pairs.csv (one row per analysed frame pair) and
// <out>/summary.json (settings, sigma, per-recording rejection rates).
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { LANDMARK_SETS, type LandmarkSetId } from '../src/core/landmark-sets';
import { parseRecording } from '../src/core/recording';
import { analyzeRecording, calibrationFromRecording, rowsToCsv, type PairRow } from '../src/core/recording-analysis';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    calibration: { type: 'string' },
    sigma: { type: 'string' },
    set: { type: 'string', default: 'rigid' },
    window: { type: 'string', default: '1500' },
    'min-motion': { type: 'string', default: '5' },
    alpha: { type: 'string', default: '0.05' },
    gap: { type: 'string', default: '5' },
    'max-cal-motion': { type: 'string', default: '1' },
    out: { type: 'string', default: 'analysis-out' },
  },
});

function die(msg: string): never {
  console.error(`error: ${msg}`);
  process.exit(1);
}

const setId = values.set as LandmarkSetId;
if (!(setId in LANDMARK_SETS)) die(`unknown landmark set "${values.set}"`);
const indices = LANDMARK_SETS[setId].indices;
const num = (s: string | undefined, name: string) => {
  const v = Number(s);
  if (!Number.isFinite(v)) die(`--${name} must be a number`);
  return v;
};
const opts = {
  indices,
  windowMs: num(values.window, 'window'),
  minMotionDeg: num(values['min-motion'], 'min-motion'),
  alpha: num(values.alpha, 'alpha'),
};
if (!positionals.length) die('no trial recordings given');

let sigmaPx: number | null = null;
let calibrationInfo: unknown = null;
if (values.sigma !== undefined) {
  sigmaPx = num(values.sigma, 'sigma');
  calibrationInfo = { source: 'fixed --sigma', sigmaPx };
} else if (values.calibration) {
  const cal = parseRecording(readFileSync(values.calibration, 'utf8'));
  const c = calibrationFromRecording(cal, indices, num(values.gap, 'gap'), num(values['max-cal-motion'], 'max-cal-motion'));
  if (c.state !== 'ok' || !c.estimate) die(`calibration failed (${c.state}): ${c.message ?? ''}`);
  sigmaPx = c.estimate.sigmaPx;
  calibrationInfo = { source: values.calibration, recordingId: cal.meta.id, sigmaPx, dof: c.estimate.dof, interval95: c.interval95, maxRotationDeg: c.maxRotationDeg };
} else {
  die('give --calibration <file> or --sigma <px>');
}

const rows: PairRow[] = [];
const summaries = [];
for (const file of positionals) {
  const rec = parseRecording(readFileSync(file, 'utf8'));
  if (rec.meta.role !== 'trial') die(`${file} is a ${rec.meta.role} recording; pass it with --calibration`);
  const r = analyzeRecording(rec, { ...opts, sigmaPx });
  rows.push(...r.rows);
  summaries.push({ file, ...r.summary });
  const rate = r.summary.rejectionRate === null ? '—' : `${r.summary.rejected}/${r.summary.tested} = ${r.summary.rejectionRate.toFixed(3)} [${r.summary.ci95![0].toFixed(3)}, ${r.summary.ci95![1].toFixed(3)}]`;
  console.log(`${rec.meta.condition.padEnd(20)} ${r.summary.pairs} pairs, rejected ${rate}${r.summary.blockedStage ? `  (${r.summary.blockedStage})` : ''}`);
}

mkdirSync(values.out!, { recursive: true });
writeFileSync(join(values.out!, 'pairs.csv'), rowsToCsv(rows));
writeFileSync(
  join(values.out!, 'summary.json'),
  JSON.stringify(
    {
      note: 'Pairs within one recording are not independent (same object, overlapping time). Treat per-recording rates as descriptive; the protocol must define the unit of analysis.',
      settings: { landmarkSet: setId, ...opts, indices: undefined },
      calibration: calibrationInfo,
      recordings: summaries,
    },
    null,
    2,
  ) + '\n',
);
console.log(`wrote ${join(values.out!, 'pairs.csv')} and summary.json`);
