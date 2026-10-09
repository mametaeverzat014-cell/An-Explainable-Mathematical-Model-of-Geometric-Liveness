// Coverage check of the interval reported per condition in research mode
// (stats.ts, clusteredRateInterval), against the naive Wilson interval that
// treats all frame pairs as independent.
//
// Model: k recordings; recording i has m_i tested pairs; its own rejection
// probability is drawn from a Beta distribution with mean p and
// intra-recording correlation icc (icc = 0: all pairs independent). Each
// pair is rejected with that probability.
//
// Run: npm run simulate-interval   (about a minute)
import { clusteredRateInterval, wilsonInterval } from '../src/core/stats';
import { Rng } from '../src/core/rng';

function gammaSample(rng: Rng, shape: number): number {
  // Marsaglia & Tsang (2000); boost for shape < 1.
  if (shape < 1) return gammaSample(rng, shape + 1) * Math.pow(rng.uniform(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = rng.normal();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.uniform();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function betaSample(rng: Rng, a: number, b: number): number {
  const x = gammaSample(rng, a);
  return x / (x + gammaSample(rng, b));
}

const REPS = 2000;
const rng = new Rng(11);
console.log('Coverage of 95 % intervals; "low"/"high": interval entirely below/above the true rate (nominal 0.025 each).');
console.log('p     icc  k   m    | clustered: cover  low    high  | naive Wilson: cover  low    high');
for (const p of [0.05, 0.1]) {
  for (const icc of [0, 0.1, 0.3]) {
    for (const k of [5, 10, 20]) {
      for (const mSpec of ['3', '6', '2-8']) {
        const c = [0, 0, 0];
        const w = [0, 0, 0];
        for (let r = 0; r < REPS; r++) {
          const x: number[] = [];
          const m: number[] = [];
          for (let i = 0; i < k; i++) {
            const mi = mSpec === '2-8' ? 2 + Math.floor(rng.uniform() * 7) : Number(mSpec);
            const s = (1 - icc) / icc;
            const pi = icc > 0 ? betaSample(rng, p * s, (1 - p) * s) : p;
            let xi = 0;
            for (let j = 0; j < mi; j++) if (rng.uniform() < pi) xi++;
            x.push(xi);
            m.push(mi);
          }
          const tally = (ci: [number, number], t: number[]) => {
            if (ci[0] <= p && p <= ci[1]) t[0]++;
            else if (ci[1] < p) t[1]++;
            else t[2]++;
          };
          tally(clusteredRateInterval(x, m)!.ci, c);
          tally(wilsonInterval(x.reduce((a, b) => a + b, 0), m.reduce((a, b) => a + b, 0)), w);
        }
        const f = (t: number[]) => t.map((v) => (v / REPS).toFixed(3)).join('  ');
        console.log(`${p.toFixed(2)}  ${icc.toFixed(1)}  ${String(k).padEnd(3)} ${mSpec.padEnd(4)} |            ${f(c)}  |               ${f(w)}`);
      }
    }
  }
}
