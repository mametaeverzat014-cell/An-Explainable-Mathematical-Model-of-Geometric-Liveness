// Standard statistical distribution functions (infrastructure).
// Implementations follow the classical series / continued-fraction
// expansions of the regularised incomplete gamma function
// (Press et al., Numerical Recipes, 3rd ed., section 6.2) and are verified
// against tabulated values in src/core/__tests__/stats.test.ts.

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** Natural log of the gamma function, for x > 0 (Lanczos approximation, g = 7). */
export function logGamma(x: number): number {
  if (!(x > 0)) throw new Error('logGamma: x must be positive');
  if (x < 0.5) {
    // Reflection formula.
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const z = x - 1;
  let sum = LANCZOS[0];
  for (let i = 1; i < 9; i++) sum += LANCZOS[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Regularised lower incomplete gamma function P(a, x). */
export function regularizedGammaP(a: number, x: number): number {
  if (!(a > 0)) throw new Error('regularizedGammaP: a must be positive');
  if (x <= 0) return 0;
  if (!Number.isFinite(x)) return 1;
  if (x < a + 1) {
    // Series representation.
    let ap = a;
    let sum = 1 / a;
    let del = sum;
    for (let n = 0; n < 10000; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-16) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
  }
  return 1 - regularizedGammaQ(a, x);
}

/** Regularised upper incomplete gamma function Q(a, x) = 1 - P(a, x). */
export function regularizedGammaQ(a: number, x: number): number {
  if (!(a > 0)) throw new Error('regularizedGammaQ: a must be positive');
  if (x <= 0) return 1;
  if (!Number.isFinite(x)) return 0;
  if (x < a + 1) return 1 - regularizedGammaP(a, x);
  // Continued fraction (modified Lentz).
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 10000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

/** CDF of the chi-square distribution with k degrees of freedom. */
export function chiSquareCdf(x: number, k: number): number {
  if (!(k > 0)) throw new Error('chiSquareCdf: k must be positive');
  return regularizedGammaP(k / 2, x / 2);
}

/** Survival function 1 - CDF (upper-tail probability), accurate in the far tail. */
export function chiSquareSf(x: number, k: number): number {
  if (!(k > 0)) throw new Error('chiSquareSf: k must be positive');
  return regularizedGammaQ(k / 2, x / 2);
}

/** Quantile: the x with chiSquareCdf(x, k) = p, for 0 < p < 1. */
export function chiSquareQuantile(p: number, k: number): number {
  if (!(p > 0 && p < 1)) throw new Error('chiSquareQuantile: p must be in (0, 1)');
  let lo = 0;
  let hi = Math.max(1, k);
  while (chiSquareCdf(hi, k) < p) hi *= 2;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (chiSquareCdf(mid, k) < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12 * Math.max(1, hi)) break;
  }
  return 0.5 * (lo + hi);
}

/** Probability density of the chi-square distribution (used for plots). */
export function chiSquarePdf(x: number, k: number): number {
  if (x < 0) return 0;
  if (x === 0) return k === 2 ? 0.5 : k < 2 ? Number.POSITIVE_INFINITY : 0;
  const h = k / 2;
  return Math.exp((h - 1) * Math.log(x) - x / 2 - h * Math.log(2) - logGamma(h));
}

/** Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9). */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new Error('normalQuantile: p must be in (0, 1)');
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pLow) return -normalQuantile(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/**
 * Wilson score interval for a binomial proportion (k successes in n trials).
 * Used for Monte-Carlo rejection rates. Returns [lower, upper].
 */
export function wilsonInterval(k: number, n: number, confidence = 0.95): [number, number] {
  if (n <= 0) return [0, 1];
  const z = normalQuantile(1 - (1 - confidence) / 2);
  const phat = k / n;
  const denom = 1 + (z * z) / n;
  const centre = (phat + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((phat * (1 - phat)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}

/** Sample variance with denominator n - 1. */
export function sampleVariance(values: readonly number[]): number {
  if (values.length < 2) return Number.NaN;
  const m = mean(values);
  let s = 0;
  for (const v of values) s += (v - m) * (v - m);
  return s / (values.length - 1);
}

export function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

/** Linear-interpolation quantile (type 7, as in R / NumPy default). */
export function quantile(values: readonly number[], q: number): number {
  if (values.length === 0) return Number.NaN;
  const s = [...values].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
