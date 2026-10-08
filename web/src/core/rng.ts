/**
 * Seeded pseudo-random number generator (mulberry32). Every synthetic
 * experiment takes an explicit seed so that results are reproducible.
 */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform in [0, 1). */
  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal variate (Box-Muller). */
  normal(): number {
    let u = 0;
    while (u === 0) u = this.uniform();
    const v = this.uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Uniform in [lo, hi). */
  between(lo: number, hi: number): number {
    return lo + (hi - lo) * this.uniform();
  }
}
