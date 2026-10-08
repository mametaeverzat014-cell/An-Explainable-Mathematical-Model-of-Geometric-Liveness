import { h } from './dom';

// Small canvas chart component: line/scatter series with optional interval
// bars, histograms with an overlaid reference curve, horizontal reference
// lines, and a hover tooltip. Colours come from CSS custom properties so
// light and dark themes are handled in styles.css.

export type Marker = 'circle' | 'square' | 'triangle';

export interface SeriesPoint {
  x: number;
  y: number;
  lo?: number;
  hi?: number;
}

export interface Series {
  label: string;
  /** CSS custom property holding the colour, e.g. '--series-1'. */
  colorVar: string;
  marker: Marker;
  points: SeriesPoint[];
  line?: boolean;
  showMarkers?: boolean;
}

export interface Histogram {
  label: string;
  colorVar: string;
  /** Bin edges (length = counts.length + 1). */
  edges: number[];
  /** Bar heights (already normalised as wanted, e.g. density). */
  heights: number[];
}

export interface RefLine {
  y?: number;
  x?: number;
  label: string;
}

export interface Curve {
  label: string;
  colorVar: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

export interface PlotSpec {
  xLabel: string;
  yLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  series?: Series[];
  histogram?: Histogram;
  curves?: Curve[];
  refLines?: RefLine[];
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  emptyMessage?: string;
}

const PAD = { left: 52, right: 16, top: 14, bottom: 40 };

function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const start = Math.ceil(lo / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}

function defaultFmt(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-2) return v.toExponential(0);
  return Number(v.toPrecision(3)).toString();
}

export class Plot {
  readonly element: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly tooltip: HTMLElement;
  private readonly legend: HTMLElement;
  private spec: PlotSpec | null = null;
  private xScale = (v: number) => v;
  private yScale = (v: number) => v;
  private xInv = (p: number) => p;

  constructor(ariaLabel: string, height = 220) {
    this.canvas = h('canvas', { role: 'img', 'aria-label': ariaLabel, style: `height:${height}px` }) as HTMLCanvasElement;
    this.tooltip = h('div', { class: 'plot-tooltip', hidden: true });
    this.legend = h('div', { class: 'plot-legend' });
    this.element = h('figure', { class: 'plot' }, this.legend, h('div', { class: 'plot-canvas-wrap' }, this.canvas, this.tooltip));
    this.canvas.addEventListener('pointermove', (e) => this.onHover(e));
    this.canvas.addEventListener('pointerleave', () => {
      this.tooltip.hidden = true;
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(this.canvas);
    window.addEventListener('themechange', () => this.draw());
  }

  render(spec: PlotSpec): void {
    this.spec = spec;
    this.renderLegend();
    this.draw();
  }

  private color(varName: string): string {
    return getComputedStyle(this.element).getPropertyValue(varName).trim() || '#888';
  }

  private renderLegend(): void {
    const spec = this.spec;
    this.legend.replaceChildren();
    if (!spec) return;
    const items: { label: string; colorVar: string; marker?: Marker; dashed?: boolean }[] = [];
    const n = (spec.series?.length ?? 0) + (spec.histogram ? 1 : 0) + (spec.curves?.length ?? 0);
    if (n < 2) return; // a single series is named by the card title
    spec.series?.forEach((s) => items.push({ label: s.label, colorVar: s.colorVar, marker: s.marker }));
    if (spec.histogram) items.push({ label: spec.histogram.label, colorVar: spec.histogram.colorVar, marker: 'square' });
    spec.curves?.forEach((c) => items.push({ label: c.label, colorVar: c.colorVar, dashed: c.dashed }));
    for (const it of items) {
      const swatch = h('span', { class: `swatch swatch-${it.marker ?? (it.dashed ? 'dash' : 'line')}`, style: `--swatch:var(${it.colorVar})` });
      this.legend.append(h('span', { class: 'legend-item' }, swatch, it.label));
    }
  }

  private domain(): { x: [number, number]; y: [number, number] } {
    const spec = this.spec!;
    const xs: number[] = [];
    const ys: number[] = [];
    spec.series?.forEach((s) =>
      s.points.forEach((p) => {
        xs.push(p.x);
        ys.push(p.y, p.lo ?? p.y, p.hi ?? p.y);
      }),
    );
    if (spec.histogram) {
      xs.push(spec.histogram.edges[0], spec.histogram.edges[spec.histogram.edges.length - 1]);
      ys.push(0, ...spec.histogram.heights);
    }
    spec.curves?.forEach((c) => c.points.forEach((p) => (xs.push(p.x), ys.push(p.y))));
    spec.refLines?.forEach((r) => (r.y !== undefined ? ys.push(r.y) : r.x !== undefined ? xs.push(r.x) : null));
    const finite = (a: number[]) => a.filter(Number.isFinite);
    const fx = finite(xs);
    const fy = finite(ys);
    const x: [number, number] = spec.xDomain ?? (fx.length ? [Math.min(...fx), Math.max(...fx)] : [0, 1]);
    let y: [number, number] = spec.yDomain ?? (fy.length ? [Math.min(0, ...fy), Math.max(...fy)] : [0, 1]);
    if (x[0] === x[1]) x[1] = x[0] + 1;
    if (y[0] === y[1]) y = [y[0], y[0] + 1];
    if (!spec.yDomain) y[1] += 0.05 * (y[1] - y[0]);
    return { x, y };
  }

  private draw(): void {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const hgt = canvas.clientHeight;
    if (w === 0 || hgt === 0) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(hgt * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    const spec = this.spec;
    const ink2 = this.color('--text-secondary');
    const muted = this.color('--text-muted');
    const grid = this.color('--grid');
    const axis = this.color('--axis');
    ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
    const hasData = spec && ((spec.series?.some((s) => s.points.length) ?? false) || spec.histogram || spec.curves?.length);
    if (!spec || !hasData) {
      ctx.fillStyle = muted;
      ctx.textAlign = 'center';
      ctx.fillText(spec?.emptyMessage ?? 'No data yet', w / 2, hgt / 2);
      return;
    }
    const { x: xd, y: yd } = this.domain();
    const pw = w - PAD.left - PAD.right;
    const ph = hgt - PAD.top - PAD.bottom;
    this.xScale = (v) => PAD.left + ((v - xd[0]) / (xd[1] - xd[0])) * pw;
    this.yScale = (v) => PAD.top + ph - ((v - yd[0]) / (yd[1] - yd[0])) * ph;
    this.xInv = (p) => xd[0] + ((p - PAD.left) / pw) * (xd[1] - xd[0]);
    const fx = spec.formatX ?? defaultFmt;
    const fy = spec.formatY ?? defaultFmt;

    // Grid and ticks (recessive).
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of niceTicks(yd[0], yd[1])) {
      const y = Math.round(this.yScale(t)) + 0.5;
      ctx.strokeStyle = grid;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(w - PAD.right, y);
      ctx.stroke();
      ctx.fillStyle = muted;
      ctx.fillText(fy(t), PAD.left - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of niceTicks(xd[0], xd[1], Math.max(3, Math.floor(pw / 80)))) {
      ctx.fillStyle = muted;
      ctx.fillText(fx(t), this.xScale(t), PAD.top + ph + 6);
    }
    ctx.strokeStyle = axis;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + ph + 0.5);
    ctx.lineTo(w - PAD.right, PAD.top + ph + 0.5);
    ctx.stroke();
    ctx.fillStyle = ink2;
    ctx.fillText(spec.xLabel, PAD.left + pw / 2, hgt - 16);
    ctx.save();
    ctx.translate(12, PAD.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle';
    ctx.fillText(spec.yLabel, 0, 0);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    // Clip slightly outside the plot area so markers at the domain edges stay whole.
    ctx.rect(PAD.left - 8, PAD.top - 8, pw + 16, ph + 16);
    ctx.clip();

    // Histogram bars: 2px surface gap between adjacent bars, rounded top.
    if (spec.histogram) {
      const hist = spec.histogram;
      ctx.fillStyle = this.color(hist.colorVar);
      hist.heights.forEach((v, i) => {
        const x0 = this.xScale(hist.edges[i]) + 1;
        const x1 = this.xScale(hist.edges[i + 1]) - 1;
        const y = this.yScale(v);
        const base = this.yScale(Math.max(0, yd[0]));
        if (x1 - x0 < 0.5 || base - y < 0.5) return;
        roundedTopRect(ctx, x0, y, x1 - x0, base - y, Math.min(4, (x1 - x0) / 2));
      });
    }

    // Reference lines (dashed, muted).
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = ink2;
    ctx.lineWidth = 1;
    for (const r of spec.refLines ?? []) {
      ctx.beginPath();
      if (r.y !== undefined) {
        const y = this.yScale(r.y);
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(w - PAD.right, y);
      } else if (r.x !== undefined) {
        const x = this.xScale(r.x);
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + ph);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    for (const c of spec.curves ?? []) {
      ctx.strokeStyle = this.color(c.colorVar);
      ctx.lineWidth = 2;
      ctx.setLineDash(c.dashed ? [6, 4] : []);
      ctx.beginPath();
      c.points.forEach((p, i) => (i ? ctx.lineTo(this.xScale(p.x), this.yScale(p.y)) : ctx.moveTo(this.xScale(p.x), this.yScale(p.y))));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const surface = this.color('--surface-1');
    for (const s of spec.series ?? []) {
      const col = this.color(s.colorVar);
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      if (s.line !== false && s.points.length > 1) {
        ctx.beginPath();
        s.points.forEach((p, i) => (i ? ctx.lineTo(this.xScale(p.x), this.yScale(p.y)) : ctx.moveTo(this.xScale(p.x), this.yScale(p.y))));
        ctx.stroke();
      }
      for (const p of s.points) {
        if (p.lo !== undefined && p.hi !== undefined) {
          const x = this.xScale(p.x);
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(x, this.yScale(p.lo));
          ctx.lineTo(x, this.yScale(p.hi));
          ctx.moveTo(x - 4, this.yScale(p.lo));
          ctx.lineTo(x + 4, this.yScale(p.lo));
          ctx.moveTo(x - 4, this.yScale(p.hi));
          ctx.lineTo(x + 4, this.yScale(p.hi));
          ctx.stroke();
        }
      }
      if (s.showMarkers !== false) {
        for (const p of s.points) drawMarker(ctx, s.marker, this.xScale(p.x), this.yScale(p.y), col, surface);
      }
    }
    ctx.restore();

    // Reference line labels, drawn outside the clip.
    ctx.fillStyle = ink2;
    // Labels sit at the left end of the line, where data rarely is.
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    for (const r of spec.refLines ?? []) {
      if (r.y !== undefined) ctx.fillText(r.label, PAD.left + 14, this.yScale(r.y) - 3);
      else if (r.x !== undefined) {
        ctx.fillText(r.label, this.xScale(r.x) + 4, PAD.top + 12);
      }
    }
  }

  private onHover(e: PointerEvent): void {
    const spec = this.spec;
    if (!spec) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const x = this.xInv(px);
    const fx = spec.formatX ?? defaultFmt;
    const fy = (v: number) => (spec.formatY ?? defaultFmt)(v);
    const lines: string[] = [];
    let anchorX = px;
    if (spec.histogram) {
      const hist = spec.histogram;
      const i = hist.edges.findIndex((edge, k) => k < hist.heights.length && x >= edge && x < hist.edges[k + 1]);
      if (i >= 0) lines.push(`${fx(hist.edges[i])} – ${fx(hist.edges[i + 1])}: ${fy(hist.heights[i])}`);
    }
    for (const s of spec.series ?? []) {
      if (!s.points.length) continue;
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
      if (Math.abs(this.xScale(best.x) - px) > 40) continue;
      anchorX = this.xScale(best.x);
      const ci = best.lo !== undefined && best.hi !== undefined ? ` [${fy(best.lo)}, ${fy(best.hi)}]` : '';
      lines.push(`${s.label} @ ${fx(best.x)}: ${fy(best.y)}${ci}`);
    }
    if (!lines.length) {
      this.tooltip.hidden = true;
      return;
    }
    this.tooltip.replaceChildren(...lines.map((l) => h('div', {}, l)));
    this.tooltip.hidden = false;
    const tw = this.tooltip.offsetWidth;
    const left = Math.min(Math.max(0, anchorX + 12), rect.width - tw - 4);
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${PAD.top}px`;
    this.draw();
    const ctx = this.canvas.getContext('2d');
    if (ctx) {
      ctx.strokeStyle = this.color('--axis');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(anchorX + 0.5, PAD.top);
      ctx.lineTo(anchorX + 0.5, rect.height - PAD.bottom);
      ctx.stroke();
    }
  }
}

function roundedTopRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, hgt: number, r: number): void {
  const rr = Math.min(r, hgt);
  ctx.beginPath();
  ctx.moveTo(x, y + hgt);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + hgt);
  ctx.closePath();
  ctx.fill();
}

function drawMarker(ctx: CanvasRenderingContext2D, marker: Marker, x: number, y: number, color: string, ring: string): void {
  const r = 4.5;
  ctx.beginPath();
  if (marker === 'circle') ctx.arc(x, y, r, 0, 2 * Math.PI);
  else if (marker === 'square') ctx.rect(x - r, y - r, 2 * r, 2 * r);
  else {
    ctx.moveTo(x, y - r - 1);
    ctx.lineTo(x + r + 1, y + r);
    ctx.lineTo(x - r - 1, y + r);
    ctx.closePath();
  }
  ctx.fillStyle = color;
  ctx.strokeStyle = ring;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fill();
}
