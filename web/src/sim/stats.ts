/** Statistics helpers (mirror python/optimizer.py and research/staffing_methods.py). */
import { DAY_MINUTES, NUM_SLOTS, type SimConfig } from './model';
import { type DayResult, simulateDay } from './simulate';

const T_TABLE_95: Record<number, number> = {
  1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
  8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.16,
  14: 2.145, 15: 2.131, 16: 2.12, 17: 2.11, 18: 2.101, 19: 2.093,
  20: 2.086, 21: 2.08, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.06,
  26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
};

export function tCritical95(df: number): number {
  return T_TABLE_95[df] ?? 1.96;
}

export type Interval = [number, number];

/** Mean with a two-sided 95% t interval. */
export function meanCi(xs: readonly number[]): { mean: number; ci: Interval } {
  const n = xs.length;
  if (n === 0) return { mean: 0, ci: [0, 0] };
  let s = 0;
  for (const x of xs) s += x;
  const mean = s / n;
  if (n < 2) return { mean, ci: [mean, mean] };
  let v = 0;
  for (const x of xs) v += (x - mean) ** 2;
  const margin = tCritical95(n - 1) * Math.sqrt(v / (n - 1) / n);
  return { mean, ci: [mean - margin, mean + margin] };
}

/** Ratio estimator sum(L)/sum(A) with a delta-method CI over independent days. */
export function ratioCi(late: readonly number[], arrivals: readonly number[], z = 1.96): { p: number; ci: Interval } {
  const n = late.length;
  let sumL = 0, sumA = 0;
  for (let i = 0; i < n; i++) {
    sumL += late[i];
    sumA += arrivals[i];
  }
  if (sumA === 0) return { p: 0, ci: [0, 0] };
  const p = sumL / sumA;
  if (n < 2) return { p, ci: [p, p] };
  let r2 = 0;
  for (let i = 0; i < n; i++) r2 += (late[i] - p * arrivals[i]) ** 2;
  const se = Math.sqrt(r2 / (n * (n - 1))) / (sumA / n);
  return { p, ci: [Math.max(0, p - z * se), Math.min(1, p + z * se)] };
}

export const HIST_EDGES = [5, 10, 15, 20, 25, 30]; // bins: 0, (0,5], ..., (25,30], >30
export const HIST_LABELS = ['0', '0–5', '5–10', '10–15', '15–20', '20–25', '25–30', '30+'];
export const QUEUE_BIN = 10; // minutes per queue-curve bin

export interface Aggregate {
  days: number;
  meanWait: number;
  meanWaitCi: Interval;
  /** Mean over days of each day's P90 wait. */
  p90: number;
  p90Ci: Interval;
  /** Share of days whose own P90 is within the threshold. */
  fracDaysOk: number;
  lateByHour: number[];
  lateCi: Interval[];
  overallLate: number;
  utilization: number[];
  overtime: number;
  arrivalsPerDay: number;
  /** Share of all citizens in each histogram bin. */
  histogram: number[];
  /** Time-average queue length per QUEUE_BIN minutes over the open day. */
  queueCurve: number[];
  /** Per-day values, kept for paired (common random numbers) comparisons. */
  dailyMeanWait: number[];
  dailyP90: number[];
}

/** Streams days in and summarizes them. */
export class Accumulator {
  private meanW: number[] = [];
  private p90s: number[] = [];
  private late: number[][] = Array.from({ length: NUM_SLOTS }, () => []);
  private arr: number[][] = Array.from({ length: NUM_SLOTS }, () => []);
  private util = new Array<number>(NUM_SLOTS).fill(0);
  private overtime = 0;
  private arrivals = 0;
  private hist = new Array<number>(HIST_LABELS.length).fill(0);
  private queue = new Array<number>(DAY_MINUTES / QUEUE_BIN).fill(0);
  private okDays = 0;
  constructor(private threshold: number) {}

  add(d: DayResult) {
    this.meanW.push(d.meanWait);
    this.p90s.push(d.p90);
    if (d.p90 <= this.threshold) this.okDays++;
    for (let s = 0; s < NUM_SLOTS; s++) {
      this.late[s].push(d.lateBySlot[s]);
      this.arr[s].push(d.arrivalsBySlot[s]);
      this.util[s] += d.utilization[s];
    }
    this.overtime += d.overtime;
    this.arrivals += d.citizens.length;
    for (const w of d.waits) {
      let b = 0;
      if (w > 1e-9) {
        b = 1;
        while (b <= HIST_EDGES.length && w > HIST_EDGES[b - 1]) b++;
      }
      this.hist[b]++;
    }
    // Queue curve: time each waiting interval overlaps each bin
    for (const c of d.citizens) {
      const a = c.arrival, e = Math.min(c.start, DAY_MINUTES);
      for (let k = Math.floor(a / QUEUE_BIN); k * QUEUE_BIN < e && k < this.queue.length; k++) {
        const lo = Math.max(a, k * QUEUE_BIN), hi = Math.min(e, (k + 1) * QUEUE_BIN);
        if (hi > lo) this.queue[k] += hi - lo;
      }
    }
  }

  result(): Aggregate {
    const n = this.meanW.length || 1;
    const mw = meanCi(this.meanW);
    const p9 = meanCi(this.p90s);
    const lateByHour: number[] = [];
    const lateCi: Interval[] = [];
    let totL = 0, totA = 0;
    for (let s = 0; s < NUM_SLOTS; s++) {
      const r = ratioCi(this.late[s], this.arr[s]);
      lateByHour.push(r.p);
      lateCi.push(r.ci);
      for (const x of this.late[s]) totL += x;
      for (const x of this.arr[s]) totA += x;
    }
    const histTotal = this.hist.reduce((a, b) => a + b, 0) || 1;
    return {
      days: this.meanW.length,
      meanWait: mw.mean,
      meanWaitCi: mw.ci,
      p90: p9.mean,
      p90Ci: p9.ci,
      fracDaysOk: this.okDays / n,
      lateByHour,
      lateCi,
      overallLate: totA ? totL / totA : 0,
      utilization: this.util.map((u) => u / n),
      overtime: this.overtime / n,
      arrivalsPerDay: this.arrivals / n,
      histogram: this.hist.map((h) => h / histTotal),
      queueCurve: this.queue.map((q) => q / (n * QUEUE_BIN)),
      dailyMeanWait: this.meanW,
      dailyP90: this.p90s,
    };
  }
}

/** Run `days` replications with seeds baseSeed, baseSeed + 1, ... (like run_replications in C++). */
export function runDays(cfg: SimConfig, days: number, baseSeed: number): Aggregate {
  const acc = new Accumulator(cfg.threshold);
  for (let i = 0; i < days; i++) acc.add(simulateDay(cfg, baseSeed + i));
  return acc.result();
}
