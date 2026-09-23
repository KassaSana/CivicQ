/**
 * Queueing formulas: Erlang-C, SIPP and offered-load staffing.
 * Ports of python/optimizer.py (erlang_c, mmc_*) and research/staffing_methods.py.
 */
import { DAY_MINUTES, NUM_SLOTS, SLOT_MINUTES, type ServiceDist, slotOf } from './model';

/** P(wait > 0) in M/M/c via the stable Erlang-B recursion; 1 when unstable. */
export function erlangC(c: number, load: number): number {
  if (load >= c) return 1;
  let b = 1;
  for (let k = 1; k <= c; k++) b = (load * b) / (k + load * b);
  const rho = load / c;
  return b / (1 - rho * (1 - b));
}

/** Steady-state mean wait in queue (minutes); Infinity when unstable. */
export function mmcMeanWait(c: number, perHour: number, meanService: number): number {
  const lam = perHour / 60, mu = 1 / meanService;
  if (lam >= c * mu) return Infinity;
  return erlangC(c, lam / mu) / (c * mu - lam);
}

/** Steady-state P(W > threshold minutes) for an offered load in Erlangs. */
export function probWaitExceeds(c: number, load: number, meanService: number, threshold: number): number {
  if (load <= 0) return 0;
  if (load >= c) return 1;
  return erlangC(c, load) * Math.exp((-(c - load) * threshold) / meanService);
}

export function serversForLoad(load: number, meanService: number, threshold: number, alpha: number, min = 1): number {
  let c = Math.max(min, 1);
  while (probWaitExceeds(c, load, meanService, threshold) > alpha) c++;
  return c;
}

/** Complementary error function (Numerical Recipes erfcc, |error| < 1.2e-7). */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

/** P(S > u) for the service-time distribution. */
function survival(u: number, meanService: number, dist: ServiceDist, cv: number): number {
  if (dist === 'det') return u < meanService ? 1 : 0;
  if (dist === 'lognormal') {
    const s2 = Math.log(1 + cv * cv);
    const mu = Math.log(meanService) - s2 / 2;
    return 0.5 * erfc((Math.log(Math.max(u, 1e-12)) - mu) / Math.sqrt(2 * s2));
  }
  return Math.exp(-u / meanService);
}

/**
 * Infinite-server offered load m(t) = ∫ λ(t−u) P(S > u) du, with the office
 * opening empty (λ = 0 before t = 0). Returns samples every dt minutes.
 */
export function offeredLoad(
  ratesPerHour: readonly number[], meanService: number, dist: ServiceDist = 'exp', cv = 1, dt = 0.5,
): { t: number[]; m: number[] } {
  const n = Math.round(DAY_MINUTES / dt);
  const lam = new Array<number>(n), kernel = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    lam[i] = ratesPerHour[slotOf(i * dt + dt / 2)] / 60;
    kernel[i] = survival(i * dt + dt / 2, meanService, dist, cv);
  }
  const t = [0], m = [0];
  for (let k = 1; k <= n; k++) {
    let s = 0;
    for (let i = 0; i < k; i++) s += lam[k - 1 - i] * kernel[i];
    t.push(k * dt);
    m.push(s * dt);
  }
  return { t, m };
}

/** Hourly averages of λ(t − lag), with λ = 0 before opening. */
export function laggedRates(ratesPerHour: readonly number[], lag: number, dt = 0.25): number[] {
  const out: number[] = [];
  for (let i = 0; i < NUM_SLOTS; i++) {
    let s = 0, n = 0;
    for (let x = i * SLOT_MINUTES + dt / 2; x < (i + 1) * SLOT_MINUTES; x += dt) {
      const idx = Math.floor((x - lag) / SLOT_MINUTES);
      s += idx < 0 ? 0 : ratesPerHour[Math.min(idx, NUM_SLOTS - 1)];
      n++;
    }
    out.push(s / n);
  }
  return out;
}

export type PlanName = 'SIPP' | 'Lag-SIPP' | 'OL-avg' | 'OL-max';

/** All analytic staffing rules for one scenario (research/staffing_methods.analytic_plans). */
export function analyticPlans(
  rates: readonly number[], meanService: number, threshold: number, alpha: number, minWindows = 1,
): Record<PlanName, number[]> {
  const staff = (loads: number[]) => loads.map((a) => serversForLoad(a, meanService, threshold, alpha, minWindows));
  const sipp = rates.map((r) => (r / 60) * meanService);
  const lag = laggedRates(rates, meanService).map((r) => (r / 60) * meanService);
  const { t, m } = offeredLoad(rates, meanService);
  const avg: number[] = [], max: number[] = [];
  for (let s = 0; s < NUM_SLOTS; s++) {
    let sum = 0, n = 0, mx = 0;
    for (let i = 0; i < t.length; i++) {
      if (slotOf(t[i]) !== s) continue;
      sum += m[i];
      n++;
      mx = Math.max(mx, m[i]);
    }
    avg.push(sum / n);
    max.push(mx);
  }
  return { SIPP: staff(sipp), 'Lag-SIPP': staff(lag), 'OL-avg': staff(avg), 'OL-max': staff(max) };
}

/** Per-hour SIPP view of a plan: load, utilization, P(W > T), mean wait and queue length. */
export function sippHourly(plan: readonly number[], rates: readonly number[], meanService: number, threshold: number) {
  return plan.map((c, i) => {
    const load = (rates[i] / 60) * meanService;
    const unstable = load >= c;
    const wq = unstable ? Infinity : mmcMeanWait(c, rates[i], meanService);
    return {
      load,
      rho: load / c,
      unstable,
      late: probWaitExceeds(c, load, meanService, threshold),
      meanWait: wq,
      queueLength: unstable ? Infinity : (rates[i] / 60) * wq,
    };
  });
}
