/** Shared model types and defaults (match cpp/include/simulation.hpp and python/optimizer.py). */

export const NUM_SLOTS = 8;
export const SLOT_MINUTES = 60;
export const DAY_MINUTES = 480;

export const DEFAULT_ARRIVALS = [12, 15, 10, 8, 8, 12, 14, 10];
/** Simulation-optimal plan under the per-hour target (research/results/e1b_sgs_optimality.csv). */
export const DEFAULT_PLAN = [2, 3, 3, 2, 2, 3, 3, 3];
/** Cost-optimized plan under the pooled P90 target (README scenario analysis). */
export const OPTIMIZED_PLAN = [3, 3, 3, 2, 2, 3, 3, 2];

export const HOUR_LABELS = ['8a', '9a', '10a', '11a', '12p', '1p', '2p', '3p'];
export const HOUR_RANGES = ['8–9', '9–10', '10–11', '11–12', '12–1', '1–2', '2–3', '3–4'];

export type ServiceDist = 'exp' | 'lognormal' | 'det';

export interface SimConfig {
  plan: number[];
  /** Arrivals per hour for each slot (already scaled by any demand multiplier). */
  arrivals: number[];
  meanService: number;
  serviceDist: ServiceDist;
  /** Service-time CV, used by the lognormal distribution only. */
  serviceCv: number;
  /** CV of the random daily demand multiplier M ~ Gamma(mean 1). 0 disables it. */
  rateCv: number;
  /** A citizen is "late" if they wait longer than this many minutes. */
  threshold: number;
  duration: number;
}

export function makeConfig(partial: Partial<SimConfig> = {}): SimConfig {
  return {
    plan: DEFAULT_PLAN,
    arrivals: DEFAULT_ARRIVALS,
    meanService: 8,
    serviceDist: 'exp',
    serviceCv: 1,
    rateCv: 0,
    threshold: 15,
    duration: DAY_MINUTES,
    ...partial,
  };
}

export function slotOf(time: number): number {
  return Math.min(Math.floor(time / SLOT_MINUTES), NUM_SLOTS - 1);
}

export function sum(xs: readonly number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}
