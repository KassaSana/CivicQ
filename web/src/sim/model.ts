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
  /** Booked arrival times in minutes from opening (walk-ins are `arrivals`). */
  appointments: number[];
  /** Probability a booked citizen does not come. */
  noShow: number;
  /** SD (minutes) of arrival around the booked time. */
  punctualitySd: number;
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
    appointments: [],
    noShow: 0,
    punctualitySd: 0,
    ...partial,
  };
}

export type Placement = 'proportional' | 'flat' | 'counter';

/**
 * Move `share` of expected daily demand to appointments (port of
 * research/experiments.py appointment_book). Returns walk-in hourly rates and
 * booked times, overbooked by 1/(1 - noShow) so expected shows equal the
 * demand moved. Placement of expected shows per hour:
 *   proportional  same shape as demand
 *   flat          evenly across the day
 *   counter       water-filled into quiet hours so walk-ins plus shows are as flat as possible
 */
export function appointmentBook(
  rates: readonly number[], share: number, placement: Placement, noShow: number,
): { walk: number[]; times: number[] } {
  const walk = rates.map((r) => r * (1 - share));
  const moved = share * sum(rates);
  let shows: number[];
  if (placement === 'proportional') shows = rates.map((r) => share * r);
  else if (placement === 'flat') shows = rates.map(() => moved / rates.length);
  else {
    let lo = Math.min(...walk), hi = Math.max(...walk) + moved;
    for (let i = 0; i < 100; i++) { // bisection on the water level
      const level = (lo + hi) / 2;
      if (sum(walk.map((w) => Math.max(0, level - w))) > moved) hi = level;
      else lo = level;
    }
    shows = walk.map((w) => Math.max(0, lo - w));
    const total = sum(shows);
    shows = shows.map((x) => (total > 0 ? (x * moved) / total : 0));
  }
  const wanted = shows.map((x) => x / (1 - noShow));
  const total = Math.round(sum(wanted));
  const counts = wanted.map((w) => Math.floor(w)); // largest-remainder rounding
  const order = wanted.map((_, i) => i).sort((a, b) => wanted[b] - counts[b] - (wanted[a] - counts[a]));
  for (const i of order) {
    if (sum(counts) >= total) break;
    counts[i]++;
  }
  const times: number[] = [];
  counts.forEach((c, i) => {
    for (let k = 0; k < c; k++) times.push(SLOT_MINUTES * i + (SLOT_MINUTES * (k + 0.5)) / c);
  });
  return { walk, times };
}

/** Expected arrivals per hour, walk-ins plus booked citizens who show (for the analytic rules). */
export function expectedRates(cfg: SimConfig): number[] {
  const r = cfg.arrivals.slice();
  for (const t of cfg.appointments) r[slotOf(t)] += 1 - cfg.noShow;
  return r;
}

export function slotOf(time: number): number {
  return Math.min(Math.floor(time / SLOT_MINUTES), NUM_SLOTS - 1);
}

export function sum(xs: readonly number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}
