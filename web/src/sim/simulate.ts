/**
 * One simulated day: a TypeScript port of QueueSimulator (cpp/src/simulation.cpp).
 *
 * - Non-homogeneous Poisson arrivals by thinning.
 * - Service time drawn when the citizen arrives (common random numbers).
 * - Single FIFO queue; the first free *open* window serves the head of the line.
 * - Staffing changes on the hour: newly opened windows pull from the queue at
 *   once, a closing window finishes its current citizen.
 * - Doors close at `duration`; everyone inside is served (overtime).
 */
import { exponential, gamma, makeRng, normal } from './rng';
import { NUM_SLOTS, SLOT_MINUTES, type SimConfig, slotOf } from './model';

const ARRIVAL = 0;
const DEPARTURE = 1;
const STAFFING = 2;

interface Event {
  time: number;
  type: number;
  seq: number; // FIFO tie-break for equal times, keeps runs deterministic
  window: number;
  citizen: number;
}

/** Minimal binary min-heap on (time, seq). */
class EventHeap {
  private a: Event[] = [];
  get size() {
    return this.a.length;
  }
  private less(i: number, j: number) {
    const x = this.a[i], y = this.a[j];
    return x.time < y.time || (x.time === y.time && x.seq < y.seq);
  }
  push(e: Event) {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop(): Event {
    const a = this.a;
    const top = a[0];
    const last = a.pop()!;
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && this.less(l, m)) m = l;
        if (r < a.length && this.less(r, m)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}

export interface Citizen {
  arrival: number;
  service: number;
  start: number;
  departure: number;
  window: number;
}

export interface DayResult {
  citizens: Citizen[];
  waits: number[];
  meanWait: number;
  /** Nearest-rank 90th percentile of the day's waits. */
  p90: number;
  overtime: number;
  rateMultiplier: number;
  arrivalsBySlot: number[];
  lateBySlot: number[];
  utilization: number[];
}

export function simulateDay(cfg: SimConfig, seed: number): DayResult {
  const arrivalRng = makeRng(seed, 1);
  const serviceRng = makeRng(seed, 2);
  const rateRng = makeRng(seed, 3);
  const { plan, duration } = cfg;

  // Day-level demand multiplier M ~ Gamma(1/cv^2, cv^2): mean 1, CV = rateCv
  let mult = 1;
  if (cfg.rateCv > 0) {
    const shape = 1 / (cfg.rateCv * cfg.rateCv);
    mult = gamma(rateRng, shape, 1 / shape);
  }

  const lambdaMax = (mult * Math.max(...cfg.arrivals)) / 60; // per minute
  const sigma2 = Math.log(1 + cfg.serviceCv * cfg.serviceCv);
  const logMu = Math.log(cfg.meanService) - sigma2 / 2;

  const drawService = (): number => {
    if (cfg.serviceDist === 'det') return cfg.meanService;
    if (cfg.serviceDist === 'lognormal') return Math.exp(logMu + Math.sqrt(sigma2) * normal(serviceRng));
    return exponential(serviceRng, cfg.meanService);
  };

  let now = 0;
  const nextArrival = (): number => {
    if (lambdaMax <= 0) return Infinity;
    let t = now;
    while (t < duration) {
      t += exponential(arrivalRng, 1 / lambdaMax);
      if (t >= duration) return Infinity;
      const lambdaT = (mult * cfg.arrivals[slotOf(t)]) / 60;
      if (arrivalRng() <= lambdaT / lambdaMax) return t;
    }
    return Infinity;
  };

  const maxWindows = Math.max(...plan);
  const busy = new Array<boolean>(maxWindows).fill(false);
  const citizens: Citizen[] = [];
  const queue: number[] = [];
  let qHead = 0;
  const slotBusy = new Array<number>(NUM_SLOTS).fill(0);
  let lastDeparture = 0;
  let seq = 0;
  const heap = new EventHeap();

  const push = (time: number, type: number, window = -1, citizen = -1) =>
    heap.push({ time, type, seq: seq++, window, citizen });

  for (let s = 1; s < NUM_SLOTS; s++) {
    if (s * SLOT_MINUTES < duration) push(s * SLOT_MINUTES, STAFFING);
  }
  const first = nextArrival();
  if (first <= duration) push(first, ARRIVAL);

  const serveWaiting = () => {
    const open = plan[slotOf(now)];
    while (qHead < queue.length) {
      let w = -1;
      for (let i = 0; i < open && i < maxWindows; i++) {
        if (!busy[i]) {
          w = i;
          break;
        }
      }
      if (w < 0) break;
      const id = queue[qHead++];
      busy[w] = true;
      const c = citizens[id];
      c.start = now;
      c.window = w;
      push(now + c.service, DEPARTURE, w, id);
    }
  };

  const addBusy = (start: number, end: number) => {
    end = Math.min(end, duration);
    for (let s = slotOf(start); s < NUM_SLOTS && start < end; s++) {
      const slotEnd = s === NUM_SLOTS - 1 ? end : Math.min(end, (s + 1) * SLOT_MINUTES);
      if (slotEnd > start) {
        slotBusy[s] += slotEnd - start;
        start = slotEnd;
      }
    }
  };

  while (heap.size) {
    const ev = heap.pop();
    now = ev.time;
    if (ev.type === ARRIVAL) {
      citizens.push({ arrival: now, service: drawService(), start: -1, departure: -1, window: -1 });
      queue.push(citizens.length - 1);
      serveWaiting();
      const next = nextArrival();
      if (next <= duration) push(next, ARRIVAL);
    } else if (ev.type === DEPARTURE) {
      const c = citizens[ev.citizen];
      c.departure = now;
      lastDeparture = Math.max(lastDeparture, now);
      addBusy(c.start, now);
      busy[ev.window] = false;
      serveWaiting();
    } else {
      serveWaiting();
    }
  }

  const arrivalsBySlot = new Array<number>(NUM_SLOTS).fill(0);
  const lateBySlot = new Array<number>(NUM_SLOTS).fill(0);
  const waits: number[] = new Array(citizens.length);
  let total = 0;
  for (let i = 0; i < citizens.length; i++) {
    const c = citizens[i];
    const w = c.start - c.arrival;
    waits[i] = w;
    total += w;
    const s = slotOf(c.arrival);
    arrivalsBySlot[s]++;
    if (w > cfg.threshold) lateBySlot[s]++;
  }
  const sorted = waits.slice().sort((a, b) => a - b);
  const rank = Math.ceil(0.9 * sorted.length);
  const utilization = slotBusy.map((b, s) => {
    const len = s === NUM_SLOTS - 1 ? duration - s * SLOT_MINUTES : SLOT_MINUTES;
    const avail = plan[s] * Math.max(0, len);
    return avail > 0 ? b / avail : 0;
  });

  return {
    citizens,
    waits,
    meanWait: waits.length ? total / waits.length : 0,
    p90: sorted.length ? sorted[Math.max(rank, 1) - 1] : 0,
    overtime: Math.max(0, lastDeparture - duration),
    rateMultiplier: mult,
    arrivalsBySlot,
    lateBySlot,
    utilization,
  };
}

/** Number waiting in line at time t (for the replay and queue curves). */
export function queueLengthAt(citizens: Citizen[], t: number): number {
  let n = 0;
  for (const c of citizens) if (c.arrival <= t && c.start > t) n++;
  return n;
}
