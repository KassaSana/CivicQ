/**
 * One simulated day: a TypeScript port of QueueSimulator (cpp/src/simulation.cpp).
 *
 * - Non-homogeneous Poisson arrivals by thinning.
 * - Service time drawn when the citizen arrives (common random numbers).
 * - Single FIFO queue; the first free *open* window serves the head of the line.
 * - Staffing changes on the hour: newly opened windows pull from the queue at
 *   once, a closing window finishes its current citizen.
 * - Doors close at `duration`; everyone inside is served (overtime).
 * - Optional appointments: each booked citizen shows with probability
 *   1 - noShow and arrives at the booked time plus Normal(0, punctualitySd),
 *   then joins the same FIFO line as walk-ins.
 * - Optional walk-in abandonment (booked citizens never leave): reneging from
 *   a hidden queue once the wait exceeds patience, or balking at a visible
 *   line when the expected wait (q + 1) S / c exceeds patience. Patience has
 *   its own random stream, drawn for every citizen in arrival order.
 */
import { exponential, gamma, makeRng, normal } from './rng';
import { NUM_SLOTS, SLOT_MINUTES, type SimConfig, slotOf } from './model';

const ARRIVAL = 0;
const DEPARTURE = 1;
const STAFFING = 2;
const APPOINTMENT = 3;
const RENEGE = 4;

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
  booked: boolean;
  /** Minutes this citizen will wait (0 when abandonment is off). */
  patience: number;
  abandoned: boolean;
  /** When they left unserved (their arrival time for a balk), else -1. */
  leave: number;
}

/** When a citizen stops waiting in line: served or gave up. */
export function lineExit(c: Citizen): number {
  return c.abandoned ? c.leave : c.start;
}

export interface DayResult {
  citizens: Citizen[];
  /** Waits of served citizens (those who left are not in the wait statistics, as in C++). */
  waits: number[];
  meanWait: number;
  /** Nearest-rank 90th percentile of the day's waits. */
  p90: number;
  overtime: number;
  rateMultiplier: number;
  /** Served citizens by arrival hour (C++ arrivals_per_slot). */
  arrivalsBySlot: number[];
  lateBySlot: number[];
  /** Walk-ins who balked or reneged, by arrival hour, and the minutes they spent before leaving. */
  abandonedBySlot: number[];
  abandonedWaitSum: number;
  utilization: number[];
  /** Booked citizens who showed up, how many of them waited over the threshold, and their total wait. */
  apptArrived: number;
  apptLate: number;
  apptWaitSum: number;
}

export function simulateDay(cfg: SimConfig, seed: number): DayResult {
  const arrivalRng = makeRng(seed, 1);
  const serviceRng = makeRng(seed, 2);
  const rateRng = makeRng(seed, 3);
  const apptRng = makeRng(seed, 4);
  const patienceRng = makeRng(seed, 5);
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

  const pSigma2 = Math.log(1 + cfg.patienceCv * cfg.patienceCv);
  const pLogMu = Math.log(cfg.meanPatience) - pSigma2 / 2;
  const drawPatience = (): number => {
    if (cfg.patienceDist === 'det') return cfg.meanPatience;
    if (cfg.patienceDist === 'lognormal') return Math.exp(pLogMu + Math.sqrt(pSigma2) * normal(patienceRng));
    return exponential(patienceRng, cfg.meanPatience);
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
  let waitingCount = 0; // excludes reneged citizens still in `queue`
  const slotBusy = new Array<number>(NUM_SLOTS).fill(0);
  let lastDeparture = 0;
  let seq = 0;
  const heap = new EventHeap();

  const push = (time: number, type: number, window = -1, citizen = -1) =>
    heap.push({ time, type, seq: seq++, window, citizen });

  for (let s = 1; s < NUM_SLOTS; s++) {
    if (s * SLOT_MINUTES < duration) push(s * SLOT_MINUTES, STAFFING);
  }
  // Both draws are always taken so the stream stays aligned across settings
  for (const booked of cfg.appointments) {
    const u = apptRng();
    const z = normal(apptRng);
    if (u < cfg.noShow) continue;
    push(Math.min(Math.max(booked + cfg.punctualitySd * z, 0), duration), APPOINTMENT);
  }
  const first = nextArrival();
  if (first <= duration) push(first, ARRIVAL);

  const freeWindow = (): number => {
    const open = plan[slotOf(now)];
    for (let i = 0; i < open && i < maxWindows; i++) if (!busy[i]) return i;
    return -1;
  };

  const serveWaiting = () => {
    while (qHead < queue.length) {
      if (citizens[queue[qHead]].abandoned) {
        qHead++; // reneged while in line
        continue;
      }
      const w = freeWindow();
      if (w < 0) break;
      const id = queue[qHead++];
      waitingCount--;
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

  const admit = (booked: boolean) => {
    const id = citizens.length;
    const c: Citizen = {
      arrival: now, service: drawService(), start: -1, departure: -1, window: -1, booked,
      patience: 0, abandoned: false, leave: -1,
    };
    // Drawn for every citizen, in arrival order, so patience stays aligned across plans
    let mayLeave = false;
    if (cfg.abandonment !== 'none') {
      c.patience = drawPatience();
      mayLeave = !booked;
    }
    citizens.push(c);
    if (mayLeave && cfg.abandonment === 'balk') {
      // They see q waiting at c open windows: (q + 1) completions at rate c / S
      const freeNow = waitingCount === 0 && freeWindow() >= 0;
      const estimate = freeNow ? 0 : ((waitingCount + 1) * cfg.meanService) / plan[slotOf(now)];
      if (estimate > c.patience) {
        c.abandoned = true;
        c.leave = now;
        return;
      }
    }
    queue.push(id);
    waitingCount++;
    serveWaiting();
    if (mayLeave && cfg.abandonment === 'renege' && c.start < 0) push(now + c.patience, RENEGE, -1, id);
  };

  while (heap.size) {
    const ev = heap.pop();
    now = ev.time;
    if (ev.type === ARRIVAL) {
      admit(false);
      const next = nextArrival();
      if (next <= duration) push(next, ARRIVAL);
    } else if (ev.type === DEPARTURE) {
      const c = citizens[ev.citizen];
      c.departure = now;
      lastDeparture = Math.max(lastDeparture, now);
      addBusy(c.start, now);
      busy[ev.window] = false;
      serveWaiting();
    } else if (ev.type === APPOINTMENT) {
      admit(true);
    } else if (ev.type === RENEGE) {
      const c = citizens[ev.citizen];
      if (c.start < 0 && !c.abandoned) {
        c.abandoned = true;
        c.leave = now;
        waitingCount--; // removed lazily from `queue`
      }
    } else {
      serveWaiting();
    }
  }

  const arrivalsBySlot = new Array<number>(NUM_SLOTS).fill(0);
  const lateBySlot = new Array<number>(NUM_SLOTS).fill(0);
  const abandonedBySlot = new Array<number>(NUM_SLOTS).fill(0);
  let abandonedWaitSum = 0;
  const waits: number[] = [];
  let total = 0;
  let apptArrived = 0, apptLate = 0, apptWaitSum = 0;
  for (const c of citizens) {
    if (c.abandoned) {
      abandonedBySlot[slotOf(c.arrival)]++;
      abandonedWaitSum += c.leave - c.arrival;
      continue;
    }
    const w = c.start - c.arrival;
    waits.push(w);
    total += w;
    const s = slotOf(c.arrival);
    arrivalsBySlot[s]++;
    if (w > cfg.threshold) lateBySlot[s]++;
    if (c.booked) {
      apptArrived++;
      apptWaitSum += w;
      if (w > cfg.threshold) apptLate++;
    }
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
    abandonedBySlot,
    abandonedWaitSum,
    utilization,
    apptArrived,
    apptLate,
    apptWaitSum,
  };
}

/** Number waiting in line at time t (for the replay and queue curves). */
export function queueLengthAt(citizens: Citizen[], t: number): number {
  let n = 0;
  for (const c of citizens) if (c.arrival <= t && lineExit(c) > t) n++;
  return n;
}
