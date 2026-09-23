/// <reference lib="webworker" />
/**
 * Runs replications off the main thread. Work is chunked and yields between
 * chunks so a newer request of the same kind cancels an older one.
 */
import { NUM_SLOTS, type SimConfig, sum } from './model';
import { simulateDay } from './simulate';
import { Accumulator, type Aggregate, runDays } from './stats';

export type WorkerRequest =
  | { kind: 'run'; id: number; cfg: SimConfig; days: number; seed: number }
  | { kind: 'compare'; id: number; cfg: SimConfig; plans: number[][]; days: number; seed: number }
  | { kind: 'frontier'; id: number; cfg: SimConfig; seed: number };

export interface FrontierPoint {
  hours: number;
  plan: number[];
  p90: number;
  p90Ci: [number, number];
  meanWait: number;
}

export type WorkerResponse =
  | { kind: WorkerRequest['kind']; id: number; progress: number }
  | { kind: 'run'; id: number; result: Aggregate }
  | { kind: 'compare'; id: number; results: Aggregate[] }
  | { kind: 'frontier'; id: number; points: FrontierPoint[] };

const latest: Record<string, number> = {};
// Yield so newer requests can arrive. MessageChannel avoids the timer
// throttling browsers apply to setTimeout in background tabs.
const channel = new MessageChannel();
const waiting: (() => void)[] = [];
channel.port1.onmessage = () => waiting.shift()?.();
const yieldNow = () => new Promise<void>((r) => { waiting.push(r); channel.port2.postMessage(0); });
const post = (m: WorkerResponse) => (self as unknown as Worker).postMessage(m);

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const req = e.data;
  latest[req.kind] = req.id;
  const stale = () => latest[req.kind] !== req.id;
  if (req.kind === 'run') void run(req, stale);
  else if (req.kind === 'compare') void compare(req, stale);
  else void frontier(req, stale);
};

async function run(req: Extract<WorkerRequest, { kind: 'run' }>, stale: () => boolean) {
  const acc = new Accumulator(req.cfg.threshold);
  const chunk = 100;
  for (let i = 0; i < req.days; i += chunk) {
    for (let j = i; j < Math.min(i + chunk, req.days); j++) acc.add(simulateDay(req.cfg, req.seed + j));
    post({ kind: 'run', id: req.id, progress: Math.min(1, (i + chunk) / req.days) });
    await yieldNow();
    if (stale()) return;
  }
  post({ kind: 'run', id: req.id, result: acc.result() });
}

async function compare(req: Extract<WorkerRequest, { kind: 'compare' }>, stale: () => boolean) {
  const results: Aggregate[] = [];
  for (let k = 0; k < req.plans.length; k++) {
    // Same seeds for every plan: common random numbers
    results.push(runDays({ ...req.cfg, plan: req.plans[k] }, req.days, req.seed));
    post({ kind: 'compare', id: req.id, progress: (k + 1) / req.plans.length });
    await yieldNow();
    if (stale()) return;
  }
  post({ kind: 'compare', id: req.id, results });
}

/**
 * Two-stage search like python/optimizer.py: screen every plan with
 * s_i in {2,3,4} (sum <= 28) on a few days, then confirm the best few per
 * staff-hour budget on 300 days and keep the lowest P90 per budget.
 */
async function frontier(req: Extract<WorkerRequest, { kind: 'frontier' }>, stale: () => boolean) {
  const plans: number[][] = [];
  const rec = (p: number[]) => {
    if (p.length === NUM_SLOTS) {
      if (sum(p) <= 28) plans.push(p);
      return;
    }
    for (const s of [2, 3, 4]) rec([...p, s]);
  };
  rec([]);

  const screenDays = 10, confirmDays = 300, topK = 4;
  const byBudget = new Map<number, { plan: number[]; p90: number }[]>();
  for (let i = 0; i < plans.length; i++) {
    const a = runDays({ ...req.cfg, plan: plans[i] }, screenDays, 1);
    const h = sum(plans[i]);
    if (!byBudget.has(h)) byBudget.set(h, []);
    byBudget.get(h)!.push({ plan: plans[i], p90: a.p90 });
    if (i % 200 === 0) {
      post({ kind: 'frontier', id: req.id, progress: (0.6 * i) / plans.length });
      await yieldNow();
      if (stale()) return;
    }
  }

  const budgets = [...byBudget.keys()].sort((a, b) => a - b);
  const points: FrontierPoint[] = [];
  for (let b = 0; b < budgets.length; b++) {
    const shortlist = byBudget.get(budgets[b])!.sort((x, y) => x.p90 - y.p90).slice(0, topK);
    let best: FrontierPoint | null = null;
    for (const c of shortlist) {
      const a = runDays({ ...req.cfg, plan: c.plan }, confirmDays, req.seed);
      if (!best || a.p90 < best.p90) best = { hours: budgets[b], plan: c.plan, p90: a.p90, p90Ci: a.p90Ci, meanWait: a.meanWait };
    }
    points.push(best!);
    post({ kind: 'frontier', id: req.id, progress: 0.6 + (0.4 * (b + 1)) / budgets.length });
    await yieldNow();
    if (stale()) return;
  }
  post({ kind: 'frontier', id: req.id, points });
}
