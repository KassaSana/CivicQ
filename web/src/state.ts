/** App state: one reducer, mirrored into the URL so a view can be shared. */
import { DEFAULT_ARRIVALS, DEFAULT_PLAN, type ServiceDist, type SimConfig, makeConfig } from './sim/model';

export interface Params {
  plan: number[];
  meanService: number;
  serviceDist: ServiceDist;
  serviceCv: number;
  demandMult: number;
  rateCv: number;
  threshold: number;
  alpha: number;
  seed: number;
  days: number;
}

export const DEFAULTS: Params = {
  plan: DEFAULT_PLAN,
  meanService: 8,
  serviceDist: 'exp',
  serviceCv: 1,
  demandMult: 1,
  rateCv: 0,
  threshold: 15,
  alpha: 0.1,
  seed: 100000,
  days: 1000,
};

export const MIN_WINDOWS = 1;
export const MAX_WINDOWS = 6;

export type Action =
  | { type: 'set'; patch: Partial<Params> }
  | { type: 'step'; hour: number; delta: number }
  | { type: 'reset' };

export function reducer(s: Params, a: Action): Params {
  switch (a.type) {
    case 'set':
      return { ...s, ...a.patch };
    case 'step': {
      const plan = s.plan.slice();
      plan[a.hour] = Math.max(MIN_WINDOWS, Math.min(MAX_WINDOWS, plan[a.hour] + a.delta));
      return { ...s, plan };
    }
    case 'reset':
      return DEFAULTS;
  }
}

export function toConfig(p: Params): SimConfig {
  return makeConfig({
    plan: p.plan,
    arrivals: DEFAULT_ARRIVALS.map((r) => r * p.demandMult),
    meanService: p.meanService,
    serviceDist: p.serviceDist,
    serviceCv: p.serviceCv,
    rateCv: p.rateCv,
    threshold: p.threshold,
  });
}

export function isDefault(p: Params): boolean {
  return JSON.stringify(p) === JSON.stringify(DEFAULTS);
}

const NUM_KEYS: (keyof Params)[] = ['meanService', 'serviceCv', 'demandMult', 'rateCv', 'threshold', 'alpha', 'seed', 'days'];

export function fromUrl(search: string): Params {
  const q = new URLSearchParams(search);
  const p: Params = { ...DEFAULTS };
  const plan = q.get('plan')?.split(',').map(Number);
  if (plan && plan.length === 8 && plan.every((v) => Number.isInteger(v) && v >= MIN_WINDOWS && v <= MAX_WINDOWS)) p.plan = plan;
  const dist = q.get('dist');
  if (dist === 'exp' || dist === 'lognormal' || dist === 'det') p.serviceDist = dist;
  for (const k of NUM_KEYS) {
    const v = Number(q.get(k));
    if (q.has(k) && Number.isFinite(v)) (p[k] as number) = v;
  }
  return p;
}

export function toUrl(p: Params): string {
  const q = new URLSearchParams();
  if (p.plan.join() !== DEFAULTS.plan.join()) q.set('plan', p.plan.join(','));
  if (p.serviceDist !== DEFAULTS.serviceDist) q.set('dist', p.serviceDist);
  for (const k of NUM_KEYS) if (p[k] !== DEFAULTS[k]) q.set(k, String(p[k]));
  const s = q.toString();
  return s ? `?${s}` : location.pathname;
}
