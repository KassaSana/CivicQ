/** App state: one reducer, mirrored into the URL so a view can be shared. */
import { type Abandonment, DEFAULT_ARRIVALS, DEFAULT_PLAN, type Placement, type ServiceDist, type SimConfig, appointmentBook, makeConfig } from './sim/model';

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
  /** Share of expected daily demand booked as appointments (0 = walk-ins only). */
  apptShare: number;
  placement: Placement;
  noShow: number;
  punctualitySd: number;
  abandonment: Abandonment;
  meanPatience: number;
  patienceDist: ServiceDist;
  patienceCv: number;
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
  apptShare: 0,
  placement: 'counter',
  noShow: 0.15,
  punctualitySd: 5,
  abandonment: 'none',
  meanPatience: 30,
  patienceDist: 'exp',
  patienceCv: 1,
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
  const rates = DEFAULT_ARRIVALS.map((r) => r * p.demandMult);
  const { walk, times } = p.apptShare > 0
    ? appointmentBook(rates, p.apptShare, p.placement, p.noShow)
    : { walk: rates, times: [] };
  return makeConfig({
    plan: p.plan,
    arrivals: walk,
    appointments: times,
    noShow: p.noShow,
    punctualitySd: p.punctualitySd,
    abandonment: p.abandonment,
    meanPatience: p.meanPatience,
    patienceDist: p.patienceDist,
    patienceCv: p.patienceCv,
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

const NUM_KEYS: (keyof Params)[] = ['meanService', 'serviceCv', 'demandMult', 'rateCv', 'threshold', 'alpha', 'seed', 'days', 'apptShare', 'noShow', 'punctualitySd', 'meanPatience', 'patienceCv'];

const APPT_KEYS: (keyof Params)[] = ['noShow', 'punctualitySd'];
const PATIENCE_KEYS: (keyof Params)[] = ['meanPatience', 'patienceCv'];

const isDist = (v: string | null): v is ServiceDist => v === 'exp' || v === 'lognormal' || v === 'det';

export function fromUrl(search: string): Params {
  const q = new URLSearchParams(search);
  const p: Params = { ...DEFAULTS };
  const plan = q.get('plan')?.split(',').map(Number);
  if (plan && plan.length === 8 && plan.every((v) => Number.isInteger(v) && v >= MIN_WINDOWS && v <= MAX_WINDOWS)) p.plan = plan;
  const dist = q.get('dist');
  if (isDist(dist)) p.serviceDist = dist;
  const aband = q.get('aband');
  if (aband === 'renege' || aband === 'balk') p.abandonment = aband;
  const pdist = q.get('pdist');
  if (isDist(pdist)) p.patienceDist = pdist;
  const place = q.get('place');
  if (place === 'proportional' || place === 'flat' || place === 'counter') p.placement = place;
  for (const k of NUM_KEYS) {
    const v = Number(q.get(k));
    if (q.has(k) && Number.isFinite(v)) (p[k] as number) = v;
  }
  // Keep hand-edited links inside the ranges the booking rule can handle
  p.apptShare = Math.min(0.9, Math.max(0, p.apptShare));
  p.noShow = Math.min(0.5, Math.max(0, p.noShow));
  p.punctualitySd = Math.min(30, Math.max(0, p.punctualitySd));
  p.meanPatience = Math.min(120, Math.max(1, p.meanPatience));
  p.patienceCv = Math.min(2, Math.max(0.25, p.patienceCv));
  return p;
}

export function toUrl(p: Params): string {
  const q = new URLSearchParams();
  if (p.plan.join() !== DEFAULTS.plan.join()) q.set('plan', p.plan.join(','));
  if (p.serviceDist !== DEFAULTS.serviceDist) q.set('dist', p.serviceDist);
  // Booking settings only matter when some demand is booked
  const booked = p.apptShare > 0;
  if (booked && p.placement !== DEFAULTS.placement) q.set('place', p.placement);
  // Patience settings only matter when walk-ins may leave
  const leaving = p.abandonment !== 'none';
  if (leaving) q.set('aband', p.abandonment);
  if (leaving && p.patienceDist !== DEFAULTS.patienceDist) q.set('pdist', p.patienceDist);
  for (const k of NUM_KEYS) {
    if (!booked && APPT_KEYS.includes(k)) continue;
    if (!leaving && PATIENCE_KEYS.includes(k)) continue;
    if (p[k] !== DEFAULTS[k]) q.set(k, String(p[k]));
  }
  const s = q.toString();
  return s ? `?${s}` : location.pathname;
}
