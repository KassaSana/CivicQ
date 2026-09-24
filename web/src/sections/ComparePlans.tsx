import { type Dispatch, useMemo } from 'react';
import { SectionHead, pct } from '../components/ui';
import { analyticPlans } from '../sim/analytic';
import { HOUR_RANGES, OPTIMIZED_PLAN, type SimConfig, expectedRates, sum } from '../sim/model';
import { meanCi } from '../sim/stats';
import type { Action, Params } from '../state';
import { useCompare } from '../useSim';

const COMPARE_DAYS = 500;

function MiniBars({ plan }: { plan: number[] }) {
  const mx = Math.max(4, ...plan);
  return (
    <svg viewBox="0 0 160 52" width="100%" aria-hidden="true">
      {plan.map((v, i) => (
        <rect key={i} className="bar" x={2 + i * 20} width={16} y={52 - (v / mx) * 50} height={(v / mx) * 50} rx={2} fill="var(--accent)" opacity={0.8} />
      ))}
    </svg>
  );
}

export function ComparePlans({ params, dispatch, cfg }: { params: Params; dispatch: Dispatch<Action>; cfg: SimConfig }) {
  const candidates = useMemo(() => {
    const a = analyticPlans(expectedRates(cfg), cfg.meanService, cfg.threshold, params.alpha, 1);
    const list: { names: string[]; plan: number[] }[] = [];
    const add = (name: string, plan: number[]) => {
      const hit = list.find((x) => x.plan.join() === plan.join());
      if (hit) hit.names.push(name);
      else list.push({ names: [name], plan });
    };
    add('Your plan', params.plan);
    add('SIPP', a.SIPP);
    add('Lag-SIPP', a['Lag-SIPP']);
    add('OL-max', a['OL-max']);
    add('Optimized (README)', OPTIMIZED_PLAN);
    return list;
  }, [cfg, params.alpha, params.plan]);

  const { result, progress } = useCompare(cfg, candidates.map((c) => c.plan), COMPARE_DAYS, params.seed);
  const ready = result && result.length === candidates.length ? result : null;
  const base = ready?.[0];

  return (
    <section id="compare" className="section">
      <SectionHead num="05" title="Compare plans">
        {progress !== null && <span className="pill ok">simulating {Math.round(progress * 100)}%</span>}
      </SectionHead>
      <p className="lede">
        The analytic rules recomputed for your settings, next to your plan. Every plan sees the same citizens with
        the same service needs (common random numbers, {COMPARE_DAYS} days), so the difference in P90 against your
        plan is measured day by day and its interval is much narrower than comparing two separate averages.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 12 }}>
        {candidates.map((c, k) => {
          const r = ready?.[k];
          let worst = 0;
          r?.lateByHour.forEach((p, i) => { if (p > r.lateByHour[worst]) worst = i; });
          const diff = r && base && k > 0 ? meanCi(r.dailyP90.map((v, i) => v - base.dailyP90[i])) : null;
          const mine = k === 0;
          return (
            <div key={c.plan.join()} className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10, borderColor: mine ? 'var(--accent)' : undefined }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <span className="card-title">{c.names.join(' = ')}</span>
                <span className="num card-sub">{sum(c.plan)} h</span>
              </div>
              <MiniBars plan={c.plan} />
              <div className="num card-sub">[{c.plan.join(', ')}]</div>
              <div className={r ? '' : 'stale'} style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                <div><div className="card-sub">P90</div><div className={`num ${r && r.p90 > cfg.threshold ? 'warn' : ''}`} style={{ fontSize: 16 }}>{r ? r.p90.toFixed(1) : '…'}</div></div>
                <div><div className="card-sub">Mean</div><div className="num" style={{ fontSize: 16 }}>{r ? r.meanWait.toFixed(2) : '…'}</div></div>
                <div><div className="card-sub">Worst hour</div><div className={`num ${r && r.lateByHour[worst] > params.alpha ? 'warn' : ''}`} style={{ fontSize: 16 }}>{r ? pct(r.lateByHour[worst]) : '…'}</div></div>
              </div>
              <div className="card-sub" style={{ minHeight: 18 }}>
                {mine ? 'Baseline for the paired differences.' : diff ? (
                  <>ΔP90 vs yours <span className="num">{diff.mean >= 0 ? '+' : ''}{diff.mean.toFixed(2)}</span> min
                    {' '}<span className="num">({diff.ci[0].toFixed(2)} to {diff.ci[1].toFixed(2)})</span>
                    {r && ` · worst ${HOUR_RANGES[worst]}`}</>
                ) : ''}
              </div>
              {!mine && (
                <button className="btn sm" style={{ alignSelf: 'flex-start' }} onClick={() => dispatch({ type: 'set', patch: { plan: c.plan } })}>
                  Load into editor
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
