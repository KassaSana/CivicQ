import { type Dispatch, useMemo } from 'react';
import { SectionHead, pct } from '../components/ui';
import { analyticPlans } from '../sim/analytic';
import { HOUR_RANGES, OPTIMIZED_PLAN, type SimConfig, expectedRates, sum } from '../sim/model';
import { meanCi } from '../sim/stats';
import type { Action, Params } from '../state';
import { useCompare } from '../useSim';

const COMPARE_DAYS = 500;

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
      <SectionHead title="The textbook rules, side by side">
        {progress !== null && <span className="status">simulating, {Math.round(progress * 100)}%</span>}
      </SectionHead>
      <p className="lede">
        SIPP, its lagged variant and the offered-load rule (OL-max) are recomputed below for your settings, next to your
        plan. All of them face the same {COMPARE_DAYS} simulated days, with the same visitors and the same service times.
        So the difference in P90 against your plan is measured day by day, and its interval is much narrower than it
        would be for two separate averages.
      </p>
      <div className="table-wrap wide">
        <table className="plans">
          <caption><b>Table 1.</b> Candidate plans, {COMPARE_DAYS} paired days each. Waits in minutes; ΔP90 is against your plan, with a 95% interval.</caption>
          <thead>
            <tr>
              <th>Plan</th><th>Windows, 8am–4pm</th><th className="r">Staff-h</th><th className="r">Mean</th>
              <th className="r">P90</th><th>Worst hour</th><th>ΔP90</th><th />
            </tr>
          </thead>
          <tbody className={ready ? '' : 'stale'}>
            {candidates.map((c, k) => {
              const r = ready?.[k];
              let worst = 0;
              r?.lateByHour.forEach((p, i) => { if (p > r.lateByHour[worst]) worst = i; });
              const diff = r && base && k > 0 ? meanCi(r.dailyP90.map((v, i) => v - base.dailyP90[i])) : null;
              const mine = k === 0;
              return (
                <tr key={c.plan.join()} className={mine ? 'mine' : ''}>
                  <td>{c.names.join(' = ')}</td>
                  <td className="num">{c.plan.join(' ')}</td>
                  <td className="r num">{sum(c.plan)}</td>
                  <td className="r num">{r ? r.meanWait.toFixed(2) : '…'}</td>
                  <td className={`r num ${r && r.p90 > cfg.threshold ? 'warn' : ''}`}>{r ? r.p90.toFixed(1) : '…'}</td>
                  <td className={`num ${r && r.lateByHour[worst] > params.alpha ? 'warn' : ''}`}>
                    {r ? <>{pct(r.lateByHour[worst])} <span className="muted">at {HOUR_RANGES[worst]}</span></> : '…'}
                  </td>
                  <td className="num">
                    {mine ? <span className="muted">baseline</span> : diff ? (
                      <>{diff.mean >= 0 ? '+' : ''}{diff.mean.toFixed(2)} <span className="muted">({diff.ci[0].toFixed(2)} to {diff.ci[1].toFixed(2)})</span></>
                    ) : '…'}
                  </td>
                  <td>
                    {!mine && (
                      <button className="linkish" onClick={() => dispatch({ type: 'set', patch: { plan: c.plan } })}>use</button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
