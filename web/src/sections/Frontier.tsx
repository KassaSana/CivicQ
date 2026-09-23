import { type Dispatch, useState } from 'react';
import { SectionHead, useTip } from '../components/ui';
import { type SimConfig, sum } from '../sim/model';
import type { FrontierPoint } from '../sim/worker';
import type { Action, Params } from '../state';
import { useFrontier } from '../useSim';

/** README frontier table (python/optimizer.py --frontier, 300 replications, default settings). */
const README_FRONTIER: FrontierPoint[] = [
  { hours: 17, plan: [2, 2, 2, 2, 2, 2, 3, 2], p90: 19.6, p90Ci: [18.2, 21.0], meanWait: 6.6 },
  { hours: 18, plan: [2, 3, 2, 2, 2, 2, 3, 2], p90: 13.5, p90Ci: [12.5, 14.5], meanWait: 4.2 },
  { hours: 19, plan: [2, 3, 2, 2, 2, 3, 3, 2], p90: 11.7, p90Ci: [10.8, 12.6], meanWait: 3.4 },
  { hours: 20, plan: [2, 3, 3, 2, 2, 3, 3, 2], p90: 10.0, p90Ci: [9.2, 10.8], meanWait: 2.8 },
  { hours: 21, plan: [3, 3, 3, 2, 2, 3, 3, 2], p90: 8.2, p90Ci: [7.5, 8.9], meanWait: 2.2 },
  { hours: 22, plan: [3, 3, 3, 2, 2, 3, 3, 3], p90: 6.6, p90Ci: [6.0, 7.2], meanWait: 1.8 },
  { hours: 24, plan: [3, 4, 3, 2, 3, 3, 3, 3], p90: 4.5, p90Ci: [4.0, 5.0], meanWait: 1.2 },
];

const W = 560, H = 280, X0 = 48, X1 = 540, TOP = 20, BASE = 250;

export function Frontier({ params, dispatch, cfg }: { params: Params; dispatch: Dispatch<Action>; cfg: SimConfig }) {
  const [token, setToken] = useState(0);
  const { result, progress } = useFrontier(cfg, params.seed, token);
  const points = result ?? README_FRONTIER;
  const [selHours, setSelHours] = useState(21);
  const sel = points.find((p) => p.hours === selHours) ?? points[0];
  const { node, bind } = useTip(W, H);

  const hMin = Math.min(...points.map((p) => p.hours)), hMax = Math.max(...points.map((p) => p.hours));
  const yMax = Math.max(cfg.threshold + 5, Math.ceil(Math.min(60, Math.max(...points.map((p) => p.p90Ci[1]))) / 5) * 5);
  const x = (h: number) => X0 + ((h - hMin) / Math.max(1, hMax - hMin)) * (X1 - X0);
  const y = (v: number) => BASE - (Math.min(v, yMax) / yMax) * (BASE - TOP);
  const cheapestOk = points.find((p) => p.p90Ci[1] <= cfg.threshold);
  const yours = sum(params.plan);

  return (
    <section id="frontier" className="section">
      <SectionHead num="06" title="Cost vs. service">
        <span className="spacer" />
        <button className="btn sm soft" disabled={progress !== null} onClick={() => setToken((t) => t + 1)}>
          {progress !== null ? `Searching ${Math.round(progress * 100)}%` : 'Recompute for current settings'}
        </button>
      </SectionHead>
      <p className="lede">
        The best P90 wait for each staff-hour budget.{' '}
        {result
          ? 'Searched live: all plans with 2–4 windows per hour screened on 10 days, then the best 4 per budget confirmed on 300 days.'
          : 'Showing the README frontier (300 days per plan, default settings). Recompute to search with your current settings; it takes about 10 seconds.'}
      </p>
      <div className="card frontier">
        <div className="rel">
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Best P90 wait for each staff-hour budget">
            {Array.from({ length: yMax / 5 + 1 }, (_, i) => i * 5).map((v) => (
              <g key={v}>
                <line className="gridline" x1={X0} x2={X1} y1={y(v)} y2={y(v)} />
                <text className="tick" x={X0 - 8} y={y(v) + 4} textAnchor="end">{v}</text>
              </g>
            ))}
            <line className="target" x1={X0} x2={X1} y1={y(cfg.threshold)} y2={y(cfg.threshold)} />
            <text x={X1} y={y(cfg.threshold) - 6} fontSize="11" fill="var(--warn)" textAnchor="end">P90 target {cfg.threshold} min</text>
            {yours >= hMin && yours <= hMax && (
              <g>
                <line x1={x(yours)} x2={x(yours)} y1={TOP} y2={BASE} stroke="var(--dot)" strokeDasharray="3 3" />
                <text className="tick" x={x(yours) + 4} y={TOP + 10} fontSize="10">your plan</text>
              </g>
            )}
            <polyline points={points.map((p) => `${x(p.hours)},${y(p.p90)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="1.5" opacity={0.6} />
            {points.map((p) => {
              const on = p.hours === sel.hours, miss = p.p90 > cfg.threshold;
              return (
                <g key={p.hours} style={{ cursor: 'pointer' }} onClick={() => setSelHours(p.hours)}
                  {...bind(x(p.hours), y(p.p90Ci[1]) - 4, `${p.hours} h: P90 ${p.p90.toFixed(1)} (${p.p90Ci[0].toFixed(1)}–${p.p90Ci[1].toFixed(1)})`)}>
                  <line x1={x(p.hours)} x2={x(p.hours)} y1={y(p.p90Ci[0])} y2={y(p.p90Ci[1])} stroke={miss ? 'var(--warn)' : 'var(--accent)'} strokeWidth="2" opacity={0.5} />
                  <circle cx={x(p.hours)} cy={y(p.p90)} r={on ? 8 : 5} fill={miss ? 'var(--warn)' : 'var(--accent)'}
                    stroke={on ? 'var(--panel)' : 'none'} strokeWidth={3} />
                  <circle cx={x(p.hours)} cy={y(p.p90)} r={14} fill="transparent" />
                  <text className="tick num" x={x(p.hours)} y={H - 8} textAnchor="middle">{p.hours}h</text>
                </g>
              );
            })}
            <text className="tick" x={X0} y={12} fontSize="11">P90 wait (min)</text>
          </svg>
          {node}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="card-sub">Selected budget</div>
          <div className="num" style={{ fontSize: 28 }}>{sel.hours} h</div>
          <div className="num" style={{ fontSize: 13 }}>[{sel.plan.join(', ')}]</div>
          <div className="muted">
            P90 <span className="num" style={{ color: 'var(--text)' }}>{sel.p90.toFixed(1)}</span> min
            ({sel.p90Ci[0].toFixed(1)}–{sel.p90Ci[1].toFixed(1)}), mean <span className="num" style={{ color: 'var(--text)' }}>{sel.meanWait.toFixed(1)}</span> min.
          </div>
          {cheapestOk && <div className="muted">{cheapestOk.hours} h is the cheapest budget whose whole CI is under {cfg.threshold} min.</div>}
          <button className="btn soft" onClick={() => dispatch({ type: 'set', patch: { plan: sel.plan } })}>Load into editor</button>
        </div>
      </div>
    </section>
  );
}
