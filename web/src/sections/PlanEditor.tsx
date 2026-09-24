import { type Dispatch, type KeyboardEvent, useRef } from 'react';
import { Figure, Num, SectionHead, Tile, useTip } from '../components/ui';
import { HOUR_LABELS, HOUR_RANGES, sum } from '../sim/model';
import type { sippHourly } from '../sim/analytic';
import { type Action, MAX_WINDOWS, type Params } from '../state';

type Hourly = ReturnType<typeof sippHourly>;

const W = 800, H = 250, X0 = 40, Y0 = 20, Y1 = 220;
const colW = (W - 10 - X0) / 8;
const y = (v: number) => Y1 - (Math.min(v, MAX_WINDOWS) / MAX_WINDOWS) * (Y1 - Y0);

export function PlanEditor({ params, dispatch, hourly, arrivals, meanWait, meanWaitSimulated }: {
  params: Params; dispatch: Dispatch<Action>; hourly: Hourly; arrivals: number[];
  meanWait: number; meanWaitSimulated: boolean;
}) {
  const refs = useRef<(SVGGElement | null)[]>([]);
  const { node: tip, bind } = useTip(W, H);
  const unstable = hourly.filter((h) => h.unstable).length;
  const staffHours = sum(params.plan);
  const cost = meanWait + 0.5 * staffHours;

  const onKey = (i: number) => (e: KeyboardEvent) => {
    const moves: Record<string, () => void> = {
      ArrowUp: () => dispatch({ type: 'step', hour: i, delta: 1 }),
      ArrowDown: () => dispatch({ type: 'step', hour: i, delta: -1 }),
      ArrowRight: () => refs.current[Math.min(7, i + 1)]?.focus(),
      ArrowLeft: () => refs.current[Math.max(0, i - 1)]?.focus(),
    };
    if (moves[e.key]) {
      e.preventDefault();
      moves[e.key]();
    }
  };

  const loadPts = hourly.map((h, i) => `${X0 + colW * (i + 0.5)},${y(h.load)}`).join(' ');

  return (
    <section id="plan" className="section">
      <SectionHead title="Setting the staffing plan" />
      <p className="lede">
        The plan is just eight numbers: how many windows are open in each hour. The line over the bars is the
        offered load λ/μ, the number of windows the visitors would keep busy on average. If an hour’s load reaches
        its window count, that hour can never catch up and the line grows until the rush passes.
      </p>
      <Figure n={2} wide caption={<>Windows per hour (bars) against offered load (line). Click a bar to add a window,
        Shift-click to remove one, or focus it and use the arrow keys. Hours drawn in red are unstable.{' '}
        <button className="linkish" onClick={() => dispatch({ type: 'set', patch: { plan: [2, 3, 3, 2, 2, 3, 3, 3] } })}>Reset the plan</button>.</>}>
        <div className="statline">
          <Tile variant="flat" label="Staff-hours" value={staffHours} />
          <Tile variant="flat" label="Cost, w₁·W̄ + w₂·Σs" value={<Num value={cost} />}
            sub={meanWaitSimulated ? 'W̄ simulated' : 'W̄ from Erlang-C, updating'} />
          <Tile variant="flat" label="Visitors a day" value={Math.round(sum(arrivals))} />
          <Tile variant={unstable ? 'bad' : 'flat'} label="Unstable hours" value={unstable} />
        </div>
        <div className="rel scroll-x">
          <svg className="chart plan-chart" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Windows per hour. Use arrow keys to edit.">
            {[0, 2, 4, 6].map((v) => (
              <g key={v}>
                <line className="gridline" x1={X0} x2={W - 10} y1={y(v)} y2={y(v)} />
                <text className="tick" x={X0 - 10} y={y(v) + 4} textAnchor="end">{v}</text>
              </g>
            ))}
            {params.plan.map((c, i) => {
              const h = hourly[i], cx = X0 + colW * (i + 0.5);
              return (
                <g key={i} className="plan-col" tabIndex={0} role="slider" ref={(el) => { refs.current[i] = el; }}
                  aria-label={`Windows ${HOUR_RANGES[i]}`} aria-valuenow={c} aria-valuemin={1} aria-valuemax={MAX_WINDOWS}
                  onKeyDown={onKey(i)} style={{ cursor: 'pointer' }}
                  onClick={(e) => dispatch({ type: 'step', hour: i, delta: e.shiftKey ? -1 : 1 })}
                  {...bind(cx, y(c) - 12, `${HOUR_RANGES[i]}: ${c} windows · λ ${arrivals[i].toFixed(1)}/h · load ${h.load.toFixed(2)}`)}>
                  <rect x={X0 + colW * i} y={Y0} width={colW} height={Y1 - Y0} fill="transparent" />
                  <rect className="bar" x={cx - 28} width={56} y={y(c)} height={Y1 - y(c)}
                    fill={h.unstable ? 'var(--warn-soft)' : 'var(--accent-soft)'}
                    stroke={h.unstable ? 'var(--warn)' : 'var(--accent)'} strokeWidth={h.unstable ? 1.5 : 1} rx={1} />
                  <text className="num" x={cx} y={y(c) - 8} textAnchor="middle" fontSize="12" fill="var(--text)">{c}</text>
                  <text className="tick" x={cx} y={H - 10} textAnchor="middle">{HOUR_LABELS[i]}</text>
                </g>
              );
            })}
            <polyline points={loadPts} fill="none" stroke="var(--text)" strokeWidth="2" pointerEvents="none" />
            {hourly.map((h, i) => (
              <circle key={i} cx={X0 + colW * (i + 0.5)} cy={y(h.load)} r={4} pointerEvents="none"
                fill={h.unstable ? 'var(--warn)' : 'var(--text)'} />
            ))}
          </svg>
          {tip}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: `36px repeat(8, minmax(0, 1fr))`, rowGap: 6, alignItems: 'center' }}>
          <span className="card-sub">s</span>
          {params.plan.map((_, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'center', gap: 4, flexWrap: 'wrap' }}>
              <button className="btn step" aria-label={`Remove a window at ${HOUR_RANGES[i]}`}
                onClick={() => dispatch({ type: 'step', hour: i, delta: -1 })}>−</button>
              <button className="btn step" aria-label={`Add a window at ${HOUR_RANGES[i]}`}
                onClick={() => dispatch({ type: 'step', hour: i, delta: 1 })}>+</button>
            </div>
          ))}
          <span className="card-sub">λ/h</span>
          {arrivals.map((l, i) => <span key={i} className="num card-sub" style={{ textAlign: 'center' }}>{l.toFixed(l % 1 ? 1 : 0)}</span>)}
          <span className="card-sub">ρ</span>
          {hourly.map((h, i) => (
            <span key={i} className={`num ${h.unstable ? 'warn' : 'muted'}`} style={{ textAlign: 'center', fontSize: 12 }}>
              {h.unstable ? '≥100%' : `${Math.round(h.rho * 100)}%`}
            </span>
          ))}
        </div>
      </Figure>
    </section>
  );
}
