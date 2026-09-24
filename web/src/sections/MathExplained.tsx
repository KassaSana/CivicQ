import { type Dispatch, useMemo, useState } from 'react';
import { Figure, SectionHead, pct, useTip } from '../components/ui';
import { erlangC, offeredLoad, type sippHourly } from '../sim/analytic';
import { DAY_MINUTES, HOUR_RANGES, type SimConfig, expectedRates } from '../sim/model';
import { type Aggregate, QUEUE_BIN } from '../sim/stats';
import type { Action, Params } from '../state';

type Hourly = ReturnType<typeof sippHourly>;

const W = 400, H = 230, X0 = 32, X1 = 396, TOP = 16, BASE = 200;
const xm = (m: number) => X0 + (m / DAY_MINUTES) * (X1 - X0);
const X_TICKS: [number, string][] = [[0, '8 AM'], [120, '10'], [240, '12 PM'], [360, '2'], [480, '4 PM']];

function Axis({ yMax, ticks, label }: { yMax: number; ticks: number[]; label: string }) {
  const y = (v: number) => BASE - (v / yMax) * (BASE - TOP);
  return (
    <>
      {ticks.map((v) => (
        <g key={v}>
          <line className="gridline" x1={X0} x2={X1} y1={y(v)} y2={y(v)} />
          <text className="tick" x={X0 - 6} y={y(v) + 4} textAnchor="end" fontSize="10">{v}</text>
        </g>
      ))}
      {X_TICKS.map(([m, l]) => (
        <text key={m} className="tick" x={xm(m)} y={H - 8} textAnchor="middle" fontSize="10">{l}</text>
      ))}
      <text className="tick" x={X0} y={10} fontSize="10">{label}</text>
    </>
  );
}

export function MathExplained({ params, dispatch, hourly, agg, cfg }: {
  params: Params; dispatch: Dispatch<Action>; hourly: Hourly; agg: Aggregate | null; cfg: SimConfig;
}) {
  const [hour, setHour] = useState(0);
  const h = hourly[hour];
  const c = params.plan[hour];
  const lam = expectedRates(cfg)[hour];
  const sim = agg?.lateByHour[hour];
  const simCi = agg?.lateCi[hour];

  const note =
    sim === undefined ? '' :
    h.late > sim + 0.02
      ? 'Theory is more pessimistic here. It assumes the hour starts already congested, but the real office opens with nobody in line, or demand has just dropped so the hour starts calmer than steady state.'
      : h.late < sim - 0.02
        ? 'Theory is too optimistic here: the backlog from the previous, busier hour carries over. Congestion lags demand.'
        : 'Here steady-state theory and the simulation roughly agree.';

  // Queue length: simulation vs SIPP steady state per hour
  const lq = hourly.map((x) => x.queueLength);
  const qMax = Math.max(1, Math.ceil(Math.max(...(agg?.queueCurve ?? [0]), ...lq.filter(Number.isFinite)) + 0.2));
  const yq = (v: number) => BASE - (Math.min(v, qMax) / qMax) * (BASE - TOP);
  const qPts = (agg?.queueCurve ?? []).map((v, k) => `${xm(k * QUEUE_BIN + QUEUE_BIN / 2)},${yq(v)}`).join(' ');
  const sippPath = lq.map((v, i) => `${i ? 'L' : 'M'}${xm(i * 60)},${yq(Number.isFinite(v) ? v : qMax)} L${xm(i * 60 + 60)},${yq(Number.isFinite(v) ? v : qMax)}`).join(' ');

  // Offered load m(t) vs the hourly λ/μ steps
  const ol = useMemo(() => offeredLoad(expectedRates(cfg), cfg.meanService, cfg.serviceDist, cfg.serviceCv, 2), [cfg]);
  const mMax = Math.ceil(Math.max(...ol.m, ...hourly.map((x) => x.load), ...params.plan) + 0.3);
  const ym = (v: number) => BASE - (v / mMax) * (BASE - TOP);
  const mPts = ol.t.map((t, i) => `${xm(t)},${ym(ol.m[i])}`).join(' ');
  const stepPath = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${xm(i * 60)},${ym(v)} L${xm(i * 60 + 60)},${ym(v)}`).join(' ');
  const tipQ = useTip(W, H);

  return (
    <section id="math" className="section">
      <SectionHead title="Why the textbook formula disagrees" />
      <p className="lede">
        The standard way to staff an office like this is SIPP: treat each hour as its own steady-state queue and
        use the Erlang-C formula to pick the smallest number of windows that meets the target. The formula gives
        the chance that a visitor has to wait at all, C(s, a), and from it the chance of waiting longer than T:
      </p>
      <div className="formula">{`a = λ/μ                  (offered load)
C(s,a) = aˢ/s!·s/(s−a) ÷ [Σₖ₌₀ˢ⁻¹ aᵏ/k! + aˢ/s!·s/(s−a)]
P(W > T) = C(s,a) · e^(−(sμ−λ)T)`}</div>
      <div className="seg hours" role="group" aria-label="Hour">
        {HOUR_RANGES.map((r, i) => (
          <button key={r} className={hour === i ? 'on' : ''} aria-pressed={hour === i} onClick={() => setHour(i)}>{r}</button>
        ))}
      </div>
      <p style={{ lineHeight: 2 }}>
        In the {HOUR_RANGES[hour]} hour, <span className="chip num">λ = {lam.toFixed(1)}</span> people arrive and each takes{' '}
        <span className="chip num">{cfg.meanService}</span> minutes, so the load is <span className="chip num">a = {h.load.toFixed(2)}</span>. With{' '}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, verticalAlign: 'middle' }}>
          <button className="btn step" aria-label="Fewer windows" onClick={() => dispatch({ type: 'step', hour, delta: -1 })}>−</button>
          <span className="chip num">{c}</span>
          <button className="btn step" aria-label="More windows" onClick={() => dispatch({ type: 'step', hour, delta: 1 })}>+</button>
        </span>{' '}
        windows, the formula says <b className="num">{h.unstable ? 'everyone' : pct(h.late)}</b> wait more than {params.threshold} minutes.
      </p>
      <div className="statline">
        <div className="tile"><div className="label">Erlang-C, steady state</div><div className="value num">{pct(h.late)}</div>
          <div className="sub">{pct(erlangC(c, h.load))} wait at all</div></div>
        <div className="tile"><div className="label">Simulated, office opens empty</div>
          <div className="value num accent">{sim === undefined ? '…' : pct(sim)}</div>
          <div className="sub">{simCi ? `95% interval ${pct(simCi[0])}–${pct(simCi[1])}` : ''}</div></div>
      </div>
      <p>{note}</p>
      <div className="figs wide">
        <Figure n={6} caption={<>Average number waiting through the day. SIPP (dashed) assumes every hour is already
          in steady state. The simulated line (solid) starts at zero and carries each hour’s backlog into the next,
          so congestion lags demand.</>}>
          <div className="rel">
            <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mean queue length through the day, simulated versus SIPP">
              <Axis yMax={qMax} ticks={Array.from({ length: qMax + 1 }, (_, i) => i)} label="waiting" />
              <path d={sippPath} fill="none" stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="5 3" />
              <polyline points={qPts} fill="none" stroke="var(--accent)" strokeWidth="2.2" />
              {lq.map((v, i) => (
                <rect key={i} x={xm(i * 60)} y={TOP} width={xm(60) - X0} height={BASE - TOP} fill="transparent"
                  {...tipQ.bind(xm(i * 60 + 30), TOP + 10, `${HOUR_RANGES[i]}: SIPP ${Number.isFinite(v) ? v.toFixed(2) : '∞'} · sim ${
                    agg ? (agg.queueCurve.slice(i * 6, i * 6 + 6).reduce((a, b) => a + b, 0) / 6).toFixed(2) : '…'}`)} />
              ))}
            </svg>
            {tipQ.node}
          </div>
        </Figure>
        <Figure n={7} caption={<>Offered load m(t) = ∫ λ(t−u)·P(S &gt; u) du: how many windows would be busy if there
          were unlimited windows. It starts at zero when the doors open and trails the hourly λ/μ steps (dashed).
          The grey steps are your windows.</>}>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Offered load over the day">
            <Axis yMax={mMax} ticks={Array.from({ length: mMax + 1 }, (_, i) => i)} label="windows" />
            <path d={stepPath(params.plan)} fill="none" stroke="var(--dot)" strokeWidth="1.5" />
            <path d={stepPath(hourly.map((x) => x.load))} fill="none" stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="5 3" />
            <polyline points={mPts} fill="none" stroke="var(--accent)" strokeWidth="2.2" />
          </svg>
        </Figure>
      </div>
    </section>
  );
}
