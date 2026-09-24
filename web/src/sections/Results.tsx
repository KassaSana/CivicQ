import { Figure, Num, SectionHead, pct, useTip } from '../components/ui';
import type { sippHourly } from '../sim/analytic';
import { HOUR_LABELS, HOUR_RANGES } from '../sim/model';
import { type Aggregate, HIST_EDGES, HIST_LABELS } from '../sim/stats';

type Hourly = ReturnType<typeof sippHourly>;

function Histogram({ agg, threshold }: { agg: Aggregate; threshold: number }) {
  const W = 380, H = 200, x0 = 16, bw = 36, gap = 8, base = 172;
  const { node, bind } = useTip(W, H);
  const maxP = Math.max(0.05, ...agg.histogram);
  // Position a wait value on the categorical axis (bin 0 is exactly zero wait)
  const xOf = (w: number) => {
    if (w <= 0) return x0 + bw / 2;
    const k = HIST_EDGES.findIndex((e) => w <= e);
    const b = k < 0 ? HIST_EDGES.length : k;
    const lo = b === 0 ? 0 : HIST_EDGES[b - 1];
    const frac = k < 0 ? 0.5 : (w - lo) / (HIST_EDGES[b] - lo);
    return x0 + (b + 1) * (bw + gap) + frac * bw;
  };
  return (
    <Figure n={3} caption={<>Share of all visitors by wait in minutes, over {agg.days.toLocaleString()} simulated days.
      Most are served at once. Red bars are past the {threshold}-minute target.</>}>
      <div className="rel">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Wait time histogram">
          {agg.histogram.map((p, i) => {
            const h = (p / maxP) * 140, x = x0 + i * (bw + gap);
            const late = i > 0 && HIST_EDGES[i - 1] >= threshold;
            return (
              <g key={i} {...bind(x + bw / 2, base - h, `${HIST_LABELS[i]} min: ${pct(p)}`)}>
                <rect className="bar" x={x} y={base - h} width={bw} height={h}
                  fill={late ? 'var(--warn)' : 'var(--accent)'} opacity={i === 0 ? 0.45 : 0.8} />
                <text className="tick num" x={x + bw / 2} y={base - h - 4} textAnchor="middle" fontSize="10">{(p * 100).toFixed(1)}</text>
                <text className="tick" x={x + bw / 2} y={H - 8} textAnchor="middle" fontSize="10">{HIST_LABELS[i]}</text>
              </g>
            );
          })}
          <line className="target" x1={xOf(threshold)} x2={xOf(threshold)} y1={14} y2={base} />
          <text x={xOf(threshold) + 4} y={24} fontSize="11" fill="var(--warn)">{threshold}-min target</text>
          <line x1={xOf(agg.p90)} x2={xOf(agg.p90)} y1={30} y2={base} stroke="var(--text)" strokeWidth="1.5" />
          <text x={xOf(agg.p90) - 4} y={40} fontSize="11" fill="var(--text)" textAnchor="end">P90 ≈ {agg.p90.toFixed(1)}</text>
        </svg>
        {node}
      </div>
    </Figure>
  );
}

function Utilization({ agg, hourly }: { agg: Aggregate; hourly: Hourly }) {
  const W = 380, H = 200, base = 172, full = 130;
  const { node, bind } = useTip(W, H);
  return (
    <Figure n={4} caption={<>How busy the open windows are each hour. Bars are simulated; the black ticks are the
      offered utilization ρ = λ / (s·μ). The two differ because the office opens empty and the line carries over between hours.</>}>
      <div className="rel">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Utilization per hour">
          <line className="target" x1={30} x2={376} y1={base - full} y2={base - full} strokeWidth={1} />
          <text className="tick" x={26} y={base - full + 4} textAnchor="end" fontSize="10">100%</text>
          {agg.utilization.map((u, i) => {
            const x = 36 + i * 42, h = Math.min(u, 1.2) * full, r = Math.min(hourly[i].rho, 1.2) * full;
            return (
              <g key={i} {...bind(x + 15, base - h, `${HOUR_RANGES[i]}: ${pct(u)} busy (offered ${pct(hourly[i].rho, 0)})`)}>
                <rect className="bar" x={x} y={base - h} width={30} height={h}
                  fill={hourly[i].unstable ? 'var(--warn)' : 'var(--accent)'} opacity={0.8} />
                <line x1={x - 3} x2={x + 33} y1={base - r} y2={base - r} stroke="var(--text)" strokeWidth="1.5" />
                <text className="tick num" x={x + 15} y={base - Math.max(h, r) - 5} textAnchor="middle" fontSize="10">{Math.round(u * 100)}</text>
                <text className="tick" x={x + 15} y={H - 8} textAnchor="middle" fontSize="10">{HOUR_LABELS[i]}</text>
              </g>
            );
          })}
        </svg>
        {node}
      </div>
    </Figure>
  );
}

function LateByHour({ agg, hourly, alpha, threshold }: { agg: Aggregate; hourly: Hourly; alpha: number; threshold: number }) {
  const W = 800, H = 210, x0 = 44, top = 12, base = 182;
  const maxP = Math.max(0.2, Math.ceil(Math.max(...agg.lateCi.map((c) => c[1]), ...hourly.map((h) => h.late), alpha) * 10) / 10);
  const yl = (p: number) => base - (Math.min(p, maxP) / maxP) * (base - top);
  const step = (W - 10 - x0) / 8;
  const ticks = Array.from({ length: 5 }, (_, k) => (maxP * k) / 4);
  const { node, bind } = useTip(W, H);
  return (
    <Figure n={5} wide caption={<>Share of each hour’s visitors who wait more than {threshold} minutes. Dots are simulated,
      with 95% intervals; squares are what the steady-state Erlang-C formula predicts for the same hour. The dashed
      line is the {pct(alpha, 0)} allowance.</>}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="legend">
          <span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>Simulated, 95% CI</span>
          <span><svg width="10" height="10"><rect x="1" y="1" width="8" height="8" fill="none" stroke="var(--muted)" strokeWidth="1.5" /></svg>Erlang-C (SIPP)</span>
          <span className="warn">– – α = {pct(alpha, 0)}</span>
        </div>
      </div>
      <div className="rel">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-hour late share, simulated versus Erlang-C">
          {ticks.map((v) => (
            <g key={v}>
              <line className="gridline" x1={x0} x2={W - 10} y1={yl(v)} y2={yl(v)} />
              <text className="tick" x={x0 - 8} y={yl(v) + 4} textAnchor="end">{pct(v, 0)}</text>
            </g>
          ))}
          <line className="target" x1={x0} x2={W - 10} y1={yl(alpha)} y2={yl(alpha)} />
          {agg.lateByHour.map((p, i) => {
            const cx = x0 + step * (i + 0.5), e = hourly[i].late, ci = agg.lateCi[i];
            const miss = ci[0] > alpha;
            return (
              <g key={i}>
                <g {...bind(cx + 15, yl(e), `Erlang-C: ${pct(e)}`)}>
                  <rect x={cx + 10} y={yl(e) - 5} width={10} height={10} fill="var(--panel)"
                    stroke={e > alpha ? 'var(--warn)' : 'var(--muted)'} strokeWidth="1.5" />
                </g>
                <g {...bind(cx, yl(ci[1]), `Simulated: ${pct(p)} (${pct(ci[0])}–${pct(ci[1])})`)}>
                  <line x1={cx} x2={cx} y1={yl(ci[0])} y2={yl(ci[1])} stroke={miss ? 'var(--warn)' : 'var(--accent)'} strokeWidth="2" />
                  <circle cx={cx} cy={yl(p)} r={5} fill={p > alpha ? 'var(--warn)' : 'var(--accent)'} style={{ transition: 'cy 0.18s' }} />
                  <rect x={cx - 12} y={top} width={24} height={base - top} fill="transparent" />
                </g>
                <text className="tick" x={cx} y={H - 6} textAnchor="middle">{HOUR_RANGES[i]}</text>
              </g>
            );
          })}
        </svg>
        {node}
      </div>
    </Figure>
  );
}

export function Results({ agg, hourly, alpha, threshold, updating }: {
  agg: Aggregate | null; hourly: Hourly; alpha: number; threshold: number; updating: boolean;
}) {
  if (!agg) {
    return (
      <section id="results" className="section">
        <SectionHead title="How long people wait" />
        <p className="status">Running the first simulation…</p>
      </section>
    );
  }
  let worst = 0;
  agg.lateByHour.forEach((p, i) => { if (p > agg.lateByHour[worst]) worst = i; });
  const worstP = agg.lateByHour[worst];
  return (
    <section id="results" className="section">
      <SectionHead title="How long people wait">
        {updating && <span className="status">updating…</span>}
      </SectionHead>
      <p className={updating ? 'stale' : ''}>
        Over <span className="num">{agg.days.toLocaleString()}</span> simulated days, the average visitor waits{' '}
        <b className="num"><Num value={agg.meanWait} /> minutes</b> (95% interval{' '}
        <span className="num">{agg.meanWaitCi[0].toFixed(2)}–{agg.meanWaitCi[1].toFixed(2)}</span>). On a typical day
        nine in ten are served within <b className={`num ${agg.p90 > threshold ? 'warn' : ''}`}><Num value={agg.p90} digits={1} /> minutes</b>,
        and <span className="num">{(agg.fracDaysOk * 100).toFixed(1)}%</span> of days meet the {threshold}-minute P90 target.
        The hardest hour is {HOUR_RANGES[worst]}, when <span className={`num ${agg.lateCi[worst][0] > alpha ? 'warn' : ''}`}>{pct(worstP)}</span> of
        visitors wait longer than {threshold} minutes, against an allowance of {pct(alpha, 0)}. Clearing the line after
        closing takes <span className="num">{agg.overtime.toFixed(1)}</span> minutes of overtime on average, for{' '}
        <span className="num">{agg.arrivalsPerDay.toFixed(1)}</span> visitors a day.
      </p>
      <div className={`figs wide ${updating ? 'stale' : ''}`}>
        <Histogram agg={agg} threshold={threshold} />
        <Utilization agg={agg} hourly={hourly} />
      </div>
      <LateByHour agg={agg} hourly={hourly} alpha={alpha} threshold={threshold} />
      {agg.apptMeanWait !== null && agg.walkinMeanWait !== null && (
        <p>
          <b>Booked visitors.</b> About <span className="num">{agg.apptPerDay.toFixed(1)}</span> people a day come with an
          appointment. <span className="num">{pct(agg.apptLate ?? 0)}</span> of them wait over {threshold} minutes (mean{' '}
          <span className="num">{agg.apptMeanWait.toFixed(2)}</span> min), against{' '}
          <span className="num">{pct(agg.walkinLate ?? 0)}</span> of walk-ins (mean{' '}
          <span className="num">{agg.walkinMeanWait.toFixed(2)}</span> min). Evenly spaced bookings avoid the random
          clusters that make walk-ins wait, unless they are moved into hours this plan staffs thinly.
        </p>
      )}
    </section>
  );
}
