import { useEffect, useMemo, useRef, useState } from 'react';
import { Figure, SectionHead } from '../components/ui';
import { DAY_MINUTES, type SimConfig, slotOf } from '../sim/model';
import { queueLengthAt, simulateDay } from '../sim/simulate';

const SPEEDS = [1, 10, 30, 60]; // simulated minutes per real second
const MAX_DOTS = 16;

export function clockLabel(t: number): string {
  const h = 8 + Math.floor(t / 60), m = Math.floor(t % 60);
  const h12 = ((h + 11) % 12) + 1;
  return `${h12}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
}

export function LiveQueue({ cfg, seed }: { cfg: SimConfig; seed: number }) {
  const day = useMemo(() => simulateDay(cfg, seed), [cfg, seed]);
  const end = DAY_MINUTES + day.overtime;
  const [t, setT] = useState(102);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(10);
  const tRef = useRef(t);
  tRef.current = t;

  useEffect(() => {
    if (!playing) return;
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const next = tRef.current + ((now - last) / 1000) * speed;
      last = now;
      if (next >= end) {
        setT(end);
        setPlaying(false);
        return;
      }
      setT(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, end]);

  // Space toggles playback unless the user is typing or on a control
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      const el = e.target as HTMLElement;
      if (el.closest('input, select, textarea, button, [role="slider"]')) return;
      e.preventDefault();
      setPlaying((p) => {
        if (!p && tRef.current >= end) setT(0);
        return !p;
      });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [end]);

  const maxW = Math.max(...cfg.plan);
  const open = t < DAY_MINUTES ? cfg.plan[slotOf(t)] : cfg.plan[7];
  const waiting = day.citizens.filter((c) => c.arrival <= t && c.start > t);
  const inService = day.citizens.filter((c) => c.start <= t && c.departure > t);
  const served = day.citizens.filter((c) => c.departure <= t).length;
  const justArrived = day.citizens.filter((c) => c.arrival <= t && t - c.arrival < 0.6 && c.start > t).length;
  const headWait = waiting.length ? t - waiting[0].arrival : 0;
  const busyCount = inService.length;

  // Sparkline of this day's queue length, doubling as the scrubber track
  const spark = useMemo(() => {
    const pts: string[] = [];
    let mx = 1;
    const vals: number[] = [];
    for (let m = 0; m <= end; m += 2) vals.push(queueLengthAt(day.citizens, m));
    for (const v of vals) mx = Math.max(mx, v);
    vals.forEach((v, i) => pts.push(`${((i * 2) / end) * 1000},${40 - (v / mx) * 36}`));
    return { d: `M0,40 L${pts.join(' L')} L1000,40 Z`, max: mx };
  }, [day, end]);

  const rowH = 44, winTop = 14;
  const svgH = Math.max(170, winTop * 2 + maxW * rowH);
  const midY = svgH / 2;

  return (
    <section id="queue" className="section">
      <SectionHead title="One simulated day" />
      <p className="lede">
        Start with a single day. Visitors arrive at random, but more often at the busy hours. They join one line
        and go to the first free window. Each dot below is a person. Press play (or Space) to run the day, or drag
        along the strip underneath to jump to a time.
      </p>
      <Figure n={1} caption={<>One day under the current plan, seed <span className="num">{seed}</span>. The strip under the
        clock is this day’s line length (peak <span className="num">{spark.max}</span>); the dashed mark is closing time.
        Red dots have waited longer than {cfg.threshold} minutes.</>}>
      <div className="live">
        <svg className="chart" viewBox={`0 0 600 ${svgH}`} role="img"
          aria-label={`At ${clockLabel(t)}: ${waiting.length} waiting, ${busyCount} of ${open} windows busy`}>
          <rect x="8" y={midY - 25} width="40" height="50" rx="2" fill="none" stroke="var(--line)" strokeWidth="1.5" />
          <text x="28" y={midY + 42} className="tick" textAnchor="middle">{t >= DAY_MINUTES ? 'closed' : 'door'}</text>
          {justArrived > 0 && <circle cx="70" cy={midY} r="7" fill="var(--dot)" />}
          <line x1="110" y1={midY} x2="410" y2={midY} stroke="var(--grid)" strokeWidth="26" strokeLinecap="round" />
          {waiting.slice(0, MAX_DOTS).map((c, i) => (
            <circle key={c.arrival} cx={400 - i * 18} cy={midY} r="7.5"
              fill={t - c.arrival > cfg.threshold ? 'var(--warn)' : 'var(--accent)'}>
              <title>{`waiting ${(t - c.arrival).toFixed(1)} min`}</title>
            </circle>
          ))}
          {waiting.length > MAX_DOTS && (
            <text x="110" y={midY - 20} className="tick num">+{waiting.length - MAX_DOTS} more</text>
          )}
          <text x="260" y={midY + 36} className="tick" textAnchor="middle">line (first come, first served)</text>
          {Array.from({ length: maxW }, (_, w) => {
            const isOpen = w < open && t < DAY_MINUTES;
            const c = inService.find((x) => x.window === w);
            const y = winTop + w * rowH;
            return (
              <g key={w} opacity={isOpen || c ? 1 : 0.4}>
                <rect x="450" y={y} width="140" height={rowH - 8} rx="2" fill="var(--panel)"
                  stroke={c ? 'var(--accent)' : 'var(--line)'} strokeWidth="1.5" strokeDasharray={isOpen || c ? undefined : '4 3'} />
                {c && <circle cx="472" cy={y + (rowH - 8) / 2} r="7.5" fill="var(--accent)" />}
                <text x="488" y={y + (rowH - 8) / 2 + 4} className="tick">
                  {`Window ${w + 1} · ${c ? 'busy' : isOpen ? 'idle' : 'closed'}`}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="live-stats">
          <div><div className="card-sub">Clock</div><div className="num" style={{ fontSize: 24 }}>{clockLabel(t)}</div>
            {t > DAY_MINUTES && <div className="card-sub warn">overtime +{(t - DAY_MINUTES).toFixed(0)} min</div>}</div>
          <div><div className="card-sub">In line</div><div className="num">{waiting.length}</div></div>
          <div><div className="card-sub">Busy windows</div><div className="num">{busyCount} / {open}</div></div>
          <div><div className="card-sub">Front of the line has waited</div><div className="num">{headWait.toFixed(1)} min</div></div>
          <div><div className="card-sub">Served so far</div><div className="num">{served} / {day.citizens.length}</div></div>
        </div>
        <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <button className="btn primary icon" style={{ width: 40, height: 40 }} aria-label={playing ? 'Pause' : 'Play'}
            onClick={() => {
              if (!playing && t >= end) setT(0);
              setPlaying(!playing);
            }}>
            {playing ? (
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2" y="1" width="3.5" height="12" fill="currentColor" /><rect x="8.5" y="1" width="3.5" height="12" fill="currentColor" /></svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1 L13 7 L3 13 Z" fill="currentColor" /></svg>
            )}
          </button>
          <div className="rel" style={{ flex: 1, minWidth: 200 }}>
            <svg viewBox="0 0 1000 40" preserveAspectRatio="none" style={{ width: '100%', height: 36, display: 'block' }} aria-hidden="true">
              <path d={spark.d} fill="var(--accent-soft)" stroke="var(--accent)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
              <line x1={(DAY_MINUTES / end) * 1000} x2={(DAY_MINUTES / end) * 1000} y1="0" y2="40" stroke="var(--warn)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
              <line x1={(t / end) * 1000} x2={(t / end) * 1000} y1="0" y2="40" stroke="var(--text)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
            </svg>
            <label htmlFor="clock" className="sr-only">Clock</label>
            <input id="clock" type="range" min={0} max={end} step={0.5} value={t}
              onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }} />
          </div>
          <div className="seg" role="group" aria-label="Playback speed">
            {SPEEDS.map((s) => (
              <button key={s} className={`num ${speed === s ? 'on' : ''}`} onClick={() => setSpeed(s)} aria-pressed={speed === s}>{s}×</button>
            ))}
          </div>
        </div>
      </div>
      </Figure>
    </section>
  );
}
