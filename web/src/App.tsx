import { useEffect, useMemo, useReducer, useState } from 'react';
import { ComparePlans } from './sections/ComparePlans';
import { Controls } from './sections/Controls';
import { Findings } from './sections/Findings';
import { Frontier } from './sections/Frontier';
import { LiveQueue } from './sections/LiveQueue';
import { MathExplained } from './sections/MathExplained';
import { PlanEditor } from './sections/PlanEditor';
import { Results } from './sections/Results';
import { sippHourly } from './sim/analytic';
import { expectedRates, sum } from './sim/model';
import { fromUrl, reducer, toConfig, toUrl } from './state';
import { useRun } from './useSim';

const REPO = 'https://github.com/KassaSana/CivicQ';

const CONTENTS: [string, string][] = [
  ['queue', 'One simulated day'], ['plan', 'Setting the staffing plan'], ['results', 'How long people wait'],
  ['math', 'Why the textbook formula disagrees'], ['compare', 'The textbook rules, side by side'],
  ['frontier', 'What each extra staff-hour buys'], ['findings', 'What the study found'],
];

type Theme = 'light' | 'dark';

function useTheme(): [Theme, () => void] {
  const system = (): Theme => (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.dataset.theme as Theme) || system());
  const toggle = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('civicq-theme', next); } catch { /* storage may be blocked */ }
    setTheme(next);
  };
  return [theme, toggle];
}

export function App() {
  const [params, dispatch] = useReducer(reducer, undefined, () => fromUrl(location.search));
  const [theme, toggleTheme] = useTheme();
  const [railOpen, setRailOpen] = useState(false);

  useEffect(() => { history.replaceState(null, '', toUrl(params)); }, [params]);

  const cfg = useMemo(() => toConfig(params), [params]);
  // Analytic views see walk-ins plus expected booked shows
  const rates = useMemo(() => expectedRates(cfg), [cfg]);
  const hourly = useMemo(() => sippHourly(params.plan, rates, cfg.meanService, cfg.threshold), [params.plan, rates, cfg]);
  const { result: agg, progress } = useRun(cfg, params.days, params.seed);
  const updating = progress !== null;

  // Erlang-C mean wait (arrival-weighted) until the simulation for this plan lands
  const erlangWait = useMemo(() => {
    let s = 0, n = 0;
    hourly.forEach((h, i) => { s += h.meanWait * rates[i]; n += rates[i]; });
    return s / n;
  }, [hourly, rates]);

  // Escape closes the assumptions drawer
  useEffect(() => {
    if (!railOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setRailOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [railOpen]);

  const openAssumptions = () => setRailOpen(true);
  const perDay = Math.round(sum(rates));

  return (
    <>
      <header className="header">
        <div className="header-inner">
          <a className="brand" href="#top">CivicQ</a>
          <span className="spacer" />
          <a href={`${REPO}/blob/master/research/REPORT.md`} className="only-wide">Report</a>
          <a href={REPO}>Code</a>
          <button className="linkish" onClick={openAssumptions} aria-expanded={railOpen}>Assumptions</button>
          <button className="linkish" onClick={toggleTheme} aria-label="Toggle light or dark mode">
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
        {updating && <div className="progress" style={{ width: `${Math.max(4, (progress ?? 0) * 100)}%` }} />}
      </header>

      <main className="page" id="top">
        <div className="title-block">
          <h1>How many windows does a walk-in office need?</h1>
          <div className="byline">
            KassaSana · September 2026 · <a href={`${REPO}/blob/master/research/REPORT.md`}>full report</a> · <a href={REPO}>source</a>
          </div>
          <p>
            A permit office opens at 8 and sees about <span className="num">{perDay}</span> people a day, in waves: a
            morning rush, a lull at lunch, a second rush after 1. Each visit takes <span className="num">{params.meanService}</span> minutes
            on average. The manager has to decide, hour by hour, how many windows to open. Too few and the line runs
            out the door; too many and staff sit idle on the public’s money.
          </p>
          <p>
            This page runs a queue simulator in your browser. Change the plan below and it replays{' '}
            <span className="num">{params.days.toLocaleString()}</span> days to show what people would wait. The current plan
            uses <span className="num">{sum(params.plan)}</span> staff-hours. The target is that at most{' '}
            <span className="num">{Math.round(params.alpha * 100)}%</span> of each hour’s visitors wait more than{' '}
            <span className="num">{params.threshold}</span> minutes. Every assumption can be changed under{' '}
            <button className="linkish" onClick={openAssumptions}>Assumptions</button>.
          </p>
          <ol className="contents">
            {CONTENTS.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
          </ol>
        </div>

        <LiveQueue cfg={cfg} seed={params.seed} />
        <PlanEditor params={params} dispatch={dispatch} hourly={hourly} arrivals={rates}
          meanWait={!updating && agg ? agg.meanWait : erlangWait} meanWaitSimulated={!updating && !!agg} />
        <Results agg={agg} hourly={hourly} alpha={params.alpha} threshold={params.threshold} updating={updating}
          leaving={params.abandonment !== 'none'} />
        <MathExplained params={params} dispatch={dispatch} hourly={hourly} agg={agg} cfg={cfg} />
        <ComparePlans params={params} dispatch={dispatch} cfg={cfg} />
        <Frontier params={params} dispatch={dispatch} cfg={cfg} />
        <Findings />

        <footer className="footer">
          <p>
            <b>Reproducing this.</b> The simulator is a TypeScript port of <code>cpp/src/simulation.cpp</code>. Its tests
            (<code>web/src/sim/sim.test.ts</code>) check it against Erlang-C, the Python research code and the C++
            executable. Day <i>k</i> always uses seed <span className="num">{params.seed}</span> + <i>k</i>, so every plan
            sees the same visitors, and a copied link reproduces this exact page.
          </p>
          <p><a href={REPO}>github.com/KassaSana/CivicQ</a></p>
        </footer>
      </main>

      <aside className={`rail ${railOpen ? 'open' : ''}`} aria-label="Model assumptions" aria-hidden={!railOpen}>
        <Controls params={params} dispatch={dispatch} onClose={() => setRailOpen(false)} />
      </aside>
      <div className={`backdrop ${railOpen ? 'open' : ''}`} onClick={() => setRailOpen(false)} />
    </>
  );
}
