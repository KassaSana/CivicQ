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
import { fromUrl, reducer, toConfig, toUrl } from './state';
import { useRun } from './useSim';

const SECTIONS: [string, string][] = [
  ['queue', 'Live queue'], ['plan', 'Staffing plan'], ['results', 'Results'], ['math', 'The math'],
  ['compare', 'Compare plans'], ['frontier', 'Cost vs. service'], ['findings', 'Findings'],
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

function useActiveSection(): string {
  const [active, setActive] = useState('queue');
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id);
      },
      { rootMargin: '-70px 0px -60% 0px' },
    );
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, []);
  return active;
}

export function App() {
  const [params, dispatch] = useReducer(reducer, undefined, () => fromUrl(location.search));
  const [theme, toggleTheme] = useTheme();
  const [railOpen, setRailOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const active = useActiveSection();

  useEffect(() => { history.replaceState(null, '', toUrl(params)); }, [params]);

  const cfg = useMemo(() => toConfig(params), [params]);
  const hourly = useMemo(() => sippHourly(params.plan, cfg.arrivals, cfg.meanService, cfg.threshold), [params.plan, cfg]);
  const { result: agg, progress } = useRun(cfg, params.days, params.seed);
  const updating = progress !== null;

  // Erlang-C mean wait (arrival-weighted) until the simulation for this plan lands
  const erlangWait = useMemo(() => {
    let s = 0, n = 0;
    hourly.forEach((h, i) => { s += h.meanWait * cfg.arrivals[i]; n += cfg.arrivals[i]; });
    return s / n;
  }, [hourly, cfg]);

  return (
    <>
      <header className="header">
        <button className="btn icon only-mobile" aria-label="Sections" onClick={() => setNavOpen(!navOpen)} aria-expanded={navOpen}>
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" /></svg>
        </button>
        <div className="brand">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
            <rect x="1" y="1" width="20" height="20" rx="5" fill="none" stroke="var(--accent)" strokeWidth="1.6" />
            <circle cx="7" cy="11" r="1.8" fill="var(--accent)" /><circle cx="11" cy="11" r="1.8" fill="var(--accent)" />
            <rect x="14" y="7" width="3.5" height="8" rx="1" fill="var(--accent)" />
          </svg>
          <b>CivicQ</b>
          <span>Staffing a walk-in office, visually</span>
        </div>
        <span className="spacer" />
        <span className="num card-sub only-wide">seed {params.seed} · {params.days.toLocaleString()} days</span>
        <button className="btn only-narrow" onClick={() => setRailOpen(!railOpen)} aria-expanded={railOpen}>Model</button>
        <button className="btn" onClick={toggleTheme} aria-label="Toggle light or dark mode">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
            <path d="M8 2 A6 6 0 0 1 8 14 Z" fill="currentColor" />
          </svg>
          <span className="only-wide">{theme === 'dark' ? 'Dark' : 'Light'}</span>
        </button>
        {updating && <div className="progress" style={{ width: `${Math.max(4, (progress ?? 0) * 100)}%` }} />}
      </header>

      <div className="shell">
        <nav className={`nav ${navOpen ? 'open' : ''}`} aria-label="Sections">
          <div className="nav-inner">
            <div className="nav-title">On this page</div>
            {SECTIONS.map(([id, label], i) => (
              <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} onClick={() => setNavOpen(false)}>
                <span className="num">0{i + 1}</span>{label}
              </a>
            ))}
          </div>
        </nav>

        <main className="main">
          <LiveQueue cfg={cfg} seed={params.seed} />
          <PlanEditor params={params} dispatch={dispatch} hourly={hourly} arrivals={cfg.arrivals}
            meanWait={!updating && agg ? agg.meanWait : erlangWait} meanWaitSimulated={!updating && !!agg} />
          <Results agg={agg} hourly={hourly} alpha={params.alpha} threshold={params.threshold} updating={updating} />
          <MathExplained params={params} dispatch={dispatch} hourly={hourly} agg={agg} cfg={cfg} />
          <ComparePlans params={params} dispatch={dispatch} cfg={cfg} />
          <Frontier params={params} dispatch={dispatch} cfg={cfg} />
          <Findings />
          <footer className="card-sub">
            Simulator ported from <code>cpp/src/simulation.cpp</code> and checked against it in <code>web/src/sim/sim.test.ts</code>.
          </footer>
        </main>

        <aside className={`rail ${railOpen ? 'open' : ''}`} aria-label="Model controls">
          <Controls params={params} dispatch={dispatch} />
        </aside>
        <div className={`backdrop ${railOpen || navOpen ? 'open' : ''}`} onClick={() => { setRailOpen(false); setNavOpen(false); }} />
      </div>
    </>
  );
}
