import { SectionHead } from '../components/ui';

interface Bar { label: string; frac: number; value: string; warn?: boolean }
interface Finding { big: string; title: string; bars: Bar[]; body: string }

/** Headline results from research/REPORT.md (24-setting factorial design, fixed seeds). */
const FINDINGS: Finding[] = [
  {
    big: '+6.7%',
    title: 'SIPP overstaffs; no analytic rule significantly understaffed',
    bars: [
      { label: 'SIPP average', frac: 6.7 / 18.6, value: '+6.7%', warn: true },
      { label: 'SIPP worst case', frac: 14.5 / 18.6, value: '+14.5%', warn: true },
      { label: 'OL-max average', frac: 1, value: '+18.6%', warn: true },
    ],
    body: 'Extra staff-hours versus the simulation-based plan. Most of the excess sits in the opening hour and the early-afternoon ramp.',
  },
  {
    big: '≈ ½',
    title: 'Lag corrections recover about half of the excess',
    bars: [
      { label: 'Service ≥ 8 min', frac: 0.5, value: '~half' },
      { label: 'Service 4 min', frac: 0.1, value: 'little' },
    ],
    body: 'Shifting demand forward by one mean service time helps when services are long relative to the one-hour block.',
  },
  {
    big: '8% → 22%',
    title: 'Demand uncertainty is a scale effect',
    bars: [
      { label: '2-window office', frac: 8 / 22, value: '+8%' },
      { label: '24-window office', frac: 1, value: '+22%' },
    ],
    body: 'Extra staff needed to absorb 20% day-to-day demand uncertainty. Big offices lose the most because their Poisson noise is relatively small.',
  },
  {
    big: '18 vs 22 h',
    title: 'The service-level definition is a policy choice',
    bars: [
      { label: 'Loosest definition', frac: 18 / 22, value: '18 h' },
      { label: 'Strictest definition', frac: 1, value: '22 h' },
    ],
    body: 'Same office, same demand. Only the definition of “good service” changes the required staffing.',
  },
];

export function Findings() {
  return (
    <section id="findings" className="section">
      <SectionHead num="07" title="Research findings" />
      <p className="lede">From <code>research/REPORT.md</code>. Common random numbers also cut the variance of plan-vs-plan comparisons by a median of 40×.</p>
      <div className="grid2">
        {FINDINGS.map((f) => (
          <div key={f.title} className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div className="num accent" style={{ fontSize: 26 }}>{f.big}</div>
            <div className="card-title">{f.title}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {f.bars.map((b) => (
                <div key={b.label} style={{ display: 'grid', gridTemplateColumns: '130px minmax(0, 1fr) 56px', alignItems: 'center', gap: 8 }}>
                  <span className="card-sub">{b.label}</span>
                  <div style={{ height: 8, borderRadius: 4, background: 'var(--grid)' }}>
                    <div style={{ height: 8, borderRadius: 4, width: `${b.frac * 100}%`, background: b.warn ? 'var(--warn)' : 'var(--accent)' }} />
                  </div>
                  <span className="num" style={{ fontSize: 12, textAlign: 'right' }}>{b.value}</span>
                </div>
              ))}
            </div>
            <div className="card-sub" style={{ lineHeight: 1.5 }}>{f.body}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
