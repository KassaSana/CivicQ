import { type Dispatch, useState } from 'react';
import type { Placement, ServiceDist } from '../sim/model';
import type { Action, Params } from '../state';

function Slider({ id, label, value, min, max, step, format, onChange }: {
  id: string; label: string; value: number; min: number; max: number; step: number;
  format: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div className="field">
      <div className="field-row"><label htmlFor={id}>{label}</label><span className="num">{format(value)}</span></div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Controls({ params, dispatch }: { params: Params; dispatch: Dispatch<Action> }) {
  const set = (patch: Partial<Params>) => dispatch({ type: 'set', patch });
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard can be blocked; the URL bar still has the link */
    }
  };
  return (
    <div className="rail-inner">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 600 }}>Model</span>
        <button className="btn sm" onClick={() => dispatch({ type: 'reset' })}>Reset to defaults</button>
      </div>
      <Slider id="ms" label="Mean service time" value={params.meanService} min={2} max={20} step={0.5}
        format={(v) => `${v} min`} onChange={(v) => set({ meanService: v })} />
      <div className="field">
        <label htmlFor="dist" style={{ fontSize: 13 }}>Service distribution</label>
        <select id="dist" value={params.serviceDist} onChange={(e) => set({ serviceDist: e.target.value as ServiceDist })}>
          <option value="exp">Exponential (CV 1)</option>
          <option value="lognormal">Lognormal</option>
          <option value="det">Deterministic (CV 0)</option>
        </select>
      </div>
      {params.serviceDist === 'lognormal' && (
        <Slider id="cv" label="Service CV" value={params.serviceCv} min={0.25} max={2} step={0.05}
          format={(v) => v.toFixed(2)} onChange={(v) => set({ serviceCv: v })} />
      )}
      <Slider id="mult" label="Demand × forecast" value={params.demandMult} min={0.6} max={1.5} step={0.05}
        format={(v) => `${v.toFixed(2)}×`} onChange={(v) => set({ demandMult: v })} />
      <Slider id="dcv" label="Day-to-day demand CV" value={params.rateCv} min={0} max={0.4} step={0.05}
        format={(v) => v.toFixed(2)} onChange={(v) => set({ rateCv: v })} />
      <Slider id="thr" label="Wait threshold T" value={params.threshold} min={5} max={30} step={1}
        format={(v) => `${v} min`} onChange={(v) => set({ threshold: v })} />
      <Slider id="alpha" label="Allowed late share α" value={params.alpha} min={0.02} max={0.3} step={0.01}
        format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ alpha: v })} />
      <div className="divider" />
      <span style={{ fontWeight: 600 }}>Appointments</span>
      <Slider id="appt" label="Share of demand booked" value={params.apptShare} min={0} max={0.75} step={0.05}
        format={(v) => (v ? `${Math.round(v * 100)}%` : 'Walk-ins only')} onChange={(v) => set({ apptShare: v })} />
      {params.apptShare > 0 && (
        <>
          <div className="field">
            <label htmlFor="place" style={{ fontSize: 13 }}>Where bookings go</label>
            <select id="place" value={params.placement} onChange={(e) => set({ placement: e.target.value as Placement })}>
              <option value="counter">Quiet hours first (counter-cyclical)</option>
              <option value="flat">Evenly across the day</option>
              <option value="proportional">Same shape as demand</option>
            </select>
          </div>
          <Slider id="ns" label="No-show rate" value={params.noShow} min={0} max={0.3} step={0.05}
            format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ noShow: v })} />
          <Slider id="punct" label="Punctuality SD" value={params.punctualitySd} min={0} max={15} step={1}
            format={(v) => `${v} min`} onChange={(v) => set({ punctualitySd: v })} />
          <div className="card-sub" style={{ lineHeight: 1.5 }}>
            Slots are overbooked by 1/(1 − no-show) so expected arrivals stay the same; walk-ins shrink by the booked share.
          </div>
        </>
      )}
      <div className="divider" />
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 10 }}>
        <div className="field">
          <label htmlFor="seed" style={{ fontSize: 13 }}>Seed</label>
          <input id="seed" className="num" type="number" value={params.seed}
            onChange={(e) => Number.isFinite(e.target.valueAsNumber) && set({ seed: Math.round(e.target.valueAsNumber) })} />
        </div>
        <div className="field">
          <label htmlFor="days" style={{ fontSize: 13 }}>Days simulated</label>
          <select id="days" className="num" value={params.days} onChange={(e) => set({ days: Number(e.target.value) })}>
            {[100, 300, 1000, 3000, 10000].map((d) => <option key={d} value={d}>{d.toLocaleString()}</option>)}
          </select>
        </div>
      </div>
      <button className="btn" onClick={copy}>{copied ? 'Link copied' : 'Copy link to this view'}</button>
      <div className="card-sub" style={{ lineHeight: 1.5 }}>
        Erlang-C numbers update as you drag. Simulated numbers rerun in the background; the thin bar under the
        header shows progress. Seeds are shared across plans (common random numbers).
      </div>
    </div>
  );
}
