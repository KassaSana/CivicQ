import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { analyticPlans, erlangC, mmcMeanWait, probWaitExceeds } from './analytic';
import { DEFAULT_ARRIVALS, DEFAULT_PLAN, appointmentBook, expectedRates, makeConfig, sum } from './model';
import { simulateDay } from './simulate';
import { meanCi, runDays, tCritical95 } from './stats';

describe('Erlang-C (python/test_validation.py cases)', () => {
  it('matches known values', () => {
    expect(erlangC(1, 0.5)).toBeCloseTo(0.5, 12);
    expect(erlangC(2, 1)).toBeCloseTo(1 / 3, 12);
    expect(erlangC(3, 2)).toBeCloseTo(4 / 9, 12);
  });
  it('is 1 when unstable', () => {
    expect(erlangC(2, 2)).toBe(1);
    expect(mmcMeanWait(2, 15, 8)).toBe(Infinity);
  });
  it('gives the steady-state 8–9 AM late share quoted in the design (33.6%)', () => {
    expect(probWaitExceeds(2, 1.6, 8, 15)).toBeCloseTo(0.3359, 3);
  });
  it('reproduces SIPP staffing from the README scenario analysis', () => {
    const plans = analyticPlans(DEFAULT_ARRIVALS, 8, 15, 0.1, 2);
    expect(plans.SIPP).toEqual([3, 3, 3, 2, 2, 3, 3, 3]);
  });
  it('uses small-sample t values', () => {
    expect(tCritical95(2)).toBe(4.303);
    const { mean, ci } = meanCi([1, 2, 3]);
    expect(mean).toBe(2);
    expect(ci[1] - 2).toBeCloseTo(4.303 / Math.sqrt(3), 9);
  });
});

describe('simulator', () => {
  it('is deterministic for a seed', () => {
    const cfg = makeConfig();
    const a = simulateDay(cfg, 123), b = simulateDay(cfg, 123);
    expect(a.waits).toEqual(b.waits);
    expect(a.overtime).toBe(b.overtime);
  });

  it('uses common random numbers: same citizens under every plan', () => {
    const a = simulateDay(makeConfig({ plan: [2, 2, 2, 2, 2, 2, 2, 2] }), 7);
    const b = simulateDay(makeConfig({ plan: [4, 4, 4, 4, 4, 4, 4, 4] }), 7);
    expect(a.citizens.map((c) => c.arrival)).toEqual(b.citizens.map((c) => c.arrival));
    expect(a.citizens.map((c) => c.service)).toEqual(b.citizens.map((c) => c.service));
    expect(b.meanWait).toBeLessThanOrEqual(a.meanWait);
  });

  it('serves everyone who arrived, FIFO', () => {
    const d = simulateDay(makeConfig(), 42);
    for (const c of d.citizens) expect(c.departure).toBeGreaterThan(c.arrival);
    for (let i = 1; i < d.citizens.length; i++) {
      expect(d.citizens[i].start).toBeGreaterThanOrEqual(d.citizens[i - 1].start);
    }
  });

  it('matches Erlang-C under constant load (M/M/3, a = 2)', () => {
    // Long horizon with constant rate so the start-up transient is negligible
    const cfg = makeConfig({ plan: new Array(8).fill(3), arrivals: new Array(8).fill(15), duration: 20000 });
    const exact = mmcMeanWait(3, 15, 8);
    const ms: number[] = [];
    for (let s = 0; s < 30; s++) ms.push(simulateDay(cfg, 1000 + s).meanWait);
    const { ci } = meanCi(ms);
    expect(exact).toBeGreaterThan(ci[0]);
    expect(exact).toBeLessThan(ci[1]);
  });

  it('agrees with research/staffing_methods.evaluate for the default plan', () => {
    // Reference: evaluate([2,3,3,2,2,3,3,3], OFFICE_RATES, 8.0), 1,000 days
    const ref = [0.0872, 0.067, 0.0285, 0.0367, 0.0522, 0.015, 0.0234, 0.0136];
    const refHalfWidth = [0.0128, 0.0127, 0.0078, 0.0083, 0.011, 0.0057, 0.0067, 0.0057];
    const agg = runDays(makeConfig({ plan: DEFAULT_PLAN }), 1000, 100000);
    for (let h = 0; h < 8; h++) {
      const ours = (agg.lateCi[h][1] - agg.lateCi[h][0]) / 2;
      // Two independent estimates: allow 3 combined standard errors
      const tol = 3 * Math.hypot(ours, refHalfWidth[h]) / 1.96;
      expect(Math.abs(agg.lateByHour[h] - ref[h])).toBeLessThan(tol);
    }
    expect(agg.meanWait).toBeGreaterThan(2.1);
    expect(agg.meanWait).toBeLessThan(2.6);
  });
});

describe('appointments', () => {
  const slotCounts = (times: number[]) => {
    const c = new Array(8).fill(0);
    for (const t of times) c[Math.floor(t / 60)]++;
    return c;
  };

  it('books like research/experiments.appointment_book', () => {
    // Reference: appointment_book(OFFICE_RATES, 0.5, placement, 0.15)
    const ref = {
      proportional: [7, 9, 6, 5, 4, 7, 8, 6],
      flat: [7, 7, 7, 7, 6, 6, 6, 6],
      counter: [6, 4, 7, 9, 8, 6, 5, 7],
    } as const;
    for (const [placement, counts] of Object.entries(ref)) {
      const { walk, times } = appointmentBook(DEFAULT_ARRIVALS, 0.5, placement as keyof typeof ref, 0.15);
      expect(slotCounts(times)).toEqual(counts);
      expect(walk).toEqual(DEFAULT_ARRIVALS.map((r) => r / 2));
    }
    expect(appointmentBook(DEFAULT_ARRIVALS, 0.5, 'counter', 0.15).times.slice(0, 2)).toEqual([5, 15]);
  });

  it('keeps expected arrivals unchanged', () => {
    const { walk, times } = appointmentBook(DEFAULT_ARRIVALS, 0.5, 'counter', 0.15);
    const rates = expectedRates(makeConfig({ arrivals: walk, appointments: times, noShow: 0.15 }));
    expect(Math.abs(sum(rates) - sum(DEFAULT_ARRIVALS))).toBeLessThan(1);
  });

  it('gives zero waits for perfectly spaced bookings with fixed service', () => {
    const times = Array.from({ length: 48 }, (_, k) => 10 * k);
    const d = simulateDay(makeConfig({
      plan: new Array(8).fill(1), arrivals: new Array(8).fill(0), serviceDist: 'det', appointments: times,
    }), 3);
    expect(d.citizens.length).toBe(48);
    expect(d.apptArrived).toBe(48);
    expect(Math.max(...d.waits)).toBe(0);
  });

  it('shows up at the rate 1 - no-show', () => {
    const times = Array.from({ length: 40 }, (_, k) => 12 * k);
    const cfg = makeConfig({ arrivals: new Array(8).fill(0), appointments: times, noShow: 0.2, punctualitySd: 5 });
    let shown = 0;
    for (let s = 0; s < 500; s++) shown += simulateDay(cfg, s).apptArrived;
    expect(Math.abs(shown / (500 * 40) - 0.8)).toBeLessThan(0.01);
  });

  it('leaves the walk-in stream untouched (common random numbers)', () => {
    const base = simulateDay(makeConfig(), 11);
    const withAppts = simulateDay(makeConfig({ appointments: [30, 90, 150], noShow: 0.1, punctualitySd: 5 }), 11);
    const walkins = withAppts.citizens.filter((c) => !c.booked).map((c) => c.arrival);
    expect(walkins).toEqual(base.citizens.map((c) => c.arrival));
    expect(base.apptArrived).toBe(0);
  });

  it('aggregates booked and walk-in waits separately', () => {
    // Proportional placement keeps the plan's fit, so booked citizens (evenly spaced) wait less:
    // research/REPORT.md §5.7 finding 4
    const { walk, times } = appointmentBook(DEFAULT_ARRIVALS, 0.5, 'proportional', 0.15);
    const agg = runDays(makeConfig({ arrivals: walk, appointments: times, noShow: 0.15, punctualitySd: 5 }), 1000, 1);
    expect(agg.apptPerDay).toBeGreaterThan(40);
    expect(agg.apptLate!).toBeLessThan(agg.walkinLate!);
    expect(agg.apptMeanWait!).toBeLessThan(agg.walkinMeanWait!);
    const blended = (agg.apptMeanWait! * agg.apptPerDay + agg.walkinMeanWait! * (agg.arrivalsPerDay - agg.apptPerDay)) / agg.arrivalsPerDay;
    // Pooled mean vs mean of daily means: close, not identical
    expect(Math.abs(blended - agg.meanWait)).toBeLessThan(0.1);
  });
});

describe('abandonment', () => {
  const thin = [2, 2, 2, 2, 2, 2, 2, 2];
  const renege = makeConfig({ plan: thin, abandonment: 'renege', meanPatience: 20 });
  const balk = makeConfig({ plan: thin, abandonment: 'balk', meanPatience: 20 });

  it("'none' is the model without abandonment", () => {
    const a = simulateDay(makeConfig(), 11), b = simulateDay(makeConfig({ abandonment: 'none', meanPatience: 5 }), 11);
    expect(b.waits).toEqual(a.waits);
    expect(b.citizens.every((c) => !c.abandoned && c.patience === 0)).toBe(true);
  });

  it('keeps arrivals, service and patience aligned across modes and plans', () => {
    const a = simulateDay(renege, 3), b = simulateDay(balk, 3), c = simulateDay({ ...renege, plan: [4, 4, 4, 4, 4, 4, 4, 4] }, 3);
    const none = simulateDay({ ...renege, abandonment: 'none' }, 3);
    for (const d of [b, c, none]) {
      expect(d.citizens.map((x) => x.arrival)).toEqual(a.citizens.map((x) => x.arrival));
      expect(d.citizens.map((x) => x.service)).toEqual(a.citizens.map((x) => x.service));
    }
    expect(b.citizens.map((x) => x.patience)).toEqual(a.citizens.map((x) => x.patience));
    expect(c.citizens.map((x) => x.patience)).toEqual(a.citizens.map((x) => x.patience));
  });

  it('reneging citizens leave exactly when patience runs out, and the served waited less', () => {
    let left = 0;
    for (let seed = 1; seed <= 20; seed++) {
      for (const c of simulateDay(renege, seed).citizens) {
        if (c.abandoned) {
          left++;
          expect(c.start).toBe(-1);
          expect(c.leave).toBeCloseTo(c.arrival + c.patience, 9);
        } else if (!c.booked) {
          expect(c.start - c.arrival).toBeLessThanOrEqual(c.patience + 1e-9);
        }
      }
    }
    expect(left).toBeGreaterThan(0);
  });

  it('balking citizens leave on arrival', () => {
    let left = 0;
    for (let seed = 1; seed <= 20; seed++) {
      for (const c of simulateDay(balk, seed).citizens) {
        if (!c.abandoned) continue;
        left++;
        expect(c.leave).toBe(c.arrival);
        expect(c.start).toBe(-1);
      }
    }
    expect(left).toBeGreaterThan(0);
  });

  it('never loses booked citizens and conserves everyone who came', () => {
    const { walk, times } = appointmentBook(DEFAULT_ARRIVALS, 0.5, 'counter', 0.15);
    for (const mode of ['renege', 'balk'] as const) {
      const cfg = makeConfig({ plan: thin, arrivals: walk, appointments: times, noShow: 0.15, abandonment: mode, meanPatience: 10 });
      for (let seed = 1; seed <= 10; seed++) {
        const d = simulateDay(cfg, seed);
        expect(d.citizens.some((c) => c.booked && c.abandoned)).toBe(false);
        expect(d.waits.length + sum(d.abandonedBySlot)).toBe(d.citizens.length);
        expect(sum(d.arrivalsBySlot)).toBe(d.waits.length);
      }
    }
  });

  it('loses fewer people with more windows, and failure is at least the served-late rate', () => {
    for (const cfg of [renege, balk]) {
      const few = runDays(cfg, 200, 1), many = runDays({ ...cfg, plan: [4, 4, 4, 4, 4, 4, 4, 4] }, 200, 1);
      expect(few.abandonedPerDay).toBeGreaterThan(many.abandonedPerDay);
      expect(few.overallFail).toBeGreaterThan(few.overallLate);
      expect(few.arrivalsPerDay).toBeCloseTo(few.servedPerDay + few.abandonedPerDay, 9);
    }
  });
});

// Cross-check against the C++ executable when it has been built
const exeCandidates = ['queue_sim.exe', 'queue_sim', 'Release/queue_sim.exe'].map((p) =>
  resolve(__dirname, '../../../cpp/build', p),
);
const exe = exeCandidates.find((p) => existsSync(p));

describe.skipIf(!exe)('cross-check with the C++ simulator', () => {
  const plans = [DEFAULT_PLAN, [3, 3, 3, 2, 2, 3, 3, 2], [2, 3, 2, 2, 2, 2, 3, 2], [3, 4, 3, 2, 3, 3, 3, 3]];
  for (const plan of plans) {
    it(`mean and P90 agree for [${plan}]`, () => {
      const csv = execFileSync(exe!, ['--staffing', plan.join(','), '--replications', '600', '--seed', '5000', '--per-replication'], {
        encoding: 'utf8',
      });
      const rows = csv.trim().split(/\r?\n/).slice(1).map((l) => l.split(',').map(Number));
      const cpp = { mean: meanCi(rows.map((r) => r[1])), p90: meanCi(rows.map((r) => r[2])) };
      const ts = runDays(makeConfig({ plan }), 600, 900000);
      for (const [a, b] of [
        [ts.meanWait, cpp.mean],
        [ts.p90, cpp.p90],
      ] as const) {
        const hw = (b.ci[1] - b.ci[0]) / 2;
        expect(Math.abs(a - b.mean)).toBeLessThan(3 * hw * Math.SQRT2 / 1.96 + 1e-9);
      }
    });
  }

  it('mean and P90 agree with 50% counter-cyclical appointments', () => {
    const { walk, times } = appointmentBook(DEFAULT_ARRIVALS, 0.5, 'counter', 0.15);
    const csv = execFileSync(exe!, [
      '--staffing', DEFAULT_PLAN.join(','), '--arrivals', walk.join(','), '--appointments', times.join(','),
      '--no-show', '0.15', '--punctuality-sd', '5', '--replications', '600', '--seed', '5000', '--per-replication',
    ], { encoding: 'utf8' });
    const rows = csv.trim().split(/\r?\n/).slice(1).map((l) => l.split(',').map(Number));
    const cpp = { mean: meanCi(rows.map((r) => r[1])), p90: meanCi(rows.map((r) => r[2])), appt: meanCi(rows.map((r) => r[32])) };
    const ts = runDays(makeConfig({ arrivals: walk, appointments: times, noShow: 0.15, punctualitySd: 5 }), 600, 900000);
    for (const [a, b] of [
      [ts.meanWait, cpp.mean],
      [ts.p90, cpp.p90],
      [ts.apptPerDay, cpp.appt],
    ] as const) {
      const hw = (b.ci[1] - b.ci[0]) / 2;
      expect(Math.abs(a - b.mean)).toBeLessThan(3 * hw * Math.SQRT2 / 1.96 + 1e-9);
    }
  });

  for (const mode of ['renege', 'balk'] as const) {
    for (const plan of [DEFAULT_PLAN, [2, 2, 2, 2, 2, 2, 2, 2]]) {
      it(`mean, P90 and walk-aways agree with ${mode} for [${plan}]`, () => {
        const csv = execFileSync(exe!, [
          '--staffing', plan.join(','), '--abandonment', mode, '--patience', '30',
          '--replications', '600', '--seed', '5000', '--per-replication',
        ], { encoding: 'utf8' });
        const [head, ...lines] = csv.trim().split(/\r?\n/);
        const col = Object.fromEntries(head.split(',').map((name, i) => [name, i]));
        const rows = lines.map((l) => l.split(',').map(Number));
        const aband = (r: number[]) => sum(Array.from({ length: 8 }, (_, j) => r[col[`aband_${j}`]]));
        const cpp = {
          mean: meanCi(rows.map((r) => r[col.mean_wait])),
          p90: meanCi(rows.map((r) => r[col.p90_wait])),
          left: meanCi(rows.map(aband)),
        };
        const ts = runDays(makeConfig({ plan, abandonment: mode, meanPatience: 30 }), 600, 900000);
        expect(cpp.left.mean).toBeGreaterThan(0);
        for (const [a, b] of [
          [ts.meanWait, cpp.mean],
          [ts.p90, cpp.p90],
          [ts.abandonedPerDay, cpp.left],
        ] as const) {
          const hw = (b.ci[1] - b.ci[0]) / 2;
          expect(Math.abs(a - b.mean)).toBeLessThan(3 * hw * Math.SQRT2 / 1.96 + 1e-9);
        }
      });
    }
  }
});
