import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { analyticPlans, erlangC, mmcMeanWait, probWaitExceeds } from './analytic';
import { DEFAULT_ARRIVALS, DEFAULT_PLAN, makeConfig } from './model';
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
});
