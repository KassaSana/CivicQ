"""
Experiments for the CivicQ staffing study. Every result is written to
research/results/*.csv and is reproducible from fixed seeds.

    python research/experiments.py --all
    python research/experiments.py e1b e1      # or any subset: e1b e1 e2 e2b e3a e3b

Design seeds (DESIGN_SEED..) choose plans; evaluation seeds (EVAL_SEED..) score
them, so no reported number is biased by the search that produced the plan.
"""

import argparse
import csv
import itertools
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import (  # noqa: E402
    DESIGN_REPS, DESIGN_SEED, EVAL_REPS, EVAL_SEED, OFFICE_RATES, SLOTS,
    analytic_plans, arrival_profile, evaluate, simulation_staffing,
)
from optimizer import run_simulation  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
VALID_SEED = 200_000   # Third seed block: validates a screened choice before it is scored
THRESHOLD = 15.0     # minutes
ALPHA = 0.10         # P(W > THRESHOLD | arrival hour) target
OFFICE_S = 8.0
# optimizer.py's cost-weighted choice when E2 was designed; with 300-day
# confirmation it is one of several statistically tied 20-21 h plans
RECOMMENDED = [2, 3, 3, 2, 2, 3, 3, 2]

SERVICE_TIMES = [4.0, 8.0, 16.0, 32.0]
AMPLITUDES = [0.3, 0.6]
MEAN_LOADS = [2.0, 8.0, 24.0]


def write_csv(name: str, rows: list[dict]):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(RESULTS.parent.parent)} ({len(rows)} rows)")


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def feasible_design(plan, rates, s, **kw):
    ev = evaluate(plan, rates, s, THRESHOLD, reps=DESIGN_REPS, seed=DESIGN_SEED, **kw)
    return all(p <= ALPHA for p in ev.late_prob)


# ============================================================================
# E1b: is simulation greedy staffing (SGS) optimal?
# ============================================================================

def exhaustive_check(rates, s, sgs_plan, max_candidates=20000):
    """
    Search every plan cheaper than SGS for one that meets the per-hour target.

    Per-hour lower bounds L_i come from plans with every other hour generously
    staffed (more servers elsewhere cannot make hour i worse under FIFO), so any
    feasible plan has s_i >= L_i. Returns a dict describing the check.
    """
    generous = [x + 3 for x in sgs_plan]
    lower = []
    for i in range(SLOTS):
        for c in range(1, sgs_plan[i] + 1):
            trial = list(generous)
            trial[i] = c
            ev = evaluate(trial, rates, s, THRESHOLD, reps=DESIGN_REPS, seed=DESIGN_SEED)
            if ev.late_prob[i] <= ALPHA:
                lower.append(c)
                break
        else:
            lower.append(sgs_plan[i])
    budget = sum(sgs_plan) - 1 - sum(lower)
    if budget < 0:
        return {"lower_bounds": lower, "candidates": 0, "cheaper_feasible": 0,
                "status": "SGS equals the lower bound"}
    candidates = [
        [l + d for l, d in zip(lower, deltas)]
        for deltas in itertools.product(range(budget + 1), repeat=SLOTS)
        if sum(deltas) <= budget
    ]
    if len(candidates) > max_candidates:
        return {"lower_bounds": lower, "candidates": len(candidates), "cheaper_feasible": None,
                "status": "skipped (too many candidates)"}
    ok = pmap(lambda p: feasible_design(p, rates, s), candidates)
    cheaper = [p for p, good in zip(candidates, ok) if good]
    return {"lower_bounds": lower, "candidates": len(candidates),
            "cheaper_feasible": len(cheaper),
            "status": "SGS optimal" if not cheaper else f"SGS suboptimal, e.g. {cheaper[0]}"}


def run_e1b():
    print("E1b: optimality of SGS (exhaustive search below the SGS cost)")
    cases = [("office", OFFICE_RATES, OFFICE_S)]
    for s in SERVICE_TIMES:
        for a in AMPLITUDES:
            cases.append((f"R2_S{s:g}_A{a}", arrival_profile(2.0, s, a), s))
    rows = []
    for name, rates, s in cases:
        plan, calls = simulation_staffing(rates, s, THRESHOLD, ALPHA)
        check = exhaustive_check(rates, s, plan)
        rows.append({"case": name, "service_time": s, "sgs_plan": json.dumps(plan),
                     "sgs_staff_hours": sum(plan), "sgs_calls": calls,
                     "lower_bounds": json.dumps(check["lower_bounds"]),
                     "candidates_checked": check["candidates"],
                     "cheaper_feasible": check["cheaper_feasible"],
                     "status": check["status"]})
        print(f"  {name:<16} SGS {plan} ({sum(plan)} h) -> {check['status']} "
              f"[{check['candidates']} candidates]")
    write_csv("e1b_sgs_optimality.csv", rows)


# ============================================================================
# E1: analytical rules vs simulation staffing across a factorial design
# ============================================================================

def run_e1():
    print("E1: SIPP / Lag-SIPP / OL-avg / OL-max vs SGS, "
          f"{len(SERVICE_TIMES) * len(AMPLITUDES) * len(MEAN_LOADS)} settings")
    rows = []
    for load, s, amp in itertools.product(MEAN_LOADS, SERVICE_TIMES, AMPLITUDES):
        rates = arrival_profile(load, s, amp)
        plans = analytic_plans(rates, s, THRESHOLD, ALPHA)
        plans["SGS"], _ = simulation_staffing(rates, s, THRESHOLD, ALPHA)
        plans["SGS-UCB"], _ = simulation_staffing(rates, s, THRESHOLD, ALPHA,
                                                  start=plans["SGS"], criterion="ucb")
        names = list(plans)
        evals = pmap(lambda n: evaluate(plans[n], rates, s, THRESHOLD), names)
        sgs_hours = sum(plans["SGS"])
        ucb_hours = sum(plans["SGS-UCB"])
        for name, ev in zip(names, evals):
            rows.append({
                "mean_load": load, "service_time": s, "amplitude": amp, "method": name,
                "plan": json.dumps(plans[name]), "staff_hours": ev.staff_hours,
                "gap_vs_sgs": ev.staff_hours - sgs_hours,
                "gap_vs_sgs_ucb": ev.staff_hours - ucb_hours,
                "diff_by_hour": json.dumps([a - b for a, b in zip(plans[name], plans["SGS"])]),
                "worst_late": round(max(ev.late_prob), 4),
                "worst_hour": ev.worst_hour(),
                "hours_over_alpha": sum(p > ALPHA for p in ev.late_prob),
                "hours_significantly_over": ev.hours_missing(ALPHA),
                "late_by_hour": json.dumps([round(p, 4) for p in ev.late_prob]),
                "overall_late": round(ev.overall_late, 4),
                "mean_wait": round(ev.mean_wait, 3),
            })
        summary = ", ".join(f"{n} {sum(plans[n])}h/{max(e.late_prob):.2f}"
                            for n, e in zip(names, evals))
        print(f"  R={load:<4g} S={s:<4g} A={amp}: {summary}")
    write_csv("e1_methods.csv", rows)


# ============================================================================
# E2: sensitivity to service-time variability and demand uncertainty
# ============================================================================

def run_e2():
    print("E2: service-time CV x daily demand CV (office)")
    service_models = [("exp", 1.0), ("lognormal", 0.5), ("lognormal", 1.0), ("lognormal", 1.5)]
    rate_cvs = [0.0, 0.1, 0.2]
    rows = []
    for (dist, cv), rcv in itertools.product(service_models, rate_cvs):
        kw = {"service_dist": dist, "service_cv": cv, "rate_cv": rcv}
        plan, _ = simulation_staffing(OFFICE_RATES, OFFICE_S, THRESHOLD, ALPHA, **kw)
        ev_sgs = evaluate(plan, OFFICE_RATES, OFFICE_S, THRESHOLD, **kw)
        ev_rec = evaluate(RECOMMENDED, OFFICE_RATES, OFFICE_S, THRESHOLD, **kw)
        rows.append({
            "service_dist": dist, "service_cv": cv, "rate_cv": rcv,
            "sgs_plan": json.dumps(plan), "sgs_staff_hours": sum(plan),
            "sgs_worst_late": round(max(ev_sgs.late_prob), 4),
            "rec_worst_late": round(max(ev_rec.late_prob), 4),
            "rec_worst_hour": ev_rec.worst_hour(),
            "rec_overall_late": round(ev_rec.overall_late, 4),
            "rec_mean_daily_p90": round(ev_rec.mean_daily_p90, 2),
            "rec_frac_days_p90_ok": round(ev_rec.frac_days_p90_ok, 3),
        })
        print(f"  {dist:<9} cv={cv:<4g} rate_cv={rcv:<4g}: SGS {plan} ({sum(plan)} h) | "
              f"recommended plan worst hour {max(ev_rec.late_prob):.3f}, "
              f"days meeting P90 {ev_rec.frac_days_p90_ok:.0%}")
    write_csv("e2_sensitivity.csv", rows)


def run_e2b():
    print("E2b: daily demand CV x office size (S = 8, A = 0.6, exponential service)")
    rows = []
    for load, rcv in itertools.product(MEAN_LOADS, [0.0, 0.1, 0.2]):
        rates = arrival_profile(load, OFFICE_S, 0.6)
        plan, _ = simulation_staffing(rates, OFFICE_S, THRESHOLD, ALPHA, rate_cv=rcv)
        ev = evaluate(plan, rates, OFFICE_S, THRESHOLD, rate_cv=rcv)
        rows.append({"mean_load": load, "rate_cv": rcv, "sgs_plan": json.dumps(plan),
                     "sgs_staff_hours": sum(plan),
                     "eval_worst_late": round(max(ev.late_prob), 4)})
    for row in rows:
        base = next(r["sgs_staff_hours"] for r in rows
                    if r["mean_load"] == row["mean_load"] and r["rate_cv"] == 0.0)
        row["extra_staff_hours"] = row["sgs_staff_hours"] - base
        row["extra_pct"] = round(100.0 * row["extra_staff_hours"] / base, 1)
        print(f"  R={row['mean_load']:<4g} rate_cv={row['rate_cv']:<4g}: "
              f"{row['sgs_staff_hours']} h (+{row['extra_staff_hours']}, "
              f"{row['extra_pct']:+.1f}%)")
    write_csv("e2b_scale.csv", rows)


# ============================================================================
# E3a: variance reduction from common random numbers
# ============================================================================

def run_e3a(pairs=20, reps=300):
    print("E3a: common random numbers, variance of paired differences")
    rng = random.Random(7)
    rows = []
    for k in range(pairs):
        base = [rng.choice([2, 3, 4]) for _ in range(SLOTS)]
        other = list(base)
        other[rng.randrange(SLOTS)] += 1
        a = run_simulation(base, replications=reps, seed=DESIGN_SEED)
        b_crn = run_simulation(other, replications=reps, seed=DESIGN_SEED)
        b_ind = run_simulation(other, replications=reps, seed=DESIGN_SEED + 50_000)
        for metric, get in [("mean_wait", lambda r: r.daily_mean_waits),
                            ("p90_wait", lambda r: r.daily_p90s)]:
            d_crn = np.array(get(a)) - np.array(get(b_crn))
            d_ind = np.array(get(a)) - np.array(get(b_ind))
            rows.append({"pair": k, "base": json.dumps(base), "other": json.dumps(other),
                         "metric": metric,
                         "var_crn": round(float(d_crn.var(ddof=1)), 4),
                         "var_independent": round(float(d_ind.var(ddof=1)), 4),
                         "variance_ratio": round(float(d_ind.var(ddof=1) / d_crn.var(ddof=1)), 2),
                         "mean_diff": round(float(d_crn.mean()), 4)})
    for metric in ("mean_wait", "p90_wait"):
        ratios = [r["variance_ratio"] for r in rows if r["metric"] == metric]
        print(f"  {metric}: variance ratio median {np.median(ratios):.1f} "
              f"(min {min(ratios):.1f}, max {max(ratios):.1f})")
    write_csv("e3a_crn.csv", rows)


# ============================================================================
# E3b: does the service-level definition change the recommendation?
# ============================================================================

DEFINITIONS = {
    "D1 mean daily P90 <= 15": lambda ev: ev.mean_daily_p90 <= THRESHOLD,
    "D2 pooled late <= 10%": lambda ev: ev.overall_late <= ALPHA,
    "D3 >= 90% of days P90 <= 15": lambda ev: ev.frac_days_p90_ok >= 0.90,
    "D4 every hour late <= 10%": lambda ev: all(p <= ALPHA for p in ev.late_prob),
}


def run_e3b(reps=200):
    print("E3b: cheapest plan under four service-level definitions (all 6,561 plans)")
    plans = [list(p) for p in itertools.product([2, 3, 4], repeat=SLOTS)]
    t0 = time.time()
    evals = pmap(lambda p: evaluate(p, OFFICE_RATES, OFFICE_S, THRESHOLD,
                                    reps=reps, seed=DESIGN_SEED), plans)
    print(f"  screened {len(plans)} plans in {time.time() - t0:.0f}s")
    rows = []
    for name, ok in DEFINITIONS.items():
        feasible = sorted((e for e in evals if ok(e)), key=lambda e: (e.staff_hours, e.mean_wait))
        naive = feasible[0]
        naive_holds = ok(evaluate(naive.staffing, OFFICE_RATES, OFFICE_S, THRESHOLD))
        # Walk up the screened ranking until a plan also passes on an independent
        # validation block; the cheapest screen winner is biased toward lucky plans
        rejected = 0
        for cand in feasible:
            if ok(evaluate(cand.staffing, OFFICE_RATES, OFFICE_S, THRESHOLD,
                           reps=EVAL_REPS, seed=VALID_SEED)):
                best = cand
                break
            rejected += 1
        check = evaluate(best.staffing, OFFICE_RATES, OFFICE_S, THRESHOLD)
        rows.append({
            "definition": name,
            "naive_plan": json.dumps(list(naive.staffing)),
            "naive_staff_hours": naive.staff_hours,
            "naive_holds_on_eval": naive_holds,
            "rejected_by_validation": rejected,
            "plan": json.dumps(list(best.staffing)),
            "staff_hours": best.staff_hours, "n_feasible": len(feasible),
            "eval_meets_definition": ok(check),
            "eval_mean_daily_p90": round(check.mean_daily_p90, 2),
            "eval_overall_late": round(check.overall_late, 4),
            "eval_frac_days_p90_ok": round(check.frac_days_p90_ok, 3),
            "eval_worst_hour_late": round(max(check.late_prob), 4),
        })
        print(f"  {name:<30} naive {list(naive.staffing)} {naive.staff_hours} h "
              f"(holds: {naive_holds}) -> validated {list(best.staffing)} "
              f"{best.staff_hours} h (holds: {ok(check)}, {rejected} rejected)")
    write_csv("e3b_definitions.csv", rows)


EXPERIMENTS = {"e1b": run_e1b, "e1": run_e1, "e2": run_e2, "e2b": run_e2b,
               "e3a": run_e3a, "e3b": run_e3b}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", choices=[*EXPERIMENTS, []], default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    names = list(EXPERIMENTS) if args.all else args.names
    if not names:
        parser.error("name at least one experiment or pass --all")
    for name in names:
        t0 = time.time()
        EXPERIMENTS[name]()
        print(f"  ({name} took {time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    main()
