"""
Experiments for the CivicQ staffing study. Every result is written to
research/results/*.csv and is reproducible from fixed seeds.

    python research/experiments.py --all
    python research/experiments.py e1b e1      # or any subset: e1b e1 e2 e2b e3a e3b e4 e5 e6 e7a e7 e7c e8a e8b e9a e9b

Design seeds (DESIGN_SEED..) choose plans; evaluation seeds (EVAL_SEED..) score
them, so no reported number is biased by the search that produced the plan.
"""

import argparse
import csv
import itertools
import json
import math
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


# ============================================================================
# E4: shift-feasible staffing, two-step vs integrated simulation search
# ============================================================================

def run_e4():
    from shifts import FLEXIBLE, STANDARD, Menu
    from staffing_methods import _feasible
    print("E4: two-step (requirement -> shift IP) vs integrated simulation search, "
          "standard and flexible shift menus")
    settings = [("office", OFFICE_RATES, OFFICE_S)]
    for load in MEAN_LOADS:
        for s in (8.0, 32.0):
            settings.append((f"R{load:g}_S{s:g}_A0.6", arrival_profile(load, s, 0.6), s))
    rows = []
    for name, rates, s in settings:
        reqs = {k: v for k, v in analytic_plans(rates, s, THRESHOLD, ALPHA).items()
                if k in ("SIPP", "OL-avg")}
        reqs["SGS-UCB"], _ = simulation_staffing(rates, s, THRESHOLD, ALPHA, criterion="ucb")
        standard_iss = None
        for menu_name, shifts in (("standard", STANDARD), ("flexible", FLEXIBLE)):
            menu = Menu(shifts)
            schedules = {f"{k}-IP": menu.cover_ip(r) for k, r in reqs.items()}

            # Integrated search starts from the cheapest two-step schedule that
            # is feasible on the design seeds. The flexible menu contains every
            # standard shift, so it also starts from the standard menu's result
            # (multi-start: local search alone can end above that point)
            starts = dict(schedules)
            if standard_iss is not None:
                names = [n for n, _, _ in shifts]
                starts["standard ISS"] = [0] * len(shifts)
                for shift_name, count in standard_iss.items():
                    starts["standard ISS"][names.index(shift_name)] = count
            feasible_starts = [
                (menu.paid_hours(x), k) for k, x in starts.items()
                if _feasible(menu.profile(x), rates, s, THRESHOLD, ALPHA,
                             DESIGN_REPS, DESIGN_SEED, "ucb")[0]
            ]
            start_name = min(feasible_starts)[1]
            schedules["ISS"], calls = menu.integrated_search(rates, s, THRESHOLD, ALPHA,
                                                             starts[start_name])
            if menu_name == "standard":
                standard_iss = menu.describe(schedules["ISS"])
            evals = pmap(lambda k: evaluate(menu.profile(schedules[k]), rates, s, THRESHOLD),
                         list(schedules))
            best_two_step = min(menu.paid_hours(x) for k, x in schedules.items() if k != "ISS")
            for (k, x), ev in zip(schedules.items(), evals):
                req = reqs.get(k.replace("-IP", ""))
                profile = menu.profile(x)
                rows.append({
                    "setting": name, "service_time": s, "menu": menu_name, "method": k,
                    "shifts": json.dumps(menu.describe(x)),
                    "profile": json.dumps(profile), "paid_hours": menu.paid_hours(x),
                    "requirement_hours": sum(req) if req else "",
                    "surplus_window_hours": sum(p - r for p, r in zip(profile, req)) if req else "",
                    "saving_vs_best_two_step_pct": round(
                        100 * (best_two_step - menu.paid_hours(x)) / best_two_step, 1),
                    "worst_late": round(max(ev.late_prob), 4),
                    "hours_significantly_over": ev.hours_missing(ALPHA),
                    "iss_start": start_name if k == "ISS" else "",
                    "simulator_calls": calls if k == "ISS" else "",
                })
            print(f"  {name:<14} {menu_name:<9}" + ", ".join(
                f"{k} {menu.paid_hours(x)}h/{max(e.late_prob):.2f}"
                for (k, x), e in zip(schedules.items(), evals))
                + f"  (ISS from {start_name}, {calls} calls)")
    write_csv("e4_shifts.csv", rows)


# ============================================================================
# E6: appointments mixed with walk-ins
# ============================================================================

def appointment_book(rates, share, placement, no_show):
    """
    Move `share` of expected daily demand to appointments.

    Returns (walk-in hourly rates, booked times in minutes). Bookings are
    overbooked by 1/(1 - no_show) so expected shows equal the demand moved.
    Placement of expected shows per hour:
        proportional  same shape as demand
        flat          evenly across the day
        counter       water-filling into quiet hours so that walk-ins plus
                      expected shows are as flat as possible
    """
    walk = [r * (1 - share) for r in rates]
    moved = share * sum(rates)
    if placement == "proportional":
        shows = [share * r for r in rates]
    elif placement == "flat":
        shows = [moved / SLOTS] * SLOTS
    elif placement == "counter":
        lo, hi = min(walk), max(walk) + moved
        for _ in range(100):                      # bisection on the water level
            level = (lo + hi) / 2
            if sum(max(0.0, level - w) for w in walk) > moved:
                hi = level
            else:
                lo = level
        shows = [max(0.0, lo - w) for w in walk]
        shows = [s * moved / sum(shows) for s in shows]
    else:
        raise ValueError(placement)
    wanted = [s / (1 - no_show) for s in shows]
    total = round(sum(wanted))
    counts = [int(w) for w in wanted]            # largest-remainder rounding
    for i in sorted(range(SLOTS), key=lambda i: wanted[i] - counts[i], reverse=True):
        if sum(counts) >= total:
            break
        counts[i] += 1
    times = [60.0 * i + 60.0 * (k + 0.5) / c for i, c in enumerate(counts) for k in range(c)]
    return walk, times


def _class_split(r):
    """Walk-in vs appointment late rate and mean wait from per-day totals."""
    arr = np.array(r.daily_arrivals, float).sum(axis=1)
    late = np.array(r.daily_late, float).sum(axis=1)
    wait = np.array(r.daily_mean_waits) * arr
    a_arr = np.array(r.daily_appt_arrived, float)
    a_late = np.array(r.daily_appt_late, float)
    a_wait = np.array(r.daily_appt_wait_sum, float)
    w_arr = arr - a_arr
    return {
        "walkin_late": round(float((late - a_late).sum() / w_arr.sum()), 4),
        "walkin_mean_wait": round(float((wait - a_wait).sum() / w_arr.sum()), 3),
        "appt_late": round(float(a_late.sum() / a_arr.sum()), 4) if a_arr.sum() else "",
        "appt_mean_wait": round(float(a_wait.sum() / a_arr.sum()), 3) if a_arr.sum() else "",
        "appt_show_rate": "",
    }


def run_e6():
    from shifts import STANDARD, Menu
    from staffing_methods import _feasible
    print("E6: appointments + walk-ins (hourly optimum and standard-menu roster)")
    offices = [("office", OFFICE_RATES, OFFICE_S),
               ("R8_S16_A0.6", arrival_profile(8.0, 16.0, 0.6), 16.0)]
    cells = [(0.0, "none", 0.15)]
    cells += [(f, pl, 0.15) for f in (0.25, 0.5, 0.75)
              for pl in ("proportional", "flat", "counter")]
    cells += [(0.5, pl, p) for p in (0.05, 0.30) for pl in ("proportional", "counter")]
    menu = Menu(STANDARD)
    rows = []
    for name, rates, s in offices:
        for share, placement, p in cells:
            walk, times = appointment_book(rates, share, placement if share else "flat", p)
            kw = {"appointments": times, "no_show": p, "punctuality_sd": 5.0}
            hourly, _ = simulation_staffing(walk, s, THRESHOLD, ALPHA, criterion="ucb", **kw)
            k = 1
            while not _feasible(menu.profile(menu.full_days(k)), walk, s, THRESHOLD, ALPHA,
                                DESIGN_REPS, DESIGN_SEED, "ucb", **kw)[0]:
                k += 1
            roster, calls = menu.integrated_search(walk, s, THRESHOLD, ALPHA,
                                                   menu.full_days(k), **kw)
            r = run_simulation(menu.profile(roster), walk, replications=EVAL_REPS,
                               seed=EVAL_SEED, mean_service=s, wait_threshold=THRESHOLD, **kw)
            split = _class_split(r)
            if times:
                split["appt_show_rate"] = round(float(np.mean(r.daily_appt_arrived)) / len(times), 3)
            ev_hourly = evaluate(hourly, walk, s, THRESHOLD, **kw)
            rows.append({
                "office": name, "service_time": s, "share": share, "placement": placement,
                "no_show": p, "booked": len(times),
                "hourly_plan": json.dumps(hourly), "hourly_window_hours": sum(hourly),
                "hourly_worst_late": round(max(ev_hourly.late_prob), 4),
                "roster": json.dumps(menu.describe(roster)),
                "roster_paid_hours": menu.paid_hours(roster),
                "roster_worst_late": round(max(r.late_prob_per_slot), 4),
                "roster_overtime": round(r.avg_overtime, 1),
                **split,
            })
            print(f"  {name:<12} f={share:<4} {placement:<12} p={p:<4}: hourly {sum(hourly)} h, "
                  f"roster {menu.paid_hours(roster)} h | walk-in late {split['walkin_late']:.3f}"
                  + (f", appt late {split['appt_late']:.3f}" if times else ""))
    write_csv("e6_appointments.csv", rows)


# ============================================================================
# E5: cross-validation against an independent simulator (Ciw)
# ============================================================================

def _civicq_days(plan, rates, s, reps, dist="exp", cv=1.0):
    r = run_simulation(plan, rates, replications=reps, seed=EVAL_SEED, mean_service=s,
                       wait_threshold=THRESHOLD, service_dist=dist, service_cv=cv)
    return (np.array(r.daily_arrivals, float), np.array(r.daily_late, float),
            np.array(r.daily_mean_waits))


def run_e5(reps=2000):
    from scipy.stats import norm
    from crossval_ciw import ciw_days, ratio_and_se
    print(f"E5: CivicQ vs Ciw, {reps} independent days per simulator and case")
    big_rates = arrival_profile(24.0, 32.0, 0.6)
    big_plan = next(json.loads(r["plan"]) for r in csv.DictReader(open(RESULTS / "e1_methods.csv"))
                    if r["method"] == "SGS-UCB" and float(r["mean_load"]) == 24.0
                    and float(r["service_time"]) == 32.0 and float(r["amplitude"]) == 0.6)
    strict = [  # Constant staffing: the two models are exactly equivalent
        ("flat rates, 2 windows", [2] * 8, [11.125] * 8, 8.0, "exp", 1.0),
        ("office rates, 3 windows", [3] * 8, OFFICE_RATES, 8.0, "exp", 1.0),
        ("office, lognormal CV 0.5", [3] * 8, OFFICE_RATES, 8.0, "lognormal", 0.5),
        ("office, lognormal CV 1.5", [3] * 8, OFFICE_RATES, 8.0, "lognormal", 1.5),
        ("office, deterministic", [3] * 8, OFFICE_RATES, 8.0, "det", 1.0),
        ("24 Erlangs, S=32, 31 windows", [31] * 8, big_rates, 32.0, "exp", 1.0),
    ]
    bracket = [  # Staffing changes: Ciw's options bound CivicQ's semantics
        ("office plan [2,3,3,2,2,3,3,2]", RECOMMENDED, OFFICE_RATES, 8.0),
        ("office SIPP [3,3,3,2,2,3,3,3]", [3, 3, 3, 2, 2, 3, 3, 3], OFFICE_RATES, 8.0),
        ("burst [1,4,4,...] 30/h then 0", [1, 4, 4, 4, 4, 4, 4, 4], [30] + [0] * 7, 8.0),
        ("24 Erlangs, S=32, SGS-UCB plan", big_plan, big_rates, 32.0),
    ]
    rows = []
    for name, plan, rates, s, dist, cv in strict:
        a1, l1, w1 = _civicq_days(plan, rates, s, reps, dist, cv)
        a2, l2, w2 = ciw_days(plan, rates, s, THRESHOLD, reps, dist=dist, cv=cv)
        for i in range(SLOTS):
            if a1[:, i].sum() == 0:
                continue
            p1, se1 = ratio_and_se(l1[:, i], a1[:, i])
            p2, se2 = ratio_and_se(l2[:, i], a2[:, i])
            z = (p1 - p2) / math.hypot(se1, se2) if se1 + se2 > 0 else 0.0
            rows.append({"case": name, "kind": "strict", "metric": f"late_hour_{i}",
                         "civicq": round(p1, 5), "ciw": round(p2, 5), "ciw_upper": "",
                         "se_civicq": round(se1, 5), "se_ciw": round(se2, 5),
                         "z": round(z, 3), "p": 2 * norm.sf(abs(z))})
        se = math.sqrt(w1.var(ddof=1) / reps + w2.var(ddof=1) / reps)
        z = (w1.mean() - w2.mean()) / se
        rows.append({"case": name, "kind": "strict", "metric": "mean_wait",
                     "civicq": round(w1.mean(), 4), "ciw": round(w2.mean(), 4), "ciw_upper": "",
                     "se_civicq": round(w1.std(ddof=1) / math.sqrt(reps), 4),
                     "se_ciw": round(w2.std(ddof=1) / math.sqrt(reps), 4),
                     "z": round(z, 3), "p": 2 * norm.sf(abs(z))})

    # Holm step-down over the strict family
    strict_rows = sorted((r for r in rows if r["kind"] == "strict"), key=lambda r: r["p"])
    m = len(strict_rows)
    running = 0.0
    for k, r in enumerate(strict_rows):
        running = max(running, min(1.0, (m - k) * r["p"]))
        r["p_holm"] = round(running, 4)
        r["p"] = round(r["p"], 4)
    raw_rej = sum(r["p"] < 0.05 for r in strict_rows)
    holm_rej = sum(r["p_holm"] < 0.05 for r in strict_rows)
    print(f"  strict: {m} tests, {raw_rej} reject at 5% unadjusted "
          f"(expected by chance ~{0.05 * m:.1f}), {holm_rej} after Holm")

    inside = total = 0
    for name, plan, rates, s in bracket:
        a1, l1, _ = _civicq_days(plan, rates, s, reps)
        a2, l2, _ = ciw_days(plan, rates, s, THRESHOLD, reps, preemption=False)
        a3, l3, _ = ciw_days(plan, rates, s, THRESHOLD, reps, seed=700_000, preemption="resume")
        for i in range(SLOTS):
            if a1[:, i].sum() == 0:
                continue
            p1, se1 = ratio_and_se(l1[:, i], a1[:, i])
            lo, se_lo = ratio_and_se(l2[:, i], a2[:, i])
            hi, se_hi = ratio_and_se(l3[:, i], a3[:, i])
            ok = (lo - 1.96 * math.hypot(se1, se_lo) <= p1 <= hi + 1.96 * math.hypot(se1, se_hi))
            inside += ok
            total += 1
            rows.append({"case": name, "kind": "bracket", "metric": f"late_hour_{i}",
                         "civicq": round(p1, 5), "ciw": round(lo, 5), "ciw_upper": round(hi, 5),
                         "se_civicq": round(se1, 5), "se_ciw": round(se_lo, 5),
                         "z": "", "p": "", "p_holm": "", "within_bracket": ok})
    print(f"  bracket: CivicQ inside [Ciw non-preemptive, Ciw resume] in {inside}/{total} hours")
    for r in rows:
        r.setdefault("p_holm", "")
        r.setdefault("within_bracket", "")
    write_csv("e5_crossval.csv", rows)


# ============================================================================
# E7: walk-in abandonment (visible vs hidden queues, ticket-log metrics,
#     mandatory services)
# ============================================================================

PATIENCE = [("exp30", 30.0, "exp", 1.0), ("exp60", 60.0, "exp", 1.0),
            ("logn30", 30.0, "lognormal", 0.5)]
MODES = ["renege", "balk"]


def _e7_offices():
    return [("office", OFFICE_RATES, OFFICE_S),
            ("R8_S16_A0.6", arrival_profile(8.0, 16.0, 0.6), 16.0)]


def _patience_kw(mode, mean, dist, cv):
    return {"abandonment": mode, "patience": mean, "patience_dist": dist, "patience_cv": cv}


def run_e7a(reps=200):
    """Simulator vs exact stationary models under constant demand and staffing."""
    from abandonment import hour_metrics, score
    print("E7a: abandonment simulator vs exact stationary models")
    cases = [(3, 15.0, 30.0, "exp", 1.0), (2, 15.0, 30.0, "exp", 1.0),
             (2, 12.0, 60.0, "exp", 1.0), (4, 40.0, 20.0, "exp", 1.0),
             (2, 15.0, 30.0, "lognormal", 0.5), (3, 15.0, 30.0, "lognormal", 0.5)]
    rows = []
    for mode in MODES:
        for c, lam, pat, dist, cv in cases:
            if mode == "renege" and dist != "exp":
                continue          # Erlang-A is exact only for exponential patience
            exact = hour_metrics(mode, c, lam, 8.0, pat, THRESHOLD, dist, cv)
            ev = score([c] * SLOTS, [lam] * SLOTS, 8.0, THRESHOLD, reps=reps, seed=300_000,
                       duration=20000, **_patience_kw(mode, pat, dist, cv))
            row = {"mode": mode, "windows": c, "rate_per_hour": lam, "patience": pat,
                   "patience_dist": dist, "patience_cv": cv}
            for name, sim, ci, ex in [("fail", ev.fail[7], ev.fail_ci[7], exact.fail),
                                      ("served_late", ev.served_late[7], ev.served_late_ci[7],
                                       exact.served_late)]:
                se = (ci[1] - ci[0]) / (2 * 1.96)
                row.update({f"{name}_exact": round(ex, 5), f"{name}_sim": round(sim, 5),
                            f"{name}_ci_low": round(ci[0], 5), f"{name}_ci_high": round(ci[1], 5),
                            f"{name}_z": round((sim - ex) / se, 2) if se > 0 else 0.0})
            row.update({"abandon_exact": round(exact.abandon, 5),
                        "abandon_sim": round(ev.abandon[7], 5)})
            rows.append(row)
            print(f"  {mode:<6} c={c} lam={lam:g} {dist}{pat:g}: fail {ev.fail[7]:.4f} vs "
                  f"{exact.fail:.4f} (z={row['fail_z']}), served-late {ev.served_late[7]:.4f} "
                  f"vs {exact.served_late:.4f} (z={row['served_late_z']}), abandon "
                  f"{ev.abandon[7]:.4f} vs {exact.abandon:.4f}")
    write_csv("e7a_validation.csv", rows)


def _e7_exhaustive(rates, s, sgs_plan, metric, kw, cap=4):
    """
    Every plan in {1..cap}^8 cheaper than SGS, checked with the SGS-UCB rule.
    No lower bounds: the served-late metric is not monotone in staffing (extra
    windows serve impatient citizens who would otherwise have left).
    """
    budget = sum(sgs_plan) - 1
    candidates = [list(p) for p in itertools.product(range(1, cap + 1), repeat=SLOTS)
                  if sum(p) <= budget]

    def ok(p):
        ev = evaluate(p, rates, s, THRESHOLD, reps=DESIGN_REPS, seed=DESIGN_SEED,
                      metric=metric, **kw)
        return all(hi <= ALPHA for _, hi in ev.late_ci)

    good = [p for p, g in zip(candidates, pmap(ok, candidates, workers=16)) if g]
    return len(candidates), good


def run_e7():
    from abandonment import score, sipp_abandonment
    print("E7: staffing when walk-ins leave (served-late vs failure targets)")
    rows, exhaustive_rows = [], []
    for office, rates, s in _e7_offices():
        base, _ = simulation_staffing(rates, s, THRESHOLD, ALPHA, criterion="ucb")
        sipp_c = analytic_plans(rates, s, THRESHOLD, ALPHA)["SIPP"]
        for pname, mean, dist, cv in PATIENCE:
            plans = {("no-abandonment SGS-UCB", "-"): base, ("SIPP (Erlang-C)", "-"): sipp_c}
            for mode in MODES:
                kw = _patience_kw(mode, mean, dist, cv)
                plans[("SIPP-A (fail)", mode)] = sipp_abandonment(
                    rates, s, THRESHOLD, ALPHA, mode, mean, "fail", dist, cv)
                for metric in ("late", "fail"):
                    plan, _ = simulation_staffing(rates, s, THRESHOLD, ALPHA, criterion="ucb",
                                                  start=sipp_c, metric=metric, **kw)
                    plans[(f"SGS-UCB ({metric})", mode)] = plan
                    if office == "office" and pname == "exp30":
                        n, cheaper = _e7_exhaustive(rates, s, plan, metric, kw)
                        exhaustive_rows.append({
                            "mode": mode, "metric": metric, "sgs_plan": json.dumps(plan),
                            "sgs_staff_hours": sum(plan), "candidates": n,
                            "cheaper_feasible": len(cheaper),
                            "example": json.dumps(cheaper[0]) if cheaper else ""})
                        print(f"    exhaustive {mode}/{metric}: SGS {sum(plan)} h, "
                              f"{len(cheaper)} of {n} cheaper plans feasible")
            # Score every plan under both behaviours on the evaluation days
            unique = {tuple(p) for p in plans.values()}
            for plan in sorted(unique):
                labels = [f"{m}@{b}" for (m, b), p in plans.items() if tuple(p) == plan]
                for mode in MODES:
                    ev = score(list(plan), rates, s, THRESHOLD,
                               **_patience_kw(mode, mean, dist, cv))
                    rows.append({
                        "office": office, "service_time": s, "patience": pname,
                        "mode": mode, "plan": json.dumps(list(plan)),
                        "staff_hours": sum(plan), "found_by": "; ".join(labels),
                        "worst_served_late": round(ev.worst("late"), 4),
                        "served_late_misses": ev.misses("late", ALPHA),
                        "worst_fail": round(ev.worst("fail"), 4),
                        "fail_misses": ev.misses("fail", ALPHA),
                        "overall_fail": round(ev.overall_fail, 4),
                        "overall_abandon": round(ev.overall_abandon, 4),
                        "worst_hour_abandon": round(max(ev.abandon), 4),
                        "abandoned_per_day": round(ev.abandoned_per_day, 2),
                        "wasted_min_per_leaver": round(ev.wasted_minutes_per_day
                                                       / max(ev.abandoned_per_day, 1e-9), 2),
                        "mean_wait_served": round(ev.mean_wait_served, 3),
                        "served_late_by_hour": json.dumps([round(x, 4) for x in ev.served_late]),
                        "fail_by_hour": json.dumps([round(x, 4) for x in ev.fail]),
                    })
            found = {k: sum(v) for k, v in plans.items()}
            print(f"  {office:<12} {pname:<7} " + ", ".join(
                f"{m}@{b}={h}h" for (m, b), h in found.items()))
    write_csv("e7_abandonment.csv", rows)
    write_csv("e7_exhaustive.csv", exhaustive_rows)


def run_e7c():
    """Mandatory services: citizens who leave return on a later day (r = 1)."""
    from abandonment import return_fixed_point, score
    print("E7c: return visits (mandatory service, exponential patience, mean 30)")
    plans = {}
    with open(RESULTS / "e7_abandonment.csv") as f:
        for row in csv.DictReader(f):
            if row["patience"] != "exp30":
                continue
            for label in row["found_by"].split("; "):
                method, mode = label.split("@")
                if method.startswith("SGS-UCB") and mode == row["mode"]:
                    plans[(row["office"], mode, method)] = json.loads(row["plan"])
    offices = {name: (rates, s) for name, rates, s in _e7_offices()}
    rows = []
    for (office, mode, method), plan in sorted(plans.items()):
        rates, s = offices[office]
        kw = _patience_kw(mode, 30.0, "exp", 1.0)
        none = score(plan, rates, s, THRESHOLD, **kw)
        for timing in ("profile", "opening"):
            st = return_fixed_point(plan, rates, s, 1.0, timing, THRESHOLD,
                                    reps=EVAL_REPS, **kw)
            row = {"office": office, "mode": mode, "plan_from": method,
                   "plan": json.dumps(plan), "staff_hours": sum(plan), "timing": timing,
                   "stable": st.stable,
                   "no_return_abandon": round(none.overall_abandon, 4),
                   "no_return_worst_served_late": round(none.worst("late"), 4),
                   "no_return_fail_hour0": round(none.fail[0], 4)}
            if st.stable:
                ev = st.evaluation
                row.update({
                    "returns_per_day": round(st.returns_per_day, 2),
                    "repeat_visits_per_100": round(st.repeat_visits_per_100, 2),
                    "overall_abandon": round(ev.overall_abandon, 4),
                    "worst_served_late": round(ev.worst("late"), 4),
                    "served_late_hours_over": sum(1 for x in ev.served_late if x > ALPHA),
                    "served_late_misses": ev.misses("late", ALPHA),
                    "worst_fail": round(ev.worst("fail"), 4),
                    "fail_hour0": round(ev.fail[0], 4),
                    "fail_by_hour": json.dumps([round(x, 4) for x in ev.fail]),
                    "served_late_by_hour": json.dumps([round(x, 4) for x in ev.served_late]),
                })
            rows.append(row)
            if st.stable:
                print(f"  {office:<12} {mode:<6} {method:<16} {timing:<8}: "
                      f"{row['repeat_visits_per_100']:.1f} repeat visits/100, worst served-late "
                      f"{row['worst_served_late']:.3f} (was {row['no_return_worst_served_late']:.3f}), "
                      f"8AM fail {row['fail_hour0']:.3f} (was {row['no_return_fail_hour0']:.3f})")
            else:
                print(f"  {office:<12} {mode:<6} {method:<16} {timing:<8}: UNSTABLE")
    write_csv("e7c_returns.csv", rows)


# ============================================================================
# E8: when does abandonment raise or lower the staffing need? (regimes)
# ============================================================================

def _crossover_utilization(c, patience, s=8.0):
    """
    Utilization rho* at which abandonment (Erlang-A, exponential patience)
    stops raising the failure rate at a fixed c: below it early leavers
    outweigh queue thinning. Found by bisection on
    Delta(rho) = P(late or left) - Erlang-C P(W > T).
    """
    from abandonment import renege_metrics
    from staffing_methods import prob_wait_exceeds

    def delta(rho):
        load = rho * c
        return (renege_metrics(c, load / s, s, patience, THRESHOLD).fail
                - prob_wait_exceeds(c, load, s, THRESHOLD))

    lo, hi = 0.2, 0.999
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if delta(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def run_e8a():
    """Stationary per-hour regime maps (exact models, no simulation)."""
    from abandonment import (fluid_discount, required_windows,
                             required_windows_with_returns)
    from staffing_methods import prob_wait_exceeds
    print("E8a: stationary regimes of abandonment (S = 8, T = 15)")
    s = 8.0

    rows = []
    for patience in (15.0, 30.0, 60.0, 120.0):
        for c in (1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 80):
            rho = _crossover_utilization(c, patience, s)
            rows.append({"patience": patience, "windows": c, "rho_star": round(rho, 4),
                         "erlang_c_late_at_rho_star": round(prob_wait_exceeds(c, rho * c, s,
                                                                              THRESHOLD), 4),
                         "beta_star": round((1 - rho) * math.sqrt(c), 4)})
        print(f"  crossover, exp {patience:g}: rho* = "
              + ", ".join(f"{r['rho_star']:.2f}@c={r['windows']}" for r in rows[-12:][::3]))
    write_csv("e8a_crossover.csv", rows)

    loads = np.round(np.arange(0.1, 16.01, 0.1), 2)
    rows = []
    for patience in (30.0, 60.0):
        for alpha in (0.2, 0.1, 0.05, 0.02, 0.01):
            c_c = [required_windows(R / s * 60, s, THRESHOLD, alpha) for R in loads]
            c_a = [required_windows(R / s * 60, s, THRESHOLD, alpha, "renege", patience)
                   for R in loads]
            diff = np.array(c_a) - np.array(c_c)
            rows.append({"patience": patience, "alpha": alpha, "loads": len(loads),
                         "adds_window": int((diff > 0).sum()),
                         "removes_window": int((diff < 0).sum()),
                         "ties": int((diff == 0).sum())})
            print(f"  exp {patience:g}, alpha {alpha}: abandonment adds a window at "
                  f"{rows[-1]['adds_window']} loads, removes at {rows[-1]['removes_window']}")
    write_csv("e8a_alpha_sign.csv", rows)

    cases = [("renege", 30.0, "exp", 1.0), ("renege", 120.0, "exp", 1.0),
             ("balk", 30.0, "exp", 1.0), ("balk", 30.0, "lognormal", 0.5),
             ("balk", 60.0, "lognormal", 0.5)]
    rows = []
    for load in (2, 5, 10, 25, 50, 100, 200, 400):
        rate = load / s * 60
        row = {"load": load, "erlang_c": round(required_windows(rate, s, THRESHOLD, ALPHA) / load, 4)}
        for mode, patience, dist, cv in cases:
            if mode == "renege" and patience > 60 and load > 200:
                continue
            row[f"{mode}_{dist}{patience:g}"] = round(
                required_windows(rate, s, THRESHOLD, ALPHA, mode, patience, dist, cv) / load, 4)
        for r in (0.0, 0.5, 1.0):
            if load <= 200:
                c, _ = required_windows_with_returns(rate, s, THRESHOLD, ALPHA, 30.0, r)
                row[f"renege_exp30_returns{r:g}"] = round(c / load, 4)
        rows.append(row)
        print(f"  load {load:>3}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "load"))
    fluid = {"load": "fluid limit", "erlang_c": 1.0}
    for mode, patience, dist, cv in cases:
        fluid[f"{mode}_{dist}{patience:g}"] = round(1 - fluid_discount(THRESHOLD, ALPHA, patience,
                                                                       dist, cv), 4)
    d = fluid_discount(THRESHOLD, ALPHA, 30.0)
    for r in (0.0, 0.5, 1.0):
        fluid[f"renege_exp30_returns{r:g}"] = round((1 - d) / (1 - r * d), 4)
    rows.append(fluid)
    write_csv("e8a_fluid.csv", rows)


def run_e8b():
    """Time-varying tests of the regime predictions (H15-H17)."""
    from abandonment import score
    print("E8b: abandonment across office sizes and targets (time-varying, hidden queue)")
    s = 8.0
    patience = {"none": None, "exp30": (30.0, "exp", 1.0), "logn60": (60.0, "lognormal", 0.5)}

    def plan_for(rates, alpha, pname):
        if patience[pname] is None:
            plan, _ = simulation_staffing(rates, s, THRESHOLD, alpha, criterion="ucb")
            return plan
        mean, dist, cv = patience[pname]
        plan, _ = simulation_staffing(rates, s, THRESHOLD, alpha, criterion="ucb",
                                      metric="fail", **_patience_kw("renege", mean, dist, cv))
        return plan

    rows = []
    cells = [(f"R{load:g}", arrival_profile(load, s, 0.6), 0.10, load)
             for load in (1, 2, 4, 8, 16, 32)]
    cells += [(name, rates, alpha, load)
              for name, rates, load in (("office", OFFICE_RATES, 1.5),
                                        ("R8", arrival_profile(8.0, s, 0.6), 8.0))
              for alpha in (0.02, 0.20)]
    for name, rates, alpha, load in cells:
        base = plan_for(rates, alpha, "none")
        for pname in ("exp30", "logn60") if alpha == 0.10 else ("exp30",):
            plan = plan_for(rates, alpha, pname)
            mean, dist, cv = patience[pname]
            ev = score(plan, rates, s, THRESHOLD, **_patience_kw("renege", mean, dist, cv))
            rows.append({"office": name, "mean_load": load, "alpha": alpha, "patience": pname,
                         "no_abandonment_plan": json.dumps(base),
                         "no_abandonment_hours": sum(base),
                         "fail_target_plan": json.dumps(plan), "fail_target_hours": sum(plan),
                         "saving": round(1 - sum(plan) / sum(base), 4),
                         "worst_fail": round(ev.worst("fail"), 4),
                         "fail_misses": ev.misses("fail", alpha)})
            print(f"  {name:<7} alpha={alpha:<4} {pname:<6}: {sum(base)} h -> {sum(plan)} h "
                  f"({100 * rows[-1]['saving']:+.1f}% saving), worst fail {ev.worst('fail'):.3f}")
    write_csv("e8b_regimes.csv", rows)

    # H17: the office's no-abandonment plan under different patience shapes
    base = plan_for(OFFICE_RATES, 0.10, "none")
    ref = score(base, OFFICE_RATES, s, THRESHOLD)
    rows = [{"patience": "none", "worst_hour_rate": round(ref.worst("late"), 4), "delta": 0.0,
             "rate_by_hour": json.dumps([round(x, 4) for x in ref.served_late])}]
    for label, mean, dist, cv in [("exp30", 30.0, "exp", 1.0), ("exp60", 60.0, "exp", 1.0),
                                  ("exp120", 120.0, "exp", 1.0),
                                  ("logn30", 30.0, "lognormal", 0.5),
                                  ("logn60", 60.0, "lognormal", 0.5),
                                  ("logn120", 120.0, "lognormal", 0.5)]:
        ev = score(base, OFFICE_RATES, s, THRESHOLD, **_patience_kw("renege", mean, dist, cv))
        rows.append({"patience": label, "worst_hour_rate": round(ev.worst("fail"), 4),
                     "delta": round(ev.worst("fail") - ref.worst("late"), 4),
                     "rate_by_hour": json.dumps([round(x, 4) for x in ev.fail])})
        print(f"  office plan {base}, {label:<7}: worst-hour failure {ev.worst('fail'):.4f} "
              f"(no abandonment: late {ref.worst('late'):.4f}, delta {rows[-1]['delta']:+.4f})")
    write_csv("e8b_patience_shape.csv", rows)


# ============================================================================
# E9: the fixed-threshold regime (Round 6, H19-H21; exact models, no simulation)
# ============================================================================

def k1(mean_service, mean_patience):
    """Limit of sqrt(c) P(abandon) in a critically loaded Erlang-A queue (Round 6)."""
    r = math.sqrt(mean_service / mean_patience)
    return r * math.sqrt(2 / math.pi) / (1 + r)


def crossover_slack(c, s, threshold, patience):
    """
    Slack delta* = c - R* at which abandonment (Erlang-A) stops raising the
    failure rate at fixed c: below R* early leavers dominate, above it
    queue thinning. None if no load 0 < R < c has a crossover.
    """
    from scipy.optimize import brentq
    from abandonment import renege_metrics
    from staffing_methods import prob_wait_exceeds

    def diff(delta):
        load = c - delta
        return (renege_metrics(c, load / s, s, patience, threshold, offered=False).fail
                - prob_wait_exceeds(c, load, s, threshold))

    lo = 1e-3                      # Near critical load Erlang-C is far worse
    cap = 0.999 * c                # Keep some load: R = c - delta > 0
    hi = min(cap, (s / threshold) * (0.5 * math.log(c) + 3.0) + 1.0)
    while diff(hi) < 0:
        if hi >= cap:
            return None
        hi = min(cap, hi * 1.5)
    if diff(lo) > 0:
        return None
    return brentq(diff, lo, hi, xtol=1e-7)


def pmap_processes(fn, items, workers=8):
    """Like pmap, for CPU-bound pure-Python work (threads would share the GIL)."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        out = []
        for k, f in enumerate(futures, 1):
            out.append(f.result())
            if k % 20 == 0 or k == len(futures):
                print(f"    {k}/{len(futures)} cases", flush=True)
        return out


def _e9a_case(case):
    from staffing_methods import prob_wait_exceeds
    s, t, p, c = case
    row = {"service_time": s, "threshold": t, "patience": p, "windows": c,
           "delta_star": "", "scaled_delta": "", "beta_star": "", "alpha_star": "",
           "alpha_star_sqrt_c": "", "k1": round(k1(s, p), 5)}
    delta = crossover_slack(c, s, t, p)
    if delta is not None:
        alpha_star = prob_wait_exceeds(c, c - delta, s, t)
        row.update({"delta_star": round(delta, 5), "scaled_delta": round(delta * t / s, 5),
                    "beta_star": round(delta / math.sqrt(c), 5),
                    "alpha_star": round(alpha_star, 7),
                    "alpha_star_sqrt_c": round(alpha_star * math.sqrt(c), 5)})
    return row


def run_e9a():
    """H19-H20: asymptotics of the crossover slack and of alpha*(c)."""
    print("E9a: crossover slack and alpha* for c up to 10,000 (exact Erlang-A)")
    windows = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    cases = [(s, t, p, c) for s in (4.0, 8.0, 16.0, 32.0) for t in (5.0, 15.0, 30.0)
             for p in (30.0, 120.0) for c in windows]
    # Largest (slowest) cases first so the pool stays busy to the end
    order = sorted(range(len(cases)), key=lambda i: -cases[i][3] * cases[i][1] / cases[i][0])
    done = pmap_processes(_e9a_case, [cases[i] for i in order])
    rows = [None] * len(cases)
    for i, row in zip(order, done):
        rows[i] = row
    write_csv("e9a_log_slack.csv", rows)
    for s in (4.0, 8.0, 16.0, 32.0):
        for t in (5.0, 15.0, 30.0):
            for p in (30.0, 120.0):
                pts = [(math.log(r["windows"]), r["delta_star"]) for r in rows
                       if (r["service_time"], r["threshold"], r["patience"]) == (s, t, p)
                       and r["windows"] >= 1000 and r["delta_star"] != ""]
                top = next(r for r in rows if (r["service_time"], r["threshold"], r["patience"],
                                               r["windows"]) == (s, t, p, 10000))
                slope = np.polyfit(*zip(*pts), 1)[0] if len(pts) > 1 else float("nan")
                print(f"  S={s:<4g} T={t:<4g} patience {p:<5g}: slope {slope:.3f} "
                      f"(S/2T {s / (2 * t):.3f}), beta* {top['beta_star']}, "
                      f"alpha* sqrt(c) {top['alpha_star_sqrt_c']} (K1 {top['k1']})")


E9B_KINK = THRESHOLD / math.log(1 / (1 - ALPHA))     # Patience mean with G(T) = alpha


def _e9b_case(case):
    from abandonment import fluid_discount, required_windows
    name, p, R = case
    d = fluid_discount(THRESHOLD, ALPHA, p)
    c = required_windows(R / OFFICE_S * 60, OFFICE_S, THRESHOLD, ALPHA, "renege", p)
    x = c - (1 - d) * R
    return {"patience": name, "mean_patience": round(p, 3),
            "G_T": round(1 - math.exp(-THRESHOLD / p), 5), "load": R, "windows": c,
            "fluid_windows": round((1 - d) * R, 3), "excess": round(x, 3),
            "excess_over_sqrt_load": round(x / math.sqrt(R), 4)}


def run_e9b():
    """H21: order of the staffing correction above the fluid limit."""
    print("E9b: extra windows above the fluid staffing (S = 8, T = 15, alpha = 0.10)")
    loads = [50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]
    patiences = [("exp30", 30.0), ("exp300", 300.0), ("exp_kink", E9B_KINK), ("exp120", 120.0)]
    cases = [(name, p, R) for name, p in patiences for R in loads]
    rows = pmap_processes(_e9b_case, cases)
    write_csv("e9b_fluid_order.csv", rows)
    for name, _ in patiences:
        print(f"  {name:<8} excess: " + ", ".join(
            f"{r['load']}:{r['excess']:g}" for r in rows if r["patience"] == name))


EXPERIMENTS = {"e1b": run_e1b, "e1": run_e1, "e2": run_e2, "e2b": run_e2b,
               "e3a": run_e3a, "e3b": run_e3b, "e4": run_e4, "e5": run_e5, "e6": run_e6,
               "e7a": run_e7a, "e7": run_e7, "e7c": run_e7c, "e8a": run_e8a, "e8b": run_e8b,
               "e9a": run_e9a, "e9b": run_e9b}


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
