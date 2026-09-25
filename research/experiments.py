"""
Experiments for the CivicQ staffing study. Every result is written to
research/results/*.csv and is reproducible from fixed seeds.

    python research/experiments.py --all
    python research/experiments.py e1b e1      # or any subset: e1b e1 e2 e2b e3a e3b e4 e5 e6 e7a e7 e7c e8a e8b e9a e9b e10a-h e11a-e e12a-e
    (run e10b before e10a, e10c, e10d and e10e: they read its fluid constants)

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


# ============================================================================
# E10: a finite-horizon fluid theory of the walk-in day (Round 7, H22-H25)
# ============================================================================

E10_SHAPES = ("double", "single", "ramp")


def _fluid_case(case):
    """Fluid constants for one (shape, A, S, T): alpha = 0 LP and the alpha MILP."""
    from fluid import fluid_day, fluid_staffing
    from staffing_methods import SHAPES
    shape, amp, s, t = case
    rates = arrival_profile(1.0, s, amp, SHAPES[shape])       # 1 Erlang; scales with R
    row = {"shape": shape, "amplitude": amp, "service_time": s, "threshold": t,
           "load_by_hour": json.dumps([round(r / 60 * s, 4) for r in rates])}
    for tag, a in (("f0", 0.0), ("f_alpha", ALPHA)):
        sol = fluid_staffing(rates, s, t, a, time_limit=900)
        day = fluid_day(sol["plan"], rates, s, t, dt=min(0.25, s / 32))
        dt = day["t"][1] - day["t"][0]
        c = np.array(sol["plan"])[np.minimum((day["t"] // 60).astype(int), SLOTS - 1)]
        idle = float(np.sum(np.maximum(c - day["X"], 0.0)) * dt)     # window-minutes
        backlog = float(day["X"][-1] + (rates[-1] / 60 - min(day["X"][-1], sol["plan"][-1])
                                       / s) * dt)                 # X at closing
        row.update({tag: round(sol["cost"] / SLOTS, 5),
                    f"{tag}_plan": json.dumps([round(x, 4) for x in sol["plan"]]),
                    f"{tag}_optimal": sol["optimal"],
                    f"{tag}_idle_share": round(idle / (SLOTS * 60), 5),
                    f"{tag}_backlog_share": round(s * backlog / (SLOTS * 60), 5),
                    f"{tag}_worst_fluid_late": round(max(day["late"]), 4)})
    return row


def run_e10b():
    """Fluid constants f = window-hours / (8R) for every shape, swing and service time."""
    print("E10b: fluid constants (T = 15, and T = 1.875 S for the D2 settings)")
    cases = [(sh, a, s, THRESHOLD) for sh in E10_SHAPES for a in AMPLITUDES
             for s in SERVICE_TIMES]
    cases += [("double", 0.6, s, 1.875 * s) for s in SERVICE_TIMES if s != 8.0]
    rows = pmap_processes(_fluid_case, cases)
    write_csv("e10b_fluid_constants.csv", rows)
    for r in rows:
        print(f"  {r['shape']:<6} A={r['amplitude']:<4} S={r['service_time']:<4g} "
              f"T={r['threshold']:<5g}: f0 {r['f0']:.4f}, f_alpha {r['f_alpha']:.4f} "
              f"(idle {r['f_alpha_idle_share']:.4f}, backlog {r['f_alpha_backlog_share']:.4f})")


def _fluid_row(shape, amp, s, t):
    """E10b's fluid constants for one setting; solved directly if E10b has not run yet."""
    path = RESULTS / "e10b_fluid_constants.csv"
    rows = load_results(path.name) if path.exists() else []
    for r in rows:
        if (r["shape"], float(r["amplitude"]), float(r["service_time"]),
                float(r["threshold"])) == (shape, amp, s, t):
            return r
    return _fluid_case((shape, amp, s, t))


def load_results(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def _method_rows(label, rates, s, t, fluid_plan_1e, load, extra=None):
    """SIPP / Lag-SIPP / OL-avg / SGS / SGS-UCB for one setting, scored on eval days."""
    plans = analytic_plans(rates, s, t, ALPHA)
    plans["SGS"], _ = simulation_staffing(rates, s, t, ALPHA)
    plans["SGS-UCB"], _ = simulation_staffing(rates, s, t, ALPHA, start=plans["SGS"],
                                              criterion="ucb")
    fluid = [x * load for x in fluid_plan_1e]
    names = list(plans)
    evals = pmap(lambda n: evaluate(plans[n], rates, s, t), names)
    ucb = sum(plans["SGS-UCB"])
    rows = []
    for name, ev in zip(names, evals):
        rows.append({"setting": label, "mean_load": load, "service_time": s, "threshold": t,
                     **(extra or {}), "method": name, "plan": json.dumps(plans[name]),
                     "staff_hours": ev.staff_hours,
                     "gap_vs_ucb_pct": round(100 * (ev.staff_hours - ucb) / ucb, 2),
                     "fluid_hours": round(sum(fluid), 2),
                     "corr_with_fluid": round(float(np.corrcoef(plans[name], fluid)[0, 1]), 4),
                     "worst_late": round(max(ev.late_prob), 4),
                     "hours_significantly_over": ev.hours_missing(ALPHA),
                     "late_by_hour": json.dumps([round(p, 4) for p in ev.late_prob])})
    print(f"  {label}: " + ", ".join(f"{n} {sum(plans[n])}h" for n in names)
          + f" | fluid {sum(fluid):.1f}h")
    return rows


def run_e10a():
    """H22 (D2): the E1 comparison at 24 E with T/S held at 15/8."""
    print("E10a: analytic rules vs SGS-UCB at 24 E, A = 0.6, T = 1.875 S")
    rows = []
    for s in SERVICE_TIMES:
        t = 1.875 * s
        rates = arrival_profile(24.0, s, 0.6)
        fluid = json.loads(_fluid_row("double", 0.6, s, t)["f_alpha_plan"])
        rows += _method_rows(f"R24_S{s:g}_T{t:g}", rates, s, t, fluid, 24.0)
    write_csv("e10a_threshold_ratio.csv", rows)


def run_e10c():
    """H23: SGS-UCB against the fluid constant at 64 and 128 Erlangs (S = 8, A = 0.6)."""
    print("E10c: SGS-UCB at 64 and 128 Erlangs against the fluid")
    s = OFFICE_S
    fluid = json.loads(_fluid_row("double", 0.6, s, THRESHOLD)["f_alpha_plan"])
    slack = s / THRESHOLD * math.log(1 / ALPHA)          # Round 6 stationary slack per hour
    rows = []
    for load in (64.0, 128.0):
        rates = arrival_profile(load, s, 0.6)
        start = [math.ceil(x * load + slack) for x in fluid]
        plan, calls = simulation_staffing(rates, s, THRESHOLD, ALPHA, start=start,
                                          criterion="ucb")
        ev = evaluate(plan, rates, s, THRESHOLD)
        rows.append({"mean_load": load, "plan": json.dumps(plan), "staff_hours": sum(plan),
                     "start": json.dumps(start), "simulator_calls": calls,
                     "worst_late": round(max(ev.late_prob), 4),
                     "hours_significantly_over": ev.hours_missing(ALPHA)})
        print(f"  R={load:g}: {plan} ({sum(plan)} h, from {sum(start)} h, {calls} calls), "
              f"worst hour {max(ev.late_prob):.3f}")
    write_csv("e10c_scaling.csv", rows)


def run_e10d():
    """H24 robustness: the two new demand shapes at 24 E (S = 8 and 32, A = 0.6)."""
    from staffing_methods import SHAPES
    print("E10d: new demand shapes at 24 E")
    rows = []
    for shape in ("single", "ramp"):
        for s in (8.0, 32.0):
            rates = arrival_profile(24.0, s, 0.6, SHAPES[shape])
            fluid = json.loads(_fluid_row(shape, 0.6, s, THRESHOLD)["f_alpha_plan"])
            rows += _method_rows(f"{shape}_R24_S{s:g}", rates, s, THRESHOLD, fluid, 24.0,
                                 {"shape": shape})
    write_csv("e10d_shapes.csv", rows)


def _fluid_roster_case(case):
    from fluid import fluid_staffing
    from shifts import FLEXIBLE, STANDARD, Menu
    menu_name, s = case
    menu = Menu(STANDARD if menu_name == "standard" else FLEXIBLE)
    rates = arrival_profile(1.0, s, 0.6)
    sol = fluid_staffing(rates, s, THRESHOLD, ALPHA, menu=menu, time_limit=900)
    return {"menu": menu_name, "service_time": s, "roster_per_erlang": round(sol["cost"], 5),
            "units": json.dumps([round(x, 4) for x in sol["units"]]),
            "profile": json.dumps([round(x, 4) for x in sol["plan"]]),
            "optimal": sol["optimal"]}


def run_e10e():
    """H25: fluid rosters against the E4 integrated-search rosters."""
    print("E10e: fluid rosters (paid hours per Erlang) vs E4")
    rows = pmap_processes(_fluid_roster_case,
                          [(m, s) for m in ("standard", "flexible") for s in (8.0, 32.0)])
    e4 = load_results("e4_shifts.csv")
    e1 = load_results("e1_methods.csv")
    out = []
    for r in rows:
        s = r["service_time"]
        hourly = 8 * float(_fluid_row("double", 0.6, s, THRESHOLD)["f_alpha"])
        for load in (8.0, 24.0):
            setting = f"R{load:g}_S{s:g}_A0.6"
            iss = next(int(x["paid_hours"]) for x in e4 if x["setting"] == setting
                       and x["menu"] == r["menu"] and x["method"] == "ISS")
            best_two = min(int(x["paid_hours"]) for x in e4 if x["setting"] == setting
                           and x["menu"] == r["menu"] and x["method"] != "ISS")
            ucb = next(int(x["staff_hours"]) for x in e1 if x["method"] == "SGS-UCB"
                       and float(x["mean_load"]) == load and float(x["service_time"]) == s
                       and float(x["amplitude"]) == 0.6)
            fluid_roster, fluid_hourly = r["roster_per_erlang"] * load, hourly * load
            out.append({**r, "mean_load": load, "fluid_roster_hours": round(fluid_roster, 2),
                        "fluid_hourly_hours": round(fluid_hourly, 2), "iss_paid_hours": iss,
                        "best_two_step_hours": best_two, "sgs_ucb_hours": ucb,
                        "iss_over_fluid_pct": round(100 * (iss / fluid_roster - 1), 2),
                        "ucb_over_fluid_pct": round(100 * (ucb / fluid_hourly - 1), 2),
                        "fluid_price_of_shifts_pct": round(100 * (fluid_roster / fluid_hourly - 1), 2),
                        "measured_price_of_shifts_pct": round(100 * (iss / ucb - 1), 2)})
            print(f"  {r['menu']:<9} {setting}: fluid roster {fluid_roster:.1f} h vs ISS {iss} h "
                  f"({out[-1]['iss_over_fluid_pct']:+.1f}%); SGS-UCB {ucb} h vs fluid hourly "
                  f"{fluid_hourly:.1f} h ({out[-1]['ucb_over_fluid_pct']:+.1f}%)")
    write_csv("e10e_rosters.csv", out)


def _fluid_np_case(case):
    """Corrected fluid (closing windows finish their citizen), alpha = 0; plans per Erlang."""
    from fluid import fluid_staffing_nonpreemptive
    from shifts import FLEXIBLE, STANDARD, Menu
    from staffing_methods import SHAPES
    shape, amp, s, t, menu_name = case
    menu = {"hourly": None, "standard": Menu(STANDARD), "flexible": Menu(FLEXIBLE)}[menu_name]
    rates = arrival_profile(1.0, s, amp, SHAPES[shape])
    sol = fluid_staffing_nonpreemptive(rates, s, t, 0.0, menu=menu, time_limit=900)
    return {"shape": shape, "amplitude": amp, "service_time": s, "threshold": t,
            "menu": menu_name, "cost_per_erlang": round(sol["cost"], 5),
            "f0_np": round(sol["cost"] / SLOTS, 5) if menu is None else "",
            "plan": json.dumps([round(x, 4) for x in sol["plan"]]), "optimal": sol["optimal"]}


def run_e10f():
    """Post hoc (Round 7b): constants of the corrected, non-preemptive fluid."""
    print("E10f: corrected fluid (non-preemptive closing), alpha = 0")
    cases = [(sh, a, s, THRESHOLD, "hourly") for sh in E10_SHAPES for a in AMPLITUDES
             for s in SERVICE_TIMES]
    cases += [("double", 0.6, s, 1.875 * s, "hourly") for s in SERVICE_TIMES if s != 8.0]
    cases += [("double", 0.6, s, THRESHOLD, m) for m in ("standard", "flexible")
              for s in (8.0, 32.0)]
    rows = pmap_processes(_fluid_np_case, cases)
    write_csv("e10f_fluid_nonpreemptive.csv", rows)
    for r in rows:
        print(f"  {r['shape']:<6} A={r['amplitude']:<4} S={r['service_time']:<4g} "
              f"T={r['threshold']:<5g} {r['menu']:<9}: {r['cost_per_erlang']:.4f} per Erlang")


def run_e10g():
    """H26 (confirmatory): SGS-UCB at 256 Erlangs against the corrected fluid."""
    print("E10g: SGS-UCB at 256 Erlangs")
    s, load = OFFICE_S, 256.0
    row = next(r for r in load_results("e10f_fluid_nonpreemptive.csv")
               if r["shape"] == "double" and float(r["amplitude"]) == 0.6
               and float(r["service_time"]) == s and float(r["threshold"]) == THRESHOLD
               and r["menu"] == "hourly")
    fluid = json.loads(row["plan"])
    rates = arrival_profile(load, s, 0.6)
    start = [math.ceil(x * load) + 4 for x in fluid]      # Neutral start: E(start) ~ 32
    plan, calls = simulation_staffing(rates, s, THRESHOLD, ALPHA, start=start, criterion="ucb")
    ev = evaluate(plan, rates, s, THRESHOLD)
    excess = sum(plan) - 8 * load * float(row["f0_np"])
    write_csv("e10g_confirm.csv", [{
        "mean_load": load, "plan": json.dumps(plan), "staff_hours": sum(plan),
        "start": json.dumps(start), "simulator_calls": calls,
        "fluid_hours": round(8 * load * float(row["f0_np"]), 2), "excess": round(excess, 2),
        "worst_late": round(max(ev.late_prob), 4),
        "hours_significantly_over": ev.hours_missing(ALPHA)}])
    print(f"  R=256: {plan} ({sum(plan)} h from {sum(start)} h, {calls} calls), "
          f"excess over fluid {excess:.1f}, worst hour {max(ev.late_prob):.3f}")


def run_e10h():
    """Validation: the corrected fluid plan at 256 Erlangs, scaled by +-5%, in the simulator."""
    print("E10h: corrected fluid plan x (1 +- 5%) at 256 Erlangs, simulated")
    s, load = OFFICE_S, 256.0
    row = next(r for r in load_results("e10f_fluid_nonpreemptive.csv")
               if r["shape"] == "double" and float(r["amplitude"]) == 0.6
               and float(r["service_time"]) == s and float(r["threshold"]) == THRESHOLD
               and r["menu"] == "hourly")
    rates = arrival_profile(load, s, 0.6)
    rows = []
    for scale in (0.95, 1.0, 1.05):
        plan = [round(x * load * scale) for x in json.loads(row["plan"])]
        ev = evaluate(plan, rates, s, THRESHOLD)
        rows.append({"scale": scale, "plan": json.dumps(plan), "staff_hours": sum(plan),
                     "late_by_hour": json.dumps([round(p, 4) for p in ev.late_prob]),
                     "worst_late": round(max(ev.late_prob), 4),
                     "hours_over_alpha": sum(p > ALPHA for p in ev.late_prob)})
        print(f"  x{scale}: {sum(plan)} h, late by hour "
              + ", ".join(f"{p:.3f}" for p in ev.late_prob))
    write_csv("e10h_fluid_validation.csv", rows)


# ============================================================================
# E11: paying for overtime and spill (Round 8, H27-H30)
# ============================================================================

KAPPAS = (1.0, 1.5)


def _fluid_paid_case(case):
    from overtime import fluid_paid
    from staffing_methods import SHAPES
    shape, amp, s, t, kappa = case
    sol = fluid_paid(arrival_profile(1.0, s, amp, SHAPES[shape]), s, t, kappa, time_limit=900)
    return {"shape": shape, "amplitude": amp, "service_time": s, "threshold": t,
            "kappa": kappa, "f_paid": round(sol["paid_cost"] / SLOTS, 5),
            "window_share": round(sol["window_hours"] / SLOTS, 5),
            "spill_share": round(sol["spill_hours"] / SLOTS, 5),
            "overtime_share": round(sol["overtime_hours"] / SLOTS, 5),
            "plan": json.dumps([round(x, 4) for x in sol["plan"]]), "optimal": sol["optimal"]}


def run_e11a():
    """Paid-overtime fluid constants (theory): paid cost / (8R)."""
    print("E11a: fluid with unpaid work charged at kappa")
    cases = [(sh, 0.6, s, THRESHOLD, k) for sh in E10_SHAPES for s in SERVICE_TIMES
             for k in (0.0,) + KAPPAS]
    cases += [("double", 0.6, s, 1.875 * s, k) for s in SERVICE_TIMES if s != 8.0
              for k in (0.0,) + KAPPAS]
    rows = pmap_processes(_fluid_paid_case, cases)
    write_csv("e11a_fluid_paid.csv", rows)
    for r in rows:
        print(f"  {r['shape']:<6} S={r['service_time']:<4g} T={r['threshold']:<5g} "
              f"kappa={r['kappa']:<4g}: {r['f_paid']:.4f} (windows {r['window_share']:.4f}, "
              f"spill {r['spill_share']:.4f}, overtime {r['overtime_share']:.4f})")


def _paid_rows(label, rates, s, t, load, window_opt, kappa, extra=None):
    """Paid cost of every rule's plan and of the paid-cost optimum, on the eval days."""
    from overtime import paid_evaluation, paid_staffing
    plans = analytic_plans(rates, s, t, ALPHA)
    plans["SGS-UCB (window-hours)"] = window_opt
    plans["paid optimum"], calls = paid_staffing(rates, s, t, ALPHA, kappa, window_opt)
    names = list(plans)
    evals = pmap(lambda n: paid_evaluation(plans[n], rates, s, t, kappa), names)
    best = evals[names.index("paid optimum")]["paid_cost"]
    rows = []
    for name, ev in zip(names, evals):
        rows.append({"setting": label, "mean_load": load, "service_time": s, "threshold": t,
                     "kappa": kappa, **(extra or {}), "method": name,
                     "plan": json.dumps(plans[name]), "window_hours": ev["window_hours"],
                     "spill_hours": round(ev["spill_hours"], 3),
                     "overtime_hours": round(ev["overtime_hours"], 3),
                     "paid_cost": round(ev["paid_cost"], 3),
                     "paid_cost_all_stay": round(ev["paid_cost_all_stay"], 3),
                     "excess_pct": round(100 * (ev["paid_cost"] / best - 1), 2),
                     "worst_late": round(max(ev["late"]), 4),
                     "hours_significantly_over": sum(lo > ALPHA for lo, _ in ev["late_ci"]),
                     "search_calls": calls if name == "paid optimum" else ""})
    print(f"  {label} kappa={kappa:g}: " + ", ".join(
        f"{n} {e['paid_cost']:.1f}" for n, e in zip(names, evals)))
    return rows


def run_e11b():
    """H27 (D2 with paid overtime): the fixed-T/S settings of E10a."""
    print("E11b: 24 E, A = 0.6, T = 1.875 S, unpaid work charged")
    ucb = {float(r["service_time"]): json.loads(r["plan"])
           for r in load_results("e10a_threshold_ratio.csv") if r["method"] == "SGS-UCB"}
    rows = []
    for kappa in KAPPAS:
        for s in SERVICE_TIMES:
            rows += _paid_rows(f"R24_S{s:g}_T{1.875 * s:g}", arrival_profile(24.0, s, 0.6), s,
                               1.875 * s, 24.0, ucb[s], kappa)
    write_csv("e11b_paid_threshold_ratio.csv", rows)


def run_e11c():
    """H28: E1's 24 E settings (T = 15, A = 0.6) with paid overtime."""
    print("E11c: E1 settings at 24 E, A = 0.6, T = 15, unpaid work charged at 1")
    e1 = load_results("e1_methods.csv")
    rows = []
    for s in SERVICE_TIMES:
        ucb = next(json.loads(r["plan"]) for r in e1 if r["method"] == "SGS-UCB"
                   and float(r["mean_load"]) == 24.0 and float(r["service_time"]) == s
                   and float(r["amplitude"]) == 0.6)
        rows += _paid_rows(f"R24_S{s:g}_T15", arrival_profile(24.0, s, 0.6), s, THRESHOLD,
                           24.0, ucb, 1.0)
    write_csv("e11c_paid_e1.csv", rows)


def run_e11d():
    """H29: the paid optimum against the paid fluid as the office grows (S = 8, A = 0.6)."""
    print("E11d: paid optimum at 8, 32 and 128 Erlangs")
    e8b = {float(r["mean_load"]): json.loads(r["no_abandonment_plan"])
           for r in load_results("e8b_regimes.csv")
           if r["patience"] == "exp30" and float(r["alpha"]) == 0.10 and r["office"] != "office"}
    starts = {8.0: e8b[8.0], 32.0: e8b[32.0],
              128.0: json.loads(next(r["plan"] for r in load_results("e10c_scaling.csv")
                                     if float(r["mean_load"]) == 128.0))}
    rows = []
    for load, start in starts.items():
        rows += _paid_rows(f"R{load:g}_S8_T15", arrival_profile(load, OFFICE_S, 0.6), OFFICE_S,
                           THRESHOLD, load, start, 1.0)
    write_csv("e11d_paid_scaling.csv", rows)


def run_e11e():
    """H31 (confirmatory): the paid optimum at 256 Erlangs against the paid fluid."""
    print("E11e: paid optimum at 256 Erlangs")
    start = json.loads(load_results("e10g_confirm.csv")[0]["plan"])
    rows = _paid_rows("R256_S8_T15", arrival_profile(256.0, OFFICE_S, 0.6), OFFICE_S,
                      THRESHOLD, 256.0, start, 1.0)
    write_csv("e11e_paid_confirm.csv", rows)


# ============================================================================
# Round 9: tipping points of mandatory services (REPORT section 5.13)
# ============================================================================

E12_RATES = arrival_profile(8.0, 16.0, 0.6)
E12_S = 16.0
E12_PATIENCE = 30.0
E12_FAIL_PLAN = [8, 13, 9, 6, 5, 9, 12, 9]     # E7 SGS-UCB (fail), renege, exp 30
E12_LATE_PLAN = [6, 10, 8, 5, 4, 7, 10, 7]     # E7 SGS-UCB (late): collapsed in E7c
E12_PHIS = [1.0, 0.95, 0.9, 0.85, 0.8]
E12_TIMINGS = ["profile", "opening"]
E12_KW = {"abandonment": "renege", "patience": E12_PATIENCE,
          "patience_dist": "exp", "patience_cv": 1.0}


def _e12_plans():
    """The fail plan scaled by phi (rounded half up), plus E7c's collapsed plan."""
    plans = [(f"phi={phi:.2f}", [int(math.floor(phi * c + 0.5)) for c in E12_FAIL_PLAN])
             for phi in E12_PHIS]
    return plans + [("late plan", E12_LATE_PLAN)]


def run_e12a():
    """H32: the stationary Erlang-A return model has at most one fixed point."""
    from tipping import stationary_return_roots
    print("E12a: stationary return fixed points")
    rows = []
    for c, rho, s, pat, r in itertools.product([1, 2, 4, 8, 16, 32],
                                               [0.5, 0.8, 0.95, 1.0, 1.1, 1.5, 2.0],
                                               [4.0, 16.0], [10.0, 30.0, 120.0],
                                               [0.5, 0.9, 1.0]):
        rate = rho * c * 60.0 / s
        roots = stationary_return_roots(c, rate, s, pat, r)
        rows.append({"c": c, "rho_fresh": rho, "S": s, "patience": pat, "r": r,
                     "n_roots": len(roots),
                     "total_rate_per_hour": round(roots[0] * 60.0, 4) if roots else ""})
    counts = {k: sum(1 for x in rows if x["n_roots"] == k) for k in (0, 1, 2, 3)}
    print(f"  {len(rows)} cases, roots: {counts}")
    write_csv("e12a_stationary_roots.csv", rows)


def run_e12b():
    """Fluid of the day with returns: steady state, slope, recovery after a closure."""
    from tipping import (classify_roots, fluid_fixed_point, fluid_recovery_days,
                         fluid_return_curve)
    print("E12b: fluid return dynamics (8 E, S = 16, exp patience 30, r = 1)")
    fresh = sum(E12_RATES)
    grid = np.concatenate([np.linspace(0.0, fresh, 41),
                           np.geomspace(1.05 * fresh, 30 * fresh, 40)])
    rows = []
    for (name, plan), timing in itertools.product(_e12_plans(), E12_TIMINGS):
        h = np.array(fluid_return_curve(plan, E12_RATES, E12_S, E12_PATIENCE, 1.0,
                                        timing, grid))
        cls = classify_roots(grid, h)
        fp = fluid_fixed_point(plan, E12_RATES, E12_S, E12_PATIENCE, 1.0, timing)
        rec = (fluid_recovery_days(plan, E12_RATES, E12_S, E12_PATIENCE, 1.0, timing,
                                   fp["R"]) if math.isfinite(fp["R"]) else "")
        rows.append({"plan_name": name, "plan": json.dumps(plan), "window_hours": sum(plan),
                     "timing": timing, "fluid_R": round(fp["R"], 3),
                     "repeat_per_100": round(100 * fp["R"] / fresh, 2),
                     "slope": round(fp["slope"], 4), "relax_days": round(fp["relax_days"], 2),
                     "recovery_days": rec, "max_dh_step": round(float(np.max(np.diff(h))), 4),
                     "n_roots_grid": len(cls["roots"]), "bistable": cls["bistable"]})
        print(f"  {name:<10} {sum(plan):>3} h {timing:<8} "
              f"R*/100 = {rows[-1]['repeat_per_100']:>8} slope {rows[-1]['slope']:.3f} "
              f"recovery {rec} days; max dh {rows[-1]['max_dh_step']}")
    write_csv("e12b_fluid_returns.csv", rows)


E12_R_FACTORS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def run_e12c(reps=400):
    """H33, H34: simulated h(R) = L(R) - R on common random numbers."""
    from tipping import return_curve, sign_changes
    print("E12c: simulated return curves")
    fresh = sum(E12_RATES)
    grid = [f * fresh for f in E12_R_FACTORS]
    jobs = list(itertools.product(_e12_plans(), E12_TIMINGS))
    curves = pmap(lambda job: return_curve(job[0][1], E12_RATES, E12_S, 1.0, job[1], grid,
                                           THRESHOLD, reps, EVAL_SEED, **E12_KW), jobs)
    rows = []
    for ((name, plan), timing), curve in zip(jobs, curves):
        roots = sign_changes([p["R"] for p in curve], [p["h"] for p in curve])
        rises = sum(1 for p in curve[1:] if p["dh_low"] > 0)
        for p in curve:
            rows.append({"plan_name": name, "plan": json.dumps(plan), "window_hours": sum(plan),
                         "timing": timing, "R": round(p["R"], 3), "L": round(p["L"], 3),
                         "h": round(p["h"], 3), "h_low": round(p["h_low"], 3),
                         "h_high": round(p["h_high"], 3),
                         "dh": round(p.get("dh", math.nan), 4),
                         "dh_low": round(p.get("dh_low", math.nan), 4),
                         "dh_high": round(p.get("dh_high", math.nan), 4),
                         "sim_R": round(roots[0], 3) if roots else "",
                         "n_roots": len(roots), "significant_rises": rises})
        star = 100 * roots[0] / fresh if roots else math.inf
        print(f"  {name:<10} {timing:<8} roots {[round(x, 1) for x in roots]} "
              f"(R*/100 {star:.1f}), significant rises {rises}")
    write_csv("e12c_return_curves.csv", rows)


def run_e12d(chains=20, burn_in=30, max_after=400):
    """H35: day-to-day chains with one closure day; recovery time vs the fluid."""
    from tipping import simulate_return_chain
    print("E12d: day-to-day chains with a closure day")
    fresh = sum(E12_RATES)
    fluid = {(r["plan_name"], r["timing"]): r for r in load_results("e12b_fluid_returns.csv")}
    jobs = []
    for (name, plan), timing in itertools.product(_e12_plans(), E12_TIMINGS):
        fr = fluid[(name, timing)]
        R0 = float(fr["fluid_R"])
        if not math.isfinite(R0) or R0 > 3 * fresh:
            R0 = 0.0
        after = max_after if fr["recovery_days"] == "" else \
            min(max_after, max(60, 3 * int(fr["recovery_days"])))
        for k in range(chains):
            jobs.append((name, plan, timing, R0, burn_in + 1 + after, k))
    paths = pmap(lambda j: simulate_return_chain(
        j[1], E12_RATES, E12_S, 1.0, j[2], j[4], shock_day=burn_in, start_returns=j[3],
        seed=300_000 + 10_000 * j[5], threshold=THRESHOLD, **E12_KW), jobs)
    rows = []
    for (name, plan, timing, R0, days, k), path in zip(jobs, paths):
        before = [p["returns"] for p in path[10:burn_in]]
        base = float(np.mean(before)) if before else math.nan
        after = [p["returns"] for p in path[burn_in + 1:]]
        collapsed = len(path) < days
        recovery = ""
        if not collapsed:
            for d in range(len(after) - 6):
                if np.mean(after[d:d + 7]) <= base + 0.1 * fresh:
                    recovery = d + 1
                    break
        rows.append({"plan_name": name, "window_hours": sum(plan), "timing": timing,
                     "chain": k, "pre_shock_mean_R": round(base, 2),
                     "collapsed": collapsed, "recovery_days": recovery,
                     "final_R": round(path[-1]["returns"], 1),
                     "path": json.dumps([round(p["returns"], 1) for p in path])})
    for (name, plan), timing in itertools.product(_e12_plans(), E12_TIMINGS):
        sub = [r for r in rows if r["plan_name"] == name and r["timing"] == timing]
        rec = [r["recovery_days"] for r in sub if r["recovery_days"] != ""]
        med = float(np.median(rec)) if rec else math.nan
        print(f"  {name:<10} {timing:<8} recovered {len(rec)}/{len(sub)}, median {med:.0f} "
              f"days (fluid {fluid[(name, timing)]['recovery_days']})")
    write_csv("e12d_chains.csv", rows)


def run_e12e(reps=200):
    """H35 (confirmatory): the fluid's error on R* at 32 E, spread returns."""
    from tipping import fluid_fixed_point, return_curve, sign_changes
    print("E12e: steady state at 32 E vs the fluid")
    rates = arrival_profile(32.0, E12_S, 0.6)
    fresh = sum(rates)
    grid = [f * fresh for f in np.arange(0.0, 1.001, 0.025)]
    base = {r["plan_name"]: r for r in load_results("e12c_return_curves.csv")
            if r["timing"] == "profile"}
    fl8 = {r["plan_name"]: r for r in load_results("e12b_fluid_returns.csv")
           if r["timing"] == "profile"}
    rows = []
    for name, plan in _e12_plans():
        if name not in ("phi=0.95", "phi=0.90", "phi=0.85"):
            continue
        plan32 = [4 * c for c in plan]
        curve = return_curve(plan32, rates, E12_S, 1.0, "profile", grid, THRESHOLD, reps,
                             EVAL_SEED, **E12_KW)
        roots = sign_changes([p["R"] for p in curve], [p["h"] for p in curve])
        fp = fluid_fixed_point(plan32, rates, E12_S, E12_PATIENCE, 1.0, "profile")
        sim100 = 100 * roots[0] / fresh if roots else math.inf
        fl100 = 100 * fp["R"] / fresh
        ex8 = 100 * float(base[name]["sim_R"]) / sum(E12_RATES) - float(fl8[name]["repeat_per_100"])
        rows.append({"plan_name": name, "plan": json.dumps(plan32), "window_hours": sum(plan32),
                     "sim_per_100": round(sim100, 2), "fluid_per_100": round(fl100, 2),
                     "excess_32": round(sim100 - fl100, 2), "excess_8": round(ex8, 2),
                     "ratio": round((sim100 - fl100) / ex8, 3),
                     "significant_rises": sum(1 for p in curve[1:] if p["dh_low"] > 0)})
        print(f"  {name}: R*/100 sim {sim100:.1f}, fluid {fl100:.1f}; excess {sim100 - fl100:.1f} "
              f"vs {ex8:.1f} at 8 E (ratio {rows[-1]['ratio']})")
    write_csv("e12e_scaling_confirm.csv", rows)


# ============================================================================
# Round 10: learning patience from the office's own ticket log (E13)
# ============================================================================

# (office, patience, plan type, plan) from results/e7_abandonment.csv, hidden queue
E13_SETTINGS = [
    ("office", "exp30", "lean", [2, 2, 2, 2, 2, 2, 2, 2]),
    ("office", "exp30", "citizen", [3, 3, 3, 2, 2, 3, 3, 3]),
    ("office", "logn30", "lean", [2, 3, 2, 2, 2, 2, 3, 2]),
    ("office", "logn30", "citizen", [2, 3, 2, 2, 2, 3, 3, 3]),
    ("R8_S16_A0.6", "exp30", "lean", [6, 10, 8, 5, 4, 7, 10, 7]),
    ("R8_S16_A0.6", "exp30", "citizen", [8, 13, 9, 6, 5, 9, 12, 9]),
    ("R8_S16_A0.6", "logn30", "lean", [7, 13, 9, 6, 5, 8, 12, 9]),
    ("R8_S16_A0.6", "logn30", "citizen", [7, 14, 9, 6, 5, 9, 13, 9]),
]
E13_PATIENCE = {"exp30": (30.0, "exp", 1.0), "logn30": (30.0, "lognormal", 0.5)}
E13_POOL = 20_000
E13_RATE_POOL = 50_000                     # H38's two settings
E13_RATE_SETTINGS = [("office", "exp30", "lean"), ("R8_S16_A0.6", "exp30", "citizen")]
E13_RATE_DAYS = [10, 30, 100, 300, 1000]
E13_FAMILY_DAYS = [5, 10, 20, 30, 45, 60, 90, 120, 180, 250, 365, 500]
E13_REPS = 200
E13_T = 15.0


def _e13_office(office):
    return next((r, s) for name, r, s in _e7_offices() if name == office)


def _e13_log(office, pname, kind, plan):
    """Pooled ticket log; H38's settings get the larger pool, whose first days are E13_POOL's."""
    from patience_logs import pooled_ticket_log
    rates, s = _e13_office(office)
    mean, dist, cv = E13_PATIENCE[pname]
    days = E13_RATE_POOL if (office, pname, kind) in E13_RATE_SETTINGS else E13_POOL
    return pooled_ticket_log(plan, rates, s, mean, dist, cv, days=days)


def _e13_restrict(log, days):
    """The first `days` days of a pooled log."""
    from patience_logs import TicketLog
    keep = log.day < days
    return TicketLog(day=log.day[keep], v=log.v[keep], absent=log.absent[keep],
                     patience=log.patience[keep], days=days,
                     walkins_per_day=log.walkins_per_day)


def run_e13a():
    """H36: the ticket log is current-status data, and the CS-NPMLE recovers G."""
    from scipy.stats import kendalltau
    from patience_logs import cs_npmle, true_cdf
    print("E13a: is the ticket log current-status data?")
    rows = []
    for office, pname, kind, plan in E13_SETTINGS:
        log = _e13_restrict(_e13_log(office, pname, kind, plan), E13_POOL)
        mean, dist, cv = E13_PATIENCE[pname]
        tau, _ = kendalltau(log.v, log.patience)
        t, g = cs_npmle(log.v, log.absent)
        lo, hi = np.quantile(log.v, [0.05, 0.95])
        inside = (t >= lo) & (t <= hi)
        err = np.abs(g[inside] - true_cdf(t[inside], mean, dist, cv))
        # Consistency check of the censoring rule itself
        agree = float(np.mean(log.absent == (log.patience < log.v)))
        rows.append({"office": office, "patience": pname, "plan_type": kind,
                     "plan": json.dumps(plan), "staff_hours": sum(plan), "days": log.days,
                     "tickets_per_day": round(len(log.v) / log.days, 2),
                     "absent_per_day": round(log.absent.sum() / log.days, 2),
                     "walkins_per_day": round(log.walkins_per_day, 2),
                     "kendall_tau": round(float(tau), 5),
                     "v_p05": round(float(lo), 2), "v_p95": round(float(hi), 2),
                     "sup_error": round(float(err.max()), 4),
                     "absent_iff_patience_below_v": agree})
        print(f"  {office:<12} {pname:<6} {kind:<7} tau {tau:+.4f}, sup|G^-G| {err.max():.4f} "
              f"on V in [{lo:.1f}, {hi:.1f}], {rows[-1]['absent_per_day']} absent/day")
    write_csv("e13a_validation.csv", rows)


def run_e13b():
    """H37: bias of the call-center estimator; plus the curves for fig17."""
    from patience_logs import (cs_mle, cs_npmle, naive_km, naive_km_limit, pick_family,
                               step_at, true_cdf)
    print("E13b: what each estimator says about G(15) on 20,000 days")
    rows, curves = [], []
    for office, pname, kind, plan in E13_SETTINGS:
        log = _e13_restrict(_e13_log(office, pname, kind, plan), E13_POOL)
        mean, dist, cv = E13_PATIENCE[pname]
        g_true = float(true_cdf(E13_T, mean, dist, cv))
        tk, gk = naive_km(log.v, log.absent)
        tn, gn = cs_npmle(log.v, log.absent)
        fam, fits = pick_family(log.v, log.absent)
        g_km, g_np = step_at(tk, gk, E13_T), step_at(tn, gn, E13_T)
        # The call-center estimate of mean patience: area under the KM curve
        # (restricted to the largest observed V)
        km_mean = float(np.sum(np.diff(np.concatenate([[0.0], tk])) *
                               np.concatenate([[1.0], 1 - gk[:-1]])))
        rows.append({"office": office, "patience": pname, "plan_type": kind,
                     "staff_hours": sum(plan), "G15_true": round(g_true, 4),
                     "G15_naive_km": round(g_km, 4), "G15_npmle": round(g_np, 4),
                     "G15_exp_mle": round(float(fits["exp"].cdf(E13_T)), 4),
                     "G15_logn_mle": round(float(fits["lognormal"].cdf(E13_T)), 4),
                     "naive_rel_bias": round((g_km - g_true) / g_true, 4),
                     # Post hoc: KM's limit G(t) h_V(t), from the log's own waits
                     "naive_limit_posthoc": round(naive_km_limit(
                         log.v, lambda t: true_cdf(t, mean, dist, cv), E13_T), 4),
                     "v_mean": round(float(log.v.mean()), 2),
                     "aic_family": fam,
                     "mean_exp_mle": round(fits["exp"].mean, 2),
                     "mean_logn_mle": round(fits["lognormal"].mean, 2),
                     "mean_naive_km_restricted": round(km_mean, 2)})
        print(f"  {office:<12} {pname:<6} {kind:<7} G(15) true {g_true:.3f}  naive KM {g_km:.3f} "
              f"({rows[-1]['naive_rel_bias']:+.1%}; limit {rows[-1]['naive_limit_posthoc']:.3f})  "
              f"NPMLE {g_np:.3f}  AIC: {fam}")
        # fig17: one office's log over 30 and 300 days, in two settings
        if (office, pname, kind) in (("office", "exp30", "lean"),
                                     ("R8_S16_A0.6", "logn30", "citizen")):
            grid = np.arange(0.0, 60.01, 0.5)
            for n in (30, 300):
                v, a = log.first(n)
                tk, gk = naive_km(v, a)
                tn, gn = cs_npmle(v, a)
                fam_n, fits_n = pick_family(v, a)
                for x in grid:
                    curves.append({"office": office, "patience": pname, "plan_type": kind,
                                   "days": n, "t": x,
                                   "truth": round(float(true_cdf(x, mean, dist, cv)), 5),
                                   "naive_km": round(step_at(tk, gk, x), 5),
                                   "npmle": round(step_at(tn, gn, x), 5),
                                   "parametric": round(float(fits_n[fam_n].cdf(x)), 5),
                                   "family": fam_n, "v_max": round(float(v.max()), 2)})
    write_csv("e13b_estimators.csv", rows)
    write_csv("e13b_curves.csv", curves)


def run_e13c():
    """H38: RMSE of G(15) against days of log, NPMLE vs parametric."""
    from patience_logs import cs_mle, cs_npmle, step_at, true_cdf
    print("E13c: how fast does G(15) converge with days of log?")
    rows = []
    for office, pname, kind, plan in E13_SETTINGS:
        if (office, pname, kind) not in E13_RATE_SETTINGS:
            continue
        log = _e13_log(office, pname, kind, plan)
        mean, dist, cv = E13_PATIENCE[pname]
        g_true = float(true_cdf(E13_T, mean, dist, cv))
        rng = np.random.default_rng(1310)
        for n in E13_RATE_DAYS:
            est = {"npmle": [], "exp_mle": []}
            for _ in range(E13_REPS):
                v, a = log.sample(n, rng)
                t, g = cs_npmle(v, a)
                est["npmle"].append(step_at(t, g, E13_T))
                est["exp_mle"].append(float(cs_mle(v, a, "exp").cdf(E13_T)))
            for name, xs in est.items():
                xs = np.array(xs)
                rows.append({"office": office, "patience": pname, "plan_type": kind,
                             "estimator": name, "days": n, "reps": E13_REPS,
                             "G15_true": round(g_true, 4),
                             "bias": round(float(xs.mean() - g_true), 5),
                             "sd": round(float(xs.std(ddof=1)), 5),
                             "rmse": round(float(np.sqrt(np.mean((xs - g_true) ** 2))), 5)})
            print(f"  {office:<12} {kind:<7} n={n:<5} RMSE NPMLE {rows[-2]['rmse']:.4f}, "
                  f"exp MLE {rows[-1]['rmse']:.4f}")
        for name in ("npmle", "exp_mle"):
            sub = [r for r in rows if r["office"] == office and r["estimator"] == name]
            slope = np.polyfit(np.log([r["days"] for r in sub]),
                               np.log([r["rmse"] for r in sub]), 1)[0]
            for r in sub:
                r["slope"] = round(float(slope), 3)
            print(f"  {office:<12} {name}: log-log slope {slope:+.3f}")
    write_csv("e13c_rates.csv", rows)


def run_e13d():
    """H39, H40: days of log before AIC picks the true family."""
    from patience_logs import pick_family
    print("E13d: days needed to tell exponential from lognormal patience")
    rows = []
    for office, pname, kind, plan in E13_SETTINGS:
        log = _e13_restrict(_e13_log(office, pname, kind, plan), E13_POOL)
        truth = E13_PATIENCE[pname][1]
        rng = np.random.default_rng(1320)
        for n in E13_FAMILY_DAYS:
            right = 0
            for _ in range(E13_REPS):
                v, a = log.sample(n, rng)
                right += pick_family(v, a)[0] == truth
            rows.append({"office": office, "patience": pname, "plan_type": kind,
                         "staff_hours": sum(plan), "days": n, "reps": E13_REPS,
                         "correct": round(right / E13_REPS, 3)})
        print(f"  {office:<12} {pname:<6} {kind:<7} " +
              " ".join(f"{r['days']}:{r['correct']:.2f}" for r in rows[-len(E13_FAMILY_DAYS):]))
    write_csv("e13d_family.csv", rows)
    summary = []
    for office in ("office", "R8_S16_A0.6"):
        for kind in ("lean", "citizen"):
            need = {}
            for pname in ("exp30", "logn30"):
                sub = [r for r in rows if (r["office"], r["patience"], r["plan_type"])
                       == (office, pname, kind)]
                ok = [r["correct"] >= 0.9 for r in sub]
                # First grid point from which the rate stays >= 0.9
                stays = next((sub[i]["days"] for i in range(len(sub)) if all(ok[i:])), None)
                first = next((r["days"] for r in sub if r["correct"] >= 0.9), None)
                need[pname] = (stays, first)
            both = [x[0] for x in need.values()]
            summary.append({"office": office, "plan_type": kind,
                            "days_exp": need["exp30"][0], "days_logn": need["logn30"][0],
                            "first_exp": need["exp30"][1], "first_logn": need["logn30"][1],
                            "days_needed": (max(both) if all(b is not None for b in both)
                                            else f">{E13_FAMILY_DAYS[-1]}")})
            print(f"  {office:<12} {kind:<7} days needed: {summary[-1]['days_needed']} "
                  f"(exp {need['exp30'][0]}, logn {need['logn30'][0]})")
    write_csv("e13d_days_needed.csv", summary)


def run_e13e(blocks=20, days=60):
    """H41: Erlang-A SIPP with an exponential fitted to 60 days of the office's log."""
    from abandonment import score, sipp_abandonment
    from patience_logs import cs_mle
    print("E13e: restaffing the 8 E lognormal office from its own log")
    office, pname, kind, plan = E13_SETTINGS[7]
    rates, s = _e13_office(office)
    mean, dist, cv = E13_PATIENCE[pname]
    kw = _patience_kw("renege", mean, dist, cv)
    log = _e13_restrict(_e13_log(office, pname, kind, plan), E13_POOL)
    scored = {}

    def evaluate(p):
        key = tuple(p)
        if key not in scored:
            scored[key] = score(p, rates, s, THRESHOLD, **kw)
        return scored[key]

    rows = []
    for b in range(blocks):
        v, a = log.first(days, offset=b * days)
        fit = cs_mle(v, a, "exp")
        new = sipp_abandonment(rates, s, THRESHOLD, ALPHA, "renege", fit.mean, "fail")
        ev = evaluate(new)
        rows.append({"route": "exp CS-MLE, 60 days", "block": b, "fitted_mean": round(fit.mean, 2),
                     "plan": json.dumps(new), "staff_hours": sum(new),
                     "fail_misses": ev.misses("fail", ALPHA), "worst_fail": round(ev.worst("fail"), 4)})
        print(f"  block {b:>2}: fitted mean {fit.mean:6.1f} min -> {sum(new)} h, "
              f"worst hour {ev.worst('fail'):.3f}, misses {ev.misses('fail', ALPHA)}")
    # Reference routes on the same evaluation days
    for route, m in (("mean-matched (true mean 30)", mean),):
        p = sipp_abandonment(rates, s, THRESHOLD, ALPHA, "renege", m, "fail")
        ev = evaluate(p)
        rows.append({"route": route, "block": "", "fitted_mean": m, "plan": json.dumps(p),
                     "staff_hours": sum(p), "fail_misses": ev.misses("fail", ALPHA),
                     "worst_fail": round(ev.worst("fail"), 4)})
    ok = sum(1 for r in rows[:blocks] if r["fail_misses"] == 0)
    mean_h = np.mean([r["staff_hours"] for r in rows[:blocks]])
    print(f"  {ok}/{blocks} plans meet the failure target; mean {mean_h:.1f} staff-hours")
    write_csv("e13e_restaff.csv", rows)


def run_e13f():
    """Exploratory: the visible line, learned from a timestamped door counter."""
    from patience_logs import cs_npmle, pooled_door_counts, step_at, true_cdf
    print("E13f: balking patience from door counts (exploratory)")
    rows = []
    for office, pname, kind, plan in E13_SETTINGS:
        if kind != "lean":
            continue
        rates, s = _e13_office(office)
        mean, dist, cv = E13_PATIENCE[pname]
        door = pooled_door_counts(plan, rates, s, mean, dist, cv, days=E13_POOL)
        g_true = float(true_cdf(E13_T, mean, dist, cv))
        t, g = cs_npmle(door.v, door.absent)
        # Expected waits sit on a grid (q + 1) S / c; report the pooled fit there
        pts = np.unique(np.round(door.v, 6))
        pts = pts[pts <= 30.0]
        grid_err = max(abs(step_at(t, g, x) - float(true_cdf(x, mean, dist, cv))) for x in pts)
        rng = np.random.default_rng(1360)
        rmse = {}
        for n in (30, 100, 300):
            xs = []
            for _ in range(100):
                v, a = door.sample(n, rng)
                tt, gg = cs_npmle(v, a)
                xs.append(step_at(tt, gg, E13_T))
            rmse[n] = float(np.sqrt(np.mean((np.array(xs) - g_true) ** 2)))
        below = pts[pts <= E13_T]
        rows.append({"office": office, "patience": pname, "plan": json.dumps(plan),
                     "arrivals_facing_a_line_per_day": round(len(door.v) / door.days, 2),
                     "balks_per_day": round(door.absent.sum() / door.days, 2),
                     "grid_points_le_30": len(pts),
                     "nearest_grid_point_le_15": round(float(below.max()), 3) if len(below) else "",
                     "G15_true": round(g_true, 4), "G15_pooled": round(step_at(t, g, E13_T), 4),
                     "G_at_grid_point": round(float(true_cdf(below.max(), mean, dist, cv)), 4)
                     if len(below) else "",
                     "max_grid_error": round(grid_err, 4),
                     "rmse_30d": round(rmse[30], 4), "rmse_100d": round(rmse[100], 4),
                     "rmse_300d": round(rmse[300], 4)})
        print(f"  {office:<12} {pname:<6} G(15) {g_true:.3f}, pooled {rows[-1]['G15_pooled']}, "
              f"grid pts <=30: {len(pts)}, RMSE 30/100/300 d: "
              f"{rmse[30]:.3f}/{rmse[100]:.3f}/{rmse[300]:.3f}")
    write_csv("e13f_door_counter.csv", rows)

# ============================================================================
# Round 11: learning by staffing (E14)
# ============================================================================

E14_TRUTHS = {"exp30": ("exp", 30.0, 1.0), "logn30": ("lognormal", 30.0, 0.5)}


def _e14_truth(pname):
    from learning import Patience
    fam, mean, cv = E14_TRUTHS[pname]
    return Patience(fam, mean, cv)


def _e14_starts(office, pname):
    """Erlang-C SIPP, and the E7 plans for this truth (hidden queue)."""
    rates, s = _e13_office(office)
    starts = {"Erlang-C SIPP": analytic_plans(rates, s, THRESHOLD, ALPHA)["SIPP"]}
    for r in load_results("e7_abandonment.csv"):
        if r["office"] == office and r["patience"] == pname and r["mode"] == "renege":
            for label in r["found_by"].split("; "):
                if label in ("SGS-UCB (fail)@renege", "SGS-UCB (late)@renege",
                             "SIPP-A (fail)@renege"):
                    starts[label.replace("@renege", "")] = json.loads(r["plan"])
    return starts


def run_e14a(reps=200):
    """The exact M/M/c+G model against the simulator (constant demand and staffing)."""
    from abandonment import renege_metrics, score
    from learning import Patience, mmcg_hour
    print("E14a: exact M/M/c+G vs the simulator and vs Erlang-A")
    cases = [(3, 15.0, 8.0, "exp30"), (2, 15.0, 8.0, "logn30"), (3, 15.0, 8.0, "logn30"),
             (2, 20.0, 8.0, "logn30"), (8, 30.0, 16.0, "logn30"), (10, 40.0, 16.0, "logn30"),
             (12, 48.0, 16.0, "logn30"), (10, 40.0, 16.0, "exp30")]
    rows = []
    for c, lam, s, pname in cases:
        truth = _e14_truth(pname)
        fam, mean, cv = E14_TRUTHS[pname]
        exact = mmcg_hour(c, lam / 60.0, s, truth, THRESHOLD)
        ev = score([c] * SLOTS, [lam] * SLOTS, s, THRESHOLD, reps=reps, seed=300_000,
                   duration=20000, **_patience_kw("renege", mean, fam, cv))
        row = {"windows": c, "rate_per_hour": lam, "service_time": s, "patience": pname,
               "erlang_a_fail": (round(renege_metrics(c, lam / 60, s, mean, THRESHOLD).fail, 5)
                                 if fam == "exp" else "")}
        for name, sim, ci, ex in [("fail", ev.fail[7], ev.fail_ci[7], exact.fail),
                                  ("served_late", ev.served_late[7], ev.served_late_ci[7],
                                   exact.served_late)]:
            se = (ci[1] - ci[0]) / (2 * 1.96)
            row.update({f"{name}_exact": round(ex, 5), f"{name}_sim": round(sim, 5),
                        f"{name}_z": round((sim - ex) / se, 2) if se > 0 else 0.0})
        row.update({"abandon_exact": round(exact.abandon, 5), "abandon_sim": round(ev.abandon[7], 5)})
        rows.append(row)
        print(f"  c={c:<2} lam={lam:g} S={s:g} {pname}: fail {ev.fail[7]:.4f} vs {exact.fail:.4f} "
              f"(z={row['fail_z']}), served-late z={row['served_late_z']}, abandon "
              f"{ev.abandon[7]:.4f} vs {exact.abandon:.4f}")
    write_csv("e14a_mmcg_validation.csv", rows)


E14_CASES = [(8.0, 0.10), (8.0, 0.20), (32.0, 0.10), (32.0, 0.20)]   # (Erlangs, alpha), S = 16
E14_S = 16.0
E14_EXPLORE = [(0.1, 0.9), (0.2, 0.9), (0.1, 0.8), (0.2, 0.8)]      # (share of days p, scale phi)


def run_e14b():
    """Theory: the refit-and-restaff loop in the stationary large-sample limit."""
    from learning import (explore_plan, failures_per_day, iterate_explore, iterate_map,
                          plan_distribution, sipp_g)
    print("E14b: fixed points of the learning loop (stationary limit)")
    rows = []
    # (1) The E7 offices at alpha = 0.10 from every E7 start
    for office in ("office", "R8_S16_A0.6"):
        rates, s = _e13_office(office)
        for pname in E14_TRUTHS:
            truth = _e14_truth(pname)
            oracle = sipp_g(rates, s, THRESHOLD, ALPHA, truth)
            for rule in ("A", "B"):
                for sname, start in _e14_starts(office, pname).items():
                    path = iterate_map(start, rates, s, truth, rule)
                    end, fitted = path[-1]
                    rows.append({"part": "starts", "office": office, "alpha": ALPHA,
                                 "patience": pname, "rule": rule, "start": sname,
                                 "p": 0.0, "phi": "", "start_hours": sum(start),
                                 "fixed_point": json.dumps(end), "fixed_hours": sum(end),
                                 "oracle_hours": sum(oracle), "fitted_mean": round(fitted.mean, 1),
                                 "path_hours": json.dumps([sum(q) for q, _ in path])})
                    print(f"  {office:<12} {pname:<6} rule {rule} from {sname:<22} {sum(start):>3} h "
                          f"-> {sum(end):>3} h (oracle {sum(oracle)})")
    # (2) Size x target, from Erlang-C, with and without exploration days
    for R, alpha in E14_CASES:
        rates = arrival_profile(R, E14_S, 0.6)
        erl = analytic_plans(rates, E14_S, THRESHOLD, alpha)["SIPP"]
        for pname in E14_TRUTHS:
            truth = _e14_truth(pname)
            oracle = sipp_g(rates, E14_S, THRESHOLD, alpha, truth)
            for rule in ("A", "B"):
                for p, phi in [(0.0, 1.0)] + (E14_EXPLORE if rule == "A" else []):
                    path = iterate_explore(erl, rates, E14_S, truth, rule, p, phi, alpha=alpha)
                    end, fitted = path[-1]
                    ex = explore_plan(end, phi) if p > 0 else end
                    fails = ((1 - p) * failures_per_day(end, rates, E14_S, truth)
                             + p * failures_per_day(ex, rates, E14_S, truth))
                    row = {"part": "size", "office": f"R{R:g}", "alpha": alpha, "patience": pname,
                           "rule": rule, "start": "Erlang-C SIPP", "p": p,
                           "phi": phi if p > 0 else "", "start_hours": sum(erl),
                           "fixed_point": json.dumps(end), "fixed_hours": sum(end),
                           "oracle_hours": sum(oracle), "fitted_mean": round(fitted.mean, 1),
                           "path_hours": json.dumps([sum(q) for q, _ in path]),
                           "paid_hours_per_day": round((1 - p) * sum(end) + p * sum(ex), 2),
                           "failures_per_day": round(fails, 1),
                           "oracle_failures_per_day": round(
                               failures_per_day(oracle, rates, E14_S, truth), 1)}
                    if p == 0:
                        h, _ = plan_distribution([(erl, 1.0)], rates, E14_S, truth, rule, 30,
                                                 draws=200, alpha=alpha)
                        row.update({"first_refit_30d_mean": round(float(h.mean()), 2),
                                    "first_refit_30d_sd": round(float(h.std()), 2)})
                    rows.append(row)
                    print(f"  R={R:g} a={alpha} {pname:<6} rule {rule} p={p} phi={phi}: "
                          f"{sum(erl)} -> {sum(end)} h (path {row['path_hours']}; oracle "
                          f"{sum(oracle)}), fitted mean {fitted.mean:.0f}, failures/day {fails:.1f}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    write_csv("e14b_fixed_points.csv", [{k: r.get(k, "") for k in keys} for r in rows])


E14_POLICIES = [("A", False), ("B", False), ("A", True)]     # (rule, explore)
E14_HISTORIES = 10
E14_PERIODS = 8
E14_DAYS = 30


def run_e14c():
    """H42-H46: simulated histories of refitting and restaffing from the office's own log."""
    from abandonment import score
    from learning import run_history
    print("E14c: learning-by-staffing histories")
    oracle = {(r["office"], float(r["alpha"]), r["patience"]): int(r["oracle_hours"])
              for r in load_results("e14b_fixed_points.csv") if r["part"] == "size"}
    tasks, keys = [], []
    for R, alpha in E14_CASES:
        rates = arrival_profile(R, E14_S, 0.6)
        erl = analytic_plans(rates, E14_S, THRESHOLD, alpha)["SIPP"]
        for pname in E14_TRUTHS:
            truth = _e14_truth(pname)
            for rule, explore in E14_POLICIES:
                for h in range(E14_HISTORIES):
                    tasks.append((rates, E14_S, alpha, truth, rule, explore, erl, h, E14_PERIODS,
                                  E14_DAYS, THRESHOLD))
                    keys.append((R, alpha, pname, rule, explore, h))
    histories = pmap_processes(run_history, tasks)
    rows = []
    for (R, alpha, pname, rule, explore, h), hist in zip(keys, histories):
        for rec in hist:
            rows.append({"office": f"R{R:g}", "alpha": alpha, "patience": pname, "rule": rule,
                         "explore": explore, "history": h, "period": rec["period"],
                         "hours_run": rec["hours_run"], "new_hours": rec["new_hours"],
                         "new_plan": json.dumps(rec["new_plan"]),
                         "failures_per_day": round(rec["failures_per_day"], 2),
                         "paid_hours_per_day": round(rec["paid_hours_per_day"], 2),
                         "fitted_family": rec["fitted_family"],
                         "fitted_mean": round(rec["fitted_mean"], 1),
                         "fitted_cv": round(rec["fitted_cv"], 3),
                         "oracle_hours": oracle[(f"R{R:g}", alpha, pname)]})
    write_csv("e14c_histories.csv", rows)

    # Score every distinct final plan on the evaluation days
    finals = {}
    for r in rows:
        if r["period"] == E14_PERIODS:
            finals.setdefault((r["office"], r["alpha"], r["patience"], r["new_plan"]), None)
    print(f"  scoring {len(finals)} distinct final plans")

    def score_one(key):
        office, alpha, pname, plan = key
        R = float(office[1:])
        fam, mean, cv = E14_TRUTHS[pname]
        ev = score(json.loads(plan), arrival_profile(R, E14_S, 0.6), E14_S, THRESHOLD,
                   **_patience_kw("renege", mean, fam, cv))
        return ev.misses("fail", alpha), round(ev.worst("fail"), 4)

    for key, res in zip(finals, pmap(score_one, list(finals))):
        finals[key] = res
    summary = []
    for R, alpha in E14_CASES:
        office = f"R{R:g}"
        for pname in E14_TRUTHS:
            for rule, explore in E14_POLICIES:
                sub = [r for r in rows if (r["office"], r["alpha"], r["patience"], r["rule"],
                                           r["explore"]) == (office, alpha, pname, rule, explore)]
                orc = sub[0]["oracle_hours"]
                first = [r["new_hours"] for r in sub if r["period"] == 1]
                last = [r for r in sub if r["period"] == E14_PERIODS]
                path = [float(np.mean([r["new_hours"] for r in sub if r["period"] == k]))
                        for k in range(1, E14_PERIODS + 1)]
                scored = [finals[(office, alpha, pname, r["new_plan"])] for r in last]
                summary.append({
                    "office": office, "alpha": alpha, "patience": pname, "rule": rule,
                    "explore": explore, "oracle_hours": orc,
                    "start_hours": sub[0]["hours_run"],
                    "first_refit_mean": round(float(np.mean(first)), 2),
                    "first_within": sum(abs(x - orc) <= (3 if R == 8 else 5) for x in first),
                    "final_mean": round(float(np.mean([r["new_hours"] for r in last])), 2),
                    "final_min": min(r["new_hours"] for r in last),
                    "final_max": max(r["new_hours"] for r in last),
                    "final_within": sum(abs(r["new_hours"] - orc) <= (2 if R == 8 else 3)
                                        for r in last),
                    "max_rise": round(max(b - a for a, b in zip(path, path[1:])), 2),
                    "mean_path": json.dumps([round(x, 1) for x in path]),
                    "failures_per_day": round(float(np.mean([r["failures_per_day"] for r in sub])), 2),
                    "paid_hours_per_day": round(float(np.mean([r["paid_hours_per_day"] for r in sub])), 2),
                    "final_fitted_mean": round(float(np.mean([r["fitted_mean"] for r in last])), 1),
                    "final_family_lognormal": sum(r["fitted_family"] == "lognormal" for r in last),
                    "final_safe": sum(m == 0 for m, _ in scored),
                    "final_worst_fail": max(w for _, w in scored)})
                x = summary[-1]
                print(f"  {office} a={alpha} {pname:<6} {rule}{'+explore' if explore else '':<8} "
                      f"{x['start_hours']} -> first {x['first_refit_mean']:.1f}, final "
                      f"{x['final_mean']:.1f} [{x['final_min']}-{x['final_max']}] (oracle {orc}); "
                      f"fails/day {x['failures_per_day']:.1f}; safe {x['final_safe']}/10")
    write_csv("e14c_summary.csv", summary)



def run_e14d():
    """H47 (confirmatory): rule B restaffing by Lag-SIPP-G, new histories."""
    from abandonment import score
    from learning import run_history, sipp_g
    from staffing_methods import lagged_rates
    print("E14d: learn the patience curve, staff for the lag")
    rule_a = {(r["office"], float(r["alpha"]), r["patience"]): float(r["final_mean"])
              for r in load_results("e14c_summary.csv")
              if r["rule"] == "A" and r["explore"] == "False"}
    tasks, keys = [], []
    for R, alpha in E14_CASES:
        rates = arrival_profile(R, E14_S, 0.6)
        erl = analytic_plans(rates, E14_S, THRESHOLD, alpha)["SIPP"]
        for pname in E14_TRUTHS:
            truth = _e14_truth(pname)
            for h in range(E14_HISTORIES):
                tasks.append((rates, E14_S, alpha, truth, "B", False, erl, h, E14_PERIODS,
                              E14_DAYS, THRESHOLD, True, 800_000))
                keys.append((R, alpha, pname, h))
    histories = pmap_processes(run_history, tasks)
    rows = []
    for (R, alpha, pname, h), hist in zip(keys, histories):
        for rec in hist:
            rows.append({"office": f"R{R:g}", "alpha": alpha, "patience": pname, "history": h,
                         "period": rec["period"], "hours_run": rec["hours_run"],
                         "new_hours": rec["new_hours"], "new_plan": json.dumps(rec["new_plan"]),
                         "failures_per_day": round(rec["failures_per_day"], 2),
                         "fitted_family": rec["fitted_family"],
                         "fitted_mean": round(rec["fitted_mean"], 1),
                         "fitted_cv": round(rec["fitted_cv"], 3)})
    write_csv("e14d_histories.csv", rows)
    finals = {}
    for r in rows:
        if r["period"] == E14_PERIODS:
            finals.setdefault((r["office"], r["alpha"], r["patience"], r["new_plan"]), None)

    def score_one(key):
        office, alpha, pname, plan = key
        fam, mean, cv = E14_TRUTHS[pname]
        ev = score(json.loads(plan), arrival_profile(float(office[1:]), E14_S, 0.6), E14_S,
                   THRESHOLD, **_patience_kw("renege", mean, fam, cv))
        return ev.misses("fail", alpha), round(ev.worst("fail"), 4)

    for key, res in zip(finals, pmap(score_one, list(finals))):
        finals[key] = res
    summary = []
    for R, alpha in E14_CASES:
        office = f"R{R:g}"
        rates = arrival_profile(R, E14_S, 0.6)
        for pname in E14_TRUTHS:
            target = sum(sipp_g(lagged_rates(rates, E14_S), E14_S, THRESHOLD, alpha,
                                _e14_truth(pname)))
            last = [r for r in rows if (r["office"], r["alpha"], r["patience"], r["period"])
                    == (office, alpha, pname, E14_PERIODS)]
            scored = [finals[(office, alpha, pname, r["new_plan"])] for r in last]
            final_mean = float(np.mean([r["new_hours"] for r in last]))
            summary.append({"office": office, "alpha": alpha, "patience": pname,
                            "lag_sipp_g_true": target,
                            "final_mean": round(final_mean, 2),
                            "final_min": min(r["new_hours"] for r in last),
                            "final_max": max(r["new_hours"] for r in last),
                            "within": sum(abs(r["new_hours"] - target) <= (2 if R == 8 else 3)
                                          for r in last),
                            "safe": sum(m == 0 for m, _ in scored),
                            "worst_fail": max(w for _, w in scored),
                            "rule_a_final_mean": rule_a[(office, alpha, pname)],
                            "saving_vs_rule_a": round(rule_a[(office, alpha, pname)] - final_mean, 2),
                            "failures_per_day": round(float(np.mean(
                                [r["failures_per_day"] for r in rows if (r["office"], r["alpha"],
                                 r["patience"]) == (office, alpha, pname)])), 2)})
            x = summary[-1]
            print(f"  {office} a={alpha} {pname:<6}: final {x['final_mean']:.1f} "
                  f"[{x['final_min']}-{x['final_max']}] vs Lag-SIPP-G(true) {target}, within "
                  f"{x['within']}/10, safe {x['safe']}/10 (worst {x['worst_fail']}), "
                  f"rule A {x['rule_a_final_mean']}")
    write_csv("e14d_summary.csv", summary)



# ============================================================================
# Round 12: should offices show the wait? (E15)
# ============================================================================

E15_DISPLAYS = {"T": "tickets", "C": "count", "L": "les"}
E15_REGIMES = {"H": {}, "T": {"announce": "tickets"}, "C": {"announce": "count"},
               "L": {"announce": "les"}, "V": {"announce": "count", "commit": True}}
E15_LOG_SEED = 900_000


def run_e15a(days=2000):
    """H48 (and H50's counterfactual): how each display compares with the wait V."""
    from patience_logs import (ARRIVAL, BOOKED, CALL, EST_COUNT, EST_LES, EST_TICKETS,
                               OUTCOME, PATIENCE, raw_log)
    print("E15a: accuracy of wait displays, read from the hidden queue's own log")
    rows = []
    for office, pname, kind, plan in E13_SETTINGS:
        rates, s = _e13_office(office)
        mean, dist, cv = E13_PATIENCE[pname]
        log = raw_log(plan, rates, s, "renege", mean, dist, cv, days=days, seed=E15_LOG_SEED)
        walk = log[log[:, BOOKED] == 0]
        v = walk[:, CALL] - walk[:, ARRIVAL]
        tau = walk[:, PATIENCE]
        waits = walk[:, EST_COUNT] > 0            # A window was not free on arrival
        reneged = walk[:, OUTCOME] == 1
        row = {"office": office, "patience": pname, "plan_type": kind, "staff_hours": sum(plan),
               "days": days, "facing_a_wait_per_day": round(waits.sum() / days, 2),
               "reneged_per_day": round(reneged.sum() / days, 2),
               "wasted_min_per_day": round(float((tau * reneged).sum()) / days, 2)}
        for key, col in (("T", EST_TICKETS), ("C", EST_COUNT), ("L", EST_LES)):
            w = walk[:, col]
            false_balk = waits & (v <= tau) & (tau < w)
            row.update({
                f"{key}_overstates": round(float(np.mean(w[waits] > v[waits])), 4),
                f"{key}_mean_error": round(float(np.mean(w[waits] - v[waits])), 3),
                f"{key}_mae": round(float(np.mean(np.abs(w[waits] - v[waits]))), 3),
                f"{key}_false_balks_per_day": round(false_balk.sum() / days, 3),
                f"{key}_ontime_false_balks_per_day": round(
                    (false_balk & (v <= THRESHOLD)).sum() / days, 3),
                # Post hoc: false balkers who would have been served late (failures anyway)
                f"{key}_late_false_balks_per_day": round(
                    (false_balk & (v > THRESHOLD)).sum() / days, 3),
                # Renegers the display would have sent home at once
                f"{key}_time_saved_min_per_day": round(
                    float((tau * (reneged & (tau < w))).sum()) / days, 2)})
        rows.append(row)
        print(f"  {office:<12} {pname:<6} {kind:<7} overstates T {row['T_overstates']:.2f} "
              f"C {row['C_overstates']:.2f} L {row['L_overstates']:.2f}; on-time false balks/day "
              f"T {row['T_ontime_false_balks_per_day']:.2f} C {row['C_ontime_false_balks_per_day']:.2f} "
              f"L {row['L_ontime_false_balks_per_day']:.2f}")
    write_csv("e15a_display_accuracy.csv", rows)


def run_e15b():
    """H49-H52: the five regimes on the evaluation days (common random numbers)."""
    from abandonment import score
    print("E15b: hidden queue, three wait displays, and the visible line")
    jobs = []
    for office, pname, kind, plan in E13_SETTINGS:
        for regime, extra in E15_REGIMES.items():
            jobs.append((office, pname, kind, plan, regime, extra))

    def one(job):
        office, pname, kind, plan, regime, extra = job
        rates, s = _e13_office(office)
        mean, dist, cv = E13_PATIENCE[pname]
        return score(plan, rates, s, THRESHOLD, **_patience_kw("renege", mean, dist, cv), **extra)

    rows = []
    for (office, pname, kind, plan, regime, _), ev in zip(jobs, pmap(one, jobs)):
        served = ev.arrivals_per_day - ev.abandoned_per_day
        rows.append({"office": office, "patience": pname, "plan_type": kind,
                     "staff_hours": sum(plan), "regime": regime,
                     "overall_fail": round(ev.overall_fail, 5),
                     "failures_per_day": round(ev.overall_fail * ev.arrivals_per_day, 3),
                     "worst_fail": round(ev.worst("fail"), 4),
                     "fail_misses": ev.misses("fail", ALPHA),
                     "overall_abandon": round(ev.overall_abandon, 5),
                     "left_per_day": round(ev.abandoned_per_day, 3),
                     "balked_per_day": round(ev.balked_per_day, 3),
                     "wasted_min_per_day": round(ev.wasted_minutes_per_day, 2),
                     "mean_wait_served": round(ev.mean_wait_served, 3),
                     "lost_min_per_arrival": round((ev.mean_wait_served * served
                                                    + ev.wasted_minutes_per_day)
                                                   / ev.arrivals_per_day, 3),
                     "arrivals_per_day": round(ev.arrivals_per_day, 2)})
    write_csv("e15b_regimes.csv", rows)
    for office, pname, kind, _ in E13_SETTINGS:
        sub = {r["regime"]: r for r in rows if (r["office"], r["patience"], r["plan_type"])
               == (office, pname, kind)}
        print(f"  {office:<12} {pname:<6} {kind:<7} fail " + "  ".join(
            f"{k} {sub[k]['overall_fail']:.3f}" for k in E15_REGIMES) + " | wasted min/day " +
            "  ".join(f"{k} {sub[k]['wasted_min_per_day']:.0f}" for k in E15_REGIMES))



def run_e15c():
    """Supplementary (post hoc): paired day-level 95% CIs for the regime differences."""
    from optimizer import run_simulation
    print("E15c: paired differences in failures per day (1,000 evaluation days)")
    rows = []
    for office, pname, kind, plan in E13_SETTINGS:
        rates, s = _e13_office(office)
        mean, dist, cv = E13_PATIENCE[pname]
        daily = {}
        for regime, extra in E15_REGIMES.items():
            r = run_simulation(plan, rates, replications=EVAL_REPS, seed=EVAL_SEED,
                               mean_service=s, wait_threshold=THRESHOLD,
                               **_patience_kw("renege", mean, dist, cv), **extra)
            daily[regime] = (np.array(r.daily_late).sum(axis=1)
                             + np.array(r.daily_abandoned).sum(axis=1))
        for a, b in (("T", "H"), ("C", "H"), ("L", "H"), ("V", "C"), ("V", "H")):
            d = daily[a] - daily[b]
            half = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
            rows.append({"office": office, "patience": pname, "plan_type": kind,
                         "contrast": f"{a}-{b}", "mean": round(float(d.mean()), 3),
                         "ci_low": round(float(d.mean() - half), 3),
                         "ci_high": round(float(d.mean() + half), 3),
                         "significant": bool(abs(d.mean()) > half)})
        print(f"  {office:<12} {pname:<6} {kind:<7} " + "  ".join(
            f"{r['contrast']} {r['mean']:+.2f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]"
            for r in rows[-5:]))
    write_csv("e15c_paired_contrasts.csv", rows)


# ============================================================================
# Round 13: a theory of wait displays (E16)
# ============================================================================

E16_OFFICES = {"R4_S8_A0.6": (4.0, 8.0), "R16_S16_A0.6": (16.0, 16.0)}   # (Erlangs, S)
E16_PATIENCE = {"exp30": (30.0, "exp", 1.0), "logn30": (30.0, "lognormal", 0.5),
                "logn30cv15": (30.0, "lognormal", 1.5)}
E16_CUTOFFS = [5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 30.0]
E16_KAPPAS = [1.5, 2.0, 3.0]
E16_LOG_SEED = 950_000


def _e16_truth(pname):
    from learning import Patience
    mean, dist, cv = E16_PATIENCE[pname]
    return Patience("exp" if dist == "exp" else "lognormal", mean, cv)


def _e16_settings():
    """12 offices not used before; plans fixed by rule (lean: SIPP-G at 0.20;
    safe: Lag-SIPP-G at 0.10, Round 11b's safe rule)."""
    from learning import sipp_g
    from staffing_methods import lagged_rates
    out = []
    for office, (load, s) in E16_OFFICES.items():
        rates = arrival_profile(load, s, 0.6)
        for pname in E16_PATIENCE:
            truth = _e16_truth(pname)
            out.append((office, pname, "lean", rates, s,
                        sipp_g(rates, s, THRESHOLD, 0.20, truth)))
            out.append((office, pname, "safe", rates, s,
                        sipp_g(lagged_rates(rates, s), s, THRESHOLD, 0.10, truth)))
    return out


def _e16_regimes():
    """name -> (simulator keywords, theory display or None, (predictor, kappa, cutoff))."""
    from displays import HIDDEN, cutoff, scaled
    reg = {"H": ({}, HIDDEN, None)}
    for k in E16_KAPPAS:
        reg[f"O*{k:g}"] = ({"announce": "oracle", "display_scale": k}, scaled(k),
                           ("oracle", k, math.inf))
    for m in E16_CUTOFFS:
        reg[f"O-M{m:g}"] = ({"announce": "oracle", "display_cutoff": [m]}, cutoff(m),
                            ("oracle", 1.0, m))
    reg["C"] = ({"announce": "count"}, None, ("count", 1.0, math.inf))
    for m in E16_CUTOFFS:
        reg[f"C0-M{m:g}"] = ({"announce": "count", "display_scale": 0.0, "display_cutoff": [m]},
                             None, ("count", 0.0, m))
        reg[f"C1-M{m:g}"] = ({"announce": "count", "display_cutoff": [m]}, None,
                             ("count", 1.0, m))
    return reg


def run_e16p():
    """Theory only: the settings, their plans and the predicted failure changes."""
    from displays import plan_prediction
    print("E16p: stationary per-hour predictions for the 12 new settings")
    rows = []
    for office, pname, kind, rates, s, plan in _e16_settings():
        truth = _e16_truth(pname)
        base = plan_prediction(plan, rates, s, truth, THRESHOLD)
        row = {"office": office, "patience": pname, "plan_type": kind,
               "plan": "-".join(map(str, plan)), "staff_hours": sum(plan),
               "H_fail": round(base["fail"], 5)}
        for name, (_, show, _) in _e16_regimes().items():
            if show is None or name == "H":
                continue
            pr = plan_prediction(plan, rates, s, truth, THRESHOLD, show)
            row[f"{name}_dfail"] = round(pr["fail"] - base["fail"], 5)
        row["argmin_cutoff"] = min(E16_CUTOFFS, key=lambda m: row[f"O-M{m:g}_dfail"])
        rows.append(row)
        print(f"  {office:<13} {pname:<10} {kind:<5} {plan} H {base['fail']:.3f} | " + " ".join(
            f"{k[:-6]} {100 * v:+.2f}" for k, v in row.items() if k.endswith("_dfail")))
    write_csv("e16p_predictions.csv", rows)


def run_e16a(reps=200):
    """H53: the exact stationary law under oracle displays, against the simulator."""
    from abandonment import score
    from displays import cutoff, display_hour, scaled
    print("E16a: display theory vs the simulator (constant demand and staffing)")
    cases = []
    for c, s in [(2, 8.0), (8, 16.0), (16, 16.0)]:
        for rho in (0.9, 1.2):
            for pname in E16_PATIENCE:
                for dname, kw, show in [("O*2", {"display_scale": 2.0}, scaled(2.0)),
                                        ("O-M15", {"display_cutoff": [15.0]}, cutoff(15.0)),
                                        ("O-M10", {"display_cutoff": [10.0]}, cutoff(10.0))]:
                    cases.append((c, s, rho, pname, dname, kw, show))

    def one(case):
        c, s, rho, pname, dname, kw, show = case
        lam = rho * c / s * 60.0
        mean, dist, cv = E16_PATIENCE[pname]
        ev = score([c] * SLOTS, [lam] * SLOTS, s, THRESHOLD, reps=reps, seed=310_000,
                   duration=20000, announce="oracle", **kw,
                   **_patience_kw("renege", mean, dist, cv))
        th = display_hour(c, lam / 60.0, s, _e16_truth(pname), THRESHOLD, show)
        return ev, th

    rows = []
    for case, (ev, th) in zip(cases, pmap(one, cases)):
        c, s, rho, pname, dname, _, _ = case
        ci = ev.fail_ci[7]
        se = (ci[1] - ci[0]) / (2 * 1.96)
        row = {"windows": c, "service_time": s, "rho": rho, "patience": pname, "display": dname,
               "fail_theory": round(th.fail, 5), "fail_sim": round(ev.fail[7], 5),
               "se": round(se, 5), "z": round((ev.fail[7] - th.fail) / se, 2) if se > 0 else 0.0,
               "abandon_theory": round(th.balk + th.renege, 5),
               "abandon_sim": round(ev.abandon[7], 5)}
        row["ok"] = bool(abs(row["z"]) <= 3 or abs(ev.fail[7] - th.fail) <= 0.002)
        rows.append(row)
        print(f"  c={c:<2} rho={rho} {pname:<10} {dname:<6} fail {ev.fail[7]:.4f} vs "
              f"{th.fail:.4f} (z={row['z']:+.2f})")
    write_csv("e16a_theory_validation.csv", rows)
    print(f"  within 3 SE (or 0.002): {sum(r['ok'] for r in rows)} of {len(rows)}; "
          f"max |z| {max(abs(r['z']) for r in rows):.2f}")


def run_e16b():
    """H54-H56: 26 display regimes in the 12 new offices (1,000 evaluation days, CRN)."""
    from displays import plan_prediction
    print("E16b: display regimes in the 12 new offices")
    settings = _e16_settings()
    regimes = _e16_regimes()
    jobs = [(st, name) for st in settings for name in regimes]

    def one(job):
        (office, pname, kind, rates, s, plan), name = job
        mean, dist, cv = E16_PATIENCE[pname]
        r = run_simulation(plan, rates, replications=EVAL_REPS, seed=EVAL_SEED, mean_service=s,
                           wait_threshold=THRESHOLD, **_patience_kw("renege", mean, dist, cv),
                           **regimes[name][0])
        late = np.array(r.daily_late, dtype=float).sum(axis=1)
        aband = np.array(r.daily_abandoned, dtype=float).sum(axis=1)
        served = np.array(r.daily_arrivals, dtype=float).sum(axis=1)
        return {"fail_day": late + aband, "arrivals": served + aband,
                "balked": np.array(r.daily_balked, dtype=float),
                "wasted": np.array(r.daily_abandoned_wait, dtype=float),
                "wait_sum": r.mean_wait * served.sum()}

    out = pmap(one, jobs)
    res = {(st[0], st[1], st[2], name): o for (st, name), o in zip(jobs, out)}
    rows = []
    for office, pname, kind, rates, s, plan in settings:
        truth = _e16_truth(pname)
        base_pred = plan_prediction(plan, rates, s, truth, THRESHOLD)["fail"]
        h = res[(office, pname, kind, "H")]
        o15 = res[(office, pname, kind, "O-M15")]
        hf = h["fail_day"].sum() / h["arrivals"].sum()
        for name, (_, show, _) in regimes.items():
            o = res[(office, pname, kind, name)]
            fail = o["fail_day"].sum() / o["arrivals"].sum()
            row = {"office": office, "patience": pname, "plan_type": kind,
                   "staff_hours": sum(plan), "regime": name,
                   "fail": round(float(fail), 5),
                   "failures_per_day": round(float(o["fail_day"].mean()), 3),
                   "arrivals_per_day": round(float(o["arrivals"].mean()), 2),
                   "balked_per_day": round(float(o["balked"].mean()), 3),
                   "wasted_min_per_day": round(float(o["wasted"].mean()), 2),
                   "lost_min_per_arrival": round(float((o["wait_sum"] + o["wasted"].sum())
                                                       / o["arrivals"].sum()), 3)}
            row.update({"d_C1": "", "d_C1_lo": "", "d_C1_hi": ""})
            refs = [("H", h), ("O-M15", o15)]
            if name.startswith("C0-M"):
                # H56(b): nothing below the cutoff against the count estimate below it
                refs.append(("C1", res[(office, pname, kind, "C1-" + name[3:])]))
            for ref, ro in refs:
                d = o["fail_day"] - ro["fail_day"]
                half = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
                row[f"d_{ref}"] = round(float(d.mean()), 3)
                row[f"d_{ref}_lo"] = round(float(d.mean() - half), 3)
                row[f"d_{ref}_hi"] = round(float(d.mean() + half), 3)
            row["dfail_sim"] = round(float(fail - hf), 5)
            row["dfail_pred"] = (round(plan_prediction(plan, rates, s, truth, THRESHOLD, show)
                                       ["fail"] - base_pred, 5) if show is not None else "")
            rows.append(row)
        sub = [r for r in rows if (r["office"], r["patience"], r["plan_type"])
               == (office, pname, kind)]
        print(f"  {office:<13} {pname:<10} {kind:<5} H {sub[0]['fail']:.3f} | " + " ".join(
            f"{r['regime']} {100 * r['dfail_sim']:+.2f}" for r in sub[1:]))
    write_csv("e16b_regimes.csv", rows)


def run_e16c(days=2000):
    """H55(c): counterfactual false balkers for every display, from the hidden queue's log."""
    from patience_logs import ARRIVAL, BOOKED, CALL, EST_COUNT, PATIENCE, raw_log
    print("E16c: who each display would send home, read from the hidden queue's log")
    rows = []
    for office, pname, kind, rates, s, plan in _e16_settings():
        mean, dist, cv = E16_PATIENCE[pname]
        log = raw_log(plan, rates, s, "renege", mean, dist, cv, days=days, seed=E16_LOG_SEED)
        walk = log[log[:, BOOKED] == 0]
        v = walk[:, CALL] - walk[:, ARRIVAL]
        tau = walk[:, PATIENCE]
        for name, (_, _, spec) in _e16_regimes().items():
            if spec is None:
                continue
            predictor, kappa, m = spec
            est = v if predictor == "oracle" else walk[:, EST_COUNT]
            shown = np.where((est > 0) & (est >= m), np.inf, kappa * est)
            false_balk = (v <= tau) & (tau < shown)
            on_time = false_balk & (v <= THRESHOLD)
            n = int(false_balk.sum())
            rows.append({"office": office, "patience": pname, "plan_type": kind,
                         "regime": name, "false_balks_per_day": round(n / days, 3),
                         "ontime_false_balks_per_day": round(int(on_time.sum()) / days, 3),
                         "ontime_share": round(int(on_time.sum()) / n, 4) if n else ""})
        print(f"  {office:<13} {pname:<10} {kind:<5} done")
    write_csv("e16c_false_balkers.csv", rows)


EXPERIMENTS = {"e1b": run_e1b, "e1": run_e1, "e2": run_e2, "e2b": run_e2b,
               "e3a": run_e3a, "e3b": run_e3b, "e4": run_e4, "e5": run_e5, "e6": run_e6,
               "e7a": run_e7a, "e7": run_e7, "e7c": run_e7c, "e8a": run_e8a, "e8b": run_e8b,
               "e9a": run_e9a, "e9b": run_e9b,
               "e10b": run_e10b, "e10a": run_e10a, "e10c": run_e10c, "e10d": run_e10d,
               "e10e": run_e10e, "e10f": run_e10f, "e10g": run_e10g, "e10h": run_e10h,
               "e11a": run_e11a, "e11b": run_e11b, "e11c": run_e11c, "e11d": run_e11d,
               "e11e": run_e11e, "e12a": run_e12a, "e12b": run_e12b, "e12c": run_e12c,
               "e12d": run_e12d, "e12e": run_e12e,
               "e13a": run_e13a, "e13b": run_e13b, "e13c": run_e13c, "e13d": run_e13d,
               "e13e": run_e13e, "e13f": run_e13f, "e14a": run_e14a, "e14b": run_e14b,
               "e14c": run_e14c, "e14d": run_e14d,
               "e15a": run_e15a, "e15b": run_e15b,
               "e15c": run_e15c, "e16p": run_e16p, "e16a": run_e16a, "e16b": run_e16b,
               "e16c": run_e16c}


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
