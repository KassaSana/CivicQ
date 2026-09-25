"""
Figures for research/REPORT.md, built only from research/results/*.csv
(plus the analytic offered-load curve, which needs no simulation).

    python research/figures.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import arrival_profile, offered_load  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Reference palette (dataviz skill), validated light-mode categorical order.
# Color follows the method everywhere, whatever subset a chart shows.
METHOD_COLOR = {
    "SIPP": "#2a78d6", "Lag-SIPP": "#eb6834", "OL-avg": "#1baf7a",
    "OL-max": "#eda100", "SGS": "#e87ba4", "SGS-UCB": "#008300",
}
METHOD_MARKER = {"SIPP": "o", "Lag-SIPP": "s", "OL-avg": "^", "OL-max": "D",
                 "SGS": "v", "SGS-UCB": "P"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf",
                                                      "#0d366b"])
HOURS = ["8", "9", "10", "11", "12", "1", "2", "3"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9.5, "legend.frameon": False,
    "font.family": ["Segoe UI", "DejaVu Sans"],
})


def load(name):
    return list(csv.DictReader(open(RESULTS / name)))


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote research/figures/{name}")


# ----------------------------------------------------------------------------
def fig_gap_heatmap():
    rows = load("e1_methods.csv")
    methods = ["SIPP", "Lag-SIPP", "OL-avg", "OL-max"]
    loads = sorted({float(r["mean_load"]) for r in rows})
    cols = sorted({(float(r["service_time"]), float(r["amplitude"])) for r in rows})
    fig, axes = plt.subplots(1, len(loads), figsize=(13, 3.2), sharey=True)
    vmax = 25
    for ax, load_ in zip(axes, loads):
        grid = np.zeros((len(methods), len(cols)))
        for r in rows:
            if float(r["mean_load"]) != load_ or r["method"] not in methods:
                continue
            ucb = int(r["staff_hours"]) - int(r["gap_vs_sgs_ucb"])
            j = cols.index((float(r["service_time"]), float(r["amplitude"])))
            grid[methods.index(r["method"]), j] = 100.0 * int(r["gap_vs_sgs_ucb"]) / ucb
        im = ax.imshow(grid, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(methods)):
            for j in range(len(cols)):
                v = grid[i, j]
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 15 else INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"S{s:g}\nA{a:g}" for s, a in cols], fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title(f"Mean load {load_:g} Erlangs")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label("Staff-hours vs SGS-UCB (%)", color=INK_2)
    fig.suptitle("No analytic rule beat simulation; at scale their excess grows with "
                 "service time S (minutes)", x=0.02, ha="left", fontsize=12,
                 fontweight="bold", y=1.04)
    save(fig, "fig1_gap_heatmap.png")


# ----------------------------------------------------------------------------
def fig_hourly(load_=24.0, s=32.0, amp=0.6):
    rows = [r for r in load("e1_methods.csv") if float(r["mean_load"]) == load_
            and float(r["service_time"]) == s and float(r["amplitude"]) == amp]
    show = ["SIPP", "Lag-SIPP", "OL-avg", "SGS-UCB"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(8)
    for r in rows:
        m = r["method"]
        if m not in show:
            continue
        kw = dict(color=METHOD_COLOR[m], marker=METHOD_MARKER[m], lw=2, ms=7,
                  label=f"{m} ({r['staff_hours']} h)")
        ax1.plot(x, json.loads(r["plan"]), **kw)
        ax2.plot(x, np.array(json.loads(r["late_by_hour"])) * 100, **kw)
    ax2.axhline(10, color=INK_2, ls="--", lw=1)
    ax2.text(7.3, 10.4, "target 10%", ha="right", color=INK_2, fontsize=8.5)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(HOURS)
        ax.set_xlabel("Hour of arrival (8AM-4PM)")
    ax1.set_ylabel("Open windows")
    ax1.set_title("Where each rule puts staff")
    ax2.set_ylabel("Arrivals waiting > 15 min (%)")
    ax2.set_title("Resulting service level by hour")
    ax2.set_ylim(bottom=0)
    ax1.legend(loc="upper center", ncol=2, fontsize=8.5)
    fig.suptitle(f"High-lag office (mean load {load_:g}, S = {s:g} min, A = {amp}): "
                 "SIPP overstaffs the opening hour and the 1PM ramp, understaffs 10AM",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig2_hourly.png")


# ----------------------------------------------------------------------------
def fig_offered_load(load_=24.0, s=32.0, amp=0.6):
    rates = arrival_profile(load_, s, amp)
    t, m = offered_load(rates, s)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    step_t = np.repeat(np.arange(9) * 60, 2)[1:-1]
    step_v = np.repeat(np.array(rates) / 60 * s, 2)
    ax.plot(step_t / 60, step_v, color=METHOD_COLOR["SIPP"], lw=2,
            label="SIPP load: hourly rate x S")
    ax.plot(t / 60, m, color=METHOD_COLOR["Lag-SIPP"], lw=2,
            label="Offered load m(t): busy servers with unlimited capacity")
    ax.set_xticks(range(9))
    ax.set_xticklabels(["8AM", "9", "10", "11", "12PM", "1", "2", "3", "4PM"])
    ax.set_ylabel("Load (Erlangs)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.annotate("office opens empty:\nreal load starts at 0", xy=(0.25, m[60]),
                xytext=(0.55, 5), fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("load stays high\nafter demand drops", xy=(2.3, m[int(2.3 * 240)]),
                xytext=(3.0, 38), fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_title(f"Why SIPP misallocates: the real load lags demand by about S = {s:g} min",
                 loc="left")
    save(fig, "fig3_offered_load.png")


# ----------------------------------------------------------------------------
def fig_sensitivity():
    e2b = load("e2b_scale.csv")
    e2 = load("e2_sensitivity.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    loads = sorted({float(r["mean_load"]) for r in e2b})
    x = np.arange(len(loads))
    for k, (rcv, color) in enumerate([(0.1, "#6da7ec"), (0.2, "#1c5cab")]):
        vals = [float(r["extra_pct"]) for r in e2b if float(r["rate_cv"]) == rcv]
        bars = ax1.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color,
                       label=f"daily demand CV {rcv:g}")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"+{v:.0f}%", ha="center",
                     fontsize=8.5, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l:g} Erlangs" for l in loads])
    ax1.set_xlabel("Office size (mean offered load)")
    ax1.set_ylabel("Extra staff-hours needed (%)")
    ax1.set_title("Demand uncertainty costs more in bigger offices")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="x", visible=False)

    svc = [("exp", 1.0), ("lognormal", 0.5), ("lognormal", 1.0), ("lognormal", 1.5)]
    rcvs = [0.0, 0.1, 0.2]
    grid = np.array([[next(float(r["rec_worst_late"]) for r in e2
                           if r["service_dist"] == d and float(r["service_cv"]) == c
                           and float(r["rate_cv"]) == rc) for rc in rcvs] for d, c in svc])
    im = ax2.imshow(grid * 100, cmap=SEQUENTIAL, vmin=0, vmax=22, aspect="auto")
    for i in range(len(svc)):
        for j in range(len(rcvs)):
            v = grid[i, j] * 100
            ax2.text(j, i, f"{v:.2f}%" + (" x" if v > 10 else ""), ha="center", va="center",
                     fontsize=9, color="white" if v > 12 else INK)
    ax2.set_xticks(range(len(rcvs)))
    ax2.set_xticklabels([f"{r:g}" for r in rcvs])
    ax2.set_xlabel("Daily demand CV")
    ax2.set_yticks(range(len(svc)))
    ax2.set_yticklabels(["exponential (CV 1)", "lognormal CV 0.5", "lognormal CV 1.0",
                         "lognormal CV 1.5"])
    ax2.set_title("20 h plan [2,3,3,2,2,3,3,2]: worst-hour late % (x = misses 10%)")
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, shrink=0.85, pad=0.02).set_label("% waiting > 15 min",
                                                              color=INK_2)
    fig.tight_layout()
    save(fig, "fig4_sensitivity.png")


# ----------------------------------------------------------------------------
def fig_methodology():
    e3a = load("e3a_crn.csv")
    e3b = load("e3b_definitions.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
    for k, (metric, label, m) in enumerate([("mean_wait", "mean wait", "SIPP"),
                                            ("p90_wait", "daily P90", "Lag-SIPP")]):
        vals = sorted(float(r["variance_ratio"]) for r in e3a if r["metric"] == metric)
        ax1.scatter(vals, np.full(len(vals), k) + np.linspace(-0.12, 0.12, len(vals)),
                    s=40, color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                    edgecolors=SURFACE, linewidths=1, label=label)
        med = float(np.median(vals))
        ax1.plot([med, med], [k - 0.25, k + 0.25], color=INK, lw=2)
        ax1.text(med, k + 0.3, f"median {med:.0f}x", ha="center", fontsize=8.5)
    ax1.set_xscale("log")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["mean wait", "daily P90"])
    ax1.set_ylim(-0.6, 1.7)
    ax1.set_xlabel("Variance of plan-vs-plan difference: independent / common seeds")
    ax1.set_title("Common random numbers: 20 plan pairs")

    names = [r["definition"] for r in e3b]
    hours = [int(r["staff_hours"]) for r in e3b]
    y = np.arange(len(names))[::-1]
    ax2.barh(y, hours, color="#2a78d6", height=0.55)
    for yi, h, r in zip(y, hours, e3b):
        note = "" if r["naive_holds_on_eval"] == "True" else \
            f"  (naive pick {r['naive_staff_hours']} h failed on fresh seeds)"
        ax2.text(h + 0.2, yi, f"{h} h{note}", va="center", fontsize=8.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("Cheapest validated plan (staff-hours)")
    ax2.set_title("Same office, four service-level definitions")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig5_methodology.png")


# ----------------------------------------------------------------------------
def fig_shifts():
    rows = load("e4_shifts.csv")
    methods = [("SIPP-IP", "SIPP"), ("OL-avg-IP", "OL-avg"), ("SGS-UCB-IP", "SGS-UCB"),
               ("ISS", None)]
    iss_color, iss_marker = "#4a3aa7", "X"   # categorical slot 7: its own entity
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    labels = ["office" if s == "office" else
              s.replace("_A0.6", "").replace("R", "").replace("_S", " E, S") for s in settings]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, menu in zip(axes, ("standard", "flexible")):
        x = np.arange(len(settings))
        for k, (m, color_key) in enumerate(methods):
            vals = []
            for s in settings:
                r = next(r for r in rows if r["setting"] == s and r["menu"] == menu
                         and r["method"] == m)
                bench = int(next(q["requirement_hours"] for q in rows if q["setting"] == s
                                 and q["method"] == "SGS-UCB-IP"))
                vals.append(100.0 * (int(r["paid_hours"]) / bench - 1))
            color = iss_color if color_key is None else METHOD_COLOR[color_key]
            label = "Integrated search (ISS)" if color_key is None else f"{color_key} -> shift IP"
            ax.bar(x + (k - 1.5) * 0.2, vals, 0.18, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"{menu.capitalize()} shift menu"
                     + (" (8h + 4h)" if menu == "standard" else " (8h + 6h + 4h)"))
        ax.axhline(0, color=AXIS, lw=1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Paid hours above hour-by-hour optimum (%)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("The price of shifts: +14-33% paid hours even with integrated search; "
                 "SIPP-then-IP up to +68%", x=0.02, ha="left", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_shifts.png")


# ----------------------------------------------------------------------------
def fig_crossval():
    rows = load("e5_crossval.csv")
    strict = [r for r in rows if r["kind"] == "strict" and r["metric"].startswith("late")]
    bracket = [r for r in rows if r["kind"] == "bracket"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    c = np.array([float(r["civicq"]) for r in strict]) * 100
    w = np.array([float(r["ciw"]) for r in strict]) * 100
    ec = np.array([float(r["se_civicq"]) for r in strict]) * 196
    ew = np.array([float(r["se_ciw"]) for r in strict]) * 196
    ax1.errorbar(w, c, xerr=ew, yerr=ec, fmt="o", ms=5, color=METHOD_COLOR["SIPP"],
                 ecolor=MUTED, elinewidth=0.8, capsize=0)
    lim = max(c.max(), w.max()) * 1.08
    ax1.plot([0, lim], [0, lim], color=INK_2, ls="--", lw=1)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Ciw: arrivals waiting > 15 min (%)")
    ax1.set_ylabel("CivicQ (%)")
    ax1.set_title(f"Constant staffing: {len(strict)} hour-level tests, 0 rejections")

    x = np.arange(len(bracket))
    lo = np.array([float(r["ciw"]) for r in bracket]) * 100
    hi = np.array([float(r["ciw_upper"]) for r in bracket]) * 100
    cv = np.array([float(r["civicq"]) for r in bracket]) * 100
    ax2.vlines(x, lo, hi, color="#86b6ef", lw=6, label="Ciw bounds (non-preemptive to resume)")
    ax2.scatter(x, cv, color=METHOD_COLOR["SIPP"], s=28, zorder=3, label="CivicQ")
    cases = list(dict.fromkeys(r["case"] for r in bracket))
    for case in cases[1:]:
        ax2.axvline(next(i for i, r in enumerate(bracket) if r["case"] == case) - 0.5,
                    color=GRID, lw=1)
    ax2.set_xticks([np.mean([i for i, r in enumerate(bracket) if r["case"] == cs])
                    for cs in cases])
    ax2.set_xticklabels([cs.split(" [")[0].replace(", SGS-UCB plan", "") for cs in cases],
                        fontsize=8.5)
    ax2.set_ylabel("Arrivals waiting > 15 min (%)")
    ax2.set_title("Changing staffing: CivicQ inside Ciw's bounds in 25/25 hours")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig7_crossval.png")


# ----------------------------------------------------------------------------
def fig_appointments():
    rows = [r for r in load("e6_appointments.csv") if float(r["no_show"]) == 0.15]
    colors = {"proportional": "#2a78d6", "flat": "#eb6834", "counter": "#1baf7a"}
    markers = {"proportional": "o", "flat": "s", "counter": "^"}
    names = {"proportional": "proportional to demand", "flat": "flat across the day",
             "counter": "counter-cyclical (quiet hours)"}
    big = [r for r in rows if r["office"] == "R8_S16_A0.6"]
    base = next(r for r in big if float(r["share"]) == 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    for pl in ("proportional", "flat", "counter"):
        pts = [base] + sorted((r for r in big if r["placement"] == pl),
                              key=lambda r: float(r["share"]))
        x = [100 * float(r["share"]) for r in pts]
        ax1.plot(x, [int(r["roster_paid_hours"]) for r in pts], color=colors[pl],
                 marker=markers[pl], lw=2, ms=7, label=f"roster, {names[pl]}")
        ax1.plot(x, [int(r["hourly_window_hours"]) for r in pts], color=colors[pl],
                 lw=1, ls="--", alpha=0.8)
    ax1.text(76, 72, "hour-by-hour optimum\n(dashed, all placements)", fontsize=8.5,
             color=INK_2, va="top", ha="right")
    ax1.set_xticks([0, 25, 50, 75])
    ax1.set_xlabel("Share of demand booked as appointments (%)")
    ax1.set_ylabel("Staff-hours per day")
    ax1.set_ylim(60, 105)
    ax1.set_title("8-Erlang office: booking quiet hours shrinks the shift roster",
                  loc="left")
    ax1.legend(loc="lower left", fontsize=8.5)

    appt = [r for r in load("e6_appointments.csv") if r["appt_late"]]
    for pl in ("proportional", "flat", "counter"):
        pts = [r for r in appt if r["placement"] == pl]
        ax2.scatter([100 * float(r["walkin_late"]) for r in pts],
                    [100 * float(r["appt_late"]) for r in pts], color=colors[pl],
                    marker=markers[pl], s=45, edgecolors=SURFACE, linewidths=1,
                    label=names[pl])
    lim = 5
    ax2.plot([0, lim], [0, lim], color=INK_2, ls="--", lw=1)
    ax2.text(lim * 0.97, lim * 0.93, "equal", ha="right", fontsize=8.5, color=INK_2)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_xlabel("Walk-ins waiting > 15 min (%)")
    ax2.set_ylabel("Appointment holders waiting > 15 min (%)")
    ax2.set_title("Appointment holders wait less in all 26 settings", loc="left")
    ax2.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig8_appointments.png")


def fig_abandonment():
    rows = load("e7_abandonment.csv")
    returns = load("e7c_returns.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1.1]})

    # (a) Staff-hours each target needs, per setting and behaviour
    settings = [(o, p) for o in ("office", "R8_S16_A0.6") for p in ("exp30", "exp60", "logn30")]
    label = {"office": "Office", "R8_S16_A0.6": "8 E"}
    pname = {"exp30": "exp 30", "exp60": "exp 60", "logn30": "logn 30"}
    series = [("SGS-UCB (late)", "served-late target", "#e34948", "v", -0.1),
              ("SGS-UCB (fail)", "failure target", "#008300", "P", 0.0),
              ("SIPP-A (fail)", "SIPP-A (failure target)", "#2a78d6", "o", 0.1)]
    ys, ylabels = [], []
    y = 0
    for office, pat in settings:
        base = next(int(r["staff_hours"]) for r in rows if r["office"] == office
                    and r["patience"] == pat and "no-abandonment" in r["found_by"])
        for mode, shift in (("renege", 0.22), ("balk", -0.22)):
            yy = y + shift
            for method, name, color, marker, dy in series:
                h = next(int(r["staff_hours"]) for r in rows if r["office"] == office
                         and r["patience"] == pat and f"{method}@{mode}" in r["found_by"])
                ax1.scatter(100 * (h / base - 1), yy + dy, color=color, marker=marker, s=40,
                            edgecolors=SURFACE, linewidths=0.8,
                            label=name if (y, mode) == (0, "renege") else None, zorder=3)
            ax1.text(-49, yy, "hidden" if mode == "renege" else "visible", fontsize=7.5,
                     color=MUTED, va="center")
        ys.append(y)
        ylabels.append(f"{label[office]}, {pname[pat]}")
        y += 1
    ax1.axvline(0, color=INK_2, lw=1, ls="--")
    ax1.set_yticks(ys)
    ax1.set_yticklabels(ylabels)
    ax1.invert_yaxis()
    ax1.set_xlim(-50, 20)
    ax1.set_xlabel("Staff-hours vs the no-abandonment optimum (%)")
    ax1.set_title("Ticket-log targets let an office cut staff", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)

    # (b) Same plan, same patience: citizens lost behind a visible vs hidden queue
    colors = {"exp30": "#2a78d6", "exp60": "#eb6834", "logn30": "#1baf7a"}
    names = {"exp30": "exponential, mean 30", "exp60": "exponential, mean 60",
             "logn30": "lognormal CV 0.5, mean 30"}
    pairs = {}
    for r in rows:
        pairs.setdefault((r["office"], r["patience"], r["plan"]), {})[r["mode"]] = r
    for pat in ("exp30", "exp60", "logn30"):
        pts = [(100 * float(v["renege"]["overall_abandon"]),
                float(v["balk"]["overall_abandon"]) / float(v["renege"]["overall_abandon"]))
               for (o, p, _), v in pairs.items() if p == pat]
        ax2.scatter(*zip(*pts), color=colors[pat], s=40, edgecolors=SURFACE, linewidths=0.8,
                    label=names[pat])
    ax2.axhline(1, color=INK_2, ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Citizens lost, hidden queue (%, log scale)")
    ax2.set_ylabel("Lost at visible queue / lost at hidden queue")
    ax2.text(0.98, 0.97, "visible queue loses more", transform=ax2.transAxes, ha="right",
             va="top", fontsize=8.5, color=INK_2)
    ax2.text(0.98, 0.03, "visible queue loses fewer", transform=ax2.transAxes, ha="right",
             va="bottom", fontsize=8.5, color=INK_2)
    ax2.set_title("Which queue loses more depends on patience", loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=8)

    # (c) Office, served-late plan, hidden queue: failure by hour with returns
    sel = [r for r in returns if r["office"] == "office" and r["mode"] == "renege"
           and r["plan_from"] == "SGS-UCB (late)" and r["stable"] == "True"]
    base_row = next(r for r in rows if r["office"] == "office" and r["patience"] == "exp30"
                    and r["mode"] == "renege" and "SGS-UCB (late)@renege" in r["found_by"])
    x = np.arange(8)
    ax3.plot(x, 100 * np.array(json.loads(base_row["fail_by_hour"])), color=INK_2, marker="o",
             lw=1.6, ms=5, label="no returns")
    for r, color, ls in zip(sorted(sel, key=lambda r: r["timing"]), ("#e34948", "#eda100"),
                            ("-", "--")):
        who = "return at opening" if r["timing"] == "opening" else "return any time"
        ax3.plot(x, 100 * np.array(json.loads(r["fail_by_hour"])), color=color, ls=ls,
                 marker="o", lw=1.8, ms=5,
                 label=f"{who} ({float(r['repeat_visits_per_100']):.0f} repeat visits/100)")
    ax3.plot(x, 100 * np.array(json.loads(base_row["served_late_by_hour"])), color=MUTED,
             ls=":", lw=1.4, label="ticket log (served-late), no returns")
    ax3.axhline(10, color=INK_2, lw=1, ls="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels(HOURS)
    ax3.set_xlabel("Arrival hour")
    ax3.set_ylabel("Citizens late or lost (%)")
    ax3.set_title("Office on the served-late plan, hidden queue", loc="left")
    ax3.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig9_abandonment.png")


def fig_regimes():
    from staffing_methods import OFFICE_RATES
    cross = load("e8a_crossover.csv")
    fluid = load("e8a_fluid.csv")
    sweep = [r for r in load("e8b_regimes.csv") if float(r["alpha"]) == 0.10]
    e7 = load("e7_abandonment.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) Where abandonment hurts at fixed staffing, and where real hours sit
    colors = {"15.0": "#eb6834", "30.0": "#2a78d6", "60.0": "#1baf7a", "120.0": "#eda100"}
    for pat, color in colors.items():
        pts = sorted((int(r["windows"]), float(r["rho_star"])) for r in cross
                     if r["patience"] == pat)
        ax1.plot(*zip(*pts), color=color, lw=1.8, label=f"crossover, patience exp {float(pat):g}")
    # Curves assume S = 8, so only S = 8 offices are placed on them
    office_plan = next(json.loads(r["plan"]) for r in e7 if r["office"] == "office"
                       and "no-abandonment" in r["found_by"])
    r8_plan = next(json.loads(r["no_abandonment_plan"]) for r in sweep if r["office"] == "R8")
    offices = [(OFFICE_RATES, office_plan, INK, "o", "office's hours (1.5 E)"),
               (arrival_profile(8.0, 8.0, 0.6), r8_plan, "#e34948", "s",
                "8-Erlang office's hours (S = 8)")]
    for rates, plan, color, marker, label in offices:
        ax1.scatter(plan, [r * 8.0 / 60 / c for r, c in zip(rates, plan)], color=color,
                    marker=marker, s=34, edgecolors=SURFACE, linewidths=0.8, zorder=3,
                    label=label)
    ax1.set_xscale("log")
    ax1.set_xticks([1, 2, 4, 10, 20, 40, 80])
    ax1.set_xticklabels(["1", "2", "4", "10", "20", "40", "80"])
    ax1.set_ylim(0.3, 1.2)
    ax1.set_xlabel("Open windows c")
    ax1.set_ylabel("Utilization (offered load / windows)")
    ax1.text(1.1, 1.12, "abandonment lowers failures", fontsize=8.5, color=INK_2)
    ax1.text(12, 0.62, "abandonment raises failures", fontsize=8.5, color=INK_2)
    ax1.set_title("Early leavers vs queue thinning, per hour", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5)

    # (b) Staffing per unit of load converges to the fluid limit
    limit = fluid[-1]
    rows = [r for r in fluid if r["load"] != "fluid limit"]
    series = [("erlang_c", "Erlang-C (no abandonment)", INK_2, "-"),
              ("renege_exp30", "exp 30", "#2a78d6", "-"),
              ("balk_lognormal30", "lognormal 30 (visible queue)", "#1baf7a", "-"),
              ("balk_lognormal60", "lognormal 60 (visible queue)", "#eda100", "-"),
              ("renege_exp30_returns0.5", "exp 30, half return", "#2a78d6", "--"),
              ("renege_exp30_returns1", "exp 30, all return", "#2a78d6", ":")]
    for key, label, color, ls in series:
        pts = [(float(r["load"]), float(r[key])) for r in rows if r.get(key)]
        ax2.plot(*zip(*pts), color=color, ls=ls, marker="o", ms=3.5, lw=1.6, label=label)
        ax2.axhline(float(limit[key]), color=color, ls=ls, lw=0.8, alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_ylim(0.85, 1.25)
    ax2.set_xlabel("Offered load (Erlangs, log scale)")
    ax2.set_ylabel("Windows needed / offered load")
    ax2.set_title("Large offices approach the fluid limit (thin lines)", loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5)

    # (c) The time-varying office sweep
    names = {"exp30": ("exponential, mean 30", "#2a78d6", 0.10),
             "logn60": ("lognormal CV 0.5, mean 60", "#eda100", 0.0035)}
    for pat, (label, color, d) in names.items():
        pts = sorted((float(r["mean_load"]), 100 * float(r["saving"])) for r in sweep
                     if r["patience"] == pat)
        ax3.plot(*zip(*pts), color=color, marker="o", lw=1.8, ms=5, label=label)
        ax3.axhline(100 * d, color=color, lw=0.8, ls="--", alpha=0.7)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks([1, 2, 4, 8, 16, 32])
    ax3.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax3.set_xlabel("Mean offered load (Erlangs)")
    ax3.set_ylabel("Staff-hours saved by abandonment (%)")
    ax3.set_title("Time-varying offices (dashed: fluid limit)", loc="left")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=8)
    fig.tight_layout()
    save(fig, "fig10_regimes.png")


def fig_log_regime():
    slack = [r for r in load("e9a_log_slack.csv") if r["delta_star"]]
    order = load("e9b_fluid_order.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))
    ratios = sorted({float(r["service_time"]) / float(r["threshold"]) for r in slack})
    shade = {q: SEQUENTIAL(0.15 + 0.85 * i / (len(ratios) - 1)) for i, q in enumerate(ratios)}
    marker = {"30.0": "o", "120.0": "s"}

    # (a) Every case collapses onto 1/2 ln c once the patience intercept is removed
    for key in sorted({(r["service_time"], r["threshold"], r["patience"]) for r in slack}):
        pts = sorted((int(r["windows"]), float(r["scaled_delta"]) + np.log(float(r["k1"])),
                      float(r["beta_star"])) for r in slack
                     if (r["service_time"], r["threshold"], r["patience"]) == key
                     and int(r["windows"]) >= 50)     # Below that some cases lose the crossover
        q = float(key[0]) / float(key[1])
        c, y, beta = zip(*pts)
        ax1.plot(c, y, color=shade[q], lw=1.0, alpha=0.8)
        ax1.scatter([a for a, b in zip(c, beta) if b < 0.15],
                    [v for v, b in zip(y, beta) if b < 0.15], color=shade[q], s=12,
                    marker=marker[key[2]], zorder=3)
    grid = np.array([50, 10000])
    ax1.plot(grid, 0.5 * np.log(grid), color="#e34948", lw=2, ls="--",
             label="theory: ½ ln c")
    ax1.set_xscale("log")
    ax1.set_xlabel("Open windows c (log scale)")
    ax1.set_ylabel("δ*·T/S + ln K₁")
    ax1.set_title("Crossover slack grows like ln c", loc="left")
    ax1.legend(loc="upper left", fontsize=8)

    # (b) alpha* sqrt(c) approaches K1 from below; slower when S/T is large
    for key in sorted({(r["service_time"], r["threshold"], r["patience"]) for r in slack}):
        pts = sorted((int(r["windows"]), float(r["alpha_star_sqrt_c"]) / float(r["k1"]))
                     for r in slack if (r["service_time"], r["threshold"], r["patience"]) == key)
        q = float(key[0]) / float(key[1])
        ax2.plot(*zip(*pts), color=shade[q], lw=1.2, marker=marker[key[2]], ms=3)
    ax2.axhline(1.0, color="#e34948", lw=1.5, ls="--")
    for q in (ratios[0], ratios[-1]):
        ax2.plot([], [], color=shade[q], lw=2, label=f"S/T = {q:.2g}")
    ax2.plot([], [], color=MUTED, marker="o", ls="", label="patience mean 30")
    ax2.plot([], [], color=MUTED, marker="s", ls="", label="patience mean 120")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("Open windows c (log scale)")
    ax2.set_ylabel("α*(c)·√c / K₁")
    ax2.set_title("Walk-out boundary α* → K₁/√c", loc="left")
    ax2.legend(loc="lower right", fontsize=7.5)

    # (c) Three orders of the correction above the fluid staffing
    styles = {"exp30": ("G(T) = 0.39 > α (mean 30)", "#2a78d6"),
              "exp120": ("G(T) = 0.118, just above α (mean 120)", "#1baf7a"),
              "exp_kink": ("G(T) = α (mean 142)", "#eda100"),
              "exp300": ("G(T) = 0.049 < α (mean 300)", "#e34948")}
    for name, (label, color) in styles.items():
        pts = [(float(r["load"]), float(r["excess"])) for r in order if r["patience"] == name]
        ax3.plot(*zip(*pts), color=color, marker="o", ms=4, lw=1.6, label=label)
    loads = np.array([50, 5000])
    ax3.plot(loads, 0.26 * np.sqrt(loads), color=MUTED, ls=":", lw=1.2,
             label="predicted 0.26 √R")
    ax3.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Offered load R (Erlangs, log scale)")
    ax3.set_ylabel("Windows above fluid staffing (1 − d)R")
    ax3.set_title("Extra windows above the fluid limit", loc="left")
    ax3.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig11_log_regime.png")


def fig_fluid_day(load_=24.0, s=32.0, amp=0.6):
    from fluid import fluid_day
    fluid = next(r for r in load("e10b_fluid_constants.csv") if r["shape"] == "double"
                 and float(r["amplitude"]) == amp and float(r["service_time"]) == s
                 and float(r["threshold"]) == 15.0)
    e1 = [r for r in load("e1_methods.csv") if float(r["mean_load"]) == load_
          and float(r["service_time"]) == s and float(r["amplitude"]) == amp]
    rates = arrival_profile(load_, s, amp)
    plan = [x * load_ for x in json.loads(fluid["f_alpha_plan"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.9))
    x = np.arange(8)
    ax1.bar(x, [r / 60 * s for r in rates], color=GRID, width=0.8, label="offered load λᵢS")
    for m in ("SIPP", "SGS-UCB"):
        r = next(r for r in e1 if r["method"] == m)
        ax1.plot(x, json.loads(r["plan"]), color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                 lw=2, ms=6, label=f"{m} ({r['staff_hours']} h)")
    ax1.plot(x, plan, color=INK, lw=2, ls="--", marker="o", ms=4,
             label=f"fluid optimum ({sum(plan):.0f} h)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(HOURS)
    ax1.set_xlabel("Hour (8AM-4PM)")
    ax1.set_ylabel("Open windows")
    ax1.set_title("The fluid plan tracks simulation, not SIPP", loc="left")
    ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    day = fluid_day(plan, rates, s, 15.0)            # On the grid the plan was solved on
    c = np.array(plan)[np.minimum((day["t"] // 60).astype(int), 7)]
    ax2.plot(day["t"] / 60, day["X"], color="#2a78d6", lw=1.8, label="in office X(t)")
    ax2.plot(day["t"] / 60, c, color=INK, lw=1.2, ls="--", label="open windows c(t)")
    ax2.set_xlabel("Hours after opening")
    ax2.set_ylabel("Citizens / windows")
    tw = ax2.twinx()
    tw.plot(day["t"] / 60, day["wait"], color="#e34948", lw=1.5, label="fluid wait w(t)")
    tw.axhline(15, color="#e34948", lw=0.8, ls=":")
    tw.set_ylabel("Wait (min)", color="#e34948")
    tw.set_ylim(0, 30)
    tw.grid(False)
    ax2.set_title("Backlog carried through the peaks (≤ α late per hour)", loc="left")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig12_fluid_day.png")


def fig_fluid_scaling():
    fluid = next(r for r in load("e10b_fluid_constants.csv") if r["shape"] == "double"
                 and float(r["amplitude"]) == 0.6 and float(r["service_time"]) == 8.0
                 and float(r["threshold"]) == 15.0)
    f = float(fluid["f_alpha"])
    pts = [(float(r["mean_load"]), int(r["no_abandonment_hours"]))
           for r in load("e8b_regimes.csv")
           if r["patience"] == "exp30" and float(r["alpha"]) == 0.10 and r["office"] != "office"]
    pts += [(float(r["mean_load"]), int(r["staff_hours"])) for r in load("e10c_scaling.csv")]
    pts += [(float(r["mean_load"]), int(r["staff_hours"])) for r in load("e10g_confirm.csv")]
    pts = sorted(set(pts))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.9))
    R = np.array([p[0] for p in pts])
    hours = np.array([p[1] for p in pts])
    ax1.plot(R, hours / (8 * R), color=METHOD_COLOR["SGS-UCB"], marker="P", ms=7, lw=2,
             label="SGS-UCB (simulation)")
    f_np = next(float(r["f0_np"]) for r in load("e10f_fluid_nonpreemptive.csv")
                if r["shape"] == "double" and float(r["amplitude"]) == 0.6
                and float(r["service_time"]) == 8.0 and float(r["threshold"]) == 15.0
                and r["menu"] == "hourly")
    grid = np.geomspace(1, 256, 100)
    e128 = hours[R == 128][0] - 8 * 128 * f_np
    ax1.plot(grid, f_np + e128 * np.sqrt(grid / 128) / (8 * grid), color=INK_2, ls="--",
             lw=1.2, label="corrected fluid + κ√R windows")
    ax1.axhline(f, color=MUTED, lw=1.2, ls=":", label=f"registered fluid f = {f:.3f}")
    ax1.axhline(f_np, color=INK, lw=1.2, label=f"corrected fluid f = {f_np:.3f}")
    ax1.axhline(1.0, color=GRID, lw=1)
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(0.88, 1.6)
    ax1.set_xlabel("Mean offered load R (Erlangs)")
    ax1.set_ylabel("Window-hours / raw workload 8R")
    ax1.set_title("Large offices converge to the fluid, below the workload", loc="left")
    ax1.legend(fontsize=8)

    rows = load("e10e_rosters.csv")
    labels = [f"{r['menu'][:4]} R{float(r['mean_load']):g} S{float(r['service_time']):g}"
              for r in rows]
    y = np.arange(len(rows))
    npf = load("e10f_fluid_nonpreemptive.csv")

    def corrected_price(r):
        cost = {m: next(float(x["cost_per_erlang"]) for x in npf if x["shape"] == "double"
                        and float(x["amplitude"]) == 0.6 and x["menu"] == m
                        and float(x["service_time"]) == float(r["service_time"])
                        and float(x["threshold"]) == 15.0) for m in (r["menu"], "hourly")}
        return 100 * (cost[r["menu"]] / cost["hourly"] - 1)

    ax2.barh(y - 0.27, [float(r["measured_price_of_shifts_pct"]) for r in rows], height=0.27,
             color="#2a78d6", label="measured: ISS vs SGS-UCB")
    ax2.barh(y, [corrected_price(r) for r in rows], height=0.27, color=INK,
             label="corrected fluid (post hoc)")
    ax2.barh(y + 0.27, [float(r["fluid_price_of_shifts_pct"]) for r in rows], height=0.27,
             color=AXIS, label="registered fluid")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Price of shifts (% more paid hours)")
    ax2.set_title("The fluid predicts the price of shifts", loc="left")
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    save(fig, "fig13_fluid_scaling.png")


def fig_paid_overtime():
    fluid = load("e11a_fluid_paid.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))
    services = [4.0, 8.0, 16.0, 32.0]

    # (a) Paying for unpaid work removes the "below the workload" effect
    styles = {0.0: ("overtime free", "#e34948", "o"), 1.0: ("paid (κ = 1)", INK, "s"),
              1.5: ("time and a half (κ = 1.5)", "#2a78d6", "^")}
    for kappa, (label, color, marker) in styles.items():
        ys = [next(float(r["f_paid"]) for r in fluid if r["shape"] == "double"
                   and float(r["service_time"]) == s and float(r["threshold"]) == 15.0
                   and float(r["kappa"]) == kappa) for s in services]
        ax1.plot(services, ys, color=color, marker=marker, lw=2, label=label)
    ax1.axhline(1.0, color=GRID, lw=1)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(services)
    ax1.set_xticklabels(["4", "8", "16", "32"])
    ax1.set_xlabel("Mean service time S (min)")
    ax1.set_ylabel("Fluid cost / raw workload")
    ax1.set_title("Below-workload staffing was unpaid work", loc="left")
    ax1.legend(fontsize=8)

    # (b) SIPP's excess: free vs paid overtime, at fixed T and at fixed T/S
    def sipp_excess(rows, key):
        out = []
        for s in services:
            sub = [r for r in rows if float(r["service_time"]) == s and key(r)]
            out.append(next(float(r["excess_pct"]) for r in sub if r["method"] == "SIPP"))
        return out

    e1 = load("e1_methods.csv")
    free_t15 = []
    for s in services:
        sub = {r["method"]: int(r["staff_hours"]) for r in e1 if float(r["mean_load"]) == 24.0
               and float(r["service_time"]) == s and float(r["amplitude"]) == 0.6}
        free_t15.append(100 * (sub["SIPP"] / sub["SGS-UCB"] - 1))
    free_ratio = [next(float(r["gap_vs_ucb_pct"]) for r in load("e10a_threshold_ratio.csv")
                       if float(r["service_time"]) == s and r["method"] == "SIPP")
                  for s in services]
    paid_t15 = sipp_excess(load("e11c_paid_e1.csv"), lambda r: True)
    paid_ratio = sipp_excess(load("e11b_paid_threshold_ratio.csv"),
                             lambda r: float(r["kappa"]) == 1.0)
    for ys, label, color, ls in [(free_t15, "T = 15, overtime free", "#e34948", "--"),
                                 (paid_t15, "T = 15, paid", "#e34948", "-"),
                                 (free_ratio, "T = 1.875 S, overtime free", "#2a78d6", "--"),
                                 (paid_ratio, "T = 1.875 S, paid", "#2a78d6", "-")]:
        ax2.plot(services, ys, color=color, ls=ls, marker="o", ms=4, lw=1.8, label=label)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(services)
    ax2.set_xticklabels(["4", "8", "16", "32"])
    ax2.set_xlabel("Mean service time S (min)")
    ax2.set_ylabel("SIPP's excess over the optimum (%)")
    ax2.set_title("SIPP's excess still grows with S when paid", loc="left")
    ax2.legend(fontsize=8)

    # (c) Excess over the fluid: grows like sqrt(R) when overtime is free, flat when paid
    f_np = next(float(r["f0_np"]) for r in load("e10f_fluid_nonpreemptive.csv")
                if r["shape"] == "double" and float(r["amplitude"]) == 0.6
                and float(r["service_time"]) == 8.0 and float(r["threshold"]) == 15.0
                and r["menu"] == "hourly")
    free = [(float(r["mean_load"]), int(r["no_abandonment_hours"]))
            for r in load("e8b_regimes.csv") if r["patience"] == "exp30"
            and float(r["alpha"]) == 0.10 and r["office"] != "office"
            and float(r["mean_load"]) >= 4]
    free += [(float(r["mean_load"]), int(r["staff_hours"])) for r in load("e10c_scaling.csv")]
    free += [(float(r["mean_load"]), int(r["staff_hours"])) for r in load("e10g_confirm.csv")]
    free = sorted(set(free))
    ax3.plot([R for R, _ in free], [h - 8 * R * f_np for R, h in free], color="#e34948",
             marker="o", lw=1.8, label="overtime free (Round 7)")
    f1 = next(float(r["f_paid"]) for r in fluid if r["shape"] == "double"
              and float(r["service_time"]) == 8.0 and float(r["threshold"]) == 15.0
              and float(r["kappa"]) == 1.0)
    paid = [(float(r["mean_load"]), float(r["paid_cost"]))
            for name in ("e11d_paid_scaling.csv", "e11e_paid_confirm.csv")
            for r in load(name) if r["method"] == "paid optimum"]
    paid = sorted(paid)
    ax3.plot([R for R, _ in paid], [c - 8 * R * f1 for R, c in paid], color=INK, marker="s",
             lw=1.8, label="paid overtime (Round 8)")
    ax3.axhline(8 * 8.0 / 15.0 * np.log(10), color=MUTED, ls=":", lw=1.2,
                label="8 (S/T) ln(1/α) = 9.8")
    ax3.set_xscale("log", base=2)
    ax3.set_xlabel("Mean offered load R (Erlangs)")
    ax3.set_ylabel("Optimum − 8R × fluid constant (window-hours)")
    ax3.set_title("Excess over the fluid: free vs paid overtime", loc="left")
    ax3.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig14_paid_overtime.png")


# ----------------------------------------------------------------------------
def fig_tipping():
    """Round 9: return curves, steady states and recovery times (fluid vs simulation)."""
    from experiments import E12_PATIENCE, E12_RATES, E12_S
    from tipping import fluid_return_curve
    fresh = sum(E12_RATES)
    curves = load("e12c_return_curves.csv")
    fl = load("e12b_fluid_returns.csv")
    chains = load("e12d_chains.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) h(R) = L(R) - R, simulated (band) vs fluid (line), returns spread over the day
    colors = {"phi=1.00": "#2a78d6", "phi=0.90": "#1baf7a", "phi=0.85": "#eda100",
              "phi=0.80": "#e34948"}
    grid = np.linspace(0.0, 3 * fresh, 61)
    for name, color in colors.items():
        sub = [r for r in curves if r["plan_name"] == name and r["timing"] == "profile"]
        R = np.array([float(r["R"]) for r in sub]) / fresh
        ax1.fill_between(R, [float(r["h_low"]) / fresh for r in sub],
                         [float(r["h_high"]) / fresh for r in sub], color=color, alpha=0.25, lw=0)
        ax1.plot(R, [float(r["h"]) / fresh for r in sub], color=color, marker="o", ms=3, lw=1.5,
                 label=f"{name.replace('phi', 'φ')} ({sub[0]['window_hours']} h)")
        plan = json.loads(sub[0]["plan"])
        h = fluid_return_curve(plan, E12_RATES, E12_S, E12_PATIENCE, 1.0, "profile", grid)
        ax1.plot(grid / fresh, np.array(h) / fresh, color=color, ls="--", lw=1)
    ax1.axhline(0.0, color=INK_2, lw=0.8)
    ax1.set_xlabel("Returners per day R / fresh demand")
    ax1.set_ylabel("h(R) = L(R) − R, per fresh citizen")
    ax1.set_title("One crossing: no tipping point", loc="left")
    ax1.legend(fontsize=8, title="simulated (dots), fluid (dashed)", title_fontsize=8)

    # (b) Steady state R* against staffing
    for timing, color, marker in [("profile", INK, "o"), ("opening", "#e34948", "s")]:
        fsub = sorted([r for r in fl if r["timing"] == timing and r["plan_name"] != "late plan"],
                      key=lambda r: int(r["window_hours"]))
        xs = [int(r["window_hours"]) for r in fsub]
        ys = [min(float(r["repeat_per_100"]), 1e4) for r in fsub]
        ax2.plot(xs, ys, color=color, ls="--", lw=1.2)
        pts = {}
        for r in curves:
            if r["timing"] == timing and r["plan_name"] != "late plan":
                pts[int(r["window_hours"])] = (100 * float(r["sim_R"]) / fresh
                                               if r["sim_R"] else None)
        xs2 = sorted(pts)
        ax2.plot([x for x in xs2 if pts[x]], [pts[x] for x in xs2 if pts[x]], color=color,
                 marker=marker, lw=0, ms=6,
                 label=f"returns {'spread over the day' if timing == 'profile' else 'at opening'}")
        for x in xs2:
            if pts[x] is None:
                ax2.annotate("none ≤ 3×", (x, 300), color=color, fontsize=7, ha="center",
                             xytext=(0, 4), textcoords="offset points")
                ax2.plot([x], [300], color=color, marker="^", ms=6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Window-hours (E7 fail plan × φ)")
    ax2.set_ylabel("Steady-state repeat visits per 100")
    ax2.set_title("The steady state diverges smoothly", loc="left")
    ax2.legend(fontsize=8, title="simulated (markers), fluid (dashed)", title_fontsize=8)

    # (c) Recovery after one closure day: critical slowing down
    for timing, color, marker in [("profile", INK, "o"), ("opening", "#e34948", "s")]:
        fsub = sorted([r for r in fl if r["timing"] == timing and r["plan_name"] != "late plan"
                       and r["recovery_days"] not in ("", "2000")],
                      key=lambda r: int(r["window_hours"]))
        ax3.plot([int(r["window_hours"]) for r in fsub],
                 [float(r["recovery_days"]) for r in fsub], color=color, ls="--", lw=1.2)
        xs, med, lo, hi = [], [], [], []
        for name in sorted({r["plan_name"] for r in chains if r["plan_name"] != "late plan"}):
            sub = [r for r in chains if r["plan_name"] == name and r["timing"] == timing]
            rec = [float(r["recovery_days"]) for r in sub if r["recovery_days"] != ""]
            if len(rec) == len(sub) and rec:
                xs.append(int(sub[0]["window_hours"]))
                med.append(np.median(rec))
                lo.append(np.percentile(rec, 10))
                hi.append(np.percentile(rec, 90))
        if not xs:              # no case where every chain recovered
            continue
        order = np.argsort(xs)
        xs, med = np.array(xs)[order], np.array(med)[order]
        lo, hi = np.array(lo)[order], np.array(hi)[order]
        ax3.errorbar(xs, med, yerr=[med - lo, hi - med], color=color, marker=marker, lw=0,
                     elinewidth=1, capsize=3, ms=6,
                     label=f"returns {'spread over the day' if timing == 'profile' else 'at opening'}")
    ax3.set_yscale("log")
    ax3.set_xlabel("Window-hours (E7 fail plan × φ)")
    ax3.set_ylabel("Days to recover from one closure day")
    ax3.set_title("Critical slowing down near collapse", loc="left")
    ax3.legend(fontsize=8, title="simulated median, 10–90% (markers); fluid (dashed)",
               title_fontsize=8)
    fig.tight_layout()
    save(fig, "fig15_tipping.png")


def fig_return_chains():
    """Round 9: day-to-day returners around one closure day (day 30)."""
    from experiments import E12_RATES
    fresh = sum(E12_RATES)
    chains = load("e12d_chains.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, name, color in zip(axes, ["phi=1.00", "phi=0.90", "phi=0.80"],
                               ["#2a78d6", "#1baf7a", "#e34948"]):
        sub = [r for r in chains if r["plan_name"] == name and r["timing"] == "profile"]
        paths = [np.array(json.loads(r["path"])) / fresh for r in sub]
        n = min(len(p) for p in paths)
        for p in paths:
            ax.plot(np.arange(len(p)), p, color=color, alpha=0.15, lw=0.8)
        ax.plot(np.arange(n), np.median([p[:n] for p in paths], axis=0), color=INK, lw=1.6,
                label="median of 20 chains")
        ax.axvline(30, color=MUTED, ls=":", lw=1)
        ax.set_title(f"{name.replace('phi', 'φ')} ({sub[0]['window_hours']} h)", loc="left")
        ax.set_xlabel("Day (office closed on day 30)")
        ax.set_xlim(0, min(n, 250))
    axes[0].set_ylabel("Returners / fresh demand")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig16_return_chains.png")


def fig_patience_estimates():
    """Round 10: G(t) from the office's own ticket log, 30 and 300 days."""
    curves = load("e13b_curves.csv")
    cases = [("office", "exp30", "Office, exponential patience, lean plan (16 h)"),
             ("R8_S16_A0.6", "logn30", "8 E, lognormal patience, citizen-optimal plan (72 h)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, (office, pname, title) in enumerate(cases):
        for j, days in enumerate((30, 300)):
            ax = axes[i, j]
            sub = [r for r in curves if r["office"] == office and r["patience"] == pname
                   and int(r["days"]) == days]
            t = np.array([float(r["t"]) for r in sub])
            col = lambda k: np.array([float(r[k]) for r in sub])  # noqa: E731
            ax.plot(t, col("truth"), color=INK, lw=2.0, label="truth")
            ax.step(t, col("naive_km"), where="post", color="#e34948", lw=1.4,
                    label="call-center KM (call time as departure)")
            ax.step(t, col("npmle"), where="post", color="#2a78d6", lw=1.4,
                    label="current-status NPMLE")
            ax.plot(t, col("parametric"), color="#1baf7a", lw=1.4, ls="--",
                    label=f"parametric CS-MLE ({sub[0]['family']}, by AIC)")
            ax.axvline(15, color=MUTED, ls=":", lw=1)
            ax.axvspan(float(sub[0]["v_max"]), 60, color=GRID, alpha=0.5, lw=0)
            ax.set_title(f"{title}: {days} days" if j == 0 else f"{days} days", loc="left")
            ax.set_ylim(0, 0.9 if pname == "exp30" else 0.45)
            ax.set_xlim(0, 60)
            if j == 0:
                ax.set_ylabel("G(t) = P(patience < t)")
            if i == 1:
                ax.set_xlabel("Minutes (shaded: beyond the longest wait in the log)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig17_patience_estimates.png")


def fig_patience_learning():
    """Round 10: how fast an office learns G(15), and the family, from its log."""
    rates = load("e13c_rates.csv")
    fam = load("e13d_family.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = {("office", "npmle"): ("#2a78d6", "o", "office, NPMLE"),
              ("office", "exp_mle"): ("#2a78d6", "s", "office, exponential CS-MLE"),
              ("R8_S16_A0.6", "npmle"): ("#e34948", "o", "8 E, NPMLE"),
              ("R8_S16_A0.6", "exp_mle"): ("#e34948", "s", "8 E, exponential CS-MLE")}
    for (office, est), (color, marker, label) in styles.items():
        sub = [r for r in rates if r["office"] == office and r["estimator"] == est]
        n = np.array([int(r["days"]) for r in sub])
        y = np.array([float(r["rmse"]) for r in sub])
        a.loglog(n, y, color=color, marker=marker, lw=1.4,
                 ls="-" if est == "npmle" else "--",
                 label=f"{label} (slope {float(sub[0]['slope']):+.2f})")
    ref = np.array([10, 1000])
    a.loglog(ref, 0.07 * (ref / 10) ** (-1 / 3), color=MUTED, lw=0.8, ls=":")
    a.loglog(ref, 0.03 * (ref / 10) ** (-1 / 2), color=MUTED, lw=0.8, ls=":")
    a.text(1100, 0.07 * 100 ** (-1 / 3), "n^(-1/3)", color=MUTED, fontsize=8, va="center")
    a.text(1100, 0.03 * 100 ** (-1 / 2), "n^(-1/2)", color=MUTED, fontsize=8, va="center")
    a.set_xlabel("Days of ticket log")
    a.set_ylabel("RMSE of G(15)")
    a.set_title("(a) Learning G(15), exponential patience", loc="left")
    a.legend(fontsize=7.5)
    for office, color in (("office", "#2a78d6"), ("R8_S16_A0.6", "#e34948")):
        for kind, ls in (("lean", "--"), ("citizen", "-")):
            sub = [r for r in fam if r["office"] == office and r["patience"] == "exp30"
                   and r["plan_type"] == kind]
            b.plot([int(r["days"]) for r in sub], [float(r["correct"]) for r in sub],
                   color=color, ls=ls, marker="o", ms=3, lw=1.4,
                   label=f"{'office' if office == 'office' else '8 E'}, {kind} plan "
                         f"({sub[0]['staff_hours']} h)")
    b.axhline(0.9, color=MUTED, ls=":", lw=1)
    b.set_xscale("log")
    b.set_ylim(0.7, 1.01)
    b.set_xlabel("Days of ticket log")
    b.set_ylabel("Share of logs where AIC picks exponential")
    b.set_title("(b) Recognising exponential patience (lognormal: ≥ 95% by day 5)", loc="left")
    b.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig18_patience_learning.png")


POLICY_STYLE = {("A", "False"): ("#e34948", "-", "A: exponential fit, Erlang-A SIPP"),
                ("A", "True"): ("#eda100", "--", "A + exploration (p = 0.1, φ = 0.9)"),
                ("B", "False"): ("#2a78d6", "-", "B: AIC family, SIPP-G"),
                ("B+lag", ""): ("#1baf7a", "-", "B, staffing by Lag-SIPP-G (H47)")}


def fig_learning_paths():
    """Round 11: plan staff-hours after each refit, lognormal patience."""
    hist = load("e14c_histories.csv")
    lag = load("e14d_histories.csv")
    lag_sum = load("e14d_summary.csv")
    cases = [("R8", "0.1"), ("R8", "0.2"), ("R32", "0.1"), ("R32", "0.2")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (office, alpha) in zip(axes, cases):
        sub = [r for r in hist if r["office"] == office and r["alpha"] == alpha
               and r["patience"] == "logn30"]
        start = int(sub[0]["hours_run"])
        for (rule, explore), (color, ls, label) in POLICY_STYLE.items():
            if rule == "B+lag":
                rows = [r for r in lag if r["office"] == office and r["alpha"] == alpha
                        and r["patience"] == "logn30"]
            else:
                rows = [r for r in sub if r["rule"] == rule and r["explore"] == explore]
            path = [start] + [np.mean([float(r["new_hours"]) for r in rows if int(r["period"]) == k])
                              for k in range(1, 9)]
            ax.plot(np.arange(0, 9) * 30, path, color=color, ls=ls, marker="o", ms=3, lw=1.5,
                    label=label)
        orc = int(sub[0]["oracle_hours"])
        tgt = next(int(r["lag_sipp_g_true"]) for r in lag_sum if r["office"] == office
                   and r["alpha"] == alpha and r["patience"] == "logn30")
        ax.axhline(orc, color="#2a78d6", ls=":", lw=1)
        ax.axhline(tgt, color="#1baf7a", ls=":", lw=1)
        unsafe = office == "R32"
        ax.text(240, orc, "SIPP-G oracle" + (" (misses!)" if unsafe else ""), fontsize=7.5,
                color="#2a78d6", ha="right", va="bottom")
        ax.text(240, tgt, "Lag-SIPP-G", fontsize=7.5, color="#1baf7a", ha="right", va="top")
        ax.set_title(f"{office[1:]} E, α = {float(alpha):.2f}", loc="left")
        ax.set_xlabel("Days of log")
    axes[0].set_ylabel("Staff-hours per day after refit")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Refitting on the office's own log, starting from Erlang-C SIPP "
                 "(lognormal patience, mean of 10 histories)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig19_learning_paths.png")


def fig_learning_tradeoff():
    """Round 11: final staff-hours against the worst hour's failure rate."""
    summ = load("e14c_summary.csv")
    lag = load("e14d_summary.csv")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
    for ax, (office, alpha) in zip(axes, [("R8", "0.1"), ("R8", "0.2"), ("R32", "0.1"),
                                          ("R32", "0.2")]):
        for pname, marker in (("logn30", "o"), ("exp30", "s")):
            for (rule, explore), (color, _, label) in POLICY_STYLE.items():
                if rule == "B+lag":
                    r = next(x for x in lag if x["office"] == office and x["alpha"] == alpha
                             and x["patience"] == pname)
                else:
                    r = next(x for x in summ if x["office"] == office and x["alpha"] == alpha
                             and x["patience"] == pname and x["rule"] == rule
                             and x["explore"] == explore)
                ax.scatter(float(r["final_mean"]), float(r["worst_fail" if rule == "B+lag"
                                                           else "final_worst_fail"]),
                           color=color, marker=marker, s=40, zorder=3,
                           edgecolor=INK if pname == "logn30" else "none", lw=0.5)
        ax.axhline(float(alpha), color=MUTED, ls=":", lw=1)
        ax.set_title(f"{office[1:]} E, α = {float(alpha):.2f}", loc="left")
        ax.set_xlabel("Final staff-hours per day")
    axes[0].set_ylabel("Worst hour: late or left")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=l)
               for c, _, l in POLICY_STYLE.values()]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", ls="", label="lognormal truth"),
                plt.Line2D([], [], color=MUTED, marker="s", ls="", label="exponential truth")]
    axes[3].legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig20_learning_tradeoff.png")


E15_LABELS = [("office", "exp30", "lean", "Office, exp, lean"),
              ("office", "exp30", "citizen", "Office, exp, citizen"),
              ("office", "logn30", "lean", "Office, logn, lean"),
              ("office", "logn30", "citizen", "Office, logn, citizen"),
              ("R8_S16_A0.6", "exp30", "lean", "8 E, exp, lean"),
              ("R8_S16_A0.6", "exp30", "citizen", "8 E, exp, citizen"),
              ("R8_S16_A0.6", "logn30", "lean", "8 E, logn, lean"),
              ("R8_S16_A0.6", "logn30", "citizen", "8 E, logn, citizen")]


def fig_wait_displays():
    """Round 12: what showing the wait does, per setting (paired 95% CIs)."""
    con = load("e15c_paired_contrasts.csv")
    reg = load("e15b_regimes.csv")
    contrasts = [("T-H", "#e34948", "tickets display vs hidden"),
                 ("L-H", "#2a78d6", "LES display vs hidden"),
                 ("V-C", "#eda100", "commitment: visible line vs count display"),
                 ("V-H", INK, "visible line vs hidden")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    y0 = np.arange(len(E15_LABELS))[::-1]
    for j, (name, color, label) in enumerate(contrasts):
        for yi, (o, p, k, _) in zip(y0, E15_LABELS):
            r = next(x for x in con if (x["office"], x["patience"], x["plan_type"], x["contrast"])
                     == (o, p, k, name))
            y = yi + 0.3 - 0.2 * j
            a.plot([float(r["ci_low"]), float(r["ci_high"])], [y, y], color=color, lw=1.6)
            a.plot(float(r["mean"]), y, "o", color=color, ms=4,
                   label=label if yi == y0[0] else None)
    a.axvline(0, color=MUTED, lw=1)
    a.set_yticks(y0)
    a.set_yticklabels([lab for *_, lab in E15_LABELS])
    a.set_xlabel("Change in citizens late or left per day (paired, 1,000 days)")
    a.set_title("(a) Failures: displays, commitment, and the visible line", loc="left")
    a.legend(fontsize=7.5, loc="upper right")
    for regime, color, marker in (("T", "#e34948", "o"), ("C", "#b0301f", "^"),
                                  ("L", "#2a78d6", "s"), ("V", INK, "D")):
        xs, ys = [], []
        for o, p, k, _ in E15_LABELS:
            sub = {r["regime"]: r for r in reg if (r["office"], r["patience"], r["plan_type"])
                   == (o, p, k)}
            xs.append(100 * (1 - float(sub[regime]["lost_min_per_arrival"])
                             / float(sub["H"]["lost_min_per_arrival"])))
            ys.append(100 * (float(sub[regime]["overall_fail"]) - float(sub["H"]["overall_fail"])))
        b.scatter(xs, ys, color=color, marker=marker, s=30,
                  label={"T": "tickets display", "C": "count display", "L": "LES display",
                         "V": "visible line"}[regime])
    b.axhline(0, color=MUTED, lw=1)
    b.set_xlabel("Citizen minutes lost per arrival, % below the hidden queue")
    b.set_ylabel("Failure rate vs hidden (points)")
    b.set_title("(b) Time saved against failures added", loc="left")
    b.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig21_wait_displays.png")


# ----------------------------------------------------------------------------
E16_PAT_COLOR = {"exp30": "#2a78d6", "logn30": "#e34948", "logn30cv15": "#1baf7a"}
E16_PAT_LABEL = {"exp30": "exponential", "logn30": "lognormal CV 0.5",
                 "logn30cv15": "lognormal CV 1.5"}


def fig_display_theory():
    """Round 13: the exact law, its optimum at the threshold, and out-of-sample predictions."""
    from displays import cutoff, display_hour
    from learning import Patience
    pats = {"exp30": Patience("exp", 30.0), "logn30": Patience("lognormal", 30.0, 0.5),
            "logn30cv15": Patience("lognormal", 30.0, 1.5)}
    val = load("e16a_theory_validation.csv")
    reg = load("e16b_regimes.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    c, s, rho = 8, 16.0, 1.2
    lam = rho * c / s
    ms = np.linspace(2.0, 45.0, 87)
    for pname, p in pats.items():
        col = E16_PAT_COLOR[pname]
        a.plot(ms, [100 * display_hour(c, lam, s, p, 15.0, cutoff(m)).fail for m in ms],
               color=col, lw=1.8, label=E16_PAT_LABEL[pname])
        a.axhline(100 * display_hour(c, lam, s, p, 15.0).fail, color=col, lw=1, ls=":")
        for r in val:
            if (int(r["windows"]), float(r["rho"]), r["patience"]) == (c, rho, pname)                     and r["display"].startswith("O-M"):
                a.errorbar(float(r["display"][3:]), 100 * float(r["fail_sim"]),
                           yerr=196 * float(r["se"]), fmt="o", color=col, ms=5, capsize=2)
    a.axvline(15.0, color=MUTED, lw=1)
    a.set_xlabel("Cutoff M (minutes): the display says \"over M\" from there on")
    a.set_ylabel("Late or left (%)")
    a.set_title(f"(a) Stationary, c = {c}, ρ = {rho}: theory (lines), simulation (dots)",
                loc="left")
    a.text(15.4, a.get_ylim()[0] + 0.5, "T = 15", color=MUTED, va="bottom", fontsize=8)
    a.legend(fontsize=7.5, title="dotted: hidden queue", title_fontsize=7.5)
    lim = 0.0
    for pname in pats:
        pts = [(100 * float(r["dfail_pred"]), 100 * float(r["dfail_sim"])) for r in reg
               if r["patience"] == pname and r["dfail_pred"] != "" and r["regime"] != "H"]
        xs, ys = zip(*pts)
        lim = max(lim, max(map(abs, xs)), max(map(abs, ys)))
        b.scatter(xs, ys, s=16, color=E16_PAT_COLOR[pname], label=E16_PAT_LABEL[pname])
    b.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--")
    b.axhline(0, color=AXIS, lw=0.8)
    b.axvline(0, color=AXIS, lw=0.8)
    b.set_xlabel("Predicted change vs hidden (points; per-hour stationary theory)")
    b.set_ylabel("Simulated change vs hidden (points)")
    b.set_title("(b) 12 new offices × 10 oracle displays", loc="left")
    b.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig22_display_theory.png")


def fig_display_practice():
    """Round 13b: how much of the oracle's gain a real office can reach, and staffing."""
    twin = load("e16d_twin.csv") + load("e16f_twin_high_quantiles.csv")
    st = load("e16e_staffing.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    keys = []
    for r in twin:
        k = (r["office"], r["patience"], r["plan_type"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rows = sorted((r for r in twin if (r["office"], r["patience"], r["plan_type"]) == k),
                      key=lambda r: float(r["quantile"]))
        qs = [float(r["quantile"]) for r in rows]
        share = [100 * float(r["share_of_oracle_gain"]) for r in rows]
        ls = "-" if k[0].startswith("R16") else "--"
        mk = "o" if k[2] == "lean" else "s"
        a.plot(qs, share, ls, marker=mk, ms=3.5, lw=1.3, color=E16_PAT_COLOR[k[1]],
               label=f"{k[0][:3].rstrip('_')} E, {k[2]}" if k[1] == "logn30" else None)
    a.axhline(0, color=MUTED, lw=1)
    a.axvspan(0.3, 0.7, color=GRID, alpha=0.4, lw=0)
    a.set_ylim(-60, 75)
    a.text(0.5, 70, "registered grid", ha="center", va="top", color=MUTED, fontsize=8)
    a.set_xlabel("Quantile q: say \"over 15\" when P(wait ≥ 15) ≥ 1 − q")
    a.set_ylabel("Share of the oracle cutoff's reduction (%)")
    a.set_title("(a) Twin display: what a ticket office can predict", loc="left")
    handles = [plt.Line2D([], [], color=c, lw=2, label=E16_PAT_LABEL[p])
               for p, c in E16_PAT_COLOR.items()]
    handles += [plt.Line2D([], [], color=INK_2, ls="-", label="16 E"),
                plt.Line2D([], [], color=INK_2, ls="--", label="4 E"),
                plt.Line2D([], [], color=INK_2, ls="", marker="o", label="lean plan"),
                plt.Line2D([], [], color=INK_2, ls="", marker="s", label="safe plan")]
    a.legend(handles=handles, fontsize=7, loc="lower left", ncol=2)
    hid = {(r["office"], r["patience"], r["alpha"]): int(r["staff_hours"]) for r in st
           if r["plan"] == "hidden" and r["regime"] == "H"}
    for regime, marker, label in (("O-M15", "o", "oracle cutoff"), ("twin", "^", "twin (q = 0.5)")):
        for r in st:
            if r["plan"] != "display" or r["regime"] != regime:
                continue
            k = (r["office"], r["patience"], r["alpha"])
            saved = 100 * (1 - int(r["staff_hours"]) / hid[k])
            ratio = float(r["worst_fail"]) / float(r["alpha"])
            miss = int(r["fail_misses"]) > 0
            b.scatter(saved, ratio, marker=marker, s=34, color=E16_PAT_COLOR[r["patience"]],
                      facecolors=E16_PAT_COLOR[r["patience"]] if miss else "none",
                      linewidths=1.3)
    b.axhline(1.0, color=MUTED, lw=1, ls="--")
    b.set_xlabel("Staff-hours saved by staffing for the display (%)")
    b.set_ylabel("Worst hour's failure rate / target")
    b.set_title("(b) Staffing for the display (filled = significant miss)", loc="left")
    b.legend(handles=[plt.Line2D([], [], color=INK_2, ls="", marker="o", mfc="none",
                                 label="oracle cutoff"),
                      plt.Line2D([], [], color=INK_2, ls="", marker="^", mfc="none",
                                 label="twin display, q = 0.5")], fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig23_display_practice.png")


# ----------------------------------------------------------------------------
E17_C_COLOR = {1: "#e34948", 2: "#eb6834", 4: "#eda100", 16: "#1baf7a", 64: "#2a78d6"}


def fig_display_returns_theory():
    """Round 14: the exchange rate of the cutoff display with returns (stationary)."""
    ex = [r for r in load("e17a_exchange.csv") if r["eps"] != ""]
    col = load("e17e_nT_collapse.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for c, color in E17_C_COLOR.items():
        for pname, ls in (("exp30", "-"), ("logn30", "--")):
            pts = sorted((float(r["rho"]), float(r["eps"])) for r in ex
                         if int(r["c"]) == c and r["S"] == "16.0" and r["patience"] == pname
                         and float(r["H_late_share"]) >= 1e-3)
            if pts:
                xs, ys = zip(*pts)
                a.plot([1 - x for x in xs], ys, ls, marker="o", ms=3, lw=1.5, color=color)
        a.plot([], [], color=color, lw=1.5, label=f"c = {c}")
    a.set_xscale("log")
    a.invert_xaxis()
    a.set_yscale("log")
    a.set_ylim(0.01, 5.0)
    a.axhline(1.0, color=INK_2, lw=1)
    a.axhspan(0.01, 0.5, color=GRID, alpha=0.35, lw=0)
    a.text(0.0011, 0.3, "shaded: registered H60(a), ε < 0.5", color=MUTED, fontsize=8, va="top", ha="right")
    a.text(0.0011, 3.9, "above 1: returns cancel the gain", color=INK_2, fontsize=8, va="top",
           ha="right")
    a.set_xlabel("1 − ρ (fresh load; toward collapse →)")
    a.set_ylabel("ε: extra visits per late service avoided")
    a.set_title("(a) Exchange rate by office size (S = 16, T = 15)", loc="left")
    a.legend(fontsize=7, title="solid exponential, dashed lognormal CV 0.5",
             title_fontsize=7, loc="lower left", ncol=3)
    t_color = {7.5: "#2a78d6", 15.0: "#1baf7a", 30.0: "#e34948"}
    for r in col:
        if r["eps"] == "" or float(r["H_late_share"]) < 1e-3:
            continue
        b.scatter(float(r["n_T"]), float(r["eps"]), s=18, color=t_color[float(r["T"])],
                  marker="o" if r["patience"] == "exp30" else "^", alpha=0.85)
    b.set_xscale("log")
    b.set_yscale("log")
    b.axhline(1.0, color=INK_2, lw=1)
    b.set_xlabel("n_T = c·T/S: citizens the office can serve within the target")
    b.set_ylabel("ε at ρ = 0.95")
    b.set_title("(b) Post hoc: ε collapses onto n_T (T, S and c varied)", loc="left")
    b.legend(handles=[plt.Line2D([], [], color=v, ls="", marker="o", label=f"T = {k:g} min")
                      for k, v in t_color.items()] +
             [plt.Line2D([], [], color=INK_2, ls="", marker="o", mfc="none", label="exponential"),
              plt.Line2D([], [], color=INK_2, ls="", marker="^", mfc="none",
                         label="lognormal CV 0.5")], fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig24_display_returns_theory.png")


E17_REG_STYLE = {"O-M15": ("#2a78d6", "o", "oracle cutoff"),
                 "C0-M15": ("#eda100", "s", "head count, cutoff 15"),
                 "T0.7": ("#e34948", "^", "twin, q = 0.7")}


def fig_display_returns_sim():
    """Round 14: simulated exchange rates in the 16 offices, and returns at opening."""
    co = load("e17b_contrasts.csv")
    law = {(r["office"], r["patience"], r["plan_type"]): r for r in load("e17p_predictions.csv")
           if r["timing"] == "profile"}
    col = [r for r in load("e17e_nT_collapse.csv")
           if r["eps"] != "" and float(r["H_late_share"]) >= 1e-3]
    op = load("e17f_opening_curves.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    a.scatter([float(r["n_T"]) for r in col], [float(r["eps"]) for r in col], s=10,
              color=AXIS, label="stationary law (E17e)", zorder=1)
    for i, (reg, (color, marker, label)) in enumerate(E17_REG_STYLE.items()):
        xs, ys, lo, hi = [], [], [], []
        for r in co:
            if r["regime"] != reg:
                continue
            k = (r["office"], r["patience"], r["plan_type"])
            s_ = 8.0 if r["office"].startswith("R4") else 16.0
            xs.append(int(law[k]["law_c"]) * 15.0 / s_ * (1 + 0.04 * (i - 1)))
            ys.append(float(r["eps"]))
            lo.append(float(r["eps"]) - float(r["eps_low"]))
            hi.append(float(r["eps_high"]) - float(r["eps"]))
        a.errorbar(xs, ys, yerr=[lo, hi], fmt=marker, color=color, ms=5, capsize=2, lw=1,
                   label=label, zorder=3)
    a.set_xscale("log")
    a.set_yscale("log")
    a.axhline(1.0, color=INK_2, lw=1)
    a.text(0.62, 1.07, "above 1: returns cancel the gain", color=INK_2, fontsize=8)
    a.set_xlabel("n_T = c̄·T/S of the office (c̄: mean windows)")
    a.set_ylabel("ε: extra visits per late service avoided")
    a.set_title("(a) 13 offices with a steady state (3 more have none under any display)",
                loc="left")
    a.legend(fontsize=7.5, loc="lower left")
    for reg, color, label in (("H", INK_2, "hidden queue"), ("O-M15", "#2a78d6", "oracle cutoff")):
        rows = [r for r in op if r["regime"] == reg]
        R = [float(r["R"]) for r in rows]
        b.fill_between(R, [float(r["h_low"]) for r in rows], [float(r["h_high"]) for r in rows],
                       color=color, alpha=0.15, lw=0)
        b.plot(R, [float(r["h"]) for r in rows], marker="o", ms=3.5, lw=1.6, color=color,
               label=label)
    b.axhline(0, color=MUTED, lw=1)
    b.set_xlabel("Returners a day, R (all arrive in the first hour)")
    b.set_ylabel("h(R) = losses − R (above 0: backlog grows)")
    b.set_title("(b) Post hoc: returns at opening, 8 E office, φ = 1.0", loc="left")
    b.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig25_display_returns_sim.png")


E18_OFFICE_LABEL = {"R4_S8_A0.6": "4 E", "R16_S16_A0.6": "16 E"}
E18_PAT_LABEL = {"exp30": "exp", "logn30": "logn 0.5", "logn30cv15": "logn 1.5"}


def _psi_table(name, office, pname, kind):
    key = f"{office}_{pname}_{kind}".replace("=", "").replace(".", "p")
    rows = load(f"e18_psi/{name}_{key}.csv")
    out = {}
    for r in rows:
        out.setdefault(int(r["hour"]), []).append((float(r["x"]), float(r["psi"])))
    return {h: np.array(v) for h, v in out.items()}


def fig_bayes_display_theory():
    """Round 15: the influence of turning a citizen away, and the threshold it implies."""
    pred = [r for r in load("e18p_predictions.csv") if r["table"] == "B1"]
    twin = load("e16d_twin.csv") + load("e16f_twin_high_quantiles.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    styles = [("R16_S16_A0.6", "logn30", "lean", "#2a78d6"),
              ("R16_S16_A0.6", "exp30", "lean", "#e34948"),
              ("R4_S8_A0.6", "logn30", "lean", "#1baf7a"),
              ("R4_S8_A0.6", "logn30cv15", "safe", "#eda100")]
    for office, pname, kind, color in styles:
        t = _psi_table("B1", office, pname, kind)[1]          # 9-10 AM, the morning peak
        a.plot(t[:, 0], t[:, 1], color=color, lw=1.7,
               label=f"{E18_OFFICE_LABEL[office]}, {E18_PAT_LABEL[pname]}, {kind}")
    a.step([0, 15, 15, 30], [1, 1, -1, -1], where="post", color=MUTED, lw=1, ls="--")
    a.text(1, 1.04, "a quantile rule acts as if ψ were this step (±1)", color=MUTED,
           fontsize=7.5, va="bottom")
    a.axhline(0, color=INK_2, lw=1)
    a.axvline(15, color=AXIS, lw=1)
    a.set_xlim(0, 30)
    a.set_ylim(-1.2, 1.25)
    a.set_xlabel("Offered wait V of the citizen turned away (min)")
    a.set_ylabel("ψ(V): failures added per citizen turned away")
    a.set_title("(a) Influence under the oracle cutoff, 9–10 AM", loc="left")
    a.legend(fontsize=7.5, loc="lower left")
    for r in pred:
        k = (r["office"], r["patience"], r["plan_type"])
        rows = [t for t in twin if (t["office"], t["patience"], t["plan_type"]) == k]
        best = min(rows, key=lambda t: float(t["fail"]))
        near = sorted(float(t["quantile"]) for t in rows
                      if float(t["fail"]) <= float(best["fail"]) + 0.0015)
        x = float(r["implied_threshold_mean"])
        color = "#2a78d6" if r["office"].startswith("R16") else "#e34948"
        b.errorbar(x, 1 - float(best["quantile"]),
                   yerr=[[1 - float(best["quantile"]) - (1 - near[-1])],
                         [(1 - near[0]) - (1 - float(best["quantile"]))]],
                   fmt="o" if r["plan_type"] == "lean" else "s", color=color, ms=5, capsize=2,
                   lw=1)
    b.plot([0, 0.6], [0, 0.6], color=MUTED, lw=1, ls="--")
    b.set_xlim(0, 0.6)
    b.set_ylim(0, 0.8)
    b.set_xlabel("Threshold ψ implies: flag once P(late | X) exceeds ψ_on / (ψ_on + |ψ_late|)")
    b.set_ylabel("1 − q of the best twin quantile (E16d/f)")
    b.set_title("(b) Retrodiction: theory against Round 13b's tuned quantile", loc="left")
    b.legend(handles=[plt.Line2D([], [], color="#e34948", ls="", marker="o", label="4 E"),
                      plt.Line2D([], [], color="#2a78d6", ls="", marker="o", label="16 E"),
                      plt.Line2D([], [], color=INK_2, ls="", marker="o", mfc="none",
                                 label="lean plan"),
                      plt.Line2D([], [], color=INK_2, ls="", marker="s", mfc="none",
                                 label="safe plan")], fontsize=7.5, loc="upper left")
    b.text(0.59, 0.02, "bars: quantiles within 0.15 points of failure rate of the best",
           color=MUTED,
           fontsize=7.5, ha="right")
    fig.tight_layout()
    save(fig, "fig26_bayes_display_theory.png")


E18_REG_STYLE = {"C0-M12.5": ("#eda100", "D", "head count, cutoff 12.5"),
                 "C0-M15": ("#1baf7a", "s", "head count, cutoff 15"),
                 "T0.7": ("#e34948", "^", "twin, q = 0.7"),
                 "B30": ("#2a78d6", "o", "Bayes, K = 30"),
                 "B60": ("#8a5cd6", "v", "Bayes, K = 60")}


def fig_bayes_display_sim():
    """Round 15: the Bayes display against the tuned twin, and what the posterior knows."""
    rows = load("e18b_bayes.csv")
    info = load("e18a_information.csv")
    rel = load("e18a_reliability.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.6),
                                  gridspec_kw={"width_ratios": [1.5, 1, 1]})
    keys = [(r["office"], r["patience"], r["plan_type"]) for r in rows if r["regime"] == "H"]
    share = {(r["office"], r["patience"], r["plan_type"], r["regime"]): r["share_of_oracle_gain"]
             for r in rows}
    order = sorted(keys, key=lambda k: float(share[(*k, "Tbest")]))
    for regime, color, marker, label, dx in (("Tbest", "#e34948", "^", "twin, best q in hindsight", -0.2),
                                              ("B1", "#2a78d6", "o", "Bayes B1 (no tuning)", 0.0),
                                              ("Bstar", MUTED, "x", "B* (three refits)", 0.2)):
        ys = [float(share[(*k, regime)]) for k in order]
        a.scatter(np.arange(len(order)) + dx, ys, color=color, marker=marker, s=28, zorder=3,
                  label=label)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([f"{E18_OFFICE_LABEL[o]} {E18_PAT_LABEL[p]}\n{k}" for o, p, k in order],
                      fontsize=6.8, rotation=90)
    a.axhline(0, color=INK_2, lw=1)
    a.set_ylim(-1.05, 0.8)
    a.set_ylabel("Share of the oracle cutoff's cut in failures")
    a.set_title("(a) Theory matches the tuned rule; refitting overshoots", loc="left")
    a.legend(fontsize=7.5, loc="lower right")
    for r in info:
        color = "#2a78d6" if r["office"].startswith("R16") else "#e34948"
        b.scatter(float(r["auc_count"]), float(r["auc_twin"]), color=color, s=26,
                  marker="o" if r["plan_type"] == "lean" else "s")
    b.plot([0.89, 0.97], [0.89, 0.97], color=MUTED, lw=1, ls="--")
    b.set_xlim(0.895, 0.97)
    b.set_ylim(0.895, 0.97)
    b.set_xlabel("AUC of the head count for V > 15")
    b.set_ylabel("AUC of the twin's P(V > 15 | X)")
    b.set_title("(b) The posterior ranks barely better", loc="left")
    b.legend(handles=[plt.Line2D([], [], color="#e34948", ls="", marker="o", label="4 E"),
                      plt.Line2D([], [], color="#2a78d6", ls="", marker="o", label="16 E"),
                      plt.Line2D([], [], color=INK_2, ls="", marker="s", mfc="none",
                                 label="safe plan")], fontsize=7.5, loc="lower right")
    for key in {(r["office"], r["patience"], r["plan_type"]) for r in rel}:
        pts = [(float(r["predicted"]), float(r["observed"])) for r in rel
               if (r["office"], r["patience"], r["plan_type"]) == key]
        xs, ys = zip(*pts)
        c.plot(xs, ys, marker="o", ms=2.5, lw=0.9, alpha=0.8,
               color="#2a78d6" if key[0].startswith("R16") else "#e34948")
    c.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    c.set_xlim(0, 0.75)
    c.set_ylim(0, 0.75)
    c.set_xlabel("Twin's predicted P(V > 15 | X), decile mean")
    c.set_ylabel("Observed share with V > 15")
    c.set_title("(c) The twin is calibrated", loc="left")
    fig.tight_layout()
    save(fig, "fig27_bayes_display_sim.png")


def fig_bayes_display_returns():
    """Round 15: precision sets a display's price in visits; the returns-aware display."""
    co = {(r["office"], r["patience"], r["plan_type"], r["regime"]): r
          for r in load("e18c_contrasts.csv")}
    pr = load("e18d_precision.csv")
    mk = load("e18c_mk.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for reg, (color, marker, label) in E18_REG_STYLE.items():
        xs, ys = [], []
        for r in pr:
            if r["regime"] != reg or r["eps"] in ("", "nan"):
                continue
            o = co.get((r["office"], r["patience"], r["plan_type"], "O-M15"))
            if not o or o["eps"] in ("", "nan") or float(r["precision"]) <= 0:
                continue
            xs.append(1.0 / float(r["precision"]))
            ys.append(float(r["eps"]) / float(o["eps"]))
        a.scatter(xs, ys, color=color, marker=marker, s=26, label=label, zorder=3)
    grid = np.geomspace(1, 5000, 50)
    a.plot(grid, np.exp(0.428) * grid ** 0.446, color=INK_2, lw=1.2,
           label="least squares: slope 0.45")
    a.plot(grid, grid, color=MUTED, lw=1, ls="--", label="ε / ε_oracle = 1 / precision")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(0.8, 200)
    a.set_xlabel("1 / precision (would-be-served citizens turned away per one who "
                 "would have been late)")
    a.set_ylabel("ε of the display / ε of the oracle cutoff")
    a.set_title("(a) Precision sets the price in repeat visits", loc="left")
    a.legend(fontsize=7, loc="upper left")
    k30 = [r for r in mk if float(r["K"]) == 30.0 and r["M_K"] != "inf"]
    h = {(r["office"], r["patience"], r["plan_type"]): float(r["M_K"]) for r in k30
         if r["regime"] == "H"}
    keys = sorted((k for k in h if sum((r["office"], r["patience"], r["plan_type"]) == k
                                       for r in k30) > 1), key=lambda k: h[k])
    for i, (reg, (color, marker, label)) in enumerate([("O-M15", ("#0d366b", "*", "oracle cutoff"))]
                                                       + [(k, v) for k, v in E18_REG_STYLE.items()
                                                          if k != "B60"]):
        xs, ys = [], []
        for j, k in enumerate(keys):
            r = next((r for r in k30 if (r["office"], r["patience"], r["plan_type"]) == k
                      and r["regime"] == reg), None)
            if r is not None:
                xs.append(j + 0.12 * (i - 2))
                ys.append(float(r["M_K"]) - h[k])
        b.scatter(xs, ys, color=color, marker=marker, s=24, label=label, zorder=3)
    b.axhline(0, color=INK_2, lw=1)
    b.set_xticks(range(len(keys)))
    b.set_xticklabels([f"{E18_OFFICE_LABEL.get(o, '8 E')} {E18_PAT_LABEL[p]}\n{k}"
                       for o, p, k in keys], fontsize=6.3, rotation=90)
    b.set_ylabel("Minutes lost + 30 × visits, minus the hidden queue's (per citizen)")
    b.set_title("(b) With returns, K = 30: below 0 beats showing nothing\n"
                "(13 settings with a steady state)", loc="left")
    b.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig28_bayes_display_returns.png")


if __name__ == "__main__":
    fig_gap_heatmap()
    fig_hourly()
    fig_offered_load()
    fig_sensitivity()
    fig_methodology()
    fig_shifts()
    fig_crossval()
    fig_appointments()
    fig_abandonment()
    fig_regimes()
    fig_log_regime()
    fig_fluid_day()
    fig_fluid_scaling()
    fig_paid_overtime()
    fig_tipping()
    fig_return_chains()
    fig_patience_estimates()
    fig_patience_learning()
    fig_learning_paths()
    fig_learning_tradeoff()
    fig_wait_displays()
    fig_display_theory()
    fig_display_practice()
    fig_display_returns_theory()
    fig_display_returns_sim()
    fig_bayes_display_theory()
    fig_bayes_display_sim()
    fig_bayes_display_returns()
