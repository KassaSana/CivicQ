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
