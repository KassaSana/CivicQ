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


if __name__ == "__main__":
    fig_gap_heatmap()
    fig_hourly()
    fig_offered_load()
    fig_sensitivity()
    fig_methodology()
    fig_shifts()
    fig_crossval()
    fig_appointments()
