"""Turn results/runs and results/eval into figures/ and results/summary.md.

    python scripts/make_plots.py
"""

import csv
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

# colours: validated categorical slots in fixed order, grey for reference lines
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COLOR = {"airfed": "#2a78d6", "independent": "#eb6834", "static_balanced": "#1baf7a", "random": "#8c8b86"}
LABEL = {
    "airfed": "AirFed", "no_gat": "w/o GAT", "no_reputation": "w/o reputation", "no_fed": "w/o federation",
    "no_quant": "w/o quantization", "independent": "Independent learner", "random": "Random",
    "static_local": "Static + local", "static_balanced": "Static + balanced",
}
MAIN = ["airfed", "no_gat", "no_reputation", "no_fed", "no_quant", "independent", "static_balanced", "static_local", "random"]


def load(folder):
    rows = []
    for path in glob.glob(str(ROOT / "results" / folder / "*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                for key in ("uavs", "devices", "seed", "episode"):
                    r[key] = int(r[key])
                for key in ("time", "energy_kj", "deadline_rate", "dropped", "coverage", "reward", "comm_mb"):
                    r[key] = float(r[key])
                rows.append(r)
    return rows


def add_cost(rows):
    """Weighted cost of Eq. 31 with min-max bounds over every run of the same setup (Eqs. 32-33)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["uavs"], r["devices"])].append(r)
    for group in groups.values():
        t = np.array([r["time"] for r in group])
        e = np.array([r["energy_kj"] for r in group])
        for r in group:
            nt = (r["time"] - t.min()) / max(t.max() - t.min(), 1e-9)
            ne = (r["energy_kj"] - e.min()) / max(e.max() - e.min(), 1e-9)
            r["cost"] = 0.5 * nt + 0.5 * ne


def style(ax, title, ylabel, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def label_ends(ax, ends):
    """Direct labels at the right end of each line, pushed apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, text, x in ends:
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)


def smooth(y, w=5):
    return np.convolve(np.pad(y, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")


def curve(rows, method, key):
    seeds = sorted({r["seed"] for r in rows if r["method"] == method})
    series = []
    for s in seeds:
        mine = sorted((r for r in rows if r["method"] == method and r["seed"] == s), key=lambda r: r["episode"])
        series.append(smooth(np.array([r[key] for r in mine])))
    n = min(len(x) for x in series)
    arr = np.array([x[:n] for x in series])
    return np.arange(1, n + 1), arr.mean(0), arr.std(0)


def learning_curves(train):
    rows = [r for r in train if r["uavs"] == 6 and r["devices"] == 40]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, key, title, ylabel in [(axes[0], "cost", "Weighted cost per episode (lower is better)", "weighted cost"),
                                   (axes[1], "coverage", "IoT coverage per episode", "share of devices covered")]:
        ends = []
        for method in ["airfed", "independent", "static_balanced", "random"]:
            if not any(r["method"] == method for r in rows):
                continue
            x, mean, std = curve(rows, method, key)
            dashed = method in ("static_balanced", "random")
            ax.plot(x, mean, color=COLOR[method], lw=2, ls="--" if dashed else "-", label=LABEL[method])
            if not dashed:
                ax.fill_between(x, mean - std, mean + std, color=COLOR[method], alpha=0.12, lw=0)
            ends.append([mean[-1], LABEL[method], x[-1]])
        style(ax, title, ylabel, "episode")
        ax.set_xlim(1, 128)
        label_ends(ax, ends)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "learning_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def mean_std(rows, method, key, uavs=6, devices=40):
    per_seed = defaultdict(list)
    for r in rows:
        if r["method"] == method and r["uavs"] == uavs and r["devices"] == devices:
            per_seed[r["seed"]].append(r[key])
    vals = [np.mean(v) for v in per_seed.values()]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)


def bar_chart(names, values, errors, title, xlabel, path, fmt):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.2), facecolor=SURFACE)
    y = np.arange(len(names))[::-1]
    colors = [COLOR["airfed"] if n == "AirFed" else "#9ec5f4" for n in names]
    ax.barh(y, values, xerr=errors, color=colors, height=0.6, error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
    for yi, v, e in zip(y, values, errors):
        ax.annotate(fmt.format(v), (v + e, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    style(ax, title, "", xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v + e for v, e in zip(values, errors)) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def scalability(evals):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, axis_key, points, xlabel, title in [
        (axes[0], "uavs", [(4, 40), (6, 40), (8, 40)], "number of UAVs (40 devices)", "Weighted cost vs UAV count"),
        (axes[1], "devices", [(6, 20), (6, 40), (6, 60)], "number of IoT devices (6 UAVs)", "Weighted cost vs device density"),
    ]:
        for method in ["airfed", "static_balanced", "random"]:
            xs, ys = [], []
            for k, m in points:
                v, _ = mean_std(evals, method, "cost", k, m)
                if not np.isnan(v):
                    xs.append(k if axis_key == "uavs" else m)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, color=COLOR[method], lw=2, marker="o", ms=5, ls="--" if method == "random" else "-",
                        label=LABEL[method])
                ax.annotate(LABEL[method], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK_2)
        style(ax, title, "weighted cost (lower is better)", xlabel)
        ax.set_xticks([k if axis_key == "uavs" else m for k, m in points])
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(FIG / "scalability.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def summary(train, evals):
    lines = ["# Results", "", "All numbers are means over seeds (± standard deviation across seeds).",
             "Final numbers come from 5 evaluation episodes per seed on a task stream never used in training.", ""]

    lines += ["## Main comparison (6 UAVs, 40 devices)", "",
              "| Method | Time (s) | Energy per UAV (kJ) | Weighted cost | Deadline met | Coverage | Tasks dropped |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in MAIN:
        t, ts = mean_std(evals, m, "time")
        if np.isnan(t):
            continue
        e, es = mean_std(evals, m, "energy_kj")
        c, cs = mean_std(evals, m, "cost")
        d, _ = mean_std(evals, m, "deadline_rate")
        cov, covs = mean_std(evals, m, "coverage")
        dr, _ = mean_std(evals, m, "dropped")
        lines.append(f"| {LABEL[m]} | {t:.3f} ± {ts:.3f} | {e:.2f} ± {es:.2f} | {c:.3f} ± {cs:.3f} | "
                     f"{100 * d:.1f}% | {100 * cov:.1f}% ± {100 * covs:.1f} | {100 * dr:.1f}% |")

    lines += ["", "## Federated communication (per UAV, whole training run)", "",
              "| Method | Data sent (MB) |", "| --- | --- |"]
    totals = {}
    for m in ["airfed", "no_reputation", "no_quant"]:
        per_seed = defaultdict(float)
        for r in train:
            if r["method"] == m and r["uavs"] == 6 and r["devices"] == 40:
                per_seed[r["seed"]] += r["comm_mb"]
        if per_seed:
            totals[m] = (np.mean(list(per_seed.values())), np.std(list(per_seed.values())))
            lines.append(f"| {LABEL[m]} | {totals[m][0]:.1f} ± {totals[m][1]:.1f} |")
    if "airfed" in totals and "no_quant" in totals:
        cut = 100 * (1 - totals["airfed"][0] / totals["no_quant"][0])
        lines.append(f"\nQuantization cuts the data AirFed sends by **{cut:.1f}%**.")

    lines += ["", "## Scalability (weighted cost, evaluation)", "",
              "| Setup (UAVs-devices) | AirFed | Static + balanced | Random |", "| --- | --- | --- | --- |"]
    for k, m in [(4, 40), (6, 40), (8, 40), (6, 20), (6, 60)]:
        cells = []
        for method in ["airfed", "static_balanced", "random"]:
            v, s = mean_std(evals, method, "cost", k, m)
            cells.append("-" if np.isnan(v) else f"{v:.3f} ± {s:.3f}")
        lines.append(f"| {k}-{m} | " + " | ".join(cells) + " |")
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n")
    return totals


def main():
    FIG.mkdir(exist_ok=True)
    train, evals = load("runs"), load("eval")
    add_cost(train + evals)
    learning_curves(train)

    names, vals, errs = [], [], []
    for m in ["airfed", "no_gat", "no_reputation", "no_fed", "no_quant", "independent"]:
        v, s = mean_std(evals, m, "cost")
        if not np.isnan(v):
            names.append(LABEL[m]); vals.append(v); errs.append(s)
    if names:
        bar_chart(names, vals, errs, "Ablation: weighted cost after training (lower is better)",
                  "weighted cost", FIG / "ablation.png", "{:.3f}")

    names, vals, errs = [], [], []
    for m in ["airfed", "no_gat", "no_reputation", "no_fed", "no_quant", "independent", "static_balanced"]:
        v, sd = mean_std(evals, m, "coverage")
        if not np.isnan(v):
            names.append(LABEL[m]); vals.append(100 * v); errs.append(100 * sd)
    if names:
        bar_chart(names, vals, errs, "Ablation: IoT coverage after training (higher is better)",
                  "devices covered (%)", FIG / "ablation_coverage.png", "{:.1f}%")

    totals = summary(train, evals)
    if totals:
        names = [LABEL[m] for m in totals]
        bar_chart(names, [v[0] for v in totals.values()], [v[1] for v in totals.values()],
                  "Federated learning traffic per UAV over training", "megabytes sent", FIG / "communication.png", "{:.0f} MB")
    scalability(evals)
    print((ROOT / "results" / "summary.md").read_text())


if __name__ == "__main__":
    main()
