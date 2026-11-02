"""Is a site's value error correlated with its reference-range shift? (thesis Chapter 4)

The perturbation draws the value shift and the range shift independently, because
reference ranges tend to be inherited from external sources while values are measured
at the local site (Friedberg 2007). No source reports the joint distribution of the
two shifts. Adeli 2017 Table 2 reports both quantities broken down by instrument
manufacturer, so this script measures their correlation across manufacturer groups
within each analyte.

A manufacturer group pools many laboratories running the same platform, and here it
stands in for a site. The question is whether platforms that measure less reproducibly
also state more divergent reference intervals. Pooled over the 29 manufacturer ×
analyte groups (standardised within each analyte), Pearson r = −0.003. Chapter 4
quotes this number.

The data are transcribed from the page image of Adeli 2017 Table 2 (Clin Biochem
50:925-935, p.933). The text layer of the PDF was not used. Prints the correlations
and writes a figure, results/analysis/a26/F1_value_range_coupling.png.

  python analysis/a26_value_range_coupling.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

OUT = locations.REPO / "results/analysis/a26"

#: analyte -> [(manufacturer, CV_BL %, %V at LRL or None, %V at URL or None)]
#: The "All" rows are left out because they aggregate the manufacturers.
TABLE2 = {
    "ALT": [("Abbott", 7.5, None, 21.9), ("Beckman", 15.0, None, 20.4),
            ("Ortho", 5.3, None, 22.6), ("Roche", 9.7, None, 6.4),
            ("Siemens", 19.7, None, 36.8)],
    "ALP": [("Abbott", 3.8, 18.2, 52.3), ("Beckman", 5.2, 46.5, 35.7),
            ("Ortho", 2.1, 41.7, 43.2), ("Roche", 2.8, None, 23.1),
            ("Siemens", 3.1, None, 41.1)],
    "Calcium": [("Abbott", 1.2, 2.0, 3.5), ("Beckman", 2.2, 2.3, 1.5),
                ("Ortho", 2.2, 2.9, 2.7), ("Roche", 1.9, 2.2, 1.5),
                ("Siemens", 2.8, 4.5, 1.9)],
    "Creatinine": [("Abbott", 3.7, None, 19.8), ("Beckman", 1.3, 22.2, 17.7),
                   ("Ortho", 1.3, 14.8, 15.7), ("Roche", 4.1, None, 14.1),
                   ("Siemens", 6.9, None, 15.5)],
    "FT4": [("Abbott", 6.0, 10.0, 4.8), ("Beckman", 7.5, 9.5, 19.8),
            ("Roche", 4.7, 17.1, 12.6), ("Siemens", 7.5, 12.7, 14.6)],
    "Sodium": [("Abbott", 0.7, 0.7, 1.3), ("Beckman", 0.8, 0.8, 1.1),
               ("Ortho", 1.2, 1.2, 1.2), ("Roche", 0.9, 0.6, 0.6),
               ("Siemens", 0.6, 0.7, 1.2)],
}


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    print("Correlation of CV_BL with %V(URL), across manufacturer groups, per analyte")
    print(f"{'analyte':11s} {'n':>2s} {'Pearson r':>10s} {'Spearman':>9s}   points (CV_BL, %V URL)")
    print("-" * 96)
    zx, zy, per = [], [], []
    for a, rows in TABLE2.items():
        pts = [(cv, v) for _m, cv, _l, v in rows if v is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        r, rho = pearson(xs, ys), spearman(xs, ys)
        per.append((a, len(pts), r, rho))
        print(f"{a:11s} {len(pts):>2d} {r:10.3f} {rho:9.3f}   "
              f"{', '.join(f'({x:g},{y:g})' for x, y in pts)}")
        # standardise within analyte so analytes can be pooled on one scale
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sx = (sum((v - mx) ** 2 for v in xs) / n) ** 0.5
        sy = (sum((v - my) ** 2 for v in ys) / n) ** 0.5
        if sx and sy:
            zx += [(v - mx) / sx for v in xs]
            zy += [(v - my) / sy for v in ys]
    pooled_r, pooled_rho = pearson(zx, zy), spearman(zx, zy)
    print("-" * 96)
    rs = [r for _a, _n, r, _ in per]
    print(f"\nper-analyte Pearson r ranges {min(rs):.3f} to {max(rs):.3f}, "
          f"median {sorted(rs)[len(rs)//2]:.3f}")
    print(f"pooled (within-analyte standardised, n={len(zx)} manufacturer-analyte points):")
    print(f"   Pearson  rho = {pooled_r:.3f}")
    print(f"   Spearman rho = {pooled_rho:.3f}")
    print("\nNote: each manufacturer group pools many laboratories, so this correlation\n"
          "    stands in for the site-level one.")

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
        COLORS = ["#2a78d6", "#eb6834", "#3f9e5c", "#9b59b6", "#c0392b", "#16a085"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        for i, (a, rows) in enumerate(TABLE2.items()):
            pts = [(cv, v) for _m, cv, _l, v in rows if v is not None]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=44,
                       color=COLORS[i % len(COLORS)], label=a, zorder=3,
                       edgecolor="white", linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("CV$_{BL}$ (%), between-laboratory value variation", color=INK2)
        ax.set_ylabel("%V at the upper reference limit", color=INK2)
        ax.set_title("Raw scale: analytes occupy different regions", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[1]
        k = 0
        for i, (a, rows) in enumerate(TABLE2.items()):
            pts = [(cv, v) for _m, cv, _l, v in rows if v is not None]
            n = len(pts)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            mx, my = sum(xs)/n, sum(ys)/n
            sx = (sum((v-mx)**2 for v in xs)/n) ** 0.5
            sy = (sum((v-my)**2 for v in ys)/n) ** 0.5
            if not (sx and sy):
                continue
            ax.scatter([(v-mx)/sx for v in xs], [(v-my)/sy for v in ys], s=44,
                       color=COLORS[i % len(COLORS)], zorder=3,
                       edgecolor="white", linewidth=1.2)
            k += n
        lo, hi = -2.2, 2.2
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.2, linestyle="--",
                zorder=2, label="perfect coupling (ρ = 1)")
        b = pooled_r
        ax.plot([lo, hi], [b*lo, b*hi], color="#2a78d6", linewidth=2, zorder=2,
                label=f"fitted ρ = {pooled_r:.2f}")
        ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)
        ax.axvline(0, color=GRID, linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("CV$_{BL}$, standardised within analyte", color=INK2)
        ax.set_ylabel("%V (URL), standardised within analyte", color=INK2)
        ax.set_title(f"Pooled, n = {k} manufacturer×analyte points", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle("Adeli 2017 Table 2: does a platform's value error track its "
                     "reference-range shift?", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "F1_value_range_coupling.png", dpi=160)
        print(f"\nwrote {OUT/'F1_value_range_coupling.png'}")
    except Exception as e:
        print(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    sys.exit(main())
