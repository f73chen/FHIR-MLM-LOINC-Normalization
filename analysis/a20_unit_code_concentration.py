"""
How many codes does each unit identify, and is that real (thesis Chapter 4 and Appendix A)?

Counts the LOINC and MIMIC codes associated with each unit. Where a unit is seen for only one or a
few codes, it looks deeper and measures how many observations and patients sit on it. The expected
case is that different sites use the same unit type for a given test. The exact scale may differ,
but the base dimension should be the same, because the same thing is being measured.

A unit that maps to only one code identifies that code outright, before the model reads anything
else. How much that matters depends on two things that this script measures:

  1. How much data sits on such units, in observations and patients (a one-code unit may carry four
     observations or 110,000).
  2. Whether the uniqueness is real or an artefact of the restricted label space. Suppose `pg` is
     used by many LOINC codes but only MCH is in the 679. Then the "shortcut" disappears at a site
     with a fuller catalogue, and the analysis would be describing the label space.

The external evidence is in a local file. `loinc_measurements_enriched.jsonl` records the attested
units per code across five systems (MIMIC, LOINC, UCUM, CDISC, CIEL) and covers all 679 label-space
codes. This lets the script measure the expected case instead of citing it. For each
code, the script asks whether the external systems attest a unit of the same kind as MIMIC's. A kind
is the dimension, the mole exponent and the analyte-dependent unit. The expected case is agreement
on kind with a difference in scale, checked against four terminologies.

Reads the a15 cache (test split by default), the terminology files in UPSTREAM_DIR and the label
space. Writes a20_codes_per_unit.csv, a20_few_code_units.csv, a20_external_attestation.csv and two
figures to results/analysis/a20/. analysis/a31_unit_disagreement_resolution.py and the Appendix A
code read the attestation table. Appendix A also reads the per-unit counts. Outputs and stdout are
aggregate only. They include patient counts but no identifiers.

  python analysis/a20_unit_code_concentration.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(HERE))

import unit_standardisation as us  # noqa: E402
from input_preprocessing_utils import load_unit_alias_map  # noqa: E402

_s = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
a15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a15)

OUT = locations.REPO / "results/analysis/a20"
FHIR = locations.UPSTREAM
ENRICHED = FHIR / "loinc_measurements_enriched.jsonl"
LOINC_CSV = FHIR / "Loinc.csv"
LABELSPACE = ROOT / "data" / "labelspace" / "labelspace.csv"
EXTERNAL = ("loinc", "ucum", "cdisc", "ciel")

#: categorical colours, checked for contrast and separation on the light background
C_LOINC, C_MIMIC = "#2a78d6", "#eb6834"
SEQ = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load_observations(amap, split="test"):
    """One row per observation carrying a unit, with its LOINC code, MIMIC item and unit at three
    levels. `patient_uuid` is used only to count distinct patients and never leaves this function."""
    from fhir_data_utils import build_lab_code_mappings
    _, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    arr = np.array([c2l.get(i, f"?{i}") for i in range(1623)], dtype=object)
    df = a15.load_cache(split)
    df = df[df.val_unit.notna()].copy()
    df["loinc"] = arr[df.class_idx.to_numpy()]
    cache = {}
    for u in df.val_unit.unique():
        cache[u] = (us.canonical_unit(u, amap) or u, us.standardise_unit(u, amap)[0])
    df["canon"] = [cache[u][0] for u in df.val_unit]
    df["std"] = [cache[u][1] for u in df.val_unit]
    df["pid"] = pd.factorize(df.patient_uuid)[0]
    return df.drop(columns=["patient_uuid"])


def concentration(df, unit_col):
    """Per unit: distinct LOINC codes, distinct MIMIC items, observations, distinct patients."""
    g = df.groupby(unit_col)
    out = pd.DataFrame({
        "n_loinc": g.loinc.nunique(),
        "n_mimic_item": g.item_code.nunique(),
        "n_obs": g.size(),
        "n_patients": g.pid.nunique(),
    }).sort_values("n_loinc")
    out["pct_obs"] = 100 * out.n_obs / len(df)
    return out.reset_index().rename(columns={unit_col: "unit"})


# --------------- is the uniqueness real, or an artefact of the label space? ---------------
def loinc_wide_unit_usage():
    """UCUM unit -> the set of codes in the full LOINC release that list it as an example unit.

    This checks whether the uniqueness of a unit is real. `pg` maps to only one code in the 679. If
    the rest of LOINC rarely uses `pg`, the unit names the analyte and the signal is real. If LOINC
    uses it for dozens of codes, the uniqueness comes from the chosen label space. It would not hold
    at a site with a fuller catalogue, and the result would be an artefact."""
    per_unit = collections.defaultdict(set)
    with open(LOINC_CSV, newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("EXAMPLE_UCUM_UNITS") or "").strip()
            if not raw:
                continue
            # LOINC packs alternatives into one field, separated by ; or |
            for tok in (t.strip() for part in raw.split("|") for t in part.split(";")):
                if tok:
                    per_unit[tok].add(r["LOINC_NUM"])
    return per_unit


def external_attestation(amap):
    """Per code: does any external terminology attest a unit of the same kind as MIMIC's?

    "Kind" is (dimension, mole exponent, analyte-dependent unit), the same equivalence class the unit
    standardisation of the v2 and v3 representations uses. Agreeing on kind while differing on scale
    is the expected case. Scale differences are therefore recorded separately and are not counted as
    disagreement."""
    label_space = {r["loinc"] for r in csv.DictReader(open(LABELSPACE))
                   if r["in_primary_ge10occ"] == "1"}
    rows = []
    for line in open(ENRICHED):
        d = json.loads(line)
        code = d.get("loinc_code")
        if code not in label_space:
            continue
        m = d.get("measurements") or {}

        def kinds(system):
            out = {}
            for u in (m.get(system, {}) or {}).get("quantitative", []) or []:
                c = us.canonical_unit(u, amap) or u
                k = _kind(c)
                if k:
                    out.setdefault(k, set()).add(c)
            return out

        mine = kinds("mimic")
        ext = {s: kinds(s) for s in EXTERNAL}
        self_agree = external_self_agreement({s: ext[s] for s in EXTERNAL})
        ext_all = set().union(*[set(v) for v in ext.values()]) if ext else set()
        n_ext = sum(1 for s in EXTERNAL if ext[s])
        if not mine or not ext_all:
            verdict = "no_mimic_unit" if not mine else "no_external_unit"
            same_scale = None
        else:
            shared = set(mine) & ext_all
            verdict = "kind_agrees" if shared else "kind_differs"
            # among agreeing kinds, is the exact unit string shared too?
            same_scale = bool(shared and any(
                mine[k] & set().union(*[ext[s].get(k, set()) for s in EXTERNAL])
                for k in shared))
        diff_class = mimic_u = ext_u = None
        if verdict == "kind_differs":
            diff_class, ka, kb = best_pair(mine, ext_all)
            mimic_u = ",".join(sorted(mine.get(ka, set())))
            ext_u = ",".join(sorted(set().union(
                *[ext[s].get(kb, set()) for s in EXTERNAL]) or {""}))
        rows.append({"loinc": code, "verdict": verdict, "n_external_systems": n_ext,
                     "same_exact_unit": same_scale, "external_self_agreement": self_agree,
                     "disagreement_class": diff_class,
                     "mimic_unit": mimic_u, "external_unit": ext_u,
                     "mimic_kinds": len(mine), "external_kinds": len(ext_all)})
    return pd.DataFrame(rows)


#: a19 and its UCUM unit table, loaded by _kind() on first use
_A19 = None
_UTAB = None


def _kind(canonical_unit):
    """(dim, moleExp, atom_class) for a canonical unit, or None if the UCUM table cannot parse it.

    Delegates to a19 so that the rule is not restated here. The equivalence classes must be the ones
    the unit standardisation uses, or this analysis would describe a different partition from the
    one the models see."""
    global _A19, _UTAB
    if _A19 is None:
        spec = importlib.util.spec_from_file_location(
            "a19", HERE / "a19_scale_only_standardisation.py")
        _A19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(_A19)
        _UTAB = _A19.a16.UnitTable()
    return _A19.unit_kind(_UTAB, canonical_unit)


# --------------- what kind of disagreement is it? ---------------
#: dimension vector slots, in the order lab2clean stores them
DIM_NAMES = ("length", "time", "mass", "current", "temperature", "amount", "luminous")


def classify_pair(a, b):
    """Describe how two kinds differ. Each kind is (dim_vector, mole_exponent, atom_class)."""
    (da, ma, _), (db, mb, _) = a, b
    if a == b:
        return "identical"
    dim_same = da == db
    if dim_same and ma == mb:
        return "atom class only"          # e.g. [IU]/L vs /L: same physics, different arbitrary unit
    if ma != mb and da != db:
        # the important case. Mass-per-volume vs moles-per-volume is a different LOINC property,
        # and so in principle a different code
        return "mass vs molar"
    if ma != mb:
        return "mole exponent only"
    zero = tuple([0] * 7)
    if (tuple(da) == zero) != (tuple(db) == zero):
        return "dimensionless vs dimensioned"
    changed = [n for n, x, y in zip(DIM_NAMES, da, db) if x != y]
    return "dimension differs: " + "+".join(changed) if changed else "other"


def best_pair(mine, theirs):
    """The closest (mimic_kind, external_kind) pair and its classification.

    "Closest" means the fewest differing components. A code whose units differ only in the
    analyte-dependent unit is then not reported as a dimensional disagreement when some other
    attested unit is further away."""
    best, label = None, None
    for a in mine:
        for b in theirs:
            c = classify_pair(a, b)
            rank = {"identical": 0, "atom class only": 1, "mole exponent only": 2,
                    "dimensionless vs dimensioned": 3, "mass vs molar": 4}.get(c, 5)
            if best is None or rank < best[0]:
                best, label = (rank, a, b), c
    return label, (best[1] if best else None), (best[2] if best else None)


def external_self_agreement(per_system_kinds):
    """Do the external systems agree with each other, ignoring MIMIC?

    This is the cleanest test of the expected case. It asks whether independent terminologies agree
    on a unit kind for the same test, with MIMIC left out. If they disagree among themselves, the
    claim that every site uses the same unit type is weaker than it looks. MIMIC differing from them
    then says less."""
    sysk = {s: set(k) for s, k in per_system_kinds.items() if k}
    if len(sysk) < 2:
        return "single_system" if len(sysk) == 1 else "no_system"
    common = set.intersection(*sysk.values())
    union = set.union(*sysk.values())
    if common:
        return "all_agree" if len(union) == len(common) else "overlap"
    return "disjoint"


# --------------------------- figures ---------------------------
def figures(std, raw, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # F1: codes per unit, as sorted horizontal bars because it compares magnitudes over named units.
    # The code count and the observation count have different scales, so they get two panels that
    # share the unit labels, which avoids a dual axis.
    d = std.sort_values(["n_loinc", "n_obs"]).reset_index(drop=True)
    y = np.arange(len(d))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 8.6), sharey=True,
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    h = 0.38
    ax.barh(y + h / 2, d.n_loinc, height=h, color=C_LOINC, label="LOINC codes", zorder=3)
    ax.barh(y - h / 2, d.n_mimic_item, height=h, color=C_MIMIC, label="MIMIC item codes", zorder=3)
    for i, r in d.iterrows():                      # direct labels, but not a number on every mark
        if r.n_loinc >= 10 or r.n_loinc <= 2:
            ax.text(r.n_loinc + 1.5, i + h / 2, f"{int(r.n_loinc)}", va="center", fontsize=7,
                    color=INK2)
    ax.axvline(1, color="#d03b3b", lw=1.2, zorder=4)
    # the one-code units sort to the bottom (ascending), so the callout goes there. At the top it
    # would point at the many-code rows and say the opposite of what it means.
    n_one = int((d.n_loinc == 1).sum())
    ax.annotate(f"{n_one} units identify a single LOINC code",
                xy=(1, n_one / 2), xytext=(14, n_one / 2 - 1.5),
                color="#d03b3b", fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=1.1))
    ax.set_yticks(y); ax.set_yticklabels(d.unit, fontsize=8)
    ax.set_xlabel("distinct codes using this unit")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    ax2.barh(y, d.n_obs, height=.62, color=SEQ, zorder=3)
    # A linear axis, because a log axis flatters the tail and makes a 4-observation unit look
    # comparable to one with hundreds of thousands. On a linear axis the low-volume units are
    # invisible, which is the finding. The direct labels still give their exact values.
    ax2.set_xlabel("observations")
    ax2.grid(axis="x", color=GRID, lw=.6, zorder=0); ax2.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    for i, r in d.iterrows():
        ax2.text(d.n_obs.max() * 0.015 + r.n_obs, i, f"{int(r.n_obs):,}", va="center",
                 fontsize=7, color=INK2)
    ax2.set_xlim(0, d.n_obs.max() * 1.28)
    ax2.ticklabel_format(axis="x", style="plain")

    fig.suptitle("Codes per unit after v2 standardisation, and observations per unit",
                 fontsize=11, color=INK, x=.5, y=.975)
    fig.text(.5, .945, "a unit used by one code identifies it; the right panel shows how much "
             "data such units carry", ha="center", fontsize=8.5,
             color=INK2)
    fig.tight_layout(rect=[0, 0, 1, .935])
    fig.savefig(outdir / "F1_codes_per_unit.png", dpi=170, facecolor="#fcfcfb")
    plt.close(fig)

    # F2: did standardising concentrate or dilute? Before/after on the same axis.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist([raw.n_loinc, std.n_loinc], bins=[0.5, 1.5, 2.5, 5.5, 10.5, 25.5, 60.5, 140.5],
            color=[C_MIMIC, C_LOINC], label=["raw units", "standardised (v2)"], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("distinct LOINC codes using the unit")
    ax.set_ylabel("units")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Standardising reduced one-code units, "
                 f"{int((raw.n_loinc == 1).sum())} → {int((std.n_loinc == 1).sum())}",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(outdir / "F2_concentration_shift.png", dpi=170, facecolor="#fcfcfb")
    plt.close(fig)


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--few", type=int, default=2, help="deep-dive units with <= this many codes")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    amap = load_unit_alias_map(str(FHIR / "unit_synonym_map.json"))

    df = load_observations(amap, args.split)
    print(f"{len(df):,} observations carrying a unit ({args.split} split), "
          f"{df.loinc.nunique()} LOINC codes, {df.item_code.nunique()} MIMIC items\n")

    levels = {"raw": "val_unit", "canonical": "canon", "standardised": "std"}
    tables = {k: concentration(df, c) for k, c in levels.items()}
    print(f"{'level':14s} {'units':>6s} {'median':>7s} {'mean':>6s} {'max':>5s} "
          f"{'1-code units':>13s} {'their obs':>11s}")
    for k, t in tables.items():
        one = t[t.n_loinc == 1]
        print(f"{k:14s} {len(t):6d} {int(t.n_loinc.median()):7d} {t.n_loinc.mean():6.1f} "
              f"{int(t.n_loinc.max()):5d} {len(one):13d} {100*one.n_obs.sum()/len(df):10.1f}%")
        t.insert(0, "level", k)
    pd.concat(tables.values()).to_csv(OUT / "a20_codes_per_unit.csv", index=False)

    # -- the deep dive --
    std = tables["standardised"]
    few = std[std.n_loinc <= args.few].sort_values("n_obs", ascending=False)
    wide = loinc_wide_unit_usage()
    print(f"\n{'='*96}\nUNITS USED BY <= {args.few} CODES (the deep-dive set)\n{'='*96}")
    print(f"{'unit':14s} {'obs':>9s} {'%obs':>6s} {'patients':>9s} {'LOINC':>6s} {'MIMIC':>6s} "
          f"{'codes in FULL LOINC':>20s}  verdict")
    rows = []
    for r in few.itertuples():
        n_wide = len(wide.get(r.unit, ()))
        # Is the uniqueness local or general? This only means something when LOINC lists the unit.
        # The count alone cannot say whether a unit really names an analyte. That needs the
        # identities, which are printed below. `pg` is listed by 10 LOINC codes and all ten are
        # "Entitic mass" measurements, so it names a measurement type. A verdict phrased as "real vs
        # artefact" would read that as dilution. So these labels only describe the candidate set
        # that the unit narrows to.
        if n_wide == 0:
            v = "not a LOINC example unit"
        elif n_wide <= 20:
            v = f"narrows to {n_wide} candidates LOINC-wide"
        else:
            v = f"common LOINC-wide ({n_wide}) — uniqueness is local to our label space"
        print(f"{r.unit:14s} {r.n_obs:9,d} {r.pct_obs:5.2f}% {r.n_patients:9,d} "
              f"{r.n_loinc:6d} {r.n_mimic_item:6d} {n_wide:20d}  {v}")
        rows.append({"unit": r.unit, "n_obs": r.n_obs, "pct_obs": r.pct_obs,
                     "n_patients": r.n_patients, "n_loinc": r.n_loinc,
                     "n_mimic_item": r.n_mimic_item, "n_codes_full_loinc": n_wide, "verdict": v})
    dd = pd.DataFrame(rows)
    dd.to_csv(OUT / "a20_few_code_units.csv", index=False)

    # Which other codes, including those outside the label space, use these units? If they are all
    # the same kind of analyte, the unit names a measurement type.
    names = {}
    with open(LOINC_CSV, newline="") as f:
        for r in csv.DictReader(f):
            names[r["LOINC_NUM"]] = (r.get("LONG_COMMON_NAME") or r.get("COMPONENT") or "").strip()
    print(f"\n{'='*96}\nWHO ELSE USES THESE UNITS, ACROSS ALL OF LOINC?\n{'='*96}")
    for r in few.sort_values("n_obs", ascending=False).head(4).itertuples():
        others = sorted(wide.get(r.unit, ()))
        ours = set(df[df["std"] == r.unit].loinc)
        print(f"\n  {r.unit}: {len(others)} LOINC codes list it as an example unit "
              f"({len(ours)} of them in our label space)")
        for c in others[:12]:
            mark = "  <-- ours" if c in ours else ""
            print(f"      {c:10s} {names.get(c, '')[:62]}{mark}")
        if len(others) > 12:
            print(f"      … and {len(others) - 12} more")
    print(f"\n  total on <= {args.few}-code units: {few.n_obs.sum():,} observations "
          f"({100*few.n_obs.sum()/len(df):.2f}%), {few.n_patients.max():,} patients at most in one")

    # -- the external attestation check --
    print(f"\n{'='*96}\nDO OTHER TERMINOLOGIES ATTEST THE SAME UNIT KIND?\n{'='*96}")
    ext = external_attestation(amap)
    # One denominator. Everything here is on the chosen split (test by default), and so is the
    # volume weighting. Label-space codes that never appear in it contribute 0 and are counted as such.
    vol = df.groupby("loinc").size()
    ext["n_obs"] = ext.loinc.map(vol).fillna(0).astype(int)
    ext.to_csv(OUT / "a20_external_attestation.csv", index=False)
    n_obs_total = int(ext.n_obs.sum())

    print(f"{'verdict':20s} {'codes':>6s} {'% codes':>8s} {'observations':>13s} {'% obs':>7s}")
    for k in ("kind_agrees", "kind_differs", "no_external_unit", "no_mimic_unit"):
        sub = ext[ext.verdict == k]
        if len(sub):
            print(f"  {k:18s} {len(sub):6d} {100*len(sub)/len(ext):7.1f}% "
                  f"{int(sub.n_obs.sum()):13,d} {100*sub.n_obs.sum()/max(n_obs_total,1):6.1f}%")
    comp = ext[ext.verdict.isin(["kind_agrees", "kind_differs"])]
    agree = comp[comp.verdict == "kind_agrees"]
    if len(comp):
        print(f"\n  Of the {len(comp)} codes where both sides give a parseable unit, "
              f"{len(agree)} agree on kind")
        print(f"     by code        {100*len(agree)/len(comp):5.1f}%")
        print(f"     by observation {100*agree.n_obs.sum()/max(comp.n_obs.sum(),1):5.1f}%   "
              f"({int(agree.n_obs.sum()):,} of {int(comp.n_obs.sum()):,})")
        ex = agree.same_exact_unit
        print(f"     of the agreeing codes, {int(ex.sum())} ({100*ex.mean():.1f}%) share the exact "
              "unit string; the rest differ only in scale")

    # -- what kind of disagreement? --
    dis = ext[ext.verdict == "kind_differs"].sort_values("n_obs", ascending=False)
    print(f"\n{'='*96}\nWHAT KIND OF DISAGREEMENT? ({len(dis)} codes, "
          f"{int(dis.n_obs.sum()):,} observations = "
          f"{100*dis.n_obs.sum()/max(n_obs_total,1):.2f}% of the split)\n{'='*96}")
    cls = dis.groupby("disagreement_class").agg(codes=("loinc", "size"), obs=("n_obs", "sum"))
    cls = cls.sort_values("obs", ascending=False)
    print(f"{'class':34s} {'codes':>6s} {'observations':>13s} {'% of split':>11s}")
    for k, r in cls.iterrows():
        print(f"  {str(k):32s} {int(r.codes):6d} {int(r.obs):13,d} "
              f"{100*r.obs/max(n_obs_total,1):10.3f}%")
    print("\n  the highest-volume disagreements:")
    print(f"  {'code':10s} {'obs':>9s} {'class':32s} {'MIMIC':16s} external")
    for r in dis.head(12).itertuples():
        print(f"  {r.loinc:10s} {r.n_obs:9,d} {str(r.disagreement_class)[:32]:32s} "
              f"{str(r.mimic_unit)[:16]:16s} {str(r.external_unit)[:34]}")

    # -- do the external systems agree with each other? --
    print(f"\n{'='*96}\nDO THE EXTERNAL TERMINOLOGIES AGREE WITH EACH OTHER (MIMIC removed)?"
          f"\n{'='*96}")
    multi = ext[~ext.external_self_agreement.isin(["no_system", "single_system"])]
    sa = multi.groupby("external_self_agreement").agg(codes=("loinc", "size"),
                                                      obs=("n_obs", "sum"))
    for k, r in sa.sort_values("obs", ascending=False).iterrows():
        print(f"  {k:14s} {int(r.codes):5d} codes  {int(r.obs):11,d} obs  "
              f"{100*r.codes/len(multi):5.1f}% of the {len(multi)} codes with >=2 systems")
    print(f"  (single_system: {(ext.external_self_agreement=='single_system').sum()} codes, "
          f"no_system: {(ext.external_self_agreement=='no_system').sum()})")

    figures(tables["standardised"], tables["raw"], OUT)
    print(f"\nwrote {OUT}/a20_*.csv and F1/F2 .png")


if __name__ == "__main__":
    main()
