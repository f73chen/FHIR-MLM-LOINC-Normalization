"""Freeze the per-code perturbation spec into the table that the dataset builds read (thesis Chapter 4
and Appendix B).

Every magnitude, scale class and guard flag is resolved once and written to
data/variability/perturbation_spec.csv. The dataset builders read that table through
perturbation.load_spec, and the magnitude tables in Chapter 4 and Appendix B read it too. Nothing
re-derives the magnitudes at build time, because a table derived from the data at hand could not be
rebuilt at a second site. `unit_standardisation.STANDARD_UNIT` is frozen for the same reason.

Inputs, with the script that builds or checks each:
  variability_register.py   per-analyte magnitudes           (analysis/a27_register_verification.py)
  measurement_scales.csv    scale class, native precision    (analysis/a30_measurement_scales.py)
  numeric_obs_counts.csv    which codes carry numbers        (pipeline/build_numeric_obs_counts.py)
  stated_intervals.csv      each code's stated interval      (analysis/a37_stated_intervals.py)
  Loinc.csv (UPSTREAM_DIR)  CLASS and PROPERTY, for the class partition
It also uses classification code from analysis/a21, a25 and a30.

A code's magnitudes are resolved in this order: per-analyte observed, then per-analyte specification,
then the class default (the maximum over the class's anchors), then the median fallback. The guard
columns are marked below with their guard's name (G0, G1, ...). pipeline/perturbation.py applies
them.

  python pipeline/build_perturbation_spec.py
"""
import csv, importlib.util, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
HERE = ROOT / "analysis"
sys.path.insert(0, str(ROOT / "pipeline"))
import variability_register as vr  # noqa: E402

LOINC_CSV = locations.UPSTREAM / "Loinc.csv"
STATED_INTERVALS = ROOT / "data" / "variability" / "stated_intervals.csv"      # a37
OUT = ROOT / "data" / "variability" / "perturbation_spec.csv"

#: One SD of between-site range shift may be at most this many half-widths of the interval it shifts.
#: analysis/a37_stated_intervals.py measures it. It pairs each of the 12 analytes that have a
#: measured %V with MIMIC's modal interval for the same code, and forms %V / halfwidth_pct.
#: The 12 ratios run from 0.25 (sodium) to 1.16 (aPTT), with a median of 0.56. That is a 4.6x spread,
#: against 44x for %V itself, so an interval's width predicts its between-site movement far better
#: than any unconditional magnitude does. 1.16 is the largest measured ratio, so a code may move as
#: far as the most mobile analyte in the literature.
#:
#: The cap applies only where a code took a class default or the global fallback. It never overrides
#: a per-analyte measurement, and it can only lower a magnitude. It is needed because a class default
#: is the maximum anchor in the class, so narrow homeostatic codes inherit it from a wide-interval
#: member. Chloride's stated 96–108 would take magnesium's 20.8%, a shift of 3.54 half-widths per SD.
#: At that shift the chloride intervals of two sites would not overlap.
MAX_MEASURED_RATIO = 1.16

#: The weakest tier, used only where a code's class has no anchor: the median of the 33
#: measured CVs (values) and of the 12 measured %V (ranges).
VALUE_FALLBACK, RANGE_FALLBACK = 5.1, 16.2

#: G1: codes for which a negative value is valid, so the clamp must skip them. Keyed on LOINC
#: COMPONENT, because this is a property of the analyte.
NEGATIVE_OK = {"Base excess", "Anion gap", "Bilirubin.indirect", "Cholesterol in LDL"}

#: G3: derived quantities, which are recomputed from their perturbed parents.
#: Only the red-cell indices qualify. They are published as calculated (Sarma 1990, Clinical Methods
#: ch. 152: "these indices are calculated and are not directly measured"), and they match MIMIC at a
#: median error of 0.12–0.28% over about 500,000 panels. Anion gap, absolute differential counts,
#: INR and the bilirubin fractions are left out. The way their panels are built made the relation
#: impossible to verify in MIMIC. INR cannot be recomputed, because the ISI and the mean
#: normal PT are not in MIMIC.
DERIVED = {
    "MCV":  ("Hct", "RBC", lambda hct, rbc: hct / rbc * 10.0),
    "MCH":  ("Hb",  "RBC", lambda hb,  rbc: hb / rbc * 10.0),
    "MCHC": ("Hb",  "Hct", lambda hb,  hct: hb / hct * 100.0),
}
#: the parent roles, matched on LOINC LONG_COMMON_NAME
PARENT_RX = {
    "Hb":  r"^Hemoglobin \[Mass/volume\] in Blood$",
    "Hct": r"^Hematocrit \[Volume Fraction\] of Blood by Automated count$",
    "RBC": r"^Erythrocytes \[#/volume\] in Blood by Automated count$",
}
DERIVED_RX = {
    "MCV":  r"^MCV \[Entitic mean volume\]",
    "MCH":  r"^MCH \[Entitic mass\]",
    "MCHC": r"^MCHC \[Entitic Mass/volume\]",
}
#: How a magnitude is read. Each code states its kind. Deducing the kind from the scale class would
#: give log codes a relative CV that the runtime applies as an absolute offset, which turns a blood
#: pH of 7.40 into 12.5.
#:
#:   relative_pct    v * (1 + z*mag/100)              the ordinary case
#:   absolute        v + z*mag                        mag is in the value's units
#:   excess_over_1   1 + (v-1)*(1 + z*mag/100)        specific gravity: a ratio to water
#:   rung            move one step along the observed value grid
#:   poisson         v + z*sqrt(v)                    counting objects in a field
#:
#: Blood and body-fluid pH take an absolute offset of 0.014 pH units. CLIA bounds blood pH at +-0.04,
#: an absolute limit. analysis/a22_clia_divisor.py puts the
#: median ratio of a CLIA tolerance to an observed CV at k = 2.82, which gives 0.04 / 2.82. This is
#: derived from a tolerance. Over a22's k range of 1.06–4.73 it would be 0.008–0.038. Urine pH cannot
#: use it, because its grid step is 0.5 and a 0.014 offset rounds to zero. Urine pH moves by rungs
#: instead.
PH_ABSOLUTE_OFFSET = 0.014
BLOOD_PH_RX = r"^pH of (Arterial |Venous |Capillary |Mixed venous )?[Bb]lood|^pH of Body fluid"

#: G4: fractions of one whole, renormalised to the panel's original sum after perturbation.
FRACTION_PANEL_RX = r"^(Neutrophils|Lymphocytes|Monocytes|Eosinophils|Basophils)/Leukocytes in Blood"


def main():
    a21 = _load("a21", "a21_perturbation_coverage.py")
    a25 = _load("a25", "a25_partition_comparison.py")
    a30 = _load("a30", "a30_measurement_scales.py")
    scales = {r["loinc"]: r for r in csv.DictReader(
        open(ROOT / "data/variability/measurement_scales.csv"))}
    numeric = {r["loinc"]: int(r["train_obs_numeric"]) for r in csv.DictReader(
        open(ROOT / "data/variability/numeric_obs_counts.csv")) if int(r["train_obs_numeric"])}
    with open(LOINC_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        meta = {r["LOINC_NUM"]: r for r in csv.DictReader(fh) if r["LOINC_NUM"] in numeric}

    endo = re.compile(
        r"^(Thyrotropin|Thyroxine|Triiodothyronine|Cortisol|Prostate specific|"
        r"Choriogonadotropin|Cancer Ag|Calcitonin|Alpha-1-Fetoprotein|Ferritin|Folate|"
        r"Cobalamin|Vitamin D|Calcidiol|Parathyrin|Aldosterone|Insulin|Testosterone|"
        r"Estradiol|Progesterone|Prolactin|Follitropin|Lutropin|Somatotropin|"
        r"Androstenedione|Dehydroepiandrosterone|Sex hormone|Homocysteine|"
        r"17-Hydroxyprogesterone|Natriuretic peptide|Carcinoembryonic)", re.I)

    def p4_group(r):
        b = f"{r['CLASS']} / {a25.PROP_GROUP.get(r['PROPERTY'], r['PROPERTY'])}"
        if r["CLASS"] != "CHEM":
            return b
        return b + (" [endocrine]" if endo.match(r["LONG_COMMON_NAME"]) else " [general chem]")

    if not STATED_INTERVALS.exists():
        sys.exit(f"missing {STATED_INTERVALS} (run analysis/a37_stated_intervals.py first)")
    halfwidth = {r["loinc"]: float(r["halfwidth_pct"])
                 for r in csv.DictReader(open(STATED_INTERVALS)) if r["halfwidth_pct"]}

    anchors = vr.anchors_for_matching()
    per_v, per_r, grp_v, grp_r = {}, {}, defaultdict(list), defaultdict(list)
    for c, r in meta.items():
        n = r["LONG_COMMON_NAME"]
        for a, (rx, v, tier, g, src) in anchors.items():
            if re.search(rx, n, re.I):
                if v is not None:
                    per_v[c] = (v, "observed" if tier == "obs" else "specification", a)
                if g is not None:
                    per_r[c] = (g, "observed", a)
                break
        k = p4_group(r)
        if c in per_v:
            grp_v[k].append(per_v[c][0])
        if c in per_r:
            grp_r[k].append(per_r[c][0])

    rows = []
    for c, r in sorted(meta.items()):
        n, sc = r["LONG_COMMON_NAME"], scales.get(c, {})
        grp = p4_group(r)
        scale_src = "a30 (measured)"
        if not sc:
            # No measurement_scales row. a30 skips a code with fewer than MIN_OBS = 200
            # observations, because below that the value distribution says little. Falling through
            # to plain `ratio` would contradict a30's precedence, in which PROPERTY decides first
            # and needs no observations, since it is published LOINC metadata. For
            # example, 41284-1 (Non-sq Epi Cells #/area UrnS HPF) "Epithelial cells.non-squamous
            # [#/area] in Urine sediment" has 64 observations and PROPERTY=Naric. It is a
            # microscope-field count like its siblings 13945-1 (RBC #/area UrnS HPF) and
            # 5787-7 (Epi Cells #/area UrnS HPF). As `ratio`, with the 71% class CV, a count of 1
            # would clamp to 0. Fractions and log-scale codes with few observations would be
            # misclassified the same way.
            #
            # This reuses a30.classify. The empirical arguments get values that cannot satisfy the
            # empirical branches (n_distinct above LADDER_MAX, top-12 share and integer share at
            # zero). Only the PROPERTY-driven branches can then fire, and everything else still
            # lands on `ratio`.
            scale, why = a30.classify(r.get("PROPERTY", ""), r.get("EXAMPLE_UCUM_UNITS", ""),
                                      n_distinct=10 ** 9, top12_share=0.0, pct_integer=0.0,
                                      vmin=0.0, vmax=0.0)
            sc = {"scale": scale}
            scale_src = f"LOINC PROPERTY only, no a30 row: {why}"
        tier0 = a21.classify(r)[0] == "not an analytical measurement"       # G0

        # --- how the magnitude is to be read -------------------------------------
        scale = sc.get("scale", "ratio")
        ladder = sc.get("ladder_values", "")
        if re.match(BLOOD_PH_RX, n):
            mkind, vmag, vtier, vsrc = "absolute", PH_ABSOLUTE_OFFSET, "derived", "CLIA 0.04 / k 2.82"
        elif scale == "log" and ladder:
            mkind = "rung"                      # urine pH: a 9-step grid on a log scale
        elif scale == "log":
            mkind = "relative_pct"              # log viral loads: relative in log space
        elif scale == "ratio_near_1":
            mkind = "excess_over_1"
        elif scale == "ladder":
            mkind = "rung"
        elif scale == "count_poisson":
            mkind = "poisson"
        else:
            mkind = "relative_pct"
        if mkind == "rung" and not ladder:
            sys.exit(f"{c} ({n}) needs rungs but measurement_scales.csv has no ladder_values")

        if mkind != "absolute":                       # the pH branch already set these
            vmag, vtier, vsrc = per_v.get(c, (None, None, None))
        if vmag is None:
            vmag, vtier, vsrc = ((max(grp_v[grp]), "class", grp) if grp_v[grp]
                                 else (VALUE_FALLBACK, "fallback", "median of 33 measured CVs"))
        rmag, rtier, rsrc = per_r.get(c, (None, None, None))
        if mkind == "absolute":
            # magnitude_kind is a property of the scale, so it governs the value and the range
            # alike. The two magnitudes come from different places, though. Blood pH's value
            # magnitude is an absolute 0.014 pH units, while its range magnitude would otherwise be
            # the register's percentage fallback (16.2). Applied as an absolute offset, that would
            # move a reference range of 7.35–7.45 to 16.13–16.23.
            #
            # No source gives a %V for blood pH. Koerbin 2018 reports blood-gas reference intervals
            # as nationally harmonised (pH 7.35–7.45 is the Australian consensus), so the true
            # between-site spread is small. Using the value's absolute offset assumes that the
            # stated interval moves about as much as the measurement does. This is conservative and
            # on the right scale.
            rmag, rtier, rsrc = PH_ABSOLUTE_OFFSET, "derived", "same absolute offset as the value"
        if rmag is None:
            rmag, rtier, rsrc = ((max(grp_r[grp]), "class", grp) if grp_r[grp]
                                 else (RANGE_FALLBACK, "fallback", "median of 12 measured %V"))

        # --- cap a borrowed magnitude at what the interval can absorb (see MAX_MEASURED_RATIO) ----
        # Only relative_pct is capped. It is the only kind whose magnitude is a percentage of the
        # midpoint, and so the only kind whose magnitude can be set against the half-width. Urine
        # specific gravity, for example, has a nominal ratio of 9.7 but an actual interval overlap of
        # 94%, because its excess_over_1 rule perturbs only the distance from 1.0.
        hw, rmag_raw, capped = halfwidth.get(c), rmag, 0
        if hw and mkind == "relative_pct" and rtier in ("class", "fallback"):
            lim = MAX_MEASURED_RATIO * hw
            if lim < rmag:
                rmag, capped = lim, 1
                rsrc = f"{rsrc}, capped at {MAX_MEASURED_RATIO} x stated half-width"
        derived = next((k for k, rx in DERIVED_RX.items() if re.match(rx, n)), "")
        parent = next((k for k, rx in PARENT_RX.items() if re.match(rx, n)), "")
        rows.append({
            "loinc": c, "name": n, "class_group": grp,
            "tier0_no_perturb": int(tier0),                                  # G0
            "value_cv_pct": "" if tier0 else round(vmag, 3), "value_tier": "" if tier0 else vtier,
            "value_source": "" if tier0 else vsrc,
            "range_pctV": "" if tier0 else round(rmag, 3), "range_tier": "" if tier0 else rtier,
            "range_source": "" if tier0 else rsrc,
            "range_pctV_uncapped": "" if tier0 else round(rmag_raw, 3),
            "range_cap_applied": "" if tier0 else capped,
            "stated_halfwidth_pct": "" if (tier0 or not hw) else round(hw, 3),
            "scale_class": scale,                                             # G6
            "scale_source": scale_src,
            "magnitude_kind": "" if tier0 else mkind,                          # G6
            "native_decimals": sc.get("native_decimals", "2"),                # G7
            "ladder_values": sc.get("ladder_values", ""),                     # G6
            "allow_negative": int(r["COMPONENT"].strip() in NEGATIVE_OK),     # G1
            # G2 also follows the PROPERTY. The clamp is a guard ("a fraction of a whole cannot
            # exceed 100") that holds however precisely the fraction is reported, so it does not
            # depend on which scale rule won. Taken from `scale_class == "bounded_frac"` alone, it
            # would miss every fraction that a30 classes as `integer_grid`, because a30's precedence
            # puts the >= 99%-integer test first. Oxygen saturation could then reach 113% and a
            # haematocrit 104%.
            #
            # A few of these codes already hold source values above 100 (MIMIC data errors: SpO2
            # up to 932, Hct up to 288), so clamping alters them. G2 already does the same for
            # haematocrit 4544-3 (Hct VFr Bld Auto, max 248), and two haematocrit codes must be
            # treated alike.
            "clamp_0_100": int(sc.get("scale") == "bounded_frac"
                               or (r.get("PROPERTY", "") in a30.FRAC_PROPS
                                   and r.get("EXAMPLE_UCUM_UNITS", "").strip()
                                   in ("%", "% of total", ""))),                # G2
            "derived_role": derived,                                          # G3
            "parent_role": parent,                                            # G3
            "fraction_panel": int(bool(re.match(FRACTION_PANEL_RX, n))),      # G4
            "n_obs": numeric[c],
        })
    # A regex that matches nothing stops the build, because the guard would otherwise shrink. MCHC's
    # name, for example, is "MCHC [Entitic Mass/volume]". A pattern written for "MCHC [Mass/volume]"
    # would miss it, let every MCHC be perturbed independently and break the identity the guard
    # preserves.
    for role in DERIVED_RX:
        hits = [r for r in rows if r["derived_role"] == role]
        if not hits:
            sys.exit(f"G3: derived role {role!r} matched no code. Check DERIVED_RX[{role!r}].")
    for role in PARENT_RX:
        hits = [r for r in rows if r["parent_role"] == role]
        if not hits:
            sys.exit(f"G3: parent role {role!r} matched no code. Check PARENT_RX[{role!r}].")
    n_frac = sum(1 for r in rows if r["fraction_panel"])
    if n_frac < 5:
        sys.exit(f"G4: expected the 5-part differential, matched {n_frac} codes")
    # the cap is why stated_intervals.csv is read, so check here that the cap holds
    over = [r for r in rows if not r["tier0_no_perturb"] and r["stated_halfwidth_pct"]
            and r["magnitude_kind"] == "relative_pct" and r["range_tier"] in ("class", "fallback")
            # 1e-2 absorbs the 3-decimal rounding that both columns carry. A real violation is a
            # whole magnitude out (chloride would be 20.8 against a limit of 6.8), never 0.0005
            and r["range_pctV"] > MAX_MEASURED_RATIO * r["stated_halfwidth_pct"] + 1e-2]
    if over:
        sys.exit(f"cap did not hold for {len(over)} codes, e.g. {over[0]['loinc']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    TOT = sum(r["n_obs"] for r in rows)
    f = lambda p: 100 * sum(r["n_obs"] for r in rows if p(r)) / TOT
    print(f"{len(rows)} codes, {TOT:,} numeric observations\n")
    print(f"  G0 tier-0, never perturbed        {f(lambda r: r['tier0_no_perturb']):6.2f}%")
    print(f"  G1 negative allowed               {f(lambda r: r['allow_negative']):6.2f}%")
    print(f"  G2 clamped to [0,100]             {f(lambda r: r['clamp_0_100']):6.2f}%")
    print(f"  G3 derived (recomputed)           {f(lambda r: r['derived_role']):6.2f}%")
    print(f"     ...their parents               {f(lambda r: r['parent_role']):6.2f}%")
    print(f"  G4 fraction panel (renormalised)  {f(lambda r: r['fraction_panel']):6.2f}%")
    print(f"  G6 non-ratio scale                {f(lambda r: r['scale_class'] not in ('ratio','integer_grid','bounded_frac')):6.2f}%")
    print("\n  value magnitude tier:")
    for t in ("observed", "specification", "class", "fallback"):
        print(f"     {t:14s} {f(lambda r, t=t: r['value_tier'] == t):6.2f}%")
    print("  range magnitude tier:")
    for t in ("observed", "class", "fallback"):
        print(f"     {t:14s} {f(lambda r, t=t: r['range_tier'] == t):6.2f}%")
    cap = [r for r in rows if r["range_cap_applied"] == 1]
    print(f"\n  range magnitude capped at {MAX_MEASURED_RATIO} x the stated half-width: "
          f"{len(cap)} codes, {f(lambda r: r['range_cap_applied'] == 1):.2f}% of observations")
    for r in sorted(cap, key=lambda r: -r["n_obs"])[:8]:
        print(f"     {r['loinc']:<9}{r['name'][:38]:<40}{r['range_pctV_uncapped']:6.1f} ->"
              f"{r['range_pctV']:6.1f}   (half-width {r['stated_halfwidth_pct']:5.1f}%)")
    print(f"\nwrote {OUT}")


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


if __name__ == "__main__":
    sys.exit(main())
