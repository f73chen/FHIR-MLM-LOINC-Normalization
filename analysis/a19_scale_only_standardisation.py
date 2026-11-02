"""
Code-blind, scale-only unit standardisation, and how to write the resulting numbers (thesis
Chapter 4 and Appendix A).

a16's A3 standardises units within (LOINC PROPERTY, dimension, mole exponent). PROPERTY is a
per-code label, though. At deployment a receiving site should need only the unit string, with no
property table and no code mapping.

A purely dimensional rule is not safe on its own. UCUM treats International Units as dimensionless,
so `[IU]/L -> /L` comes back at factor 1.0 and would delete the IU marker. "40 [IU]/L of ALT" would
become "40 /L". This script therefore uses a rule that is both code-blind and safe. It allows a
conversion only between units of the same kind, as defined below.

     10*3/uL -> 10*9/L   allowed (both are a plain count per volume, factor 1)
     g/dL    -> g/L      allowed (both are grams per volume, factor 10)
     [IU]/L  -> /L       refused (deletes the IU atom)
     meq/L   -> mmol/L   refused code-blind (equivalents depend on valence, which the unit string
                         does not carry; a16 needs the LOINC COMPONENT to allow these)
     mg/dL   -> mmol/L   refused (different mole exponent, already blocked dimensionally)

`kind` is (dimension vector, mole exponent, analyte-dependent unit). All three are read from the
unit string alone, and `atom_class` below reads the third.
a20 uses `unit_kind`, and the table of standard units in pipeline/unit_standardisation.py follows
these kinds.

The second question is about notation. Standardising to one scale per kind makes some numbers very
large (a white cell count of 7 becomes 7000000000 per litre). Does scientific notation that keeps the
original significant figures fix that? The lookup floor cannot see notation, because it keys on the
numeric value. Part B therefore measures the cost of each notation in tokens, with the ModernBERT
tokenizer.

Reads the a15 cache. --stage kinds writes results/analysis/a19/a19_kinds.csv. --stage run writes
a19_identifiability.csv and a19_representation.csv. Stdout is aggregate only.

  python analysis/a19_scale_only_standardisation.py --stage run
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_s16 = importlib.util.spec_from_file_location("a16", HERE / "a16_unit_collision.py")
a16 = importlib.util.module_from_spec(_s16); _s16.loader.exec_module(a16)
a15 = a16.a15

OUT = locations.REPO / "results/analysis/a19"
MODERNBERT = "answerdotai/ModernBERT-base"


# --------------------------- unit kind ---------------------------
# The dimension vector plus mole exponent settles almost everything. It cannot settle units whose
# numerator is dimensionless, and those are the units where a wrong conversion would do damage.
#     /L, 10*9/L, [IU]/L, U/L, [arb'U]/L, {cells}/L   all have dim (-3,0,0,0,0,0,0), mole 0
# The analyte-dependent unit (`atom_class` below) separates them. Each pattern is matched against
# the canonical UCUM string.
_BRACKET = re.compile(r"\[[^\]]*\]")        # UCUM customary/arbitrary atoms: [IU], [HPF], [GPL'U]
_ANNOT = re.compile(r"\{[^}]*\}")           # UCUM annotations: {cells}, {score}
_PREFIX = "Y|Z|E|P|T|G|M|k|h|da|d|c|m|u|n|p|f|a|z|y"
_EQ = re.compile(rf"(?<![A-Za-z])(?:{_PREFIX})?eq(?![A-Za-z])")
_U = re.compile(rf"(?<![A-Za-z\[])(?:{_PREFIX})?U(?![A-Za-z\]])")
#: MIMIC writes `AU` to mean arbitrary units. UCUM has no `AU`, and lab2clean parses it as the
#: astronomical unit (1.496e11 m, a length dimension) on 433,001 observations. Nothing else in the
#: corpus has a length dimension, but that safety is a coincidence. Giving `AU` its own class
#: isolates it by construction, so a length-dimensioned unit cannot become convertible to it at a
#: factor of 1.5e11.
_AU = re.compile(rf"(?<![A-Za-z])(?:{_PREFIX})?AU(?![A-Za-z])")


def atom_class(u: str) -> str:
    """A signature of the unit's non-dimensional content, read from the string alone.

    Everything in square brackets is a UCUM customary or arbitrary atom whose meaning its dimensions
    do not capture. [IU], [arb'U] and [GPL'U] all parse as dimensionless. A purely dimensional rule
    would therefore convert 40 [IU]/L into 40 /L at factor 1 and delete the only part that says what
    was measured. Each bracketed atom is treated as its own class, which is the conservative choice.
    It blocks a few conversions that would have been safe, but no unsafe conversion gets through.

    `eq` and `U` are handled separately because they are written without brackets. Equivalents
    depend on valence, which the string does not carry, and enzyme units are arbitrary."""
    u = str(u)
    brackets = sorted(_BRACKET.findall(u))
    annots = sorted(_ANNOT.findall(u))
    bare = _BRACKET.sub("", u)
    # An annotation is decoration when the unit has any real atom. UCUM defines `{...}` as carrying
    # no meaning, so `{#}/[HPF]` and `/[HPF]` are the same unit and belong in the same kind. The
    # exception is a unit that is only an annotation (`{score}`). There the annotation is the unit,
    # and dropping it would merge it with every other dimensionless quantity, `%` included.
    only_annotation = not brackets and not _ANNOT.sub("", u).strip(" /1")
    parts = brackets + (annots if only_annotation else [])
    if _AU.search(_BRACKET.sub("", _ANNOT.sub("", u))):
        parts.append("AU")
    if _EQ.search(bare):
        parts.append("eq")
    if _U.search(_BRACKET.sub("", _ANNOT.sub("", u))):
        parts.append("U")
    return "|".join(sorted(parts))


def unit_kind(U, u: str):
    """(dim, moleExp, atom_class), all read from the unit string and none from the code."""
    d, m = U.dim.get(u), U.mole.get(u)
    if d is None or m is None:
        return None
    return (d, m, atom_class(u))


def scale_factor(U, src, dst):
    """Conversion factor from src to dst, defined only within a kind. Returns None across kinds."""
    if unit_kind(U, src) is None or unit_kind(U, src) != unit_kind(U, dst):
        return None
    a, b = U.mag.get(src), U.mag.get(dst)
    if not a or not b:
        return None
    return a / b


# --------------------------- notation ---------------------------
def sig_figs(v, cap=15):
    """The fewest significant figures that reproduce the value exactly.

    Two definitions are available, and they disagree:

      "digits present"   140 has three digits, so 3 s.f.
      "minimal lossless" 140 is reproduced exactly by 1.4e2, so 2 s.f.        <- used here

    Minimal lossless is used because the output goes into a model input. Writing 1.40e2 claims a
    third significant digit that the source did not establish. Writing 1.4e2 loses no information
    and adds none. Plain decimal cannot express this distinction, because "140" is ambiguous about
    whether the trailing zero was measured. This is one argument for scientific notation.

    Returns 0 for zero and for non-finite values."""
    out = np.zeros(len(v), dtype="int8")
    ok = np.isfinite(v) & (v != 0)
    if not ok.any():
        return out
    x = np.abs(v[ok])
    exp = np.floor(np.log10(x)).astype(int)
    n = np.full(len(x), cap, dtype="int8")
    found = np.zeros(len(x), dtype=bool)       # separate sentinel, because `n == 1` is a valid answer
    for k in range(1, cap + 1):
        f = np.power(10.0, k - 1 - exp)
        exact = np.abs(np.round(x * f) / f - x) <= np.abs(x) * 1e-12
        hit = exact & ~found
        n[hit] = k
        found |= exact
        if found.all():
            break
    out[ok] = n
    return out


def render_decimal(v, dec):
    """Plain decimal, at the analyte's native decimal places, as MIMIC writes the values."""
    return np.array([("" if not np.isfinite(x) else f"{x:.{d}f}")
                     for x, d in zip(v, dec)], dtype=object)


def render_fixed_sci(v, nsf=3):
    """Fixed-width scientific: always one digit, a point, nsf-1 digits, e, sign, exponent.

    The goal differs from render_scientific, which minimises what is written. This function gives
    every value the same shape, so the model sees one numeric pattern instead of dozens. 7.3 becomes
    7.30e+00 and 140 becomes 1.40e+02, which is longer than either needs. The hypothesis is that a
    uniform surface form generalises better than a compact one. The lookup floor cannot say whether
    that is worth the extra tokens. What can be measured here is the cost and the uniformity, and
    representation_report reports both.

    Lossless for 98.9% of MIMIC values. 3 s.f. covers all but 1.07% of observations."""
    return np.array([("" if not np.isfinite(x) else f"{x:.{nsf - 1}e}") for x in v], dtype=object)


def round_sig(v, nsf=3):
    """Round to nsf significant figures. This is the information that a fixed-width render loses."""
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v) & (v != 0)
    if m.any():
        f = np.power(10.0, nsf - 1 - np.floor(np.log10(np.abs(v[m]))))
        out[m] = np.round(v[m] * f) / f
    out[np.isfinite(v) & (v == 0)] = 0.0
    return out


def render_hybrid(v, sf, dec, max_chars=8):
    """Decimal when it is compact, and scientific only when it is not.

    Scientific notation for every value costs 65% more tokens. It lengthens the 97% of lab values
    that are already short (7.3 becomes 7.3e+00) to help the 2.6% that are long. Choosing per value
    keeps the short values short and the long values compact."""
    d = render_decimal(v, dec)
    sci = render_scientific(v, sf)
    return np.array([a if len(a) <= max_chars else b for a, b in zip(d, sci)], dtype=object)


def render_scientific(v, sf):
    """Scientific notation with the original significant figures: 7000000000 -> 7e+09."""
    return np.array([("" if not np.isfinite(x) else
                      ("0" if x == 0 else f"{x:.{max(int(s) - 1, 0)}e}"))
                     for x, s in zip(v, sf)], dtype=object)


# --------------------------- targets ---------------------------
def choose_targets(U, unit_counts, mode="modal"):
    """kind -> target unit.

    "modal"   the unit carrying the most observations in that kind. It is derived from the data, so
              it must be published with the model. A receiving site cannot guess MIMIC's modal unit.
    "compact" the unit that minimises total rendered digits across the kind's observations. This
              choice keeps numbers in the range that laboratories write, and it is the relevant one
              if values stay in plain decimal."""
    by_kind = collections.defaultdict(list)
    for u, n in unit_counts.items():
        k = unit_kind(U, u)
        if k is not None:
            by_kind[k].append((u, n))
    tgt = {}
    for k, members in by_kind.items():
        if mode == "modal":
            tgt[k] = max(members, key=lambda t: (t[1], -len(t[0])))[0]
        else:
            best, best_cost = None, None
            for cand, _ in members:
                cost = 0
                for u, n in members:
                    f = scale_factor(U, u, cand)
                    if f is None:
                        cost = None
                        break
                    cost += n * abs(np.log10(f) if f > 0 else 0)
                if cost is not None and (best_cost is None or cost < best_cost):
                    best, best_cost = cand, cost
            tgt[k] = best or members[0][0]
    return tgt, by_kind


def stage_kinds(U, unit_counts, n_obs):
    tgt, by_kind = choose_targets(U, unit_counts)
    rows = []
    print(f"{'kind: dimension vector | moleExp | atom class':56s} {'units':>5s} "
          f"{'obs':>11s}  target      members")
    for k, members in sorted(by_kind.items(), key=lambda kv: -sum(n for _, n in kv[1])):
        tot = sum(n for _, n in members)
        ms = sorted(members, key=lambda t: -t[1])
        lbl = f"{k[0]} mol={k[1]:g} atom={k[2] or '-'}"
        print(f"  {lbl:54s} {len(members):5d} {tot:11,d}  {tgt[k]:10s}  "
              f"{', '.join(f'{u}({n:,})' for u, n in ms[:4])}")
        for u, n in ms:
            rows.append({"dim": str(k[0]), "mole": k[1], "atom": k[2], "unit": u, "n_obs": n,
                         "target": tgt[k], "factor": scale_factor(U, u, tgt[k])})
    d = pd.DataFrame(rows)
    n_move = d[d.unit != d.target].n_obs.sum()
    print(f"\n  {len(by_kind)} kinds over {len(unit_counts)} distinct units")
    print(f"  observations whose unit string would change: {n_move:,} ({100*n_move/n_obs:.1f}%)")
    bad = d[d.factor.isna()]
    print(f"  members with no factor to their own target: {len(bad)} (should be 0)")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "a19_kinds.csv", index=False)
    print(f"\nwrote {OUT/'a19_kinds.csv'}")
    return tgt


# --------------------------- B. representation ---------------------------
def representation_report(U, v_native, u_native, v_std, u_std, ycode, ylab, sample=400_000,
                          seed=0):
    """What each rendering costs, measured in tokens with the ModernBERT tokenizer.

    The lookup floor cannot answer this, because it keys on the numeric value and does not see the
    notation. The model does read the notation, so the cost is measured here in tokens."""
    rng = np.random.default_rng(seed)
    n = len(v_native)
    idx = rng.choice(n, min(sample, n), replace=False)
    vn, un = v_native[idx], np.array(u_native, dtype=object)[idx]
    vs, us = v_std[idx], np.array(u_std, dtype=object)[idx]
    yy = ycode[idx]

    n_y = int(max(ycode.max(), 0)) + 1
    sf = sig_figs(vn)
    dec_n = _native_dec(vn, yy, n_y)
    dec_s = _native_dec(vs, yy, n_y)

    variants = {
        "native decimal":        (render_decimal(vn, dec_n[yy]), un),
        "standardised decimal":  (render_decimal(vs, dec_s[yy]), us),
        "standardised scientific": (render_scientific(vs, sf), us),
        "standardised hybrid":   (render_hybrid(vs, sf, dec_s[yy]), us),
        "standardised fixed 4sf":  (render_fixed_sci(vs, 4), us),
        "standardised fixed 3sf":  (render_fixed_sci(vs, 3), us),
        "standardised fixed 2sf":  (render_fixed_sci(vs, 2), us),
        "native fixed 3sf":      (render_fixed_sci(vn, 3), un),
        "native scientific":     (render_scientific(vn, sf), un),
    }
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODERNBERT)
    except Exception as e:
        print(f"\n  (tokenizer unavailable: {type(e).__name__}; reporting characters only)")
        tok = None

    print(f"\n{'rendering':26s} {'chars':>6s} {'maxch':>6s} {'distinct':>9s} "
          f"{'tok/obs':>8s} {'tok sd':>7s} {'tok lens':>9s} {'1 shape%':>9s}")
    rows = []
    for name, (txt, uu) in variants.items():
        lens = np.array([len(x) for x in txt])
        long = float(np.mean(lens > 6) * 100)
        ntok = tsd = nlen = modeshare = np.nan
        if tok is not None:
            sub = txt[rng.choice(len(txt), min(20000, len(txt)), replace=False)]
            counts = np.array([len(tok(x, add_special_tokens=False)["input_ids"]) for x in sub])
            ntok, tsd = float(counts.mean()), float(counts.std())
            nlen = int(len(np.unique(counts)))
            # "1 shape%" is the share of values that take the most common token count. A perfectly
            # uniform format would score 100%, with every number the same shape and only the digits
            # different. The uniformity argument for fixed-width notation rests on this quantity,
            # which is not the same as the mean.
            modeshare = float(100 * np.bincount(counts).max() / len(counts))
        print(f"{name:26s} {lens.mean():6.2f} {lens.max():6d} {len(set(txt.tolist())):9,d} "
              f"{ntok:8.2f} {tsd:7.2f} {nlen:9d} {modeshare:8.1f}%")
        rows.append({"rendering": name, "mean_chars": lens.mean(), "pct_over_6_chars": long,
                     "max_chars": int(lens.max()), "distinct_strings": len(set(txt.tolist())),
                     "mean_tokens_value_only": ntok, "sd_tokens": tsd,
                     "distinct_token_counts": nlen, "pct_in_modal_token_count": modeshare})
    pd.DataFrame(rows).to_csv(OUT / "a19_representation.csv", index=False)

    # the analytes whose numbers standardisation makes long
    grew = []
    for y in range(n_y):
        m = yy == y
        if m.sum() < 200:
            continue
        a = np.mean([len(x) for x in variants["native decimal"][0][m]])
        b = np.mean([len(x) for x in variants["standardised decimal"][0][m]])
        c = np.mean([len(x) for x in variants["standardised scientific"][0][m]])
        if b - a > 1.0:
            grew.append((ylab[y], m.sum(), a, b, c, un[m][0], us[m][0]))
    grew.sort(key=lambda t: -(t[3] - t[2]))
    if grew:
        print("\n  analytes whose numbers get longer when standardised "
              f"({len(grew)} of {n_y} codes):")
        print(f"  {'code':9s} {'n':>7s} {'native':>7s} {'std dec':>8s} {'std sci':>8s}  units")
        for code, cnt, a, b, c, uo, ut in grew[:12]:
            print(f"  {code:9s} {cnt:7,d} {a:7.1f} {b:8.1f} {c:8.1f}  {uo} -> {ut}")
    else:
        print("\n  no analyte's numbers get materially longer under standardisation")


def _native_dec(val, ycode, n_y, q=0.99, cap=6):
    dec = np.zeros(len(val), dtype="int16")
    ok = np.isfinite(val)
    need = ok.copy()
    for d in range(cap + 1):
        hit = need & (np.abs(np.round(val, d) - val) < 1e-9)
        dec[hit] = d
        need &= ~hit
    dec[need] = cap
    out = np.zeros(n_y, dtype="int16")
    for y in range(n_y):
        m = ok & (ycode == y)
        if m.any():
            out[y] = int(np.quantile(dec[m], q))
    return out


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["kinds", "run"], required=True)
    ap.add_argument("--target-mode", default="modal", choices=["modal", "compact"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from fhir_data_utils import build_lab_code_mappings
    _, class_to_loinc = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    arr = np.array([class_to_loinc.get(i, f"?{i}") for i in range(1623)], dtype=object)
    U = a16.UnitTable()

    tr_df, te_df = a15.load_cache("train"), a15.load_cache("test")
    if args.limit:
        tr_df, te_df = tr_df.head(args.limit), te_df.head(args.limit)
    tv, tur, tuc, ty_s, _ = a16.prepare(tr_df, U, arr)
    ev, eur, euc, ey_s, _ = a16.prepare(te_df, U, arr)
    del tr_df, te_df
    ycodes, ylab = pd.factorize(pd.Series(np.r_[ty_s, ey_s]))
    ty = ycodes[:len(ty_s)].astype("int32"); ey = ycodes[len(ty_s):].astype("int32")
    n_obs = len(ty) + len(ey)

    uc = collections.Counter(u for u in np.r_[tuc, euc] if u is not None)
    print(f"train {len(ty):,} / test {len(ey):,} observations; "
          f"{len(uc)} canonicalisable unit strings covering "
          f"{100*sum(uc.values())/n_obs:.1f}% of observations\n")

    if args.stage == "kinds":
        stage_kinds(U, uc, n_obs)
        return

    tgt, _ = choose_targets(U, uc, args.target_mode)

    # -- A. identifiability under scale-only standardisation --
    def standardise(val, ucan):
        out_v, out_u = val.copy(), np.array(ucan, dtype=object)
        fac = {}
        for u in uc:
            k = unit_kind(U, u)
            t = tgt.get(k)
            if t is None or t == u:
                continue
            f = scale_factor(U, u, t)
            if f is not None:
                fac[u] = (t, f)
        if fac:
            idx = collections.defaultdict(list)
            for i, u in enumerate(ucan):
                if u in fac:
                    idx[u].append(i)
            for u, rows in idx.items():
                t, f = fac[u]
                r = np.asarray(rows)
                out_v[r] = a16.keyval(val[r] * f)
                out_u[r] = t
        return out_v, out_u, fac

    t0 = time.time()
    prior = np.array(a15.topk_list(collections.Counter(ty.tolist()), max(a15.TOP_KS), tie="label"),
                     dtype="int32")
    base = a16.score(tv, tur, ty, ev, eur, ey, prior)
    tv2, tu2, fac = standardise(tv, tuc)
    ev2, eu2, _ = standardise(ev, euc)
    std = a16.score(tv2, tu2, ty, ev2, eu2, ey, prior)
    # A fixed-width scientific render also rounds every value to n significant figures. That is an
    # information cost, so it goes in the identifiability table with the standardisation cost.
    arms = [("A0 native units", base), ("S1 scale-only standardised", std)]
    for nsf in (4, 3, 2):
        r = a16.score(round_sig(tv2, nsf), tu2, ty, round_sig(ev2, nsf), eu2, ey, prior)
        arms.append((f"S1 + fixed {nsf} s.f. rounding", r))
    print(f"{'arm':34s} {'keys':>9s} {'cov%':>6s} {'H|k':>6s} {'top1cov':>8s} {'top1all':>8s} "
          f"{'collide%':>8s}")
    for nm, r in arms:
        print(f"{nm:34s} {r['n_keys']:9,d} {r['test_coverage_pct']:5.1f}% "
              f"{r['H_label_given_key_bits']:6.3f} {100*r['top1_covered']:7.2f}% "
              f"{100*r['top1_overall']:7.2f}% {r['collision_mass_pct']:7.2f}%")
    print(f"  delta top1 overall: {100*(std['top1_overall']-base['top1_overall']):+.2f} pp "
          f"({time.time()-t0:.0f}s)")
    print(f"  unit strings rewritten: {len(fac)} of {len(uc)}; "
          f"{100*sum(uc[u] for u in fac)/n_obs:.1f}% of observations")
    rows = [{"arm": nm, **r} for nm, r in arms]
    pd.DataFrame(rows).to_csv(OUT / "a19_identifiability.csv", index=False)

    # -- B. how the numbers read --
    representation_report(U, ev, euc, ev2, eu2, ey, ylab)
    print(f"\nwrote {OUT}/a19_*.csv")


if __name__ == "__main__":
    main()
