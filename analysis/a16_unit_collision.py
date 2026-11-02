"""
Does re-expressing values in equivalent units break the value+unit shortcut (thesis Chapter 4)?

a15 finds value+unit to be a strong code-identifying key. If a lab reported the same analyte in an
equivalent unit (g/L instead of g/dL) with the value scaled to match, would the key still identify
the code? a19 and a20 build on this script's unit table and scoring.

Mechanism: converting code A into unit U can only collide with code B if B natively uses U and their
value ranges overlap after conversion. Haemoglobin 12–16 g/dL becomes 120–160 g/L. Whether anything
else lives there is an empirical question, which this script answers.

Arms:
  A0    baseline, raw units as recorded                       (must reproduce a15)
  A0c   canonicalise unit spellings only, no conversion       (control: isolates spelling merges)
  A1    per-observation random draw                           (upper bound on collision damage;
                                                               unrealistic, since no lab reports one
                                                               analyte in two units)
  A2    K pseudo-sites, each picks one unit per (code,unit)   (the realistic case)
  A2x   leave-one-site-out: train on sites 0..K-2, test on    (does the key transfer to a lab whose
        site K-1, whose draw is independent                    conventions were never seen?)
  A4    train on raw MIMIC units, test on pseudo-site         (the deployment case)
        draws from a table training never saw
  A3    fully standardised: one unit per PROPERTY family      (the pre-harmonise deployment option;
                                                               the unit stops partitioning entirely)

Tiers:
  att   units actually attested for that code across MIMIC / LOINC / UCUM / CDISC / CIEL (realistic)
  prop  units sharing LOINC PROPERTY + dimension + mole exponent, decimal factor       (broader)

A conversion must keep the label. Dimension and mole exponent enforce this (lib/units.py). They
exclude conversions between mass and molar units by construction, because those are different LOINC
codes. Two independent blocklists add to this: Hauser's 285 pairs and the 814-row harmonisation map.
See tests/test_units_lib.py.

Scores go through a15's score_vec(), so every number is directly comparable to a15's table.

Reads the a15 cache. Writes results/analysis/a16/a16_alternatives.csv (--stage alts) and
a16_arms.csv (--stage run). Patient IDs are hashed to site IDs and never printed. stdout is aggregate
only.

  python analysis/a16_unit_collision.py --stage run
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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
from lib.units import (UnitTable, load_attested_units, load_ucum_property,       # noqa: E402
                       load_blocked_code_pairs, load_blocked_code_unit, _norm_key)

_spec = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
a15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a15)

OUT = locations.REPO / "results/analysis/a16"
N_SITES = 5
SEED = 0
SIGFIG = 10          # conversion factors introduce float noise; see keyval()
MISSING = "\x00"     # a15's sentinel for an absent string component


# --------------------------- helpers ---------------------------
def keyval(a):
    """Value rounded to SIGFIG significant figures.

    a15 keys the raw value at 4 decimal places, which is lossless there because MIMIC values carry
    at most 4 dp natively. It is not lossless after a conversion that scales down (0.000125 g/L
    -> 0.0001), which would merge distinct values and make collisions look worse than they are.
    Significant figures are scale-free, so every arm is keyed the same way. A0 is also reported
    under a15's exact 4 dp rule, to show the two agree on unconverted data."""
    out = np.full(a.shape, np.nan)
    m = np.isfinite(a) & (a != 0)
    if m.any():
        f = np.power(10.0, SIGFIG - 1 - np.floor(np.log10(np.abs(a[m]))))
        out[m] = np.round(a[m] * f) / f
    out[np.isfinite(a) & (a == 0)] = 0.0
    return out


def site_of(uuids, n_sites=N_SITES):
    """Stable pseudo-site per patient (a site is a lab, and a patient's labs come from one lab).

    The site comes from a hash, so the same patient lands in the same site in train and test and in
    every arm. The hash runs over the distinct patients (100k), and the result is mapped back to the
    rows (38M)."""
    codes, uniq = pd.factorize(pd.Series(uuids))
    tbl = np.array([int(hashlib.blake2b(str(u).encode(), digest_size=8).hexdigest(), 16) % n_sites
                    for u in uniq], dtype="int8")
    return tbl[codes]


# ------------------- the valence guard on equivalent-based units -------------------
# UCUM parses `eq` as a mole, so the dimension table gives meq/L to mmol/L a factor of 1.0. That
# only holds for a monovalent species. Calcium at 2 mEq/L is 1 mmol/L. Every eq/mol pair in
# MIMIC is monovalent (Na, K, Cl, HCO3, CO2, Li, anion gap, base excess, going by the COMPONENT field
# of Loinc.csv). Nothing in the unit table would stop a divalent analyte from slipping in, though.
# Hence the whitelist below. An unrecognised component is refused, so it cannot be converted at the
# wrong factor.
MONOVALENT = {"potassium", "sodium", "chloride", "bicarbonate", "carbon dioxide", "lithium",
              "anion gap", "base excess", "ammonium", "hydrogen carbonate", "nitrite", "fluoride",
              "bromide", "iodide", "hydroxide", "silver", "cesium", "rubidium"}
# an `eq` atom, optionally carrying one UCUM metric prefix (meq, ueq, keq ...). Anchored on the
# whole token so `[IU]` and annotations like `{eqv}` do not match.
_UCUM_PREFIX = "Y|Z|E|P|T|G|M|k|h|da|d|c|m|u|n|p|f|a|z|y"
_EQ_ATOM = re.compile(rf"^(?:{_UCUM_PREFIX})?eq$")
_TOKEN = re.compile(r"[A-Za-z]+")


def is_equivalent_unit(u: str) -> bool:
    """True if any atom of the unit is an equivalent. Annotations in {...} are not atoms."""
    bare = re.sub(r"\{[^}]*\}", "", str(u))
    return any(_EQ_ATOM.match(t) for t in _TOKEN.findall(bare))


def load_loinc_component(path=locations.UPSTREAM / "Loinc.csv") -> dict:
    import csv
    with open(path, newline="") as f:
        return {r["LOINC_NUM"]: (r.get("COMPONENT") or "").strip().lower()
                for r in csv.DictReader(f)}


# --------------------------- alternatives ---------------------------
def build_alternatives(pairs, verbose=True):
    """pairs: iterable of the (loinc_code, canonical_unit) pairs that were observed.

    Returns (alts_att, alts_prop, standard, U) where each alts maps (code, unit) -> [(alt, factor)]
    and `standard` maps (code, unit) -> its unit family (PROPERTY, dimension, mole exponent).
    resolve_standard() later picks the single target unit of each family for arm A3.
    Alternatives are keyed by the observed unit, because a code can be recorded in more than one unit
    and the conversion factor depends on where you start."""
    U = UnitTable()
    prop = load_ucum_property()
    hauser = load_blocked_code_pairs()
    harm = load_blocked_code_unit()

    att_canon = {}
    for code, per_system in load_attested_units().items():
        s = {c for units in per_system.values() for u in units
             if (c := U.canonical(u)) and U.known(c)}
        if s:
            att_canon[code] = s

    component = load_loinc_component()
    blocked = collections.Counter()

    def blocked_pair(code, base, alt):
        if is_equivalent_unit(base) != is_equivalent_unit(alt):
            comp = component.get(code, "")
            if not any(m == comp or comp.startswith(m + " ") for m in MONOVALENT):
                blocked["valence-unknown"] += 1
                return True
        if (code, _norm_key(alt)) in harm:
            blocked["harmonisation-map"] += 1
            return True
        for pr in hauser:
            if code in pr:
                other = next(iter(pr - {code}), None)
                if other and alt in att_canon.get(other, ()):
                    blocked["hauser-pair"] += 1
                    return True
        return False

    # every UCUM unit grouped by (PROPERTY, dimension, mole exponent)
    fam_index = collections.defaultdict(set)
    for u in U.mag:
        p = prop.get(u)
        if p:
            fam_index[(p, U.dim.get(u), U.mole.get(u))].add(u)

    alts_att, alts_prop, standard = {}, {}, {}
    for code, base in pairs:
        key = (code, base)
        a = []
        for v in sorted(att_canon.get(code, ())):
            if v == base:
                continue
            f = U.factor(base, v)
            if f is not None and not blocked_pair(code, base, v):
                a.append((v, f))
        if a:
            alts_att[key] = a

        fam = (prop.get(base), U.dim.get(base), U.mole.get(base))
        b = []
        if fam[0]:
            for v in sorted(fam_index.get(fam, ())):
                if v == base:
                    continue
                f = U.factor(base, v)
                if f is None:
                    continue
                lg = np.log10(f)
                if abs(lg - round(lg)) > 1e-9:      # decimal scalings only
                    continue
                if not blocked_pair(code, base, v):
                    b.append((v, f))
        if b:
            alts_prop[key] = b
        if fam[0]:
            standard[key] = fam                    # resolved to a unit once volumes are known
    if verbose:
        print(f"  alternatives: att {len(alts_att)} (code,unit) pairs, prop {len(alts_prop)}; "
              f"blocked {dict(blocked) or 'none'}")
    return alts_att, alts_prop, standard, U


def resolve_standard(standard, pair_counts, U):
    """A3's target unit per family: the unit carrying the most MIMIC observations in that family,
    ties broken alphabetically. The choice is deterministic and comes from the data."""
    vol = collections.defaultdict(collections.Counter)
    for (code, unit), n in pair_counts.items():
        fam = standard.get((code, unit))
        if fam:
            vol[fam][unit] += n
    chosen = {fam: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for fam, c in vol.items()}
    out = {}
    for key, fam in standard.items():
        tgt = chosen.get(fam)
        if tgt is None or tgt == key[1]:
            continue
        f = U.factor(key[1], tgt)
        if f is not None:
            out[key] = (tgt, f)
    return out


# --------------------------- arm application ---------------------------
def site_table(alt_tbl, rng, n_sites=N_SITES):
    """{(site, pair_id): index into that pair's pool}, with one draw per site per (code,unit).

    The table is precomputed over the full pair space. The draws are not taken from a live generator
    inside the row loop, because that loop only visits the pairs that occur in a split. The train and
    test splits do not contain identical pair sets, so drawing inside the loop would put the two
    streams out of step. A site would then use g/L in train and g/dL in test. The resulting drop in
    coverage would look like a collision effect, but it would be a bookkeeping artefact."""
    return {(s, p): int(rng.integers(1 + len(a[1])))
            for p, a in sorted(alt_tbl.items()) for s in range(n_sites)}


def apply_arm(arm, pair_id, groups, val, u_canon, u_raw, conv, sites, alt_tbl, std_tbl,
              rng=None, sitetbl=None):
    """Return (value, unit_string) arrays for one arm.

    `pair_id`  int32 code for (loinc, canonical unit); -1 where not canonicalisable
    `groups`   pair_id -> row indices, precomputed once
    `conv`     bool mask: this row's pair has >=1 alternative under the active tier
    `sitetbl`  the A2/A4 per-site draw, from site_table(); `rng` is used only by A1
    """
    out_v = val.copy()
    out_u = u_raw.copy()
    if arm == "A0":
        return out_v, out_u

    # every convertible row is emitted in canonical spelling, whether or not its unit changed.
    # Otherwise the identity draw would keep a raw spelling while a converted draw got a canonical
    # one, and the arm would measure spelling merges as well as conversions. A0c isolates that.
    out_u[conv] = u_canon[conv]
    if arm == "A0c":
        return out_v, out_u

    if arm == "A3":
        for pid, idx in groups.items():
            hit = std_tbl.get(pid)
            if hit is None:
                continue
            tgt, f = hit
            out_v[idx] = val[idx] * f
            out_u[idx] = tgt
        return keyval(out_v), out_u

    for pid, idx in groups.items():
        opts = alt_tbl.get(pid)
        if not opts:
            continue
        pool_u = np.array([opts[0]] + [u for u, _ in opts[1]], dtype=object)
        pool_f = np.array([1.0] + [f for _, f in opts[1]], dtype="float64")
        if arm == "A1":
            pick = rng.integers(len(pool_u), size=len(idx))
        else:                                     # A2 / A2x / A4: one draw per (site, pair)
            per_site = np.array([sitetbl[(s, pid)] for s in range(N_SITES)])
            pick = per_site[sites[idx]]
        out_v[idx] = val[idx] * pool_f[pick]
        out_u[idx] = pool_u[pick]
    return keyval(out_v), out_u


# --------------------------- scoring ---------------------------
def score(val_tr, u_tr, y_tr, val_te, u_te, y_te, prior, four_dp=False):
    """a15-comparable metrics for the value+unit key, plus collision mass."""
    def cols(v, u):
        return {"val_raw": np.round(v, 4) if four_dp else keyval(v),
                "val_u": pd.Series(u).fillna(MISSING).astype("string").to_numpy()}
    ctr, cte = cols(val_tr, u_tr), cols(val_te, u_te)
    ptr = {"v": np.isfinite(val_tr), "vu": pd.notna(u_tr)}
    pte = {"v": np.isfinite(val_te), "vu": pd.notna(u_te)}
    ktr, kte = a15.joint_codes(ctr, cte)
    r = a15.score_vec(ktr, ptr, y_tr, kte, pte, y_te, ["vraw", "unit"], prior)

    # collision mass: share of test observations whose key is ambiguous in training (>1 LOINC code)
    m_tr = ptr["v"] & ptr["vu"]
    m_te = pte["v"] & pte["vu"]
    tr = pd.DataFrame({"a": ktr["val_raw"][m_tr], "b": ktr["val_u"][m_tr], "y": y_tr[m_tr]})
    nlab = tr.drop_duplicates().groupby(["a", "b"], sort=False).size()
    te = pd.DataFrame({"a": kte["val_raw"][m_te], "b": kte["val_u"][m_te]})
    amb = te.join(nlab.rename("n"), on=["a", "b"])["n"].to_numpy()
    # fragmentation: how many distinct keys does one code spread over? n_keys/n_labels is the
    # wrong quantity, because keys are shared between codes and not every label reaches the table.
    pairs_ = tr.drop_duplicates()
    per_code = pairs_.groupby("y", sort=False).size()
    r["n_codes_in_table"] = int(len(per_code))
    r["keys_per_code_mean"] = float(per_code.mean())
    r["keys_per_code_median"] = float(per_code.median())
    r["keys_per_code_p90"] = float(per_code.quantile(0.90))
    # ... and the other direction: distinct codes per training key, an unweighted mean over keys.
    # collision_mass_pct below is the view weighted by test observations.
    r["codes_per_key_mean"] = float(nlab.mean())

    seen = np.isfinite(amb)
    r["collision_mass_pct"] = 100.0 * np.sum(amb[seen] > 1) / len(y_te)          # of all test obs
    r["collision_of_covered_pct"] = (100.0 * np.sum(amb[seen] > 1) / seen.sum()
                                     if seen.any() else float("nan"))
    return r


# --------------------------- stages ---------------------------
def prepare(df, U, arr):
    val = df["val"].to_numpy(dtype="float64", na_value=np.nan)
    u_raw = df["val_unit"].astype(object).where(df["val_unit"].notna(), None).to_numpy()
    y = arr[df["class_idx"].to_numpy()]          # LOINC strings; factorised later
    sites = site_of(df["patient_uuid"])
    # canonicalise the ~300 distinct unit strings and then gather, which avoids 38M lookups
    ucodes, uuniq = pd.factorize(df["val_unit"])
    tbl = np.empty(len(uuniq) + 1, dtype=object)
    for i, u in enumerate(uuniq):
        c = U.canonical(u)
        tbl[i] = c if (c and U.known(c)) else None
    tbl[len(uuniq)] = None                              # factorize's -1 sentinel for NaN
    u_canon = tbl[np.where(ucodes < 0, len(uuniq), ucodes)]
    return val, u_raw, u_canon, y, sites


def pair_index(y, u_canon):
    """(loinc, canonical unit) -> int32 id, with -1 where the unit is not canonicalisable.

    It is built by crossing two small factorisations, which avoids concatenating 42M strings."""
    yc, yu = pd.factorize(pd.Series(y))
    uc, uu = pd.factorize(pd.Series(u_canon))
    ok = uc >= 0
    comb = np.where(ok, yc.astype("int64") * (len(uu) + 1) + uc, -1)
    codes, cu = pd.factorize(pd.Series(comb), sort=True)
    codes = codes.astype("int32")
    off = 1 if (len(cu) and cu[0] == -1) else 0          # -1 sorts first; it is "no pair"
    codes = np.where(ok, codes - off, -1).astype("int32")
    pairs = {i - off: (yu[int(v) // (len(uu) + 1)], uu[int(v) % (len(uu) + 1)])
             for i, v in enumerate(cu) if v >= 0}
    return codes, pairs


def groups_of(pid, valid):
    """pair id -> row indices (contiguous slices of one argsort)."""
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(pid[idx], kind="stable")]
    vals = pid[order]
    bnd = np.flatnonzero(np.diff(vals)) + 1
    out = {}
    for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(vals)]):
        out[int(vals[a])] = order[a:b]
    return out


ARMS = ["A0", "A0c", "A1", "A2", "A2x", "A4", "A3"]
TIERS = ["att", "prop"]
DETERMINISTIC = ("A0", "A0c", "A3")          # no random draw, so one run is the whole answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["alts", "run"], required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows of each split")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from fhir_data_utils import build_lab_code_mappings
    _, class_to_loinc = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    arr = np.array([class_to_loinc.get(i, f"?{i}") for i in range(1623)], dtype=object)
    U = UnitTable()
    tr_df, te_df = a15.load_cache("train"), a15.load_cache("test")
    if args.limit:
        tr_df, te_df = tr_df.head(args.limit), te_df.head(args.limit)
    print(f"train {len(tr_df):,} / test {len(te_df):,} observations")

    tv, tur, tuc, ty_s, tsite = prepare(tr_df, U, arr)
    ev, eur, euc, ey_s, esite = prepare(te_df, U, arr)
    del tr_df, te_df

    # one pair space over train+test so an arm's draw is consistent across the two
    allpid, pairs = pair_index(np.r_[ty_s, ey_s], np.r_[tuc, euc])
    tpid, epid = allpid[:len(ty_s)], allpid[len(ty_s):]
    # score_vec compares label codes numerically, as a15 does: factorise once over both splits
    ycodes, ylab = pd.factorize(pd.Series(np.r_[ty_s, ey_s]))
    ty = ycodes[:len(ty_s)].astype("int32"); ey = ycodes[len(ty_s):].astype("int32")
    pair_counts = collections.Counter()
    for p, n in zip(*np.unique(allpid[allpid >= 0], return_counts=True)):
        pair_counts[pairs[int(p)]] = int(n)

    alts_att, alts_prop, std_fam, U = build_alternatives(sorted(pair_counts), verbose=True)
    std_res = resolve_standard(std_fam, pair_counts, U)
    by_tier = {"att": alts_att, "prop": alts_prop}

    n_obs = len(ty) + len(ey)
    if args.stage == "alts":
        rows = []
        for tier, tbl in by_tier.items():
            cov = sum(n for k, n in pair_counts.items() if k in tbl)
            ncodes = len({c for c, _ in pair_counts})
            hit = {c for c, _ in tbl}
            na = [len(v) for v in tbl.values()]
            print(f"\n{tier}: {len(hit)}/{ncodes} codes, {len(tbl)}/{len(pair_counts)} (code,unit) "
                  f"pairs have >=1 alternative; {100*cov/n_obs:.1f}% of observations; "
                  f"mean {np.mean(na):.1f} median {int(np.median(na))} alternatives")
            for (c, u), n in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
                rows.append({"tier": tier, "loinc": c, "unit": u, "n_obs": n,
                             "n_alts": len(tbl.get((c, u), [])),
                             "alts": "|".join(a for a, _ in tbl.get((c, u), []))})
        std_cov = sum(n for k, n in pair_counts.items() if k in std_res)
        print(f"\nA3 standardisation would move {100*std_cov/n_obs:.1f}% of observations "
              f"({len(std_res)} pairs) onto a family-canonical unit")
        pd.DataFrame(rows).to_csv(OUT / "a16_alternatives.csv", index=False)
        print(f"\nwrote {OUT/'a16_alternatives.csv'}")
        return

    # -- run the arms --
    tgrp = groups_of(tpid, tpid >= 0)
    egrp = groups_of(epid, epid >= 0)
    prior = np.array(a15.topk_list(collections.Counter(ty.tolist()), max(a15.TOP_KS),
                                   tie="label"), dtype="int32")
    H0 = a15.label_entropy(ey.tolist())
    print(f"H(label) = {H0:.3f} bits, {len(set(ey.tolist()))} of {len(ylab)} "
          "LOINC labels present in test\n")

    out_rows = []
    hdr = (f"{'arm':5s} {'tier':5s} {'sd':>2s} {'keys':>9s} {'uniq%':>6s} {'cov%':>6s} "
           f"{'H|k':>6s} {'top1cov':>8s} {'top1all':>8s} {'collide%':>8s} {'conv%':>6s} "
           f"{'k/code':>7s} {'codes':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in args.arms.split(","):
        for tier in (["att"] if arm in DETERMINISTIC else args.tiers.split(",")):
            tbl = by_tier[tier]
            alt_tbl = {p: (u, tbl[(c, u)]) for p, (c, u) in pairs.items() if (c, u) in tbl}
            std_tbl = {p: std_res[(c, u)] for p, (c, u) in pairs.items() if (c, u) in std_res}
            active = std_tbl if arm == "A3" else alt_tbl
            act = np.fromiter(active, dtype="int32", count=len(active)) if active \
                else np.empty(0, "int32")
            tconv, econv = np.isin(tpid, act), np.isin(epid, act)
            for sd in ([None] if arm in DETERMINISTIC else list(range(args.seeds))):
                t0 = time.time()
                base = "A2" if arm in ("A2x", "A4") else arm
                rng = np.random.default_rng(1000 + (sd or 0))
                # one site table shared by both splits, because a unit convention belongs to the
                # site, whichever split a row landed in.
                st = site_table(alt_tbl, rng) if base == "A2" else None
                ar = lambda pid, gr, v, uc, ur, cv, si, a=base, r=rng, t=st: apply_arm(
                    a, pid, gr, v, uc, ur, cv, si, alt_tbl, std_tbl, rng=r, sitetbl=t)

                if arm == "A4":
                    # deployment case. The model only saw the units MIMIC recorded. The test site's
                    # convention is drawn from a table that the training data did not contain.
                    v1, u1 = apply_arm("A0c", tpid, tgrp, tv, tuc, tur, tconv, tsite,
                                       alt_tbl, std_tbl)
                    st2 = site_table(alt_tbl, np.random.default_rng(9000 + (sd or 0)))
                    v2, u2 = apply_arm("A2", epid, egrp, ev, euc, eur, econv, esite,
                                       alt_tbl, std_tbl, sitetbl=st2)
                    v1s, u1s, y1s, v2s, u2s, y2s = v1, u1, ty, v2, u2, ey
                else:
                    v1, u1 = ar(tpid, tgrp, tv, tuc, tur, tconv, tsite)
                    v2, u2 = ar(epid, egrp, ev, euc, eur, econv, esite)
                    if arm == "A2x":
                        # held-out lab: its own draw is independent of the four training sites
                        st2 = site_table(alt_tbl, np.random.default_rng(9000 + (sd or 0)))
                        vx, ux = apply_arm("A2", epid, egrp, ev, euc, eur, econv, esite,
                                           alt_tbl, std_tbl, sitetbl=st2)
                        hold = esite == N_SITES - 1
                        v2, u2 = np.where(hold, vx, v2), np.where(hold, ux, u2)
                        keep = tsite != N_SITES - 1
                        v1s, u1s, y1s = v1[keep], u1[keep], ty[keep]
                        v2s, u2s, y2s = v2[hold], u2[hold], ey[hold]
                    else:
                        v1s, u1s, y1s, v2s, u2s, y2s = v1, u1, ty, v2, u2, ey

                r = score(v1s, u1s, y1s, v2s, u2s, y2s, prior)
                r.update(arm=arm, tier=tier, seed=sd, H_label_bits=H0,
                         n_train=len(y1s), n_test=len(y2s),
                         convertible_pct=100.0 * econv.mean(), secs=round(time.time() - t0, 1))
                out_rows.append(r)
                print(f"{arm:5s} {tier:5s} {'-' if sd is None else sd:>2} {r['n_keys']:9,d} "
                      f"{r['unique_key_pct']:5.1f}% {r['test_coverage_pct']:5.1f}% "
                      f"{r['H_label_given_key_bits']:6.3f} {100*r['top1_covered']:7.2f}% "
                      f"{100*r['top1_overall']:7.2f}% {r['collision_mass_pct']:7.2f}% "
                      f"{r['convertible_pct']:5.1f}% {r['keys_per_code_mean']:7.1f} "
                      f"{r['n_codes_in_table']:5d}", flush=True)
                if arm == "A0":   # a15 parity check, under a15's exact 4 dp rule
                    r4 = score(v1s, u1s, y1s, v2s, u2s, y2s, prior, four_dp=True)
                    print(f"      a15 rule (4 dp): keys {r4['n_keys']:,d}  "
                          f"top1cov {100*r4['top1_covered']:.2f}%   "
                          "[a15's own result: 79,100 keys / 81.04%]")
                    r4.update(arm="A0_4dp", tier=tier, seed=sd, H_label_bits=H0,
                              n_train=len(y1s), n_test=len(y2s), convertible_pct=0.0, secs=0)
                    out_rows.append(r4)
    pd.DataFrame(out_rows).to_csv(OUT / "a16_arms.csv", index=False)
    print(f"\nwrote {OUT/'a16_arms.csv'}")


if __name__ == "__main__":
    main()
