"""Tests for the helpers of analysis/a15_field_identifiability.py: significant-figure and magnitude
coarsening, reference-range keys, percentile bins (checked against the FHIR-MLM tokenizer), the
scoring core and prep()."""
import sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402
sys.path.insert(0, str(locations.REPO / "analysis"))
sys.path.insert(0, str(locations.TOKENIZER))
sys.path.insert(0, str(locations.THIRD_PARTY))
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("a15", str(locations.REPO / "analysis/a15_field_identifiability.py"))
a15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a15)

fails = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   {extra}"))
    if not cond: fails.append(name)


def test_a15_pure_logic():
    print("T1 sigfig")
    chk("3sf of 123.456 == 123", a15.sigfig(123.456, 3) == 123.0, a15.sigfig(123.456,3))
    chk("2sf of 123.456 == 120", a15.sigfig(123.456, 2) == 120.0, a15.sigfig(123.456,2))
    chk("1sf of 123.456 == 100", a15.sigfig(123.456, 1) == 100.0, a15.sigfig(123.456,1))
    chk("2sf of 0.004567 == 0.0046", abs(a15.sigfig(0.004567,2)-0.0046) < 1e-12, a15.sigfig(0.004567,2))
    chk("1sf of -85 == -90 or -80 (round-half-even)", a15.sigfig(-85,1) in (-90.0,-80.0), a15.sigfig(-85,1))
    chk("sigfig(0)==0", a15.sigfig(0.0, 2) == 0.0)
    chk("sigfig(None) is None", a15.sigfig(None, 2) is None)
    chk("sigfig(nan) is None", a15.sigfig(float('nan'), 2) is None)
    chk("sigfig is idempotent-ish: 2sf(2sf(x))==2sf(x)", a15.sigfig(a15.sigfig(123.456,2),2)==a15.sigfig(123.456,2))

    print("T2 magnitude")
    chk("mag(123)==2", a15.magnitude(123)==2)
    chk("mag(0.05)==-2", a15.magnitude(0.05)==-2, a15.magnitude(0.05))
    chk("mag(1.0)==0", a15.magnitude(1.0)==0)
    chk("mag(0)==-99 sentinel", a15.magnitude(0.0)==-99)
    chk("mag(-250)==2 (uses abs)", a15.magnitude(-250)==2)
    chk("mag(None) is None", a15.magnitude(None) is None)

    print("T3 refrange keys")
    chk("exact key shape", a15.rr_key_exact(3.5, 5.1, "meq/l", "meq/l") == (3.5, "meq/l", 5.1, "meq/l"))
    chk("all-None -> None", a15.rr_key_exact(None, None, None, None) is None)
    chk("4dp rounding", a15.rr_key_exact(3.50000001, 5.1, "u", "u")[0] == 3.5)
    chk("units-only key present", a15.rr_key_exact(None, None, "mg/dl", None) == (None,"mg/dl",None,None))
    chk("sf key coarsens both bounds", a15.rr_key_sf(123.4, 567.8, "u","u",2) == (120.0,"u",570.0,"u"),
        a15.rr_key_sf(123.4,567.8,"u","u",2))
    chk("sf key None when no bounds", a15.rr_key_sf(None, None, "u","u",2) is None)
    chk("mag key", a15.rr_key_mag(123.4, 5678.0, "u","u") == (2,"u",3,"u"))
    chk("strip_units", a15.strip_units((3.5,"a",5.1,"b")) == (3.5,5.1))
    chk("strip_units(None)", a15.strip_units(None) is None)
    chk("units_only picks first non-null", a15.units_only(None,"hi","v") == ("hi",))
    chk("units_only all None", a15.units_only(None,None,None) is None)

    print("T4 per-analyte percentile == fhir_tokenizer._get_numeric_token")
    nb = a15.load_bins()
    # Call the real tokenizer method, unbound, with a stub self. The test then compares against
    # fhir_tokenizer.py itself.
    from fhir_tokenizer import NdjsonTokenizer
    class _Stub:
        numeric_encoding = "percentile_bins"
        def __init__(self, bins, mapping): self.numeric_bins = bins; self.code_mapping = mapping
    _pct_ids = {f"p_{i}": 900000+i for i in range(101)}
    _stub = _Stub(nb, {"value_percentile": _pct_ids})
    _real = NdjsonTokenizer._get_numeric_token
    _inv = {v: k for k, v in _pct_ids.items()}
    def real_impl(value, item_code):
        out = _real(_stub, value, item_code)
        return _inv[out[0]] if out else None
    # Replicate the tokenizer arithmetic independently, without calling the copy in a15.
    def ref_impl(value, item_code, bins_map):
        bins = bins_map.get(item_code, [])
        if not bins: return None
        n_bins = len(bins); scale = 100/n_bins
        bucket = np.digitize(value, bins, right=True)
        bucket = bucket*scale - (scale/2)
        percentile = round(np.clip(bucket, 0, 100) + 1e-6)
        return f"p_{percentile}"
    import random
    random.seed(0)
    items = [k for k in nb if nb[k]][:200]
    mismatch = 0; seen = set()
    for it in items:
        b = nb[it]
        probes = list(b) + [min(b)-1, max(b)+1, 0.0, -5.0, 1e6] + [random.uniform(min(b)-1, max(b)+1) for _ in range(8)]
        for v in probes:
            a = a15.per_analyte_percentile(v, it, nb); r = real_impl(v, it)
            r2 = ref_impl(v, it, nb)
            if r != r2: mismatch += 1
            if a != r: mismatch += 1
            if a: seen.add(a)
    chk(f"200 items x ~24 probes match fhir_tokenizer._get_numeric_token (mismatches={mismatch})", mismatch == 0)
    chk("no-bins item -> None", a15.per_analyte_percentile(5.0, "__nope__", nb) is None)
    chk("None value -> None", a15.per_analyte_percentile(None, items[0], nb) is None)
    b4 = [k for k in nb if len(nb[k])==4][0]; b19 = [k for k in nb if len(nb[k])==19][0]
    s4 = {a15.per_analyte_percentile(v,b4,nb) for v in np.linspace(-10,10,500)}
    s19 = {a15.per_analyte_percentile(v,b19,nb) for v in np.linspace(-10,10,500)}
    chk(f"5-bin family emits <=5 distinct tokens (got {len(s4)}: {sorted(s4)})", len(s4) <= 5)
    chk(f"20-bin family emits <=20 distinct tokens (got {len(s19)})", len(s19) <= 20)
    chk(f"families near-disjoint (overlap={sorted(s4 & s19)})", len(s4 & s19) <= 2)

    print("T5 global percentile")
    vals = list(np.random.RandomState(0).lognormal(0,2,200000))
    e = a15.build_global_percentile_edges(vals, 100)
    chk("edges length 99", len(e)==99)
    chk("edges monotone", bool(np.all(np.diff(e) >= 0)))
    gs = [a15.global_percentile(v,e) for v in vals[:20000]]
    chk("all values get a token (no absence signal)", all(g is not None for g in gs))
    chk("<=100 distinct buckets", len({g for g in gs}) <= 101, len({g for g in gs}))
    chk("None -> None", a15.global_percentile(None,e) is None)

    print("T6 scoring core")
    # a deterministic toy: key 'a'->class 1 always, 'b'->{2:3,3:1}, None -> backoff
    trk = ["a"]*10 + ["b"]*3 + ["b"] + [None]*5
    trc = [1]*10 + [2]*3 + [3] + [7]*5
    tek = ["a","b","zz",None]
    tec = [1,2,1,1]
    prior = [7,1,2,3]
    r = a15.score_single(trk,trc,tek,tec,prior)
    chk("n_keys==2", r["n_keys"]==2, r["n_keys"])
    chk("unique_key_pct==50", r["unique_key_pct"]==50.0, r["unique_key_pct"])
    chk("coverage 50% (a,b seen; zz,None not)", r["test_coverage_pct"]==50.0, r["test_coverage_pct"])
    chk("top1_covered==1.0", r["top1_covered"]==1.0, r["top1_covered"])
    chk("top1_overall==0.5 (2 covered hits; zz and None back off to prior[0]=7 and miss)",
        abs(r["top1_overall"]-0.5)<1e-9, r["top1_overall"])
    # H(label|key): 'a' pure (0 bits, mass 10), 'b' 3/1 split (0.811 bits, mass 4)
    exp_h = (10/14)*0 + (4/14)*(-(0.75*math.log2(0.75)+0.25*math.log2(0.25)))
    chk("H(label|key) correct", abs(r["H_label_given_key_bits"]-exp_h)<1e-9,
        f"{r['H_label_given_key_bits']} vs {exp_h}")
    # a pure table has zero entropy and 100% unique keys
    r2 = a15.score_single(["a","b","c"],[1,2,3],["a","b","c"],[1,2,3],prior)
    chk("pure table: H=0, uniq=100, top1=1", r2["H_label_given_key_bits"]==0
        and r2["unique_key_pct"]==100.0 and r2["top1_covered"]==1.0)

    print("T7 label_entropy")
    chk("uniform 4 classes -> 2 bits", abs(a15.label_entropy([0,1,2,3])-2.0)<1e-12)
    chk("single class -> 0 bits", a15.label_entropy([5,5,5])==0.0)

    print("T8 _combo")
    d = {"x":[1,None,None],"y":[None,2,None]}
    c = a15._combo(d,["x","y"])
    chk("_combo is strict: partial -> None", c[0] is None and c[1] is None)
    chk("_combo all-None -> None", c[2] is None)
    d2 = {"x":[1,2],"y":[3,4]}
    chk("_combo both present -> tuple", a15._combo(d2,["x","y"]) == [(1,3),(2,4)])

    print("T9 _unit / _f")
    chk("_unit lowercases+strips", a15._unit({"unit":" MeQ/L "})=="meq/l")
    chk("_unit falls back to code", a15._unit({"code":"mg/dL"})=="mg/dl")
    chk("_unit empty -> None", a15._unit({"unit":"  "}) is None)
    chk("_unit None -> None", a15._unit(None) is None)
    chk("_f int->float", a15._f(5)==5.0)
    chk("_f str numeric", a15._f("3.2")==3.2)
    chk("_f garbage -> None", a15._f("abc") is None)
    chk("_f nan -> None", a15._f(float('nan')) is None)

    print("T10 prep(): missing fields read as None, not NaN")
    import pandas as pd
    _df = pd.DataFrame({
        "patient_uuid": ["u1","u2","u3","u4"],
        "obs_ordinal": np.int32([0,1,2,3]),
        "class_idx":   np.int16([1,1,2,2]),
        "item_code":   pd.Categorical(["50801","50801","99999","99999"]),
        "rr_low":      [3.5, np.nan, 3.5, np.nan],     # rows 1 and 3 have no reference range
        "rr_low_unit": pd.Categorical(["meq/l", None, "meq/l", None]),
        "rr_high":     [5.1, np.nan, 5.1, np.nan],
        "rr_high_unit":pd.Categorical(["meq/l", None, "meq/l", None]),
        "val":         [4.2, 4.2, np.nan, 4.2],
        "val_unit":    pd.Categorical(["meq/l","meq/l",None,"meq/l"]),
        "val_kind":    pd.Categorical(["quantity","quantity","none","quantity"]),
        "val_string":  pd.Categorical([None,None,None,None]),
        "priority":    pd.Categorical(["stat",None,"stat",None]),
    })
    import numpy as _np
    _arr = _np.array([f"L{i}" for i in range(1623)], dtype=object)
    _p = a15.prep(_df, nb, _arr)
    chk("missing refrange -> k_rr is None", _p["k_rr"] == [(3.5,"meq/l",5.1,"meq/l"), None,
                                                          (3.5,"meq/l",5.1,"meq/l"), None], _p["k_rr"])
    chk("present refranges are equal (not NaN-fragmented)", _p["k_rr"][0] == _p["k_rr"][2])
    chk("only 1 distinct non-None rr key", len({k for k in _p["k_rr"] if k is not None}) == 1)
    for n in (3,2,1):
        chk(f"missing refrange -> k_rr{n}sf is None", _p[f"k_rr{n}sf"][1] is None and _p[f"k_rr{n}sf"][3] is None,
            _p[f"k_rr{n}sf"])
    chk("missing refrange -> k_rrmag is None", _p["k_rrmag"][1] is None and _p["k_rrmag"][3] is None)
    chk("missing refrange -> k_rr_nounit is None", _p["k_rr_nounit"][1] is None)
    chk("missing value -> k_vraw is None", _p["k_vraw"][2] is None, _p["k_vraw"])
    chk("present values equal", _p["k_vraw"][0] == _p["k_vraw"][1] == (4.2,))
    chk("k_rr_unit strict: None unless both parts present", _p["k_rr_unit"] == [
        ((3.5,"meq/l",5.1,"meq/l"),"meq/l"), None, None, None], _p["k_rr_unit"])
    chk("k_vpct None for item with no bin table", _p["k_vpct"][3] is None, _p["k_vpct"])
    chk("k_vpct set for item with bin table", _p["k_vpct"][0] is not None)
    chk("k_hasbins never None", all(k is not None for k in _p["k_hasbins"]))
    chk("k_prio None where absent", _p["k_prio"] == [("stat",), None, ("stat",), None], _p["k_prio"])
    _hr = [k is not None for k in _p["k_rr"]]
    chk("reference-range key present on 50% of rows, not 100%", sum(_hr)/len(_hr) == 0.5)

    print()
    print(("ALL %d CHECKS PASSED" % 0) if not fails else f"{len(fails)} FAILURES: {fails}")
    if fails:
        raise AssertionError(f"{len(fails)} failures: {fails}")



if __name__ == "__main__":
    test_a15_pure_logic()
