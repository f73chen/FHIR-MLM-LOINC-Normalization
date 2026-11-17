"""How much the view of each model changes when the perturbation moves a reference-range bound (thesis
Chapter 7).

FHIR-MLM reads a range bound as a percentile token, so a bound that moves into another bin becomes a different
token (a32_bin_change_final.py measures how often). BCM and Llama read the same bound as text: fixed-width
scientific notation with four significant figures ("1.400e+02"), split by the tokenizer of each model. On the
sample and perturbation of a32, this script measures
  * how often the rendered string changes at all, its leading digit changes, and its exponent changes
  * per tokenizer (BioClinical ModernBERT-large, Llama-3.1-8B-Instruct), how often the token sequence of the
    number changes at all, and the share of its tokens that are kept (multiset overlap with the tokens of the
    original, over the length of the original)

The sample and weighting are those of a32_bin_change_final.py: the training split from the a15 cache, at most
4,000 observations per code (random_state 0), the perturbation and range draw of the build, and an
observation-weighted mean over the ordinary scales. Numbers are rendered in their MIMIC unit. The factors of
the unit standardisation are powers of ten (checked below), so they shift the exponent of every rendered number
by the same amount and change neither its digits nor whether it changes. Each number is tokenised with a
leading space, as it appears after a field label in the input.

Writes aggregates only, to results/analysis/a34/text_token_change.{csv,json}.

  python analysis/a34_text_token_change.py
"""
import collections, csv, importlib.util, json, math, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

ROOT = locations.REPO
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
import perturbation as pert  # noqa: E402
import unit_standardisation as us  # noqa: E402

CACHE = locations.WORK / "analysis_cache/a15_obs_fields"
OUT = ROOT / "results" / "analysis" / "a34"
SAMPLE_PER_CODE = 4000          # at most this many observations per code
ORDINARY = ("ratio", "integer_grid", "bounded_frac")
TOKENIZERS = {"BCM": "thomas-sounack/BioClinical-ModernBERT-large", "Llama": "meta-llama/Llama-3.1-8B-Instruct"}


def factors_are_powers_of_ten():
    src = (ROOT / "pipeline" / "unit_standardisation.py").read_text()
    facs = [float(x) for x in re.findall(r"^\s*'[^']*'\s*:\s*\('[^']*'\s*,\s*([0-9.eE+-]+)\)", src, re.M)]
    bad = [f for f in facs if f <= 0 or abs(math.log10(f) - round(math.log10(f))) > 1e-9]
    return len(facs), bad


def kept_share(a, b):
    """Share of the tokens of a that are also in b (multiset overlap / len(a))."""
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sum(min(n, cb[t]) for t, n in ca.items()) / len(a) if a else 1.0


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from fhir_data_utils import build_lab_code_mappings
    n_fac, bad = factors_are_powers_of_ten()
    print(f"unit standardisation: {n_fac} conversion factors, {len(bad)} not a power of ten {bad[:5]}")
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZERS.items()}
    _s = importlib.util.spec_from_file_location("a15", HERE / "a15_field_identifiability.py")
    a15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(a15)
    _m2c, c2l = build_lab_code_mappings(a15.CODES_TO_TOKENS_PATH, a15.SEEN_CODES_PATH)
    numeric_bins = a15.load_bins()
    spec = pert.load_spec()
    df = pq.read_table(CACHE / "train.parquet",
                       columns=["patient_uuid", "class_idx", "item_code", "val", "rr_low", "rr_high"]).to_pandas()
    df = df[df["val"].notna()]

    tokcache = {k: {} for k in toks}

    def tok(k, s):
        c = tokcache[k]
        if s not in c:
            c[s] = toks[k].tokenize(" " + s)
        return c[s]

    rows = []
    for ci, g in df.groupby("class_idx"):
        ln = c2l.get(ci)
        if ln is None:
            continue
        item = g["item_code"].iloc[0]
        if not numeric_bins.get(item):
            continue
        s = spec.get(ln)
        if s is None or s["tier0"] or s["range_pctV"] is None:
            continue
        gg = g.sample(min(len(g), SAMPLE_PER_CODE), random_state=0) if len(g) > SAMPLE_PER_CODE else g
        n = nbin = ntext = nlead = nexp = 0
        tk_any = collections.Counter(); tk_kept = collections.Counter()
        for u, lo, hi in zip(gg["patient_uuid"], gg["rr_low"], gg["rr_high"]):
            z = pert.site_z(u, ln, "range")                    # one draw for both bounds, as the build does
            for b in (lo, hi):
                if b is None or b != b:
                    continue
                c0 = a15.per_analyte_percentile(b, item, numeric_bins)
                if c0 is None:
                    continue
                bp = pert.perturb_number(b, z, s, s["range_pctV"])
                n += 1
                nbin += (a15.per_analyte_percentile(bp, item, numeric_bins) != c0)
                t0, tp = us.render_value(b), us.render_value(bp)
                ntext += (t0 != tp)
                nlead += (t0.lstrip("-")[0] != tp.lstrip("-")[0]) or (t0.startswith("-") != tp.startswith("-"))
                nexp += (t0.split("e")[1] != tp.split("e")[1])
                for k in toks:
                    a, bb = tok(k, t0), tok(k, tp)
                    tk_any[k] += (a != bb)
                    tk_kept[k] += kept_share(a, bb)
        if n:
            r = {"loinc": ln, "scale": s["scale"], "n_obs_total": len(g), "n_bounds": n,
                 "bin_change_pct": 100 * nbin / n, "text_change_pct": 100 * ntext / n,
                 "lead_digit_change_pct": 100 * nlead / n, "exponent_change_pct": 100 * nexp / n}
            for k in toks:
                r[f"{k}_tokens_change_pct"] = 100 * tk_any[k] / n
                r[f"{k}_tokens_kept_pct"] = 100 * tk_kept[k] / n
            rows.append(r)
    ordinary = [r for r in rows if r["scale"] in ORDINARY]
    W = sum(r["n_obs_total"] for r in ordinary)
    cols = [c for c in rows[0] if c.endswith("_pct")]
    head = {c: round(sum(r[c] * r["n_obs_total"] for r in ordinary) / W, 1) for c in cols}
    head.update({"codes": len(ordinary), "obs": W, "bounds_sampled": sum(r["n_bounds"] for r in ordinary),
                 "weighting": "observation-weighted over the ordinary scales (ratio, integer_grid, bounded_frac), as a32",
                 "unit_factors_checked": n_fac, "unit_factors_not_power_of_ten": len(bad)})
    print("OBSERVATION-WEIGHTED over the ordinary scales (range bounds):")
    for c in cols:
        print(f"   {c:28s} {head[c]:5.1f}%")
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows:
        for c in cols:
            r[c] = round(r[c], 2)
    with open(OUT / "text_token_change.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "text_token_change.json").write_text(json.dumps(head, indent=2) + "\n")
    print(f"wrote {OUT}/text_token_change.csv and .json")


if __name__ == "__main__":
    main()
