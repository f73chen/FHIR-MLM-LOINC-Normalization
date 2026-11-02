"""Flag the garbage among the paraphrases that two or more LOINC codes share (thesis Appendix C).

Reads the verified paraphrase pool from prep/4.2 (UPSTREAM_DIR/paraphrases_ext_verified/, one .txt
file per LOINC code). For every paraphrase that at least two files share (exact match after
lowercasing), it writes a row to UPSTREAM_DIR/shared_paraphrase_labels.csv with the columns
paraphrase, n_files, is_garbage, reason and confidence. Counts per n_files go to
shared_paraphrase_by_bucket.csv. The rules flag known patterns: prompt leakage, LOINC axis codes,
vocabulary names, and exact strings found by manual review. pipeline/04 and pipeline/05 read the
labels.

  python prep/shared_paraphrase_garbage_check.py
"""

import csv
import re
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

PARAPHRASE_DIR = locations.UPSTREAM / "paraphrases_ext_verified"
OUT_CSV = locations.UPSTREAM / "shared_paraphrase_labels.csv"
OUT_BUCKETS_CSV = locations.UPSTREAM / "shared_paraphrase_by_bucket.csv"

# Every string with n_files >= FULL_REVIEW_THRESHOLD was reviewed by hand, so
# an unflagged one there is confirmed valid. Below the threshold, unflagged
# rows are valid by default (rules plus spot checks).
FULL_REVIEW_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Garbage rules, applied by is_regex_garbage() to the lowercased string. They
# also cover patterns that the cleaning in prep/4.2 targets but lets through.
# ---------------------------------------------------------------------------

EXACT_AXIS_CODES = {
    # LOINC scale / property / axis short codes, which do not describe a
    # test on their own.
    "qn", "ql", "ord", "nom", "nar", "doc", "semiqn", "quan", "qual",
    "mcnc", "scnc", "acnc", "ncnc", "ccnc", "rcnc", "ncnt", "ccnt", "scnt",
    "srto", "mrto", "nrto", "cfr", "nfr", "mfr", "sfr", "vfr", "efr", "rfr",
    "len", "time", "rto", "angle", "vol", "temp", "pres", "freq",
    # "pt" is left out because it is a real paraphrase ("prothrombin time").
}

EXACT_VOCAB_IDS = {
    "mimic", "mimic-iv", "mimic iv", "loinc", "snomed", "snomed-ct",
    "cdisc", "ciel", "umls", "omop", "athena", "ohdsi", "fhir",
    "lis", "lis naming", "lis naming conventions",
}

PROMPT_LEAK_SUBSTR = [
    "generates test names",
    "official loinc",
    "test name, description",
    "test name,description",
    "based on legacy lis",
    "legacy lis naming",
    "here are",
    "as a lab technician",
    "the following are",
    "following paraphrases",
    "list of paraphrases",
    "5 ultra-brief",
    "5 paraphrases",
    "paraphrases:",
    "loinc code",
    "loinc name",
    "long common name",
    "short name",
    "display name",
    "alternate name",
    "alternative name",
    "for this test",
    "for this code",
    "for this loinc",
    "this loinc code",
    "synonyms for",
    "abbreviations for",
    "standard name",
    "i'll provide",
    "i'll list",
    "i cannot",
    "i can't",
    "sorry",
    "note:",
    "instruction",
    "brief test names",
    "test names using",
    "medical abbreviations",
    "lab technician",
    "recognizable to",
    "characters, unique",
    "30 characters",
    "under 30",
    "provide ",
    "list 5",
    # Additional substrings from manual n>=5 inspection
    "a list of",
    "list of test",
    "list of generated",
    "commonly written",
    "in clinical orders",
    "desired format",
    "lis format",
    "legacy lis",
    "lis convention",
    "snomed naming",
    "mimic naming",
    "mimic name",
    "mimichospital",
    "mimic ",
    "mimics ",
    "mimeic ",
    "provided criteria",
    "provided loinc",
    "test name examples",
    "use the provided",
    "show less",
    "show more",
    "show all",
    "show comment",
    "final answer",
    "end of output",
    "test names are",
    "test name,",
    "test name, ",
    "test description per",
    "legacy test names",
    "example test",
    "point in time",
    "scale type",
    "time aspect",
    "name, specimen",
    "name, description",
    "name, calculation",
    "name, qualifier",
    "name, nfr",
    "name, csf",
    "name,specimen",
    "specimen, calculation",
    "property type",
    "property nfr",
    "property time",
    "property vrat",
    "calculation, fraction",
    "fraction, method",
    "postfix qualifier",
    "standard medical abbreviation",
    "more test names",
    "number of ",
    "&#x",
]

# Additional exact matches confirmed garbage via manual n>=5 inspection.
EXTRA_EXACT_GARBAGE = {
    "e.g", "eg", "go", "choice", "good", "perfect", "final answer",
    "end", "here", "list", "item", "examples", "grammar", "markdown",
    "quote", "complete", "show", "or", "for", "add", "end of output",
    "other", "number of", "anti", "anti-", "add-", "add- on",
    "immun", "mim", "snom", "en", "lm",
    "true", "false", "yes", "no",
    # LOINC axis / property labels used as standalone paraphrase.
    "property", "component", "system", "method", "scale type",
    "time aspect", "time aspect pt", "measurement", "observation",
    "concentration", "presence", "quantitative", "nominal", "normal",
    "point", "point in time", "unit", "rate", "body", "light",
    "total", "free", "volume", "pressure", "color", "respiratory",
    "test name description",
    # Fragments / non-specific CD references lacking the marker number.
    "cd", "cells cd", "cell cd", "cells.cd", "cells/cells", "/cells",
    "cd cells", "cd cell",
    # Malformed / stray text
    "white blood", "u urine", "complete blood", "pt/in",
}


def is_regex_garbage(s: str):
    """Return (is_garbage, reason) or (False, '')."""
    if s in EXACT_AXIS_CODES:
        return True, "axis-code"
    if s in EXACT_VOCAB_IDS:
        return True, "vocab-id"
    if s in EXTRA_EXACT_GARBAGE:
        return True, "manual-exact"
    # Exact numeric / punctuation-only
    if re.fullmatch(r"[\W_\d]+", s):
        return True, "punct-or-digit-only"
    # Single letter
    if len(s) == 1:
        return True, "single-char"
    for pat in PROMPT_LEAK_SUBSTR:
        if pat in s:
            return True, f"prompt-leak:{pat}"
    # "1.", "2.", "option 1", "option 2" list markers
    if re.fullmatch(r"(option\s*)?\d+[.)]?", s):
        return True, "list-marker"
    return False, ""


def main():
    str_to_files = defaultdict(set)
    for p in sorted(PARAPHRASE_DIR.glob("*.txt")):
        for l in p.read_text().splitlines():
            l = l.strip().lower()
            if l:
                str_to_files[l].add(p.stem)

    shared = [(s, codes) for s, codes in str_to_files.items() if len(codes) > 1]
    shared.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paraphrase", "n_files", "is_garbage", "reason", "confidence"])
        for s, codes in shared:
            g, reason = is_regex_garbage(s)
            # 'reviewed': in the hand-reviewed band (n >= 5) or flagged by
            # the rules. 'tail': below the band and unflagged, so valid by
            # default.
            if g:
                conf = "reviewed"
            elif len(codes) >= FULL_REVIEW_THRESHOLD:
                conf = "reviewed"
            else:
                conf = "tail"
            w.writerow([s, len(codes), "1" if g else "0", reason, conf])

    # ---- Per-bucket stats (for cutoff plots) ----
    from collections import Counter
    bucket_total = Counter()
    bucket_garbage = Counter()
    for s, codes in shared:
        n = len(codes)
        bucket_total[n] += 1
        if is_regex_garbage(s)[0]:
            bucket_garbage[n] += 1
    with OUT_BUCKETS_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_files", "n_distinct", "n_garbage", "garbage_pct"])
        for n in sorted(bucket_total):
            w.writerow([n, bucket_total[n], bucket_garbage[n],
                        f"{bucket_garbage[n]/bucket_total[n]*100:.2f}"])

    # ---- Summary ----
    total = len(shared)
    auto_g = sum(1 for s, _ in shared if is_regex_garbage(s)[0])
    pairs_total = sum(len(c) for _, c in shared)
    pairs_auto_g = sum(len(c) for s, c in shared if is_regex_garbage(s)[0])
    print(f"distinct shared strings: {total}")
    print(f"  garbage (reviewed+regex): {auto_g} ({auto_g/total:.1%})")
    print(f"(paraphrase, loinc) pairs: {pairs_total}")
    print(f"  garbage: {pairs_auto_g} ({pairs_auto_g/pairs_total:.1%})")
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_BUCKETS_CSV}")


if __name__ == "__main__":
    main()
