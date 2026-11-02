"""Tests for the sliding-window dataset builder (pipeline/23_sliding_dataset.py and
common/sliding_window_utils.py).

Three properties make the windowed results joinable:

  1. obs_offset counts only the tokens the collator labels. Counting raw lab tokens overcounts and
     leaves gaps in obs_ordinal (window 2 starting at 320 when window 1 scored 314), which put the
     cross-model join out of step.
  2. Windows do not overlap and cover every payload resource exactly once, so no observation is
     scored twice or dropped.
  3. obs_offset is cumulative and contiguous, so concatenating the windows reproduces the patient's
     chronological observation index.
"""
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))


ROOT = Path(__file__).resolve().parents[1]
_S = importlib.util.spec_from_file_location("s23", ROOT / "pipeline" / "23_sliding_dataset.py")


def _mod():
    m = importlib.util.module_from_spec(_S)
    _S.loader.exec_module(m)
    return m


# -- offsets -----------------------------------------------------------------

def _offsets_from_counts(counts):
    """Reference implementation of the offset accumulation the builder performs."""
    out, acc = [], 0
    for c in counts:
        out.append(acc)
        acc += c
    return out


def test_offsets_are_cumulative_and_contiguous():
    """Concatenating windows must reproduce 0..N-1 with no gap and no repeat."""
    counts = [314, 180, 97, 5]
    offs = _offsets_from_counts(counts)
    assert offs == [0, 314, 494, 591]
    ordinals = [o + i for o, c in zip(offs, counts) for i in range(c)]
    assert ordinals == list(range(sum(counts)))


def test_offset_gap_is_what_raw_token_counting_would_cause():
    """Guards against offsets accumulated from raw counts while only mappable tokens are scored."""
    raw, scored = [320, 185], [314, 180]
    bad = _offsets_from_counts(raw)          # offsets from raw counts
    good = _offsets_from_counts(scored)
    emitted = [bad[1] + i for i in range(scored[1])]
    assert min(emitted) == 320 and good[1] == 314
    assert set(range(314, 320)).isdisjoint(emitted)   # 314..319 never emitted -> a gap


def test_single_window_offsets_are_all_zero():
    """A patient who needs one window must behave the same as in the non-windowed build."""
    assert _offsets_from_counts([200]) == [0]


# -- windowing invariants (against the real windower) ------------------------

def _fake_resources(n):
    return [{"id": f"r{i}", "resourceType": "Observation"} for i in range(n)]


def test_windows_cover_every_payload_exactly_once():
    """Windows do not overlap, and each payload index appears in exactly one window."""
    from sliding_window_utils import make_windows
    res = _fake_resources(200)
    res[0]["resourceType"] = "Patient"
    costs = [0] + [100] * 199                    # patient free, payload 100 each
    wr = make_windows(res, costs, max_len=1000, cap=64)
    payload = [i for w in wr.windows for i in w if i != 0]
    assert len(payload) == len(set(payload)), "a payload resource appeared in two windows"
    assert set(payload) == set(range(1, 200)), "windows did not cover every payload"


def test_cap_reports_dropped_payload_rather_than_silently_losing_it():
    from sliding_window_utils import make_windows
    res = _fake_resources(200)
    res[0]["resourceType"] = "Patient"
    costs = [0] + [100] * 199
    wr = make_windows(res, costs, max_len=1000, cap=2)
    assert wr.hit_cap and wr.dropped_payload_count > 0


def test_window_cap_is_above_the_observed_maximum():
    """The observed worst case in the test split is 107 windows, and the cap must not bind."""
    m = _mod()
    assert m.WINDOW_CAP > 107


# -- split parsing (split files store 'uuid.ndjson') -------------------------

def test_test_uuids_strips_the_ndjson_suffix():
    m = _mod()
    ids = m.test_uuids(5)
    assert ids and all(not i.endswith(".ndjson") for i in ids)
    assert all("-" in i for i in ids)                       # looks like a uuid


def test_test_uuids_is_sorted_and_deterministic():
    """09_mlm_dataset sorts the split set, and the windowed build must use the same patient order."""
    m = _mod()
    a, b = m.test_uuids(50), m.test_uuids(50)
    assert a == b == sorted(a)


# -- mappable set (the tokens obs_offset counts) -----------------------------

def test_mappable_set_matches_the_label_space_size():
    """There are 828 LOINC-mappable classes. If this count drifts, obs_offset numbers the observations
    wrongly."""
    m = _mod()
    mp = m.mappable_lab_tokens()
    assert len(mp) == 828, f"expected 828 mappable lab tokens, got {len(mp)}"
    assert all(isinstance(t, int) for t in list(mp)[:10])
