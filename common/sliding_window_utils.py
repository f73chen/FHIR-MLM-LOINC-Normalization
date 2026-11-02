"""Pack one patient's resources into non-overlapping windows for the sliding-window evaluation
(pipeline/23_sliding_dataset.py; thesis Chapters 5 and 6).

The windower works at the resource-list level so that the FHIR-MLM, BCM and
Llama pipelines can share it. Each model supplies its own per-resource
token-cost vector, computed in that model's tokenizer space.

The windower produces a list of resource-index lists. Each window has the
form:
  [patient_idx, optional_most_recent_encounter_idx, ...payload_indices]

The Patient resource comes first in every window. Each bundle has exactly
one, at chronological position 0. The most recent Encounter at or before the
first payload resource of the window is also prepended, if one exists in the
chronologically sorted resource list and is not already in the preamble.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WindowingResult:
    """Result of sliding-window packing for a single patient bundle.

    Attributes:
        windows: List of windows, each a list of resource indices into the
            input `resources` list. Indices appear in the order they should
            be re-assembled (preamble first, then payload in chronological
            order).
        dropped_payload_count: Number of payload resources that didn't fit
            into the first `cap` windows. Zero if the bundle wasn't capped.
        hit_cap: True iff we stopped emitting because `cap` was reached
            (i.e., dropped_payload_count > 0).
    """

    windows: list[list[int]]
    dropped_payload_count: int
    hit_cap: bool


def make_windows(
    resources: list[dict],
    per_resource_cost: list[int],
    max_len: int,
    cap: int = 32,
) -> WindowingResult:
    """Slide a fixed-size window over a chronologically-sorted resource list.

    Args:
        resources: List of FHIR resource dicts. Must already be
            chronologically sorted (typically via fhir_formatter.sort_key).
            Must contain exactly one Patient resource.
        per_resource_cost: Token cost of each resource in the calling
            model's tokenizer space. `len(per_resource_cost) == len(resources)`.
            FHIR-MLM uses NdjsonTokenizer costs; BCM and Llama use BPE costs.
        max_len: Per-window token budget. Sum of per_resource_cost over a
            single window's resources will be <= max_len.
        cap: Max windows per patient. Payload past this is dropped and
            counted. Default 32.

    Returns:
        WindowingResult.

    Raises:
        ValueError: if input invariants are violated (Patient count, length
            mismatch, oversized preamble).
    """
    if len(per_resource_cost) != len(resources):
        raise ValueError(
            f"per_resource_cost length ({len(per_resource_cost)}) must match "
            f"resources length ({len(resources)})"
        )

    patient_idxs = [
        i for i, r in enumerate(resources) if r.get("resourceType") == "Patient"
    ]
    if len(patient_idxs) != 1:
        raise ValueError(
            f"Expected exactly 1 Patient resource, got {len(patient_idxs)}"
        )
    patient_idx = patient_idxs[0]

    encounter_idxs = [
        i for i, r in enumerate(resources) if r.get("resourceType") == "Encounter"
    ]

    # Payload candidates are every resource except Patient, in chronological
    # order. Encounters are in this list and may also be prepended at window
    # boundaries. preamble_set keeps them from being counted twice.
    payload_candidates = [i for i in range(len(resources)) if i != patient_idx]

    windows: list[list[int]] = []
    pos = 0

    while pos < len(payload_candidates) and len(windows) < cap:
        first_payload_idx = payload_candidates[pos]

        # Most recent Encounter at or before the first payload of the window,
        # by position in the resource list. The resources are already sorted
        # by time, so walking encounter_idxs in order finds the right one. It
        # is "at or before" because, when the first payload is an Encounter,
        # that Encounter is the relevant one.
        most_recent_enc: int | None = None
        for ei in encounter_idxs:
            if ei <= first_payload_idx:
                most_recent_enc = ei
            else:
                break

        preamble = [patient_idx]
        preamble_set = {patient_idx}
        if most_recent_enc is not None:
            preamble.append(most_recent_enc)
            preamble_set.add(most_recent_enc)

        preamble_cost = sum(per_resource_cost[i] for i in preamble)
        if preamble_cost > max_len:
            raise ValueError(
                f"Preamble cost {preamble_cost} exceeds max_len {max_len} "
                f"(preamble resources: {preamble})"
            )

        cur = list(preamble)
        cur_cost = preamble_cost
        progress_this_window = False

        while pos < len(payload_candidates):
            cand = payload_candidates[pos]
            if cand in preamble_set:
                # Don't re-add the encounter we already prepended.
                pos += 1
                continue
            cand_cost = per_resource_cost[cand]
            if cand_cost > max_len - preamble_cost:
                # A single resource too big to fit beside this preamble is
                # skipped. This is only a safeguard. The largest single block
                # measured was 1,712 tokens, well under a window.
                pos += 1
                continue
            if cur_cost + cand_cost > max_len:
                break
            cur.append(cand)
            cur_cost += cand_cost
            pos += 1
            progress_this_window = True

        if progress_this_window:
            windows.append(cur)
        else:
            # No payload fit this window. Stop so the loop cannot run forever.
            break

    dropped = len(payload_candidates) - pos
    hit_cap = len(windows) >= cap and pos < len(payload_candidates)
    return WindowingResult(
        windows=windows, dropped_payload_count=dropped, hit_cap=hit_cap
    )


def summarize_windows(result: WindowingResult) -> dict:
    """Compact summary of a WindowingResult for logging/manifests."""
    return {
        "n_windows": len(result.windows),
        "window_sizes": [len(w) for w in result.windows],
        "dropped_payload_count": result.dropped_payload_count,
        "hit_cap": result.hit_cap,
    }
