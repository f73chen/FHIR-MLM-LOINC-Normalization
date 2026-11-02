"""Power and minimum detectable effect (MDE) for paired McNemar comparisons (thesis Chapter 5).

Every model comparison in the thesis is on matched observations, so the test is McNemar's, on the
discordant pairs. The formula for two independent proportions would badly overstate the sample size
needed. It ignores that the arms share the same observations, and the paired design gets its power
from this sharing.

The MDE depends on the discordance rate as well as on n. Two models that rarely disagree need a bigger
accuracy gap before that gap is detectable. The discordance is therefore taken from the observed
pairwise counts and is not assumed.

A patient's observations are not independent (the same history, the same local names, the same
perturbation draw). This is why the observation-level intervals resample patients. The MDE the thesis
reports is therefore `mde_mcnemar_patients`. It is the same calculation at Kish's effective sample size,
which is n divided by the design effect that `design_effect` measures on the compared arms.
`mde_mcnemar` alone is the MDE if every observation were independent.

Notation
    n      paired observations
    psi    discordance rate = (a_only + b_only) / n
    m      discordant pairs = n * psi
    pi     P(a correct & b wrong | discordant); null is pi = 0.5
    delta  accuracy difference = psi * (2*pi - 1)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def min_discordant_for_significance(alpha: float = 0.05) -> int:
    """Fewest discordant pairs at which a two-sided exact binomial test can reach significance.

    With m discordant pairs the most extreme outcome (all favouring one model) gives p = 2 * 0.5**m.
    If that already exceeds alpha, no effect size is detectable at any power. The test cannot reject
    the null, which is worse than being underpowered. At alpha=0.05 this is m = 6. Without this guard the
    bisection would return a finite MDE for a stratum with one discordant pair, where the right answer
    is "not estimable".
    """
    m = 1
    while 2.0 * (0.5 ** m) > alpha:
        m += 1
    return m


def power_mcnemar(n: int, psi: float, delta: float, alpha: float = 0.05) -> float:
    """Power of a two-sided McNemar test to detect an accuracy difference `delta`.

    Normal approximation to the binomial on the discordant pairs, without the usual small-delta
    simplification sqrt(pi(1-pi)) ~ 1/2, so the result stays right for large deltas too.
    """
    if psi <= 0 or n <= 0:
        return 0.0
    m = n * psi
    if m < min_discordant_for_significance(alpha):
        return 0.0                    # the exact test cannot reject at this alpha, whatever delta is
    pi = 0.5 + delta / (2.0 * psi)
    if not (0.0 < pi < 1.0):          # delta exceeds what this discordance rate can produce
        return 1.0 if abs(delta) >= psi else 0.0
    z = norm.ppf(1 - alpha / 2)
    num = abs(m * (pi - 0.5)) - z * np.sqrt(m) / 2.0
    den = np.sqrt(m * pi * (1 - pi))
    if den <= 0:
        return 0.0
    return float(norm.cdf(num / den))


def mde_mcnemar(n: int, psi: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest accuracy difference detectable at `power`. It is solved by bisection on power_mcnemar
    so that the exact variance term above is respected."""
    if psi <= 0 or n <= 0:
        return float("nan")
    lo, hi = 0.0, psi                  # delta cannot exceed the discordance rate
    if power_mcnemar(n, psi, hi, alpha) < power:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_mcnemar(n, psi, mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def design_effect(patient, a_correct, b_correct) -> float:
    """Kish's design effect of patients on a paired accuracy difference. It is the variance of the summed
    differences with patients as the independent units, divided by their variance if every observation
    were independent. A value of 1 means patients add nothing. Above 1, a patient's observations move
    together and the test holds less information than its n suggests.

    Each observation contributes d = a_correct - b_correct in {-1, 0, 1}. Patient k's sum S_k is centred
    on n_k times the overall mean difference, as the patient bootstrap (v3_rq3.paired_cluster_ci)
    centres it. So with one observation per patient the ratio is 1.
    """
    d = np.asarray(a_correct, dtype=np.int64) - np.asarray(b_correct, dtype=np.int64)
    if not len(d):
        return float("nan")
    mean = d.mean()
    v_independent = len(d) * (np.mean(d != 0) - mean ** 2)
    if v_independent <= 0:
        return float("nan")
    _, k = np.unique(np.asarray(patient), return_inverse=True)
    s, m = np.bincount(k, weights=d), np.bincount(k)
    return float(((s - m * mean) ** 2).sum() / v_independent)


def mde_mcnemar_patients(n: int, psi: float, deff: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """MDE of McNemar's test with patients as clusters, computed as `mde_mcnemar` at Kish's effective
    sample size n / deff. Suppose patients scale the variance of the summed differences by deff under
    both the null and the alternative. Then the power of the clustered test at n equals the power of the
    independent test at n / deff. tests/test_power.py checks this by simulating the clustered test. deff
    comes from `design_effect` on the compared arms."""
    if not deff > 0:
        return float("nan")
    return mde_mcnemar(n / deff, psi, power, alpha)


def simulate_power(n: int, psi: float, delta: float, alpha: float = 0.05,
                   n_sim: int = 4000, seed: int = 0) -> float:
    """Monte-Carlo power, used only to verify the analytic formula. Draws discordant pairs and runs
    the same exact binomial test the analysis uses (dumps.mcnemar, which calls scipy binomtest)."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(seed)
    m = int(round(n * psi))
    pi = 0.5 + delta / (2.0 * psi)
    if not (0.0 < pi < 1.0):
        return float("nan")
    x = rng.binomial(m, pi, size=n_sim)
    hits = 0
    for xi in x:
        if binomtest(int(xi), m, 0.5).pvalue < alpha:
            hits += 1
    return hits / n_sim
