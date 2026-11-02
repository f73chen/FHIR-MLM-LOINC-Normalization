#!/usr/bin/env python
"""
Run every analysis behind thesis Chapter 6 on v3, in order, with one command.

  python analysis/v3_all.py                    the whole sequence
  python analysis/v3_all.py --from catalogue   resume at a step
  python analysis/v3_all.py --only tables      one or more steps (comma-separated)
  python analysis/v3_all.py --list

Each step runs in its own process at nice 19 and the idle I/O class. The run stops at the first
failure. One log per run: results/analysis/logs/v3_all_<time>.log. CPU only. The TF-IDF CPU benchmark
(`v3_cost.py --stage tfidf`) needs a quiet machine, so it is run once by hand and is not in the
sequence. `v3_cost.py --stage report` uses it when present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from lib import registry as R  # noqa: E402

LOGS = R.OUT_ROOT.parent / "logs"

STEPS = [
    ("gates", [["v3_gates.py", "--stage", "codes"], ["v3_gates.py", "--stage", "structure"]]),
    ("rq3", [["v3_rq3.py", "--stage", "all"]]),
    ("reduce", [["v3_calibration.py", "--stage", "gates"], ["v3_calibration.py", "--stage", "reduce"]]),
    ("calibration", [["v3_calibration.py", "--stage", "temperature"], ["v3_calibration.py", "--stage", "apply-t"],
                     ["v3_calibration.py", "--stage", "metrics"], ["v3_calibration.py", "--stage", "relma"]]),
    ("observation", [["v3_observation.py", "--stage", "all"]]),
    ("singleobs", [["v3_singleobs.py", "--stage", "all"], ["v3_singleobs.py", "--stage", "relma"]]),
    ("errors", [["v3_errors.py", "--stage", "run"]]),
    ("checks", [["v3_checks.py", "--stage", "all"]]),
    ("cost", [["v3_cost.py", "--stage", "collect"], ["v3_cost.py", "--stage", "report"]]),
    ("sliding", [["v3_sliding.py", "--stage", "run"]]),
    ("offshelf", [["v3_offshelf.py", "--stage", "run"]]),
    ("catalogue", [["v3_catalogue.py", "--stage", "percode"], ["v3_catalogue.py", "--stage", "tables"]]),
    ("tables", [["v3_tables.py"]]),
    ("figures", [["v3_figures.py"]]),
    ("book", [["v3_book.py"]]),
]


def run(cmd: list, log: Path) -> tuple[int, float]:
    full = ["nice", "-n", "19", "ionice", "-c3", sys.executable, "-u", str(HERE / cmd[0]), *cmd[1:]]
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n{'=' * 78}\n$ {' '.join(cmd)}   [{dt.datetime.now():%Y-%m-%d %H:%M:%S}]\n{'=' * 78}\n")
        fh.flush()
        rc = subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
    return rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, choices=[s for s, _ in STEPS])
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for name, cmds in STEPS:
            print(f"  {name:12s} " + "; ".join(" ".join(c) for c in cmds))
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"v3_all_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    names = [s for s, _ in STEPS]
    start = names.index(a.start) if a.start else 0
    plan = [(n, c) for i, (n, c) in enumerate(STEPS) if i >= start and (not only or n in only)]
    print(f"log -> {log}", flush=True)
    t_all = time.time()
    for name, cmds in plan:
        for cmd in cmds:
            rc, sec = run(cmd, log)
            print(f"  {'✓' if rc == 0 else '✗'} {name:12s} {' '.join(cmd):52s} {sec / 60:6.1f} min", flush=True)
            if rc != 0:
                print(f"\nstopped: exit {rc}; see {log}")
                sys.exit(rc)
    print(f"\nall done in {(time.time() - t_all) / 60:.1f} min; log {log}")


if __name__ == "__main__":
    main()
