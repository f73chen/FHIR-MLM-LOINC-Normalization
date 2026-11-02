"""Copy the shipped resources to where the pipeline reads them (thesis Chapter 4).

The unit synonym map, the unit aliases and the two qualitative-value lists go to UPSTREAM_DIR. The
public-source perturbation registers and the module that transcribes the published tables go to
data/variability/. A file that already exists there is left alone, for example one rebuilt with
prep/4.3 or pipeline/parse_*.py.

  python prep/0_install_resources.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import locations  # noqa: E402

COPIES = [(p, locations.UPSTREAM / p.name) for p in sorted(locations.RESOURCES.glob("*.json"))]
COPIES += [(p, locations.REPO / "data" / "variability" / p.name)
           for p in sorted((locations.RESOURCES / "variability").iterdir()) if p.suffix in (".csv", ".py")]

for src, dst in COPIES:
    if dst.exists():
        print(f"kept      {dst}")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"installed {dst}")
