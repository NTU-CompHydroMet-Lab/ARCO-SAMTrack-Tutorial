"""Post-process jupytext-generated notebooks for a clean Colab read.

After `jupytext --sync notebooks/*.py` regenerates the .ipynb files, run
this script to:

  1. Drop the leading raw YAML header cell that jupytext writes on
     conversion (it's required round-trip metadata in the .py source but
     is noise in the rendered .ipynb).
  2. Clear all cell outputs and reset execution counts. Keeps the repo
     small, avoids merge conflicts, and ensures Colab readers run their
     own cells from a clean slate.

Usage:
    uv run python tools/strip_jupytext_header.py
"""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))


def clean_one(path: Path) -> tuple[bool, bool]:
    nb = json.loads(path.read_text())
    stripped = False
    cleared = False

    if nb["cells"] and nb["cells"][0]["cell_type"] == "raw":
        first = "".join(nb["cells"][0].get("source", []))
        if first.startswith("---") and "jupytext" in first:
            nb["cells"].pop(0)
            stripped = True

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                cleared = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                cleared = True

    if stripped or cleared:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return stripped, cleared


def main() -> None:
    for p in NOTEBOOKS:
        stripped, cleared = clean_one(p)
        flags = []
        if stripped:
            flags.append("header")
        if cleared:
            flags.append("outputs")
        msg = ", ".join(flags) if flags else "clean"
        print(f"  {p}: {msg}")


if __name__ == "__main__":
    main()
