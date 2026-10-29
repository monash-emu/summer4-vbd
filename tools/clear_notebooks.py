"""Clear outputs and execution counts from every notebook in ``notebooks/``.

Also gives every cell an ``id`` (required by nbformat 4.5) if it lacks one.

    pixi run clear-notebooks
"""

from __future__ import annotations

import json
import uuid
import warnings
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # missing ids are fixed below
            nb = nbformat.read(path, as_version=4)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        changed = any("id" not in cell for cell in on_disk["cells"])
        for cell in nb.cells:
            if "id" not in cell:
                cell["id"] = uuid.uuid4().hex[:8]
                changed = True
            if cell.cell_type != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                cell["outputs"] = []
                cell["execution_count"] = None
                changed = True
        if changed:
            nbformat.write(nb, path)
            print(f"cleared {path.relative_to(ROOT)}")
    print("notebooks are clear")


if __name__ == "__main__":
    main()
