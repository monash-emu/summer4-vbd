"""Run every notebook in ``notebooks/`` headless, then the tests in ``tests/``.

    pixi run check            # everything
    pixi run check --quick    # tests only

A notebook fails if any cell raises. Outputs are not written back, so notebooks stay clean
for version control. Exit status is non-zero on any failure.

Notebooks run on a kernel started from this Python (the project's pixi environment), not
whatever kernel named ``python3`` Jupyter finds, so they always see this project's code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "project-pixi-env"


def use_project_kernel(directory: Path) -> None:
    """Register a kernel that runs this interpreter, visible only to this process."""
    spec = directory / "kernels" / KERNEL
    spec.mkdir(parents=True)
    argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    (spec / "kernel.json").write_text(
        json.dumps({"argv": argv, "display_name": KERNEL, "language": "python"})
    )
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [str(directory), *filter(None, [os.environ.get("JUPYTER_PATH")])]
    )


def run_notebook(path: Path) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # missing cell ids; pixi run clear-notebooks adds them
        nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=1800, kernel_name=KERNEL, resources={"metadata": {"path": str(path.parent)}}
    )
    start = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as err:
        print(f"FAIL {path.relative_to(ROOT)}\n{err}")
        return False
    print(f"ok   {path.relative_to(ROOT)}  ({time.perf_counter() - start:.0f}s)")
    return True


def run_tests() -> bool:
    tests = ROOT / "tests"
    if not tests.is_dir() or not any(tests.rglob("test_*.py")):
        print("no tests in tests/")
        return True
    status = subprocess.call([sys.executable, "-m", "pytest", "-q", str(tests)], cwd=ROOT)
    return status == 0


def main(argv: list[str]) -> int:
    ok = run_tests()
    if "--quick" not in argv:
        with tempfile.TemporaryDirectory() as kernels:
            use_project_kernel(Path(kernels))
            for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
                ok = run_notebook(path) and ok
    print("all checks passed" if ok else "checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
