"""Run every ```python block in docs/summer4-guide.md, in order, in one namespace.

The guide is the reference agents and people read for the pinned summer4, so a snippet
that no longer runs (after changing the pin, say) must fail the checks.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "summer4-guide.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def test_guide_snippets_run() -> None:
    blocks = BLOCK.findall(GUIDE.read_text(encoding="utf-8"))
    assert blocks, "no ```python blocks found in the guide"
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        # The guide must not teach a spelling summer4 has deprecated.
        warnings.filterwarnings("error", message=".* is deprecated", category=DeprecationWarning)
        for number, source in enumerate(blocks, start=1):
            code = compile(source, f"{GUIDE.name} block {number}", "exec")
            exec(code, namespace)  # noqa: S102 - running our own documentation
