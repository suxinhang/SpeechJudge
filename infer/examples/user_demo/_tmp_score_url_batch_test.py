"""Compatibility shim.

The batch test script was moved to ``test/tmp_score_url_batch_test.py``.
Run that file instead.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_NEW = Path(__file__).resolve().parents[3] / "test" / "tmp_score_url_batch_test.py"


def main() -> None:
    if not _NEW.is_file():
        raise SystemExit(f"missing moved script: {_NEW}")
    sys.argv[0] = str(_NEW)
    runpy.run_path(str(_NEW), run_name="__main__")


if __name__ == "__main__":
    main()
