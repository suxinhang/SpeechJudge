"""Step 1 only: read testd1.xlsx, download URL audios, verify locals, write manifest.

Does **not** call the rank API. After this succeeds, run ``submit_rank_from_manifest.py``.

Requires: ``pip install openpyxl``

Example::

    python test/download_inputs_from_testd1_xlsx.py --dry-run
    python test/download_inputs_from_testd1_xlsx.py
    python test/submit_rank_from_manifest.py --base-url https://....trycloudflare.com
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from testd1_inputs_lib import manifest_dict, materialize_from_xlsx, verify_files_non_empty, write_manifest


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> int:
    _configure_stdout_utf8()
    p = argparse.ArgumentParser(description="Download + verify inputs from testd1.xlsx; write JSON manifest only.")
    p.add_argument("--xlsx", type=Path, default=Path(__file__).resolve().parent / "testd1.xlsx")
    p.add_argument("--local-root", type=Path, default=Path(r"D:\Downloads\泰语"))
    p.add_argument("--download-dir", type=Path, default=Path(r"D:\Downloads\泰语\_from_excel_urls"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "testd1_rank_inputs.json",
        help="Output manifest for submit_rank_from_manifest.py",
    )
    p.add_argument("--dry-run", action="store_true", help="Parse only; no downloads; no manifest write")
    args = p.parse_args()

    res = materialize_from_xlsx(
        args.xlsx,
        local_root=args.local_root,
        download_dir=args.download_dir,
        dry_run=args.dry_run,
        do_download_urls=True,
    )

    if res.row_errors:
        print("[warn] row issues:")
        for e in res.row_errors[:40]:
            print(f"  {e}")
        if len(res.row_errors) > 40:
            print(f"  ... and {len(res.row_errors) - 40} more")

    n = len(res.items)
    est = n * (n - 1) // 2 if n > 1 else 0
    print(f"[info] target_text length={len(res.target_text)} chars")
    print(f"[info] items={n} (~{est} pairwise comparisons on server bubble sort)")

    if args.dry_run:
        print("[dry-run] no downloads, no manifest")
        return 0

    empty_bad = verify_files_non_empty(res.items)
    if empty_bad:
        print("[error] file checks failed:")
        for e in empty_bad[:50]:
            print(f"  {e}")
        return 1

    data = manifest_dict(res)
    write_manifest(args.manifest, data)
    print(f"[ok] wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
