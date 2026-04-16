"""Batch test: POST /score-url on a remote SpeechJudge API (e.g. Cloudflare tunnel).

All multi-language cases live in one file (default ``infer/examples/user_demo/score_url_batch_cases.json``).
Each **suite** has ``id``, ``lang``, ``dataset``, ``target`` (``path`` under ``infer/examples/``),
and ``rows`` of ``{name, url, score?}``.

**Sync Thai rows from CSV manifests** (preserves ``fr_book`` and other suites; only refreshes
``thai_booksummary`` / ``thai_tts_demo``):

    python test/tmp_score_url_batch_test.py --sync-cases

**Run one language** (filter suites by case file ``lang``):

    python test/tmp_score_url_batch_test.py --lang fr
    python test/tmp_score_url_batch_test.py --lang en
    python test/tmp_score_url_batch_test.py --lang tr
    python test/tmp_score_url_batch_test.py --lang th
    python test/tmp_score_url_batch_test.py --lang th --preset tts_demo

**Run everything** (all suites that have non-empty ``rows``):

    python test/tmp_score_url_batch_test.py

**Pick suites by id** (optional; combined with ``--lang`` as intersection):

    python test/tmp_score_url_batch_test.py --suite thai_tts_demo
    python test/tmp_score_url_batch_test.py --suite thai_booksummary --suite thai_tts_demo

Override tunnel:

    python test/tmp_score_url_batch_test.py --url https://other.trycloudflare.com

Optional **YAML** case files: ``.yaml`` / ``.yml`` + ``pip install pyyaml``.

TryCloudflare hostnames change each restart. Past examples (replace with your current URL):

- https://fcc-therapy-munich-boost.trycloudflare.com
- https://disturbed-seed-proceedings-market.trycloudflare.com
- https://reads-absolute-chains-expressed.trycloudflare.com
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "infer" / "examples"
_USER_DEMO = _EXAMPLES / "user_demo"
DEFAULT_CASES_FILE = _USER_DEMO / "score_url_batch_cases.json"
DEFAULT_BASE_URL = "https://fcc-therapy-munich-boost.trycloudflare.com"

EMBEDDED_TARGET_THAI = """ตั้งแต่อายุยังน้อย คิโยซากิและไมค์ เพื่อนของเขามีความปรารถนาอย่างแรงกล้าที่จะกลายเป็นคนร่ำรวย อย่างไรก็ตาม ในตอนแรกพวกเขาไม่รู้ว่าจะทำอย่างไรจึงจะบรรลุเป้าหมายนี้ได้ เมื่อพวกเขาไปขอคำแนะนำจากพ่อของตนเอง พวกเขากลับได้รับคำตอบที่แตกต่างกันอย่างสิ้นเชิง พ่อที่ยากจนของคิโยซากิซึ่งมีการศึกษาดีแต่มีปัญหาทางการเงิน แนะนำให้พวกเขาตั้งใจเรียนและหางานที่มั่นคงทำ แม้คำแนะนำแบบดั้งเดิมนี้จะมาจากความหวังดี แต่มันมักทำให้ผู้คนติดอยู่ในวงจรของการทำงานหนักเพื่อเงิน โดยไม่สามารถสร้างความมั่งคั่งที่แท้จริงได้
พ่อที่ยากจนของคิโยซากิเป็นตัวแทนของแนวคิดแบบดั้งเดิมที่ผู้คนจำนวนมากยังคงยึดถือมาจนถึงทุกวันนี้ แนวคิดนี้มักเกิดจากความกลัวต่อความไม่มั่นคงทางการเงิน และความเชื่อว่าการมีการศึกษาที่ดีและงานที่มั่นคงคือกุญแจสู่ความสำเร็จ อย่างไรก็ตาม คิโยซากิอธิบายว่าแนวทางนี้อาจทำให้ผู้คนติดอยู่ในสิ่งที่เรียกว่า “วงจรหนูวิ่ง” หรือ rat race ซึ่งหมายถึงการทำงานอย่างหนักเพื่อรับเงินเดือน แต่เงินจำนวนมากกลับถูกใช้ไปกับภาษี ค่าบิล และค่าใช้จ่ายต่าง ๆ ในชีวิตประจำวัน ดังนั้น แม้ว่าพวกเขาอาจหลีกเลี่ยงความยากจนได้ แต่ก็ยังไม่สามารถสะสมความมั่งคั่งที่แท้จริงได้เลย"""

EMBEDDED_TARGET_FR = (
    "Chaque fois que nous nous fixons un nouvel objectif — qu\u2019il s\u2019agisse de perdre du poids, "
    "de créer une entreprise ou d\u2019écrire un livre — la société nous conditionne à croire que nous devons "
    "opérer des changements monumentaux dans notre comportement. Nous nous mettons une pression immense pour "
    "réaliser une amélioration spectaculaire, quelque chose de si marquant que tout le monde en parlera. "
    "Pourtant, cette approche est fondamentalement erronée. Pour comprendre pourquoi, il suffit de considérer "
    "l\u2019histoire remarquable de l\u2019équipe britannique de cyclisme.\n"
    "Pendant près de cent ans, les cyclistes britanniques ont connu une période étonnamment longue de médiocrité. "
    "Ils n\u2019avaient jamais remporté le Tour de France et leurs performances aux Jeux olympiques étaient "
    "constamment décevantes. En réalité, leur réputation était si mauvaise que les plus grands fabricants de "
    "vélos en Europe refusaient même de vendre leur équipement à l\u2019équipe, de peur que leur marque ne soit "
    "ternie si d\u2019autres professionnels voyaient les coureurs britanniques utiliser leur matériel."
)

# Matches infer/examples/en_book/target_en.txt (Rich Dad / Kiyosaki excerpt).
EMBEDDED_TARGET_EN = (
    "From a young age, Kiyosaki and his friend Mike were driven by a desire to become wealthy. However, "
    "they were initially unsure of how to achieve this goal. When they sought advice from their fathers, "
    "they received vastly different responses. Poor Dad, who was well-educated but financially struggling, "
    "advised them to focus on their education and secure a stable job. This conventional wisdom, while "
    "well-intentioned, often leads individuals into a cycle of working hard for money without ever truly "
    "building wealth.\n\n"
    "Kiyosaki's Poor Dad represented the traditional mindset that many people still adhere to today. "
    "This mindset is characterized by a fear of financial instability and a belief that a good education "
    "and a steady job are the keys to success. However, Kiyosaki argues that this approach can trap "
    "individuals in a 'rat race,' where they work tirelessly to earn a paycheck, only to see a significant "
    "portion of their earnings go to taxes, bills, and other expenses. As a result, they may avoid poverty "
    "but fail to accumulate real wealth."
)

# Matches infer/examples/tr_book/target_tr.txt (Kiyosaki / Fakir Baba, Turkish).
EMBEDDED_TARGET_TR = (
    "Kiyosaki ve arkadaşı Mike, küçük yaşlardan itibaren zengin olma arzusuyla hareket ediyordu. Ancak bu hedefe "
    "nasıl ulaşacaklarını başta bilmiyorlardı. Babalarından tavsiye istediklerinde ise birbirinden oldukça farklı "
    "cevaplar aldılar. İyi eğitimli olmasına rağmen maddi açıdan zorlanan Fakir Baba, onlara eğitimlerine "
    "odaklanmalarını ve güvenli, istikrarlı bir iş bulmalarını tavsiye etti. İyi niyetli olsa da bu geleneksel "
    "anlayış, çoğu zaman insanları para için sürekli çalıştıkları, fakat gerçek anlamda servet biriktiremedikleri "
    "bir döngüye sürükler.\n\n"
    "Kiyosaki'nin Fakir Babası, bugün hâlâ pek çok insanın benimsediği geleneksel zihniyeti temsil ediyordu. "
    "Bu düşünce yapısı, maddi istikrarsızlık korkusuna ve iyi bir eğitim ile düzenli bir işin başarının anahtarı "
    "olduğuna dair inanca dayanır. Ancak Kiyosaki'ye göre bu yaklaşım, insanları bir 'fare yarışına' hapsedebilir; "
    "burada insanlar maaş kazanmak için durmadan çalışır, fakat kazançlarının büyük bir kısmı vergilere, faturalara "
    "ve diğer giderlere gider. Sonuç olarak, yoksulluktan kaçınabilirler ama gerçek servet biriktirmeyi başaramazlar."
)


def load_cases_file(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped, import-not-found]
        except ImportError as e:
            raise SystemExit(
                "PyYAML is required for .yaml/.yml case files. Install: pip install pyyaml"
            ) from e
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error: {e}") from e
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}") from e
    if data is None or not isinstance(data, dict) or "suites" not in data:
        raise ValueError("case file must be a JSON/YAML object with key 'suites'")
    return data


def rows_from_manifest_csv(manifest: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if not url:
                continue
            name = (row.get("base_name") or Path(url).stem or url).strip()
            r: dict[str, Any] = {"name": name, "url": url}
            score_s = (row.get("score") or "").strip()
            if score_s:
                try:
                    r["score"] = float(score_s)
                except ValueError:
                    pass
            out.append(r)
    return out


def sync_cases_from_manifests(cases_path: Path) -> None:
    """Refresh ``thai_booksummary`` / ``thai_tts_demo`` ``rows`` from manifest CSVs; keep other suites."""
    book_m = _EXAMPLES / "thai_booksummary" / "manifest.csv"
    tts_m = _EXAMPLES / "thai_tts_demo" / "manifest.csv"
    book_rows = rows_from_manifest_csv(book_m) if book_m.is_file() else []
    tts_rows = rows_from_manifest_csv(tts_m) if tts_m.is_file() else []

    thai_book_tpl: dict[str, Any] = {
        "id": "thai_booksummary",
        "lang": "th",
        "dataset": "booksummary",
        "description": (
            "BookSummary Thai audio; reference transcript thai_booksummary/target_thai.txt"
        ),
        "target": {"path": "thai_booksummary/target_thai.txt"},
        "rows": book_rows,
    }
    thai_tts_tpl: dict[str, Any] = {
        "id": "thai_tts_demo",
        "lang": "th",
        "dataset": "tts_demo",
        "description": (
            "explore/tts_demo Thai audio; reference transcript thai_tts_demo/target_thai.txt"
        ),
        "target": {"path": "thai_tts_demo/target_thai.txt"},
        "rows": tts_rows,
    }
    fr_default: dict[str, Any] = {
        "id": "fr_book",
        "lang": "fr",
        "dataset": "fr_book",
        "description": (
            "French TTS; audio must match fr_book/target_fr.txt; edit rows in this JSON"
        ),
        "target": {"path": "fr_book/target_fr.txt"},
        "rows": [],
    }

    by_id: dict[str, dict[str, Any]] = {}
    version = 1
    if cases_path.is_file():
        try:
            raw_data = json.loads(cases_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"cannot sync — invalid JSON: {e}") from e
        if isinstance(raw_data, dict):
            version = int(raw_data.get("version") or 1)
            suites_list = raw_data.get("suites")
            if isinstance(suites_list, list):
                for s in suites_list:
                    if isinstance(s, dict) and s.get("id"):
                        by_id[str(s["id"])] = copy.deepcopy(s)

    if "thai_booksummary" in by_id:
        by_id["thai_booksummary"]["rows"] = book_rows
        for k, v in thai_book_tpl.items():
            if k != "rows":
                by_id["thai_booksummary"].setdefault(k, v)
    else:
        by_id["thai_booksummary"] = copy.deepcopy(thai_book_tpl)

    if "thai_tts_demo" in by_id:
        by_id["thai_tts_demo"]["rows"] = tts_rows
        for k, v in thai_tts_tpl.items():
            if k != "rows":
                by_id["thai_tts_demo"].setdefault(k, v)
    else:
        by_id["thai_tts_demo"] = copy.deepcopy(thai_tts_tpl)

    if "fr_book" not in by_id:
        by_id["fr_book"] = copy.deepcopy(fr_default)

    en_default: dict[str, Any] = {
        "id": "en_book",
        "lang": "en",
        "dataset": "en_book",
        "description": (
            "English TTS; audio must match en_book/target_en.txt; edit rows in this JSON"
        ),
        "target": {"path": "en_book/target_en.txt"},
        "rows": [],
    }
    if "en_book" not in by_id:
        by_id["en_book"] = copy.deepcopy(en_default)

    tr_default: dict[str, Any] = {
        "id": "tr_book",
        "lang": "tr",
        "dataset": "tr_book",
        "description": (
            "Turkish TTS; audio must match tr_book/target_tr.txt; edit rows in this JSON"
        ),
        "target": {"path": "tr_book/target_tr.txt"},
        "rows": [],
    }
    if "tr_book" not in by_id:
        by_id["tr_book"] = copy.deepcopy(tr_default)

    priority = ["thai_booksummary", "thai_tts_demo", "fr_book", "en_book", "tr_book"]
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sid in priority:
        if sid in by_id:
            ordered.append(by_id[sid])
            seen.add(sid)
    for sid in sorted(k for k in by_id if k not in seen):
        ordered.append(by_id[sid])

    data = {"version": version, "suites": ordered}
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_fr = len(by_id.get("fr_book", {}).get("rows") or [])
    n_en = len(by_id.get("en_book", {}).get("rows") or [])
    n_tr = len(by_id.get("tr_book", {}).get("rows") or [])
    print(
        f"Synced {cases_path}: thai_booksummary={len(book_rows)} rows, "
        f"thai_tts_demo={len(tts_rows)} rows, fr_book={n_fr} rows, en_book={n_en} rows, "
        f"tr_book={n_tr} rows",
        flush=True,
    )


def row_expected_score(row: dict[str, Any]) -> float:
    if "score" not in row or row["score"] is None:
        return float("nan")
    try:
        return float(row["score"])
    except (TypeError, ValueError):
        return float("nan")


def select_suites(
    data: dict[str, Any],
    preset: str,
    suite_ids: list[str] | None,
    lang_filter: str | None,
) -> list[dict[str, Any]]:
    suites_in = data["suites"]
    if not isinstance(suites_in, list):
        raise ValueError("'suites' must be a list")

    if suite_ids:
        id_want = set(suite_ids)
        sel = [copy.deepcopy(s) for s in suites_in if isinstance(s, dict) and s.get("id") in id_want]
        found = {s.get("id") for s in sel}
        missing = id_want - found
        if missing:
            known = [s.get("id") for s in suites_in if isinstance(s, dict)]
            raise SystemExit(f"Unknown suite id(s): {sorted(missing)}. Known: {known}")
    elif preset == "all":
        sel = [copy.deepcopy(s) for s in suites_in if isinstance(s, dict) and s.get("rows")]
    elif preset == "booksummary":
        sel = [copy.deepcopy(s) for s in suites_in if isinstance(s, dict) and s.get("id") == "thai_booksummary"]
    elif preset == "tts_demo":
        sel = [copy.deepcopy(s) for s in suites_in if isinstance(s, dict) and s.get("id") == "thai_tts_demo"]
    else:
        raise SystemExit(f"internal: bad preset {preset!r}")

    if lang_filter:
        lf = lang_filter.strip().lower()
        sel = [s for s in sel if str(s.get("lang", "")).strip().lower() == lf]
        if not sel:
            langs = sorted(
                {str(s.get("lang", "")).lower() for s in suites_in if isinstance(s, dict)}
            )
            raise SystemExit(
                f"No suites match --lang {lang_filter!r} for this selection "
                f"(preset={preset!r}, suite={suite_ids!r}). Languages in file: {langs}"
            )
    return sel


def resolve_suite_target_text(
    suite: dict[str, Any],
    global_target_file: Path | None,
) -> tuple[str, str]:
    """Return (target_text, source_label for logging)."""
    if global_target_file is not None and global_target_file.is_file():
        t = global_target_file.read_text(encoding="utf-8").strip()
        return t, str(global_target_file)

    tgt = suite.get("target") if isinstance(suite.get("target"), dict) else {}
    path_s = (tgt.get("path") or "").strip()
    if path_s:
        p = _EXAMPLES / path_s
        if p.is_file():
            return p.read_text(encoding="utf-8").strip(), str(p)
    inline = tgt.get("inline")
    if inline is not None and str(inline).strip():
        return str(inline).strip(), "(suite target.inline)"

    lang = (suite.get("lang") or "th").strip().lower()
    if lang == "fr":
        return EMBEDDED_TARGET_FR.strip(), "(embedded fr)"
    if lang == "en":
        return EMBEDDED_TARGET_EN.strip(), "(embedded en)"
    if lang == "tr":
        return EMBEDDED_TARGET_TR.strip(), "(embedded tr)"
    return EMBEDDED_TARGET_THAI.strip(), "(embedded th)"


def apply_manifest_override(
    suites: list[dict[str, Any]],
    manifest: Path | None,
    preset: str,
    suite_ids: list[str] | None,
) -> None:
    if manifest is None:
        return
    if preset == "all":
        print(
            "# warning: --manifest ignored with --preset all (ambiguous); use e.g. "
            "--suite thai_tts_demo --manifest path.csv",
            file=sys.stderr,
            flush=True,
        )
        return
    if not manifest.is_file():
        print(
            f"# warning: --manifest not found ({manifest}); using rows from case file.",
            file=sys.stderr,
            flush=True,
        )
        return
    ovr = rows_from_manifest_csv(manifest)
    target_id: str | None = None
    if suite_ids:
        if len(suite_ids) != 1:
            print(
                "# warning: --manifest with multiple --suite values is ambiguous; skipping override.",
                file=sys.stderr,
                flush=True,
            )
            return
        target_id = suite_ids[0]
    elif preset == "booksummary":
        target_id = "thai_booksummary"
    elif preset == "tts_demo":
        target_id = "thai_tts_demo"
    else:
        print(
            "# warning: --manifest needs --preset booksummary|tts_demo or a single --suite SUITE_ID; "
            "skipping override.",
            file=sys.stderr,
            flush=True,
        )
        return
    for s in suites:
        if s.get("id") == target_id:
            s["rows"] = ovr
            print(f"# note: rows for suite {target_id!r} replaced from {manifest}", flush=True)
            return
    print(
        f"# warning: suite {target_id!r} not in current run; manifest override skipped.",
        file=sys.stderr,
        flush=True,
    )


def warn_fr_thai_mismatch(suite: dict[str, Any]) -> None:
    # These suites legitimately use explore/tts_demo / BookSummary URLs for non-Thai audio.
    if suite.get("id") in {"fr_book", "en_book", "tr_book"}:
        return
    lang = (suite.get("lang") or "").strip().lower()
    if lang != "fr":
        return
    rows = suite.get("rows") or []
    if not rows:
        return
    thai_like = sum(
        1
        for r in rows
        if str(r.get("name") or "").startswith("th-TH")
        or "/BookSummary/" in str(r.get("url") or "")
    )
    if thai_like == len(rows):
        print(
            f"# warning: suite {suite.get('id')!r} is lang=fr but rows look like Thai demo/booksummary URLs; "
            "scores will be very low unless audio speaks the French target.",
            file=sys.stderr,
            flush=True,
        )


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "测试套件",
            "数据集",
            "参考语言",
            "音色",
            "音频URL",
            "参考分",
            "预测分",
            "分差",
            "耗时秒",
            "是否成功",
            "错误信息",
        ]
    )
    for r in rows:
        w.writerow(
            [
                r["suite_id"],
                r["dataset"],
                r["lang"],
                r["name"],
                r["url"],
                r["exp"],
                r["pred"],
                r["diff"],
                r["elapsed"],
                r["ok"],
                r["error"],
            ]
        )
    path.write_text("\ufeff" + buf.getvalue(), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sync-cases",
        action="store_true",
        help=(
            "Update thai_booksummary + thai_tts_demo rows from infer/examples/*/manifest.csv; "
            "preserve fr_book and other suites; write --cases file; then exit."
        ),
    )
    p.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_FILE,
        help=f"Case file JSON or YAML (default: {DEFAULT_CASES_FILE}).",
    )
    p.add_argument(
        "--suite",
        action="append",
        default=None,
        metavar="ID",
        help="Suite id from case file (repeat for multiple). If omitted, --preset selects suites.",
    )
    p.add_argument(
        "--preset",
        default="all",
        choices=["all", "booksummary", "tts_demo"],
        help="Legacy: all = suites with rows; booksummary / tts_demo = one Thai suite each.",
    )
    p.add_argument(
        "--base-url",
        "--url",
        dest="base_url",
        default=DEFAULT_BASE_URL,
        metavar="URL",
        help="API base URL (no trailing slash). Alias: --url",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV (url, base_name, score) replacing rows for one suite: use with "
            "--preset booksummary|tts_demo or a single --suite."
        ),
    )
    p.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help="Override reference transcript for every suite in this run (UTF-8).",
    )
    p.add_argument(
        "--lang",
        default=None,
        choices=["th", "fr", "en", "tr"],
        metavar="LANG",
        help=(
            "Run only suites whose case file ``lang`` matches (fr/en/tr/th). "
            "Combined with --suite as intersection. Default: all selected suites."
        ),
    )
    p.add_argument("--mode", default="compact", choices=["fast", "compact", "analysis"])
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (seconds)")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write results with Chinese headers (UTF-8 BOM for Excel).",
    )
    args = p.parse_args()

    cases_path = args.cases
    if args.sync_cases:
        if cases_path.suffix.lower() in {".yaml", ".yml"}:
            print(
                "--sync-cases writes JSON only; use e.g. --cases score_url_batch_cases.json",
                file=sys.stderr,
            )
            return 1
        try:
            sync_cases_from_manifests(cases_path)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        return 0

    if not cases_path.is_file():
        print(
            f"case file not found: {cases_path}\n"
            f"Hint: create it with  {Path(__file__).name}  --sync-cases",
            file=sys.stderr,
        )
        return 1

    try:
        data = load_cases_file(cases_path)
    except ValueError as e:
        print(f"invalid case file: {e}", file=sys.stderr)
        return 1

    suite_ids = args.suite if args.suite else None
    try:
        suites = select_suites(data, args.preset, suite_ids, args.lang)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1

    apply_manifest_override(suites, args.manifest, args.preset, suite_ids)

    for s in suites:
        rows = s.get("rows") or []
        if not rows:
            print(f"suite {s.get('id')!r} has no rows; add urls in {cases_path.name}", file=sys.stderr)
            return 1
        warn_fr_thai_mismatch(s)

    n_rows = sum(len(s.get("rows") or []) for s in suites)
    lang_note = args.lang or "all"
    print(
        f"# cases={cases_path.name}  lang_filter={lang_note}  "
        f"suites={[s.get('id') for s in suites]}  total_rows={n_rows}  preset={args.preset}",
        flush=True,
    )

    base_url = args.base_url.rstrip("/")
    score_url = f"{base_url}/score-url"
    ok = 0
    csv_rows: list[dict[str, object]] = []

    for suite in suites:
        suite_id = str(suite.get("id", ""))
        dataset = str(suite.get("dataset", suite_id))
        lang = str(suite.get("lang") or "th").strip().lower()
        try:
            target_text, tgt_src = resolve_suite_target_text(suite, args.target_file)
        except OSError as e:
            print(f"read target failed for suite {suite_id}: {e}", file=sys.stderr)
            return 1
        if not target_text:
            print(f"target text empty for suite {suite_id}", file=sys.stderr)
            return 1

        print(
            f"# suite={suite_id}  lang={lang}  rows={len(suite.get('rows') or [])}  target={tgt_src}",
            flush=True,
        )

        for row in suite.get("rows") or []:
            name = str(row.get("name") or "").strip()
            audio_url = str(row.get("url") or "").strip()
            if not audio_url:
                continue
            if not name:
                name = Path(audio_url).stem or audio_url
            exp = row_expected_score(row)

            body = json.dumps(
                {
                    "audio_url": audio_url,
                    "target_text": target_text,
                    "mode": args.mode,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            req = urllib.request.Request(
                score_url,
                data=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            t0 = time.perf_counter()
            crow: dict[str, object] = {
                "suite_id": suite_id,
                "dataset": dataset,
                "lang": lang,
                "name": name,
                "url": audio_url,
                "exp": exp,
                "pred": "",
                "diff": "",
                "elapsed": "",
                "ok": 0,
                "error": "",
            }
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                score = resp_data.get("score")
                dt = time.perf_counter() - t0
                crow["elapsed"] = round(dt, 2)
                if score is not None and not math.isnan(exp):
                    diff = float(score) - exp
                else:
                    diff = None
                print(
                    f"[{lang}][{suite_id}]\t{name}\texp={exp}\tgot={score}\tdiff={diff}\t{dt:.1f}s"
                )
                if score is not None:
                    ok += 1
                    crow["ok"] = 1
                    crow["pred"] = score
                    crow["diff"] = diff if diff is not None else ""
                else:
                    crow["error"] = "no score in response"
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", errors="replace")[:500]
                print(f"[{lang}][{suite_id}]\t{name}\tHTTP {e.code}\t{err[:200]}")
                crow["error"] = f"HTTP {e.code} {err[:300]}"
            except Exception as e:
                print(f"[{lang}][{suite_id}]\t{name}\tERROR\t{e}")
                crow["error"] = str(e)

            csv_rows.append(crow)

    n_done = len(csv_rows)
    print(f"\nOK responses: {ok}/{n_done}")

    if args.output_csv is not None:
        write_results_csv(args.output_csv, csv_rows)
        print(f"Wrote {args.output_csv}", flush=True)

    return 0 if ok == n_done else 1


if __name__ == "__main__":
    sys.exit(main())
