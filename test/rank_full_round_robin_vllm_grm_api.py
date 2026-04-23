"""
Full round-robin ranking for URL list via vllm_grm_api.

Rule:
1) Compare every pair once (N URLs -> N*(N-1)/2 jobs).
2) Use result.score_a_avg / result.score_b_avg as pair scores.
3) Win points: win=1, tie=0.5, loss=0.
4) Final rank score = 0.7 * (win_rate * 10) + 0.3 * avg_score.
5) Grade by rank score: >=8.0 "优秀", >=6.5 "良好", else "待提升".
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE = "http://127.0.0.1:8001"
DEFAULT_TARGET = "smoke triplet screen"


def load_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # tolerate copied "L1:https://..." style lines
        if s.startswith("L") and ":" in s and s[1 : s.index(":")].isdigit():
            s = s.split(":", 1)[1].strip()
        urls.append(s)
    if len(urls) < 2:
        raise SystemExit(f"Need >=2 URLs in {path}")
    return urls


def ensure_health(client: httpx.Client, base: str) -> None:
    r = client.get(f"{base}/health")
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict) or data.get("status") != "ok":
        raise SystemExit(
            f"{base}/health is not vllm_grm_api style response: {data!r}"
        )


def create_job(
    client: httpx.Client,
    base: str,
    target_text: str,
    url_a: str,
    url_b: str,
    num_of_generation: int,
) -> str:
    payload = {
        "target_text": target_text,
        "audio_url_a": url_a,
        "audio_url_b": url_b,
        "num_of_generation": num_of_generation,
    }
    r = client.post(f"{base}/jobs", json=payload)
    r.raise_for_status()
    return r.json()["task_id"]


def wait_job(client: httpx.Client, base: str, task_id: str, poll_interval: float) -> dict[str, Any]:
    while True:
        r = client.get(f"{base}/jobs/{task_id}")
        r.raise_for_status()
        body: dict[str, Any] = r.json()
        status = body.get("status")
        if status in ("completed", "failed"):
            return body
        time.sleep(poll_interval)


def grade_from_rank_score(rank_score: float) -> str:
    if rank_score >= 8.0:
        return "优秀"
    if rank_score >= 6.5:
        return "良好"
    return "待提升"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parent / "testdate.txt",
    )
    ap.add_argument("--target-text", default=DEFAULT_TARGET)
    ap.add_argument("--num-of-generation", type=int, default=2)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "full_round_robin_rank.json",
    )
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    urls = load_urls(args.data_file)
    n = len(urls)
    all_pairs = list(combinations(range(n), 2))

    stats: dict[int, dict[str, Any]] = {
        i: {
            "url": urls[i],
            "win_points": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "score_sum": 0.0,
            "score_count": 0,
        }
        for i in range(n)
    }
    pair_results: list[dict[str, Any]] = []

    print(f"[info] urls={n}, pairs={len(all_pairs)}")
    with httpx.Client(timeout=120.0) as client:
        ensure_health(client, base)
        print("[health] ok")

        for idx, (i, j) in enumerate(all_pairs, start=1):
            print(f"[pair {idx}/{len(all_pairs)}] submit {i} vs {j}")
            task_id = create_job(
                client,
                base,
                args.target_text,
                urls[i],
                urls[j],
                args.num_of_generation,
            )
            body = wait_job(client, base, task_id, args.poll_interval)
            status = body.get("status")
            if status != "completed":
                raise SystemExit(f"Job failed for pair ({i},{j}), task_id={task_id}: {body.get('error')}")

            result = body.get("result") or {}
            sa = result.get("score_a_avg")
            sb = result.get("score_b_avg")
            if sa is None or sb is None:
                raise SystemExit(f"Missing score_a_avg/score_b_avg in task {task_id}: {result}")
            sa = float(sa)
            sb = float(sb)

            stats[i]["score_sum"] += sa
            stats[i]["score_count"] += 1
            stats[j]["score_sum"] += sb
            stats[j]["score_count"] += 1

            if sa > sb:
                stats[i]["wins"] += 1
                stats[i]["win_points"] += 1.0
                stats[j]["losses"] += 1
                outcome = "A"
            elif sa < sb:
                stats[j]["wins"] += 1
                stats[j]["win_points"] += 1.0
                stats[i]["losses"] += 1
                outcome = "B"
            else:
                stats[i]["ties"] += 1
                stats[j]["ties"] += 1
                stats[i]["win_points"] += 0.5
                stats[j]["win_points"] += 0.5
                outcome = "Tie"

            pair_results.append(
                {
                    "pair_index": idx,
                    "i": i,
                    "j": j,
                    "url_i": urls[i],
                    "url_j": urls[j],
                    "task_id": task_id,
                    "score_i": sa,
                    "score_j": sb,
                    "winner": outcome,
                }
            )

    ranking: list[dict[str, Any]] = []
    for i in range(n):
        st = stats[i]
        matches = n - 1
        win_rate = st["win_points"] / matches if matches else 0.0
        avg_score = st["score_sum"] / st["score_count"] if st["score_count"] else 0.0
        rank_score = 0.7 * (win_rate * 10.0) + 0.3 * avg_score
        ranking.append(
            {
                "url": st["url"],
                "wins": st["wins"],
                "ties": st["ties"],
                "losses": st["losses"],
                "win_points": round(st["win_points"], 4),
                "win_rate": round(win_rate, 6),
                "avg_score": round(avg_score, 6),
                "rank_score": round(rank_score, 6),
                "grade": grade_from_rank_score(rank_score),
            }
        )

    ranking.sort(key=lambda x: (x["rank_score"], x["avg_score"]), reverse=True)
    for rank_idx, row in enumerate(ranking, start=1):
        row["rank"] = rank_idx

    output_data: dict[str, Any] = {
        "rule": {
            "method": "full_round_robin_pairwise",
            "pairs": len(all_pairs),
            "win_points": "win=1, tie=0.5, loss=0",
            "rank_score_formula": "0.7 * (win_rate * 10) + 0.3 * avg_score",
            "grade_thresholds": {"优秀": "rank_score>=8.0", "良好": "rank_score>=6.5"},
        },
        "meta": {
            "n_urls": n,
            "target_text": args.target_text,
            "num_of_generation": args.num_of_generation,
            "base_url": base,
        },
        "ranking": ranking,
        "pair_results": pair_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] output: {args.output}")
    print("[top 10]")
    for row in ranking[:10]:
        print(
            f"#{row['rank']:02d} {row['grade']} "
            f"rank_score={row['rank_score']:.3f} "
            f"avg_score={row['avg_score']:.3f} "
            f"wins={row['wins']} ties={row['ties']} losses={row['losses']} "
            f"url={row['url']}"
        )


if __name__ == "__main__":
    main()
