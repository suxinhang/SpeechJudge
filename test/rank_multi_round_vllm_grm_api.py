"""
Multi-round full ranking via vllm_grm_api.

For each round:
1) Run full round-robin pairwise comparisons over all URLs.
2) Build per-round ranking by rank_score:
   rank_score = 0.7 * (win_rate * 10) + 0.3 * avg_score
3) Grade by rank_score:
   >= 8.0 => 优秀
   >= 6.5 => 良好
   else   => 待提升

Final ranking aggregates all rounds by:
- top5_hits desc
- avg_rank asc
- rank_std asc
"""

from __future__ import annotations

import argparse
import json
import statistics
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
        raise SystemExit(f"{base}/health is not vllm_grm_api style response: {data!r}")


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


def wait_job(
    client: httpx.Client,
    base: str,
    task_id: str,
    poll_interval: float,
) -> dict[str, Any]:
    while True:
        r = client.get(f"{base}/jobs/{task_id}")
        r.raise_for_status()
        body: dict[str, Any] = r.json()
        if body.get("status") in ("completed", "failed"):
            return body
        time.sleep(poll_interval)


def grade_from_rank_score(rank_score: float) -> str:
    if rank_score >= 8.0:
        return "优秀"
    if rank_score >= 6.5:
        return "良好"
    return "待提升"


def run_one_round(
    *,
    client: httpx.Client,
    base: str,
    urls: list[str],
    target_text: str,
    num_of_generation: int,
    poll_interval: float,
    round_idx: int,
    total_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

    for pair_idx, (i, j) in enumerate(all_pairs, start=1):
        print(
            f"[round {round_idx}/{total_rounds}] "
            f"[pair {pair_idx}/{len(all_pairs)}] submit {i} vs {j}"
        )
        task_id = create_job(
            client,
            base,
            target_text,
            urls[i],
            urls[j],
            num_of_generation,
        )
        body = wait_job(client, base, task_id, poll_interval)
        if body.get("status") != "completed":
            raise SystemExit(
                f"Job failed at round={round_idx}, pair=({i},{j}), task_id={task_id}: "
                f"{body.get('error')}"
            )

        result = body.get("result") or {}
        sa = result.get("score_a_avg")
        sb = result.get("score_b_avg")
        if sa is None or sb is None:
            raise SystemExit(
                f"Missing score_a_avg/score_b_avg in task {task_id}: {result}"
            )
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
            winner = "A"
        elif sa < sb:
            stats[j]["wins"] += 1
            stats[j]["win_points"] += 1.0
            stats[i]["losses"] += 1
            winner = "B"
        else:
            stats[i]["ties"] += 1
            stats[j]["ties"] += 1
            stats[i]["win_points"] += 0.5
            stats[j]["win_points"] += 0.5
            winner = "Tie"

        pair_results.append(
            {
                "round": round_idx,
                "pair_index": pair_idx,
                "i": i,
                "j": j,
                "url_i": urls[i],
                "url_j": urls[j],
                "task_id": task_id,
                "score_i": sa,
                "score_j": sb,
                "winner": winner,
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
                "url_index": i,
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
    return ranking, pair_results


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
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "multi_round_rank",
    )
    args = ap.parse_args()

    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    base = args.base_url.rstrip("/")
    urls = load_urls(args.data_file)
    n = len(urls)
    pairs_per_round = n * (n - 1) // 2
    print(
        f"[info] urls={n}, pairs/round={pairs_per_round}, rounds={args.rounds}, "
        f"total_jobs={pairs_per_round * args.rounds}"
    )

    round_rankings: list[dict[str, Any]] = []
    all_pair_results: list[dict[str, Any]] = []

    with httpx.Client(timeout=120.0) as client:
        ensure_health(client, base)
        print("[health] ok")

        for round_idx in range(1, args.rounds + 1):
            ranking, pair_results = run_one_round(
                client=client,
                base=base,
                urls=urls,
                target_text=args.target_text,
                num_of_generation=args.num_of_generation,
                poll_interval=args.poll_interval,
                round_idx=round_idx,
                total_rounds=args.rounds,
            )
            round_rankings.append(
                {
                    "round": round_idx,
                    "ranking": ranking,
                }
            )
            all_pair_results.extend(pair_results)

    by_url_idx: dict[int, dict[str, Any]] = {
        i: {
            "url": urls[i],
            "ranks": [],
            "rank_scores": [],
            "avg_scores": [],
            "top1_hits": 0,
            "top3_hits": 0,
            "top5_hits": 0,
            "top10_hits": 0,
        }
        for i in range(n)
    }

    for rr in round_rankings:
        for row in rr["ranking"]:
            i = int(row["url_index"])
            rank = int(row["rank"])
            by_url_idx[i]["ranks"].append(rank)
            by_url_idx[i]["rank_scores"].append(float(row["rank_score"]))
            by_url_idx[i]["avg_scores"].append(float(row["avg_score"]))
            if rank == 1:
                by_url_idx[i]["top1_hits"] += 1
            if rank <= 3:
                by_url_idx[i]["top3_hits"] += 1
            if rank <= 5:
                by_url_idx[i]["top5_hits"] += 1
            if rank <= 10:
                by_url_idx[i]["top10_hits"] += 1

    final_rows: list[dict[str, Any]] = []
    for i in range(n):
        st = by_url_idx[i]
        ranks = st["ranks"]
        avg_rank = statistics.mean(ranks)
        rank_std = statistics.pstdev(ranks) if len(ranks) > 1 else 0.0
        avg_rank_score = statistics.mean(st["rank_scores"])
        avg_avg_score = statistics.mean(st["avg_scores"])
        final_rows.append(
            {
                "url_index": i,
                "url": st["url"],
                "rounds": len(ranks),
                "avg_rank": round(avg_rank, 6),
                "rank_std": round(rank_std, 6),
                "avg_rank_score": round(avg_rank_score, 6),
                "avg_score": round(avg_avg_score, 6),
                "top1_hits": st["top1_hits"],
                "top3_hits": st["top3_hits"],
                "top5_hits": st["top5_hits"],
                "top10_hits": st["top10_hits"],
                "grade": grade_from_rank_score(avg_rank_score),
            }
        )

    final_rows.sort(
        key=lambda x: (
            x["top5_hits"],
            -x["avg_rank"],
            -x["rank_std"],
            x["avg_rank_score"],
            x["avg_score"],
        ),
        reverse=True,
    )
    for rank_idx, row in enumerate(final_rows, start=1):
        row["final_rank"] = rank_idx

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.out_dir / "final_top_ranking.json"
    stability_path = args.out_dir / "stability_report.json"
    txt_path = args.out_dir / "full_ranking.txt"

    final_payload = {
        "rule": {
            "method": "multi_round_full_round_robin_pairwise",
            "rounds": args.rounds,
            "pairs_per_round": pairs_per_round,
            "win_points": "win=1, tie=0.5, loss=0",
            "rank_score_formula": "0.7 * (win_rate * 10) + 0.3 * avg_score",
            "final_sort": ["top5_hits(desc)", "avg_rank(asc)", "rank_std(asc)", "avg_rank_score(desc)"],
        },
        "meta": {
            "n_urls": n,
            "target_text": args.target_text,
            "num_of_generation": args.num_of_generation,
            "base_url": base,
        },
        "ranking": final_rows,
    }

    stability_payload = {
        "meta": {
            "rounds": args.rounds,
            "n_urls": n,
            "pairs_per_round": pairs_per_round,
            "total_jobs": pairs_per_round * args.rounds,
        },
        "per_round_rankings": round_rankings,
        "pair_results": all_pair_results,
    }

    final_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stability_path.write_text(
        json.dumps(stability_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Full ranking (multi-round stable ranking)\n")
        f.write(
            f"rounds={args.rounds}, num_of_generation={args.num_of_generation}, total={len(final_rows)}\n\n"
        )
        for row in final_rows:
            f.write(
                f"#{row['final_rank']:02d} {row['grade']} "
                f"avg_rank={row['avg_rank']:.3f} rank_std={row['rank_std']:.3f} "
                f"top5_hits={row['top5_hits']}/{args.rounds} url={row['url']}\n"
            )

    print(f"[done] final ranking: {final_path}")
    print(f"[done] stability report: {stability_path}")
    print(f"[done] full ranking txt: {txt_path}")
    print("[full ranking]")
    for row in final_rows:
        print(
            f"#{row['final_rank']:02d} {row['grade']} "
            f"avg_rank={row['avg_rank']:.3f} rank_std={row['rank_std']:.3f} "
            f"top5_hits={row['top5_hits']}/{args.rounds} url={row['url']}"
        )


if __name__ == "__main__":
    main()
