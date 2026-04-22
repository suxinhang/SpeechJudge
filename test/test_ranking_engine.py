from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / "infer"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from rank_jobs_app.core.ranking import (
    ALGORITHM_FULL_PAIRWISE,
    ALGORITHM_PHASED_ELO,
    PHASE_CHALLENGE,
    PHASE_EXPLOIT,
    PHASE_EXPLORE,
    PHASE_FULL,
    PHASE_TOP_K,
    RankingConfig,
    RankingItem,
    collapse_pairwise_votes_adaptive,
    collapse_pairwise_votes,
    majority_vote,
)
from rank_jobs_app.services.ranking_engine import RankingEngine, _ItemState, _PairStats


class RankingEngineTests(unittest.TestCase):
    def test_majority_vote_prefers_sign_with_most_votes(self) -> None:
        self.assertEqual(majority_vote([1, -1, 1]), 1)
        self.assertEqual(majority_vote([-1, 0, -1]), -1)
        self.assertEqual(majority_vote([1, -1]), 0)

    def test_collapse_pairwise_votes_groups_repeated_pair_results(self) -> None:
        self.assertEqual(collapse_pairwise_votes([1, 1, -1, -1, 0, -1], 3), [1, -1])

    def test_collapse_pairwise_votes_adaptive_skips_third_vote_when_first_two_match(self) -> None:
        self.assertEqual(collapse_pairwise_votes_adaptive([1, 1, -1, 1], [-1]), [1, -1])

    def test_collapse_pairwise_votes_adaptive_uses_third_vote_on_disagreement(self) -> None:
        self.assertEqual(collapse_pairwise_votes_adaptive([1, -1, -1, -1], [1]), [1, -1])

    def test_engine_orders_strict_quality_scores(self) -> None:
        items = [
            RankingItem(id="a", wav_path="a.wav"),
            RankingItem(id="b", wav_path="b.wav"),
            RankingItem(id="c", wav_path="c.wav"),
            RankingItem(id="d", wav_path="d.wav"),
            RankingItem(id="e", wav_path="e.wav"),
            RankingItem(id="f", wav_path="f.wav"),
        ]
        quality = {
            "a": 90,
            "b": 70,
            "c": 50,
            "d": 30,
            "e": 10,
            "f": 0,
        }

        def compare_batch(batch: list[tuple[RankingItem, RankingItem]]) -> list[int]:
            out: list[int] = []
            for left, right in batch:
                if quality[left.id] > quality[right.id]:
                    out.append(1)
                elif quality[left.id] < quality[right.id]:
                    out.append(-1)
                else:
                    out.append(0)
            return out

        engine = RankingEngine(
            RankingConfig(
                algorithm=ALGORITHM_PHASED_ELO,
                top_k=3,
                budget_multiplier=3.0,
                neighbor_window=3,
                max_pair_repeats=2,
            )
        )
        result = engine.run(items=items, compare_batch=compare_batch, batch_size=3)

        self.assertEqual([entry.item.id for entry in result.items], ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(result.comparisons_total, result.comparisons_done)
        self.assertLessEqual(result.comparisons_done, result.comparisons_total)

    def test_engine_exposes_phase_counts_and_handles_ties(self) -> None:
        items = [
            RankingItem(id="a", wav_path="a.wav"),
            RankingItem(id="b", wav_path="b.wav"),
            RankingItem(id="c", wav_path="c.wav"),
            RankingItem(id="d", wav_path="d.wav"),
        ]
        quality = {
            "a": 100,
            "b": 100,
            "c": 60,
            "d": 10,
        }

        def compare_batch(batch: list[tuple[RankingItem, RankingItem]]) -> list[int]:
            out: list[int] = []
            for left, right in batch:
                if quality[left.id] == quality[right.id]:
                    out.append(0)
                elif quality[left.id] > quality[right.id]:
                    out.append(1)
                else:
                    out.append(-1)
            return out

        engine = RankingEngine(
            RankingConfig(
                algorithm=ALGORITHM_PHASED_ELO,
                top_k=2,
                budget_multiplier=2.5,
                neighbor_window=2,
                max_pair_repeats=3,
            )
        )
        result = engine.run(items=items, compare_batch=compare_batch, batch_size=2)

        self.assertEqual(
            set(result.phase_comparisons),
            {PHASE_EXPLORE, PHASE_EXPLOIT, PHASE_CHALLENGE, PHASE_TOP_K},
        )
        self.assertLessEqual(result.comparisons_done, result.comparisons_total)
        self.assertEqual({result.items[0].item.id, result.items[1].item.id}, {"a", "b"})

    def test_challenge_refine_prefers_near_boundary_defenders(self) -> None:
        items = [
            RankingItem(id="a", wav_path="a.wav"),
            RankingItem(id="b", wav_path="b.wav"),
            RankingItem(id="c", wav_path="c.wav"),
            RankingItem(id="d", wav_path="d.wav"),
            RankingItem(id="e", wav_path="e.wav"),
            RankingItem(id="f", wav_path="f.wav"),
            RankingItem(id="g", wav_path="g.wav"),
            RankingItem(id="h", wav_path="h.wav"),
        ]
        engine = RankingEngine(
            RankingConfig(
                algorithm=ALGORITHM_PHASED_ELO,
                top_k=3,
                neighbor_window=2,
                top_k_margin=2,
                max_pair_repeats=3,
            )
        )
        states = {
            item.id: _ItemState(item=item, rating=1700 - idx * 20)
            for idx, item in enumerate(items)
        }

        pairs = engine._select_challenge_pairs(states=states, pair_stats={}, limit=3)

        self.assertEqual(pairs[0], ("f", "c"))
        self.assertNotIn(("f", "b"), pairs[:2])

    def test_challenge_refine_surfaces_undercompared_contender(self) -> None:
        items = [RankingItem(id=chr(ord("a") + idx), wav_path=f"{idx}.wav") for idx in range(8)]
        engine = RankingEngine(
            RankingConfig(
                algorithm=ALGORITHM_PHASED_ELO,
                top_k=3,
                neighbor_window=2,
                top_k_margin=2,
                max_pair_repeats=3,
            )
        )
        states = {
            "a": _ItemState(item=items[0], rating=1700, comparisons=18, wins=14),
            "b": _ItemState(item=items[1], rating=1680, comparisons=18, wins=13),
            "c": _ItemState(item=items[2], rating=1660, comparisons=18, wins=12),
            "d": _ItemState(item=items[3], rating=1640, comparisons=18, wins=11),
            "e": _ItemState(item=items[4], rating=1620, comparisons=18, wins=10),
            "f": _ItemState(item=items[5], rating=1560, comparisons=14, wins=8),
            "g": _ItemState(item=items[6], rating=1545, comparisons=16, wins=8),
            "h": _ItemState(item=items[7], rating=1530, comparisons=1, wins=1),
        }

        pairs = engine._select_challenge_pairs(states=states, pair_stats={}, limit=3)

        self.assertEqual(pairs[0][0], "h")
        self.assertEqual(pairs[0][1], "e")

    def test_top_k_refine_prefers_boundary_local_pairs(self) -> None:
        items = [
            RankingItem(id="a", wav_path="a.wav"),
            RankingItem(id="b", wav_path="b.wav"),
            RankingItem(id="c", wav_path="c.wav"),
            RankingItem(id="d", wav_path="d.wav"),
            RankingItem(id="e", wav_path="e.wav"),
            RankingItem(id="f", wav_path="f.wav"),
        ]
        engine = RankingEngine(
            RankingConfig(
                algorithm=ALGORITHM_PHASED_ELO,
                top_k=3,
                neighbor_window=2,
                max_pair_repeats=3,
            )
        )
        states = {
            item.id: _ItemState(item=item, rating=1600 - idx * 25)
            for idx, item in enumerate(items)
        }

        pairs = engine._select_top_k_pairs(states=states, pair_stats={}, limit=3)

        self.assertEqual(pairs[0], ("b", "c"))
        self.assertNotIn(("a", "c"), pairs[:2])

    def test_full_pairwise_compares_every_unique_pair_once(self) -> None:
        items = [
            RankingItem(id="a", wav_path="a.wav"),
            RankingItem(id="b", wav_path="b.wav"),
            RankingItem(id="c", wav_path="c.wav"),
            RankingItem(id="d", wav_path="d.wav"),
        ]
        seen_pairs: list[tuple[str, str]] = []
        quality = {"a": 100, "b": 80, "c": 60, "d": 40}

        def compare_batch(batch: list[tuple[RankingItem, RankingItem]]) -> list[int]:
            out: list[int] = []
            for left, right in batch:
                seen_pairs.append((left.id, right.id))
                out.append(1 if quality[left.id] > quality[right.id] else -1)
            return out

        engine = RankingEngine(RankingConfig(algorithm=ALGORITHM_FULL_PAIRWISE))
        result = engine.run(items=items, compare_batch=compare_batch, batch_size=2)

        self.assertEqual(result.phase_comparisons, {PHASE_FULL: 6})
        self.assertEqual(result.comparisons_done, 6)
        self.assertEqual(result.comparisons_total, 6)
        self.assertEqual(
            {tuple(sorted(pair)) for pair in seen_pairs},
            {("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")},
        )
        self.assertEqual([entry.item.id for entry in result.items], ["a", "b", "c", "d"])

    def test_full_pairwise_ranks_by_round_robin_points_before_elo(self) -> None:
        items = {
            "a": RankingItem(id="a", wav_path="a.wav"),
            "b": RankingItem(id="b", wav_path="b.wav"),
            "c": RankingItem(id="c", wav_path="c.wav"),
        }
        engine = RankingEngine(RankingConfig(algorithm=ALGORITHM_FULL_PAIRWISE))
        states = {
            "a": _ItemState(item=items["a"], rating=1400, comparisons=2, wins=2.0),
            "b": _ItemState(item=items["b"], rating=1600, comparisons=2, wins=1.0),
            "c": _ItemState(item=items["c"], rating=1500, comparisons=2, wins=0.0),
        }
        pair_stats = {
            ("a", "b"): _PairStats(left_id="a", right_id="b", total=1, left_wins=1, right_wins=0, ties=0),
            ("a", "c"): _PairStats(left_id="a", right_id="c", total=1, left_wins=1, right_wins=0, ties=0),
            ("b", "c"): _PairStats(left_id="b", right_id="c", total=1, left_wins=1, right_wins=0, ties=0),
        }

        ranked = engine._rank_full_pairwise(states, pair_stats)

        self.assertEqual([state.item.id for state in ranked], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
