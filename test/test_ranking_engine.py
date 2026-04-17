from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / "infer"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from rank_jobs_app.core.ranking import PHASE_EXPLOIT, PHASE_EXPLORE, PHASE_TOP_K, RankingConfig, RankingItem
from rank_jobs_app.services.ranking_engine import RankingEngine, _ItemState


class RankingEngineTests(unittest.TestCase):
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
                top_k=2,
                budget_multiplier=2.5,
                neighbor_window=2,
                max_pair_repeats=3,
            )
        )
        result = engine.run(items=items, compare_batch=compare_batch, batch_size=2)

        self.assertEqual(set(result.phase_comparisons), {PHASE_EXPLORE, PHASE_EXPLOIT, PHASE_TOP_K})
        self.assertLessEqual(result.comparisons_done, result.comparisons_total)
        self.assertEqual({result.items[0].item.id, result.items[1].item.id}, {"a", "b"})

    def test_top_k_refine_prefers_boundary_local_pairs(self) -> None:
        items = [
            RankingItem(id="a", wav_path="a.wav"),
            RankingItem(id="b", wav_path="b.wav"),
            RankingItem(id="c", wav_path="c.wav"),
            RankingItem(id="d", wav_path="d.wav"),
            RankingItem(id="e", wav_path="e.wav"),
            RankingItem(id="f", wav_path="f.wav"),
        ]
        engine = RankingEngine(RankingConfig(top_k=3, neighbor_window=2, max_pair_repeats=3))
        states = {
            item.id: _ItemState(item=item, rating=1600 - idx * 25)
            for idx, item in enumerate(items)
        }

        pairs = engine._select_top_k_pairs(states=states, pair_stats={}, limit=3)

        self.assertEqual(pairs[0], ("b", "c"))
        self.assertNotIn(("a", "c"), pairs[:2])


if __name__ == "__main__":
    unittest.main()
