from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / "infer"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from rank_jobs_app.services.triplet_screen_worker import _pick_winner_index, shuffle_into_groups


class TripletScreenTests(unittest.TestCase):
    def test_shuffle_deterministic(self) -> None:
        items = [{"id": f"id{i}"} for i in range(6)]
        g1 = shuffle_into_groups(items, seed=42, group_size=3)
        g2 = shuffle_into_groups(items, seed=42, group_size=3)
        self.assertEqual(len(g1), 2)
        self.assertEqual(g1, g2)

    def test_pick_winner_tiebreak_id(self) -> None:
        wins = [1.0, 1.0, 0.0]
        ids = ["z", "a", "m"]
        self.assertEqual(_pick_winner_index(wins, ids), 1)

    def test_pick_winner_unique(self) -> None:
        wins = [2.0, 0.0, 0.0]
        ids = ["a", "b", "c"]
        self.assertEqual(_pick_winner_index(wins, ids), 0)


if __name__ == "__main__":
    unittest.main()
