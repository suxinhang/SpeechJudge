from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_ELO_RATING = 1500.0

PHASE_EXPLORE = "explore"
PHASE_EXPLOIT = "exploit"
PHASE_CHALLENGE = "challenge_refine"
PHASE_TOP_K = "top_k_refine"


@dataclass(frozen=True)
class RankingItem:
    id: str
    wav_path: str
    label: str | None = None
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class RankingConfig:
    top_k: int = 20
    budget_multiplier: float = 2.0
    min_budget_per_item: int = 8
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.5
    challenge_ratio: float = 0.15
    refinement_ratio: float = 0.15
    neighbor_window: int = 4
    top_k_margin: int = 6
    max_pair_repeats: int = 3
    base_k_factor: float = 24.0
    min_repeat_uncertainty: float = 0.45

    def normalized(self) -> "RankingConfig":
        total = (
            self.exploration_ratio
            + self.exploitation_ratio
            + self.challenge_ratio
            + self.refinement_ratio
        )
        if total <= 0:
            return RankingConfig(
                top_k=self.top_k,
                budget_multiplier=self.budget_multiplier,
                min_budget_per_item=self.min_budget_per_item,
                exploration_ratio=0.2,
                exploitation_ratio=0.5,
                challenge_ratio=0.15,
                refinement_ratio=0.15,
                neighbor_window=self.neighbor_window,
                top_k_margin=self.top_k_margin,
                max_pair_repeats=self.max_pair_repeats,
                base_k_factor=self.base_k_factor,
                min_repeat_uncertainty=self.min_repeat_uncertainty,
            )
        return RankingConfig(
            top_k=max(1, self.top_k),
            budget_multiplier=max(1.0, self.budget_multiplier),
            min_budget_per_item=max(1, self.min_budget_per_item),
            exploration_ratio=self.exploration_ratio / total,
            exploitation_ratio=self.exploitation_ratio / total,
            challenge_ratio=self.challenge_ratio / total,
            refinement_ratio=self.refinement_ratio / total,
            neighbor_window=max(1, self.neighbor_window),
            top_k_margin=max(1, self.top_k_margin),
            max_pair_repeats=max(1, self.max_pair_repeats),
            base_k_factor=max(1.0, self.base_k_factor),
            min_repeat_uncertainty=min(max(self.min_repeat_uncertainty, 0.0), 1.0),
        )


@dataclass(frozen=True)
class RankingProgress:
    phase: str
    comparisons_done: int
    comparisons_total: int
    phase_comparisons: dict[str, int]


@dataclass(frozen=True)
class RankedItemResult:
    item: RankingItem
    rating: float
    comparisons: int
    wins: float
    ties: int


@dataclass(frozen=True)
class RankingResult:
    items: list[RankedItemResult]
    comparisons_done: int
    comparisons_total: int
    phase_comparisons: dict[str, int]


def max_comparison_budget(n_items: int, config: RankingConfig) -> int:
    if n_items <= 1:
        return 0
    normalized = config.normalized()
    unique_pairs = n_items * (n_items - 1) // 2
    return unique_pairs * normalized.max_pair_repeats


def estimate_total_budget(n_items: int, config: RankingConfig) -> int:
    if n_items <= 1:
        return 0
    normalized = config.normalized()
    levels = max(1, math.ceil(math.log2(n_items)))
    baseline = n_items * levels - (1 << levels) + 1
    estimated = max(
        int(math.ceil(baseline * normalized.budget_multiplier)),
        n_items * normalized.min_budget_per_item,
    )
    return min(estimated, max_comparison_budget(n_items, normalized))


def phase_budgets(n_items: int, config: RankingConfig) -> dict[str, int]:
    total_budget = estimate_total_budget(n_items, config)
    normalized = config.normalized()
    explore = int(total_budget * normalized.exploration_ratio)
    exploit = int(total_budget * normalized.exploitation_ratio)
    challenge = int(total_budget * normalized.challenge_ratio)
    top_k = max(0, total_budget - explore - exploit - challenge)
    return {
        PHASE_EXPLORE: max(0, explore),
        PHASE_EXPLOIT: max(0, exploit),
        PHASE_CHALLENGE: max(0, challenge),
        PHASE_TOP_K: max(0, top_k),
    }
