from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_ELO_RATING = 1500.0

ALGORITHM_FULL_PAIRWISE = "full_pairwise"
ALGORITHM_PHASED_ELO = "phased_elo"

PHASE_FULL = "full_compare"
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
    algorithm: str = ALGORITHM_FULL_PAIRWISE
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
                algorithm=self._normalized_algorithm(self.algorithm),
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
            algorithm=self._normalized_algorithm(self.algorithm),
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

    @staticmethod
    def _normalized_algorithm(value: str) -> str:
        if value == ALGORITHM_PHASED_ELO:
            return ALGORITHM_PHASED_ELO
        return ALGORITHM_FULL_PAIRWISE


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


def majority_vote(votes: list[int]) -> int:
    positive = sum(1 for vote in votes if vote > 0)
    negative = sum(1 for vote in votes if vote < 0)
    if positive > negative:
        return 1
    if negative > positive:
        return -1
    return 0


def collapse_pairwise_votes(votes: list[int], votes_per_pair: int) -> list[int]:
    if votes_per_pair <= 1:
        return list(votes)
    if len(votes) % votes_per_pair != 0:
        raise ValueError("votes length must be divisible by votes_per_pair")
    collapsed: list[int] = []
    for idx in range(0, len(votes), votes_per_pair):
        collapsed.append(majority_vote(votes[idx : idx + votes_per_pair]))
    return collapsed


def collapse_pairwise_votes_adaptive(first_two_votes: list[int], third_votes: list[int]) -> list[int]:
    if len(first_two_votes) % 2 != 0:
        raise ValueError("first_two_votes length must be divisible by 2")
    resolved: list[int] = []
    third_idx = 0
    for idx in range(0, len(first_two_votes), 2):
        vote_a = first_two_votes[idx]
        vote_b = first_two_votes[idx + 1]
        if vote_a == vote_b:
            resolved.append(vote_a)
            continue
        if third_idx >= len(third_votes):
            raise ValueError("not enough third_votes to break ties")
        resolved.append(majority_vote([vote_a, vote_b, third_votes[third_idx]]))
        third_idx += 1
    if third_idx != len(third_votes):
        raise ValueError("unused third_votes remain after resolving ties")
    return resolved


def max_comparison_budget(n_items: int, config: RankingConfig) -> int:
    if n_items <= 1:
        return 0
    normalized = config.normalized()
    unique_pairs = n_items * (n_items - 1) // 2
    if normalized.algorithm == ALGORITHM_FULL_PAIRWISE:
        return unique_pairs
    return unique_pairs * normalized.max_pair_repeats


def estimate_total_budget(n_items: int, config: RankingConfig) -> int:
    if n_items <= 1:
        return 0
    normalized = config.normalized()
    unique_pairs = n_items * (n_items - 1) // 2
    if normalized.algorithm == ALGORITHM_FULL_PAIRWISE:
        return unique_pairs
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
    if normalized.algorithm == ALGORITHM_FULL_PAIRWISE:
        return {PHASE_FULL: total_budget}
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
