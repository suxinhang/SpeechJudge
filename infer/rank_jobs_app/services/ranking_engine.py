from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.ranking import (
    DEFAULT_ELO_RATING,
    PHASE_EXPLOIT,
    PHASE_EXPLORE,
    PHASE_TOP_K,
    RankedItemResult,
    RankingConfig,
    RankingItem,
    RankingProgress,
    RankingResult,
    phase_budgets,
)

BatchComparator = Callable[[list[tuple[RankingItem, RankingItem]]], list[int]]
ProgressCallback = Callable[[RankingProgress], None]


@dataclass
class _ItemState:
    item: RankingItem
    rating: float = DEFAULT_ELO_RATING
    comparisons: int = 0
    wins: float = 0.0
    ties: int = 0


@dataclass
class _PairStats:
    left_id: str
    right_id: str
    total: int = 0
    left_wins: int = 0
    right_wins: int = 0
    ties: int = 0

    def note(self, canonical_outcome: int) -> None:
        self.total += 1
        if canonical_outcome > 0:
            self.left_wins += 1
        elif canonical_outcome < 0:
            self.right_wins += 1
        else:
            self.ties += 1

    def uncertainty(self) -> float:
        if self.total <= 0:
            return 1.0
        decisive = self.left_wins + self.right_wins
        if decisive <= 0:
            balance = 1.0
        else:
            balance = 1.0 - abs(self.left_wins - self.right_wins) / decisive
        tie_ratio = self.ties / self.total
        sample_penalty = 1.0 / (self.total + 1)
        return min(1.0, 0.55 * balance + 0.25 * tie_ratio + 0.20 * sample_penalty)


class RankingEngine:
    def __init__(self, config: RankingConfig) -> None:
        self.config = config.normalized()

    def run(
        self,
        *,
        items: Sequence[RankingItem],
        compare_batch: BatchComparator,
        batch_size: int,
        progress_callback: ProgressCallback | None = None,
    ) -> RankingResult:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        states = {item.id: _ItemState(item=item) for item in items}
        phase_limits = phase_budgets(len(items), self.config)
        phase_counts = {phase: 0 for phase in phase_limits}
        pair_stats: dict[tuple[str, str], _PairStats] = {}
        estimated_total_budget = sum(phase_limits.values())
        explore_schedule = self._round_robin_pairs([item.id for item in items])
        explore_cursor = 0
        last_phase = PHASE_EXPLORE

        def emit_progress(phase: str, *, total_budget: int = estimated_total_budget) -> None:
            if progress_callback is None:
                return
            progress_callback(
                RankingProgress(
                    phase=phase,
                    comparisons_done=sum(phase_counts.values()),
                    comparisons_total=total_budget,
                    phase_comparisons=dict(phase_counts),
                )
            )

        for phase in (PHASE_EXPLORE, PHASE_EXPLOIT, PHASE_TOP_K):
            last_phase = phase
            emit_progress(phase)
            while phase_counts[phase] < phase_limits[phase]:
                need = min(batch_size, phase_limits[phase] - phase_counts[phase])
                if phase == PHASE_EXPLORE:
                    pair_ids, explore_cursor = self._select_explore_pairs(
                        states=states,
                        pair_stats=pair_stats,
                        limit=need,
                        schedule=explore_schedule,
                        cursor=explore_cursor,
                    )
                elif phase == PHASE_EXPLOIT:
                    pair_ids = self._select_exploit_pairs(
                        states=states,
                        pair_stats=pair_stats,
                        limit=need,
                    )
                else:
                    pair_ids = self._select_top_k_pairs(
                        states=states,
                        pair_stats=pair_stats,
                        limit=need,
                    )
                if not pair_ids:
                    break

                outcomes = compare_batch([(states[a].item, states[b].item) for a, b in pair_ids])
                if len(outcomes) != len(pair_ids):
                    raise RuntimeError("compare_batch returned unexpected number of results")

                for (left_id, right_id), raw_outcome in zip(pair_ids, outcomes):
                    outcome = self._normalize_outcome(raw_outcome)
                    self._record_outcome(
                        states=states,
                        pair_stats=pair_stats,
                        left_id=left_id,
                        right_id=right_id,
                        outcome=outcome,
                    )
                    phase_counts[phase] += 1
                emit_progress(phase)

        ranked = sorted(
            states.values(),
            key=lambda state: (-state.rating, -state.wins, state.comparisons, state.item.id),
        )
        comparisons_done = sum(phase_counts.values())
        comparisons_total = estimated_total_budget if comparisons_done >= estimated_total_budget else comparisons_done
        emit_progress(last_phase, total_budget=comparisons_total)
        return RankingResult(
            items=[
                RankedItemResult(
                    item=state.item,
                    rating=state.rating,
                    comparisons=state.comparisons,
                    wins=state.wins,
                    ties=state.ties,
                )
                for state in ranked
            ],
            comparisons_done=comparisons_done,
            comparisons_total=comparisons_total,
            phase_comparisons=dict(phase_counts),
        )

    @staticmethod
    def _normalize_outcome(outcome: int) -> int:
        if outcome > 0:
            return 1
        if outcome < 0:
            return -1
        return 0

    @staticmethod
    def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
        return (left_id, right_id) if left_id < right_id else (right_id, left_id)

    def _pair_stats_for(
        self,
        pair_stats: dict[tuple[str, str], _PairStats],
        left_id: str,
        right_id: str,
    ) -> _PairStats | None:
        return pair_stats.get(self._pair_key(left_id, right_id))

    def _should_schedule(self, stats: _PairStats | None) -> bool:
        if stats is None or stats.total <= 0:
            return True
        if stats.total >= self.config.max_pair_repeats:
            return False
        return stats.uncertainty() >= self.config.min_repeat_uncertainty

    def _record_outcome(
        self,
        *,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        left_id: str,
        right_id: str,
        outcome: int,
    ) -> None:
        key = self._pair_key(left_id, right_id)
        record = pair_stats.get(key)
        if record is None:
            record = _PairStats(left_id=key[0], right_id=key[1])
            pair_stats[key] = record
        canonical_outcome = outcome if key == (left_id, right_id) else -outcome
        record.note(canonical_outcome)
        self._apply_elo(states[left_id], states[right_id], outcome)

    def _apply_elo(self, left: _ItemState, right: _ItemState, outcome: int) -> None:
        expected_left = 1.0 / (1.0 + math.pow(10.0, (right.rating - left.rating) / 400.0))
        actual_left = 1.0 if outcome > 0 else 0.0 if outcome < 0 else 0.5
        exposure = min(left.comparisons, right.comparisons)
        k_factor = self.config.base_k_factor / math.sqrt(1.0 + exposure)
        delta = k_factor * (actual_left - expected_left)
        left.rating += delta
        right.rating -= delta
        left.comparisons += 1
        right.comparisons += 1
        if outcome > 0:
            left.wins += 1.0
        elif outcome < 0:
            right.wins += 1.0
        else:
            left.wins += 0.5
            right.wins += 0.5
            left.ties += 1
            right.ties += 1

    def _select_explore_pairs(
        self,
        *,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        limit: int,
        schedule: list[tuple[str, str]],
        cursor: int,
    ) -> tuple[list[tuple[str, str]], int]:
        selected: list[tuple[str, str]] = []
        chosen: set[tuple[str, str]] = set()
        while cursor < len(schedule) and len(selected) < limit:
            pair = schedule[cursor]
            cursor += 1
            key = self._pair_key(*pair)
            if key in chosen:
                continue
            if self._pair_stats_for(pair_stats, *pair) is not None:
                continue
            selected.append(pair)
            chosen.add(key)
        if len(selected) < limit:
            selected.extend(
                self._select_fallback_pairs(
                    states=states,
                    pair_stats=pair_stats,
                    limit=limit - len(selected),
                    focus_ids=None,
                    chosen=chosen,
                )
            )
        return selected, cursor

    def _select_exploit_pairs(
        self,
        *,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        limit: int,
    ) -> list[tuple[str, str]]:
        ranked_ids = [state.item.id for state in self._sorted_states(states)]
        window = min(len(ranked_ids) - 1, self.config.neighbor_window * 2) if ranked_ids else 0
        candidates: list[tuple[tuple[float, ...], tuple[str, str]]] = []
        for idx, left_id in enumerate(ranked_ids):
            for offset in range(1, window + 1):
                right_idx = idx + offset
                if right_idx >= len(ranked_ids):
                    break
                right_id = ranked_ids[right_idx]
                stats = self._pair_stats_for(pair_stats, left_id, right_id)
                if not self._should_schedule(stats):
                    continue
                total = 0 if stats is None else stats.total
                uncertainty = 1.0 if stats is None else stats.uncertainty()
                diff = abs(states[left_id].rating - states[right_id].rating)
                score = (
                    0 if total == 0 else 1,
                    diff,
                    total,
                    -(uncertainty),
                    states[left_id].comparisons + states[right_id].comparisons,
                )
                candidates.append((score, (left_id, right_id)))
        return self._take_candidate_pairs(
            candidates=candidates,
            limit=limit,
            states=states,
            pair_stats=pair_stats,
            focus_ids=None,
        )

    def _select_top_k_pairs(
        self,
        *,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        limit: int,
    ) -> list[tuple[str, str]]:
        ranked_states = self._sorted_states(states)
        ranked_ids = [state.item.id for state in ranked_states]
        focus_count = min(
            len(ranked_ids),
            max(self.config.top_k * 2, self.config.top_k + self.config.top_k_margin),
        )
        focus_ids = ranked_ids[:focus_count]
        boundary_idx = min(max(self.config.top_k - 1, 0), max(focus_count - 1, 0))
        candidates: list[tuple[tuple[float, ...], tuple[str, str]]] = []
        for left_idx, left_id in enumerate(focus_ids):
            for right_idx in range(left_idx + 1, len(focus_ids)):
                right_id = focus_ids[right_idx]
                stats = self._pair_stats_for(pair_stats, left_id, right_id)
                if not self._should_schedule(stats):
                    continue
                total = 0 if stats is None else stats.total
                uncertainty = 1.0 if stats is None else stats.uncertainty()
                diff = abs(states[left_id].rating - states[right_id].rating)
                left_boundary_distance = abs(left_idx - boundary_idx)
                right_boundary_distance = abs(right_idx - boundary_idx)
                boundary_distance = max(left_boundary_distance, right_boundary_distance)
                span = right_idx - left_idx
                score = (
                    boundary_distance,
                    span,
                    0 if total == 0 else 1,
                    total,
                    diff,
                    -(uncertainty),
                )
                candidates.append((score, (left_id, right_id)))
        return self._take_candidate_pairs(
            candidates=candidates,
            limit=limit,
            states=states,
            pair_stats=pair_stats,
            focus_ids=focus_ids,
        )

    def _take_candidate_pairs(
        self,
        *,
        candidates: list[tuple[tuple[float, ...], tuple[str, str]]],
        limit: int,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        focus_ids: list[str] | None,
    ) -> list[tuple[str, str]]:
        selected: list[tuple[str, str]] = []
        chosen: set[tuple[str, str]] = set()
        for _score, pair in sorted(candidates, key=lambda entry: entry[0]):
            key = self._pair_key(*pair)
            if key in chosen:
                continue
            selected.append(pair)
            chosen.add(key)
            if len(selected) >= limit:
                return selected
        if len(selected) < limit:
            selected.extend(
                self._select_fallback_pairs(
                    states=states,
                    pair_stats=pair_stats,
                    limit=limit - len(selected),
                    focus_ids=focus_ids,
                    chosen=chosen,
                )
            )
        return selected

    def _select_fallback_pairs(
        self,
        *,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        limit: int,
        focus_ids: list[str] | None,
        chosen: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        if limit <= 0:
            return []
        ranked_ids = [state.item.id for state in self._sorted_states(states)]
        pool = focus_ids if focus_ids else ranked_ids
        candidates: list[tuple[tuple[float, ...], tuple[str, str]]] = []
        for left_idx, left_id in enumerate(pool):
            for right_id in pool[left_idx + 1 :]:
                key = self._pair_key(left_id, right_id)
                if key in chosen:
                    continue
                stats = pair_stats.get(key)
                if not self._should_schedule(stats):
                    continue
                total = 0 if stats is None else stats.total
                uncertainty = 1.0 if stats is None else stats.uncertainty()
                diff = abs(states[left_id].rating - states[right_id].rating)
                score = (
                    0 if total == 0 else 1,
                    total,
                    diff,
                    -(uncertainty),
                )
                candidates.append((score, (left_id, right_id)))
        out: list[tuple[str, str]] = []
        for _score, pair in sorted(candidates, key=lambda entry: entry[0]):
            key = self._pair_key(*pair)
            if key in chosen:
                continue
            out.append(pair)
            chosen.add(key)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _sorted_states(states: dict[str, _ItemState]) -> list[_ItemState]:
        return sorted(
            states.values(),
            key=lambda state: (-state.rating, -state.wins, state.comparisons, state.item.id),
        )

    @staticmethod
    def _round_robin_pairs(item_ids: list[str]) -> list[tuple[str, str]]:
        players = list(item_ids)
        if len(players) <= 1:
            return []
        if len(players) % 2 == 1:
            players.append("__bye__")
        rounds = len(players) - 1
        half = len(players) // 2
        order: list[tuple[str, str]] = []
        current = list(players)
        for _ in range(rounds):
            for idx in range(half):
                left_id = current[idx]
                right_id = current[-(idx + 1)]
                if "__bye__" not in {left_id, right_id}:
                    order.append((left_id, right_id))
            current = [current[0], current[-1], *current[1:-1]]
        return order
