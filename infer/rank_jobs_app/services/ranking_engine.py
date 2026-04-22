from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.ranking import (
    ALGORITHM_FULL_PAIRWISE,
    FULL_PAIRWISE_AGG_BRADLEY_TERRY,
    FULL_PAIRWISE_AGG_RANK_CENTRALITY_BT,
    PHASE_FULL,
    PHASE_CHALLENGE,
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
        phase_order = tuple(phase_limits)
        last_phase = phase_order[0] if phase_order else PHASE_FULL

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

        for phase in phase_order:
            last_phase = phase
            emit_progress(phase)
            while phase_counts[phase] < phase_limits[phase]:
                need = min(batch_size, phase_limits[phase] - phase_counts[phase])
                if phase in {PHASE_FULL, PHASE_EXPLORE}:
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
                elif phase == PHASE_CHALLENGE:
                    pair_ids = self._select_challenge_pairs(
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

        rating_override: dict[str, float] | None = None
        item_aux: dict[str, dict[str, float]] | None = None
        if self.config.algorithm == ALGORITHM_FULL_PAIRWISE:
            if self.config.full_pairwise_aggregation == FULL_PAIRWISE_AGG_BRADLEY_TERRY:
                ranked, rating_override = self._rank_bradley_terry(states, pair_stats)
            elif self.config.full_pairwise_aggregation == FULL_PAIRWISE_AGG_RANK_CENTRALITY_BT:
                ranked, rating_override, item_aux = self._rank_rank_centrality_bt(states, pair_stats)
            else:
                ranked = self._rank_full_pairwise(states, pair_stats)
        else:
            ranked = self._sorted_states(states)
        comparisons_done = sum(phase_counts.values())
        comparisons_total = estimated_total_budget if comparisons_done >= estimated_total_budget else comparisons_done
        emit_progress(last_phase, total_budget=comparisons_total)
        return RankingResult(
            items=[
                RankedItemResult(
                    item=state.item,
                    rating=(
                        float(rating_override[state.item.id])
                        if rating_override is not None
                        else state.rating
                    ),
                    comparisons=state.comparisons,
                    wins=state.wins,
                    ties=state.ties,
                )
                for state in ranked
            ],
            comparisons_done=comparisons_done,
            comparisons_total=comparisons_total,
            phase_comparisons=dict(phase_counts),
            item_aux=item_aux,
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
                confidence_gap = self._pair_confidence_gap(states[left_id], states[right_id])
                score = (
                    0 if total == 0 else 1,
                    confidence_gap,
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
        focus_count = self._focus_count(ranked_ids)
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
                confidence_gap = self._pair_confidence_gap(states[left_id], states[right_id])
                left_boundary_distance = abs(left_idx - boundary_idx)
                right_boundary_distance = abs(right_idx - boundary_idx)
                boundary_distance = max(left_boundary_distance, right_boundary_distance)
                span = right_idx - left_idx
                score = (
                    boundary_distance,
                    confidence_gap,
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

    def _select_challenge_pairs(
        self,
        *,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
        limit: int,
    ) -> list[tuple[str, str]]:
        ranked_ids = [state.item.id for state in self._sorted_states(states)]
        if not ranked_ids:
            return []

        boundary_start = max(0, self.config.top_k - self.config.top_k_margin)
        boundary_end = min(
            len(ranked_ids),
            self.config.top_k + max(self.config.top_k_margin, self.config.neighbor_window),
        )
        defenders = ranked_ids[boundary_start:boundary_end]
        challengers = self._challenge_pool(ranked_ids, boundary_end, states)
        if not defenders or not challengers:
            return []

        boundary_idx = min(max(self.config.top_k - 1, 0), len(ranked_ids) - 1)
        rank_index = {item_id: idx for idx, item_id in enumerate(ranked_ids)}
        candidates: list[tuple[tuple[float, ...], tuple[str, str]]] = []
        for challenger_id in challengers:
            challenger_rank = rank_index[challenger_id]
            for defender_idx, defender_id in enumerate(defenders):
                stats = self._pair_stats_for(pair_stats, challenger_id, defender_id)
                if not self._should_schedule(stats):
                    continue
                total = 0 if stats is None else stats.total
                uncertainty = 1.0 if stats is None else stats.uncertainty()
                diff = abs(states[challenger_id].rating - states[defender_id].rating)
                confidence_gap = self._promotion_gap(states[challenger_id], states[defender_id])
                defender_rank = boundary_start + defender_idx
                defender_distance = abs(defender_rank - boundary_idx)
                challenger_distance = challenger_rank - boundary_idx
                score = (
                    confidence_gap,
                    challenger_distance,
                    defender_distance,
                    0 if total == 0 else 1,
                    total,
                    diff,
                    -(uncertainty),
                    -self._optimistic_rating(states[challenger_id]),
                )
                candidates.append((score, (challenger_id, defender_id)))
        return self._take_candidate_pairs(
            candidates=candidates,
            limit=limit,
            states=states,
            pair_stats=pair_stats,
            focus_ids=defenders + challengers,
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
                confidence_gap = self._pair_confidence_gap(states[left_id], states[right_id])
                score = (
                    0 if total == 0 else 1,
                    total,
                    confidence_gap,
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

    def _rank_full_pairwise(
        self,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> list[_ItemState]:
        points = {item_id: state.wins for item_id, state in states.items()}
        sonneborn = {item_id: 0.0 for item_id in states}
        for (left_id, right_id), stats in pair_stats.items():
            left_score, right_score = self._pair_scores_from_stats(left_id, right_id, stats)
            sonneborn[left_id] += left_score * points[right_id]
            sonneborn[right_id] += right_score * points[left_id]

        grouped: dict[float, list[_ItemState]] = {}
        for state in states.values():
            grouped.setdefault(points[state.item.id], []).append(state)

        ranked: list[_ItemState] = []
        for point_value in sorted(grouped.keys(), reverse=True):
            tied_group = grouped[point_value]
            if len(tied_group) == 1:
                ranked.extend(tied_group)
                continue
            tied_ids = {state.item.id for state in tied_group}
            head_to_head = {
                state.item.id: self._group_points(state.item.id, tied_ids, pair_stats) for state in tied_group
            }
            ranked.extend(
                sorted(
                    tied_group,
                    key=lambda state: (
                        -head_to_head[state.item.id],
                        -sonneborn[state.item.id],
                        -state.rating,
                        state.item.id,
                    ),
                )
            )
        return ranked

    def _rank_bradley_terry(
        self,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> tuple[list[_ItemState], dict[str, float]]:
        """Order items by Bradley–Terry MLE strengths (Hunter MM). Ties use wins, then id."""
        item_ids = sorted({state.item.id for state in states.values()})
        gamma = self._bradley_terry_mm(item_ids, pair_stats)
        if not item_ids:
            return [], {}
        log_vals = {i: math.log(max(gamma[i], 1e-300)) for i in item_ids}
        mean_log = sum(log_vals.values()) / len(item_ids)
        rating_override = {
            i: DEFAULT_ELO_RATING + 400.0 * (log_vals[i] - mean_log) for i in item_ids
        }
        ranked_states = sorted(
            states.values(),
            key=lambda state: (
                -gamma[state.item.id],
                -state.wins,
                state.comparisons,
                state.item.id,
            ),
        )
        return ranked_states, rating_override

    def _rank_rank_centrality_bt(
        self,
        states: dict[str, _ItemState],
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> tuple[list[_ItemState], dict[str, float], dict[str, dict[str, float]]]:
        """Rank Centrality (Markov stationary) primary order; BT strengths refine ties."""
        item_ids = sorted({state.item.id for state in states.values()})
        if not item_ids:
            return [], {}, {}
        W = self._directed_wins_matrix(item_ids, pair_stats)
        rc = self._rank_centrality_stationary(item_ids, W)
        gamma = self._bradley_terry_mm(item_ids, pair_stats)
        log_vals = {i: math.log(max(gamma[i], 1e-300)) for i in item_ids}
        mean_log = sum(log_vals.values()) / len(item_ids)
        rating_override = {
            i: DEFAULT_ELO_RATING + 400.0 * (log_vals[i] - mean_log) for i in item_ids
        }
        ranked_states = sorted(
            states.values(),
            key=lambda state: (
                -rc[state.item.id],
                -gamma[state.item.id],
                -state.wins,
                state.comparisons,
                state.item.id,
            ),
        )
        item_aux = {
            i: {"rank_centrality": float(rc[i]), "bt_rating": float(rating_override[i])}
            for i in item_ids
        }
        return ranked_states, rating_override, item_aux

    @staticmethod
    def _directed_wins_matrix(
        item_ids: list[str],
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> list[list[float]]:
        n = len(item_ids)
        idx = {sid: k for k, sid in enumerate(item_ids)}
        wmat = [[0.0] * n for _ in range(n)]
        for key, stats in pair_stats.items():
            lid, rid = key[0], key[1]
            if lid not in idx or rid not in idx:
                continue
            i, j = idx[lid], idx[rid]
            t = float(stats.ties)
            wmat[i][j] += float(stats.left_wins) + 0.5 * t
            wmat[j][i] += float(stats.right_wins) + 0.5 * t
        return wmat

    @staticmethod
    def _rank_centrality_stationary(
        item_ids: list[str],
        wmat: list[list[float]],
    ) -> dict[str, float]:
        """Negahban–Shah–Wainwright-style Rank Centrality (row-stochastic walk, power method)."""
        n = len(item_ids)
        if n == 0:
            return {}
        if n == 1:
            return {item_ids[0]: 1.0}
        a_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                tot = wmat[i][j] + wmat[j][i]
                if tot > 0.0:
                    a_mat[i][j] = wmat[j][i] / tot
        degrees = [sum(a_mat[i][j] for j in range(n) if j != i) for i in range(n)]
        d_max = max(degrees)
        if d_max <= 0.0:
            return {item_ids[i]: 1.0 / n for i in range(n)}
        p_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            off = sum(a_mat[i][j] for j in range(n) if j != i)
            for j in range(n):
                if i == j:
                    p_mat[i][j] = 1.0 - off / d_max
                else:
                    p_mat[i][j] = a_mat[i][j] / d_max
        pi = [1.0 / n] * n
        for _ in range(100_000):
            new_pi = [0.0] * n
            for j in range(n):
                acc = 0.0
                for i in range(n):
                    acc += pi[i] * p_mat[i][j]
                new_pi[j] = acc
            ssum = sum(new_pi)
            if ssum <= 0.0:
                break
            new_pi = [x / ssum for x in new_pi]
            delta = max(abs(new_pi[i] - pi[i]) for i in range(n))
            pi = new_pi
            if delta < 1e-15:
                break
        return {item_ids[i]: pi[i] for i in range(n)}

    @staticmethod
    def _bradley_terry_mm(
        item_ids: list[str],
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> dict[str, float]:
        """Hunter (2004) MM iterations for Bradley–Terry strengths; ties split 0.5–0.5."""
        ids = list(item_ids)
        n = len(ids)
        if n == 0:
            return {}
        if n == 1:
            return {ids[0]: 1.0}

        id_set = set(ids)
        wins: dict[str, float] = {i: 0.0 for i in ids}
        games: dict[tuple[str, str], float] = {}
        for key, stats in pair_stats.items():
            lid, rid = key[0], key[1]
            if lid not in id_set or rid not in id_set:
                continue
            lw = float(stats.left_wins)
            rw = float(stats.right_wins)
            t = float(stats.ties)
            wins[lid] += lw + 0.5 * t
            wins[rid] += rw + 0.5 * t
            games[key] = lw + rw + t

        gamma = {i: 1.0 for i in ids}
        for _ in range(10_000):
            new_gamma: dict[str, float] = {}
            for i in ids:
                denom = 0.0
                for j in ids:
                    if j == i:
                        continue
                    key = (i, j) if i < j else (j, i)
                    tij = games.get(key, 0.0)
                    if tij <= 0.0:
                        continue
                    denom += tij / (gamma[i] + gamma[j])
                if denom <= 0.0:
                    new_gamma[i] = gamma[i]
                else:
                    new_gamma[i] = wins[i] / denom
            log_scale = sum(math.log(max(new_gamma[i], 1e-300)) for i in ids) / n
            scale = math.exp(-log_scale)
            for i in ids:
                new_gamma[i] = max(new_gamma[i] * scale, 1e-300)

            max_rel = max(
                abs(new_gamma[i] - gamma[i]) / max(gamma[i], 1e-300) for i in ids
            )
            gamma = new_gamma
            if max_rel < 1e-12:
                break

        return gamma

    def _group_points(
        self,
        item_id: str,
        group_ids: set[str],
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> float:
        total = 0.0
        for other_id in group_ids:
            if other_id == item_id:
                continue
            left_score, right_score = self._pair_scores(item_id, other_id, pair_stats)
            total += left_score
        return total

    def _pair_scores(
        self,
        left_id: str,
        right_id: str,
        pair_stats: dict[tuple[str, str], _PairStats],
    ) -> tuple[float, float]:
        stats = self._pair_stats_for(pair_stats, left_id, right_id)
        if stats is None:
            return (0.0, 0.0)
        return self._pair_scores_from_stats(left_id, right_id, stats)

    @staticmethod
    def _pair_scores_from_stats(left_id: str, right_id: str, stats: _PairStats) -> tuple[float, float]:
        if stats.left_id == left_id and stats.right_id == right_id:
            left_wins = stats.left_wins
            right_wins = stats.right_wins
        else:
            left_wins = stats.right_wins
            right_wins = stats.left_wins
        if left_wins > right_wins:
            return (1.0, 0.0)
        if left_wins < right_wins:
            return (0.0, 1.0)
        return (0.5, 0.5)

    @staticmethod
    def _sorted_states(states: dict[str, _ItemState]) -> list[_ItemState]:
        return sorted(
            states.values(),
            key=lambda state: (-state.rating, -state.wins, state.comparisons, state.item.id),
        )

    def _focus_count(self, ranked_ids: list[str]) -> int:
        return min(
            len(ranked_ids),
            max(self.config.top_k * 2, self.config.top_k + self.config.top_k_margin),
        )

    def _challenge_pool(
        self,
        ranked_ids: list[str],
        boundary_end: int,
        states: dict[str, _ItemState],
    ) -> list[str]:
        if boundary_end >= len(ranked_ids):
            return []
        pool = ranked_ids[boundary_end:]
        challenger_cap = min(
            len(pool),
            max(self.config.top_k * 3, self.config.top_k + self.config.top_k_margin * 4),
        )
        ranked_pool = sorted(
            pool,
            key=lambda item_id: (
                -self._optimistic_rating(states[item_id]),
                states[item_id].comparisons,
                -states[item_id].wins,
                -states[item_id].rating,
                item_id,
            ),
        )
        return ranked_pool[:challenger_cap]

    def _confidence_radius(self, state: _ItemState) -> float:
        return (self.config.base_k_factor * 3.0) / math.sqrt(1.0 + state.comparisons)

    def _optimistic_rating(self, state: _ItemState) -> float:
        return state.rating + self._confidence_radius(state)

    def _pessimistic_rating(self, state: _ItemState) -> float:
        return state.rating - self._confidence_radius(state)

    def _pair_confidence_gap(self, left: _ItemState, right: _ItemState) -> float:
        upper = min(self._optimistic_rating(left), self._optimistic_rating(right))
        lower = max(self._pessimistic_rating(left), self._pessimistic_rating(right))
        return max(lower - upper, 0.0)

    def _promotion_gap(self, challenger: _ItemState, defender: _ItemState) -> float:
        return max(self._pessimistic_rating(defender) - self._optimistic_rating(challenger), 0.0)

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
