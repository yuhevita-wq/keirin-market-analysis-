from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import isfinite
from typing import Iterable


TRIO_KEYS = tuple("-".join(map(str, c)) for c in combinations(range(1, 8), 3))
TRIFECTA_KEYS = tuple("-".join(map(str, p)) for p in permutations(range(1, 8), 3))


@dataclass(frozen=True)
class MarketRow:
    combination: str
    odds: float
    captured_at: str
    source: str


@dataclass(frozen=True)
class DRow:
    combination: str
    trio_odds: float
    trio_share: float
    trifecta_implied_sum: float
    trifecta_share: float
    d: float
    d_rank_desc: int


def _validate_positive_odds(rows: Iterable[MarketRow], market_name: str) -> None:
    for row in rows:
        if not isfinite(row.odds) or row.odds <= 0:
            raise ValueError(f"{market_name}: invalid odds for {row.combination}: {row.odds}")
        if not row.captured_at:
            raise ValueError(f"{market_name}: missing captured_at for {row.combination}")
        if not row.source:
            raise ValueError(f"{market_name}: missing source for {row.combination}")


def validate_complete_market(
    trio_rows: list[MarketRow],
    trifecta_rows: list[MarketRow],
) -> None:
    _validate_positive_odds(trio_rows, "trio")
    _validate_positive_odds(trifecta_rows, "trifecta")

    trio_map = {r.combination: r for r in trio_rows}
    trifecta_map = {r.combination: r for r in trifecta_rows}

    if len(trio_rows) != len(trio_map):
        raise ValueError("trio: duplicate combinations found")
    if len(trifecta_rows) != len(trifecta_map):
        raise ValueError("trifecta: duplicate combinations found")

    missing_trio = sorted(set(TRIO_KEYS) - set(trio_map))
    extra_trio = sorted(set(trio_map) - set(TRIO_KEYS))
    missing_trifecta = sorted(set(TRIFECTA_KEYS) - set(trifecta_map))
    extra_trifecta = sorted(set(trifecta_map) - set(TRIFECTA_KEYS))

    if missing_trio or extra_trio:
        raise ValueError(
            f"trio market must contain exactly 35 combinations; "
            f"missing={missing_trio}, extra={extra_trio}"
        )
    if missing_trifecta or extra_trifecta:
        raise ValueError(
            f"trifecta market must contain exactly 210 combinations; "
            f"missing_count={len(missing_trifecta)}, extra_count={len(extra_trifecta)}"
        )


def compute_d(
    trio_rows: list[MarketRow],
    trifecta_rows: list[MarketRow],
) -> list[DRow]:
    validate_complete_market(trio_rows, trifecta_rows)

    trio_map = {r.combination: r for r in trio_rows}
    trifecta_map = {r.combination: r for r in trifecta_rows}

    trio_implied = {k: 1.0 / r.odds for k, r in trio_map.items()}
    trio_total = sum(trio_implied.values())

    trifecta_implied = {k: 1.0 / r.odds for k, r in trifecta_map.items()}
    trifecta_total = sum(trifecta_implied.values())

    raw_rows: list[tuple[str, float, float, float, float]] = []
    for trio_key in TRIO_KEYS:
        cars = tuple(map(int, trio_key.split("-")))
        six_keys = ["-".join(map(str, p)) for p in permutations(cars, 3)]
        six_sum = sum(trifecta_implied[k] for k in six_keys)

        trio_share = trio_implied[trio_key] / trio_total
        trifecta_share = six_sum / trifecta_total
        d = trifecta_share / trio_share
        raw_rows.append((trio_key, trio_map[trio_key].odds, trio_share, six_sum, trifecta_share, d))

    order = sorted(range(len(raw_rows)), key=lambda i: raw_rows[i][5], reverse=True)
    rank_by_index = {idx: rank for rank, idx in enumerate(order, start=1)}

    return [
        DRow(
            combination=row[0],
            trio_odds=row[1],
            trio_share=row[2],
            trifecta_implied_sum=row[3],
            trifecta_share=row[4],
            d=row[5],
            d_rank_desc=rank_by_index[i],
        )
        for i, row in enumerate(raw_rows)
    ]
