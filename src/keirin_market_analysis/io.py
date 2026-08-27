from __future__ import annotations

import csv
from pathlib import Path

from .core import DRow, MarketRow


REQUIRED_COLUMNS = {"combination", "odds", "captured_at", "source"}


def read_market_csv(path: str | Path) -> list[MarketRow]:
    rows: list[MarketRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            try:
                odds = float(row["odds"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}:{line_no}: invalid odds: {row.get('odds')}") from e

            rows.append(
                MarketRow(
                    combination=(row["combination"] or "").strip(),
                    odds=odds,
                    captured_at=(row["captured_at"] or "").strip(),
                    source=(row["source"] or "").strip(),
                )
            )
    return rows


def write_d_csv(path: str | Path, rows: list[DRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "combination",
                "trio_odds",
                "trio_share",
                "trifecta_implied_sum",
                "trifecta_share",
                "d",
                "d_rank_desc",
            ]
        )
        for r in sorted(rows, key=lambda x: x.d_rank_desc):
            writer.writerow(
                [
                    r.combination,
                    f"{r.trio_odds:.10g}",
                    f"{r.trio_share:.12g}",
                    f"{r.trifecta_implied_sum:.12g}",
                    f"{r.trifecta_share:.12g}",
                    f"{r.d:.12g}",
                    r.d_rank_desc,
                ]
            )
