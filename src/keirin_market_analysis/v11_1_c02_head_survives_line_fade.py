from __future__ import annotations

"""v11.1-C02: HEAD SURVIVES x LINE FADE.

A deliberately sharp override for the specific market/race-card conflict:

    H_CONCENTRATED | HEAD_SUPPORTED | LINE_CHALLENGED

Psychological hypothesis:
- the market is strongly concentrated on the correct head candidate;
- the race card independently supports that head candidate;
- but the market's strongest *line* is not the strongest line by race-card F;
- therefore preserve the head, fade the line-mates that the market packages with
  that head, and use the strongest rival fundamental line for the tail.

No outcome or payout input is used. No fitted numeric thresholds are introduced.
Outside this conflict branch, v8.25-F26 remains unchanged.
"""

from typing import Iterable

from v11_0_c01_market_psychology_racecard_context import build_v11_0_c01

SCHEME_VERSION = "v11.1-C02"
BASE_SCHEME_VERSION = "v8.25-F26"
TARGET_CONTEXT = "H_CONCENTRATED|HEAD_SUPPORTED|LINE_CHALLENGED"


def _find_line_index(lines: list[list[int]], car: int) -> int | None:
    for idx, line in enumerate(lines):
        if car in line:
            return idx
    return None


def build_v11_1_c02(
    trio_odds,
    trifecta_odds,
    predicted_line_formation: str,
    race_type: str,
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    d = dict(
        build_v11_0_c01(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
            race_type,
            entry_rows,
        )
    )

    d["scheme_version"] = SCHEME_VERSION
    d["base_scheme_version"] = BASE_SCHEME_VERSION
    d["c02_override_applied"] = False

    # C02 only modifies races that v8.25 already chose to buy.
    if not d.get("buy"):
        return d

    if not d.get("racecard_context_ok"):
        return d

    if d.get("racecard_context") != TARGET_CONTEXT:
        return d

    lines_raw = d.get("racecard_market_line_order")
    fundamental_order_raw = d.get("racecard_fundamental_line_order")
    f_raw = d.get("racecard_fundamental_F")
    h1 = d.get("racecard_market_h1")

    # Reconstruct line partition from the race-card component metadata.
    # Each component includes deterministic line_position/line_size but not line_id,
    # so parse the supplied formation directly here.
    from v7_0_f01_market_hierarchy import parse_lines

    parsed = parse_lines(predicted_line_formation)
    if parsed is None:
        d["buy"] = False
        d["reason"] = "C02_INVALID_LINE_STRUCTURE"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    lines = [list(line) for line in parsed]
    if not isinstance(h1, int):
        d["buy"] = False
        d["reason"] = "C02_MISSING_MARKET_H1"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    market_line_order = list(lines_raw or [])
    fundamental_line_order = list(fundamental_order_raw or [])
    if not market_line_order or not fundamental_line_order:
        d["buy"] = False
        d["reason"] = "C02_MISSING_LINE_ORDERS"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    market_top_line = int(market_line_order[0])
    fundamental_top_line = int(fundamental_line_order[0])
    h1_line = _find_line_index(lines, h1)

    # The sharp psychological interpretation only exists when the market's
    # strongest line is H1's own line, while the fundamentals prefer another line.
    if h1_line is None or market_top_line != h1_line:
        d["buy"] = False
        d["reason"] = "C02_HEAD_NOT_ON_MARKET_TOP_LINE"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    if fundamental_top_line == h1_line:
        d["buy"] = False
        d["reason"] = "C02_NO_TRUE_LINE_FADE_CONFLICT"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    rival_line = lines[fundamental_top_line]
    if len(rival_line) < 2:
        d["buy"] = False
        d["reason"] = "C02_RIVAL_LINE_TOO_SMALL"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    f = {int(k): float(v) for k, v in (f_raw or {}).items()}
    if any(car not in f for car in rival_line):
        d["buy"] = False
        d["reason"] = "C02_MISSING_RIVAL_F"
        d["tickets"] = []
        d["ticket_count"] = 0
        return d

    # Exactly two strongest riders from the strongest rival fundamental line.
    # Buy both tail orders. This keeps the hypothesis sharp without pretending F
    # can distinguish second from third place reliably.
    tail = sorted(rival_line, key=lambda car: (-f[car], car))[:2]
    a, b = tail
    tickets = [(h1, a, b), (h1, b, a)]

    d["buy"] = True
    d["reason"] = "C02_HEAD_SURVIVES_LINE_FADE"
    d["tickets"] = tickets
    d["ticket_count"] = 2
    d["formation"] = {
        "first": [h1],
        "second": [a, b],
        "third": [a, b],
        "exclude_same_rider": True,
    }
    d["branch_policy"] = "HEAD_SURVIVES_LINE_FADE_TWO_TICKET_TAIL_SWAP"
    d["c02_override_applied"] = True
    d["c02_market_head"] = h1
    d["c02_market_head_line"] = h1_line
    d["c02_fundamental_rival_line"] = fundamental_top_line
    d["c02_rival_tail"] = tail
    return d
