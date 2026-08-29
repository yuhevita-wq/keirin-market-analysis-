from __future__ import annotations

import json
from pathlib import Path

from keirin_market_analysis import simulate_market_scenario_portfolio_2023_v1 as base

YEAR = 2024


def main():
    # Frozen-forward test: reuse the exact 2023 v1 implementation.
    # Only the year and output paths change. No strategy parameter or selection rule changes.
    base.YEAR = YEAR
    base.DATA = Path(f"data/{YEAR}/s_class_yosen")
    base.OUT = Path("data/audits/market_scenario_portfolio_2024_v1.json")
    base.DETAIL = Path("data/audits/market_scenario_portfolio_2024_v1_decisions.csv")

    base.main()

    # Correct metadata emitted by the reused 2023 implementation.
    summary = json.loads(base.OUT.read_text(encoding="utf-8"))
    summary["status"] = "MARKET_SCENARIO_PORTFOLIO_2024_V1_FROZEN_FROM_2023_HOBBY_SIMULATION"
    summary["year"] = YEAR
    summary["years_read"] = [YEAR]
    summary["strategy_source_year"] = 2023
    summary["evaluation_year_2024_used"] = True
    summary["evaluation_year_2025_used"] = False
    summary["evaluation_year_2026_used"] = False
    summary["important_limit"] = (
        "2024 frozen-forward hobby simulation using final odds. Strategy v1 was defined from 2023 exploratory work "
        "and applied to 2024 without any rule/parameter changes. This is stronger than same-year 2023 development evidence, "
        "but final odds mean it is still not T-10 executable performance."
    )
    summary["frozen_rule_verification"] = (
        "The exact 2023 v1 implementation was imported and reused; only YEAR, DATA, OUT, DETAIL and output metadata changed."
    )
    base.OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
