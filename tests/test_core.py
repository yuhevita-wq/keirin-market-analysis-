from itertools import combinations, permutations

from keirin_market_analysis.core import MarketRow, compute_d


def make_rows():
    trio = [
        MarketRow("-".join(map(str, c)), 10.0, "2026-08-28T12:00:00+09:00", "test")
        for c in combinations(range(1, 8), 3)
    ]
    trifecta = [
        MarketRow("-".join(map(str, p)), 60.0, "2026-08-28T12:00:00+09:00", "test")
        for p in permutations(range(1, 8), 3)
    ]
    return trio, trifecta


def test_equal_markets_produce_d_one():
    trio, trifecta = make_rows()
    rows = compute_d(trio, trifecta)
    assert len(rows) == 35
    for row in rows:
        assert abs(row.d - 1.0) < 1e-12


def test_missing_trio_fails():
    trio, trifecta = make_rows()
    trio.pop()
    try:
        compute_d(trio, trifecta)
    except ValueError as e:
        assert "exactly 35" in str(e)
    else:
        raise AssertionError("missing trio should fail")
