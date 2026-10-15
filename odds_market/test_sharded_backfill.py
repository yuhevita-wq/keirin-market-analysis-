"""Offline bounds and independent-cursor tests for eight parallel collectors."""
from datetime import date, timedelta
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from odds_market import sharded_backfill as s


def run_tests():
    ranges = [s.bounds(i) for i in range(s.SHARD_COUNT)]
    assert ranges[0][0] == date(2021, 1, 3)
    assert ranges[-1][1] == date(2026, 6, 30)
    all_dates = []
    for i, (begin, end) in enumerate(ranges):
        assert begin <= end
        if i:
            assert begin == ranges[i - 1][1] + timedelta(days=1)
        all_dates.extend(begin + timedelta(days=k)
                         for k in range((end-begin).days+1))
    assert len(all_dates) == len(set(all_dates)) == 2005
    with TemporaryDirectory() as temp:
        with patch.object(s, "SHARD_DIR", Path(temp)), patch.object(
                s.core, "has_real_world_proof", return_value=True):
            for i in range(s.SHARD_COUNT):
                first, end, mode = s.choose(i, 7)
                assert first == ranges[i][0] and end <= ranges[i][1]
                assert mode == "forward"
                assert s.state_path(i) != s.state_path((i + 1) % s.SHARD_COUNT)
        with patch.object(s, "SHARD_DIR", Path(temp)), patch.object(
                s.core, "has_real_world_proof", return_value=False):
            assert all(s.choose(i, 4) is None for i in range(s.SHARD_COUNT))
    print("SHARDED BACKFILL TESTS PASS")


if __name__ == "__main__":
    run_tests()
