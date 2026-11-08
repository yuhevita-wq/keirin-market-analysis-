"""Offline tests: identical pre-race features, genuine losses and frozen rules."""
import gzip
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run import (band, learn, market_tickets, normalize_combo, race_date,
                 run, scan, stage)


def make_rid(year, race=1):
    return f"01{year}010101{race:04d}"


def make_record(year, race=1):
    cars = (1, 2, 3, 4, 5)
    tri = {"-".join(map(str, p)): (15.0 if p == (1, 2, 3) else 100.0)
           for p in itertools.permutations(cars, 3)}
    trio = {"=".join(map(str, p)): (5.0 if p == (1, 2, 3) else 20.0)
            for p in itertools.combinations(cars, 3)}
    return {"race_id": make_rid(year, race), "race_type": "F1/Ｓ級予選",
            "odds": {"3連単": tri, "3連複": trio}}


def put_gz(path, documents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def fixture(root, first=2021, last=2024, missing=None):
    for year in range(first, last + 1):
        records = [make_record(year, 1), make_record(year, 2)]
        put_gz(root / "data/odds_only" / f"odds_{year}.jsonl.gz", records)
        outcomes = [{"race_id": make_rid(year, j), "finish_order": [1, 2, 3]}
                    for j in (1, 2) if (year, j) != missing]
        put_gz(root / "data/results_only" / f"results_{year}.jsonl.gz", outcomes)


class FlatBuyTests(unittest.TestCase):
    def test_date_and_type(self):
        self.assertEqual(race_date(make_rid(2021)), __import__("datetime").date(2021, 1, 1))
        self.assertEqual(stage("Ｆ１／Ｓ級準決勝"), "準決勝")
        self.assertEqual(band(10, (3.5, 10, 20)), "<20")
        self.assertEqual(normalize_combo("3=1=2", "3連複"), (1, 2, 3))

    def test_pre_race_features_never_use_result(self):
        record = make_record(2021)
        a = list(market_tickets(record, (1, 2, 3)))
        b = list(market_tickets(record, (5, 4, 3)))
        self.assertEqual([(w, c, price, f) for w, c, price, f, _, _ in a],
                         [(w, c, price, f) for w, c, price, f, _, _ in b])
        self.assertEqual(sum(hit for *_, hit, pay in a), 2)
        self.assertEqual(len(a), 70)  # 5P3 + 5C3
        self.assertEqual(next(pay for w, c, _, _, hit, pay in a if w == "3連単" and hit), 1500)

    def test_no_buy_filter_and_missing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 2021, 2021, missing=(2021, 2))
            total, _, audit = scan(root, [2021])
            self.assertEqual(total[2021].bets, 70)
            self.assertEqual(audit["2021"]["no_verified_result"], 1)
            selected, _, a = scan(root, [2021], ("券種=3連単", "人気順位帯=1"))
            self.assertEqual(selected[2021].bets, 1)
            self.assertEqual(selected[2021].paid, 1500)
            self.assertEqual(a["2021"]["buy_races"], 1)
            excluded, _, _ = scan(root, [2021], ("架空条件=無",))
            self.assertEqual(excluded[2021].bets, 0)

    def test_frozen_training_ignores_future_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 2021, 2024)
            first = learn(root, [2021, 2022], depth=2, min_bets=1, min_hits=1,
                          min_profitable_years=2, min_gain=0)
            # Change an untouched future label. Earlier discovery must be identical.
            future = root / "data/results_only/results_2023.jsonl.gz"
            put_gz(future, [{"race_id": make_rid(2023, j), "finish_order": [5, 4, 3]}
                            for j in (1, 2)])
            second = learn(root, [2021, 2022], depth=2, min_bets=1, min_hits=1,
                           min_profitable_years=2, min_gain=0)
            self.assertEqual(first["rule"], second["rule"])
            self.assertEqual(first["training"], second["training"])

    def test_end_to_end_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            args = SimpleNamespace(first_year=2021, last_year=2023,
                                   holdout_year=2024, depth=2, min_bets=1,
                                   min_hits=1, min_profitable_years=2, min_gain=0)
            out = root / "out"
            report = run(root, out, args)
            self.assertEqual(report["discovery_window"], [2021, 2023])
            self.assertEqual(report["untouched_later_year_evaluation"]["year"], 2024)
            self.assertTrue((out / "flat_buy_profitable_conditions.csv").is_file())
            self.assertTrue((out / "walk_forward.csv").is_file())
            self.assertTrue((out / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
