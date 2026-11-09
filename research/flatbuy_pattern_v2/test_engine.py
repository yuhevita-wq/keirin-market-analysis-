"""Self-contained regression tests; no actual historical data is embedded in tests."""
import gzip
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from engine import (parse_market, valid_lines, race_features, select_policies,
                    scan, run, race_day)


def rid(year, seq):
    return f"01{year}010101{seq:04d}"


def market(year, seq, grade, race_type, winner):
    tri = {}
    for combo in itertools.permutations(range(1, 6), 3):
        tri["-".join(map(str, combo))] = 5.0 if combo == (1, 2, 3) else 20.0
    trio = {}
    for combo in itertools.combinations(range(1, 6), 3):
        trio["=".join(map(str, combo))] = 5.0 if combo == (1, 2, 3) else 20.0
    return ({"race_id": rid(year, seq), "race_type": grade + "/" + race_type,
             "odds": {"3連単": tri, "3連複": trio}},
            {"race_id": rid(year, seq), "finish_order": list(winner)})


def gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def build_data(root, years):
    for year in years:
        odds, results = [], []
        seq = 1
        # For the top1 policy, only F1+予選 is profitable:
        # 3 hits/10 at 5.0 => ROI 150%; both individual features => ROI 75%.
        for grade, stage, wins in (
            ("F1", "Ｓ級予選", 3), ("F1", "Ｓ級決勝", 0),
            ("F2", "Ｓ級予選", 0), ("F2", "Ｓ級決勝", 0),
        ):
            for i in range(10):
                win = (1, 2, 3) if i < wins else (5, 4, 3)
                r, y = market(year, seq, grade, stage, win)
                odds.append(r)
                results.append(y)
                seq += 1
        gz(root / "data/odds_only" / f"odds_{year}.jsonl.gz", odds)
        gz(root / "data/results_only" / f"results_{year}.jsonl.gz", results)


class Tests(unittest.TestCase):
    def test_race_id(self):
        self.assertEqual(str(race_day(rid(2021, 1))), "2021-01-01")

    def test_complete_market_rejects_partial_and_duplicates(self):
        raw = {"1-2-3": 5.0}
        self.assertIsNone(parse_market(raw, "3連単"))
        raw2 = {"1-2-3": 5.0, "1=2=3": 5.0}
        self.assertIsNone(parse_market(raw2, "3連単"))

    def test_line_shape_not_inferred_from_positions_only(self):
        riders = {i: {"line": "", "pos": 1, "age": None} for i in range(1, 6)}
        self.assertIsNone(valid_lines(riders, set(riders)))

    def test_actual_ticket_policy_and_winner_feature_separation(self):
        race, _ = market(2021, 1, "F1", "Ｓ級予選", (1, 2, 3))
        items = parse_market(race["odds"]["3連単"], "3連単")
        def compare(finish):
            return [(name, [combo for combo, _ in choices]) for name, choices, _, _ in
                    select_policies("3連単", items, finish, None)]
        self.assertEqual(compare((1, 2, 3)), compare((5, 4, 3)))

    def test_pair_profitable_when_single_conditions_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build_data(root, [2021])
            stats, audit = scan(root, 2021, 2021, {})
            def roi(key):
                return stats[key][2021].result()["roi"]
            base = ("3連単/top1", ())
            a = ("3連単/top1", ("grade=F1",))
            b = ("3連単/top1", ("stage=予選",))
            ab = ("3連単/top1", ("grade=F1", "stage=予選"))
            self.assertEqual(roi(base), 0.375)
            self.assertEqual(roi(a), 0.75)
            self.assertEqual(roi(b), 0.75)
            self.assertEqual(roi(ab), 1.5)
            self.assertEqual(audit["2021"]["verified_races"], 40)

    def test_walk_forward_frozen_rule_and_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build_data(root, range(2021, 2027))
            report = run(root, root / "out", min_races=15, min_hits=2, min_years=2)
            self.assertGreater(report["profitable_train_cells_including_fragile"], 0)
            self.assertEqual(len(report["walk_forward"]), 4)
            for x in report["walk_forward"]:
                self.assertLess(max(x["train_years"]), x["test_year"])
                z = x["forward"]
                self.assertEqual(z["stake_yen"], z["bets"] * 100)
                self.assertEqual(z["profit_yen"], z["return_yen"] - z["stake_yen"])
            self.assertTrue((root / "out/report.json").is_file())
            self.assertTrue((root / "out/all_flatbuy_cells.csv").is_file())


if __name__ == "__main__":
    unittest.main()
