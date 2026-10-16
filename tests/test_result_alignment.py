"""Offline fixtures for race-ID-exact odds/result alignment."""
import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]/"research/result_alignment_v1/build_result_targets.py"
spec = importlib.util.spec_from_file_location("result_alignment", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["race_id","car_no","finish_position"])
        w.writeheader()
        w.writerows(rows)


def three(rid, order):
    return [{"race_id":rid, "car_no":str(c), "finish_position":str(i)}
            for i,c in enumerate(order,1)]


class AlignmentTests(unittest.TestCase):
    def test_exact_odds_ids_and_conflict_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odds = root/"data/odds_only/odds_2023.jsonl.gz"
            odds.parent.mkdir(parents=True,exist_ok=True)
            ids = ["0120230101010001","0120230101010002","0120230101010003"]
            tri = {"-".join(map(str,(a,b,c))):float(10)
                   for a in (1,2,3) for b in (1,2,3) for c in (1,2,3)
                   if len({a,b,c})==3}
            with gzip.open(odds,"wt",encoding="utf-8") as out:
                for rid in ids:
                    out.write(json.dumps({"race_id":rid,"race_type":"A級一般","odds":{"3連単":tri}})+"\n")
            write_csv(root/"data/2023/source1/results.csv",
                      three(ids[0],[1,2,3]) + three(ids[1],[2,1,3]) +
                      three("0120230101010099",[1,2,3]))
            archive = root/"data/grade_races/races.zip"
            archive.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(archive,"w") as z:
                payload = "race_id,car_no,finish_position\n" + "".join(
                    f"{r['race_id']},{r['car_no']},{r['finish_position']}\n"
                    for r in three(ids[1],[3,1,2]))
                z.writestr("results.csv",payload)
            audit=mod.build(root)
            self.assertEqual(audit["odds_race_count"],3)
            self.assertEqual(audit["matched_result_count"],1)
            self.assertEqual(audit["missing_result_count"],2)
            self.assertEqual(audit["conflicting_result_count"],1)
            with gzip.open(root/"data/results_only/results_2023.jsonl.gz","rt") as f:
                self.assertEqual([json.loads(s) for s in f],[
                    {"race_id":ids[0],"finish_order":[1,2,3]}])
            with gzip.open(root/"data/results_only/unmatched_race_ids.csv.gz","rt") as f:
                csv_rows=list(csv.DictReader(f))
            self.assertEqual({r["race_id"] for r in csv_rows},set(ids[1:]))
            self.assertEqual(mod.parse_podium(three(ids[0],[1,1,3]),{1,2,3}),None)
            self.assertEqual(mod.parse_podium(three(ids[0],[1,2,4]),{1,2,3}),None)


if __name__=="__main__":
    unittest.main()
