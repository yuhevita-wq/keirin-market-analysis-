"""Minimal schema audit for the oldest complete seven-day baby-AI training window."""
from pathlib import Path
import csv
import gzip
import json
import zipfile
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
def first_gz(path,limit=2):
    if not path.exists():
        print("MISSING:",path.relative_to(ROOT));return
    with gzip.open(path,"rt",encoding="utf-8-sig") as stream:
        for _,line in zip(range(limit),stream):
            obj=json.loads(line)
            print("sample",path.relative_to(ROOT),"keys",list(obj.keys()),"repr",json.dumps(obj,ensure_ascii=False)[:1800])
def csv_probe(path):
    if not path.exists():
        print("MISSING",path.relative_to(ROOT));return
    with path.open(encoding="utf-8-sig",newline="") as f:
        rd=csv.DictReader(f);row=next(rd,None)
    print("CSV",path.relative_to(ROOT),"columns",list(row or {}),"row",json.dumps(row,ensure_ascii=False)[:1400])
def main():
    for y in ("2021","2022","2023"):
        first_gz(ROOT/f"data/odds_only/odds_{y}.jsonl.gz",1)
        first_gz(ROOT/f"data/result_universe_v2/combined/results_{y}.jsonl.gz",1)
    for p in [
        "data/a_class_12/2023_01_02/races.csv",
        "data/a_class_12/2023_01_02/entries.csv",
        "data/a_class_12/2023_01_02/results.csv",
        "data/a3_challenge/2023_01_02/races.csv",
        "data/2023/s_class_f1_all_parts/2023_q1/races.csv",
        "data/2023/s_class_f1_all_parts/2023_q1/entries.csv",
        "data/2023/s_class_yosen/races.csv"
    ]:
        csv_probe(ROOT/p)
    for path in sorted((ROOT/"data/grade_races/g3/2023").glob("*.zip"))[:1]:
        with zipfile.ZipFile(path) as z:
            print("ZIP",path.relative_to(ROOT),"files",z.namelist()[:20])
    odds_count=Counter()
    with gzip.open(ROOT/"data/odds_only/odds_2023.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            if str(row.get("race_id")) and row.get("race_type"):
                odds_count["all"]+=1
                if "2023" in str(row.get("race_id")):odds_count["id_2023"]+=1
            if "01-0" in str(row.get("race_id")):odds_count["id_week"]+=1
    print("2023 odds index",dict(odds_count))
if __name__=="__main__":main()
