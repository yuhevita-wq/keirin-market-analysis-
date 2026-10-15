"""Compare observed nationwide daily calendar IDs against all archived races.

Output is discovery gaps, not price-model features. Each source archived race ID is
deduplicated. External calendar may itself be incomplete; treat as lower bound.
"""
from __future__ import annotations
import csv,gzip,io,json,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scope import in_scope,girls_race
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/audits/nationwide_census_vs_saved.json"

def saved_races():
    known={}
    for parent in ("a_class_12","a3_challenge","2023","2024","2025","2026_h1","2026_future_block1"):
        for path in (ROOT/"data"/parent).rglob("races.csv"):
            with path.open(encoding="utf-8-sig",newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("race_id") and in_scope(row.get("race_date")):
                        known[row["race_id"]]=(row.get("race_date",""),str(path.relative_to(ROOT)))
    for path in (ROOT/"data/grade_races").rglob("*.zip"):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.rsplit("/",1)[-1]=="races.csv"]
            for name in names:
                with z.open(name) as b:
                    with io.TextIOWrapper(b,encoding="utf-8-sig",newline="") as f:
                        for row in csv.DictReader(f):
                            if row.get("race_id") and row["race_id"] not in known and in_scope(row.get("race_date")):
                                known[row["race_id"]]=(row.get("race_date",""),str(path.relative_to(ROOT)))
    for path in (ROOT/"data/girls_universe").rglob("races.csv"):
        with path.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                if row.get("race_id") and row["race_id"] not in known and in_scope(row.get("race_date")):
                    known[row["race_id"]]=(row.get("race_date",""),str(path.relative_to(ROOT)))
    # A recovered racing card is now archived even if its final odds remain
    # incomplete; the latter is a SEPARATE market completeness problem.
    for path in (ROOT/"data/unarchived_races_v1").glob("batch_*/races.csv.gz"):
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                rid=row.get("race_id","")
                if rid and rid not in known and in_scope(row.get("race_date")):
                    known[rid]=(row["race_date"],str(path.relative_to(ROOT)))
    return known

def report():
    known=saved_races()
    sources=[]
    covered={}
    for path in (ROOT/"data/nationwide_calendar").rglob("race_index.csv.gz"):
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            rows=list(csv.DictReader(f))
        good=0;missing=0
        ignored_girls=0
        unknown_labels=0
        eligible_rows=0
        diff=[]
        for row in rows:
            rid=row["race_id"]
            if not in_scope(row.get("race_date")):
                continue
            if girls_race(row.get("race_type")):
                ignored_girls+=1
                continue
            if not row.get("race_type"):
                unknown_labels+=1
            eligible_rows+=1
            if rid in covered and covered[rid]!=row["race_date"]:
                raise ValueError("census race ID assigned conflicting event date")
            covered[rid]=row["race_date"]
            if rid in known:good+=1
            else:
                missing+=1
                diff.append(row)
        sources.append({"census_file":str(path.relative_to(ROOT)),
                        "calendar_race_ids":eligible_rows,
                        "girls_race_ids_excluded":ignored_girls,
                        "race_type_unlabeled":unknown_labels,
                        "calendar_raw_link_count":len(rows),
                        "already_in_saved_archive":good,
                        "race_ids_absent_in_saved_archive":missing,
                        "unarchived_races":diff})
    mismatch=[]
    for rid,day in covered.items():
        if rid in known and known[rid][0] and known[rid][0]!=day:
            mismatch.append({"race_id":rid,"calendar_date":day,
                             "saved_date":known[rid][0],
                             "saved_source":known[rid][1]})
    aggregate={"saved_unique_race_ids":len(known),
               "census_unique_race_ids":len(covered),
               "census_found_in_existing_archives":len(set(known)&set(covered)),
               "census_missing_from_existing_archives":len(set(covered)-set(known)),
               "date_conflicts":len(mismatch),
               "census_race_type_unlabeled":sum(x["race_type_unlabeled"] for x in sources),
               "census_girls_excluded":sum(x["girls_race_ids_excluded"] for x in sources)}
    result={"status":"NATIONWIDE_PUBLIC_CALENDAR_VS_SAVED_RACE_ID_GAPS",
            "generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "summary":aggregate,"date_conflict_examples":mismatch[:100],
            "sources":sources,
            "warning":"Calendar links are an observed sample, not proof of nationwide coverage. Nonindexed races remain possible.",
            "training_features_used":"none; this file is a dataset completeness audit"}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CALENDAR_COMPARE",json.dumps(aggregate,ensure_ascii=False),flush=True)
if __name__=="__main__":
    report()
