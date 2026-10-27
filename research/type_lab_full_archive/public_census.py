#!/usr/bin/env python3
"""Source-wide PUBLIC historical racecard/odds/result audit: NEVER call subset a full archive.

Ciphertext requires a separate secret-gated secure job. No training or betting.
"""
from __future__ import annotations
import collections
import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
OUT=ROOT/"research/type_lab_full_archive/public_connection_report.json"
RACE=re.compile(r"^[0-9]{16}$")
YEARS=list(range(2021,2027))
PLAIN_AUDIT=DATA/"audits/odds_only_export.json"

def reader(handle):
    return csv.DictReader(handle)

def rid(row):
    return str(row.get("race_id","")).strip()

def car(row):
    for key in ("car_no","car","car_number","bike_no","rider_no"):
        x=str(row.get(key,"")).strip()
        if x and x.isdigit(): return int(x)
    return None

def race_day(r):
    try:
        from datetime import date,timedelta
        return (date(int(r[2:6]),int(r[6:8]),int(r[8:10])) + timedelta(days=int(r[10:12])-1)).isoformat()
    except Exception: return None

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def main():
    src=collections.Counter()
    entrycars=collections.defaultdict(set)
    duplicate_entries=0
    line_races=set()
    score_races=set()
    cards=set()
    resultids={}
    result_conflicts=set()
    oddsids=set()
    plain_year=collections.Counter()
    year_of_odds={}
    for year in YEARS:
        path=DATA/f"result_universe_v2/combined/results_{year}.jsonl.gz"
        if not path.is_file(): raise RuntimeError("MISSING_PUBLIC_RESULT_ARCHIVE: "+str(path))
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip():continue
                src["result_rows"]+=1
                r=json.loads(line); k=rid(r); finish=r.get("finish_order")
                if not RACE.fullmatch(k):src["bad_result_ids"]+=1;continue
                if not isinstance(finish,list) or len(finish)!=3 or len(set(map(str,finish)))!=3:
                    src["bad_podium_rows"]+=1;continue
                if k in resultids and resultids[k]!=tuple(finish):
                    result_conflicts.add(k)
                resultids[k]=tuple(finish)
    for k in result_conflicts:resultids.pop(k,None)

    # All published plaintext odds, NOT the full encrypted seven-market set.
    reference=json.loads(PLAIN_AUDIT.read_text(encoding="utf-8"))
    for year in YEARS:
        path=DATA/f"odds_only/odds_{year}.jsonl.gz"
        if not path.is_file():raise RuntimeError("MISSING_PUBLIC_ODDS_ARCHIVE: "+str(path))
        descriptor=reference["years"][str(year)]
        if sha256(path)!=descriptor["sha256"]:
            raise RuntimeError("PLAIN_PUBLIC_ODDS_HASH_MISMATCH: "+str(path))
        n=0
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip():continue
                n+=1
                row=json.loads(line); key=rid(row)
                if key in oddsids:raise RuntimeError("duplicate plaintext odds race ID: "+key)
                oddsids.add(key);year_of_odds[key]=str(year)
        if n!=descriptor["complete_races"]:
            raise RuntimeError("ODDS_YEAR_DECLARED_COUNT_MISMATCH: "+str(year))
        plain_year[str(year)]=n

    # Every available CSV entry/racecard source, including ALL grade ZIP members.
    # No selective race stage, A-class restriction, or 2023-only sampling.
    csvfiles=sorted(DATA.rglob("entries.csv"))
    zipped=sorted((DATA/"grade_races").rglob("*.zip"))
    if not csvfiles or not zipped:raise RuntimeError("SOURCE_RACECARD_ARCHIVES_NOT_MOUNTED")
    def eat_entries(rows,source):
        nonlocal duplicate_entries
        src["entry_sources_opened"]+=1
        for row in rows:
            src["entry_rows"]+=1
            k=rid(row); c=car(row)
            if not RACE.fullmatch(k) or c is None or not 1<=c<=9:
                src["bad_entry_rows"]+=1;continue
            if c in entrycars[k]:duplicate_entries+=1
            entrycars[k].add(c)
            if any(row.get(name) for name in ("line_position","line_id","predicted_line_formation")):
                line_races.add(k)
            if any(row.get(name) for name in ("race_point","competition_score","score","racing_score")):
                score_races.add(k)
    for fp in csvfiles:
        with fp.open("r",encoding="utf-8-sig",newline="") as fh:
            eat_entries(reader(fh),str(fp.relative_to(ROOT)))
    for path in zipped:
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            src["zip_archives_opened"]+=1
            for name in names:
                if name.rsplit("/",1)[-1]!="entries.csv":continue
                with z.open(name) as f,io.TextIOWrapper(f,encoding="utf-8-sig",newline="") as t:
                    eat_entries(reader(t),str(path.relative_to(ROOT))+":"+name)

    # Count race-level metadata independently; this is separate from rider
    # actual-as-of-time feature availability.
    for fp in sorted(DATA.rglob("races.csv")):
        with fp.open("r",encoding="utf-8-sig",newline="") as f:
            for row in reader(f):
                src["race_rows"]+=1
                if RACE.fullmatch(rid(row)): cards.add(rid(row))
    for path in zipped:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.rsplit("/",1)[-1]!="races.csv":continue
                with z.open(name) as f,io.TextIOWrapper(f,encoding="utf-8-sig",newline="") as t:
                    for row in reader(t):
                        src["zip_race_rows"]+=1
                        if RACE.fullmatch(rid(row)):cards.add(rid(row))

    races77={k for k,cs in entrycars.items() if len(cs) in (7,9) and cs==set(range(1,len(cs)+1))}
    odds_results=oddsids & resultids.keys()
    joined3=odds_results & races77
    observed_byyear=collections.Counter(race_day(k)[:4] for k in joined3 if race_day(k))
    out={
      "schema":"type_lab_full_source_public_read_audit_v1",
      "status":"PUBLIC_SOURCE_FULL_CENSUS_COMPLETED__PRIVATE_SEVEN_MARKET_SEALED_PENDING",
      "counts":dict(src),
      "plain_complete_two_market_races":len(oddsids),
      "plain_two_market_by_year":dict(sorted(plain_year.items())),
      "independent_result_labels":len(resultids),
      "conflicted_result_ids":len(result_conflicts),
      "distinct_races_with_any_entries":len(entrycars),
      "races_with_complete_7_9_car_rosters_across_all_sources":len(races77),
      "odds_and_result_joined_plain_only":len(odds_results),
      "odds_result_complete_roster_join_plain_only":len(joined3),
      "plain_3way_joined_by_year":dict(sorted(observed_byyear.items())),
      "all_csv_entry_files_opened":len(csvfiles),
      "all_grade_zip_files_opened":len(zipped),
      "distinct_races_with_race_metadata":len(cards),
      "entry_duplicate_car_records_across_overlapping_archives":duplicate_entries,
      "races_with_any_line_columns":len(line_races),
      "races_with_any_score_columns":len(score_races),
      "encrypted_seven_market_files_decrypted":0,
      "entire_2021_2026_market_archives_connected":False,
      "model_pre_race_p1_p3_connected":False,
      "training_performed":False,
      "notes":[
         "All *available public* plaintext source files are read without sampling.",
         "PUBLIC 42,975-type two-market export is a STRICT SUBSET, not the >133k encrypted archive.",
         "Conflicting rider features and contemporaneous/retrospective timestamps need a separate audit.",
         "Combined 7/9-car roster count is not sufficient for original TypeLab model predictions.",
         "No decryption key was used and no wager was placed."
       ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TYPE_LAB_PUBLIC_FULL_SOURCE_COUNTS",json.dumps({
      "public_plain_market":len(oddsids),
      "result_labels":len(resultids),
      "complete_roster_7_9":len(races77),
      "plain_odds_results_roster_joined":len(joined3),
      "zip_files":len(zipped),"entry_files":len(csvfiles),
      "still_blocked_sealed":True
    }),flush=True)

if __name__=="__main__":
    main()
