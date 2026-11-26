#!/usr/bin/env python3
"""Audit ALL public real entrant/payout archives, independently of sealed odds.

Never substitute these public racecards for the full seven-market 134,336
odds universe. No post-event value may be marked as a pre-race feature.
Rider rows and suspect race IDs are kept in separate CMS-encrypted artifacts.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import gzip
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
CERT=ROOT/"odds_market/vault_public_cert.pem"
YEARS=tuple(range(2021,2027))
FIELDS=("player_profile","player_name","prefecture","age","term","class",
        "style","gear","score","s_count","b_count","nige_count","makuri_count",
        "sashi_count","mark_count","first_count","second_count","third_count",
        "outside_count","win_rate","top2_rate","top3_rate",
        "line_id","line_position","line_size","line_role")
TICKET_RX=re.compile(r"(?:3連単|３連単|三連単|3rentan|trifecta)",re.I)
TIE_RX=re.compile(r"同着|同順位|tie",re.I)
RID_RX=re.compile(r"^\d{16}$")
JST=dt.timezone(dt.timedelta(hours=9))
BASELINE_CLEAN={2021:23668,2022:24325,2023:24923,2024:24540,2025:24174,2026:11855}
BASELINE_MATCHED={2021:23669,2022:24331,2023:24966,2024:24598,2025:24234,2026:11875}

def race_day(rid):
    if not RID_RX.fullmatch(rid): return None
    if int(rid[10:12]) not in range(1,8): return None
    try:
        d=dt.date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))
        return d+dt.timedelta(days=int(rid[10:12])-1)
    except ValueError:
        return None

def parse_time(s):
    if not s:return None
    try:
        x=dt.datetime.fromisoformat(s.replace("Z","+00:00"))
        return x if x.tzinfo else None
    except ValueError:return None

def captured_before_event(row,day):
    captured=parse_time(row.get("captured_at_utc","").strip())
    deadline=(row.get("deadline") or "").strip()
    if captured is None or not re.fullmatch(r"\d\d:\d\d",deadline):
        return False
    try:
        hh,mm=map(int,deadline.split(":"))
        close=dt.datetime.combine(day,dt.time(hh,mm),tzinfo=JST)
        return captured<=close
    except ValueError:return False

def normalize_combo(s):
    s=str(s or "").strip()
    parts=re.findall(r"[1-9]",s)
    if len(parts)!=3 or len(set(parts))!=3:return ""
    return "-".join(parts)

class SealedWriter:
    def __init__(self,path):self.path=Path(path);self.proc=None;self.z=None;self.n=0
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.proc=subprocess.Popen(["openssl","cms","-encrypt","-binary",
            "-aes-256-gcm","-stream","-outform","DER",
            "-out",str(self.path),str(CERT)],stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        self.z=gzip.GzipFile(fileobj=self.proc.stdin,mode="wb",mtime=0)
        return self
    def write(self,record):
        self.z.write((json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n").encode())
        self.n+=1
    def __exit__(self,kind,val,tb):
        self.z.close()
        self.proc.stdin.close()
        try: rc=self.proc.wait(timeout=100)
        except subprocess.TimeoutExpired:self.proc.kill();raise
        if kind is not None or rc:
            self.path.unlink(missing_ok=True)
            if not kind:raise RuntimeError("Failed to encrypt protected audit")
        return False

def iter_tables(names):
    """Open every saved plaintext CSV and every grade ZIP member. Never sample."""
    for file in sorted(DATA.rglob("*.csv")):
        if file.name not in names:continue
        with file.open("r",encoding="utf-8-sig",newline="") as f:
            yield file.name,str(file.relative_to(ROOT)),csv.DictReader(f)
    for file in sorted((DATA/"grade_races").rglob("*.zip")):
        with zipfile.ZipFile(file) as z:
            for n in sorted(z.namelist()):
                kind=n.rsplit("/",1)[-1]
                if kind not in names:continue
                with z.open(n) as raw,io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as f:
                    yield kind,str(file.relative_to(ROOT))+":"+n,csv.DictReader(f)

def load(year):
    entrants=collections.defaultdict(lambda:collections.defaultdict(list))
    races=collections.defaultdict(list)
    results=collections.defaultdict(list)
    payouts=collections.defaultdict(list)
    seen=collections.Counter()
    for kind,source,rows in iter_tables({"entries.csv","races.csv","results.csv","payouts.csv"}):
        seen["sources_"+kind]+=1
        for row in rows:
            rid=str(row.get("race_id","")).strip()
            day=race_day(rid)
            if day is None or day.year!=year or (year==2026 and day>dt.date(2026,6,30)):continue
            seen["rows_"+kind]+=1
            row["_source"]=source
            if kind=="entries.csv":
                s=str(row.get("car_no","")).strip()
                if s in {str(i) for i in range(1,10)}:
                    entrants[rid][int(s)].append(row)
                else:seen["invalid_car"]+=1
            elif kind=="races.csv":races[rid].append(row)
            elif kind=="results.csv":results[rid].append(row)
            else:payouts[rid].append(row)
    return entrants,races,results,payouts,seen

def consensus(rows,field):
    values={str(row.get(field,"")).strip() for row in rows if str(row.get(field,"")).strip()}
    if len(values)>1:return None,True
    return (next(iter(values)) if values else None),False

def tie_evidence(rows,paid,order):
    """Different valid winning 3連単 codes or duplicate podium ranks quarantine the race."""
    tags=set();wins=set();mismatch=False
    for row in paid:
        typ=" ".join(str(row.get(k,"")) for k in ("ticket_type","bet_code"))
        if not TICKET_RX.search(typ):continue
        combo=normalize_combo(row.get("combination",""))
        if combo:wins.add(combo)
        elif row.get("combination"):tags.add("UNPARSEABLE_TRIFECTA_PAYOUT")
        if TIE_RX.search(str(row.get("status",""))+" "+str(row.get("raw_text",""))):
            tags.add("PAYOUT_TEXT_TIE")
    if len(wins)>1:tags.add("MULTIPLE_OFFICIAL_TRIFECTA_WINNERS")
    if len(wins)==1 and order is not None:
        if next(iter(wins))!="-".join(map(str,order)):tags.add("OFFICIAL_PAYOUT_ORDER_MISMATCH")
    ranks=collections.defaultdict(set)
    for r in rows:
        pos=str(r.get("finish_position","")).strip()
        c=str(r.get("car_no","")).strip()
        if pos in {"1","2","3"} and c.isdigit():ranks[pos].add(int(c))
        if TIE_RX.search(str(r.get("finish_text",""))):
            tags.add("RESULT_TEXT_TIE")
    if any(len(s)>1 for s in ranks.values()):
        tags.add("MULTIPLE_RIDERS_SAME_PODIUM_RANK")
    return sorted(tags),len(wins),len(paid)

def verified_results(year):
    src=DATA/f"result_universe_v2/combined/results_{year}.jsonl.gz"
    if not src.exists():raise RuntimeError("missing full result table "+str(src))
    known={};conflicts=set()
    with gzip.open(src,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line);rid=str(x.get("race_id",""))
            order=x.get("finish_order")
            when=race_day(rid)
            if when is None or when.year!=year or (year==2026 and when>dt.date(2026,6,30)) or not isinstance(order,list) or len(order)!=3:continue
            if rid in known and known[rid]!=order:conflicts.add(rid)
            known[rid]=order
    for k in conflicts:known.pop(k,None)
    return known,conflicts

def run(year,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    entrants,races,results,payouts,raw=load(year)
    labels,invalidlabels=verified_results(year)
    observed=collections.Counter(); car_counts=collections.Counter()
    quarantine=collections.Counter()
    all_ids=set(labels)|set(entrants)|set(races)|set(results)|set(payouts)
    private=out/f"rider_settlement_audit_{year}.jsonl.gz.cms"
    suspects=out/f"settlement_quarantine_{year}.jsonl.gz.cms"
    with SealedWriter(private) as records,SealedWriter(suspects) as rejected:
        for rid in sorted(all_ids):
            day=race_day(rid)
            all_car=entrants.get(rid,{})
            rr=races.get(rid,[])
            fin=labels.get(rid)
            payout=payouts.get(rid,[])
            tags,nwins,source_payout_rows=tie_evidence(results.get(rid,[]),payout,fin)
            if rid in invalidlabels:tags.append("LABEL_CONFLICT")
            for t in tags:quarantine[t]+=1
            if tags:
                observed["quarantine_races"]+=1
                rejected.write({"race_id":rid,"reason":tags})
            valid_roster=len(all_car) in {7,9} and set(all_car)==set(range(1,len(all_car)+1))
            if valid_roster:
                observed["complete_7_9_car_roster"]+=1
                car_counts[str(len(all_car))]+=1
                if fin is not None:observed["roster_and_result"]+=1
            elif all_car:observed["partial_or_unsupported_roster"]+=1
            points=[];any_ambiguous=False
            all_timing_valid=bool(all_car and rr)
            for car in sorted(all_car):
                samples=all_car[car]
                values={};conflicting=[]
                for key in FIELDS:
                    v,conflict=consensus(samples,key)
                    if conflict:conflicting.append(key)
                    values[key]=v
                if conflicting:any_ambiguous=True
                if not all(captured_before_event(x,day) for x in samples):
                    all_timing_valid=False
                points.append({"car":car,"values":values,"conflicting_fields":conflicting,
                               "source_count":len(samples),
                               "source_files":sorted({str(x["_source"]) for x in samples}),
                               "captured_at_utc":sorted({str(x.get("captured_at_utc","")) for x in samples})})
            if rr and not all(captured_before_event(x,day) for x in rr):
                all_timing_valid=False
            if any_ambiguous:observed["rider_attribute_conflict_races"]+=1
            if valid_roster and fin is not None and not tags:
                observed["labeled_with_complete_roster_and_no_detected_conflict"]+=1
                if all_timing_valid and not any_ambiguous:
                    observed["strictly_pre_race_timestamp_verified"]+=1
                else:
                    observed["retrospective_or_unverified_asof"]+=1
            if payout:observed["races_with_any_payout_rows"]+=1
            if nwins:observed["races_with_trifecta_payout_combination"]+=1
            if fin is not None:observed["full_result_archive_races"]+=1
            if all_car or tags:
                race_fields={}
                for field in ("race_type","meeting_grade","predicted_line_formation",
                              "track","deadline","captured_at_utc"):
                    value,conflict=consensus(rr,field)
                    race_fields[field]={"value":value,"conflict":conflict}
                records.write({"race_id":rid,"race_date":day.isoformat(),
                    "n_entry_cars":len(all_car),"complete_7_9_roster":valid_roster,
                    "result_archive_order":fin,"source_race":race_fields,
                    "riders":points,"payout_rows_present":bool(payout),
                    "distinct_trifecta_winner_count":nwins,
                    "settlement_quarantine":tags,"all_values_asof_pre_race":all_timing_valid,
                    "predictions_or_bets_created":False})
    known_ties=BASELINE_MATCHED[year]-BASELINE_CLEAN[year]
    result={
        "schema":"type_lab_full_rider_settlement_audit_v1","year":year,
        "all_csv_and_grade_zip_source_counts":dict(raw),
        "full_label_ids":len(labels),"all_race_ids_examined":len(all_ids),
        "counted":dict(sorted(observed.items())),
        "by_car":dict(sorted(car_counts.items())),
        "detected_unsafe_reasons":dict(sorted(quarantine.items())),
        "historical_independent_tie_exclusions_expected":known_ties,
        "historical_tie_188_id_list_not_recovered_by_this_audit":True,
        "detected_unsafe_not_equivalent_to_all_historical_188":True,
        "full_seven_market_odds_join_proven_in_this_public_only_run":False,
        "unverified_retrospective_features_must_not_enter_training":True,
        "private_artifacts":[private.name,suspects.name],
        "notes":[
            "Scans every available historical entries/races/results/payouts CSV and every grade ZIP, never small subset.",
            "Uses only the standard public result label archive and never infers missing winner.",
            "Duplicate official winning trifecta combinations or observed same-place podiums quarantine IDs.",
            "No payout row does not prove that a race had no tie.",
            "No source captured after a race can be marked temporally safe for model training.",
            "This audit does not certify coverage against all 134336 sealed market IDs; separate secure join required."
        ],
    }
    (out/"audit_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("TYPE_LAB_RIDER_SETTLEMENT_AUDIT",json.dumps({
       "year":year,"sources":sum(v for k,v in raw.items() if k.startswith("sources_")),
       "labels":len(labels),"complete_7_9":observed["complete_7_9_car_roster"],
       "roster_labeled":observed["roster_and_result"],
       "settlement_suspect":observed["quarantine_races"],
       "asof_certified":observed["strictly_pre_race_timestamp_verified"],
       "historical_ties_expected":known_ties
       },ensure_ascii=False),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--year",type=int,choices=YEARS,required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run(args.year,args.output)
