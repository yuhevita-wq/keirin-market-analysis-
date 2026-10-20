"""Canonical multi-year data-source connector for the baby keirin learner.

Connects existing GitHub-repository archives, NOT new scraped data or other
models. Joins racecards + complete FINAL trifecta/trio books + podium results
by exact 16-digit race ID. Strictly observational: never updates model weights.
GEN-001..007 old checkpoints and frozen 2023 exams are left untouched.

Use:
  from source_bridge import joined_races
  races, coverage = joined_races(2024, "2024-01-01", "2024-03-31")

Only use training periods that do not overlap any locked evaluation period.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import itertools
import json
import re
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

from train_week import ROOT, eligible

DATA=ROOT/"data"
AUDIT=DATA/"audits/odds_only_export.json"
YEARS=tuple(range(2021,2027))
MARKETS=("3連単","3連複")
RACEID=re.compile(r"^\d{16}$")
GIRL=re.compile(r"ガールズ|女子|L級|ガ予|ガ決|ガ一般|ガ選",re.I)
# Never allow result-derived columns as participant/racecard features.
FORBIDDEN=re.compile(
    r"(^|_)(?:result|finish|payout|paid|settlement|settled|hit|correct|"
    r"rank_after|post_race|winning|winner|dividend|recovered)(?:_|$)",re.I
)
ENTRY_DENY={
    "raw_row_json","captured_at_utc","source_url","source","race_id",
    "finish_position","finish_text","winning_method",
    "prediction_mark","evaluation","result_comment","result_source",
}
RECORD_FIELDS={"race_id","race_type","odds"}


def permitted_race_date(day,year,first=None,last=None):
    if not isinstance(day,str) or not re.fullmatch(r"20\d\d-\d\d-\d\d",day):
        return False
    try:
        dt=date.fromisoformat(day)
    except ValueError:
        return False
    if dt.year!=year or (year==2026 and dt>date(2026,6,30)):
        return False
    return (not first or day>=first) and (not last or day<=last)


def public_entries(entry):
    """A deliberately small, pre-result feature whitelist by exclusion."""
    return {
        k:v for k,v in entry.items()
        if k not in ENTRY_DENY and not FORBIDDEN.search(k)
        and k not in {"raw","result_json","odds","actual_order"}
    }


def _archive_paths(year):
    # A1/A2, A3 and graded race sources; 2023..2025 S-class parts.
    roots=[]
    if year in (2023,2024,2025):
        roots.append(DATA/str(year))
    if year==2026:
        roots.append(DATA/"2026_h1")
    for family in ("a_class_12","a3_challenge"):
        root=DATA/family
        if root.exists():
            roots.extend(p for p in sorted(root.iterdir())
                         if p.is_dir() and p.name.startswith(str(year)+"_"))
    for root in roots:
        if root.exists():
            for source in sorted(root.rglob("races.csv")):
                if (source.parent/"entries.csv").is_file():
                    yield ("csv",source.parent)
    # For early years, most racecards live in durable G1/G2/G3 zips.
    grade=DATA/"grade_races"
    if grade.exists():
        for z in sorted(grade.rglob("*.zip")):
            if str(year) in z.as_posix():
                yield ("zip",z)


def _stream_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        yield from csv.DictReader(handle)


def _archive_streams(kind,source):
    if kind=="csv":
        yield str(source.relative_to(ROOT)),_stream_csv(source/"races.csv"),_stream_csv(source/"entries.csv")
        return
    with zipfile.ZipFile(source) as z:
        race_names=[x for x in z.namelist() if x.rsplit("/",1)[-1]=="races.csv"]
        entry_names=[x for x in z.namelist() if x.rsplit("/",1)[-1]=="entries.csv"]
        if len(race_names)!=1 or len(entry_names)!=1:
            return
        with z.open(race_names[0]) as rawr, z.open(entry_names[0]) as rawe:
            with io.TextIOWrapper(rawr,encoding="utf-8-sig",newline="") as f_r, \
                 io.TextIOWrapper(rawe,encoding="utf-8-sig",newline="") as f_e:
                yield str(source.relative_to(ROOT)),csv.DictReader(f_r),csv.DictReader(f_e)


def _excluded_girl(race,entries):
    rt=str(race.get("race_type") or "")
    if GIRL.search(rt):
        return True
    return any(
        str(row.get("class") or "").strip().upper().startswith("L")
        for row in entries.values()
    )


def load_cards_year(year,first=None,last=None):
    """Return sanitized, deduplicated racecards and audited input counts."""
    if year not in YEARS:raise ValueError("Outside authorized 2021..2026 historical universe")
    cards={}
    counters=Counter()
    selected_source=Counter()
    for kind,path in _archive_paths(year):
        try:
            packages=_archive_streams(kind,path)
            for source,races,entries in packages:
                counters["source_containers"]+=1
                local={}
                for race in races:
                    day=str(race.get("race_date") or "")
                    if not permitted_race_date(day,year,first,last):
                        continue
                    rid=str(race.get("race_id") or "").strip()
                    if not RACEID.fullmatch(rid):
                        counters["invalid_race_id"]+=1;continue
                    counters["race_rows"]+=1
                    local[rid]={"race":{
                        k:v for k,v in race.items()
                        if k not in ENTRY_DENY and not FORBIDDEN.search(k)
                    },"entries":{},"source":source}
                for entry in entries:
                    rid=str(entry.get("race_id") or "").strip()
                    if rid not in local:continue
                    c=str(entry.get("car_no") or "").strip()
                    if not c.isdigit() or not (1<=int(c)<=9):
                        counters["invalid_car_no"]+=1;continue
                    local[rid]["entries"][int(c)]=public_entries(entry)
                for rid,obj in local.items():
                    if not obj["entries"]:
                        counters["no_entries"]+=1;continue
                    if _excluded_girl(obj["race"],obj["entries"]):
                        counters["girls_excluded"]+=1;continue
                    prev=cards.get(rid)
                    if prev is not None:
                        counters["duplicate_race_id"]+=1
                        if prev["race"].get("race_date")!=obj["race"].get("race_date"):
                            counters["conflicting_duplicate_date"]+=1
                            # Never silently choose one conflicting chronology.
                            del cards[rid]
                            counters["conflicting_race_ids_dropped"]+=1
                            continue
                        if len(prev["entries"])>=len(obj["entries"]):
                            continue
                        selected_source[prev["source"]]-=1
                    cards[rid]=obj
                    selected_source[source]+=1
        except (OSError,EOFError,zipfile.BadZipFile) as e:
            raise RuntimeError("Unreadable existing GitHub archive "+str(path)) from e
    counters["unique_cards"]=len(cards)
    return cards,dict(counters),{k:v for k,v in selected_source.items() if v>0}


def stream_podium(year,needed):
    path=DATA/f"result_universe_v2/combined/results_{year}.jsonl.gz"
    if not path.is_file():raise FileNotFoundError(path)
    out={}
    counters=Counter()
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            obj=json.loads(line)
            counters["source_records"]+=1
            rid=str(obj.get("race_id") or "")
            if rid not in needed:continue
            order=obj.get("finish_order")
            if not isinstance(order,list) or len(order)!=3:
                counters["invalid_podium"]+=1;continue
            try: cars=tuple(int(x) for x in order)
            except (TypeError,ValueError):
                counters["invalid_podium"]+=1;continue
            if len(set(cars))!=3 or min(cars)<1 or max(cars)>9:
                counters["invalid_podium"]+=1;continue
            old=out.get(rid)
            if old is not None and old!=cars:
                counters["conflicting_podium"]+=1
                del out[rid]
                continue
            out[rid]=cars
    return out,dict(counters)


def _sha256(path):
    hash_=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            hash_.update(block)
    return hash_.hexdigest()


def stream_full_odds(year,needed,*,verify_sha=True):
    path=DATA/f"odds_only/odds_{year}.jsonl.gz"
    if not path.is_file():raise FileNotFoundError(path)
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    declared=audit["years"][str(year)]
    if verify_sha and _sha256(path)!=declared["sha256"]:
        raise RuntimeError(f"Odds SHA256 mismatch; refusing unverified {path}")
    odds={}
    errors=Counter()
    records=0
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            obj=json.loads(line)
            records+=1
            rid=str(obj.get("race_id") or "")
            if rid not in needed:continue
            if set(obj)!=RECORD_FIELDS or any(
                not isinstance(obj["odds"].get(m),dict) or not obj["odds"][m]
                for m in MARKETS
            ):
                errors["malformed_complete_market"]+=1;continue
            odds[rid]=obj["odds"]
    if records!=declared["complete_races"]:
        raise RuntimeError(f"Corrupted market manifest: {year} records={records} claimed={declared['complete_races']}")
    return odds,{"archive_complete_races":records,"hash_verified":bool(verify_sha),
                 "archive_sha256":declared["sha256"],"matched_candidate_ids":len(odds),
                 "invalid":dict(errors)}


def joined_races(year,first=None,last=None,*,verify_sha=True):
    """Produce precisely the objects understood by feature_matrix & current learner.

    Separate label/odds/card presence conditions before applying existing
    full-grid checks. Only two fully stored markets, never invent odds.
    """
    cards,card_count,sources=load_cards_year(year,first,last)
    labels,label_count=stream_podium(year,set(cards))
    odds,market_count=stream_full_odds(year,set(cards),verify_sha=verify_sha)
    statuses=Counter()
    ready=[]
    for rid,obj in cards.items():
        if rid not in labels:
            statuses["no_result"]+=1
        if rid not in odds:
            statuses["no_complete_market"]+=1
        if rid not in labels or rid not in odds:
            continue
        item={"race_id":rid,**obj,"odds":odds[rid],"outcome":labels[rid]}
        if not eligible(item):
            statuses["incomplete_entries_or_race_mismatch"]+=1
            continue
        ready.append(item)
    ready.sort(key=lambda r:(
        r["race"]["race_date"],r["race"].get("start_time",""),
        r["race"].get("track",""),
        int(r["race"].get("race_no") or 0),r["race_id"]
    ))
    statuses["joined_and_eligible"]=len(ready)
    report={
        "year":year,"date_start":first or f"{year}-01-01",
        "date_end":last or (f"{year}-06-30" if year==2026 else f"{year}-12-31"),
        "source_count":card_count,
        "selected_racecard_sources":sources,
        "label_source":f"data/result_universe_v2/combined/results_{year}.jsonl.gz",
        "result_count":label_count,
        "odds_source":f"data/odds_only/odds_{year}.jsonl.gz",
        "market_count":market_count,
        "join_status":dict(statuses),
        "joined_race_ids_sha256":hashlib.sha256(
            ("\n".join(r["race_id"] for r in ready)).encode()
        ).hexdigest(),
        "complete_races":len(ready),
        "date_data_origin":"racecard race_date ONLY; never decoded meeting-start ID",
        "price_phase":"FINAL (historical), NOT certified pre-close",
        "markets":["3連単","3連複"],
        "result_used_only_as_label":True,
        "original_training_checkpoints_untouched":True,
    }
    return ready,report


def joined_races_for_training(start,end):
    """One explicit adapter callable from GEN-008+ trainers; no training here."""
    first=date.fromisoformat(start);last=date.fromisoformat(end)
    if first>last:raise ValueError("start after end")
    if first.year not in YEARS or last.year not in YEARS:
        raise ValueError("year not in permitted historical range")
    if last>date(2026,6,30):
        raise ValueError("2026 second half is prohibited")
    # Existing locked exams must not become training data.
    locked=((date(2023,6,1),date(2023,6,7)),
            (date(2023,7,1),date(2023,7,7)))
    for lo,hi in locked:
        if first<=hi and last>=lo:
            raise ValueError(f"Training would overlap locked exam {lo}..{hi}")
    rows=[]
    reports=[]
    for year in range(first.year,last.year+1):
        lo=max(first,date(year,1,1)).isoformat()
        hi=min(last,date(year,6,30) if year==2026 else date(year,12,31)).isoformat()
        batch,rep=joined_races(year,lo,hi)
        rows.extend(batch);reports.append(rep)
    rows.sort(key=lambda r:(
        r["race"]["race_date"],r["race"].get("start_time",""),
        r["race"].get("track",""),int(r["race"].get("race_no") or 0),r["race_id"]
    ))
    return rows,reports
