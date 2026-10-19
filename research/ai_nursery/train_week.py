"""Baby keirin AI — zero-knowledge online learner, seven calendar days.

The AI gets ONLY (1) raw racecard rows and (2) all verified final odds.
After its mark-sheet prediction has been written, it is shown the true finish.
There are no pretrained weights, betting rules, favorites, line heuristics,
market EV thresholds, or borrowed GEN-001 coefficients.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/ai_nursery/outputs/GEN-001"
WEB=ROOT/"docs/ai-lab"
START=date(2023,1,1)
END=date(2023,12,31)
D=256
LR=0.045
SEED=20261019
NUMERIC_SKIP={"race_id","race_no","car_no","race_date","captured_at_utc","start_time","deadline","source_url","line_id"}
TEXT_FIELDS=("class","style","line_role","prefecture")
POOLS=("3連単","3連複")
SKIP_FIELDS={"prediction_mark","evaluation","raw_row_json","source_url","captured_at_utc","result_comment","finish_position","finish_text","winning_method","result_source"}


def numeric(raw):
    try:
        s=str(raw).strip().replace(",","").replace("%","")
        if not s:return None
        v=float(s)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None

def keyindex(txt):
    return 4+int(hashlib.blake2b(txt.encode("utf-8"),digest_size=4).hexdigest(),16)%(D-4)

def fmt(i):
    return f"{int(i):03d}"

def load_cards():
    """Load every available A1/A2, A3, S, and graded 2023 source; no race-type filtering."""
    candidates=[]
    candidates+=sorted((ROOT/"data/a_class_12").glob("2023_*/races.csv"))
    candidates+=sorted((ROOT/"data/a3_challenge").glob("2023_*/races.csv"))
    candidates+=sorted((ROOT/"data/2023").rglob("races.csv"))
    races={}
    card_sources=Counter()
    def add(read_races,read_entries,source):
        local={}
        for row in read_races:
            day=row.get("race_date","")
            if not ("2023-01-01"<=day<="2023-12-31"):continue
            rid=str(row.get("race_id","")).strip()
            if not re.fullmatch(r"\d{16}",rid):continue
            local[rid]={"race":row,"entries":{},"source":source}
        if not local:return
        for ent in read_entries:
            rid=str(ent.get("race_id","")).strip()
            if rid not in local:continue
            car=numeric(ent.get("car_no"))
            if car is not None and int(car)==car and 1<=car<=9:
                # Only racecard attributes; never inject historical result annotations.
                local[rid]["entries"][int(car)]={k:v for k,v in ent.items() if k not in SKIP_FIELDS}
        for rid,obj in local.items():
            if not obj["entries"]:continue
            old=races.get(rid)
            if old is None or len(obj["entries"])>len(old["entries"]):
                races[rid]=obj
                card_sources[source]+=1
    def stream_csv(path):
        with path.open("r",encoding="utf-8-sig",newline="") as f:
            yield from csv.DictReader(f)
    for racefile in candidates:
        entryfile=racefile.with_name("entries.csv")
        if entryfile.exists():
            add(stream_csv(racefile),stream_csv(entryfile),str(racefile.parent.relative_to(ROOT)))
    for zipfile_path in sorted((ROOT/"data/grade_races").rglob("*2023*.zip")):
        with zipfile.ZipFile(zipfile_path) as z:
            race_names=[n for n in z.namelist() if n.endswith("/races.csv")]
            entry_names=[n for n in z.namelist() if n.endswith("/entries.csv")]
            if len(race_names)!=1 or len(entry_names)!=1:continue
            with z.open(race_names[0]) as ra,z.open(entry_names[0]) as en:
                races_reader=csv.DictReader(io.TextIOWrapper(ra,encoding="utf-8-sig"))
                entries_reader=csv.DictReader(io.TextIOWrapper(en,encoding="utf-8-sig"))
                add(races_reader,entries_reader,str(zipfile_path.relative_to(ROOT)))
    return races,card_sources

def load_results():
    res={}
    path=ROOT/"data/result_universe_v2/combined/results_2023.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            o=r.get("finish_order",[])
            if len(o)==3 and len(set(o))==3:
                res[str(r["race_id"])]=tuple(int(x) for x in o)
    return res

def load_odds(needed):
    odds={}
    path=ROOT/"data/odds_only/odds_2023.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            rid=str(r.get("race_id",""))
            if rid not in needed:continue
            both=r.get("odds",{})
            if not all(isinstance(both.get(k),dict) and both[k] for k in POOLS):continue
            clean={}
            valid=True
            for market in POOLS:
                clean[market]={}
                for key,value in both[market].items():
                    price=numeric(value)
                    if price is None or price<1:valid=False;break
                    clean[market][key]=price
            if valid:odds[rid]=clean
    return odds

def eligible(row):
    """Reject only missing/unresolvable data, never based on A/S/grade or predicted value."""
    entries=row["entries"]
    odds=row["odds"]
    cars=set(entries)
    if not (5<=len(cars)<=9):return False
    if any(c not in cars for c in row["outcome"]):return False
    expected=[len(cars)*(len(cars)-1)*(len(cars)-2),
              len(cars)*(len(cars)-1)*(len(cars)-2)//6]
    for market,count in zip(POOLS,expected):
        if len(odds[market])!=count:return False
        for key in odds[market]:
            sep="-" if market=="3連単" else "="
            try:participants=[int(x) for x in key.split(sep)]
            except ValueError:return False
            if len(participants)!=3 or len(set(participants))!=3 or not set(participants).issubset(cars):
                return False
    o=row["outcome"]
    return "-".join(map(str,o)) in odds["3連単"] and "=".join(map(str,sorted(o))) in odds["3連複"]

def feature_matrix(entries, odds):
    """Simple generic feature hashing, not manually encoded handicapping theory."""
    fields=set()
    for e in entries.values():
        for k,v in e.items():
            if k not in NUMERIC_SKIP and k not in SKIP_FIELDS and k not in TEXT_FIELDS and numeric(v) is not None:
                fields.add(k)
    standardized={}
    for k in fields:
        vals=[numeric(e.get(k)) for e in entries.values()]
        available=[v for v in vals if v is not None]
        if not available:continue
        mean=sum(available)/len(available)
        sd=max(math.sqrt(sum((v-mean)**2 for v in available)/len(available)),0.1)
        for car,e in entries.items():
            v=numeric(e.get(k))
            if v is not None:
                standardized[(car,k)]=max(-3,min(3,(v-mean)/sd))
    arms=[]
    count={m:len(odds[m]) for m in POOLS}
    for market in POOLS:
        for selection,price in sorted(odds[market].items()):
            sep="-" if market=="3連単" else "="
            cars=[int(x) for x in selection.split(sep)]
            f=np.zeros(D,dtype=np.float64)
            f[0]=1.0
            f[1]=math.log(price)/9.0
            f[2]=float(market=="3連単")
            f[3]=float(market=="3連複")
            for slot,car in enumerate(cars):
                e=entries[car]
                f[keyindex("car:"+str(slot)+":"+str(car))]+=0.3
                for k in fields:
                    v=standardized.get((car,k))
                    if v is not None:
                        f[keyindex("raw:"+str(slot)+":"+k)]+=v/5
                for key in TEXT_FIELDS:
                    if e.get(key):
                        f[keyindex("category:"+str(slot)+":"+key+":"+str(e[key]))]+=0.3
            arms.append({"market":market,"selection":selection,"odds":price,"x":f,
                         "prior":-math.log(count[market])})
    return arms

def softmax(a):
    p=np.exp(a-max(a))
    return p/p.sum()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)
    card,card_source=load_cards()
    results=load_results()
    odds=load_odds(set(card).intersection(results))
    ready=[]
    all_days=Counter()
    removed=Counter()
    for rid,obj in card.items():
        day=obj["race"].get("race_date","")
        all_days[day]+=1
        if rid not in results:removed["missing_result"]+=1;continue
        if rid not in odds:removed["missing_complete_odds"]+=1;continue
        row={**obj,"odds":odds[rid],"outcome":results[rid],"race_id":rid}
        if not eligible(row):removed["ineligible_or_inconsistent_entry"]+=1;continue
        ready.append(row)
    if not ready:raise RuntimeError("No exact-ID complete card+odds+result joins in 2023")
    # Earliest calendar week of joined races (no cherry-picking predicted results).
    first=min(date.fromisoformat(x["race"]["race_date"]) for x in ready)
    last=first+timedelta(days=6)
    batch=[x for x in ready if first<=date.fromisoformat(x["race"]["race_date"])<=last]
    batch.sort(key=lambda x:(x["race"]["race_date"],x["race"].get("start_time",""),
                             x["race"].get("track",""),int(x["race"].get("race_no","0") or 0),x["race_id"]))
    if not batch:raise RuntimeError("No seven-day training races")
    rng=np.random.default_rng(SEED)
    weights=np.zeros(D,dtype=np.float64)
    journal=[]
    losses=[]
    for i,r in enumerate(batch,1):
        # Strict phase 1: agent sees racecard and exact odds, never the result.
        pre_card=r["entries"]
        arms=feature_matrix(pre_card,r["odds"])
        X=np.stack([arm["x"] for arm in arms])
        prior=np.asarray([arm["prior"] for arm in arms])
        pre_hash=hashlib.sha256(weights.tobytes()).hexdigest()
        prob=softmax(X@weights+prior)
        choice=int(rng.choice(len(arms),p=prob))
        arm=arms[choice]
        prediction={"step":i,"race_id":r["race_id"],"date":r["race"]["race_date"],
            "track":r["race"].get("track"),"race_type":r["race"].get("race_type"),
            "race_no":r["race"].get("race_no"),"type":arm["market"],
            "selection":arm["selection"],"yen":100,
            "checkpoint_before_sha256":pre_hash,
            "probability_of_selected_mark":float(prob[choice]),
            "available_mark_options":len(arms)}
        # Write and flush the marksheet before reading the correct finishing order.
        with (OUT/"mark_sheets.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(prediction,ensure_ascii=False)+"\n")
            f.flush()
            os.fsync(f.fileno())
        # Phase 2: unseal outcome and update the model; the next race gets new weights.
        finish=r["outcome"]
        correct={
            ("3連単","-".join(map(str,finish))),
            ("3連複","=".join(map(str,sorted(finish))))
        }
        label_ids=[j for j,a in enumerate(arms) if (a["market"],a["selection"]) in correct]
        if len(label_ids)!=2:raise AssertionError("Missing winning mark in full markets")
        loss=float(-sum(math.log(max(prob[j],1e-30)) for j in label_ids)/2)
        label_mean=X[label_ids].mean(axis=0)
        expectation=prob@X
        gradient=expectation-label_mean
        weights-=LR*gradient
        hit=(arm["market"],arm["selection"]) in correct
        record={**prediction,"finish_order":"-".join(map(str,finish)),
            "hit":bool(hit),"loss_before_update":loss,
            "reward_yen_proxy":int(round(arm["odds"]*100)) if hit else 0,
            "checkpoint_after_sha256":hashlib.sha256(weights.tobytes()).hexdigest()}
        journal.append(record)
        losses.append(loss)
        if i%25==0:print("BABY_PROGRESS",i,len(batch),"last_date",prediction["date"],"hits",sum(v["hit"] for v in journal),flush=True)
    daily=[]
    for d in (first+timedelta(days=i) for i in range(7)):
        rr=[x for x in journal if x["date"]==d.isoformat()]
        daily.append({"date":d.isoformat(),"races":len(rr),"hits":sum(x["hit"] for x in rr),
                      "mean_loss":sum(x["loss_before_update"] for x in rr)/len(rr) if rr else None,
                      "paper_stake_yen":100*len(rr),
                      "paper_return_proxy_yen":sum(x["reward_yen_proxy"] for x in rr)})
    won=sum(x["hit"] for x in journal)
    stake=100*len(journal)
    payout=sum(x["reward_yen_proxy"] for x in journal)
    summary={
        "id":"GEN-001","stage":"BABY_FIRST_SEVEN_DAYS_COMPLETED",
        "date_first":first.isoformat(),"date_last":last.isoformat(),
        "instructions_only":"出走表とオッズを見る。マークシートに券種・車番・金額を記載する。結果は記載後に教わる。",
        "model":"fresh-zero-weights generic online feature learner; no previous models or handicapping",
        "prior_checkpoint":"all zeros","random_seed":SEED,"feature_dimension":D,"learning_rate":LR,
        "races":len(journal),"hits":won,"accuracy_percent":100*won/len(journal),
        "mean_loss":sum(losses)/len(losses),
        "first_30_loss":sum(losses[:30])/min(30,len(losses)),
        "last_30_loss":sum(losses[-30:])/min(30,len(losses)),
        "paper_stake_yen":stake,"paper_return_proxy_yen":payout,
        "paper_roi_proxy_percent":100*payout/stake,
        "bet_type_counts":dict(Counter(x["type"] for x in journal)),
        "race_type_counts":dict(Counter(x["race_type"] for x in journal)),
        "racecard_count_2023":len(card),"joined_2023":len(ready),
        "missing_counts_2023":dict(removed),
        "raw_racecard_counts_seven_days":{k:v for k,v in sorted(all_days.items()) if first.isoformat()<=k<=last.isoformat()},
        "daily":daily,
        "limitations":[
          "This is a historic teaching simulation. The odds supplied were FINAL odds, not historically captured pre-deadline prices.",
          "Racecards were retrospectively archived; feature timestamps cannot be certified as pre-race.",
          "Only 3連単 and 3連複 are present in the joint public odds archive. Other types were not invented.",
          "Only exact-ID joined racecard, complete odds and finishing order are played; missing races are logged.",
          "100 yen per mark is only the training sheet accounting denomination, not investment advice.",
          "Returns are odds * 100 proxy, not independently confirmed official settled payouts.",
          "First-week training scores cannot be treated as out-of-sample prediction performance."
        ]
    }
    checkpoint={"id":"GEN-001","after_step":len(journal),"dimension":D,"weights":[float(x) for x in weights],
                "random_seed":SEED,"weights_sha256":hashlib.sha256(weights.tobytes()).hexdigest(),
                "training_span":[first.isoformat(),last.isoformat()]}
    for path,obj in ((OUT/"summary.json",summary),(OUT/"checkpoint.json",checkpoint),
                     (WEB/"nursery-summary.json",summary),(WEB/"nursery-checkpoint.json",checkpoint)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"learning_history.jsonl").open("w",encoding="utf-8") as f:
        for row in journal:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    (WEB/"nursery-history.json").write_text(json.dumps(journal,ensure_ascii=False)+"\n",encoding="utf-8")
    print("BABY_ONE_WEEK_COMPLETE",json.dumps({"first":summary["date_first"],"last":summary["date_last"],
        "races":summary["races"],"hits":won,"bet_types":summary["bet_type_counts"],
        "missing":summary["missing_counts_2023"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
