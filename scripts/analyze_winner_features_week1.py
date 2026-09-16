#!/usr/bin/env python3
import csv, json
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE=Path("data/2024/s_class_f1_all_parts/2024_q1")
START="2024-01-01"; END="2024-01-07"

def read(name):
    with (BASE/name).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def num(v):
    try:return float(v)
    except:return 0.0
def ino(v):
    try:return int(float(v))
    except:return 0

races=read("races.csv"); entries=read("entries.csv"); results=read("results.csv")
target={r["race_id"]:r for r in races if START<=r["race_date"]<=END and ino(r["entry_count"])==7 and r["meeting_grade"]=="F1" and "Ｓ級" in r["race_type"]}
eby=defaultdict(list); rby=defaultdict(list)
for e in entries:
    if e["race_id"] in target: eby[e["race_id"]].append(e)
for r in results:
    if r["race_id"] in target: rby[r["race_id"]].append(r)

features=["score","b_count","nige_count","makuri_count","sashi_count","mark_count","win_rate","top2_rate","top3_rate","line_size"]
winner_vals=defaultdict(list); non_vals=defaultdict(list)
winner_rank_counts={k:defaultdict(int) for k in features}
style_win=defaultdict(int)
role_win=defaultdict(int)
n=0

for rid in target:
    es=eby[rid]
    if len(es)!=7: continue
    first=[x for x in rby[rid] if ino(x.get("finish_position"))==1]
    if len(first)!=1: continue
    wn=ino(first[0]["car_no"])
    n+=1
    for e in es:
        dest=winner_vals if ino(e["car_no"])==wn else non_vals
        for k in features: dest[k].append(num(e.get(k)))
        if ino(e["car_no"])==wn:
            style_win[e.get("style","")]+=1
            pos=ino(e.get("line_position")); size=ino(e.get("line_size"))
            role="solo" if size<=1 else ("leader" if pos==1 else ("second" if pos==2 else "third_plus"))
            role_win[role]+=1
    for k in features:
        ordered=sorted(es,key=lambda e:(-num(e.get(k)),ino(e.get("car_no"))))
        for rank,e in enumerate(ordered,1):
            if ino(e["car_no"])==wn:
                winner_rank_counts[k][rank]+=1
                break

out={"races":n,"winner_avg":{},"nonwinner_avg":{},"winner_vs_nonwinner_ratio":{},"winner_rank_distribution":{},"style_wins":dict(style_win),"role_wins":dict(role_win)}
for k in features:
    wa=mean(winner_vals[k]); na=mean(non_vals[k])
    out["winner_avg"][k]=wa; out["nonwinner_avg"][k]=na
    out["winner_vs_nonwinner_ratio"][k]=(wa/na if na else None)
    out["winner_rank_distribution"][k]=dict(sorted(winner_rank_counts[k].items()))

print(json.dumps(out,ensure_ascii=False,indent=2))
