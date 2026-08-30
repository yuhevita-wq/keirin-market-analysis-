import csv, json, math
from collections import defaultdict
from pathlib import Path

YEAR = 2025
BASE = Path(f"data/{YEAR}/s_class_yosen")
OUT = Path("data/audits/selected_134_136_146_2025.json")
TH = {"entropy_min":0.7598574338315534,"top3_conc_max":0.5122247620383481,"p123_max":0.22900352400362795,"rank1_share_min":0.24054261443743438}
PATTERNS=[(1,3,4),(1,3,6),(1,4,6)]

def f(x):
    try: return float(x)
    except: return None

def racekey(row):
    return tuple(row[k] for k in ("date","venue","race_no"))

odds=defaultdict(dict)
with open(BASE/"trio_final_odds.csv",encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        o=f(r.get("odds")); combo=tuple(sorted(int(r[k]) for k in ("first","second","third")))
        if o and o>0: odds[racekey(r)][combo]=o

payout={}
with open(BASE/"payouts.csv",encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        if r.get("bet_type")!="trio": continue
        combo=tuple(sorted(int(x) for x in r["combination"].replace("-",",").split(",")))
        payout[racekey(r)]=(combo,int(float(r["payout_yen"])))

selected=[]; by={p:{"tickets":0,"hits":0,"stake_yen":0,"payout_yen":0} for p in PATTERNS}
for rk, omap in sorted(odds.items()):
    if rk not in payout: continue
    inv={c:1/o for c,o in omap.items()}; z=sum(inv.values())
    if not z: continue
    probs={c:v/z for c,v in inv.items()}
    riders=sorted({x for c in probs for x in c}); sup={x:sum(p for c,p in probs.items() if x in c) for x in riders}
    ranks={x:i+1 for i,x in enumerate(sorted(riders,key=lambda x:(-sup[x],x)))}
    rr={v:k for k,v in ranks.items()}
    ps=sorted(probs.values(),reverse=True); entropy=-sum(p*math.log(p) for p in probs.values())/math.log(len(probs)) if len(probs)>1 else 0
    top3=sum(ps[:3]); p123=probs.get(tuple(sorted(rr[i] for i in (1,2,3))),0); r1=sup[rr[1]]
    if not (entropy>=TH["entropy_min"] and top3<=TH["top3_conc_max"] and p123<=TH["p123_max"] and r1>=TH["rank1_share_min"]): continue
    selected.append(rk); win,pay=payout[rk]
    for pat in PATTERNS:
        combo=tuple(sorted(rr[i] for i in pat))
        d=by[pat]; d["tickets"]+=1; d["stake_yen"]+=100
        if combo==win: d["hits"]+=1; d["payout_yen"]+=pay

stake=sum(d["stake_yen"] for d in by.values()); pay=sum(d["payout_yen"] for d in by.values()); hits=sum(d["hits"] for d in by.values())
for d in by.values(): d["profit_yen"]=d["payout_yen"]-d["stake_yen"]; d["roi_pct"]=100*d["payout_yen"]/d["stake_yen"] if d["stake_yen"] else None
# race losing streak
hitset=set()
for rk in selected:
    win,_=payout[rk]; omap=odds[rk]; inv={c:1/o for c,o in omap.items()}; z=sum(inv.values()); probs={c:v/z for c,v in inv.items()}; riders=sorted({x for c in probs for x in c}); sup={x:sum(p for c,p in probs.items() if x in c) for x in riders}; rr={i+1:x for i,x in enumerate(sorted(riders,key=lambda x:(-sup[x],x)))}
    if any(tuple(sorted(rr[i] for i in pat))==win for pat in PATTERNS): hitset.add(rk)
cur=mx=0
for rk in selected:
    if rk in hitset: cur=0
    else: cur+=1; mx=max(mx,cur)
res={"status":"SELECTED_134_136_146_2025_FROZEN_2023_GATE","year":YEAR,"years_read":[YEAR],"thresholds":TH,"strategy":"Buy support-rank trios 1-3-4, 1-3-6, and 1-4-6, 100 yen each, only when frozen 2023 gate passes.","payout_method":"Actual published 3連複 payout_yen from 2025 payouts.csv.","result":{"total_analyzable_races":len([rk for rk in odds if rk in payout]),"selected_races":len(selected),"selection_rate_pct":100*len(selected)/len([rk for rk in odds if rk in payout]),"tickets":sum(d["tickets"] for d in by.values()),"hit_races":hits,"race_hit_rate_pct":100*hits/len(selected) if selected else None,"stake_yen":stake,"payout_yen":pay,"profit_yen":pay-stake,"roi_pct":100*pay/stake if stake else None,"max_losing_streak":mx,"by_pattern":{"-".join(map(str,k)):v for k,v in by.items()}},"warning":"2025 comparison using frozen 2023 gate. Only 1-3-4 added to the existing 1-3-6 and 1-4-6 tickets; no retuning."}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(res,ensure_ascii=False,indent=2))
