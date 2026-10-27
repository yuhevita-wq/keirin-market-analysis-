"""Full-universe replay of the archived six atoms and their original unions.
No S-class/qualifier/year subset. Undefined strategies abstain, with explicit audit.
CSCV is retrospective selection diagnostics, not a future-profit certificate.
"""
from __future__ import annotations
import sys,json,math,csv,gzip,itertools,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import date,timedelta,datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"research/ai_nursery")]
from source_bridge import _archive_streams, public_entries
from keirin_market_analysis.simulate_formation_slots_2023_2025 import ATOMS,masks
from research.snowberg_observation_corrected_20261027 import observe as corrected
OUT=ROOT/"research/pbo_full_universe_20261028"
OUT.mkdir(parents=True,exist_ok=True)
FIRST=date(2021,1,1);LAST=date(2026,6,30)
DAYS=[(FIRST+timedelta(days=i)).isoformat() for i in range((LAST-FIRST).days+1)]
DIDX={v:i for i,v in enumerate(DAYS)}
MS=masks()
NAMES=["+".join(m["atoms"]) for m in MS]
ATOMIC=[i for i,m in enumerate(MS) if len(m["atoms"])==1]
INC=np.array([[int(a in m["atoms"]) for a in ATOMS] for m in MS],dtype=float)
POINTS=np.array([m["points"] for m in MS],dtype=float)
STAKE=np.zeros((len(DAYS),len(MS)));PAY=np.zeros_like(STAKE)
ELIGIBLE=np.zeros(len(DAYS),dtype=int)
AUDIT=Counter();BY_YEAR=defaultdict(Counter);BY_TYPE=defaultdict(Counter)
CURRENT=None;CARDS={};CARD_AUDIT={};SOURCES=[]
ledger=gzip.open(OUT/"race_coverage.csv.gz","wt",encoding="utf-8",newline="")
writer=csv.writer(ledger);writer.writerow(["race_id","day","race_type","status"])
def source_paths():
    # All saved card containers, not only the former S-class qualifier directory.
    seen=set()
    for races in sorted((ROOT/"data").rglob("races.csv")):
        p=races.parent
        if (p/"entries.csv").is_file() and p not in seen:
            seen.add(p);yield "csv",p
    for races in sorted((ROOT/"data").rglob("races.csv.gz")):
        p=races.parent
        if (p/"entries.csv.gz").is_file() and p not in seen:
            seen.add(p);yield "gz_csv",p
    import zipfile
    for p in sorted((ROOT/"data").rglob("*.zip")):
        try:
            with zipfile.ZipFile(p) as z:
                b=[n.rsplit("/",1)[-1] for n in z.namelist()]
                if b.count("races.csv")==1 and b.count("entries.csv")==1:yield "zip",p
        except zipfile.BadZipFile:AUDIT["invalid_zip_containers"]+=1
def canonical_card(obj):
    entries=obj["entries"]
    parsed={}
    # Published line text can supply missing entry-level line positions.
    text=obj["race"].get("predicted_line_formation","")
    if text and str(obj["race"].get("line_status","")).lower()=="published":
        try:
            for lid,line in enumerate(text.split("/"),1):
                for pos,c in enumerate(line.split("-"),1):
                    n=int(c)
                    if n in parsed:return None,"invalid_published_line"
                    parsed[n]=(lid,pos)
            if set(parsed)!=set(entries):return None,"published_line_car_mismatch"
        except (ValueError,TypeError):return None,"invalid_published_line"
    groups=defaultdict(list)
    for car,e in entries.items():
        try:
            raw=str(e.get("score","")).replace(",","").strip()
            score=float(raw)
            if not math.isfinite(score) or not 30<=score<=150:return None,"missing_or_invalid_score"
            if parsed: lid,pos=parsed[car]
            else:lid,pos=int(e["line_id"]),int(e["line_position"])
            if lid<0 or pos<1:return None,"invalid_line_position"
        except (ValueError,TypeError,KeyError):return None,"missing_score_or_line"
        groups[lid].append((pos,car,score))
    out=[]
    for lid,m in groups.items():
        m.sort()
        if [x[0] for x in m]!=list(range(1,len(m)+1)):return None,"noncontiguous_line_position"
        out.append(tuple((c,s) for _,c,s in m))
    # Ignore arbitrary line IDs when comparing equivalent duplicate forecasts.
    return tuple(sorted(out)),None
def load_cards(year):
    global CARD_AUDIT
    cards={};bad=set();counts=Counter()
    for kind,path in source_paths():
        for source,races,entries in _archive_streams(kind,path):
            local={}
            for r in races:
                day=str(r.get("race_date",""))
                if not day.startswith(str(year)+"-") or day not in DIDX:continue
                rid=str(r.get("race_id",""))
                local[rid]={"race":r,"entries":{}}
            for e in entries:
                rid=str(e.get("race_id",""))
                if rid not in local:continue
                try:car=int(e["car_no"])
                except (ValueError,KeyError):continue
                if car in local[rid]["entries"]:counts["duplicate_entry_row"]+=1
                local[rid]["entries"][car]=public_entries(e)
            if local:SOURCES.append({"year":year,"path":source,"races":len(local)})
            for rid,obj in local.items():
                if rid in bad:continue
                card,reason=canonical_card(obj)
                if reason:
                    counts[reason]+=1
                    if rid not in cards:cards[rid]=(None,obj["race"].get("race_date"),reason)
                    continue
                val=(card,obj["race"].get("race_date"),None)
                previous=cards.get(rid)
                if previous and previous[0] is not None:
                    if previous[:2]!=val[:2]:
                        cards.pop(rid,None);bad.add(rid);counts["conflicting_role_cards_quarantined"]+=1
                    else:counts["equivalent_duplicate_card"]+=1
                else:cards[rid]=val
    for rid in bad:cards[rid]=(None,None,"conflicting_role_cards")
    counts["unique_race_ids"]=len(cards)
    CARD_AUDIT[str(year)]=dict(counts)
    print("CARDS",year,dict(counts),flush=True)
    return cards
def roles(card):
    candidates=[m for m in card if len(m)>=2]
    if len(candidates)<3:return None,"fewer_than_three_two_plus_lines"
    # Same pair-score and leader-score priority as original main-line rule.
    candidates.sort(key=lambda m:(-(m[0][1]+m[1][1]),-m[0][1],-int(len(m)>=3),m[0][0]))
    main=candidates[0]
    priority=lambda m:(m[0][1]+m[1][1],m[0][1],int(len(m)>=3))
    if sum(priority(m)==priority(main) for m in candidates)>1:
        return None,"ambiguous_main_line_tie"
    if len(main)<3:return None,"original_main_line_under_three"
    others=candidates[1:]
    low=min(m[0][1]+m[1][1] for m in others)
    weak=[m for m in others if m[0][1]+m[1][1]==low]
    if len(weak)!=1:return None,"weak_line_score_tie"
    rival=max(others,key=lambda m:(m[0][1]+m[1][1],-m[0][0]))
    if rival==weak[0]:return None,"rival_equals_weak"
    a,b=main[0][0],main[1][0];w,l=weak[0][0][0],weak[0][1][0]
    return {"A":a,"B":b,"W1L":w,"W1B":l},None
def capture(rid,day,books,finish,rtype,source):
    global CURRENT,CARDS
    year=int(day[:4])
    if year!=CURRENT:
        CARDS=load_cards(year);CURRENT=year
    AUDIT["full_archive_joined_races"]+=1
    def status(reason):
        AUDIT[reason]+=1;BY_YEAR[str(year)][reason]+=1;BY_TYPE[rtype][reason]+=1
        writer.writerow([rid,day,rtype,reason])
    item=CARDS.get(rid)
    if item is None:status("missing_racecard");return
    card,card_day,reason=item
    if reason:status(reason);return
    if card_day!=day:status("card_market_date_mismatch");return
    cars={c for t in books["3rentan"] for c in t}
    if {c for m in card for c,_ in m}!=cars:status("card_market_car_mismatch");return
    role,reason=roles(card)
    if reason:status(reason);return
    orders=[{tuple(role[r] for r in t) for t in ATOMS[a]} for a in ATOMS]
    assert all(len(x)==2 for x in orders)
    atompay=np.array([100*books["3rentan"][tuple(finish)] if tuple(finish) in x else 0 for x in orders])
    ix=DIDX[day]
    STAKE[ix]+=100*POINTS;PAY[ix]+=INC@atompay;ELIGIBLE[ix]+=1
    status("eligible_same_56_candidates")
def score(stake,pay):
    return np.divide(pay,stake,out=np.full_like(pay,-np.inf,dtype=float),where=stake>0)
def evaluate(indices,blocks=10):
    s=STAKE[:,indices];p=PAY[:,indices]
    names=[NAMES[i] for i in indices];n=len(indices)
    # Equal-sized blocks, padding only zero-exposure accounting rows; no dates dropped.
    pad=(-len(s))%blocks
    s=np.pad(s,((0,pad),(0,0)));p=np.pad(p,((0,pad),(0,0)))
    sb=s.reshape(blocks,-1,n).sum(axis=1);pb=p.reshape(blocks,-1,n).sum(axis=1)
    results=[]
    for train in itertools.combinations(range(blocks),blocks//2):
        test=[i for i in range(blocks) if i not in train]
        a=score(sb[list(train)].sum(axis=0),pb[list(train)].sum(axis=0))
        b=score(sb[test].sum(axis=0),pb[test].sum(axis=0))
        if not np.isfinite(a).any() or not np.isfinite(b).all():
            results.append({"status":"undefined_no_exposure"});continue
        chosen=int(np.argmax(a));v=b[chosen]
        rank=1+int(np.sum(b<v))+.5*(int(np.sum(b==v))-1)
        # Rank ascending: best is n. PBO means strictly below OOS median.
        results.append({"status":"ok","selected":names[chosen],"train_roi":float(a[chosen]),
           "test_roi":float(v),"ascending_rank":rank,"below_median":bool(rank<(n+1)/2),
           "test_loss":bool(v<1),"rank_fraction":rank/(n+1)})
    ok=[x for x in results if x["status"]=="ok"]
    return {"candidate_count":n,"metric":"aggregate payout/stake, not raw profit","blocks":blocks,
       "padding_zero_exposure_rows":pad,"splits_total":len(results),"valid_splits":len(ok),
       "pbo_strict_below_median":sum(x["below_median"] for x in ok)/len(ok) if ok else None,
       "loss_frequency_separate_from_pbo":sum(x["test_loss"] for x in ok)/len(ok) if ok else None,
       "mean_train_roi":float(np.mean([x["train_roi"] for x in ok])) if ok else None,
       "mean_test_roi":float(np.mean([x["test_roi"] for x in ok])) if ok else None,"splits":results}
def forward(indices):
    out=[]
    for year in range(2022,2027):
        train=np.array([int(d[:4])<year for d in DAYS]);test=np.array([int(d[:4])==year for d in DAYS])
        s=STAKE[:,indices];p=PAY[:,indices]
        a=score(s[train].sum(axis=0),p[train].sum(axis=0));b=score(s[test].sum(axis=0),p[test].sum(axis=0))
        if not np.isfinite(a).any() or not np.isfinite(b).all():
            out.append({"test_year":year,"status":"undefined_no_exposure"});continue
        i=int(np.argmax(a));rank=1+sum(b>b[i])+.5*(sum(b==b[i])-1)
        out.append({"test_year":year,"selected":NAMES[indices[i]],"train_roi":float(a[i]),
          "test_roi":float(b[i]),"descending_rank":float(rank),"candidates":len(indices),
          "stake_yen":float(s[test,i].sum()),"payout_proxy_yen":float(p[test,i].sum())})
    return out
def write_matrix():
    with gzip.open(OUT/"daily_candidate_matrix.csv.gz","wt",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["day","eligible_races"]+[a+"_"+n for n in NAMES for a in ("stake","payout_proxy")])
        for i,day in enumerate(DAYS):
            w.writerow([day,int(ELIGIBLE[i])]+[float(v) for j in range(len(MS)) for v in (STAKE[i,j],PAY[i,j])])
def main():
    import os
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):raise RuntimeError("Missing existing private-key secret")
    corrected.base_observe=capture
    corrected.old.save=lambda status:print("CHECKPOINT",dict(AUDIT),flush=True)
    try:corrected.old.sealed()
    finally:ledger.close()
    assert AUDIT["full_archive_joined_races"]==sum(sum(v.values()) for v in BY_YEAR.values())
    if not ELIGIBLE.sum():raise RuntimeError("No usable role candidate races; coverage report required")
    write_matrix()
    result={"status":"FULL_ARCHIVE_ROLE_SELECTION_REPLAY_COMPLETE","generated_utc":datetime.now(timezone.utc).isoformat(),
       "scope":"ALL existing sealed 2021-2026H1 books; all saved card containers; no class, race-type or S-qualifier restriction",
       "audit":dict(AUDIT),"by_year":{k:dict(v) for k,v in BY_YEAR.items()},
       "by_race_type":{k:dict(v) for k,v in BY_TYPE.items()},"card_source_audit":CARD_AUDIT,
       "source_containers":SOURCES,"candidate_registry":[{"name":n,"atoms":list(m["atoms"]),"points":m["points"]} for n,m in zip(NAMES,MS)],
       "six_atom_cscv":evaluate(ATOMIC),"original_union_family_cscv":evaluate(list(range(len(MS)))),
       "six_atom_forward":forward(ATOMIC),"original_union_family_forward":forward(list(range(len(MS)))),
       "orientation_checks":list(corrected.VALIDATED.values()),
       "limits":["Retrospective candidate family already historically explored; no untouched certification.",
       "Six original atoms and all original 1-to-4-atom unions (56), NOT every historical strategy or numeric gate.",
       "Same role family cannot be defined in every race. Undefined/missing cases abstain and remain in daily calendar and coverage ledger.",
       "Final-odds payout proxy; no official settlement or T-15 claim.",
       "Year/card source conflicts quarantined; alternative forecasts never selected by results.",
       "CSCV splits overlap; reported frequencies are descriptive and not independent confidence estimates.",
       "No AI or purchase rules modified."] }
    (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in result.items() if k not in ("source_containers","candidate_registry")}
    for k in ("six_atom_cscv","original_union_family_cscv"):compact[k]={a:b for a,b in compact[k].items() if a!="splits"}
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print("FINISHED",json.dumps(compact,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
