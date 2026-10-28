"""Discovery, NOT preselected betting patterns.
Use actual finish -> whole odds -> learn conditional payout-weighted rider/line-role
membership maps. Build irregular union of cartesian rectangles via minimum
description length on train; evaluate unmodified on later dates.
No fixed 34 forms, no favorite formation chosen by ROI leaderboard.
"""
from __future__ import annotations
import itertools, json, math, os, sys, csv, tempfile
from collections import defaultdict, Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
bridge=corrected.old.bridge
OUT=Path(__file__).resolve().parent
START="2023-01-01";END="2023-04-30"
SOURCES=("data/a_class_12/2023_01_02/races.csv","data/a_class_12/2023_03_04/races.csv",
"data/2023/s_class_f1_all_parts/2023_q1/races.csv","data/2023/s_class_f1_all_parts/2023_q2/races.csv")
def binprice(v):return "<3" if v<3 else "3-5" if v<5 else "5-8" if v<8 else "8-12" if v<12 else "12+"
def load_lines():
    out={}
    for source in SOURCES:
        with (ROOT/source).open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                day=row.get("race_date","");raw=row.get("predicted_line_formation","")
                if not START<=day<=END or not raw:continue
                try:lines=tuple(tuple(int(z) for z in t.split("-")) for t in raw.split("/"))
                except ValueError:continue
                members=[z for line in lines for z in line]
                if len(members)<5 or len(members)>9 or len(set(members))!=len(members):continue
                klass="S" if "s_class" in source else "A"
                stage=row.get("race_type","").replace("Ａ級","").replace("Ｓ級","")
                shape="-".join(map(str,sorted((len(x) for x in lines),reverse=True)))
                key=(day,lines,klass,stage,shape)
                rid=row.get("race_id","")
                if rid in out and out[rid]!=key:out[rid]=None
                elif rid not in out:out[rid]=key
    return {k:v for k,v in out.items() if v is not None}
def role_features(w,lines):
    m={car:(i,j,len(line)) for i,line in enumerate(lines) for j,car in enumerate(line)}
    a,b,c=(m[z] for z in w)
    def r(x):return "H" if x[1]==0 else "B" if x[1]==1 else "T"
    return (r(a[0] if False else a),r(b),r(c),
            int(a[0]==b[0]),int(a[0]==c[0]),int(b[0]==c[0]),
            min(a[2],3),min(b[2],3),min(c[2],3))
def rectangles(cars,score,limit=25):
    """Induce whole Cartesian boxes from scored outcomes; no registered box types.
    The rectangle generator starts from high residual tickets, but expands each
    rectangle across ALL 3 axes, scoring its entire legal ticket set. Each
    resulting box is a genuine complete formation, not a list of singleton picks.
    """
    available=set(itertools.permutations(cars,3))
    remaining=set(available)
    forms=[]
    for _ in range(limit):
        if not remaining:break
        seed=max(remaining,key=lambda w:(score.get(w,-1e10),w))
        if score.get(seed,-1e10)<=0:break
        axes=[{x} for x in seed]
        def members(ax):
            return {w for w in itertools.product(*map(sorted,ax)) if len(set(w))==3}
        def objective(ax):
            ws=members(ax)
            return sum(score.get(w,-1e10) for w in ws if w in remaining)
        while True:
            opts=[]
            for j in range(3):
                for car in cars:
                    if car in axes[j]:continue
                    test=[set(x) for x in axes];test[j].add(car)
                    ws=members(test)
                    if not ws or len(ws & remaining)<2:continue
                    val=objective(test)
                    if val>objective(axes)+1e-9:opts.append((val,-len(ws),j,car,test))
            if not opts:break
            axes=max(opts,key=lambda x:x[:4])[4]
        ws=members(axes)&remaining
        if len(ws)<4:break
        forms.append(tuple(tuple(sorted(a)) for a in axes))
        remaining-=ws
    return forms
def union(forms):
    return set().union(*(set(w for w in itertools.product(*f) if len(set(w))==3) for f in forms)) if forms else set()
def main():
    lines=load_lines();audit=Counter();records=[]
    labels,_=bridge.verified_results(2023)
    catalog,_=bridge.catalog(START,END)
    temp,key=bridge.secure_key_temp()
    try:
        for day,files in catalog.items():
            merged={};bad=set()
            with tempfile.TemporaryDirectory() as tmp:
                for path,sha in files:
                    for rid,rows in bridge.decrypt_sealed(path,sha,key,tmp).items():
                        if rid in bad:continue
                        if rid in merged and sorted(merged[rid])!=sorted(rows):
                            bad.add(rid);merged.pop(rid,None);continue
                        merged[rid]=rows
            for rid,rows in merged.items():
                if rid not in lines or rid not in labels:continue
                line=lines[rid]
                if line[0]!=day:continue
                odds={}
                for row in rows:
                    if row[2]!="3rentan":continue
                    lo,hi=float(row[4]),float(row[5])
                    if lo!=hi:continue
                    w=tuple(map(int,row[3].split("-")));odds[w]=lo
                n=len(set(z for w in odds for z in w))
                if len(odds)!=n*(n-1)*(n-2):continue
                finish=tuple(labels[rid])
                if finish not in odds:continue
                linecars=set(z for l in line[1] for z in l)
                if linecars!=set(z for w in odds for z in w):continue
                records.append((day,rid,line,odds,finish))
            audit["days"]+=1
            if audit["days"]%20==0:print("LOADED",day,len(records),flush=True)
    finally:temp.cleanup()
    # Discovery from outcome-first empirical conditional distributions.
    # Weighted price ratio of whole outcome role-patterns, learned on Jan-Feb only.
    train=[r for r in records if r[0]<="2023-02-28"]
    test=[r for r in records if r[0]>="2023-03-01"]
    hist=defaultdict(lambda:defaultdict(lambda:[0.,0.,0]))
    for day,rid,line,odds,finish in train:
        _,ls,klass,stage,shape=line
        cell=(klass,stage,shape,binprice(min(odds.values())))
        invsum=sum(1/x for x in odds.values())
        for w,o in odds.items():
            f=role_features(w,ls)
            hist[cell][f][1]+=(1/o)/invsum
        hist[cell][role_features(finish,ls)][0]+=1
        hist[cell][role_features(finish,ls)][2]+=1
    # The empirical ratio is regularized to prevent small-cell overfit;
    # price-aware objective is applied to all legal ordered outcomes.
    def predict(record):
        day,rid,line,odds,finish=record
        _,ls,klass,stage,shape=line
        cell=(klass,stage,shape,binprice(min(odds.values())))
        stats=hist.get(cell,{})
        if sum(v[0] for v in stats.values())<30:return [],set()
        cars=tuple(sorted({z for w in odds for z in w}))
        invsum=sum(1/x for x in odds.values())
        # Relative frequency corrected by prior mass, with 30 race pseudo-counts.
        q={}
        for w,o in odds.items():
            f=role_features(w,ls)
            obs,market,_=stats.get(f,(0,0,0))
            total=sum(v[0] for v in stats.values())
            multiplier=(obs+30*market/max(total,1))/(market+30*market/max(total,1)) if market>0 else 1.
            # A ticket's expected value must exceed break-even, not merely exceed market frequency.
            q[w]=((1/o)/invsum)*multiplier*o-1.
        forms=rectangles(cars,q)
        return forms,union(forms)
    def summarize(part, period_name):
        sums=Counter();examples=[];details=[]
        for r in part:
            forms,ws=predict(r)
            if not ws:continue
            day,rid,line,odds,finish=r
            _,ls,klass,stage,shape=line
            npts=len(ws)
            hit=int(finish in ws)
            stake=100*npts
            payout=round(100*odds[finish]*hit)
            sums["races"]+=1
            sums["points"]+=npts
            sums["stake"]+=stake
            sums["hits"]+=hit
            sums["payout"]+=payout
            if len(examples)<12:
                examples.append({"race_id":rid,"date":day,"line":ls,"forms":forms,
                                 "points":npts,"hit":hit,"payout":payout})
            if period_name=="holdout":
                details.append({
                    "race_id":rid,"date":day,"month":day[:7],
                    "class":klass,"race_type":stage,"line_shape":shape,
                    "published_lines":["-".join(map(str,line)) for line in ls],
                    "favorite_odds":min(odds.values()),
                    "full_market_tickets":len(odds),
                    "formation_component_count":len(forms),
                    "formations":[{"first":list(f[0]),"second":list(f[1]),"third":list(f[2])} for f in forms],
                    "tickets":npts,"stake_yen":stake,
                    "result":list(finish),"hit":bool(hit),
                    "winning_ticket_odds":odds[finish],
                    "payout_yen":payout,"profit_yen":payout-stake,
                    "market_ticket_support": sum((1/odds[w] for w in ws))/sum(1/o for o in odds.values()),
                })
        summary={"races":sums["races"],"tickets":sums["points"],
                "stake_yen":sums["stake"],"payout_yen":round(sums["payout"]),
                "profit_yen":round(sums["payout"]-sums["stake"]),
                "roi_pct":round(100*sums["payout"]/sums["stake"],2) if sums["stake"] else None,
                "examples":examples}
        return summary,details
    def audit_holdout(details, original):
        # This audit never changes the original ticket generation, score, or selection.
        assert len(details)==original["races"]
        assert sum(v["tickets"] for v in details)==original["tickets"]
        assert sum(v["stake_yen"] for v in details)==original["stake_yen"]
        assert sum(v["payout_yen"] for v in details)==original["payout_yen"]
        assert sum(v["profit_yen"] for v in details)==original["profit_yen"]
        assert original["races"]==479 and original["tickets"]==19088
        assert original["stake_yen"]==1908800 and original["payout_yen"]==2131630
        def roi(p,s):return round(100*p/s,2) if s else None
        gross=original["payout_yen"];stake=original["stake_yen"]
        winners=sorted((r for r in details if r["hit"]),
             key=lambda r:(-r["payout_yen"],r["date"],r["race_id"]))
        assert gross==sum(r["payout_yen"] for r in winners)
        eliminated=[]
        for k in (0,1,3,5,10,20):
            top=winners[:k]
            payout_removed=sum(r["payout_yen"] for r in top)
            stake_removed=sum(r["stake_yen"] for r in top)
            eliminated.append({"top_winning_races_removed":k,
                "eliminated_payout_yen":payout_removed,
                "fraction_of_all_payout_pct":round(100*payout_removed/gross,2) if gross else None,
                "stress_all_stakes_kept_roi_pct":roi(gross-payout_removed,stake),
                "stress_all_stakes_kept_profit_yen":gross-payout_removed-stake,
                "omit_entire_top_races_roi_pct":roi(gross-payout_removed,stake-stake_removed),
                "omit_entire_top_races_profit_yen":gross-payout_removed-(stake-stake_removed)})
        def group_rows(key_fn):
            grouped=defaultdict(list)
            for r in details:grouped[key_fn(r)].append(r)
            result=[]
            for key,rows in grouped.items():
                stake=sum(r["stake_yen"] for r in rows)
                pay=sum(r["payout_yen"] for r in rows)
                result.append({"group":key,"races":len(rows),
                    "hits":sum(r["hit"] for r in rows),
                    "tickets":sum(r["tickets"] for r in rows),
                    "stake_yen":stake,"payout_yen":pay,"profit_yen":pay-stake,
                    "roi_pct":roi(pay,stake),
                    "average_points":round(sum(r["tickets"] for r in rows)/len(rows),2)})
            return sorted(result,key=lambda x:x["group"])
        def points_band(n):
            return "01_1-09" if n<10 else "02_10-19" if n<20 else "03_20-29" if n<30 else "04_30-39" if n<40 else "05_40-49" if n<50 else "06_50+"
        def favorite_band(o):
            return "01_under3" if o<3 else "02_3-5" if o<5 else "03_5-8" if o<8 else "04_8-12" if o<12 else "05_12+"
        gross_per_hit=[r["payout_yen"] for r in winners]
        import statistics
        total_hits=len(winners)
        return {
            "methodology":"Every previously generated full formation is unchanged. All 479 historical wagers included at 100 yen per ticket. This audit adds complete per-race accounting, no ticket removal or rule fitting.",
            "reconciliation":{"races":len(details),"total_tickets":sum(r["tickets"] for r in details),
                "stake_yen":stake,"payout_yen":gross,"profit_yen":gross-stake,
                "roi_pct":roi(gross,stake),"hits":total_hits,
                "hit_rate_pct":round(100*total_hits/len(details),2)},
            "payout_concentration":eliminated,
            "winning_race_payout_distribution":{
                "hit_count":total_hits,
                "median_payout_yen":statistics.median(gross_per_hit) if gross_per_hit else None,
                "mean_payout_yen":round(statistics.mean(gross_per_hit),2) if gross_per_hit else None,
                "max_payout_yen":max(gross_per_hit) if gross_per_hit else None,
                "median_winning_ticket_odds":statistics.median([r["winning_ticket_odds"] for r in winners]) if winners else None,
                "winning_races_that_are_individually_profitable":sum(r["profit_yen"]>0 for r in winners),
                "winning_races_that_are_individually_unprofitable":sum(r["profit_yen"]<0 for r in winners)},
            "biggest_winners":winners[:20],
            "by_month":group_rows(lambda r:r["month"]),
            "by_class":group_rows(lambda r:r["class"]),
            "by_race_type":group_rows(lambda r:r["class"]+"|"+r["race_type"]),
            "by_line_shape":group_rows(lambda r:r["line_shape"]),
            "by_favorite_band":group_rows(lambda r:favorite_band(r["favorite_odds"])),
            "by_point_band":group_rows(lambda r:points_band(r["tickets"])),
            "by_formation_count":group_rows(lambda r:str(r["formation_component_count"])),
            "by_four_axes":group_rows(lambda r:r["class"]+"|"+r["race_type"]+"|"+r["line_shape"]+"|"+favorite_band(r["favorite_odds"])),
            "all_479_wagers":sorted(details,key=lambda r:(r["date"],r["race_id"]))
        }
    OUT.mkdir(parents=True,exist_ok=True)
    train_summary,_=summarize(train,"train")
    test_summary,holdout_details=summarize(test,"holdout")
    deep_audit=audit_holdout(holdout_details,test_summary)
    result={"status":"EXPERIMENTAL_OUTCOME_FIRST_UNION_DISCOVERY","audit":dict(audit),"records":len(records),"training_records":len(train),"test_records":len(test),"method":"Learn empirical role co-occurrence vs market in Jan-Feb, then generate novel Cartesian union boxes from expected payout scores; fixed holdout Mar-Apr.","train":train_summary,"holdout":test_summary,"caveats":["Exploratory, no fixed 34 forms.","Outcome role buckets still an inductive bias; not a proof of global optimal formation.","Uses final odds; not live betting.","No fixed selection from holdout outcomes."]}
    (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (OUT/"profit_concentration_full.json").write_text(json.dumps(deep_audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    short_audit={k:v for k,v in deep_audit.items() if k not in ("all_479_wagers","biggest_winners","by_four_axes")}
    (OUT/"profit_concentration_summary.json").write_text(json.dumps(short_audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PROFIT_AUDIT",json.dumps(short_audit,ensure_ascii=False),flush=True)
    print("DISCOVERY_RESULT",json.dumps({k:v for k,v in result.items() if k not in ("train","holdout")}),flush=True)
    print("TRAIN",json.dumps({k:v for k,v in result["train"].items() if k!="examples"}),flush=True)
    print("HOLDOUT",json.dumps({k:v for k,v in result["holdout"].items() if k!="examples"}),flush=True)
if __name__=="__main__":main()
