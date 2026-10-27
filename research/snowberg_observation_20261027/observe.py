"""Descriptive cross-pool observation inspired by Snowberg & Wolfers (2010).
Not a reproduction of their competing structural models; no model training.
Only aggregate outputs; plaintext quotes and keys never leave temporary memory.
"""
import sys, json, math, itertools, tempfile, gzip, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.deletion_ai import secure_seven_bet_bridge as bridge
OUT=ROOT/"research/snowberg_observation_20261027"
CUTS=[5,10,20,50,100,300,1000]
stats=defaultdict(lambda:Counter())
audit=Counter()
def binof(x,cuts=CUTS):
    return next((i for i,c in enumerate(cuts) if x<=c),len(cuts))
def add(key,hit,q,odds):
    s=stats[key];s["tickets"]+=1;s["hits"]+=hit;s["market_expected_hits"]+=q
    s["return_units"]+=hit*odds
def observe(rid,day,books,finish,rtype,source):
    year=day[:4]
    cars=sorted({c for t in books["3rentan"] for c in t})
    n=len(cars)
    if len(finish)!=3 or not set(finish).issubset(cars):
        audit["result_not_in_book"]+=1;return
    expected={"3rentan":set(itertools.permutations(cars,3)),
              "3renpuku":set(itertools.combinations(cars,3)),
              "2shatan":set(itertools.permutations(cars,2)),
              "2shafuku":set(itertools.combinations(cars,2))}
    qs={}
    for kind,book in books.items():
        if kind not in expected:continue
        if set(book)!=expected[kind] or any(not math.isfinite(v) or v<1 for v in book.values()):
            audit["invalid_book_"+kind]+=1;continue
        z=sum(1/v for v in book.values())
        qs[kind]={t:(1/v)/z for t,v in book.items()}
    if not {"3rentan","3renpuku"}.issubset(qs):
        audit["incomplete_three_markets"]+=1;return
    audit["usable_races_"+source]+=1;audit["year_"+year]+=1
    for kind,q in qs.items():
        target=tuple(finish[:2]) if kind=="2shatan" else tuple(sorted(finish[:2])) if kind=="2shafuku" else tuple(finish) if kind=="3rentan" else tuple(sorted(finish))
        for t,p in q.items():
            hit=int(t==target);o=books[kind][t]
            add((source,year,kind,"price",str(binof(o))),hit,p,o)
            add((source,year,kind,"price_cars",str(binof(o)),str(n)),hit,p,o)
            add((source,year,kind,"price_type",str(binof(o)),rtype),hit,p,o)
    if {"2shatan","2shafuku"}.issubset(qs):
        q2,qf=qs["2shatan"],qs["2shafuku"]
        for a,b in itertools.combinations(cars,2):
            pair=(a,b);mass=q2[a,b]+q2[b,a];p=qf[pair]
            gap=math.log(mass/p)
            gb=binof(gap,[-.15,-.05,.05,.15])
            add((source,year,"2shafuku","price_cross",str(binof(books["2shafuku"][pair])),str(gb)),
                int(pair==tuple(sorted(finish[:2]))),p,books["2shafuku"][pair])
        # Conditional order check: precisely one realized top-two pair per race.
        a,b=sorted(finish[:2]);p=q2[a,b]/(q2[a,b]+q2[b,a])
        s=stats[(source,year,"2shatan","conditional_order",str(binof(p,[.1,.2,.3,.4,.5,.6,.7,.8,.9])))]
        s["races"]+=1;s["predicted_a_first"]+=p;s["actual_a_first"]+=int(finish[0]==a)
    q3,qt=qs["3rentan"],qs["3renpuku"]
    for t,p in qt.items():
        mass=sum(q3[o] for o in itertools.permutations(t))
        gb=binof(math.log(mass/p),[-.15,-.05,.05,.15])
        add((source,year,"3renpuku","price_cross",str(binof(books["3renpuku"][t])),str(gb)),
            int(t==tuple(sorted(finish))),p,books["3renpuku"][t])
def sealed():
    temp,key=bridge.secure_key_temp()
    try:
        for year in range(2021,2027):
            labels,ls=bridge.verified_results(year)
            audit["label_count_"+str(year)]=ls["verified_distinct_results"]
            first=f"{year}-01-01";last=min(f"{year}-12-31","2026-06-30")
            catalog,_=bridge.catalog(first,last)
            for day,files in catalog.items():
                union={};conflict=set()
                with tempfile.TemporaryDirectory(prefix="private_observation_") as tmp:
                    for path,sha in files:
                        for rid,rows in bridge.decrypt_sealed(path,sha,key,tmp).items():
                            if rid in conflict:continue
                            canonical=sorted(rows)
                            if rid in union and union[rid]!=canonical:
                                union.pop(rid);conflict.add(rid)
                            else:union[rid]=canonical
                audit["conflicting_books"]+=len(conflict)
                for rid,rows in union.items():
                    # Reuse existing date/type/girls/full-grid validation.
                    norm,reason=bridge.normalize_book(rid,day,rows)
                    if reason:audit[reason]+=1;continue
                    if rid not in labels:audit["missing_result"]+=1;continue
                    books=defaultdict(dict)
                    for row in rows:
                        kind=row[2]
                        if kind not in {"2shatan","2shafuku","3rentan","3renpuku"}:continue
                        lo,hi=float(row[4]),float(row[5])
                        if lo!=hi:raise ValueError("Non point price")
                        t=tuple(map(int,row[3].split("-")))
                        if kind in {"2shafuku","3renpuku"}:t=tuple(sorted(t))
                        books[kind][t]=lo
                    observe(rid,day,books,labels[rid],rows[0][1],"sealed_four_markets")
                audit["days"]+=1
                if audit["days"]%31==0:
                    print("PROGRESS",year,audit["days"],audit["usable_races_sealed_four_markets"],flush=True)
                    save("IN_PROGRESS")
    finally:temp.cleanup()
def fallback():
    # Explicit two-market fallback, never silently presented as full archive.
    for year in range(2021,2027):
        labels,_=bridge.verified_results(year)
        path=ROOT/f"data/odds_only/odds_{year}.jsonl.gz"
        if not path.exists():continue
        seen={};conflict=set()
        with gzip.open(path,"rt") as f:
            for line in f:
                row=json.loads(line);rid=str(row["race_id"])
                digest=hashlib.sha256(json.dumps(row["odds"],sort_keys=True).encode()).hexdigest()
                if rid in seen and seen[rid][0]!=digest:conflict.add(rid)
                else:seen[rid]=(digest,row)
        for rid,(_,row) in seen.items():
            if rid in conflict:audit["fallback_conflicts"]+=1;continue
            finish=labels.get(rid);day=bridge.race_day(rid)
            if not finish or not day or not "2021-01-01"<=day<="2026-06-30":
                audit["fallback_missing_label_or_date"]+=1;continue
            books={}
            for jp,kind,sep in [("3連単","3rentan","-"),("3連複","3renpuku","=")]:
                books[kind]={tuple(map(int,k.split(sep))):float(v) for k,v in row["odds"].get(jp,{}).items()}
            observe(rid,day,books,finish,"unknown_not_in_plaintext_export","plaintext_two_market_fallback")
def save(status):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key,s in sorted(stats.items()):
        r={"group":list(key),**dict(s)}
        if s["tickets"]:
            r["roi_pct_final_odds"]=100*s["return_units"]/s["tickets"]
            r["observed_over_market"]=s["hits"]/s["market_expected_hits"] if s["market_expected_hits"] else None
        if s["races"]:
            r["predicted_order_rate"]=s["predicted_a_first"]/s["races"]
            r["actual_order_rate"]=s["actual_a_first"]/s["races"]
        rows.append(r)
    report={"status":status,"generated_utc":datetime.now(timezone.utc).isoformat(),
      "source_commit":__import__("os").environ.get("GITHUB_SHA","local"),
      "audit":dict(audit),"price_upper_bounds_inclusive":CUTS,
      "cross_log_ratio_cuts":[-.15,-.05,.05,.15],"groups":rows,
      "limits":["Descriptive observation, not reproduction of risk-love vs misperception structural tests.",
      "Normalized inverse odds are market proxies, not true probabilities.",
      "Final odds return proxy, not official payout or executable T-15 return.",
      "All years descriptive; none certified untouched. No model or purchase rules modified.",
      "No rider-role or line join in this observation. Wide intervals and frame markets not analyzed.",
      "No adjustment for intraday dependence or multiple comparisons; no significance claim.",
      "Plaintext fallback supports only two markets; its race-type and girls metadata are not independently reverified."]}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (OUT/"README.md").write_text("# Snowberg–Wolfers inspired descriptive observation\n\nStatus: "+status+
      "\n\nSee summary.json for exact scope, counts, exclusions and limitations.\n"+
      "No model training, purchase selection, or racing-site collection. Only aggregates are published.\n")
def main():
    import os
    if os.environ.get(bridge.SECRET_ENV):
        try:sealed()
        except Exception:
            save("SEALED_OBSERVATION_FAILED_PARTIAL");raise
        save("SEALED_FOUR_MARKET_OBSERVATION_COMPLETE")
    else:
        audit["key_unavailable_four_market_observation_not_run"]=1
        fallback();save("PARTIAL_TWO_MARKET_ONLY_KEY_UNAVAILABLE")
    print("SAVED",OUT/"summary.json","AUDIT",json.dumps(dict(audit)),flush=True)
if __name__=="__main__":main()
