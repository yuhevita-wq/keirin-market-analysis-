"""Sequential market observation inspired by Sundali & Croson (2006).
All archived dates; no betting model changes. Only aggregate exports.
"""
import sys,json,math,os
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
old=corrected.old
OUT=ROOT/"research/casino_sequence_observation_20261028"
RECORDS={}
AUDIT=Counter()
BANDS=[5,10,20,50,100,300,1000]
def band(o):return next((str(c) for c in BANDS if o<=c),"above1000")
def capture(rid,day,books,finish,rtype,source):
    cars=sorted({c for t in books["3rentan"] for c in t})
    if len(finish)!=3 or not set(finish).issubset(cars):
        AUDIT["invalid_result"]+=1;return
    import itertools
    expected={"2shatan":set(itertools.permutations(cars,2)),
      "2shafuku":set(itertools.combinations(cars,2)),
      "3rentan":set(itertools.permutations(cars,3)),
      "3renpuku":set(itertools.combinations(cars,3))}
    markets={}
    for kind in expected:
        b=books.get(kind,{})
        if set(b)!=expected[kind] or any(not math.isfinite(x) or x<1 for x in b.values()):
            AUDIT["invalid_"+kind]+=1;continue
        z=sum(1/x for x in b.values());q={t:1/o/z for t,o in b.items()}
        target=(tuple(finish[:2]) if kind=="2shatan" else tuple(sorted(finish[:2]))
          if kind=="2shafuku" else tuple(finish) if kind=="3rentan" else tuple(sorted(finish)))
        markets[kind]={"top":max(q.values()),"hhi":sum(p*p for p in q.values()),
          "entropy":-sum(p*math.log(p) for p in q.values())/math.log(len(q)),
          "winning_odds":b[target],"winning_rank":1+sum(o<b[target] for o in b.values())}
        if kind in ("2shatan","3rentan"):
            markets[kind]["first"]={c:sum(p for t,p in q.items() if t[0]==c) for c in cars}
    if len(markets)!=4:AUDIT["incomplete_four_markets"]+=1;return
    obj={"day":day,"venue":rid[:2],"number":int(rid[-2:]),"type":rtype,
      "cars":len(cars),"winner":finish[0],"markets":markets}
    if rid in RECORDS and RECORDS[rid]!=obj:raise RuntimeError("Conflicting race ID")
    RECORDS[rid]=obj
    AUDIT["usable_races"]+=1;AUDIT["year_"+day[:4]]+=1
def add(s,values):
    s["n"]+=1
    for k,v in values.items():s[k]+=v
def finish_report():
    grouped=defaultdict(dict)
    for rid,r in RECORDS.items():
        k=(r["day"],r["venue"])
        if r["number"] in grouped[k]:raise RuntimeError("Duplicate race number")
        grouped[k][r["number"]]=r
    pairs=[]
    for races in grouped.values():
        for n,r in sorted(races.items()):
            if n-1 not in races:
                AUDIT["no_immediate_previous_usable_race"]+=1;continue
            pairs.append((races[n-1],r))
    AUDIT["adjacent_same_venue_day_pairs"]=len(pairs)
    stats=defaultdict(Counter);strata=defaultdict(lambda:defaultdict(Counter))
    for prev,cur in pairs:
        year=cur["day"][:4]
        for kind,m in cur["markets"].items():
            pm=prev["markets"][kind];b=band(pm["winning_odds"])
            values={k:m[k] for k in ("top","hhi","entropy")}
            values["previous_winning_rank"]=pm["winning_rank"]
            for y in ("ALL",year):add(stats[(y,kind,b)],values)
            sk=(year,cur["venue"],cur["number"],cur["type"],cur["cars"],kind)
            add(strata[sk][b],{k:m[k] for k in ("top","hhi","entropy")})
            if "first" in m:
                car=prev["winner"]
                if car not in m["first"]:
                    AUDIT["previous_winner_car_absent_"+kind]+=1;continue
                q=m["first"][car];hit=int(cur["winner"]==car)
                for y in ("ALL",year):
                    add(stats[(y,kind,"previous_winner_car")],
                        {"market_first_probability":q,"actual_first":hit})
                    add(stats[(y,kind,"previous_winner_car_"+str(car))],
                        {"market_first_probability":q,"actual_first":hit})
    # Remove mean differences across observed year/venue/race number/type/car-count strata.
    # Association only: no rider attributes, causal interpretation, or matched treatment design.
    residual=defaultdict(Counter)
    for sk,groups in strata.items():
        total=sum(x["n"] for x in groups.values())
        mean={k:sum(x[k] for x in groups.values())/total for k in ("top","hhi","entropy")}
        for b,s in groups.items():
            vals={k:s[k]/s["n"]-mean[k] for k in mean}
            for year in ("ALL",sk[0]):
                cell=residual[(year,sk[-1],b)]
                cell["n"]+=s["n"]
                cell["strata"]+=1
                cell["multi_band_strata"]+=int(len(groups)>1)
                for k,v in vals.items():cell[k]+=v*s["n"]
    def rows(d):
        return [{"group":list(key),"n":s["n"],
          **{k: v/s["n"] for k,v in s.items() if k not in ("n","strata","multi_band_strata")},
          **{k:int(s[k]) for k in ("strata","multi_band_strata") if k in s}}
          for key,s in sorted(d.items())]
    report={"status":"COMPLETE","paper":"Sundali and Croson (2006), Biases in casino betting",
      "source_commit":os.environ.get("GITHUB_SHA"),"generated_utc":datetime.now(timezone.utc).isoformat(),
      "scope":"2021-01-01 through 2026-06-30, all available validated four-market non-girls archives",
      "audit":dict(AUDIT),"archive_audit":dict(old.audit),
      "orientation_checks":list(corrected.VALIDATED.values()),
      "previous_winning_odds_band_upper_bounds_inclusive":BANDS,
      "raw_groups":rows(stats),"stratum_centered_groups":rows(residual),
      "stratum_fields":["year","venue","current_race_number","current_race_type","current_car_count","market"],
      "limits":["Descriptive associations only; not proof of gambler psychology, causality or profit.",
        "No individual purchase histories; hot hand cannot be observed.",
        "Only final odds; cannot measure before/after response to preceding result or confirm result-publication timing.",
        "Inverse odds normalized within each market are support proxies, not exact turnover shares.",
        "Different riders and field strength may explain associations, even after stratum centering.",
        "No significance claims; dependent races and multiple comparisons not tested.",
        "Missing previous races break adjacency; no jump across gaps, venues or dates.",
        "Markets use separately realized winning-odds bands; no arbitrary period or class restriction.",
        "Same car number is not same rider; car-specific rows expose numbering effects.",
        "Original archives and AI/purchase rules unchanged."]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in report.items() if k not in ("raw_groups","stratum_centered_groups","orientation_checks")}
    compact["raw_groups"]=[r for r in report["raw_groups"] if r["group"][0]=="ALL"]
    compact["stratum_centered_groups"]=[r for r in report["stratum_centered_groups"] if r["group"][0]=="ALL"]
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(compact,ensure_ascii=False),flush=True)
def progress(status):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"progress.json").write_text(json.dumps({"status":status,"audit":dict(AUDIT),
      "archive_audit":dict(old.audit)},ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":
    if not os.environ.get(old.bridge.SECRET_ENV):raise RuntimeError("Full archive key unavailable")
    corrected.base_observe=capture
    old.save=progress
    try:
        old.sealed()
        finish_report()
        progress("COMPLETE")
    except Exception:
        progress("FAILED_PARTIAL_NOT_CONCLUSION")
        raise
