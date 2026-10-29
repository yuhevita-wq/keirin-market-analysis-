#!/usr/bin/env python3
"""2023 no-odds full-trifecta probability observation and chronological test.

No odds/payout inputs anywhere. Training: Jan-Oct 2023; holdout: Nov-Dec.
The archived published line is a PREDICTION, not certified on-track formation.
Works only on complete, result-matched cards with published pre-race line.
"""
import csv, gzip, itertools, json, math, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/audits/independent_trifecta_observation_2023.json"
SAMPLE = ROOT / "data/audits/independent_trifecta_sample_2023.json"
RESULT = ROOT / "data/result_universe_v2/combined/results_2023.jsonl.gz"
TRAIN_END = "2023-10-31"
ALPHA_ROLE, ALPHA_FULL = 40.0, 80.0

def csv_rows(stream):
    yield from csv.DictReader(stream)

def sources():
    paths = list(sorted((ROOT/"data/a_class_12").glob("2023_*/races.csv")))
    paths += list(sorted((ROOT/"data/a3_challenge").glob("2023_*/races.csv")))
    paths += list(sorted((ROOT/"data/2023").glob("*/**/races.csv")))
    seen=set()
    for p in paths:
        ep=p.with_name("entries.csv")
        if ep.exists() and p not in seen:
            seen.add(p)
            yield str(p.relative_to(ROOT)), lambda p=p: (
                csv_rows(p.open(encoding="utf-8-sig", newline="")),
                csv_rows(p.with_name("entries.csv").open(encoding="utf-8-sig",newline="")))
    for z in sorted((ROOT/"data/grade_races").rglob("*.zip")):
        if "2023" not in str(z): continue
        with zipfile.ZipFile(z) as arch:
            names=[x for x in arch.namelist() if x.endswith("/races.csv") or x=="races.csv"]
            for name in names:
                entry=name[:-9]+"entries.csv"
                if entry not in arch.namelist(): continue
                yield str(z.relative_to(ROOT))+":"+name, lambda z=z,name=name,entry=entry: zip_pair(z,name,entry)

def zip_pair(z,name,entry):
    import io
    with zipfile.ZipFile(z) as arch:
        a=io.StringIO(arch.read(name).decode("utf-8-sig"))
        b=io.StringIO(arch.read(entry).decode("utf-8-sig"))
    return csv_rows(a),csv_rows(b)

def parse_line(s,cars):
    groups=[]
    for block in str(s or "").split("/"):
        parts=block.strip().split("-")
        if not parts or not all(x.isdigit() and len(x)==1 for x in parts): return None
        groups.append(parts)
    flat=[c for g in groups for c in g]
    if len(flat)!=len(set(flat)) or set(flat)!=set(cars): return None
    return groups

def ingest():
    orders={}
    with gzip.open(RESULT,"rt",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line); top=row.get("finish_order")
            if isinstance(top,list) and len(top)>=3:
                top=tuple(map(str,top[:3]))
                if len(set(top))==3 and all(s.isdigit() for s in top):
                    orders[str(row["race_id"])]=top
    manifests={};cards=defaultdict(dict);audit=Counter()
    for path,factory in sources():
        race_rows,entry_rows=factory()
        valid_ids=set()
        for row in race_rows:
            rid=str(row.get("race_id","")).strip()
            day=str(row.get("race_date","")).strip()
            if rid not in orders or not ("2023-01-01"<=day<="2023-12-31"): continue
            if rid not in manifests:
                manifests[rid]={"date":day,"line":str(row.get("predicted_line_formation") or ""),
                                "status":str(row.get("line_status") or ""),"source":path}
            else: audit["duplicated_race_manifests"]+=1
            valid_ids.add(rid)
        accepted=0
        for row in entry_rows:
            rid=str(row.get("race_id","")).strip()
            if rid not in valid_ids:continue
            car=str(row.get("car_no","")).strip()
            if not car.isdigit(): continue
            try:
                score=float(row.get("score") or "")
                if not math.isfinite(score):raise ValueError()
            except ValueError:
                audit["invalid_score_rows"]+=1
                continue
            if car not in cards[rid]:
                cards[rid][car]=score;accepted+=1
            else: audit["duplicated_card_rows"]+=1
        print("LOADED",path,accepted,flush=True)
    eligible={}
    for rid,metadata in manifests.items():
        runners=cards.get(rid,{})
        if len(runners)<4 or not set(orders[rid]).issubset(runners):
            audit["missing_card_or_podium"]+=1;continue
        if metadata["status"]!="published":
            audit["missing_published_line"]+=1;continue
        groups=parse_line(metadata["line"],runners)
        if groups is None:
            audit["invalid_line_coverage"]+=1;continue
        positions={car:(g+1,p+1,len(group)) for g,group in enumerate(groups) for p,car in enumerate(group)}
        sorted_score=sorted(runners,key=lambda c:(-runners[c],int(c)))
        ranks={car:min(i+1,4) for i,car in enumerate(sorted_score)}
        eligible[rid]={"date":metadata["date"],"order":orders[rid],"scores":runners,
                       "positions":positions,"groups":groups,"ranks":ranks}
    eligible=dict(sorted(eligible.items(),key=lambda x:(x[1]["date"],x[0])))
    return eligible,{"verified_2023_results":len(orders),
                     "races_with_manifest":len(manifests),"eligible_races":len(eligible),
                     "audit":dict(audit),"source_note":"Exact-ID join, 2023 original folders and grade ZIPs; never sum duplicate source folders"}

def group_rel(e,c,a):
    if a is None:return "none"
    g,pos,_=e["positions"][c];h,hpos,_=e["positions"][a]
    return ("diff" if g!=h else ("behind" if pos>hpos else "ahead"))

def role(e,c):
    _,pos,size=e["positions"][c]
    return (min(pos,4),min(size,4))

def keys(e,c,prefix,include_score):
    st=len(prefix)+1
    rel=tuple(group_rel(e,c,a) for a in prefix)
    base=(st,role(e,c),rel)
    full=base+((e["ranks"][c],) if include_score else ())
    return base,full

def train_model(train,with_score):
    base_stats=defaultdict(lambda:[0,0])
    full_stats=defaultdict(lambda:[0,0])
    size_stats=defaultdict(lambda:[0,0])
    for e in train.values():
        cars=e["scores"]
        for st in (1,2,3):
            prefix=e["order"][:st-1]; target=e["order"][st-1]
            for c in cars:
                if c in prefix:continue
                base,full=keys(e,c,prefix,with_score)
                y=int(c==target)
                base_stats[base][0]+=y;base_stats[base][1]+=1
                full_stats[full][0]+=y;full_stats[full][1]+=1
                size_stats[(st,len(cars))][0]+=y;size_stats[(st,len(cars))][1]+=1
    return base_stats,full_stats,size_stats

def prob(e,c,prefix,model,with_score):
    base_stats,full_stats,size_stats=model
    st=len(prefix)+1
    base,full=keys(e,c,prefix,with_score)
    # Hierarchical empirical-Bayes smoothing, only training period counts.
    baseprior=1/(len(e["scores"])-len(prefix))
    a,n=base_stats.get(base,(0,0))
    b=(a+ALPHA_ROLE*baseprior)/(n+ALPHA_ROLE)
    if not with_score:return b
    a,n=full_stats.get(full,(0,0))
    return (a+ALPHA_FULL*b)/(n+ALPHA_FULL)

def distribution(e,model,with_score):
    cars=sorted(e["scores"],key=int)
    p1={};p2={};p3={}
    for a in cars:p1[a]=prob(e,a,(),model,with_score)
    z=sum(p1.values());p1={k:v/z for k,v in p1.items()}
    for a in cars:
        candidates=[b for b in cars if b!=a]
        arr={b:prob(e,b,(a,),model,with_score) for b in candidates}
        z=sum(arr.values())
        for b,v in arr.items():p2[a,b]=v/z
        for b in candidates:
            cs=[c for c in cars if c!=a and c!=b]
            arr3={c:prob(e,c,(a,b),model,with_score) for c in cs}
            z3=sum(arr3.values())
            for c,v in arr3.items():p3[a,b,c]=v/z3
    out={(a,b,c):p1[a]*p2[a,b]*v for (a,b,c),v in p3.items()}
    if abs(sum(out.values())-1)>1e-9:raise AssertionError("Probability conservation failed")
    if len(out)!=len(cars)*(len(cars)-1)*(len(cars)-2):
        raise AssertionError("Missing trifecta outcomes")
    return out

def observed_stats(events):
    counts=Counter();line_sizes=Counter();winning_roles=Counter()
    for e in events.values():
        a,b,c=e["order"]
        p=e["positions"]
        groups=[p[x][0] for x in (a,b,c)]
        counts["races"]+=1
        if groups[0]==groups[1]:counts["same_line_12"]+=1
        if groups[0]==groups[2]:counts["same_line_13"]+=1
        if groups[1]==groups[2]:counts["same_line_23"]+=1
        if len(set(groups))==1:counts["same_line_123"]+=1
        if len(set(groups))==3:counts["three_distinct_lines"]+=1
        line_sizes["-".join(map(str, sorted([len(x) for x in e["groups"]],reverse=True)))]+=1
        winning_roles["place1_position_"+str(p[a][1])]+=1
        winning_roles["place2_position_"+str(p[b][1])]+=1
        winning_roles["place3_position_"+str(p[c][1])]+=1
    n=counts["races"]
    return {"n":n,"rates_pct":{k:round(v*100/n,5) for k,v in counts.items() if k!="races"},
            "line_shape_counts":dict(line_sizes.most_common()),"winning_role_counts":dict(winning_roles)}

def evaluate(events,models):
    result={}
    samples={}
    for name,model,scored in [("uniform",None,False),("line_role_only",models[0],False),
                               ("line_role_score_rank",models[1],True)]:
        loss=0.;hit1=0;hit3=0;brier=0;sum_p=0
        buckets=defaultdict(lambda:[0,0.,0])
        for rid,e in events.items():
            if model is None:
                n=len(e["scores"]);p=1/(n*(n-1)*(n-2))
                dist={triple:p for triple in itertools.permutations(sorted(e["scores"],key=int),3)}
            else: dist=distribution(e,model,scored)
            actual=e["order"]; pa=dist[actual]
            loss-=math.log(max(pa,1e-15))
            sum_p+=pa
            ordering=sorted(dist,key=lambda k:(-dist[k],k))
            hit1+=ordering[0]==actual
            hit3+=actual in ordering[:3]
            brier+=(1.-2.*pa+sum(v*v for v in dist.values()))
            for k,v in dist.items():
                band=("[0,.1%)" if v<.001 else "[.1,.5%)" if v<.005 else
                      "[.5,1%)" if v<.01 else "[1,2%)" if v<.02 else
                      "[2,5%)" if v<.05 else "[5,10%)" if v<.10 else "[10,100%]")
                z=buckets[band];z[0]+=1;z[1]+=v;z[2]+=int(k==actual)
            if name=="line_role_score_rank" and len(samples)==0 and len(e["scores"])==7:
                samples={"race_id":rid,"race_date":e["date"],"participants":7,
                         "actual_result":"-".join(actual),
                         "ticket_probabilities":[{"trifecta":"-".join(k),"probability_pct":round(v*100,8)}
                                  for k,v in sorted(dist.items())]}
        n=len(events)
        result[name]={"races":n,"nll":round(loss/n,7),"brier":round(brier/n,7),
                      "top1_hit_rate_pct":round(100*hit1/n,5),
                      "top3_hit_rate_pct":round(100*hit3/n,5),
                      "mean_probability_assigned_to_actual_pct":round(100*sum_p/n,5),
                      "calibration_bands":[{"band":k,"ticket_count":a,
                         "mean_predicted_pct":round(100*b/a,6),"empirical_hit_pct":round(100*c/a,6),
                         "hits":c} for k,(a,b,c) in sorted(buckets.items())]}
        print("EVALUATED",name,result[name]["nll"],flush=True)
    return result,samples

def main():
    events,audit=ingest()
    train={k:v for k,v in events.items() if v["date"]<=TRAIN_END}
    holdout={k:v for k,v in events.items() if v["date"]>TRAIN_END}
    if len(train)<5000 or len(holdout)<1000:raise RuntimeError("Insufficient clean 2023 train/holdout")
    print("SPLIT",len(train),len(holdout),flush=True)
    models=[train_model(train,False),train_model(train,True)]
    metrics,samples=evaluate(holdout,models)
    report={"state":"COMPLETED","title":"2023 odds-free empirical conditional triple probability model",
            "formula":"P(a,b,c|X)=P(a|X)P(b|a,X)P(c|a,b,X); hierarchical-smoothed conditional empirical scores normalized per choice set",
            "market_odds_in_training":False,"market_odds_in_evaluation":False,
            "year":2023,"train_period":["2023-01-01","2023-10-31"],
            "heldout_period":["2023-11-01","2023-12-31"],
            "counts":{"train":len(train),"heldout":len(holdout),**audit},
            "observed_2023":observed_stats(events),
            "observed_train":observed_stats(train),
            "observed_holdout":observed_stats(holdout),
            "holdout_metrics":metrics,
            "sample_7_rider_full_ticket_file":str(SAMPLE.relative_to(ROOT)),
            "fit_note":"This is a NEW low-capacity empirical hypothesis, not the previously fitted RF.",
            "limitations":["Only exact-ID matched results, cards and published lines enter.",
                           "Scores and line forecasts archived retrospectively; availability at original cutoff not independently verified.",
                           "The historical grade data might include overlapping race IDs; deduplicated on ID.",
                           "2024 and later results are not read or touched.",
                           "No market odds, revenue, payouts or profit calculations.",
                           "No selection or tuning using the Nov-Dec holdout; smoothing weights set in advance.",
                           "Calibration bands of rare triples may have small sample counts."]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    SAMPLE.write_text(json.dumps(samples,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RESEARCH_COMPLETE",json.dumps({"train":len(train),"heldout":len(holdout),
                "metrics":{k:{q:v[q] for q in ("nll","top1_hit_rate_pct","top3_hit_rate_pct")} for k,v in metrics.items()}}),
          flush=True)
if __name__=="__main__":main()
