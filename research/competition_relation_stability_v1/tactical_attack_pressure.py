#!/usr/bin/env python3
"""Pre-race ATTACK-PRESSURE rivalry: score tie alone does not imply contested lead.

Read historical b_count and tactics from PRE-RACE rider entries. These are
pre-race past statistics, NEVER actual in-race sprint/attack data.
Within similar leader-score gaps, formation, third-rival deficit and field:
 compare two line leaders with high prior B counts vs one high-B leader.
Also distinguish which outsider benefits. No changes to V6, no odds/payouts.

HIGH_B >= 6, LOW_B <= 2 fixed prior to evaluating 2024-26.
Only original entry sources + published line/score records are used.
2023 Jan-Oct discovery, Nov-Dec dev; 2024-26 retrospective confirmation,
already inspected in older research. Associations are NOT causal effects.
"""
from __future__ import annotations
import csv,io,json,math,sys,zipfile
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.competition_relation_stability_v1 import observe as rel

OUT=ROOT/"data/audits/tactical_rivalry_bcount_2023_2026.json"
RACE_CATEGORIES=("2023_train","2023_dev","2024","2025","2026","2024_to_2026")
TARGETS=("third_multi_head_win","third_multi_line_win","best_head_win",
         "best_line_win","same_line_12","three_distinct_lines","weak_score_win")

def int_count(x):
    try:
        n=int(str(x).strip())
        return n if 0<=n<=60 else None
    except (ValueError,TypeError):return None

def folders():
    for prefix in ("data/a_class_12","data/a3_challenge","data/2023",
                   "data/2024","data/2025","data/2026_h1",
                   "data/2026_future_block1"):
        p=ROOT/prefix
        if p.exists():
            for ep in sorted(p.rglob("entries.csv")):yield ("csv",ep)
    for p in sorted((ROOT/"data/grade_races").rglob("*.zip")):
        if any(y in str(p) for y in ("2023","2024","2025","2026")):
            yield ("zip",p)

def read_entries(typ,p):
    if typ=="csv":
        with p.open("r",encoding="utf-8-sig",newline="") as f:
            yield from csv.DictReader(f)
    else:
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if name.rsplit("/",1)[-1]!="entries.csv":continue
                with z.open(name) as raw:
                    with io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as f:
                        yield from csv.DictReader(f)

def b_data(sections):
    needed={}
    for period,rows in sections.items():
        if period=="2024_to_2026":continue
        for rid,e,f,y in rows:needed[rid]=e
    data=defaultdict(dict)
    conflicts=set()
    audit=Counter()
    for typ,p in folders():
        audit["source_entry_files"]+=1
        try:
            for row in read_entries(typ,p):
                rid=str(row.get("race_id","")).strip()
                e=needed.get(rid)
                if e is None:continue
                car=str(row.get("car_no","")).strip()
                if car not in e["scores"]:continue
                try:score=float(row.get("score") or "")
                except (TypeError,ValueError):continue
                if not math.isfinite(score) or abs(score-e["scores"][car])>.025:
                    audit["score_doesnt_match_published_card"]+=1;continue
                b=int_count(row.get("b_count"))
                if b is None:
                    audit["invalid_or_missing_b"]+=1;continue
                age=int_count(row.get("age"))
                v=(b,age,str(row.get("style","")).strip())
                if car in data[rid]:
                    if data[rid][car]!=v:conflicts.add((rid,car))
                    else:audit["duplicate_identical"]+=1
                else:data[rid][car]=v
                audit["matched_entry_rows"]+=1
        except (OSError,UnicodeError,zipfile.BadZipFile) as ex:
            audit["unreadable_source"]+=1
            print("TACTIC_SOURCE_FAIL",str(p),type(ex).__name__,flush=True)
    for rid,car in conflicts:data[rid].pop(car,None)
    audit["conflicting_rider_b_metadata"]=len(conflicts)
    audit["race_ids_with_at_least_one_b"]=len(data)
    print("TACTIC_B_FIELD_ARCHIVE_COVERAGE",json.dumps(dict(audit)),flush=True)
    return data,dict(audit)

def binof(x,thresholds):
    for i,z in enumerate(thresholds):
        if x<z:return i
    return len(thresholds)

def categorize(e,f,y,brows):
    multi=sorted(((e["scores"][g[0]],len(g),g[0],i+1)
             for i,g in enumerate(e["groups"]) if len(g)>=2),
             key=lambda x:(-x[0],int(x[2])))
    if len(multi)<2:return None
    h1,h2=multi[:2]
    if h1[2] not in brows or h2[2] not in brows:return None
    if f["multi_top_head_gap"]>5:return None
    b1,b2=brows[h1[2]][0],brows[h2[2]][0]
    hi=sum(v>=6 for v in (b1,b2))
    lo=sum(v<=2 for v in (b1,b2))
    if hi==2:arm="TWO_HIGH_B_LEADERS"
    elif hi==1 and lo==1:arm="ONE_HIGH_B_ONE_LOW_B"
    elif hi==0:arm="NO_HIGH_B_LEADERS"
    else:arm="OTHER_PRESSURE"
    if h1[3]!=f["num_lines"] and h1[2] is None:
        raise AssertionError("Invalid line leader")
    keys={
        "field":f["n"],"shape":f["line_shape"],
        "head_gap":binof(f["multi_top_head_gap"],(2,4)),
        "third_deficit":binof(f["third_multi_deficit"],(7,12,18,27)),
        "support_gap":binof(f["best_bante_gap"],(0,5,11)),
        "dispersion":binof(f["dispersion"],(15,25,35)),
        "class":e.get("race_class","UNKNOWN"),
    }
    y["attack_pressure_proxy_both_B_over6"]=int(hi==2)
    return arm,keys

def robust_match(rows,b):
    raw=defaultdict(lambda:[0,Counter()])
    matched=defaultdict(lambda:defaultdict(lambda:[0,Counter()]))
    alln=Counter()
    for rid,e,f,y in rows:
        info=categorize(e,f,y.copy(),b.get(rid,{}))
        if info is None:
            alln["not_matched_or_head_gap_above5"]+=1;continue
        arm,k=info
        alln["valid_rival_leader_b_pre-race"]+=1
        if arm not in ("TWO_HIGH_B_LEADERS","ONE_HIGH_B_ONE_LOW_B"):
            alln["other_B_mix"]+=1;continue
        raw[arm][0]+=1
        for m in TARGETS:raw[arm][1][m]+=y[m]
        # exact stratum as far as common support allows. Cohort-year is
        # already fixed by the caller; don't match future to past records.
        stratum=(k["field"],k["shape"],k["head_gap"],
                 k["third_deficit"],k["support_gap"])
        z=matched[stratum][arm];z[0]+=1
        for m in TARGETS:z[1][m]+=y[m]
    a,b_arm="TWO_HIGH_B_LEADERS","ONE_HIGH_B_ONE_LOW_B"
    valid=[v for v in matched.values() if v[a][0]>=4 and v[b_arm][0]>=4]
    overlap_a=sum(v[a][0] for v in valid)
    overlap_b=sum(v[b_arm][0] for v in valid)
    wtotal=0.;sums=defaultdict(float);var=defaultdict(float)
    for v in valid:
        na,va=v[a];nb,vb=v[b_arm]
        w=na*nb/(na+nb)
        wtotal+=w
        for m in TARGETS:
            pa=va[m]/na;pb=vb[m]/nb
            sums[m]+=w*(pa-pb)
            # Independence approx for matched strata (descriptive only).
            var[m]+=w*w*(pa*(1-pa)/na+pb*(1-pb)/nb)
    def show(arm):
        n,ct=raw[arm]
        return {"races":n,"observed_pct":{
            m:round(100*ct[m]/n,6) if n else None for m in TARGETS}}
    return {
      "counts":dict(alln),
      "raw_two_high_B":show(a),
      "raw_one_high_one_low_B":show(b_arm),
      "matched_strata_min4_each":len(valid),
      "matched_arm_two_high_count":overlap_a,
      "matched_arm_one_high_count":overlap_b,
      "match_score_and_formation_controlled_effect_two_high_minus_one_high_pct_points":{
          m:round(100*sums[m]/wtotal,6) if wtotal else None for m in TARGETS},
      "rough_sampling_se_percentage_points":{
          m:round(100*math.sqrt(var[m])/wtotal,6) if wtotal else None for m in TARGETS},
      "status":"OBSERVATIONAL_ASSOCIATION_ONLY_ACTUAL_DUEL_NOT_OBSERVED"
    }

def main():
    all23,cov=rel.src23.ingest()
    rows23,cover23=rel.prepare(all23,"2023")
    periods={"2023_train":[r for r in rows23 if r[1]["date"]<="2023-10-31"],
             "2023_dev":[r for r in rows23 if r[1]["date"]>"2023-10-31"]}
    del all23,rows23
    coverage={"2023":cover23}
    for year,reader in (("2024",rel.src24.ingest_2024),
                        ("2025",rel.src25.read_2025),
                        ("2026",rel.src26.ingest_2026)):
        events,_=reader()
        periods[year],coverage[year]=rel.prepare(events,year)
        del events
    b,b_audit=b_data(periods)
    periods["2024_to_2026"]=[v for p in ("2024","2025","2026") for v in periods[p]]
    out={
        "state":"TACTICAL_BCOUNT_TWO_FRONTLINE_COMPETITION_PRE_RACE_COMPLETE",
        "no_V6_changes":True,
        "no_odds_or_payout":True,
        "periods":{},"coverage":coverage,
        "b_count_source":b_audit,
        "hypothesis":"When two similarly scored opposing leaders both historically run high B counts, third-line/upset arrivals differ vs one high-B and one low-B leader.",
        "important_caveat":"Prior B counts are not observed sprint competition; observational score-matched data do not identify causal race tactics.",
        "field_study_years_previously_inspected":True,
    }
    for period,rows in periods.items():
        ans=robust_match(rows,b)
        out["periods"][period]={"races":len(rows),**ans}
        print("TACTICAL_PRESSURE_PERIOD_COMPLETE",json.dumps({"period":period,**ans},ensure_ascii=False),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TACTICAL_PRESSURE_FINAL_RESULT",json.dumps({
        "year_cohorts":{z:out["periods"][z] for z in ("2023_train","2023_dev","2024","2025","2026","2024_to_2026")},
        "b_archive_coverage":b_audit
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()
