"""Fixed-period DESCRIPTIVE 4-axis observation: line shape × class × favorite odds × race type.

Scope: 2023-01-01..2023-02-28 ONLY. Does NOT open results/quotes for later dates.
No learned purchases, no optimization, no search for best individual outcomes.
Each policy is a pre-registered UNION OF ACTUAL CARTESIAN FORMATIONS and
stakes exactly 100 JPY on EVERY distinct exact ordered outcome it covers.
"""
from __future__ import annotations
import os,sys,csv,re,json,math,itertools,tempfile
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
bridge=corrected.old.bridge
OUT=ROOT/"research/favorite_odds_line_stage_four_axis_20261028"
START="2023-01-01";END="2023-02-28"
LINE_SOURCES=(
 "data/a_class_12/2023_01_02/races.csv",
 "data/a3_challenge/2023_01_02/races.csv",
 "data/2023/s_class_f1_all_parts/2023_q1/races.csv",
 "data/2023/s_class_yosen/races.csv",
)
ODDS_EDGES=(2.,3.,5.,8.,12.,20.)
def normalized_stage(race_type):
    """Normalize recorded stage label without using any outcomes or odds."""
    rt=race_type.strip().replace("Ａ級","").replace("Ｓ級","").replace("チャレンジ","")
    if rt in ("初特選","初日特選"):return "初日特選"
    if rt in ("特予選","予選","一般","特一般","選抜","準決勝","特選","決勝"):return rt
    return "不明_"+rt if rt else "不明"
def classified_grade(source):
    if "s_class_f1_all_parts" in source or "s_class_yosen" in source:return "S級"
    if "a_class_12" in source:return "A級1・2班"
    if "a3_challenge" in source:return "A級チャレンジ"
    raise RuntimeError("Unknown source class "+source)
REPORT_GROUPS=("all","favorite_band","favorite_band_by_line_shape",
               "favorite_band_by_class","favorite_band_by_point_band",
               "line_shape","car_count")
AUDIT=Counter()
AGGS=defaultdict(lambda:[0.,0.,0.,0.,0.,0.,0.,0.])
FORM_EXAMPLES={}
LINES={};CONFLICTS=set();SOURCES=Counter()
FAMILIES=(
 "all_orders_control",
 "all_same_line_123",
 "all_same_line_12_external_3",
 "all_same_line_13_external_2",
 "all_same_line_23_external_1",
 "all_heads_12_all_3",
 "all_heads_12_followers_3",
 "all_followers_12_heads_3",
 "heads_first_followers_second",
 "followers_first_heads_second",
 "pair_all_selected_123",
 "pair_line_cross_3_internal",
 "pair_line_cross_3_external",
 "pair_line_cross_3_all",
 "pair_line_cross_3_on_first",
 "pair_line_cross_3_on_second",
 "pair_same_line_12_other_line_3",
 "pair_same_line_13_other_line_2",
 "pair_same_line_23_other_line_1",
 "pair_heads_12_pair_followers_3",
 "pair_followers_12_pair_heads_3",
 "three_line_cycle"
)
SELECTORS=("all_lines","market_top_two","longest_two")
REGISTRY=tuple([("all_lines",f) for f in FAMILIES[:10]]+
               [(s,f) for s in SELECTORS[1:] for f in FAMILIES[10:]])
# All templates are declared before loading any results, never changed by observations.
assert len(REGISTRY)==10+2*12

def load_published_lines():
    pattern=re.compile(r"[1-9](?:-[1-9])*(?:/[1-9](?:-[1-9])*)*")
    for source in LINE_SOURCES:
        path=ROOT/source
        if not path.exists():raise FileNotFoundError("Missing line source "+source)
        with path.open(encoding="utf-8-sig",newline="") as file:
            for row in csv.DictReader(file):
                day=row.get("race_date","");rid=row.get("race_id","")
                if not START<=day<=END:continue
                AUDIT["line_rows_in_period"]+=1
                raw=row.get("predicted_line_formation","").strip()
                if not pattern.fullmatch(raw):
                    AUDIT["line_invalid_published"]+=1;continue
                pieces=[tuple(int(t) for t in l.split("-")) for l in raw.split("/")]
                members=sum((list(x) for x in pieces),[])
                try:n=int(row.get("entry_count","0"))
                except(ValueError,TypeError):n=0
                if len(members)!=n or len(set(members))!=n or set(members)!=set(range(1,n+1)):
                    AUDIT["line_invalid_grid"]+=1;continue
                kind=source.split("/")[1]
                value=(day,tuple(pieces),kind,source,row.get("race_type","").strip(),row.get("meeting_grade","").strip())
                if rid in LINES:
                    existing=LINES[rid]
                    if existing[0]!=day or existing[1]!=value[1] or existing[4]!=value[4]:
                        CONFLICTS.add(rid)
                        AUDIT["line_conflicting_duplicates"]+=1
                    else:AUDIT["line_matching_duplicates"]+=1
                    continue
                LINES[rid]=value
                SOURCES[source]+=1
    for rid in CONFLICTS:LINES.pop(rid,None)
    AUDIT["unique_line_races"]=len(LINES)
    AUDIT["duplicate_conflict_races"]=len(CONFLICTS)

def fav_bin(v):
    return (("<2" if v<2 else "2-3" if v<3 else "3-5" if v<5 else
             "5-8" if v<8 else "8-12" if v<12 else "12-20" if v<20 else "20+"))

def point_bin(n):
    return "4-8" if n<=8 else "9-16" if n<=16 else "17-32" if n<=32 else "33-64" if n<=64 else "65+"

def flatten(a):
    return tuple(dict.fromkeys(itertools.chain.from_iterable(a)))

def cartesian_parts(components):
    T=set()
    for triple in components:
        if len(triple)!=3:raise ValueError("Bad triple formation")
        for a,b,c in itertools.product(*triple):
            if len({a,b,c})==3:T.add((a,b,c))
    return T

def all_family_templates(lines,cars):
    allcar=tuple(cars)
    heads=tuple(x[0] for x in lines)
    followers=flatten(tuple(x[1:]) for x in lines)
    same=[(x,x,x) for x in lines]
    ext12=[(x,x,tuple(c for c in cars if c not in x)) for x in lines]
    ext13=[(x,tuple(c for c in cars if c not in x),x) for x in lines]
    ext23=[(tuple(c for c in cars if c not in x),x,x) for x in lines]
    return {
      "all_orders_control":[(allcar,allcar,allcar)],
      "all_same_line_123":same,
      "all_same_line_12_external_3":ext12,
      "all_same_line_13_external_2":ext13,
      "all_same_line_23_external_1":ext23,
      "all_heads_12_all_3":[(heads,heads,allcar)],
      "all_heads_12_followers_3":[(heads,heads,followers)],
      "all_followers_12_heads_3":[(followers,followers,heads)],
      "heads_first_followers_second":[(heads,followers,allcar)],
      "followers_first_heads_second":[(followers,heads,allcar)]
    }

def pair_family_templates(L1,L2,lines,cars):
    U=tuple(dict.fromkeys((*L1,*L2)))
    outside=tuple(c for c in cars if c not in U)
    heads=(L1[0],L2[0])
    followers=tuple(dict.fromkeys((*L1[1:],*L2[1:])))
    cross=((L1,L2),(L2,L1))
    # All policies comprise proper Cartesian 1st x 2nd x 3rd formations;
    # order-swapped components are a symmetric union, not two handpicked outcomes.
    tri=[]
    if len(lines)>=3:
        # For at least three lines, use first three (according to selection order).
        a,b,c=lines[:3]
        tri=[(a,b,c),(b,c,a),(c,a,b)]
    return {
      "pair_all_selected_123":[(U,U,U)],
      "pair_line_cross_3_internal":[(a,b,U) for a,b in cross],
      "pair_line_cross_3_external":[(a,b,outside) for a,b in cross],
      "pair_line_cross_3_all":[(a,b,tuple(cars)) for a,b in cross],
      "pair_line_cross_3_on_first":[(a,b,a) for a,b in cross],
      "pair_line_cross_3_on_second":[(a,b,b) for a,b in cross],
      "pair_same_line_12_other_line_3":[(L1,L1,L2),(L2,L2,L1)],
      "pair_same_line_13_other_line_2":[(L1,L2,L1),(L2,L1,L2)],
      "pair_same_line_23_other_line_1":[(L2,L1,L1),(L1,L2,L2)],
      "pair_heads_12_pair_followers_3":[(heads,heads,followers)],
      "pair_followers_12_pair_heads_3":[(followers,followers,heads)],
      "three_line_cycle":tri
    }

def record(group,name,npts,win,wonodds,q,favorite):
    k=(group[0],group[1],name)
    v=AGGS[k]
    v[0]+=1
    v[1]+=npts
    v[2]+=100*npts
    v[3]+=100*wonodds*win
    v[4]+=win
    v[5]+=q
    v[6]+=favorite
    v[7]+=q*(1-q)

def observe(rid,day,books,finish,rtype,source):
    AUDIT["odds_callbacks"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Failed independent ordered odds validation")
    ldata=LINES.get(rid)
    if ldata is None:
        AUDIT["odds_without_published_line"]+=1;return
    if ldata[0]!=day:raise AssertionError("Line date mismatch")
    lines=ldata[1]
    book=books["3rentan"]
    actual=tuple(finish)
    cars=tuple(sorted({x for w in book for x in w}))
    n=len(cars)
    if set(flatten(lines))!=set(cars) or actual not in book or len(book)!=n*(n-1)*(n-2):
        AUDIT["line_odds_grid_mismatch"]+=1;return
    if any(not math.isfinite(float(o)) or float(o)<1 for o in book.values()):
        AUDIT["invalid_odds"]+=1;return
    win_odd=float(book[actual])
    qtotal=sum(1/float(o) for o in book.values())
    q={w:(1/float(o))/qtotal for w,o in book.items()}
    favorite=min(float(o) for o in book.values())
    band=fav_bin(favorite)
    marginal={x:0.0 for x in cars}
    for w,pr in q.items():marginal[w[0]]+=pr
    marketlines=sorted(lines,key=lambda z:(-sum(marginal[x] for x in z),tuple(z)))
    sizelines=sorted(lines,key=lambda z:(-len(z),tuple(z)))
    line_shape="-".join(str(len(x)) for x in sizelines)
    kind=classified_grade(ldata[3])
    race_type=ldata[4]
    stage=normalized_stage(race_type)
    meeting_grade=ldata[5] or "開催グレード未収録"
    AUDIT["joined_class_"+kind]+=1
    AUDIT["joined_stage_"+stage]+=1
    AUDIT["joined_grade_"+meeting_grade]+=1
    if len(lines)<2:
        AUDIT["one_line_race_no_crossline_comparison"]+=1
        return
    AUDIT["eligible_races"]+=1;AUDIT["fav_"+band]+=1
    AUDIT["line_shape_"+line_shape]+=1
    allforms=all_family_templates(lines,cars)
    candidates=[]
    for k in FAMILIES[:10]:candidates.append(("all_lines",k,allforms[k]))
    for selector,order in (("market_top_two",marketlines),("longest_two",sizelines)):
        if len(order)<2:continue
        pairforms=pair_family_templates(order[0],order[1],order,cars)
        for k in FAMILIES[10:]:candidates.append((selector,k,pairforms[k]))
    assert len(candidates)==len(REGISTRY)
    for selector,family,forms in candidates:
        if not forms:
            AUDIT["candidate_empty_"+family]+=1;continue
        all_tickets=cartesian_parts(forms)
        # Reject tiny pseudo-formations; this is about coverage sets, not one/two bets.
        if len(all_tickets)<4:
            AUDIT["under_four_tickets_"+family]+=1;continue
        npts=len(all_tickets)
        hit=int(actual in all_tickets)
        mass=sum(q[x] for x in all_tickets)
        pbin=point_bin(npts)
        name=selector+"/"+family
        for group in (("all","2023_Jan_Feb"),
                      ("favorite_band",band),
                      ("favorite_band_by_line_shape",band+"|"+line_shape),
                      ("favorite_band_by_class",band+"|"+kind),
                      ("favorite_band_by_race_type",band+"|"+race_type),
                      ("favorite_band_by_race_stage",band+"|"+stage),
                      ("favorite_band_by_class_stage",band+"|"+kind+"|"+stage),
                      ("favorite_band_by_class_meeting_grade",band+"|"+kind+"|"+meeting_grade),
                      ("favorite_band_by_meeting_grade",band+"|"+meeting_grade),
                      ("favorite_band_by_line_shape_class_stage",band+"|"+line_shape+"|"+kind+"|"+stage),
                      ("favorite_band_by_line_shape_class_racetype",band+"|"+line_shape+"|"+kind+"|"+race_type),
                      ("favorite_band_by_point_band",band+"|"+pbin),
                      ("line_shape",line_shape),
                      ("car_count",str(n))):
            record(group,name,npts,hit,win_odd,mass,favorite)
        # first encountered example per odds band / shape / family, not best outcome
        ek=(band,name)
        if ek not in FORM_EXAMPLES and len(FORM_EXAMPLES)<200:
            FORM_EXAMPLES[ek]={
              "race_id":rid,"date":day,"line":"/".join("-".join(map(str,l)) for l in lines),
              "favorite":favorite,"line_shape":line_shape,
              "class":kind,"race_type":race_type,"race_stage":stage,
              "meeting_grade":meeting_grade,
              "forms":[[list(slot) for slot in component] for component in forms],
              "unique_points":npts,"market_mass":round(mass,6),
              "actual_finish":list(actual),
              "hit":bool(hit),"stake_yen":100*npts,
              "actual_return_yen":100*hit*win_odd
            }

def main():
    load_published_lines()
    if AUDIT["unique_line_races"]<1000:raise RuntimeError("Too few line races in Jan-Feb")
    if not os.environ.get(bridge.SECRET_ENV):
        raise RuntimeError("Private archive key must be installed in Actions")
    labels,ls=bridge.verified_results(2023)
    AUDIT["2023_result_labels_loaded"]=len(labels)
    catalog,meta=bridge.catalog(START,END)
    AUDIT["encrypted_files"]=meta["sealed_files"]
    corrected.base_observe=observe
    temporary,key=bridge.secure_key_temp()
    try:
        for day,files in catalog.items():
            if not START<=day<=END:raise AssertionError("Reserved date read")
            merged={};conflicts=set()
            with tempfile.TemporaryDirectory(prefix="lineformation_2023_janfeb_") as tmp:
                for path,sha in files:
                    for rid,rows in bridge.decrypt_sealed(path,sha,key,tmp).items():
                        if rid in conflicts:continue
                        canonical=sorted(rows)
                        if rid in merged and merged[rid]!=canonical:
                            conflicts.add(rid);merged.pop(rid);continue
                        merged[rid]=canonical
            AUDIT["odds_duplicate_conflicts"]+=len(conflicts)
            for rid,rows in merged.items():
                if rid not in LINES:continue
                norm,reason=bridge.normalize_book(rid,day,rows)
                if reason:
                    AUDIT["normalize_"+reason]+=1;continue
                if rid not in labels:
                    AUDIT["missing_result"]+=1;continue
                books=defaultdict(dict)
                for row in rows:
                    typ=row[2]
                    if typ not in ("3rentan","3renpuku","2shatan","2shafuku"):continue
                    lo,hi=float(row[4]),float(row[5])
                    if lo!=hi:raise AssertionError("Range-priced market")
                    keytuple=tuple(int(x) for x in row[3].split("-"))
                    if typ in ("3renpuku","2shafuku"):keytuple=tuple(sorted(keytuple))
                    books[typ][keytuple]=lo
                corrected.corrected_observe(rid,day,books,labels[rid],rows[0][1],"sealed_four_markets")
            AUDIT["days"]+=1
            if AUDIT["days"]%15==0:print("PROGRESS",day,dict(AUDIT),flush=True)
    finally:temporary.cleanup()
    if AUDIT["eligible_races"]<500:
        raise RuntimeError("Insufficient actual join of lines+complete odds+finishes")
    rows=[]
    for (group,value,name),v in sorted(AGGS.items()):
        N,points,stake,pay,hits,mass,fav,variance=v
        rows.append({
           "group":group,"segment":value,"family":name,"races":int(N),
           "mean_points":points/N,"tickets":int(points),
           "stake_yen":round(stake),"payout_yen":round(pay),
           "profit_yen":round(pay-stake),
           "roi_pct":100*pay/stake if stake else None,
           "set_hit_count":int(hits),
           "set_hit_pct":100*hits/N,
           "market_expected_set_hit_count":mass,
           "set_hits_over_market":hits/mass if mass else None,
           "mean_favorite_odds":fav/N,
           "mean_market_set_probability":mass/N,
           "observed_minus_market_hits":hits-mass,
           "descriptive_z":(hits-mass)/math.sqrt(variance) if variance>0 else None
        })
    primary=[x for x in rows if x["group"]=="favorite_band"]
    all_overview=[x for x in rows if x["group"]=="all"]
    # This is a descriptive leaderboard with a sample-size floor, not an accepted policy.
    leaderboards={}
    for b in ("<2","2-3","3-5","5-8","8-12","12-20","20+"):
        valid=[x for x in primary if x["segment"]==b and x["races"]>=40]
        valid.sort(key=lambda x:(-x["roi_pct"],-x["races"],x["family"]))
        leaderboards[b]=valid[:10]
    report={
      "status":"DESCRIPTIVE_FOUR_AXIS_FAVORITE_LINE_CLASS_RACE_TYPE_OBSERVATION_COMPLETE",
      "source_commit":os.environ.get("GITHUB_SHA","unknown"),
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "audit":dict(AUDIT),"line_input_sources":dict(SOURCES),
      "scope":{"start":START,"end":END,"only_five_seven_market_verified":True,
               "later_2023_2026_odds_consumed":False,
               "note":"The results label file for calendar year 2023 is loaded but only Jan-Feb matched odds/line records are evaluated; 2023-03+ prices and lines are not processed. Other research already used these later results in prior studies."},
      "pre_registered_families":list(REGISTRY),
      "four_axis_naming":{"line_shape":"Published line-length multiset, e.g. 3-2-2",
        "class":"A級1・2班 / S級 / A級チャレンジ, validated from original source dataset",
        "favorite_price":"Min final trifecta decimal odds, before results, fixed bands",
        "race_type":"Original KDreams label Ａ級予選 / Ｓ級準決勝 etc, with normalized stage",
        "meeting_grade":"F1/F2 as supplied; not conflated with qualifying/final stages",
        "sample":"Fixed Jan-Feb 2023 join; original 34 union formations unchanged",
        "correction":"Previous class group called S-series source '2023' (directory index); fixed to S級. Prior report is not overwritten."},
      "definitions":{
         "favorite":"Lowest numeric final 3連単 exact ticket price in full book",
         "favorite_band":"[1,2),[2,3),[3,5),[5,8),[8,12),[12,20),[20,infinity)",
         "published_line":"Published predicted formation, not observed in-race actual partnership",
         "market_pair":"Two different published lines with highest aggregate market 1st-place probability from full 3連単",
         "longest_pair":"Two longest published lines; tie broken by car numbers, NOT finish or price",
         "union":"Full Cartesian slot product for each listed formation component; duplicate ordered tickets bought once",
         "cost":"100 yen per distinct ticket in full set every eligible race, regardless outcome",
         "returns":"100 * archived FINAL odds of exact finish if and only if exact finish is inside full set",
         "candidate_policy":"All predefined real-line formations; no post-result refinement, no ranking method fitting",
         "no_selection":"No 'buy if this condition' deployed; descriptive comparisons only",
         "market_expected_set_hit":"Sum of normalized inverse three-way odds across all exact tickets in the set"
      },
      "results":rows,
      "descriptive_leaderboards_not_selection":leaderboards,
      "examples":list(FORM_EXAMPLES.values()),
      "warnings":[
          "These are observational total-set portfolio returns, NOT estimates of prospective edge.",
          "The 'best' row is identified after seeing outcomes; all 4D cells reported. No such row is an executable strategy.",
          "Market conditional ranking may regress to market average; other longest-line selection offers market-independent control.",
          "Uses FINAL odds and does not prove T-15 bet availability.",
          "Only races in overlapping documented public line data and verified complete quote/result archives; not full-universe random sample.",
          "Published line formation does not prove actual tactical lines raced.",
          "All form policies overlap, therefore results are correlated; do not aggregate profits across families as if independent.",
          "Scans 34 policies, 7 favorite odds bands, line shapes, classes, and race types: severe multiple comparisons and sparse cells. No correction.",
          "No complete generic formation optimization; this is descriptive observation of documented LINE STRUCTURAL FAMILIES."
      ]
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    four=[x for x in rows if x["group"]=="favorite_band_by_line_shape_class_stage"]
    report["sample_fragmentation"]={"all_4axis_cells_including_34_policies":len(four),
       "cells_with_at_least_30_races":sum(x["races"]>=30 for x in four),
       "cells_with_fewer_than_30_races":sum(x["races"]<30 for x in four),
       "warning":"No four-axis cell was chosen as a purchase rule. Small cells have unstable ROI."}
    (OUT/"compact.json").write_text(json.dumps({k:v for k,v in report.items() if k not in ("results","examples")} |
       {"favorite_band_full_table":primary,"overall_all_families":all_overview,
        "four_axis_cells_at_least_30":[x for x in four if x["races"]>=30],
        "stage_tables":[x for x in rows if x["group"]=="favorite_band_by_race_stage"],
        "class_stage_tables":[x for x in rows if x["group"]=="favorite_band_by_class_stage"]},
       ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("OBSERVATION_COMPLETE",json.dumps({"audit":dict(AUDIT),
      "leaderboards":{k:[{"name":r["family"],"n":r["races"],"ROI":round(r["roi_pct"],2),
                       "points":round(r["mean_points"],2),
                       "profit":r["profit_yen"]} for r in v[:4]] for k,v in leaderboards.items()}},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
