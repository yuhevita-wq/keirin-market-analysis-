"""Discover genuine line-level FORMATION SETS from outcomes/whole-market odds.
NO catalog of preset 3連単 buying templates, NO ticket-by-ticket strategy selection.

Discovery window: Jan-Feb 2023. Holdout: Mar-Apr 2023 (not used for any rule choice).
The learned representation is the UNION of up to two complete slot-product
FORMATIONS whose three slot sets consist of *whole lines*, selected by their
market-aggregate line ranks within the published formation. This is a LIMITED
but truly combinatorial, data-learned set grammar; it does not claim to search
arbitrary rider-subsets or role-based formations yet.
"""
from __future__ import annotations
import os,re,sys,csv,json,math,itertools,tempfile
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
bridge=corrected.old.bridge
OUT=ROOT/"research/formation_discovery_from_outcomes_20261028"
TRAIN=("2023-01-01","2023-02-28")
TEST=("2023-03-01","2023-04-30")
INPUTS=(
 "data/a_class_12/2023_01_02/races.csv",
 "data/a_class_12/2023_03_04/races.csv",
 "data/2023/s_class_f1_all_parts/2023_q1/races.csv",
 "data/2023/s_class_f1_all_parts/2023_q2/races.csv",
 "data/2023/s_class_yosen/races.csv",
)
A=Counter()
RACES=[]
LINE={};DUP_CONFLICT=set();SOURCES=Counter()
LIMIT_LINES=3
ODDS_BANDS=(2.,3.,5.,8.,12.,20.)
SUBSETS=tuple(range(1,1<<LIMIT_LINES))
# Structural geometry: ALL nonempty Cartesian products of independently
# chosen sets of ranked lines; not a list of handicapped outcomes/patterns.
RECTS=tuple((a,b,c,sum(1<<((i*3+j)*3+k)
           for i in range(3) if a>>i&1
           for j in range(3) if b>>j&1
           for k in range(3) if c>>k&1))
           for a in SUBSETS for b in SUBSETS for c in SUBSETS)
assert len(RECTS)==343
CELL_N=LIMIT_LINES**3
assert CELL_N==27
def class_name(p):
    if "/a_class_12/" in p:return "A級1・2班"
    if "/s_class_f1_all_parts/" in p or "/s_class_yosen/" in p:return "S級"
    return "OTHER"
def normalize_stage(s):
    a=s.strip().replace("Ａ級","").replace("Ｓ級","").replace("チャレンジ","")
    return "初日特選" if a in ("初特選","初日特選") else a
def fav_band(v):
    if v<2:return "<2"
    if v<3:return "2-3"
    if v<5:return "3-5"
    if v<8:return "5-8"
    if v<12:return "8-12"
    if v<20:return "12-20"
    return "20+"
def load_lines():
    pattern=re.compile(r"[1-9](?:-[1-9])*(?:/[1-9](?:-[1-9])*)*")
    for source in INPUTS:
        p=ROOT/source
        if not p.exists():raise FileNotFoundError(str(p))
        with p.open(encoding="utf-8-sig",newline="") as h:
            for row in csv.DictReader(h):
                day=row.get("race_date","")
                if not TRAIN[0]<=day<=TEST[1]:continue
                A["published_line_rows"]+=1
                val=row.get("predicted_line_formation","")
                if not pattern.fullmatch(val):
                    A["excluded_no_published_line"]+=1;continue
                ll=tuple(tuple(map(int,x.split("-"))) for x in val.split("/"))
                cars=tuple(itertools.chain.from_iterable(ll))
                try:n=int(row.get("entry_count",0))
                except:continue
                if len(cars)!=n or len(set(cars))!=n or set(cars)!=set(range(1,n+1)):
                    A["excluded_bad_line_coverage"]+=1;continue
                rid=row["race_id"]
                val=(day,ll,class_name(source),normalize_stage(row["race_type"]),
                     row["race_type"],source)
                if rid in LINE:
                    if LINE[rid][:5]!=val[:5]:
                        DUP_CONFLICT.add(rid)
                        A["line_conflict_duplicates"]+=1
                    else:A["line_matching_duplicates"]+=1
                else:
                    LINE[rid]=val;SOURCES[source]+=1
    for rid in DUP_CONFLICT:LINE.pop(rid,None)
    A["unique_line_races"]=len(LINE)
def observe(rid,day,books,finish,rtype,source):
    if source!="sealed_four_markets_orientation_verified":raise AssertionError("odds orientation not proven")
    row=LINE.get(rid)
    if not row:return
    if day!=row[0]:raise AssertionError("date mismatch")
    ll=row[1]
    if len(ll)!=3:
        A["excluded_non_three_line_races"]+=1;return
    book=books["3rentan"]
    actual=tuple(finish)
    n=len(set(x for t in book for x in t))
    if len(book)!=n*(n-1)*(n-2) or actual not in book:
        A["excluded_incomplete_book"]+=1;return
    cars={car for line in ll for car in line}
    if len(cars)!=n or cars!=set(x for t in book for x in t):
        A["excluded_car_alignment"]+=1;return
    ps={w:float(o) for w,o in book.items()}
    if any(not math.isfinite(x) or x<1 for x in ps.values()):
        A["excluded_invalid_prices"]+=1;return
    # Full market first-place aggregate line support. Does not use finish.
    inv={w:1/o for w,o in ps.items()}
    norm=sum(inv.values())
    first=defaultdict(float)
    for w,v in inv.items():first[w[0]]+=v/norm
    ll=tuple(sorted(ll,key=lambda line:(-sum(first[x] for x in line),line)))
    lookup={x:i for i,line in enumerate(ll) for x in line}
    count=[0]*CELL_N
    implied=[0.0]*CELL_N
    for w,p in ps.items():
        i,j,k=(lookup[x] for x in w)
        c=(i*3+j)*3+k
        count[c]+=1
        implied[c]+=1.0/p/norm
    wi,wj,wk=(lookup[x] for x in actual)
    winner_cell=(wi*3+wj)*3+wk
    winodd=ps[actual]
    pnl=[-100*v for v in count]
    pnl[winner_cell]+=100*winodd
    RACES.append({
      "race_id":rid,"day":day,"split":"train" if day<=TRAIN[1] else "test",
      "class":row[2],"stage":row[3],"race_type":row[4],
      "line_shape":"-".join(str(len(x)) for x in sorted(ll,key=lambda z:-len(z))),
      "favorite_band":fav_band(min(ps.values())),
      "favorite_odds":min(ps.values()),
      "lines":ll,
      "winner":actual,
      "winner_line_cell":winner_cell,
      "winodds":winodd,
      "count":count,"implied":implied,"pnl":pnl
    })
    A["usable_3_line_races"]+=1
    A["train" if day<=TRAIN[1] else "test"]+=1
def records():
    if not os.environ.get(bridge.SECRET_ENV):raise RuntimeError("Actions secrets private sealed archive key required")
    load_lines()
    labels,_=bridge.verified_results(2023)
    A["available_result_labels_2023"]=len(labels)
    catalog,meta=bridge.catalog(TRAIN[0],TEST[1])
    A["encrypted_files"]=meta["sealed_files"]
    corrected.base_observe=observe
    tmpkey,key=bridge.secure_key_temp()
    try:
        for day,files in catalog.items():
            if not TRAIN[0]<=day<=TEST[1]:raise AssertionError("Unexpected date")
            merged={};bad=set()
            with tempfile.TemporaryDirectory(prefix="formation_discovery_decrypt_") as d:
                for path,sha in files:
                    for rid,rows in bridge.decrypt_sealed(path,sha,key,d).items():
                        if rid in bad:continue
                        canonical=sorted(rows)
                        if rid in merged and merged[rid]!=canonical:
                            bad.add(rid);merged.pop(rid);continue
                        merged[rid]=canonical
            A["odds_duplicate_conflicts"]+=len(bad)
            for rid,rows in merged.items():
                if rid not in LINE:continue
                norm,reason=bridge.normalize_book(rid,day,rows)
                if reason:
                    A["excluded_"+reason]+=1;continue
                if rid not in labels:
                    A["excluded_no_result"]+=1;continue
                books=defaultdict(dict)
                for z in rows:
                    kind=z[2]
                    if kind not in ("3rentan","3renpuku","2shatan","2shafuku"):continue
                    if float(z[4])!=float(z[5]):raise AssertionError("Non single-odds source")
                    w=tuple(map(int,z[3].split("-")))
                    if kind in ("2shafuku","3renpuku"):w=tuple(sorted(w))
                    books[kind][w]=float(z[4])
                corrected.corrected_observe(rid,day,books,labels[rid],rows[0][1],"sealed_four_markets")
            A["days"]+=1
            if A["days"]%20==0:print("PROGRESS",day,"races",len(RACES),flush=True)
    finally:tmpkey.cleanup()
    if A["train"]<1000 or A["test"]<1000:raise RuntimeError("Missing baseline usable races "+str(A))
def groups():
    grouped=defaultdict(lambda:{"train":[],"test":[]})
    for r in RACES:
        if r["class"] not in ("S級","A級1・2班"):continue
        keys=(
           ("4axis",r["line_shape"],r["class"],r["favorite_band"],r["stage"]),
           ("3axis_stage_pooled",r["line_shape"],r["class"],r["favorite_band"],"all-stages"))
        for k in keys:grouped[k][r["split"]].append(r)
    return grouped
def aggregate(records):
    stake=[0.]*CELL_N;pay=[0.]*CELL_N
    hits=[0]*CELL_N;expected=[0.]*CELL_N;priced_count=[0.]*CELL_N
    for r in records:
        for c in range(CELL_N):
            priced_count[c]+=r["count"][c]
            stake[c]+=100*r["count"][c]
            expected[c]+=r["implied"][c]
        cell=r["winner_line_cell"]
        hits[cell]+=1
        pay[cell]+=100*r["winodds"]
    return {"stake":stake,"pay":pay,"hits":hits,"exp":expected,"count":priced_count,"n":len(records)}
def bits(mask):
    while mask:
        x=mask&-mask
        yield x.bit_length()-1
        mask^=x
def totals(ag,mask):
    cs=list(bits(mask))
    stake=sum(ag["stake"][j] for j in cs)
    pay=sum(ag["pay"][j] for j in cs)
    return {"stake_yen":round(stake),"payout_yen":round(pay),
            "profit_yen":round(pay-stake),
            "roi_pct":round(100*pay/stake,2) if stake else None,
            "tickets":sum(ag["count"][j] for j in cs),
            "mean_points":round(sum(ag["count"][j] for j in cs)/ag["n"],3) if ag["n"] else None,
            "hits":sum(ag["hits"][j] for j in cs),
            "market_expected_hits":round(sum(ag["exp"][j] for j in cs),3),
            "races":ag["n"]}
def form_parts(rects):
    return [{"first_line_ranks":[i+1 for i in range(3) if a>>i&1],
             "second_line_ranks":[i+1 for i in range(3) if b>>i&1],
             "third_line_ranks":[i+1 for i in range(3) if c>>i&1]}
            for a,b,c,_ in rects]
def render_example(example,rects,mask):
    if example is None:return None
    ll=example["lines"]
    forms=[]
    for a,b,c,_ in rects:
        slots=[sorted({x for j,line in enumerate(ll) if code>>(j)&1 for x in line}) for code in (a,b,c)]
        forms.append(slots)
    actual_tickets=set()
    for a,b,c in forms:
        for t in itertools.product(a,b,c):
            if len(set(t))==3:actual_tickets.add(t)
    count=sum(example["count"][j] for j in bits(mask))
    if len(actual_tickets)!=count:raise AssertionError("Cartesian geometry not preserved")
    return {"race_id":example["race_id"],"day":example["day"],
            "lines_ranked":["-".join(map(str,x)) for x in ll],
            "favorite_odds":example["favorite_odds"],
            "formation_cartesian_union":forms,
            "distinct_tickets":count,
            "winner":list(example["winner"]),
            "example_set_hit":example["winner"] in actual_tickets,
            "example_stake_yen":100*count,
            "example_payout_yen":100*example["winodds"] if example["winner"] in actual_tickets else 0}
def selection_score(train_months,full,mask,rect_count):
    # Data *discovery* objective locked before evaluation:
    # Both months must have >=2 hits; profitable by sum-of-full-ticket payouts
    # with risk penalty for half-to-half deterioration, ticket breadth and
    # union description complexity. This only selects, not claims superiority.
    x,y=(totals(t,mask) for t in train_months)
    z=totals(full,mask)
    if min(x["races"],y["races"])<12:return None
    if min(x["hits"],y["hits"])<2:return None
    if min(x["stake_yen"],y["stake_yen"])<2000:return None
    if not 8<=z["mean_points"]<=65:return None
    if z["stake_yen"]<16000:return None
    # Conservative objective: worst-month ROI, discount for two parts,
    # and prefer 2-way monthly agreement over posthoc one-hit jackpots.
    min_roi=min(x["roi_pct"],y["roi_pct"])
    score=min_roi-1.5*(rect_count-1)-0.01*z["mean_points"]
    return score,(x,y,z)
def discover(train,test):
    months={m:aggregate([r for r in train if r["day"][:7]==m]) for m in ("2023-01","2023-02")}
    full=aggregate(train)
    hold=aggregate(test)
    arr=[]
    def consider(mask,rects):
        score=selection_score((months["2023-01"],months["2023-02"]),full,mask,len(rects))
        if score is None:return
        v,details=score
        arr.append((v,mask,rects,details))
    for rect in RECTS:consider(rect[3],[rect])
    # Data-derived union search: top 48 rectangles by conservative objective
    # plus top 48 by profitability-adjusted market hit gap are candidate atoms.
    single=sorted(arr,key=lambda x:(-x[0],-x[3][2]["stake_yen"]))[:48]
    # Allow unions even if constituents individually miss stable criteria.
    # All 343 rectangles represented; rank positive net component profit with
    # a breadth penalty to select 40 more atomic building blocks.
    def component(rect):
        mask=rect[3];z=totals(full,mask)
        return (z["profit_yen"]/(100*len(train))-0.3*z["mean_points"])
    bonus=sorted(RECTS,key=component,reverse=True)[:48]
    atoms={r[3]:r for r in bonus}
    for _,_,rects,_ in single:
        atom=rects[0];atoms[atom[3]]=atom
    atomlist=list(atoms.values())
    for ii in range(len(atomlist)):
        for jj in range(ii+1,len(atomlist)):
            aa,bb=atomlist[ii],atomlist[jj]
            union=aa[3]|bb[3]
            if union in (aa[3],bb[3]):continue
            consider(union,[aa,bb])
    if not arr:return {"status":"NO_DISCOVERABLE_STABLE_FORMATION",
                       "reason":"No set satisfied identical January/February feasibility filters.",
                       "train_races":len(train),"test_races":len(test),
                       "rectangle_universe":len(RECTS),"atoms":len(atomlist),"searched_unions":len(atomlist)*(len(atomlist)-1)//2}
    arr.sort(key=lambda x:(-x[0],len(x[2]),-x[3][2]["stake_yen"]))
    score,mask,parts,(jan,feb,train_agg)=arr[0]
    # Absolutely no holdout outcomes or payoffs used until NOW.
    testing=totals(hold,mask)
    by_test_month={m:totals(aggregate([r for r in test if r["day"][:7]==m]),mask)
                   for m in ("2023-03","2023-04")}
    # sample race chosen by earliest date+race ID, never using its outcome.
    sample=min(train,key=lambda r:(r["day"],r["race_id"]))
    return {
      "status":"FORM_FOUND_IN_TRAIN" if train_agg["roi_pct"]>=100 else "DESCRIPTIVE_FORM_ONLY_NO_TRAIN_PROFIT",
      "rank_line_order":"sum of normalized full-market first place 3連単 probability on each line, descending",
      "shape":"union of complete cartesian whole-line slot sets (no individual tickets selected)",
      "coverage_27_line_triplet_cells":mask.bit_count(),
      "union_components":len(parts),
      "line_rank_slot_formation":form_parts(parts),
      "training_score_conservative_worst_month_roi":round(score,3),
      "train_january":jan,"train_february":feb,"train_total":train_agg,
      "heldout_march":by_test_month["2023-03"],
      "heldout_april":by_test_month["2023-04"],
      "heldout_total":testing,
      "train_races":len(train),"heldout_races":len(test),
      "example_real_car_formation":render_example(sample,parts,mask),
      "geometry_search":{
        "distinct_cartesian_rectangles_exhaustively_enumerated":343,
        "union_atom_count":len(atomlist),
        "union_pair_tests_upper_bound":len(atomlist)*(len(atomlist)-1)//2,
        "feasible_candidates":len(arr),
        "note":"General all nonempty subsets of 3 line ranks per slot. Up to two complete rectangles; not a predefined list of 34."}
    }
def main():
    records()
    grp=groups()
    results=[]
    for key,part in sorted(grp.items()):
        tr=part["train"];te=part["test"]
        threshold=60 if key[0]=="4axis" else 100
        if len(tr)<threshold:continue
        discovered=discover(tr,te)
        results.append({"conditioning_axes":{"level":key[0],"line_shape":key[1],
                "class":key[2],"favorite_odds_band":key[3],"race_stage":key[4]},
             **discovered})
        print("DISCOVERED",json.dumps({"group":key,"status":discovered["status"],
             "formation":discovered.get("line_rank_slot_formation"),
             "train_ROI":discovered.get("train_total",{}).get("roi_pct"),
             "holdout_ROI":discovered.get("heldout_total",{}).get("roi_pct"),
             "train":len(tr),"holdout":len(te)},ensure_ascii=False),flush=True)
    main_key={"level":"4axis","line_shape":"3-2-2","class":"A級1・2班",
              "favorite_odds_band":"5-8","race_stage":"準決勝"}
    keyresult=next((x for x in results if x["conditioning_axes"]==main_key),None)
    report={
      "status":"COMPLETE_DISCOVERY_WITH_FROZEN_OUT_OF_SAMPLE_COMPARISON",
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "scope":{"discovery":TRAIN,"frozen_holdout":TEST,
         "years_processed":"2023 only","no_later_2023_years_consumed":True,
         "data_sources":"Sealed full four-market odds archive + published KDreams lines + verified result labels; 3連単 pays, corrected orientation"},
      "research_question":"Discover and compress actual profitable finish-order SET SHAPES, conditional on line shape × class × 1番人気 price × race type, not merely score 34 predefined buying templates.",
      "hard_limits":{"all_market_line_count_exactly":3,
         "line_selection":"ALL 2^3-1 per slot mask combinations, all three rank slots; supports multiple lines on each finishing position",
         "union_components_max":2,
         "rider_roles_in_line_separately_modeled":False,
         "rider_no_preselected":"Each slot includes ALL riders of selected lines, no cherry-picked outcome tickets.",
         "min_train_races_for_four_axes":60,"min_train_races_for_pooled_stage":100,
         "january_and_february_each_min_hits":2,
         "min_mean_tickets":8,"max_mean_tickets":65},
      "audit":dict(A),"line_sources":dict(SOURCES),
      "eligible_groups":len(results),
      "main_4axis_example_discovery":keyresult,
      "groups":results,
      "warnings":[
        "The training data and Mar-Apr dates have been previously examined in this research conversation, so this is a temporal rule-held-out replication, not proof of untouched independent data.",
        "Searching among 343+composite candidate set geometries can produce in-sample false positives. Do not reselect using holdout outcomes.",
        "Marketline ranks are built from full final 3連単 quotes, not a closing-15-minute executable snapshot.",
        "A whole-line set grammar cannot express individual head/follower role subsets; further language learning is needed for full unconstrained formations.",
        "No bet placement or investment advice. No evidence of a market edge unless a frozen rule survives holdout and broader tests.",
        "Only 3-line races and source-matched 3連単 complete odds with published line labels. S grades beyond F1 are not comprehensive.",
        "Multiple testing of cells and union structures; exact odds settle to displayed final values."]
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in report.items() if k!="groups"}
    compact["overview"]=[
       {"condition":x["conditioning_axes"],"status":x["status"],
         "shape":x.get("line_rank_slot_formation"),
         "train":x.get("train_total"),"holdout":x.get("heldout_total")}
       for x in results]
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print("DISCOVERY_COMPLETE",json.dumps({"audit":dict(A),"groups":len(results),
         "main":keyresult},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
