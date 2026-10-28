"""FROZEN prospective-style historical replication of discover_2023_four_months.py.
Read ONLY May 1-Jun 30 2023 odds/finishes. No fitting, new formation search,
post-outcome selection, point pruning, model changes, or revised odds bins.
The 4 candidate geometries below are exact output from the PRECEDING fixed
Jan-Feb train / Mar-Apr test run, committed before this source file.
"""
from __future__ import annotations
import sys,os,json,math,tempfile,itertools
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.formation_discovery_from_outcomes_20261028 import discover_2023_four_months as base
from research.snowberg_observation_corrected_20261027 import observe as corrected
bridge=base.bridge
BASE_RESULT=ROOT/"research/formation_discovery_from_outcomes_20261028/summary.json"
OUT=ROOT/"research/frozen_formation_sets_replication_2023_mayjun"
START="2023-05-01";END="2023-06-30"
SOURCE_FILES=(
 "data/a_class_12/2023_05_06/races.csv",
 "data/2023/s_class_f1_all_parts/2023_q2/races.csv")
# 1-based MARKET LINE RANKS (line first-place support from all 3連単).
# All were mined *before* May-Jun was loaded; they are not programmed templates.
FROZEN=(
 {"id":"S_322_5to8_stages_pooled",
  "source_stage":"all-stages","line_shape":"3-2-2","class":"S級","favorite_band":"5-8",
  "parts":[([1],[2,3],[2]),([3],[2],[1])]},
 {"id":"S_322_8to12_stages_pooled",
  "source_stage":"all-stages","line_shape":"3-2-2","class":"S級","favorite_band":"8-12",
  "parts":[([1],[3],[2,3]),([2],[2],[2])]},
 {"id":"S_322_5to8_preliminary",
  "source_stage":"予選","line_shape":"3-2-2","class":"S級","favorite_band":"5-8",
  "parts":[([3],[1,2],[3])]},
 {"id":"A_322_5to8_semifinal_negative_control",
  "source_stage":"準決勝","line_shape":"3-2-2","class":"A級1・2班","favorite_band":"5-8",
  "parts":[([1],[2],[2]),([2],[1],[1])]}
)
def cells_for(parts):
    mask=0
    for a,b,c in parts:
        for i in a:
            for j in b:
                for k in c:
                    assert i in (1,2,3) and j in (1,2,3) and k in (1,2,3)
                    mask|=1<<(((i-1)*3+j-1)*3+k-1)
    return mask
def verify_against_saved():
    prior=json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    assert prior["status"]=="COMPLETE_DISCOVERY_WITH_FROZEN_OUT_OF_SAMPLE_COMPARISON"
    for spec in FROZEN:
        found=[g for g in prior["groups"] if
               g["conditioning_axes"]["line_shape"]==spec["line_shape"] and
               g["conditioning_axes"]["class"]==spec["class"] and
               g["conditioning_axes"]["favorite_odds_band"]==spec["favorite_band"] and
               g["conditioning_axes"]["race_stage"]==spec["source_stage"]]
        if len(found)!=1:raise AssertionError("Missing unique frozen 2023 Jan-Apr rule "+spec["id"])
        actual=found[0]["line_rank_slot_formation"]
        expected=[{"first_line_ranks":a,"second_line_ranks":b,"third_line_ranks":c}
                  for a,b,c in spec["parts"]]
        if actual!=expected:
            raise AssertionError("Frozen geometry differs from prior discovery: "+spec["id"])
    return prior
def read_selected_months():
    if not os.environ.get(bridge.SECRET_ENV):raise RuntimeError("Sealed archive key missing")
    base.TRAIN=("2023-01-01","2023-04-30")
    base.TEST=(START,END)
    base.INPUTS=SOURCE_FILES
    base.load_lines()
    labels,_=bridge.verified_results(2023)
    catalog,meta=bridge.catalog(START,END)
    base.A["encrypted_files"]=meta["sealed_files"]
    corrected.base_observe=base.observe
    kt,key=bridge.secure_key_temp()
    try:
        for day,files in catalog.items():
            if not START<=day<=END:raise AssertionError("Unexpected historical odds date")
            combined={};bad=set()
            with tempfile.TemporaryDirectory(prefix="discovered_form_replication_") as tmp:
                for path,sha in files:
                    for rid,rows in bridge.decrypt_sealed(path,sha,key,tmp).items():
                        if rid in bad:continue
                        canon=sorted(rows)
                        if rid in combined and combined[rid]!=canon:
                            bad.add(rid);combined.pop(rid,None);continue
                        combined[rid]=canon
            base.A["archive_conflicts"]+=len(bad)
            for rid,rows in combined.items():
                if rid not in base.LINE:continue
                # Pre-outcome eligibility gate: ALL frozen rules were trained on
                # exactly 3-2-2; avoid unrelated 5-/6-/9-car races with
                # different public result labels or no actionable formation.
                published=base.LINE[rid]
                if len(published[1])!=3 or sorted((len(line) for line in published[1]),reverse=True)!=[3,2,2]:
                    base.A["excluded_not_frozen_322_before_reading_finish"]+=1
                    continue
                _,err=bridge.normalize_book(rid,day,rows)
                if err:
                    base.A["bad_norm_"+err]+=1;continue
                if rid not in labels:
                    base.A["missing_results"]+=1;continue
                books=defaultdict(dict)
                for z in rows:
                    typ=z[2]
                    if typ not in ("3rentan","3renpuku","2shatan","2shafuku"):continue
                    lo,hi=float(z[4]),float(z[5])
                    if lo!=hi:raise AssertionError("Ranges not permitted")
                    numbers=tuple(map(int,z[3].split("-")))
                    if typ in ("3renpuku","2shafuku"):numbers=tuple(sorted(numbers))
                    books[typ][numbers]=lo
                corrected.corrected_observe(rid,day,books,labels[rid],rows[0][1],"sealed_four_markets")
            base.A["days"]+=1
            if base.A["days"]%20==0:print("PROGRESS",day,len(base.RACES),flush=True)
    finally:kt.cleanup()
    if len(base.RACES)<800:raise RuntimeError("Insufficient May-Jun aligned 3-line races")
    if any(not START<=r["day"]<=END for r in base.RACES):
        raise AssertionError("Invalid OOS date used")
def totals_month(sub,mask):
    if not sub:return {"races":0,"status":"NO_MATCHING_RACES"}
    result=base.totals(base.aggregate(sub),mask)
    # Preserve proof that "training price selection" cannot be made inside OOS.
    assert result["stake_yen"]==100*result["tickets"]
    return result
def main():
    prior=verify_against_saved()
    read_selected_months()
    reports=[]
    for spec in FROZEN:
        subset=[r for r in base.RACES if
          r["line_shape"]==spec["line_shape"] and
          r["class"]==spec["class"] and
          r["favorite_band"]==spec["favorite_band"] and
          (spec["source_stage"]=="all-stages" or r["stage"]==spec["source_stage"])]
        mask=cells_for(spec["parts"])
        may=[r for r in subset if r["day"][:7]=="2023-05"]
        jun=[r for r in subset if r["day"][:7]=="2023-06"]
        if not may or not jun:raise RuntimeError("May-Jun segment empty: "+spec["id"])
        example=base.render_example(min(subset,key=lambda r:(r["day"],r["race_id"])),
                                   [tuple((sum(1<<(i-1) for i in k) for k in part))+ (0,)
                                    for part in spec["parts"]],mask)
        # Make true tuple rect (a,b,c,bitmask) for example rendering
        reports.append({"id":spec["id"],
           "class":spec["class"],"line_shape":spec["line_shape"],
           "favorite_band":spec["favorite_band"],"race_stage":spec["source_stage"],
           "frozen_rank_line_formation_parts":spec["parts"],
           "market_line_cell_mask":mask,
           "may":totals_month(may,mask),"june":totals_month(jun,mask),
           "both":totals_month(subset,mask),
           "first_matching_race_example":example})
        print("FROZEN_RESULT",json.dumps({k:reports[-1][k] for k in ("id","may","june","both")},ensure_ascii=False),flush=True)
    report={"status":"COMPLETE_NO_REFIT_MAY_JUNE_2023",
       "generated_utc":datetime.now(timezone.utc).isoformat(),
       "discovery_source":"research/formation_discovery_from_outcomes_20261028/summary.json",
       "period":[START,END],"frozen_count":len(FROZEN),
       "audit":dict(base.A),"aligned_3line_races":len(base.RACES),
       "warnings":["No retuning, not an invitation to pick best May-Jun family after seeing this table.",
          "Previous 2023 race data may have been considered in other earlier research; the locked code paths here do not access May-Jun to fit the geometries.",
          "Only full 3-line markets with validated source line information; final odds, not confirmed 15-minute placement.",
          "Full distinct-ticket formation baskets at 100 yen per ticket; not individual outcome selections.",
          "Multiple outcomes and geometry groups were inspected during 2023 Jan-Feb discovery; later positive result still exploratory without longer prospective test."],
       "results":reports}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("FROZEN_MAY_JUNE_COMPLETED",json.dumps({"audit":dict(base.A),
          "results":[{"id":v["id"],"roi":v["both"]["roi_pct"],"profit":v["both"]["profit_yen"],"n":v["both"]["races"]} for v in reports]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
