from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
from simulate_v8_1_f02_2024q1 import load,pi,pl,streak,STAKE
from v8_3_f04_modeled_profit import build_v8_3_f04
SCHEME="v8.3-F04";ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"artifacts/v8_3_f04_2024q1"
def main():
 races,trio,tf,pay=load();out=[];fail=Counter();sizes=Counter();forms=Counter();attr=defaultdict(int);attr["races_csv"]=len(races)
 for rid,r in sorted(races.items(),key=lambda kv:(kv[1].get("race_date",""),kv[0])):
  if r.get("meeting_grade")!="F1" or not (r.get("race_type") or "").startswith("Ｓ級"):continue
  attr["f1_s"]+=1
  if pi(r.get("entry_count"))!=7:continue
  attr["seven_car"]+=1
  if len(trio.get(rid,{}))!=35:continue
  attr["complete_trio35"]+=1
  if len(tf.get(rid,{}))!=210:continue
  attr["complete_tf210"]+=1
  cars=sorted({v for c in trio[rid] for v in c});lines=pl(r.get("predicted_line_formation"))
  if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars):continue
  attr["complete_line"]+=1
  if rid not in pay or not pay[rid]:continue
  attr["has_tf_payout"]+=1;attr["population"]+=1
  d=build_v8_3_f04(trio[rid],tf[rid],r.get("predicted_line_formation") or "")
  if not d.get("buy"):fail[str(d.get("reason"))]+=1;continue
  attr["entry_pass"]+=1;ts=tuple(d["tickets"]);wins=[t for t in ts if t in pay[rid]];payout=sum(pay[rid][t] for t in wins)
  size=f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}";sizes[size]+=1;forms[d["formation"]]+=1
  out.append({"race_id":rid,"race_date":r.get("race_date"),"track":r.get("track"),"race_no":pi(r.get("race_no")),"formation":d["formation"],"size_pattern":size,"ticket_count":d["ticket_count"],"modeled_profit":d["modeled_profit"],"modeled_roi":d["modeled_roi"],"q_mass":d["q_mass"],"q_star_mass":d["q_star_mass"],"hit":int(bool(wins)),"payout_yen":payout,"winning_ticket":";".join("-".join(map(str,t)) for t in wins)})
 attr["bet_races"]=len(out);h=sum(r["hit"] for r in out);n=sum(r["ticket_count"] for r in out);p=sum(r["payout_yen"] for r in out);s=n*STAKE
 result={"scheme_version":SCHEME,"dataset":"2024Q1","status":"Q1_DEVELOPMENT_SIMULATION","entry_rule":"unchanged v6.1 pre-formation gate","formation_rule":"independent clean F1-F2-F3 maximizing sum(q_star*odds-1)","fixed_place_counts":False,"fixed_point_count":False,"price_cut":False,"nested_required":False,"attrition":dict(attr),"entry_fail_reasons":dict(fail),"summary":{"bet_races":len(out),"hit_races":h,"miss_races":len(out)-h,"hit_rate_pct":100*h/len(out) if out else None,"tickets":n,"avg_tickets_per_race":n/len(out) if out else None,"min_tickets_per_race":min((r["ticket_count"] for r in out),default=None),"max_tickets_per_race":max((r["ticket_count"] for r in out),default=None),"stake_yen":s,"payout_yen":p,"profit_yen":p-s,"roi_pct":100*p/s if s else None,"max_losing_streak":streak(out),"mean_modeled_roi":sum(r["modeled_roi"] for r in out)/len(out) if out else None},"size_pattern_distribution":dict(sorted(sizes.items(),key=lambda kv:(-kv[1],kv[0]))),"top_20_exact_formations":dict(forms.most_common(20))}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"v8_3_f04_2024q1_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 if out:
  with (OUT/"v8_3_f04_2024q1_races.csv").open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
 print("V8_3_F04_Q1_RESULT_BEGIN");print(json.dumps(result,ensure_ascii=False,indent=2));print("V8_3_F04_Q1_RESULT_END")
if __name__=="__main__":main()
