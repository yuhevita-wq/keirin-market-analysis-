from __future__ import annotations
import csv,json
from pathlib import Path
SCHEME_VERSION="v6.3-D01";BASE_SCHEME_VERSION="v6.1";DEVELOPMENT_DATASET="2024Q1";ANALYSIS_SOURCE="v6.1-HM01";STATUS="DEVELOPMENT";ENTRY_FILTER_NAME="M_PRE_FORMATION_MASS";ENTRY_FILTER_THRESHOLD=0.35640013538348414;STAKE=100
ROOT=Path(__file__).resolve().parents[2];IN_CSV=ROOT/"artifacts/v6_1_hm01_2024q1/v6_1_hm01_races.csv";OUT=ROOT/"artifacts/v6_3_d01_2024q1"
def ti(x):
 try:return int(float(x))
 except:return 0
def tf(x):
 try:return float(x)
 except:return None
def streak(rs):
 cur=best=0
 for r in sorted(rs,key=lambda r:(r['race_date'],r['race_id'])):
  if ti(r['hit']):cur=0
  else:cur+=1;best=max(best,cur)
 return best
def summary(rs):
 b=len(rs);h=sum(ti(r['hit']) for r in rs);t=sum(ti(r['final_bet_count']) for r in rs);p=sum(ti(r['payout_yen']) for r in rs);s=t*STAKE
 return {'bet_races':b,'hit_races':h,'miss_races':b-h,'hit_rate_pct':100*h/b if b else None,'tickets':t,'avg_tickets_per_race':t/b if b else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None,'max_losing_streak':streak(rs)}
def main():
 with IN_CSV.open('r',encoding='utf-8-sig',newline='') as f:base=list(csv.DictReader(f))
 sel=[];rem=[]
 for r in base:(sel if tf(r.get('M_pre')) is not None and tf(r.get('M_pre'))>=ENTRY_FILTER_THRESHOLD else rem).append(r)
 result={'scheme_version':SCHEME_VERSION,'base_scheme_version':BASE_SCHEME_VERSION,'development_dataset':DEVELOPMENT_DATASET,'analysis_source':ANALYSIS_SOURCE,'status':STATUS,'base_v6_1':summary(base),'v6_3_d01':summary(sel),'removed_by_filter':summary(rem)}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'v6_3_d01_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
 main()
 from analyze_v8_4_f05_hit_payout_q1 import main as analyze_payout
 analyze_payout()
