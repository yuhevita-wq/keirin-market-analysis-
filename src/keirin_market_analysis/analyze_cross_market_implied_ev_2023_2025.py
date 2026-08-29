from __future__ import annotations
import json
from pathlib import Path
from .analyze_odds_contradictions_2023_2025 import YEARS, load_results, load_odds, normalize_market
OUT=Path('data/audits/cross_market_implied_ev_2023_2025.json')
BINS=[(-1e9,0.70,'lt_0.70'),(0.70,0.85,'0.70_0.85'),(0.85,1.0,'0.85_1.00'),(1.0,1.10,'1.00_1.10'),(1.10,1.25,'1.10_1.25'),(1.25,1e9,'ge_1.25')]
THRESHOLDS=(1.0,1.05,1.10,1.20,1.25,1.50)

def new(): return {'bets':0,'wins':0,'races_with_bet':set(),'gross_return':0.0,'sum_xev':0.0}
def add(a,rid,win,odds,xev):
 a['bets']+=1;a['wins']+=int(win);a['races_with_bet'].add(rid);a['sum_xev']+=xev
 if win:a['gross_return']+=odds
def finish(a,total_races):
 return {'bets':a['bets'],'wins':a['wins'],'races_with_bet':len(a['races_with_bet']),
 'race_purchase_rate_pct':len(a['races_with_bet'])/total_races*100 if total_races else None,
 'avg_bets_per_purchased_race':a['bets']/len(a['races_with_bet']) if a['races_with_bet'] else None,
 'hit_rate_pct':a['wins']/a['bets']*100 if a['bets'] else None,
 'roi_pct_final_odds':a['gross_return']/a['bets']*100 if a['bets'] else None,
 'mean_cross_market_implied_ev':a['sum_xev']/a['bets'] if a['bets'] else None}

def analyze_year(y):
 b=Path(f'data/{y}/s_class_yosen');res=load_results(b);trio=load_odds(b,'trio_final_odds.csv','=');tri3=load_odds(b,'trifecta_final_odds.csv','-')
 bins={name:new() for _,_,name in BINS}; th={str(t):new() for t in THRESHOLDS}; eligible=0
 for rid in sorted(set(res)&set(trio)&set(tri3)):
  wo=res[rid];ws=tuple(sorted(wo));to=trio[rid];oo=tri3[rid];cars=sorted({c for s in to for c in s})
  if len(cars)<5 or len(to)!=(len(cars)*(len(cars)-1)*(len(cars)-2)//6) or len(oo)!=(len(cars)*(len(cars)-1)*(len(cars)-2)):continue
  qt=normalize_market(to);qo=normalize_market(oo);qs={s:0.0 for s in qt}
  for o,p in qo.items():qs[tuple(sorted(o))]+=p
  eligible+=1
  for s,odds in to.items():
   xev=qs[s]*odds;win=s==ws
   for lo,hi,name in BINS:
    if lo<=xev<hi:add(bins[name],rid,win,odds,xev);break
   for t in THRESHOLDS:
    if xev>=t:add(th[str(t)],rid,win,odds,xev)
 return {'year':y,'eligible_complete_races':eligible,'bins':{k:finish(v,eligible) for k,v in bins.items()},'thresholds':{k:finish(v,eligible) for k,v in th.items()}}

def combine(years,section,keys):
 out={}
 total=sum(y['eligible_complete_races'] for y in years)
 for k in keys:
  a=new()
  # race ids collide only within year? prefix year to ensure distinct.
  for y in years:
   x=y[section][k]
   a['bets']+=x['bets'];a['wins']+=x['wins'];a['gross_return']+=x['roi_pct_final_odds']/100*x['bets'] if x['roi_pct_final_odds'] is not None else 0
   a['sum_xev']+=x['mean_cross_market_implied_ev']*x['bets'] if x['mean_cross_market_implied_ev'] is not None else 0
   for i in range(x['races_with_bet']):a['races_with_bet'].add((y['year'],i))
  out[k]=finish(a,total)
 return out

def main():
 years=[analyze_year(y) for y in YEARS]
 out={'status':'CROSS_MARKET_IMPLIED_EV_2023_2025','years_read':list(YEARS),'evaluation_year_2026_used':False,'odds_phase':'final',
 'definition':'xEV = normalized trifecta probability aggregated to the 3-rider set * archived final trio odds. xEV>=1 is the natural no-result-used threshold where the trifecta-derived set probability would imply nonnegative gross expectation at the trio price.',
 'important_limit':'Final odds only, not executable T-10 backtest. Thresholds above 1 are descriptive robustness levels, not frozen strategy optimization.',
 'years':years,'combined':{
  'bins':combine(years,'bins',[x[2] for x in BINS]),
  'thresholds':combine(years,'thresholds',[str(t) for t in THRESHOLDS])}}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
