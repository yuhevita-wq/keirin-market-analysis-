from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from . import search_three_year_conditions_v1 as s
YEARS=(2023,2024,2025)
OUT=Path('data/audits/three_year_candidate_neighborhoods.json')

def build():
 d={}
 for y in YEARS:
  races,eb,tri,seg=s.load(y); ys=defaultdict(list)
  for race in races:
   rid=race['race_id']; sec=seg.get(rid)
   if sec not in ('前半','中盤','後半'): continue
   chosen=s.choose_main_line(eb[rid])
   if not chosen: continue
   mid,main=chosen
   if len(main)<3: continue
   rival=s.strongest_rival(eb[rid],mid)
   if not rival or len(rival)<2: continue
   r=s.feature_row(race,eb[rid],mid,main,rival); r['year']=y;r['tri']=tri[rid];r['forms']=s.formations(r,eb[rid]);ys[sec].append(r)
  d[y]=ys
 return d

def stat(d,sec,form,fn):
 out={}
 for y in YEARS:
  rows=[r for r in d[y][sec] if fn(r)]; out[str(y)]=s.fin(rows,form)
 return out

def summarize(grid):
 vals=[]
 for name,st in grid.items():
  rois=[st[str(y)]['roi'] for y in YEARS]; vals.append({'variant':name,'all_years_profitable':min(rois)>1,'worst_year_roi':min(rois),'rois':{str(y):st[str(y)]['roi'] for y in YEARS},'races':{str(y):st[str(y)]['races'] for y in YEARS},'hits':{str(y):st[str(y)]['hits'] for y in YEARS}})
 vals.sort(key=lambda x:x['worst_year_roi'],reverse=True); return vals

def main():
 d=build(); out={'scope':'local coarse threshold-neighborhood audit for preidentified 3-year candidates; no new formation search','candidates':{}}
 # Early robust-large candidate: pair_score_gap<=6 and pair_win_gap<=0, MIX2
 g={}
 for sg in (2,6,10):
  for wg in (-10,0,10): g[f'score_gap_le_{sg}__win_gap_le_{wg}']=stat(d,'前半','MIX2',lambda r,sg=sg,wg=wg:r['pair_score_gap']<=sg and r['pair_win_gap']<=wg)
 out['candidates']['early_pair_gap_mix2']=summarize(g)
 # Middle simple: rival pair score >=200, MIX2
 g={}
 for t in (196,200,204): g[f'rival_score_ge_{t}']=stat(d,'中盤','MIX2',lambda r,t=t:r['rival_score']>=t)
 out['candidates']['middle_rival_score_mix2']=summarize(g)
 # Middle aggressive-large: rival_top3>=80 & pair_top3_gap>=10, MIX2
 g={}
 for rt in (70,80,90):
  for gap in (0,10,25): g[f'rival_top3_ge_{rt}__top3_gap_ge_{gap}']=stat(d,'中盤','MIX2',lambda r,rt=rt,gap=gap:r['rival_top3']>=rt and r['pair_top3_gap']>=gap)
 out['candidates']['middle_top3_mix2']=summarize(g)
 # Late large: R1L score>=100 & main_top2<=70, MAIN4_X
 g={}
 for rs in (98,100,102):
  for mt in (60,70,80): g[f'r1l_score_ge_{rs}__main_top2_le_{mt}']=stat(d,'後半','MAIN4_X',lambda r,rs=rs,mt=mt:r['r1l_score']>=rs and r['main_top2']<=mt)
 out['candidates']['late_r1l_main_top2_main4x']=summarize(g)
 # Late simple: pair_win_gap<=-10, MAIN4_X
 g={}
 for t in (-20,-10,0): g[f'pair_win_gap_le_{t}']=stat(d,'後半','MAIN4_X',lambda r,t=t:r['pair_win_gap']<=t)
 out['candidates']['late_pair_win_gap_main4x']=summarize(g)
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
