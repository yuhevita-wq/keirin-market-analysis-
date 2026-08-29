from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from . import search_three_year_conditions_v1 as s

YEARS=(2023,2024,2025)
FORMS={'前半':['RIVAL4','MAIN4_RIVAL','MIX2'],'中盤':['MAIN6','MAIN4_RIVAL','RIVAL4','MIX2'],'後半':['MAIN4_X','MAIN6','MAIN4_RIVAL','RIVAL4']}
OUT=Path('data/audits/three_year_conservative_selection.json')

def build():
 d={}
 for y in YEARS:
  races,eb,tri,seg=s.load(y); ys=defaultdict(list)
  for race in races:
   rid=race['race_id']; sec=seg.get(rid)
   if sec not in FORMS: continue
   chosen=s.choose_main_line(eb[rid])
   if not chosen: continue
   mid,main=chosen
   if len(main)<3: continue
   rival=s.strongest_rival(eb[rid],mid)
   if not rival or len(rival)<2: continue
   r=s.feature_row(race,eb[rid],mid,main,rival); r['year']=y; r['tri']=tri[rid]; r['forms']=s.formations(r,eb[rid]); ys[sec].append(r)
  d[y]=ys
 return d

def q(st,minr): return st['races']>=minr and st['hits']>=3 and st['roi']>1 and st['top1_payout_share']<=.70 and min(h['hits'] for h in st['halves'])>=1

def score(stats,years,complexity):
 st=sum(stats[y]['stake_yen'] for y in years); pay=sum(stats[y]['payout_yen'] for y in years)
 return (min(stats[y]['roi'] for y in years),pay/st if st else 0,-complexity)

def select(d,sec,train,mode):
 if mode=='simple': rules=[(a[0],[a]) for a in s.ATOMS]; minr=10
 else:
  rules=[(a[0],[a]) for a in s.ATOMS]+[(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(s.ATOMS) for b in s.ATOMS[i+1:] if s.compatible(a,b)]; minr=30
 best=None; bk=None; ba=None
 for rn,atoms in rules:
  sel={y:[r for r in d[y][sec] if all(s.passes(r,a) for a in atoms)] for y in train}
  if min(len(sel[y]) for y in train)<minr: continue
  for form in FORMS[sec]:
   stats={y:s.fin(sel[y],form) for y in train}
   if not all(q(stats[y],minr) for y in train): continue
   k=score(stats,train,len(atoms))
   if best is None or k>bk:
    st=sum(stats[y]['stake_yen'] for y in train); pay=sum(stats[y]['payout_yen'] for y in train)
    best={'rule':rn,'complexity':len(atoms),'formation':form,'train':{str(y):stats[y] for y in train},'train_worst_roi':min(stats[y]['roi'] for y in train),'train_combined_roi':pay/st}; bk=k; ba=atoms
 return best,ba

def main():
 d=build(); out={'scope':'conservative leave-one-year-out selection on 2023-2025; fresh 2022 still required','modes':{'simple':'one condition only','large':'one/two conditions but >=30 races in each training year'},'segments':{}}
 for sec in FORMS:
  out['segments'][sec]={}
  for mode in ('simple','large'):
   z={}
   for held in YEARS:
    train=[y for y in YEARS if y!=held]; best,atoms=select(d,sec,train,mode)
    if not best: z[str(held)]={'selected':None}; continue
    rows=[r for r in d[held][sec] if all(s.passes(r,a) for a in atoms)]; hs=s.fin(rows,best['formation'])
    z[str(held)]={'selected':best,'heldout':hs,'heldout_profitable':hs['roi']>1}
   out['segments'][sec][mode]=z
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
