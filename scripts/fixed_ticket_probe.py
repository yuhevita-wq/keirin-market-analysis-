import csv,glob,itertools,json,re
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SETS={
 '2024':glob.glob(str(ROOT/'data/2024/s_class_f1_all_parts/2024_q*')),
 '2025':glob.glob(str(ROOT/'data/2025/s_class_f1_all_parts/2025_q*')),
 '2026H1':[str(ROOT/'data/2026_h1/s_class_f1_all')],
}
TYPES=['2車複','2車単','3連複','3連単','ワイド','2枠複','2枠単']
UNORD={'2車複','3連複','ワイド','2枠複'}

def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f: yield from csv.DictReader(f)

def norm(tt,s):
 a=[int(x) for x in re.findall(r'\d+',s or '')]
 n=3 if tt.startswith('3連') else 2
 if len(a)!=n:return None
 if tt in UNORD:a.sort()
 return tuple(a)

def universe(tt,n=6):
 k=3 if tt.startswith('3連') else 2
 return list(itertools.combinations(range(1,n+1),k)) if tt in UNORD else list(itertools.permutations(range(1,n+1),k))

# per year: active race ids, seven-car race ids, paid returns by (type,combo), blanket refunds
D={}
for y,dirs in SETS.items():
 active=defaultdict(set); seven=set(); paid=defaultdict(lambda:defaultdict(int)); refund=defaultdict(set); allr=set()
 for d in dirs:
  rp=Path(d)/'races.csv'
  if rp.exists():
   for r in rows(rp):
    rid=r['race_id'];allr.add(rid)
    if int(r.get('entry_count') or 0)==7:seven.add(rid)
  pp=Path(d)/'payouts.csv'
  if pp.exists():
   for r in rows(pp):
    tt=r.get('ticket_type');rid=r.get('race_id');st=r.get('status','')
    if tt not in TYPES:continue
    if st!='not_offered':active[tt].add(rid)
    if st=='refund' and not r.get('combination'):refund[tt].add(rid)
    if st=='paid':
     c=norm(tt,r.get('combination',''))
     if c: paid[(tt,c)][rid]=max(paid[(tt,c)][rid],int(float(r.get('payout_yen') or 0)))
 D[y]=(active,seven,paid,refund,allr)

def ev(y,tt,c,seven_only=False):
 active,seven,paid,refund,_=D[y]; rs=active[tt] & seven if seven_only else active[tt]
 n=len(rs)
 ret=sum(v for rid,v in paid[(tt,c)].items() if rid in rs)+100*len(refund[tt]&rs)
 hits=sum(1 for rid in paid[(tt,c)] if rid in rs)
 return {'races':n,'hits':hits,'return':ret,'stake':100*n,'roi':ret/(100*n) if n else None,'profit':ret-100*n}

def best(tt,year,seven_only=False):
 n=6 if tt.startswith('2枠') else (7 if seven_only else 6)
 vals=[(ev(year,tt,c,seven_only)['roi'] or -1,c) for c in universe(tt,n)]
 return max(vals)[1]

out={'race_counts':{y:len(D[y][4]) for y in D},'strict_1to6':{},'seven_car_only':{}}
for mode,seven_only in [('strict_1to6',False),('seven_car_only',True)]:
 for tt in TYPES:
  c=best(tt,'2024',seven_only)
  yrs={y:ev(y,tt,c,seven_only) for y in D}
  n=6 if tt.startswith('2枠') else (7 if seven_only else 6)
  pos=[]
  for x in universe(tt,n):
   z={y:ev(y,tt,x,seven_only) for y in D}
   if all(z[y]['roi'] is not None and z[y]['roi']>1 for y in D): pos.append((min(z[y]['roi'] for y in D),x,z))
  pos.sort(reverse=True,key=lambda q:q[0])
  allbest=[]
  for x in universe(tt,n):
   z=[ev(y,tt,x,seven_only) for y in D]; st=sum(v['stake'] for v in z); rt=sum(v['return'] for v in z)
   if st: allbest.append((rt/st,x,rt-st,sum(v['hits'] for v in z)))
  allbest.sort(reverse=True)
  out[mode][tt]={'pick_2024':list(c),'by_year':yrs,'cross_year_positive_count':len(pos),'cross_year_positive_top':[{'combo':list(x),'min_roi':m,'by_year':z} for m,x,z in pos[:10]],'hindsight_best':({'combo':list(allbest[0][1]),'roi':allbest[0][0],'profit':allbest[0][2],'hits':allbest[0][3]} if allbest else None)}

p=ROOT/'results/fixed_ticket_all_f1_20260904.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
