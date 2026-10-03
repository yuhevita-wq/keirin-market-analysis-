from __future__ import annotations
import csv,io,json,math,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ARCH=ROOT/'data'/'grade_races'/'g3'; OUT=ROOT/'results'/'g3_day3_reality'
def dec(b):
 for e in ('utf-8-sig','utf-8','cp932','shift_jis'):
  try:return b.decode(e)
  except:pass
 return b.decode('utf-8',errors='replace')
def read(z,n):
 m=[x for x in z.namelist() if Path(x).name==n]; return list(csv.DictReader(io.StringIO(dec(z.read(m[0]))))) if m else []
def f(x,d=-999):
 try:return float(x)
 except:return d
def day(rid):
 try:return int(rid[10:12]) if len(rid)==16 and rid.isdigit() else None
 except:return None
def lines(es):
 d=defaultdict(list)
 for e in es:
  if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():d[int(e['line_id'])].append(e)
 o=[]
 for lid,m in d.items():
  m.sort(key=lambda x:int(x['line_position']))
  if len(m)>=2:o.append(((f(m[0].get('score'))+f(m[1].get('score')),f(m[0].get('score')),1 if len(m)>=3 else 0,-lid),m))
 return [m for _,m in sorted(o,reverse=True)]
def comb(xs):return '='.join(map(str,sorted(xs)))
def evals(rows,sel):
 n=h=st=ret=0; pts=[]
 for r in rows:
  cs=sel(r)
  if not cs:continue
  n+=1;pts.append(len(cs));st+=100*len(cs);got=sum(r['pay'].get(c,0) for c in cs);ret+=got;h+=got>0
 return {'races':n,'hits':h,'hit_rate':h/n if n else 0,'avg_points':sum(pts)/len(pts) if pts else 0,'roi':ret/st if st else 0,'stake':st,'return':ret}
def main():
 races={};E=defaultdict(list);R=defaultdict(list);P=defaultdict(list);O=defaultdict(dict)
 for p in ARCH.rglob('*.zip'):
  with zipfile.ZipFile(p) as z:
   for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
   for x in read(z,'entries.csv'):E[x['race_id']].append(x)
   for x in read(z,'results.csv'):R[x['race_id']].append(x)
   for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
   for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x
 rows=[]
 for rid,r in races.items():
  if day(rid)!=3:continue
  es=E[rid];ls=lines(es)
  if len(ls)<2:continue
  fin=[]
  for x in R[rid]:
   try:p=int(x.get('finish_position') or 99);c=int(x.get('car_no') or 0)
   except:continue
   if p<=3:fin.append((p,c))
  if len(fin)!=3 or sorted(p for p,_ in fin)!=[1,2,3]:continue
  m,rv=ls[0],ls[1];mc=[int(x['car_no']) for x in m];d=int(rv[0]['car_no']);e=int(rv[1]['car_no'])
  a,b=mc[:2];c=mc[2] if len(mc)>=3 else None
  abc=comb((a,b,c)) if c else None; ao=O[rid].get(abc,{}) if abc else {};rank=int(ao['market_rank']) if ao.get('market_rank','').isdigit() else 999
  pair_top3=f(m[0].get('top3_rate'),0)+f(m[1].get('top3_rate'),0); second_top3=f(m[1].get('top3_rate'),0)
  weak=pair_top3<=106.1 and second_top3<=35.7
  # route: main line is only 2 riders OR 3+ line whose ABC trio is outside top5, AND weak main-pair profile
  route=(len(mc)==2 or rank>5) and weak
  if not route:continue
  score=sorted(es,key=lambda x:(-f(x.get('score')),int(x['car_no'])));thirds=[int(x['car_no']) for x in score if int(x['car_no']) not in {d,e}][:4]
  cand=[comb((d,e,x)) for x in thirds]
  odds={c:f(O[rid].get(c,{}).get('odds'),-1) for c in cand}
  pay={}
  for q in P[rid]:
   if q.get('ticket_type')=='3連複' and q.get('status')=='paid':
    try:pay[q['combination']]=int(float(q.get('payout_yen') or 0))
    except:pass
  rows.append({'year':int(r['race_date'][:4]),'cand':cand,'odds':odds,'pay':pay})
 tr=[x for x in rows if x['year']<=2024];te=[x for x in rows if x['year']>=2025]
 sels={
 'all4':lambda r:r['cand'],
 'top2_high_odds':lambda r:sorted(r['cand'],key=lambda c:r['odds'].get(c,-1),reverse=True)[:2],
 'top3_high_odds':lambda r:sorted(r['cand'],key=lambda c:r['odds'].get(c,-1),reverse=True)[:3],
 'odds>=10':lambda r:[c for c in r['cand'] if r['odds'].get(c,-1)>=10],
 'odds>=15':lambda r:[c for c in r['cand'] if r['odds'].get(c,-1)>=15],
 'odds>=20':lambda r:[c for c in r['cand'] if r['odds'].get(c,-1)>=20],
 'odds>=25':lambda r:[c for c in r['cand'] if r['odds'].get(c,-1)>=25],
 'odds15-80':lambda r:[c for c in r['cand'] if 15<=r['odds'].get(c,-1)<=80],
 'odds20-100':lambda r:[c for c in r['cand'] if 20<=r['odds'].get(c,-1)<=100],
 }
 out={k:{'train':evals(tr,v),'test':evals(te,v)} for k,v in sels.items()}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'outside_value.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 md=['# G3三日目 外型3連複 価格フィルター','',f'route races train={len(tr)} test={len(te)}','']
 for k,v in out.items():
  a=v['train'];b=v['test'];md.append(f"- {k}: train {a['races']}R {a['avg_points']:.1f}点 hit {a['hit_rate']:.1%} ROI {a['roi']:.1%} / test {b['races']}R {b['avg_points']:.1f}点 hit {b['hit_rate']:.1%} ROI {b['roi']:.1%}")
 (OUT/'outside_value.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
if __name__=='__main__':main()
