from __future__ import annotations

import csv, io, json, math, zipfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCH=ROOT/'data'/'grade_races'/'g3'
OUT=ROOT/'results'/'g3_day3_reality'

def dec(b):
    for e in ('utf-8-sig','utf-8','cp932','shift_jis'):
        try:return b.decode(e)
        except UnicodeDecodeError:pass
    return b.decode('utf-8',errors='replace')

def read(z,n):
    m=[x for x in z.namelist() if Path(x).name==n]
    return list(csv.DictReader(io.StringIO(dec(z.read(m[0]))))) if m else []

def f(v,d=-999):
    try:return float(v)
    except:return d

def day(rid):
    try:return int(rid[10:12]) if len(rid)==16 and rid.isdigit() else None
    except:return None

def lines(es):
    d=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit(): d[int(e['line_id'])].append(e)
    out=[]
    for lid,mem in d.items():
        mem.sort(key=lambda x:int(x['line_position']))
        if len(mem)>=2:
            key=(f(mem[0].get('score'))+f(mem[1].get('score')),f(mem[0].get('score')),1 if len(mem)>=3 else 0,-lid)
            out.append((key,lid,mem))
    return sorted(out,reverse=True)

def combo(xs):return '='.join(map(str,sorted(xs)))
def form(groups):
    out=set()
    for a in groups[0]:
      for b in groups[1]:
       for c in groups[2]:
        if len({a,b,c})==3:out.add(combo((a,b,c)))
    return out

def valid(xs):return [x for x in xs if x is not None]
def topoutside(es,exc,n):
    xs=sorted(es,key=lambda x:(-f(x.get('score')),int(x['car_no'])))
    return [int(x['car_no']) for x in xs if int(x['car_no']) not in exc][:n]

def metrics(rows,selector,pred):
    z=[r for r in rows if pred(r)]
    R=H=S=P=0; pts=[]; off=offh=0
    for r in z:
        cs=selector(r)
        if not cs:continue
        R+=1; pts.append(len(cs)); S+=100*len(cs)
        got=sum(r['pay'].get(c,0) for c in cs); P+=got; H+=got>0
        if not r['win_main']:off+=1;offh+=got>0
    return {'races':R,'hits':H,'hit_rate':H/R if R else 0,'avg_points':sum(pts)/len(pts) if pts else 0,'stake':S,'return':P,'roi':P/S if S else 0,'off_rate':off/R if R else 0,'off_capture':offh/off if off else 0}

def main():
    races={}; E=defaultdict(list); R=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for p in sorted(ARCH.rglob('*.zip')):
      with zipfile.ZipFile(p) as z:
       for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
       for x in read(z,'entries.csv'):E[x['race_id']].append(x)
       for x in read(z,'results.csv'):R[x['race_id']].append(x)
       for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
       for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x
    rows=[]
    for rid,r in races.items():
      if day(rid)!=3:continue
      es=E[rid]; ls=lines(es)
      if not ls:continue
      fin=[]
      for x in R[rid]:
       try:pos=int(x.get('finish_position') or 99);car=int(x.get('car_no') or 0)
       except:continue
       if pos<=3:fin.append((pos,car))
      if len(fin)!=3 or sorted(x[0] for x in fin)!=[1,2,3]:continue
      top3=[c for _,c in sorted(fin)]
      m=ls[0][2]; rv=ls[1][2] if len(ls)>1 else []
      mc=[int(x['car_no']) for x in m]; rc=[int(x['car_no']) for x in rv]
      a,b=mc[:2]; c=mc[2] if len(mc)>2 else None
      d=rc[0] if rc else None;e=rc[1] if len(rc)>1 else None
      ent={int(x['car_no']):x for x in es}
      main_pair=f(m[0].get('score'))+f(m[1].get('score')); rival_pair=(f(rv[0].get('score'))+f(rv[1].get('score'))) if len(rv)>=2 else -999
      pair_top3=f(m[0].get('top3_rate'),0)+f(m[1].get('top3_rate'),0)
      pair_win=f(m[0].get('win_rate'),0)+f(m[1].get('win_rate'),0)
      second_top3=f(m[1].get('top3_rate'),0)
      pm={}
      for q in P[rid]:
       if q.get('ticket_type')=='3連複' and q.get('status')=='paid':
        try:pm[q['combination']]=int(float(q.get('payout_yen') or 0))
        except:pass
      abc=combo((a,b,c)) if c else None; od=O[rid].get(abc,{}) if abc else {}
      rows.append({'date':r.get('race_date',''),'year':int(r.get('race_date','')[:4]),'race_type':r.get('race_type',''),'es':es,'a':a,'b':b,'c':c,'d':d,'e':e,'main_size':len(mc),'gap':main_pair-rival_pair if rival_pair>-900 else 999,'pair_top3':pair_top3,'pair_win':pair_win,'second_top3':second_top3,'abc_rank':int(od['market_rank']) if od.get('market_rank','').isdigit() else 999,'abc_odds':f(od.get('odds'),999),'win_main':top3[0] in mc,'pay':pm})
    train=[r for r in rows if r['year']<=2024]; test=[r for r in rows if r['year']>=2025]

    def MAIN(r):return {combo((r['a'],r['b'],r['c']))} if r['c'] else set()
    def O4(r):return form((valid([r['d'],r['e']]),valid([r['a'],r['b'],r['d'],r['e']]),valid([r['a'],r['b'],r['d'],r['e']])))
    def OPAIR(r):
      if not r['d'] or not r['e']:return set()
      x=topoutside(r['es'],{r['d'],r['e']},4)
      return {combo((r['d'],r['e'],q)) for q in x}
    def OCROSS(r):
      if not r['d'] or not r['e']:return set()
      x=topoutside(r['es'],set(valid([r['a'],r['b'],r['d'],r['e']])),2)
      return form((valid([r['d'],r['e']]),valid([r['a'],r['b']]),valid([r['a'],r['b'],r['d'],r['e']])+x))
    def O9(r):
      x=topoutside(r['es'],set(valid([r['a'],r['b'],r['d'],r['e']])),1)
      return form((valid([r['d'],r['e']]),valid([r['a'],r['b'],r['d'],r['e']]),valid([r['a'],r['b'],r['d'],r['e']])+x))
    forms={'O4_ABDE_box':O4,'O4_DE_pair_plus_top4':OPAIR,'O_cross_DE_AB_plus2':OCROSS,'O9_DE_ABDE_plus1':O9}
    rules={
      'all':lambda r:True,
      'main3':lambda r:r['c'] is not None,
      'abc_rank<=3':lambda r:r['c'] is not None and r['abc_rank']<=3,
      'abc_rank<=5':lambda r:r['c'] is not None and r['abc_rank']<=5,
      'abc_rank>5':lambda r:r['c'] is None or r['abc_rank']>5,
      'abc_odds<=15':lambda r:r['c'] is not None and r['abc_odds']<=15,
      'abc_odds>15':lambda r:r['c'] is None or r['abc_odds']>15,
      'weakB':lambda r:r['pair_top3']<=106.1 and r['second_top3']<=35.7,
      'weakB_or_main2':lambda r:r['main_size']==2 or (r['pair_top3']<=106.1 and r['second_top3']<=35.7),
      'weakB_and_gap<=5':lambda r:r['pair_top3']<=106.1 and r['second_top3']<=35.7 and r['gap']<=5,
      'abc_rank>5_and_gap<=5':lambda r:(r['c'] is None or r['abc_rank']>5) and r['gap']<=5,
      'abc_rank>5_and_weakB':lambda r:(r['c'] is None or r['abc_rank']>5) and r['pair_top3']<=106.1 and r['second_top3']<=35.7,
      'strongA':lambda r:r['c'] is not None and r['pair_top3']>106.1 and r['pair_win']>54.7,
      'strongA_rank<=5':lambda r:r['c'] is not None and r['pair_top3']>106.1 and r['pair_win']>54.7 and r['abc_rank']<=5,
    }
    out={'n':len(rows),'train_n':len(train),'test_n':len(test),'main':{},'off':{}}
    for rn,pred in rules.items():
      out['main'][rn]={'train':metrics(train,MAIN,pred),'test':metrics(test,MAIN,pred)}
      out['off'][rn]={fn:{'train':metrics(train,sel,pred),'test':metrics(test,sel,pred)} for fn,sel in forms.items()}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'trio_routes.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=['# G3三日目 3連複 二系統ルーティング','',f"対象 {len(rows)}R / train 2022-24={len(train)}R / test 2025-26={len(test)}R",'', '## 本命ABC 1点']
    for rn,v in out['main'].items():
      t=v['train'];q=v['test'];md.append(f"- {rn}: train {t['races']}R ROI {t['roi']:.1%} hit {t['hit_rate']:.1%} / test {q['races']}R ROI {q['roi']:.1%} hit {q['hit_rate']:.1%}")
    md += ['', '## 外型']
    for rn,formsx in out['off'].items():
      for fn,v in formsx.items():
       t=v['train'];q=v['test'];
       if t['races']>=80 and q['races']>=40:
        md.append(f"- {rn} × {fn}: train {t['races']}R {t['avg_points']:.1f}点 ROI {t['roi']:.1%} hit {t['hit_rate']:.1%} off率 {t['off_rate']:.1%} / test {q['races']}R {q['avg_points']:.1f}点 ROI {q['roi']:.1%} hit {q['hit_rate']:.1%} off率 {q['off_rate']:.1%}")
    (OUT/'trio_routes.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
if __name__=='__main__':main()
