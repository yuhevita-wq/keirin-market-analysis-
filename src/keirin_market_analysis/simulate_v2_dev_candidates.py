from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_branching_v1 import classify, mainline_bets

ROOT=Path('data')
OUT=ROOT/'audits'/'v2_dev_candidate_financials.json'
YEARS=(2024,2025)


def read_csv(path:Path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def m(e,k):
    v=num(e.get(k,'')); return 0.0 if v==float('-inf') else float(v)

def line_map(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():by[int(e['line_id'])].append(e)
    for lid in by:by[lid].sort(key=lambda e:int(e['line_position']))
    return by

def rival(es,mid):
    cs=[]
    for lid,x in line_map(es).items():
        if lid==mid or len(x)<2:continue
        cs.append(((m(x[0],'score')+m(x[1],'score'),m(x[0],'score'),-lid),x))
    return max(cs,key=lambda x:x[0])[1] if cs else None

def segmap(races):
    g=defaultdict(list); out={}
    for r in races:g[(r['race_date'],r['track'])].append(r)
    for xs in g.values():
        xs.sort(key=lambda r:int(r['race_no']))
        for i,r in enumerate(xs,1):out[r['race_id']]=segment_for(i,len(xs))
    return out

def payout_map(rows):
    out=defaultdict(dict)
    for p in rows:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            try:out[p['race_id']][p['combination']]=int(float(p['payout_yen']))
            except:pass
    return out

def dedupe(xs):return sorted(set(xs))

def mid_core(a,b,m3,r1l,r1b):
    return dedupe([f'{x}-{y}-{z}' for x,y in ((a,b),(b,a)) for z in (m3,r1l,r1b) if z not in {x,y}])

def mid_bets(name,a,b,m3,r1l,r1b):
    xs=mid_core(a,b,m3,r1l,r1b)
    if name in {'M7','M8'}:xs.append(f'{b}-{m3}-{a}')
    if name=='M8':xs.append(f'{a}-{m3}-{b}')
    return dedupe(xs)

def late_bets(name,es,main,r1):
    a=int(main[0]['car_no']); b=int(main[1]['car_no']); m3=int(main[2]['car_no'])
    r1l=int(r1[0]['car_no']); r1b=int(r1[1]['car_no'])
    if name=='L4_current':return mainline_bets(es,main)
    thirds=(m3,r1l,r1b) if name=='L6_core' else (r1l,r1b)
    return dedupe([f'{x}-{y}-{z}' for x,y in ((a,b),(b,a)) for z in thirds if z not in {x,y}])

def stats(rs):
    n=len(rs); stake=sum(r['stake'] for r in rs); pay=sum(r['payout'] for r in rs); hits=sum(r['hit'] for r in rs)
    return {'races':n,'hits':hits,'hit_rate':hits/n if n else 0,'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi':pay/stake if stake else 0,'avg_points':sum(r['points'] for r in rs)/n if n else 0}

def run_year(year):
    d=ROOT/str(year)/'s_class_yosen'; races=read_csv(d/'races.csv'); entries=read_csv(d/'entries.csv'); payouts=read_csv(d/'payouts.csv')
    eb=defaultdict(list)
    for e in entries:eb[e['race_id']].append(e)
    seg=segmap(races); pm=payout_map(payouts)
    out=[]
    for race in sorted(races,key=lambda r:(r['race_date'],r['track'],int(r['race_no']))):
        rid=race['race_id']; chosen=choose_main_line(eb[rid])
        if not chosen:continue
        mid,main=chosen
        if len(main)<3:continue
        r1=rival(eb[rid],mid)
        if not r1 or len(r1)<2:continue
        a=int(main[0]['car_no']); b=int(main[1]['car_no']); m3=int(main[2]['car_no']); r1l=int(r1[0]['car_no']); r1b=int(r1[1]['car_no'])
        if seg[rid]=='中盤':
            pair_top2=m(main[0],'top2_rate')+m(main[1],'top2_rate'); pair_top3=m(main[0],'top3_rate')+m(main[1],'top3_rate')
            filters={
              'ALL':True,
              'PAIR_TOP2_GT55':pair_top2>55,
              'PAIR_TOP2_GT70':pair_top2>70,
              'PAIR_TOP3_GT110':pair_top3>110,
              'PAIR_WIN_GT40':m(main[0],'win_rate')+m(main[1],'win_rate')>40,
            }
            for filt,ok in filters.items():
                if not ok:continue
                for form in ('M6','M7','M8'):
                    bets=mid_bets(form,a,b,m3,r1l,r1b); ret=sum(pm[rid].get(x,0) for x in bets)
                    out.append({'year':year,'part':'middle','candidate':f'{filt}__{form}','stake':100*len(bets),'payout':ret,'hit':int(ret>0),'points':len(bets)})
        elif seg[rid]=='後半' and classify(main)[0]=='A_mainline':
            for form in ('L4_current','L6_core','L4_rivals'):
                bets=late_bets(form,eb[rid],main,r1); ret=sum(pm[rid].get(x,0) for x in bets)
                out.append({'year':year,'part':'late_mainline','candidate':form,'stake':100*len(bets),'payout':ret,'hit':int(ret>0),'points':len(bets)})
    return out

def main():
    records={y:run_year(y) for y in YEARS}
    out={'scope':'2024+2025 development financial comparison of a small candidate set frozen after structure-only audit. 2023 remains untouched OOS.','selection_rule':'Prefer candidates with the highest worst-year ROI; require >=20 races in each year; break near-ties toward fewer points. Do not add candidates after seeing this output.','middle':{},'late_mainline':{}}
    for part in ('middle','late_mainline'):
        names=sorted({r['candidate'] for y in YEARS for r in records[y] if r['part']==part})
        rows=[]
        for name in names:
            ys={str(y):stats([r for r in records[y] if r['part']==part and r['candidate']==name]) for y in YEARS}
            combined=stats([r for y in YEARS for r in records[y] if r['part']==part and r['candidate']==name])
            worst=min(ys['2024']['roi'],ys['2025']['roi'])
            rows.append({'candidate':name,'2024':ys['2024'],'2025':ys['2025'],'combined':combined,'worst_year_roi':worst})
        rows.sort(key=lambda x:(x['2024']['races']>=20 and x['2025']['races']>=20,float(x['worst_year_roi']),float(x['combined']['roi']),-float(x['combined']['avg_points'])),reverse=True)
        out[part]={'ranked':rows}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
