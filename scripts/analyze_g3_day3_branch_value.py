from __future__ import annotations
import csv,io,json,math,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ARCH=ROOT/'data'/'grade_races'/'g3'; OUT=ROOT/'results'/'g3_day3_reality'
def dec(b):
    for e in ('utf-8-sig','utf-8','cp932','shift_jis'):
        try:return b.decode(e)
        except UnicodeDecodeError:pass
    return b.decode('utf-8',errors='replace')
def read(z,n):
    xs=[x for x in z.namelist() if Path(x).name==n]; return list(csv.DictReader(io.StringIO(dec(z.read(xs[0]))))) if xs else []
def f(v,d=float('-inf')):
    try:x=float(v or ''); return x if math.isfinite(x) else d
    except:return d
def day3(r):return len(r)==16 and r.isdigit() and r[10:12]=='03'
def tk(*xs):return '='.join(map(str,sorted(xs)))
def rank(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():by[int(e['line_id'])].append(e)
    ls=[]; ss=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda x:int(x['line_position']))
        if len(ms)>=2:
            ps=f(ms[0].get('score'))+f(ms[1].get('score'));ls.append(((ps,f(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid),ms))
        elif len(ms)==1:ss.append(ms[0])
    ls.sort(reverse=True,key=lambda x:x[0]);ss.sort(key=lambda x:(-f(x.get('score')),int(x.get('car_no') or 99)));return ls,ss
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0
def main():
    races={};E=defaultdict(list);P=defaultdict(list);O=defaultdict(dict)
    for p in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x
    acc=defaultdict(lambda:defaultdict(lambda:{'n':0,'hit':0,'ret':0,'mp':0.0,'stake':0}))
    templates=['ABC','ABD','ACD','BCD','ABT','CDU','ACT','ADT','BCT','BDT','ACU','ADU','BCU','BDU','ACW','ADW','BCW','BDW','ACS','ADS','BCS','BDS']
    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        ls,ss=rank(E[rid])
        if len(ls)<2:continue
        m,q=ls[0][1],ls[1][1]; weak=ls[-1][1] if len(ls)>=3 else []
        A,B=int(m[0]['car_no']),int(m[1]['car_no']);T=int(m[2]['car_no']) if len(m)>=3 else None
        C,D=int(q[0]['car_no']),int(q[1]['car_no']);U=int(q[2]['car_no']) if len(q)>=3 else None
        W=int(weak[2]['car_no']) if len(weak)>=3 else None;S=int(ss[0]['car_no']) if ss else None
        roles={'A':A,'B':B,'C':C,'D':D,'T':T,'U':U,'W':W,'S':S}
        win,pay=paid(P[rid]);
        if not win:continue
        inv={}
        for co,o in O[rid].items():
            try:od=float(o.get('odds') or 0)
            except:continue
            if od>0:inv[co]=1/od
        z=sum(inv.values())
        if z<=0:continue
        sp='train' if int(race['race_date'][:4])<=2024 else 'test'
        for t in templates:
            cars=[roles[x] for x in t]
            if None in cars or len(set(cars))<3:continue
            co=tk(*cars)
            if co not in inv:continue
            a=acc[t][sp];a['n']+=1;a['stake']+=100;a['mp']+=inv[co]/z
            if co==win:a['hit']+=1;a['ret']+=pay
    md=['# G3三日目 フォーメーション枝別価値','', '市場期待は3連複全組合せの1/oddsを正規化。',''];out={}
    for t in templates:
        out[t]={};md.append(f'## {t}')
        for sp in ('train','test'):
            a=acc[t][sp];n=a['n'];hr=a['hit']/n if n else 0;mp=a['mp']/n if n else 0;roi=a['ret']/a['stake'] if a['stake'] else 0
            s={'n':n,'hit_rate':hr,'market_rate':mp,'edge_ratio':hr/mp if mp else None,'diff':hr-mp,'roi':roi};out[t][sp]=s
            md.append(f"- {sp}: n={n} hit {hr:.2%} market {mp:.2%} ratio {(hr/mp if mp else 0):.3f} diff {hr-mp:+.2%} ROI {roi:.1%}")
        md.append('')
    (OUT/'branch_value.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(OUT/'branch_value.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('\n'.join(md))
if __name__=='__main__':main()
