from __future__ import annotations
import csv,io,json,math,zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ARCH=ROOT/'data'/'grade_races'/'g3';OUT=ROOT/'results'/'g3_day3_reality'
def dec(b):
    for e in ('utf-8-sig','utf-8','cp932','shift_jis'):
        try:return b.decode(e)
        except UnicodeDecodeError:pass
    return b.decode('utf-8',errors='replace')
def read(z,n):
    xs=[x for x in z.namelist() if Path(x).name==n];return list(csv.DictReader(io.StringIO(dec(z.read(xs[0]))))) if xs else []
def f(v,d=float('-inf')):
    try:x=float(v or '');return x if math.isfinite(x) else d
    except:return d
def day3(r):return len(r)==16 and r.isdigit() and r[10:12]=='03'
def tk(*xs):return '='.join(map(str,sorted(xs)))
def rank(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():by[int(e['line_id'])].append(e)
    ls=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda x:int(x['line_position']))
        if len(ms)>=2:
            ps=f(ms[0].get('score'))+f(ms[1].get('score'));ls.append(((ps,f(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid),ms))
    ls.sort(reverse=True,key=lambda x:x[0]);return ls
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0
def add(s,*xs):
    if None not in xs and len(set(xs))==3:s.add(tk(*xs))
def metrics(rs):
    n=len(rs);pts=sum(len(r['tickets']) for r in rs);hit=sum(r['hit'] for r in rs);st=pts*100;ret=sum(r['ret'] for r in rs);mp=sum(r['mp'] for r in rs)/n if n else 0;hr=hit/n if n else 0
    def rr(pred):
        q=[r for r in rs if pred(r)];return (sum(r['hit'] for r in q)/len(q) if q else 0,len(q))
    ab=rr(lambda r:not r['AB']);ne=rr(lambda r:not r['AB'] and not r['CD']);off=rr(lambda r:r['off'])
    return {'races':n,'avg_points':pts/n if n else 0,'hit_rate':hr,'market_rate':mp,'edge_ratio':hr/mp if mp else None,'diff':hr-mp,'roi':ret/st if st else 0,'ab_broken_hit_rate':ab[0],'neither_hit_rate':ne[0],'offmain_hit_rate':off[0],'neither_n':ne[1]}
def main():
    races={};E=defaultdict(list);R=defaultdict(list);P=defaultdict(list);O=defaultdict(dict)
    for p in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'results.csv'):R[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x
    rows=defaultdict(list)
    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        ls=rank(E[rid])
        if len(ls)<2:continue
        m,q=ls[0][1],ls[1][1];A,B=int(m[0]['car_no']),int(m[1]['car_no']);T=int(m[2]['car_no']) if len(m)>=3 else None;C,D=int(q[0]['car_no']),int(q[1]['car_no']);U=int(q[2]['car_no']) if len(q)>=3 else None
        win,pay=paid(P[rid]);
        if not win:continue
        top=[]
        for x in R[rid]:
            try:p=int(x.get('finish_position') or 0);c=int(x.get('car_no') or 0)
            except:continue
            if p in (1,2,3):top.append((p,c))
        if len(top)!=3:continue
        t3=[c for _,c in sorted(top)];AB=A in t3 and B in t3;CD=C in t3 and D in t3;off=t3[0] not in [int(x['car_no']) for x in m]
        inv={}
        for co,o in O[rid].items():
            try:od=float(o.get('odds') or 0)
            except:continue
            if od>0:inv[co]=1/od
        z=sum(inv.values())
        if z<=0:continue
        base=set();add(base,A,B,T);add(base,C,D,U);add(base,B,D,T)
        f4=set(base);add(f4,A,C,D)
        f4b=set(base);add(f4b,B,C,U)
        f5=set(f4);add(f5,B,C,U)
        f6=set(f5);add(f6,B,C,D)
        # fallback: only add ACD when fewer than 2 value branches exist
        adaptive=set(base)
        if len(adaptive)<2:add(adaptive,A,C,D)
        plans={'VALUE3':base,'VALUE4_ACD':f4,'VALUE4_BCU':f4b,'VALUE5':f5,'VALUE6':f6,'ADAPTIVE':adaptive}
        sp='train' if int(race['race_date'][:4])<=2024 else 'test'
        for name,tix in plans.items():
            tix={t for t in tix if t in inv}
            if not tix:continue
            h=int(win in tix)
            rows[name].append({'split':sp,'tickets':tix,'hit':h,'ret':pay if h else 0,'mp':sum(inv[t]/z for t in tix),'AB':AB,'CD':CD,'off':off})
    md=['# G3三日目 市場乖離ロバスト型 最終比較',''];out={}
    for name in rows:
        out[name]={};md.append(f'## {name}')
        for sp in ('train','test'):
            s=metrics([r for r in rows[name] if r['split']==sp]);out[name][sp]=s
            md.append(f"- {sp}: {s['races']}R 平均{s['avg_points']:.2f}点 hit {s['hit_rate']:.2%} market {s['market_rate']:.2%} ratio {s['edge_ratio']:.3f} diff {s['diff']:+.2%} ROI {s['roi']:.1%}")
            md.append(f"  - AB崩れ時 {s['ab_broken_hit_rate']:.2%} / AB・CD両ペア不成立時 {s['neither_hit_rate']:.2%} ({s['neither_n']}R) / 主力外勝者時 {s['offmain_hit_rate']:.2%}")
        md.append('')
    (OUT/'value_formation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(OUT/'value_formation.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('\n'.join(md))
if __name__=='__main__':main()
