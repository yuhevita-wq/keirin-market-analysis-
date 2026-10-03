from __future__ import annotations
import csv, io, json, math, statistics, zipfile
from collections import defaultdict
from itertools import combinations, product
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
    xs=[x for x in z.namelist() if Path(x).name==n]
    return list(csv.DictReader(io.StringIO(dec(z.read(xs[0]))))) if xs else []
def f(v,d=float('-inf')):
    try:
        x=float(v or ''); return x if math.isfinite(x) else d
    except:return d
def day3(r): return len(r)==16 and r.isdigit() and r[10:12]=='03'
def tk(*xs): return '='.join(map(str,sorted(xs)))
def lines(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit(): by[int(e['line_id'])].append(e)
    ranked=[]; singles=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda e:int(e['line_position']))
        if len(ms)>=2:
            ps=f(ms[0].get('score'))+f(ms[1].get('score'))
            ranked.append(((ps,f(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid),lid,ms))
        elif len(ms)==1: singles.append(ms[0])
    ranked.sort(reverse=True,key=lambda x:x[0]); singles.sort(key=lambda e:(-f(e.get('score')),int(e.get('car_no') or 99)))
    return ranked,singles

def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0

def add(rows, *cars):
    cars=[c for c in cars if c is not None]
    if len(cars)==3 and len(set(cars))==3: rows.add(tk(*cars))
def row_form(r1,r2,r3):
    out=set()
    for a,b,c in product([x for x in r1 if x],[x for x in r2 if x],[x for x in r3 if x]): add(out,a,b,c)
    return out

def metrics(rs):
    n=len(rs); pts=sum(len(r['tickets']) for r in rs); hits=sum(r['hit'] for r in rs); stake=pts*100; ret=sum(r['ret'] for r in rs)
    mp=sum(r['market_p'] for r in rs)/n if n else 0
    hr=hits/n if n else 0
    def rate(pred):
        sub=[r for r in rs if pred(r)]
        return (sum(r['hit'] for r in sub)/len(sub) if sub else 0, len(sub))
    abbroken=rate(lambda r:not r['AB_pair']); neither=rate(lambda r:not r['AB_pair'] and not r['CD_pair']); offmain=rate(lambda r:r['offmain_winner'])
    deep_hits=sum(r['hit'] and (not r['AB_pair']) and (not r['CD_pair']) for r in rs)
    return {'races':n,'avg_points':pts/n if n else 0,'hit_rate':hr,'market_rate':mp,'edge_ratio':hr/mp if mp else None,
            'diff':hr-mp,'roi':ret/stake if stake else 0,'ab_broken_hit_rate':abbroken[0],'ab_broken_races':abbroken[1],
            'neither_pair_hit_rate':neither[0],'neither_pair_races':neither[1],'offmain_winner_hit_rate':offmain[0],'offmain_winner_races':offmain[1],
            'deep_hit_share':deep_hits/hits if hits else 0}

def main():
    races={}; E=defaultdict(list); R=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'): races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'): E[x['race_id']].append(x)
            for x in read(z,'results.csv'): R[x['race_id']].append(x)
            for x in read(z,'payouts.csv'): P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'): O[x['race_id']][x.get('combination','')]=x
    rows=defaultdict(list)
    for race in races.values():
        rid=race['race_id']
        if not day3(rid): continue
        ranked,singles=lines(E[rid])
        if len(ranked)<2: continue
        m=ranked[0][2]; q=ranked[1][2]
        A,B=int(m[0]['car_no']),int(m[1]['car_no']); T=int(m[2]['car_no']) if len(m)>=3 else None
        C,D=int(q[0]['car_no']),int(q[1]['car_no']); U=int(q[2]['car_no']) if len(q)>=3 else None
        weak=ranked[-1][2] if len(ranked)>=3 else []
        WT=int(weak[2]['car_no']) if len(weak)>=3 else None
        S=int(singles[0]['car_no']) if singles else None
        win,pay=paid(P[rid])
        if not win: continue
        # actual top3 for robustness strata
        top=[]
        for rr in R[rid]:
            try: pos=int(rr.get('finish_position') or 0); car=int(rr.get('car_no') or 0)
            except: continue
            if pos in (1,2,3): top.append((pos,car))
        if len(top)!=3: continue
        top3=[c for _,c in sorted(top)]
        AB_pair=A in top3 and B in top3; CD_pair=C in top3 and D in top3; offmain=top3[0] not in [int(x['car_no']) for x in m]
        # normalized market probabilities
        inv={}
        for combo,o in O[rid].items():
            try: od=float(o.get('odds') or 0)
            except: continue
            if od>0 and math.isfinite(od): inv[combo]=1/od
        z=sum(inv.values())
        if z<=0: continue
        mp={k:v/z for k,v in inv.items()}
        core=set(tk(*x) for x in combinations((A,B,C,D),3))
        line=set(core); add(line,A,B,T); add(line,C,D,U)
        cross_tails=row_form([A,B],[C,D],[A,B,C,D,T,U])
        cross_value=row_form([A,B],[C,D],[A,B,C,D,T,U,WT])
        cross_single=row_form([A,B],[C,D],[A,B,C,D,T,U,S])
        insurance=set(line)
        for x in (WT,):
            for a in (A,B):
                for b in (C,D): add(insurance,a,b,x)
        insurance_single=set(insurance)
        for a in (A,B):
            for b in (C,D): add(insurance_single,a,b,S)
        follower=row_form([B,D],[A,B,C,D],[A,B,C,D,T,U,WT,S])
        leader=row_form([A,C],[A,B,C,D],[A,B,C,D,T,U,WT,S])
        families={'CORE4':core,'CORE4_LINE':line,'CROSS_TAILS':cross_tails,'CROSS_VALUE':cross_value,
                  'CROSS_SINGLE':cross_single,'INSURANCE_TAIL':insurance,'INSURANCE_TAIL_SINGLE':insurance_single,
                  'FOLLOWER_HUB':follower,'LEADER_HUB':leader}
        split='train' if int(race['race_date'][:4])<=2024 else 'test'
        for name,tickets in families.items():
            tickets={t for t in tickets if t in mp}
            if not tickets or len(tickets)>16: continue
            hit=int(win in tickets)
            rows[name].append({'split':split,'tickets':tickets,'hit':hit,'ret':pay if hit else 0,'market_p':sum(mp[t] for t in tickets),
                               'AB_pair':AB_pair,'CD_pair':CD_pair,'offmain_winner':offmain})
    out={}; md=['# G3三日目 ロバスト3連複フォーメーション','', '目的: 想定外展開を拾いつつ、市場期待確率より実現率が上回る形を探す。train=2022-2024 / test=2025-2026H1','']
    for name in rows:
        out[name]={}; md.append(f'## {name}')
        for sp in ('train','test'):
            s=metrics([r for r in rows[name] if r['split']==sp]); out[name][sp]=s
            md.append(f"- {sp}: {s['races']}R 平均{s['avg_points']:.2f}点 hit {s['hit_rate']:.1%} / market {s['market_rate']:.1%} / ratio {s['edge_ratio']:.3f} / diff {s['diff']:+.2%} / ROI {s['roi']:.1%}")
            md.append(f"  - AB崩れ時 hit {s['ab_broken_hit_rate']:.1%} ({s['ab_broken_races']}R) / AB・CD両ペア不成立時 hit {s['neither_pair_hit_rate']:.1%} ({s['neither_pair_races']}R) / 主力外勝者時 hit {s['offmain_winner_hit_rate']:.1%} / 的中のうち深いズレ {s['deep_hit_share']:.1%}")
        md.append('')
    (OUT/'robust_formations.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'robust_formations.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__': main()
