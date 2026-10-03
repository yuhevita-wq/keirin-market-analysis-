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
        except UnicodeDecodeError: pass
    return b.decode('utf-8',errors='replace')
def read(z,n):
    xs=[x for x in z.namelist() if Path(x).name==n]
    return list(csv.DictReader(io.StringIO(dec(z.read(xs[0]))))) if xs else []
def f(v,d=float('-inf')):
    try:
        x=float(v or ''); return x if math.isfinite(x) else d
    except:return d
def rank_lines(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit(): by[int(e['line_id'])].append(e)
    out=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda e:int(e['line_position']))
        if len(ms)<2: continue
        pair=f(ms[0].get('score'))+f(ms[1].get('score'))
        key=(pair,f(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid)
        out.append((key,lid,ms))
    out.sort(reverse=True,key=lambda x:x[0]); return out
def day3(r): return len(r)==16 and r.isdigit() and r[10:12]=='03'
def tk(a,b,c): return '='.join(map(str,sorted((a,b,c))))
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0
def uniq(xs):
    o=[]; s=set()
    for x in xs:
        if x is None or x in s: continue
        s.add(x); o.append(x)
    return o
def result(rows):
    n=len(rows); st=sum(r['points']*100 for r in rows); ret=sum(r['ret'] for r in rows); h=sum(r['hit'] for r in rows)
    return {'races':n,'avg_points':sum(r['points'] for r in rows)/n if n else 0,'hit_rate':h/n if n else 0,'roi':ret/st if st else 0,'stake':st,'return':ret}

def main():
    races={}; E=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'): races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'): E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'): P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'): O[x['race_id']][x.get('combination','')]=x
    defs={}
    rows_by=defaultdict(list)
    for race in races.values():
        rid=race['race_id']
        if not day3(rid): continue
        ls=rank_lines(E[rid])
        if len(ls)<2: continue
        main=ls[0][2]; rival=ls[1][2]; third=ls[2][2] if len(ls)>=3 else []
        A,B=int(main[0]['car_no']),int(main[1]['car_no'])
        M3=int(main[2]['car_no']) if len(main)>=3 else None
        C,D=int(rival[0]['car_no']),int(rival[1]['car_no'])
        R3=int(rival[2]['car_no']) if len(rival)>=3 else None
        E1=int(third[0]['car_no']) if len(third)>=1 else None
        E2=int(third[1]['car_no']) if len(third)>=2 else None
        win,pay=paid(P[rid])
        if not win: continue
        split='train' if int(race['race_date'][:4])<=2024 else 'test'
        # Structural candidates only, no result information.
        strategies={
          'AB2_M3D': ((A,B), uniq([M3,D,C])[:2]),
          'AB2_M3C': ((A,B), uniq([M3,C,D])[:2]),
          'AB3_M3CD': ((A,B), uniq([M3,C,D])[:3]),
          'CD2_AR3': ((C,D), uniq([A,R3,B,E1])[:2]),
          'CD2_BR3': ((C,D), uniq([B,R3,A,E1])[:2]),
          'CD2_AB': ((C,D), uniq([A,B,R3,E1])[:2]),
          'CD3_ABR3': ((C,D), uniq([A,B,R3,E1,E2])[:3]),
          'CD4_ABR3_E1': ((C,D), uniq([A,B,R3,E1,E2])[:4]),
          'CD5_ABR3_E12': ((C,D), uniq([A,B,R3,E1,E2])[:5]),
        }
        odds=O[rid]
        for name,(pair,cands) in strategies.items():
            combos=[tk(pair[0],pair[1],x) for x in cands if x not in pair]
            combos=list(dict.fromkeys(combos))
            hit=int(win in combos); ret=pay if hit else 0
            rows_by[name].append({'split':split,'points':len(combos),'hit':hit,'ret':ret})
            # Value filters applied per ticket, but only structural candidates.
            for floor in (5,7,10,12,15,20):
                kept=[]
                for combo in combos:
                    try: od=float(odds.get(combo,{}).get('odds') or '')
                    except: continue
                    if od>=floor: kept.append(combo)
                if kept:
                    h=int(win in kept); rows_by[f'{name}_floor{floor}'].append({'split':split,'points':len(kept),'hit':h,'ret':pay if h else 0})
    md=['# G3三日目 構造固定ペア 最終検証','', 'train=2022-2024 / test=2025-2026H1','']
    order=['AB2_M3D','AB2_M3C','AB3_M3CD','CD2_AR3','CD2_BR3','CD2_AB','CD3_ABR3','CD4_ABR3_E1','CD5_ABR3_E12']
    out={}
    for name in order:
        out[name]={}; md.append(f'## {name}')
        for split in ('train','test'):
            s=result([r for r in rows_by[name] if r['split']==split]); out[name][split]=s
            md.append(f"- base {split}: {s['races']}R {s['avg_points']:.1f}点 hit {s['hit_rate']:.1%} ROI {s['roi']:.1%}")
        for floor in (5,7,10,12,15,20):
            vals=[]
            out[name][f'floor{floor}']={}
            for split in ('train','test'):
                s=result([r for r in rows_by[f'{name}_floor{floor}'] if r['split']==split]); out[name][f'floor{floor}'][split]=s
                vals.append(f"{split} {s['races']}R {s['avg_points']:.1f}点 hit {s['hit_rate']:.1%} ROI {s['roi']:.1%}")
            md.append(f"- odds>={floor}: "+' / '.join(vals))
        md.append('')
    # Combined structural plans, same floor to all included tickets.
    combo_defs={'PLAN5_AB2_CD3':['AB2_M3D','CD3_ABR3'], 'PLAN6_AB3_CD3':['AB3_M3CD','CD3_ABR3']}
    md.append('## 組み合わせ')
    for plan,names in combo_defs.items():
        out[plan]={}
        for floor in (0,5,7,10,12,15,20):
            vals=[]; out[plan][floor]={}
            suffix='' if floor==0 else f'_floor{floor}'
            for split in ('train','test'):
                # rows align by eligible race except floor variants can omit. Recompute aggregate by summing strategy stats is valid for stake/return; hit overlap is not, so report ROI/avg points only here.
                source=[]
                for n in names:
                    key=n+suffix
                    source += [r for r in rows_by[key] if r['split']==split]
                st=sum(r['points']*100 for r in source); ret=sum(r['ret'] for r in source); pts=sum(r['points'] for r in source)
                # denominator use 1463/793 baseline race counts for comparable avg points.
                denom=1463 if split=='train' else 793
                s={'avg_points':pts/denom if denom else 0,'roi':ret/st if st else 0,'stake':st,'return':ret}
                out[plan][floor][split]=s; vals.append(f"{split} {s['avg_points']:.1f}点 ROI {s['roi']:.1%}")
            md.append(f"- {plan} floor>={floor}: "+' / '.join(vals))
    (OUT/'logical_pair_strategy.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'logical_pair_strategy.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
