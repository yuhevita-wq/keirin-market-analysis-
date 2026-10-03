from __future__ import annotations
import csv, io, json, math, statistics, zipfile
from collections import defaultdict
from itertools import combinations
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
def day3(rid): return len(rid)==16 and rid.isdigit() and rid[10:12]=='03'
def tk3(xs): return '='.join(map(str,sorted(xs)))
def lines(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit(): by[int(e['line_id'])].append(e)
    ranked=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda x:int(x['line_position']))
        if len(ms)>=2:
            strength=f(ms[0].get('score'))+f(ms[1].get('score'))
            ranked.append((strength,f(ms[0].get('score')),lid,ms))
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2]))
    return by,ranked
def market_probs(od):
    inv={}
    for c,row in od.items():
        try:o=float(row.get('odds') or 0)
        except:o=0
        if o>0: inv[c]=1/o
    s=sum(inv.values()); return {c:v/s for c,v in inv.items()} if s else {}
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0
def uniq(xs):
    out=[]; seen=set()
    for x in xs:
        if x is None or x in seen: continue
        seen.add(x); out.append(x)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    races={}; E=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'): races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'): E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'): P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'): O[x['race_id']][x.get('combination','')]=x

    stats=defaultdict(lambda: defaultdict(lambda:{'races':0,'points':0,'hits':0,'market':0.0,'ret':0,'odds':[]}))
    def add(name,split,combos,win,pay,mp,od):
        combos={c for c in combos if c in mp}
        if not combos:return
        s=stats[name][split]; s['races']+=1; s['points']+=len(combos)
        h=win in combos; s['hits']+=int(h); s['market']+=sum(mp[c] for c in combos); s['ret']+=pay if h else 0
        if h:
            try:s['odds'].append(float(od[win]['odds']))
            except:pass

    for race in races.values():
        rid=race['race_id']
        if not day3(rid): continue
        win,pay=paid(P[rid]); od=O[rid]
        if not win or not od: continue
        es=E[rid]; by,ranked=lines(es)
        if len(ranked)<2: continue
        mp=market_probs(od)
        if not mp: continue
        split='train' if int(race['race_date'][:4])<=2024 else 'test'
        top1,top2=ranked[0][3],ranked[1][3]
        A,B=int(top1[0]['car_no']),int(top1[1]['car_no'])
        C,D=int(top2[0]['car_no']),int(top2[1]['car_no'])
        core=[A,B,C,D]
        pairsets={
            'AB':[(A,B)], 'CD':[(C,D)], 'BD':[(B,D)],
            'SAME':[(A,B),(C,D)],
            'CROSS':[(A,C),(A,D),(B,C),(B,D)],
            'ALL2':list(combinations(core,2)),
        }
        # line-rank residual role groups
        third=ranked[2][3] if len(ranked)>=3 else []
        weak=ranked[-1][3] if len(ranked)>=3 else []
        lower=ranked[2:] if len(ranked)>=3 else []
        # true singletons = line_id missing or line with <2 members
        single=[]
        for e in es:
            car=int(e['car_no'])
            lid=int(e['line_id']) if e.get('line_id','').isdigit() else None
            if lid is None or len(by.get(lid,[]))<2: single.append(car)
        scored=sorted(es,key=lambda e:(-f(e.get('score')),int(e.get('car_no','99'))))
        sr={int(e['car_no']):i for i,e in enumerate(scored)}
        best_single=min(single,key=lambda c:(sr.get(c,99),c)) if single else None
        groups={
          'R3_LEADER':[int(third[0]['car_no'])] if third else [],
          'R3_SECOND':[int(third[1]['car_no'])] if len(third)>=2 else [],
          'R3_TAIL':[int(x['car_no']) for x in third[2:]],
          'WEAK_LEADER':[int(weak[0]['car_no'])] if weak else [],
          'WEAK_SECOND':[int(weak[1]['car_no'])] if len(weak)>=2 else [],
          'WEAK_TAIL':[int(x['car_no']) for x in weak[2:]],
          'LOWER_LEADERS':[int(ms[0]['car_no']) for _,_,_,ms in lower if ms],
          'LOWER_SECONDS':[int(ms[1]['car_no']) for _,_,_,ms in lower if len(ms)>=2],
          'LOWER_TAILS':[int(x['car_no']) for _,_,_,ms in lower for x in ms[2:]],
          'LOWER_ALL':[int(x['car_no']) for _,_,_,ms in lower for x in ms],
          'SINGLE_BEST':[best_single] if best_single is not None else [],
          'SINGLE_ALL':single,
        }
        for pname,pairs in pairsets.items():
            for gname,cars in groups.items():
                cars=uniq(cars)
                combos=[]
                for p in pairs:
                    for x in cars:
                        if x in p: continue
                        combos.append(tk3((p[0],p[1],x)))
                add(f'{pname}+{gname}',split,combos,win,pay,mp,od)
        # combined robust outsider slots using only residual roles, not all lower members
        for name,cars in {
            'RESID_ROLE_ALL':uniq(groups['LOWER_TAILS']+groups['SINGLE_ALL']),
            'RESID_BACK_ONLY':uniq(groups['R3_TAIL']+groups['WEAK_TAIL']+groups['SINGLE_ALL']),
            'RESID_R3_SINGLE':uniq(groups['R3_LEADER']+groups['R3_SECOND']+groups['R3_TAIL']+groups['SINGLE_ALL']),
        }.items():
            for pname,pairs in pairsets.items():
                combos=[]
                for p in pairs:
                    for x in cars:
                        if x not in p: combos.append(tk3((p[0],p[1],x)))
                add(f'{pname}+{name}',split,combos,win,pay,mp,od)

    rows=[]; out={}
    for name in sorted(stats):
        out[name]={}
        parts=[]
        for split in ('train','test'):
            s=stats[name][split]; n=s['races']; pts=s['points']; hits=s['hits']
            actual=hits/n if n else 0; market=s['market']/n if n else 0
            ratio=actual/market if market else 0; roi=s['ret']/(pts*100) if pts else 0
            avg=pts/n if n else 0; med=statistics.median(s['odds']) if s['odds'] else None
            rec={'races':n,'avg_points':avg,'hit_rate':actual,'market_rate':market,'ratio':ratio,'roi':roi,'median_hit_odds':med}
            out[name][split]=rec
            parts.append(f"{split} n={n} pts={avg:.2f} hit={actual:.2%} market={market:.2%} ratio={ratio:.3f} ROI={roi:.1%}"+(f" medhit={med:.1f}" if med is not None else ''))
        # shortlist only sufficiently sampled shapes for readable report
        if out[name]['train']['races']>=150 and out[name]['test']['races']>=80:
            rows.append((name,parts,out[name]))
    # sort by conservative market divergence (min ratio) then test ROI
    rows.sort(key=lambda x:(min(x[2]['train']['ratio'],x[2]['test']['ratio']), min(x[2]['train']['roi'],x[2]['test']['roi'])), reverse=True)
    md=['# G3三日目 上位2ライン×単騎・下位ライン 市場乖離','',
        'A-B=最上位ライン、C-D=2番手ライン。下位ラインは強度3位以下。市場期待は3連複1/oddsのレース内正規化。','',
        '## 市場乖離が両期間で残る上位候補']
    for name,parts,rec in rows[:35]:
        md.append(f"- **{name}**: "+' / '.join(parts))
    md += ['', '## 全結果はJSON参照']
    (OUT/'core_residual_matrix.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'core_residual_matrix.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
