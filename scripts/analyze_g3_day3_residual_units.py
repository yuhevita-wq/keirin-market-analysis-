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
        except UnicodeDecodeError:pass
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
        strength=(f(ms[0].get('score'))+f(ms[1].get('score'))) if len(ms)>=2 else f(ms[0].get('score'))
        if len(ms)>=2: ranked.append((strength,f(ms[0].get('score')),lid,ms))
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2]))
    return by,ranked
def market_probs(od):
    inv={}
    for c,row in od.items():
        try:o=float(row.get('odds') or 0)
        except:o=0
        if o>0:inv[c]=1/o
    s=sum(inv.values()); return {c:v/s for c,v in inv.items()} if s else {}
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    races={}; E=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x

    rows=[]
    branch=defaultdict(lambda: {'n':0,'hit':0,'market':0.0,'ret':0,'odds':[]})
    cat=defaultdict(lambda: {'n':0,'actual':0,'market':0.0,'win_odds':[]})

    for race in races.values():
        rid=race['race_id']
        if not day3(rid): continue
        es=E[rid]; win,pay=paid(P[rid]); od=O[rid]
        if not win or not od: continue
        by,ranked=lines(es)
        if len(ranked)<2: continue
        mp=market_probs(od)
        winset=set(map(int,win.split('=')))
        entry={int(e['car_no']):e for e in es if e.get('car_no','').isdigit()}
        line_rank={lid:i+1 for i,(_,_,lid,_) in enumerate(ranked)}
        top1=ranked[0][3]; top2=ranked[1][3]
        A,B=int(top1[0]['car_no']),int(top1[1]['car_no'])
        C,D=int(top2[0]['car_no']),int(top2[1]['car_no'])
        T=int(top1[2]['car_no']) if len(top1)>=3 else None
        U=int(top2[2]['car_no']) if len(top2)>=3 else None
        residual=[]; single=[]; lower=[]
        for e in es:
            car=int(e['car_no'])
            lid=int(e['line_id']) if e.get('line_id','').isdigit() else None
            pos=int(e['line_position']) if e.get('line_position','').isdigit() else None
            mem=by.get(lid,[]) if lid is not None else []
            if len(mem)<2:
                single.append(car); residual.append(car)
            elif line_rank.get(lid,99)>=3:
                lower.append(car); residual.append(car)
        # categories conditional on availability
        defs={
          'ANY_SINGLETON': set(single),
          'ANY_LOWER': set(lower),
          'ANY_RESIDUAL': set(residual),
          'SINGLETON_AND_LOWER': None,
          'LOWER_LEADER': {int(ms[0]['car_no']) for _,_,_,ms in ranked[2:] if ms},
          'LOWER_SECOND': {int(ms[1]['car_no']) for _,_,_,ms in ranked[2:] if len(ms)>=2},
          'LOWER_TAIL': {int(x['car_no']) for _,_,_,ms in ranked[2:] for x in ms[2:]},
        }
        for name,S in defs.items():
            if name=='SINGLETON_AND_LOWER':
                avail=bool(single and lower)
                hit=bool(winset & set(single) and winset & set(lower))
                def pred(combo):
                    s=set(map(int,combo.split('='))); return bool(s&set(single) and s&set(lower))
            else:
                avail=bool(S)
                hit=bool(winset&S)
                def pred(combo,S=S): return bool(set(map(int,combo.split('=')))&S)
            if not avail: continue
            cat[name]['n']+=1; cat[name]['actual']+=int(hit)
            cat[name]['market']+=sum(prob for combo,prob in mp.items() if pred(combo))
            try:cat[name]['win_odds'].append(float(od[win]['odds']))
            except:pass
        # structural one-ticket branches using a single residual rider selected before result
        scored=sorted(es,key=lambda e:(-f(e.get('score')),int(e.get('car_no','99'))))
        score_rank={int(e['car_no']):i+1 for i,e in enumerate(scored)}
        def best(cars):
            return min(cars,key=lambda c:(score_rank.get(c,99),c)) if cars else None
        S=best(single); L=best(lower)
        # best residual by score and by longest odds when paired with BD (pre-race market information)
        R=best(residual)
        candidates={
          'AB_SINGLE': (A,B,S),'CD_SINGLE':(C,D,S),'BD_SINGLE':(B,D,S),'AC_SINGLE':(A,C,S),
          'AB_LOWER':(A,B,L),'CD_LOWER':(C,D,L),'BD_LOWER':(B,D,L),'AC_LOWER':(A,C,L),
          'AB_RESID':(A,B,R),'CD_RESID':(C,D,R),'BD_RESID':(B,D,R),'AC_RESID':(A,C,R),
        }
        for name,tr in candidates.items():
            if tr[2] is None or len(set(tr))<3: continue
            combo=tk3(tr)
            if combo not in mp: continue
            b=branch[name]; b['n']+=1; b['market']+=mp[combo]
            hit=int(combo==win); b['hit']+=hit; b['ret']+=pay if hit else 0
            try:b['odds'].append(float(od[combo]['odds']))
            except:pass

    md=['# G3三日目 単騎・下位ライン残差分析','', '頻度ではなく市場期待との比較。下位ライン=強度順位3位以下。','']
    md.append('## カテゴリ全体')
    out={'categories':{},'branches':{}}
    for name,v in cat.items():
        n=v['n']; actual=v['actual']/n if n else 0; market=v['market']/n if n else 0
        ratio=actual/market if market else 0
        med=statistics.median(v['win_odds']) if v['win_odds'] else None
        out['categories'][name]={'n':n,'actual':actual,'market':market,'ratio':ratio,'median_winning_odds':med}
        md.append(f"- {name}: n={n} actual {actual:.1%} / market {market:.1%} / ratio {ratio:.3f} / winning median odds {med:.1f}")
    md.append('')
    md.append('## 代表的な残差1点枝')
    for name,v in branch.items():
        n=v['n']; hit=v['hit']/n if n else 0; market=v['market']/n if n else 0; ratio=hit/market if market else 0
        roi=v['ret']/(n*100) if n else 0; med=statistics.median(v['odds']) if v['odds'] else None
        out['branches'][name]={'n':n,'hit':hit,'market':market,'ratio':ratio,'roi':roi,'median_odds':med}
        md.append(f"- {name}: n={n} hit {hit:.2%} / market {market:.2%} / ratio {ratio:.3f} / ROI {roi:.1%} / median odds {med:.1f}")
    (OUT/'residual_units.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'residual_units.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
