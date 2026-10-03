from __future__ import annotations
import csv, io, json, math, zipfile
from collections import defaultdict, Counter
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
def lines(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():
            by[int(e['line_id'])].append(e)
    ranked=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda x:int(x['line_position']))
        if len(ms)>=2:
            strength=f(ms[0].get('score'))+f(ms[1].get('score'))
            ranked.append((strength,f(ms[0].get('score')),lid,ms))
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2]))
    return by,ranked
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            return p.get('combination')
    return None
def market_probs(od):
    inv={}
    for c,row in od.items():
        try:o=float(row.get('odds') or 0)
        except:o=0
        if o>0:inv[c]=1/o
    s=sum(inv.values()); return {c:v/s for c,v in inv.items()} if s else {}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    races={}; E=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x

    split={'train':lambda y:y<=2024,'test':lambda y:y>=2025}
    out={k:{'n':0,'main_survivors':Counter(),'one_main_outside_shape':Counter(),'zero_main_shape':Counter(),'outside_roles':Counter(),'outside_units':Counter(),'market':defaultdict(float),'actual':Counter()} for k in split}

    def rider_meta(car, es, by, line_rank):
        e=next(x for x in es if int(x['car_no'])==car)
        lid=int(e['line_id']) if e.get('line_id','').isdigit() else None
        pos=int(e['line_position']) if e.get('line_position','').isdigit() else None
        mem=by.get(lid,[]) if lid is not None else []
        if len(mem)<2:
            return ('S','singleton','S')
        rank=line_rank.get(lid,99)
        role='leader' if pos==1 else ('second' if pos==2 else 'tail')
        unit=f'L{rank}'
        return (unit,role,unit)

    def shape_for(combo, A,B, es, by, line_rank):
        cars=list(map(int,combo.split('=')))
        mains=[c for c in cars if c in (A,B)]
        outs=[c for c in cars if c not in (A,B)]
        metas=[rider_meta(c,es,by,line_rank) for c in outs]
        units=[m[0] for m in metas]
        roles=[m[1] for m in metas]
        if len(mains)==1:
            same = len(outs)==2 and units[0]==units[1] and units[0] != 'S'
            if same: sh='1M+same_out_line_pair'
            elif 'S' in units and len(set(units))==2: sh='1M+singleton+other'
            elif units.count('S')==2: sh='1M+2singletons'
            else: sh='1M+two_different_out_units'
        elif len(mains)==0:
            cnt=Counter(units)
            mx=max(cnt.values()) if cnt else 0
            if mx==3 and 'S' not in cnt: sh='0M+same_out_line_triple'
            elif mx==2 and any(u!='S' and n==2 for u,n in cnt.items()): sh='0M+same_out_line_pair+other'
            elif cnt.get('S',0)>=2: sh='0M+2plus_singletons'
            elif cnt.get('S',0)==1: sh='0M+singleton+two_other_units'
            else: sh='0M+three_different_out_units'
        else:
            sh='2M+one_out'
        return len(mains), sh, units, roles

    for race in races.values():
        rid=race['race_id'];
        if not day3(rid): continue
        try:y=int(rid[:4])
        except: continue
        key='train' if y<=2024 else 'test'
        if key not in out: continue
        es=E[rid]; win=paid(P[rid]); od=O[rid]
        if not win or not od: continue
        by,ranked=lines(es)
        if not ranked: continue
        main=ranked[0][3]
        if len(main)<2: continue
        A,B=int(main[0]['car_no']),int(main[1]['car_no'])
        line_rank={lid:i+1 for i,(_,_,lid,_) in enumerate(ranked)}
        mp=market_probs(od)
        if not mp: continue
        d=out[key]; d['n']+=1
        ms,sh,units,roles=shape_for(win,A,B,es,by,line_rank)
        d['main_survivors'][str(ms)]+=1
        if ms==1:d['one_main_outside_shape'][sh]+=1
        if ms==0:d['zero_main_shape'][sh]+=1
        for u in units:d['outside_units'][u]+=1
        for r in roles:d['outside_roles'][r]+=1
        d['actual'][sh]+=1
        # structural market mass for each shape
        for combo,p in mp.items():
            try:cms,csh,_,_=shape_for(combo,A,B,es,by,line_rank)
            except:continue
            if csh==sh: pass
            d['market'][csh]+=p

    md=['# G3三日目 本命以外ゼロベース再分析','', '本命A-Bだけ固定。2番手ラインを特別扱いせず、外側をライン順位・単騎・同一ユニット/異ユニットで分解。','']
    jout={}
    for key,d in out.items():
        n=d['n']; md.append(f'## {key}: {n}R')
        md.append('### 本命A/Bの3着内残存人数')
        for k in ('2','1','0'):
            v=d['main_survivors'][k]; md.append(f'- {k}人: {v} ({v/n:.1%})')
        one=sum(d['one_main_outside_shape'].values()) or 1
        md.append('### A/Bが1人だけ残った時、外2人の構造')
        for k,v in d['one_main_outside_shape'].most_common(): md.append(f'- {k}: {v} ({v/one:.1%})')
        zero=sum(d['zero_main_shape'].values()) or 1
        md.append('### A/Bが2人とも消えた時、外3人の構造')
        for k,v in d['zero_main_shape'].most_common(): md.append(f'- {k}: {v} ({v/zero:.1%})')
        md.append('### 構造別 実現率 vs 市場期待（全レース基準）')
        keys=sorted(set(d['actual'])|set(d['market']))
        for k in keys:
            a=d['actual'][k]/n; m=d['market'][k]/n; ratio=a/m if m else 0
            md.append(f'- {k}: actual {a:.2%} / market {m:.2%} / ratio {ratio:.3f}')
        md.append('')
        jout[key]={
            'n':n,
            'main_survivors':dict(d['main_survivors']),
            'one_main_outside_shape':dict(d['one_main_outside_shape']),
            'zero_main_shape':dict(d['zero_main_shape']),
            'outside_units':dict(d['outside_units']),
            'outside_roles':dict(d['outside_roles']),
            'actual':dict(d['actual']),
            'market':dict(d['market'])
        }
    (OUT/'nonmain_structure.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'nonmain_structure.json').write_text(json.dumps(jout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
