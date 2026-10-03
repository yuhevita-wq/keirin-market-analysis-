from __future__ import annotations
import csv, io, math, zipfile
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
def day3(rid):return len(rid)==16 and rid.isdigit() and rid[10:12]=='03'
def lines(es):
    by=defaultdict(list); ranked=[]
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():by[int(e['line_id'])].append(e)
    for lid,ms in by.items():
        ms.sort(key=lambda x:int(x['line_position']))
        if len(ms)>=2: ranked.append((f(ms[0].get('score'))+f(ms[1].get('score')),f(ms[0].get('score')),lid,ms))
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2])); return by,ranked
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':return p.get('combination')
    return None

def main():
    races={};E=defaultdict(list);P=defaultdict(list)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
    out={k:defaultdict(Counter) for k in ('train','test')}
    totals={k:Counter() for k in ('train','test')}
    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        try:y=int(race.get('race_date','9999')[:4])
        except:continue
        sp='train' if y<=2024 else 'test'
        win=paid(P[rid]); es=E[rid]
        if not win:continue
        by,ranked=lines(es)
        if not ranked:continue
        main=ranked[0][3]; A,B=int(main[0]['car_no']),int(main[1]['car_no'])
        line_rank={lid:i+1 for i,(_,_,lid,_) in enumerate(ranked)}
        meta={}
        for e in es:
            c=int(e['car_no']); lid=int(e['line_id']) if e.get('line_id','').isdigit() else None
            pos=int(e['line_position']) if e.get('line_position','').isdigit() else None
            mem=by.get(lid,[]) if lid is not None else []
            if len(mem)<2: meta[c]=('S',None,'singleton')
            else: meta[c]=(f'L{line_rank.get(lid,99)}',line_rank.get(lid,99),'leader' if pos==1 else ('second' if pos==2 else 'tail'))
        cars=list(map(int,win.split('='))); mains=[c for c in cars if c in (A,B)]; outs=[c for c in cars if c not in (A,B)]
        totals[sp]['races']+=1; totals[sp][f'm{len(mains)}']+=1
        if len(mains)==1:
            totals[sp]['m1_A' if A in mains else 'm1_B']+=1
            u=[meta[c][0] for c in outs]
            if len(u)==2 and u[0]==u[1] and u[0]!='S':
                rank=meta[outs[0]][1]; out[sp]['m1_same_pair_rank'][str(rank)]+=1
                roles='+'.join(sorted(meta[c][2] for c in outs)); out[sp]['m1_same_pair_roles'][roles]+=1
            else:
                for c in outs:
                    unit,rank,role=meta[c]
                    out[sp]['m1_mixed_unit'][unit]+=1
                    out[sp]['m1_mixed_role'][role]+=1
        elif len(mains)==0:
            units=[meta[c][0] for c in outs]; cnt=Counter(units)
            nonS=[(u,n) for u,n in cnt.items() if u!='S']
            if any(n==3 for u,n in nonS):
                u=next(u for u,n in nonS if n==3); rank=int(u[1:]); out[sp]['m0_triple_rank'][str(rank)]+=1
            elif any(n==2 for u,n in nonS):
                u=next(u for u,n in nonS if n==2); rank=int(u[1:]); out[sp]['m0_pair_rank'][str(rank)]+=1
                paircars=[c for c in outs if meta[c][0]==u]
                roles='+'.join(sorted(meta[c][2] for c in paircars)); out[sp]['m0_pair_roles'][roles]+=1
                other=next(c for c in outs if meta[c][0]!=u)
                out[sp]['m0_pair_other_unit'][meta[other][0]]+=1
                out[sp]['m0_pair_other_role'][meta[other][2]]+=1
            else:
                for c in outs:
                    out[sp]['m0_no_pair_unit'][meta[c][0]]+=1
                    out[sp]['m0_no_pair_role'][meta[c][2]]+=1
    md=['# G3三日目 本命外のライン順位詳細','']
    for sp in ('train','test'):
        md.append(f'## {sp}')
        t=totals[sp]; md.append(f"- races {t['races']} / A-B両消え {t['m0']} / A-B片方 {t['m1']} / A-B両残り {t['m2']}")
        if t['m1']:
            md.append(f"- A/B片残り内訳: Aのみ {t['m1_A']} ({t['m1_A']/t['m1']:.1%}) / Bのみ {t['m1_B']} ({t['m1_B']/t['m1']:.1%})")
        for section,title in [
            ('m1_same_pair_rank','A/B片残り + 外同一ライン2人: そのライン順位'),
            ('m1_same_pair_roles','A/B片残り + 外同一ライン2人: 役割'),
            ('m0_triple_rank','A/B両消え + 外1ライン3車完結: ライン順位'),
            ('m0_pair_rank','A/B両消え + 外同一ライン2人+別1人: ペアのライン順位'),
            ('m0_pair_roles','A/B両消え + 外同一ライン2人: ペア役割'),
            ('m0_pair_other_unit','A/B両消え + 外同一ライン2人+別1人: 別1人の所属'),
            ('m0_pair_other_role','A/B両消え + 外同一ライン2人+別1人: 別1人の役割')]:
            md.append(f'### {title}')
            c=out[sp][section]; den=sum(c.values()) or 1
            for k,v in sorted(c.items(),key=lambda kv:(-kv[1],kv[0])): md.append(f'- {k}: {v} ({v/den:.1%})')
        md.append('')
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'nonmain_rank_detail.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); print('\n'.join(md))
if __name__=='__main__':main()

# workflow trigger
