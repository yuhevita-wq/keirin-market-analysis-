from __future__ import annotations
import csv, io, math, zipfile
from collections import Counter, defaultdict
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

def f(v,d=-1e9):
    try:
        x=float(v or ''); return x if math.isfinite(x) else d
    except:return d

def semifinal(r): return '準決勝' in (r.get('race_type') or '')

def line_info(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():
            by[int(e['line_id'])].append(e)
    ranked=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda x:int(x['line_position']))
        if len(ms)>=2:
            score=f(ms[0].get('score'))+f(ms[1].get('score'))
            ranked.append((score,f(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid,lid,ms))
    ranked.sort(reverse=True)
    rank={lid:i+1 for i,(*_,lid,ms) in enumerate(ranked)}
    return by,ranked,rank

def role(pos,size):
    if size<2:return '単騎'
    if pos==1:return '先頭'
    if pos==2:return '番手'
    return '3番手以降'

def main():
    races={};E=defaultdict(list);R=defaultdict(list);O=defaultdict(dict)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'results.csv'):R[x['race_id']].append(x)
            for x in read(z,'trifecta_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x
    data={sp:defaultdict(Counter) for sp in ('train','test')}; totals=Counter()
    for race in races.values():
        if not semifinal(race):continue
        rid=race['race_id']; es=E[rid]
        try:y=int(race['race_date'][:4])
        except:continue
        sp='train' if y<=2024 else 'test'
        by,ranked,lrank=line_info(es)
        if not ranked:continue
        finish={}
        for x in R[rid]:
            try:p=int(x.get('finish_position') or 99); c=int(x['car_no'])
            except:continue
            if p in (1,2,3):finish[p]=c
        if len(finish)!=3:continue
        meta={}
        for e in es:
            c=int(e['car_no']); lid=int(e['line_id']) if e.get('line_id','').isdigit() else None
            pos=int(e['line_position']) if e.get('line_position','').isdigit() else None
            size=int(e['line_size']) if e.get('line_size','').isdigit() else 1
            if lid is None or size<2 or lid not in lrank:
                meta[c]=('S','単騎',1,None)
            else:
                meta[c]=(f'L{lrank[lid]}',role(pos,size),size,lid)
        a,b,c=finish[1],finish[2],finish[3]
        m1,m2,m3=meta[a],meta[b],meta[c]
        totals[sp]+=1
        # per finishing position line rank and role
        for p,car in ((1,a),(2,b),(3,c)):
            lr,ro,sz,lid=meta[car]
            data[sp][f'finish{p}_line'][lr]+=1
            data[sp][f'finish{p}_role'][ro]+=1
            data[sp][f'finish{p}_linesize'][str(sz)]+=1
        sig=f'{m1[0]}>{m2[0]}>{m3[0]}'
        data[sp]['ordered_line_signature'][sig]+=1
        rsig=f'{m1[1]}>{m2[1]}>{m3[1]}'
        data[sp]['ordered_role_signature'][rsig]+=1
        # line relation
        l1,l2,l3=m1[3],m2[3],m3[3]
        if l1 is not None and l1==l2==l3:shape='1-2-3同一ライン'
        elif l1 is not None and l1==l2:shape='1-2同一＋3別'
        elif l1 is not None and l1==l3:shape='1-3同一＋2別'
        elif l2 is not None and l2==l3:shape='2-3同一＋1別'
        else:shape='3人別ユニット'
        data[sp]['shape'][shape]+=1
        # condition on winner line rank
        data[sp][f'winner_{m1[0]}_second_line'][m2[0]]+=1
        data[sp][f'winner_{m1[0]}_third_line'][m3[0]]+=1
        data[sp][f'winner_{m1[0]}_second_role'][m2[1]]+=1
        data[sp][f'winner_{m1[0]}_third_role'][m3[1]]+=1
        # strongest line top3 survivor count and role positions
        top_lid=ranked[0][-2]
        surv=[car for car in (a,b,c) if meta[car][3]==top_lid]
        data[sp]['L1_survivors'][str(len(surv))]+=1
        if len(surv)==1:
            data[sp]['L1_one_survivor_finish'][str([a,b,c].index(surv[0])+1)]+=1
            data[sp]['L1_one_survivor_role'][meta[surv[0]][1]]+=1
        # top 3 lines pair/full survival
        for rr,(_,_,_,_,lid,ms) in enumerate(ranked[:3],start=1):
            members={int(e['car_no']) for e in ms}
            k=sum(car in members for car in (a,b,c))
            data[sp][f'L{rr}_survivors'][str(k)]+=1
            if len(ms)>=3 and all(int(e['car_no']) in (a,b,c) for e in ms[:3]):
                data[sp]['threecar_sweep_rank'][f'L{rr}']+=1
        # strongest 3-car line details
        topms=ranked[0][-1]
        if len(topms)>=3:
            data[sp]['L1_size3_races']['n']+=1
            members={int(e['car_no']) for e in topms[:3]}
            k=sum(car in members for car in (a,b,c))
            data[sp]['L1_size3_survivors'][str(k)]+=1
            if k==3:
                order=[]
                posmap={int(e['car_no']):int(e['line_position']) for e in topms[:3]}
                for car in (a,b,c):order.append(str(posmap[car]))
                data[sp]['L1_size3_sweep_order']['>'.join(order)]+=1
        # same-line 1-2 conditional third line/role
        if l1 is not None and l1==l2:
            data[sp]['same12_third_line'][m3[0]]+=1
            data[sp]['same12_third_role'][m3[1]]+=1
            data[sp]['same12_pair_roles'][f'{m1[1]}>{m2[1]}']+=1
            data[sp]['same12_pair_line_rank'][m1[0]]+=1
    md=['# G3準決勝 ライン徹底分析','', 'ライン強さ順位は各ライン先頭+番手の競走得点合計で順位化。L1=最強、L2=2番手、L3=3番手。単騎=S。','']
    sections=[
      ('finish1_line','1着はどのラインから出るか'),('finish2_line','2着はどのラインから出るか'),('finish3_line','3着はどのラインから出るか'),
      ('finish1_role','1着のライン内位置'),('finish2_role','2着のライン内位置'),('finish3_role','3着のライン内位置'),
      ('shape','1-2-3のライン関係'),('ordered_line_signature','着順どおりのライン順位パターン'),('ordered_role_signature','着順どおりのライン内位置パターン'),
      ('L1_survivors','最強ラインL1が3着内に何人残るか'),('L2_survivors','L2が3着内に何人残るか'),('L3_survivors','L3が3着内に何人残るか'),
      ('L1_one_survivor_finish','L1が1人だけ残る時、その着順'),('L1_one_survivor_role','L1が1人だけ残る時、その役割'),
      ('same12_pair_roles','1-2着同ライン時の位置順'),('same12_pair_line_rank','1-2着同ライン時、そのラインの強さ順位'),('same12_third_line','1-2着同ライン時、3着はどのラインか'),('same12_third_role','1-2着同ライン時、3着の位置'),
      ('threecar_sweep_rank','3車ラインが1-2-3独占した時のライン順位'),('L1_size3_survivors','L1が3車ラインの時の3着内残存人数'),('L1_size3_sweep_order','L1三車独占時の着順（1=先頭,2=番手,3=三番手）')]
    for sp in ('train','test'):
        md += [f'## {sp}: {totals[sp]}R','']
        for key,title in sections:
            c=data[sp][key]; den=sum(c.values()) or 1
            md.append(f'### {title}')
            for k,v in c.most_common(20):md.append(f'- {k}: {v} ({v/den:.1%})')
            md.append('')
        for wr in ('L1','L2','L3','S'):
            k=f'winner_{wr}_second_line'
            if data[sp][k]:
                md.append(f'### 1着が{wr}の時：2着ライン')
                den=sum(data[sp][k].values())
                for x,v in data[sp][k].most_common():md.append(f'- {x}: {v} ({v/den:.1%})')
                md.append(f'### 1着が{wr}の時：3着ライン')
                den=sum(data[sp][f'winner_{wr}_third_line'].values())
                for x,v in data[sp][f'winner_{wr}_third_line'].most_common():md.append(f'- {x}: {v} ({v/den:.1%})')
                md.append('')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'semifinal_line_patterns.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':main()
