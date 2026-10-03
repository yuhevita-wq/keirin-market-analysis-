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

def role(e):
    try:size=int(e.get('line_size') or 1); pos=int(e.get('line_position') or 1)
    except:return '単騎'
    if size<=1:return '単騎'
    if pos==1:return '先頭'
    if pos==2:return '番手'
    return '3番手以降'

def unit(e):
    try:
        size=int(e.get('line_size') or 1)
        if size<=1:return ('S',int(e['car_no']))
        return ('L',int(e.get('line_id') or -1))
    except:return ('S',e.get('car_no'))

def tier(r):
    if r<=2:return '上位2'
    if r<=5:return '中位3-5'
    return '下位6+'

def pct(n,d): return 100*n/d if d else 0

def main():
    races={}; E=defaultdict(list); R=defaultdict(list)
    for p in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'results.csv'):R[x['race_id']].append(x)

    splits={'train':lambda y:y<=2024,'test':lambda y:y>=2025}
    data={sp:defaultdict(Counter) for sp in splits}
    totals=Counter()

    for rid,race in races.items():
        if '準決勝' not in (race.get('race_type') or ''): continue
        try:y=int((race.get('race_date') or '')[:4])
        except:continue
        sp='train' if y<=2024 else 'test'
        es=E[rid]; rs=R[rid]
        if not es or not rs:continue
        bycar={int(e['car_no']):e for e in es if e.get('car_no','').isdigit()}
        fin={}
        for r in rs:
            try:p=int(r.get('finish_position') or 99); c=int(r['car_no'])
            except:continue
            if p in (1,2,3):fin[p]=(c,r)
        if len(fin)<3 or any(fin[p][0] not in bycar for p in (1,2,3)):continue
        ranked=sorted(bycar.values(), key=lambda e:(-f(e.get('score')),int(e['car_no'])))
        sr={int(e['car_no']):i+1 for i,e in enumerate(ranked)}
        c1,r1=fin[1]; c2,r2=fin[2]; c3,r3=fin[3]
        e1,e2,e3=bycar[c1],bycar[c2],bycar[c3]
        u1,u2,u3=unit(e1),unit(e2),unit(e3)
        rr1,rr2,rr3=sr[c1],sr[c2],sr[c3]
        t1,t2,t3=tier(rr1),tier(rr2),tier(rr3)
        totals[sp]+=1
        d=data[sp]

        # Exact score-rank path and tiers.
        d['score_path'][(rr1,rr2,rr3)]+=1
        d['tier_path'][(t1,t2,t3)]+=1

        # 1着の具体像 -> 2着/3着条件分岐
        wkey=f'{t1}|{role(e1)}|{r1.get("winning_method") or "?"}'
        d['winner_profile'][wkey]+=1
        d['winner_to_second_same'][f'{t1}|'+('同ライン' if u1==u2 else '別ライン')]+=1
        d['winner_to_second_tier'][f'{t1}->{t2}']+=1
        d['winner_to_third_tier'][f'{t1}->{t3}']+=1

        # 1-2関係の中身
        rel12='同ライン' if u1==u2 else '別ライン'
        rel3='1着と同ライン' if u3==u1 else ('2着と同ライン' if u3==u2 else '第三ユニット')
        d['rel12_rel3'][f'{rel12}|{rel3}']+=1
        d['role123'][f'{role(e1)}>{role(e2)}>{role(e3)}']+=1
        if u1==u2:
            d['same12_roles'][f'{role(e1)}>{role(e2)}']+=1
            d['same12_third_tier'][t3]+=1
            d['same12_third_role'][role(e3)]+=1
            d['same12_third_relation'][rel3]+=1
        else:
            d['diff12_third_relation'][rel3]+=1
            d['diff12_second_tier'][t2]+=1
            d['diff12_third_tier'][t3]+=1

        # 強-強-穴を具体化: 1着得点上位2、2着得点上位2、3着得点6位以下
        if rr1<=2 and rr2<=2 and rr3>=6:
            d['SSO_third_role'][role(e3)]+=1
            d['SSO_third_style'][e3.get('style') or '?']+=1
            d['SSO_third_relation'][rel3]+=1
            d['SSO_third_rank'][str(rr3)]+=1
            d['SSO_12_relation'][rel12]+=1
            try: sc=int(e3.get('sashi_count') or 0); mk=int(e3.get('mark_count') or 0); th=int(e3.get('third_count') or 0)
            except: sc=mk=th=0
            d['SSO_third_traits']['差し回数>=3' if sc>=3 else '差し回数0-2']+=1
            d['SSO_third_traits']['マーク回数>=3' if mk>=3 else 'マーク回数0-2']+=1
            d['SSO_third_traits']['3着回数>=3' if th>=3 else '3着回数0-2']+=1

        # 1着が得点上位3の時、2着・3着の具体像
        if rr1<=3:
            d['Wtop3_second_tier'][t2]+=1
            d['Wtop3_second_role'][role(e2)]+=1
            d['Wtop3_second_relation']['同ライン' if u1==u2 else '別ライン']+=1
            d['Wtop3_third_tier'][t3]+=1
            d['Wtop3_third_role'][role(e3)]+=1
            d['Wtop3_third_relation'][rel3]+=1

        # 3着穴(得点6位以下)の時、1-2の形と3着の姿
        if rr3>=6:
            d['low3_rel12'][rel12]+=1
            d['low3_role'][role(e3)]+=1
            d['low3_style'][e3.get('style') or '?']+=1
            d['low3_relation'][rel3]+=1
            d['low3_first_tier'][t1]+=1
            d['low3_second_tier'][t2]+=1

    md=['# G3準決勝 条件分岐で言語化する1着→2着→3着','',
        '「強い・残る」の抽象語を避けるため、得点順位・ライン位置・1-2関係・3着の所属を条件付きで分解。','']
    for sp in ('train','test'):
        n=totals[sp]; d=data[sp]
        md += [f'## {sp}: {n}R','']
        def sec(key,title,limit=None):
            md.append(f'### {title}')
            c=d[key]; den=sum(c.values()) or 1
            items=c.most_common(limit)
            for k,v in items: md.append(f'- {k}: {v} ({pct(v,den):.1f}%)')
            md.append('')
        sec('tier_path','得点階層の1着→2着→3着 上位パターン',12)
        sec('winner_to_second_same','1着得点階層ごとの2着同ライン/別ライン')
        sec('winner_to_second_tier','1着得点階層→2着得点階層')
        sec('winner_to_third_tier','1着得点階層→3着得点階層')
        sec('rel12_rel3','1-2の関係から見た3着の所属')
        sec('same12_roles','1-2同ライン時の位置ペア')
        sec('same12_third_tier','1-2同ライン時の3着得点階層')
        sec('same12_third_role','1-2同ライン時の3着ライン位置')
        sec('diff12_third_relation','1-2別ライン時の3着所属')
        sec('Wtop3_second_tier','1着が得点上位3人だった時の2着得点階層')
        sec('Wtop3_second_role','1着が得点上位3人だった時の2着位置')
        sec('Wtop3_third_tier','1着が得点上位3人だった時の3着得点階層')
        sec('Wtop3_third_role','1着が得点上位3人だった時の3着位置')
        sec('low3_rel12','3着が得点6位以下だった時の1-2関係')
        sec('low3_role','3着が得点6位以下だった時の3着位置')
        sec('low3_style','3着が得点6位以下だった時の脚質')
        sec('low3_relation','3着が得点6位以下だった時の所属')
        sec('SSO_12_relation','得点上位2→上位2→6位以下の時の1-2関係')
        sec('SSO_third_rank','得点上位2→上位2→6位以下の3着得点順位')
        sec('SSO_third_role','得点上位2→上位2→6位以下の3着位置')
        sec('SSO_third_style','得点上位2→上位2→6位以下の3着脚質')
        sec('SSO_third_relation','得点上位2→上位2→6位以下の3着所属')
        sec('SSO_third_traits','得点上位2→上位2→6位以下の3着個体特徴')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'semifinal_conditional_roles.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':main()
