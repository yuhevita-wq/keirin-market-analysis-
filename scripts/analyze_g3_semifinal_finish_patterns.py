from __future__ import annotations
import csv, io, json, math, zipfile
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
def num(v,d=0.0):
    try:
        x=float(v or 0); return x if math.isfinite(x) else d
    except:return d
def ival(v):
    try:return int(v or '')
    except:return None
def parse_tri(s):
    p=(s or '').split('-')
    if len(p)==3 and all(x.strip().isdigit() for x in p):return tuple(int(x) for x in p)
    return None
def is_semi(r): return '準決勝' in (r.get('race_type') or '')
def rank_map(es, key, reverse=True):
    def kval(e):return key(e)
    arr=sorted(es,key=lambda e:((-kval(e)) if reverse else kval(e), ival(e.get('car_no')) or 99))
    return {ival(e.get('car_no')):i+1 for i,e in enumerate(arr)}
def tier(rank):
    if rank is None:return 'NA'
    if rank<=2:return '上位2'
    if rank<=5:return '中位3-5'
    return '下位6+'
def role_of(e):
    sz=ival(e.get('line_size')) or 0; pos=ival(e.get('line_position')) or 0
    if sz<=1:return '単騎'
    if pos==1:return '先頭'
    if pos==2:return '番手'
    return '3番手以降'
def unit_of(e):
    sz=ival(e.get('line_size')) or 0; lid=ival(e.get('line_id')); car=ival(e.get('car_no'))
    if sz<=1 or lid is None:return f'S{car}'
    return f'L{lid}'
def line_shape(cars, emap):
    u=[unit_of(emap[c]) for c in cars]
    if u[0]==u[1]==u[2]:return '1-2-3同一ライン'
    if u[0]==u[1]:return '1-2同一＋3別'
    if u[0]==u[2]:return '1-3同一＋2別'
    if u[1]==u[2]:return '2-3同一＋1別'
    return '3人別ユニット'
def market_probs(rows):
    raw=[]
    for x in rows:
        c=parse_tri(x.get('combination')); o=num(x.get('odds'))
        if c and o>0:raw.append((c,1/o))
    s=sum(w for _,w in raw)
    return {c:w/s for c,w in raw} if s else {}
def pct(n,d):return n/d if d else 0

def main():
    races={}; E=defaultdict(list); R=defaultdict(list); O=defaultdict(list)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'results.csv'):R[x['race_id']].append(x)
            for x in read(z,'trifecta_final_odds.csv'):O[x['race_id']].append(x)

    data={sp:{'n':0,'pos':{1:defaultdict(Counter),2:defaultdict(Counter),3:defaultdict(Counter)},'ordered':defaultdict(Counter),'actual_cat':defaultdict(Counter),'market_cat':defaultdict(Counter)} for sp in ('train','test')}
    exact_role_sigs={sp:Counter() for sp in ('train','test')}
    exact_score_sigs={sp:Counter() for sp in ('train','test')}

    for race in races.values():
        if not is_semi(race):continue
        rid=race['race_id']; es=E[rid]; rs=R[rid]
        if len(es)<7:continue
        top={}
        for x in rs:
            fp=ival(x.get('finish_position')); c=ival(x.get('car_no'))
            if fp in (1,2,3) and c is not None:top[fp]=c
        if len(top)!=3:continue
        emap={ival(e.get('car_no')):e for e in es if ival(e.get('car_no')) is not None}
        if any(top[p] not in emap for p in (1,2,3)):continue
        y=ival((race.get('race_date') or '')[:4]); sp='train' if y and y<=2024 else 'test'

        score=rank_map(es,lambda e:num(e.get('score')))
        wrank=rank_map(es,lambda e:num(e.get('win_rate')))
        q2rank=rank_map(es,lambda e:max(0,num(e.get('top2_rate'))-num(e.get('win_rate'))))
        q3rank=rank_map(es,lambda e:max(0,num(e.get('top3_rate'))-num(e.get('top2_rate'))))
        t3rank=rank_map(es,lambda e:num(e.get('top3_rate')))
        ranks={'score':score,'W':wrank,'Q2':q2rank,'Q3':q3rank,'T3':t3rank}
        d=data[sp]; d['n']+=1

        for pos in (1,2,3):
            c=top[pos]; e=emap[c]; pr=d['pos'][pos]
            for name,rm in ranks.items():
                pr[f'{name}_rank'][str(rm[c])]+=1
                pr[f'{name}_tier'][tier(rm[c])]+=1
            pr['style'][e.get('style') or '不明']+=1
            pr['line_role'][role_of(e)]+=1
            pr['line_size'][str(ival(e.get('line_size')) or 0)]+=1
            if pos==1:
                rr=next((x for x in rs if ival(x.get('finish_position'))==1),{})
                pr['winning_method'][rr.get('winning_method') or '不明']+=1

        cars=(top[1],top[2],top[3])
        shape=line_shape(cars,emap); d['ordered']['line_shape'][shape]+=1
        d['ordered']['singleton_count'][str(sum(1 for c in cars if role_of(emap[c])=='単騎'))]+=1
        role_sig=(wrank[cars[0]],q2rank[cars[1]],q3rank[cars[2]])
        score_sig=(score[cars[0]],score[cars[1]],score[cars[2]])
        exact_role_sigs[sp][role_sig]+=1; exact_score_sigs[sp][score_sig]+=1
        rtiers=' / '.join((tier(wrank[cars[0]]),tier(q2rank[cars[1]]),tier(q3rank[cars[2]])))
        stiers=' / '.join((tier(score[cars[0]]),tier(score[cars[1]]),tier(score[cars[2]])))
        d['actual_cat']['role_tier'][rtiers]+=1
        d['actual_cat']['score_tier'][stiers]+=1
        d['actual_cat']['line_shape'][shape]+=1

        mp=market_probs(O[rid])
        if mp:
            for tri,p in mp.items():
                if any(c not in emap for c in tri):continue
                rr=' / '.join((tier(wrank[tri[0]]),tier(q2rank[tri[1]]),tier(q3rank[tri[2]])))
                ss=' / '.join((tier(score[tri[0]]),tier(score[tri[1]]),tier(score[tri[2]])))
                ls=line_shape(tri,emap)
                d['market_cat']['role_tier'][rr]+=p
                d['market_cat']['score_tier'][ss]+=p
                d['market_cat']['line_shape'][ls]+=p

    md=['# G3準決勝 1着・2着・3着ゼロベース分析','', '対象: G3アーカイブの「準決勝」。train=2022-2024 / test=2025-2026H1。','役割指標: W=勝率、Q2=2連対率-勝率、Q3=3連対率-2連対率。ラインは候補選定ではなく結果構造の説明にのみ使用。','']
    jout={}
    for sp in ('train','test'):
        d=data[sp]; n=d['n']; md.append(f'## {sp}: {n}R')
        jout[sp]={'n':n,'position':{},'ordered':{},'actual_cat':{},'market_cat':{}}
        for pos,label in ((1,'1着'),(2,'2着'),(3,'3着')):
            md.append(f'### {label}')
            pr=d['pos'][pos]; jout[sp]['position'][str(pos)]={k:dict(v) for k,v in pr.items()}
            role_metric={1:'W',2:'Q2',3:'Q3'}[pos]
            c=pr[f'{role_metric}_rank']
            top1=c['1']; top2=top1+c['2']; top3=top2+c['3']
            md.append(f'- 本来の役割順位 {role_metric}: 1位 {pct(top1,n):.1%} / 上位2 {pct(top2,n):.1%} / 上位3 {pct(top3,n):.1%}')
            sc=pr['score_rank']; s1=sc['1']; s2=s1+sc['2']; s3=s2+sc['3']; s5=s3+sc['4']+sc['5']
            md.append(f'- 競走得点順位: 1位 {pct(s1,n):.1%} / 上位2 {pct(s2,n):.1%} / 上位3 {pct(s3,n):.1%} / 上位5 {pct(s5,n):.1%}')
            md.append('- 脚質: '+ ' / '.join(f'{k} {pct(v,n):.1%}' for k,v in pr['style'].most_common()))
            md.append('- ライン位置: '+ ' / '.join(f'{k} {pct(v,n):.1%}' for k,v in pr['line_role'].most_common()))
            if pos==1:md.append('- 決まり手: '+ ' / '.join(f'{k} {pct(v,n):.1%}' for k,v in pr['winning_method'].most_common()))
        md.append('### 1-2-3の組み合わせ構造')
        ls=d['ordered']['line_shape']; md.append('- ライン関係: '+ ' / '.join(f'{k} {pct(v,n):.1%}' for k,v in ls.most_common()))
        sg=d['ordered']['singleton_count']; md.append('- 単騎の3着内人数: '+ ' / '.join(f'{k}人 {pct(v,n):.1%}' for k,v in sorted(sg.items())))
        md.append('### 役割順位シグネチャ上位（勝者W順位 / 2着Q2順位 / 3着Q3順位）')
        for sig,v in exact_role_sigs[sp].most_common(12):md.append(f'- {sig}: {v} ({pct(v,n):.1%})')
        md.append('### 得点順位シグネチャ上位（1着 / 2着 / 3着）')
        for sig,v in exact_score_sigs[sp].most_common(12):md.append(f'- {sig}: {v} ({pct(v,n):.1%})')
        for cat,title in [('role_tier','役割3階層'),('score_tier','得点3階層'),('line_shape','ライン関係')]:
            md.append(f'### 市場との比較: {title}')
            rows=[]
            keys=set(d['actual_cat'][cat])|set(d['market_cat'][cat])
            for k in keys:
                a=d['actual_cat'][cat][k]/n if n else 0; m=d['market_cat'][cat][k]/n if n else 0
                if a<0.005 and m<0.005:continue
                rows.append((a/m if m else 0,a-m,a,m,k))
            for ratio,diff,a,m,k in sorted(rows,reverse=True)[:12]:
                md.append(f'- {k}: actual {a:.2%} / market {m:.2%} / ratio {ratio:.3f} / diff {diff:+.2%}')
        md.append('')
        jout[sp]['ordered']={k:dict(v) for k,v in d['ordered'].items()}
        jout[sp]['actual_cat']={k:dict(v) for k,v in d['actual_cat'].items()}
        jout[sp]['market_cat']={k:dict(v) for k,v in d['market_cat'].items()}

    # stable market residuals in both periods
    md.append('## 両期間で同方向に市場より多い役割パターン')
    ntr=data['train']['n']; nte=data['test']['n']; stable=[]
    keys=set(data['train']['actual_cat']['role_tier'])|set(data['test']['actual_cat']['role_tier'])
    for k in keys:
        vals=[]; ok=True
        for sp,n in [('train',ntr),('test',nte)]:
            a=data[sp]['actual_cat']['role_tier'][k]/n if n else 0; m=data[sp]['market_cat']['role_tier'][k]/n if n else 0
            if not (m>0 and a>m):ok=False
            vals.append((a,m,a/m if m else 0))
        if ok and vals[0][0]>=0.01 and vals[1][0]>=0.01:
            stable.append((min(vals[0][2],vals[1][2]),k,vals))
    for score,k,vals in sorted(stable,reverse=True):
        md.append(f"- {k}: train {vals[0][0]:.2%}/{vals[0][1]:.2%} x{vals[0][2]:.3f} / test {vals[1][0]:.2%}/{vals[1][1]:.2%} x{vals[1][2]:.3f}")

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'semifinal_finish_patterns.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'semifinal_finish_patterns.json').write_text(json.dumps(jout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
