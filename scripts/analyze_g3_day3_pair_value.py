from __future__ import annotations

import csv, io, json, math, statistics, zipfile
from collections import defaultdict
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
        x=float(v or '')
        return x if math.isfinite(x) else d
    except:return d

def i(v):
    try:return int(v or '')
    except:return None

def day3(rid):
    return len(rid)==16 and rid.isdigit() and rid[10:12]=='03'

def rank_lines(es):
    by=defaultdict(list)
    for e in es:
        if e.get('line_id','').isdigit() and e.get('line_position','').isdigit():by[int(e['line_id'])].append(e)
    out=[]
    for lid,ms in by.items():
        ms=sorted(ms,key=lambda e:int(e['line_position']))
        if len(ms)<2:continue
        ps=f(ms[0].get('score'))+f(ms[1].get('score'))
        key=(ps,f(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid)
        out.append((key,lid,ms))
    out.sort(reverse=True,key=lambda x:x[0]);return out

def tkey(a,b,c):return '='.join(map(str,sorted((a,b,c))))

def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            try:return p.get('combination'),int(float(p.get('payout_yen') or 0))
            except:return p.get('combination'),0
    return None,0

def summarize_tickets(ts):
    n=len(ts); stake=n*100; ret=sum(x['ret'] for x in ts); hit=sum(x['hit'] for x in ts)
    return {'tickets':n,'hit_rate':hit/n if n else 0,'roi':ret/stake if stake else 0,
            'median_odds':statistics.median([x['odds'] for x in ts]) if ts else None}

def summarize_races(rs):
    n=len(rs); stake=sum(len(r['tickets'])*100 for r in rs); ret=sum(sum(t['ret'] for t in r['tickets']) for r in rs)
    hits=sum(any(t['hit'] for t in r['tickets']) for r in rs); pts=sum(len(r['tickets']) for r in rs)
    return {'races':n,'avg_points':pts/n if n else 0,'hit_rate':hits/n if n else 0,'roi':ret/stake if stake else 0}

def main():
    races={}; E=defaultdict(list); R=defaultdict(list); P=defaultdict(list); O=defaultdict(dict)
    for path in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(path) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'results.csv'):R[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']][x.get('combination','')]=x

    per=[]
    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        lr=rank_lines(E[rid])
        if len(lr)<2:continue
        m=lr[0][2]; q=lr[1][2]
        A,B=int(m[0]['car_no']),int(m[1]['car_no']); C,D=int(q[0]['car_no']),int(q[1]['car_no'])
        win,pay=paid(P[rid])
        if not win:continue
        year=int(race['race_date'][:4]); split='train' if year<=2024 else 'test'
        cars=sorted(int(e['car_no']) for e in E[rid] if e.get('car_no','').isdigit())
        od=O[rid]
        rec={'rid':rid,'split':split,'win':win,'pay':pay,'AB':[],'CD':[]}
        for label,pair,kmax in [('AB',(A,B),4),('CD',(C,D),5)]:
            pool=[x for x in cars if x not in pair]
            cand=[]
            for x in pool:
                c=tkey(pair[0],pair[1],x); ox=od.get(c,{})
                odds=f(ox.get('odds'),1e18)
                if odds>=1e17:continue
                cand.append((odds,x,c))
            cand.sort(key=lambda z:(z[0],z[1]))
            for pos,(odds,x,c) in enumerate(cand[:kmax],1):
                rec[label].append({'pos':pos,'car':x,'combo':c,'odds':odds,'hit':int(c==win),'ret':pay if c==win else 0})
        per.append(rec)

    md=['# G3三日目 AB/CD固定ペア 候補順位と価格','']
    for label,maxp in [('AB',4),('CD',5)]:
        md.append(f'## {label} 市場候補順位ごとの1点成績')
        for pos in range(1,maxp+1):
            for split in ('train','test'):
                ts=[t for r in per if r['split']==split for t in r[label] if t['pos']==pos]
                s=summarize_tickets(ts)
                med=s['median_odds'] if s['median_odds'] is not None else 0
                md.append(f"- {label}候補{pos} {split}: {s['tickets']}券 hit {s['hit_rate']:.1%} ROI {s['roi']:.1%} 中央odds {med:.1f}")
        md.append('')

    floors=[0,5,7,10,12,15,20,25,30]
    configs=[('AB2','AB',2),('AB3','AB',3),('CD2','CD',2),('CD3','CD',3),('CD4','CD',4)]
    summaries={}
    md.append('## 固定ペア別 オッズ下限')
    for name,label,k in configs:
        summaries[name]={}
        for floor in floors:
            summaries[name][floor]={}
            seg=[]
            for r in per:
                tickets=[t for t in r[label] if t['pos']<=k and t['odds']>=floor]
                if tickets:seg.append({'split':r['split'],'tickets':tickets})
            vals=[]
            for split in ('train','test'):
                ss=summarize_races([x for x in seg if x['split']==split]);summaries[name][floor][split]=ss
                vals.append(f"{split} {ss['races']}R {ss['avg_points']:.1f}点 hit {ss['hit_rate']:.1%} ROI {ss['roi']:.1%}")
            md.append(f"- {name} odds>={floor}: "+' / '.join(vals))
        md.append('')

    md.append('## AB + CD 同時運用')
    combos=[('AB2+CD3',2,3),('AB3+CD3',3,3),('AB2+CD2',2,2),('AB3+CD2',3,2)]
    combined={}
    for name,ka,kc in combos:
        combined[name]={}
        for floor in floors:
            combined[name][floor]={}
            for split in ('train','test'):
                rr=[]
                for r in per:
                    if r['split']!=split:continue
                    ts=[t for t in r['AB'] if t['pos']<=ka and t['odds']>=floor]+[t for t in r['CD'] if t['pos']<=kc and t['odds']>=floor]
                    uniq={t['combo']:t for t in ts}
                    if uniq:rr.append({'tickets':list(uniq.values())})
                s=summarize_races(rr);combined[name][floor][split]=s
            tr=combined[name][floor]['train'];te=combined[name][floor]['test']
            md.append(f"- {name} odds>={floor}: train {tr['avg_points']:.1f}点 hit {tr['hit_rate']:.1%} ROI {tr['roi']:.1%} / test {te['avg_points']:.1f}点 hit {te['hit_rate']:.1%} ROI {te['roi']:.1%}")
        md.append('')

    out={'summaries':summaries,'combined':combined,'floors':floors}
    (OUT/'pair_value.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'pair_value.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':main()
