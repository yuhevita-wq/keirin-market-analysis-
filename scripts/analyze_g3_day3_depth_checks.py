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
def num(v,d=float('-inf')):
    try:
        x=float(v or ''); return x if math.isfinite(x) else d
    except:return d
def iv(v):
    try:return int(v or '')
    except:return None
def day3(r):return len(r)==16 and r.isdigit() and r[10:12]=='03'
def pc(s):
    if not s:return None
    for sep in ('=','-'):
        p=s.split(sep)
        if len(p)==3 and all(x.strip().isdigit() for x in p):return tuple(sorted(int(x) for x in p))
    return None
def groups(es):
    by=defaultdict(list);assigned=set()
    for e in es:
        lid=iv(e.get('line_id'));pos=iv(e.get('line_position'));car=iv(e.get('car_no'))
        if lid is not None and pos is not None and car is not None:by[lid].append(e);assigned.add(car)
    for lid in by:by[lid].sort(key=lambda e:iv(e.get('line_position')) or 99)
    singles=[]
    for ms in by.values():
        if len(ms)==1:singles.append(ms[0])
    for e in es:
        c=iv(e.get('car_no'))
        if c is not None and c not in assigned:singles.append(e)
    ss=[];seen=set()
    for e in singles:
        c=iv(e.get('car_no'))
        if c is not None and c not in seen:seen.add(c);ss.append(e)
    multi=[]
    for lid,ms in by.items():
        if len(ms)<2:continue
        st=num(ms[0].get('score'))+num(ms[1].get('score'))
        multi.append(((st,num(ms[0].get('score')),1 if len(ms)>=3 else 0,-lid),lid,ms))
    multi.sort(reverse=True,key=lambda x:x[0]);return multi,ss
def paid(ps):
    for p in ps:
        if p.get('ticket_type')=='3連複' and p.get('status')=='paid':
            c=pc(p.get('combination',''))
            if c:
                try:y=int(float(p.get('payout_yen') or 0))
                except:y=0
                return c,y
    return None,0
def mq(rows):
    raw=[]
    for o in rows:
        c=pc(o.get('combination',''));od=num(o.get('odds'))
        if c and od>0 and math.isfinite(od):raw.append((c,od,1/od))
    den=sum(x[2] for x in raw)
    return ({c:w/den for c,od,w in raw},{c:od for c,od,w in raw}) if den else ({},{})
def cal(rows):
    n=len(rows);a=sum(r['actual'] for r in rows);m=sum(r['market'] for r in rows)
    return {'n':n,'actual':a/n if n else 0,'market':m/n if n else 0,'ratio':a/m if m else 0,'diff':a/n-m/n if n else 0}
def tick(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);ret=sum(r['pay'] if r['hit'] else 0 for r in rows);ods=[r['odds'] for r in rows]
    return {'n':n,'hits':h,'hit_rate':h/n if n else 0,'roi':ret/(n*100) if n else 0,'median_odds':statistics.median(ods) if ods else None}
def bucket_rank(r):
    return '1-3' if r<=3 else ('4-6' if r<=6 else '7-9')

def main():
    races={};E=defaultdict(list);P=defaultdict(list);O=defaultdict(list)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'payouts.csv'):P[x['race_id']].append(x)
            for x in read(z,'trio_final_odds.csv'):O[x['race_id']].append(x)
    weak=defaultdict(list);sing=defaultdict(list);r3_year=defaultdict(list);r3_type=defaultdict(list);r3_split=defaultdict(list)
    for race in races.values():
        rid=race['race_id']
        if not day3(rid):continue
        ls,ss=groups(E[rid]);win,pay=paid(P[rid]);q,odds=mq(O[rid])
        if not win or not q or win not in q:continue
        split='train' if int(race['race_date'][:4])<=2024 else 'test';year=race['race_date'][:4];typ=race.get('race_type','')
        all_lines=[[iv(e.get('car_no')) for e in ms if iv(e.get('car_no')) is not None] for _,_,ms in ls]
        # True weakest line means at least three multi-rider lines.
        if len(all_lines)>=3:
            w=all_lines[-1];ws=set(w)
            defs={
                'weak_any':lambda c:any(x in ws for x in c),
                'weak_pair':lambda c:set(w[:2])<=set(c),
                'weak_leader':lambda c:w[0] in c,
                'weak_second':lambda c:w[1] in c,
                'weak_tail':lambda c:any(x in c for x in w[2:]),
            }
            for name,fn in defs.items():
                weak[(name,split)].append({'actual':int(fn(win)),'market':sum(p for c,p in q.items() if fn(c))})
            weak[(f'weak_size_{len(w)}',split)].append({'actual':int(any(x in ws for x in win)),'market':sum(p for c,p in q.items() if any(x in ws for x in c))})
        # Singleton rider-level by pre-race score rank.
        scored=sorted([e for e in E[rid] if iv(e.get('car_no')) is not None],key=lambda e:(-num(e.get('score')),iv(e.get('car_no')) or 99))
        ranks={iv(e.get('car_no')):j+1 for j,e in enumerate(scored)}
        for e in ss:
            c=iv(e.get('car_no'))
            if c is None:continue
            b=bucket_rank(ranks[c]);market=sum(p for comb,p in q.items() if c in comb)
            sing[(b,split)].append({'actual':int(c in win),'win':int(win[0]==c),'market':market})
        # Second-strongest line intact 3-rider trio, only when line has 3+.
        if len(all_lines)>=2 and len(all_lines[1])>=3:
            c=tuple(sorted(all_lines[1][:3]))
            if c in q:
                row={'hit':int(c==win),'pay':pay,'odds':odds[c]}
                r3_year[year].append(row);r3_type[typ].append(row);r3_split[split].append(row)
    out={'weak':{},'singleton_rank':{},'rank2_third':{'by_year':{},'by_type':{},'by_split':{}}}
    md=['# G3三日目 深掘り再検証','', '最弱ラインは3本以上の複数ラインがあるレースだけ。単騎は得点順位別。第2強度3車ラインは年別・競走種別で再検証。','']
    md.append('## 本当の最弱ライン 市場較正')
    for name in ('weak_any','weak_pair','weak_leader','weak_second','weak_tail','weak_size_2','weak_size_3','weak_size_4'):
        out['weak'][name]={};md.append(f'### {name}')
        for split in ('train','test'):
            s=cal(weak[(name,split)]);out['weak'][name][split]=s
            md.append(f"- {split}: n={s['n']} actual {s['actual']:.1%} / market {s['market']:.1%} / diff {s['diff']*100:+.2f}pt / ratio {s['ratio']:.3f}")
        md.append('')
    md.append('## 単騎 得点順位別 top3市場較正')
    for b in ('1-3','4-6','7-9'):
        out['singleton_rank'][b]={};md.append(f'### score_rank {b}')
        for split in ('train','test'):
            rows=sing[(b,split)];n=len(rows);a=sum(x['actual'] for x in rows);w=sum(x['win'] for x in rows);m=sum(x['market'] for x in rows)
            s={'n':n,'top3':a/n if n else 0,'win':w/n if n else 0,'market':m/n if n else 0,'ratio':a/m if m else 0};out['singleton_rank'][b][split]=s
            md.append(f"- {split}: n={n} top3 {s['top3']:.1%} / win {s['win']:.1%} / market {s['market']:.1%} / ratio {s['ratio']:.3f}")
        md.append('')
    md.append('## 第2強度ライン3車丸ごと 3連複')
    for split in ('train','test'):
        s=tick(r3_split[split]);out['rank2_third']['by_split'][split]=s
        md.append(f"- {split}: n={s['n']} hits={s['hits']} hit {s['hit_rate']:.2%} ROI {s['roi']:.1%} median_odds {s['median_odds']:.1f}")
    md.append('### 年別')
    for y in sorted(r3_year):
        s=tick(r3_year[y]);out['rank2_third']['by_year'][y]=s
        md.append(f"- {y}: n={s['n']} hits={s['hits']} hit {s['hit_rate']:.2%} ROI {s['roi']:.1%} median_odds {s['median_odds']:.1f}")
    md.append('### 競走種別 n>=40')
    for typ,rows in sorted(r3_type.items(),key=lambda kv:-len(kv[1])):
        if len(rows)<40:continue
        s=tick(rows);out['rank2_third']['by_type'][typ]=s
        md.append(f"- {typ}: n={s['n']} hits={s['hits']} hit {s['hit_rate']:.2%} ROI {s['roi']:.1%} median_odds {s['median_odds']:.1f}")
    (OUT/'depth_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'depth_checks.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
