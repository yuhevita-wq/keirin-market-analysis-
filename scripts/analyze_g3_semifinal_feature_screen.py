from __future__ import annotations
import csv, io, math, zipfile
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
def num(v):
    try:
        x=float(v or 0); return x if math.isfinite(x) else 0
    except:return 0
def ii(v):
    try:return int(v or '')
    except:return None
def rank(es,fn):
    a=sorted(es,key=lambda e:(-fn(e),ii(e.get('car_no')) or 99))
    return {ii(e.get('car_no')):i+1 for i,e in enumerate(a)}
def val(e,k):return num(e.get(k))

FEATURES={
 'score':lambda e:val(e,'score'),
 'win_rate':lambda e:val(e,'win_rate'),
 'top2_rate':lambda e:val(e,'top2_rate'),
 'top3_rate':lambda e:val(e,'top3_rate'),
 'q2_only':lambda e:max(0,val(e,'top2_rate')-val(e,'win_rate')),
 'q3_only':lambda e:max(0,val(e,'top3_rate')-val(e,'top2_rate')),
 'first_count':lambda e:val(e,'first_count'),
 'second_count':lambda e:val(e,'second_count'),
 'third_count':lambda e:val(e,'third_count'),
 'B_count':lambda e:val(e,'b_count'),
 'S_count':lambda e:val(e,'s_count'),
 'nige_count':lambda e:val(e,'nige_count'),
 'makuri_count':lambda e:val(e,'makuri_count'),
 'sashi_count':lambda e:val(e,'sashi_count'),
 'mark_count':lambda e:val(e,'mark_count'),
 'self_move':lambda e:val(e,'nige_count')+val(e,'makuri_count'),
 'follow_move':lambda e:val(e,'sashi_count')+val(e,'mark_count'),
}

def main():
    races={};E=defaultdict(list);R=defaultdict(list)
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            for x in read(z,'races.csv'):races.setdefault(x['race_id'],x)
            for x in read(z,'entries.csv'):E[x['race_id']].append(x)
            for x in read(z,'results.csv'):R[x['race_id']].append(x)
    out={sp:{p:{f:[0,0,0,0,0,0] for f in FEATURES} for p in (1,2,3)} for sp in ('train','test')}
    n={sp:0 for sp in ('train','test')}
    for race in races.values():
        if '準決勝' not in (race.get('race_type') or ''):continue
        rid=race['race_id'];es=E[rid];rs=R[rid]
        top={ii(x.get('finish_position')):ii(x.get('car_no')) for x in rs if ii(x.get('finish_position')) in (1,2,3)}
        if len(top)!=3 or len(es)<7:continue
        y=ii((race.get('race_date') or '')[:4]);sp='train' if y and y<=2024 else 'test';n[sp]+=1
        maps={f:rank(es,fn) for f,fn in FEATURES.items()}
        for p in (1,2,3):
            c=top[p]
            for f,rm in maps.items():
                r=rm.get(c,99); arr=out[sp][p][f]
                arr[0]+=int(r<=1);arr[1]+=int(r<=2);arr[2]+=int(r<=3);arr[3]+=int(r<=4);arr[4]+=int(r<=5);arr[5]+=r
    md=['# G3準決勝 1-3着 特徴量スクリーニング','', '各特徴をレース内順位化し、実際の各着順選手が上位何人に含まれたかを比較。ライン情報・予想印は不使用。','']
    for sp in ('train','test'):
        md.append(f'## {sp}: {n[sp]}R')
        for p in (1,2,3):
            md.append(f'### {p}着')
            rows=[]
            for f,a in out[sp][p].items():
                rows.append((a[2]/n[sp],a[1]/n[sp],a[0]/n[sp],a[4]/n[sp],a[5]/n[sp],f))
            for t3,t2,t1,t5,mean,f in sorted(rows,reverse=True):
                md.append(f'- {f}: top1 {t1:.1%} / top2 {t2:.1%} / top3 {t3:.1%} / top5 {t5:.1%} / mean_rank {mean:.2f}')
            md.append('')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'semifinal_feature_screen.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('\n'.join(md))
if __name__=='__main__':main()
