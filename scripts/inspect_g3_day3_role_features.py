from __future__ import annotations
import csv, io, zipfile
from pathlib import Path
from collections import Counter

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

def main():
    headers=Counter(); nonempty=Counter(); samples={}
    total=0
    for pth in sorted(ARCH.rglob('*.zip')):
        with zipfile.ZipFile(pth) as z:
            rows=read(z,'entries.csv')
            for r in rows:
                rid=r.get('race_id','')
                if not (len(rid)==16 and rid.isdigit() and rid[10:12]=='03'): continue
                total+=1
                for k,v in r.items():
                    headers[k]+=1
                    if str(v or '').strip():
                        nonempty[k]+=1
                        samples.setdefault(k,str(v))
    md=['# G3三日目 entries 個体特徴量監査','',f'- rows: {total}','', '## columns']
    for k in headers:
        md.append(f"- {k}: nonempty {nonempty[k]}/{headers[k]} ({nonempty[k]/headers[k]:.1%}) sample={samples.get(k,'')}")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'role_feature_schema.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
