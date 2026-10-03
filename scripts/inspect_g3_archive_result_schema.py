from __future__ import annotations
import csv, io, zipfile
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

files=Counter(); schemas=defaultdict(Counter); samples={}
for p in sorted(ARCH.rglob('*.zip')):
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            b=Path(n).name
            if not b.endswith('.csv'): continue
            files[b]+=1
            try:
                txt=dec(z.read(n)); r=csv.reader(io.StringIO(txt)); hdr=next(r,[]); row=next(r,[])
            except Exception:
                continue
            schemas[b][tuple(hdr)]+=1
            samples.setdefault(b,row)
OUT.mkdir(parents=True,exist_ok=True)
md=['# G3 archive CSV schema','']
for b,c in files.most_common():
    md.append(f'## {b} ({c} zip)')
    for hdr,n in schemas[b].most_common(3):
        md.append(f'- schema x{n}: {list(hdr)}')
    md.append(f'- sample: {samples.get(b)}')
    md.append('')
(OUT/'archive_result_schema.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print('\n'.join(md))
