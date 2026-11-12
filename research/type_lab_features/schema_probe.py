#!/usr/bin/env python3
"""Print ONLY headers and archive counts for full historical rider/settlement audit."""
from __future__ import annotations
import csv, gzip, io, json, zipfile
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/"data";out=ROOT/"research/type_lab_features/schema_inventory.json"
names=["races.csv","entries.csv","results.csv","payouts.csv"]
fields=defaultdict(lambda:defaultdict(Counter));counts=Counter();samplepaths=defaultdict(list)
for p in sorted(DATA.rglob("*.csv")):
    if p.name not in names:continue
    kind=p.name;family=p.relative_to(DATA).parts[0]
    with p.open(encoding="utf-8-sig",newline="") as fp:
        reader=csv.DictReader(fp); f=tuple(reader.fieldnames or [])
    fields[kind][family][f]+=1;counts["csv_"+kind]+=1
    if len(samplepaths[kind])<12:samplepaths[kind].append(str(p.relative_to(ROOT)))
for zippath in sorted((DATA/"grade_races").rglob("*.zip")):
    counts["zip_archives"]+=1
    with zipfile.ZipFile(zippath) as z:
        for name in z.namelist():
            kind=name.rsplit("/",1)[-1]
            if kind not in names:continue
            with z.open(name) as f:
                reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig",newline=""))
                hdr=tuple(reader.fieldnames or [])
            fields[kind]["grade_zip"][hdr]+=1;counts["zip_"+kind]+=1
            if len(samplepaths[kind])<12:samplepaths[kind].append(str(zippath.relative_to(ROOT))+":"+name)
obj={"file_counts":dict(counts),
    "schemas":{kind:{family:[{"fields":list(hdr),"source_files":count} for hdr,count in counter.most_common()]
                     for family,counter in families.items()}
               for kind,families in fields.items()},
    "source_samples":dict(samplepaths)}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
print("TYPE_LAB_SCHEMAS",json.dumps(obj["schemas"],ensure_ascii=False))
print("TYPE_LAB_SOURCE_COUNTS",json.dumps(obj["file_counts"],ensure_ascii=False))
