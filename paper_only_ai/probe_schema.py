"""Inspect only the shapes of GitHub 2023 source data, without fitting or inventing features."""
from __future__ import annotations
import csv, gzip, io, json
from pathlib import Path
from github_bridge import get

def first_csv(path):
    reader=csv.DictReader(io.StringIO(get(path).decode('utf-8-sig')))
    rows=[]
    for _,r in zip(range(3),reader):
        rows.append({k:str(v)[:200] for k,v in r.items()})
    return {'path':path,'columns':reader.fieldnames,'rows':rows}

def first_gzip(path):
    data=get(path)
    with gzip.open(io.BytesIO(data),'rt',encoding='utf-8') as fp:
        return {'path':path,'rows':[json.loads(next(fp)) for _ in range(3)]}

if __name__ == '__main__':
    files=[
        'data/a_class_12/2023_01_02/entries.csv',
        'data/a_class_12/2023_01_02/results.csv',
        'data/a3_challenge/2023_01_02/entries.csv',
    ]
    probe={'csv':[first_csv(x) for x in files],
           'finish':[first_gzip('data/result_universe_v2/combined/results_2023.jsonl.gz')]}
    out=Path('docs/paper-only-ai-2023/schema_probe.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(probe,ensure_ascii=False,indent=2)+'\n')
    print('Inspection published',out,[(r['path'],r['columns']) for r in probe['csv']])
