import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'data'/'2023'/'s_class_yosen'/'payouts.csv'
with p.open('r',encoding='utf-8-sig',newline='') as f:
    r=csv.DictReader(f); rows=[]
    for row in r:
        if any('2車複' in str(v) for v in row.values()):
            rows.append(row)
            if len(rows)>=5: break
out={'fieldnames':r.fieldnames,'sample_quinella_rows':rows}
q=ROOT/'data'/'audits'/'payout_schema_probe_2023.json'; q.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
