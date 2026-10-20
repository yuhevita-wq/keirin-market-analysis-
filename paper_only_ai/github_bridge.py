"""Paper-only AI: source bridge. Data provenance only, no knowledge, no model training.
Only the independent 2023+ data streams are accessed. Source reports are not training features.
"""
import argparse, gzip, hashlib, json, tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from urllib.request import urlopen, Request

REPO = 'yuhevita-wq/keirin-market-analysis-'
ROOT = f'https://raw.githubusercontent.com/{REPO}/main/'
START = '2023-01-01'
ODDS_AUDIT = 'data/audits/odds_only_export.json'
def get(path):
    if '..' in path or not path.startswith('data/'):
        raise ValueError('Invalid source path')
    with urlopen(Request(ROOT+path,headers={'User-Agent':'paper-only-ai-source-bridge'}),timeout=120) as response:
        return response.read()
def survey_gzip(data):
    import io
    unique=set()
    count=0
    fields=set()
    duplicates=0
    with gzip.open(io.BytesIO(data),'rt',encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);count+=1
            if not isinstance(row,dict) or not row.get('race_id'):
                raise ValueError('invalid record: missing race_id')
            key=str(row['race_id'])
            fields.update(row)
            if key in unique:duplicates+=1
            unique.add(key)
    return {'records':count,'unique_races':len(unique),'duplicates':duplicates,'fields':sorted(fields)},unique

def month_from_id(race_id, year):
    # Mechanical date decode for the stored 16-digit archive ID;
    # unrecognised IDs are excluded from monthly plots, never guessed.
    ident = str(race_id)
    if len(ident) != 16 or not ident.isdigit():
        return None
    candidate = ident[2:10]
    try:
        date = datetime.strptime(candidate, '%Y%m%d')
    except ValueError:
        return None
    return date.strftime('%Y-%m') if date.year == year else None

def connect(year):
    if year < 2023 or year > 2026:
        raise ValueError('Only 2023 onward through archived 2026 supported')
    audit=json.loads(get(ODDS_AUDIT))
    info=audit['years'][str(year)]
    odds_path=f'data/odds_only/odds_{year}.jsonl.gz'
    results_path=f'data/result_universe_v2/combined/results_{year}.jsonl.gz'
    odds_data=get(odds_path)
    digest=hashlib.sha256(odds_data).hexdigest()
    if digest!=info['sha256']:
        raise ValueError('SHA-256 integrity verification failed')
    odds,oid=survey_gzip(odds_data)
    if odds['unique_races']!=info['complete_races']:
        raise ValueError('Audit complete-races does not match archive')
    results,rid=survey_gzip(get(results_path))
    oc = Counter(month_from_id(i, year) for i in oid)
    mc = Counter(month_from_id(i, year) for i in (oid & rid))
    monthly = [{'month': f'{year}-{m:02d}',
                'odds_races': oc[f'{year}-{m:02d}'],
                'matched_races': mc[f'{year}-{m:02d}']}
               for m in range(1, 13)]
    report={
        'source_status':'CONNECTED_AWAITING_TARGET',
        'model_status':'UNTRAINED',
        'start_date':START,
        'year':year,
        'repository':REPO,
        'updated_at':datetime.now(timezone.utc).isoformat(),
        'odds_path':odds_path,
        'results_path':results_path,
        'odds_races':odds['unique_races'],
        'monthly':monthly,
        'undated_odds_races':oc[None],
        'undated_matched_races':mc[None],
        'result_races':results['unique_races'],
        'matched_races':len(oid&rid),
        'unmatched_odds_races':len(oid-rid),
        'unmatched_result_races':len(rid-oid),
        'odds_fields':odds['fields'],
        'result_fields':results['fields'],
        'odds_sha256_verified':True,
        'price_phase':'final',
        'odds_types':audit.get('supported_odds_types',[]),
        'model_input_count':0,
        'evaluated_predictions':0,
        'binary_target':None,
        'note':'Source archives linked and integrity checked. No labels or features chosen; no learning performed. Final odds are not pre-deadline quotes.'
    }
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--year',type=int,default=2023)
    parser.add_argument('--out',default='docs/paper-only-ai-2023/source_status.json')
    args=parser.parse_args()
    report=connect(args.year)
    destination=Path(args.out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('odds_fields','result_fields')},ensure_ascii=False,indent=2))
