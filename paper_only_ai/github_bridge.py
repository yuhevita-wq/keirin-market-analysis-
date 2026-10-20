"""Paper-only AI: source bridge. Data provenance only, no knowledge, no model training.
Only the independent 2023+ data streams are accessed. Source reports are not training features.
"""
import argparse, csv, gzip, hashlib, io, json
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
    race_types = Counter()
    with gzip.open(io.BytesIO(data),'rt',encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);count+=1
            if not isinstance(row,dict) or not row.get('race_id'):
                raise ValueError('invalid record: missing race_id')
            key=str(row['race_id'])
            fields.update(row)
            if 'race_type' in row and row['race_type']:
                race_types[str(row['race_type'])] += 1
            if key in unique:duplicates+=1
            unique.add(key)
    return {'records':count,'unique_races':len(unique),'duplicates':duplicates,'fields':sorted(fields),'race_types':dict(race_types)},unique

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


def read_race_manifests(paths, year):
    ids = set()
    by_month = Counter()
    duplicates = 0
    file_counts = []
    for path in paths:
        raw = get(path).decode('utf-8-sig')
        rows = csv.DictReader(io.StringIO(raw))
        if not rows.fieldnames or 'race_id' not in rows.fieldnames or 'race_date' not in rows.fieldnames:
            raise ValueError('Missing race_id/race_date in '+path)
        n = 0
        for row in rows:
            race_id = str(row['race_id']).strip()
            date = str(row['race_date']).strip()
            if not race_id or not date.startswith(str(year)+'-'):
                raise ValueError('Missing or out-of-year race id/date in '+path)
            try:
                date_obj = datetime.strptime(date,'%Y-%m-%d')
            except ValueError:
                raise ValueError('Invalid date in '+path)
            n += 1
            if race_id in ids:
                duplicates += 1
                continue
            ids.add(race_id)
            by_month[date_obj.strftime('%Y-%m')] += 1
        file_counts.append({'path':path,'rows':n})
    return {'unique_races':len(ids),'duplicate_rows':duplicates,'files':file_counts,
            'monthly':[{'month':f'{year}-{m:02d}','races':by_month[f'{year}-{m:02d}']}
                       for m in range(1,13)]}, ids

def load_2023_all_a_inventory(result_ids, odds_ids):
    year=2023
    intervals=['01_02','03_04','05_06','07_08','09_10','11_12']
    a12_paths=[f'data/a_class_12/2023_{interval}/races.csv' for interval in intervals]
    # 2023_03_04 is absent in the A3 archive; report the missing interval explicitly.
    a3_intervals=['01_02','05_06','07_08','09_10','11_12']
    a3_paths=[f'data/a3_challenge/2023_{interval}/races.csv' for interval in a3_intervals]
    a12,a12_ids=read_race_manifests(a12_paths,year)
    a3,a3_ids=read_race_manifests(a3_paths,year)
    overlap=a12_ids&a3_ids
    if overlap:
        raise ValueError('A1/A2 and A3 manifests contain overlapping race IDs')
    identified=a12_ids|a3_ids
    matched=identified & result_ids
    return {
        'scope':'Race-entry manifests, not odds-only records; distinct race IDs',
        'a1_a2':a12,
        'a3':a3,
        'a3_missing_intervals':['2023-03-01/2023-04-30'],
        'identified_a_races':len(identified),
        'identified_a_with_results':len(matched),
        'identified_a_with_odds':len(identified&odds_ids),
        'identified_a_missing_results':len(identified-result_ids),
        'results_not_in_identified_a':len(result_ids-matched),
        'result_universe_a_majority_confirmed':len(matched)>len(result_ids-matched),
        'notes':'Source inventory only. No race type or result label is used as a model input.'
    }

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
    all_a = load_2023_all_a_inventory(rid, oid) if year == 2023 else None
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
        'a_class_inventory':all_a,
        'matched_races':len(oid&rid),
        'unmatched_odds_races':len(oid-rid),
        'unmatched_result_races':len(rid-oid),
        'odds_fields':odds['fields'],
        'odds_race_types':odds['race_types'],
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
