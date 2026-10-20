"""Audit 2023 source-vs-learner exact race-ID coverage. No model fitting."""
import csv, gzip, io, json, zipfile
from pathlib import Path
from datetime import datetime,timezone
from train_conditional_2023 import collect
from train_2023 import fetch, sources

OUT=Path('docs/paper-only-ai-2023/coverage_audit_2023.json')
ARCHIVE='data/result_universe_v2/results_2023.jsonl.gz'
COMBINED='data/result_universe_v2/combined/results_2023.jsonl.gz'
ZIPS=['data/grade_races/g1g2/g1-g2-2023-non-girls.zip']+[
  f'data/grade_races/g3/2023/g3-all-2023-{q}-non-girls.zip' for q in ('q1','q2','q3','q4')]
def result_ids(path):
    with gzip.open(io.BytesIO(fetch(path)),'rt',encoding='utf-8') as f:
        return {str(a['race_id']) for a in map(json.loads,f) if a.get('race_id')}
def get_csv_ids(data):
    try:
        data=data.decode('utf-8-sig')
    except UnicodeDecodeError:
        data=data.decode('cp932',errors='replace')
    return {str(r.get('race_id','')).strip() for r in csv.DictReader(io.StringIO(data))
            if str(r.get('race_id','')).strip()}
def main():
    old=result_ids(ARCHIVE);all_results=result_ids(COMBINED)
    events,learner_info=collect()
    trained=set(events)
    manifests=set()
    for entry in sources():
        name=entry.replace('/entries.csv','/races.csv')
        try:
            manifests|=get_csv_ids(fetch(name))
        except Exception as exc:
            print('MANIFEST_ERROR',name,str(exc),flush=True)
    archive_ids=set()
    zips=[]
    for path in ZIPS:
        raw=fetch(path)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            files=z.namelist()
            found=set()
            processed=[]
            for name in files:
                if name.lower().endswith(('/races.csv','/entries.csv')) or name.lower() in ('races.csv','entries.csv'):
                    try:
                        ids=get_csv_ids(z.read(name));found.update(ids)
                        processed.append({'path':name,'races':len(ids)})
                    except Exception as e:
                        processed.append({'path':name,'error':str(e)})
            archive_ids|=found
            zips.append({'zip':path,'zip_bytes':len(raw),'members':len(files),
                         'names_sample':files[:10],'race_ids':len(found),
                         'processed_csv':processed[:25]})
        print('ZIP_PROBED',path,len(found),flush=True)
    report={
      'audited_at':datetime.now(timezone.utc).isoformat(),
      'year':2023,'old_verified_result_source':ARCHIVE,
      'merged_verified_result_source':COMBINED,
      'old_verified_result_races':len(old),
      'merged_verified_result_races':len(all_results),
      'old_subset_of_merged':old<=all_results,
      'new_results_since_older_archive':len(all_results-old),
      'training_model_races':len(trained),
      'training_races_in_old_results':len(trained&old),
      'training_races_outside_old_results':len(trained-old),
      'old_results_not_used_in_training':len(old-trained),
      'old_results_not_used_sample':sorted(old-trained)[:25],
      'merged_results_not_used_in_training':len(all_results-trained),
      'merged_results_not_used_sample':sorted(all_results-trained)[:25],
      'all_studied_entry_manifest_races':len(manifests),
      'old_result_races_without_studied_manifest':len(old-manifests),
      'old_result_races_with_studied_manifest_not_train':len((old&manifests)-trained),
      'old_result_races_without_studied_manifest_sample':sorted(old-manifests)[:25],
      'g1g2_g3_zip_races':len(archive_ids),
      'g1g2_g3_zip_races_not_trained':len(archive_ids-trained),
      'old_not_trained_but_g1g2_g3_zip':len((old-trained)&archive_ids),
      'merged_not_trained_but_g1g2_g3_zip':len((all_results-trained)&archive_ids),
      'new_merged_result_races_in_g1g2_g3_zip_not_trained':len(((all_results-old)-trained)&archive_ids),
      'zips':zips,
      'warnings':['A race result does not guarantee a valid pre-race feature row exists',
                  'ZIP manifest rows alone do not prove model-ready numeric features',
                  'ZIP source records may duplicate existing entry sources',
                  'The difference in dataset totals is not equal to the exact missing-ID count'],
      'learner_existing_collection_info':{
         'duplicate_entry_rows':learner_info['duplicate_entry_rows'],
         'incomplete_card_rows':learner_info['incomplete_card_rows'],
         'races_missing_podium_rider':learner_info['races_missing_podium_rider']}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('AUDIT',json.dumps({k:v for k,v in report.items() if isinstance(v,(bool,int))},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
