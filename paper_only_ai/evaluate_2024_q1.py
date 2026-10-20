"""Pure out-of-time evaluation: 2023 frozen joblib -> 2024 Q1 data.
No fit/partial_fit, no model selection, no odds, no 2024 labels in predictors.
"""
from __future__ import annotations
import csv,gzip,io,json,os,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from train_2023 import FEATURES, fetch, ece

START='2024-01-01'
END='2024-03-31'
OUTPUT=Path('docs/paper-only-ai-2023/validation_2024_q1')
ARTIFACT=Path(os.environ.get('FROZEN_MODEL_DIR','frozen_2023'))
SOURCE_RUN_ID=39354641293
def sources():
    return [
      'data/a_class_12/2024_01_02/entries.csv',
      'data/a_class_12/2024_03_04/entries.csv',
      'data/a3_challenge/2024_01_02/entries.csv',
      'data/a3_challenge/2024_03_04/entries.csv',
      'data/2024/s_class_f1_all_parts/2024_q1/entries.csv',
      'data/2024/s_class_yosen/entries.csv',
    ]
def dataset():
    labels={}
    result_path='data/result_universe_v2/combined/results_2024.jsonl.gz'
    with gzip.open(io.BytesIO(fetch(result_path)),'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            order=row.get('finish_order')
            if isinstance(order,list) and order and str(order[0]).isdigit():
                labels[str(row['race_id'])]=str(order[0])
    records={}
    audit={'files':{},'duplicated_entry_rows':0,'numeric_missing':0,
           'race_labels_absent':0,'race_with_missing_winner':0,
           'excluded_other_dates':0,'pre_race_feature_certified':False}
    for path in sources():
        raw=fetch(path).decode('utf-8-sig');count=0
        for row in csv.DictReader(io.StringIO(raw)):
            date=str(row.get('race_date',''))
            if not START<=date<=END:
                audit['excluded_other_dates']+=1
                continue
            race=str(row.get('race_id','')).strip()
            if race not in labels:
                audit['race_labels_absent']+=1
                continue
            car=str(row.get('car_no','')).strip()
            key=(race,car)
            if key in records:
                audit['duplicated_entry_rows']+=1
                continue
            try:
                vals={k:float(row[k]) for k in FEATURES}
                if not all(np.isfinite(x) for x in vals.values()):
                    raise ValueError('non-finite')
            except (KeyError,ValueError,TypeError):
                audit['numeric_missing']+=1
                continue
            count+=1
            records[key]={**vals,'race_id':race,'race_date':date,
                          'car_no':int(car),'label':int(car==labels[race]),
                          'source_category':('A1_A2' if '/a_class_12/' in path
                          else 'A3' if '/a3_challenge/' in path else 'S')}
        audit['files'][path]=count
        print('READ',path,count,flush=True)
    df=pd.DataFrame(records.values())
    if df.empty:raise RuntimeError('No 2024 Q1 labelled entry records')
    group=df.groupby('race_id')
    valid=group.label.sum()
    invalid=valid[valid!=1].index
    audit['race_with_missing_winner']=int(len(invalid))
    df=df[~df.race_id.isin(invalid)].copy().sort_values(['race_date','race_id','car_no'])
    if len(df)<100 or df['label'].sum()<10:raise RuntimeError('Too few valid evaluation records')
    if not df.race_date.between(START,END).all():raise RuntimeError('Date leakage')
    if df.groupby('race_id').label.sum().ne(1).any():raise RuntimeError('Not one winner/race')
    audit['evaluation_races']=int(df.race_id.nunique())
    audit['evaluation_rider_rows']=len(df)
    audit['winners']=int(df.label.sum())
    audit['outcome_prevalence']=float(df.label.mean())
    audit['periods']=df.groupby(df.race_date.str[:7]).agg(races=('race_id','nunique'),rider_rows=('label','size')).reset_index().rename(columns={'race_date':'month'}).to_dict(orient='records')
    audit['categories']=df.groupby('source_category').agg(races=('race_id','nunique'),rows=('label','size')).reset_index().to_dict(orient='records')
    return df.reset_index(drop=True),audit

def scored(df, probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(len(df),2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise RuntimeError('Invalid class probability matrix')
    y=df.label.to_numpy(dtype=int)
    if not set(y)=={0,1}:raise RuntimeError('Both classes required')
    strict,coverage=ece(y,p,True)
    auc=roc_auc_score(y,p[:,1])
    selected=df[['race_id','label','car_no']].copy()
    selected['prob']=p[:,1]
    selected=selected.sort_values(['race_id','prob','car_no'],ascending=[True,False,True]).drop_duplicates('race_id')
    expected_random=float(df.groupby('race_id').size().rdiv(1).mean())
    # Top-1 hit is a per-race metric, unlike the imbalanced rider-level accuracy.
    return {'rider_binary_accuracy':float(accuracy_score(y,p[:,1]>=.5)),
            'all_negative_baseline_accuracy':float((y==0).mean()),
            'brier':float(brier_score_loss(y,p[:,1])),
            'base_rate_brier':float(brier_score_loss(y,np.repeat(y.mean(),len(y)))),
            'log_loss':float(log_loss(y,p[:,1],labels=[0,1])),
            'roc_auc':float(auc),
            'ece_20bin_classwise':float(strict),
            'ece_20bin_unconstrained':float(ece(y,p,False)[0]),
            'ece_occupied_bins':coverage,
            'ece_gate_passed':min(coverage)>=16,
            'top1_race_win_rate':float(selected['label'].mean()),
            'top1_random_rider_reference':expected_random,
            'top1_races':int(len(selected)),
            'winner_mean_predicted_probability':float(df.loc[df.label==1].assign(prob=p[df.label.to_numpy()==1,1])['prob'].mean()),
            'rider_rows':int(len(df))}
def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    (OUTPUT/'progress.json').write_text(json.dumps({'state':'STARTED','at':started,'model_source_run':SOURCE_RUN_ID},ensure_ascii=False))
    df,audit=dataset()
    branches={}
    for branch in ('calibration','accuracy'):
        model_path=ARTIFACT/(branch+'.joblib')
        if not model_path.is_file():raise RuntimeError('Frozen model artifact absent: '+str(model_path))
        model=joblib.load(model_path)
        print('MODEL_LOADED',branch,str(model_path),flush=True)
        columns=list(model.feature_names_in_) if hasattr(model,'feature_names_in_') else None
        # sklearn Pipeline exposes the fitted feature names through first step.
        if columns is None:
            first=model.steps[0][1]
            columns=list(first.feature_names_in_)
        if not columns or not set(columns).issubset(FEATURES):
            raise RuntimeError('Unexpected model features')
        print('FROZEN_FEATURES',branch,columns,flush=True)
        probs=model.predict_proba(df[columns])
        if list(model.classes_)!=[0,1]:
            raise RuntimeError('Unexpected class order')
        branch_rows={}
        branch_rows['overall']=scored(df,probs)
        branch_rows['months']={}
        for month in ['2024-01','2024-02','2024-03']:
            selected=np.asarray(df.race_date.str.startswith(month))
            if not selected.any():raise RuntimeError('Missing month '+month)
            branch_rows['months'][month]=scored(df.loc[selected],probs[selected])
        branch_rows['categories']={}
        for cat in ['A1_A2','A3','S']:
            selected=np.asarray(df.source_category==cat)
            if selected.any():
                branch_rows['categories'][cat]=scored(df.loc[selected],probs[selected])
        branch_rows['features']=columns
        branches[branch]=branch_rows
        print('OUT_OF_TIME_SCORES',branch,json.dumps(branch_rows['overall']),flush=True)
    report={'state':'VALIDATED_2024_Q1','prediction_period':[START,END],
            'training_data_end':'2023-12-31',
            'model_source_run':SOURCE_RUN_ID,
            'models_changed':False,'fit_calls_2024':0,'feature_selection_2024':False,
            'source_result':'data/result_universe_v2/combined/results_2024.jsonl.gz',
            'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
            'dataset':audit,'branches':branches,
            'betting_roi':None,'reason_roi_unavailable':'No single-rider win bet/decimal odds match in keirin; simulated ROI unsupported',
            'limitations':['Single rider one-vs-rest classification; no market-odds testing',
                           '2023 model had been fitted on retrospective snapshots whose race-time availability is not certified',
                           'A3 March-Apr 2023 missing; 2023 S-class source incomplete',
                           '2023 model final fit capped at 80000 sampled rows',
                           'Binary classification accuracy is strongly influenced by nonwinner prevalence']}
    (OUTPUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'progress.json').write_text(json.dumps({'state':report['state'],'at':report['finished_at'],
       'model_source_run':SOURCE_RUN_ID,'races':audit['evaluation_races'],'rows':audit['evaluation_rider_rows']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINISHED',report['state'],flush=True)
if __name__=='__main__':
    main()
