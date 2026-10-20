"""Final 2023 line-aware AI retraining.

The LINE RF method/hyperparameters were selected in the paired 2023 Jan-Oct
train / Nov-Dec test experiment. They are frozen here. This pass refits the
same three conditional race-outcome stages on ALL model-ready 2023 races with
published line formations, without re-using test outcomes for model selection.
Output is a reproducible three-model bundle for genuinely later evaluation.
"""
from __future__ import annotations
import hashlib
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from compare_lines_2023 import (
    FEATURES, SEED, TRAIN_END, YEAR_END,
    load, matrix, classifier, feature, normalise
)

OUT=Path('docs/paper-only-ai-2023/line_full_2023')
ARTIFACT=OUT/'line_aware_rf_2023.tar.gz'
LABELS={'stage_1':'winner','stage_2':'second_given_first',
        'stage_3':'third_given_first_and_second'}
COMPARISON_REPORT='docs/paper-only-ai-2023/line_ab_2023/report.json'
SELECTION='RF with_line selected from the strict 2023 November-December paired holdout'
PAPER_BOUNDARY='The ordered-finish conditional extension and line structure features are not specified in Walsh & Joshi; paper-inspired calibration requires independent testing.'

def integrity(race,models):
    cars=sorted(race['cars'],key=int)
    cards=race['cars']
    p1=normalise(models[1].predict_proba(
        np.vstack([feature(race,a,use_line=True) for a in cars]))[:,1],
        [0]*len(cars))
    trifecta={}
    for ia,a in enumerate(cars):
        seconds=[b for b in cars if b!=a]
        p2=normalise(models[2].predict_proba(np.vstack([
            feature(race,b,a,use_line=True) for b in seconds]))[:,1],
            [0]*len(seconds))
        for ib,b in enumerate(seconds):
            thirds=[c for c in seconds if c!=b]
            p3=normalise(models[3].predict_proba(np.vstack([
                feature(race,c,a,b,use_line=True) for c in thirds]))[:,1],
                [0]*len(thirds))
            for ic,c in enumerate(thirds):
                trifecta[(a,b,c)]=float(p1[ia]*p2[ib]*p3[ic])
    if len(trifecta)!=len(cars)*(len(cars)-1)*(len(cars)-2):
        raise RuntimeError('Not all trifecta combinations enumerated')
    total=math.fsum(trifecta.values())
    if abs(total-1.0)>1e-8:
        raise RuntimeError('Invalid trifecta sum '+str(total))
    trio={}
    for (a,b,c),p in trifecta.items():
        ticket=tuple(sorted((a,b,c),key=int))
        trio[ticket]=trio.get(ticket,0.0)+p
    if len(trio)!=math.comb(len(cars),3) or abs(math.fsum(trio.values())-1.0)>1e-8:
        raise RuntimeError('Invalid trio distribution')
    return {'participants':len(cars),'trifecta_tickets':len(trifecta),
            'trio_tickets':len(trio),
            'trifecta_sum':total,'trio_sum':math.fsum(trio.values())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    (OUT/'progress.json').write_text(json.dumps({
        'state':'RUNNING','phase':'2023_read','started_at':started,
        'period':['2023-01-01','2023-12-31'],
        'model_family':'RF with published line formations'},ensure_ascii=False,indent=2)+'\n')
    jan_oct,nov_dec,audit=load()
    if set(jan_oct).intersection(nov_dec):
        raise RuntimeError('Same race found in both original partitions')
    combined={**jan_oct,**nov_dec}
    combined=dict(sorted(combined.items(),key=lambda pair:(pair[1]['date'],pair[0])))
    n=len(combined)
    if n<10000:
        raise RuntimeError('Suspiciously small 2023 training set: '+str(n))
    if n!=audit['races_with_valid_line_and_all_card_features']:
        raise RuntimeError('Race counts do not match prior paired extraction')
    if any(not '2023-01-01'<=v['date']<='2023-12-31' for v in combined.values()):
        raise RuntimeError('Non-2023 date contamination')
    outcomes=len({tuple(v['order']) for v in combined.values()})
    class_counts={}
    month_counts={}
    for r in combined.values():
        class_counts[r['source']]=class_counts.get(r['source'],0)+1
        m=r['date'][:7]
        month_counts[m]=month_counts.get(m,0)+1
    if len(month_counts)!=12:
        raise RuntimeError('Missing month from full-year training')
    (OUT/'progress.json').write_text(json.dumps({
        'state':'RUNNING','phase':'RF_three_stages','started_at':started,
        'eligible_2023_races':n,'completed_stages':0},ensure_ascii=False,indent=2)+'\n')
    models={}
    stages={}
    for stage in (1,2,3):
        X,y,group=matrix(combined,stage,True)
        if len(group)!=len(y) or int(y.sum())!=n:
            raise RuntimeError('Not exactly one winning candidate per conditional group')
        model=classifier('RF')
        print('FIT_START',stage,len(y),X.shape[1],flush=True)
        model.fit(X,y)
        models[stage]=model
        path=OUT/f'stage_{stage}.joblib'
        joblib.dump(model,path,compress=3)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        stages[str(stage)]={
            'target':LABELS[f'stage_{stage}'],
            'candidate_rows':int(len(y)),
            'race_groups':n,'positive_targets':int(y.sum()),
            'features_per_candidate':int(X.shape[1]),
            'output_path':str(path),'file_bytes':path.stat().st_size,
            'sha256':digest}
        del X,y,group
        print('FIT_DONE',stage,digest,flush=True)
        (OUT/'progress.json').write_text(json.dumps({
            'state':'RUNNING','phase':'RF_three_stages','started_at':started,
            'eligible_2023_races':n,'completed_stages':stage},ensure_ascii=False,indent=2)+'\n')
    samples={}
    for rid,race in list(combined.items())[:25]:
        samples[rid]=integrity(race,models)
    finished=datetime.now(timezone.utc).isoformat()
    manifest={
        'schema':'line_aware_full_year_RF_2023_v1',
        'model_family':'RandomForestClassifier',
        'selection':SELECTION,
        'selected_2023_AB_report':COMPARISON_REPORT,
        'training_date_range':['2023-01-01','2023-12-31'],
        'trained_races':n,
        'line_source':'historical races.csv published KDreams predicted_line_formation',
        'fit_only_2023':True,
        'first_place_2023_train':True,
        'stage2_2023_train':True,
        'stage3_2023_train':True,
        'line_features_only_difference_from_baseline_ab':True,
        'features_base':FEATURES,
        'seed':SEED,
        'algorithm_hyperparameters':{
            'n_estimators':90,'max_depth':10,
            'min_samples_leaf':12,'max_features':0.7,
            'n_jobs':2,'random_state':SEED},
        'stages':stages,
        'training_races_by_class':class_counts,
        'training_races_by_month':month_counts,
        'excluded_source_data_audit':audit,
        'trifecta_and_trio_probability_normalization_sample':samples,
        'selection_test_now_included_in_fit':'2023 November-December, now part of 2023 training set',
        'new_out_of_time_assessment_required':'2024 onward; never claim the 2023 A/B test remains independent of the full-year refit',
        'market_odds_used_for_training':False,
        'profit_tested':False,
        'limitations':[
            PAPER_BOUNDARY,
            'Only 2023 races with published complete line and matching numeric cards and top-three finish used; missing line information is NOT inferred.',
            'G1/G2/G3 ZIP archives are not additional sources in this exact paired-comparison pipeline.',
            'A3 January-February lacks line information and March-April source is missing.',
            'No 2024 or later labels used in training or model selection.']
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with tarfile.open(ARTIFACT,'w:gz') as t:
        for filename in ['manifest.json','stage_1.joblib','stage_2.joblib','stage_3.joblib']:
            t.add(OUT/filename,arcname=filename)
    report={
        'state':'TRAINED_LINE_RF_FULL_YEAR_2023',
        'started_at':started,'finished_at':finished,
        'training_races':n,
        'training_range':['2023-01-01','2023-12-31'],
        'stage_results':stages,
        'selection_basis':SELECTION,
        'training_counts_by_class':class_counts,
        'training_counts_by_month':month_counts,
        'manifest_path':str(OUT/'manifest.json'),
        'bundle_path':str(ARTIFACT),
        'bundle_bytes':ARTIFACT.stat().st_size,
        'bundle_sha256':hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        'fit_2024_calls':0,
        '2024_unseen_validation':'not_run',
        'strict_probability_calibration':'not_evaluated_on_new_out_of_time_data',
        'sanity_check_races':len(samples),
        'line_data_races_audit':audit,
        'limitations':manifest['limitations']
    }
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps({
        'state':report['state'],'finished_at':finished,
        'trained_races':n,'stages_trained':3,
        'bundle_sha256':report['bundle_sha256']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FULL_YEAR_TRAINED',n,'BUNDLE',report['bundle_sha256'],flush=True)

if __name__=='__main__':
    main()
