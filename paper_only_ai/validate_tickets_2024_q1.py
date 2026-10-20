"""Strict out-of-time ticket probability assessment of frozen 2023 three-stage AI.
2024 Q1 inputs/results are used ONLY for assessment; no model fit or retuning.
Tickets: trifecta, trio, exacta, quinella. 2023 methodology's 20-bin
classwise-ECE coverage gate is reported transparently, not silently relaxed.
"""
from __future__ import annotations
import csv, gzip, io, json, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_2024_q1 import sources, START, END
from train_2023 import FEATURES, fetch, ece
from train_conditional_2023 import conditional_features, group_normalise

OUT = Path('docs/paper-only-ai-2023/validation_tickets_2024_q1')
BUNDLE = Path(os.environ.get('FROZEN_TICKET_MODEL_DIR', 'frozen_conditional_bundle'))
LABELS = 'data/result_universe_v2/combined/results_2024.jsonl.gz'
TYPES = ('3連単', '3連複', '2車単', '2車複')
SPLIT = 75
EPS = 1e-12

def cards_and_results():
    results = {}
    with gzip.open(io.BytesIO(fetch(LABELS)), 'rt', encoding='utf-8') as handle:
        for line in handle:
            obj = json.loads(line)
            order = obj.get('finish_order')
            if isinstance(order,list) and len(order)>=3:
                top = tuple(str(x) for x in order[:3])
                if len(set(top))==3 and all(x.isdigit() and int(x)>0 for x in top):
                    results[str(obj['race_id'])] = top
    all_races = {}
    audit=Counter()
    by_source={}
    for path in sources():
        found=0
        for row in csv.DictReader(io.StringIO(fetch(path).decode('utf-8-sig'))):
            day = str(row.get('race_date','')).strip()
            if not START <= day <= END:
                continue
            rid = str(row.get('race_id','')).strip()
            if rid not in results:
                audit['entry_rows_without_complete_top_three_result']+=1
                continue
            car = str(row.get('car_no','')).strip()
            try:
                vec=np.asarray([float(row[k]) for k in FEATURES],dtype=np.float64)
                if not car.isdigit() or not np.isfinite(vec).all():
                    raise ValueError('missing')
            except (ValueError,TypeError,KeyError):
                audit['entry_rows_with_missing_fields']+=1
                continue
            race=all_races.setdefault(rid,{'date':day,'riders':{},'first3':results[rid],
                                           'source':('A1_A2' if '/a_class_12/' in path else
                                                     'A3' if '/a3_challenge/' in path else 'S')})
            if car in race['riders']:
                audit['duplicate_entry_rows']+=1
                continue
            race['riders'][car]=vec
            found+=1
        by_source[path]=found
        print('SOURCE',path,found,flush=True)
    valid={}
    for rid,r in all_races.items():
        if len(r['riders']) < 4 or any(x not in r['riders'] for x in r['first3']):
            audit['races_excluded_missing_podium_in_card']+=1
        else:
            valid[rid]=r
    valid=dict(sorted(valid.items(),key=lambda x:(x[1]['date'],x[0])))
    if not valid:
        raise RuntimeError('No valid 2024 Q1 entries/results')
    return valid, {**dict(audit),'files':by_source,'race_count':len(valid),
                   'rider_rows':sum(len(x['riders']) for x in valid.values()),
                   'months':dict(Counter(r['date'][:7] for r in valid.values())),
                   'categories':dict(Counter(r['source'] for r in valid.values()))}

def load_models():
    paths={
      'first':BUNDLE/'frozen_2023/calibration.joblib',
      'second':BUNDLE/'docs/paper-only-ai-2023/conditional_2023/stage2.joblib',
      'third':BUNDLE/'docs/paper-only-ai-2023/conditional_2023/stage3.joblib'}
    for kind,path in paths.items():
        if not path.is_file():
            raise RuntimeError('Missing immutable 2023 model '+kind+' at '+str(path))
    return tuple(joblib.load(paths[k]) for k in ('first','second','third'))

def infer_batch(batch, models):
    first, second, third = models
    cols=list(first.steps[0][1].feature_names_in_)
    if not set(cols).issubset(FEATURES):
        raise RuntimeError('Unrecognized first-place frozen features')
    f_indices=[FEATURES.index(name) for name in cols]
    x1=[];g1=[];key1={}
    x2=[];g2=[];key2={}
    x3=[];g3=[];keys3=[]
    for gid,(rid,race) in enumerate(batch):
        cards=race['riders']
        cars=sorted(cards,key=int)
        for a in cars:
            key1[(rid,a)]=len(x1)
            x1.append(cards[a][f_indices])
            g1.append(gid)
        for a in cars:
            g2id=len(set_group2)
            set_group2.append((rid,a))
            for b in cars:
                if a==b:continue
                key2[(rid,a,b)]=len(x2)
                x2.append(conditional_features(cards[b],cards[a]))
                g2.append(g2id)
                g3id=len(set_group3)
                set_group3.append((rid,a,b))
                for c in cars:
                    if c==a or c==b:continue
                    x3.append(conditional_features(cards[c],cards[a],cards[b]))
                    g3.append(g3id)
                    keys3.append((rid,a,b,c))
    p1=group_normalise(first.predict_proba(pd.DataFrame(np.asarray(x1),columns=cols))[:,1],g1)
    p2=group_normalise(second.predict_proba(np.asarray(x2,dtype=np.float32))[:,1],g2)
    p3=group_normalise(third.predict_proba(np.asarray(x3,dtype=np.float32))[:,1],g3)
    output={rid: {'3連単':{},'3連複':defaultdict(float),
                  '2車単':defaultdict(float),'2車複':defaultdict(float)}
            for rid,_ in batch}
    for (rid,a,b,c),p in zip(keys3,p3):
        prob=float(p*p1[key1[(rid,a)]]*p2[key2[(rid,a,b)]])
        dest=output[rid]
        dest['3連単'][(a,b,c)]=prob
        dest['3連複'][tuple(sorted((a,b,c),key=int))]+=prob
        dest['2車単'][(a,b)]+=prob
        dest['2車複'][tuple(sorted((a,b),key=int))]+=prob
    for rid,v in output.items():
        for kind in TYPES:
            total=sum(v[kind].values())
            if abs(total-1)>1e-8:
                raise RuntimeError('Exclusive exhaustive ticket probabilities do not sum to 1 '+rid+' '+kind+' '+str(total))
    return output

# Resettable per-batch grouping arrays, used only for deterministic group IDs.
set_group2=[]
set_group3=[]

def truth_for(race):
    a,b,c=race['first3']
    return {'3連単':(a,b,c),'3連複':tuple(sorted((a,b,c),key=int)),
            '2車単':(a,b),'2車複':tuple(sorted((a,b),key=int))}

def gather_metrics(records):
    if not records: return None
    ps=[];ys=[]
    nll=[];brier=[];top1=[];uniform_nll=[];random_hit=[]
    count=0
    coverage={1:[],3:[],5:[],10:[]}
    for t, d in records:
        keys=sorted(d,key=lambda k:tuple(int(x) for x in k))
        p=np.asarray([d[k] for k in keys],dtype=float)
        y=np.asarray([int(k==t) for k in keys],dtype=np.int8)
        if int(y.sum())!=1 or abs(p.sum()-1)>1e-8 or np.min(p)<0:
            raise RuntimeError('Missing real ticket or probability invalid')
        n=len(p)
        nll.append(float(-np.log(np.clip(p[y==1][0],EPS,1))))
        brier.append(float(np.mean((p-y)**2)))
        best=np.argsort(-p,kind='stable')
        top1.append(int(y[best[0]]))
        random_hit.append(1.0/n)
        uniform_nll.append(float(np.log(n)))
        for k in coverage:
            coverage[k].append(int(np.any(y[best[:min(k,n)]]==1)))
        ps.append(p);ys.append(y)
        count+=1
    flat_p=np.concatenate(ps)
    flat_y=np.concatenate(ys)
    matrix=np.stack([1-flat_p,flat_p],axis=1)
    raw_ece,occupied = ece(flat_y,matrix,False)
    # Paper requires >=16 of 20 bins occupied for EACH class. Sparse ticket
    # distributions naturally fail; preserve failure rather than waive it.
    _, strict_coverage=ece(flat_y,matrix,True)
    diff=np.abs(flat_p-flat_y)
    qs=np.quantile(flat_p,np.linspace(0,1,11))
    quantile_reliability=[]
    # Quantile range bins are supplemental; not a substitute for the paper's gate.
    qbins=np.searchsorted(qs[1:-1],flat_p,side='right')
    for k in range(10):
        mask=qbins==k
        if mask.any():
            quantile_reliability.append({'bin':k,'tickets':int(mask.sum()),
                 'mean_predicted_probability':float(flat_p[mask].mean()),
                 'empirical_hit_rate':float(flat_y[mask].mean())})
    return {'races':count,'ticket_outcomes_evaluated':len(flat_p),
        'top1_hit_rate':float(np.mean(top1)),
        'uniform_random_ticket_top1_reference':float(np.mean(random_hit)),
        'top_k_hit_rate':{str(k):float(np.mean(v)) for k,v in coverage.items()},
        'mean_realized_ticket_probability':float(np.mean([p[np.flatnonzero(y)[0]] for p,y in zip(ps,ys)])),
        'mean_nll_real_ticket':float(np.mean(nll)),
        'uniform_mean_nll_reference':float(np.mean(uniform_nll)),
        'mean_ticket_brier':float(np.mean(brier)),
        'pooled_20bin_classwise_ece_raw':float(raw_ece),
        'pooled_ece_bins_per_class':occupied,
        'paper_20bin_80pct_gate_passed':bool(min(occupied)>=16),
        'quantile_calibration_display_only':quantile_reliability}

def main():
    global set_group2,set_group3
    OUT.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    (OUT/'progress.json').write_text(json.dumps({'state':'RUNNING','started_at':started},ensure_ascii=False)+'\n')
    models=load_models()
    races,audit=cards_and_results()
    full={k:[] for k in TYPES}
    by_month={month:{k:[] for k in TYPES} for month in ('2024-01','2024-02','2024-03')}
    by_class={cat:{k:[] for k in TYPES} for cat in ('A1_A2','A3','S')}
    integrity=Counter()
    all_items=list(races.items())
    for start in range(0,len(all_items),SPLIT):
        chunk=all_items[start:start+SPLIT]
        set_group2=[];set_group3=[]
        result=infer_batch(chunk,models)
        for rid,race in chunk:
            true=truth_for(race)
            monthly=by_month[race['date'][:7]]
            group=by_class[race['source']]
            for k in TYPES:
                row=(true[k],result[rid][k])
                full[k].append(row)
                monthly[k].append(row)
                group[k].append(row)
                integrity[k]+=1
        count=start+len(chunk)
        if count%750< SPLIT or count==len(all_items):
            print('PREDICTED',count,'/',len(all_items),flush=True)
            (OUT/'progress.json').write_text(json.dumps({
                'state':'RUNNING','started_at':started,'processed_races':count,
                'total_races':len(all_items),'2024_refit_count':0},ensure_ascii=False)+'\n')
    report={
        'state':'VALIDATED_TICKET_PROBABILITIES_2024_Q1',
        'scope':{'2023_training_finished':'2023-12-31',
                 '2024_evaluation_start':START,'2024_evaluation_end':END,
                 '2024_train_fit_calls':0,'2024_model_selection_calls':0,
                 '2023_frozen_model_bundle_run':39362270286},
        'started_at':started,
        'finished_at':datetime.now(timezone.utc).isoformat(),
        'data_audit':audit,
        'per_ticket_type':{k:gather_metrics(v) for k,v in full.items()},
        'by_month':{m:{k:gather_metrics(v) for k,v in group.items()} for m,group in by_month.items()},
        'by_class':{c:{k:gather_metrics(v) for k,v in group.items()} for c,group in by_class.items()},
        'probability_normalization_checks':dict(integrity),
        'validation_only':True,
        'closing_odds_joined':False,
        'roi_computed':False,
        'notes':['Ordered and unordered ticket probabilities are from chain-rule construction, not directly specified in Walsh-Joshi.',
                 '2023 original rider winner model was normalized within a race; normalized probabilities need fresh calibration.',
                 'Second/third learner choice-model ECE coverage was not passed on 2023 holdout.',
                 '20-bin 80% gate is recorded for each ticket type and may be unattainable for sparse multi-rider tickets; separate quantile reliability is supplementary, not a replacement.',
                 'No odds data used in this probability experiment; 3連単・3連複 historical odds backtesting is a separate next operation.',
                 'Absent source files, missing features and podium mismatches are excluded from this evaluation, with audit counts.']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps({
        'state':report['state'],'started_at':started,
        'finished_at':report['finished_at'],
        'races':len(races),'2024_refit_count':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for k in TYPES:
        v=report['per_ticket_type'][k]
        print('TICKET',k,'races',v['races'],'top1',v['top1_hit_rate'],
              'NLL',v['mean_nll_real_ticket'],
              'ECE_GATE',v['paper_20bin_80pct_gate_passed'],flush=True)
    print('FINISHED',report['state'],flush=True)

if __name__=='__main__':
    main()
