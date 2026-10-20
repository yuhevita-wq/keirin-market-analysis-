"""2023 conditional finish-order learning (research extension; not a formula from Walsh-Joshi).
The 2023 first-place classifier is FROZEN. All inputs are archived race-card numbers.
Stage 2 learns P(second rider | known first rider, race card).
Stage 3 learns P(third rider | known first and second riders, race card).
Hypothetical first/second riders at inference are supplied from the model's own enumeration,
not actual race results. Full ticket probabilities satisfy the chain rule.
"""
import csv, gzip, io, itertools, json, os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from train_2023 import FEATURES, fetch, sources, ece, SEED

OUT = Path('docs/paper-only-ai-2023/conditional_2023')
FROZEN = Path(os.environ.get('FROZEN_MODEL_DIR', 'frozen_2023'))
START, END = '2023-01-01', '2023-12-31'
RESULTS = 'data/result_universe_v2/combined/results_2023.jsonl.gz'
EPS = 1e-9
# Feature selection is determined by the 2023 data alone; no 2024 observations are used.
CANDIDATES = ('LR', 'RF')
def make_model(name):
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(
            solver='liblinear', C=1.0, random_state=SEED, max_iter=350))
    if name == 'RF':
        return RandomForestClassifier(n_estimators=110, max_depth=11,
                                      min_samples_leaf=10, max_features=0.7,
                                      n_jobs=2, random_state=SEED)
    raise ValueError(name)

def collect():
    orders = {}
    with gzip.open(io.BytesIO(fetch(RESULTS)), 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            order = item.get('finish_order')
            if isinstance(order, list) and len(order) >= 3:
                val = [str(i) for i in order[:3]]
                if len(set(val)) == 3 and all(i.isdigit() and int(i) > 0 for i in val):
                    orders[str(item['race_id'])] = val
    events = {}
    audit = Counter()
    source_counts = {}
    for path in sources():
        count = 0
        raw = fetch(path).decode('utf-8-sig')
        for r in csv.DictReader(io.StringIO(raw)):
            rid = str(r.get('race_id','')).strip()
            day = str(r.get('race_date','')).strip()
            car = str(r.get('car_no','')).strip()
            if rid not in orders or not (START <= day <= END):
                continue
            try:
                vals = np.array([float(r[k]) for k in FEATURES], dtype=np.float64)
                if not np.all(np.isfinite(vals)) or not car.isdigit():
                    raise ValueError('bad numeric')
            except (KeyError, ValueError, TypeError):
                audit['incomplete_card_rows'] += 1
                continue
            if rid not in events:
                events[rid] = {'date': day, 'riders': {}, 'first3': orders[rid]}
            if car in events[rid]['riders']:
                audit['duplicate_entry_rows'] += 1
                continue
            events[rid]['riders'][car] = vals
            count += 1
        source_counts[path] = count
        print('LOADED', path, count, flush=True)
    kept = {}
    for rid, event in events.items():
        cars = event['riders']
        if (len(cars) < 4 or not all(car in cars for car in event['first3'])
                or len(event['first3']) != 3):
            audit['races_missing_podium_rider'] += 1
            continue
        kept[rid] = event
    if not kept:
        raise RuntimeError('No complete 2023 top-three training races')
    events = dict(sorted(kept.items(), key=lambda kv: (kv[1]['date'], kv[0])))
    info = {'source_files': source_counts, **dict(audit),
            'valid_races': len(events),
            'full_podium_races': len(events),
            'rider_rows': sum(len(v['riders']) for v in events.values()),
            'label_source': RESULTS,
            'omitted_a3_source_period': '2023-03-01 through 2023-04-30',
            'known_predate_numeric_fields': FEATURES}
    return events, info

def conditional_features(candidate, winner1, winner2=None):
    if winner2 is None:
        return np.concatenate((candidate, winner1, candidate - winner1))
    return np.concatenate((candidate, winner1, winner2,
                           candidate - winner1, candidate - winner2))

def build_stage(events, stage):
    X, y, group, day, groupsizes = [], [], [], [], []
    for gid, (rid, race) in enumerate(events.items()):
        a, b, c = race['first3']
        riders = race['riders']
        others = sorted(k for k in riders if k not in ([a] if stage == 2 else [a,b]),
                        key=int)
        if not others:
            raise RuntimeError('Empty conditional choice set')
        target = b if stage == 2 else c
        if target not in others:
            raise RuntimeError('Training target absent')
        for k in others:
            X.append(conditional_features(riders[k],riders[a],
                                          riders[b] if stage == 3 else None))
            y.append(int(k == target))
            group.append(gid)
            day.append(race['date'])
        groupsizes.append(len(others))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    group = np.asarray(group, dtype=np.int32)
    day = np.asarray(day)
    if len(events) != np.sum(y):
        raise RuntimeError('Exactly one labelled chosen rider per conditional group is required')
    return {'X': X, 'y': y, 'group': group, 'date': day,
            'races': len(events), 'candidate_rows': len(y),
            'choice_set_sizes': dict(Counter(groupsizes))}

def group_normalise(probs, groups):
    scores = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1)
    g = np.asarray(groups, dtype=np.int64)
    if scores.size != g.size:
        raise RuntimeError('Score and group size mismatch')
    denom = np.bincount(g,weights=scores)
    result = scores / denom[g]
    deviation = np.max(np.abs(np.bincount(g, weights=result) - 1))
    if deviation > 1e-8:
        raise RuntimeError('Conditional probabilities fail to sum to 1')
    return result

def evaluated(model, data):
    X,y,g = data['X'],data['y'],data['group']
    p = group_normalise(model.predict_proba(X)[:,1], g)
    prob = np.stack([1-p,p],axis=1)
    strict, bin_coverage = ece(y,prob,True)
    raw_ece = ece(y,prob,False)[0]
    best = np.full(data['races'], -np.inf)
    predicted = np.full(data['races'], -1, dtype=int)
    for j, (i,score) in enumerate(zip(g,p)):
        if score > best[i]:
            predicted[i] = j
            best[i] = score
    top1 = float(np.mean(y[predicted]))
    nll = float(-np.log(np.clip(p[y==1],EPS,1)).mean())
    return {'races':int(data['races']),'rows':int(len(y)),
            'conditional_top1':top1,
            'mean_correct_choice_log_loss':nll,
            'conditional_brier':float(np.mean((y-p)**2)),
            'classwise_ece_raw':float(raw_ece),
            'classwise_ece_strict':float(strict),
            'occupied_20bins':bin_coverage,
            'paper_ece_gate_passed':bool(min(bin_coverage)>=16),
            'group_sum_error':float(np.max(np.abs(np.bincount(g,weights=p)-1)))}

def fit_stage(stage, events):
    training = {rid:r for rid,r in events.items() if r['date']<='2023-08-31'}
    selection = {rid:r for rid,r in events.items() if '2023-09-01'<=r['date']<='2023-10-31'}
    holdout = {rid:r for rid,r in events.items() if '2023-11-01'<=r['date']<='2023-12-31'}
    if min(map(len,(training,selection,holdout)))<100:
        raise RuntimeError('Training/validation split insufficient')
    train_d,select_d,hold_d=(build_stage(x,stage) for x in (training,selection,holdout))
    experiments={}
    for name in CANDIDATES:
        model=make_model(name)
        model.fit(train_d['X'],train_d['y'])
        experiments[name] = evaluated(model,select_d)
        print('STAGE',stage,'SELECT',name,experiments[name],flush=True)
    # Models with the paper's ECE occupancy gate pass ahead of those failing it.
    # If no model passes, select lower ECE for research only and make gate failure explicit.
    chosen=min(CANDIDATES, key=lambda z:(not experiments[z]['paper_ece_gate_passed'],
                                         experiments[z]['classwise_ece_raw']))
    extended={**training,**selection}
    extended_d=build_stage(extended,stage)
    model=make_model(chosen)
    model.fit(extended_d['X'],extended_d['y'])
    hold_results=evaluated(model,hold_d)
    print('STAGE',stage,'HOLDOUT',chosen,hold_results,flush=True)
    # Final 2023 refit follows paper's selection -> all historical training procedure.
    full_d=build_stage(events,stage)
    final=make_model(chosen)
    final.fit(full_d['X'],full_d['y'])
    OUT.mkdir(parents=True,exist_ok=True)
    joblib.dump(final,OUT/f'stage{stage}.joblib',compress=3)
    summary={'stage':stage,'target':('second conditioned on first' if stage==2
                                else 'third conditioned on first and second'),
             'trained_model':chosen,'model_candidates':experiments,
             'selection_period':'2023-09-01 through 2023-10-31',
             'holdout_period':'2023-11-01 through 2023-12-31',
             'holdout':hold_results,'final_fit_races':full_d['races'],
             'final_fit_rows':full_d['candidate_rows'],
             'features_per_candidate':int(full_d['X'].shape[1]),
             'paper_gate_on_holdout':bool(hold_results['paper_ece_gate_passed']),
             'methodology_extension':'True-result conditional training; non-winners ranked by a binary candidate score then normalized within each race/conditional set'}
    return final,summary

def probs_of_one_race(first,second,third,race):
    cars=sorted(race['riders'], key=int)
    if len(cars)<4:
        raise ValueError('At least four participants needed')
    originals=race['riders']
    stage1_cols=list(first.steps[0][1].feature_names_in_)
    if not set(stage1_cols).issubset(FEATURES):
        raise RuntimeError('Unknown stage1 trained columns')
    input_frame=pd.DataFrame([{f:float(originals[car][FEATURES.index(f)]) for f in stage1_cols} for car in cars],
                             columns=stage1_cols)
    raw1=first.predict_proba(input_frame)[:,1]
    p1=group_normalise(raw1,np.zeros(len(cars),dtype=int))
    trifecta={}
    for ia,a in enumerate(cars):
        rem2=[x for x in cars if x!=a]
        f2=np.asarray([conditional_features(originals[b],originals[a]) for b in rem2],dtype=np.float32)
        p2=group_normalise(second.predict_proba(f2)[:,1],np.zeros(len(rem2),dtype=int))
        for ib,b in enumerate(rem2):
            rem3=[x for x in rem2 if x!=b]
            f3=np.asarray([conditional_features(originals[c],originals[a],originals[b]) for c in rem3],dtype=np.float32)
            p3=group_normalise(third.predict_proba(f3)[:,1],np.zeros(len(rem3),dtype=int))
            for ic,c in enumerate(rem3):
                trifecta[(a,b,c)] = float(p1[ia]*p2[ib]*p3[ic])
    exacta=defaultdict(float);quinella=defaultdict(float);trio=defaultdict(float)
    for (a,b,c),p in trifecta.items():
        exacta[(a,b)] += p
        quinella[tuple(sorted([a,b],key=int))] += p
        trio[tuple(sorted([a,b,c],key=int))] += p
    totals={'3連単':sum(trifecta.values()),'2車単':sum(exacta.values()),
            '2車複':sum(quinella.values()),'3連複':sum(trio.values())}
    if any(abs(x-1)>1e-8 for x in totals.values()):
        raise RuntimeError('Ticket probability integrity broken')
    return {'3連単':dict(trifecta),'2車単':dict(exacta),
            '2車複':dict(quinella),'3連複':dict(trio)},totals

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=datetime.now(timezone.utc).isoformat()
    (OUT/'progress.json').write_text(json.dumps(
        {'state':'STARTING','start_date':START,'end_date':END,'started_at':start},
        ensure_ascii=False)+'\n')
    p=FROZEN/'calibration.joblib'
    if not p.exists():
        raise RuntimeError('Original frozen first-place calibration AI is not available')
    first=joblib.load(p)
    if not hasattr(first,'predict_proba') or not hasattr(first.steps[0][1],'feature_names_in_'):
        raise RuntimeError('Original first-stage model is incompatible')
    events,audit=collect()
    second,a2=fit_stage(2,events)
    third,a3=fit_stage(3,events)
    sample=[]
    for rid,race in itertools.islice(events.items(),20):
        tickets,totals=probs_of_one_race(first,second,third,race)
        sample.append({'race_id':rid,'participants':len(race['riders']),
                       'ticket_count':{k:len(v) for k,v in tickets.items()},
                       'ticket_probability_sums':totals})
    done=datetime.now(timezone.utc).isoformat()
    report={'state':'TRAINED_CONDITIONAL_RESEARCH',
            'frozen_first_place_model':'2023 calibration branch, GitHub Actions run 39354641293',
            'first_place_model_changed':False,
            'target_data_period':[START,END],
            'started_at':start,'finished_at':done,
            'methodological_boundary':'Walsh-Joshi classwise calibration model-selection principle; conditional finish-order construction is an original keirin extension, NOT part of that paper',
            'stage2':a2,'stage3':a3,'race_dataset':audit,
            'probability_inference':'P(first)*P(second|first)*P(third|first,second); normalize each exclusive choice set',
            'market_odds_used':False,'roi_computed':False,
            'full_ticket_probability_coverage':['3連単','3連複','2車単','2車複'],
            'sample_integrity_checks':sample,
            'risk_notes':['Conditional models were trained given ACTUAL previous finishers (teacher forcing); at inference they evaluate every hypothetical previous finisher. Joint calibration requires out-of-time testing.',
                          'Win-probability normalization changes the original binary probabilities; resulting per-race calibration is not guaranteed.',
                          'A3 entries for 2023 March-April are missing; source coverage is not nationwide complete.',
                          'Other ticket types require additional definitions or fields; no inferred odds.',
                          'ECE occupancy 80% is reported; a failed gate does not certify calibration']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps(
        {'state':'TRAINED_CONDITIONAL_RESEARCH','races':audit['valid_races'],'started_at':start,'finished_at':done},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINISHED',done,'races',audit['valid_races'],flush=True)
if __name__=='__main__':main()
