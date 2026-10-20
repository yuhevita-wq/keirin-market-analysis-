"""2023 paired experiment: identical race sample, identical features except published line structure.

Train: Jan 1 - Oct 31 2023. Strict untouched 2023 evaluation: Nov 1 - Dec 31.
Four configs: LR and RF, each LINE_ON and LINE_OFF. Same model hyperparameters per pair.
Data source: published KDreams predicted_line_formation in historical races.csv.
NO reconstructed lines from race results; no historical odds or 2024 labels used.
"""
import csv, gzip, io, itertools, json, math, os, warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from train_2023 import FEATURES, fetch, sources, SEED

OUT = Path('docs/paper-only-ai-2023/line_ab_2023')
TRAIN_END = '2023-10-31'
HOLDOUT_START = '2023-11-01'
YEAR_END = '2023-12-31'
RESULTS = 'data/result_universe_v2/combined/results_2023.jsonl.gz'
EPS = 1e-12
BATCH = 30
ALGORITHMS = ('LR','RF')
BRANCHES = ('no_line','with_line')
TYPES=('3連単','3連複')

def parse_line(text,cars):
    if not isinstance(text,str) or not text.strip():
        return None
    groups=[]; used=[]
    for chunk in text.strip().split('/'):
        parts=chunk.strip().split('-')
        if not parts or any(not p.isdigit() for p in parts):
            return None
        if any(len(p)>1 for p in parts):
            return None
        groups.append(tuple(parts));used.extend(parts)
    if len(groups)<1 or len(set(used))!=len(used) or set(used)!=set(cars):
        return None
    return tuple(groups)

def lines_lookup(groups):
    positions={}
    for group_number,group in enumerate(groups):
        for i,car in enumerate(group):
            positions[car]=(group_number,i+1,len(group))
    return positions

def load():
    outcome={}
    with gzip.open(io.BytesIO(fetch(RESULTS)),'rt',encoding='utf-8') as z:
        for line in z:
            row=json.loads(line)
            top=row.get('finish_order')
            if isinstance(top,list) and len(top)>=3:
                top=tuple(map(str,top[:3]))
                if len(set(top))==3 and all(t.isdigit() for t in top):
                    outcome[str(row['race_id'])]=top
    entries={}
    metadata={}
    audit=Counter()
    manifests=[]
    for path in sources():
        source_path=path.replace('/entries.csv','/races.csv')
        try:
            source=fetch(source_path).decode('utf-8-sig')
            card=fetch(path).decode('utf-8-sig')
        except Exception as exc:
            audit['source_failures']+=1
            raise RuntimeError('Cannot inspect input source '+path) from exc
        this=0;line_count=0
        for row in csv.DictReader(io.StringIO(source)):
            rid=str(row.get('race_id','')).strip()
            date=str(row.get('race_date','')).strip()
            if not '2023-01-01'<=date<=YEAR_END or rid not in outcome:
                continue
            line=str(row.get('predicted_line_formation') or '').strip()
            if row.get('line_status')!='published' or not line:
                audit['published_line_unavailable']+=1
                continue
            if rid in metadata:
                audit['duplicate_race_manifest']+=1
                continue
            metadata[rid]={'date':date,'published_line':line,
                           'source':('A1_A2' if '/a_class_12/' in path else
                                     'A3' if '/a3_challenge/' in path else 'S')}
            line_count+=1
        for row in csv.DictReader(io.StringIO(card)):
            rid=str(row.get('race_id','')).strip()
            if rid not in metadata:
                continue
            car=str(row.get('car_no','')).strip()
            try:
                nums=np.array([float(row[k]) for k in FEATURES],dtype=np.float32)
                if not car.isdigit() or not all(np.isfinite(nums)):
                    raise ValueError('non-finite')
            except (TypeError,ValueError,KeyError):
                audit['bad_numeric_rows']+=1
                continue
            batch=entries.setdefault(rid,{})
            if car in batch:
                audit['duplicate_entry_rows']+=1
                continue
            batch[car]=nums
            this+=1
        manifests.append({'source':path,'rows_accepted':this,'line_manifests':line_count})
        print('SOURCE',path,'rows',this,'published lines',line_count,flush=True)
    valid={}
    for rid,info in metadata.items():
        cards=entries.get(rid,{})
        if len(cards)<4:
            audit['exclude_insufficient_runners']+=1;continue
        if not all(car in cards for car in outcome[rid]):
            audit['exclude_missing_podium_car']+=1;continue
        groups=parse_line(info['published_line'],cards.keys())
        if groups is None:
            audit['exclude_bad_line_coverage']+=1;continue
        valid[rid]={'date':info['date'],'source':info['source'],'cars':cards,
                    'order':outcome[rid], 'groups':groups,
                    'positions':lines_lookup(groups)}
    valid=dict(sorted(valid.items(),key=lambda x:(x[1]['date'],x[0])))
    training={k:v for k,v in valid.items() if v['date']<=TRAIN_END}
    heldout={k:v for k,v in valid.items() if v['date']>=HOLDOUT_START}
    if min(len(training),len(heldout))<200:
        raise RuntimeError('Not enough paired train/test races')
    return training, heldout, {'sources':manifests,
                               'exclusions':dict(audit),
                               'races_with_valid_line_and_all_card_features':len(valid),
                               'training_races':len(training),
                               'heldout_races':len(heldout),
                               'heldout_categories':dict(Counter(r['source'] for r in heldout.values())),
                               'heldout_months':dict(Counter(r['date'][:7] for r in heldout.values())),
                               'using_only_exactly_identical_race_ids_for_both_branches':True}

def extras(race,k,a=None,b=None):
    lookup=race['positions']; nlines=len(race['groups'])
    def single(car):
        g,i,size=lookup[car]
        return [float(i),float(size),float(nlines),float(size-i)]
    vec=single(k)
    if a is not None:
        ga,pa,sa=lookup[a]
        gk,pk,sk=lookup[k]
        vec+=single(a)+[float(gk==ga),float(pk-pa if gk==ga else 0),
                         float(abs(pk-pa) if gk==ga else 0)]
    if b is not None:
        ga,pa,sa=lookup[a]; gb,pb,sb=lookup[b]; gk,pk,sk=lookup[k]
        vec+=single(b)+[float(gk==gb),float(gb==ga),
                        float(pk-pb if gk==gb else 0),
                        float(pa-pb if ga==gb else 0),
                        float(len({gk,ga,gb}))]
    return np.asarray(vec,dtype=np.float32)

def base_features(cards,k,a=None,b=None):
    cur=cards[k]
    if a is None: return cur
    other=cards[a]
    if b is None:
        return np.concatenate([cur,other,cur-other])
    next_=cards[b]
    return np.concatenate([cur,other,next_,cur-other,cur-next_])

def feature(race,k,a=None,b=None,use_line=False):
    x=base_features(race['cars'],k,a,b)
    if use_line:
        x=np.concatenate([x,extras(race,k,a,b)])
    return x

def matrix(events,stage,use_line):
    X=[];y=[];groups=[]
    for group_id,(rid,race) in enumerate(events.items()):
        order=race['order']
        first=order[0] if stage>=2 else None
        second=order[1] if stage>=3 else None
        exclude=set(order[:stage-1])
        candidates=sorted(set(race['cars'])-exclude,key=int)
        truth=order[stage-1]
        if truth not in candidates:
            raise RuntimeError('Target not in candidates')
        for c in candidates:
            X.append(feature(race,c,first,second,use_line))
            y.append(int(c==truth))
            groups.append(group_id)
    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int8),np.asarray(groups,dtype=np.int32)

def classifier(kind):
    if kind=='LR':
        return make_pipeline(StandardScaler(),LogisticRegression(
            solver='liblinear',C=1.0,max_iter=350,random_state=SEED))
    if kind=='RF':
        return RandomForestClassifier(n_estimators=90,max_depth=10,
            min_samples_leaf=12,max_features=0.7,n_jobs=2,random_state=SEED)
    raise ValueError(kind)

def fit_all(training):
    models={}
    for stage in (1,2,3):
        for branch in BRANCHES:
            use_line=branch=='with_line'
            X,y,g=matrix(training,stage,use_line)
            print('TRAINING',stage,branch,len(y),'features',X.shape[1],flush=True)
            for kind in ALGORITHMS:
                model=classifier(kind)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning)
                    model.fit(X,y)
                models[(kind,branch,stage)]=model
                path=OUT/'models'/kind/branch
                path.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,path/f'stage_{stage}.joblib',compress=3)
                print('TRAINED',kind,branch,stage,flush=True)
    return models

def normalise(raw,group_ids):
    raw=np.maximum(np.asarray(raw,dtype=float),EPS)
    gid=np.asarray(group_ids,dtype=np.int64)
    denom=np.bincount(gid,weights=raw)
    p=raw/denom[gid]
    if np.max(np.abs(np.bincount(gid,weights=p)-1))>1e-8:
        raise RuntimeError('Probability normalization failed')
    return p

def infer_chunk(chunk,models):
    # All four model variants are evaluated on precisely the same hypothetical
    # first/second outcomes; actual finish order does not enter feature generation.
    template={}
    for branch in BRANCHES:
        stage_vectors={1:[],2:[],3:[]}
        group_ids={1:[],2:[],3:[]}
        keys={1:[],2:[],3:[]}
        for rid,race in chunk:
            cars=sorted(race['cars'],key=int)
            gid1=len(keys[1]) # replaced by stable per-race group in group key lookup
            for a in cars:
                keys[1].append((rid,a))
                stage_vectors[1].append(feature(race,a,use_line=branch=='with_line'))
                group_ids[1].append(rid)
            for a in cars:
                for b in cars:
                    if b==a:continue
                    keys[2].append((rid,a,b))
                    stage_vectors[2].append(feature(race,b,a,use_line=branch=='with_line'))
                    group_ids[2].append((rid,a))
                    for c in cars:
                        if c==a or c==b:continue
                        keys[3].append((rid,a,b,c))
                        stage_vectors[3].append(feature(race,c,a,b,use_line=branch=='with_line'))
                        group_ids[3].append((rid,a,b))
        group_int={}
        probs={}
        for kind in ALGORITHMS:
            p_by_stage={}
            for stage in (1,2,3):
                model=models[(kind,branch,stage)]
                raw=model.predict_proba(np.asarray(stage_vectors[stage],dtype=np.float32))[:,1]
                unique={}
                ids=[]
                for key in group_ids[stage]:
                    if key not in unique:unique[key]=len(unique)
                    ids.append(unique[key])
                p_by_stage[stage]=normalise(raw,ids)
            p1=dict(zip(keys[1],p_by_stage[1]))
            p2=dict(zip(keys[2],p_by_stage[2]))
            joint=defaultdict(lambda:{'3連単':{},'3連複':defaultdict(float)})
            for key,third in zip(keys[3],p_by_stage[3]):
                rid,a,b,c=key
                p=float(p1[(rid,a)]*p2[(rid,a,b)]*third)
                joint[rid]['3連単'][(a,b,c)]=p
                joint[rid]['3連複'][tuple(sorted((a,b,c),key=int))]+=p
            for rid,race in chunk:
                for t in TYPES:
                    total=sum(joint[rid][t].values())
                    if abs(total-1)>1e-8:
                        raise RuntimeError('Ticket probabilities do not sum to one')
            probs[(kind,branch)]=joint
    return probs

def race_scoring(race,distribution,kind):
    a,b,c=race['order']
    correct=(a,b,c) if kind=='3連単' else tuple(sorted((a,b,c),key=int))
    keys=sorted(distribution,key=lambda z:tuple(map(int,z)))
    values=np.array([distribution[x] for x in keys])
    if correct not in distribution:raise RuntimeError('True ticket absent')
    p=float(distribution[correct])
    ranked=sorted(keys,key=lambda x:(-distribution[x],tuple(map(int,x))))
    return {'nll':float(-math.log(max(p,EPS))),
            'top1':int(ranked[0]==correct),
            'top3':int(correct in ranked[:3]),
            'brier':float(1-2*p+np.square(values).sum()),
            'p_true':p}

def aggregates(rows):
    if not rows:return None
    return {k:float(np.mean([row[k] for row in rows]))
            for k in ('nll','top1','top3','brier','p_true')}|{'races':len(rows)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    (OUT/'progress.json').write_text(json.dumps({'state':'RUNNING','phase':'loading','started_at':started},
                                                ensure_ascii=False)+'\n')
    tr,te,audit=load()
    (OUT/'progress.json').write_text(json.dumps({'state':'RUNNING','phase':'training',
          'training_races':len(tr),'holdout_races':len(te),'started_at':started},ensure_ascii=False)+'\n')
    models=fit_all(tr)
    rows={alg:{branch:{t:[] for t in TYPES} for branch in BRANCHES} for alg in ALGORITHMS}
    subsets=defaultdict(lambda:{alg:{branch:{t:[] for t in TYPES} for branch in BRANCHES} for alg in ALGORITHMS})
    sample=list(te.items())
    for idx in range(0,len(sample),BATCH):
        chunk=sample[idx:idx+BATCH]
        pred=infer_chunk(chunk,models)
        for rid,race in chunk:
            for alg in ALGORITHMS:
                for branch in BRANCHES:
                    for t in TYPES:
                        item=race_scoring(race,pred[(alg,branch)][rid][t],t)
                        rows[alg][branch][t].append(item)
                        subsets[race['source']][alg][branch][t].append(item)
        if (idx+BATCH)%300 < BATCH or idx+BATCH>=len(sample):
            print('TESTED',min(idx+BATCH,len(sample)),'/',len(sample),flush=True)
            (OUT/'progress.json').write_text(json.dumps({'state':'RUNNING',
                'phase':'holdout','tested_races':min(idx+BATCH,len(sample)),
                'total_races':len(sample),'started_at':started},ensure_ascii=False)+'\n')
    total={alg:{branch:{t:aggregates(rows[alg][branch][t]) for t in TYPES}
                for branch in BRANCHES} for alg in ALGORITHMS}
    comparison={}
    for alg in ALGORITHMS:
        comparison[alg]={}
        for kind in TYPES:
            w=rows[alg]['with_line'][kind]
            n=rows[alg]['no_line'][kind]
            delta=np.array([x['nll']-y['nll'] for x,y in zip(n,w)],dtype=float)
            comparison[alg][kind]={
                'delta_nll_without_minus_with_positive_means_line_better':float(delta.mean()),
                'line_better_nll_races':int(np.sum(delta>0)),
                'no_line_better_nll_races':int(np.sum(delta<0)),
                'tie_nll_races':int(np.sum(delta==0)),
                'delta_top1_line_minus_without':total[alg]['with_line'][kind]['top1']-
                                                   total[alg]['no_line'][kind]['top1'],
                'delta_brier_without_minus_with':total[alg]['no_line'][kind]['brier']-
                                                   total[alg]['with_line'][kind]['brier']}
    report={
      'state':'COMPLETE_PAIRED_2023_LINE_AB',
      'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
      'objective':'Assess whether supplied published pre-race line grouping/order improves probabilistic prediction in controlled paired conditions',
      'train_dates':['2023-01-01',TRAIN_END],
      'strict_heldout_dates':[HOLDOUT_START,YEAR_END],
      'line_source_field':'races.csv predicted_line_formation',
      'line_semantics':'published KDreams line prediction, not verified realized on-track arrangement',
      'split_isolation':'Both branches train on same race ids; no Nov-Dec labels used in fitting or feature selection',
      'algorithms_and_hyperparameters':{
          'LR':{'solver':'liblinear','C':1.0,'max_iter':350,'preprocessing':'StandardScaler'},
          'RF':{'trees':90,'depth':10,'min_samples_leaf':12,'max_features':0.7,'n_jobs':2}},
      'retraining_not_comparable_directly_to_earlier_2023_full_year_model':True,
      'baseline_features':FEATURES,
      'extra_line_features':'line position, line length, line counts, within-line distance, same-line flags and relations among hypothetical 1/2/3 finishers',
      'fair_same_training_samples':True,
      'audit':audit,'metrics':total,'paired_improvement':comparison,
      'by_class':{source:{alg:{branch:{t:aggregates(rows) for t,rows in tt.items()}
                                  for branch,tt in bs.items()} for alg,bs in aa.items()}
                  for source,aa in subsets.items()},
      'calibration_note':'Proper multiclass log-loss and Brier on complete 3連単/3連複 distributions are used; no claim of Walsh-Joshi 20-bin-ECE coverage pass',
      'odds_roi_not_computed':True,
      'no_2024_data_used':True,
      'limitations':['2023 November-December is the evaluation holdout. A new 2024+ untouched-period check is needed for external generalization.',
                     'The experiment is a fixed-architecture ablation and does not constitute paper-compliant full feature search/hyperparameter calibration.',
                     'The published line forecast is an archived prediction; not an official observed line result.',
                     'A3 2023 Jan-Feb has no published line column in source, and Mar-Apr source is absent. Only paired available-line rows enter.',
                     'No arbitrary bets, stake amounts, or odds-based selection tested in this comparison.']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps({'state':report['state'],
        'finished_at':report['finished_at'],'heldout_races':len(te)},ensure_ascii=False)+'\n')
    print('RESULT_COMPARISON',json.dumps(comparison,ensure_ascii=False),flush=True)
    print('FINISHED',len(tr),len(te),flush=True)

if __name__=='__main__':
    main()
