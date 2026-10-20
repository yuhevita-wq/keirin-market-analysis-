"""Independent 2023 learner. Only the cited paper's ML methodology; no other models."""
import csv,gzip,io,json,os,warnings
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,urlopen
import numpy as np,pandas as pd,optuna,joblib
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT='https://raw.githubusercontent.com/yuhevita-wq/keirin-market-analysis-/main/'
OUTPUT=Path('docs/paper-only-ai-2023/training')
FEATURES=['car_no','age','term','gear','score','s_count','b_count','nige_count',
'makuri_count','sashi_count','mark_count','first_count','second_count','third_count',
'outside_count','win_rate','top2_rate','top3_rate']
SEED=230306021
N_TRIALS=3
SVM_CAP=8000
def fetch(path):
    if not path.startswith('data/') or '..' in path:raise ValueError(path)
    with urlopen(Request(ROOT+path,headers={'User-Agent':'paper-only-ai-2023'}),timeout=180) as r:return r.read()
def sources():
    p=['01_02','03_04','05_06','07_08','09_10','11_12']
    a=['01_02','05_06','07_08','09_10','11_12']
    return [f'data/a_class_12/2023_{x}/entries.csv' for x in p]+[
        f'data/a3_challenge/2023_{x}/entries.csv' for x in a]+[
        f'data/2023/s_class_f1_all_parts/2023_{q}/entries.csv' for q in ('q1','q2','q3','q4')]+[
        'data/2023/s_class_yosen/entries.csv']
def build_data():
    label_path='data/result_universe_v2/combined/results_2023.jsonl.gz'
    winners={}
    with gzip.open(io.BytesIO(fetch(label_path)),'rt',encoding='utf-8') as handle:
        for line in handle:
            r=json.loads(line)
            order=r.get('finish_order')
            if isinstance(order,list) and order and str(order[0]).isdigit():
                winners[str(r['race_id'])]=str(order[0])
    items={};stats={'duplicate_entries':0,'missing_numeric':0,'absent_first_place_races':0,'files':{}}
    for path in sources():
        found=0
        data=fetch(path).decode('utf-8-sig')
        for r in csv.DictReader(io.StringIO(data)):
            rid=str(r.get('race_id','')).strip(); car=str(r.get('car_no','')).strip()
            date=str(r.get('race_date',''))
            if rid not in winners or not ('2023-01-01'<=date<='2023-12-31'):continue
            key=(rid,car)
            if key in items:stats['duplicate_entries']+=1;continue
            try:
                vals={c:float(r[c]) for c in FEATURES}
                if not all(np.isfinite(x) for x in vals.values()):raise ValueError('nonfinite')
            except (ValueError,KeyError,TypeError):stats['missing_numeric']+=1;continue
            items[key]={**vals,'race_id':rid,'race_date':date,
                        'event_id':rid+'_'+car,'outcome':int(car==winners[rid])}
            found+=1
        stats['files'][path]=found
        print('SOURCE',path,found,flush=True)
    df=pd.DataFrame(items.values())
    if df.empty:raise RuntimeError('No 2023 labelled entries')
    counts=df.groupby('race_id').outcome.sum()
    stats['absent_first_place_races']=int((counts!=1).sum())
    df=df[df.race_id.isin(counts[counts==1].index)].sort_values(['race_date','race_id','event_id'])
    stats['races']=int(df.race_id.nunique())
    stats['rows']=len(df)
    stats['outcome_1']=int(df.outcome.sum())
    stats['outcome_0']=len(df)-stats['outcome_1']
    stats['source_result']=label_path
    stats['a3_missing_period']='2023-03-01 to 2023-04-30 (dedicated archive absent)'
    stats['pre_race_feature_verification']='Historical entries snapshots; retrospective source capture, availability at race time not independently certified'
    return df.reset_index(drop=True),stats
def take(data,limit):
    return data if len(data)<=limit else data.sample(n=limit,random_state=SEED)
def ece(y,p,strict=True):
    y=np.asarray(y,dtype=int);total=0;occupied=[]
    for cls in (0,1):
        idx=np.clip((p[:,cls]*20).astype(int),0,19)
        occupied.append(int(len(np.unique(idx))))
        for b in range(20):
            mask=idx==b
            if mask.any():total+=mask.mean()*abs((y[mask]==cls).mean()-p[mask,cls].mean())/2
    return (1.0 if strict and min(occupied)<16 else float(total)),occupied
def make(kind,p):
    if kind=='LR':c=LogisticRegression(C=p.get('C',1.),solver='liblinear',max_iter=250,random_state=SEED)
    elif kind=='RF':c=RandomForestClassifier(n_estimators=p.get('trees',70),max_depth=p.get('depth',8),min_samples_leaf=p.get('leaf',5),n_jobs=2,random_state=SEED)
    elif kind=='SVM':c=SVC(C=p.get('C',1.),kernel=p.get('kernel','rbf'),gamma=p.get('gamma','scale'),probability=True,random_state=SEED)
    elif kind=='MLP':c=MLPClassifier(hidden_layer_sizes=(p.get('hidden',24),),alpha=p.get('alpha',.001),
                                     learning_rate_init=p.get('lr',.001),max_iter=120,random_state=SEED)
    else:raise ValueError(kind)
    return make_pipeline(StandardScaler(),c)
def suggest(kind,t):
    if kind=='LR':return {'C':t.suggest_float('C',.01,20,log=True)}
    if kind=='RF':return {'trees':t.suggest_int('trees',45,95,step=25),'depth':t.suggest_int('depth',4,11),'leaf':t.suggest_int('leaf',2,12)}
    if kind=='SVM':return {'C':t.suggest_float('C',.03,10,log=True),'kernel':t.suggest_categorical('kernel',['linear','rbf']),'gamma':t.suggest_categorical('gamma',['scale','auto'])}
    if kind=='MLP':return {'hidden':t.suggest_categorical('hidden',[16,32]),'alpha':t.suggest_float('alpha',.0001,.03,log=True),'lr':t.suggest_float('lr',.0003,.003,log=True)}
    raise ValueError(kind)
def fit(kind,p,df,cols):
    m=make(kind,p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        m.fit(df[cols],df.outcome)
    return m
def loss(m,df,cols,metric):
    p=m.predict_proba(df[cols])
    return ece(df.outcome,p,True)[0] if metric=='calibration' else 1-float(accuracy_score(df.outcome,p[:,1]>=.5))
def select(tr,va,cols,metric):
    tr=take(tr,10000);va=take(va,12000)
    rest=cols[:];current=[];best=10.;bestcols=[];trace=[]
    # Computational cap is a reported deviation from paper's complete SFS.
    while rest and len(current)<8:
        candidates=[]
        for c in rest:
            use=current+[c]
            model=fit('LR',{'C':1.},tr,use)
            candidates.append((loss(model,va,use,metric),c))
        v,c=min(candidates);rest.remove(c);current.append(c)
        trace.append({'feature':c,'objective':float(v)})
        if v<best-1e-9:best=float(v);bestcols=current[:]
    return bestcols,trace
def learn_branch(tr,va,te,ho,cols,metric):
    selected,trace=select(tr,va,cols,metric)
    if not selected:raise RuntimeError('no selected features')
    scores=[]
    for kind in ['LR','RF','SVM','MLP']:
        train=take(tr,SVM_CAP if kind=='SVM' else 28000)
        def objective(t):
            p=suggest(kind,t)
            return loss(fit(kind,p,train,selected),va,selected,metric)
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
        params=study.best_params
        before_test=pd.concat([tr,va],ignore_index=True)
        fit_set=take(before_test,SVM_CAP if kind=='SVM' else 55000)
        mod=fit(kind,params,fit_set,selected)
        probs=mod.predict_proba(te[selected])
        calibrated,coverage=ece(te.outcome,probs,True)
        accuracy=float(accuracy_score(te.outcome,probs[:,1]>=.5))
        score={'algorithm':kind,'params':params,'validation_score':float(study.best_value),
               'test_accuracy':accuracy,'test_ece_strict':calibrated,'test_occupied_bins':coverage,
               'test_loss':calibrated if metric=='calibration' else 1-accuracy}
        scores.append(score)
        print('TRIAL',metric,kind,round(accuracy,5),round(calibrated,5),coverage,flush=True)
    chosen=min(scores,key=lambda x:x['test_loss'])
    kind=chosen['algorithm'];p=chosen['params']
    before_ho=pd.concat([tr,va,te],ignore_index=True)
    mod=fit(kind,p,take(before_ho,SVM_CAP if kind=='SVM' else 60000),selected)
    probs=mod.predict_proba(ho[selected])
    strict,coverage=ece(ho.outcome,probs,True)
    held={'accuracy':float(accuracy_score(ho.outcome,probs[:,1]>=.5)),
          'ece_unconstrained':ece(ho.outcome,probs,False)[0],
          'ece_20bin_80pct_rule':strict,'bins_per_class':coverage,'rows':len(ho)}
    full_year=pd.concat([tr,va,te,ho],ignore_index=True)
    cap=SVM_CAP if kind=='SVM' else 80000
    final=fit(kind,p,take(full_year,cap),selected)
    result={'metric':metric,'features':selected,'sfs':trace,'candidates':scores,
            'selection':chosen,'passes_paper_coverage':bool(metric!='calibration' or chosen['test_ece_strict']<1),
            'holdout':held,'final_fit_rows':min(len(full_year),cap)}
    return result,final
def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    status={'state':'RUNNING','start_date':'2023-01-01','end_date':'2023-12-31',
            'started_at':started,'trainings_completed':0}
    (OUTPUT/'progress.json').write_text(json.dumps(status,ensure_ascii=False))
    df,audit=build_data()
    dates={'train':('2023-01-01','2023-06-30'),'validation':('2023-07-01','2023-08-31'),
           'selection_test':('2023-09-01','2023-10-31'),'holdout':('2023-11-01','2023-12-31')}
    parts={k:df[df.race_date.between(*range_)].copy() for k,range_ in dates.items()}
    if min(len(v) for v in parts.values())<50:raise RuntimeError('Insufficient 2023 time-series data')
    tr=parts['train'];va=parts['validation'];keep=[];dropped={}
    for col in FEATURES:
        if tr[col].std()<1e-10:dropped[col]='constant';continue
        if ks_2samp(take(tr,9000)[col],take(va,8000)[col]).pvalue<.01:
            dropped[col]='KS p<0.01';continue
        related=next((x for x in keep if abs(tr[[x,col]].corr(method='spearman').iloc[0,1])>.7),None)
        if related:dropped[col]='Spearman >0.7 with '+related
        else:keep.append(col)
    if not keep:raise RuntimeError('All features filtered')
    outputs={}
    for metric in ('calibration','accuracy'):
        output,model=learn_branch(parts['train'],parts['validation'],parts['selection_test'],parts['holdout'],keep,metric)
        outputs[metric]=output
        joblib.dump(model,OUTPUT/(metric+'.joblib'))
        print('BRANCH_COMPLETE',metric,flush=True)
    guard=outputs['calibration']['passes_paper_coverage']
    result={'state':'RESEARCH_ONLY' if guard else 'CALIBRATION_GATE_FAILED',
            'methodology':'Walsh and Joshi (2024), arXiv:2303.06021v4',
            'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
            'start_date':'2023-01-01','end_date':'2023-12-31','pretrained_models':False,
            'existing_keirin_models_used':False,'dataset':audit,
            'chronological_partitions':{k:{'range':list(dates[k]),'rows':len(v),'races':int(v.race_id.nunique())} for k,v in parts.items()},
            'feature_filter':{'retained':keep,'excluded':dropped},'branches':outputs,
            'no_odds_comparison':True,'no_roi_computed':True,
            'limitations':['Binary target = individual rider wins versus does not win',
                           'No published odds exist for the constructed binary target',
                           'Time availability of historically captured pre-race statistics is not independently certified',
                           '2023 A3 March-Apr dedicated entry file is absent',
                           'Forward selection capped at 8 features; tuning uses 3 trials/model',
                           'SVM fitting capped at 8000 records, other fitting capped at 80000 records',
                           '20-bin 80%-coverage rule must pass for a calibrated model to be validated']}
    (OUTPUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUTPUT/'progress.json').write_text(json.dumps(
        {'state':result['state'],'start_date':result['start_date'],'end_date':result['end_date'],
         'started_at':started,'finished_at':result['finished_at'],'races':audit['races'],'rows':audit['rows']},
        ensure_ascii=False,indent=2)+'\n')
    print('FINISHED',result['state'],flush=True)
if __name__=='__main__':main()
