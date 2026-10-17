"""Predeclared: 2023-24 train; 2025 and 2026 H1 untouched temporal tests.
Binary unordered top-two event learner, normalized over ALL 21 unordered pairs.
No result, payout or odds in features. A/S fit separately. Fixed hyperparameters.
"""
import sys,json,subprocess,importlib.util
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old',ROOT/'research/cross_joint_v3/train.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
WORK=ROOT/'.coexistence-work';WORK.mkdir(exist_ok=True)
def export_model(m):
    trees=[]
    for stage in m._predictors:
        trees.append([{'leaf':bool(n['is_leaf']),'value':float(n['value']),
          'feature':int(n['feature_idx']),'threshold':float(n['num_threshold']),
          'left':int(n['left']),'right':int(n['right'])} for n in stage[0].nodes])
    gain=np.zeros(m.n_features_in_)
    for stage in m._predictors:
        for n in stage[0].nodes:
            if not n['is_leaf']:gain[int(n['feature_idx'])]+=max(0,float(n['gain']))
    return {'intercept':float(m._baseline_prediction[0,0]),'trees':trees,'trainingSplitGain':gain.tolist()}
def evaluate(meta,q):
    n=len(meta);targets=np.array([r['target'] for r in meta]);base=np.array([r['base'] for r in meta]);ii=np.arange(n)
    report={'n':n,'basePairNLL':float(-np.log(base[ii,targets]).mean()),'newPairNLL':float(-np.log(q[ii,targets]).mean())}
    delta=np.log(q[ii,targets])-np.log(base[ii,targets]);se=float(delta.std(ddof=1)/np.sqrt(n))
    report['pairedMeanNLLImprovement95CI']=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)]
    for name,probs in [('base',base),('new',q),('new_gated',q)]:
        hits=0;selected=0;priced=0;payout=0;changed=0
        for i,r in enumerate(meta):
            cs=r['candidates']
            if not cs:continue
            vals=[c['p']*probs[i,c['pair']]/r['base'][c['pair']] for c in cs]
            if name=='new_gated':
                agg={}
                for j,lp in enumerate(r['linePairs']):
                    k=tuple(lp);agg[k]=agg.get(k,0)+probs[i,j]
                cross={k:v for k,v in agg.items() if k[0]!=k[1]};banned=set()
                if len(r['lines'])==3:
                    weak=r['third'];ss=next(k for k in cross if weak not in k);sw=max((k for k in cross if weak in k),key=cross.get)
                    ordered=sorted(r['lineWin'],reverse=True)
                    if ordered[1]>ordered[2]+1e-12 and cross[sw]>cross[ss]+1e-12:banned.update(j for j,lp in enumerate(r['linePairs']) if tuple(lp)==ss)
                wins=r['win']
                for lp in cross:
                    aa=sorted(r['lines'][lp[0]],key=lambda n:-wins[str(n)]);bb=sorted(r['lines'][lp[1]],key=lambda n:-wins[str(n)])
                    if (len(aa)>1 and wins[str(aa[0])]<=wins[str(aa[1])]+1e-12) or (len(bb)>1 and wins[str(bb[0])]<=wins[str(bb[1])]+1e-12):continue
                    indexes=[j for j,k in enumerate(r['linePairs']) if tuple(k)==lp]
                    ssj=next(j for j in indexes if set(r['pairs'][j])=={aa[0],bb[0]})
                    swjs=[j for j in indexes if (aa[0] in r['pairs'][j]) != (bb[0] in r['pairs'][j])]
                    if swjs and max(probs[i,j] for j in swjs)>probs[i,ssj]+1e-12:banned.add(ssj)
                cars=sorted(wins,key=lambda n:-wins[n]);strong=set(map(int,cars[:2]))
                ssj=next(j for j,p in enumerate(r['pairs']) if set(p)==strong)
                swjs=[j for j,p in enumerate(r['pairs']) if len(set(p)&strong)==1 and r['linePairs'][j][0]!=r['linePairs'][j][1]]
                if wins[cars[1]]>wins[cars[2]]+1e-12 and r['linePairs'][ssj][0]!=r['linePairs'][ssj][1] and swjs and max(probs[i,j] for j in swjs)>probs[i,ssj]+1e-12:banned.add(ssj)
                vals=[v if cs[j]['pair'] not in banned else -1 for j,v in enumerate(vals)]
                if max(vals)<0:continue
            k=int(np.argmax(vals));b=int(np.argmax([c['p'] for c in cs]));changed+=k!=b
            selected+=1;hits+=cs[k]['hit']
            if r['payout'] is not None:priced+=1;payout+=r['payout']*cs[k]['hit']
        report[name]={'selected':selected,'hits':hits,'changed':changed,'priced':priced,'returnPct':payout/(400*priced)*100 if priced else None}
    # 322 only: observable categories with no retrospective selection.
    groups={}
    for i,r in enumerate(meta):
        if sorted(map(len,r['lines']))!=[2,2,3]:continue
        # rank1/2 strong relative to the lowest-win line; no score cut-off.
        weak=r['third'];agg={}
        for j,lp in enumerate(r['linePairs']):
            k=tuple(lp);agg[k]=agg.get(k,0)+q[i,j]
        cross={k:v for k,v in agg.items() if k[0]!=k[1]}
        ss=next(k for k in cross if weak not in k);sw=max((k for k in cross if weak in k),key=cross.get)
        switch=cross[sw]>cross[ss];actual=tuple(r['linePairs'][r['target']]);kind='SAME' if actual[0]==actual[1] else 'SW' if weak in actual else 'SS'
        key='line_switch' if switch else 'line_keep'
        g=groups.setdefault(key,{'n':0,'SS':0,'SW':0,'SAME':0,'chosenLineHit':0});g['n']+=1;g[kind]+=1;g['chosenLineHit']+=actual==(sw if switch else ss)
        wins=r['win'];cars=sorted(wins,key=lambda n:-wins[n]);strong=set(map(int,cars[:2]))
        ssj=next(j for j,p in enumerate(r['pairs']) if set(p)==strong)
        swjs=[j for j,p in enumerate(r['pairs']) if len(set(p)&strong)==1 and r['linePairs'][j][0]!=r['linePairs'][j][1]]
        if wins[cars[1]]>wins[cars[2]]+1e-12 and r['linePairs'][ssj][0]!=r['linePairs'][ssj][1] and swjs:
            swj=max(swjs,key=lambda j:q[i,j]);key='global_player_switch' if q[i,swj]>q[i,ssj]+1e-12 else 'global_player_keep'
            g=groups.setdefault(key,{'n':0,'strongStrongExact':0,'chosenStrongWeakExact':0,'sameLineActual':0})
            g['n']+=1;g['strongStrongExact']+=r['target']==ssj;g['chosenStrongWeakExact']+=r['target']==swj;g['sameLineActual']+=kind=='SAME'
        # Within the ACTUAL cross lines: conditional diagnostic, not a pre-race selection score.
        if kind=='SAME':continue
        a,b=actual;wins=r['win'];aa=sorted(r['lines'][a],key=lambda n:-wins[str(n)]);bb=sorted(r['lines'][b],key=lambda n:-wins[str(n)])
        if abs(wins[str(aa[0])]-wins[str(aa[1])])<1e-12 or abs(wins[str(bb[0])]-wins[str(bb[1])])<1e-12:continue
        indexes=[j for j,lp in enumerate(r['linePairs']) if tuple(lp)==actual]
        ssj=next(j for j in indexes if set(r['pairs'][j])=={aa[0],bb[0]})
        swjs=[j for j in indexes if (aa[0] in r['pairs'][j]) != (bb[0] in r['pairs'][j])]
        swj=max(swjs,key=lambda j:q[i,j]);pswitch=q[i,swj]>q[i,ssj]
        at=r['pairs'][r['target']];pk='SS' if set(at)=={aa[0],bb[0]} else 'SW' if ((aa[0] in at) != (bb[0] in at)) else 'WW'
        g=groups.setdefault('player_switch' if pswitch else 'player_keep',{'n':0,'SS':0,'SW':0,'WW':0,'chosenPairHit':0});g['n']+=1;g[pk]+=1;g['chosenPairHit']+=r['target']==(swj if pswitch else ssj)
    report['groups322']=groups
    return report
bundle={'version':'coexistence-v1','trainThrough':'2024-12-31','hardGateEnabled':False,'models':{},'featureNames':None,'protocol':'2023-24 fixed SHA256 cap12000/class; all 2025 and 2026H1 temporal evaluation; fixed learner hyperparameters; old conditional order/third retained; hard gate exploratory and disabled'}
report={'protocol':bundle['protocol'],'classes':{}}
for cls in ['A','S']:
    prefix=WORK/cls
    if not prefix.with_suffix('.json').exists():
        cards,excluded=old.load(cls)
        train=old.sample([r for r in cards if r['date']<'2025-01-01'],12000)
        tests=[r for r in cards if r['date']>='2025-01-01']
        cards=train+tests
        print(cls,'prepare',len(cards),'train',len(train),flush=True)
        inp=WORK/(cls+'-cards.json');inp.write_text(json.dumps(cards,ensure_ascii=False))
        subprocess.run(['node',str(Path(__file__).with_name('prepare.mjs')),str(inp),str(prefix)],check=True)
    data=json.loads(prefix.with_suffix('.json').read_text());meta=data['meta'];bundle['featureNames']=data['featureNames']
    x=np.fromfile(prefix.with_suffix('.f32'),np.float32).reshape(len(meta),21,-1)
    y=np.zeros((len(meta),21),np.int8);y[np.arange(len(meta)),[r['target'] for r in meta]]=1
    train=np.array([r['date']<'2025-01-01' for r in meta])
    model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=200,l2_regularization=5,learning_rate=.06,early_stopping=False,random_state=0)
    print(cls,'fit',int(train.sum()),flush=True)
    model.fit(x[train].reshape(-1,x.shape[-1]),y[train].reshape(-1))
    bundle['models'][cls]=export_model(model);bundle['models'][cls]['trainRaces']=int(train.sum())
    q=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(-1,21);q/=q.sum(axis=1,keepdims=True)
    report['classes'][cls]={}
    for period,start,end in [('2025','2025-01-01','2025-12-31'),('2026H1','2026-01-01','2026-06-30')]:
        inds=[i for i,r in enumerate(meta) if start<=r['date']<=end]
        result=evaluate([meta[i] for i in inds],q[inds]);report['classes'][cls][period]=result
        print(cls,period,json.dumps(result),flush=True)
    # Cross-language reference is never used to fit or select parameters.
    ci=next(i for i,r in enumerate(meta) if r['date']>='2025-01-01')
    fixture=json.loads((WORK/(cls+'-cards.json')).read_text())[ci]
    fixture['expectedPairs']=q[ci].tolist()
    (WORK/(cls+'-reference.json')).write_text(json.dumps(fixture,ensure_ascii=False))
(ROOT/'docs/cross-lab/coexistence_frozen.json').write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':'))+'\n')
(ROOT/'research/cross_coexistence_v1/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
