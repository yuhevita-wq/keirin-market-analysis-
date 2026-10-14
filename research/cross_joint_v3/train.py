"""Frozen historical A1/A2 and S joint top-two, then conditional third.

No 2026-10-14 card, odds or outcome enters fitting. Same JS feature builder as UI.
Regularization and sampling cap fixed before evaluation; no holdout tuning.
"""
import csv, json, hashlib, subprocess, os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
TMP=ROOT/'.joint-v3-work'; TMP.mkdir(exist_ok=True)
RIDGE=0.01
TRAIN_CAP=12000
EVAL_CAP=6000
FIELDS=['score','win_rate','top3_rate','b_count','nige_count','makuri_count','sashi_count','mark_count','age']
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def number(x):
    try:
        v=float(str(x).replace('%',''))
        return v if np.isfinite(v) and v>=0 else None
    except (ValueError,TypeError):return None
def folders(cls):
    if cls=='A':return sorted((ROOT/'data/a_class_12').glob('20*'))
    return sorted((ROOT/'data').glob('20*/s_class_f1_all_parts/*'))+[ROOT/'data/2026_h1/s_class_f1_all']
def load(cls):
    found={}; excluded=Counter()
    for d in folders(cls):
        if not all((d/f).exists() for f in ['races.csv','entries.csv','results.csv','payouts.csv']):continue
        cards={r['race_id']:r for r in rows(d/'races.csv') if r['race_date']<='2026-06-30' and r.get('entry_count')=='7'}
        ent=defaultdict(list); result=defaultdict(list); payouts={};invalid=set()
        for e in rows(d/'entries.csv'):
            rid=e['race_id']
            if rid not in cards:continue
            vals={k:number(e.get(k)) for k in FIELDS}
            if any(v is None for v in vals.values()) or not vals['score'] or not vals['age'] or e.get('style') not in ['逃','両','追']:
                invalid.add(rid);continue
            if sum(number(e.get(k)) or 0 for k in ['first_count','second_count','third_count','outside_count'])==0:
                invalid.add(rid);continue
            aliases={'b_count':'b','nige_count':'nige','makuri_count':'makuri','sashi_count':'sashi','mark_count':'mark'}
            ent[rid].append({'no':int(e['car_no']),'style':e['style'],**{aliases.get(k,k):v for k,v in vals.items()}})
        for z in rows(d/'results.csv'):
            rid=z['race_id']
            if rid not in cards:continue
            if any(s in str(z.get('finish_text',''))+str(z.get('finish_position','')) for s in ['落','失','棄','欠','故']):invalid.add(rid)
            try:
                pos=int(z['finish_position']);car=int(z['car_no'])
                if 1<=pos<=3:result[rid].append((pos,car))
            except (ValueError,TypeError):invalid.add(rid)
        for p in rows(d/'payouts.csv'):
            if p['race_id'] in cards and p.get('bet_code')=='trifecta' and p.get('status')=='paid':
                payouts[p['race_id']]=number(p['payout_yen'])
        for rid,c in cards.items():
            if rid in invalid:excluded['invalid_card_or_incident']+=1;continue
            if len(ent[rid])!=7 or len({e['no'] for e in ent[rid]})!=7:excluded['entries']+=1;continue
            rr=result[rid]
            if len(rr)!=3 or {p for p,_ in rr}!={1,2,3}:excluded['finish']+=1;continue
            try:
                lines=[[int(x) for x in l.split('-')] for l in c['predicted_line_formation'].split('/')]
                if len(lines)<2 or sorted(sum(lines,[]))!=sorted(e['no'] for e in ent[rid]):raise ValueError()
            except (ValueError,KeyError):excluded['lines']+=1;continue
            actual=[car for _,car in sorted(rr)]
            found[rid]={'input':{'race_id':rid,'race_type':c['race_type'],'meeting_grade':c['meeting_grade'],'lines':lines,'entries':sorted(ent[rid],key=lambda e:e['no'])},
                        'target':actual,'date':c['race_date'],'payout':payouts.get(rid)}
    return list(found.values()),dict(excluded)
def sample(data,cap):return sorted(data,key=lambda r:hashlib.sha256(r['input']['race_id'].encode()).hexdigest())[:cap]
def matrix(data,prefix):
    path=TMP/(prefix+'.json');path.write_text(json.dumps(data,ensure_ascii=False))
    out=TMP/prefix
    subprocess.run(['node',str(Path(__file__).with_name('features.mjs')),str(path),str(out)],check=True)
    sh=json.loads(Path(str(out)+'.shape.json').read_text());n=sh['races']
    xp=np.fromfile(str(out)+'.pair.f32',np.float32).reshape(n,42,sh['pairFeatures']).astype(np.float64)
    xt=np.fromfile(str(out)+'.third.f32',np.float32).reshape(n,5,sh['thirdFeatures']).astype(np.float64)
    yp=np.fromfile(str(out)+'.yp.i32',np.int32);yt=np.fromfile(str(out)+'.yt.i32',np.int32)
    return xp,xt,yp,yt,sh
def fit(x,y):
    n,_,d=x.shape;ii=np.arange(n)
    def fg(w):
        z=x@w;logz=logsumexp(z,axis=1);p=np.exp(z-logz[:,None]);p[ii,y]-=1
        return np.mean(logz-z[ii,y])+RIDGE*np.dot(w,w)/2,np.einsum('nc,ncd->d',p,x,optimize=True)/n+RIDGE*w
    res=minimize(fg,np.zeros(d),jac=True,method='L-BFGS-B',options={'maxiter':120,'ftol':1e-9})
    return res.x,{'success':bool(res.success),'iterations':int(res.nit),'objective':float(res.fun),'message':str(res.message)}
def nll(x,y,w):
    z=x@w
    return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
def evaluate_cards(data,models,prefix):
    path=TMP/(prefix+'-eval.json');path.write_text(json.dumps(data,ensure_ascii=False))
    modelpath=TMP/(prefix+'-bundle.json');modelpath.write_text(json.dumps(models))
    out=TMP/(prefix+'-eval-out.json')
    subprocess.run(['node',str(Path(__file__).with_name('evaluate.mjs')),str(path),str(modelpath),str(out)],check=True)
    return json.loads(out.read_text())
def main():
    bundle={'version':'conditional-pair-v3','trainThrough':'2024-12-31','models':{},'regularization':RIDGE,
            'scope':'A1/A2 and S trained separately; 7-car races; incidents excluded','features':'joint_model.mjs featurePack; pair 53 / third 29',
            'sampling':'SHA256(race_id), train cap 12000/class, eval cap 6000/class/period; no post-holdout tuning'}
    report={'protocol':{k:v for k,v in bundle.items() if k!='models'},'classes':{}}
    all_data={}
    for cls in ['A','S']:
        data,excluded=load(cls);all_data[cls]=data
        train=sample([r for r in data if r['date']<'2025-01-01'],TRAIN_CAP)
        print(cls,'eligible',len(data),'train',len(train),flush=True)
        if len(train)<1000:raise RuntimeError('insufficient historical cards')
        xp,xt,yp,yt,sh=matrix(train,cls+'-train')
        wp,info=fit(xp,yp);wb,baseinfo=fit(xp[:,:,:sh['baseFeatures']],yp);wt,tinfo=fit(xt,yt)
        if not all(x['success'] for x in [info,baseinfo,tinfo]):raise RuntimeError('optimizer did not converge')
        base=np.pad(wb,(0,len(wp)-len(wb)))
        bundle['models'][cls]={'pair':wp.tolist(),'third':wt.tolist(),'baselinePair':base.tolist(),'trainRaces':len(train),'fit':{'pair':info,'baseline':baseinfo,'third':tinfo}}
        report['classes'][cls]={'eligible':len(data),'excluded':excluded,'trainRaces':len(train),'trainDateRange':[min(r['date'] for r in train),max(r['date'] for r in train)],'holdouts':{}}
        for name,start,end in [('2025','2025-01-01','2025-12-31'),('2026_h1','2026-01-01','2026-06-30')]:
            eligible=[r for r in data if start<=r['date']<=end];test=sample(eligible,EVAL_CAP)
            print(cls,name,'eligible',len(eligible),'evaluated',len(test),flush=True)
            xpt,xtt,ypt,ytt,_=matrix(test,cls+'-'+name)
            report['classes'][cls]['holdouts'][name]={'eligible':len(eligible),'evaluated':len(test),'pairNLL':nll(xpt,ypt,wp),'baselinePairNLL':nll(xpt,ypt,base),
                'thirdConditionalNLL':nll(xtt,ytt,wt),'jointNLL':nll(xpt,ypt,wp)+nll(xtt,ytt,wt),
                'baselineJointNLL':nll(xpt,ypt,base)+nll(xtt,ytt,wt),**evaluate_cards(test,bundle,cls+'-'+name)}
        bundle['models'][cls]['validationSummary']={period:{'evaluated':h['evaluated'],'pairNLL':h['pairNLL'],
            'baselinePairNLL':h['baselinePairNLL'],'selected':h['joint']['selected'],'hitRate':h['joint']['hitRate'],
            'returnRatio':h['joint']['returnRatio']} for period,h in report['classes'][cls]['holdouts'].items()}
        del xp,xt
    (ROOT/'docs/cross-lab/joint_model_frozen.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'research/cross_joint_v3/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
