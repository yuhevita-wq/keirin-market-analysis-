"""Train S9/A3 scopes separately, joint ordered pair then conditional third.
Frozen protocol: <=2024 train; 2025 and 2026H1 holdouts; no target-day results.
"""
import csv,io,json,zipfile,hashlib,subprocess,os
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'.extended-work';WORK.mkdir(exist_ok=True)
FIELDS=['score','age','win_rate','top3_rate','b_count','nige_count','makuri_count','sashi_count','mark_count']
ALIAS={'b_count':'b','nige_count':'nige','makuri_count':'makuri','sashi_count':'sashi','mark_count':'mark'}
def csvrows(raw):return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
def sources(scope):
 if scope=='S9':
  for p in sorted((ROOT/'data/grade_races').rglob('*.zip')):
   with zipfile.ZipFile(p) as z:
    if not all(n in z.namelist() for n in ['races.csv','entries.csv','results.csv']):continue
    yield {n:csvrows(z.read(n+'.csv')) for n in ['races','entries','results']}
 else:
  for p in sorted((ROOT/'data/a3_challenge').glob('*')):
   if not all((p/(n+'.csv')).exists() for n in ['races','entries','results']):continue
   yield {n:csvrows((p/(n+'.csv')).read_bytes()) for n in ['races','entries','results']}
def data(scope):
 out={};bad=Counter();n=9 if scope=='S9' else 7
 for batch in sources(scope):
  cards={r['race_id']:r for r in batch['races'] if '2022-01-01'<=r.get('race_date','')<='2026-06-30'}
  ent=defaultdict(list);res=defaultdict(list);invalid=set()
  for e in batch['entries']:
   rid=e['race_id']
   if rid not in cards:continue
   try:
    vals={ALIAS.get(k,k):float(str(e[k]).replace('%','')) for k in FIELDS}
    if not all(np.isfinite(v) and v>=0 for v in vals.values()) or vals['score']<=0 or vals['age']<=0 or e['style'] not in ['逃','両','追']:raise ValueError()
    if sum(float(e.get(k) or 0) for k in ['first_count','second_count','third_count','outside_count'])==0:raise ValueError()
    ent[rid].append(dict(no=int(e['car_no']),style=e['style'],**{'class':e['class']},line_id=int(e['line_id']),line_position=int(e['line_position']),**vals))
   except (ValueError,KeyError):invalid.add(rid)
  for r in batch['results']:
   rid=r['race_id']
   if rid not in cards:continue
   if any(s in str(r.get('finish_text',''))+str(r.get('finish_position','')) for s in ['落','失','棄','欠','故']):invalid.add(rid)
   try:
    p=int(r['finish_position']);no=int(r['car_no'])
    if 1<=p<=3:res[rid].append((p,no))
   except (ValueError,KeyError):invalid.add(rid)
  for rid,c in cards.items():
   es=ent[rid];rs=res[rid]
   if rid in invalid or len(es)!=n or len(set(e['no'] for e in es))!=n or len(rs)!=3 or set(p for p,_ in rs)!={1,2,3}:bad['invalid']+=1;continue
   if scope=='S9' and (not all(e['class'] in ['SS','S1','S2'] for e in es) or c.get('meeting_grade') not in ['G1','G2','G3']):continue
   if scope=='A3' and not all(e['class']=='A3' for e in es):continue
   lines=defaultdict(list)
   for e in es:lines[e['line_id']].append(e)
   if any(e['line_id']<=0 or e['line_position']<=0 for e in es) or len(lines)<2:continue
   ls=[[e['no'] for e in sorted(arr,key=lambda e:e['line_position'])] for arr in lines.values()]
   out[rid]={'input':{'race_id':rid,'race_type':'S級' if scope=='S9' else 'A級チャレンジ','meeting_grade':c.get('meeting_grade'),'entries':es,'lines':ls},'target':[no for p,no in sorted(rs)],'date':c['race_date']}
 return list(out.values()),dict(bad)
def sample(ds,n):return sorted(ds,key=lambda r:hashlib.sha256(r['input']['race_id'].encode()).digest())[:n]
def matrix(ds):
 allx=[]
 for i in range(0,len(ds),250):
  p=WORK/'batch';(WORK/'input.json').write_text(json.dumps(ds[i:i+250],ensure_ascii=False))
  subprocess.run(['node',str(Path(__file__).with_name('features.mjs')),str(WORK/'input.json'),str(p)],check=True)
  s=json.loads(p.with_suffix('.json').read_text());n=s['n']
  allx.append([np.fromfile(str(p)+'.pair',np.float32).reshape(n,s['pairs'],s['pf']),np.fromfile(str(p)+'.third',np.float32).reshape(n,s['rest'],s['tf']),np.fromfile(str(p)+'.yp',np.int32),np.fromfile(str(p)+'.yt',np.int32)])
 return [np.concatenate([z[k] for z in allx]).astype(np.float64 if k<2 else np.int32) for k in range(4)]
def fit(x,y):
 def fg(w):
  z=x@w;l=logsumexp(z,axis=1);p=np.exp(z-l[:,None]);p[np.arange(len(y)),y]-=1
  return np.mean(l-z[np.arange(len(y)),y])+.005*(w@w),np.einsum('nc,ncd->d',p,x,optimize=True)/len(y)+.01*w
 r=minimize(fg,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':160,'ftol':1e-9})
 if not r.success:raise RuntimeError(str(r.message))
 return r.x,{'converged':bool(r.success),'iterations':int(r.nit)}
def nll(x,y,w):
 z=x@w;return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
bundle={'version':'conditional-pair-extended-v1','trainThrough':'2024-12-31','models':{},'protocol':{'ridge':.01,'trainCap':3000,'holdoutCap':1500,'sampling':'SHA256(race_id)','targetDayUsed':False,'profitValidated':False}}
report={}
for scope in ['S9','A3']:
 ds,bad=data(scope);train=sample([r for r in ds if r['date']<'2025-01-01'],3000)
 print(scope,'eligible',len(ds),'train',len(train),flush=True)
 if len(train)<500:raise RuntimeError('insufficient training data')
 xp,xt,yp,yt=matrix(train);wp,fp=fit(xp,yp);wt,ft=fit(xt,yt)
 bundle['models'][scope]={'carCount':len(train[0]['input']['entries']),'pair':wp.tolist(),'third':wt.tolist(),'trainRaces':len(train),'fit':{'pair':fp,'third':ft}}
 report[scope]={'eligible':len(ds),'excluded':bad,'trainRaces':len(train),'trainStart':min(r['date'] for r in train),'trainEnd':max(r['date'] for r in train),'holdouts':{}}
 del xp,xt,yp,yt
 for period,start,end in [('2025','2025-01-01','2025-12-31'),('2026H1','2026-01-01','2026-06-30')]:
  test=sample([r for r in ds if start<=r['date']<=end],1500)
  if not test:raise RuntimeError('missing holdout')
  xp,xt,yp,yt=matrix(test);pn=nll(xp,yp,wp);tn=nll(xt,yt,wt)
  report[scope]['holdouts'][period]={'races':len(test),'pairNLL':pn,'conditionalThirdNLL':tn,'jointNLL':pn+tn,'uniformJointNLL':float(np.log(xp.shape[1]*xt.shape[1]))}
  print(scope,period,report[scope]['holdouts'][period],flush=True)
  del xp,xt,yp,yt
bundle['validation']=report
(ROOT/'docs/cross-lab/extended_joint_frozen.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
(Path(__file__).with_name('validation.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
