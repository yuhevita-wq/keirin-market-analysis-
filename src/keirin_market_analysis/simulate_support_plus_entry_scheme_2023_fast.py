from __future__ import annotations
import csv,itertools,json,math,random,statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'2023'/'s_class_yosen'
OUT=ROOT/'data'/'audits'/'support_plus_entry_scheme_2023_v2.json'
NUM=['score','win_rate','top2_rate','top3_rate','b_count','nige_count','makuri_count','sashi_count','mark_count','age','line_size','line_position','car_no']
CAT=['class','style','line_role']
RIDGE=0.02
EPOCHS=18
CAL=(5,6)
EVAL=(7,8,9,10,11,12)

def rows(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sf(v,d=0.0):
    try:
        x=float(str(v).replace(',','').strip());return x if math.isfinite(x) else d
    except:return d
def combo(s):
    try:return tuple(sorted(int(x) for x in str(s or '').replace('=','-').replace(',','-').split('-') if x.strip()))
    except:return ()
def clip(p):return min(1-1e-9,max(1e-9,p))
def logit(p):
    p=clip(p);return math.log(p/(1-p))
def sig(z):
    if z>=0:
        e=math.exp(-z);return 1/(1+e)
    e=math.exp(z);return e/(1+e)
def qn(xs,q):
    xs=sorted(xs);return xs[max(0,math.ceil(q*len(xs))-1)] if xs else None

def load():
    ent=defaultdict(dict)
    for r in rows(DATA/'entries.csv'):
        try:c=int(r['car_no'])
        except:continue
        ent[str(r['race_id'])][c]=r
    od=defaultdict(list);dates={}
    for r in rows(DATA/'trio_final_odds.csv'):
        if r.get('odds_status')!='available':continue
        c=combo(r.get('combination'));o=sf(r.get('odds'),-1);rid=str(r.get('race_id',''))
        if len(c)==3 and o>0:
            od[rid].append((c,o))
            try:dates[rid]=date.fromisoformat(r['race_date'])
            except:pass
    wins=defaultdict(list)
    for r in rows(DATA/'payouts.csv'):
        if r.get('ticket_type')=='3連複' and r.get('status')=='paid':
            c=combo(r.get('combination'))
            if len(c)==3:wins[str(r['race_id'])].append(c)
    out=[];acc={'market_races':len(od),'excluded_market':0,'excluded_entries':0,'excluded_winner':0}
    for rid,rs in od.items():
        riders=sorted({x for c,_ in rs for x in c})
        if len(riders)!=7:acc['excluded_market']+=1;continue
        cs=list(itertools.combinations(riders,3));oo={c:o for c,o in rs}
        if set(oo)!=set(cs):acc['excluded_market']+=1;continue
        if set(ent.get(rid,{}))!=set(riders):acc['excluded_entries']+=1;continue
        if len(wins.get(rid,[]))!=1:acc['excluded_winner']+=1;continue
        if rid not in dates:continue
        iv={c:1/oo[c] for c in cs};z=sum(iv.values());m={c:iv[c]/z for c in cs}
        S={i:sum(v for c,v in m.items() if i in c) for i in riders};wset=set(wins[rid][0])
        ps=[{'car':i,'row':ent[rid][i],'s':S[i],'y':int(i in wset)} for i in riders]
        out.append({'race_id':rid,'date':dates[rid],'persons':ps,'win':wset})
    out.sort(key=lambda r:(r['date'],r['race_id']));acc['eligible_complete_races']=len(out);return out,acc

def encoder(train):
    vals={k:[] for k in NUM};lev={k:set() for k in CAT}
    for r in train:
        for p in r['persons']:
            for k in NUM:vals[k].append(sf(p['row'].get(k)))
            for k in CAT:lev[k].add(str(p['row'].get(k,'') or '').strip())
    mu={k:statistics.mean(v) for k,v in vals.items()};sd={}
    for k,v in vals.items():
        x=statistics.pstdev(v);sd[k]=x if x>1e-12 else 1.0
    levels={k:sorted(v) for k,v in lev.items()};names=['intercept']+NUM[:]
    for k in CAT:names += [f'{k}={v}' for v in levels[k]]
    return {'mu':mu,'sd':sd,'levels':levels,'names':names}
def vec(row,e):
    x=[1.0]+[(sf(row.get(k))-e['mu'][k])/e['sd'][k] for k in NUM]
    for k in CAT:
        v=str(row.get(k,'') or '').strip();x += [1.0 if v==lv else 0.0 for lv in e['levels'][k]]
    return x

def fit(train,seed):
    e=encoder(train);sam=[]
    for r in train:
        for p in r['persons']:sam.append((vec(p['row'],e),logit(p['s']),p['y']))
    w=[0.0]*len(e['names']);rng=random.Random(seed);n=max(1,len(sam))
    for ep in range(EPOCHS):
        rng.shuffle(sam);lr=0.025/(1+0.12*ep)
        for x,off,y in sam:
            pr=sig(off+sum(a*b for a,b in zip(w,x)));err=pr-y
            for j,xj in enumerate(x):
                reg=0.0 if j==0 else RIDGE*w[j]/n
                w[j]-=lr*(err*xj+reg)
    return e,w

def pred_race(r,e,w):
    zz={}
    for p in r['persons']:
        x=vec(p['row'],e);zz[p['car']]=logit(p['s'])+sum(a*b for a,b in zip(w,x))
    lo,hi=-12.0,12.0
    for _ in range(60):
        mid=(lo+hi)/2
        if sum(sig(z+mid) for z in zz.values())>3:hi=mid
        else:lo=mid
    sh=(lo+hi)/2;F={c:sig(z+sh) for c,z in zz.items()};S={p['car']:p['s'] for p in r['persons']};E={c:F[c]-S[c] for c in F}
    return {'race':r,'s':S,'f':F,'e':E,'dmax':max(abs(v) for v in E.values()),'dtotal':sum(abs(v) for v in E.values())}
def walk(races):
    out=[];mods={}
    for mo in range(5,13):
        tr=[r for r in races if r['date'].month<mo];te=[r for r in races if r['date'].month==mo]
        if not tr or not te:continue
        e,w=fit(tr,2023+mo);mods[str(mo)]={'train_races':len(tr),'test_races':len(te),'features':len(w)}
        out += [pred_race(r,e,w) for r in te]
    return out,mods

def metrics(ps):
    if not ps:return {}
    bs=bf=ls=lf=0.0;n=0;es=ef=os=of=0
    for z in ps:
        yset=z['race']['win']
        for p in z['race']['persons']:
            c=p['car'];y=p['y'];a=clip(z['s'][c]);b=clip(z['f'][c]);bs+=(a-y)**2;bf+=(b-y)**2;ls-=y*math.log(a)+(1-y)*math.log(1-a);lf-=y*math.log(b)+(1-y)*math.log(1-b);n+=1
        ts=set(sorted(z['s'],key=lambda c:(-z['s'][c],c))[:3]);tf=set(sorted(z['f'],key=lambda c:(-z['f'][c],c))[:3]);es+=int(ts==yset);ef+=int(tf==yset);os+=len(ts&yset);of+=len(tf&yset)
    r=len(ps)
    return {'races':r,'support_brier':bs/n,'adjusted_brier':bf/n,'brier_improvement_pct':100*(bs-bf)/bs,'support_logloss':ls/n,'adjusted_logloss':lf/n,'logloss_improvement_pct':100*(ls-lf)/ls,'support_top3_exact_rate_pct':100*es/r,'adjusted_top3_exact_rate_pct':100*ef/r,'support_avg_top3_overlap':os/r,'adjusted_avg_top3_overlap':of/r}

def main():
    races,acc=load();ps,mods=walk(races);cal=[p for p in ps if p['race']['date'].month in CAL];ev=[p for p in ps if p['race']['date'].month in EVAL]
    tdmax=qn([p['dmax'] for p in cal],.75);tdtot=qn([p['dtotal'] for p in cal],.75);pe=[v for p in cal for v in p['e'].values() if v>0];te=qn(pe,.5)
    nplus=lambda p:sum(v>te for v in p['e'].values())
    enter=[p for p in ev if p['dmax']>=tdmax and p['dtotal']>=tdtot and nplus(p)>=2];non=[p for p in ev if p not in enter]
    monthly={}
    for mo in EVAL:
        a=[p for p in ev if p['race']['date'].month==mo];b=[p for p in enter if p['race']['date'].month==mo]
        monthly[str(mo)]={'eligible':len(a),'must_enter':len(b),'rate_pct':100*len(b)/len(a) if a else None,'metrics':metrics(b)}
    out={'status':'SUPPORT_PLUS_ENTRY_SCHEME_2023_V2','scope':'Race-entry simulation only; no ticket rule, no ROI.','data_note':'Support uses archived KDreams final trio odds; historical diagnostic, not T-10 executable.','design':{'baseline':'3連複 rider support S','model':'offset logistic logit(F)=logit(S)+race-card correction, normalized to sum F=3','numeric_features':NUM,'categorical_features':CAT,'excluded':['prediction_mark','evaluation'],'walk_forward':'Jan-Apr bootstrap; monthly refit May-Dec using prior outcomes only','calibration':'May-Jun outputs only; thresholds do not use outcomes','evaluation':'Jul-Dec','must_enter':'Dmax>=May-Jun Q75 AND Dtotal>=May-Jun Q75 AND Nplus>=2','Dmax':'max abs(F-S)','Dtotal':'sum abs(F-S)','Nplus_threshold':'median positive E=F-S in May-Jun'},'accounting':acc,'models':mods,'calibration':{'races':len(cal),'dmax_q75':tdmax,'dtotal_q75':tdtot,'positive_e_median':te},'evaluation':{'eligible_races':len(ev),'must_enter_races':len(enter),'must_enter_rate_pct':100*len(enter)/len(ev) if ev else None,'all_metrics':metrics(ev),'must_enter_metrics':metrics(enter),'non_enter_metrics':metrics(non),'avg_dmax':statistics.mean(p['dmax'] for p in enter) if enter else None,'avg_dtotal':statistics.mean(p['dtotal'] for p in enter) if enter else None,'avg_nplus':statistics.mean(nplus(p) for p in enter) if enter else None,'monthly':monthly},'pass_rule':'Proceed to 2-point formation only if adjusted F improves BOTH Brier and logloss vs S on MUST_ENTER subset.'}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
