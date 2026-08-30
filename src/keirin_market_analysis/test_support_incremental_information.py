from __future__ import annotations

import csv, itertools, json, math, statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data'/'audits'/'support_incremental_information_2023dev_forward.json'
DATASETS={
 '2023':ROOT/'data'/'2023'/'s_class_yosen',
 '2024':ROOT/'data'/'2024'/'s_class_yosen',
 '2025':ROOT/'data'/'2025'/'s_class_yosen',
 '2026_h1':ROOT/'data'/'2026_h1'/'s_class_yosen',
}

def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def combo(s):
    s=str(s).strip().replace('=','-').replace(',','-')
    try:return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except:return ()

def quantile(xs,q):
    a=sorted(xs); p=(len(a)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(p-lo)

def q_from_weights(combos,weights):
    raw={c:math.prod(weights[x] for x in c) for c in combos}; z=sum(raw.values())
    return {c:v/z for c,v in raw.items()}

def fit_maxent(riders,combos,target,tol=1e-12,max_iter=20000):
    w={r:1.0 for r in riders}
    for it in range(max_iter):
        for r in riders:
            q=q_from_weights(combos,w); cur=sum(v for c,v in q.items() if r in c); t=target[r]
            f=(t*(1-cur))/(cur*(1-t)); w[r]*=f
        g=math.exp(sum(math.log(max(w[r],1e-300)) for r in riders)/len(riders))
        for r in riders:w[r]/=g
        q=q_from_weights(combos,w)
        err=max(abs(sum(v for c,v in q.items() if r in c)-target[r]) for r in riders)
        if err<tol:return q,err
    return q,err

def load_year(name):
    data=DATASETS[name]
    by=defaultdict(list)
    for r in read_csv(data/'trio_final_odds.csv'):
        if r.get('odds_status')!='available':continue
        try:o=float(r['odds'])
        except:continue
        c=combo(r.get('combination',''))
        if len(c)==3 and o>0:by[str(r['race_id'])].append((c,o))
    paid=defaultdict(set)
    for r in read_csv(data/'payouts.csv'):
        if r.get('ticket_type')=='3連複' and r.get('status')=='paid':
            c=combo(r.get('combination',''))
            if len(c)==3:paid[str(r['race_id'])].add(c)
    races=[]; excluded=0; multi=0; solver_fail=0
    for rid,rows in sorted(by.items()):
        riders=sorted({x for c,_ in rows for x in c})
        if len(riders)!=7:excluded+=1;continue
        combos=list(itertools.combinations(riders,3)); odds={c:o for c,o in rows}
        if set(odds)!=set(combos):excluded+=1;continue
        if len(paid.get(rid,set()))!=1:multi+=1;continue
        win=next(iter(paid[rid]))
        if win not in odds:excluded+=1;continue
        inv={c:1/o for c,o in odds.items()}; z=sum(inv.values()); m={c:v/z for c,v in inv.items()}
        support={r:sum(v for c,v in m.items() if r in c) for r in riders}
        q,err=fit_maxent(riders,combos,support)
        if err>=1e-9:solver_fail+=1;continue
        R={c:m[c]/q[c] for c in combos}
        races.append({'race_id':rid,'combos':combos,'m':m,'R':R,'win':win})
    return races,{'market_races':len(by),'excluded_non7_or_incomplete':excluded,'multi_or_nonunique_paid_trio':multi,'solver_failures':solver_fail,'evaluated_races':len(races)}

def loglik_beta(races,beta):
    s=0.0
    for r in races:
        vals={c:r['m'][c]*(r['R'][c]**beta) for c in r['combos']}; z=sum(vals.values()); p=vals[r['win']]/z
        s+=math.log(max(p,1e-300))
    return s

def fit_beta(races,lo=-5.0,hi=5.0,iters=120):
    gr=(math.sqrt(5)-1)/2
    a,b=lo,hi; c=b-gr*(b-a); d=a+gr*(b-a); fc=loglik_beta(races,c); fd=loglik_beta(races,d)
    for _ in range(iters):
        if fc>fd:
            b,d,fd=d,c,fc; c=b-gr*(b-a); fc=loglik_beta(races,c)
        else:
            a,c,fc=c,d,fd; d=a+gr*(b-a); fd=loglik_beta(races,d)
    beta=(a+b)/2
    return beta,loglik_beta(races,beta)

def probs(r,beta):
    vals={c:r['m'][c]*(r['R'][c]**beta) for c in r['combos']}; z=sum(vals.values())
    return {c:v/z for c,v in vals.items()}

def tv(a,b):return .5*sum(abs(a[c]-b[c]) for c in a)

def metrics(races,beta,must_threshold=None):
    selected=[]
    for r in races:
        p=probs(r,beta); distance=tv(p,r['m'])
        if must_threshold is not None and distance<must_threshold:continue
        selected.append((r,p,distance))
    if not selected:return {'races':0}
    ll=[]; brier=[]; top1=0; top3=0; tvs=[]
    for r,p,d in selected:
        ll.append(-math.log(max(p[r['win']],1e-300)))
        brier.append(sum((p[c]-(1.0 if c==r['win'] else 0.0))**2 for c in r['combos']))
        order=sorted(r['combos'],key=lambda c:(-p[c],c))
        top1+=r['win']==order[0]; top3+=r['win'] in order[:3]; tvs.append(d)
    return {'races':len(selected),'log_loss':statistics.mean(ll),'brier':statistics.mean(brier),'top1_hit_pct':100*top1/len(selected),'top3_hit_pct':100*top3/len(selected),'mean_tv_vs_market':statistics.mean(tvs)}

def market_metrics(races,must_threshold,beta):
    selected=[]
    for r in races:
        p1=probs(r,beta); d=tv(p1,r['m'])
        if must_threshold is not None and d<must_threshold:continue
        selected.append(r)
    if not selected:return {'races':0}
    ll=[]; brier=[]; top1=0; top3=0
    for r in selected:
        p=r['m']; ll.append(-math.log(max(p[r['win']],1e-300))); brier.append(sum((p[c]-(1.0 if c==r['win'] else 0.0))**2 for c in r['combos']))
        order=sorted(r['combos'],key=lambda c:(-p[c],c)); top1+=r['win']==order[0]; top3+=r['win'] in order[:3]
    return {'races':len(selected),'log_loss':statistics.mean(ll),'brier':statistics.mean(brier),'top1_hit_pct':100*top1/len(selected),'top3_hit_pct':100*top3/len(selected)}

def decile_calibration(races,cuts):
    bins=[{'expected_market_hits':0.0,'actual_hits':0,'candidate_rows':0} for _ in range(10)]
    for r in races:
        for c in r['combos']:
            x=r['R'][c]; idx=0
            while idx<9 and x>cuts[idx]:idx+=1
            b=bins[idx]; b['expected_market_hits']+=r['m'][c]; b['candidate_rows']+=1
            if c==r['win']:b['actual_hits']+=1
    out=[]
    for i,b in enumerate(bins):
        e=b['expected_market_hits']; out.append({'decile':i+1,**b,'actual_over_market_expected':b['actual_hits']/e if e else None})
    return out

def main():
    datasets={}; acct={}
    for y in DATASETS:
        datasets[y],acct[y]=load_year(y)
    dev=datasets['2023']
    beta,ll=fit_beta(dev)
    allR=[r['R'][c] for r in dev for c in r['combos']]
    cuts=[quantile(allR,i/10) for i in range(1,10)]
    dev_tvs=[tv(probs(r,beta),r['m']) for r in dev]
    must_threshold=quantile(dev_tvs,.75)
    evals={}
    for y,races in datasets.items():
        base_all=market_metrics(races,None,beta); aug_all=metrics(races,beta,None)
        base_must=market_metrics(races,must_threshold,beta); aug_must=metrics(races,beta,must_threshold)
        evals[y]={
          'all_races':{'market_baseline':base_all,'support_augmented':aug_all,'log_loss_improvement_pct':100*(base_all['log_loss']-aug_all['log_loss'])/base_all['log_loss'],'brier_improvement_pct':100*(base_all['brier']-aug_all['brier'])/base_all['brier']},
          'must_enter_races':{'market_baseline':base_must,'support_augmented':aug_must,'log_loss_improvement_pct':100*(base_must['log_loss']-aug_must['log_loss'])/base_must['log_loss'],'brier_improvement_pct':100*(base_must['brier']-aug_must['brier'])/base_must['brier']},
          'R_decile_calibration_using_2023_fixed_cuts':decile_calibration(races,cuts),
        }
    out={
      'status':'SUPPORT_INCREMENTAL_INFORMATION_TEST',
      'conceptual_note':'Support and R are deterministic transforms of the full 35-way market, so they cannot add information to a model that already uses the full 35-way vector without restriction. This test asks the practical question: do support-derived structural features improve prediction over using each combination own normalized market probability M alone?',
      'development':'2023 only',
      'forward_checks':['2024','2025','2026_h1'],
      'model_lock':{
        'baseline':'P0(c)=M(c), normalized inverse 3連複 odds.',
        'support_augmented':'P1(c) proportional to M(c) * R(c)^beta, where R=M/Q and Q is maximum entropy constrained by rider support marginals.',
        'beta_fit':'Single beta fitted on 2023 by maximizing race-level log likelihood; frozen unchanged afterward.',
        'fitted_beta':beta,
        'must_enter_definition':'A race is MUST_ENTER when total-variation distance TV(P1,M) is at or above the 2023 75th percentile. This identifies the top quartile of races where the support model materially disagrees with the raw market. Threshold frozen before 2024+ scoring.',
        'must_enter_tv_threshold':must_threshold,
        'R_decile_cuts_2023':cuts,
      },
      'accounting':acct,
      'evaluation':evals,
      'decision_rule':'Call support practically additive only if the frozen support-augmented model improves log loss and Brier versus M in the same direction across forward years, with special attention to MUST_ENTER races. Do not call a one-year improvement proof.',
      'project_rule_for_all_future_models':'Before any new model is evaluated on outcomes, define its MUST_ENTER race rule from pre-race observables/model outputs, freeze it, and report performance both on all eligible races and MUST_ENTER races. No post-result race inclusion/exclusion.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'beta':beta,'must_enter_tv_threshold':must_threshold,'accounting':acct,'evaluation':evals},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
