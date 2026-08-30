from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'2023'/'s_class_yosen'
OUT=ROOT/'data'/'audits'/'support_maxent_residuals_2023.json'


def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def combo(s):
    s=str(s).strip().replace('=','-').replace(',','-')
    try:return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except:return ()


def quantile(xs,q):
    a=sorted(xs)
    if not a:return None
    p=(len(a)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    if lo==hi:return a[lo]
    return a[lo]+(a[hi]-a[lo])*(p-lo)


def summarize(xs):
    return {
        'n':len(xs),
        'mean':statistics.mean(xs) if xs else None,
        'median':statistics.median(xs) if xs else None,
        'q10':quantile(xs,.10),'q25':quantile(xs,.25),'q75':quantile(xs,.75),'q90':quantile(xs,.90),
        'min':min(xs) if xs else None,'max':max(xs) if xs else None,
    }


def q_from_weights(combos,weights):
    raw={c:math.prod(weights[x] for x in c) for c in combos}
    z=sum(raw.values())
    return {c:v/z for c,v in raw.items()}


def fit_maxent(riders, combos, target, tol=1e-12, max_iter=20000):
    weights={r:1.0 for r in riders}
    q=q_from_weights(combos,weights)
    for it in range(1,max_iter+1):
        for r in riders:
            q=q_from_weights(combos,weights)
            cur=sum(v for c,v in q.items() if r in c)
            t=target[r]
            if not (0<t<1) or not (0<cur<1):
                raise RuntimeError(f'bad marginal target/current rider={r} target={t} current={cur}')
            factor=(t*(1-cur))/(cur*(1-t))
            weights[r]*=factor
        # remove irrelevant common scale
        g=math.exp(sum(math.log(max(weights[r],1e-300)) for r in riders)/len(riders))
        for r in riders: weights[r]/=g
        q=q_from_weights(combos,weights)
        marg={r:sum(v for c,v in q.items() if r in c) for r in riders}
        err=max(abs(marg[r]-target[r]) for r in riders)
        if err<tol:
            return q,it,err,weights
    return q,max_iter,err,weights


def main():
    by_race=defaultdict(list)
    for r in read_csv(DATA/'trio_final_odds.csv'):
        if r.get('odds_status')!='available': continue
        try:o=float(r['odds'])
        except: continue
        c=combo(r.get('combination',''))
        if len(c)==3 and o>0: by_race[str(r['race_id'])].append((c,o))

    pattern_rows=defaultdict(list)
    race_stats=[]
    solver_fail=[]
    excluded_non7=0

    for rid,rows in sorted(by_race.items()):
        riders=sorted({x for c,_ in rows for x in c})
        if len(riders)!=7:
            excluded_non7+=1; continue
        combos=list(itertools.combinations(riders,3))
        odds={c:o for c,o in rows}
        if set(odds)!=set(combos):
            continue
        inv={c:1/o for c,o in odds.items()}
        z=sum(inv.values())
        m={c:v/z for c,v in inv.items()}
        support={r:sum(v for c,v in m.items() if r in c) for r in riders}
        ranked=sorted(riders,key=lambda r:(-support[r],r))
        rank={r:i+1 for i,r in enumerate(ranked)}
        try:
            q,it,err,w=fit_maxent(riders,combos,support)
        except Exception as e:
            solver_fail.append({'race_id':rid,'error':repr(e)})
            continue
        kl=sum(m[c]*math.log(m[c]/q[c]) for c in combos)
        tv=.5*sum(abs(m[c]-q[c]) for c in combos)
        abslog=[]
        for c in combos:
            pat=tuple(sorted(rank[x] for x in c))
            ratio=m[c]/q[c]
            lr=math.log(ratio)
            abslog.append(abs(lr))
            pattern_rows[pat].append({'ratio':ratio,'log_ratio':lr,'m':m[c],'q':q[c]})
        race_stats.append({'race_id':rid,'kl_m_q':kl,'tv_distance':tv,'mean_abs_log_ratio':statistics.mean(abslog),'solver_iterations':it,'max_marginal_error':err})

    patterns=[]
    for pat in itertools.combinations(range(1,8),3):
        rows=pattern_rows.get(pat,[])
        ratios=[r['ratio'] for r in rows]; logs=[r['log_ratio'] for r in rows]
        patterns.append({
            'support_rank_pattern':''.join(map(str,pat)),
            'races':len(rows),
            'ratio_M_over_Q':summarize(ratios),
            'log_ratio':summarize(logs),
            'geometric_mean_ratio':math.exp(statistics.mean(logs)) if logs else None,
            'pct_market_above_maxent':100*sum(x>0 for x in logs)/len(logs) if logs else None,
            'pct_ratio_ge_1_10':100*sum(x>=1.10 for x in ratios)/len(ratios) if ratios else None,
            'pct_ratio_le_0_90':100*sum(x<=0.90 for x in ratios)/len(ratios) if ratios else None,
            'mean_market_mass':statistics.mean(r['m'] for r in rows) if rows else None,
            'mean_maxent_mass':statistics.mean(r['q'] for r in rows) if rows else None,
        })

    by_positive=sorted(patterns,key=lambda x:x['geometric_mean_ratio'] if x['geometric_mean_ratio'] is not None else -1,reverse=True)
    by_negative=sorted(patterns,key=lambda x:x['geometric_mean_ratio'] if x['geometric_mean_ratio'] is not None else 999)
    out={
        'status':'SUPPORT_MAXENT_MARKET_DECOMPOSITION_2023',
        'year':2023,
        'scope':'7-rider exact S-class qualifying races only, so each market has exactly 35 trio combinations.',
        'uses_results_or_payouts':False,
        'source':'data/2023/s_class_yosen/trio_final_odds.csv',
        'definition':{
            'M':'normalized inverse-odds mass over the 35 actual trio prices',
            'support':'rider marginal of M; sum across riders = 3',
            'Q':'maximum-entropy distribution over 35 trios subject only to reproducing the seven rider-support marginals',
            'residual_ratio':'R = M / Q',
            'log_residual':'L = log(M/Q)',
            'interpretation':'R>1 means the actual market gives that support-rank trio more mass than rider marginals alone require; R<1 means less.'
        },
        'accounting':{
            'races_with_any_market':len(by_race),
            'excluded_non7_rider_races':excluded_non7,
            'analyzed_7_rider_races':len(race_stats),
            'solver_failures':len(solver_fail),
            'max_solver_marginal_error':max((r['max_marginal_error'] for r in race_stats),default=None),
        },
        'race_level_market_nonseparability':{
            'kl_M_to_Q':summarize([r['kl_m_q'] for r in race_stats]),
            'total_variation_M_vs_Q':summarize([r['tv_distance'] for r in race_stats]),
            'mean_abs_log_ratio':summarize([r['mean_abs_log_ratio'] for r in race_stats]),
        },
        'patterns_all_35':patterns,
        'top10_market_premium_by_geometric_mean_ratio':by_positive[:10],
        'top10_market_discount_by_geometric_mean_ratio':by_negative[:10],
        'solver_failure_examples':solver_fail[:10],
        'warning':'This is a market-structure decomposition only. It does not establish profitability, predictive accuracy, or fair value.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'accounting':out['accounting'],'race_level':out['race_level_market_nonseparability'],'premium':[(x['support_rank_pattern'],x['geometric_mean_ratio']) for x in by_positive[:10]],'discount':[(x['support_rank_pattern'],x['geometric_mean_ratio']) for x in by_negative[:10]]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
