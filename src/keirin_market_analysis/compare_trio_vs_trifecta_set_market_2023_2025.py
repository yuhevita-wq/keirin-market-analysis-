from __future__ import annotations

import json, math
from pathlib import Path
from .analyze_odds_contradictions_2023_2025 import YEARS, load_results, load_odds, normalize_market

OUT=Path('data/audits/trio_vs_trifecta_set_market_2023_2025.json')
ALPHAS=(0.0,0.25,0.5,0.75,1.0)


def analyze_year(year):
    base=Path(f'data/{year}/s_class_yosen')
    res=load_results(base); trio=load_odds(base,'trio_final_odds.csv','='); tri3=load_odds(base,'trifecta_final_odds.csv','-')
    sums={a:{'n':0,'logloss':0.0,'brier':0.0,'top1':0,'top3':0,'top5':0} for a in ALPHAS}
    eligible=0
    for rid in sorted(set(res)&set(trio)&set(tri3)):
        wo=res[rid]; ws=tuple(sorted(wo)); to=trio[rid]; oo=tri3[rid]
        cars=sorted({c for s in to for c in s})
        if len(cars)<5 or len(to)!=(len(cars)*(len(cars)-1)*(len(cars)-2)//6) or len(oo)!=(len(cars)*(len(cars)-1)*(len(cars)-2)): continue
        qt=normalize_market(to); qo=normalize_market(oo)
        if ws not in qt: continue
        qs={s:0.0 for s in qt}
        for o,p in qo.items(): qs[tuple(sorted(o))]+=p
        if abs(sum(qs.values())-1)>1e-8: continue
        eligible+=1
        for a in ALPHAS:
            p={s:(1-a)*qt[s]+a*qs[s] for s in qt}
            x=sums[a]; x['n']+=1
            x['logloss'] += -math.log(max(p[ws],1e-15))
            x['brier'] += sum((v-(1.0 if s==ws else 0.0))**2 for s,v in p.items())
            ranked=sorted(p,key=lambda s:(-p[s],s))
            rank=ranked.index(ws)+1
            x['top1'] += rank<=1; x['top3'] += rank<=3; x['top5'] += rank<=5
    out={}
    for a,x in sums.items():
        n=x['n']
        out[str(a)]={
            'races':n,
            'mean_logloss':x['logloss']/n if n else None,
            'mean_brier':x['brier']/n if n else None,
            'top1_hit_pct':x['top1']/n*100 if n else None,
            'top3_hit_pct':x['top3']/n*100 if n else None,
            'top5_hit_pct':x['top5']/n*100 if n else None,
        }
    return {'year':year,'eligible_complete_races':eligible,'alpha_results':out}


def main():
    years=[analyze_year(y) for y in YEARS]
    # Combined weighted by races.
    combined={}
    for a in ALPHAS:
        k=str(a); n=sum(y['alpha_results'][k]['races'] for y in years)
        combined[k]={'races':n}
        for metric in ('mean_logloss','mean_brier','top1_hit_pct','top3_hit_pct','top5_hit_pct'):
            combined[k][metric]=sum(y['alpha_results'][k][metric]*y['alpha_results'][k]['races'] for y in years)/n
    out={
        'status':'TRIO_VS_TRIFECTA_SET_MARKET_SCORING_2023_2025',
        'years_read':list(YEARS),'evaluation_year_2026_used':False,'odds_phase':'final',
        'method':'alpha=0 is normalized 3連複 inverse-odds distribution; alpha=1 is normalized 3連単 distribution aggregated over six orders per 3-rider set; intermediate alpha is a linear probability blend. Lower logloss/Brier is better; higher top-k is better.',
        'years':years,'combined':combined
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
