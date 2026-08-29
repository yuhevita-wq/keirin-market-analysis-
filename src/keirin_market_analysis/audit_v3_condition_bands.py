from __future__ import annotations

import json
from pathlib import Path

from .audit_v3_failure_decomposition import PERIODS, load_period
from .simulate_mainline_v1 import choose_main_line
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import feature_row, formations

OUT = Path('data/audits/v3_condition_band_audit.json')

EARLY_RULES = []
for top2 in (50, 60, 70, 80):
    for gap in (0, 2, 4):
        EARLY_RULES.append((f'main_top2_ge_{top2}__leader_gap_le_{gap}', lambda r,t=top2,g=gap: float(r['main_top2'])>=t and float(r['leader_score_gap'])<=g))
for top3 in (80, 90, 100, 110):
    for gap in (0, 2, 4):
        EARLY_RULES.append((f'main_top3_ge_{top3}__leader_gap_le_{gap}', lambda r,t=top3,g=gap: float(r['main_top3'])>=t and float(r['leader_score_gap'])<=g))
for a in (102, 104, 106):
    for wg in (-10, 0, 10):
        EARLY_RULES.append((f'a_score_le_{a}__pair_win_gap_le_{wg}', lambda r,a=a,wg=wg: float(r['a_score'])<=a and float(r['pair_win_gap'])<=wg))

LATE_RULES = []
for wg in (-20, -10, 0):
    LATE_RULES.append((f'pair_win_gap_le_{wg}', lambda r,wg=wg: float(r['pair_win_gap'])<=wg))
    for tg in (0, 10, 25, 40):
        LATE_RULES.append((f'pair_win_gap_le_{wg}__pair_top3_gap_le_{tg}', lambda r,wg=wg,tg=tg: float(r['pair_win_gap'])<=wg and float(r['pair_top3_gap'])<=tg))
    for mt2 in (70, 80, 90):
        LATE_RULES.append((f'pair_win_gap_le_{wg}__main_top2_le_{mt2}', lambda r,wg=wg,mt2=mt2: float(r['pair_win_gap'])<=wg and float(r['main_top2'])<=mt2))
    for rt2 in (70, 80, 90):
        LATE_RULES.append((f'pair_win_gap_le_{wg}__rival_top2_le_{rt2}', lambda r,wg=wg,rt2=rt2: float(r['pair_win_gap'])<=wg and float(r['rival_top2'])<=rt2))


def fin(rows):
    n=len(rows); st=sum(x[0] for x in rows); py=sum(x[1] for x in rows); hp=[x[1] for x in rows if x[1]>0]
    return {'races':n,'hits':len(hp),'stake_yen':st,'payout_yen':py,'profit_yen':py-st,'roi':py/st if st else 0.0,'hit_rate':len(hp)/n if n else 0.0,'top1_payout_share':max(hp)/py if hp and py else 0.0}


def build(segment, form):
    out={p:[] for p in PERIODS}
    for period,path in PERIODS.items():
        races,eb,tri,_rb,seg=load_period(path)
        for race in races:
            rid=race['race_id']
            if seg.get(rid)!=segment: continue
            es=eb[rid]; chosen=choose_main_line(es)
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(es,mid)
            if not rival or len(rival)<2: continue
            row=feature_row(race,es,mid,main,rival)
            bets=formations(row,es)[form]
            out[period].append((row,100*len(bets),sum(tri[rid].get(b,0) for b in bets)))
    return out


def audit(data,rules):
    rows=[]
    for name,pred in rules:
        ps={}
        for p in PERIODS:
            ps[p]=fin([(st,py) for r,st,py in data[p] if pred(r)])
        rois=[ps[p]['roi'] for p in PERIODS]
        rows.append({'rule':name,'periods':ps,'profitable_periods':sum(x>1 for x in rois),'worst_period_roi':min(rois),'min_races':min(ps[p]['races'] for p in PERIODS),'combined_roi':sum(ps[p]['payout_yen'] for p in PERIODS)/sum(ps[p]['stake_yen'] for p in PERIODS) if sum(ps[p]['stake_yen'] for p in PERIODS) else 0.0})
    rows.sort(key=lambda x:(x['profitable_periods'],x['worst_period_roi'],x['min_races'],x['combined_roi']),reverse=True)
    return rows


def main():
    protocol=json.loads(Path('data/audits/v3_research_protocol.json').read_text(encoding='utf-8'))
    assert protocol['future_forward_oos']=='2026-07-01 onward'
    out={'status':'V3_CONDITION_BAND_AUDIT_COMPLETE','note':'focused coarse neighborhood audit of recurring early and late clusters; no 2026 H2 read','early':audit(build('前半','MIX2'),EARLY_RULES),'late':audit(build('後半','MAIN4_X'),LATE_RULES)}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'early':out['early'][:12],'late':out['late'][:15]},ensure_ascii=False,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
