from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from fake_favorite_formation_sim_2023 import (
    load_audited_gate, load_trio_odds, load_trifecta_odds, load_results,
    compute_delta, entrants_from_trio, norm_combo,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'all_pair_axis_search_2023.json'
STAKE = 100


def stats(pair, entrants, delta, trio, trifecta):
    others = [x for x in entrants if x not in pair]
    combos = [norm_combo(pair[0], pair[1], x) for x in others]
    vals = [delta[c] for c in combos if c in delta]
    pos = [v for v in vals if v > 0]
    ptrio = [1.0 / trio[c] for c in combos if c in trio and trio[c] > 0]
    ptri = []
    for c in combos:
        s = 0.0
        for perm in itertools.permutations(c):
            o = trifecta.get(perm)
            if o and o > 0: s += 1.0 / o
        ptri.append(s)
    return {'positive_count':len(pos),'positive_sum':sum(pos),'min_delta':min(vals) if vals else float('-inf'),'mean_delta':sum(vals)/len(vals) if vals else float('-inf'),'mean_ptrio':sum(ptrio)/len(ptrio) if ptrio else 0.0,'mean_ptrifecta_set':sum(ptri)/len(ptri) if ptri else 0.0}


def keys(st,pair):
    tie=tuple(-x for x in pair)
    return {'count_sum_min_ptrio':(st['positive_count'],st['positive_sum'],st['min_delta'],st['mean_ptrio'],tie),'count_sum_min_ptrifecta':(st['positive_count'],st['positive_sum'],st['min_delta'],st['mean_ptrifecta_set'],tie),'count_min_sum_ptrio':(st['positive_count'],st['min_delta'],st['positive_sum'],st['mean_ptrio'],tie),'count_mean_sum_ptrio':(st['positive_count'],st['mean_delta'],st['positive_sum'],st['mean_ptrio'],tie),'min_count_sum_ptrio':(st['min_delta'],st['positive_count'],st['positive_sum'],st['mean_ptrio'],tie)}


def main():
    gate=load_audited_gate();trio=load_trio_odds();trifecta=load_trifecta_odds();results=load_results()
    rules=['count_sum_min_ptrio','count_sum_min_ptrifecta','count_min_sum_ptrio','count_mean_sum_ptrio','min_count_sum_ptrio'];acc={r:defaultdict(int) for r in rules}
    for rid in sorted(gate):
        entrants=entrants_from_trio(trio[rid]);delta=compute_delta(trio[rid],trifecta[rid]);result=norm_combo(*results[rid]);rset=set(result)
        rows=[(tuple(p),stats(tuple(p),entrants,delta,trio[rid],trifecta[rid])) for p in itertools.combinations(entrants,2)]
        for rule in rules:
            pair,st=max(rows,key=lambda ps:keys(ps[1],ps[0])[rule]);tickets=[norm_combo(pair[0],pair[1],x) for x in entrants if x not in pair];a=acc[rule];a['bet_races']+=1;a['tickets']+=len(tickets);a['axis_survived']+=int(set(pair).issubset(rset));hit=result in tickets;a['hit_races']+=int(hit)
            if hit:a['payout']+=round(STAKE*trio[rid][result])
    out={'status':'ALL_PAIR_AXIS_SEARCH_2023','year':2023,'years_read':[2023],'population':'Audited 208 fake-favorite races only; every actual entrant pair is eligible as axis.','rules':{},'note':'Development search only. No 2024/2025/2026 data read.'}
    for rule,a in acc.items():
        stake=a['tickets']*STAKE;payout=a['payout'];out['rules'][rule]={'bet_races':a['bet_races'],'tickets':a['tickets'],'hit_races':a['hit_races'],'race_hit_rate_pct':100*a['hit_races']/a['bet_races'],'axis_pair_survival_pct':100*a['axis_survived']/a['bet_races'],'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
# trigger: literal all-pair axis search
