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
OUT = ROOT / 'data' / 'audits' / 'pair_rank_axis_2023.json'
STAKE = 100


def pair_stats(pair, entrants, delta, trio):
    combos=[norm_combo(pair[0],pair[1],x) for x in entrants if x not in pair]
    vals=[delta[c] for c in combos if c in delta]
    pos=[v for v in vals if v>0]
    ptrio=[1.0/trio[c] for c in combos if c in trio and trio[c]>0]
    return {
      'positive_count':len(pos),
      'positive_sum':sum(pos),
      'min_delta':min(vals) if vals else float('-inf'),
      'mean_ptrio':sum(ptrio)/len(ptrio) if ptrio else 0.0,
    }


def support_key(st,pair):
    return (st['positive_count'],st['positive_sum'],st['min_delta'],st['mean_ptrio'],tuple(-x for x in pair))


def main():
    gate=load_audited_gate(); trio=load_trio_odds(); trifecta=load_trifecta_odds(); results=load_results()
    acc={1:defaultdict(int),2:defaultdict(int),3:defaultdict(int)}
    for rid in sorted(gate):
        entrants=entrants_from_trio(trio[rid]); delta=compute_delta(trio[rid],trifecta[rid]); result=norm_combo(*results[rid])
        ranked=[]
        for p in itertools.combinations(entrants,2):
            pair=tuple(p); st=pair_stats(pair,entrants,delta,trio[rid]); ranked.append((pair,st))
        ranked=sorted(ranked,key=lambda ps:support_key(ps[1],ps[0]),reverse=True)
        for rank in (1,2,3):
            pair,st=ranked[rank-1]
            tickets=[norm_combo(pair[0],pair[1],x) for x in entrants if x not in pair]
            a=acc[rank]; a['bet_races']+=1; a['tickets']+=len(tickets)
            hit=result in tickets; a['hit_races']+=int(hit)
            if hit: a['payout']+=round(STAKE*trio[rid][result])
    out={'status':'PAIR_SUPPORT_RANK_AXIS_2023','year':2023,'years_read':[2023],
         'population':'Audited 208 fake-favorite races only.',
         'pair_support_definition':'Rank all actual entrant pairs by positive_count, then positive_sum, then min_delta, then mean trio implied mass, then lower car-number tuple.',
         'strategy':'Use pair support rank #1, #2, or #3 directly as the two-head axis; literal full flow to every other entrant.',
         'results':{},'note':'Development simulation only. No 2024/2025/2026 data read.'}
    for rank,a in acc.items():
        stake=a['tickets']*STAKE; payout=a['payout']
        out['results'][f'pair_rank_{rank}']={'bet_races':a['bet_races'],'tickets':a['tickets'],'hit_races':a['hit_races'],'race_hit_rate_pct':100*a['hit_races']/a['bet_races'],'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
