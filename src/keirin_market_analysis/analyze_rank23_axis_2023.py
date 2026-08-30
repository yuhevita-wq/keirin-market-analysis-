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
OUT = ROOT / 'data' / 'audits' / 'rank23_axis_2023.json'
STAKE = 100


def rider_stats(rider, entrants, delta):
    vals=[]
    others=[x for x in entrants if x!=rider]
    for a,b in itertools.combinations(others,2):
        c=norm_combo(rider,a,b)
        if c in delta:
            vals.append(delta[c])
    pos=[v for v in vals if v>0]
    return {
        'positive_count': len(pos),
        'positive_sum': sum(pos),
        'all_delta_sum': sum(vals),
        'mean_delta': sum(vals)/len(vals) if vals else float('-inf'),
    }


def rider_key(st, rider):
    return (st['positive_count'], st['positive_sum'], st['all_delta_sum'], st['mean_delta'], -rider)


def main():
    gate=load_audited_gate(); trio=load_trio_odds(); trifecta=load_trifecta_odds(); results=load_results()
    acc=defaultdict(int)
    rank1_top3=0
    rank2_top3=0
    rank3_top3=0
    rank1_with_axis_hit=0
    rank1_with_axis_total=0
    rows=[]
    for rid in sorted(gate):
        entrants=entrants_from_trio(trio[rid]); delta=compute_delta(trio[rid],trifecta[rid]); result=norm_combo(*results[rid]); rset=set(result)
        ranked=sorted([(r,rider_stats(r,entrants,delta)) for r in entrants], key=lambda x:rider_key(x[1],x[0]), reverse=True)
        r1,r2,r3=ranked[0][0],ranked[1][0],ranked[2][0]
        pair=norm_combo(r2,r3)
        tickets=[norm_combo(r2,r3,x) for x in entrants if x not in pair]
        hit=result in tickets
        acc['bet_races']+=1; acc['tickets']+=len(tickets); acc['hit_races']+=int(hit); acc['axis_survived']+=int(set(pair).issubset(rset))
        rank1_top3+=int(r1 in rset); rank2_top3+=int(r2 in rset); rank3_top3+=int(r3 in rset)
        if set(pair).issubset(rset):
            rank1_with_axis_total += 1
            rank1_with_axis_hit += int(r1 in rset)
        if hit: acc['payout'] += round(STAKE*trio[rid][result])
        rows.append({'race_id':rid,'rank1':r1,'rank2':r2,'rank3':r3,'axis':pair,'result':result,'hit':hit})
    stake=acc['tickets']*STAKE; payout=acc['payout']
    out={
      'status':'RANK23_AXIS_2023','year':2023,'years_read':[2023],
      'population':'Audited 208 fake-favorite races only.',
      'support_definition':'For each rider, aggregate all trio-combination delta values containing that rider. Rank by positive_count, then positive_sum, then all_delta_sum, then mean_delta, then lower car number.',
      'strategy':'Use support-rank #2 and #3 riders as the two-head axis. Support-rank #1 remains eligible only as a flow target, not an axis rider.',
      'result':{
        'bet_races':acc['bet_races'],'tickets':acc['tickets'],'hit_races':acc['hit_races'],
        'race_hit_rate_pct':100*acc['hit_races']/acc['bet_races'],
        'axis_pair_survival_pct':100*acc['axis_survived']/acc['bet_races'],
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake,
        'rank1_top3_pct':100*rank1_top3/acc['bet_races'],'rank2_top3_pct':100*rank2_top3/acc['bet_races'],'rank3_top3_pct':100*rank3_top3/acc['bet_races'],
        'rank1_top3_when_axis23_survived_pct':100*rank1_with_axis_hit/rank1_with_axis_total if rank1_with_axis_total else None,
        'axis23_survived_races':rank1_with_axis_total,
      },
      'note':'Development simulation only. No 2024/2025/2026 data read.'
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
