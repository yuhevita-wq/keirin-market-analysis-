from __future__ import annotations

import json
from collections import defaultdict

from simulate_v8_1_f02_2024q1 import load, pl
from evaluate_v8_17_q2_oos import load_q2
from evaluate_v8_25_q3_oos import load_q3
from evaluate_v8_25_q1q2 import summary
from evaluate_v9_0_q1q2q3 import load_entries
from v7_0_f01_market_hierarchy import trio_implied_probabilities
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_25_f26_final_set_lock_only import build_v8_25_f26
from v9_0_f27_fundamental_first_branch import fundamental_scores


def line_state(trio_odds, line_text, entry_rows):
    lines = pl(line_text)
    if not lines:
        return None
    fd = fundamental_scores(entry_rows)
    if not fd.get('ok'):
        return None
    p = trio_implied_probabilities(trio_odds)
    cars = sorted({c for combo in trio_odds for c in combo})
    s = {c: sum(prob for combo, prob in p.items() if c in combo) / 3.0 for c in cars}
    market_ls = [sum(s[c] for c in line) for line in lines]
    fund_ls = [sum(fd['F'][c] for c in line) for line in lines]
    mo = tuple(sorted(range(len(lines)), key=lambda i: (-market_ls[i], i)))
    fo = tuple(sorted(range(len(lines)), key=lambda i: (-fund_ls[i], i)))
    top1 = mo[0] == fo[0]
    top2 = len(mo) >= 2 and len(fo) >= 2 and set(mo[:2]) == set(fo[:2])
    order2 = len(mo) >= 2 and len(fo) >= 2 and mo[:2] == fo[:2]
    return {
        'market_order': mo,
        'fund_order': fo,
        'top1_agree': top1,
        'top2_set_agree': top2,
        'top2_order_agree': order2,
    }


def evaluate(label, data, entries):
    races, trio, tf, pay = data
    buckets = defaultdict(list)
    group_buckets = defaultdict(lambda: defaultdict(list))
    for rid, r in sorted(races.items(), key=lambda x: (x[1].get('race_date',''), x[0])):
        if r.get('meeting_grade') != 'F1' or not (r.get('race_type') or '').startswith('Ｓ級'):
            continue
        if r.get('entry_count') not in ('7', 7):
            try:
                if int(float(r.get('entry_count'))) != 7: continue
            except Exception:
                continue
        if len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210 or rid not in pay or not pay[rid]:
            continue
        lines = pl(r.get('predicted_line_formation'))
        cars = sorted({v for c in trio[rid] for v in c})
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        rt = r.get('race_type') or ''
        d = build_v8_25_f26(trio[rid], tf[rid], r.get('predicted_line_formation') or '', rt)
        if not d.get('buy'):
            continue
        st = line_state(trio[rid], r.get('predicted_line_formation') or '', entries.get(rid, []))
        if not st:
            continue
        wins = [t for t in d['tickets'] if t in pay[rid]]
        row = {
            'race_id': rid, 'race_date': r.get('race_date'), 'race_type': rt,
            'group': classify_race_type(rt), 'hit': int(bool(wins)),
            'payout': sum(pay[rid][t] for t in wins), 'tickets': int(d['ticket_count']),
        }
        keys = [
            'ALL',
            'TOP1_AGREE' if st['top1_agree'] else 'TOP1_DISAGREE',
            'TOP2_SET_AGREE' if st['top2_set_agree'] else 'TOP2_SET_DISAGREE',
            'TOP2_ORDER_AGREE' if st['top2_order_agree'] else 'TOP2_ORDER_DISAGREE',
        ]
        for k in keys:
            buckets[k].append(row)
            group_buckets[k][row['group']].append(row)
    return {
        'dataset': label,
        'overall': {k: summary(v) for k, v in sorted(buckets.items())},
        'by_group': {
            k: {g: summary(rows) for g, rows in sorted(groups.items())}
            for k, groups in sorted(group_buckets.items())
        },
    }


def main():
    result = {
        'diagnostic': 'v9.1 universal market-vs-fundamental line-rank agreement',
        'status': 'DEVELOPMENT_Q1Q2Q3_NO_NEW_THRESHOLD',
        'candidate_semantics': [
            'market top line == fundamental top line',
            'market top-2 line set == fundamental top-2 line set',
            'market top-2 line order == fundamental top-2 line order',
        ],
        'Q1': evaluate('2024Q1', load(), load_entries('2024_q1')),
        'Q2': evaluate('2024Q2', load_q2(), load_entries('2024_q2')),
        'Q3': evaluate('2024Q3', load_q3(), load_entries('2024_q3')),
    }
    print('V9_1_LINE_AGREEMENT_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V9_1_LINE_AGREEMENT_END')

if __name__ == '__main__':
    main()
